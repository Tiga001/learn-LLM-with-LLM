# Day 2：ZeRO 显存优化详解 — 从「切了就放得下」到精确显存分析

> **目标**：深入理解 ZeRO（Zero Redundancy Optimizer）三个阶段的核心思想——如何通过分片消除数据并行中的显存冗余；精确推导每个阶段的显存占用公式（面试高频！）；掌握 ZeRO-Offload / ZeRO-Infinity 的卸载策略；对比 DeepSpeed ZeRO 与 PyTorch FSDP。
>
> **前置知识**：Day 1 数据并行原理、AllReduce 通信、显存四大组成。

---

## 一、模型训练显存组成精确分析

### 1.1 显存四大组成

以参数量 $\Phi$ 的模型为例，使用 **AdamW + 混合精度训练**（FP16 计算 + FP32 优化器）：

| 组成 | 说明 | 每参数字节数 | 总量 |
|------|------|:----------:|------|
| 参数（FP16） | 用于前向 / 反向 | 2 bytes | $2\Phi$ |
| 梯度（FP16） | 反向传播的梯度 | 2 bytes | $2\Phi$ |
| 优化器状态 | FP32 参数副本 + $m$ + $v$ | $4 + 4 + 4 = 12$ bytes | $12\Phi$ |
| **合计** | | **16 bytes** | $16\Phi$ |

> 注意：激活值的显存与 batch size / 序列长度相关，这里暂不计入（可用 activation checkpointing 优化）。

### 1.2 关键洞察

$$\boxed{\text{优化器状态占 } \frac{12}{16} = 75\% \text{ 的显存！}}$$

这意味着：如果我们能把优化器状态分片到多张 GPU 上，就能显著减少每卡显存。

### 1.3 数值例子

| 模型 | 参数量 $\Phi$ | 参数+梯度 (FP16) | 优化器状态 (FP32) | 合计 |
|------|:-----------:|:---------------:|:---------------:|:----:|
| 7B | $7 \times 10^9$ | 28 GB | 84 GB | 112 GB |
| 13B | $13 \times 10^9$ | 52 GB | 156 GB | 208 GB |
| 70B | $70 \times 10^9$ | 280 GB | 840 GB | 1120 GB |

---

## 二、ZeRO 核心思想

### 2.1 DP 的冗余分析

在标准 DP 中，$N_d$ 张 GPU 每卡都持有完整的：

$$\text{每卡显存} = 2\Phi + 2\Phi + 12\Phi = 16\Phi \text{ bytes}$$

总系统显存 = $N_d \times 16\Phi$，但有效数据只有 $16\Phi$。

$$\boxed{\text{冗余率} = \frac{(N_d - 1) \times 16\Phi}{N_d \times 16\Phi} = \frac{N_d - 1}{N_d} \approx 1 - \frac{1}{N_d}}$$

8 张 GPU 时冗余率 = 87.5%！

### 2.2 ZeRO 的三步切分策略

ZeRO 的核心思想：**既然每卡不需要完整数据，那就把数据切片分给不同 GPU**。

```
                    DP (无优化)        ZeRO-1         ZeRO-2         ZeRO-3
每卡存储:
  参数 (2Φ)          完整              完整            完整           分片 (2Φ/Nd)
  梯度 (2Φ)          完整              完整            分片 (2Φ/Nd)   分片 (2Φ/Nd)
  优化器 (12Φ)       完整              分片 (12Φ/Nd)   分片 (12Φ/Nd)  分片 (12Φ/Nd)

每卡显存:            16Φ             4Φ + 12Φ/Nd    2Φ + 14Φ/Nd    16Φ/Nd
```

---

## 三、ZeRO Stage 1 — 优化器状态分片

### 3.1 原理

ZeRO-1 将优化器状态（FP32 参数副本 + $m$ + $v$）均分到 $N_d$ 张 GPU：

- 每卡只维护 $1/N_d$ 的优化器状态
- 参数和梯度仍然每卡完整持有
- 反向传播时照常 AllReduce 梯度
- 参数更新时，每卡只更新自己负责的那 $1/N_d$ 参数
- 更新后通过 AllGather 把更新后的参数同步给所有卡

### 3.2 显存公式

$$\boxed{\text{ZeRO-1 每卡显存} = 2\Phi + 2\Phi + \frac{12\Phi}{N_d} = 4\Phi + \frac{12\Phi}{N_d}}$$

### 3.3 通信分析

- **反向传播**：AllReduce 梯度（与 DP 相同）= $2\Phi$
- **参数更新后**：AllGather 更新后的参数 = $\Phi$（FP16）

$$\text{ZeRO-1 通信量} = 2\Phi + \Phi = 3\Phi \quad (\text{略高于 DP 的 } 2\Phi)$$

### 3.4 数值对比

以 7B 模型 + 8 卡为例：

| | DP | ZeRO-1 | 节省 |
|------|-----|--------|:----:|
| 参数 (FP16) | 14 GB | 14 GB | - |
| 梯度 (FP16) | 14 GB | 14 GB | - |
| 优化器 (FP32) | 84 GB | 10.5 GB | 87.5% |
| **每卡合计** | **112 GB** | **38.5 GB** | **65.6%** |

> ZeRO-1 已经把 7B 模型的每卡显存从 112 GB 降到 38.5 GB，一张 A100 80GB 可以放下了！

---

## 四、ZeRO Stage 2 — 梯度 + 优化器分片

### 4.1 原理

ZeRO-2 在 ZeRO-1 基础上，进一步将梯度也分片：

- 反向传播时，不做 AllReduce，而是做 **ReduceScatter**
- 每卡只保留自己负责的 $1/N_d$ 梯度（归约后）
- 其余梯度可以立即释放
- 参数更新后同样需要 AllGather

### 4.2 显存公式

$$\boxed{\text{ZeRO-2 每卡显存} = 2\Phi + \frac{2\Phi}{N_d} + \frac{12\Phi}{N_d} = 2\Phi + \frac{14\Phi}{N_d}}$$

### 4.3 通信分析

- **反向传播**：ReduceScatter 梯度 = $\Phi$
- **参数更新后**：AllGather 参数 = $\Phi$

$$\text{ZeRO-2 通信量} = \Phi + \Phi = 2\Phi \quad (\text{与 DP 相同！})$$

> 关键：ZeRO-2 在减少显存的同时，**通信量没有增加**！

### 4.4 数值对比

以 7B 模型 + 8 卡为例：

| | DP | ZeRO-1 | ZeRO-2 | 节省(vs DP) |
|------|-----|--------|--------|:----------:|
| 参数 (FP16) | 14 GB | 14 GB | 14 GB | - |
| 梯度 (FP16) | 14 GB | 14 GB | 1.75 GB | 87.5% |
| 优化器 (FP32) | 84 GB | 10.5 GB | 10.5 GB | 87.5% |
| **每卡合计** | **112 GB** | **38.5 GB** | **26.25 GB** | **76.6%** |

---

## 五、ZeRO Stage 3 — 全分片

### 5.1 原理

ZeRO-3 把参数、梯度、优化器状态**全部分片**：

- 每卡只持有 $1/N_d$ 的参数
- **前向传播时**：需要某层参数 → AllGather 收集完整参数 → 计算 → 释放
- **反向传播时**：需要某层参数 → AllGather → 计算梯度 → ReduceScatter 梯度 → 释放参数和非本卡梯度
- **参数更新**：每卡只更新自己负责的 $1/N_d$ 参数

### 5.2 显存公式

$$\boxed{\text{ZeRO-3 每卡显存} = \frac{2\Phi}{N_d} + \frac{2\Phi}{N_d} + \frac{12\Phi}{N_d} = \frac{16\Phi}{N_d}}$$

> 理论上，显存随 GPU 数量线性下降！$N_d$ 足够大时，可以训练任意大的模型。

### 5.3 通信分析

- **前向传播**：AllGather 参数（$L$ 层，每层收集后释放）= $\Phi$
- **反向传播**：AllGather 参数 + ReduceScatter 梯度 = $\Phi + \Phi = 2\Phi$

$$\text{ZeRO-3 通信量} = \Phi + 2\Phi = 3\Phi \quad (\text{是 DP 的 1.5 倍})$$

### 5.4 数值对比

以 7B 模型 + 8 卡为例：

| | DP | ZeRO-1 | ZeRO-2 | ZeRO-3 | 节省(vs DP) |
|------|-----|--------|--------|--------|:----------:|
| 参数 (FP16) | 14 GB | 14 GB | 14 GB | 1.75 GB | 87.5% |
| 梯度 (FP16) | 14 GB | 14 GB | 1.75 GB | 1.75 GB | 87.5% |
| 优化器 (FP32) | 84 GB | 10.5 GB | 10.5 GB | 10.5 GB | 87.5% |
| **每卡合计** | **112 GB** | **38.5 GB** | **26.25 GB** | **14 GB** | **87.5%** |

### 5.5 前向 / 反向时的参数流动

```
前向传播 (Layer l):
  ┌──────────────────────────────────────────────────────┐
  │ GPU k 持有参数分片 W_l^(k) (1/Nd 大小)                │
  │                                                      │
  │ Step 1: AllGather → 收集完整 W_l                      │
  │ Step 2: 用完整 W_l 计算前向  output_l = f(input_l, W_l) │
  │ Step 3: 释放非本卡的参数分片 (只保留 W_l^(k))           │
  └──────────────────────────────────────────────────────┘

反向传播 (Layer l):
  ┌──────────────────────────────────────────────────────┐
  │ Step 1: AllGather → 收集完整 W_l                      │
  │ Step 2: 计算梯度 ∂L/∂W_l                              │
  │ Step 3: ReduceScatter → 每卡只保留 (∂L/∂W_l)^(k)     │
  │ Step 4: 释放完整 W_l 和非本卡梯度                      │
  └──────────────────────────────────────────────────────┘
```

---

## 六、ZeRO 三阶段显存对比总览

### 6.1 公式汇总

| 阶段 | 每卡显存 | 通信量 | 通信 vs DP |
|------|---------|--------|:---------:|
| DP (baseline) | $16\Phi$ | $2\Phi$ | 1× |
| ZeRO-1 (OS) | $4\Phi + \frac{12\Phi}{N_d}$ | $3\Phi$ | 1.5× |
| ZeRO-2 (OS+G) | $2\Phi + \frac{14\Phi}{N_d}$ | $2\Phi$ | 1× |
| ZeRO-3 (OS+G+P) | $\frac{16\Phi}{N_d}$ | $3\Phi$ | 1.5× |

> OS = Optimizer States, G = Gradients, P = Parameters

### 6.2 7B / 13B / 70B 模型显存对比表

**8 张 GPU，AdamW + FP16 混合精度**：

| 模型 | DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|---:|-------:|-------:|-------:|
| 7B | 112 GB | 38.5 GB | 26.25 GB | 14 GB |
| 13B | 208 GB | 71.5 GB | 48.75 GB | 26 GB |
| 70B | 1120 GB | 385 GB | 262.5 GB | 140 GB |

**64 张 GPU**：

| 模型 | DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|---:|-------:|-------:|-------:|
| 7B | 112 GB | 15.3 GB | 15.5 GB | 1.75 GB |
| 13B | 208 GB | 28.4 GB | 28.8 GB | 3.25 GB |
| 70B | 1120 GB | 153.1 GB | 155 GB | 17.5 GB |

### 6.3 显存对比图（ASCII）

```
每卡显存 (7B 模型, 8 GPU)

    112 GB  ████████████████████████████████████████████████████  DP
            ▓▓▓▓▓▓▓ 参数  ▓▓▓▓▓▓▓ 梯度  ████████████████████████████ 优化器

    38.5 GB ▓▓▓▓▓▓▓ 参数  ▓▓▓▓▓▓▓ 梯度  ████ 优化器                   ZeRO-1

    26.25GB ▓▓▓▓▓▓▓ 参数  ▒ 梯度  ████ 优化器                          ZeRO-2

    14 GB   ▒参 ▒梯  ████ 优化器                                       ZeRO-3
            ├────────┼────────┼────────┼────────┼────────┤
            0        25       50       75       100      112 GB
```

---

## 七、ZeRO-Offload 与 ZeRO-Infinity

### 7.1 ZeRO-Offload

当 GPU 显存仍然不够时，可以将部分数据卸载到 CPU 内存：

```
┌─────────────────┐     ┌─────────────────┐
│      GPU        │     │      CPU        │
│                 │     │                 │
│  FP16 参数      │     │  FP32 参数副本   │
│  FP16 梯度      │──→  │  FP32 梯度      │
│  前向/反向计算   │     │  Adam m, v     │
│                 │  ←──│  参数更新        │
└─────────────────┘     └─────────────────┘
```

- 前向 / 反向传播在 GPU 上执行（需要速度）
- 优化器状态和参数更新在 CPU 上执行（计算量小，但数据量大）
- 通信瓶颈：GPU ↔ CPU 的 PCIe 带宽（~32 GB/s）

### 7.2 ZeRO-Infinity

ZeRO-Infinity 进一步将数据卸载到 NVMe SSD：

```
GPU 显存  ←→  CPU 内存  ←→  NVMe SSD
(快但小)      (中)          (大但慢)
80 GB         1 TB          数 TB
```

- 理论上可以训练任意大的模型（只要 SSD 够大）
- 实际受限于 I/O 带宽：NVMe ~3.5 GB/s
- 适合资源极度受限的场景（如学术界单机训练超大模型）

### 7.3 Offload 适用场景

| 方案 | GPU 显存 | CPU 内存 | NVMe | 速度 | 适用 |
|------|---------|---------|------|------|------|
| ZeRO-3 | ✅ 最小 | — | — | ✅ 最快 | 多卡训练 |
| ZeRO-Offload | ✅ 更小 | ✅ 需要 | — | 较慢 | GPU 不足 |
| ZeRO-Infinity | ✅ 最小 | ✅ 需要 | ✅ 需要 | 最慢 | 极端场景 |

---

## 八、DeepSpeed 配置实践

### 8.1 ZeRO Stage 1 配置

```json
{
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    },
    "bf16": {"enabled": true},
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 1
}
```

### 8.2 ZeRO Stage 2 配置

```json
{
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    },
    "bf16": {"enabled": true},
    "train_batch_size": 32,
    "gradient_accumulation_steps": 2
}
```

### 8.3 ZeRO Stage 3 配置

```json
{
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "sub_group_size": 1e12,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "bf16": {"enabled": true},
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4
}
```

### 8.4 ZeRO-Offload 配置

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### 8.5 关键配置参数说明

| 参数 | 作用 |
|------|------|
| `overlap_comm` | 通信与计算重叠（推荐开启） |
| `contiguous_gradients` | 梯度连续存储（减少碎片） |
| `reduce_bucket_size` | 梯度分桶大小（影响通信效率） |
| `stage3_prefetch_bucket_size` | ZeRO-3 预取大小（计算-通信重叠） |
| `stage3_param_persistence_threshold` | 小参数不分片的阈值（减少通信） |
| `gradient_accumulation_steps` | 梯度累积步数（有效增大 batch） |

---

## 九、FSDP vs DeepSpeed ZeRO

### 9.1 PyTorch FSDP 简介

PyTorch FSDP（Fully Sharded Data Parallel）是 PyTorch 原生的 ZeRO-3 实现：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 等价 ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    device_id=local_rank,
)
```

### 9.2 Sharding Strategy 对应关系

| FSDP Strategy | 等价 ZeRO Stage | 说明 |
|--------------|:---------------:|------|
| `NO_SHARD` | DP | 不分片 |
| `SHARD_GRAD_OP` | ZeRO-2 | 梯度+优化器分片 |
| `FULL_SHARD` | ZeRO-3 | 全分片 |
| `HYBRID_SHARD` | — | node 内 FULL_SHARD + node 间 NO_SHARD |

### 9.3 DeepSpeed vs FSDP 对比

| 特性 | DeepSpeed ZeRO | PyTorch FSDP |
|------|---------------|-------------|
| 实现方 | Microsoft | PyTorch (Meta) |
| ZeRO-1 支持 | ✅ | ❌（无等价） |
| ZeRO-2 支持 | ✅ | ✅ `SHARD_GRAD_OP` |
| ZeRO-3 支持 | ✅ | ✅ `FULL_SHARD` |
| CPU Offload | ✅ 成熟 | ✅ 支持 |
| NVMe Offload | ✅ ZeRO-Infinity | ❌ |
| 混合精度 | ✅ | ✅ |
| 配置方式 | JSON 配置文件 | Python API |
| 与 HuggingFace 集成 | ✅ Trainer 集成 | ✅ Trainer 集成 |
| 社区生态 | 成熟 | 快速成长 |
| **推荐场景** | 极大模型、Offload 需求 | PyTorch 原生偏好 |

---

## 十、Activation Checkpointing（激活重计算）

### 10.1 激活值显存问题

除了参数/梯度/优化器，**激活值**也占大量显存。以 Transformer 为例：

$$\text{Memory}_{\text{act}} \approx L \times s \times b \times d \times \text{bytes\_per\_element} \times \text{constant}$$

其中 $L$ = 层数，$s$ = 序列长度，$b$ = batch size，$d$ = 隐层维度。

### 10.2 Activation Checkpointing 原理

用**时间换空间**：前向传播时不保存中间层激活，反向传播时**重新计算**。

```
标准训练:     前向 → 保存所有激活 → 反向（用保存的激活计算梯度）
                     显存: O(L) 激活

Checkpointing: 前向 → 只保存 checkpoint 层激活 → 反向时重算中间激活
                     显存: O(√L) 激活     计算: ~1.33× 前向
```

### 10.3 在 DeepSpeed / FSDP 中使用

```python
# DeepSpeed
from deepspeed.runtime.activation_checkpointing import checkpointing
deepspeed.checkpointing.configure(num_checkpoints=num_layers)

# PyTorch
from torch.utils.checkpoint import checkpoint
output = checkpoint(layer, input, use_reentrant=False)
```

---

## 十一、自检题

1. 写出 ZeRO Stage 1 / 2 / 3 各阶段的每卡显存公式，并解释每个阶段切分了什么。
2. 以 LLaMA-70B + AdamW + FP16 + 64 张 GPU 为例，计算 ZeRO-3 的每卡显存占用。
3. ZeRO-2 的通信量为什么与标准 DP 相同？ZeRO-3 为什么多了 50%？
4. 解释 ZeRO-3 前向传播时为什么需要 AllGather，反向传播时为什么需要 AllGather + ReduceScatter。
5. ZeRO-Offload 和 ZeRO-Infinity 分别将什么数据卸载到哪里？各自的瓶颈是什么？
6. DeepSpeed ZeRO-3 和 PyTorch FSDP `FULL_SHARD` 有哪些核心差异？
7. Activation Checkpointing 用什么换什么？计算开销大约增加多少？
8. 在 DeepSpeed 配置中，`overlap_comm` 和 `stage3_prefetch_bucket_size` 分别优化什么？
9. 为什么说「优化器状态占 75% 的显存」？这对 ZeRO-1 的设计有什么启发？

---

## 十二、产出要求

- [ ] 默写 ZeRO Stage 1/2/3 的每卡显存公式（**面试必考！**）
- [ ] 画出 ZeRO 三阶段的显存对比图（类似本文 6.3 节）
- [ ] 计算 7B / 13B / 70B 模型在 8 卡和 64 卡下 ZeRO 各阶段的每卡显存
- [ ] 写出 ZeRO-3 前向 / 反向传播时参数的 AllGather / ReduceScatter 流程
- [ ] 撰写 DeepSpeed ZeRO vs PyTorch FSDP 对比表
- [ ] 配置一份完整的 DeepSpeed ZeRO-3 JSON 配置文件（含关键参数注释）
