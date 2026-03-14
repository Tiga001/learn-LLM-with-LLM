# Day 1：分布式训练概览与数据并行 — 从单卡到多卡的第一步

> **目标**：理解大模型训练为什么需要分布式——从显存和计算两个维度分析单卡瓶颈；建立分布式训练五大并行维度（DP / TP / PP / CP / EP）的全景认知；深入数据并行（DP / DDP）的原理与 AllReduce 通信机制；掌握分布式优化器和混合精度训练（AMP）的工程实践。
>
> **前置知识**：W4 LLaMA 模型架构、W13 DeepSpeed 多卡实操基础。

---

## 一、为什么需要分布式训练

### 1.1 单卡显存瓶颈

训练一个模型，GPU 显存需要容纳四部分内容：

$$\text{Memory}_{\text{total}} = \text{Memory}_{\text{params}} + \text{Memory}_{\text{gradients}} + \text{Memory}_{\text{optimizer}} + \text{Memory}_{\text{activations}}$$

以 LLaMA-7B 为例，使用 AdamW 优化器 + FP16 混合精度训练：

| 组成 | 计算方式 | 大小 |
|------|---------|------|
| 参数（FP16） | $7B \times 2$ bytes | 14 GB |
| 梯度（FP16） | $7B \times 2$ bytes | 14 GB |
| 优化器状态（FP32） | $7B \times (4 + 4 + 4)$ bytes（参数副本 + $m$ + $v$） | 84 GB |
| 激活值（依赖 batch/seq） | 与 batch size、序列长度相关 | 数 GB ~ 数十 GB |
| **合计** | | **> 112 GB** |

> 单张 A100 80GB 连参数 + 梯度 + 优化器状态都放不下，更不用说激活值。

### 1.2 单卡计算瓶颈

即使显存够用，单卡计算速度也有上限：

$$\text{训练时间} = \frac{\text{总 FLOPs}}{\text{GPU 吞吐量}} = \frac{6 \times N \times D}{\text{TFLOPS} \times \text{GPU 利用率}}$$

其中 $N$ 为参数量，$D$ 为 token 数。以 LLaMA-7B 训练 1T tokens 为例：

- 总 FLOPs ≈ $6 \times 7 \times 10^9 \times 10^{12} = 4.2 \times 10^{22}$
- A100 BF16 吞吐 ≈ 312 TFLOPS，实际利用率 ~50%
- 单卡训练时间 ≈ $\frac{4.2 \times 10^{22}}{312 \times 10^{12} \times 0.5} \approx 2.7 \times 10^8$ 秒 ≈ **8.5 年**

> 结论：大模型训练**既是显存问题，也是计算问题**，必须多卡并行。

### 1.3 Scaling Law 驱动

Chinchilla Scaling Law 表明，最优训练需要参数量和数据量同步增长：

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

随着算力预算 $C$ 增长，模型越来越大，分布式训练成为必然。

---

## 二、分布式训练全景图

### 2.1 五大并行维度

```
┌─────────────────────────────────────────────────────────────┐
│                    分布式训练五大并行维度                       │
├──────────────┬──────────────────────────────────────────────┤
│  并行维度     │  切分什么                                     │
├──────────────┼──────────────────────────────────────────────┤
│  DP (数据)    │  数据 batch → 每卡一份子 batch，模型完整复制    │
│  TP (张量)    │  层内权重矩阵 → 列切分 / 行切分               │
│  PP (流水线)  │  层间模型 → 不同层放不同卡                     │
│  CP (上下文)  │  序列维度 → 长序列切分给多卡 (Ring Attention)   │
│  EP (专家)    │  MoE 专家 → 不同专家放不同卡                   │
└──────────────┴──────────────────────────────────────────────┘
```

### 2.2 并行维度的作用对比

| 维度 | 解决什么问题 | 通信模式 | 典型场景 |
|------|------------|---------|---------|
| DP | 加速计算（同一模型处理更多数据） | AllReduce 梯度 | 几乎所有训练 |
| TP | 单层放不下一张卡 | AllReduce 激活 | 大模型（node 内） |
| PP | 模型太深放不下一张卡 | 点对点发送激活 | 超大模型（跨 node） |
| CP | 序列太长，注意力计算放不下 | Ring 通信 KV | 长上下文训练 |
| EP | MoE 专家太多放不下 | AllToAll token 路由 | MoE 模型 |

### 2.3 并行维度的组合

实际训练中通常组合多种并行：

```
3D 并行 = DP + TP + PP                    (Megatron-LM)
4D 并行 = DP + TP + PP + CP               (LLaMA-3)
5D 并行 = DP + TP + PP + CP + EP          (DeepSeek-V3)
```

---

## 三、数据并行（DP）原理

### 3.1 基本思想

数据并行是最简单、最常用的并行方式：

```
                    ┌─────────────┐
                    │  全量数据     │
                    └──────┬──────┘
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       ┌─────────┐   ┌─────────┐   ┌─────────┐
       │ GPU 0   │   │ GPU 1   │   │ GPU 2   │
       │ Model₀  │   │ Model₁  │   │ Model₂  │
       │ Batch₀  │   │ Batch₁  │   │ Batch₂  │
       └────┬────┘   └────┬────┘   └────┬────┘
            │  前向 + 反向  │              │
            ▼              ▼              ▼
       ┌─────────┐   ┌─────────┐   ┌─────────┐
       │  Grad₀  │   │  Grad₁  │   │  Grad₂  │
       └────┬────┘   └────┬────┘   └────┬────┘
            │              │              │
            └──────── AllReduce ──────────┘
                           │
                    ┌──────┴──────┐
                    │ Avg Gradient │
                    └──────┬──────┘
                           │
                    每卡独立更新参数
```

**核心步骤**：
1. 每张 GPU 持有**完整模型副本**
2. 将 mini-batch 均分到各 GPU
3. 各 GPU 独立做前向 + 反向传播
4. 通过 AllReduce 同步梯度（求平均）
5. 各 GPU 用相同梯度独立更新参数

### 3.2 数学等价性

设总 batch size 为 $B$，使用 $N_d$ 张 GPU，每卡 batch size 为 $B/N_d$。

单卡全量梯度：

$$g = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \mathcal{L}(x_i, \theta)$$

DP 等价梯度：

$$g_{\text{DP}} = \frac{1}{N_d} \sum_{k=1}^{N_d} g_k = \frac{1}{N_d} \sum_{k=1}^{N_d} \frac{1}{B/N_d} \sum_{i \in \mathcal{B}_k} \nabla_\theta \mathcal{L}(x_i, \theta) = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \mathcal{L}(x_i, \theta) = g$$

> DP 在数学上严格等价于单卡大 batch 训练（同步 SGD 情况下）。

### 3.3 Parameter Server vs AllReduce

**Parameter Server（PS）架构**：

```
Worker 0 ──push grad──→ ┌────────────┐ ←──push grad── Worker 2
                         │   PS Node   │
Worker 1 ──push grad──→ │  聚合 + 更新  │ ←──push grad── Worker 3
                         └──────┬─────┘
                          pull params
```

- 中心化架构，PS 是通信瓶颈
- 支持异步更新（但会导致梯度 staleness）
- 已基本被 AllReduce 取代

**AllReduce 架构**：

- 去中心化，所有 GPU 对等通信
- 每张 GPU 最终都持有完整的聚合结果
- 通信量与 GPU 数量几乎无关（Ring AllReduce）

---

## 四、AllReduce 通信机制

### 4.1 Naive AllReduce

最简单的方案：所有 GPU 把梯度发给一个 GPU，聚合后广播回去。

- 通信量：$2(N_d - 1) \times M$，其中 $M$ 为参数量（以字节计）
- 瓶颈：一个 GPU 承担所有通信

### 4.2 Ring AllReduce

Ring AllReduce 将通信均匀分摊到所有 GPU，分两个阶段：

**阶段一：ReduceScatter**（每个 GPU 拥有 1/N 的聚合结果）

```
步骤 0:  GPU0[A₀|B₀|C₀]  GPU1[A₁|B₁|C₁]  GPU2[A₂|B₂|C₂]
步骤 1:  GPU0 发 C₀→GPU1, GPU1 发 A₁→GPU2, GPU2 发 B₂→GPU0
         GPU0[A₀|B₀+B₂|C₀]  GPU1[A₁|B₁|C₀+C₁]  GPU2[A₁+A₂|B₂|C₂]
步骤 2:  继续环形传递，每步聚合一个 chunk
         GPU0[A₀|Σ B|C₀]  GPU1[A₁|B₁|Σ C]  GPU2[Σ A|B₂|C₂]
```

**阶段二：AllGather**（将聚合结果广播给所有 GPU）

```
步骤 3:  GPU0 发 Σ B→GPU1, GPU1 发 Σ C→GPU2, GPU2 发 Σ A→GPU0
步骤 4:  继续环形传递
结果:    每个 GPU 都有 [Σ A | Σ B | Σ C]
```

**通信量分析**：

每个 GPU 在 ReduceScatter 和 AllGather 各发送 $\frac{N_d - 1}{N_d} \times M$：

$$\boxed{\text{Ring AllReduce 总通信量（每 GPU）} = 2 \times \frac{N_d - 1}{N_d} \times M \approx 2M}$$

> 关键洞察：Ring AllReduce 的每 GPU 通信量与 GPU 数量 $N_d$ 几乎无关！随着 GPU 增加，通信量恒定。

### 4.3 通信原语总览

| 原语 | 操作 | 每 GPU 通信量 | 用途 |
|------|------|:------------:|------|
| AllReduce | 全局归约 + 广播 | $2 \frac{N_d-1}{N_d} M$ | DP 梯度同步 |
| AllGather | 每 GPU 的数据拼接广播 | $\frac{N_d-1}{N_d} M$ | ZeRO-3 参数收集 |
| ReduceScatter | 归约后分片 | $\frac{N_d-1}{N_d} M$ | ZeRO-2 梯度分片 |
| AllToAll | 全交换（每对 GPU 交换不同数据） | 取决于数据分布 | EP 专家路由 |
| Broadcast | 一对多广播 | $M$ | 参数初始化 |
| Reduce | 多对一归约 | $M$ | 聚合到单 GPU |

### 4.4 通信带宽与延迟

实际通信时间由两部分组成：

$$T_{\text{comm}} = \alpha + \frac{M}{\beta}$$

- $\alpha$：延迟（latency），与消息数量相关
- $\beta$：带宽（bandwidth），与数据量相关
- NVLink (intra-node): $\beta \approx 600$ GB/s (A100)
- InfiniBand (inter-node): $\beta \approx 200$ GB/s (HDR)

> **工程规则**：TP 通信频繁但数据量小 → 放 node 内（NVLink）；PP/DP 通信量大但频率低 → 可跨 node。

---

## 五、PyTorch DDP 机制

### 5.1 从 DP 到 DDP

PyTorch 的 `DataParallel`（DP）和 `DistributedDataParallel`（DDP）的区别：

| 特性 | DP (`nn.DataParallel`) | DDP (`nn.parallel.DistributedDataParallel`) |
|------|----------------------|-------------------------------------------|
| 通信 | GPU 0 聚合（瓶颈） | Ring AllReduce（均衡） |
| 进程模型 | 单进程多线程 | 多进程（每 GPU 一个进程） |
| GIL 影响 | 受 Python GIL 限制 | 无 GIL 问题 |
| 跨节点 | 不支持 | 支持 |
| 性能 | 差（GPU 0 瓶颈） | 好（通信均衡） |
| **推荐** | ❌ 不推荐 | ✅ 生产标准 |

### 5.2 DDP 核心机制

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

for batch in dataloader:
    loss = model(batch)
    loss.backward()       # 反向传播时自动触发 AllReduce
    optimizer.step()
    optimizer.zero_grad()
```

**Gradient Bucketing**：DDP 不是等所有梯度计算完再做 AllReduce，而是将梯度分桶，反向传播计算与通信**重叠**：

```
反向传播:  ← Layer N ← Layer N-1 ← ... ← Layer 2 ← Layer 1
AllReduce:           [Bucket 3]  [Bucket 2]  [Bucket 1]
                     ↑ 反向传播到这里时，Bucket 1 已经开始 AllReduce
```

### 5.3 DDP 使用注意事项

```python
train_sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=per_gpu_batch)

for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)   # 保证每 epoch shuffle 不同
    for batch in dataloader:
        ...
```

---

## 六、分布式优化器

### 6.1 大 Batch 训练的挑战

DP 的等效 batch size = 单卡 batch × GPU 数。大 batch 带来的问题：

- **学习率需要调整**：线性缩放规则 $\text{lr}_{\text{large}} = \text{lr}_{\text{base}} \times \frac{B_{\text{large}}}{B_{\text{base}}}$
- **训练不稳定**：大 batch 初期容易 diverge
- **泛化性下降**：sharp minima 问题

### 6.2 Warmup 策略

线性 warmup 是大 batch 训练的标配：

$$\text{lr}(t) = \begin{cases} \text{lr}_{\text{max}} \times \frac{t}{T_{\text{warmup}}} & t \leq T_{\text{warmup}} \\ \text{lr}_{\text{max}} \times \text{decay}(t) & t > T_{\text{warmup}} \end{cases}$$

### 6.3 LAMB / LARS

针对大 batch 训练设计的优化器：

**LARS**（Layer-wise Adaptive Rate Scaling）：

$$\theta_{l}^{(t+1)} = \theta_{l}^{(t)} - \eta \cdot \frac{\|\theta_l\|}{\|g_l\| + \lambda \|\theta_l\|} \cdot (g_l + \lambda \theta_l)$$

- 对每层独立计算自适应学习率
- 解决不同层梯度尺度差异大的问题

**LAMB**（Layer-wise Adaptive Moments optimizer for Batch training）：

- 在 LARS 基础上结合 Adam 的一二阶矩估计
- 支持 batch size 高达 64K+ 的稳定训练

### 6.4 分布式 Adam

标准 Adam 在 DP 下的工作方式：

1. 每卡独立计算梯度 $g_k$
2. AllReduce 得到平均梯度 $\bar{g} = \frac{1}{N_d}\sum g_k$
3. 每卡用 $\bar{g}$ 独立更新 Adam 状态（$m, v$）和参数

> 注意：每卡维护**完整的**优化器状态（$m, v$），这是 ZeRO 要优化的重点（Day 2）。

---

## 七、混合精度训练（AMP）

### 7.1 为什么用混合精度

| 精度 | 位宽 | 范围 | 用途 |
|------|:----:|------|------|
| FP32 | 32 bit | $\pm 3.4 \times 10^{38}$ | 传统训练精度 |
| FP16 | 16 bit | $\pm 6.5 \times 10^{4}$ | 混合精度训练 |
| BF16 | 16 bit | $\pm 3.4 \times 10^{38}$ | 大模型标配（范围同 FP32） |

混合精度的好处：
- **显存减半**：FP16 参数 + 梯度只需 FP32 的一半
- **计算加速**：Tensor Core 对 FP16/BF16 的吞吐是 FP32 的 2~8 倍
- **通信减半**：AllReduce 传输量减半

### 7.2 AMP 工作流程

```
┌─────────────────────────────────────────────┐
│            混合精度训练流程                    │
│                                             │
│  FP32 Master Weights ──cast──→ FP16 Weights │
│         │                         │         │
│         │                    Forward (FP16)  │
│         │                         │         │
│         │                    Loss (FP32)     │
│         │                         │         │
│         │              Loss × scale factor   │
│         │                         │         │
│         │                   Backward (FP16)  │
│         │                         │         │
│         │              Gradients / scale      │
│         │                         │         │
│         ←── Update (FP32) ────────┘         │
└─────────────────────────────────────────────┘
```

### 7.3 Loss Scaling

FP16 的动态范围小，梯度容易 underflow（变为 0）。Loss Scaling 的策略：

$$\text{scaled\_loss} = \text{loss} \times S$$
$$\text{scaled\_grad} = \nabla(\text{scaled\_loss}) = S \times \nabla \text{loss}$$
$$\text{grad} = \text{scaled\_grad} / S$$

**Dynamic Loss Scaling**：
- 初始 $S = 2^{16}$
- 若梯度无 inf/nan，每 $N$ 步 $S \times 2$
- 若梯度出现 inf/nan，跳过更新，$S / 2$

### 7.4 BF16 vs FP16

| 特性 | FP16 | BF16 |
|------|------|------|
| 指数位 | 5 bit | 8 bit |
| 尾数位 | 10 bit | 7 bit |
| 动态范围 | 小（$10^{-8}$ ~ $10^{4}$） | 大（与 FP32 相同） |
| 精度 | 高（尾数更多） | 低（但够用） |
| Loss Scaling | 必需 | 通常不需要 |
| 硬件支持 | A100+ | A100+ |
| **推荐** | 旧代码 | ✅ 大模型训练首选 |

```python
# PyTorch AMP 使用示例
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 八、DP 的局限性与后续方向

### 8.1 显存冗余

DP 的核心问题：每张 GPU 都持有**完整的**模型参数 + 梯度 + 优化器状态。

以 7B 模型 + AdamW + FP16 为例，每卡显存开销：

| 组成 | 每卡大小 | 冗余程度（8 卡） |
|------|---------|:--------------:|
| 参数（FP16） | 14 GB | 8× |
| 梯度（FP16） | 14 GB | 8× |
| 优化器 $m$（FP32） | 28 GB | 8× |
| 优化器 $v$（FP32） | 28 GB | 8× |
| FP32 参数副本 | 28 GB | 8× |
| **合计** | **112 GB** | **8×** |

> 8 卡总共消耗 $112 \times 8 = 896$ GB，但实际有效数据只有 112 GB。**7/8 是冗余的！**

### 8.2 解决方向

| 方向 | 方法 | Day |
|------|------|-----|
| 消除显存冗余 | ZeRO 分片 | Day 2 |
| 切分模型权重 | Tensor Parallelism | Day 4 |
| 切分模型层 | Pipeline Parallelism | Day 4 |
| 切分长序列 | Context Parallelism | Day 4 |
| 切分 MoE 专家 | Expert Parallelism | Day 5-6 |

---

## 九、自检题

1. 训练一个 70B 参数的模型，使用 AdamW + FP16 混合精度，仅参数 + 梯度 + 优化器状态需要多少 GPU 显存？
2. Ring AllReduce 中每张 GPU 的通信量是多少？为什么说它与 GPU 数量近似无关？
3. 数据并行为什么在数学上严格等价于大 batch 训练？写出证明。
4. PyTorch DDP 中 Gradient Bucketing 的作用是什么？它如何实现计算与通信的重叠？
5. 解释 BF16 相比 FP16 的优势，为什么大模型训练优先选择 BF16？
6. LAMB 优化器相比标准 Adam 解决了什么问题？
7. DP 模式下 8 张 GPU 训练 7B 模型，有多少比例的显存是冗余的？
8. 简述 AllReduce、AllGather、ReduceScatter、AllToAll 四种通信原语的区别与适用场景。

---

## 十、产出要求

- [ ] 画出分布式训练五大并行维度全景图（DP / TP / PP / CP / EP），标注各维度切分什么、解决什么问题
- [ ] 手写 Ring AllReduce 的通信步骤（3 GPU 为例），推导每 GPU 通信量
- [ ] 实现一个简单的 PyTorch DDP 训练脚本
- [ ] 计算 7B / 13B / 70B 模型在 DP 模式下的单卡显存开销
- [ ] 撰写 1 页 AMP 混合精度训练笔记（FP16 vs BF16 对比表 + Loss Scaling 流程）
