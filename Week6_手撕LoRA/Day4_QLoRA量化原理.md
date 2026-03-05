# Day 4：QLoRA 量化原理 — NF4 / 双重量化 / 分页优化器

> **目标**：深入理解 QLoRA 的三大核心创新——NF4（4-bit NormalFloat）量化的信息论最优性、双重量化（Double Quantization）的显存节省机制、分页优化器（Paged Optimizers）的统一内存管理；掌握量化的数学基础（均匀量化、仿射量化、分位数量化）；理解 QLoRA 训练流程中前向/反向传播的精度混合策略；为 Day 6 的 QLoRA 实践打下理论基础。

---

## 一、量化基础

### 1.1 什么是量化？

量化（Quantization）是将高精度数值（如 FP32 / FP16）映射到低精度数值（如 INT8 / INT4）的过程。

$$Q(x) = \text{round}\left(\frac{x}{s}\right) + z$$

其中 $s$ 是缩放因子（scale），$z$ 是零点（zero-point）。

```
为什么要量化？

FP32 → FP16:  显存减半，速度提升 ~2×
FP16 → INT8:  显存减半，速度提升 ~2×（需要硬件支持）
FP16 → INT4:  显存减至 1/4，但精度损失更大

LLaMA-7B 的显存:
  FP32: ~26 GB
  FP16: ~13 GB
  INT8: ~6.5 GB
  INT4: ~3.5 GB  ← QLoRA 的基座模型精度
```

### 1.2 量化类型

| 类型 | 时机 | 代表方法 | 说明 |
|------|------|---------|------|
| **训练后量化 (PTQ)** | 训练完成后 | GPTQ, AWQ | 不需要训练数据 |
| **量化感知训练 (QAT)** | 训练过程中 | — | 训练时模拟量化误差 |
| **训练时量化 + 微调** | 微调过程中 | **QLoRA** | 量化基座 + LoRA 微调 |

### 1.3 数值格式速览

| 格式 | 位数 | 范围 | 精度 | 用途 |
|------|------|------|------|------|
| FP32 | 32 | $\pm 3.4 \times 10^{38}$ | 7 位有效数字 | 优化器状态 |
| BF16 | 16 | $\pm 3.4 \times 10^{38}$ | 3 位有效数字 | 训练计算 |
| FP16 | 16 | $\pm 6.5 \times 10^{4}$ | 4 位有效数字 | 推理 / 训练 |
| FP8 (E4M3) | 8 | $\pm 448$ | 3 位有效数字 | 量化缩放因子 |
| INT8 | 8 | $[-128, 127]$ | 整数 | 推理量化 |
| INT4 | 4 | $[-8, 7]$ | 整数 | 激进量化 |
| **NF4** | 4 | $[-1, 1]$ | 非均匀 | **QLoRA** |

---

## 二、从均匀量化到 NF4

### 2.1 均匀量化（Uniform Quantization）

最简单的量化方式——将数值范围均匀分成 $2^b$ 个区间：

$$Q_{\text{uniform}}(x) = \text{round}\left(\frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (2^b - 1)\right)$$

反量化：

$$\hat{x} = Q \cdot \frac{x_{\max} - x_{\min}}{2^b - 1} + x_{\min}$$

```
INT4 均匀量化 (16 个量化点):

量化点分布:
  -8  -7  -6  -5  -4  -3  -2  -1   0   1   2   3   4   5   6   7
   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑
  等间距分布

问题: 神经网络权重的分布呈钟形（近似正态分布）
  权重集中在 0 附近 → 0 附近需要更高分辨率
  INT4 的等间距分布 → 0 附近分辨率不够
  → 量化误差在权重密集区域更大
```

### 2.2 absmax 量化

实践中更常用的对称量化方式：

$$Q_{\text{absmax}}(x) = \text{round}\left(\frac{x}{\max(|x|)} \cdot (2^{b-1} - 1)\right)$$

$$s = \frac{\max(|x|)}{2^{b-1} - 1}$$

```
absmax INT4 量化示例:
  权重: [0.3, -0.7, 1.2, -0.1, 0.5, -1.2]
  absmax = 1.2
  scale = 1.2 / 7 ≈ 0.171
  
  量化: round([0.3/0.171, -0.7/0.171, ...]) = [2, -4, 7, -1, 3, -7]
  反量化: [2*0.171, -4*0.171, ...] = [0.343, -0.686, 1.2, -0.171, 0.514, -1.2]
  
  误差: [0.043, 0.014, 0.0, 0.071, 0.014, 0.0]
```

### 2.3 分块量化（Block-wise Quantization）

全局共享一个缩放因子会导致某些区域精度损失过大。分块量化将权重分成小块，每块独立计算缩放因子：

$$\text{Block size} = 64 \quad (\text{QLoRA 的默认设置})$$

```
分块量化:

原始权重 (4096 个):
  [w₁, w₂, ..., w₆₄ | w₆₅, ..., w₁₂₈ | ... | w₄₀₃₃, ..., w₄₀₉₆]
    Block 1 (scale₁)    Block 2 (scale₂)       Block 64 (scale₆₄)

每个 block 独立量化:
  block_i: scale_i = max(|w|) in block_i
  quantized_w = round(w / scale_i × 7)

→ 每个 block 的量化误差只受 block 内部的数值范围影响
→ 精度显著优于全局量化
```

### 2.4 NF4：信息论最优量化

**核心洞察**：预训练模型的权重近似服从正态分布 $\mathcal{N}(0, \sigma^2)$。

如果我们知道数据的分布，可以设计**信息论最优**的量化方案——让每个量化区间包含等量的数据点。

**NF4 的构造步骤**：

**Step 1**：计算标准正态分布的 $2^b + 1$ 个等概率分位数。

对于 $b=4$（16 个量化点），需要将 $\mathcal{N}(0,1)$ 的概率质量等分为 $2^b = 16$ 个区间：

$$q_i = \Phi^{-1}\left(\frac{i}{2^b}\right), \quad i = 0, 1, \ldots, 2^b$$

其中 $\Phi^{-1}$ 是标准正态分布的逆 CDF（分位数函数）。

**Step 2**：取相邻分位数的中点作为量化值。

$$c_i = \frac{q_i + q_{i+1}}{2}, \quad i = 0, 1, \ldots, 2^b - 1$$

**Step 3**：归一化到 $[-1, 1]$。

$$c_i' = \frac{c_i}{\max(|c_i|)}$$

**Step 4**：确保包含精确的 0（重要的量化点）。

实际的 NF4 量化点：

```python
NF4_QUANTIZATION_POINTS = [
    -1.0000, -0.6962, -0.5251, -0.3949, -0.2844,
    -0.1848, -0.0911,  0.0000,
     0.0796,  0.1609,  0.2461,  0.3379,  0.4407,
     0.5626,  0.7230,  1.0000
]
```

```
NF4 vs INT4 量化点可视化:

INT4 (均匀):
 -1.0  -0.87 -0.73 -0.60 -0.47 -0.33 -0.20 -0.07  0.07  0.20  0.33  0.47  0.60  0.73  0.87  1.0
  |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
  等间距分布

NF4 (正态分位数):
 -1.0 -0.70 -0.53 -0.39 -0.28 -0.18 -0.09  0.00  0.08  0.16  0.25  0.34  0.44  0.56  0.72  1.0
  |    |     |    |    |   |   ||    |  |   |    |    |     |     |     |
  0 附近更密集 ← 因为权重集中在这里

正态分布下的量化误差:
  INT4: 每个区间包含的权重数量不均匀 → 密集区域误差大
  NF4:  每个区间包含相同数量的权重 → 误差均匀分布 → 信息论最优
```

**NF4 的信息论最优性**：

对于已知分布的数据，当每个量化区间包含等量概率质量时，量化的均方误差（MSE）最小化。NF4 正是这样构造的——在正态分布假设下，它是**最优的 4-bit 量化方案**。

### 2.5 NF4 量化与反量化过程

```
NF4 量化过程:

1. 将权重分成 block (每 64 个一组)
2. 每个 block 计算 absmax:  s = max(|w|)
3. 归一化: w_norm = w / s  (映射到 [-1, 1])
4. 找到最近的 NF4 量化点: q = argmin |w_norm - NF4[i]|
5. 存储: (4-bit 量化索引 q, FP32 缩放因子 s)

NF4 反量化过程:

1. 读取 4-bit 索引 q → 查表得到 NF4 量化值 c
2. 读取缩放因子 s
3. 反量化: w_hat = c × s
```

---

## 三、双重量化（Double Quantization）

### 3.1 问题：缩放因子的存储开销

4-bit 量化需要为每个 block 存储一个 FP32 缩放因子：

$$\text{每个权重的平均位数} = 4 + \frac{32}{\text{block\_size}} = 4 + \frac{32}{64} = 4.5 \text{ bit}$$

对于 LLaMA-65B（65B 参数）：

$$\text{缩放因子总显存} = \frac{65 \times 10^9}{64} \times 4 \text{ bytes} \approx 4.07 \text{ GB}$$

### 3.2 双重量化的思想

**将缩放因子本身也量化！**

```
第一层量化:
  权重 → NF4 (4-bit)
  每 64 个权重 → 1 个 FP32 缩放因子

第二层量化:
  FP32 缩放因子 → FP8 (8-bit)
  每 256 个 FP32 缩放因子 → 1 个 FP32 "超级缩放因子"
```

### 3.3 数学计算

**不使用双重量化**（单重量化）：

$$\text{位/参数} = 4 + \frac{32}{64} = 4.5 \text{ bit}$$

**使用双重量化**：

$$\text{位/参数} = 4 + \frac{8}{64} + \frac{32}{64 \times 256} \approx 4.127 \text{ bit}$$

分解：
- 权重本身：4 bit
- 第一层缩放因子（FP8）：$\frac{8}{64} = 0.125$ bit/参数
- 第二层缩放因子（FP32）：$\frac{32}{64 \times 256} \approx 0.002$ bit/参数

**显存节省**：

| 模型 | 单重量化 | 双重量化 | 节省 |
|------|---------|---------|------|
| LLaMA-7B | 3.77 GB | 3.47 GB | 0.30 GB |
| LLaMA-13B | 7.29 GB | 6.72 GB | 0.57 GB |
| LLaMA-33B | 18.4 GB | 17.0 GB | 1.4 GB |
| LLaMA-65B | 36.5 GB | 33.7 GB | **2.8 GB** |

对于 65B 模型，双重量化节省的 2.8 GB 可能是"能放下"和"放不下"的区别。

### 3.4 双重量化的精度影响

```
量化链路:
  权重 (FP16) → NF4 量化 → 引入误差 ε₁
  缩放因子 (FP32) → FP8 量化 → 引入误差 ε₂

总量化误差:
  ε_total ≈ ε₁ + ε₂ × scale_factor

关键: ε₂ 的影响被 "稀释" 了
  每个 FP8 缩放因子影响 64 个权重
  但 FP8 精度（3位有效数字）足以保持缩放因子的准确性
  → 双重量化引入的额外误差可忽略不计

实验验证 (QLoRA 论文):
  NF4 单重量化: MMLU = 63.9
  NF4 双重量化: MMLU = 63.9  → 几乎无差异
```

---

## 四、分页优化器（Paged Optimizers）

### 4.1 GPU 显存的峰值问题

训练过程中，显存需求不是恒定的：

```
显存使用随时间变化:

前向传播:
  ████████████  模型参数 (固定)
  ██            LoRA 参数 + 优化器 (固定)
  ████          激活值 (逐层增长) ← 显存逐渐增加

反向传播:
  ████████████  模型参数 (固定)
  ██            LoRA 参数 + 优化器 (固定)
  ████████      激活值 + 梯度 ← 显存峰值！
  
优化器更新:
  ████████████  模型参数 (固定)
  ██████        LoRA 参数 + 优化器 (FP32 动量) ← 优化器状态膨胀

问题: 在某些时刻（长序列、大 batch），显存需求出现尖峰
  → 即使平均显存够用，峰值也可能导致 OOM
```

### 4.2 NVIDIA 统一内存（Unified Memory）

CUDA 的统一内存机制允许 GPU 和 CPU 共享虚拟地址空间：

```
传统 GPU 内存管理:
  GPU 显存:  [模型 | 激活值 | 优化器 | ...]  → 满了就 OOM
  CPU 内存:  [...]                            → 未利用

统一内存:
  虚拟地址空间:
    [模型 | 激活值 | 优化器状态 ...]
         ↕           ↕
    GPU 物理显存  CPU 物理内存

  当 GPU 显存不够:
    不常用的页面自动迁移到 CPU 内存
    需要时自动迁回 GPU
    → 类似操作系统的虚拟内存分页
```

### 4.3 分页优化器的工作方式

```
QLoRA 的分页 AdamW:

正常情况 (显存充足):
  所有优化器状态在 GPU 上 → 训练正常

显存不足时 (长序列 / 大 batch):
  1. 检测到 GPU 显存不足
  2. 将部分优化器状态 "换出" 到 CPU 内存
  3. 模型前向/反向传播正常进行
  4. 优化器更新时，按需将状态从 CPU "换入" GPU
  5. 更新完成后，状态可以留在 CPU 或换回 GPU

关键性能优化:
  → 在 GPU 计算空闲时 (等待数据加载/前向传播) 进行 CPU↔GPU 传输
  → 隐藏传输延迟，几乎不影响训练速度
```

### 4.4 实现细节

```python
# bitsandbytes 库中分页优化器的使用
import bitsandbytes as bnb

# 标准 AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# 分页 AdamW (QLoRA 推荐)
optimizer = bnb.optim.PagedAdamW(model.parameters(), lr=2e-4)

# 分页 AdamW 8-bit (更极致的显存节省)
optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=2e-4)
```

### 4.5 8-bit 优化器

bitsandbytes 还提供 8-bit 优化器，进一步压缩优化器状态：

$$\text{标准 AdamW}: 2 \times |\phi| \times 4 \text{ bytes (FP32 动量)}$$
$$\text{8-bit AdamW}: 2 \times |\phi| \times 1 \text{ byte (INT8 动量)}$$

```
优化器状态显存 (LoRA r=16, 全部 Linear, 25.2M 参数):
  FP32 AdamW:    2 × 25.2M × 4 bytes = 201 MB
  8-bit AdamW:   2 × 25.2M × 1 byte  = 50 MB
  → 节省 75%

对 LoRA 微调来说:
  优化器状态本身就很小 (几百 MB)
  8-bit 优化器的收益相对有限
  分页优化器的主要价值在于避免 OOM，而非绝对显存节省
```

---

## 五、QLoRA 完整训练流程详解

### 5.1 精度混合策略

```
QLoRA 中各组件的精度:

组件                   存储精度      计算精度
──────────────────────────────────────────
基座模型权重 W₀         NF4 (4-bit)   → BF16 (反量化后计算)
LoRA 矩阵 A, B         BF16          BF16
优化器状态 (m, v)       FP32          FP32
梯度                    BF16          BF16
激活值                  BF16          BF16
缩放因子 (第1层)        FP8           → FP32 (反量化)
缩放因子 (第2层)        FP32          FP32
```

### 5.2 前向传播

```
前向传播的计算流程:

x (BF16)
    │
    ├─────────────────────────────────┐
    │                                 │
    ▼                                 ▼
NF4 反量化:                        LoRA 分支:
  W₀_nf4 → W₀_bf16               A (BF16): r × d_in
  (查表 + 乘以缩放因子)              │
    │                              Dropout
    ▼                                │
  W₀_bf16 @ x                       ▼
  = base_output (BF16)            B (BF16): d_out × r
    │                                │
    │                              × (α/r)
    │                                │
    └──────────── + ─────────────────┘
                  │
                  ▼
            h = base_output + lora_output (BF16)

注意:
  W₀ 在前向传播中需要反量化为 BF16
  这是 QLoRA 比 LoRA 慢的原因
  但反量化操作可以高度并行 → 开销可控
```

### 5.3 反向传播

```
反向传播的关键点:

1. 梯度不通过 W₀ 传播 (W₀ 冻结)
   → 不需要对 NF4 权重计算梯度
   → 但需要 W₀ 参与前向传播 (反量化)

2. 梯度只通过 LoRA 参数 (A, B) 传播
   ∂L/∂B = (α/r) × ∂L/∂h × (Ax)^T    (BF16)
   ∂L/∂A = (α/r) × B^T × ∂L/∂h × x^T  (BF16)

3. 激活值检查点 (Gradient Checkpointing)
   为节省显存，可以不保存中间激活值
   反向传播时重新计算 → 用时间换空间
   QLoRA 论文建议: 始终开启 gradient checkpointing
```

### 5.4 优化器更新

```
优化器更新:

1. 收集 LoRA 参数的 BF16 梯度
2. 将梯度转为 FP32 (避免精度损失)
3. AdamW 更新 (FP32):
   m = β₁ × m + (1-β₁) × g          (一阶动量)
   v = β₂ × v + (1-β₂) × g²         (二阶动量)
   θ = θ - lr × m / (√v + ε) - wd×θ  (参数更新)
4. 将更新后的 LoRA 参数转回 BF16

使用分页优化器时:
  如果步骤 3 的优化器状态在 CPU 上 → 先换入 GPU
  更新完成后 → 可以换出到 CPU
```

---

## 六、QLoRA 实验验证

### 6.1 NF4 vs 其他 4-bit 格式

| 量化格式 | LLaMA-7B MMLU | LLaMA-13B MMLU | 说明 |
|---------|-------------|--------------|------|
| FP16 (基线) | 45.3 | 52.1 | 无量化 |
| INT4 | 43.8 | 50.5 | 均匀量化 |
| FP4 | 44.5 | 51.2 | 浮点量化 |
| **NF4** | **45.1** | **52.0** | 正态分位数量化 |
| NF4 + Double Quant | 45.0 | 51.9 | 几乎无损 |

### 6.2 QLoRA 的实际训练效果

QLoRA 论文训练了 Guanaco 模型家族：

| 模型 | 基座 | 微调方法 | 数据量 | GPU 显存 | Vicuna Score |
|------|------|---------|--------|---------|-------------|
| Vicuna-13B | LLaMA-13B | Full FT | 70K | 多卡 | 91.4% |
| **Guanaco-7B** | LLaMA-7B | QLoRA | 9K | **1×RTX 3090** | 85.1% |
| **Guanaco-13B** | LLaMA-13B | QLoRA | 9K | **1×RTX 3090** | 90.7% |
| **Guanaco-33B** | LLaMA-33B | QLoRA | 9K | **1×A100 40G** | 93.8% |
| **Guanaco-65B** | LLaMA-65B | QLoRA | 9K | **1×A100 48G** | **99.3%** |

**震撼结论**：
- 单卡 QLoRA 微调的 Guanaco-65B 达到了 ChatGPT 99.3% 的水平
- 仅用 9K 条数据 + 24 小时训练

---

## 七、bitsandbytes 库的使用

### 7.1 4-bit 量化加载模型

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# QLoRA 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                # 4-bit 量化加载
    bnb_4bit_quant_type="nf4",        # NF4 量化类型
    bnb_4bit_use_double_quant=True,   # 双重量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### 7.2 QLoRA 完整微调流程

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 准备量化模型用于 LoRA 训练
model = prepare_model_for_kbit_training(model)

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 25,165,824 || all params: 3,740,xxx,xxx || trainable%: 0.67%
```

---

## 八、自检题

1. **NF4 量化的核心思想是什么？** 为什么它比 INT4 更适合量化神经网络权重？
2. **手写 NF4 量化点的构造过程**。为什么要用正态分布的分位数？
3. **什么是分块量化？** 为什么 block_size=64 是一个好的选择？
4. **双重量化是什么？** 它能将平均位数从多少降到多少？对 65B 模型节省多少显存？
5. **QLoRA 中各组件（基座权重/LoRA/优化器/梯度）分别是什么精度？**
6. **QLoRA 前向传播时，NF4 权重如何参与计算？** 画出完整的计算流程图。
7. **分页优化器解决了什么问题？** 它的 "换入/换出" 策略是怎样的？
8. **为什么 QLoRA 的反向传播不需要对 NF4 权重计算梯度？**
9. **对比 NF4 / FP4 / INT4 在 MMLU 上的效果。** NF4 的优势有多大？
10. **Guanaco-65B 用了多少数据、多少GPU、多长时间？** 效果如何？

---

## 九、产出要求

- [ ] 推导 NF4 量化点的构造过程（从正态分布分位数出发）
- [ ] 计算双重量化的平均位数（$4 + 8/64 + 32/(64 \times 256) \approx 4.127$ bit）
- [ ] 画出 QLoRA 前向传播的完整精度混合流程图
- [ ] 对比 INT4、FP4、NF4 量化的信息论效率
- [ ] 计算 LLaMA-65B 在 QLoRA 下的显存需求
- [ ] 理解 bitsandbytes 库中 `BitsAndBytesConfig` 的各参数含义
- [ ] 解释 gradient checkpointing 在 QLoRA 中的作用
