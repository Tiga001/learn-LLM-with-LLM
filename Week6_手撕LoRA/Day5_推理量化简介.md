# Day 5：推理量化简介 — GPTQ / AWQ / GGUF

> **目标**：区分训练量化（QLoRA）和推理量化（GPTQ / AWQ / GGUF）的定位与目标；理解 GPTQ 的逐层后训练量化思路（基于 OBQ / OBS）；理解 AWQ 的激活感知权重量化策略；了解 GGUF 格式与 CPU 推理生态（llama.cpp）；建立"何时用什么量化方法"的清晰决策框架；为第 14 周的系统量化深化做好铺垫。

---

## 一、训练量化 vs 推理量化

### 1.1 两种量化的定位

```
训练量化 (QLoRA):
  目标: 降低训练时的显存需求
  时机: 微调过程中
  对象: 基座模型权重（冻结）→ 4-bit 存储
  精度: 训练计算时反量化为 BF16
  特点: 需要配合 LoRA 使用
  
  典型场景:
    "我只有一张 3090，想微调 LLaMA-7B"
    → QLoRA: NF4 基座 + BF16 LoRA

推理量化 (GPTQ / AWQ / GGUF):
  目标: 降低推理时的显存/计算需求
  时机: 训练完成后
  对象: 完整模型权重 → 低精度
  精度: 推理全程使用低精度
  特点: 不需要训练，直接压缩已有模型
  
  典型场景:
    "我想在消费级硬件上运行 LLaMA-70B"
    → GPTQ/AWQ 4-bit: 70B 模型 → ~35 GB
```

### 1.2 两类方法的对比

| 维度 | QLoRA（训练量化） | GPTQ / AWQ（推理量化） |
|------|-----------------|---------------------|
| 目标 | 降低训练显存 | 降低推理显存/延迟 |
| 时机 | 训练时 | 训练后 |
| 是否需要训练数据 | 需要微调数据 | 需要少量校准数据 |
| 权重参与梯度更新？ | 否（基座冻结）| 否（量化后固定）|
| 典型精度 | NF4 + BF16 LoRA | INT4 / FP4 |
| 典型框架 | bitsandbytes + PEFT | AutoGPTQ / vLLM / llama.cpp |
| 代表方法 | QLoRA | GPTQ, AWQ, GGUF |

---

## 二、GPTQ：后训练量化

### 2.1 论文基本信息

- **标题**：*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*
- **作者**：Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
- **时间**：2023 年 1 月 (ICLR 2023)
- **核心贡献**：基于 OBQ（Optimal Brain Quantization）的高效逐层量化方法，可在单张 GPU 上 4 小时内量化 175B 模型

### 2.2 核心思想

GPTQ 的目标：找到量化权重 $\hat{W}$ 使得量化误差最小：

$$\hat{W} = \arg\min_{\hat{W}} \|WX - \hat{W}X\|_F^2$$

其中 $X$ 是校准数据的激活值。

**关键**：不是简单地逐元素量化（round-to-nearest），而是**考虑量化一个权重对其他权重的影响**。

### 2.3 OBS / OBQ 思想

GPTQ 基于 OBS（Optimal Brain Surgeon）的框架：

```
Round-to-Nearest (RTN) — 最简单的量化:
  对每个权重独立四舍五入 → 简单但误差大
  
OBQ — 考虑权重间相关性:
  量化一个权重 w_i 会引入误差 δ_i
  通过调整未量化的权重来补偿 δ_i
  → 最小化整体输出误差
  
GPTQ — OBQ 的高效版本:
  OBQ 的复杂度是 O(d³)，对大模型不可行
  GPTQ 通过巧妙的矩阵分解将复杂度降低
  → 在单 GPU 上几小时即可量化 175B 模型
```

### 2.4 GPTQ 的逐列量化过程

```
GPTQ 算法（简化版）:

输入: 权重矩阵 W ∈ R^{m×n}, 校准数据的 Hessian H = XX^T
输出: 量化后的权重 Ŵ

for i = 1 to n (逐列):
    1. 量化第 i 列: ŵ_i = quantize(w_i)
    2. 计算量化误差: δ_i = w_i - ŵ_i
    3. 用 Hessian 信息更新未量化的列:
       W[:, i+1:] += δ_i × H[i, i+1:] / H[i, i]
       → 补偿量化 w_i 引入的误差

关键: 每量化一列，都调整剩余列来补偿
→ 误差在列间传播和吸收，而不是简单累加
```

### 2.5 GPTQ 实验结果

| 模型 | 方法 | 位数 | WikiText-2 PPL | 模型大小 |
|------|------|------|---------------|---------|
| LLaMA-7B | FP16 | 16 | 5.68 | 13 GB |
| LLaMA-7B | RTN | 4 | 6.29 | 3.5 GB |
| LLaMA-7B | **GPTQ** | **4** | **5.85** | **3.5 GB** |
| LLaMA-13B | FP16 | 16 | 5.09 | 26 GB |
| LLaMA-13B | **GPTQ** | **4** | **5.20** | **6.5 GB** |
| LLaMA-30B | FP16 | 16 | 4.10 | 60 GB |
| LLaMA-30B | **GPTQ** | **4** | **4.20** | **15 GB** |

**关键发现**：
- GPTQ 4-bit 的 PPL 仅比 FP16 高 0.1~0.2
- 显存减少 4×
- 远优于简单的 Round-to-Nearest (RTN) 量化

### 2.6 GPTQ 的使用

```python
# 使用 AutoGPTQ 量化模型
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,  # 每 128 个权重一组
    desc_act=True,   # 按激活值排序列顺序
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config,
)

# 用校准数据量化
model.quantize(calibration_dataset)
model.save_quantized("llama2-7b-gptq-4bit")

# 加载量化模型进行推理
model = AutoGPTQForCausalLM.from_quantized(
    "llama2-7b-gptq-4bit",
    device="cuda:0",
)
```

---

## 三、AWQ：激活感知权重量化

### 3.1 论文基本信息

- **标题**：*AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*
- **作者**：Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Daquanta Dettmers, Song Han
- **机构**：MIT
- **时间**：2024 年 (MLSys 2024)

### 3.2 核心洞察

AWQ 的关键观察：**不是所有权重同等重要**。

```
权重重要性分析:

传统思路: 所有权重等同对待 → 均匀量化
AWQ 思路: 找到"重要"权重 → 保护它们 → 更好的量化效果

如何判断权重重要性？
  → 看它乘以的激活值有多大！
  
  权重 × 激活值 = 输出
  
  如果某个权重对应的激活值平均很大
  → 这个权重的量化误差会被放大
  → 这个权重更"重要"
```

### 3.3 AWQ 方法

AWQ 的做法是**对重要通道进行缩放**（scale up），减少量化误差：

$$Q(\mathbf{w} \cdot \mathbf{s}) \cdot (\mathbf{x} / \mathbf{s})$$

其中 $\mathbf{s}$ 是根据激活值统计量确定的缩放因子。

```
AWQ 的直觉:

假设某个通道 c 的激活值平均 = 100
  权重 w_c = 0.01
  输出 = 100 × 0.01 = 1.0
  
  量化 w_c: 0.01 → 0.00 (被量化到 0!)
  量化后输出 = 100 × 0.00 = 0.0
  误差 = 1.0  → 非常大！

AWQ 的做法:
  缩放: w_c' = w_c × s = 0.01 × 10 = 0.1
  量化: 0.1 → 0.1 (更容易被量化保留)
  推理: 0.1 × (100/10) = 0.1 × 10 = 1.0
  误差 ≈ 0  → 显著减少！

本质:
  对重要通道放大权重 → 量化精度更高
  对激活值缩小 → 数学等价性保持
  → 关键权重得到更好的量化保护
```

### 3.4 AWQ 的搜索策略

最优缩放因子 $\mathbf{s}$ 的搜索：

$$\mathbf{s}^* = \arg\min_{\mathbf{s}} \|Q(W \cdot \text{diag}(\mathbf{s})) \cdot \text{diag}(\mathbf{s})^{-1} X - WX\|_F^2$$

AWQ 使用一个简单的启发式：

$$s_c = \left(\frac{\bar{|x_c|}}{\max(\bar{|x_c|})}\right)^\alpha$$

其中 $\bar{|x_c|}$ 是通道 $c$ 的平均激活值绝对值，$\alpha$ 是搜索的超参数（通常在 $[0, 1]$ 的网格上搜索）。

### 3.5 AWQ vs GPTQ 对比

| 维度 | GPTQ | AWQ |
|------|------|-----|
| 核心方法 | 逐列量化 + 误差补偿 | 激活感知缩放 |
| 量化速度 | 中等 | **快**（无需逐列迭代）|
| 推理速度 | 取决于后端 | 有专门优化的推理核 |
| PPL 精度 | 优秀 | 优秀（略优） |
| 硬件兼容 | CUDA | CUDA + 专用核 |
| 校准数据 | 需要（~128 样本）| 需要（~128 样本）|
| 主要优势 | 成熟、生态好 | 更快、推理友好 |

### 3.6 AWQ 实验结果

| 模型 | 方法 | WikiText-2 PPL | 量化时间 |
|------|------|---------------|---------|
| LLaMA-2-7B | FP16 | 5.47 | — |
| LLaMA-2-7B | GPTQ-4bit | 5.63 | ~10 min |
| LLaMA-2-7B | **AWQ-4bit** | **5.60** | **~5 min** |
| LLaMA-2-70B | FP16 | 3.32 | — |
| LLaMA-2-70B | GPTQ-4bit | 3.41 | ~4 hr |
| LLaMA-2-70B | **AWQ-4bit** | **3.39** | **~30 min** |

---

## 四、GGUF：CPU 推理生态

### 4.1 背景

- **GGUF**（GPT-Generated Unified Format）是 llama.cpp 项目定义的模型格式
- **llama.cpp**：纯 C/C++ 实现的 LLM 推理引擎，支持 CPU 推理
- 核心目标：让任何人在**没有 GPU** 的情况下都能运行 LLM

### 4.2 GGUF 的量化方案

GGUF 定义了多种量化级别：

| 量化类型 | 位数 | 方法 | PPL 损失 | 大小 (7B) |
|---------|------|------|---------|---------|
| Q2_K | 2-3 | K-quant 混合 | 较大 | ~2.8 GB |
| Q3_K_M | 3-4 | K-quant medium | 中等 | ~3.3 GB |
| Q4_K_M | 4-5 | K-quant medium | 较小 | ~4.1 GB |
| Q5_K_M | 5-6 | K-quant medium | 很小 | ~4.8 GB |
| Q6_K | 6 | K-quant | 极小 | ~5.5 GB |
| Q8_0 | 8 | 均匀 | 几乎无 | ~7.2 GB |
| F16 | 16 | 无量化 | 无 | ~13 GB |

```
K-quant (K-量化):
  GGUF 的核心量化方法
  "K" 代表 "K-means inspired"
  
  不同层使用不同的量化位数:
    重要的层 (如 Attention) → 使用更高精度
    不重要的层 (如某些 FFN) → 使用更低精度
    
  Q4_K_M 的 "M" 代表 medium 混合策略:
    部分层用 4-bit，部分层用 5-bit
    平均约 4.5 bit/参数
```

### 4.3 GGUF 的使用

```bash
# 下载 GGUF 模型
# HuggingFace 上有大量社区量化的 GGUF 模型
# 例如: TheBloke/Llama-2-7B-GGUF

# 使用 llama.cpp 推理
./main -m llama-2-7b.Q4_K_M.gguf \
       -p "What is machine learning?" \
       --n-predict 128 \
       --threads 8

# 使用 Python 绑定 (llama-cpp-python)
from llama_cpp import Llama

llm = Llama(
    model_path="llama-2-7b.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
)

output = llm(
    "What is machine learning?",
    max_tokens=128,
    temperature=0.7,
)
print(output["choices"][0]["text"])
```

### 4.4 GGUF 的优缺点

| 优势 | 劣势 |
|------|------|
| CPU 推理，无需 GPU | 推理速度慢于 GPU |
| 极低入门门槛 | 不支持训练/微调 |
| 多种量化级别可选 | 高级量化方法精度不如 GPTQ/AWQ |
| 活跃的开源社区 | 只支持推理 |
| 跨平台（Mac/Windows/Linux） | — |

---

## 五、量化方法综合对比

### 5.1 方法定位

```
量化方法选择决策树:

你要做什么？
  │
  ├─ 微调模型
  │    │
  │    └─ 显存不够？
  │         ├─ 是 → QLoRA (NF4 + LoRA)
  │         └─ 否 → LoRA (FP16) 或 Full FT
  │
  └─ 部署推理
       │
       ├─ 有 GPU？
       │    │
       │    ├─ 是 → GPTQ 或 AWQ
       │    │         ├─ 需要 vLLM 兼容 → AWQ
       │    │         └─ 成熟方案 → GPTQ
       │    │
       │    └─ 否 (CPU) → GGUF + llama.cpp
       │
       └─ 精度要求？
            ├─ 高 → Q8_0 / INT8
            ├─ 中 → Q4_K_M / INT4
            └─ 低 → Q2_K / Q3_K_M
```

### 5.2 全面对比表

| 维度 | QLoRA | GPTQ | AWQ | GGUF |
|------|-------|------|-----|------|
| **用途** | 训练（微调） | 推理 | 推理 | 推理（CPU） |
| **量化对象** | 基座模型（冻结） | 完整模型 | 完整模型 | 完整模型 |
| **量化方法** | NF4（信息论最优） | 逐层 OBQ | 激活感知缩放 | K-quant 混合 |
| **精度损失** | 微（训练补偿） | 小 | 小 | 取决于量化级别 |
| **典型位数** | 4-bit (NF4) | 4-bit | 4-bit | 2~8 bit |
| **需要校准数据** | 需要微调数据 | 128 样本 | 128 样本 | 否 |
| **量化时间** | — | 中等 | 快 | 快 |
| **推理速度** | — | GPU 快 | GPU 最快 | CPU 中等 |
| **硬件要求** | GPU (训练) | GPU (推理) | GPU (推理) | CPU 即可 |
| **生态** | PEFT / TRL | AutoGPTQ | vLLM / TGI | llama.cpp |
| **代表库** | bitsandbytes | auto-gptq | autoawq | llama.cpp |

### 5.3 精度对比（LLaMA-2-7B, WikiText-2 PPL）

| 方法 | PPL | 相对 FP16 |
|------|-----|---------|
| FP16（基线） | 5.47 | 0 |
| INT8 RTN | 5.48 | +0.01 |
| GPTQ 4-bit | 5.63 | +0.16 |
| AWQ 4-bit | 5.60 | +0.13 |
| GGUF Q4_K_M | ~5.7 | ~+0.23 |
| GGUF Q2_K | ~7.5 | ~+2.0 |

---

## 六、推理量化与 vLLM / TGI 的集成

### 6.1 vLLM

vLLM 是目前最流行的 LLM 推理引擎之一，原生支持 AWQ 和 GPTQ 量化模型：

```python
from vllm import LLM, SamplingParams

# 加载 AWQ 量化模型
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="half",
)

params = SamplingParams(temperature=0.7, max_tokens=128)
outputs = llm.generate(["What is machine learning?"], params)
```

### 6.2 部署方案选择

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 企业级 GPU 推理 | vLLM + AWQ | 高吞吐、低延迟 |
| 研究/小规模 GPU | AutoGPTQ | 成熟、灵活 |
| 边缘设备 / CPU | llama.cpp + GGUF | 无需 GPU |
| API 服务 | vLLM + AWQ/GPTQ | 支持批处理和流式输出 |
| 本地桌面应用 | Ollama + GGUF | 一键部署 |

---

## 七、自检题

1. **训练量化和推理量化的核心区别是什么？** 各举一个代表方法。
2. **GPTQ 的核心思想是什么？** 为什么逐列量化 + 误差补偿优于 Round-to-Nearest？
3. **AWQ 如何判断权重的重要性？** 激活感知缩放的数学原理是什么？
4. **GPTQ 和 AWQ 哪个更快？精度哪个更好？** 各适用于什么场景？
5. **GGUF 的目标用户是谁？** 它的 K-quant 是什么意思？
6. **Q4_K_M 中的 "4"、"K"、"M" 分别代表什么？**
7. **如果你只有一台 MacBook，想运行 LLaMA-2-13B，应该选什么方案？**
8. **如果你要搭建一个企业级 LLM 推理服务，应该选什么方案？**
9. **量化模型可以继续用 LoRA 微调吗？** QLoRA 和 GPTQ+LoRA 有什么区别？
10. **INT4 RTN、GPTQ-4bit、AWQ-4bit、NF4 四种 4-bit 方案，精度从高到低排序。**

---

## 八、产出要求

- [ ] 撰写训练量化 vs 推理量化的对比表
- [ ] 用简洁语言解释 GPTQ 的逐列量化思想（无需推导细节）
- [ ] 用简洁语言解释 AWQ 的激活感知缩放思想
- [ ] 画出量化方法选择决策树（训练 vs 推理 × GPU vs CPU）
- [ ] 对比 GPTQ / AWQ / GGUF 的精度、速度、适用场景
- [ ] 了解 vLLM 和 llama.cpp 的基本使用方式
- [ ] 为第 14 周的系统量化深化做好知识铺垫
