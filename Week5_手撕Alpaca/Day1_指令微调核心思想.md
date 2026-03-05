# Day 1：指令微调核心思想 — 从「补全」到「遵循指令」

> **目标**：理解预训练与指令微调（SFT）的本质区别，掌握 SFT 数据格式的设计原理，从数学上理解为什么指令微调可以用极少量数据改变模型的行为模式，建立"预训练给知识，SFT 教行为"的核心直觉。

---

## 一、预训练 vs 指令微调：两种截然不同的范式

### 1.1 预训练：学习世界知识

回顾第 3-4 周，GPT / LLaMA 的预训练目标是 **Causal Language Modeling（因果语言模型）**：

$$\mathcal{L}_{\text{pretrain}} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1}; \theta)$$

模型在海量文本上学习"给定前文，预测下一个 token"。这个过程让模型学到了：

| 学到了什么 | 示例 |
|-----------|------|
| 语法规则 | 主谓宾结构、时态一致性 |
| 世界知识 | "巴黎是法国的首都" |
| 推理模式 | 因果关系、逻辑推导 |
| 代码能力 | 编程语言语法、常见模式 |
| 多语言能力 | 不同语言的表达方式 |

但预训练模型有一个根本问题——**它只会"续写"，不会"回答"**：

```
用户输入: "法国的首都是什么？"

预训练模型的行为（补全）:
  "法国的首都是什么？这是一个常见的地理问题。在小学地理课上..."
  → 它在"续写"一篇讨论这个问题的文章

期望的行为（遵循指令）:
  "法国的首都是巴黎。"
  → 直接回答问题
```

### 1.2 指令微调（SFT）：学习行为模式

**Supervised Fine-Tuning（SFT）** 的目标是教模型"如何回应用户的指令"：

$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T_{\text{output}}} \log P(y_t \mid \text{instruction}, \text{input}, y_1, \ldots, y_{t-1}; \theta)$$

注意关键区别：**SFT 的 loss 只计算在 output 部分**，不计算在 instruction 和 input 部分。

```
┌──────────────────────────────────────────────────────┐
│  预训练 Loss 计算:                                      │
│                                                        │
│  [The] [cat] [sat] [on] [the] [mat]                   │
│   ↓     ↓     ↓     ↓     ↓     ↓                    │
│  loss  loss  loss  loss  loss  loss   ← 每个 token 都算 │
│                                                        │
│  SFT Loss 计算:                                        │
│                                                        │
│  [Instruction: 翻译成英文] [Input: 猫坐在垫子上]         │
│   ↓     ↓     ↓     ↓      ↓    ↓    ↓    ↓          │
│  忽略  忽略  忽略  忽略   忽略  忽略  忽略  忽略          │
│                                                        │
│  [Output: The cat sat on the mat]                      │
│   ↓    ↓    ↓    ↓   ↓    ↓   ↓                      │
│  loss  loss  loss  loss loss loss loss  ← 只算输出部分   │
└──────────────────────────────────────────────────────┘
```

### 1.3 一个深刻的类比

```
预训练 ≈ 读遍了图书馆所有的书
  → 模型获得了丰富的知识
  → 但不知道怎么和人对话

指令微调 ≈ 上了几天"客服培训班"
  → 学会了"用户问什么就回答什么"的行为模式
  → 知识没有增加，但使用知识的方式改变了

这就是为什么 SFT 只需要很少的数据（52K ~ 100K 条）就能显著改变模型的行为。
```

---

## 二、SFT 数据格式详解

### 2.1 标准三元组格式

指令微调数据的核心格式是 **(Instruction, Input, Output)** 三元组：

```json
{
  "instruction": "将以下文本翻译成英文。",
  "input": "机器学习是人工智能的一个子领域。",
  "output": "Machine learning is a subfield of artificial intelligence."
}
```

| 字段 | 说明 | 是否必需 |
|------|------|---------|
| `instruction` | 任务描述，告诉模型要做什么 | 必需 |
| `input` | 任务的具体输入（可为空） | 可选 |
| `output` | 期望的输出 | 必需 |

### 2.2 有 input 和无 input 的区别

```json
// 有 input 的示例 — 任务需要处理特定输入
{
  "instruction": "对以下文本进行情感分析。",
  "input": "这部电影太棒了，我看了三遍！",
  "output": "正面情感。该评论表达了对电影的高度赞赏。"
}

// 无 input 的示例 — 任务本身就是完整的
{
  "instruction": "解释什么是梯度下降算法。",
  "input": "",
  "output": "梯度下降是一种一阶迭代优化算法，用于寻找函数的局部最小值。其核心思想是：沿着目标函数梯度的反方向更新参数，因为梯度指向函数值增长最快的方向，反方向就是下降最快的方向。..."
}
```

### 2.3 Prompt Template：从三元组到模型输入

SFT 的关键工程细节是 **Prompt Template**——将三元组转换为模型实际看到的文本格式。

**Stanford Alpaca 的 Prompt Template**：

```python
# 有 input 的模板
PROMPT_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# 无 input 的模板
PROMPT_WITHOUT_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
```

**为什么需要 Prompt Template？**

1. **统一格式**：让模型学会识别 `### Instruction:` 和 `### Response:` 等标记
2. **角色分离**：明确区分"用户说了什么"和"模型应该输出什么"
3. **推理一致性**：训练和推理时使用相同的模板

### 2.4 ChatML 格式（现代对话模型标准）

随着对话模型的发展，ChatML 格式逐渐成为主流：

```
<|im_start|>system
你是一个有帮助的 AI 助手。<|im_end|>
<|im_start|>user
法国的首都是什么？<|im_end|>
<|im_start|>assistant
法国的首都是巴黎。巴黎位于法国北部，是法国最大的城市和政治、经济、文化中心。<|im_end|>
```

| 格式 | 代表模型 | 特点 |
|------|---------|------|
| Alpaca 格式 | Stanford Alpaca, Vicuna | 简单、适合单轮指令 |
| ChatML | Qwen, Yi | 支持多轮对话、系统提示 |
| Llama-2-Chat | LLaMA-2-Chat | `[INST]...[/INST]` 标记 |
| Chatml-function | GPT-4 | 支持函数调用 |

### 2.5 Loss Mask 的实现

SFT 中最关键的工程细节是 **只在 output 部分计算 loss**：

```python
def create_sft_labels(input_ids, response_start_idx):
    """
    创建 SFT 训练的 labels。
    instruction + input 部分标记为 -100（忽略），
    output 部分保持原始 token ID。
    """
    labels = input_ids.clone()
    labels[:response_start_idx] = -100  # 忽略 instruction + input
    return labels

# 示例
# input_ids:  [Below, is, an, instruction, ..., ### Response:, The, capital, is, Paris, .]
# labels:     [-100, -100, -100, -100, ..., -100,          The, capital, is, Paris, .]
#              ←── 不计算 loss ──→                          ←── 计算 loss ──→
```

**为什么不在 instruction 部分计算 loss？**

1. **目标明确**：我们想让模型学会"给出好的回答"，而不是"重复用户的问题"
2. **避免混淆**：如果在 instruction 上也算 loss，模型会倾向于复述问题
3. **效率更高**：减少了梯度计算量，聚焦于关键部分

---

## 三、SFT 的数学直觉：为什么少量数据就够？

### 3.1 "Superficial Alignment Hypothesis"

2024 年的研究（LIMA: Less Is More for Alignment）提出了一个重要假说：

> **表面对齐假说**（Superficial Alignment Hypothesis）：模型在预训练中已经学到了几乎所有需要的知识和能力。指令微调的作用仅仅是学习与用户交互的**格式和风格**——这是一个非常浅层的学习。

```
预训练（1T+ tokens）:              指令微调（1K~100K 条）:
  知识 ✅                           知识 = 不变
  推理能力 ✅                       推理能力 = 不变
  语言理解 ✅                       语言理解 = 不变
  交互格式 ❌ ← 缺这个！            交互格式 ✅ ← 学会了！
```

### 3.2 LIMA 实验验证

LIMA（Less Is More for Alignment）使用 **仅 1000 条**高质量指令数据微调 LLaMA-65B：

| 方案 | 数据量 | 效果 |
|------|--------|------|
| Stanford Alpaca | 52K 条（GPT-3.5 生成） | 基础对话能力 |
| Vicuna | 70K 条（ShareGPT 数据） | 较好对话能力 |
| **LIMA** | **1000 条**（人工精选） | **匹敌 Alpaca/Vicuna** |

**关键洞察**：数据质量远比数据量重要。1000 条精心挑选的高质量数据，效果可以匹敌数万条自动生成的数据。

### 3.3 从优化视角理解

预训练模型的参数空间中已经存在"好的对话模式"，SFT 只是给模型一个"微推"：

$$\theta_{\text{SFT}} = \theta_{\text{pretrain}} + \Delta\theta_{\text{small}}$$

- $\|\Delta\theta_{\text{small}}\|$ 远小于 $\|\theta_{\text{pretrain}}\|$
- 这也是 LoRA 等参数高效微调方法的理论基础（第 6 周详解）
- 说明 SFT 本质上是在预训练模型的参数空间中做一个小幅调整

---

## 四、SFT 数据的多样性：任务类别全景

### 4.1 Alpaca 的任务分类

Stanford Alpaca 的 52K 指令数据覆盖了多种任务类型：

```
指令微调数据的典型任务分布:

┌────────────────────────────┐
│  开放式生成 (30%)            │  写文章、编故事、创意写作
├────────────────────────────┤
│  问答 (20%)                 │  知识问答、常识推理
├────────────────────────────┤
│  分类与分析 (15%)            │  情感分析、主题分类
├────────────────────────────┤
│  改写与摘要 (10%)            │  文本摘要、改写、翻译
├────────────────────────────┤
│  代码 (10%)                 │  代码生成、调试、解释
├────────────────────────────┤
│  推理 (8%)                  │  数学、逻辑推理
├────────────────────────────┤
│  信息提取 (7%)              │  实体识别、关系抽取
└────────────────────────────┘
```

### 4.2 数据多样性的重要性

```
差的数据集:
  100K 条全是翻译任务
  → 模型只会翻译，问别的就乱答

好的数据集:
  10K 条覆盖 50+ 种任务类型
  → 模型学会了"看指令做事"的通用能力
  → 甚至能处理训练集中没见过的任务类型！
```

---

## 五、InstructGPT：指令微调的先驱

### 5.1 InstructGPT 的核心贡献

InstructGPT（Ouyang et al., 2022）是指令微调领域的奠基之作，提出了三阶段训练范式：

```
阶段 1: SFT（Supervised Fine-Tuning）
  收集人类标注的 (prompt, response) 数据
  有监督微调 GPT-3
       │
       ▼
阶段 2: RM（Reward Model）
  收集人类偏好数据（对同一 prompt 的多个回答排序）
  训练奖励模型
       │
       ▼
阶段 3: RLHF（PPO 强化学习）
  用 PPO 算法优化模型，使其输出获得高奖励
       │
       ▼
  = ChatGPT 的前身
```

**本周聚焦阶段 1（SFT）**，阶段 2-3 将在第 9-11 周详细展开。

### 5.2 InstructGPT 的 SFT 数据

| 数据来源 | 数量 | 说明 |
|---------|------|------|
| 人工编写的 Prompt | ~13K | 标注员从头编写指令 |
| API 用户的 Prompt | ~30K | 用户实际提交给 API 的请求 |
| 总计 | ~13K 用于 SFT | 只使用了高质量子集 |

**InstructGPT 的关键发现**：
- 1.3B 参数的 InstructGPT **在人类评估中优于** 175B 的 GPT-3
- 指令微调可以让小模型"超越"未经微调的大模型
- 这个发现直接启发了 Alpaca 等后续工作

---

## 六、SFT 的工程实践要点

### 6.1 学习率设置

SFT 的学习率通常比预训练**低 1-2 个数量级**：

| 阶段 | 典型学习率 | 原因 |
|------|-----------|------|
| 预训练 | $1\times10^{-4}$ ~ $3\times10^{-4}$ | 需要大幅更新参数 |
| SFT | $1\times10^{-5}$ ~ $2\times10^{-5}$ | 微调，避免灾难性遗忘 |

### 6.2 训练轮数

SFT 通常只需要 **1~3 个 epoch**：

```
Epoch 过多的风险:
  → 过拟合到训练数据的特定表达方式
  → 丢失预训练阶段学到的通用能力（灾难性遗忘）
  → 模型回答变得"模板化"，缺乏多样性

推荐:
  → 高质量数据: 1-2 epochs
  → 中等质量数据: 2-3 epochs
  → 数据量极少时: 3-5 epochs（但要注意过拟合）
```

### 6.3 灾难性遗忘（Catastrophic Forgetting）

SFT 面临的核心挑战之一：

$$\text{过度微调} → \text{模型忘记预训练知识} → \text{性能退化}$$

缓解策略：
1. **低学习率**：减小参数更新幅度
2. **少 epoch**：避免过度适应微调数据
3. **混入预训练数据**：在 SFT 数据中混入一定比例的预训练数据
4. **参数高效微调**：只更新少量参数（Adapter / LoRA），保护预训练权重

### 6.4 数据配比

当 SFT 数据包含多种任务时，数据配比很重要：

```python
# 常见的数据混合策略
data_mix = {
    "general_instruction": 0.3,    # 通用指令
    "conversation": 0.2,           # 多轮对话
    "code": 0.15,                  # 代码相关
    "math": 0.1,                   # 数学推理
    "creative_writing": 0.1,       # 创意写作
    "safety": 0.05,                # 安全相关
    "pretrain_replay": 0.1,        # 预训练数据回放
}
```

---

## 七、指令微调的演进路线

```
2022.03  InstructGPT (OpenAI)
  │      13K 人工标注数据 + RLHF → "指令微调 + 对齐"范式的起点
  ▼
2022.12  ChatGPT (OpenAI)
  │      InstructGPT 的升级版 → 引爆全球 AI 热潮
  ▼
2023.02  LLaMA (Meta)
  │      开源预训练模型 → 开源生态的基础设施
  ▼
2023.03  Stanford Alpaca
  │      52K Self-Instruct 数据 + LLaMA SFT → 证明廉价 SFT 可行
  ▼
2023.03  Vicuna
  │      ShareGPT 对话数据 + LLaMA SFT → 更好的多轮对话
  ▼
2023.05  LIMA
  │      仅 1000 条数据 → 证明"数据质量 > 数据量"
  ▼
2023.07  LLaMA-2-Chat
  │      大规模 SFT + RLHF → 开源模型首次系统对齐
  ▼
2024+    更精细的数据工程
         合成数据、多阶段 SFT、DPO 替代 RLHF...
```

---

## 八、自检题

1. **预训练和 SFT 的优化目标有什么区别？** 为什么 SFT 只在 output 部分计算 loss？
2. **什么是"表面对齐假说"？** LIMA 的实验如何支持这个假说？
3. **解释 Prompt Template 的作用。** 为什么训练和推理必须使用相同的模板？
4. **SFT 的学习率为什么比预训练低？** 过高会有什么风险？
5. **什么是灾难性遗忘？** 列举 3 种缓解策略。
6. **InstructGPT 提出的三阶段训练范式是什么？** 本周聚焦哪个阶段？
7. **比较 Alpaca 格式和 ChatML 格式的异同。**

---

## 九、产出要求

- [ ] 画出"预训练 → SFT → RLHF"三阶段的流程图
- [ ] 手写 SFT 数据的 Loss Mask 实现代码
- [ ] 用自己的话解释"表面对齐假说"
- [ ] 准备 3 个不同任务类型的 SFT 数据样本（Alpaca 格式）
- [ ] 比较 Alpaca 格式和 ChatML 格式的优缺点
