# Day 1：LLM 发展历程 — 从 Word2Vec 到 DeepSeek

## 一、技术演进时间线

```
2013  Word2Vec          ← 词向量革命，但无法捕捉上下文
  │
2017  Transformer       ← "Attention Is All You Need"，奠基石
  │
  ├─ 2018  GPT-1        ← Decoder-only, 无监督预训练 + 有监督微调
  ├─ 2018  BERT         ← Encoder-only, 双向掩码语言模型 (MLM)
  ├─ 2019  GPT-2        ← 更大规模, zero-shot 能力涌现
  ├─ 2019  T5           ← Encoder-Decoder, "Text-to-Text" 统一范式
  │
2020  GPT-3             ← 175B 参数, few-shot 能力, Scaling Law 验证
  │
  ├─ 2022  InstructGPT  ← RLHF 首次大规模应用 (GPT-3.5 前身)
  ├─ 2022  ChatGPT      ← 对话式 AI 引爆行业
  ├─ 2023  GPT-4        ← 多模态, 推理增强
  │
  ├─ 2023  LLaMA-1/2    ← Meta 开源, 开源生态起点
  ├─ 2023  Mistral-7B   ← Sliding Window Attention, GQA
  ├─ 2023  Qwen         ← 阿里通义, 中文大模型
  ├─ 2024  LLaMA-3      ← 更大规模开源
  ├─ 2024  DeepSeek-V2  ← MLA + MoE, 训练效率革命
  ├─ 2025  DeepSeek-R1  ← RL 驱动推理, GRPO
  └─ ...
```

### 关键转折点解读

#### 1. Transformer (2017) — 一切的起点

**核心创新**：Self-Attention 机制替代 RNN/LSTM 的序列建模方式。

为什么重要？
- **并行化**：RNN 必须按时间步顺序计算，Transformer 可以并行处理所有位置
- **长距离依赖**：Attention 可以直接连接任意两个位置，无需逐步传递
- **可扩展性**：架构简洁，易于 scale up

#### 2. GPT 系列 — Decoder-only 的胜利

| 模型 | 参数量 | 核心创新 |
|------|--------|---------|
| GPT-1 (2018) | 117M | 无监督预训练 + 有监督微调 |
| GPT-2 (2019) | 1.5B | zero-shot, 不需要微调就能做任务 |
| GPT-3 (2020) | 175B | few-shot/in-context learning, Scaling Law |
| InstructGPT (2022) | ~175B | RLHF 对齐人类偏好 |
| GPT-4 (2023) | ~1.8T (推测) | 多模态, MoE (推测) |

**GPT 的核心哲学**：一个足够大的语言模型，通过预测下一个 token，就能「涌现」出各种能力。

#### 3. BERT (2018) — Encoder-only 的辉煌

- **双向注意力**：每个 token 能看到左右所有上下文
- **预训练任务**：Masked Language Model (MLM) + Next Sentence Prediction (NSP)
- **影响**：NLU 任务（分类、NER、QA）的统治者，但**不适合生成任务**
- **现状**：在 LLM 时代，BERT 系列退居「嵌入模型」角色（检索、分类）

#### 4. 开源生态 (2023~) — 竞争格局

| 模型 | 机构 | 特点 |
|------|------|------|
| LLaMA-1/2/3 | Meta | 开源标杆，社区生态最丰富 |
| Mistral/Mixtral | Mistral AI | 高效架构 (GQA, Sliding Window, MoE) |
| Qwen-1/2 | 阿里 | 中文能力强，多模态 |
| DeepSeek-V2/V3/R1 | DeepSeek | MLA, MoE, GRPO, 训练效率极高 |
| Yi | 零一万物 | 中英双语 |
| Gemma | Google | 轻量开源 |

---

## 二、为什么 Decoder-only 赢了？

这是理解现代 LLM 最重要的问题之一。

### 三种架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    Encoder-only (BERT)                           │
│  输入: [CLS] The cat sat on the [MASK] . [SEP]                  │
│  注意力: 双向 (每个token看所有token)                               │
│  任务: 填空 (MLM), 分类, NER                                     │
│  缺点: 不擅长生成                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Encoder-Decoder (T5)                          │
│  Encoder输入: "translate English to French: The cat sat..."      │
│  Decoder输出: "Le chat s'est assis..."                           │
│  注意力: Encoder双向, Decoder因果, Cross-Attention连接            │
│  适合: 翻译, 摘要, Seq2Seq                                       │
│  缺点: 架构复杂, 参数效率低(encoder参数在生成时利用率低)            │
├─────────────────────────────────────────────────────────────────┤
│                    Decoder-only (GPT)                            │
│  输入: "The cat sat on the"                                      │
│  输出: 预测下一个token → "mat"                                    │
│  注意力: 因果 (每个token只看它左边的token)                         │
│  适合: 一切文本生成 + 通过prompt也能做分类/NER                     │
│  优势: 架构简洁, 扩展性好, 统一范式                                │
└─────────────────────────────────────────────────────────────────┘
```

### Decoder-only 胜出的原因

1. **统一范式**：所有任务都可以转化为「生成下一个 token」—— 分类、翻译、摘要、代码、数学推理...
2. **Scaling 友好**：架构简洁，容易 scale 到万亿参数
3. **涌现能力**：规模足够大后，出现 in-context learning、chain-of-thought 等「涌现」能力
4. **训练效率**：因果语言模型的训练目标简单（预测下一个 token），数据利用效率高

---

## 三、Scaling Law — 大力出奇迹的理论基础

### Kaplan Scaling Law (OpenAI, 2020)

模型性能（以 loss 衡量）主要由三个因素决定：

```
L(N, D, C) ∝ N^{-αN} + D^{-αD} + C^{-αC}

其中:
  N = 模型参数量
  D = 训练数据量 (token 数)
  C = 计算量 (FLOPs)
  α = 幂律指数
```

**核心发现**：
- 性能随 N, D, C 的增长呈**幂律**(power law)提升
- 在固定计算预算下，**模型参数和数据量应等比例增长**
- 架构细节（层数 vs 宽度）的影响相对小

### Chinchilla Scaling Law (DeepMind, 2022)

修正了 Kaplan 的结论：
- 最优策略：**参数量和数据量应以 1:20 的比例增长**
- 即 1B 参数模型需要 ~20B token 训练
- LLaMA 的成功证实了这一点：7B 模型用了 1T+ token 训练

### 为什么博士生要理解 Scaling Law？

- 它解释了为什么「大」模型能涌现出小模型没有的能力
- 它指导了工业界的资源分配（应该花钱在更大模型还是更多数据？）
- 它是当前 LLM 研究的核心范式之一
- **局限性**：Scaling Law 告诉你 loss 的变化，但不能精确预测下游任务能力的涌现

---

## 四、自检题

完成 Day 1-2 的学习后，你应该能回答：

1. **为什么 Decoder-only 架构成为主流？** 至少说出 3 个原因。
2. **GPT-1 到 GPT-4 的核心创新分别是什么？** 各用一句话概括。
3. **Scaling Law 说了什么？Chinchilla 修正了什么？**
4. **BERT 在当前 LLM 时代的角色是什么？** 为什么它没有被完全淘汰？
5. **LLaMA 和 DeepSeek 各自的核心技术创新是什么？**

---

## 五、产出要求

- [ ] 手绘一张 LLM 技术演进时间线思维导图（从 2017 Transformer 到 2025 DeepSeek-R1）
- [ ] 写一篇 2 页笔记：三大架构对比 + Scaling Law
