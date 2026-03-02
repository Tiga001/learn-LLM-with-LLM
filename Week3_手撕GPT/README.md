# 第 3 周：手撕 GPT

> **目标**：精读 GPT 系列论文，理解 Decoder-only 架构与 Causal Language Modeling 的设计哲学；从零手写 GPT 模型并实现 Top-K/Top-P/Temperature 采样；入门 FlashAttention 的 IO 感知思想；在小规模语料上跑通预训练与文本生成。

---

## 每日安排（建议每日 3~5 小时）

| 天 | 主题 | 核心任务 | 产出 | 难度 |
|----|------|---------|------|:----:|
| Day 1 | GPT 系列论文精读 | GPT-1/2/3/3.5(InstructGPT)/4 核心创新对比 | 论文对比表 | ⭐⭐⭐ |
| Day 2 | GPT 模型架构详解 | Decoder-only 数据流、Causal LM 目标函数、与 Transformer 的差异 | 架构图 + 数学推导笔记 | ⭐⭐⭐ |
| Day 3 | **手写 GPT 模型** | 从零实现 CausalSelfAttention → GPTBlock → GPT | `gpt.py` 可运行 | ⭐⭐⭐⭐ |
| Day 4 | 采样策略 | Top-K / Top-P (Nucleus) / Temperature 三种采样实现 | `sampling.py` 可运行 | ⭐⭐⭐ |
| Day 5 | FlashAttention 入门 | IO 感知算法原理、分块计算、在线 softmax；tiling 思想 | 理解笔记 | ⭐⭐⭐⭐ |
| Day 6 | **GPT 预训练实践** | 小规模语料预训练 + 文本生成推理 | 训练脚本 + 生成示例 | ⭐⭐⭐⭐ |
| Day 7 | GPT-3 论文精读 + 周复盘 | 精读 *Language Models are Few-Shot Learners*；串联全周知识 | 论文笔记 + 自检 | ⭐⭐⭐⭐ |

---

## 知识图谱

```
Day 1: 论文精读 — GPT 系列的演进脉络
  GPT-1 (无监督预训练+微调) → GPT-2 (zero-shot) → GPT-3 (few-shot/ICL)
       → InstructGPT (RLHF) → GPT-4 (多模态+MoE)
       │
       ▼
Day 2: 架构详解 — 从 Transformer Decoder 到 GPT
  Causal Mask → Decoder-only Block → CLM 目标函数 → 与 W2 Transformer 的差异
       │
       ▼
Day 3: 手写 GPT — 核心代码实现（本周最重要！）
  CausalSelfAttention → GPTBlock (Attn + FFN + LN) → GPT (Embedding + Blocks + LM Head)
       │
       ▼
Day 4: 采样策略 — 让模型「说话」
  Greedy → Temperature → Top-K → Top-P (Nucleus) → 组合策略
       │
       ▼
Day 5: FlashAttention — 理解高效注意力的基石
  GPU 内存层次 → IO 瓶颈分析 → Tiling → Online Softmax → 复杂度对比
       │                                                    │
       ▼                                                    ▼
Day 6: GPT 预训练实践                              → 第 14 周深化
  数据加载 → 训练循环 → Loss 曲线 → 文本生成 → 效果分析
       │
       ▼
Day 7: GPT-3 论文精读 + 全周复盘
  Few-shot / ICL 能力分析 → 第 4 周手撕 LLaMA 铺路
```

---

## 与前序周次的关系

| 本周内容 | 前序基础 | 后续衔接 |
|---------|---------|---------|
| GPT Decoder-only 架构 | W2 Transformer Encoder-Decoder | W4 LLaMA 架构改进 |
| Causal Self-Attention | W2 Multi-Head Attention + Causal Mask | W4 GQA / W14 FlashAttention 深化 |
| BPE Tokenizer | W1 Day3-4 已手写 BPE + tiktoken | W4 LLaMA Tokenizer / W7 词表扩展 |
| FlashAttention 入门 | W2 Attention 计算原理 | W14 FlashAttention 1/2 / FlashDecoding 深化 |
| CLM 训练目标 | W1 PPL / CE Loss | W4 LLaMA 预训练 / W5 SFT |

---

## 文件结构

```
Week3_手撕GPT/
├── README.md                        ← 你在这里
├── Day1_GPT系列论文精读.md           ← GPT-1/2/3/3.5/4 核心创新对比
├── Day2_GPT模型架构详解.md           ← Decoder-only 架构 + Causal LM 数学
├── Day3_手写GPT模型.ipynb           ← 手写 GPT 完整实现 (核心!)
├── Day4_采样策略.ipynb               ← Top-K / Top-P / Temperature 实现
├── Day5_FlashAttention入门.md       ← IO 感知算法 + tiling 思想
├── Day6_GPT预训练实践.ipynb          ← 小规模预训练 + 文本生成
└── Day7_GPT3论文精读与复盘.md        ← GPT-3 精读 + 周复盘
```

---

## 关键检查点 ✅

完成本周学习后，你应该能够：

- [ ] 列出 GPT-1 到 GPT-4 各代的核心创新（各一句话）
- [ ] 画出 GPT 的完整架构图（含维度标注）
- [ ] 写出 Causal LM 的目标函数 $\mathcal{L} = -\sum_t \log P(x_t | x_{<t})$
- [ ] **闭卷手写 CausalSelfAttention（含 Causal Mask）**
- [ ] **闭卷手写 GPTBlock（Attention + FFN + LayerNorm + Residual）**
- [ ] **闭卷手写完整 GPT 模型（Embedding + N × Block + LM Head）**
- [ ] **手写 Top-K / Top-P / Temperature 采样函数**
- [ ] 解释 FlashAttention 的 tiling 思想和 IO 复杂度优势
- [ ] 在小规模语料上跑通 GPT 预训练并生成文本
- [ ] 精读 GPT-3 论文，理解 In-Context Learning

---

## 本周必读论文

1. **Language Models are Few-Shot Learners** (GPT-3, Brown et al., 2020) — **精读**
2. **FlashAttention: Fast and Memory-Efficient Exact Attention** (Dao et al., 2022) — **泛读**，理解核心思想

## 参考论文

- *Improving Language Understanding by Generative Pre-Training* (GPT-1, Radford et al., 2018)
- *Language Models are Unsupervised Multitask Learners* (GPT-2, Radford et al., 2019)
- *Training language models to follow instructions with human feedback* (InstructGPT, Ouyang et al., 2022)
- *GPT-4 Technical Report* (OpenAI, 2023)

## 推荐资源

- Andrej Karpathy: [*Let's build GPT from scratch*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (YouTube) — 强烈推荐
- Andrej Karpathy: [nanoGPT](https://github.com/karpathy/nanoGPT) — 最小 GPT 实现
- Jay Alammar: [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- Lilian Weng: [Large Language Model Training](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
