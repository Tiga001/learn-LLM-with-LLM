# 第 1 周：LLM 概述与预备知识

> **目标**：建立大模型全景认知，掌握现代 Tokenizer 与评估体系，能手写 BPE 编码器和经典评价指标，精读 Transformer 奠基论文为第 2 周做准备。

---

## 每日安排（建议每日 3~5 小时）

| 天 | 主题 | 核心任务 | 产出 | 难度 |
|----|------|---------|------|:----:|
| Day 1 | LLM 发展脉络 | GPT→BERT→T5→ChatGPT→开源生态；Decoder-only 为何胜出 | 时间线思维导图 | ⭐⭐ |
| Day 2 | 三大架构 + Scaling Law | Enc-only / Dec-only / Enc-Dec 对比；Kaplan & Chinchilla Scaling Law | 2 页笔记 + 自检题 | ⭐⭐⭐ |
| Day 3 | BPE Tokenizer 原理与实现 | BPE 算法推导、**手写 BPE 编码器**；Byte-level BPE 原理 | `bpe.py` 可运行 | ⭐⭐⭐ |
| Day 4 | SentencePiece + tiktoken + Embedding | Unigram 算法对比；tiktoken 实践；Embedding 层设计与词表扩展 | 实验 notebook | ⭐⭐ |
| Day 5 | 经典指标：BLEU / ROUGE / PPL | 数学推导 + **从零实现三大指标** | `metrics.py` 可运行 | ⭐⭐⭐ |
| Day 6 | 现代评估体系 | MMLU / HumanEval / MT-Bench / Arena Elo 定位与使用 | 评估笔记 | ⭐⭐ |
| Day 7 | 论文精读 + 周复盘 | 精读 *Attention Is All You Need*；串联全周知识 | 论文笔记 + 自检 | ⭐⭐⭐⭐ |

---

## 知识图谱

```
Day 1-2: 宏观认知
  LLM 发展历程 ─→ 三大架构对比 ─→ Scaling Law
       │                │
       ▼                ▼
Day 3-4: Tokenizer（模型的入口）
  BPE 原理 ─→ 手写 BPE ─→ Byte-level BPE ─→ tiktoken/SentencePiece
       │                                          │
       ▼                                          ▼
  Embedding 层设计 ←─────────────── 词表扩展（→W7 Chinese-LLaMA）
       │
       ▼
Day 5-6: 评估（如何衡量好坏）
  经典指标 (BLEU/ROUGE/PPL) ─→ 现代基准 (MMLU/HumanEval/MT-Bench/Arena)
       │
       ▼
Day 7: 论文精读（为第 2 周手撕 Transformer 铺路）
  Attention Is All You Need → 理解 Multi-Head Attention, Position Encoding
```

---

## 文件结构

```
Week1_LLM概述与预备知识/
├── README.md                        ← 你在这里
├── Day1_LLM发展历程.md              ← ✅ LLM 技术演进 + 三大架构简介
├── Day2_三大架构与ScalingLaw.md      ← ✅ 架构深度对比 + Scaling Law 数学
├── Day3_BPE_Tokenizer.ipynb         ← ✅ BPE 手写实现 (核心!)
├── Day4_SentencePiece与Embedding.ipynb ← ✅ Unigram/SentencePiece/tiktoken/Embedding
├── Day5_评价指标.ipynb               ← ✅ BLEU/ROUGE/PPL 手写实现
├── Day6_现代评估体系.md              ← ✅ MMLU/HumanEval/MT-Bench/Arena
└── Day7_论文精读与复盘.md            ← ✅ Attention Is All You Need 精读指南
```

---

## 关键检查点 ✅

完成本周学习后，你应该能够：

- [ ] 画出 2017→2025 LLM 技术演进时间线
- [ ] 解释 Decoder-only 胜出的 4 个原因（训练效率、参数利用、ICL、工程简洁）
- [ ] 写出 Chinchilla Scaling Law 的最优配比公式
- [ ] **闭卷手写 BPE 训练 + 编码算法**
- [ ] 解释 Byte-level BPE 为何解决了 OOV 问题
- [ ] **手写 BLEU 计算（含 Modified Precision 和 Brevity Penalty）**
- [ ] **手写 PPL 计算公式并解释其与 CE Loss 的关系**
- [ ] 说出 MMLU / HumanEval / MT-Bench / Arena 各自测什么
- [ ] 画出 Transformer 的整体架构图（为第 2 周做准备）

---

## 本周必读论文

1. **Attention Is All You Need** (Vaswani et al., 2017) — **精读**，为第 2 周手撕 Transformer 做准备
2. **Language Models are Few-Shot Learners** (GPT-3, 选读) — 建立 Scaling 直觉

## 推荐资源

- Andrej Karpathy: *Let's build GPT from scratch* (YouTube)
- Jay Alammar: [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)
- 李沐《动手学深度学习》第 10-11 章
- [3Blue1Brown: Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc)
