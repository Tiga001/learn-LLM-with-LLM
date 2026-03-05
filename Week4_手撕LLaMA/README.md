# 第 4 周：手撕 LLaMA

> **目标**：精读 LLaMA 系列论文，理解 LLaMA 相比 GPT 的所有架构改进（RMSNorm、RoPE、SwiGLU、GQA）；从零手写完整 LLaMA 模型；掌握 KV Cache 推理加速原理；在小规模语料上跑通预训练。

---

## 每日安排（建议每日 3~5 小时）

| 天 | 主题 | 核心任务 | 产出 | 难度 |
|----|------|---------|------|:----:|
| Day 1 | LLaMA 系列论文精读 | LLaMA-1/2 核心创新、数据配比、训练策略 | LLaMA vs GPT 技术差异分析表 | ⭐⭐⭐ |
| Day 2 | LLaMA 模型架构详解 | 逐模块拆解：RMSNorm / SwiGLU / RoPE / GQA 数学推导 | 架构图 + 数学推导笔记 | ⭐⭐⭐⭐ |
| Day 3 | **手写 LLaMA 模型** | 从零实现 RMSNorm → RoPE → SwiGLU → GQA → LLaMABlock → LLaMA | `llama.py` 可运行 | ⭐⭐⭐⭐⭐ |
| Day 4 | RoPE 旋转位置编码 | 复数推导、旋转矩阵实现、外推性分析（**面试高频！**） | 手写 RoPE + 可视化 | ⭐⭐⭐⭐⭐ |
| Day 5 | KV Cache 推理加速 | KV Cache 原理、显存分析、带 Cache 的推理实现 | KV Cache 实现 + 加速对比 | ⭐⭐⭐⭐ |
| Day 6 | **LLaMA 预训练实践** | 小规模语料预训练 + 带 KV Cache 的文本生成 | 训练脚本 + 生成示例 | ⭐⭐⭐⭐ |
| Day 7 | LLaMA-2 论文精读 + 周复盘 | 精读 LLaMA-2 论文中的训练与安全对齐；串联全周知识 | 论文笔记 + 自检 | ⭐⭐⭐⭐ |

---

## 知识图谱

```
Day 1: 论文精读 — LLaMA 系列的设计哲学
  LLaMA-1 (开源+高效数据) → LLaMA-2 (更大规模+RLHF)
  "在 GPT 基础上，每一处改进都有严格的数学/工程动机"
       │
       ▼
Day 2: 架构详解 — 从 GPT 到 LLaMA 的四大改进
  RMSNorm (更快的归一化) → RoPE (旋转位置编码) → SwiGLU (门控 FFN) → GQA (分组注意力)
       │
       ▼
Day 3: 手写 LLaMA — 核心代码实现（本周最重要！）
  RMSNorm → RoPE → SwiGLU FFN → GQA Attention → LLaMABlock → LLaMA
       │
       ▼
Day 4: RoPE 深入 — 面试必考的位置编码
  复数表示 → 旋转矩阵 → 远程衰减性 → 外推与 NTK-aware 扩展
       │
       ▼
Day 5: KV Cache — 推理加速的核心技巧
  自回归推理的冗余 → KV Cache 原理 → 显存分析 → GQA 如何减少 Cache
       │                                                    │
       ▼                                                    ▼
Day 6: LLaMA 预训练实践                              → 第 14 周深化
  数据加载 → 训练循环 → Loss 曲线 → 带 KV Cache 的生成
       │
       ▼
Day 7: LLaMA-2 论文精读 + 全周复盘
  训练改进 → RLHF 对齐 → 安全评估 → 第 5 周指令微调铺路
```

---

## 与前序周次的关系

| 本周内容 | 前序基础 | 后续衔接 |
|---------|---------|---------|
| RMSNorm | W2-3 LayerNorm + Pre-Norm | W5-7 微调中使用 |
| RoPE 旋转位置编码 | W3 GPT 可学习位置编码 | W14 长上下文 / NTK-aware / YaRN |
| SwiGLU FFN | W3 GELU FFN | W5-7 微调模型架构 |
| GQA 分组注意力 | W3 Multi-Head Attention | W14 MQA / KV Cache 深化 |
| KV Cache | W3 自回归生成 | W14 PagedAttention / vLLM |
| LLaMA 预训练 | W3 GPT 预训练实践 | W5 Alpaca 指令微调 / W7 二次预训练 |

---

## 文件结构

```
Week4_手撕LLaMA/
├── README.md                          ← 你在这里
├── Day1_LLaMA系列论文精读.md           ← LLaMA-1 论文核心创新与数据策略
├── Day2_LLaMA模型架构详解.md           ← 四大改进的完整数学推导
├── Day3_手写LLaMA模型.ipynb           ← 手写 LLaMA 完整实现 (核心!)
├── Day4_RoPE旋转位置编码.md            ← RoPE 复数推导 + 可视化 (面试重点!)
├── Day5_KVCache推理加速.md             ← KV Cache 原理 + 显存分析
├── Day6_LLaMA预训练实践.ipynb          ← 小规模预训练 + KV Cache 推理
└── Day7_LLaMA2论文精读与复盘.md        ← LLaMA-2 精读 + 周复盘
```

---

## 关键检查点 ✅

完成本周学习后，你应该能够：

- [ ] 列出 LLaMA 相比 GPT 的所有架构改进及其动机
- [ ] 画出 LLaMA Block 的完整架构图（含维度标注）
- [ ] **闭卷手写 RMSNorm（含数学公式和代码）**
- [ ] **闭卷手写 RoPE（含 `precompute_freqs_cis` + `apply_rotary_emb`）— 面试高频！**
- [ ] **闭卷手写 SwiGLU FFN（含门控机制）**
- [ ] **理解 GQA 并能手写实现**
- [ ] 解释 KV Cache 的原理和显存开销计算
- [ ] 实现带 KV Cache 的自回归推理
- [ ] 在小规模语料上跑通 LLaMA 预训练并生成文本
- [ ] 精读 LLaMA-2 论文，理解训练策略与安全对齐

---

## 本周必读论文

1. **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023) — **精读**
2. **Llama 2: Open Foundation and Fine-Tuned Chat Models** (Touvron et al., 2023) — **精读**

## 参考论文

- *RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su et al., 2021)
- *GLU Variants Improve Transformer* (Shazeer, 2020)
- *Root Mean Square Layer Normalization* (Zhang & Sennrich, 2019)
- *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (Ainslie et al., 2023)
- *Fast Transformer Decoding: One Write-Head is All You Need* (Shazeer, 2019) — MQA 原始论文

## 推荐资源

- Meta AI: [LLaMA 官方仓库](https://github.com/meta-llama/llama)
- Andrej Karpathy: [llama2.c](https://github.com/karpathy/llama2.c) — 最小 LLaMA-2 推理实现
- EleutherAI: [Rotary Embeddings 博客](https://blog.eleuther.ai/rotary-embeddings/)
- HuggingFace: [LLaMA 模型文档](https://huggingface.co/docs/transformers/model_doc/llama)
