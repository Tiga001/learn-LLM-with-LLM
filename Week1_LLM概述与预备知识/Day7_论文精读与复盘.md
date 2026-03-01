# Day 7：论文精读 *Attention Is All You Need* + 第一周复盘

> **目标**：精读 Transformer 奠基论文，为第 2 周手撕 Transformer 建立完整的理论地基；回顾本周所学，查漏补缺。

---

## Part 1：论文精读 — Attention Is All You Need

**论文信息**：Vaswani et al., NeurIPS 2017, Google Brain & Google Research

**论文地址**：[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

### 精读指南

这篇论文是整个 LLM 时代的基石。精读时按以下层次递进：

```
第一遍（30 min）：读 Abstract + Introduction + Conclusion，抓住核心贡献
第二遍（60 min）：读 Section 3（Model Architecture），理解每个组件
第三遍（60 min）：看数学公式和图表，对照代码理解
```

---

### 1. 论文解决了什么问题？

2017 年之前，序列建模的主流是 RNN/LSTM + Attention：

| 问题 | RNN/LSTM 的局限 |
|------|----------------|
| 并行化 | 必须按时间步顺序计算，无法并行 → 训练慢 |
| 长距离依赖 | 信息需要逐步传递，梯度消失/爆炸 |
| 可扩展性 | 架构复杂，难以 scale to 大模型 |

**Transformer 的核心主张**：完全抛弃循环结构，**仅用 Attention 机制**就能胜任序列建模。

---

### 2. 模型架构精读

论文的 Figure 1 是最重要的图，包含 Transformer 的完整架构：

```
                     ┌──────────────┐
                     │  Output Prob │
                     │   (Softmax)  │
                     └──────┬───────┘
                            │
                     ┌──────┴───────┐
                     │   Linear     │
                     └──────┬───────┘
                            │
                   ┌────────┴────────┐
                   │  Decoder Block  │ × N
                   │                 │
                   │ ┌─────────────┐ │
                   │ │ Feed Forward│ │
                   │ └──────┬──────┘ │
                   │ ┌──────┴──────┐ │
                   │ │Cross-Attn   │←──── Encoder Output
                   │ └──────┬──────┘ │
                   │ ┌──────┴──────┐ │
                   │ │Masked Self- │ │
                   │ │  Attention  │ │
                   │ └──────┬──────┘ │
                   └────────┬────────┘
                            │
                   Positional Encoding
                            +
                   Output Embedding
                            │
                     Output (shifted)


     ┌────────┴────────┐
     │  Encoder Block  │ × N
     │                 │
     │ ┌─────────────┐ │
     │ │ Feed Forward│ │
     │ └──────┬──────┘ │
     │ ┌──────┴──────┐ │
     │ │  Self-Attn  │ │
     │ └──────┬──────┘ │
     └────────┬────────┘
              │
     Positional Encoding
              +
     Input Embedding
              │
          Input
```

#### 2.1 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**精读要点**：
- **为什么要除以 $\sqrt{d_k}$？** 当 $d_k$ 很大时，$QK^T$ 的值会很大，导致 softmax 的梯度极小（进入饱和区）。除以 $\sqrt{d_k}$ 使方差回到 1。
- **论文原文解释**（Section 3.2.1）：假设 $Q, K$ 的元素均为均值 0、方差 1 的独立随机变量，则 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$。

#### 2.2 Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**精读要点**：
- 多头的作用：让不同 head 关注不同类型的模式（语法关系、语义关系、位置关系等）
- **维度设计**：$d_k = d_v = d_{\text{model}} / h$，确保总计算量与单头 attention 相同
  - 论文使用 $d_{\text{model}} = 512$，$h = 8$，所以 $d_k = d_v = 64$

#### 2.3 Position-wise Feed-Forward Network

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

**精读要点**：
- 对每个位置独立做相同的两层全连接（所以叫 position-wise）
- 中间维度 $d_{ff} = 2048 = 4 \times d_{\text{model}}$（4 倍扩展是经验值）
- 可以理解为每个位置的「思考」过程：Attention 负责「看到了什么」，FFN 负责「怎么处理」

#### 2.4 Positional Encoding

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

**精读要点**：
- 为什么用正弦/余弦？论文指出：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性变换 → 模型可以学到相对位置关系
- 不同维度使用不同频率的正弦波 → 形成一种「位置的傅里叶编码」
- **局限**：这种编码缺乏外推性，现代模型（LLaMA）已经换成了 RoPE（第 4 周学）

#### 2.5 Residual Connection + Layer Normalization

每个子层的输出：

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

**精读要点**：
- 残差连接：缓解深层网络的梯度消失
- Layer Norm：稳定训练，加速收敛
- 这是 **Post-Norm** 结构，现代 LLM（如 LLaMA）使用 **Pre-Norm**：$x + \text{Sublayer}(\text{Norm}(x))$

#### 2.6 Masking

论文中提到两种 mask：
- **Padding Mask**：将 padding 位置的 attention score 设为 $-\infty$
- **Look-ahead Mask（Causal Mask）**：Decoder 中防止看到未来位置

---

### 3. 训练细节

| 配置 | 值 |
|------|-----|
| 数据集 | WMT 2014 English-German (4.5M 句对) |
| 模型 | 6 层 Encoder + 6 层 Decoder |
| $d_{\text{model}}$ | 512 |
| $d_{ff}$ | 2048 |
| $h$ (头数) | 8 |
| 参数量 | ~65M (base) / ~213M (big) |
| 优化器 | Adam ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$) |
| 学习率 | Warmup + 衰减调度 |
| 训练时间 | 3.5 天 (8 P100 GPUs) |

#### Learning Rate Schedule（经典的 warmup）

$$lr = d_{\text{model}}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup\_steps^{-1.5})$$

前 $warmup\_steps$（4000步）线性增长，之后按 $step^{-0.5}$ 衰减。

---

### 4. 实验结果

| 模型 | EN-DE BLEU | EN-FR BLEU | 训练成本 |
|------|:----------:|:----------:|:-------:|
| 之前 SOTA | 26.4 | 41.0 | — |
| Transformer (base) | 27.3 | 38.1 | 12 GPU-days |
| **Transformer (big)** | **28.4** | **41.8** | **96 GPU-days** |

Transformer 在翻译任务上超越所有 RNN 模型，且训练速度大幅提升。

---

### 5. 论文的历史意义

| 贡献 | 影响 |
|------|------|
| 证明 Attention 可以完全替代 RNN | 开启了非循环序列建模时代 |
| Multi-Head Attention | 成为所有后续模型的标准组件 |
| 简洁可扩展的架构 | 使 Scaling Law 成为可能 |
| 训练速度的巨大提升 | 使大规模预训练变得经济可行 |

**一句话总结**：这篇论文不是提出了最强的翻译模型，而是提出了一个**通用的、可扩展的序列建模架构**——后来的 GPT、BERT、LLaMA 都是它的变体。

---

## Part 2：第一周知识串联与复盘

### 知识链路回顾

```
文本 → [Day 3-4: Tokenizer] → token ID 序列 → [Day 4: Embedding] → 向量序列
  → [Day 7: Transformer] → 隐藏状态 → Output Head → logits → next token 概率
  → [Day 5: PPL] 衡量模型质量
  → [Day 5: BLEU/ROUGE] 衡量生成质量
  → [Day 6: MMLU/Arena] 衡量综合能力

Day 1-2 提供了宏观背景:
  为什么是 Decoder-only？ → Scaling Law → 涌现能力 → 开源生态
```

### 全周自检清单

#### 理论层
- [ ] 能画出 2017→2025 LLM 技术演进时间线
- [ ] 能说出 Decoder-only 胜出的 4 个原因
- [ ] 能写出 Chinchilla Scaling Law 公式 $D_{opt} \approx 20N$
- [ ] 能写出 Scaled Dot-Product Attention 公式
- [ ] 能解释为什么要除以 $\sqrt{d_k}$
- [ ] 能说出 Multi-Head Attention 的维度关系
- [ ] 能解释 Positional Encoding 的设计动机
- [ ] 能区分 Padding Mask 和 Causal Mask
- [ ] 能说出 BLEU 的 Modified Precision 解决了什么问题
- [ ] 能写出 PPL = $\exp(\text{CE Loss})$
- [ ] 能说出 MMLU / HumanEval / MT-Bench / Arena 各自测什么

#### 代码层
- [ ] 能手写 BPE 训练算法（合并循环）
- [ ] 能手写 BPE 编码（应用合并规则）
- [ ] 能手写 BLEU 计算（含 BP）
- [ ] 能手写 PPL 计算

#### 直觉层
- [ ] 理解 Tokenizer 对模型效率的影响（中文 vs 英文）
- [ ] 理解经典指标 vs 现代评估的演进逻辑
- [ ] 理解 Transformer 相对于 RNN 的根本优势

---

### 常见疑惑解答

**Q1：BPE 和 Unigram 到底该用哪个？**

实际上，对于 Decoder-only LLM，BPE 是绝对主流。Unigram 主要在 T5/ALBERT 等 Encoder-Decoder 模型中使用。选择更多是历史原因而非本质差异。

**Q2：Transformer 论文是 Encoder-Decoder，为什么后来 GPT 只用 Decoder？**

Transformer 论文的任务是翻译（Seq2Seq），需要 Encoder-Decoder。GPT 发现：只用 Decoder + 足够大的模型 + 足够多的数据，就能用「预测下一个 token」这一个目标统一所有任务。

**Q3：为什么 PPL 低不代表下游任务好？**

PPL 衡量的是「预测下一个 token 的准确度」。一个只背诵了维基百科的模型 PPL 很低，但它不会推理、不会写代码、不会遵循指令。LLM 的「能力」远超 token 预测。

**Q4：Chatbot Arena 的 Elo 分数有绝对意义吗？**

没有。Elo 分只有相对意义——它反映的是两个模型在对战中的相对胜率。1200 分本身不代表什么，但 1200 vs 1100 意味着前者有约 64% 的胜率。

---

### 下周预告：手撕 Transformer

第 2 周你将从零手写完整的 Transformer：

| 组件 | 从论文到代码 |
|------|-------------|
| `MultiHeadAttention` | Day 7 学的 Attention 公式 → 代码实现 |
| `PositionalEncoding` | 正弦/余弦公式 → 代码 + 可视化 |
| `TransformerBlock` | Attention + FFN + LayerNorm + Residual |
| 完整模型 | 6层 Encoder + 6层 Decoder |
| 训练 | 英→法翻译任务 |

**准备工作**：确保你能闭卷写出 Attention 公式和 Multi-Head Attention 的维度关系。这是第 2 周的起点。
