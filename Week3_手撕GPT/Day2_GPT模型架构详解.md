# Day 2：GPT 模型架构详解 — 从 Transformer Decoder 到 GPT

> **目标**：深入理解 GPT 的 Decoder-only 架构，从数据流、数学推导和代码视角全面拆解 GPT 模型，为 Day 3 手写实现打下坚实基础。

---

## 一、GPT 架构全景图

GPT 是一个纯 Decoder 架构的自回归语言模型。它的结构极其简洁：

```
输入 token IDs: [x_1, x_2, ..., x_T]
       │
       ▼
┌──────────────────┐
│  Token Embedding  │  E_tok ∈ R^{V × d}
│  + Position Emb   │  E_pos ∈ R^{T_max × d}
└──────────────────┘
       │
       ▼  h_0 = E_tok[x] + E_pos[0:T]
┌──────────────────┐
│   GPT Block × N   │  ← 重复 N 次（GPT-2 Small: N=12）
│  ┌──────────────┐ │
│  │  LayerNorm 1  │ │
│  │  Causal MHA   │ │  ← 因果多头自注意力（核心！）
│  │  + Residual    │ │
│  ├──────────────┤ │
│  │  LayerNorm 2  │ │
│  │  FFN (MLP)    │ │  ← Position-wise Feed-Forward
│  │  + Residual    │ │
│  └──────────────┘ │
└──────────────────┘
       │
       ▼  h_N
┌──────────────────┐
│   Final LayerNorm │
│   LM Head (线性层)│  W ∈ R^{d × V}，通常与 E_tok 共享权重
└──────────────────┘
       │
       ▼
  logits ∈ R^{T × V}  → softmax → P(x_{t+1} | x_{≤t})
```

**关键数字（GPT-2 Small 为例）**：

| 超参数 | 符号 | GPT-2 Small | GPT-2 Medium | GPT-2 Large | GPT-2 XL |
|--------|------|-------------|--------------|-------------|----------|
| 层数 | $N$ | 12 | 24 | 36 | 48 |
| 隐藏维度 | $d$ | 768 | 1024 | 1280 | 1600 |
| 注意力头数 | $h$ | 12 | 16 | 20 | 25 |
| 每头维度 | $d_k = d/h$ | 64 | 64 | 64 | 64 |
| FFN 中间维度 | $d_{ff}$ | 3072 | 4096 | 5120 | 6400 |
| 上下文长度 | $T_{max}$ | 1024 | 1024 | 1024 | 1024 |
| 词表大小 | $V$ | 50257 | 50257 | 50257 | 50257 |
| 总参数量 | — | 124M | 355M | 774M | 1558M |

注意 $d_{ff} = 4d$ 是标准设定。

---

## 二、GPT 与原始 Transformer Decoder 的差异

GPT 不是简单地"拿来" Transformer 的 Decoder。理解差异对手写实现至关重要：

| 维度 | 原始 Transformer Decoder | GPT |
|------|-------------------------|-----|
| **Cross-Attention** | 有（attend to Encoder 输出）| **无**（没有 Encoder，不需要 Cross-Attention）|
| **Self-Attention** | Causal（下三角 mask） | Causal（下三角 mask）— 相同 |
| **LayerNorm 位置** | Post-Norm（原始论文） | **Pre-Norm**（GPT-2 起，更稳定）|
| **位置编码** | 正弦/余弦固定编码 | **可学习位置编码**（GPT-1/2/3）|
| **激活函数** | ReLU | **GELU**（GPT-2 起）|
| **输入来源** | 依赖 Encoder 输出 | **纯自回归**，只有自身输入 |

### Pre-Norm vs Post-Norm

这是 GPT 架构中最重要的工程决策之一：

```
Post-Norm（原始 Transformer）:
  x → Sublayer(x) → x + Sublayer(x) → LayerNorm(x + Sublayer(x))
  
Pre-Norm（GPT-2 及之后）:
  x → LayerNorm(x) → Sublayer(LayerNorm(x)) → x + Sublayer(LayerNorm(x))
```

数学表达：

$$
\text{Post-Norm: } x_{l+1} = \text{LN}(x_l + \text{Sublayer}(x_l))
$$

$$
\text{Pre-Norm: } x_{l+1} = x_l + \text{Sublayer}(\text{LN}(x_l))
$$

**Pre-Norm 的优势**：
1. 残差连接形成从输入到输出的"高速公路"，梯度可以无损回传
2. 每个子层的输入都被归一化，训练更稳定
3. 不需要精细的学习率预热

**直觉理解**：Pre-Norm 下，主干通路 $x_l \to x_{l+1}$ 是恒等映射加上一个"修正项"，模型更容易学习。

---

## 三、逐模块数学推导

### 3.1 Token Embedding + Position Embedding

$$
h_0 = E_{\text{tok}}[x_1, \ldots, x_T] + E_{\text{pos}}[0, 1, \ldots, T-1]
$$

- $E_{\text{tok}} \in \mathbb{R}^{V \times d}$：token 嵌入矩阵，每个 token ID 查表得到一个 $d$ 维向量
- $E_{\text{pos}} \in \mathbb{R}^{T_{\max} \times d}$：位置嵌入矩阵，可学习参数
- $h_0 \in \mathbb{R}^{T \times d}$：输入到第一个 Block 的隐状态

GPT-1/2/3 使用**可学习位置编码**（而非 Transformer 原始的正弦编码）。优势是模型可以学习到任意的位置模式，劣势是无法外推到超过训练长度的序列（这也是后来 RoPE 出现的原因之一）。

### 3.2 Causal Self-Attention（因果自注意力）

这是 GPT 的核心。对每个注意力头：

**Step 1：线性投影**

$$
Q = h \cdot W_Q, \quad K = h \cdot W_K, \quad V = h \cdot W_V
$$

其中 $W_Q, W_K \in \mathbb{R}^{d \times d_k}$，$W_V \in \mathbb{R}^{d \times d_v}$，$d_k = d_v = d / h$。

**Step 2：计算注意力分数**

$$
S = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}
$$

**Step 3：应用因果掩码（Causal Mask）**

$$
S_{\text{masked}} = S + M, \quad \text{where } M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}
$$

这是 GPT 区别于 BERT 的核心——因果掩码确保位置 $i$ 的 token 只能 attend 到位置 $\leq i$ 的 token。

```
因果掩码 M（加到 attention score 上）:

      t=0   t=1   t=2   t=3   t=4
t=0 [  0    -∞    -∞    -∞    -∞  ]
t=1 [  0     0    -∞    -∞    -∞  ]
t=2 [  0     0     0    -∞    -∞  ]
t=3 [  0     0     0     0    -∞  ]
t=4 [  0     0     0     0     0  ]

softmax 后 -∞ 变成 0，实现了"看不到未来"
```

**Step 4：Softmax + Value 加权**

$$
A = \text{softmax}(S_{\text{masked}}) \in \mathbb{R}^{T \times T}
$$

$$
\text{head}_i = A \cdot V \in \mathbb{R}^{T \times d_k}
$$

**Step 5：多头拼接 + 输出投影**

$$
\text{MHA}(h) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W_O
$$

其中 $W_O \in \mathbb{R}^{d \times d}$。

**参数量统计**（单层 Attention）：

$$
\text{Params}_{\text{attn}} = 4 \times d^2 \quad (W_Q, W_K, W_V, W_O \text{ 各 } d \times d)
$$

加上 bias 项则为 $4d^2 + 4d$。

### 3.3 Position-wise Feed-Forward Network (FFN)

$$
\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2
$$

- $W_1 \in \mathbb{R}^{d \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d}$
- $d_{ff} = 4d$（标准设定）
- GELU（Gaussian Error Linear Unit）是 GPT-2 的选择，比 ReLU 更平滑

**GELU 的定义**：

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

近似实现：

$$
\text{GELU}(x) \approx 0.5x\left[1 + \tanh\left(\sqrt{2/\pi}(x + 0.044715x^3)\right)\right]
$$

**参数量统计**（单层 FFN）：

$$
\text{Params}_{\text{ffn}} = 2 \times d \times d_{ff} + d_{ff} + d = 8d^2 + 5d
$$

### 3.4 GPT Block（完整一层）

使用 Pre-Norm 的一个 GPT Block：

$$
h'_l = h_l + \text{CausalMHA}(\text{LN}(h_l))
$$

$$
h_{l+1} = h'_l + \text{FFN}(\text{LN}(h'_l))
$$

其中 LN 是 LayerNorm：

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

$\gamma, \beta \in \mathbb{R}^d$ 是可学习参数。

### 3.5 LM Head（输出层）

经过 N 层 Block 后：

$$
\text{logits} = \text{LN}(h_N) \cdot E_{\text{tok}}^T \in \mathbb{R}^{T \times V}
$$

注意这里使用了**权重共享（Weight Tying）**：LM Head 的权重矩阵就是 Token Embedding 的转置。这减少了参数量（$V \times d$ 个参数），且在实践中效果更好。

**直觉**：如果一个 token 的嵌入向量是 $e$，那么生成该 token 的 logit 就是隐状态与 $e$ 的点积——越"像"这个 token，分数越高。

---

## 四、完整数据流追踪

以一个具体例子追踪 GPT-2 Small 的完整数据流。假设输入 `"The cat sat"` 被 tokenize 为 3 个 token：

```
输入 token IDs: [464, 3797, 3332]    # "The", "cat", "sat"
                 shape: (1, 3)        # batch=1, seq_len=3

  ┌─── Token Embedding ────────────────────────┐
  │ E_tok[464], E_tok[3797], E_tok[3332]       │
  │ shape: (1, 3, 768)                          │
  └────────────────────────────────────────────┘
      +
  ┌─── Position Embedding ─────────────────────┐
  │ E_pos[0], E_pos[1], E_pos[2]               │
  │ shape: (1, 3, 768)                          │
  └────────────────────────────────────────────┘
      =
  h_0: shape (1, 3, 768)

  ─── Block 1 ───────────────────────────────────
  │ LN_1:      (1, 3, 768) → (1, 3, 768)
  │ Q, K, V:   (1, 3, 768) → 各 (1, 12, 3, 64)   # 12 heads, d_k=64
  │ QK^T/√64:  (1, 12, 3, 64) × (1, 12, 64, 3) → (1, 12, 3, 3)
  │ + Causal Mask + Softmax → (1, 12, 3, 3)
  │ × V:       (1, 12, 3, 3) × (1, 12, 3, 64) → (1, 12, 3, 64)
  │ Concat:    (1, 12, 3, 64) → (1, 3, 768)
  │ W_O:       (1, 3, 768) → (1, 3, 768)
  │ + Residual → (1, 3, 768)
  │
  │ LN_2:      (1, 3, 768) → (1, 3, 768)
  │ W_1 + GELU: (1, 3, 768) → (1, 3, 3072)
  │ W_2:       (1, 3, 3072) → (1, 3, 768)
  │ + Residual → (1, 3, 768)
  ─────────────────────────────────────────────
  
  ... (重复 12 次) ...

  h_12: shape (1, 3, 768)
  
  Final LN: (1, 3, 768) → (1, 3, 768)
  LM Head:  (1, 3, 768) × (768, 50257) → (1, 3, 50257)
  
  logits[:, -1, :] → softmax → P(next_token | "The cat sat")
```

**训练时**：logits 的每个位置都计算 cross-entropy loss（Causal LM 目标）

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t}) = -\frac{1}{T} \sum_{t=1}^{T} \log \text{softmax}(\text{logits}_{t-1})[x_t]
$$

注意 logits 的位置 $t-1$ 预测 token $x_t$——这是 teacher forcing。

**推理时**：只取最后一个位置的 logits，通过采样策略生成下一个 token。

---

## 五、参数量计算

精确计算 GPT 模型的参数量是面试常考题，也是理解模型规模的基础。

### 5.1 单层 Block 参数

| 组件 | 参数量 | GPT-2 Small |
|------|--------|-------------|
| Attention $W_Q, W_K, W_V, W_O$ | $4d^2$ | $4 \times 768^2 = 2,359,296$ |
| Attention biases | $4d$ | $4 \times 768 = 3,072$ |
| LN_1 ($\gamma, \beta$) | $2d$ | $2 \times 768 = 1,536$ |
| FFN $W_1$ | $d \times d_{ff}$ | $768 \times 3072 = 2,359,296$ |
| FFN $b_1$ | $d_{ff}$ | $3,072$ |
| FFN $W_2$ | $d_{ff} \times d$ | $3072 \times 768 = 2,359,296$ |
| FFN $b_2$ | $d$ | $768$ |
| LN_2 ($\gamma, \beta$) | $2d$ | $2 \times 768 = 1,536$ |
| **单层合计** | $\approx 12d^2$ | **7,087,872** |

### 5.2 完整模型参数

| 组件 | 参数量 | GPT-2 Small |
|------|--------|-------------|
| Token Embedding | $V \times d$ | $50257 \times 768 = 38,597,376$ |
| Position Embedding | $T_{max} \times d$ | $1024 \times 768 = 786,432$ |
| N 层 Block | $N \times 12d^2$ | $12 \times 7,087,872 = 85,054,464$ |
| Final LayerNorm | $2d$ | $1,536$ |
| LM Head | 与 Token Emb 共享 | $0$（权重共享）|
| **总计** | | **≈ 124,439,808 ≈ 124M** |

### 5.3 参数量的通用公式

对于 $N$ 层、隐藏维度 $d$、词表大小 $V$、上下文长度 $T_{max}$ 的 GPT：

$$
\text{Total} \approx V \cdot d + T_{max} \cdot d + N \cdot 12d^2
$$

当模型很大时，$N \cdot 12d^2$ 占绝对主导，Embedding 的占比很小。

---

## 六、Causal Language Modeling 训练目标的深入分析

### 6.1 目标函数

$$
\mathcal{L}_{\text{CLM}} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

等价于交叉熵损失：模型输出的概率分布 $P_\theta$ 与 one-hot 真实分布之间的 KL 散度。

### 6.2 为什么 CLM 足够强大？

**信息论视角**：

$$
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

最小化 CLM loss 等价于最大化数据的似然，等价于建模文本的联合概率分布。**任何可以用文本表述的任务，都隐含在这个分布中。**

### 6.3 Teacher Forcing

训练时，不管模型预测对不对，下一步的输入始终用真实 token（ground truth）：

```
输入:  [BOS] The  cat  sat  on   the
标签:   The  cat  sat  on   the  mat
       ↑    ↑    ↑    ↑    ↑    ↑
       logits[0] logits[1] ... logits[5] 分别预测这些

每个位置的 loss:
  L_0 = -log P("The"  | [BOS])
  L_1 = -log P("cat"  | [BOS], "The")
  L_2 = -log P("sat"  | [BOS], "The", "cat")
  ...
```

因为使用了 Causal Mask，**一次前向传播就能并行计算所有位置的 loss**——这是训练效率的关键。

### 6.4 训练 vs 推理的区别

| 维度 | 训练 | 推理 |
|------|------|------|
| 输入 | 完整序列（teacher forcing） | 逐 token 自回归 |
| Mask | 因果掩码（并行计算） | 因果掩码（KV Cache 优化） |
| 计算 | 所有位置并行 | 一次一个 token |
| Loss | 每个位置都计算 | 无 loss |

---

## 七、GPT 的关键设计决策总结

### 7.1 为什么选择可学习位置编码？

| 位置编码 | 优点 | 缺点 |
|---------|------|------|
| 正弦/余弦（Transformer 原始）| 理论上可外推 | 实际外推效果差 |
| 可学习（GPT-1/2/3）| 更灵活，效果更好 | 无法外推到训练长度之外 |
| RoPE（LLaMA）| 旋转编码，相对位置 | 需要额外实现——W4 将详细学习 |

### 7.2 为什么选择 GELU？

与 ReLU 对比：

| 特性 | ReLU | GELU |
|------|------|------|
| 定义 | $\max(0, x)$ | $x \cdot \Phi(x)$ |
| 零点处 | 不可导 | 平滑可导 |
| 负值 | 直接归零 | 保留微小负值 |
| 训练稳定性 | 有 dying ReLU 问题 | 更稳定 |

GELU 在负值区域不完全为零，保留了更多梯度信息，对大规模训练更友好。

### 7.3 权重共享（Weight Tying）

$$
\text{LM Head weight} = E_{\text{tok}}^T
$$

这不只是为了省参数。**数学直觉**：Token Embedding 将离散 token 映射到连续空间，LM Head 做的是逆映射——将连续隐状态映射回 token 空间。共享权重确保这两个映射一致。

Press & Wolf (2017) 证明权重共享在大多数情况下不降低性能甚至略有提升。

---

## 八、GPT 架构代码骨架预览（为 Day 3 做准备）

明天将从零手写，这里先给出骨架，建立全局认知：

```python
class CausalSelfAttention(nn.Module):
    """因果多头自注意力。"""
    def __init__(self, config):
        # W_Q, W_K, W_V, W_O 投影矩阵
        # 注册因果掩码 buffer
        ...
    
    def forward(self, x):
        # 投影 → 分头 → QK^T/√d_k → 因果掩码 → softmax → 加权求和 → 合并 → 输出投影
        ...

class GPTBlock(nn.Module):
    """一个 GPT Block = Pre-LN + CausalMHA + Pre-LN + FFN + 两个残差连接。"""
    def __init__(self, config):
        # LayerNorm × 2, CausalSelfAttention, FFN (MLP)
        ...
    
    def forward(self, x):
        # x = x + attn(ln1(x))
        # x = x + ffn(ln2(x))
        ...

class GPT(nn.Module):
    """完整的 GPT 模型。"""
    def __init__(self, config):
        # Token Embedding, Position Embedding
        # N × GPTBlock
        # Final LayerNorm, LM Head (权重共享)
        ...
    
    def forward(self, idx, targets=None):
        # Embedding → Blocks → LN → LM Head → logits
        # 如果有 targets, 计算 CLM loss
        ...
    
    def generate(self, idx, max_new_tokens, temperature, top_k):
        # 自回归生成循环
        ...
```

---

## 九、自检题

1. **画出完整的 GPT Block 数据流图**，标注每一步的 tensor shape（假设 batch=1, seq_len=T, d=768, h=12）。
2. **Pre-Norm 和 Post-Norm 的数学表达是什么？** 为什么 Pre-Norm 训练更稳定？
3. **精确计算 GPT-2 Medium (d=1024, N=24) 的参数量**，分别列出各组件。
4. **为什么 Causal Mask 用 $-\infty$ 而不是 $0$？** softmax 之后会变成什么？
5. **解释权重共享（Weight Tying）的数学含义和实际好处。**
6. **CLM 训练时，为什么一次前向传播就能计算所有位置的 loss？** 关键是什么？

---

## 十、产出要求

- [ ] 手绘一份 GPT 架构图，标注所有维度和关键操作
- [ ] 推导 GPT-2 Small 的参数量（精确到各组件）
- [ ] 写出 Causal Self-Attention 的完整数学公式（5 步）
- [ ] 能解释 Pre-Norm、GELU、Weight Tying 各自的设计动机
