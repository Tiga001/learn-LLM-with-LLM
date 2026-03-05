# Day 2：LLaMA 模型架构详解 — 四大改进的数学推导

> **目标**：逐模块拆解 LLaMA 相比 GPT 的四大架构改进（RMSNorm / RoPE / SwiGLU / GQA），从数学公式到代码骨架，全面理解每一处改进的设计动机。为 Day 3 手写实现打下坚实基础。

---

## 一、LLaMA 架构全景图

LLaMA 与 GPT 一样是 Decoder-only 架构，但每个组件都做了精心改进：

```
输入 token IDs: [x_1, x_2, ..., x_T]
       │
       ▼
┌──────────────────┐
│  Token Embedding  │  E_tok ∈ R^{V × d}
│  （无位置编码加法）│  ← 注意：RoPE 不在这里加！
└──────────────────┘
       │
       ▼  h_0 = E_tok[x]
┌──────────────────┐
│  LLaMA Block × N  │  ← 重复 N 次（LLaMA-7B: N=32）
│  ┌──────────────┐ │
│  │  RMSNorm 1   │ │  ← 替代 LayerNorm
│  │  GQA + RoPE  │ │  ← 替代 MHA + 可学习位置编码
│  │  + Residual    │ │
│  ├──────────────┤ │
│  │  RMSNorm 2   │ │
│  │  SwiGLU FFN  │ │  ← 替代 GELU FFN
│  │  + Residual    │ │
│  └──────────────┘ │
└──────────────────┘
       │
       ▼  h_N
┌──────────────────┐
│   RMSNorm         │  ← 最终归一化
│   LM Head (线性层)│  W ∈ R^{d × V}，不与 E_tok 共享
└──────────────────┘
       │
       ▼
  logits ∈ R^{T × V}  → softmax → P(x_{t+1} | x_{≤t})
```

**关键数字（LLaMA-7B）**：

| 超参数 | 符号 | LLaMA-7B | LLaMA-13B | LLaMA-33B | LLaMA-65B |
|--------|------|----------|-----------|-----------|-----------|
| 层数 | $N$ | 32 | 40 | 60 | 80 |
| 隐藏维度 | $d$ | 4096 | 5120 | 6656 | 8192 |
| 注意力头数 | $n_h$ | 32 | 40 | 52 | 64 |
| 每头维度 | $d_k = d/n_h$ | 128 | 128 | 128 | 128 |
| FFN 中间维度 | $d_{ff}$ | 11008 | 13824 | 17920 | 22016 |
| 上下文长度 | $T_{max}$ | 2048 | 2048 | 2048 | 2048 |
| 词表大小 | $V$ | 32000 | 32000 | 32000 | 32000 |
| GQA KV 头数 | $n_{kv}$ | 32 (MHA) | 40 (MHA) | 52 (MHA) | 64 (MHA) |
| 总参数量 | — | 6.7B | 13.0B | 32.5B | 65.2B |

注：LLaMA-1 全部使用 MHA（每个头有独立的 KV），LLaMA-2 的 70B 引入了 GQA（8 个 KV 头）。

---

## 二、改进一：RMSNorm — 更简单更快的归一化

### 2.1 回顾 LayerNorm

GPT 使用的 LayerNorm 定义为：

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ — 均值
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$ — 方差
- $\gamma, \beta \in \mathbb{R}^d$ — 可学习的缩放和偏移参数

LayerNorm 做了两件事：**中心化**（减均值）+ **缩放**（除标准差）。

### 2.2 RMSNorm 的定义

RMSNorm（Root Mean Square Layer Normalization）去掉了中心化，只保留缩放：

$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x) + \epsilon}$$

其中：

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

注意：
- **没有**减均值（$\mu$）操作
- **没有** $\beta$（偏移参数），只有 $\gamma$（缩放参数）
- 分母是 RMS（均方根），而非标准差

### 2.3 数学对比

| 操作 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算均值 $\mu$ | ✓ | ✗ |
| 中心化 $x - \mu$ | ✓ | ✗ |
| 计算方差 $\sigma^2$ | ✓ | ✗ |
| 计算 RMS | ✗ | ✓ |
| 缩放 $\gamma$ | ✓ | ✓ |
| 偏移 $\beta$ | ✓ | ✗ |
| 可学习参数 | $2d$ ($\gamma, \beta$) | $d$ ($\gamma$) |

**为什么去掉中心化？**

Zhang & Sennrich (2019) 的实验表明：
- LayerNorm 的归一化效果主要来自缩放（re-scaling），而非中心化（re-centering）
- 去掉中心化后，训练效果基本不变，但计算更快（少一次 mean 和一次减法）
- RMSNorm 比 LayerNorm 快约 **10-15%**

### 2.4 RMSNorm 的代码实现

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm
```

**等价的更高效写法**（避免创建中间张量）：

```python
def forward(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```

`torch.rsqrt` 计算 $1/\sqrt{x}$，比先 `sqrt` 再除法更高效。

---

## 三、改进二：RoPE — 旋转位置编码

### 3.1 问题：为什么需要改进位置编码？

GPT-1/2/3 使用可学习的绝对位置编码：

$$h_0 = E_{\text{tok}}[x] + E_{\text{pos}}[0:T]$$

这有两个问题：
1. **无法外推**：训练时最大长度是 2048，推理时遇到位置 2049 就没有对应的编码了
2. **绝对位置 vs 相对位置**："the cat" 在位置 (0,1) 和位置 (100,101) 的关系应该是一样的，但绝对编码无法表达这一点

### 3.2 RoPE 的核心思想

RoPE（Rotary Position Embedding）的核心思想：

> **不是给 token 加上位置信号，而是根据位置旋转 Query 和 Key 向量。两个位置的点积自然只依赖于它们的相对距离。**

数学表达：

$$\text{RoPE}(x_m, m) = x_m \cdot e^{im\theta}$$

其中 $m$ 是位置，$\theta$ 是频率参数。这是一个复数域的旋转操作。

### 3.3 为什么旋转能编码相对位置？

设两个位置 $m, n$ 的 Query 和 Key 分别为：

$$q_m = \text{RoPE}(W_Q x_m, m), \quad k_n = \text{RoPE}(W_K x_n, n)$$

它们的内积：

$$q_m^T k_n = \text{Re}[(W_Q x_m \cdot e^{im\theta})^* \cdot (W_K x_n \cdot e^{in\theta})]$$

$$= \text{Re}[(W_Q x_m)^* \cdot W_K x_n \cdot e^{i(n-m)\theta}]$$

**关键**：内积只依赖于 $(n - m)$，即相对距离，而非绝对位置！

### 3.4 RoPE 的实际计算

在实际实现中，RoPE 将 $d$ 维向量两两配对，每对做一个 2D 旋转：

$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

其中频率参数：

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1$$

展开为完整的旋转矩阵（以 $d=4$ 为例）：

$$R_m = \begin{pmatrix} \cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 \\ \sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 \\ 0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) \\ 0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) \end{pmatrix}$$

### 3.5 RoPE 的高效实现（复数形式）

LLaMA 官方实现使用复数形式，更简洁高效：

```python
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)                    # (seq_len, dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^{i*freq}
    return freqs_cis  # (seq_len, dim/2), complex64

def apply_rotary_emb(xq, xk, freqs_cis):
    # xq, xk: (batch, seq_len, n_heads, d_head) → 转为复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]  # 广播到多头
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**直觉**：
- 将 $d$ 维实向量看成 $d/2$ 维复向量
- 乘以 $e^{im\theta}$（单位复数）就是旋转
- 不同维度对使用不同的频率 $\theta_i$

### 3.6 RoPE 的关键特性

| 特性 | 说明 |
|------|------|
| **相对位置编码** | 内积只依赖相对距离 $n-m$ |
| **远程衰减** | 相对距离越大，注意力分数自然衰减（高频分量快速振荡导致抵消） |
| **理论上可外推** | 旋转角度可以取任意值，不受训练长度限制 |
| **不增加参数** | RoPE 无可学习参数（纯数学变换） |
| **不需要 Position Embedding** | 位置信息编码在旋转中，不需要额外的位置嵌入层 |

Day 4 将深入推导 RoPE 的复数数学、远程衰减性质和外推问题。

---

## 四、改进三：SwiGLU — 门控 FFN

### 4.1 回顾 GPT 的 FFN

GPT 的 Position-wise FFN：

$$\text{FFN}_{\text{GPT}}(x) = \text{GELU}(xW_1 + b_1) \cdot W_2 + b_2$$

- 两个线性层：$W_1 \in \mathbb{R}^{d \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d}$
- 参数量：$2 \times d \times d_{ff} \approx 8d^2$（当 $d_{ff} = 4d$）

### 4.2 GLU 门控机制

GLU（Gated Linear Unit）是一种门控机制，核心思想是让模型学习"哪些信息应该通过"：

$$\text{GLU}(x) = (xW_1) \otimes \sigma(xW_{\text{gate}})$$

其中 $\sigma$ 是激活函数（sigmoid / swish / gelu 等），$\otimes$ 是逐元素乘法。

**直觉**：$xW_{\text{gate}}$ 经过激活函数后产生一个 0~1 的"门"，控制 $xW_1$ 中哪些维度的信息能通过。

### 4.3 SwiGLU 的定义

SwiGLU 使用 Swish（SiLU）作为门控激活函数：

$$\text{SwiGLU}(x) = (\text{SiLU}(xW_{\text{gate}})) \otimes (xW_{\text{up}})$$

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

完整的 SwiGLU FFN：

$$\text{FFN}_{\text{LLaMA}}(x) = (\text{SiLU}(xW_{\text{gate}}) \otimes xW_{\text{up}}) \cdot W_{\text{down}}$$

三个权重矩阵（**无 bias**）：
- $W_{\text{gate}} \in \mathbb{R}^{d \times d_{ff}}$ — 门控投影
- $W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}}$ — 上投影
- $W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$ — 下投影

### 4.4 SiLU / Swish 激活函数

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

```
x < 0 时: SiLU(x) ≈ 0（类似 ReLU）
x > 0 时: SiLU(x) ≈ x（类似恒等映射）
x = 0 时: SiLU(0) = 0
最小值: SiLU(-1.28) ≈ -0.28（不像 ReLU 那样严格非负）
```

SiLU 的优势：
- 比 ReLU 平滑（处处可导）
- 比 GELU 计算更简单（不需要 erf）
- 允许小的负值通过，保留更多梯度信息

### 4.5 参数量分析

由于 SwiGLU 有 3 个权重矩阵（而非 2 个），为保持总参数量不变，需要调整 $d_{ff}$：

**GPT FFN 参数量**：$2 \times d \times d_{ff} = 2 \times d \times 4d = 8d^2$

**SwiGLU FFN 参数量**：$3 \times d \times d_{ff}$

令两者相等：

$$3 \times d \times d_{ff}' = 8d^2 \implies d_{ff}' = \frac{8d}{3} \approx 2.67d$$

实际取 $\frac{2}{3} \times 4d$ 并向上取整到 256 的倍数。

验证 LLaMA-7B：

$$d_{ff} = \text{round}_{256}\left(\frac{2}{3} \times 4 \times 4096\right) = \text{round}_{256}(10922.67) = 11008$$

### 4.6 SwiGLU FFN 代码骨架

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

**数据流**（LLaMA-7B, $d=4096, d_{ff}=11008$）：

```
x: (B, T, 4096)
  → w_gate: (B, T, 4096) → (B, T, 11008) → SiLU → (B, T, 11008)
  → w_up:   (B, T, 4096) → (B, T, 11008)
  → 逐元素乘法: (B, T, 11008)
  → w_down:  (B, T, 11008) → (B, T, 4096)
```

---

## 五、改进四：GQA — 分组查询注意力

### 5.1 回顾 MHA（Multi-Head Attention）

GPT 和 LLaMA-1 使用标准 MHA，每个注意力头有独立的 Q、K、V：

$$\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$$

参数量：$W_Q, W_K, W_V, W_O$ 各为 $d \times d$，共 $4d^2$。

**问题**：推理时需要缓存所有头的 K、V（KV Cache），显存开销大。

### 5.2 MQA（Multi-Query Attention）

Shazeer (2019) 提出 MQA：所有 Q 头共享同一组 K、V：

$$\text{head}_i = \text{Attention}(QW_Q^i, KW_K^{\text{shared}}, VW_V^{\text{shared}})$$

KV Cache 减少到 $1/n_h$。但缺点是**质量下降**。

### 5.3 GQA（Grouped Query Attention）

GQA 是 MHA 和 MQA 的折中。将 $n_h$ 个 Q 头分成 $n_g$ 组，每组共享一组 KV：

```
MHA (n_h=8, n_kv=8):  每个 Q 头有独立的 KV
  Q₁→KV₁, Q₂→KV₂, Q₃→KV₃, Q₄→KV₄, Q₅→KV₅, Q₆→KV₆, Q₇→KV₇, Q₈→KV₈

GQA (n_h=8, n_kv=2):  每 4 个 Q 头共享 1 组 KV
  Q₁,Q₂,Q₃,Q₄ → KV₁     Q₅,Q₆,Q₇,Q₈ → KV₂

MQA (n_h=8, n_kv=1):  所有 Q 头共享 1 组 KV
  Q₁,Q₂,Q₃,Q₄,Q₅,Q₆,Q₇,Q₈ → KV₁
```

### 5.4 GQA 的参数与 Cache 分析

设 $n_h$ 为 Q 头数，$n_{kv}$ 为 KV 头数，$d_k = d / n_h$：

| 方案 | KV 参数量 | KV Cache / token | LLaMA-2 70B 设定 |
|------|----------|-----------------|-----------------|
| MHA ($n_{kv} = n_h$) | $2 \times n_h \times d_k^2 = 2d^2$ | $2 \times n_h \times d_k$ | — |
| GQA ($n_{kv} < n_h$) | $2 \times n_{kv} \times d \times d_k$ | $2 \times n_{kv} \times d_k$ | $n_h=64, n_{kv}=8$ |
| MQA ($n_{kv} = 1$) | $2 \times d \times d_k$ | $2 \times d_k$ | — |

LLaMA-2 70B 使用 GQA ($n_h=64, n_{kv}=8$)，KV Cache 减少到 MHA 的 $8/64 = 1/8$。

### 5.5 GQA 的实现

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # 每组的 Q 头数
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # 应用 RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # 扩展 KV 头以匹配 Q 头数量
        k = repeat_kv(k, self.n_rep)  # (B, T, n_heads, head_dim)
        v = repeat_kv(v, self.n_rep)

        # 标准 Attention 计算...
        ...

def repeat_kv(x, n_rep):
    """将 KV 头重复 n_rep 次以匹配 Q 头数量。"""
    if n_rep == 1:
        return x
    B, T, n_kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(B, T, n_kv_heads, n_rep, head_dim)
    return x.reshape(B, T, n_kv_heads * n_rep, head_dim)
```

---

## 六、完整 LLaMA Block 数学表达

一个 LLaMA Block 的完整公式（Pre-RMSNorm + GQA + RoPE + SwiGLU）：

$$h_l' = h_l + \text{GQA}(\text{RMSNorm}(h_l), \text{freqs\_cis})$$

$$h_{l+1} = h_l' + \text{SwiGLU\_FFN}(\text{RMSNorm}(h_l'))$$

其中 GQA 内部：

$$Q = \text{RMSNorm}(h_l) \cdot W_Q, \quad K = \text{RMSNorm}(h_l) \cdot W_K, \quad V = \text{RMSNorm}(h_l) \cdot W_V$$

$$Q' = \text{RoPE}(Q, \text{pos}), \quad K' = \text{RoPE}(K, \text{pos})$$

$$\text{GQA Output} = \text{softmax}\left(\frac{Q' {K'}^T}{\sqrt{d_k}} + \text{CausalMask}\right) \cdot V \cdot W_O$$

---

## 七、完整数据流追踪

以 LLaMA-7B 为例，输入 `"The cat sat"` 的数据流：

```
输入 token IDs: [450, 6635, 3290]      # SentencePiece tokenize
                 shape: (1, 3)

  ┌─── Token Embedding ─────────────────────────┐
  │ E_tok[450], E_tok[6635], E_tok[3290]        │
  │ shape: (1, 3, 4096)                          │
  │ 注意：没有加位置编码！RoPE 在 Attention 中处理  │
  └─────────────────────────────────────────────┘

  h_0: shape (1, 3, 4096)

  ─── Block 1 ──────────────────────────────────────
  │ RMSNorm_1:    (1, 3, 4096) → (1, 3, 4096)
  │ Q projection: (1, 3, 4096) → (1, 3, 32, 128) → 32 heads
  │ K projection: (1, 3, 4096) → (1, 3, 32, 128) → 32 KV heads (MHA)
  │ V projection: (1, 3, 4096) → (1, 3, 32, 128)
  │
  │ Apply RoPE to Q, K:
  │   Q: (1, 3, 32, 128) → 旋转 → (1, 3, 32, 128)
  │   K: (1, 3, 32, 128) → 旋转 → (1, 3, 32, 128)
  │
  │ Attention:
  │   Q·K^T/√128: (1, 32, 3, 128)×(1, 32, 128, 3) → (1, 32, 3, 3)
  │   + Causal Mask + Softmax → (1, 32, 3, 3)
  │   × V: (1, 32, 3, 3)×(1, 32, 3, 128) → (1, 32, 3, 128)
  │   Concat: → (1, 3, 4096)
  │   W_O: → (1, 3, 4096)
  │ + Residual → (1, 3, 4096)
  │
  │ RMSNorm_2:    (1, 3, 4096) → (1, 3, 4096)
  │ SwiGLU FFN:
  │   W_gate + SiLU: (1, 3, 4096) → (1, 3, 11008) → SiLU
  │   W_up:          (1, 3, 4096) → (1, 3, 11008)
  │   逐元素乘:      (1, 3, 11008)
  │   W_down:        (1, 3, 11008) → (1, 3, 4096)
  │ + Residual → (1, 3, 4096)
  ──────────────────────────────────────────────────

  ... (重复 32 次) ...

  h_32: shape (1, 3, 4096)

  Final RMSNorm: (1, 3, 4096) → (1, 3, 4096)
  LM Head:       (1, 3, 4096) × (4096, 32000) → (1, 3, 32000)

  logits[:, -1, :] → softmax → P(next_token | "The cat sat")
```

---

## 八、参数量计算

### 8.1 单层 Block 参数（LLaMA-7B）

| 组件 | 参数量公式 | LLaMA-7B |
|------|----------|----------|
| RMSNorm_1 ($\gamma$) | $d$ | 4,096 |
| $W_Q$ | $d \times n_h \times d_k = d^2$ | 16,777,216 |
| $W_K$ | $d \times n_{kv} \times d_k$ | 16,777,216 |
| $W_V$ | $d \times n_{kv} \times d_k$ | 16,777,216 |
| $W_O$ | $n_h \times d_k \times d = d^2$ | 16,777,216 |
| RMSNorm_2 ($\gamma$) | $d$ | 4,096 |
| $W_{\text{gate}}$ | $d \times d_{ff}$ | 45,088,768 |
| $W_{\text{up}}$ | $d \times d_{ff}$ | 45,088,768 |
| $W_{\text{down}}$ | $d_{ff} \times d$ | 45,088,768 |
| **单层合计** | | **~201,326,592 ≈ 201M** |

### 8.2 完整模型参数

| 组件 | 参数量 | LLaMA-7B |
|------|--------|----------|
| Token Embedding | $V \times d$ | $32000 \times 4096 = 131M$ |
| 32 层 Block | $32 \times 201M$ | $6,442M$ |
| Final RMSNorm | $d$ | $4K$ |
| LM Head | $d \times V$ | $131M$ |
| **总计** | | **≈ 6,738M ≈ 6.7B** |

注意：LLaMA 不共享 Embedding 和 LM Head 的权重，所以 LM Head 有独立的 131M 参数。

### 8.3 GPT vs LLaMA 参数量分布对比

| 组件 | GPT-2 Small (124M) | LLaMA-7B (6.7B) |
|------|-------------------|-----------------|
| Embedding | 31.2% | 1.9% |
| Attention | 27.7% | 31.7% (4d² per layer) |
| FFN | 38.1% | 64.4% (3×d×d_ff per layer) |
| Norm | 0.03% | 0.004% |
| LM Head | 共享 (0%) | 1.9% |

**关键观察**：大模型中 FFN 占比远高于小模型。SwiGLU 的 3 个权重矩阵使 FFN 成为参数量的主要来源。

---

## 九、LLaMA vs GPT 代码级完整对比

```python
# ============================================================
# GPT Block
# ============================================================
class GPTBlock(nn.Module):
    def __init__(self, config):
        self.ln_1 = nn.LayerNorm(config.n_embd)              # LayerNorm
        self.attn = CausalSelfAttention(config)               # MHA
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),     # W1 + bias
            nn.GELU(),                                         # GELU
            nn.Linear(4 * config.n_embd, config.n_embd),     # W2 + bias
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# ============================================================
# LLaMA Block
# ============================================================
class LLaMABlock(nn.Module):
    def __init__(self, config):
        self.attention_norm = RMSNorm(config.dim)             # RMSNorm
        self.attention = GQAttention(config)                   # GQA + RoPE
        self.ffn_norm = RMSNorm(config.dim)
        self.feed_forward = SwiGLUFFN(config.dim, config.ffn_dim)  # SwiGLU

    def forward(self, x, freqs_cis, mask=None):
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
```

---

## 十、自检题

1. **RMSNorm 与 LayerNorm 的数学区别是什么？** 为什么去掉中心化不影响效果？
2. **RoPE 如何实现"只编码相对位置"？** 写出数学证明。
3. **SwiGLU 中的 SiLU 激活函数是什么？** 它与 GELU、ReLU 的对比。
4. **为什么 SwiGLU FFN 需要 3 个权重矩阵？** $d_{ff}$ 为什么不再是 $4d$？
5. **GQA 是 MHA 和 MQA 的什么关系？** 画图说明 $n_h=8, n_{kv}=2$ 的情况。
6. **精确计算 LLaMA-13B 的单层参数量和总参数量。**
7. **对比 GPT Block 和 LLaMA Block 的完整数据流。** 哪些步骤是新增的？

---

## 十一、产出要求

- [ ] 画出 LLaMA Block 的详细架构图，标注所有维度
- [ ] 手写 RMSNorm 的数学公式和 PyTorch 实现
- [ ] 手写 SwiGLU FFN 的数学公式和代码骨架
- [ ] 画出 GQA 的 Q-KV 分组图（$n_h=8, n_{kv}=2$）
- [ ] 推导 LLaMA-7B 的 $d_{ff} = 11008$
- [ ] 对比 GPT Block 和 LLaMA Block 的代码差异
