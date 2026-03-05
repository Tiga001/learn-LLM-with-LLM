# Day 4：RoPE 旋转位置编码 — 面试高频，必须手撕！

> **目标**：从复数域出发，严格推导 RoPE 的数学原理，理解旋转矩阵的实际含义；实现完整的 RoPE 代码并可视化其关键性质；深入分析远程衰减性和外推问题，了解 NTK-aware / YaRN 等扩展方案。

---

## 一、位置编码的演进：从绝对到旋转

### 1.1 为什么位置编码如此重要？

Transformer 的 Self-Attention 本身是 **置换不变的（permutation invariant）**——如果你打乱输入序列的顺序，不加位置编码的 Attention 输出也只是相应地打乱，模型根本"不知道"顺序信息。

```
没有位置编码的 Attention:
  "猫 吃 鱼" 和 "鱼 吃 猫" 的 Attention 分数相同（只是 permute 了）
  → 模型无法区分语序 → 灾难性的

加了位置编码:
  "猫 吃 鱼" 中，"猫"在位置 0，"吃"在位置 1，"鱼"在位置 2
  → 模型能利用位置信息理解语义
```

### 1.2 位置编码方案总览

| 方案 | 代表模型 | 类型 | 核心思想 | 局限 |
|------|---------|------|---------|------|
| 正弦位置编码 | Transformer (原始) | 绝对 | $\sin/\cos$ 函数编码位置 | 理论上可外推，实践中效果有限 |
| 可学习绝对编码 | GPT-1/2/3, BERT | 绝对 | 学习每个位置的嵌入向量 | 无法外推到训练未见过的位置 |
| 相对位置编码 | T5, Transformer-XL | 相对 | 编码 token 之间的距离 | 实现复杂，计算开销大 |
| ALiBi | BLOOM | 相对 | 在注意力分数上加线性偏置 | 简单但表达能力有限 |
| **RoPE** | **LLaMA, Qwen, Mistral** | **相对** | **在复数域旋转 Q/K** | **优雅、高效、业界标准** |

**结论**：RoPE 已成为当今大语言模型的事实标准位置编码方案。

---

## 二、RoPE 的数学推导

### 2.1 出发点：我们需要什么性质？

设输入序列为 $x_1, x_2, \ldots, x_T$，我们希望找到一个函数 $f(x, m)$，满足：

$$\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)$$

即：**Query 和 Key 的内积只依赖于它们的内容 $(q, k)$ 和相对位置 $(m-n)$，而不依赖于绝对位置 $m, n$。**

这就是 RoPE 要解决的核心数学问题。

### 2.2 二维情形：从复数域开始

**关键洞察**：在复数域中，乘以单位复数就是旋转，而旋转的角度差就是相对位置。

将 2D 向量视为复数。设 $q, k \in \mathbb{C}$（二维情形），定义：

$$f(q, m) = q \cdot e^{im\theta}$$

其中 $\theta$ 是预定义的频率参数。

验证内积性质：

$$\langle f(q, m), f(k, n) \rangle = \text{Re}[f(q, m)^* \cdot f(k, n)]$$

$$= \text{Re}[(q \cdot e^{im\theta})^* \cdot (k \cdot e^{in\theta})]$$

$$= \text{Re}[q^* \cdot e^{-im\theta} \cdot k \cdot e^{in\theta}]$$

$$= \text{Re}[q^* k \cdot e^{i(n-m)\theta}]$$

**内积只依赖于 $(n-m)$**。证毕。

### 2.3 展开为实数运算

$e^{im\theta} = \cos(m\theta) + i\sin(m\theta)$

设 $q = q_0 + iq_1$（对应 2D 向量 $(q_0, q_1)$），则：

$$f(q, m) = (q_0 + iq_1)(\cos(m\theta) + i\sin(m\theta))$$

$$= (q_0\cos(m\theta) - q_1\sin(m\theta)) + i(q_1\cos(m\theta) + q_0\sin(m\theta))$$

写成矩阵形式：

$$f\begin{pmatrix} q_0 \\ q_1 \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}$$

这就是一个 **2D 旋转矩阵**！旋转角度为 $m\theta$。

### 2.4 推广到高维：两两配对旋转

对于 $d$ 维向量，RoPE 将其两两配对，每对独立做 2D 旋转：

$$\text{RoPE}(x, m) = R_m \cdot x$$

其中旋转矩阵 $R_m$ 为分块对角矩阵（以 $d=6$ 为例）：

$$R_m = \begin{pmatrix} \cos(m\theta_0) & -\sin(m\theta_0) & & & & \\ \sin(m\theta_0) & \cos(m\theta_0) & & & & \\ & & \cos(m\theta_1) & -\sin(m\theta_1) & & \\ & & \sin(m\theta_1) & \cos(m\theta_1) & & \\ & & & & \cos(m\theta_2) & -\sin(m\theta_2) \\ & & & & \sin(m\theta_2) & \cos(m\theta_2) \end{pmatrix}$$

频率参数：

$$\theta_i = \frac{1}{10000^{2i/d}}, \quad i = 0, 1, \ldots, d/2 - 1$$

**为什么用 10000 作为基数？**

- 低维（$i$ 小）：$\theta_i$ 大 → 旋转速度快 → 捕捉近距离位置关系
- 高维（$i$ 大）：$\theta_i$ 小 → 旋转速度慢 → 捕捉远距离位置关系
- 10000 是一个经验值，使得在 2048 长度内频率分布合理

### 2.5 等价的乘法形式

旋转矩阵可以改写为更高效的逐元素运算（避免矩阵乘法）：

$$\text{RoPE}(x, m) = x \odot \cos(\Theta_m) + \text{rotate\_half}(x) \odot \sin(\Theta_m)$$

其中：
- $\Theta_m = [m\theta_0, m\theta_0, m\theta_1, m\theta_1, \ldots, m\theta_{d/2-1}, m\theta_{d/2-1}]$
- $\text{rotate\_half}(x)$ 将 $x$ 中每对相邻元素交换并取反：$[x_0, x_1, x_2, x_3, \ldots] \to [-x_1, x_0, -x_3, x_2, \ldots]$

---

## 三、RoPE 的两种实现方式

### 3.1 方式一：复数形式（LLaMA 官方实现）

这是最简洁的实现方式，利用 PyTorch 的复数运算：

```python
import torch

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    预计算 RoPE 的复数频率。
    
    Args:
        dim: 每个注意力头的维度 d_head
        seq_len: 最大序列长度
        theta: 频率基数（默认 10000）
    
    Returns:
        freqs_cis: (seq_len, dim//2) 的复数张量
    """
    # Step 1: 计算频率 θ_i = 1 / (10000^(2i/d))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # freqs: (dim//2,)  例如 dim=128 → (64,)
    
    # Step 2: 计算外积 m * θ_i，得到每个位置每个维度的旋转角度
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)  # (seq_len, dim//2)
    
    # Step 3: 转为复数 e^{i * m * θ_i}
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # polar(r, angle) = r * e^{i*angle}
    # freqs_cis: (seq_len, dim//2), dtype=complex64
    
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor, 
    freqs_cis: torch.Tensor
) -> tuple:
    """
    将 RoPE 应用到 Query 和 Key 上。
    
    Args:
        xq: (batch, seq_len, n_heads, head_dim) — Query
        xk: (batch, seq_len, n_kv_heads, head_dim) — Key
        freqs_cis: (seq_len, head_dim//2) — 预计算的复数频率
    
    Returns:
        xq_out, xk_out: 旋转后的 Q, K，形状不变
    """
    # Step 1: 将实数张量视为复数
    # (B, T, H, D) → (B, T, H, D//2)  [每两个实数组成一个复数]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Step 2: 调整 freqs_cis 形状以广播
    # (T, D//2) → (1, T, 1, D//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    # Step 3: 复数乘法 = 旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**数据流追踪**（以 LLaMA-7B 为例，$d_{\text{head}} = 128$）：

```
xq: (B, T, 32, 128)  → reshape → (B, T, 32, 64, 2)  → view_as_complex → (B, T, 32, 64)
freqs_cis: (T, 64)    → unsqueeze → (1, T, 1, 64)
xq_ * freqs_cis:      → (B, T, 32, 64) [complex]
→ view_as_real:        → (B, T, 32, 64, 2)
→ flatten(-2):         → (B, T, 32, 128)
```

### 3.2 方式二：旋转矩阵形式（HuggingFace 实现）

```python
def rotary_emb_huggingface(x, cos, sin):
    """
    HuggingFace Transformers 中的 RoPE 实现方式。
    
    x:   (batch, n_heads, seq_len, head_dim)
    cos: (seq_len, head_dim)
    sin: (seq_len, head_dim)
    """
    def rotate_half(x):
        """将 x 的后半部分取反并与前半部分交换。"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    return (x * cos) + (rotate_half(x) * sin)
```

**两种实现的等价性证明**：

设 $x = [x_0, x_1]$，旋转角 $\alpha = m\theta$：

- 复数形式：$(x_0 + ix_1) \cdot (\cos\alpha + i\sin\alpha) = (x_0\cos\alpha - x_1\sin\alpha) + i(x_0\sin\alpha + x_1\cos\alpha)$
- 旋转矩阵形式：$[x_0, x_1] \odot [\cos\alpha, \cos\alpha] + [-x_1, x_0] \odot [\sin\alpha, \sin\alpha] = [x_0\cos\alpha - x_1\sin\alpha, x_0\sin\alpha + x_1\cos\alpha]$

结果完全一致。

### 3.3 两种实现方式的对比

| 维度 | 复数形式 | 旋转矩阵形式 |
|------|---------|-------------|
| 代码量 | 更少 | 稍多 |
| 可读性 | 需要理解复数 | 更直观 |
| 计算效率 | 略好（复数乘法优化） | 稍差（多一次 rotate_half） |
| PyTorch 支持 | 需要 complex 类型 | 纯实数运算 |
| 实际采用 | LLaMA 官方、vLLM | HuggingFace Transformers |

---

## 四、RoPE 的关键性质

### 4.1 性质一：相对位置编码

**定理**：对任意位置 $m, n$，有：

$$\langle R_m q, R_n k \rangle = \langle R_{m-n} q, k \rangle$$

**证明**：

$$\langle R_m q, R_n k \rangle = (R_m q)^T (R_n k) = q^T R_m^T R_n k$$

由于旋转矩阵是正交矩阵（$R^T R = I$，$R^T = R^{-1}$），且 $R_m^{-1} = R_{-m}$：

$$= q^T R_{-m} R_n k = q^T R_{n-m} k = \langle R_{n-m} q, k \rangle$$

等价地写成：$\langle q, R_{m-n} k \rangle$。

**直觉**：两个向量分别旋转 $m\theta$ 和 $n\theta$，它们之间的角度差为 $(m-n)\theta$，只取决于相对距离。

### 4.2 性质二：远程衰减（Long-term Decay）

这是 RoPE 的一个重要但不太直观的性质：**随着相对距离 $|m-n|$ 增大，注意力分数趋向于减小。**

**数学分析**：

考虑在随机初始化条件下，$q$ 和 $k$ 的各维度独立同分布。旋转后的内积：

$$\langle R_m q, R_n k \rangle = \sum_{i=0}^{d/2-1} \left[ (q_{2i}k_{2i} + q_{2i+1}k_{2i+1})\cos(\Delta m \cdot \theta_i) + (q_{2i}k_{2i+1} - q_{2i+1}k_{2i})\sin(\Delta m \cdot \theta_i) \right]$$

其中 $\Delta m = m - n$。

当 $\Delta m$ 增大时：
- $\cos(\Delta m \cdot \theta_i)$ 和 $\sin(\Delta m \cdot \theta_i)$ 在不同维度 $i$ 上快速振荡
- 对于高频维度（$\theta_i$ 大），振荡最快，贡献互相抵消
- 整体内积的期望保持为 0，但**方差减小**

```
相对距离小 (|m-n| 小):
  各维度的 cos/sin 值比较一致 → 内积波动大 → 可能有大的注意力分数

相对距离大 (|m-n| 大):
  各维度的 cos/sin 值快速振荡 → 内积趋向平均 → 注意力分数减弱

直觉：就像多个不同频率的波叠加，频率越多、越分散，振幅越互相抵消
```

### 4.3 性质三：无可学习参数

RoPE 是一个**纯数学变换**，不引入任何可学习参数：
- 频率 $\theta_i$ 是预定义的
- 旋转矩阵 $R_m$ 是确定性计算的
- 相比可学习位置编码，减少了 $T \times d$ 的参数量

### 4.4 性质四：RoPE 只作用于 Q 和 K

RoPE 不作用于 Value（V）和 Embedding，只旋转 Attention 计算中的 Q 和 K：

```python
# 正确的 RoPE 使用位置
q = self.wq(x)   # 线性投影
k = self.wk(x)
v = self.wv(x)

q, k = apply_rotary_emb(q, k, freqs_cis)  # 只旋转 Q, K

scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
output = torch.matmul(F.softmax(scores, dim=-1), v)  # V 不旋转
```

**为什么 V 不需要旋转？**

位置信息的目的是影响"注意力分布"（哪些 token 关注哪些 token），而注意力分布由 $QK^T$ 决定。V 是被加权求和的内容，不需要位置信息。

---

## 五、可视化 RoPE

### 5.1 频率分布可视化

```python
import matplotlib.pyplot as plt
import numpy as np

dim = 128  # d_head
theta_base = 10000.0

# 计算频率
i = np.arange(0, dim // 2)
freqs = 1.0 / (theta_base ** (2 * i / dim))

# 计算波长（一个完整旋转周期对应的 token 数）
wavelengths = 2 * np.pi / freqs

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 频率分布
axes[0].semilogy(i, freqs, 'b-', linewidth=2)
axes[0].set_xlabel('维度对索引 i')
axes[0].set_ylabel('频率 θ_i (log)')
axes[0].set_title('RoPE 频率分布')
axes[0].grid(True, alpha=0.3)

# 波长分布
axes[1].semilogy(i, wavelengths, 'r-', linewidth=2)
axes[1].set_xlabel('维度对索引 i')
axes[1].set_ylabel('波长 (tokens)')
axes[1].set_title('RoPE 波长分布')
axes[1].axhline(y=2048, color='gray', linestyle='--', label='训练长度 2048')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rope_freq_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"最高频率 θ_0 = {freqs[0]:.4f}, 波长 = {wavelengths[0]:.1f} tokens")
print(f"最低频率 θ_{dim//2-1} = {freqs[-1]:.6f}, 波长 = {wavelengths[-1]:.0f} tokens")
```

**预期输出**：

```
最高频率 θ_0 = 1.0000, 波长 = 6.3 tokens
最低频率 θ_63 = 0.000100, 波长 = 62832 tokens
```

- 低维度对：波长约 6 个 token → 对局部位置变化敏感
- 高维度对：波长超过 60000 个 token → 对远距离位置变化敏感
- 这种多尺度设计使 RoPE 能同时编码局部和全局位置信息

### 5.2 旋转轨迹可视化

```python
import matplotlib.pyplot as plt
import numpy as np

seq_len = 64
dim = 128
theta_base = 10000.0

# 选择 3 个不同频率的维度对
dims_to_show = [0, 16, 63]  # 高频、中频、低频

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, dim_idx in zip(axes, dims_to_show):
    freq = 1.0 / (theta_base ** (2 * dim_idx / dim))
    positions = np.arange(seq_len)
    angles = positions * freq
    
    x = np.cos(angles)
    y = np.sin(angles)
    
    # 画旋转轨迹
    scatter = ax.scatter(x, y, c=positions, cmap='viridis', s=30)
    ax.plot(x, y, 'gray', alpha=0.3)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title(f'维度对 i={dim_idx}\nθ={freq:.4f}, 波长={2*np.pi/freq:.0f}')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='位置')

plt.suptitle('RoPE 在单位圆上的旋转轨迹', fontsize=14)
plt.tight_layout()
plt.savefig('rope_rotation_trajectory.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.3 远程衰减可视化

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def compute_rope_attention_pattern(dim=128, seq_len=256, theta=10000.0):
    """计算纯 RoPE 的注意力分数模式（不含内容信息）。"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)  # (T, D//2)
    
    # 构造单位 Q 和 K（去除内容影响，只看位置）
    # 实际上计算 sum_i cos((m-n) * theta_i)
    cos_freqs = torch.cos(freqs)  # (T, D//2)
    sin_freqs = torch.sin(freqs)  # (T, D//2)
    
    # RoPE 内积 = sum_i [cos(m*θ_i)cos(n*θ_i) + sin(m*θ_i)sin(n*θ_i)]
    #           = sum_i cos((m-n)*θ_i)
    scores = cos_freqs @ cos_freqs.T + sin_freqs @ sin_freqs.T
    return scores


scores = compute_rope_attention_pattern(dim=128, seq_len=256)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 注意力分数热力图
im = axes[0].imshow(scores.numpy(), cmap='RdBu_r', aspect='auto')
axes[0].set_xlabel('Key 位置')
axes[0].set_ylabel('Query 位置')
axes[0].set_title('RoPE 位置注意力分数')
plt.colorbar(im, ax=axes[0])

# 固定 Query 位置，看注意力随距离的变化
query_pos = 128
relative_scores = scores[query_pos, :].numpy()
axes[1].plot(np.arange(256) - query_pos, relative_scores, 'b-', linewidth=1)
axes[1].set_xlabel('相对距离 (n - m)')
axes[1].set_ylabel('注意力分数')
axes[1].set_title(f'Query 位置={query_pos} 的注意力分数 vs 相对距离')
axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rope_decay.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 六、RoPE 的外推问题

### 6.1 什么是外推问题？

虽然 RoPE 理论上可以处理任意长度的序列（旋转矩阵对任意角度都有定义），但实际中存在严重的外推问题：

```
训练时: 位置 m ∈ [0, 2048)，旋转角 m*θ_i 分布在一定范围内
推理时: 位置 m = 4096，旋转角 m*θ_i 可能超出训练分布

比喻：
  训练时只见过 0° ~ 360° 的旋转
  推理时要处理 720° 的旋转
  → 模型没见过这种角度组合 → 注意力分布崩溃
```

### 6.2 位置插值（Position Interpolation, PI）

**核心思想**：不改变模型，而是将超长序列的位置"压缩"到训练时的范围内。

$$\theta_i^{PI} = \frac{\theta_i}{s}, \quad s = \frac{L'}{L}$$

其中 $L$ 是训练长度，$L'$ 是目标长度，$s$ 是缩放因子。

```
原始 RoPE (L=2048):
  位置 0 → 角度 0
  位置 2048 → 角度 2048*θ_i

PI 扩展到 L'=8192 (s=4):
  位置 0 → 角度 0
  位置 8192 → 角度 8192*(θ_i/4) = 2048*θ_i  ← 映射回训练范围
```

**优点**：简单有效
**缺点**：分辨率下降（原来 1 个位置差对应的角度差变为 1/4）→ 需要少量微调恢复

### 6.3 NTK-aware 缩放

**动机**：PI 对所有频率均匀压缩，但高频维度（编码近距离关系）的分辨率下降更严重。

NTK-aware 方法修改基数 $\theta_{\text{base}}$ 而非频率本身：

$$\theta_{\text{base}}' = \theta_{\text{base}} \cdot s^{d/(d-2)}$$

**效果**：
- 高频维度（$\theta_i$ 大）：几乎不变 → 保持近距离分辨率
- 低频维度（$\theta_i$ 小）：被拉伸 → 适应更长距离

```
NTK-aware 的直觉：
  - 高频维度负责编码 "cat" 和 "sat" 之间的距离（近距离关系）
    → 这些关系不随序列变长而改变 → 不压缩
  - 低频维度负责编码段落级别的位置关系
    → 序列变长后需要编码更远的距离 → 适当拉伸

类比：
  地图缩放时，你不会把城市内部的街道和省际公路按同一比例缩小
```

### 6.4 YaRN（Yet another RoPE extensioN）

YaRN 结合了 PI 和 NTK-aware 的优势，对不同频率采用不同的缩放策略：

$$\theta_i' = \begin{cases} \theta_i & \text{if } \lambda_i < \alpha \text{ (高频，不变)} \\ \theta_i / s & \text{if } \lambda_i > \beta \text{ (低频，线性插值)} \\ \text{interpolate} & \text{otherwise (中频，渐变)} \end{cases}$$

其中 $\lambda_i = 2\pi / \theta_i$ 是波长，$\alpha, \beta$ 是阈值。

**YaRN 还加了一个 Attention 缩放因子**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d} \cdot t}\right) V$$

其中 $t = 0.1 \ln(s) + 1$ 补偿因序列变长导致的注意力分布变平。

### 6.5 各方案对比

| 方案 | 原理 | 是否需要微调 | 有效扩展倍数 | 质量 |
|------|------|:----------:|:---------:|------|
| 直接外推 | 不做任何修改 | 否 | ~1.1x | 崩溃 |
| PI | 均匀压缩频率 | 少量微调 | ~4-8x | 好 |
| NTK-aware | 修改基数 | 免微调/少量微调 | ~4x | 较好 |
| Code LLaMA | PI + 微调 | 较多微调 | 16K→100K | 好 |
| YaRN | 分频段缩放 + attention 缩放 | 少量微调 | ~8-128x | 最好 |
| LongRoPE | 两阶段搜索最优缩放 | 微调 | ~128x+ | 极好 |

---

## 七、面试手撕代码

### 7.1 完整的 RoPE 手写实现（面试版本）

面试时需要手写的核心函数只有两个：`precompute_freqs_cis` 和 `apply_rotary_emb`。

```python
import torch

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """预计算 RoPE 的复数频率。"""
    # θ_i = 1 / (10000^(2i/d))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # m * θ_i
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)       # (seq_len, dim//2)
    # e^{i * m * θ_i}
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """将 RoPE 应用到 Q 和 K 上。"""
    # 实数 → 复数: (B, T, H, D) → (B, T, H, D//2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 广播: (T, D//2) → (1, T, 1, D//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    # 复数乘法 = 旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

### 7.2 面试常见问题

**Q1：RoPE 是绝对位置编码还是相对位置编码？**

A：RoPE 是相对位置编码。虽然每个位置 $m$ 有确定的旋转角 $m\theta$，看起来像绝对编码，但 Q 和 K 的内积只依赖于相对距离 $m-n$，本质是相对编码。

**Q2：RoPE 为什么不作用在 Value 上？**

A：位置信息需要影响的是"注意力分布"——哪些 token 关注哪些 token，这由 $QK^T$ 的内积决定。Value 是被加权聚合的内容，不需要位置旋转。

**Q3：RoPE 的外推性如何？实际能外推多少？**

A：理论上 RoPE 可以外推到任意长度（旋转角度可以取任意值），但实际中直接外推效果很差——超出训练长度 10% 就开始明显退化。需要使用 PI / NTK-aware / YaRN 等方法扩展。

**Q4：为什么 $\theta_{\text{base}} = 10000$？能改吗？**

A：10000 是经验值，使频率覆盖合理范围。可以改——LLaMA-3 使用 500000 作为基数以支持更长上下文。更大的基数意味着更慢的旋转，能编码更远的距离。

**Q5：RoPE 与 Sinusoidal 位置编码的区别？**

A：Sinusoidal 编码加在 embedding 上（影响所有层），是绝对编码。RoPE 在每层的 Attention 中旋转 Q/K（每层都独立编码位置），是相对编码。RoPE 不增加 embedding 维度，也不引入额外参数。

**Q6：RoPE 的计算开销是多少？**

A：几乎可以忽略。对于每个 token，RoPE 需要 $d$ 次乘法（或 $d/2$ 次复数乘法），远远小于 $W_Q, W_K$ 的线性投影（$d \times d$ 矩阵乘法）。

---

## 八、RoPE 与其他位置编码的综合对比

| 维度 | Sinusoidal | 可学习绝对 | Relative (T5) | ALiBi | RoPE |
|------|-----------|----------|--------------|-------|------|
| 类型 | 绝对 | 绝对 | 相对 | 相对 | 相对 |
| 额外参数 | 0 | $T \times d$ | $O(1)$ | 0 | 0 |
| 作用位置 | Embedding | Embedding | Attention | Attention | Attention |
| 外推能力 | 理论有限 | 无 | 有限 | 较好 | 需扩展方案 |
| 代表模型 | Transformer | GPT-1/2/3 | T5 | BLOOM | LLaMA/Qwen/Mistral |
| 2024 主流度 | ✗ | ✗ | ✗ | ✗ | **✓ (事实标准)** |

---

## 九、自检题

### 基础理解

1. **用自己的话解释 RoPE 的核心思想。** 为什么"旋转"能编码位置？
2. **写出 2D 情形下 RoPE 的旋转矩阵。** 证明内积只依赖相对距离。
3. **$\theta_i = 1/10000^{2i/d}$ 中，低维度和高维度分别编码什么？** 画出频率分布图。
4. **RoPE 为什么不作用于 Value？** 如果也旋转 Value 会怎样？

### 面试手撕

5. **闭卷手写 `precompute_freqs_cis`。** 说明每一步的含义。
6. **闭卷手写 `apply_rotary_emb`。** 解释 `view_as_complex` 和 `view_as_real` 的作用。
7. **解释复数形式和旋转矩阵形式的等价性。**

### 进阶分析

8. **什么是远程衰减性？** 为什么多频率叠加会导致远程衰减？
9. **RoPE 的外推问题是什么？** 为什么不能直接外推？
10. **对比 PI、NTK-aware、YaRN 三种扩展方案的核心差异。** 各自的优缺点。
11. **LLaMA-3 将 $\theta_{\text{base}}$ 从 10000 改为 500000，这意味着什么？**

---

## 十、产出要求

- [ ] 手写 RoPE 的完整数学推导（从复数到旋转矩阵）
- [ ] **闭卷手写 `precompute_freqs_cis` + `apply_rotary_emb`（面试核心！）**
- [ ] 运行可视化代码，理解频率分布和远程衰减
- [ ] 对比至少两种外推方案（PI vs NTK-aware 或 YaRN）
- [ ] 能回答上述面试常见问题的每一个
