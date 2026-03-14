# Day 2：ViT 与 CLIP 架构详解 — 从 Patch Embedding 到 InfoNCE 的完整数学

> **目标**：逐模块拆解 ViT 和 CLIP 的架构细节，从 Patch Embedding 到双塔对比学习，完成每个组件的数学推导与维度分析。为 Day 3 手写实现打下坚实的数学基础。

---

## 一、ViT 架构全景图

ViT（Vision Transformer）的核心思路是：**把图像当成一句话，patch 当成 token**。

```
输入图像: x ∈ R^{H × W × C}    (例如 224 × 224 × 3)
       │
       ▼  切分为 P×P 的 patch
┌─────────────────────────────┐
│  Patch Embedding            │
│  x_p^i ∈ R^{P²·C} → R^D    │  线性投影: E ∈ R^{(P²·C) × D}
│  + [CLS] token              │  x_cls ∈ R^D (可学习)
│  + Position Embedding       │  E_pos ∈ R^{(N+1) × D} (可学习)
└─────────────────────────────┘
       │
       ▼  z_0 ∈ R^{(N+1) × D}     N = (H/P)×(W/P) = 196 (for 224/16)
┌─────────────────────────────┐
│  Transformer Encoder × L    │  重复 L 次 (ViT-B: L=12)
│  ┌───────────────────────┐  │
│  │  LayerNorm             │  │
│  │  Multi-Head Self-Attn  │  │  Pre-Norm (与 BERT 不同)
│  │  + Residual            │  │
│  ├───────────────────────┤  │
│  │  LayerNorm             │  │
│  │  MLP (GELU)            │  │  隐藏维度 4D
│  │  + Residual            │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
       │
       ▼  z_L ∈ R^{(N+1) × D}
┌─────────────────────────────┐
│  LayerNorm                  │
│  取 [CLS] token → z_L^0    │  z_L^0 ∈ R^D
│  Classification Head        │  Linear(D, num_classes)
└─────────────────────────────┘
       │
       ▼
  logits ∈ R^{num_classes}
```

---

## 二、Patch Embedding 详解

### 2.1 图像切分

将 $H \times W \times C$ 的图像切分为 $N$ 个大小为 $P \times P$ 的 patch：

$$N = \frac{H}{P} \times \frac{W}{P}$$

以 ViT-B/16 为例：$H = W = 224$，$P = 16$，则 $N = 14 \times 14 = 196$。

每个 patch 展平后为一个 $P^2 \cdot C = 16^2 \times 3 = 768$ 维的向量。

### 2.2 线性投影

将展平的 patch 通过线性层投影到 $D$ 维：

$$\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C} \xrightarrow{\mathbf{E}} \mathbb{R}^D$$

其中 $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 是可学习的投影矩阵。

**实现技巧**：实际中用 `nn.Conv2d(C, D, kernel_size=P, stride=P)` 替代展平+线性层，效果等价但更高效。

```
Conv2d 实现:
输入: (B, C, H, W) = (B, 3, 224, 224)
卷积核: (D, C, P, P) = (768, 3, 16, 16), stride=16
输出: (B, D, H/P, W/P) = (B, 768, 14, 14)
reshape: (B, D, N) → transpose → (B, N, D) = (B, 196, 768)
```

### 2.3 CLS Token

在 patch 序列前添加一个可学习的 `[CLS]` token：

$$\mathbf{z}_0 = [\mathbf{x}_\text{cls};\; \mathbf{x}_p^1 \mathbf{E};\; \mathbf{x}_p^2 \mathbf{E};\; \dots;\; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_\text{pos}$$

- $\mathbf{x}_\text{cls} \in \mathbb{R}^D$：可学习的分类 token
- $\mathbf{E}_\text{pos} \in \mathbb{R}^{(N+1) \times D}$：可学习的位置编码
- 序列长度从 $N$ 变为 $N + 1$

**为什么用 CLS token？**

- 借鉴 BERT：用一个特殊 token 聚合全局信息
- 替代方案是 Global Average Pooling（对所有 patch token 取平均），ViT 论文实验表明两者效果接近
- 在 CLIP 中使用 CLS token 作为图像的全局表示

### 2.4 位置编码

ViT 使用**可学习的 1D 位置编码**：

$$\mathbf{E}_\text{pos} \in \mathbb{R}^{(N+1) \times D}$$

论文实验了多种位置编码方案：

| 方案 | 描述 | ImageNet Top-1 |
|------|------|:---:|
| 无位置编码 | 不加位置信息 | 61.32% |
| 1D 可学习 | 标准做法 | **79.39%** |
| 2D 可学习 | 分别编码行/列 | 79.35% |
| 相对位置 | 编码 patch 间相对距离 | 79.38% |

结论：**1D 可学习位置编码已经足够好**。模型能从数据中自发学到 2D 空间结构——相邻位置编码的余弦相似度呈现出 2D 网格模式。

---

## 三、ViT Encoder Block

### 3.1 Pre-Norm Transformer Block

ViT 使用 Pre-Norm 结构（与 GPT-2、LLaMA 一致，但与原始 Transformer 不同）：

$$\mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$

$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l$$

其中 $l = 1, \dots, L$，$\text{LN}$ 为 LayerNorm，$\text{MSA}$ 为 Multi-Head Self-Attention。

### 3.2 Multi-Head Self-Attention

与标准 Transformer Attention 完全一致（**无 Causal Mask**，因为 ViT 是 Encoder）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

其中 $Q = \mathbf{z} W_Q$，$K = \mathbf{z} W_K$，$V = \mathbf{z} W_V$，$W_Q, W_K, W_V \in \mathbb{R}^{D \times d_k}$。

**关键区别**：
- GPT / LLaMA 使用 **Causal Mask**（下三角），每个 token 只能看到前面的 token
- ViT 使用**无 Mask** 的全注意力，每个 patch 可以关注所有其他 patch

$$\text{MSA}(\mathbf{z}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O$$

其中 $h$ 是头数，$d_k = D / h$，$W_O \in \mathbb{R}^{D \times D}$。

### 3.3 MLP（前馈网络）

$$\text{MLP}(\mathbf{z}) = \text{GELU}(\mathbf{z} W_1 + b_1) W_2 + b_2$$

- $W_1 \in \mathbb{R}^{D \times 4D}$：扩展到 4 倍隐藏维度
- $W_2 \in \mathbb{R}^{4D \times D}$：投影回原维度
- 激活函数使用 GELU（与 GPT 相同，LLaMA 使用 SwiGLU）

### 3.4 ViT Block 完整数据流

```
输入: z_{l-1} ∈ R^{(N+1) × D}
  │
  ├──────────────────┐
  │                  │ (residual)
  ▼                  │
LayerNorm            │
  │                  │
  ▼                  │
MSA (h heads)        │     Q, K, V ∈ R^{(N+1) × D}
  │                  │     Attn: R^{(N+1) × (N+1)}  ← 注意：无 causal mask
  │                  │
  ▼                  │
  +  ◄───────────────┘
  │
  ├──────────────────┐
  │                  │ (residual)
  ▼                  │
LayerNorm            │
  │                  │
  ▼                  │
MLP (GELU, 4D)       │
  │                  │
  ▼                  │
  +  ◄───────────────┘
  │
  ▼
输出: z_l ∈ R^{(N+1) × D}
```

---

## 四、ViT 参数量与计算量分析

### 4.1 各组件参数量

| 组件 | 参数量 | ViT-B/16 数值 |
|------|-------|:---:|
| Patch Embedding | $P^2 \cdot C \cdot D + D$ | $768 \times 768 + 768 = 590K$ |
| CLS Token | $D$ | $768$ |
| Position Embedding | $(N+1) \cdot D$ | $197 \times 768 = 151K$ |
| 单层 MSA | $4D^2 + 4D$ | $4 \times 768^2 + 3072 = 2.36M$ |
| 单层 MLP | $8D^2 + 5D$ | $8 \times 768^2 + 3840 = 4.72M$ |
| 单层 LN (×2) | $4D$ | $3072$ |
| **单层 Block 总计** | $\approx 12D^2$ | $\approx 7.09M$ |
| **L 层 + Head** | $\approx 12LD^2 + DK$ | — |

### 4.2 ViT 各型号配置

| 型号 | 层数 $L$ | 隐藏维度 $D$ | 头数 $h$ | MLP 维度 | Patch 大小 | 参数量 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| ViT-S/16 | 12 | 384 | 6 | 1536 | 16 | 22M |
| ViT-B/16 | 12 | 768 | 12 | 3072 | 16 | 86M |
| ViT-L/16 | 24 | 1024 | 16 | 4096 | 16 | 304M |
| ViT-L/14 | 24 | 1024 | 16 | 4096 | 14 | 304M |
| ViT-H/14 | 32 | 1280 | 16 | 5120 | 14 | 632M |

**命名规则**：ViT-{大小}/{Patch大小}，如 ViT-B/16 = Base 模型 + 16×16 patch。

注意 ViT-L/14 比 ViT-L/16 的**序列更长**（$256$ vs $196$ 个 patch），但参数量相同（因为参数不依赖序列长度），计算量更大。

### 4.3 FLOPs 分析

单层 Transformer Block 的 FLOPs：

$$\text{FLOPs}_\text{MSA} = 4ND^2 + 2N^2D$$

$$\text{FLOPs}_\text{MLP} = 8ND^2$$

$$\text{FLOPs}_\text{Block} = 12ND^2 + 2N^2D$$

其中 $N$ 是序列长度。当 $N \ll 6D$ 时，MLP 主导计算；当 $N > 6D$ 时，注意力主导。

---

## 五、CLIP 双塔架构详解

### 5.1 架构总览

CLIP 由两个独立的编码器组成，通过对比学习在共享嵌入空间中对齐：

```
┌──── Image Branch ────┐          ┌──── Text Branch ────┐
│                      │          │                      │
│  图像 (224×224×3)     │          │  文本 "a dog"         │
│       │              │          │       │              │
│       ▼              │          │       ▼              │
│  Patch Embedding     │          │  Token Embedding     │
│  + CLS + Pos Embed   │          │  + Position Embed    │
│       │              │          │       │              │
│       ▼              │          │       ▼              │
│  ViT Encoder × L     │          │  Transformer × L'    │
│  (无 causal mask)    │          │  (有 causal mask)    │
│       │              │          │       │              │
│       ▼              │          │       ▼              │
│  [CLS] token         │          │  [EOS] token         │
│       │              │          │       │              │
│       ▼              │          │       ▼              │
│  LayerNorm           │          │  LayerNorm           │
│       │              │          │       │              │
│       ▼              │          │       ▼              │
│  Image Projection    │          │  Text Projection     │
│  W_I ∈ R^{D_I × D_E} │          │  W_T ∈ R^{D_T × D_E} │
│       │              │          │       │              │
│       ▼              │          │       ▼              │
│  I ∈ R^{D_E}        │          │  T ∈ R^{D_E}        │
│  (L2 归一化)         │          │  (L2 归一化)         │
└──────────────────────┘          └──────────────────────┘
           │                                │
           └──────── 余弦相似度 ─────────────┘
                   sim(I, T) = I · T
                   (因为已 L2 归一化)
```

### 5.2 Image Encoder

CLIP 的 Image Encoder 有两种选择：

| 型号 | 架构 | 输入分辨率 | 嵌入维度 $D_E$ | 参数量 |
|------|------|:---:|:---:|:---:|
| RN50 | ResNet-50 (改进) | 224 | 1024 | 38M |
| RN101 | ResNet-101 (改进) | 224 | 512 | 56M |
| ViT-B/32 | ViT-Base, P=32 | 224 | 512 | 88M |
| ViT-B/16 | ViT-Base, P=16 | 224 | 512 | 87M |
| ViT-L/14 | ViT-Large, P=14 | 224 | 768 | 304M |
| ViT-L/14@336 | ViT-Large, P=14 | 336 | 768 | 304M |

最常用的是 **ViT-L/14**（CLIP 中效果最好的开源 Image Encoder），后续 LLaVA 也使用它。

CLIP ViT 与标准 ViT 的细微差异：
- 在 Transformer 最后一层之后加 LayerNorm（而非之前）
- 使用 `[CLS]` token 的输出作为图像表示（同标准 ViT）
- 通过线性投影层映射到共享嵌入空间

### 5.3 Text Encoder

CLIP 的 Text Encoder 是一个 **GPT-like Transformer**（Decoder-only，带 Causal Mask）：

| 参数 | 值 |
|------|------|
| 层数 | 12 |
| 隐藏维度 | 512 |
| 注意力头数 | 8 |
| 上下文长度 | 77 tokens |
| 词表大小 | 49152 (BPE) |
| 参数量 | ~63M |

**关键设计**：
- 使用 **Causal Mask**（与 GPT 相同），但这里不是为了自回归生成，而是论文发现 causal mask 的训练效率更高
- 使用 `[EOS]` token（序列末尾）的输出作为文本的全局表示（类似 GPT 取最后一个 token）
- 通过线性投影层映射到共享嵌入空间

### 5.4 投影层与共享嵌入空间

两个编码器的输出维度可能不同（$D_I \neq D_T$），需要投影到同一维度 $D_E$：

$$\mathbf{I} = \frac{f_I(\text{image}) \cdot W_I}{\|f_I(\text{image}) \cdot W_I\|}$$

$$\mathbf{T} = \frac{f_T(\text{text}) \cdot W_T}{\|f_T(\text{text}) \cdot W_T\|}$$

- $W_I \in \mathbb{R}^{D_I \times D_E}$：图像投影矩阵
- $W_T \in \mathbb{R}^{D_T \times D_E}$：文本投影矩阵
- L2 归一化确保所有向量落在单位超球面上

归一化后，余弦相似度等价于内积：

$$\text{sim}(\mathbf{I}, \mathbf{T}) = \mathbf{I} \cdot \mathbf{T} = \cos(\theta)$$

### 5.5 温度参数 $\tau$

CLIP 使用可学习的温度参数控制 softmax 的锐度：

$$\tau = \exp(t), \quad t \text{ 是可学习的标量，初始化使得 } \tau = 0.07$$

- $\tau$ 小 → softmax 更"尖锐"，模型更确信匹配关系
- $\tau$ 大 → softmax 更"平滑"，容忍更多不确定性
- 在训练中 $\tau$ 会被 clamp 在 $[0.01, 100]$ 范围内

---

## 六、InfoNCE Loss 数学推导

### 6.1 直觉理解

给定一个 batch 的 $N$ 个图文对 $\{(I_i, T_i)\}_{i=1}^N$：

- **正例**：$(I_i, T_i)$ —— 匹配的图文对
- **负例**：$(I_i, T_j)$，$j \neq i$ —— 不匹配的图文对

目标：让正例的相似度高，负例的相似度低。

相似度矩阵 $S \in \mathbb{R}^{N \times N}$：

$$S_{ij} = \frac{\mathbf{I}_i \cdot \mathbf{T}_j}{\tau}$$

```
           T_1    T_2    T_3   ...   T_N
    I_1  [ s_11   s_12   s_13  ...  s_1N ]    ← 第 1 行: I_1 与所有文本的相似度
    I_2  [ s_21   s_22   s_23  ...  s_2N ]
    I_3  [ s_31   s_32   s_33  ...  s_3N ]
    ...  [ ...    ...    ...   ...  ...  ]
    I_N  [ s_N1   s_N2   s_N3  ...  s_NN ]

    对角线 s_ii 是正例 ✓，其余是负例 ✗
```

### 6.2 Image-to-Text Loss

对每张图像 $I_i$，在所有文本中找到匹配的 $T_i$（$N$ 分类问题）：

$$\mathcal{L}_{i2t} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})}$$

展开：

$$\mathcal{L}_{i2t} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\mathbf{I}_i \cdot \mathbf{T}_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{I}_i \cdot \mathbf{T}_j / \tau)}$$

这本质上是 **Cross-Entropy Loss**，标签是 $y_i = i$（每张图对应第 $i$ 个文本）。

### 6.3 Text-to-Image Loss

对称地，对每个文本 $T_i$，在所有图像中找到匹配的 $I_i$：

$$\mathcal{L}_{t2i} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ji})}$$

### 6.4 总损失

$$\mathcal{L}_\text{CLIP} = \frac{1}{2}(\mathcal{L}_{i2t} + \mathcal{L}_{t2i})$$

**等价理解**：对相似度矩阵 $S$，沿行做 cross-entropy（标签 = 列索引）和沿列做 cross-entropy（标签 = 行索引），取平均。

### 6.5 与标准 Cross-Entropy 的关系

令 $\text{logits}_i = [S_{i1}, S_{i2}, \dots, S_{iN}]$，$\text{label}_i = i$，则：

$$\mathcal{L}_{i2t} = \frac{1}{N}\sum_{i=1}^{N} \text{CE}(\text{logits}_i, i) = \frac{1}{N}\sum_{i=1}^{N} \text{CE}(S[i,:], i)$$

$$\mathcal{L}_{t2i} = \frac{1}{N}\sum_{i=1}^{N} \text{CE}(S[:,i], i)$$

**代码实现极其简洁**：

```python
# 伪代码
labels = torch.arange(N)                          # [0, 1, 2, ..., N-1]
logits = (I @ T.T) / tau                           # (N, N)
loss_i2t = F.cross_entropy(logits, labels)         # 沿行
loss_t2i = F.cross_entropy(logits.T, labels)       # 沿列
loss = (loss_i2t + loss_t2i) / 2
```

### 6.6 InfoNCE 与互信息的关系

InfoNCE Loss 是互信息 $I(X; Y)$ 的一个下界估计：

$$I(X; Y) \geq \log N - \mathcal{L}_\text{InfoNCE}$$

其中 $N$ 是负例数量（等于 batch size）。这意味着：
- **Batch size 越大**，下界越紧，估计越准确
- 这解释了 CLIP 为什么使用极大的 batch size（32768）

### 6.7 温度参数的梯度分析

对正例 $(I_i, T_i)$，InfoNCE 对 $I_i$ 的梯度（简化）：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{I}_i} \propto \frac{1}{\tau}\left(-\mathbf{T}_i + \sum_{j=1}^{N} p_{ij} \mathbf{T}_j\right)$$

其中 $p_{ij} = \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})}$。

- $\tau$ 小 → $p_{ij}$ 接近 one-hot → 梯度主要来自最难的负例（hard negatives）
- $\tau$ 大 → $p_{ij}$ 更均匀 → 所有负例贡献相近

---

## 七、CLIP 的 Zero-Shot 推理

### 7.1 推理流程

```
Step 1: 构造文本 prompt（每个类别一个）
  "a photo of a dog"
  "a photo of a cat"
  "a photo of a car"
  ...
  "a photo of a {class_K}"

Step 2: 编码
  Text Encoder → T_1, T_2, ..., T_K ∈ R^{D_E}    (K 个类别的文本特征)
  Image Encoder → I ∈ R^{D_E}                      (待分类图像的特征)

Step 3: 相似度计算
  sim_k = I · T_k / τ    for k = 1, ..., K

Step 4: 预测
  ŷ = argmax_k sim_k
```

### 7.2 Prompt Engineering

CLIP 论文发现 prompt 模板的选择对性能影响很大：

| Prompt 模板 | ImageNet Top-1 |
|------------|:---:|
| `"{class}"` | 低 |
| `"a photo of a {class}"` | 中 |
| `"a photo of a {class}, a type of pet"` | 高 |
| 80 个模板的集成 | 最高 |

**Prompt Ensemble**：对每个类别使用多个模板，将文本特征取平均后归一化：

$$\mathbf{T}_k = \frac{1}{M}\sum_{m=1}^{M} f_T(\text{prompt}_m(\text{class}_k))$$

---

## 八、ViT + CLIP 与 GPT / LLaMA 的对比

### 8.1 架构选择对比

| 维度 | ViT (Encoder) | GPT/LLaMA (Decoder) | CLIP Text (Decoder) |
|------|:---:|:---:|:---:|
| Attention Mask | 无（全注意力） | Causal（下三角） | Causal（下三角） |
| 位置编码 | 可学习 1D | 可学习 / RoPE | 可学习 |
| 归一化 | LayerNorm (Pre-Norm) | LayerNorm / RMSNorm | LayerNorm |
| FFN 激活 | GELU | GELU / SwiGLU | GELU |
| 输出表示 | CLS token | 最后 token | EOS token |
| 任务 | 编码/分类 | 自回归生成 | 编码表示 |

### 8.2 为什么 ViT 用 Encoder？

图像理解需要**双向注意力**：
- 一张图中每个 patch 都可能与任意其他 patch 相关
- 不存在"顺序"依赖（不像文本有从左到右的因果关系）
- Encoder 的全注意力允许每个 patch 看到所有其他 patch

### 8.3 为什么 CLIP Text Encoder 用 Decoder？

CLIP 论文的解释是训练效率：
- Causal Attention 训练时可以同时利用所有前缀子序列
- 实验中 Decoder 架构的训练效率高于 Encoder

取 `[EOS]` token 作为文本表示，因为在 causal attention 中只有最后一个 token 能看到完整输入。

---

## 九、Day 3 代码骨架预览

Day 3 将从零实现以下组件：

```python
# Part 1: 配置
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768     # D
    num_layers: int = 12       # L
    num_heads: int = 12        # h
    mlp_ratio: int = 4
    num_classes: int = 1000

# Part 2: Patch Embedding
class PatchEmbedding(nn.Module):
    # Conv2d(C, D, P, P) + CLS token + Position Embedding
    # 输入: (B, C, H, W) → 输出: (B, N+1, D)

# Part 3: ViT Encoder Block
class ViTBlock(nn.Module):
    # LayerNorm → MSA → Residual → LayerNorm → MLP → Residual
    # 输入/输出: (B, N+1, D)

# Part 4: 完整 ViT
class ViT(nn.Module):
    # PatchEmbedding + L × ViTBlock + LayerNorm + Classification Head
    # 输入: (B, C, H, W) → 输出: (B, num_classes)

# Part 5-7: CLIP
class CLIPImageEncoder(nn.Module):  # 基于 ViT，输出 CLS 特征
class CLIPTextEncoder(nn.Module):   # GPT-like，输出 EOS 特征
class CLIP(nn.Module):              # 双塔 + 投影 + Temperature

# Part 8: InfoNCE Loss
def clip_loss(image_features, text_features, temperature):
    # 对称 cross-entropy
```

---

## 十、自检题

### 基础题

1. 写出 ViT 的 Patch Embedding 公式，包括 CLS token 和位置编码。
2. ViT-B/16 处理 224×224 图像时，序列长度是多少？每个 patch 的原始维度是多少？
3. ViT 的 Attention 与 GPT 的 Attention 有什么关键区别？
4. CLIP 的 Image Encoder 和 Text Encoder 分别使用什么架构？
5. 写出 InfoNCE Loss 的公式（image-to-text 方向）。

### 进阶题

6. 推导 ViT-B/16 单层 Transformer Block 的参数量。
7. 解释为什么 CLIP 的 InfoNCE Loss 等价于对相似度矩阵做行/列方向的 cross-entropy。
8. CLIP 的温度参数 $\tau$ 如何影响梯度中 hard negatives 的权重？
9. 用 3-4 行伪代码实现 CLIP Loss。

### 面试题

10. 面试官问："ViT 的可学习 1D 位置编码和 LLaMA 的 RoPE 有什么本质区别？各自的优缺点？"
11. 面试官问："写出 CLIP InfoNCE Loss 的代码。"（面试 Tier 3 必会）

---

## 十一、产出要求

- [ ] 画出 ViT 的完整架构图（含每层输入输出维度）
- [ ] 画出 CLIP 的双塔架构图（含投影层和归一化）
- [ ] 手写 Patch Embedding 的数学公式
- [ ] 手写 InfoNCE Loss 的数学公式（双向）
- [ ] 计算 ViT-B/16 的参数量（逐组件拆解）
- [ ] 用伪代码写出 CLIP Loss 的实现
- [ ] 解释 CLIP zero-shot 分类的完整流程
- [ ] 完成全部自检题
