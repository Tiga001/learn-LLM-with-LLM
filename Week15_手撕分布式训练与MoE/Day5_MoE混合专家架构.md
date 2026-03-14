# Day 5：MoE 混合专家架构 — 用稀疏换规模

> **目标**：系统掌握 Mixture of Experts（MoE）的核心设计——理解稀疏 vs 稠密模型的 trade-off；深入 Router / Expert / Gate 的架构与路由机制（Top-K、Expert Choice、Token Choice）；掌握负载均衡的辅助损失设计；精读 Mixtral 8x7B 和 DeepSeek-V2/V3 的 MoE 实现细节；理解 MoE 训练挑战与推理优化；建立总参数量 vs 激活参数量的清晰区分。
>
> **前置知识**：Day 1 通信原语、Day 4 Expert Parallelism（AllToAll）、W4 Transformer FFN 结构。

---

## 一、MoE 基本概念

### 1.1 稠密模型的困境

传统 Transformer 是**稠密模型**——每个 token 经过所有参数的计算：

$$\text{FLOPs}_{\text{dense}} = 6 \times N_{\text{params}} \times D_{\text{tokens}}$$

增大模型 → 参数增多 → FLOPs 线性增长 → **训练和推理成本同步增加**。

> 问题：能否让模型拥有大量参数，但每个 token 只用其中一小部分？

### 1.2 条件计算与稀疏性

MoE 的核心思想是**条件计算（Conditional Computation）**：

$$\text{输出} = \sum_{i=1}^{E} g_i(x) \cdot f_i(x)$$

- $f_i$：第 $i$ 个专家（Expert），通常是标准 FFN
- $g_i(x)$：门控函数（Gate），决定 token $x$ 分配给哪些专家
- 关键：$g_i(x)$ 大部分为 0 → **稀疏激活**

### 1.3 稀疏 vs 稠密对比

| 特性 | 稠密模型 | MoE 稀疏模型 |
|------|---------|-------------|
| 每 token 计算 | 所有参数 | 部分参数（Top-K 专家） |
| 总参数量 | $N$ | $N_{\text{shared}} + E \times N_{\text{expert}}$（远大于稠密） |
| 激活参数量 | $N$ | $N_{\text{shared}} + K \times N_{\text{expert}}$（与稠密相近） |
| FLOPs/token | 与总参数正比 | 与**激活参数**正比 |
| 显存 | 与总参数正比 | 与**总参数**正比（需加载所有专家） |
| 模型能力 | 受参数量限制 | 更大参数 → 更强表示能力 |

$$\boxed{\text{MoE 的精髓：用更多参数换更强模型，但保持计算量不变}}$$

### 1.4 MoE 的发展历程

```
1991: MoE 概念提出 (Jacobs et al.)
2017: Sparsely-Gated MoE (Shazeer et al.) — 首次应用于大规模 NLP
2021: Switch Transformer — Top-1 路由
2021: GShard — 大规模 MoE 训练
2023: Mixtral 8x7B — 高质量开源 MoE
2024: DeepSeek-V2 — 细粒度专家 + 共享专家
2024: DeepSeek-V3 — 无辅助损失负载均衡 + DualPipe
```

---

## 二、MoE 架构设计

### 2.1 标准 MoE Layer

MoE Layer 替换 Transformer 中的 FFN 层：

```
标准 Transformer Block:
  Input → LayerNorm → Attention → Residual → LayerNorm → FFN → Residual → Output

MoE Transformer Block:
  Input → LayerNorm → Attention → Residual → LayerNorm → MoE Layer → Residual → Output
                                                           │
                                                    ┌──────┴──────┐
                                                    │   Router    │
                                                    │  (Gating)   │
                                                    └──────┬──────┘
                                               ┌────┬─────┼─────┬────┐
                                               ▼    ▼     ▼     ▼    ▼
                                             [E₀] [E₁]  [E₂]  [E₃] [E₄] ...
                                               │    │     │     │    │
                                               └────┴─────┼─────┴────┘
                                                    加权求和
                                                       │
                                                    Output
```

### 2.2 MoE 的三个核心组件

**Router（路由器）**：

$$\text{Router}(x) = \text{Softmax}(x \cdot W_g), \quad W_g \in \mathbb{R}^{d \times E}$$

- 输入 token 的隐层表示 $x \in \mathbb{R}^d$
- 输出 $E$ 个专家的路由概率

**Expert（专家）**：

$$\text{Expert}_i(x) = \text{FFN}_i(x) = (\text{SiLU}(x W_{\text{gate}}^{(i)}) \odot x W_{\text{up}}^{(i)}) \cdot W_{\text{down}}^{(i)}$$

- 每个专家是独立的 FFN（结构相同，参数不同）
- 通常与标准 Transformer 的 FFN 结构一致

**Gate（门控）**：

$$y = \sum_{i \in \text{TopK}} g_i(x) \cdot \text{Expert}_i(x)$$

- 选择 Top-K 个专家，用路由权重加权求和

### 2.3 MoE 放置策略

不是每层都用 MoE，通常交替放置：

```
Layer 0:  Attention + FFN (稠密)
Layer 1:  Attention + MoE Layer
Layer 2:  Attention + FFN (稠密)
Layer 3:  Attention + MoE Layer
  ...
```

**为什么交替？**
- 减少总参数量和显存
- 部分共享的稠密层有助于特征学习的稳定性
- Mixtral 和 DeepSeek 均采用此策略

### 2.4 Switch Transformer

Switch Transformer（Google, 2021）的关键创新——**Top-1 路由**：

$$y = g_{\text{top1}}(x) \cdot \text{Expert}_{\text{top1}}(x)$$

- 每个 token 只分配给 1 个专家（$K=1$）
- 极大简化路由和计算
- 但需要更精细的负载均衡

---

## 三、路由机制详解

### 3.1 Top-K Routing（Token Choice）

最常用的路由方式：每个 token 选择 Top-K 个专家。

$$\text{logits} = x \cdot W_g \in \mathbb{R}^E$$

$$\text{gates} = \text{Softmax}(\text{logits})$$

$$\text{TopK\_indices}, \text{TopK\_values} = \text{TopK}(\text{gates}, k=K)$$

$$y = \sum_{i \in \text{TopK}} \frac{g_i}{\sum_{j \in \text{TopK}} g_j} \cdot \text{Expert}_i(x)$$

> 注意：Top-K 后重新归一化（renormalize），使得权重之和为 1。

```python
def top_k_routing(x, W_gate, k=2):
    """Top-K routing implementation"""
    logits = x @ W_gate                         # [batch*seq, num_experts]
    gates = F.softmax(logits, dim=-1)
    topk_vals, topk_ids = torch.topk(gates, k)  # [batch*seq, k]
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
    return topk_ids, topk_vals
```

### 3.2 Expert Choice Routing

Expert Choice（Zhou et al., 2022）反转了路由方向：**专家选择 token**。

$$S = \text{Softmax}(W_g^T X^T) \in \mathbb{R}^{E \times T}$$

每个专家选择 Top-C 个 token（C = capacity）：

$$C = \frac{K \times T}{E}$$

**优势**：
- 天然负载均衡（每个专家处理相同数量的 token）
- 避免 token 被丢弃
- 允许重要 token 被多个专家选择

**劣势**：
- 每个 token 被选中的专家数量不固定（可能 0 个或很多个）
- 自回归生成时 token 粒度不确定，实现复杂

### 3.3 Token Choice vs Expert Choice 对比

| 特性 | Token Choice (Top-K) | Expert Choice |
|------|---------------------|---------------|
| 选择方向 | Token → Expert | Expert → Token |
| 每 token 专家数 | 固定 K 个 | 不固定（0 ~ 多个） |
| 负载均衡 | 需要辅助 loss | 天然均衡 |
| Token 丢弃 | 容量溢出时丢弃 | 不丢弃 |
| 推理友好 | ✅ 容易 | ❌ 自回归困难 |
| 代表 | Mixtral, DeepSeek | Google Expert Choice |

### 3.4 容量因子（Capacity Factor）

Token Choice 路由需要设置每个专家的**容量上限**：

$$C_i = \text{CF} \times \frac{T}{E}$$

- $T$：总 token 数
- $E$：专家数
- $\text{CF}$：容量因子（通常 1.0 ~ 1.5）

$$\text{理想情况}：\text{CF} = 1.0 \text{（每个专家处理 } T/E \text{ 个 token）}$$

CF 太小 → token 被丢弃（dropped）→ 信息损失
CF 太大 → 填充（padding）多 → 计算浪费

```
CF=1.0:  Expert 0: [t₁,t₅,__]   Expert 1: [t₂,t₃,t₇]  (t₈ 被丢弃!)
CF=1.5:  Expert 0: [t₁,t₅,__,__] Expert 1: [t₂,t₃,t₇,t₈] (无丢弃，但有 padding)
```

---

## 四、负载均衡

### 4.1 为什么需要负载均衡

没有负载均衡时，路由器倾向于把大部分 token 发给少数专家 → **路由崩塌（Routing Collapse）**：

```
理想分布:                   路由崩塌:
E₀: ████████ (25%)          E₀: ████████████████████████████ (70%)
E₁: ████████ (25%)          E₁: ████ (10%)
E₂: ████████ (25%)          E₂: ██ (5%)
E₃: ████████ (25%)          E₃: ██████ (15%)
```

后果：
- 多数专家得不到训练 → 参数浪费
- 热门专家过载 → 性能下降
- 训练不稳定

### 4.2 辅助损失（Auxiliary Load Balancing Loss）

Switch Transformer 提出的辅助损失：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

其中：
- $f_i = \frac{1}{T}\sum_{x \in \mathcal{B}} \mathbb{1}[\text{argmax}(g(x)) = i]$：实际分配到专家 $i$ 的 token 比例
- $P_i = \frac{1}{T}\sum_{x \in \mathcal{B}} g_i(x)$：路由器给专家 $i$ 的平均概率
- $\alpha$：辅助损失系数（通常 $10^{-2}$）
- $E$：专家数

**直觉**：$f_i \cdot P_i$ 在均匀分布时取最小值 $1/E^2$（当 $f_i = P_i = 1/E$）。

$$\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i}$$

```python
def load_balancing_loss(router_logits, expert_indices, num_experts, alpha=0.01):
    """计算辅助负载均衡 loss"""
    gates = F.softmax(router_logits, dim=-1)  # [T, E]
    T = gates.shape[0]

    # f_i: 每个专家实际被选中的 token 比例
    one_hot = F.one_hot(expert_indices, num_experts).float()  # [T, E]
    f = one_hot.mean(dim=0)  # [E]

    # P_i: 路由器给每个专家的平均概率
    P = gates.mean(dim=0)  # [E]

    return alpha * num_experts * (f * P).sum()
```

### 4.3 Z-Loss

ST-MoE（Zoph et al., 2022）提出 Z-Loss，惩罚路由 logits 过大：

$$\mathcal{L}_z = \frac{1}{T} \sum_{x \in \mathcal{B}} \left(\log \sum_{i=1}^{E} e^{z_i(x)}\right)^2$$

其中 $z_i(x) = x \cdot W_g$ 是未归一化的路由 logits。

**作用**：
- 防止 logits 过大导致路由过于尖锐（某个专家概率 → 1）
- 提高训练稳定性
- 与辅助损失配合使用

### 4.4 DeepSeek-V3 的无辅助损失均衡

DeepSeek-V3 提出不使用辅助损失的负载均衡策略：

$$g_i'(x) = g_i(x) + b_i$$

- $b_i$ 是每个专家的可学习偏置
- 训练时根据专家利用率动态调整 $b_i$：负载过高的专家降低 $b_i$，反之增加
- 不会影响路由权重（只影响路由决策），因此不干扰训练目标

**优势**：消除了辅助损失系数 $\alpha$ 的调参需求，避免辅助损失对主任务的干扰。

---

## 五、Mixtral 8x7B 架构解析

### 5.1 与 LLaMA 的关系

Mixtral 8x7B（Mistral AI, 2024）基于 Mistral 7B 架构，**只替换 FFN 为 MoE**：

```
Mistral 7B Block:                    Mixtral 8x7B Block:
┌─────────────────┐                  ┌─────────────────┐
│   Attention      │                  │   Attention      │ ← 完全相同
│   (GQA, RoPE)   │                  │   (GQA, RoPE)   │
├─────────────────┤                  ├─────────────────┤
│   SwiGLU FFN    │  ─── 替换为 ──→  │   MoE Layer     │
│   d→4d→d        │                  │   8 Expert FFNs  │
│                 │                  │   + Top-2 Router │
└─────────────────┘                  └─────────────────┘
```

### 5.2 Mixtral 架构参数

| 参数 | Mistral 7B | Mixtral 8x7B |
|------|:----------:|:------------:|
| 层数 $L$ | 32 | 32 |
| 隐层维度 $d$ | 4096 | 4096 |
| FFN 维度 $d_{ff}$ | 14336 | 14336 |
| 注意力头数 | 32 | 32 |
| KV 头数 (GQA) | 8 | 8 |
| 专家数 $E$ | — | 8 |
| Top-K | — | 2 |
| 上下文长度 | 8K (SWA) | 32K |
| **总参数** | **7B** | **46.7B** |
| **激活参数** | **7B** | **~12.9B** |

### 5.3 参数量分析

Mixtral 的总参数 = Attention 参数 + MoE 参数：

$$N_{\text{total}} = \underbrace{N_{\text{attn}} \times L}_{\text{32层 Attention}} + \underbrace{E \times N_{\text{ffn}} \times L}_{\text{32层 × 8专家 FFN}} + N_{\text{embed}}$$

每层 FFN 参数（SwiGLU）：$3 \times d \times d_{ff} = 3 \times 4096 \times 14336 \approx 176M$

8 个专家 × 32 层 × 176M ≈ 45B（FFN 部分）

### 5.4 为什么只替换 FFN

```
Attention 层:                    FFN 层:
  参数与 head 数相关                参数与 d_ff 相关
  已有天然并行（多头）              计算密集，参数占比大
  不同 token 需要交互               每个 token 独立计算 ✓
  → 不适合 MoE                    → 非常适合 MoE
```

**FFN 适合 MoE 的原因**：
1. FFN 对每个 token **独立计算**（point-wise），天然可以按 token 分配
2. FFN 参数量约占 Transformer 的 **2/3**，替换收益大
3. Attention 需要 token 间交互，不同专家处理会破坏注意力机制

### 5.5 Mixtral 的路由分析

Mixtral 论文中观察到的路由模式：

```
不同领域的专家激活热力图:
        E0  E1  E2  E3  E4  E5  E6  E7
数学:   ██  ░░  ██  ░░  ░░  ░░  ██  ░░   → 主要 E0, E2, E6
代码:   ░░  ██  ░░  ░░  ██  ██  ░░  ░░   → 主要 E1, E4, E5
文学:   ░░  ░░  ░░  ██  ░░  ░░  ░░  ██   → 主要 E3, E7
多语言: ██  ██  ░░  ░░  ░░  ██  ░░  ░░   → 主要 E0, E1, E5

██ = 高激活, ░░ = 低激活
```

- 专家呈现一定的**领域专业化**
- 但并非完全分离，存在共享
- 底层（靠近输入）的专家选择更随机，顶层更有语义区分

---

## 六、DeepSeek-V2/V3 的 MoE 设计

### 6.1 DeepSeekMoE 的核心创新

DeepSeek-V2 提出了两个关键改进：

**1. 细粒度专家（Fine-Grained Experts）**

传统 MoE：$E$ 个大专家，Top-K 激活

DeepSeek：$mE$ 个小专家，Top-$mK$ 激活

$$\text{传统}: E=8, K=2 \quad \longrightarrow \quad \text{DeepSeek}: E=64, K=6$$

- 每个小专家的 FFN 维度为 $d_{ff}/m$
- 总参数量和计算量不变
- 但路由**更灵活**——可以组合更多样的专家子集

```
传统 MoE (8 专家 Top-2):       DeepSeek (64 专家 Top-6):
┌────────┐ ┌────────┐          ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐
│Expert 0│ │Expert 3│          │E2││E7││E15││E23││E41││E58│
│ (大)   │ │ (大)   │          │小││小││小 ││小 ││小 ││小 │
└────────┘ └────────┘          └──┘└──┘└──┘└──┘└──┘└──┘
  2 种组合可能                   C(64,6) ≈ 74M 种组合可能!
```

**2. 共享专家（Shared Experts）**

$$y = \underbrace{\text{Expert}_{\text{shared}}(x)}_{\text{每个 token 都经过}} + \underbrace{\sum_{i \in \text{TopK}} g_i(x) \cdot \text{Expert}_i(x)}_{\text{路由专家}}$$

- 1~2 个共享专家处理**所有 token**
- 捕捉通用知识，减轻路由专家的负担
- 路由专家可以更专注于专业知识

### 6.2 DeepSeek-V2 架构参数

| 参数 | 值 |
|------|:--:|
| 总参数 | 236B |
| 激活参数 | 21B |
| 层数 | 60 |
| 隐层维度 | 5120 |
| 路由专家数 | 160（每层） |
| 共享专家数 | 2（每层） |
| Top-K | 6 |
| 注意力 | MLA（Multi-head Latent Attention） |

### 6.3 MLA（Multi-head Latent Attention）

DeepSeek-V2 还创新了注意力机制——MLA：

$$c_{KV} = x W_{DKV}, \quad \begin{cases} k = c_{KV} W_{UK} \\ v = c_{KV} W_{UV} \end{cases}$$

- 将 KV 投影到低维潜在空间 $c_{KV} \in \mathbb{R}^{d_c}$（$d_c \ll d$）
- 推理时只需缓存 $c_{KV}$，大幅减少 KV Cache

$$\boxed{\text{KV Cache 压缩比} = \frac{n_h \times d_h}{d_c} \approx \frac{128 \times 128}{512} = 32\times}$$

> MLA 虽然不属于 MoE，但与 MoE 共同构成 DeepSeek-V2 的效率优势。

### 6.4 DeepSeek-V3 的进一步优化

| 创新 | 说明 |
|------|------|
| 无辅助损失均衡 | 动态偏置 $b_i$ 替代 $\mathcal{L}_{\text{aux}}$ |
| DualPipe | 双向流水线，bubble ≈ 0（Day 4 已详述） |
| FP8 训练 | 专家计算用 FP8，降低显存和计算 |
| Multi-Token Prediction | 预测多个未来 token，提升训练效率 |
| 总参数 671B | 激活参数仅 37B |

### 6.5 DeepSeek-V3 MoE 配置

| 参数 | DeepSeek-V3 |
|------|:-----------:|
| 总参数 | 671B |
| 激活参数 | 37B |
| 层数 | 61 |
| 隐层维度 | 7168 |
| 路由专家数 | 256（每层） |
| 共享专家数 | 1（每层） |
| Top-K | 8 |
| 每专家 FFN 维度 | 2048（细粒度） |

```
DeepSeek-V3 每层 MoE:
┌─────────────────────────────────────────────────────┐
│                    输入 x                            │
│                      │                              │
│  ┌───────────────────┼───────────────────┐          │
│  │                   │                   │          │
│  ▼                   ▼                   │          │
│ Shared Expert    Router(x)               │          │
│ (始终激活)         │                     │          │
│  │           ┌────┬┴──┬────┬────┐       │          │
│  │           ▼    ▼   ▼    ▼    ▼       │          │
│  │         [E₃] [E₁₇][E₉₂][E₁₃₅]...(Top-8)      │
│  │           │    │   │    │    │       │          │
│  │           └────┴───┼────┴────┘       │          │
│  │                加权求和               │          │
│  │                    │                  │          │
│  └────────────── + ───┘                  │          │
│                  │                       │          │
│              输出 y                       │          │
└─────────────────────────────────────────────────────┘
```

---

## 七、MoE 训练挑战

### 7.1 路由崩塌（Routing Collapse）

**现象**：训练过程中，路由器逐渐只选择少数专家，其他专家被"遗忘"。

**原因**——正反馈循环：

```
Expert A 被选多 → 梯度更新多 → Expert A 变得更好
   ↑                                          │
   └──── Router 更倾向选 Expert A ←────────────┘

Expert B 被选少 → 梯度更新少 → Expert B 停滞
   ↑                                          │
   └──── Router 更不选 Expert B ←─────────────┘
```

**解决方案**：
- 辅助负载均衡损失（4.2 节）
- Z-Loss（4.3 节）
- Random routing（部分 token 随机分配）
- Expert 参数重初始化（定期 reset 不活跃的专家）

### 7.2 专家利用率

衡量指标——**专家利用率（Expert Utilization）**：

$$U_i = \frac{\text{Expert } i \text{ 被选中的 token 数}}{\text{总 token 数} \times K / E}$$

- $U_i = 1.0$：理想均匀分配
- $U_i \ll 1.0$：专家利用不足
- 目标：所有专家的 $U_i$ 接近 1.0

### 7.3 训练不稳定性

MoE 模型训练比稠密模型更容易不稳定：

| 问题 | 原因 | 缓解策略 |
|------|------|---------|
| 路由波动 | 离散选择（TopK）引入噪声 | Noisy Top-K（添加噪声后 softmax） |
| 梯度尺度不一致 | 不同专家更新频率不同 | Expert-wise 学习率调整 |
| 辅助损失权重 | $\alpha$ 太大影响主任务 | 仔细调参或无辅助损失方案 |
| 参数量巨大 | 训练初期不稳定 | Warmup + 较小初始学习率 |

### 7.4 Noisy Top-K Gating

Shazeer et al. (2017) 提出在路由 logits 上添加噪声以促进探索：

$$H(x) = x \cdot W_g + \epsilon \cdot \text{Softplus}(x \cdot W_{\text{noise}})$$

$$\epsilon \sim \mathcal{N}(0, 1)$$

- 训练时添加噪声 → 增加专家探索
- 推理时不加噪声 → 确定性路由

---

## 八、MoE 推理优化

### 8.1 MoE 推理的显存挑战

MoE 推理需要加载**所有专家**的参数，但每个 token 只用其中 K 个：

$$\text{显存} \propto N_{\text{total}}(\text{所有专家}), \quad \text{计算} \propto N_{\text{active}}(\text{K 个专家})$$

以 Mixtral 8x7B 为例：
- 推理需加载 46.7B 参数 ≈ 93.4 GB（FP16）
- 但每 token 只用 ~12.9B 参数的计算量

> 显存需求按总参数计，但计算速度按激活参数计 → 显存是瓶颈。

### 8.2 Expert Offloading

将不活跃的专家卸载到 CPU 内存或 SSD，只在需要时加载：

```
GPU 显存:   [Attention] [Shared] [Active Expert 3] [Active Expert 7]
CPU 内存:   [Expert 0] [Expert 1] [Expert 2] [Expert 4] [Expert 5] [Expert 6]

Token → Router → 决定需要 Expert 5
  → 从 CPU 加载 Expert 5 到 GPU
  → 计算
  → 可选：驱逐 Expert 7 到 CPU（LRU 策略）
```

**挑战**：
- GPU ↔ CPU 传输延迟（PCIe ~32 GB/s）
- 需要预测下一步需要哪些专家（prefetch）
- 命中率取决于专家激活的局部性

### 8.3 Expert Parallelism（EP）推理

Day 4 已介绍 EP 的训练场景。推理时 EP 同样关键：

$$\text{EP 推理}：\text{AllToAll} \rightarrow \text{Expert 计算} \rightarrow \text{AllToAll}$$

- 每张 GPU 持有部分专家
- Router 决定分配后，通过 AllToAll 将 token 发送到对应 GPU
- 计算后再通过 AllToAll 收回结果

### 8.4 量化与 MoE

MoE 模型特别适合量化：

| 量化方案 | Mixtral 8x7B 显存 | 速度影响 |
|---------|:-----------------:|:-------:|
| FP16 | ~93 GB | 基线 |
| INT8 (GPTQ) | ~47 GB | ~1.1× |
| INT4 (GPTQ) | ~24 GB | ~1.2× |
| GGUF Q4_K_M | ~26 GB | CPU 推理可行 |

> MoE 的专家参数冗余性高 → 量化后精度损失较小。

### 8.5 Speculative Decoding + MoE

Speculative Decoding（W14 Day 7）与 MoE 结合：

- Draft model 可以是 MoE 的子集（只用 Top-1 或共享专家）
- 验证时用完整 MoE（Top-K）
- 进一步减少推理延迟

---

## 九、MoE 参数量 vs 计算量分析

### 9.1 总参数 vs 激活参数

$$\boxed{N_{\text{total}} = N_{\text{non-MoE}} + E \times N_{\text{expert}}}$$

$$\boxed{N_{\text{active}} = N_{\text{non-MoE}} + K \times N_{\text{expert}}}$$

其中 $N_{\text{non-MoE}}$ 包括 Attention 层、Embedding、LayerNorm 等非 MoE 参数。

### 9.2 常见模型参数对比

| 模型 | 总参数 | 激活参数 | $E$ | $K$ | 专家粒度 |
|------|:------:|:-------:|:---:|:---:|:-------:|
| Switch-Base | 7.4B | 0.1B | 128 | 1 | 粗 |
| Mixtral 8x7B | 46.7B | 12.9B | 8 | 2 | 粗 |
| DeepSeek-V2 | 236B | 21B | 160 | 6 | 细 |
| DeepSeek-V3 | 671B | 37B | 256 | 8 | 细 |
| Qwen2.5-MoE | 57.4B | 14.3B | 64 | 8 | 中 |

### 9.3 FLOPs 分析

每 token 的前向传播 FLOPs（简化）：

$$\text{FLOPs}_{\text{MoE}} \approx 2 \times N_{\text{active}} = 2 \times (N_{\text{non-MoE}} + K \times N_{\text{expert}})$$

对比等计算量的稠密模型：

$$\frac{N_{\text{total}}}{N_{\text{active}}} = \frac{N_{\text{non-MoE}} + E \times N_{\text{expert}}}{N_{\text{non-MoE}} + K \times N_{\text{expert}}}$$

Mixtral 8x7B：$46.7 / 12.9 \approx 3.6\times$（总参数是激活参数的 3.6 倍）

> MoE 用 3.6× 的参数量（显存成本），换取与 12.9B 稠密模型相同的计算速度，但性能远超 12.9B。

### 9.4 MoE 的 Scaling 特性

```
性能
  │
  │         ╱ MoE (固定计算量，增加专家)
  │       ╱
  │     ╱     ╱ Dense (增加参数)
  │   ╱     ╱
  │ ╱     ╱
  │╱    ╱
  ├───╱───────────────→ 总参数量
  │ ╱
  │╱
  │

在相同计算预算下，MoE 可以通过增加专家数量
获得更好的性能（更多参数 = 更强表示能力）
```

**Scaling Law for MoE**：

$$L_{\text{MoE}}(N, E) \propto \left(\frac{N_{\text{active}}}{N_0}\right)^{-\alpha} \cdot \left(\frac{E}{E_0}\right)^{-\beta}$$

增加专家数 $E$ 可以在不增加计算的情况下提升性能，但收益递减（$\beta < \alpha$）。

---

## 十、MoE 的工程实践要点

### 10.1 MoE 设计选择清单

| 决策 | 选项 | 推荐 |
|------|------|------|
| 放置层 | 每层 / 交替 / 自定义 | 交替（偶数层 MoE） |
| 专家数 $E$ | 8 / 16 / 64 / 256 | 看显存预算和 EP GPU 数 |
| Top-K | 1 / 2 / 4 / 8 | $K=2$（常见）或 $K=1$（极致效率） |
| 路由方式 | Token Choice / Expert Choice | Token Choice（推理友好） |
| 负载均衡 | 辅助 Loss / Z-Loss / 无辅助 | 辅助 Loss + Z-Loss |
| 专家粒度 | 粗（标准 FFN）/ 细（FFN/m） | 细粒度（DeepSeek 验证有效） |
| 共享专家 | 有 / 无 | 有（提升稳定性和性能） |

### 10.2 MoE 训练配置示例

```python
@dataclass
class MoEConfig:
    d_model: int = 4096
    num_experts: int = 8
    num_shared_experts: int = 1
    top_k: int = 2
    expert_ffn_dim: int = 14336
    aux_loss_alpha: float = 0.01
    z_loss_weight: float = 0.001
    capacity_factor: float = 1.25
    num_moe_layers: int = 16       # 32 层中交替放 16 层 MoE
    router_jitter: float = 0.1     # 训练时路由噪声
```

### 10.3 MoE 与其他并行的关系

```
┌──────────────────────────────────────────┐
│            MoE + 分布式训练               │
│                                          │
│  DP: 数据并行（梯度 AllReduce）            │
│   └── ZeRO: 分片优化器/梯度/参数          │
│                                          │
│  EP: 专家并行（AllToAll 路由 token）       │
│   └── 通常 EP = 专家数 / 每GPU专家数       │
│                                          │
│  TP: 张量并行（单个专家内部切分）           │
│   └── 用于超大专家                        │
│                                          │
│  PP: 流水线并行（层间切分）                │
│   └── MoE 层和普通层一起切分              │
│                                          │
│  DeepSeek-V3: DP=128, PP=16, EP=64      │
└──────────────────────────────────────────┘
```

---

## 十一、自检题

1. MoE 的核心思想是什么？它如何实现「更多参数但不增加计算量」？
2. 写出 Top-K routing 的数学公式——从路由 logits 到最终输出。
3. Token Choice 和 Expert Choice 路由各有什么优缺点？推理时为什么更倾向 Token Choice？
4. 什么是路由崩塌（Routing Collapse）？它的正反馈循环是怎样的？
5. 写出 Switch Transformer 辅助负载均衡损失的公式，并解释 $f_i$ 和 $P_i$ 的含义。
6. DeepSeek-V2 的两个关键 MoE 创新是什么？细粒度专家为什么能带来更灵活的路由？
7. Mixtral 8x7B 的总参数量和激活参数量分别是多少？为什么只替换 FFN 为 MoE？
8. MoE 推理的显存瓶颈是什么？Expert Offloading 如何缓解？
9. 区分 MoE 的「总参数」和「激活参数」。为什么说 MoE 的计算量由激活参数决定？
10. DeepSeek-V3 如何在不使用辅助损失的情况下实现负载均衡？
11. Z-Loss 惩罚什么？它如何提高训练稳定性？

---

## 十二、产出要求

- [ ] 画出 MoE Layer 的完整架构图（Router → Top-K → Experts → 加权求和），标注数据流和维度
- [ ] 手写 Top-K routing + 辅助负载均衡 Loss 的实现代码
- [ ] 撰写 Mixtral 8x7B 架构分析笔记——参数量计算、与 LLaMA 的对比、路由分析
- [ ] 撰写 DeepSeek-V2/V3 MoE 设计总结——细粒度专家、共享专家、MLA、无辅助损失均衡
- [ ] 计算 Mixtral 8x7B 的总参数 / 激活参数 / 每 token FLOPs
- [ ] 画出 MoE 参数量 vs 计算量的对比图（与等参数稠密模型 / 等计算量稠密模型对比）
- [ ] 整理 MoE 训练挑战与解决方案表（路由崩塌 / 不稳定性 / 负载均衡）
