# Day 4：张量并行与流水线并行 — 模型切分的两个维度

> **目标**：深入 Megatron-LM 的张量并行（TP）实现——列切分与行切分如何应用于 Transformer 的 Attention 和 MLP；掌握流水线并行（PP）的调度策略——从 GPipe 到 1F1B 再到 DualPipe，理解 bubble ratio 分析；理解 Context Parallelism（Ring Attention）和 Expert Parallelism（AllToAll）；建立 3D/4D/5D 并行组合的全局视角。
>
> **前置知识**：Day 1 通信原语、Day 2 ZeRO 显存分析、Day 3 手写 TP Linear。

---

## 一、张量并行（TP）原理回顾

### 1.1 核心思想

张量并行将**单层**的权重矩阵切分到多张 GPU 上，每卡只持有一部分参数，通过通信组合得到完整结果。

与其他并行方式的区别：

| 并行方式 | 切分粒度 | 通信频率 |
|---------|---------|---------|
| DP | 数据 batch | 每步 1 次 AllReduce |
| TP | 层内权重 | 每层 1-2 次通信 |
| PP | 层间 | 每个 micro-batch 的层间传递 |

### 1.2 矩阵分块乘法基础

对于 $Y = XA$，其中 $X \in \mathbb{R}^{b \times d_1}$，$A \in \mathbb{R}^{d_1 \times d_2}$：

**列切分**（沿 $d_2$ 维度切分 $A$）：

$$A = [A_1 | A_2], \quad Y = X[A_1 | A_2] = [XA_1 | XA_2]$$

- 输入不需要切分，输出按列拼接

**行切分**（沿 $d_1$ 维度切分 $A$）：

$$A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}, \quad X = [X_1 | X_2], \quad Y = X_1 A_1 + X_2 A_2$$

- 输入需要按列切分，输出 AllReduce 求和

---

## 二、Megatron-LM TP：MLP 的切分

### 2.1 Transformer MLP 回顾

标准 Transformer MLP（以 LLaMA SwiGLU 为例）：

$$h = (\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}) \cdot W_{\text{down}}$$

- $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}}$（上投影）
- $W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$（下投影）

### 2.2 MLP 的 TP 策略

Megatron 的方案：**上投影列切分，下投影行切分**。

```
输入 x ∈ R^{b×d}（每卡完整，通过 identity/broadcast）
    │
    ├─ GPU 0: x @ W_gate₁ → SiLU → ⊙ (x @ W_up₁) → · @ W_down₁ ──┐
    ├─ GPU 1: x @ W_gate₂ → SiLU → ⊙ (x @ W_up₂) → · @ W_down₂ ──┤ AllReduce
    ├─ GPU 2: x @ W_gate₃ → SiLU → ⊙ (x @ W_up₃) → · @ W_down₃ ──┤  (求和)
    └─ GPU 3: x @ W_gate₄ → SiLU → ⊙ (x @ W_up₄) → · @ W_down₄ ──┘
                                                                      │
                                                                  输出 h ∈ R^{b×d}
```

$$W_{\text{gate}} = [W_{\text{gate}_1} | \cdots | W_{\text{gate}_N}], \quad W_{\text{gate}_i} \in \mathbb{R}^{d \times d_{ff}/N}$$

$$W_{\text{down}} = \begin{bmatrix} W_{\text{down}_1} \\ \vdots \\ W_{\text{down}_N} \end{bmatrix}, \quad W_{\text{down}_i} \in \mathbb{R}^{d_{ff}/N \times d}$$

**关键**：SiLU 和逐元素乘法 $\odot$ 可以在切分后独立计算，因为它们是逐元素操作。整个 MLP 只需要 **1 次 AllReduce**。

### 2.3 前向传播通信分析

| 操作 | 输入 | 输出 | 通信 |
|------|------|------|------|
| $f$（forward identity） | $x$（每卡完整） | $x$（复制到每卡） | 无通信 |
| Column Linear | $x$ | $[y_1, ..., y_N]$（分片） | 无通信 |
| 激活函数 | 分片 | 分片 | 无通信 |
| Row Linear | 分片 | 部分结果 | **AllReduce** |
| $g$（forward allreduce） | 部分结果 | 完整输出 | AllReduce |

### 2.4 反向传播通信

前向时的 identity（$f$）在反向时变成 AllReduce（$\bar{f}$），反之亦然：

$$\text{前向}: f \text{ (identity)} \rightarrow g \text{ (AllReduce)}$$
$$\text{反向}: \bar{f} \text{ (AllReduce)} \rightarrow \bar{g} \text{ (identity)}$$

> 前向和反向各 1 次 AllReduce，**MLP 每层总共 2 次 AllReduce**。

---

## 三、Megatron-LM TP：Attention 的切分

### 3.1 Multi-Head Attention 的天然并行性

MHA 有 $n_h$ 个头，每个头独立计算：

$$\text{head}_i = \text{Attention}(xW_Q^{(i)}, xW_K^{(i)}, xW_V^{(i)})$$

$$\text{MHA}(x) = \text{Concat}(\text{head}_1, \ldots, \text{head}_{n_h}) W_O$$

这天然适合 TP：**将不同的头分配给不同的 GPU**。

### 3.2 Attention 的 TP 策略

```
输入 x ∈ R^{b×s×d}（每卡完整）
    │
    ├─ GPU 0: Q₀K₀V₀ → head_0, head_1 → concat → @ W_O₁ ──┐
    ├─ GPU 1: Q₁K₁V₁ → head_2, head_3 → concat → @ W_O₂ ──┤ AllReduce
    ├─ GPU 2: Q₂K₂V₂ → head_4, head_5 → concat → @ W_O₃ ──┤  (求和)
    └─ GPU 3: Q₃K₃V₃ → head_6, head_7 → concat → @ W_O₄ ──┘
                                                              │
                                                          输出（每卡完整）
```

- $W_Q, W_K, W_V$：按列切分（每卡处理 $n_h/N$ 个头）
- $W_O$：按行切分（输出 AllReduce 求和）
- **同样只需 1 次 AllReduce**（前向）

### 3.3 GQA 下的 TP

GQA（Grouped Query Attention）中，KV 头数 $n_{\text{kv}}$ 少于 Q 头数 $n_h$：

- Q 头正常按 GPU 分配
- KV 头可能需要在多个 GPU 上复制（如果 $n_{\text{kv}} < N_{\text{GPU}}$）
- 或者确保 $n_{\text{kv}} \geq N_{\text{GPU}}$，每 GPU 至少一个 KV 头

---

## 四、TP 通信分析与最佳实践

### 4.1 每层通信量

对于一个 Transformer Block（Attention + MLP），TP 通信：

| 子层 | 前向 AllReduce | 反向 AllReduce | 合计 |
|------|:------------:|:------------:|:----:|
| Attention | 1 次 | 1 次 | 2 次 |
| MLP | 1 次 | 1 次 | 2 次 |
| **每层合计** | 2 次 | 2 次 | **4 次** |

每次 AllReduce 通信量 ≈ $2 \times b \times s \times d$（激活值大小）。

$$\text{每层 TP 通信量} \approx 4 \times 2 \times b \times s \times d \times \text{bytes}$$

### 4.2 TP vs DP 通信特点

| 特性 | TP | DP (AllReduce) |
|------|-----|----------------|
| 通信内容 | 激活值（与 batch/seq 相关） | 梯度（与参数量相关） |
| 通信频率 | 每层 4 次 | 每步 1 次 |
| 通信量级 | 较小（激活值） | 较大（所有参数） |
| 延迟敏感 | ✅ 极度敏感 | 可用 bucketing 掩盖 |
| **部署位置** | **node 内（NVLink）** | **可跨 node** |

### 4.3 TP 最佳实践

1. **TP 度数 ≤ 单 node GPU 数**（通常 TP=8 for 8-GPU node）
2. TP 通信走 NVLink（低延迟、高带宽）
3. TP 与 DP 正交：TP 在 node 内，DP 跨 node
4. 大模型（如 70B）通常 TP=8 + DP=N/8

---

## 五、流水线并行（PP）原理

### 5.1 核心思想

将模型的 $L$ 层按顺序分配到 $P$ 个 GPU（stage）上：

```
Stage 0 (GPU 0): Layer 0 ~ Layer L/P-1
Stage 1 (GPU 1): Layer L/P ~ Layer 2L/P-1
  ...
Stage P-1 (GPU P-1): Layer (P-1)L/P ~ Layer L-1
```

数据像流水线一样依次通过各 stage：

```
时间 →
Stage 0:  [F₀] ────── [F₁] ────── [F₂] ──── ...
Stage 1:        [F₀] ────── [F₁] ────── [F₂] ── ...
Stage 2:              [F₀] ────── [F₁] ────── [F₂]
Stage 3:                    [F₀] ────── [F₁] ────── [F₂]
```

### 5.2 Naive PP 的问题：Bubble

如果只有一个 micro-batch，大部分 GPU 在等待：

```
时间 →
Stage 0:  [F] ──────────────── [B] ────
Stage 1:       [F] ──────── [B] ────────
Stage 2:            [F] ── [B] ──────────
Stage 3:                 [F][B] ──────────

F = Forward, B = Backward
空白 = Bubble (GPU 空闲)
```

**Bubble 比例**非常高，GPU 利用率极低。

---

## 六、PP 调度策略

### 6.1 GPipe（Fill-Drain）

将 mini-batch 分成 $M$ 个 micro-batch，先全部做前向，再全部做反向：

```
时间 →  (M=4 micro-batches, P=4 stages)

Stage 0:  [F₁][F₂][F₃][F₄]                    [B₄][B₃][B₂][B₁]
Stage 1:       [F₁][F₂][F₃][F₄]          [B₄][B₃][B₂][B₁]
Stage 2:            [F₁][F₂][F₃][F₄][B₄][B₃][B₂][B₁]
Stage 3:                 [F₁][F₂][F₃][F₄][B₄][B₃][B₂][B₁]
```

**Bubble ratio**：

$$\boxed{r_{\text{bubble}} = \frac{P - 1}{P - 1 + M}}$$

当 $M \gg P$ 时 bubble 趋近 0，但需要缓存所有 micro-batch 的激活值 → **显存压力大**。

### 6.2 1F1B（One Forward One Backward）

PipeDream 提出的调度：每个 stage 交替执行前向和反向，尽早释放激活值。

```
时间 →  (M=8, P=4)

Stage 0:  [F₁][F₂][F₃][F₄]  [B₁][F₅][B₂][F₆][B₃][F₇][B₄][F₈]  [B₅][B₆][B₇][B₈]
Stage 1:       [F₁][F₂][F₃]  [B₁][F₄][B₂][F₅][B₃][F₆][B₄][F₇]  [B₅][F₈][B₆][B₇][B₈]
Stage 2:            [F₁][F₂]  [B₁][F₃][B₂][F₄][B₃][F₅][B₄][F₆]  [B₅][F₇][B₆][F₈][B₇][B₈]
Stage 3:                 [F₁] [B₁][F₂][B₂][F₃][B₃][F₄][B₄][F₅]  [B₅][F₆][B₆][F₇][B₇][F₈][B₈]
```

**优势**：
- Bubble ratio 与 GPipe 相同：$\frac{P-1}{P-1+M}$
- 但激活值显存从 $O(M)$ 降到 $O(P)$（只需缓存 $P$ 个 micro-batch 的激活）

### 6.3 Interleaved 1F1B

Megatron-LM v2 的改进：每个 GPU 不连续放层，而是交错放置。

例如 8 层 4 GPU：

```
标准 PP:      GPU 0: [L0,L1]  GPU 1: [L2,L3]  GPU 2: [L4,L5]  GPU 3: [L6,L7]
Interleaved:  GPU 0: [L0,L4]  GPU 1: [L1,L5]  GPU 2: [L2,L6]  GPU 3: [L3,L7]
```

$$\boxed{r_{\text{bubble,interleaved}} = \frac{P - 1}{(P - 1 + M) \times v}}$$

其中 $v$ 是每个 GPU 上的 virtual stage 数，bubble 减少 $v$ 倍。

### 6.4 DualPipe（DeepSeek-V3）

DeepSeek-V3 提出的 DualPipe 调度，将前向和反向**双向**流动：

```
Pipe 1 (正向):  F₁ → F₂ → F₃ → F₄ → B₄ → B₃ → B₂ → B₁
Pipe 2 (反向):  F₄ → F₃ → F₂ → F₁ → B₁ → B₂ → B₃ → B₄
```

两个 pipe 的计算可以**重叠**（overlap），大幅减少 bubble。

关键创新：
- 将 Transformer block 拆分为 attention 部分和 MLP 部分
- Attention 计算时隐藏 MLP 的通信（计算-通信重叠）
- Bubble ratio 可降低到近 0

### 6.5 调度策略对比

| 策略 | Bubble Ratio | 激活显存 | 复杂度 | 代表 |
|------|:----------:|:-------:|:-----:|------|
| Naive PP | $(P-1)/P$ | $O(1)$ | 低 | — |
| GPipe | $(P-1)/(P-1+M)$ | $O(M)$ | 低 | GPipe |
| 1F1B | $(P-1)/(P-1+M)$ | $O(P)$ | 中 | PipeDream |
| Interleaved 1F1B | $(P-1)/((P-1+M)\times v)$ | $O(P)$ | 高 | Megatron v2 |
| DualPipe | $\approx 0$（重叠） | $O(P)$ | 高 | DeepSeek-V3 |

---

## 七、Context Parallelism（CP）

### 7.1 动机

长序列训练时，注意力计算的显存 $O(s^2)$ 成为瓶颈：

$$\text{Attention 显存} \propto s^2 \cdot n_h \cdot b$$

当 $s = 128K$ 时，即使 FlashAttention 也可能 OOM。

### 7.2 Ring Attention

Ring Attention 将序列维度切分到多张 GPU，通过环形通信实现完整注意力计算：

```
序列 [0, 1, 2, ..., S-1] 切分到 4 GPU:
  GPU 0: tokens [0 ~ S/4-1]     的 Q 和 KV
  GPU 1: tokens [S/4 ~ S/2-1]   的 Q 和 KV
  GPU 2: tokens [S/2 ~ 3S/4-1]  的 Q 和 KV
  GPU 3: tokens [3S/4 ~ S-1]    的 Q 和 KV
```

每个 GPU 计算自己 Q 与所有 KV 的注意力，KV 通过环形传递：

```
Step 0: GPU 0 用本地 KV₀ 计算 Attn(Q₀, KV₀)
        同时将 KV₀ 发给 GPU 1，接收 GPU 3 的 KV₃

Step 1: GPU 0 用 KV₃ 计算 Attn(Q₀, KV₃)，累加到结果
        同时将 KV₃ 发给 GPU 1，接收 GPU 3 的 KV₂

Step 2: ...

经过 N-1 步，每个 GPU 完成了与所有 KV 的注意力计算
```

### 7.3 CP 与 FlashAttention 的配合

Ring Attention 可以与 FlashAttention 结合：
- 每一步的局部注意力用 FlashAttention 计算（高效）
- 通过 Online Softmax 技巧合并来自不同 KV 块的注意力分数

$$\text{Ring Attention 通信量} \approx 2 \times (N-1) \times \frac{s}{N} \times d \times b \times \text{bytes}$$

### 7.4 Causal Mask 优化

因果注意力下，很多 Q-KV 对的 attention 为 0（Q 在 KV 之前）。可以跳过这些计算：

```
GPU 0 的 Q₀ 只需要 KV₀ (不需要 KV₁, KV₂, KV₃)
GPU 1 的 Q₁ 需要 KV₀, KV₁
GPU 2 的 Q₂ 需要 KV₀, KV₁, KV₂
GPU 3 的 Q₃ 需要 KV₀, KV₁, KV₂, KV₃
```

负载不均衡 → 可通过交叉分配（zigzag）优化。

---

## 八、Expert Parallelism（EP）

### 8.1 MoE 中的 EP

MoE 模型中有多个专家（Expert），EP 将不同专家放在不同 GPU 上：

```
GPU 0: Expert 0, Expert 1
GPU 1: Expert 2, Expert 3
GPU 2: Expert 4, Expert 5
GPU 3: Expert 6, Expert 7
```

### 8.2 AllToAll 通信

EP 的核心通信是 **AllToAll**：将每个 GPU 上被路由到其他专家的 token 发送到对应 GPU。

```
Router 决定: Token t₁ → Expert 3, Token t₂ → Expert 0, ...

AllToAll 前:
  GPU 0: [t₁(→E3), t₂(→E0), t₃(→E5)]
  GPU 1: [t₄(→E1), t₅(→E7), t₆(→E2)]

AllToAll 后:
  GPU 0: [t₂(E0), t₄(E1)]       ← 收集到 E0/E1 的 tokens
  GPU 1: [t₆(E2), t₁(E3)]       ← 收集到 E2/E3 的 tokens
```

### 8.3 EP 通信开销

$$\text{AllToAll 每 GPU 通信量} \approx \frac{N_{\text{EP}} - 1}{N_{\text{EP}}} \times T \times d \times \text{bytes}$$

其中 $T$ 是每 GPU 的 token 数。

### 8.4 EP 与 TP/DP 的组合

```
Expert Parallelism: 不同专家放不同 GPU
    └── 与 DP 正交: EP GPU 间 AllToAll，DP GPU 间 AllReduce
    └── 与 TP 组合: 每个专家内部可以再做 TP（超大专家时）
```

---

## 九、3D / 4D / 5D 并行组合

### 9.1 经典 3D 并行（Megatron-LM）

$$\text{3D Parallelism} = \text{DP} \times \text{TP} \times \text{PP}$$

$$N_{\text{GPU}} = N_{\text{DP}} \times N_{\text{TP}} \times N_{\text{PP}}$$

```
例：64 GPU 训练 70B 模型

TP = 8  (一个 node 内)
PP = 4  (跨 2 个 node)
DP = 2  (剩余并行度)

64 = 2 × 8 × 4
```

### 9.2 4D 并行（LLaMA-3）

$$\text{4D} = \text{DP} \times \text{TP} \times \text{PP} \times \text{CP}$$

增加 CP 支持长序列训练（128K context）。

### 9.3 5D 并行（DeepSeek-V3）

$$\text{5D} = \text{DP} \times \text{TP} \times \text{PP} \times \text{CP} \times \text{EP}$$

DeepSeek-V3 的并行配置：

| 维度 | 值 | 作用 |
|------|:--:|------|
| TP | 1 | 不使用 TP（减少通信） |
| PP | 16 | 16-stage 流水线 |
| DP | 128 | 数据并行（含 ZeRO-1） |
| EP | 64 | 64 路专家并行 |
| CP | — | 视序列长度启用 |

### 9.4 并行维度选择指南

| 瓶颈 | 解决方案 | 优先级 |
|------|---------|:-----:|
| 数据量太大，训练太慢 | DP（增加数据吞吐） | 1 |
| 单层参数放不下 | TP（层内切分） | 2 |
| 模型太深放不下 | PP（层间切分） | 3 |
| 序列太长 | CP（序列切分） | 4 |
| MoE 专家太多 | EP（专家切分） | 5 |

**原则**：
- TP 放 node 内（需要低延迟高带宽）
- PP 可跨 node（点对点通信，延迟不敏感）
- DP 跨 node（AllReduce 通信量大但频率低）
- EP 根据专家数量灵活配置

---

## 十、计算-通信重叠（Overlap）

### 10.1 为什么需要 Overlap

分布式训练中，通信时间可能占总时间的 30~50%。Overlap 的目标是让计算和通信同时进行：

```
无 Overlap:
  [计算] ── [通信] ── [计算] ── [通信]  总时间 = T_comp + T_comm

有 Overlap:
  [计算 ──────────────]
       [通信 ────────]                  总时间 ≈ max(T_comp, T_comm)
```

### 10.2 常见 Overlap 技术

| 技术 | 重叠对象 | 实现 |
|------|---------|------|
| DDP Gradient Bucketing | 反向计算 + 梯度 AllReduce | PyTorch DDP 内置 |
| ZeRO Prefetch | 当前层计算 + 下一层参数 AllGather | DeepSpeed `prefetch_bucket_size` |
| TP 计算-通信 | Attention 计算 + MLP 通信 | DualPipe (DeepSeek-V3) |
| PP micro-batch | 当前 micro-batch 反向 + 下一个前向 | 1F1B 调度 |

### 10.3 DeepSeek-V3 的 Overlap 设计

DeepSeek-V3 不使用 TP（TP=1），改用更细粒度的 overlap：

```
一个 Transformer Block 拆分:
  [Attention 计算] ─── [MLP 的 AllToAll 通信]    ← 重叠
  [MLP 计算]      ─── [Attention 的 AllReduce]   ← 重叠
```

通过精心调度，几乎消除了通信 overhead。

---

## 十一、自检题

1. Megatron-LM 中 MLP 的 TP 策略是什么？为什么第一个 Linear 用列切分、第二个用行切分？
2. TP 中 MLP 前向传播需要几次 AllReduce？反向传播呢？
3. 为什么 TP 必须放在 node 内（NVLink）？
4. GQA 下做 TP 时，KV 头少于 GPU 数怎么办？
5. 1F1B 相比 GPipe 的优势是什么？Bubble ratio 是否相同？
6. 写出 GPipe 和 Interleaved 1F1B 的 bubble ratio 公式，并说明变量含义。
7. Ring Attention 的通信模式是什么？为什么因果注意力下会有负载不均衡？
8. EP 的核心通信是什么？与 TP 的 AllReduce 有什么区别？
9. 如果有 128 张 GPU 训练 70B 模型，你会如何配置 TP / PP / DP？说明理由。
10. 计算-通信重叠的基本思想是什么？DeepSeek-V3 为什么选择 TP=1？

---

## 十二、产出要求

- [ ] 画出 Megatron-LM 中 MLP 和 Attention 的 TP 数据流图（标注列切分/行切分/AllReduce 位置）
- [ ] 画出 1F1B Pipeline 调度时序图（4 stages × 8 micro-batches）
- [ ] 推导 GPipe 和 Interleaved 1F1B 的 bubble ratio 公式
- [ ] 画出 Ring Attention 的通信步骤（4 GPU 为例）
- [ ] 设计一个 128 GPU 训练 70B 模型的并行策略方案（TP / PP / DP 各多少），并计算每卡显存
- [ ] 撰写 TP vs PP vs CP vs EP 的对比表（切分维度 / 通信模式 / 适用场景）
