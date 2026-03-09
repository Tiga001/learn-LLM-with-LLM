# Day 2：FlashAttention 深化 — 从理解到能写伪代码（面试核心）

> **目标**：从 W3 Day5"理解 tiling 思想"进阶到"能写伪代码 + 理解反向传播"；完整回顾 FlashAttention-1 的前向算法与 Online Softmax 技巧；理解 FA1 的反向传播为何不需要存储 $S, P$ 矩阵（recomputation）；掌握 FlashAttention-2 的三个核心改进；了解 FlashDecoding 如何通过沿 KV 序列并行加速推理 Decode 阶段；简要了解 FlashAttention-3 对 Hopper GPU 的优化；对比不同序列长度和 GPU 下的性能 benchmark。

---

## 一、FlashAttention-1 完整回顾

### 1.1 标准 Attention 的 IO 瓶颈

标准 Self-Attention 的计算：

$$
O = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

分解为四步：
1. $S = QK^T / \sqrt{d_k}$ → 写入 HBM
2. $P = \text{softmax}(S)$ → 从 HBM 读 $S$，写回 $P$ 到 HBM
3. $O = PV$ → 从 HBM 读 $P$，写回 $O$ 到 HBM

```
标准 Attention 的 IO 分析 (N=序列长度, d=头维度):

矩阵         大小        HBM 读写
───────────────────────────────────
Q, K, V      N×d         读: 3Nd
S = QK^T     N×N         写: N²    ← 瓶颈!
P = softmax  N×N         读: N²    ← 瓶颈!
                         写: N²    ← 瓶颈!
O = PV       N×d         写: Nd

总 HBM IO ≈ O(N²)    ← 序列长度的平方!

GPU 显存层次:
  SRAM (片上):  ~20 MB,  带宽 ~19 TB/s
  HBM  (显存):  ~80 GB,  带宽 ~2 TB/s
  
  差距: SRAM 带宽是 HBM 的 ~10倍
  
问题: N×N 矩阵 S 和 P 太大，放不进 SRAM
  N=4096, d=128 时:
    S 大小: 4096² × 2 bytes = 32 MB → 放不进 20 MB 的 SRAM!
```

### 1.2 FlashAttention 的核心思想：Tiling + Recomputation

**关键洞察**：不需要把整个 $N \times N$ 矩阵 $S$ 实体化（materialize），可以分块计算并在线更新 softmax。

```
FlashAttention 的 IO 改进:

标准 Attention:   IO = O(N² + Nd)   ← 受 N² 项支配
FlashAttention:   IO = O(N²d / M)    ← M 是 SRAM 大小

当 M >> d 时 (通常 SRAM = 20MB >> d=128):
  N²d / M << N²
  → IO 显著减少!

核心做法:
  1. 将 Q, K, V 分成小块 (tiles)，每块能放进 SRAM
  2. 在 SRAM 中完成注意力计算（包括 softmax）
  3. 只把最终结果 O 写回 HBM
  4. 永远不把 N×N 矩阵写入 HBM
```

### 1.3 Online Softmax 技巧

FlashAttention 的核心数学难点：softmax 需要全局信息（分母的求和），但 tiling 只能看到局部。

**解决方案**：Online Softmax（Milakov & Gimelshein, 2018）

标准 softmax：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
$$

Online 版本：当新块到来时，更新 $m$ 和 $\ell$（归一化因子），并修正之前的结果。

```
Online Softmax 更新规则:

处理第 1 块 (scores s1):
  m1 = max(s1)
  ℓ1 = sum(exp(s1 - m1))
  o1 = softmax(s1) × V1 = diag(1/ℓ1) × exp(s1 - m1) × V1

处理第 2 块 (scores s2):
  m2 = max(s2)
  m_new = max(m1, m2)
  
  修正第 1 块:
    ℓ1_new = ℓ1 × exp(m1 - m_new)        ← 修正分母
    o1_new = o1 × exp(m1 - m_new) *  ℓ1       ← 修正分子
    
  计算第 2 块:
    ℓ2 = sum(exp(s2 - m_new))
    o2 = exp(s2 - m_new) × V2
    
  合并:
    ℓ_total = ℓ1_new + ℓ2
    o_total = (o1_new + o2) / ℓ_total

关键: 不需要回到第 1 块重新计算!
     只需用标量乘法 exp(m_old - m_new) 修正
```

### 1.4 FlashAttention-1 前向传播完整伪代码

```
算法: FlashAttention 前向传播


输入: Q, K, V ∈ R^{N×d}  (存储在 HBM)
输出: O ∈ R^{N×d}        (存储在 HBM)

1. 设置块大小:
   B_c = ceil(M / (4d))     ← KV 块大小 (列方向)
   B_r = min(ceil(M / (4d)), d)  ← Q 块大小 (行方向)

2. 初始化:
   O = 0 ∈ R^{N×d}         (在 HBM)
   ℓ = 0 ∈ R^N             (行方向的 log-sum-exp)
   m = -∞ ∈ R^N            (行方向的最大值)

3. 将 Q 分成 T_r = ceil(N / B_r) 个块: Q_1, ..., Q_{T_r}
   将 K 分成 T_c = ceil(N / B_c) 个块: K_1, ..., K_{T_c}
   将 V 分成 T_c 个块: V_1, ..., V_{T_c}

4. for j = 1 to T_c:                         ← 外循环: 遍历 KV 块
     从 HBM 加载 K_j, V_j 到 SRAM
     
     for i = 1 to T_r:                       ← 内循环: 遍历 Q 块
       从 HBM 加载 Q_i, O_i, ℓ_i, m_i 到 SRAM
       
       // 在 SRAM 中计算注意力
       S_ij = Q_i × K_j^T / √d              ← B_r × B_c 矩阵
       
       m̃_ij = rowmax(S_ij)                   ← B_r 维向量
       P̃_ij = exp(S_ij - m̃_ij)               ← 局部 softmax 分子
       ℓ̃_ij = rowsum(P̃_ij)                   ← 局部分母
       
       // Online Softmax 更新
       m_new = max(m_i, m̃_ij)
       ℓ_new = exp(m_i - m_new) × ℓ_i + exp(m̃_ij - m_new) × ℓ̃_ij
       
       // 更新输出
       O_i ← diag(ℓ_new)^{-1} × (
               diag(ℓ_i) × exp(m_i - m_new) × O_i   ← 修正旧结果
             + exp(m̃_ij - m_new) × P̃_ij × V_j        ← 新贡献
       )
       
       // 更新统计量
       m_i ← m_new
       ℓ_i ← ℓ_new
       
       将 O_i, ℓ_i, m_i 写回 HBM
       
5. 返回 O
```

**IO 复杂度分析（面试硬核考点：为什么是 $N^2 d^2 / M$？）**：

- 标准 Attention：需要读写 $N \times N$ 的 $S$ 和 $P$ 矩阵，总 IO 复杂度为 $\Theta(N^2 + Nd)$。
- FlashAttention：$\Theta(N^2 d^2 / M)$ 次 HBM 访问（$M$ 是 SRAM 大小）。

**复杂度推导过程（按分块循环计算）**：
1. **分块大小由 SRAM 决定**：SRAM 的大小是 $M$。我们要把矩阵分成小块放进 SRAM，每个小块的维度是 $B_r \times d$ (如 Q, O 块) 或 $B_c \times d$ (如 K, V 块)。为了能装下，单块占用内存必须是 $\Theta(M)$，即 $B_r \times d \approx \Theta(M)$，推导出块的行数 $B_r = \Theta(M/d)$。同理 $B_c = \Theta(M/d)$。
2. **总块数计算**：序列长度为 $N$，所以 Q 矩阵被分成了 $T_r = \frac{N}{B_r} = \Theta(\frac{Nd}{M})$ 个块；K 和 V 矩阵被分成了 $T_c = \frac{N}{B_c} = \Theta(\frac{Nd}{M})$ 个块。
3. **两层嵌套循环的 IO 开销**：
   - **外循环**遍历 KV 块，共 $T_c$ 次。每次循环加载当前的 K, V 块（数据量 $\Theta(B_c \times d) = \Theta(M)$）。
   - **内循环**遍历 Q 块，共 $T_r$ 次。每次循环加载并写回当前的 Q, O 块及标量（数据量 $\Theta(B_r \times d) = \Theta(M)$）。
4. **总读写量（Total IO）**：
   $$ \text{Total IO} = \text{外循环次数} \times (\text{加载KV的IO} + \text{内循环次数} \times \text{读写Q和O的IO}) $$
   主要开销集中在内循环的反复读写上，所以：
   $$ \text{Total IO} \approx T_c \times T_r \times \Theta(M) $$
   代入 $T_c$ 和 $T_r$：
   $$ \text{Total IO} \approx \Theta\left(\frac{Nd}{M}\right) \times \Theta\left(\frac{Nd}{M}\right) \times \Theta(M) = \Theta\left(\frac{N^2 d^2}{M^2} \times M\right) = \Theta\left(\frac{N^2 d^2}{M}\right) $$

**结论**：
- FlashAttention 的 HBM 访问量为 $\Theta(N^2 d^2 / M)$。
- 理论上，当 SRAM 足够大能装下所有的序列和特征即 $M = \Theta(Nd)$（实际中 SRAM 容量通常达数十MB，远大于 $Nd$ 的数百KB），带入公式得到 $\Theta(N^2 d^2 / Nd) = \Theta(Nd)$。这就说明，**IO 复杂度从标准注意力的平方级 $\Theta(N^2)$ 降到了线性的亚二次方 $\Theta(Nd)$**！这正是彻底打破显存墙（Memory Wall）的关键。

---

## 二、FlashAttention-1 的反向传播

### 2.1 为什么反向传播是难点

标准反向传播需要：
- 前向保存的 $S = QK^T$ （$N \times N$）
- 前向保存的 $P = \text{softmax}(S)$ （$N \times N$）

这些矩阵恰好是 FlashAttention 前向中没有存储到 HBM 的！

### 2.2 Recomputation 策略

FlashAttention 的反向传播使用 **recomputation**（重计算）策略：

```
标准反向传播:
  前向: 保存 S, P 到 HBM → 显存 O(N²)
  反向: 从 HBM 读取 S, P → IO = O(N²)

FlashAttention 反向传播:
  前向: 不存 N×N 的 S 和 P 矩阵，只保存 O, ℓ, m 
        (对于单个头，O 大小为 N×d，ℓ 和 m 大小为 N)
        → 显存复杂度为 O(Nd) ← 大幅减少！
  反向: 用 Q, K, V 重新计算 S, P (在 SRAM 中) → 多一次计算，但省了 IO

  为什么值得?
    重计算 S_ij = Q_i K_j^T 的计算量: O(B_r × B_c × d) FLOPs
    省下的 IO: O(N²) 次 HBM 读写
    
    在现代 GPU 上: 计算速度 >> 内存带宽
    → 重计算反而更快!
```

### 2.3 反向传播的梯度公式

$$
dV = P^T dO, \quad dP = dO \cdot V^T
$$

$$
dS = P \odot (dP - \text{rowsum}(dP \odot P)), \quad dQ = dS \cdot K, \quad dK = dS^T \cdot Q
$$

其中 $\odot$ 是逐元素乘法。

**关键**：所有涉及 $P$ 的计算都可以分块重计算，利用前向保存的 $O, \ell, m$。

### 2.4 反向传播 IO 复杂度

与前向相同，反向传播的 IO 复杂度也是 $O(N^2 d^2 / M)$，远优于标准反向传播的 $O(N^2)$。

---

## 三、FlashAttention-2 核心改进

### 3.1 FA2 论文信息

- **标题**：*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
- **作者**：Tri Dao
- **时间**：2023
- **核心贡献**：在 FA1 基础上通过优化工作分配和减少非矩阵乘运算，达到理论 FLOPS 的 50-73%（FA1 仅达 25-40%）

### 3.2 改进 1：减少非矩阵乘运算

```
FA1 的问题: 大量标量运算浪费 GPU 算力

GPU 的计算单元:
  Tensor Core: 专门做矩阵乘法 (GEMM)
    → 理论峰值: 312 TFLOPS (A100 FP16)
  CUDA Core: 做通用计算（exp, max, 加法等）
    → 理论峰值: 19.5 TFLOPS (A100 FP16)
    
  差距: Tensor Core 是 CUDA Core 的 16 倍!
  
FA1 中的非 GEMM 运算:
  - exp(S_ij - m)  ← 逐元素指数
  - rowmax, rowsum ← 规约操作
  - diag(ℓ)^{-1} × O ← 缩放
  这些都用 CUDA Core，拖慢整体速度
  
FA2 的改进:
  1. 将 Online Softmax 的缩放推迟到最后
  2. 减少中间的 rescale 操作
  3. 在内循环中只维护未归一化的 O
  → 非 GEMM FLOPs 减少约 50%
```

具体来说，FA2 在内循环中维护未归一化的输出，只在最后做一次归一化：

$$
\tilde{O}_i = \sum_j \exp(m_{ij} - m_i^{\text{final}}) \cdot \tilde{P}_{ij} V_j
$$

$$
O_i = \text{diag}(\ell_i)^{-1} \tilde{O}_i
$$

### 3.3 改进 2：内外循环调换

```
FA1 循环顺序:
  外循环: 遍历 KV 块 (j)
    内循环: 遍历 Q 块 (i)
    
  问题: 每次处理完一个 KV 块后，
        需要更新所有 Q 块对应的 O_i, ℓ_i, m_i
        → 频繁读写 HBM

FA2 循环顺序:
  外循环: 遍历 Q 块 (i)        ← 调换!
    内循环: 遍历 KV 块 (j)
    
  优势: 
    - Q_i, O_i 在外循环加载一次，内循环全程驻留 SRAM
    - KV 块在内循环顺序加载
    - O_i 只在内循环结束后写回一次
    → HBM 读写显著减少!
```

### 3.4 改进 3：更好的并行化策略（把训练、Prefill、Decode 分开看）

这一节最容易混淆的点是：**“推理”并不是一个单一阶段。**

- **训练阶段**：Q 的序列长度 = N，K/V 的序列长度 = N
- **Prefill 阶段**（处理整段 Prompt）：Q 的序列长度 = N，K/V 的序列长度 = N
- **Decode 阶段**（逐 token 生成）：Q 的序列长度 = 1，K/V 的序列长度 = context length

所以，FA2 的序列维度并行对 **训练阶段** 和 **Prefill 阶段** 都非常有用；真正不够用的是 **Decode 阶段**。

先看 GPU 为什么需要“高并行度”。

```
GPU 由很多 SM（Streaming Multiprocessor）组成。
以 A100 为例，有 108 个 SM。

如果你只给 GPU 32 个并行任务：
  → 最多只有 32 个 SM 同时干活
  → 其余 76 个 SM 会闲置
  → GPU 算力吃不满，这就是 GPU starvation（GPU 饥饿）
```

#### FA1：只在 batch 和 head 维度并行

FA1 的任务划分比较粗：

```
一个并行任务 ≈ 一个 batch 中的一个 attention head

总并行度 ≈ batch_size × n_heads
```

这在 batch 很大时还能接受，但在大模型常见的场景里会出问题：

- 训练有时还能靠较大的 batch 撑起并行度
- 在线推理经常是 `batch=1`
- 很多模型的 `n_heads` 也就 32 或 40 或 64

例如 LLaMA-7B：

```
batch_size = 1
n_heads = 32

FA1 并行度 = 1 × 32 = 32
```

而 A100 有 108 个 SM：

```
SM 利用率上限 ≈ 32 / 108 ≈ 29.6%
```

这说明不是算子本身不会算，而是**没有足够多的独立任务可以同时派发给 GPU**。

#### FA2：再沿着 Q 的序列维度切分

FA2 的关键观察是：

- 一个 attention head 里的整段 Q 序列，并不一定要作为一个整体交给一个 SM
- 可以把 Q 再切成很多个小块 `Q_1, Q_2, ..., Q_{T_r}`
- 每个 Q 块对应输出中的一个 O 块
- **不同 Q 块的最终输出彼此独立**，因此可以并行

于是，FA2 的任务粒度变成：

```
一个并行任务 ≈ 一个 batch 中的一个 attention head 里的一个 Q 块

总并行度 ≈ batch_size × n_heads × T_r
其中:
  T_r = ceil(seq_len_q / B_r)
  seq_len_q = Q 的序列长度
```

这里最关键的是 `seq_len_q`：

- 在训练阶段，`seq_len_q = N`
- 在 Prefill 阶段，`seq_len_q = N`
- 在 Decode 阶段，`seq_len_q = 1`

也就是说，**是否能靠 Q 块并行把 GPU 喂饱，根本取决于 Q 的长度到底是不是很长。**

#### 场景 1：训练阶段 / Prefill 阶段

这两个场景都要处理整段序列，所以 Q 是完整的长序列。

例如：

```
LLaMA-7B
batch_size = 1
n_heads = 32
Prompt 长度 N = 4096
B_r = 128

T_r = 4096 / 128 = 32
FA2 并行度 = 1 × 32 × 32 = 1024
```

1024 个任务远大于 A100 的 108 个 SM：

```
1024 >> 108
```

结果就是：

- 所有 SM 都能拿到活
- GPU 利用率显著提高
- Prefill 吞吐和训练吞吐都明显提升

这也是为什么 **FA2 不只是训练有用，对长 Prompt 的推理 Prefill 也同样非常有效**。

#### 场景 2：Decode 阶段

Decode 阶段就完全不同了。自回归生成时，每一步只会新来 **1 个 query token**：

```
Q shape: (batch, n_heads, 1, d_head)
K/V shape: (batch, n_heads, context_len, d_head)
```

因此：

```
seq_len_q = 1
T_r = ceil(1 / B_r) = 1
```

FA2 的并行度就退化成：

```
并行度 = batch_size × n_heads × 1
```

如果还是单用户推理：

```
batch_size = 1
n_heads = 32

FA2 并行度 = 1 × 32 × 1 = 32
```

又回到了和 FA1 类似的困境：

- 可并行任务太少
- 很多 SM 依然闲着
- GPU 利用率上不去

所以结论非常重要：

```
FA2 的序列维度并行:
  对训练阶段：有效
  对 Prefill 阶段：有效
  对 Decode 阶段：不够
```

这也正好引出下一节的核心动机：

```
既然 Decode 阶段 Q 只有 1，没法沿 Q 维度切分，
那就只能改成沿 KV 序列维度切分。

这就是 FlashDecoding 的核心思想。
```

### 3.5 FA1 vs FA2 性能对比

| 指标 | FlashAttention-1 | FlashAttention-2 |
|------|-----------------|-----------------|
| A100 FP16 TFLOPS 利用率 | 25-40% | **50-73%** |
| 相对标准 Attention 加速 | 2-4× | **5-9×** |
| 非 GEMM FLOPs 占比 | ~50% | ~25% |
| 内存效率 | $O(N)$ | $O(N)$ |
| 反向传播速度 | 基准 | 快 1.3-1.5× |

---

## 四、FlashDecoding：推理 Decode 阶段的优化

### 4.1 Decode 阶段的问题

```
Prefill 阶段 (处理整个 prompt):
  Q: (batch, n_heads, seq_len, d_head)   ← seq_len 可能很长
  K: (batch, n_heads, seq_len, d_head)
  → 矩阵乘法 Q×K^T 有充足的并行度
  → FlashAttention-2 工作良好

Decode 阶段 (每步生成 1 个 token):
  Q: (batch, n_heads, 1, d_head)          ← 只有 1 个 query!
  K: (batch, n_heads, context_len, d_head) ← context 很长
  
  FA2 的并行度 = batch × n_heads × T_r
  当 batch=1: T_r = ceil(1/B_r) = 1
  → 并行度 = 1 × 32 × 1 = 32 → 严重不足!
  
  问题: Decode 阶段 GPU 利用率极低
```

### 4.2 FlashDecoding 的解决方案

FlashDecoding（Tri Dao et al., 2023）的核心思想：**沿 KV 序列维度并行**。

```
标准 FA2 (Decode):
  Q(1个token) × K^T → 遍历所有 KV → 串行累积结果
  并行度: batch × n_heads

FlashDecoding:
  将 KV 序列分成 P 个分片 (splits):
    KV_1 = K[0:s/P],     KV_2 = K[s/P:2s/P], ...

  Step 1: 每个分片独立计算 partial attention
    O_1 = Attn(Q, K_1, V_1), m_1, ℓ_1    ← 在 SM_1 上
    O_2 = Attn(Q, K_2, V_2), m_2, ℓ_2    ← 在 SM_2 上
    O_3 = Attn(Q, K_3, V_3), m_3, ℓ_3    ← 在 SM_3 上
    ...                                    ← P 个分片并行!
    
  Step 2: reduce — 用 Online Softmax 合并所有分片结果
    m_global = max(m_1, m_2, ..., m_P)
    对每个分片: 用 exp(m_i - m_global) 修正
    O_final = weighted_sum / total_ℓ

  并行度: batch × n_heads × P
  当 P=32: 并行度从 32 → 1024 → 充分利用 GPU!
```

### 4.3 FlashDecoding 的 reduce 步骤

```python
def flash_decoding_reduce(partial_outputs, partial_maxes, partial_sums):
    """
    partial_outputs: list of (d_head,) tensors — 每个分片的部分输出
    partial_maxes:   list of scalars — 每个分片的行最大值
    partial_sums:    list of scalars — 每个分片的 exp 行和
    """
    m_global = max(partial_maxes)

    corrected_outputs = []
    corrected_sums = []
    for o_i, m_i, l_i in zip(partial_outputs, partial_maxes, partial_sums):
        correction = math.exp(m_i - m_global)
        corrected_outputs.append(o_i * correction * l_i)
        corrected_sums.append(l_i * correction)

    total_output = sum(corrected_outputs)
    total_sum = sum(corrected_sums)
    return total_output / total_sum
```

### 4.4 FlashDecoding 性能

| 序列长度 | FA2 Decode | FlashDecoding | 加速比 |
|---------|-----------|--------------|--------|
| 2K | 基准 | 快 | 1.5-2× |
| 8K | 基准 | 快 | 3-5× |
| 32K | 基准 | 快 | 5-8× |
| 128K | 基准 | 快 | 8-16× |

**核心价值**：序列越长，FlashDecoding 的加速效果越显著。对 128K 上下文模型尤为重要。

---

## 五、FlashAttention-3：Hopper GPU 优化

### 5.1 背景

- **目标 GPU**：NVIDIA H100（Hopper 架构）
- **新硬件特性**：WGMMA 指令、TMA（Tensor Memory Accelerator）、FP8 支持

### 5.2 核心优化

```
FA3 的三大优化:

1. 异步流水线 (Warp Specialization):
   将计算分为 producer 和 consumer warp:
     Producer warp: 负责从 HBM 加载数据到共享内存 (TMA)
     Consumer warp: 负责矩阵乘法计算 (WGMMA)
   → 数据加载与计算完全重叠!
   
   FA2: 加载 → 计算 → 加载 → 计算 (串行)
   FA3: 加载──────────────────────
        　　 计算──────────────────  (并行流水线)

2. FP8 低精度支持:
   H100 FP8 Tensor Core: 约 2× FP16 吞吐
   但 FP8 精度低 → 需要逐块量化 + 不连续布局处理
   FA3 提供 FP8 路径，在精度可接受时获得 2× 加速

3. 硬件感知分块:
   利用 H100 的 cluster-level 同步
   更大的 tile size → 更高的计算/IO 比
```

### 5.3 FA3 性能

| GPU | 方法 | FP16 TFLOPS | 利用率 |
|-----|------|------------|--------|
| A100 | FA2 | ~170 | 55% |
| H100 | FA2 | ~350 | 35% |
| H100 | **FA3** | **~620** | **62%** |
| H100 | FA3 (FP8) | ~1200 | ~75% |

---

## 六、FlashAttention 全家族性能 Benchmark

### 6.1 不同序列长度的加速比（A100，FP16，头维度 128）

| 序列长度 | 标准 Attention | FA1 | FA2 | 备注 |
|---------|--------------|-----|-----|------|
| 512 | 基准 | 2× | 3× | 短序列收益有限 |
| 1K | 基准 | 2.5× | 4× | |
| 2K | 基准 | 3× | 5× | |
| 4K | 基准 | 3.5× | 6× | |
| 8K | OOM | 4× | 7× | 标准 Attention 显存不足 |
| 16K | OOM | 5× | 8× | |
| 64K | OOM | — | 9× | FA1 也可能 OOM |

### 6.2 为什么短序列加速小

```
FlashAttention 的加速来源: 减少 HBM IO

短序列 (N=512):
  S 矩阵: 512² × 2 = 0.5 MB → 可能放进 L2 cache
  → HBM IO 本身不是瓶颈
  → FlashAttention 的 tiling 开销反而增加了
  
长序列 (N=8K):
  S 矩阵: 8192² × 2 = 128 MB → 远超 L2 cache
  → HBM IO 是严重瓶颈
  → FlashAttention 大幅减少 IO → 显著加速

结论: FlashAttention 对长序列越有效
```

### 6.3 显存对比

| 序列长度 | 标准 Attention | FlashAttention |
|---------|--------------|---------------|
| 1K | $O(N^2) = 2$ MB | $O(N) = 4$ KB |
| 4K | 32 MB | 16 KB |
| 16K | 512 MB | 64 KB |
| 64K | 8 GB | 256 KB |
| 256K | 128 GB → OOM | 1 MB |

---

## 七、FlashAttention 在 PyTorch 中的使用

### 7.1 `torch.nn.functional.scaled_dot_product_attention`

从 PyTorch 2.0 开始，SDPA 自动选择后端（包括 FlashAttention-2）：

```python
import torch
import torch.nn.functional as F

Q = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)
K = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)
V = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)

# 自动选择 FlashAttention-2 后端
output = F.scaled_dot_product_attention(Q, K, V)
# shape: (1, 32, 4096, 128)
```

### 7.2 选择后端

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

# 强制使用 FlashAttention
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(Q, K, V)

# 强制使用 Memory-Efficient Attention (xformers)
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    output = F.scaled_dot_product_attention(Q, K, V)

# 强制使用数学实现（用于调试/验证）
with sdpa_kernel(SDPBackend.MATH):
    output = F.scaled_dot_product_attention(Q, K, V)
```

### 7.3 GQA 场景

```python
# GQA: n_heads=32, n_kv_heads=8
Q = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)
K = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.float16)
V = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.float16)

# 需要 expand KV heads
K = K.repeat_interleave(4, dim=1)  # (1, 32, 4096, 128)
V = V.repeat_interleave(4, dim=1)

output = F.scaled_dot_product_attention(Q, K, V)
```

---

## 八、面试要点总结

### 8.1 必须掌握的知识点

```
面试高频问题:

Q1: "FlashAttention 的核心思想是什么?"
  → 通过 tiling 避免将 N×N 矩阵写入 HBM
  → Online Softmax 实现分块计算
  → IO 复杂度从 O(N²) 降至 O(N²d/M)

Q2: "FA 的反向传播怎么做? 不存 S 和 P 怎么算梯度?"
  → Recomputation: 反向传播时重新从 Q,K 计算 S=QK^T
  → 用前向保存的 m, ℓ 恢复 P
  → 多计算一次 GEMM，但省了 O(N²) 的 HBM IO
  → 在现代 GPU 上，重计算比读 HBM 更快

Q3: "FA2 比 FA1 快在哪?"
  → 三个改进: 减少非 GEMM 运算、调换内外循环、序列维度并行

Q4: "FlashDecoding 解决什么问题?"
  → Decode 阶段 Q 只有 1 个 token，并行度不足
  → 沿 KV 序列维度分片并行，最后 reduce 合并
  → 长序列加速最高 16×

Q5: "手写 FlashAttention 前向伪代码"
  → 见第一章 1.4 节（明天 Day3 会实际实现）
```

---

## 九、自检题

1. **标准 Attention 的 IO 瓶颈在哪里？** 为什么 $N \times N$ 矩阵是问题？
2. **Online Softmax 如何解决分块计算 softmax 的问题？** 写出更新公式。
3. **FlashAttention 的 IO 复杂度是多少？** 与标准 Attention 相比减少了多少？
4. **FA 反向传播的 recomputation 策略：** 为什么重新计算反而更快？
5. **FA2 的三个核心改进分别是什么？** 各带来了多少性能提升？
6. **FA2 调换内外循环的好处是什么？** 对 HBM 读写有什么影响？
7. **Decode 阶段为什么 FA2 的 GPU 利用率低？** FlashDecoding 如何解决？
8. **FlashDecoding 的 reduce 步骤是什么？** 为什么需要 Online Softmax？
9. **FA3 利用了 H100 的哪些新硬件特性？**
10. **FlashAttention 对短序列加速不明显，为什么？**

---

## 十、产出要求

- [ ] 能手写 FlashAttention 前向传播伪代码（带 Online Softmax）
- [ ] 能解释反向传播的 recomputation 策略
- [ ] 列出 FA2 相比 FA1 的三个改进并说明原理
- [ ] 能解释 FlashDecoding 的工作原理和适用场景
- [ ] 对比 FA1 / FA2 / FlashDecoding 的加速效果
- [ ] 知道如何在 PyTorch 中使用 `scaled_dot_product_attention`
