# Day 5：FlashAttention 入门 — IO 感知的高效注意力

> **目标**：理解标准 Attention 的内存瓶颈、GPU 内存层次结构、FlashAttention 的 tiling 思想和 Online Softmax 算法，建立对高效注意力计算的直觉。本日为"理解原理"，深入实现将在第 14 周展开。

---

## 一、为什么需要 FlashAttention？

### 1.1 标准 Attention 的计算与内存代价

回顾标准的 Self-Attention 计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

对于序列长度 $T$、隐藏维度 $d$：

| 步骤 | 计算 | 结果形状 | FLOPs |
|------|------|---------|-------|
| $S = QK^T / \sqrt{d_k}$ | 矩阵乘法 | $T \times T$ | $O(T^2 d)$ |
| $P = \text{softmax}(S)$ | 逐行 softmax | $T \times T$ | $O(T^2)$ |
| $O = PV$ | 矩阵乘法 | $T \times d$ | $O(T^2 d)$ |

**关键瓶颈**：中间矩阵 $S$ 和 $P$ 的形状是 $T \times T$，必须完整存储在内存中。

当 $T = 4096$（GPT-3.5 级别）：

$$T^2 = 16,777,216 \approx 16M \text{ 个浮点数}$$

以 FP16 计算：$16M \times 2B = 32MB$（**每个注意力头**）。如果有 96 个头，仅注意力分数就需要 $\sim 3GB$。

当 $T = 128K$（GPT-4 Turbo）：

$$T^2 = 16,384^2 = 268M \text{ 个浮点数} \approx 512MB \text{ 每头}$$

这远超 GPU 的高速缓存容量。

### 1.2 计算是瓶颈还是内存是瓶颈？

现代 GPU 的计算能力增长速度远超内存带宽增长速度：

| 指标 | A100 GPU |
|------|----------|
| FP16 计算能力 | 312 TFLOPS |
| HBM 带宽 | 2.0 TB/s |
| 计算/带宽比 | ~156 FLOPS/Byte |

这意味着：**GPU 每读取 1 字节数据，可以做 156 次浮点运算。** 如果算法需要频繁读写内存但计算不多，GPU 的计算单元就在"等数据"——这就是 **IO 瓶颈（Memory-bound）**。

标准 Attention 正是 IO-bound 的：
- $QK^T$：计算密集（好）
- softmax：逐元素操作，要读写整个 $T \times T$ 矩阵（坏）
- $PV$：计算密集（好）
- **但 $S, P$ 必须在 HBM 中存储和读取** → 大量 IO

---

## 二、GPU 内存层次结构

理解 FlashAttention 必须先理解 GPU 的内存层次：

```
┌─────────────────────────────────────────────┐
│                  GPU Chip                     │
│                                               │
│  ┌───────────────────────────────────────┐   │
│  │           SRAM (Shared Memory)         │   │
│  │  容量: ~20MB (总, A100 108 SMs × 192KB)│   │
│  │  带宽: ~19 TB/s                        │   │
│  │  延迟: ~几个时钟周期                     │   │
│  └───────────────────────────────────────┘   │
│                    ↕ (~10×)                    │
│  ┌───────────────────────────────────────┐   │
│  │          HBM (High Bandwidth Memory)   │   │
│  │  容量: 40~80GB (A100)                  │   │
│  │  带宽: 1.5~2.0 TB/s                   │   │
│  │  延迟: ~数百时钟周期                     │   │
│  └───────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

| 层级 | 容量 | 带宽 | 比喻 |
|------|------|------|------|
| **SRAM** (片上) | ~20MB | ~19 TB/s | 工作台（小但很快） |
| **HBM** (片外) | 40-80GB | ~2 TB/s | 仓库（大但取货慢） |

**FlashAttention 的核心思想**：不要把整个 $T \times T$ 矩阵存到 HBM（仓库），而是分块在 SRAM（工作台）上计算，每次只处理一小块。

---

## 三、标准 Attention 的 IO 分析

### 3.1 标准算法的 HBM 读写

```
标准 Attention（朴素实现）:

Step 1: S = QK^T         → 从 HBM 读 Q, K，写 S 到 HBM
         HBM 读: O(Td)    写: O(T²)

Step 2: P = softmax(S)    → 从 HBM 读 S，写 P 到 HBM
         HBM 读: O(T²)    写: O(T²)

Step 3: O = PV            → 从 HBM 读 P, V，写 O 到 HBM
         HBM 读: O(T² + Td)  写: O(Td)

总 HBM 访问量: O(T² + Td)
```

关键：$S$ 和 $P$（各 $T \times T$）必须在 HBM 中实体化（materialize），这是最大的 IO 开销。

### 3.2 为什么不能简单地"不存 S"？

因为 softmax 是**全局归一化**操作：

$$\text{softmax}(s_i) = \frac{e^{s_i}}{\sum_{j=1}^{T} e^{s_j}}$$

分母需要知道**整行的所有值**。如果 $T$ 很大，一行 $S$ 放不进 SRAM，就必须先算完整行写到 HBM，再读回来做 softmax。

FlashAttention 的突破在于：**用 Online Softmax 解决了这个问题**。

---

## 四、Online Softmax — FlashAttention 的数学基石

### 4.1 标准 Softmax 的三遍遍历

标准 softmax 计算 $\text{softmax}(x_1, \ldots, x_T)$ 需要三遍遍历输入：

```
Pass 1: 计算 m = max(x_1, ..., x_T)                ← 遍历一次
Pass 2: 计算 d = Σ exp(x_i - m)                     ← 遍历一次
Pass 3: 计算 softmax_i = exp(x_i - m) / d           ← 遍历一次
```

每一遍都需要读取全部数据。如果数据在 HBM 中，就是 3 次完整的 HBM 读取。

### 4.2 Online Softmax（两遍遍历）

**关键洞察**：可以将 Pass 1 和 Pass 2 合并为一遍！

算法：

$$m_j = \max(m_{j-1}, x_j), \quad d_j = d_{j-1} \cdot e^{m_{j-1} - m_j} + e^{x_j - m_j}$$

初始值：$m_0 = -\infty$，$d_0 = 0$。

**推导**：

假设已经处理了 $x_1, \ldots, x_{j-1}$，维护了：
- $m_{j-1} = \max(x_1, \ldots, x_{j-1})$
- $d_{j-1} = \sum_{i=1}^{j-1} e^{x_i - m_{j-1}}$

当新元素 $x_j$ 到来时：

$$m_j = \max(m_{j-1}, x_j)$$

$$d_j = \sum_{i=1}^{j} e^{x_i - m_j} = \underbrace{\sum_{i=1}^{j-1} e^{x_i - m_j}}_{\text{修正旧的求和}} + e^{x_j - m_j}$$

其中旧的求和需要从 $m_{j-1}$ 修正到 $m_j$：

$$\sum_{i=1}^{j-1} e^{x_i - m_j} = \sum_{i=1}^{j-1} e^{x_i - m_{j-1}} \cdot e^{m_{j-1} - m_j} = d_{j-1} \cdot e^{m_{j-1} - m_j}$$

因此：

$$d_j = d_{j-1} \cdot e^{m_{j-1} - m_j} + e^{x_j - m_j}$$

这只需要 $O(1)$ 额外空间！

### 4.3 数值示例

计算 $\text{softmax}([3.0, 1.0, 4.0, 2.0])$：

| 步骤 $j$ | $x_j$ | $m_j$ | 修正因子 $e^{m_{j-1} - m_j}$ | $d_j$ |
|-----------|--------|--------|-------------------------------|-------|
| 0 | — | $-\infty$ | — | 0 |
| 1 | 3.0 | 3.0 | $e^{-\infty - 3} = 0$ | $0 \cdot 0 + e^{3-3} = 1.0$ |
| 2 | 1.0 | 3.0 | $e^{3-3} = 1$ | $1.0 \cdot 1 + e^{1-3} = 1.135$ |
| 3 | 4.0 | 4.0 | $e^{3-4} = 0.368$ | $1.135 \cdot 0.368 + e^{4-4} = 1.417$ |
| 4 | 2.0 | 4.0 | $e^{4-4} = 1$ | $1.417 \cdot 1 + e^{2-4} = 1.553$ |

最终：$m = 4.0$，$d = 1.553$

$$\text{softmax} = \left[\frac{e^{3-4}}{1.553}, \frac{e^{1-4}}{1.553}, \frac{e^{4-4}}{1.553}, \frac{e^{2-4}}{1.553}\right] = [0.237, 0.032, 0.644, 0.087]$$

验证：$0.237 + 0.032 + 0.644 + 0.087 = 1.0$ ✓

---

## 五、FlashAttention 算法：Tiling

### 5.1 核心思想

将 $Q, K, V$（形状 $T \times d$）沿序列维度切成小块（tiles），每块大小 $B_r \times d$ 或 $B_c \times d$（$B_r, B_c$ 远小于 $T$），使得每个 tile 能放进 SRAM。

```
标准 Attention:
  Q (T×d) × K^T (d×T) → S (T×T) → softmax → P (T×T) × V (T×d) → O (T×d)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  S 和 P 是 T×T，太大了！

FlashAttention:
  将 Q 分成 T_r 个块 (每块 B_r × d)
  将 K, V 分成 T_c 个块 (每块 B_c × d)
  
  对每个 Q 块:
    对每个 K, V 块:
      在 SRAM 中计算小的 B_r × B_c 注意力分数
      使用 Online Softmax 增量更新输出
    → 最终输出块 O_i (B_r × d)
```

### 5.2 FlashAttention 伪代码

```python
def flash_attention(Q, K, V):
    """
    Q, K, V: (T, d) 存储在 HBM 中
    返回 O: (T, d)
    """
    T, d = Q.shape
    
    # 选择块大小，使 tile 能放进 SRAM
    B_r = min(SRAM_SIZE // (d * sizeof(float)), T)
    B_c = min(SRAM_SIZE // (d * sizeof(float)), T)
    
    # 初始化输出和统计量（在 HBM 中）
    O = zeros(T, d)           # 输出
    l = zeros(T)              # softmax 分母
    m = full(T, -inf)         # 行最大值
    
    # 外循环：遍历 K, V 的块
    for j in range(0, T, B_c):
        K_j = K[j:j+B_c]     # 从 HBM 加载到 SRAM, (B_c, d)
        V_j = V[j:j+B_c]     # 从 HBM 加载到 SRAM, (B_c, d)
        
        # 内循环：遍历 Q 的块
        for i in range(0, T, B_r):
            Q_i = Q[i:i+B_r]     # 从 HBM 加载到 SRAM, (B_r, d)
            O_i = O[i:i+B_r]     # 从 HBM 加载到 SRAM
            l_i = l[i:i+B_r]     # 从 HBM 加载
            m_i = m[i:i+B_r]     # 从 HBM 加载
            
            # --- 以下全部在 SRAM 中计算 ---
            
            # 计算小的注意力分数块: (B_r, B_c)
            S_ij = Q_i @ K_j.T / sqrt(d)
            
            # 块内的行最大值
            m_ij = S_ij.max(dim=-1)           # (B_r,)
            P_ij = exp(S_ij - m_ij[:, None])  # (B_r, B_c)
            l_ij = P_ij.sum(dim=-1)           # (B_r,)
            
            # Online Softmax 更新
            m_new = max(m_i, m_ij)                    # 新的全局最大值
            l_new = l_i * exp(m_i - m_new) + l_ij * exp(m_ij - m_new)
            
            # 更新输出（关键步骤！）
            O_i = (O_i * l_i[:, None] * exp(m_i - m_new)[:, None] 
                   + P_ij @ V_j * exp(m_ij - m_new)[:, None]) / l_new[:, None]
            
            # 写回 HBM
            O[i:i+B_r] = O_i
            l[i:i+B_r] = l_new
            m[i:i+B_r] = m_new
    
    return O
```

### 5.3 关键操作解读

**输出更新公式**（最难理解的部分）：

$$O_i^{(new)} = \frac{l_i^{(old)} \cdot e^{m_i^{(old)} - m_i^{(new)}} \cdot O_i^{(old)} + e^{m_{ij} - m_i^{(new)}} \cdot \tilde{P}_{ij} V_j}{l_i^{(new)}}$$

**直觉**：每次处理一个新的 KV 块时：
1. 旧的输出 $O_i$ 中包含的 softmax 权重是基于旧的统计量（$m_i^{(old)}, l_i^{(old)}$）计算的
2. 新的 KV 块可能改变了全局的 max 和 sum
3. 所以需要用指数修正因子 $e^{m^{(old)} - m^{(new)}}$ 来调整旧的贡献
4. 然后加上新块的贡献，用新的归一化因子重新归一化

这本质上就是 **Online Softmax 在矩阵乘法中的推广**。

---

## 六、复杂度分析

### 6.1 IO 复杂度对比

| 指标 | 标准 Attention | FlashAttention |
|------|---------------|----------------|
| **HBM 读写** | $O(T^2 + Td)$ | $O(T^2 d / M)$ |
| **额外 HBM 空间** | $O(T^2)$（存 $S, P$） | $O(T)$（只存 $l, m$） |
| **FLOPs** | $O(T^2 d)$ | $O(T^2 d)$（相同！） |

其中 $M$ 是 SRAM 的大小。

**关键洞察**：FlashAttention 不减少计算量（FLOPs 相同），而是减少了 HBM 访问次数（IO）。在 IO-bound 场景下，这带来了巨大的加速。

### 6.2 加速比分析

当 $T = 4096, d = 64, M = 100KB$（典型 SRAM 配置）：

$$\text{标准 IO} = O(T^2) \approx 16M$$

$$\text{FlashAttention IO} = O\left(\frac{T^2 d}{M}\right) \approx \frac{16M \times 64}{100K} = 10M$$

IO 减少了约 $\frac{T \times d}{M}$ 倍 ≈ 2.5 倍。但实际加速比通常更大（2-4×），因为还有其他优化（如减少 kernel launch 等）。

### 6.3 内存节省

标准 Attention 需要存储 $S, P \in \mathbb{R}^{T \times T}$。FlashAttention 只需额外存储 $l, m \in \mathbb{R}^{T}$。

内存节省：$O(T^2) → O(T)$，这是**从二次到线性的改进**。

对于 $T = 128K$：
- 标准：$128K \times 128K \times 2B = 32GB$（每头，FP16）→ **不可能**
- FlashAttention：$128K \times 2 \times 4B = 1MB$（每头）→ **轻松**

---

## 七、FlashAttention 在 GPT 中的应用

### 7.1 与 GPT 的集成

在我们 Day 3 的 GPT 实现中，替换 Attention 计算：

```python
# Day 3 的标准实现
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
y = att @ v

# 使用 FlashAttention（PyTorch 2.0+ 内置支持）
y = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,      # causal mask 在 is_causal=True 时自动处理
    is_causal=True,
    dropout_p=0.0,
)
```

PyTorch 2.0+ 的 `scaled_dot_product_attention` 会自动选择最优后端：
- FlashAttention（Dao-AILab 实现）
- Memory-Efficient Attention（xFormers）
- 标准数学实现（fallback）

### 7.2 因果掩码的处理

FlashAttention 中因果掩码的实现很巧妙：

```
对于块 (i, j)（Q 的第 i 块，K 的第 j 块）:
  - 如果 j > i: 整个块都被 mask → 直接跳过（不计算！）
  - 如果 j < i: 整个块都不被 mask → 正常计算
  - 如果 j == i: 对角线块 → 需要在块内应用部分掩码
```

这意味着因果 Attention 可以跳过约一半的块计算，进一步加速。

---

## 八、FlashAttention 的发展脉络

| 版本 | 时间 | 关键改进 |
|------|------|---------|
| **FlashAttention 1** | 2022.05 | Tiling + Online Softmax，2-4× 加速 |
| **FlashAttention 2** | 2023.07 | 更好的并行策略（沿序列长度并行），减少非矩阵乘 FLOPs，接近理论峰值 |
| **FlashAttention 3** | 2024.07 | 针对 Hopper GPU (H100) 优化，利用 FP8 和异步硬件特性 |
| **FlashDecoding** | 2023.10 | 推理时的优化：沿 KV 序列维度并行，解决长序列推理瓶颈 |

### FlashAttention 2 的核心改进

1. **更好的工作划分**：沿 $T$（序列长度）维度而非 $d$（特征维度）并行 → 减少 SM 间的通信
2. **减少非矩阵乘运算**：将 online softmax 的标量运算最小化 → 接近 Tensor Core 理论 FLOPS
3. **更好的内循环/外循环顺序**：Flash-2 内循环遍历 KV，外循环遍历 Q → 更少的 HBM 读写

---

## 九、FlashAttention 的局限性

| 局限 | 说明 |
|------|------|
| **不返回注意力权重** | $P$ 矩阵不被实体化 → 无法直接可视化注意力模式 |
| **需要特定 GPU** | 依赖 Tensor Core，需要 Ampere/Hopper 架构 |
| **自定义掩码受限** | 支持 causal 和无掩码，但复杂的稀疏掩码可能不支持 |
| **实现复杂** | CUDA kernel 编写难度极高，调试困难 |
| **d 维度限制** | 早期版本要求 $d \leq 128$，现已放宽 |

**注意力权重不可用的解决方案**：
- 用标准 attention 做可视化（仅分析时）
- FlashAttention 论文证明可以通过重计算（recomputation）在反向传播时精确恢复注意力权重

---

## 十、总结与核心要点

### 10.1 三个核心思想

1. **IO 感知（IO-Aware）**：分析算法的内存访问模式比分析 FLOPs 更重要
2. **Tiling（分块计算）**：将大矩阵切成小块，在快速的 SRAM 中计算，避免大矩阵在 HBM 中实体化
3. **Online Softmax**：通过维护递增的统计量 $(m, l)$，使得 softmax 可以分块增量计算

### 10.2 FlashAttention 为什么不减少 FLOPs 却能加速？

```
标准 Attention:
  计算: O(T²d) FLOPs → 快（GPU 擅长）
  IO:   O(T²) 次 HBM 访问 → 慢（GPU 的瓶颈）
  
FlashAttention:
  计算: O(T²d) FLOPs → 相同
  IO:   O(T²d/M) 次 HBM 访问 → 大幅减少！
  
核心: Attention 是 IO-bound 的，减少 IO 比减少计算更有效。
```

### 10.3 知识链接

| 本日内容 | 前序基础 | 后续深化 |
|---------|---------|---------|
| GPU 内存层次 | 计算机体系结构基础 | W14 CUDA 编程 |
| Online Softmax | W2 标准 Softmax 实现 | W14 FlashAttention 实现 |
| Tiling 思想 | 矩阵分块乘法 | W14 FlashAttention 2, FlashDecoding |
| IO 分析 | — | W15 推理优化 / vLLM |

---

## 十一、自检题

1. **为什么标准 Attention 是 IO-bound 的？** GPU 的计算/带宽比意味着什么？
2. **Online Softmax 的更新公式是什么？** 为什么需要修正因子 $e^{m^{(old)} - m^{(new)}}$？
3. **FlashAttention 的分块策略是什么？** 块大小由什么决定？
4. **FlashAttention 的 IO 复杂度是多少？** 相比标准 Attention 节省了多少？
5. **FlashAttention 为什么不减少 FLOPs 却能加速？** 用一句话解释。
6. **FlashAttention 的内存从 $O(T^2)$ 降到了什么？** 为什么这对长序列至关重要？
7. **在 PyTorch 中如何使用 FlashAttention？** 调用哪个函数？

---

## 十二、产出要求

- [ ] 用自己的话解释 GPU 的 SRAM-HBM 内存层次
- [ ] 手推一个 4 元素向量的 Online Softmax 过程
- [ ] 画出 FlashAttention 的分块计算示意图
- [ ] 对比标准 Attention 和 FlashAttention 的 IO / 内存 / FLOPs 复杂度
- [ ] 解释 FlashAttention 如何处理因果掩码
