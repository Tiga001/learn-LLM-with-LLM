# Day 7：Megatron 与全并行策略复盘 — 论文精读 + 知识串联

> **目标**：精读两篇分布式训练的奠基论文——Megatron-LM（张量并行）和 ZeRO（显存优化）；串联本周 Day 1~6 的全部知识——从 DP 到 ZeRO 到 TP/PP 到 MoE/EP；建立完整的分布式训练知识体系；理解计算-通信重叠的工程优化思路。
>
> **前置知识**：本周 Day 1-6 全部内容。

---

## Part 1：Megatron-LM 论文精读

### 1.1 论文信息

| 项目 | 信息 |
|------|------|
| 标题 | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism |
| 作者 | Shoeybi, Patwary, Puri, Carber, Treviso, Catanzaro (NVIDIA) |
| 年份 | 2019 (arXiv: 1909.08053) |
| 贡献 | 首次提出高效张量并行方案，在 Transformer 的 MLP 和 Attention 中实现层内并行 |
| 后续 | Megatron-LM v2 (3D 并行, 2021), v3 (Sequence Parallelism, 2022) |

### 1.2 精读指南（3-Pass）

**第一遍（10 分钟）**：
- 读 Abstract + Introduction + Conclusion
- 把握核心贡献：Transformer 的高效 TP 实现
- 关键词：Column Parallel, Row Parallel, 通信量分析

**第二遍（30 分钟）**：
- 重点读 Section 3 (Model Parallel Transformers)
- 理解 MLP 和 Attention 的切分策略
- 理解 $f$ / $g$ 通信算子的设计
- 看 Figure 3 和 Figure 4（TP 数据流图）

**第三遍（30 分钟）**：
- 读 Section 4 (实验结果)
- 理解 scaling 效率分析
- 对比不同 TP 度数的吞吐量
- 思考论文的局限性

### 1.3 核心贡献

#### 贡献一：MLP 的张量并行

Megatron 的核心洞察是利用 GELU 的非线性性质设计 TP 策略：

$$Y = \text{GELU}(XA)$$

**关键**：GELU 是逐元素操作，可以在列切分后独立计算。

$$A = [A_1 | A_2], \quad \text{GELU}(X[A_1|A_2]) = [\text{GELU}(XA_1) | \text{GELU}(XA_2)]$$

> 这一性质对 ReLU、SiLU 等逐元素激活函数都成立，但对 Softmax 不成立。

因此 MLP 的 TP 策略：

```
前向传播:
  x ─── f(identity) ───→ [列并行 W₁] → GELU → [行并行 W₂] → g(AllReduce) → y

反向传播:
  ∂x ← f̄(AllReduce) ← [列并行 W₁] ← GELU ← [行并行 W₂] ← ḡ(identity) ← ∂y
```

- 前向：1 次 AllReduce（$g$ 操作）
- 反向：1 次 AllReduce（$\bar{f}$ 操作）
- 合计：每 MLP 层 **2 次 AllReduce**

#### 贡献二：Attention 的张量并行

Multi-Head Attention 的多头结构天然适合 TP：

$$\text{head}_i = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

每个 GPU 负责 $n_h / N_{\text{TP}}$ 个头，$W_Q, W_K, W_V$ 列切分，$W_O$ 行切分。

```
前向: x → f → [Q₁K₁V₁→head₁] → [W_O₁] → g(AllReduce) → y
               [Q₂K₂V₂→head₂] → [W_O₂] → ↑
```

- 同样只需 **2 次 AllReduce**（前向 + 反向各 1 次）
- 每个 Transformer Block（Attention + MLP）总共 **4 次 AllReduce**

#### 贡献三：通信算子 $f$ 和 $g$ 的对偶设计

| 算子 | 前向 | 反向 |
|------|------|------|
| $f$ | identity（复制到每卡） | AllReduce（梯度聚合） |
| $g$ | AllReduce（输出聚合） | identity（梯度直接传递） |

$$f + g \text{ 的总通信} = 2 \text{ 次 AllReduce（前向 1 + 反向 1）}$$

这个对偶设计非常优雅：前向和反向各只需 1 次 AllReduce，通信量最小化。

### 1.4 性能结果

论文中的关键实验结果：

| 模型规模 | GPU | TP | 吞吐量 | 扩展效率 |
|---------|:---:|:--:|:------:|:-------:|
| 1.2B | 1 | 1 | 基线 | 100% |
| 1.2B | 2 | 2 | — | ~89% |
| 1.2B | 4 | 4 | — | ~76% |
| 1.2B | 8 | 8 | — | ~72% |
| 8.3B | 8 | 8 | — | — |
| 8.3B | 512 | 8 | — | ~77% |

**关键发现**：
- TP=8（单 node 内 NVLink）时效率约 72~77%
- TP 扩展效率随 TP 度数增大而下降（通信延迟成为瓶颈）
- TP 必须限制在 node 内（NVLink），跨 node 效率急剧下降

### 1.5 论文局限与后续

| 局限 | 后续解决方案 |
|------|------------|
| 只有 TP，无 PP | Megatron v2 引入 PP + Interleaved 1F1B |
| 不支持 ZeRO | 可与 DeepSpeed ZeRO 结合使用 |
| TP 跨 node 效率低 | 限制 TP 在 node 内，跨 node 用 DP/PP |
| 未考虑长序列 | Ring Attention / Context Parallelism |
| 未涉及 MoE | EP + MoE 路由后续被 GShard 等工作解决 |

### 1.6 Megatron-LM 自检题

1. Megatron 的 MLP TP 为什么选择「第一个 Linear 列切分，第二个行切分」？能否反过来？
2. 为什么 GELU 可以在列切分后独立计算，但 Softmax 不行？
3. $f$ 和 $g$ 算子在前向和反向时的行为分别是什么？为什么它们构成对偶关系？
4. 每个 Transformer Block 需要几次 AllReduce？通信量是多少？
5. 为什么 TP 的扩展效率随 TP 度数增大而下降？

---

## Part 2：ZeRO 论文精读

### 2.1 论文信息

| 项目 | 信息 |
|------|------|
| 标题 | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models |
| 作者 | Rajbhandari, Rasley, Ruwase, He (Microsoft) |
| 年份 | 2020 (SC '20) |
| 贡献 | 提出三阶段显存优化，在不增加（或少量增加）通信的前提下消除 DP 的显存冗余 |
| 后续 | ZeRO-Offload (2021), ZeRO-Infinity (2021), FSDP (PyTorch 原生实现) |

### 2.2 核心主张

ZeRO 论文的核心论点：

> 数据并行中每张 GPU 都持有完整的参数、梯度和优化器状态，这是**不必要的冗余**。通过将这些数据分片到多张 GPU，可以在保持 DP 通信效率的同时大幅减少每卡显存。

$$\text{DP 每卡显存} = \underbrace{2\Phi}_{\text{参数}} + \underbrace{2\Phi}_{\text{梯度}} + \underbrace{12\Phi}_{\text{优化器}} = 16\Phi$$

### 2.3 显存分析的精确推导

#### ZeRO Stage 1（优化器分片）

- 分片：优化器状态（FP32 参数副本 + $m$ + $v$）
- 不分片：参数（FP16）、梯度（FP16）

$$\text{ZeRO-1} = \underbrace{2\Phi}_{\text{参数}} + \underbrace{2\Phi}_{\text{梯度}} + \underbrace{\frac{12\Phi}{N_d}}_{\text{优化器分片}} = 4\Phi + \frac{12\Phi}{N_d}$$

#### ZeRO Stage 2（梯度 + 优化器分片）

- 分片：梯度 + 优化器状态
- 不分片：参数

$$\text{ZeRO-2} = \underbrace{2\Phi}_{\text{参数}} + \underbrace{\frac{2\Phi}{N_d}}_{\text{梯度分片}} + \underbrace{\frac{12\Phi}{N_d}}_{\text{优化器分片}} = 2\Phi + \frac{14\Phi}{N_d}$$

#### ZeRO Stage 3（全分片）

- 分片：参数 + 梯度 + 优化器状态

$$\text{ZeRO-3} = \underbrace{\frac{2\Phi}{N_d}}_{\text{参数分片}} + \underbrace{\frac{2\Phi}{N_d}}_{\text{梯度分片}} + \underbrace{\frac{12\Phi}{N_d}}_{\text{优化器分片}} = \frac{16\Phi}{N_d}$$

### 2.4 通信分析

ZeRO 论文的一个关键贡献是通信量分析——证明 ZeRO-2 不增加通信：

| 阶段 | 通信操作 | 通信量 | vs DP |
|------|---------|:------:|:-----:|
| DP | AllReduce 梯度 | $2\Phi$ | 1× |
| ZeRO-1 | AllReduce 梯度 + AllGather 参数 | $3\Phi$ | 1.5× |
| ZeRO-2 | ReduceScatter 梯度 + AllGather 参数 | $2\Phi$ | **1×** |
| ZeRO-3 | AllGather 参数(fwd) + AllGather(bwd) + ReduceScatter 梯度 | $3\Phi$ | 1.5× |

**为什么 ZeRO-2 通信量 = DP？**

- DP 的 AllReduce = ReduceScatter + AllGather（通信量 $2\Phi$）
- ZeRO-2 将 AllReduce 拆成独立的 ReduceScatter（梯度，$\Phi$）和 AllGather（参数更新，$\Phi$）
- 总通信量相同！但中间可以释放梯度显存

### 2.5 ZeRO 的三个关键洞察

**洞察一：优化器状态占 75% 显存**

$$\frac{12\Phi}{16\Phi} = 75\%$$

→ ZeRO-1 只分片优化器就能节省大量显存（边际收益最大）

**洞察二：分片 ≠ 通信**

分片数据不意味着通信量增加——关键在于何时需要完整数据：
- 前向/反向：需要完整参数（ZeRO-3 需 AllGather）
- 梯度聚合：只需保留自己的分片（ReduceScatter 即可）
- 参数更新：只更新自己的分片

**洞察三：通信与计算可以重叠**

```
计算 Layer l 的反向 ──── 同时 ReduceScatter Layer l 的梯度
                    ──── 同时 AllGather Layer l+1 的参数 (prefetch)
```

### 2.6 ZeRO 的工程实现细节

#### 参数分片粒度

ZeRO-3 的分片不是按层分，而是按**连续内存块**分：

```
模型参数 (展平后):  [θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ θ₇ θ₈ θ₉ θ₁₀ θ₁₁ θ₁₂]

GPU 0 持有: [θ₁ θ₂ θ₃]
GPU 1 持有: [θ₄ θ₅ θ₆]
GPU 2 持有: [θ₇ θ₈ θ₉]
GPU 3 持有: [θ₁₀ θ₁₁ θ₁₂]
```

不按层分的原因：
- 各层参数量不同，按层分会导致显存不均衡
- 连续分片更适合高效通信

#### Prefetch 与 Overlap

ZeRO-3 通过预取（prefetch）实现计算-通信重叠：

```
计算 Layer l 的前向  ──── 同时 AllGather Layer l+1 的参数
计算 Layer l 的反向  ──── 同时 AllGather Layer l-1 的参数
                    ──── 同时 ReduceScatter Layer l 的梯度
```

DeepSpeed 通过 `stage3_prefetch_bucket_size` 控制预取粒度。

#### 小参数的特殊处理

`stage3_param_persistence_threshold` 决定哪些参数不分片：

- LayerNorm 的 $\gamma, \beta$（维度 $d$，很小）
- Embedding（虽然大，但访问模式特殊）
- 小于阈值的参数每卡完整持有，避免频繁 AllGather 的延迟

### 2.7 ZeRO 自检题

1. ZeRO 的核心动机是什么？它解决了 DP 的什么问题？
2. 写出 ZeRO Stage 1/2/3 的每卡显存公式，并说明每个阶段切分了什么。
3. 为什么 ZeRO-2 的通信量与 DP 相同？从 AllReduce = ReduceScatter + AllGather 的角度解释。
4. ZeRO-3 的通信量为什么是 DP 的 1.5 倍？额外的通信来自哪里？
5. 如果只能选一个 ZeRO 阶段，在什么场景下选 ZeRO-1？什么场景下选 ZeRO-3？

---

## Part 3：全周复盘

### 3.1 Day 1~7 知识链路

```
Day 1: 分布式训练概览与数据并行
┌─────────────────────────────────────────────────────┐
│ 单卡瓶颈(显存+计算) → 五大并行维度 → DP/DDP 原理     │
│ → AllReduce(Ring) → 通信原语 → AMP 混合精度          │
│ → DP 的显存冗余问题                                  │
└───────────────────────┬─────────────────────────────┘
                        ↓
Day 2: ZeRO 显存优化
┌─────────────────────────────────────────────────────┐
│ 显存四大组成(16Φ) → ZeRO-1(OS分片, 4Φ+12Φ/Nd)      │
│ → ZeRO-2(+梯度, 2Φ+14Φ/Nd) → ZeRO-3(全分片, 16Φ/Nd)│
│ → Offload/Infinity → DeepSpeed配置 → FSDP对比        │
└───────────────────────┬─────────────────────────────┘
                        ↓
Day 3: 手写分布式训练核心组件
┌─────────────────────────────────────────────────────┐
│ Ring AllReduce 实现 → DP 训练循环 → ZeRO-1/2 实现    │
│ → Column Parallel Linear → Row Parallel Linear       │
│ → TP 应用于 MLP → 显存验证                           │
└───────────────────────┬─────────────────────────────┘
                        ↓
Day 4: 张量并行与流水线并行
┌─────────────────────────────────────────────────────┐
│ Megatron TP (MLP+Attention) → TP 通信分析            │
│ → PP 原理 → GPipe → 1F1B → Interleaved → DualPipe   │
│ → CP(Ring Attention) → EP(AllToAll)                  │
│ → 3D/4D/5D 并行组合 → 计算-通信重叠                  │
└───────────────────────┬─────────────────────────────┘
                        ↓
Day 5: MoE 混合专家架构
┌─────────────────────────────────────────────────────┐
│ 稀疏vs稠密 → Router/Expert/Gate → Top-K路由          │
│ → 负载均衡(Aux Loss + Z-Loss) → Switch Transformer   │
│ → Mixtral 8x7B → DeepSeek-V2/V3 MoE                │
│ → 训练挑战(路由崩塌) → 推理优化 → 参数vs计算分析      │
└───────────────────────┬─────────────────────────────┘
                        ↓
Day 6: 手写 MoE 与分布式训练实践
┌─────────────────────────────────────────────────────┐
│ TopK Router 实现 → SwiGLU Expert → MoE Layer 实现    │
│ → MoE Transformer Block → Routing 可视化             │
│ → 负载均衡验证 → 参数量分析 → EP 模拟                │
│ → 端到端 MoE 训练循环                                │
└───────────────────────┬─────────────────────────────┘
                        ↓
Day 7: 论文精读 + 全周复盘（你在这里）
┌─────────────────────────────────────────────────────┐
│ Megatron-LM 精读 → ZeRO 精读 → 全周知识串联          │
│ → 核心概念关系图 → 公式速查 → 常见疑惑 Q&A           │
└─────────────────────────────────────────────────────┘
```

### 3.2 核心概念关系图

```
                        ┌──────────────────────────┐
                        │     分布式训练全景         │
                        └────────────┬─────────────┘
              ┌──────────┬───────────┼───────────┬──────────┐
              ▼          ▼           ▼           ▼          ▼
          ┌──────┐  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
          │  DP  │  │  TP  │   │  PP  │   │  CP  │   │  EP  │
          │数据  │  │张量  │   │流水线 │   │上下文│   │专家  │
          │并行  │  │并行  │   │并行   │   │并行  │   │并行  │
          └──┬───┘  └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘
             │         │          │          │          │
        AllReduce   AllReduce   P2P Send   Ring      AllToAll
          梯度       激活值     激活/梯度  KV传递     token路由
             │         │          │          │          │
             ▼         │          │          │          │
          ┌──────┐     │          │          │          │
          │ ZeRO │     │          │          │          │
          │显存  │     │          │          │          │
          │优化  │     │          │          │          │
          └──┬───┘     │          │          │          │
             │         │          │          │          │
     ┌───────┼─────────┼──────────┼──────────┘          │
     │       │         │          │                     │
     ▼       ▼         ▼          ▼                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │              3D/4D/5D 并行组合                           │
  │  Megatron-LM: DP × TP × PP                             │
  │  LLaMA-3:     DP × TP × PP × CP                        │
  │  DeepSeek-V3: DP × TP × PP × CP × EP                   │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
                   ┌───────────┐
                   │    MoE    │
                   │ 混合专家  │──→ EP (专家并行)
                   │          │──→ Router (Top-K)
                   │          │──→ 负载均衡 (Aux Loss)
                   └───────────┘
```

### 3.3 全周自检清单

#### 理论维度

- [ ] **DP**：为什么数据并行在数学上等价于大 batch 训练？
- [ ] **AllReduce**：Ring AllReduce 的两个阶段分别做什么？每 GPU 通信量公式是什么？
- [ ] **ZeRO**：默写 Stage 1/2/3 的每卡显存公式（面试必考！）
- [ ] **ZeRO**：为什么 ZeRO-2 通信量与 DP 相同？ZeRO-3 多 50%？
- [ ] **TP**：为什么 MLP 的第一个 Linear 用列切分、第二个用行切分？
- [ ] **TP**：每个 Transformer Block 需要几次 AllReduce？
- [ ] **PP**：写出 GPipe 和 1F1B 的 bubble ratio 公式。
- [ ] **PP**：Interleaved 1F1B 为什么能减少 bubble？
- [ ] **CP**：Ring Attention 的通信模式是什么？为什么需要 Online Softmax？
- [ ] **EP**：AllToAll 通信与 AllReduce 有什么区别？
- [ ] **MoE**：什么是路由崩塌？如何通过辅助损失缓解？
- [ ] **MoE**：区分总参数和激活参数，为什么说「计算量由激活参数决定」？

#### 数学维度

- [ ] 显存公式：$\text{Memory} = 2\Phi + 2\Phi + 12\Phi = 16\Phi$
- [ ] ZeRO-1：$4\Phi + 12\Phi/N_d$
- [ ] ZeRO-2：$2\Phi + 14\Phi/N_d$
- [ ] ZeRO-3：$16\Phi/N_d$
- [ ] Ring AllReduce 通信量：$2 \times \frac{N_d - 1}{N_d} \times M$
- [ ] PP bubble ratio：$\frac{P-1}{P-1+M}$
- [ ] MoE 辅助损失：$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_i f_i \cdot P_i$
- [ ] MoE 激活参数：$N_{\text{active}} = N_{\text{non-MoE}} + K \times N_{\text{expert}}$

#### 代码维度

- [ ] 能否手写 Ring AllReduce 的 ReduceScatter + AllGather？
- [ ] 能否手写 ZeRO-1 的优化器分片逻辑？
- [ ] 能否手写 Column Parallel Linear 和 Row Parallel Linear？
- [ ] 能否手写 Top-K Router（含 Softmax、TopK、归一化）？
- [ ] 能否手写 MoE Layer（Router + Expert 分发 + 加权求和）？
- [ ] 能否手写辅助负载均衡 Loss？

#### 工程维度

- [ ] 能否配置 DeepSpeed ZeRO Stage 1/2/3 的 JSON 文件？
- [ ] 理解 PyTorch DDP 的 Gradient Bucketing 机制？
- [ ] 理解 TP 必须放在 node 内（NVLink）的原因？
- [ ] 理解计算-通信重叠（Overlap）的基本原理？
- [ ] 能否设计一个 128 GPU 训练 70B 模型的并行策略？

### 3.4 重要公式速查卡

#### 显存分析

$$\boxed{\text{混合精度 AdamW 训练每参数 16 bytes} = 2(\text{P}) + 2(\text{G}) + 4(\text{FP32 copy}) + 4(m) + 4(v)}$$

| 模型 | $\Phi$ | DP (每卡) | ZeRO-3 (8 GPU) | ZeRO-3 (64 GPU) |
|------|:------:|:---------:|:--------------:|:---------------:|
| 7B | $7 \times 10^9$ | 112 GB | 14 GB | 1.75 GB |
| 13B | $13 \times 10^9$ | 208 GB | 26 GB | 3.25 GB |
| 70B | $70 \times 10^9$ | 1120 GB | 140 GB | 17.5 GB |

#### 通信分析

| 原语 | 每 GPU 通信量 | 用途 |
|------|:------------:|------|
| AllReduce | $2\frac{N_d-1}{N_d}M$ | DP 梯度、TP 激活 |
| AllGather | $\frac{N_d-1}{N_d}M$ | ZeRO-3 参数收集 |
| ReduceScatter | $\frac{N_d-1}{N_d}M$ | ZeRO-2/3 梯度分片 |
| AllToAll | 取决于路由分布 | EP 专家路由 |

#### TP 通信

$$\text{每 Transformer Block TP 通信} = 4 \times \text{AllReduce} = 4 \times 2bsd$$

#### PP Bubble

$$r_{\text{GPipe}} = r_{\text{1F1B}} = \frac{P-1}{P-1+M}$$

$$r_{\text{interleaved}} = \frac{P-1}{(P-1+M) \times v}$$

#### MoE

$$N_{\text{total}} = N_{\text{non-MoE}} + E \times N_{\text{expert}}$$

$$N_{\text{active}} = N_{\text{non-MoE}} + K \times N_{\text{expert}}$$

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i, \quad f_i = \frac{\text{count}_i}{T}, \quad P_i = \frac{1}{T}\sum_{x} g_i(x)$$

### 3.5 常见疑惑 Q&A

**Q1：ZeRO-3 和 FSDP 到底有什么区别？应该选哪个？**

A：核心原理完全相同（参数+梯度+优化器全分片）。区别在于实现和生态：
- DeepSpeed ZeRO-3：JSON 配置驱动，支持 Offload/Infinity，生态成熟
- PyTorch FSDP：Python API 驱动，PyTorch 原生，社区快速发展
- 选择建议：需要 NVMe Offload → DeepSpeed；纯 PyTorch 生态 → FSDP

**Q2：TP 和 PP 都是切分模型，什么时候用哪个？**

A：
- TP 切分层内（更细粒度），通信频繁但数据量小 → **需要 NVLink**，限 node 内
- PP 切分层间（更粗粒度），通信量大但频率低 → **可跨 node**
- 实际中 TP + PP 组合使用：TP 在 node 内（8 GPU），PP 跨 node

**Q3：MoE 的总参数这么多，推理时显存不是很大吗？**

A：是的。MoE 推理的显存瓶颈在于需要加载所有专家参数。解决方案：
- Expert Offloading：不活跃的专家放 CPU 内存
- 量化：INT4/INT8 大幅减少显存
- EP 推理：多卡分担专家

**Q4：为什么 DeepSeek-V3 不用 TP（TP=1）？**

A：DeepSeek-V3 用 EP 替代 TP 的角色。原因：
- MoE 模型中大量参数是专家参数，EP 自然地分片了这些参数
- EP 的 AllToAll 通信可以与 Attention 计算重叠（DualPipe 设计）
- 避免了 TP 在每层多次 AllReduce 的通信开销

**Q5：训练一个 70B 模型，到底需要多少 GPU？**

A：取决于并行策略和显存预算。以 A100 80GB 为例：

| 策略 | 最少 GPU 数 | 每卡显存 |
|------|:---------:|:-------:|
| ZeRO-3 (8 GPU) | 8 | ~14 GB (参数) + 激活 |
| TP=8 + PP=2 + DP=4 (64 GPU) | 64 | ~9 GB + 激活 |
| TP=8 + PP=4 + DP=4 (128 GPU) | 128 | ~4.5 GB + 激活 |

> 实际还需考虑激活值显存、通信 overhead、吞吐量需求等因素。

**Q6：什么是 Gradient Accumulation？和 DP 有什么关系？**

A：Gradient Accumulation 在单卡上模拟大 batch：

$$g_{\text{accum}} = \frac{1}{K}\sum_{k=1}^{K} g_k$$

- 前向 + 反向 K 次，梯度累加，然后一次更新
- 效果等价于 K 倍 batch size
- 与 DP 正交：DP 跨卡并行，GA 跨步串行
- 实际 effective batch size = GPU数 × 单卡 batch × GA steps

**Q7：BF16 和 FP16 到底用哪个？**

A：
- BF16：动态范围与 FP32 相同，通常不需要 Loss Scaling → **大模型训练首选**
- FP16：精度更高但动态范围小，需要 Loss Scaling → 旧系统兼容
- A100+ 同时支持两者，强烈推荐 BF16

**Q8：辅助负载均衡 Loss 的 $\alpha$ 怎么调？调大了会怎样？**

A：
- 典型值：$\alpha = 0.01$
- 太小：负载均衡效果差，路由崩塌
- 太大：辅助 Loss 主导训练，模型质量下降（路由过于均匀，失去专业化能力）
- DeepSeek-V3 的创新：用动态偏置替代辅助 Loss，从根本上避免了这个问题

### 3.6 计算-通信重叠（Overlap）深入

#### 3.6.1 Overlap 的核心原则

$$T_{\text{total}} = T_{\text{comp}} + T_{\text{comm}} \quad \xrightarrow{\text{overlap}} \quad T_{\text{total}} \approx \max(T_{\text{comp}}, T_{\text{comm}})$$

理想情况下，通信完全隐藏在计算之后。实际受限于：
- GPU 的计算和通信是否可以并发（需要不同的硬件单元）
- 数据依赖关系（需要 A 的通信结果才能计算 B）

#### 3.6.2 各并行维度的 Overlap 机会

| 并行 | Overlap 策略 | 实现 |
|------|-------------|------|
| DP (DDP) | 反向计算 + 梯度 AllReduce | Gradient Bucketing |
| ZeRO-3 | 当前层计算 + 下一层参数 AllGather | Prefetch |
| TP | 无法 overlap（计算依赖通信结果） | — |
| PP | 当前 micro-batch 反向 + 下一个前向 | 1F1B 调度 |
| EP | Attention 计算 + MLP 的 AllToAll | DualPipe |

#### 3.6.3 DDP Gradient Bucketing 详解

```
反向传播计算:  ← Layer 8 ← Layer 7 ← Layer 6 ← ... ← Layer 1

Bucket 分配:   [Bucket 3: L8,L7] [Bucket 2: L6,L5] [Bucket 1: L4~L1]

AllReduce:              [AR Bucket 3]   [AR Bucket 2]   [AR Bucket 1]

时间轴:      ├──────────┼──────────┼──────────┼──────────┤
             反向 L8-L7   反向 L6-L5   反向 L4-L1
                      AR Bucket 3   AR Bucket 2   AR Bucket 1
```

- 越早完成反向的参数（靠后的层）越早开始 AllReduce
- Bucket 大小影响 overlap 效率和通信效率的 trade-off

#### 3.6.4 DeepSeek-V3 的精细 Overlap

DeepSeek-V3 不使用 TP（TP=1），通过更精细的调度实现 overlap：

```
Transformer Block 拆分:
┌────────────────────────────────────────────────────────┐
│ Phase A: Attention 计算  │ 同时: MoE 的 AllToAll 通信  │
│ Phase B: MoE 专家计算    │ 同时: Attention 的 AllReduce│
└────────────────────────────────────────────────────────┘
```

关键：将一个 Transformer Block 内的计算和通信交错排列，最大化硬件利用率。

### 3.7 并行策略设计实战

#### 案例 1：128 GPU 训练 LLaMA-70B

```
硬件: 16 nodes × 8 GPUs (A100 80GB), NVLink node 内, IB 跨 node

分析:
  参数显存 (BF16): 70B × 2 = 140 GB → 单卡放不下
  优化器 (FP32):   70B × 12 = 840 GB → 更放不下

方案 1: ZeRO-3 + DP=128
  每卡: 16Φ / 128 = 140/128 ≈ 1.1 GB (参数/梯度/优化器)
  + 激活值 → 总显存 < 40 GB ✓
  通信: 3Φ = 420 GB（频繁 AllGather/ReduceScatter）
  适合: 简单配置，但通信可能成为瓶颈

方案 2: TP=8 + PP=2 + DP=8  (128 = 8 × 2 × 8)
  TP=8 在 node 内 (NVLink)
  PP=2 跨 2 个 node
  DP=8 跨剩余 node
  每卡参数: 70B / 8(TP) / 2(PP) ≈ 4.4B → 8.8 GB (BF16)
  + 优化器分片(ZeRO-1 in DP group): 4.4B × 12 / 8 ≈ 6.6 GB
  → 总约 15~20 GB + 激活 ✓

方案 3: TP=8 + PP=4 + DP=4  (128 = 8 × 4 × 4)
  更深的 PP 切分 → 每卡层更少 → 显存更小
  但 bubble ratio = (4-1)/(4-1+M)，需要更多 micro-batches
  适合显存更紧张的场景

推荐: 方案 2（TP=8, PP=2, DP=8 + ZeRO-1）
```

#### 案例 2：2048 GPU 训练 DeepSeek-V3 (671B MoE)

```
硬件: 256 nodes × 8 GPUs (H800 80GB)

分析:
  总参数 671B, 激活参数 37B
  256 路由专家 + 1 共享专家 (每层)
  61 层

DeepSeek-V3 实际方案:
  TP=1  → 不用 TP（用更细的 overlap 替代）
  PP=16 → 16-stage 流水线 (DualPipe)
  DP=128 → 含 ZeRO-1
  EP=64 → 64 路专家并行

为什么 TP=1:
  MoE 模型的大量参数是专家参数
  EP 天然分片了专家参数（每 GPU 4 个路由专家）
  避免了 TP 在每层多次 AllReduce 的通信开销
  用 DualPipe 的 Attention-MLP 交错 overlap 替代 TP

GPU 分组 (2048 = 128 × 16 = 64 × 32):
  EP group: 64 GPUs → 每 GPU 持 256/64 = 4 个路由专家
  PP group: 16 GPUs → 61 层 / 16 ≈ 4 层/stage
  DP group: 128 GPUs → 数据并行
```

#### 策略选择决策树

```
开始
  │
  ├─ 模型能放进单卡？
  │   ├─ 是 → DP（增加吞吐量）
  │   └─ 否 ↓
  │
  ├─ 优化器放不下？
  │   ├─ 是 → 先试 ZeRO-1（最小通信开销）
  │   └─ 仍不够 ↓
  │
  ├─ 梯度也放不下？
  │   ├─ 是 → ZeRO-2 或 ZeRO-3
  │   └─ 不是 → ZeRO-1 + 减小 batch
  │
  ├─ ZeRO-3 通信太慢？
  │   ├─ 是 → 考虑 TP（node 内）+ PP（跨 node）
  │   └─ 否 → ZeRO-3 够用
  │
  ├─ 模型是 MoE？
  │   ├─ 是 → 加 EP（专家并行）
  │   └─ 否 → DP + TP + PP 足矣
  │
  └─ 序列很长（>32K）？
      ├─ 是 → 加 CP（Context Parallelism）
      └─ 否 → 标准方案
```

### 3.8 硬件拓扑与通信带宽

理解硬件拓扑对设计并行策略至关重要：

```
单 Node (DGX A100):
┌─────────────────────────────────────────────┐
│  GPU 0 ══NVLink══ GPU 1 ══NVLink══ GPU 2   │
│    ║                                  ║     │
│  GPU 3 ══NVLink══ GPU 4 ══NVLink══ GPU 5   │
│    ║                                  ║     │
│  GPU 6 ══NVLink══ GPU 7                    │
│                                             │
│  NVLink: 600 GB/s (A100), 900 GB/s (H100)  │
│  NVSwitch 全连接拓扑                         │
└─────────────────────────────────────────────┘
         ║ InfiniBand / RoCE
         ║ 200~400 GB/s
┌─────────────────────────────────────────────┐
│  Node 1 (同样 8 GPU)                        │
└─────────────────────────────────────────────┘
```

| 互联 | 带宽 | 延迟 | 适合 |
|------|:----:|:----:|------|
| NVLink (A100) | 600 GB/s | ~μs | TP |
| NVLink (H100) | 900 GB/s | ~μs | TP |
| InfiniBand HDR | 200 GB/s | ~μs | PP, DP |
| InfiniBand NDR | 400 GB/s | ~μs | PP, DP |
| PCIe Gen4 | 32 GB/s | ~μs | CPU Offload |
| NVMe SSD | 3.5 GB/s | ~ms | ZeRO-Infinity |

> **工程原则**：通信频率高的并行（TP）放 NVLink，通信量大但频率低的（DP）可跨 node。

### 3.10 与第 16 周的衔接

本周（W15）建立的分布式训练和 MoE 知识将直接服务于后续学习：

| W15 内容 | W16 衔接点 |
|---------|-----------|
| DP / ZeRO | VLM（视觉语言模型）训练的分布式策略 |
| TP / PP | 多模态大模型（如 InternVL）的并行方案 |
| MoE | 多模态 MoE（如 MoE-VLM）的设计 |
| EP | 多模态 MoE 的专家并行部署 |
| 计算-通信重叠 | 大规模 VLM 训练的性能优化 |

### 3.11 下周预告：第 16 周 — 多模态大模型（VLM）

```
Day 1: 多模态学习概览与视觉编码器
Day 2: VLM 架构设计 (LLaVA / InternVL / Qwen-VL)
Day 3: 手写视觉-语言对齐模块
Day 4: 多模态训练策略与数据工程
Day 5: 视频理解与多模态推理
Day 6: 手写多模态 VLM 实践
Day 7: 前沿多模态模型复盘
```

> 核心转变：从「如何高效训练大模型」到「如何让大模型理解多模态信息」。

---

## 十一、自检题（全周综合）

### 基础题

1. 画出分布式训练五大并行维度的全景图，标注每种并行切分什么、通信模式是什么。
2. 写出 Ring AllReduce 每 GPU 的通信量公式，并解释为什么与 GPU 数量近似无关。
3. 默写 ZeRO Stage 1/2/3 的每卡显存公式（面试高频！）。
4. 写出 Megatron-LM 中 MLP 和 Attention 的 TP 通信次数。

### 进阶题

5. 以 LLaMA-70B + AdamW + BF16 + 64 GPU 为例，计算 ZeRO-3 每卡参数显存，并估算总显存（含激活）。
6. 设计一个 256 GPU 训练 671B 参数 MoE 模型（如 DeepSeek-V3）的并行策略：DP / TP / PP / EP 分别是多少？说明理由。
7. 解释 DualPipe 如何将 bubble ratio 降低到近 0。
8. Mixtral 8x7B 的总参数 46.7B，激活参数 12.9B。计算一个 token 的前向 FLOPs，并与等激活参数的稠密模型对比。

### 面试题

9. 面试官问：「ZeRO-3 和 Tensor Parallelism 都能减少每卡显存，你会如何选择？」请给出回答。
10. 面试官问：「MoE 模型训练时可能出现什么问题？如何解决？」请给出回答。
11. 面试官问：「解释一下计算-通信重叠（Overlap）的原理，并举一个具体例子。」请给出回答。

---

## 十二、产出要求

- [ ] 写出 Megatron-LM 论文的核心贡献总结（TP 的 MLP 策略、Attention 策略、$f$/$g$ 对偶设计）
- [ ] 写出 ZeRO 论文的核心贡献总结（三阶段显存公式 + 通信分析）
- [ ] 画出本周完整的知识关系图（DP ↔ ZeRO ↔ TP ↔ PP ↔ CP ↔ EP ↔ MoE）
- [ ] 默写全部核心公式（ZeRO 三阶段 + AllReduce 通信量 + PP bubble ratio + MoE 辅助 Loss）
- [ ] 完成全周自检清单（理论 / 数学 / 代码 / 工程四个维度全部打勾）
- [ ] 设计一个大模型训练的并行策略方案（指定 GPU 数和模型大小，给出 DP/TP/PP/EP 配置）
- [ ] 撰写 Megatron-LM TP vs DeepSpeed ZeRO 的使用场景对比（1 页笔记）
