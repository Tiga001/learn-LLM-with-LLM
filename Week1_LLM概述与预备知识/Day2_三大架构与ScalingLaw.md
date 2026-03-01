# Day 2：三大架构对比与 Scaling Law

> **目标**：深入理解 Encoder-only / Decoder-only / Encoder-Decoder 的设计哲学与适用边界，掌握 Scaling Law 的数学表达与工程意义。

---

## 一、Self-Attention 的数学基础（为架构对比做铺垫）

在讨论三大架构之前，先统一 Attention 的数学语言：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

- $Q \in \mathbb{R}^{n \times d_k}$：Query 矩阵
- $K \in \mathbb{R}^{m \times d_k}$：Key 矩阵
- $V \in \mathbb{R}^{m \times d_v}$：Value 矩阵
- $\sqrt{d_k}$：缩放因子，防止 softmax 进入饱和区（梯度消失）

**核心直觉**：每个 Query 向量通过与所有 Key 做点积来计算"关注度"，然后用这个关注度对 Value 加权求和。

三种架构的核心区别在于：**哪些位置之间可以互相 Attend？**

---

## 二、三大架构深度对比

### 2.1 Encoder-only（BERT 家族）

**注意力模式**：双向（Bidirectional）— 每个 token 可以看到序列中所有其他 token。

```
Attention Mask（全1，无遮挡）:

      The  cat  sat  on  mat
The  [ 1    1    1   1    1 ]
cat  [ 1    1    1   1    1 ]
sat  [ 1    1    1   1    1 ]
on   [ 1    1    1   1    1 ]
mat  [ 1    1    1   1    1 ]
```

**预训练目标**：Masked Language Model (MLM)
- 随机遮住 15% 的 token，让模型预测被遮住的 token
- 例：`The [MASK] sat on the mat` → 预测 `[MASK]` = `cat`

**数学形式**：
$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid x_{\setminus \mathcal{M}})
$$

其中 $\mathcal{M}$ 是被遮住的 token 集合，$x_{\setminus \mathcal{M}}$ 是未被遮住的上下文。

**优势**：
- 双向上下文 → 对语义理解任务（分类、NER、语义相似度）效果极好
- 输出的每个 token 表示都融合了完整上下文

**劣势**：
- **不能自回归生成**：训练时用 mask，推理时无法逐 token 生成
- 被 Decoder-only 的 zero-shot/few-shot 能力取代了大部分场景

**代表模型**：BERT, RoBERTa, ALBERT, DeBERTa

**当前角色**：文本嵌入模型（用于检索、向量数据库）、轻量分类器

---

### 2.2 Decoder-only（GPT 家族）

**注意力模式**：因果（Causal）— 每个 token 只能看到自己和左边的 token。

```
Causal Mask（下三角矩阵）:

      The  cat  sat  on  mat
The  [ 1    0    0   0    0 ]
cat  [ 1    1    0   0    0 ]
sat  [ 1    1    1   0    0 ]
on   [ 1    1    1   1    0 ]
mat  [ 1    1    1   1    1 ]
```

**预训练目标**：Causal Language Modeling (CLM) / Next Token Prediction
$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

每个位置预测下一个 token，使用因果掩码确保不会"偷看"未来。

**为什么这个目标足够强大？**

1. **信息论角度**：预测下一个 token 等价于建模文本的联合概率分布
   $$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

2. **任务统一**：任何 NLP 任务都可以表述为条件生成
   - 分类：`"This movie is great. Sentiment: " → "positive"`
   - 翻译：`"Translate to French: Hello → "Bonjour"`
   - 推理：`"Q: What is 2+3? A: " → "5"`

**优势**：
- 架构极简，只有 Decoder block 堆叠
- 天然支持自回归生成
- Scaling 友好 → 涌现能力（ICL, CoT）
- 训练目标简单，数据利用率高（每个 token 都是训练样本）

**劣势**：
- 单向注意力在纯理解任务上不如双向（但规模够大后差距可忽略）

**代表模型**：GPT 系列, LLaMA, Mistral, Qwen, DeepSeek

---

### 2.3 Encoder-Decoder（T5 家族）

**注意力模式**：三种 Attention 并存
1. **Encoder Self-Attention**：双向
2. **Decoder Self-Attention**：因果（下三角）
3. **Cross-Attention**：Decoder attend to Encoder 输出

```
Encoder                     Decoder
[双向Self-Attn]             [因果Self-Attn]
       ↓                         ↓
  Encoder输出  ──────→  [Cross-Attention]
                              ↓
                         [FFN + 输出]
```

**预训练目标**（T5 的 Span Corruption）：
- 随机遮住连续 span → 用特殊 token 替换 → 让 Decoder 生成被遮住的内容
- `"The <X> on the mat"` → `"<X> cat sat <Y>"`

**优势**：
- Encoder 提供双向理解，Decoder 负责生成 → 理论上兼具两者优势
- 适合 Seq2Seq 任务（翻译、摘要）

**劣势**：
- 架构复杂，参数分布在 Encoder 和 Decoder → Encoder 参数在纯生成时利用率低
- Scaling 不如 Decoder-only 干净

**代表模型**：T5, BART, Flan-T5, UL2

---

### 2.4 总结对比表

| 维度 | Encoder-only | Decoder-only | Encoder-Decoder |
|------|:----------:|:----------:|:----------:|
| 注意力 | 双向 | 因果（单向） | Enc 双向 + Dec 因果 + Cross |
| 预训练目标 | MLM | CLM (Next Token) | Span Corruption / Prefix LM |
| 生成能力 | ❌ 不支持 | ✅ 强 | ✅ 强 |
| 理解能力 | ✅ 强 | ⚡ 够大后也强 | ✅ 强 |
| Scaling 性价比 | 中 | **最高** | 中 |
| 训练效率 | 只用 15% mask 位置算 loss | **每个 token 都算 loss** | 介于两者之间 |
| 当前主流 | 嵌入模型 | **LLM 主流** | 逐渐被取代 |

### 2.5 为什么 Decoder-only 赢了？（深度分析）

这是面试高频问题，Day1 简要提过，这里做数学层面的补充：

**1. 训练效率优势**

- Encoder-only (MLM): 一个序列只有 ~15% 的 token 贡献 loss
- Decoder-only (CLM): **每个 token 都贡献 loss** → 相同数据量下看到的"训练信号"多 ~6 倍

**2. 参数利用率**

- Encoder-Decoder: 假设总参数 N，大约 N/2 在 Encoder，N/2 在 Decoder。做生成时 Encoder 的 N/2 只用一次
- Decoder-only: 全部 N 参数都参与每一步生成

**3. In-Context Learning 的自然支持**

- Decoder-only 的自回归特性天然支持 `prompt + 示例 → 生成` 范式
- 这使得 few-shot learning 变得极其简洁

**4. 工程简洁性**

- 只有一种 block 反复堆叠 → 代码简洁，分布式切分简单
- Megatron-LM 的张量并行/流水线并行对 Decoder-only 更容易实现

---

## 三、Scaling Law — 深入理解

### 3.1 Kaplan Scaling Law (OpenAI, 2020)

**论文**：*Scaling Laws for Neural Language Models*

核心发现——模型的测试 loss 遵循幂律(power law)关系：

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076
$$
$$
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095
$$
$$
L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050
$$

其中 $N$ = 参数量，$D$ = 数据量(tokens)，$C$ = 计算量(FLOPs)。

**关键推论**：
- Loss 随三个因素的增长**平滑**下降（没有突变点）
- 在固定计算预算 $C$ 下，应大幅增加模型参数 $N$，数据 $D$ 只需适量
- 架构细节（宽度 vs 深度、头数等）影响很小

### 3.2 Chinchilla Scaling Law (DeepMind, 2022)

**论文**：*Training Compute-Optimal Large Language Models*

Chinchilla 修正了 Kaplan 的结论：

$$
\text{Optimal: } N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}
$$

即**参数量和数据量应等比例增长**，最优比例约为：

$$
D_{\text{opt}} \approx 20 \times N
$$

| 模型参数 | 最优训练 Token 数 | 
|---------|----------------|
| 1B | ~20B tokens |
| 7B | ~140B tokens |
| 70B | ~1.4T tokens |

**验证**：Chinchilla (70B 参数，1.4T tokens) 性能超过了 Gopher (280B 参数，300B tokens)，证明**数据量不足是更大的瓶颈**。

### 3.3 后 Chinchilla 时代的实践

LLaMA 的训练策略直接受 Chinchilla 启发：

| 模型 | 参数量 | 训练 Tokens | N:D 比 |
|------|--------|------------|--------|
| LLaMA-1 7B | 7B | 1T | 1:143 |
| LLaMA-1 65B | 65B | 1.4T | 1:22 |
| LLaMA-2 70B | 70B | 2T | 1:29 |
| LLaMA-3 8B | 8B | 15T | 1:1875 (!) |

注意 LLaMA-3 的数据量远超 Chinchilla 最优比例 —— 这反映了一个新趋势：**推理最优(inference-optimal)**。如果模型要被大量用户频繁调用，多花训练计算换取更小更强的模型是值得的。

### 3.4 涌现能力（Emergent Abilities）

Scaling Law 描述的是 loss 的平滑下降，但下游任务能力的涌现是**阶跃式**的：

```
模型规模 →     小       中       大       超大
             ──────────────────────────────
算术推理      ❌       ❌       ❌       ✅ 突然出现！
ICL Few-shot  ❌       ❌       ✅       ✅
CoT 推理      ❌       ❌       ❌       ✅
翻译          ⚠️       ✅       ✅       ✅
```

**争议**：Wei et al. (2022) 提出的涌现能力概念存在争议 —— Schaeffer et al. (2023) 指出涌现可能是评估指标的非线性造成的假象（使用连续指标则能力平滑增长）。但无论如何，**足够大的模型确实展现出小模型不具备的能力**。

---

## 四、Compute 计算估算（实用技能）

面试和研究中常需要估算训练/推理成本：

### 4.1 前向传播 FLOPs

对于一个 Transformer 模型（近似计算）：

$$
\text{FLOPs}_{\text{forward}} \approx 2 \times N \times T
$$

其中 $N$ = 模型参数量，$T$ = 序列中 token 数。乘以 2 是因为矩阵乘法中每个参数参与一次乘法和一次加法。

### 4.2 训练 FLOPs

训练时反向传播约为前向的 2 倍（计算梯度 + 更新）：

$$
\text{FLOPs}_{\text{train}} \approx 6 \times N \times D
$$

其中 $D$ = 训练总 token 数。

**例**：训练 LLaMA-2 70B on 2T tokens
$$
C \approx 6 \times 70 \times 10^9 \times 2 \times 10^{12} = 8.4 \times 10^{23} \text{ FLOPs}
$$

使用 A100 (312 TFLOPS FP16，利用率 ~40%)：
$$
\text{时间} \approx \frac{8.4 \times 10^{23}}{312 \times 10^{12} \times 0.4 \times 2000 \text{ GPUs}} \approx 33 \text{ 天}
$$

---

## 五、自检题

1. **画出 Causal Mask 的矩阵形式**，并解释它如何保证 token $t$ 只能看到 $x_1, \ldots, x_t$。
2. **为什么 Decoder-only 的训练信号密度高于 MLM？** 定量分析。
3. **一个 7B 参数模型，按 Chinchilla 最优应该训练多少 tokens？** LLaMA-3 为什么选择远超这个数？
4. **估算训练一个 13B 模型在 1T tokens 上的 FLOPs。** 需要多少 A100-天？

---

## 六、产出要求

- [ ] 撰写 2 页笔记：三大架构对比（含 Attention Mask 图示）+ Scaling Law 公式与直觉
- [ ] 能口述回答上面 4 道自检题
