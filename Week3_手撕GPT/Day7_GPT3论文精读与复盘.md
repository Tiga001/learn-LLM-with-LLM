# Day 7：GPT-3 论文精读 + 第三周复盘

> **目标**：精读 GPT-3 论文 *Language Models are Few-Shot Learners*，深入理解 In-Context Learning 的核心机制；回顾本周 Day 1 ~ Day 6 全部内容，串联知识链路，为第 4 周手撕 LLaMA 打下基础。

---

## Part 1：论文精读 — Language Models are Few-Shot Learners

**论文信息**：Brown et al., NeurIPS 2020, OpenAI

**论文地址**：[https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

**页数**：75 页（含附录），正文约 35 页

### 精读指南

这篇论文是 LLM 时代最具里程碑意义的工作之一。建议按以下节奏精读：

```
第一遍（30 min）：Abstract + Introduction + Section 5（Limitations）+ Conclusion
  → 核心贡献：大规模语言模型可以 few-shot 完成任务，无需梯度更新

第二遍（90 min）：Section 2（Approach）+ Section 3（Results），重点看 Figure 和 Table
  → 理解模型规模、训练细节、各任务的 scaling 曲线

第三遍（60 min）：Section 4（Measuring Bias）+ Section 6（Broader Impacts）+ 附录
  → 理解局限性、社会影响，以及各任务的详细评估设置
```

---

### 1. 论文要解决什么问题？

GPT-3 发表于 2020 年，当时 NLP 领域的主流范式是「预训练 + 微调」：

```
主流范式（2018-2020）:
  大规模预训练 → 收集任务标注数据 → 在预训练模型上微调 → 部署

每个新任务都需要：
  1. 标注数据（昂贵、耗时）
  2. 微调过程（调超参数、防过拟合）
  3. 任务专用模型（一个模型只做一件事）
```

GPT-3 论文的核心问题是：**能否不做微调，仅通过 prompt 中的几个示例，就让模型完成各种任务？**

---

### 2. 核心贡献

#### 2.1 In-Context Learning (ICL)

论文定义了三种使用模式，从无示例到多示例：

| 模式 | Prompt 格式 | 特点 |
|------|-----------|------|
| **Zero-shot** | 任务描述 + 输入 | 无需示例，依赖预训练知识 |
| **One-shot** | 任务描述 + 1 个示例 + 输入 | 最小化示例需求 |
| **Few-shot** | 任务描述 + K 个示例 + 输入 | 充分利用上下文窗口 |

```
Few-shot 示例（翻译任务）:

  "Translate English to French:
   sea otter => loutre de mer
   peppermint => menthe poivrée
   plush girafe => girafe en peluche
   cheese =>"

  模型输出: "fromage"
```

**关键**：ICL 不更新任何参数。模型在前向传播中"理解"了任务并给出答案。这是一种根本不同于微调的范式。

#### 2.2 In-Context Learning 不是传统的学习

ICL 与微调有本质区别：

| 维度 | 微调（Fine-tuning） | ICL（In-Context Learning） |
|------|-------------------|---------------------------|
| 参数更新 | 有，需要反向传播 | **无**，只有前向传播 |
| 需要训练数据 | 通常数百到数千条 | 0 ~ 几十个示例 |
| 任务切换 | 需要重新微调 | 换个 prompt 就行 |
| 灾难性遗忘 | 可能丢失通用能力 | 不影响模型参数 |
| 性能上限 | 通常更高 | 受限于上下文长度 |

---

### 3. 模型规模与训练

#### 3.1 模型家族

GPT-3 论文训练了 **8 个不同规模的模型**，这对理解 scaling 至关重要：

| 模型 | 参数量 | 层数 | $d_{\text{model}}$ | 头数 | $d_{\text{head}}$ | 批大小（tokens） |
|------|--------|------|---------------------|------|-----------|----------------|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 0.5M |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 0.5M |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 1M |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M |
| GPT-3 13B | 13B | 40 | 5140 | 40 | 128 | 2M |
| **GPT-3 175B** | **175B** | **96** | **12288** | **96** | **128** | **3.2M** |

**关键设计决策**：
- 所有模型的 $d_{\text{head}} = 64$ 或 $128$，与 GPT-2 一致
- $d_{ff} = 4 \times d_{\text{model}}$ 的惯例被保留
- 上下文窗口统一为 2048 tokens
- 批大小随模型增大而增大——大模型的梯度更稳定，可以用更大的批次

#### 3.2 训练数据

| 数据集 | 原始大小 | Token 数 | 采样权重 | 训练 Epoch |
|--------|---------|---------|---------|-----------|
| Common Crawl（过滤后） | 45TB → 570GB | 410B | 60% | 0.44 |
| WebText2 | 40GB | 19B | 22% | 2.9 |
| Books1 | — | 12B | 8% | 1.9 |
| Books2 | — | 55B | 8% | 0.43 |
| Wikipedia | — | 3B | 3% | 3.4 |

**论文的数据处理要点**：

1. **Common Crawl 过滤**：用一个二分类器（训练在高质量参考语料上）来判断每个文档的质量，模糊去重（fuzzy dedup），去除 benchmark 数据的污染
2. **采样权重 ≠ 数据占比**：高质量数据集（WebText2、Books）被过采样，低质量的 Common Crawl 被欠采样。这意味着 Wikipedia 被训练了 3.4 个 epoch，而 Common Crawl 只被训练了 0.44 个 epoch
3. **总训练 token 数**：约 300B tokens（虽然总数据约 500B，但按采样权重计算实际见到约 300B）

#### 3.3 训练配置

| 配置 | 175B 模型 |
|------|----------|
| 优化器 | Adam ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$) |
| 学习率 | $6 \times 10^{-5}$, cosine decay |
| Warmup | 375M tokens |
| 权重衰减 | 0.1 |
| 梯度裁剪 | 1.0 |
| 批大小 | 逐步增大 (32K → 3.2M tokens) |
| 序列长度 | 2048 |
| 训练硬件 | V100 集群 (推测 ~1000 GPU) |
| 训练时间 | ~34 天（推测） |
| 计算量 | ~$3.14 \times 10^{23}$ FLOPs |

**批大小渐增策略**（Section 2.2）是一个重要的工程技巧：

```
训练初期：小批量 → 梯度噪声大 → 有正则化效果，帮助跳出尖锐极小值
训练后期：大批量 → 梯度估计更准确 → 稳定收敛到平坦极小值
```

---

### 4. 核心实验结果精读

#### 4.1 Scaling 曲线 — 论文的灵魂

论文 Figure 1.2 和 Figure 1.3 展示了所有 8 个模型规模在各种任务上的 few-shot 性能：

```
关键发现：

1. 模型越大，few-shot 能力越强（近似对数线性）
2. 从 13B 到 175B，许多任务出现「跃迁」而非渐进提升
3. few-shot > one-shot > zero-shot，差距随模型增大而增大

具象例子（2 位数加法）:
  125M 模型:  few-shot 准确率 ~5%
  1.3B 模型:  few-shot 准确率 ~10%
  13B  模型:  few-shot 准确率 ~30%
  175B 模型:  few-shot 准确率 ~100%   ← 几乎完美！

这就是后来被称为「涌现能力 (Emergent Abilities)」的现象。
```

#### 4.2 各领域的关键结果

**语言建模（Section 3.1）**

GPT-3 175B 在 Penn Treebank 上达到了 **20.50 PPL**（zero-shot），超越当时所有微调模型。在 LAMBADA 数据集（需要长距离上下文理解）上，few-shot 准确率达到 86.4%。

**翻译（Section 3.3）**

| 方向 | GPT-3 Few-shot | 有监督 SOTA |
|------|---------------|------------|
| Fr→En | 32.6 BLEU | 36.0 |
| De→En | 29.7 BLEU | 40.2 |
| En→Fr | 25.2 BLEU | 45.6 |
| En→De | 24.3 BLEU | 41.2 |

**关键洞察**：GPT-3 在翻译到英文（X→En）时表现远好于翻译出英文（En→X）。原因：训练数据以英文为主，模型的英文生成能力更强。

**问答（Section 3.4）**

在 TriviaQA 上，GPT-3 175B 的 few-shot 准确率（71.2%）超过了微调的 T5-11B，且无需访问任何外部知识库。这说明 GPT-3 将大量事实知识存储在了模型参数中。

**算术推理（Section 3.9）**

| 任务 | GPT-3 175B Few-shot |
|------|-------------------|
| 2 位数加法 | **100%** |
| 3 位数加法 | **80.2%** |
| 4-5 位数加法 | **25-9%** |
| 2 位数减法 | **98.9%** |
| 2 位数乘法 | **29.2%** |

规模越大，算术能力越强，但超过一定复杂度后急剧下降。这揭示了 LLM 的一个根本局限：**它们做的是模式匹配，不是真正的数学推理**。

#### 4.3 ICL 的失败案例

论文诚实地展示了 GPT-3 的弱点：

| 任务类型 | 表现 | 分析 |
|---------|------|------|
| **自然语言推理（NLI）** | 显著弱于微调模型 | 蕴涵/矛盾判断需要精确推理 |
| **阅读理解（部分）** | 中等 | 需要精确的信息抽取 |
| **对比型选择题** | 弱 | 两个选项仅有微小差异时判断力不足 |
| **常识物理推理** | 弱 | 需要隐含的世界知识 |

论文的 Section 5（Limitations）值得精读：

> "GPT-3 有时会生成在句子层面看起来合理但段落层面语义不连贯的长文本。"

> "GPT-3 的双向性不足——它只能从左到右生成，不能像 BERT 那样同时看到上下文。"

> "GPT-3 的预训练目标（预测下一个 token）对每个 token 一视同仁，但直觉上更重要的 token 应该得到更多关注。"

---

### 5. In-Context Learning 的深入分析

ICL 是 GPT-3 最重要的贡献，但它的机制至今仍有争议。

#### 5.1 ICL 究竟是什么？

```
微调视角:
  数据 → 梯度下降 → 参数更新 → 模型学会了任务
  ↑ 这是传统的「学习」

ICL 视角:
  (示例 + 输入) → 一次前向传播 → 输出
  ↑ 没有任何参数变化，但模型「学会了」任务

问题：ICL 是真的在「学习」吗？还是只是在做模式匹配？
```

#### 5.2 四种主要假说

**假说 1：隐式贝叶斯推理**（Xie et al., 2022）

$$P(y | x_{\text{test}}, \text{examples}) = \sum_{\text{task}} P(y | x_{\text{test}}, \text{task}) \cdot P(\text{task} | \text{examples})$$

模型在预训练中学到了所有「任务」的先验分布。prompt 中的示例帮助模型推断出当前是什么「任务」（$P(\text{task} | \text{examples})$），然后按照该任务的条件分布生成。

**假说 2：隐式梯度下降**（Dai et al., 2023; von Oswald et al., 2023）

线性 Attention 层的前向传播在数学上等价于在示例上做了一步梯度下降：

$$\text{Attention}(Q, K, V) \approx W_0 + \eta \sum_i v_i k_i^T$$

其中 $\sum_i v_i k_i^T$ 的形式类似于梯度更新。Transformer 的多层结构相当于多步梯度更新。

**假说 3：任务识别**（Min et al., 2022）

实验发现：即使将 few-shot 示例中的标签随机替换（打乱 x-y 对应关系），GPT-3 的表现下降但仍远好于随机。这说明 ICL 主要依赖示例的**输入分布**和**标签空间**，而非具体的输入-标签映射。

```
正常 few-shot:
  "positive: I love this movie"
  "negative: This movie is terrible"
  → 模型学到了情感分类

随机标签:
  "negative: I love this movie"    ← 标签是错的
  "positive: This movie is terrible" ← 标签是错的
  → 模型仍然能做情感分类！（准确率下降但远高于随机）

这说明：模型不是在「学」示例，而是在「识别」任务
```

**假说 4：Induction Head**（Olsson et al., 2022）

Transformer 中存在特定的注意力头（Induction Head），它们可以做「复制 + 偏移」的模式匹配：

```
如果 Attention Head 看到:
  "A B ... A"
它会预测下一个 token 是 B（复制 A 之后出现过的 token）

在 ICL 中:
  "X₁ → Y₁, X₂ → Y₂, X₃ →"
Induction Head 识别模式 "X → Y"，在看到 X₃ 后预测对应的 Y₃
```

#### 5.3 ICL 的实践启示

| 启示 | 说明 |
|------|------|
| 示例的格式比内容更重要 | 保持格式一致比挑选"最好的"示例更重要 |
| 示例数量有收益递减 | 通常 4-8 个示例就够，更多未必更好 |
| 示例顺序有影响 | 最后一个示例的影响最大（recency bias） |
| 提示模板很重要 | 同一任务，不同 prompt 模板的性能差异可达 20%+ |

---

### 6. 论文的数据污染分析（Section 4）

GPT-3 论文中有一个容易被忽略但非常重要的章节：**数据污染**。

由于 GPT-3 的训练数据来自互联网，很多 benchmark 的测试集可能出现在训练数据中。论文做了详细的污染分析：

```
方法：检测训练数据与测试集的 13-gram 重叠
结果：
  - 大部分 benchmark 有一定程度的污染
  - 但移除污染数据后，性能变化通常 < 1-3%
  - 少数任务（如 PIQA）污染严重，结果不可靠
```

**这提醒我们**：在评估大模型时，数据污染是一个必须考虑的问题。这也是后来社区发展 LiveBench、GPQA 等"防污染"评测的原因。

---

### 7. GPT-3 的社会影响分析（Section 6）

论文用了相当篇幅讨论社会影响，这在 2020 年的 AI 论文中是少见的：

| 风险 | 论文的分析 |
|------|---------|
| **虚假信息** | GPT-3 生成的新闻文章，人类评估者仅能以 52% 的准确率区分真假（近乎随机猜测） |
| **偏见** | 在性别、种族、宗教方面存在系统性偏见。如 "He was very" 后更可能接正面词，而 "She was very" 后更可能接外貌描述 |
| **能源消耗** | 175B 模型训练的碳排放约 500 吨 CO₂ |

这些讨论直接催生了后来的对齐研究（InstructGPT）和安全研究方向。

---

### 8. 论文的历史地位

| 维度 | GPT-3 的贡献 |
|------|-------------|
| **范式变革** | 从「每个任务训一个模型」到「一个模型做所有任务」 |
| **In-Context Learning** | 首次大规模验证，成为 LLM 的标志性能力 |
| **Scaling Law 的下游验证** | 证明规模增大不仅降低 perplexity，还能涌现新能力 |
| **Prompt Engineering** | 开启了 prompt 设计作为一种"编程"方式的时代 |
| **对齐问题的暴露** | 展示了大模型的能力和风险并存，催生了 InstructGPT/RLHF |

---

### 9. GPT-3 论文的自检题

1. **GPT-3 的三种使用模式是什么？** 它们分别在 prompt 中包含什么？
2. **ICL 与微调的核心区别是什么？** 为什么 ICL 不需要梯度更新就能完成任务？
3. **GPT-3 的训练数据中，为什么高质量数据集（WebText2）的采样权重远高于其数据量占比？**
4. **GPT-3 翻译到英文（X→En）比翻译出英文（En→X）好得多，为什么？**
5. **为什么将 few-shot 示例的标签随机打乱后，模型表现仍然远好于随机？** 这对 ICL 的机制理解意味着什么？
6. **GPT-3 在算术任务上的表现呈现什么模式？** 这揭示了 LLM 的什么局限？
7. **什么是数据污染？为什么评估 GPT-3 时必须考虑这个问题？**

---

## Part 2：第三周知识串联与复盘

### 全周知识链路

```
Day 1: GPT 系列论文精读
  GPT-1 (预训练+微调) → GPT-2 (zero-shot, Pre-Norm) → GPT-3 (few-shot, ICL)
  → InstructGPT (RLHF) → GPT-4 (多模态, MoE)
       │
       │ "GPT 的核心架构是什么？与 Transformer 有何不同？"
       ▼
Day 2: GPT 模型架构详解
  Decoder-only = Token Emb + Pos Emb → N × (Pre-LN + CausalMHA + Pre-LN + FFN) → Final LN → LM Head
  关键决策: Pre-Norm / GELU / 可学习位置编码 / 权重共享
       │
       │ "能把这些组件全部写成代码吗？"
       ▼
Day 3: 手写 GPT 模型 ★★★★★ 本周最核心
  CausalSelfAttention → MLP → GPTBlock → GPT
  验证: 加载 HuggingFace GPT-2 权重，与官方输出一致
       │
       │ "模型有了，如何控制它的生成方式？"
       ▼
Day 4: 采样策略
  Greedy → Temperature (控制锐度) → Top-K (固定候选数) → Top-P (自适应候选集)
  组合使用: Temperature → Top-K → Top-P → Softmax → 采样
       │
       │ "标准 Attention 的 O(T²) 内存瓶颈怎么解决？"
       ▼
Day 5: FlashAttention 入门
  GPU SRAM vs HBM → IO 瓶颈分析 → Tiling + Online Softmax → O(T²d/M) IO 复杂度
       │
       │ "能实际跑一个预训练吗？"
       ▼
Day 6: GPT 预训练实践
  TinyShakespeare → Tokenize → DataLoader → AdamW + Cosine LR → 训练 → 生成
  观察: 随机 → 英文单词 → 语法正确 → 莎士比亚风格
       │
       │ "GPT-3 的核心创新（ICL）到底是什么？"
       ▼
Day 7: GPT-3 论文精读 + 全周复盘 ← 你在这里
  ICL 机制 → Scaling 曲线 → 局限性 → 为 W4 LLaMA 做准备
```

---

### 核心概念关系图

```
                    ┌──────────────────────────┐
                    │      Causal LM 训练       │
                    │ L = -Σ log P(xₜ | x<t)   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
              ┌─────┴──────┐          ┌────────┴────────┐
              │   训练时     │          │    推理时        │
              │  Teacher     │          │  自回归生成      │
              │  Forcing     │          │  + 采样策略      │
              │  (Day 2,6)   │          │  (Day 4)        │
              └─────────────┘          └─────────────────┘

     ┌──────────────────────────────────────────────┐
     │                GPT 模型 (Day 2, 3)             │
     │                                                │
     │  Token Emb + Pos Emb                           │
     │      ↓                                         │
     │  ┌─────────────────────────────────────┐      │
     │  │  GPT Block × N                       │      │
     │  │  Pre-LN → CausalMHA → Residual      │      │
     │  │  Pre-LN → FFN (GELU) → Residual     │      │
     │  └─────────────────────────────────────┘      │
     │      ↓                                         │
     │  Final LN → LM Head (权重共享)                 │
     └──────────────────────────────────────────────┘
                           │
              ┌────────────┼─────────────┐
              │            │             │
              ▼            ▼             ▼
         标准 Attn    FlashAttention   GPT-3 的
         O(T²) 内存   O(T) 内存       In-Context
         (Day 2,3)    (Day 5)         Learning
                                       (Day 1,7)
```

---

### 全周自检清单

#### 理论层 — GPT 论文与发展

- [ ] 列出 GPT-1/2/3/InstructGPT/GPT-4 各代的**一句话核心创新**
- [ ] 解释 Pre-Norm vs Post-Norm 的数学公式与 Pre-Norm 的优势
- [ ] 解释 In-Context Learning 的工作机制（至少说出 2 种假说）
- [ ] 解释 GPT-3 的数据混合策略（采样权重 vs 数据量）
- [ ] 区分 Zero-shot / One-shot / Few-shot 三种模式

#### 理论层 — 架构与数学

- [ ] 写出 Causal Self-Attention 的完整 5 步公式
- [ ] 写出 CLM 目标函数 $\mathcal{L} = -\frac{1}{T}\sum_t \log P(x_t | x_{<t})$
- [ ] 计算 GPT-2 Small (d=768, N=12) 的参数量（精确到各组件）
- [ ] 解释 $d_{ff} = 4d$ 的设计惯例和各层参数量占比
- [ ] 解释权重共享（Weight Tying）的数学含义

#### 理论层 — 采样与效率

- [ ] 写出 Temperature Sampling 的公式 $P(x_i) = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$
- [ ] 解释 Top-P 相比 Top-K 的自适应优势
- [ ] 解释 FlashAttention 的 3 个核心思想（IO 感知 / Tiling / Online Softmax）
- [ ] 写出 Online Softmax 的递推公式
- [ ] 对比标准 Attention 与 FlashAttention 的 IO / 内存 / FLOPs 复杂度

#### 代码层 — 核心手写能力

- [ ] **闭卷手写 `CausalSelfAttention`**（合并投影 → 分头 → QK^T/√dk → 因果掩码 → softmax → V加权 → 合并 → 输出投影）
- [ ] **闭卷手写 `GPTBlock`**（Pre-LN + Attention + Residual + Pre-LN + FFN + Residual）
- [ ] **闭卷手写完整 `GPT` 模型**（Embedding + N×Block + Final LN + LM Head + 权重共享 + 初始化）
- [ ] **手写 Temperature / Top-K / Top-P 采样函数**
- [ ] 手写 CLM 训练循环（DataLoader → forward → loss → backward → optimizer → lr schedule）
- [ ] 手写贪心解码和组合策略生成函数

#### 工程层 — 训练技巧

- [ ] 解释 AdamW 的权重衰减分组策略（哪些参数做衰减，哪些不做）
- [ ] 解释 Cosine LR Schedule + Linear Warmup 的设计动机
- [ ] 解释梯度裁剪（Gradient Clipping）的作用
- [ ] 解释训练初始 Loss ≈ ln(vocab_size) 的原因
- [ ] 区分训练时（Teacher Forcing + 并行）和推理时（自回归 + 串行）的差异

---

### 重要公式速查卡

| 公式 | 来源 |
|------|------|
| $h_0 = E_{\text{tok}}[x] + E_{\text{pos}}[0:T]$ | Embedding (Day 2) |
| $Q = hW_Q,\ K = hW_K,\ V = hW_V$ | Attention 投影 (Day 2, 3) |
| $\text{Att} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Mask}\right) V$ | Causal Attention (Day 2, 3) |
| $h' = h + \text{MHA}(\text{LN}(h))$ | Pre-Norm 残差 (Day 2, 3) |
| $\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$ | FFN (Day 2, 3) |
| $\text{logits} = \text{LN}(h_N) \cdot E_{\text{tok}}^T$ | 权重共享 LM Head (Day 2, 3) |
| $\mathcal{L} = -\frac{1}{T}\sum_t \log P(x_t \mid x_{<t})$ | CLM Loss (Day 2, 6) |
| $P(x_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$ | Temperature (Day 4) |
| $d_j = d_{j-1} \cdot e^{m_{j-1} - m_j} + e^{x_j - m_j}$ | Online Softmax (Day 5) |
| $\text{Params}_{\text{block}} \approx 12d^2$ | 参数量估算 (Day 2) |

---

### 常见疑惑解答

**Q1：GPT-3 的 ICL 能力是怎么来的？预训练目标明明只是「预测下一个 token」。**

预训练数据中天然包含了大量「示例 → 答案」的模式。比如网页中的 Q&A、教程中的示例、文档中的格式模板。模型在预训练中隐式地学到了：「当上文给出几个同格式的示例时，接下来应该按同样的格式给出答案。」

ICL 不是一种全新的能力，而是预训练中"任务模式识别"能力的自然延伸。

**Q2：为什么 Day 3 手写的 GPT 可以直接加载 HuggingFace 的权重？**

因为架构完全一致。HuggingFace 的 GPT-2 实现和我们的手写版在数学上做的事情完全一样——区别仅在于命名习惯和是否用 `Conv1D`（HuggingFace 历史原因用 Conv1D 替代 Linear，需要转置权重）。这验证了我们的实现正确性。

**Q3：FlashAttention 不减少 FLOPs，为什么能加速？**

因为标准 Attention 是 IO-bound（内存带宽瓶颈），不是 compute-bound（计算瓶颈）。GPU 的算力增长远超内存带宽增长，所以减少内存访问（IO）比减少计算量更能提升实际速度。FlashAttention 通过 tiling 将 $T \times T$ 中间矩阵保持在 SRAM 中，避免了反复读写 HBM。

**Q4：Temperature、Top-K、Top-P 三个参数该怎么组合？**

实践中的经验法则：
- 先选 Temperature 控制整体多样性（0.7 ~ 1.0 是安全区间）
- 再加 Top-P（0.9 ~ 0.95）自适应截断长尾
- Top-K 作为保底（50 ~ 100），防止极端情况
- 不要同时大幅调整多个参数

**Q5：为什么我们的预训练只用了 ~10M 参数，真实 GPT-3 用了 175B？**

这是 Day 6 的教学设计。10M 参数的"Baby GPT"足以在 TinyShakespeare 上学到英语模式，但不可能涌现出 ICL 等高级能力——这些需要百亿参数级别的模型和 TB 级别的数据。教学实验的目标是理解训练流程，而非复现完整能力。

---

### 本周与课程整体的连接

| 本周学到的 | 第 4 周将如何演进（手撕 LLaMA） |
|----------|---------------------------|
| 可学习位置编码 | → **RoPE** (Rotary Position Embedding)，支持外推 |
| LayerNorm + Pre-Norm | → **RMSNorm**，更简单、更高效 |
| GELU 激活函数 | → **SwiGLU**，FFN 内部带门控机制 |
| Multi-Head Attention | → **GQA** (Grouped Query Attention)，减少 KV Cache |
| Weight Tying (Emb = LM Head) | → LLaMA 不使用权重共享 |
| 带 Bias 的 Linear | → LLaMA 去掉所有 bias |
| GPT-2 BPE Tokenizer | → SentencePiece / LLaMA Tokenizer |
| FlashAttention 入门 | → 第 14 周 FlashAttention 1/2 深化 |
| RLHF 概念（InstructGPT） | → 第 9~13 周 RL 与对齐技术深化 |

---

### 下周预告：第 4 周 · 手撕 LLaMA

LLaMA 是当前开源 LLM 的基石架构。第 4 周你将理解 GPT → LLaMA 的所有架构改进，并从零手写 LLaMA。

| 组件 | 从 GPT 到 LLaMA 的改进 |
|------|---------------------|
| **RMSNorm** | 去掉均值中心化，只保留缩放 → 更快 |
| **RoPE** | 旋转位置编码，支持长度外推 → Day 2, 3 |
| **SwiGLU** | 带门控的 FFN → Day 3 |
| **GQA** | 分组查询注意力，减少 KV Cache → Day 3 |
| **KV Cache** | 推理加速的核心技巧 → Day 4 |
| **LLaMA 预训练** | 更大规模的预训练实践 → Day 6 |

**准备工作**：确保你能闭卷写出完整的 GPT 模型（从 Embedding 到 LM Head）。这是第 4 周的起点——你将在此基础上逐模块替换为 LLaMA 的改进版本。
