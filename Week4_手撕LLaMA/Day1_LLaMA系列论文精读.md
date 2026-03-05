# Day 1：LLaMA 系列论文精读 — 开源 LLM 的基石

> **目标**：系统梳理 LLaMA 系列（LLaMA-1 / LLaMA-2 / LLaMA-3）的技术演进，理解其架构改进的设计动机和训练数据策略，对比 GPT 系列的技术路线差异，建立"从 GPT 到 LLaMA"的完整认知。

---

## 一、LLaMA 系列演进总览

```
LLaMA-1 (2023.02)           LLaMA-2 (2023.07)           LLaMA-3 (2024.04)
  7B/13B/33B/65B              7B/13B/70B                  8B/70B/405B
  纯预训练模型                 预训练 + Chat 版本            预训练 + Instruct
  1.0T/1.4T tokens            2.0T tokens                 15T+ tokens
       │                         │                            │
       ▼                         ▼                            ▼
  "开源 + 高效数据            "更大规模训练               "数据和规模的
   胜过闭源大模型"             + RLHF 对齐"               极致工程"
```

**LLaMA 的历史地位**：LLaMA-1 是开源 LLM 生态的真正起点。在此之前，强大的 LLM（GPT-3/3.5/4、PaLM）都是闭源的。LLaMA 证明了：**用高质量公开数据训练的中等规模模型，可以匹敌甚至超越更大的闭源模型。**

---

## 二、LLaMA-1：用更少的参数，做更好的模型

**论文**：*LLaMA: Open and Efficient Foundation Language Models* (Touvron et al., 2023)

**论文地址**：[https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)

### 2.1 核心主张

LLaMA-1 的核心哲学可以用一句话概括：

> **给定固定的推理预算（inference budget），最优的模型不是最大的模型，而是在更多数据上训练更久的较小模型。**

这直接挑战了 GPT-3 时代"模型越大越好"的 Scaling Law 理解。

#### Chinchilla Scaling Law 的启示

2022 年 DeepMind 发表的 Chinchilla 论文提出了关键洞察：

$$C \approx 6ND$$

其中 $C$ 是计算量（FLOPs），$N$ 是模型参数量，$D$ 是训练 token 数。

**Chinchilla 最优**：给定计算预算 $C$，参数量 $N$ 和数据量 $D$ 应该同比例增长，即 $D \approx 20N$。

| 模型 | 参数量 | 训练 Token 数 | $D/N$ 比值 | 是否 Chinchilla 最优 |
|------|--------|-------------|------------|-------------------|
| GPT-3 | 175B | 300B | 1.7 | 远低于最优 |
| Chinchilla | 70B | 1.4T | 20 | 最优 |
| **LLaMA-65B** | **65B** | **1.4T** | **21.5** | **接近最优** |
| **LLaMA-7B** | **7B** | **1.0T** | **143** | **远超最优** |

LLaMA 的策略更激进：**让小模型训练远超 Chinchilla 最优的 token 数**。虽然这在训练时不是计算最优的，但推理时更便宜（小模型推理成本低）。

```
GPT-3 的策略:  175B 参数 × 300B tokens → 训练不充分
Chinchilla:    70B 参数 × 1.4T tokens → 计算最优
LLaMA-7B:     7B 参数 × 1.0T tokens → 训练充分，推理便宜
LLaMA-65B:    65B 参数 × 1.4T tokens → 训练充分 + 强性能
```

### 2.2 训练数据

LLaMA-1 只使用**公开可获取的数据**，这是它与 GPT-3/PaLM 的一个关键区别：

| 数据集 | Token 数 | 占比 | 数据描述 |
|--------|---------|------|---------|
| CommonCrawl | 827B | 67.0% | 网页数据（经过严格过滤） |
| C4 | 175B | 15.0% | Colossal Clean Crawled Corpus |
| GitHub | 100B | 4.5% | 开源代码 |
| Wikipedia | 25B | 4.5% | 20 种语言的维基百科 |
| Books | 25B | 4.5% | Gutenberg + Books3 |
| ArXiv | 33B | 2.5% | 学术论文（去除参考文献） |
| StackExchange | 25B | 2.0% | 高票回答 |
| **总计** | **~1.4T** | **100%** | |

#### 数据处理要点

**CommonCrawl 过滤**（论文 Section 2.1 的关键）：

1. **语言过滤**：用 fastText 分类器识别语言
2. **质量过滤**：训练一个二分类器，正样本为 Wikipedia 文本，负样本为随机网页。用该分类器的得分过滤 CommonCrawl
3. **去重**：CCNet pipeline（n-gram 级别去重，跨文档和句子级别）
4. **数据比例调整**：Wikipedia 和 Books 被上采样约 2 个 epoch，CommonCrawl 只训练 ~0.85 个 epoch

**关键洞察**：数据质量比数据量更重要。LLaMA 的训练数据全部公开可获取，但经过了严格的清洗和过滤。

### 2.3 模型架构

LLaMA-1 基于 Transformer Decoder-only 架构，但相比 GPT 做了四处关键改进：

| 改进 | GPT-2/3 | LLaMA | 来源 |
|------|---------|-------|------|
| **归一化** | LayerNorm (Post-Norm) | **RMSNorm (Pre-Norm)** | GPT-3 用 Pre-LN，LLaMA 进一步简化为 RMSNorm |
| **激活函数** | GELU | **SwiGLU** | Shazeer (2020)，PaLM 也用了 |
| **位置编码** | 可学习绝对位置编码 | **RoPE（旋转位置编码）** | Su et al. (2021) |
| **注意力** | MHA（全部头独立 KV） | **MHA**（LLaMA-1），**GQA**（LLaMA-2） | GQA: Ainslie et al. (2023) |
| **Bias** | 所有线性层有 bias | **去掉所有 bias** | 减少参数，实践中无性能损失 |
| **权重共享** | Embedding = LM Head | **不共享** | LLaMA 选择不共享 |

每个改进的详细数学推导将在 Day 2 展开。

### 2.4 模型规模

| 模型 | 参数量 | 层数 | $d_{\text{model}}$ | 头数 | $d_{\text{head}}$ | $d_{ff}$ | 训练 Token |
|------|--------|------|---------------------|------|-----------|----------|-----------|
| LLaMA-7B | 6.7B | 32 | 4096 | 32 | 128 | 11008 | 1.0T |
| LLaMA-13B | 13.0B | 40 | 5120 | 40 | 128 | 13824 | 1.0T |
| LLaMA-33B | 32.5B | 60 | 6656 | 52 | 128 | 17920 | 1.4T |
| LLaMA-65B | 65.2B | 80 | 8192 | 64 | 128 | 22016 | 1.4T |

**注意**：$d_{ff}$ 不再是简单的 $4d$。SwiGLU 的 FFN 有 3 个权重矩阵（而非 2 个），为保持参数量不变，$d_{ff}$ 被调整为 $\frac{2}{3} \times 4d$，再取最近的 256 的倍数。

$$d_{ff} = \text{round\_to\_multiple}\left(\frac{2}{3} \times 4 \times d_{\text{model}}, 256\right)$$

以 LLaMA-7B 为例：$\frac{2}{3} \times 4 \times 4096 = 10922.67 → \text{round to } 11008$。

### 2.5 训练配置

| 配置 | 值 |
|------|-----|
| 优化器 | AdamW ($\beta_1=0.9, \beta_2=0.95$) |
| 学习率 | cosine decay, 峰值 $3 \times 10^{-4}$ (7B/13B), $1.5 \times 10^{-4}$ (33B/65B) |
| Warmup | 2000 steps |
| 权重衰减 | 0.1 |
| 梯度裁剪 | 1.0 |
| 序列长度 | 2048 |
| 批大小 | 4M tokens |
| Tokenizer | SentencePiece BPE, 词表大小 32K |
| 精度 | 混合精度训练 (BF16) |
| 硬件 | 2048 × A100-80GB (65B 模型) |
| 训练时间 | ~21 天 (65B 在 2048 GPU) |

### 2.6 核心实验结果

LLaMA-1 的结果令人震惊——较小的开源模型超越了更大的闭源模型：

**与 GPT-3 175B 的对比**：

| 基准 | GPT-3 175B | LLaMA-13B | LLaMA-65B |
|------|-----------|-----------|-----------|
| MMLU (5-shot) | 43.9 | 46.9 | **63.4** |
| HellaSwag (0-shot) | 78.9 | 79.2 | **84.2** |
| ARC-e (0-shot) | — | 74.8 | **78.9** |

**关键发现**：
- **LLaMA-13B（13B 参数）在多数基准上超越 GPT-3（175B 参数）**
- LLaMA-65B 与 Chinchilla-70B 和 PaLM-540B 竞争
- 训练 token 数的增加持续带来性能提升（1.0T → 1.4T 仍有明显收益）

### 2.7 Tokenizer

LLaMA-1 使用 **SentencePiece** 实现的 BPE tokenizer：

| 配置 | 值 |
|------|-----|
| 算法 | BPE (Byte Pair Encoding) |
| 实现 | SentencePiece |
| 词表大小 | 32,000 |
| 字节回退 | 将未知字符分解为 UTF-8 字节序列 |
| 数字处理 | 将数字拆分为单个数字 token |

与 GPT-2 的 50,257 词表相比，LLaMA 的 32K 词表更小。这是一个有意的设计——更小的词表意味着更小的 Embedding 层和 LM Head。

---

## 三、LLaMA 的设计哲学

### 3.1 "小模型 + 多数据" vs "大模型 + 少数据"

LLaMA 论文的核心论点可以总结为：

```
传统思路（GPT-3 时代）:
  追求最大的模型 → 175B 参数 → 训练 300B tokens → 每次推理都很贵

LLaMA 的思路:
  给定推理预算 → 选择合适大小的模型 → 训练足够多的 tokens → 推理便宜
```

这对工业界的影响深远：
- 在边缘设备部署：选 7B 而非 175B
- 在服务器部署：选 13B/33B 而非 175B，成本低 10-25×
- 持续预训练的成本更低

### 3.2 开源的意义

LLaMA-1 发布前后的开源 LLM 生态：

```
LLaMA-1 之前（2023.02 前）：
  强力 LLM 全部闭源 → GPT-3/4, PaLM, Claude
  开源选择有限 → OPT-175B（效果一般）, BLOOM（训练数据偏向多语言）

LLaMA-1 之后：
  Alpaca (Stanford) → 指令微调的起点
  Vicuna → 对话微调的起点
  Chinese-LLaMA → 中文扩展的起点
  CodeLlama → 代码生成
  → 整个开源 LLM 生态爆发
```

---

## 四、LLaMA vs GPT：核心技术差异全面对比

| 维度 | GPT-2/3 | LLaMA-1 | 改进动机 |
|------|---------|---------|---------|
| **归一化** | LayerNorm | RMSNorm | 去掉均值中心化，计算更快，效果相当 |
| **Norm 位置** | Pre-Norm (GPT-2+) | Pre-Norm | 相同 |
| **位置编码** | 可学习绝对编码 | RoPE | 支持长度外推，编码相对位置信息 |
| **激活函数** | GELU | SwiGLU | 门控机制 + Swish，实验效果更好 |
| **FFN 结构** | 2 个线性层 | 3 个线性层（门控） | SwiGLU 需要额外一个线性层 |
| **$d_{ff}$** | $4d$ | $\frac{8}{3}d$（取整） | 保持与 2 层 FFN 相同的参数量 |
| **注意力** | MHA | MHA (v1) / GQA (v2) | GQA 减少 KV Cache |
| **Bias** | 有 | 无 | 减少参数，无性能损失 |
| **权重共享** | Emb = LM Head | 不共享 | 大模型中不共享效果更好 |
| **Tokenizer** | GPT-2 BPE (50257) | SentencePiece BPE (32000) | 更小词表 + 字节回退 |
| **上下文长度** | 1024 (GPT-2) / 2048 (GPT-3) | 2048 (v1) / 4096 (v2) | 更长上下文 |
| **训练数据** | 私有数据 | 公开数据 | 可复现 |

### 逐项改进的代码级差异预览

```python
# ========== 归一化 ==========
# GPT: LayerNorm
x = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True) + eps).sqrt()
x = gamma * x + beta  # 有 beta（偏移）

# LLaMA: RMSNorm（更简单）
x = x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
x = gamma * x          # 没有 beta

# ========== FFN ==========
# GPT: standard FFN
h = GELU(x @ W1 + b1) @ W2 + b2

# LLaMA: SwiGLU FFN（3 个权重矩阵，无 bias）
h = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

# ========== 位置编码 ==========
# GPT: 可学习绝对编码（加在 Embedding 上）
h_0 = tok_emb + pos_emb  # pos_emb 是可学习参数

# LLaMA: RoPE（在 Attention 的 Q, K 上做旋转）
q, k = apply_rotary_emb(q, k, freqs_cis)  # 旋转，不加
```

---

## 五、LLaMA-1 论文的关键 Figure 解读

### Figure 3: 训练 Loss 曲线

```
Token 数 →      0         250B       500B       750B      1.0T      1.4T
             ──────────────────────────────────────────────────
  7B          2.2        1.95       1.85       1.80      1.77
  13B         2.0        1.80       1.72       1.68      1.65
  33B         1.9        1.70       1.62       1.58      1.56
  65B         1.8        1.65       1.57       1.53      1.50

关键观察：
1. 更大的模型始终有更低的 loss
2. 到 1.4T tokens 时，loss 仍在下降 → 模型仍然 underfitting → 可以继续训练
3. 7B 模型在 1.0T tokens 处 loss ≈ 1.77 → 这就是 LLaMA-7B
```

### Table 2-9: 基准评估

LLaMA 论文在大量 benchmark 上进行了评估。最令人印象深刻的结果：

```
Common Sense Reasoning:
  LLaMA-13B > GPT-3 175B （在 5/5 个基准上）
  
Closed-book QA:
  LLaMA-65B ≈ Chinchilla-70B ≈ PaLM-540B （用 1/8 的参数量）

Code Generation (HumanEval):
  LLaMA-65B: 23.7% → 虽然不如专用代码模型，但作为通用模型已经不错
  
MMLU (5-shot):
  LLaMA-65B: 63.4% → 接近 Chinchilla-70B (67.5%) 和 PaLM-540B (69.3%)
```

---

## 六、训练工程细节

### 6.1 训练效率

| 模型 | GPU 数量 | GPU 类型 | 训练时间 | GPU·小时 |
|------|---------|---------|---------|---------|
| 7B | 未公开 | A100-80GB | — | ~82K |
| 13B | 未公开 | A100-80GB | — | ~135K |
| 33B | 未公开 | A100-80GB | — | ~530K |
| 65B | 2048 | A100-80GB | ~21 天 | ~1022K |

**65B 模型**：约 380 tokens/sec/GPU，训练 1.4T tokens 需要 $\frac{1.4 \times 10^{12}}{380 \times 2048} \approx 1.8M \text{ seconds} \approx 21 \text{ days}$。

### 6.2 训练稳定性

论文提到在训练 65B 模型时遇到了不稳定性（loss spike），并通过以下方式缓解：
- 降低学习率
- 使用更激进的梯度裁剪
- 从最近的 checkpoint 恢复

这是大模型训练中的常见挑战——模型越大，训练越不稳定。

### 6.3 碳排放

论文诚实报告了碳排放：
- 65B 模型训练：约 **2638 MWh** 电力，**1015 吨 CO₂**
- 这反映了大模型训练的环境成本

---

## 七、LLaMA-1 的局限性

| 局限 | 说明 |
|------|------|
| **仅预训练，无对齐** | LLaMA-1 没有做 RLHF/SFT → 不适合直接作为助手使用 |
| **中文能力弱** | 训练数据以英文为主，中文 token 覆盖率低 |
| **上下文长度限制** | 2048 tokens，在当时已经偏短 |
| **许可证限制** | 初始发布为非商用许可（后来 LLaMA-2 改为商用） |
| **安全风险** | 无安全对齐，可能生成有害内容 |

---

## 八、LLaMA 对开源生态的影响

LLaMA-1 发布后，催生了庞大的开源生态：

```
LLaMA-1 (Meta, 2023.02)
├── Stanford Alpaca → 用 GPT-3.5 生成指令数据，在 LLaMA 上 SFT
│   └── 证明了 52K 条指令数据就能显著提升对话能力
├── Vicuna → 用 ShareGPT 数据微调 LLaMA
│   └── 开源 chatbot 的早期标杆
├── Chinese-LLaMA → 扩展中文词表 + 中文预训练
│   └── 开源中文 LLM 的起点
├── WizardLM → 进化式指令数据生成
├── Orca → 用 GPT-4 的 reasoning traces 训练
├── CodeLlama → 代码生成专用微调
└── 无数其他衍生模型...

LLaMA-2 (Meta, 2023.07) → 商用许可 + RLHF
├── LLaMA-2-Chat → 对齐后的对话模型
├── Qwen / DeepSeek / Mistral 等新架构借鉴了 LLaMA 的设计
└── 开源 LLM 进入"可商用"时代

LLaMA-3 (Meta, 2024.04) → 更大数据 + 更强性能
└── 8B 模型匹敌 LLaMA-2-70B → 数据工程的胜利
```

---

## 九、自检题

1. **LLaMA-1 的核心主张是什么？** 它与 GPT-3 的 scaling 策略有何不同？
2. **Chinchilla Scaling Law 说了什么？** LLaMA 为什么选择"过度训练"小模型？
3. **LLaMA 的训练数据有什么特点？** 为什么只使用公开数据是一个重要的设计决策？
4. **列出 LLaMA 相比 GPT 的 4 大架构改进及其各自的动机。**
5. **LLaMA 的 $d_{ff}$ 为什么不是简单的 $4d$？** 计算 LLaMA-7B 的 $d_{ff}$。
6. **LLaMA-13B 为什么能超过 GPT-3 175B？** 这说明了什么？
7. **LLaMA-1 对开源 LLM 生态有什么影响？** 列举至少 3 个衍生项目。

---

## 十、产出要求

- [ ] 撰写一份 LLaMA vs GPT 技术差异分析表（含架构 / 训练数据 / 训练策略 / 设计动机四个维度）
- [ ] 画出 LLaMA 系列的技术演进时间线
- [ ] 用自己的话解释 "小模型 + 多数据" 策略的优势
- [ ] 计算 LLaMA-7B 的 $d_{ff}$，验证 $d_{ff} = 11008$
