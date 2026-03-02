# Day 1：GPT 系列论文精读 — 从 GPT-1 到 GPT-4

> **目标**：系统梳理 GPT 系列的技术演进，理解每一代的核心创新、架构变化、训练策略和能力跃迁，建立从「预训练+微调」到「RLHF 对齐」的完整认知。

---

## 一、GPT 系列演进总览

```
GPT-1 (2018.06)       GPT-2 (2019.02)       GPT-3 (2020.05)
  117M 参数              1.5B 参数              175B 参数
  预训练+微调            Zero-shot              Few-shot / ICL
       │                    │                      │
       ▼                    ▼                      ▼
  "无监督预训练           "语言模型就是           "规模即能力
   是有效的"              无监督多任务学习器"       涌现出 ICL"
                                                   │
                                                   ▼
                    InstructGPT (2022.03)       GPT-4 (2023.03)
                      ~175B 参数                 ~1.8T? (MoE, 推测)
                      SFT + RLHF                多模态 + 推理增强
                           │                        │
                           ▼                        ▼
                    "对齐人类偏好               "多模态统一
                     比规模更重要"               推理能力质变"
```

---

## 二、GPT-1：证明无监督预训练的价值

**论文**：*Improving Language Understanding by Generative Pre-Training* (Radford et al., 2018)

### 核心创新

GPT-1 的核心主张：**先用大量无标注文本做无监督预训练，再用少量标注数据做有监督微调**。

```
阶段 1: 无监督预训练（Unsupervised Pre-training）
  大量无标注文本 → Causal Language Model → 学习通用语言表示

阶段 2: 有监督微调（Supervised Fine-tuning）
  少量标注数据 → 在预训练模型上微调 → 下游任务
```

### 模型配置

| 配置 | 值 |
|------|-----|
| 架构 | 12 层 Transformer Decoder |
| 参数量 | 117M |
| $d_{\text{model}}$ | 768 |
| 注意力头数 | 12 |
| $d_{ff}$ | 3072 |
| 上下文长度 | 512 tokens |
| 训练数据 | BooksCorpus (~7000 本未出版书籍, ~800M words) |
| 优化器 | Adam (lr = 2.5e-4, warmup + cosine decay) |

### 预训练目标

标准的因果语言模型：

$$\mathcal{L}_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \ldots, u_{i-1}; \Theta)$$

其中 $k$ 是上下文窗口大小，$\mathcal{U} = \{u_1, \ldots, u_n\}$ 是无标注语料。

### 微调策略

微调时加入辅助语言模型损失：

$$\mathcal{L}_3(\mathcal{C}) = \mathcal{L}_2(\mathcal{C}) + \lambda \cdot \mathcal{L}_1(\mathcal{C})$$

其中 $\mathcal{L}_2$ 是有监督的分类/回归损失，$\lambda$ 控制辅助损失权重。

**关键设计**：不同下游任务通过**输入格式转换**统一：

```
分类:    [Start] 文本 [Extract] → 线性层 → 类别
蕴涵:    [Start] 前提 [Delim] 假设 [Extract] → 线性层 → 蕴涵/矛盾/中立
相似度:  [Start] 文本A [Delim] 文本B [Extract] + [Start] 文本B [Delim] 文本A [Extract] → 相加
多选:    [Start] 上下文 [Delim] 选项i [Extract] → softmax
```

### 实验结果

在 12 个 NLP 基准中的 9 个上达到 SOTA，证明了预训练+微调范式的有效性。

### 历史意义

- **第一次**系统证明：无监督预训练 + 微调的 Transformer Decoder 可以在多种 NLP 任务上取得优异效果
- 奠定了「预训练 → 微调」的两阶段范式
- 但仍然需要对每个任务做微调

---

## 三、GPT-2：语言模型是无监督多任务学习器

**论文**：*Language Models are Unsupervised Multitask Learners* (Radford et al., 2019)

### 核心创新

**GPT-2 的革命性主张：一个足够好的语言模型，不需要微调，直接就能做各种任务。**

这是从「预训练+微调」到「zero-shot」的范式转变。

### 与 GPT-1 的关键差异

| 维度 | GPT-1 | GPT-2 |
|------|-------|-------|
| 参数量 | 117M | 1.5B (**13×**) |
| 训练数据 | BooksCorpus (~800M words) | WebText (~40GB, ~10B words) |
| 上下文长度 | 512 | 1024 |
| 使用方式 | 预训练 + 微调 | **Zero-shot** (无需微调) |
| LayerNorm | Post-Norm | **Pre-Norm** (移到子层之前) |
| 初始化 | 标准 | 残差层权重缩放 $1/\sqrt{N}$ |

### 架构改进

```python
# GPT-1: Post-Norm（Transformer 原始设计）
x = x + Sublayer(LayerNorm(x))  # 不对，实际是
x = LayerNorm(x + Sublayer(x))

# GPT-2: Pre-Norm（更稳定的训练）
x = x + Sublayer(LayerNorm(x))
```

Pre-Norm 使梯度流动更稳定，成为后续所有大模型的标准做法。

### 残差层初始化

随模型深度增加，残差路径的累积方差会增大。GPT-2 对残差连接的投影层做了特殊初始化：

$$W_{\text{proj}} \sim \mathcal{N}\left(0, \frac{0.02}{\sqrt{2N}}\right)$$

其中 $N$ 是 Transformer 层数。直觉：层数越多，每层的贡献越小，防止信号爆炸。

### Zero-shot 范式

不微调，直接用自然语言提示模型做任务：

```
翻译: "translate English to French: cheese →"
摘要: "TL;DR:" 放在文章后面
问答: 给出段落后 "Q: xxx A:"
```

### 历史意义

- 证明了规模足够大时，语言模型可以 zero-shot 做任务
- Pre-Norm 成为标准
- 引发了「大模型涌现能力」的讨论

---

## 四、GPT-3：规模的胜利，In-Context Learning 的诞生

**论文**：*Language Models are Few-Shot Learners* (Brown et al., 2020)

### 核心创新

GPT-3 的核心发现：**足够大的语言模型可以通过 In-Context Learning (ICL)，仅凭 prompt 中的几个示例就能做任务，无需任何参数更新。**

### 三种使用范式

```
Zero-shot:
  "Translate English to French: cheese →"
  模型输出: "fromage"

One-shot:
  "Translate English to French:
   sea otter → loutre de mer
   cheese →"
  模型输出: "fromage"

Few-shot:
  "Translate English to French:
   sea otter → loutre de mer
   peppermint → menthe poivrée
   plush girafe → girafe en peluche
   cheese →"
  模型输出: "fromage"
```

### 模型规模

| 模型 | 参数量 | 层数 | $d_{\text{model}}$ | 头数 | $d_{head}$ | 上下文 | 批大小 |
|------|--------|------|---------------------|------|-----------|--------|--------|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 2048 | 0.5M |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 2048 | 0.5M |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 2048 | 0.5M |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 2048 | 1M |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 2048 | 1M |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2048 | 2M |
| GPT-3 13B | 13B | 40 | 5140 | 40 | 128 | 2048 | 2M |
| **GPT-3 175B** | **175B** | **96** | **12288** | **96** | **128** | **2048** | **3.2M** |

### 训练数据

| 数据集 | Token 数 | 采样比例 |
|--------|---------|---------|
| Common Crawl (过滤) | 410B | 60% |
| WebText2 | 19B | 22% |
| Books1 | 12B | 8% |
| Books2 | 55B | 8% |
| Wikipedia | 3B | 3% |
| **总计** | **~300B** | **~500B tokens seen** (含重复) |

### Scaling Law 验证

GPT-3 论文中展示了清晰的 scaling 曲线——从 125M 到 175B，loss 平滑下降，且 few-shot 能力随规模增长显著提升：

```
模型规模 →     125M    1.3B    13B     175B
             ──────────────────────────
算术 (2位加法)  ~5%    ~10%    ~30%    ~100%
翻译 (En→Fr)   ~10    ~20     ~30     ~40+ BLEU
ICL 能力       弱      中      强      很强
```

### 关键技术细节

**1. 交替使用稀疏注意力**

GPT-3 175B 在某些层使用了局部带状稀疏注意力模式（类似 Sparse Transformer），以降低超长序列的计算成本。

**2. 训练配置**

- 优化器：Adam ($\beta_1=0.9, \beta_2=0.95$)
- 批大小：逐步增大（从 32K tokens 到 3.2M tokens）
- 学习率：cosine decay，warmup 375M tokens
- 权重衰减：0.1

### In-Context Learning 的本质

ICL 是 GPT-3 最重要的发现。其本质仍有争议：

| 假说 | 解释 |
|------|------|
| **隐式贝叶斯推理** | 模型在预训练中学到了任务分布的先验，prompt 中的示例提供了似然 |
| **梯度下降类比** | ICL 类似于在前向传播中隐式地做了一步梯度更新 |
| **任务识别** | 模型已经学会了各种任务，示例只是帮它「识别」要做哪个任务 |
| **Induction Head** | Transformer 中的特定注意力头可以做「复制 + 偏移」模式匹配 |

### 历史意义

- In-Context Learning 成为大模型的标志性能力
- 验证了 Scaling Law 在下游任务上的有效性
- 开启了 prompt engineering 时代
- 证明了「大力出奇迹」—— 但也暴露了对齐问题（模型可能生成有害内容）

---

## 五、InstructGPT (GPT-3.5)：对齐人类偏好

**论文**：*Training language models to follow instructions with human feedback* (Ouyang et al., 2022)

### 核心问题

GPT-3 虽然强大，但有严重问题：

```
用户: "请用简洁的语言解释量子力学"
GPT-3: (可能输出维基百科式的冗长文本，或者跑题，或者输出有害内容)

用户想要: 简洁、准确、有帮助的回答
模型实际: 只在做下一个 token 预测，不在乎用户意图
```

这就是**对齐问题 (Alignment Problem)**：模型的训练目标（预测下一个 token）与用户的真实需求（有帮助、无害、诚实）不一致。

### RLHF 三阶段流程

```
阶段 1: Supervised Fine-Tuning (SFT)
  收集人类标注的 (指令, 期望回答) 数据
  在 GPT-3 上做有监督微调
  → SFT 模型

阶段 2: Reward Model Training (RM)
  收集人类对模型输出的排序偏好数据
  训练一个 Reward Model 来预测人类偏好
  → Reward Model

阶段 3: PPO (Proximal Policy Optimization)
  用 RM 的打分作为奖励信号
  通过 PPO 算法优化 SFT 模型
  → InstructGPT (对齐后的模型)
```

### 数据规模

| 阶段 | 数据量 | 数据内容 |
|------|--------|---------|
| SFT | ~13K 条 | 人类标注员写的高质量回答 |
| RM | ~33K 条 | 对模型输出的排名（4-9 个输出排序） |
| PPO | ~31K 条 | 用户 prompt（无需标注） |

### 核心发现

**一个经过 RLHF 对齐的 1.3B 模型，在人类评估中优于未对齐的 175B GPT-3。**

这说明**对齐比单纯增大规模更重要**。

### Bradley-Terry 偏好模型

RM 训练使用 Bradley-Terry 模型：

$$P(y_w \succ y_l) = \sigma(r(x, y_w) - r(x, y_l))$$

Loss：

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} [\log \sigma(r(x, y_w) - r(x, y_l))]$$

其中 $y_w$ 是人类偏好的输出，$y_l$ 是不偏好的输出，$r(x, y)$ 是 RM 的打分。

### 历史意义

- **首次大规模验证了 RLHF 在 LLM 上的有效性**
- 成为 ChatGPT 的直接技术基础
- 建立了「SFT → RM → PPO」三阶段对齐范式
- 影响了后续所有对齐工作（DPO、Constitutional AI 等）

---

## 六、GPT-4：多模态 + 推理跃迁

**论文**：*GPT-4 Technical Report* (OpenAI, 2023)

注意：GPT-4 技术报告刻意隐藏了大量细节（架构、训练数据、计算量等均未公开）。以下部分信息来自可靠推测和泄露。

### 已知信息

| 维度 | GPT-4 |
|------|-------|
| 多模态 | 支持图像输入 + 文本输出 |
| 上下文长度 | 8K / 32K / 128K (Turbo) |
| 训练数据 | 未公开，推测 10T+ tokens |
| 安全 | RLHF + 红队测试 + 系统级安全措施 |

### 推测信息（来自社区分析）

| 维度 | 推测值 |
|------|--------|
| 架构 | 8 × ~220B 专家的 MoE（Mixture of Experts）|
| 总参数量 | ~1.8T |
| 每次推理激活 | ~220B |
| 训练计算 | ~$100M+ |

### 能力跃迁

GPT-4 相比 GPT-3.5 的最大进步在于**推理能力**：

```
考试成绩对比（百分位数）:
                    GPT-3.5    GPT-4
  统一律师资格考试    ~10%     ~90%
  SAT 数学           ~70%     ~90%
  GRE 写作           ~54%     ~80%
  AP 生物学          ~65%     ~85%
```

### 可预测的 Scaling

GPT-4 论文的一个重要贡献是展示了训练 loss 的可预测性：

> "在训练开始前，通过小规模实验就能准确预测最终模型的 loss。"

这意味着 OpenAI 可以在投入 $100M+ 训练前，就知道模型大概能达到什么性能。

---

## 七、GPT 系列核心创新对比表

| 维度 | GPT-1 | GPT-2 | GPT-3 | InstructGPT | GPT-4 |
|------|-------|-------|-------|-------------|-------|
| **年份** | 2018 | 2019 | 2020 | 2022 | 2023 |
| **参数量** | 117M | 1.5B | 175B | ~175B | ~1.8T (MoE) |
| **层数** | 12 | 48 | 96 | 96 | 未公开 |
| **上下文** | 512 | 1024 | 2048 | 2048 | 8K/32K/128K |
| **训练数据** | ~800M words | ~10B words | ~300B tokens | 同 GPT-3 | 10T+ tokens |
| **核心创新** | 预训练+微调 | Zero-shot, Pre-Norm | Few-shot, ICL, Scaling | RLHF 对齐 | 多模态, MoE |
| **使用范式** | 微调 | Zero-shot | Few-shot | 对话/指令 | 多模态对话 |
| **Norm** | Post-Norm | **Pre-Norm** | Pre-Norm | Pre-Norm | 未公开 |
| **关键 Loss** | CLM | CLM | CLM | CLM + RM + PPO | 未公开 |

---

## 八、技术演进中的关键洞察

### 1. 从「任务特化」到「通用」的转变

```
GPT-1: 每个任务需要微调 → 任务特化模型
GPT-2: Zero-shot，无需微调 → 开始通用化
GPT-3: Few-shot + ICL → 真正的通用模型
InstructGPT: + 对齐 → 可用的通用助手
GPT-4: + 多模态 → 通用智能体雏形
```

### 2. 规模 vs 对齐的平衡

InstructGPT 1.3B > GPT-3 175B（人类评估），说明：
- 规模是能力的基础
- 对齐决定了能力能否被正确利用
- 两者缺一不可

### 3. Pre-Norm 的胜出

GPT-2 引入 Pre-Norm 后，成为事实标准：
- 训练更稳定（梯度不会因 LayerNorm 而被缩放）
- 深层模型训练不需要精细的学习率调整
- LLaMA、Mistral、DeepSeek 等全部采用 Pre-Norm

### 4. 数据是关键的隐变量

- GPT-1: BooksCorpus → 高质量但小规模
- GPT-2: WebText → 网页数据质量控制
- GPT-3: 多来源混合 + 采样比例调整
- GPT-4: 数据工程是竞争力核心（OpenAI 对此最为保密）

---

## 九、自检题

1. **GPT-1 的两阶段训练是什么？** 为什么微调时加入辅助语言模型损失？
2. **GPT-2 相比 GPT-1 的架构改进有哪些？** Pre-Norm 为什么更稳定？
3. **什么是 In-Context Learning？** 它和微调有什么本质区别？
4. **InstructGPT 的 RLHF 三阶段分别做什么？** 为什么 1.3B 对齐模型能胜过 175B 未对齐模型？
5. **GPT-4 的 MoE 架构意味着什么？** 总参数 1.8T 但推理成本为什么可控？
6. **从 GPT-1 到 GPT-4，「使用范式」经历了怎样的变迁？** 用一句话概括每代。

---

## 十、产出要求

- [ ] 撰写一份 GPT 系列对比表（包含架构 / 数据 / 训练策略 / 涌现能力四个维度）
- [ ] 画出 GPT 系列的技术演进时间线
- [ ] 用自己的话解释 In-Context Learning 的工作原理
