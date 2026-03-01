# Day 1：LLM 发展历程 — 从 Word2Vec 到 DeepSeek

## 一、技术演进时间线

```
2013  Word2Vec          ← 词向量革命，但无法捕捉上下文
  │
2017  Transformer       ← "Attention Is All You Need"，奠基石
  │
  ├─ 2018  GPT-1        ← Decoder-only, 无监督预训练 + 有监督微调
  ├─ 2018  BERT         ← Encoder-only, 双向掩码语言模型 (MLM)
  ├─ 2019  GPT-2        ← 更大规模, zero-shot 能力涌现
  ├─ 2019  T5           ← Encoder-Decoder, "Text-to-Text" 统一范式
  │
2020  GPT-3             ← 175B 参数, few-shot 能力, Scaling Law 验证
  │
  ├─ 2022  InstructGPT  ← RLHF 首次大规模应用 (GPT-3.5 前身)
  ├─ 2022  ChatGPT      ← 对话式 AI 引爆行业
  ├─ 2023  GPT-4        ← 多模态, 推理增强
  │
  ├─ 2023  LLaMA-1/2    ← Meta 开源, 开源生态起点
  ├─ 2023  Mistral-7B   ← Sliding Window Attention, GQA
  ├─ 2023  Qwen         ← 阿里通义, 中文大模型
  ├─ 2024  LLaMA-3      ← 更大规模开源
  ├─ 2024  DeepSeek-V2  ← MLA + MoE, 训练效率革命
  ├─ 2025  DeepSeek-R1  ← RL 驱动推理, GRPO
  └─ ...
```

### 关键转折点解读

#### 1. Transformer (2017) — 一切的起点

**核心创新**：Self-Attention 机制替代 RNN/LSTM 的序列建模方式。

为什么重要？
- **并行化**：RNN 必须按时间步顺序计算，Transformer 可以并行处理所有位置
- **长距离依赖**：Attention 可以直接连接任意两个位置，无需逐步传递
- **可扩展性**：架构简洁，易于 scale up

#### 2. GPT 系列 — Decoder-only 的胜利

| 模型 | 参数量 | 核心创新 |
|------|--------|---------|
| GPT-1 (2018) | 117M | 无监督预训练 + 有监督微调 |
| GPT-2 (2019) | 1.5B | zero-shot, 不需要微调就能做任务 |
| GPT-3 (2020) | 175B | few-shot/in-context learning, Scaling Law |
| InstructGPT (2022) | ~175B | RLHF 对齐人类偏好 |
| GPT-4 (2023) | ~1.8T (推测) | 多模态, MoE (推测) |

**GPT 的核心哲学**：一个足够大的语言模型，通过预测下一个 token，就能「涌现」出各种能力。

#### 3. BERT (2018) — Encoder-only 的辉煌

- **双向注意力**：每个 token 能看到左右所有上下文
- **预训练任务**：Masked Language Model (MLM) + Next Sentence Prediction (NSP)
- **影响**：NLU 任务（分类、NER、QA）的统治者，但**不适合生成任务**
- **现状**：在 LLM 时代，BERT 系列退居「嵌入模型」角色（检索、分类）

#### 4. 开源生态 (2023~) — 竞争格局

| 模型 | 机构 | 特点 |
|------|------|------|
| LLaMA-1/2/3 | Meta | 开源标杆，社区生态最丰富 |
| Mistral/Mixtral | Mistral AI | 高效架构 (GQA, Sliding Window, MoE) |
| Qwen-1/2 | 阿里 | 中文能力强，多模态 |
| DeepSeek-V2/V3/R1 | DeepSeek | MLA, MoE, GRPO, 训练效率极高 |
| Yi | 零一万物 | 中英双语 |
| Gemma | Google | 轻量开源 |

---

### 主流开源模型详解

以下按「机构/系列」介绍当前最重要的开源大模型，便于你选型、对比和面试时回答「你了解哪些开源模型」。

---

#### （1）LLaMA 系列 — Meta，开源标杆

| 版本 | 发布时间 | 参数量 | 核心特点 |
|------|----------|--------|----------|
| LLaMA-1 | 2023.02 | 7B / 13B / 33B / 65B | 仅研究许可，证明「小模型+海量数据」可行 |
| LLaMA-2 | 2023.07 | 7B / 13B / 70B | 商用开放，2T token 预训练，RLHF 对话版 Llama-2-Chat |
| LLaMA-3 | 2024.04 | 8B / 70B / 405B(部分) | 15T+ token，128K 上下文，128K 词表，GQA |
| LLaMA-3.1 / 3.2 / 3.3 | 2024~2025 | 8B~405B | 多尺寸、多模态（3.2 视觉），3.3 70B 性能对标 405B |

**为什么重要**  
- 生态最成熟：Hugging Face、vLLM、Llama.cpp、各种微调/量化版本都优先支持 LLaMA。  
- 架构成为事实标准：RoPE、SwiGLU、GQA 等设计被后续开源模型广泛沿用。  
- 许可证友好：LLaMA-2 起可商用，企业选型时常用基线。

**适合**：做研究、做产品基线、学习「标准」Decoder-only 架构。

---

#### （2）Mistral / Mixtral — Mistral AI，效率优先

| 模型 | 参数量 / 激活参数 | 核心特点 |
|------|-------------------|----------|
| Mistral-7B | 7B | Sliding Window Attention(4K 窗口)、GQA，同规模推理快、长上下文省显存 |
| Mixtral 8×7B | 总 46B，激活约 13B | 8 个专家 MoE，每 token 激活 2 个专家，性能接近 LLaMA-2-70B，速度远快 |
| Mixtral 8×22B | 总约 176B，激活约 39B | 更大 MoE，更强能力 |
| Mistral Large / Nemo 等 | 多规格 | 闭源/开源混合，多语言与指令跟随强 |

**关键技术**  
- **Sliding Window Attention (SWA)**：每个位置只对局部窗口（如 4096 token）做注意力，层数够多时信息仍可间接传递到很远，长上下文显存近似 O(窗口大小) 而非 O(序列长度²)。  
- **Grouped Query Attention (GQA)**：多 head 共享部分 K/V，在 MHA 和 MQA 之间折中，减少 KV cache，加速推理。  
- **MoE (Mixtral)**：多个「专家」MLP，每 token 只走少数专家，总参数量大但算力/显存成本接近小模型。

**适合**：显存或延迟敏感、需要长上下文、希望「小显存跑出大模型效果」的场景。

---

#### （3）Qwen 系列 — 阿里通义，中文与多模态

| 系列 | 典型规模 | 核心特点 |
|------|----------|----------|
| Qwen-1 | 1.8B~72B | 中文数据占比高，中英双语均衡 |
| Qwen2 | 0.5B~72B | 词表 128K，多尺寸，数学/代码增强 |
| Qwen2.5 | 0.5B~72B，含 MoE 版 | 18T token 训练，72B-Instruct 可对标 LLaMA-3-405B；VL 多模态 |
| Qwen2.5-VL / Qwen2-VL | 2B~32B | 视觉-语言模型，支持图像理解与生成，可本地部署 |

**为什么重要**  
- 中文能力在开源模型中第一梯队，适合中文产品、客服、内容生成。  
- 尺寸覆盖 0.5B~72B，便于从端侧到云侧选型。  
- 多模态（VL）版本开源且可用，做「图文理解」或多模态 RAG 时常用。

**适合**：中文场景、多模态应用、需要从 0.5B 到 72B 统一技术栈的团队。

---

#### （4）DeepSeek 系列 — 训练效率与推理创新

| 模型 | 参数量 / 激活 | 核心特点 |
|------|----------------|----------|
| DeepSeek-V2 | 236B 总，21B 激活 | MLA（Multi-head Latent Attention）+ MoE，训练成本远低于同性能稠密模型 |
| DeepSeek-V3 | 671B 总，37B 激活 | 14.8T token 训练，性能第一梯队；MoE 架构 |
| DeepSeek-R1 | 多规格 | 推理阶段强化学习（GRPO 等），「思考链」显式、可解释，强推理与数学 |

**关键技术**  
- **MLA**：用低秩/潜在表示压缩注意力，减少计算与显存。  
- **MoE**：极多专家、每 token 激活少，总参数大、单次推理成本可控。  
- **R1**：推理时「先想再答」，模型输出中间推理过程，再给出最终答案，便于对齐与调试。

**适合**：关注训练/推理效率、数学与推理能力、想理解「MoE + 高效注意力」的工程师。

---

#### （5）Yi — 零一万物，中英双语

| 模型 | 规模 | 特点 |
|------|------|------|
| Yi-1.5 | 6B / 9B / 34B / 65B 等 | 中英双语优化，长上下文（200K），Apache 2.0 |
| Yi-VL | 多规格 | 视觉-语言，图文理解与生成 |

**定位**：中英平衡、长上下文、完全开源（Apache 2.0），适合需要商用且重视中文的团队。

---

#### （6）Gemma — Google，轻量开源

| 版本 | 规模 | 特点 |
|------|------|------|
| Gemma-1 | 2B / 7B | 从 Gemini 技术拆出，轻量、易部署 |
| Gemma-2 | 2B / 9B / 27B | 27B 用 2K 上下文训练，质量高 |
| Gemma-3 | 1B / 4B / 12B / 27B | 多模态、128K 上下文（部分规格），27B 在多项基准表现突出 |

**定位**：Google 官方开源、商用友好，适合需要「小参数、高质量」的端侧或中等规模服务。

---

#### （7）其他值得关注的开源模型

- **Phi 系列 (Microsoft)**：3B、14B 等，强调「小模型、高质量数据」；Phi-3 可做端侧推理。  
- **InternLM (上海 AI Lab)**：7B~20B，中文与工具调用；InternLM2.5 等版本在中文基准上表现好。  
- **Falcon (TII)**：7B、40B、180B，训练数据强调多语言与高质量；商用需留意许可。  
- **Code Llama / StarCoder 等**：专注代码生成与补全，做 Code 场景时可与通用模型对比。

---

#### 如何选型？（简要对照）

| 需求 | 可优先考虑 |
|------|------------|
| 生态、教程、微调资源最多 | LLaMA 系列 |
| 显存紧、要长上下文、要性价比 | Mistral / Mixtral |
| 中文为主、多模态 | Qwen2.5 / Qwen2.5-VL |
| 极致性能、数学/推理 | DeepSeek-V3 / R1 |
| 中英 + 长上下文 + 完全开源 | Yi-1.5 |
| 小参数、高质量、多模态 | Gemma-2 / Gemma-3 |

完成 Day 1 学习后，建议能口头说出：**LLaMA / Mistral / Qwen / DeepSeek 各一条核心差异**（例如：LLaMA 生态标杆、Mistral 效率与 MoE、Qwen 中文与多模态、DeepSeek MoE+推理创新）。

---

## 二、为什么 Decoder-only 赢了？

这是理解现代 LLM 最重要的问题之一。

### 三种架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    Encoder-only (BERT)                           │
│  输入: [CLS] The cat sat on the [MASK] . [SEP]                  │
│  注意力: 双向 (每个token看所有token)                               │
│  任务: 填空 (MLM), 分类, NER                                     │
│  缺点: 不擅长生成                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Encoder-Decoder (T5)                          │
│  Encoder输入: "translate English to French: The cat sat..."      │
│  Decoder输出: "Le chat s'est assis..."                           │
│  注意力: Encoder双向, Decoder因果, Cross-Attention连接            │
│  适合: 翻译, 摘要, Seq2Seq                                       │
│  缺点: 架构复杂, 参数效率低(encoder参数在生成时利用率低)            │
├─────────────────────────────────────────────────────────────────┤
│                    Decoder-only (GPT)                            │
│  输入: "The cat sat on the"                                      │
│  输出: 预测下一个token → "mat"                                    │
│  注意力: 因果 (每个token只看它左边的token)                         │
│  适合: 一切文本生成 + 通过prompt也能做分类/NER                     │
│  优势: 架构简洁, 扩展性好, 统一范式                                │
└─────────────────────────────────────────────────────────────────┘
```

### Decoder-only 胜出的原因

1. **统一范式**：所有任务都可以转化为「生成下一个 token」—— 分类、翻译、摘要、代码、数学推理...
2. **Scaling 友好**：架构简洁，容易 scale 到万亿参数
3. **涌现能力**：规模足够大后，出现 in-context learning、chain-of-thought 等「涌现」能力
4. **训练效率**：因果语言模型的训练目标简单（预测下一个 token），数据利用效率高

---

## 三、Scaling Law — 大力出奇迹的理论基础

### Kaplan Scaling Law (OpenAI, 2020)

模型性能（以 loss 衡量）主要由三个因素决定：

```
L(N, D, C) ∝ N^{-αN} + D^{-αD} + C^{-αC}

其中:
  N = 模型参数量
  D = 训练数据量 (token 数)
  C = 计算量 (FLOPs)
  α = 幂律指数
```

**核心发现**：
- 性能随 N, D, C 的增长呈**幂律**(power law)提升
- 在固定计算预算下，**模型参数和数据量应等比例增长**
- 架构细节（层数 vs 宽度）的影响相对小

### Chinchilla Scaling Law (DeepMind, 2022)

修正了 Kaplan 的结论：
- 最优策略：**参数量和数据量应以 1:20 的比例增长**
- 即 1B 参数模型需要 ~20B token 训练
- LLaMA 的成功证实了这一点：7B 模型用了 1T+ token 训练

### 为什么博士生要理解 Scaling Law？

- 它解释了为什么「大」模型能涌现出小模型没有的能力
- 它指导了工业界的资源分配（应该花钱在更大模型还是更多数据？）
- 它是当前 LLM 研究的核心范式之一
- **局限性**：Scaling Law 告诉你 loss 的变化，但不能精确预测下游任务能力的涌现

---

## 四、自检题

完成 Day 1-2 的学习后，你应该能回答：

1. **为什么 Decoder-only 架构成为主流？** 至少说出 3 个原因。
2. **GPT-1 到 GPT-4 的核心创新分别是什么？** 各用一句话概括。
3. **Scaling Law 说了什么？Chinchilla 修正了什么？**
4. **BERT 在当前 LLM 时代的角色是什么？** 为什么它没有被完全淘汰？
5. **LLaMA 和 DeepSeek 各自的核心技术创新是什么？**
6. **主流开源模型选型**：若你要做「中文客服助手」「长上下文文档问答」「端侧小模型」，分别更适合选哪一类开源模型（可答系列名+简要理由）？

---

## 五、产出要求

- [ ] 手绘一张 LLM 技术演进时间线思维导图（从 2017 Transformer 到 2025 DeepSeek-R1）
- [ ] 写一篇 2 页笔记：三大架构对比 + Scaling Law
