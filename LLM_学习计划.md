<p align="center">
  <h1 align="center">手撕 LLM + RLHF 系统化学习计划</h1>
  <p align="center">
    <em>从 Transformer 底层原理到 RLHF 对齐，从单卡训练到分布式系统，从文本到多模态与推理增强</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/版本-v2.0-blue" alt="version" />
    <img src="https://img.shields.io/badge/周期-18~22_周-green" alt="duration" />
    <img src="https://img.shields.io/badge/难度-博士级-red" alt="level" />
    <img src="https://img.shields.io/badge/更新-2026.03-orange" alt="update" />
  </p>
</p>

---

> **适用人群** — 有深度学习基础、希望系统掌握大模型核心技术的研究者与博士生
>
> **学习目标** — 建立完整的大模型技术栈，覆盖 MoE、数据工程、现代评估、量化、安全与部署等常被忽略的关键领域
>
> **预计周期** — 18～22 周（可根据基础弹性压缩至 16 周）

## 使用说明

| 标记 | 含义 |
|:---:|------|
| 📌 | **进阶拓展** — 容易被忽略但对实际研究和工程至关重要的模块 |
| ⚡ | **知识点边界** — 同一知识点分「入门」与「深化」两次出现，已标注衔接关系 |
| ✅ | **关键检查点** — 建议完成后自测的核心能力项 |

---

## 目录

- [学习路线总览](#学习路线总览)
- [阶段一：基础筑基（第 1～4 周）](#阶段一基础筑基第-14-周)
- [阶段二：高效微调（第 5～7 周）](#阶段二高效微调第-57-周)
- [阶段三：应用落地（第 8 周）](#阶段三应用落地第-8-周)
- [阶段四：强化学习对齐（第 9～13 周）](#阶段四强化学习对齐第-913-周)
- [阶段五：系统优化与前沿（第 14～18 周）](#阶段五系统优化与前沿第-1418-周)
- [进阶拓展路线图](#进阶拓展路线图)
- [面试高频手撕代码清单](#面试高频手撕代码清单)
- [推荐学习资源](#推荐学习资源)
- [学习建议](#学习建议)
- [周进度追踪表](#周进度追踪表)

---

## 学习路线总览

```
阶段一：基础筑基（第 1-4 周）
  LLM 概述 → Transformer → GPT → LLaMA
        ↓
阶段二：高效微调（第 5-7 周）
  Alpaca / Instruction Tuning → LoRA / QLoRA → Chinese-LLaMA2 + 数据工程
        ↓
阶段三：应用落地（第 8 周）
  Agent（Tool Use / Memory）→ RAG → 多轮对话部署
        ↓
阶段四：强化学习对齐（第 9-13 周）
  RL 基础 → RLHF 原理与实现 → DPO / R1 / GRPO → 垂域 Chatbot → 多卡训练
        ↓
阶段五：系统优化与前沿（第 14-18 周）
  推理加速 + 量化 + 部署 → 分布式训练 + MoE → 多模态 VLM → o1 推理 → 综合收尾
```

---

## 阶段一：基础筑基（第 1～4 周）

> **核心思想**：没有扎实的基础，后面的一切都是空中楼阁。本阶段目标——能从零手写 Transformer、GPT、LLaMA 的核心代码。

### 第 1 周 · LLM 概述与预备知识

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| LLM 发展历程 | GPT → BERT → T5 → ChatGPT → 开源生态（LLaMA / Mistral / Qwen / DeepSeek）技术演进 | 手绘时间线思维导图 |
| LLM 技术路线 | Encoder-only / Decoder-only / Encoder-Decoder 架构对比；Scaling Law | 撰写 2 页笔记 |
| 数据预处理与 Tokenizer | **现代 Tokenizer**：BPE / Unigram / SentencePiece 原理与选型；Embedding 层设计；tiktoken | 手写 BPE 编码器；理解 subword 与词表扩展 |
| Word2Vec（选学） | CBOW / Skip-gram / 负采样；**时间紧可略过** | 可选：用 gensim 训练小模型 |
| 评价指标 | **经典**：BLEU / ROUGE / PPL 数学定义与适用场景 | 实现 BLEU 和 PPL 计算函数 |
| 📌 现代评估体系 | MMLU / HumanEval / MT-Bench / Arena Elo 的定位与使用场景 | 了解各评测方式与排行榜 |

**必读论文**

- *Attention Is All You Need* (Vaswani et al., 2017)
- *Language Models are Few-Shot Learners* (GPT-3, 选读，与第 3 周衔接)

**补充资源**

- Andrej Karpathy — *Let's build GPT from scratch* (YouTube)
- 李沐 —《动手学深度学习》第 10–11 章

---

### 第 2 周 · 手撕 Transformer

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| 模型架构 | 完整 Transformer：6 层 Encoder + 6 层 Decoder 的数据流 | 画出完整架构图（含维度标注） |
| Position Encoding | 正弦 / 余弦位置编码的数学推导及相对位置表示能力 | 手写位置编码 + 可视化热力图 |
| Attention 原理与实现 | Scaled Dot-Product Attention、Multi-Head Attention 完整推导 | **从零实现 Multi-Head Attention** |
| Encoder-Decoder | 完整 Encoder / Decoder 模块实现，交叉注意力机制 | 代码实现 |
| Masked 原理 | Padding Mask / Causal Mask（Look-ahead Mask）的作用与实现 | 理解并实现两种 Mask |
| **实践** | **英→法文本翻译** | 用手写 Transformer 跑通训练与推理 |

**关键检查点**

- [ ] 独立手写 `MultiHeadAttention`
- [ ] 独立手写 `PositionalEncoding`
- [ ] 手写完整 `TransformerBlock`
- [ ] 翻译任务 BLEU ≥ 30

---

### 第 3 周 · 手撕 GPT

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| 论文精读 | GPT-1 / 2 / 3 / 3.5（InstructGPT）/ 4 核心创新点对比 | 撰写论文对比表（架构 / 数据 / 训练策略 / 涌现能力） |
| GPT 模型架构 | Decoder-only 架构、Causal Language Modeling 目标函数 | 手写 GPT 模型 |
| BPE 编码原理 | Byte Pair Encoding 完整算法；tiktoken 使用 | 手写 BPE 训练 + 编解码 |
| Generate Top-K | Top-K / Top-P（Nucleus）/ Temperature 采样策略 | 实现三种采样函数 |
| ⚡ FlashAttention（入门） | IO 感知算法原理、分块计算、内存优化；**系统深化 → 第 14 周** | 阅读论文 + 理解 tiling 思想 |
| **实践** | **GPT 预训练及推理** | 在小规模语料上完成预训练与文本生成 |

**必读论文**

- *Language Models are Few-Shot Learners* (GPT-3, Brown et al., 2020)
- *FlashAttention: Fast and Memory-Efficient Exact Attention* (Dao et al., 2022)

---

### 第 4 周 · 手撕 LLaMA

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| 论文精读 | LLaMA-1 / 2 核心创新：数据配比、训练策略、模型改进 | 撰写 LLaMA vs GPT 技术差异分析 |
| LLaMA 模型架构 | Pre-Norm、GQA（Grouped Query Attention）等与 GPT 的区别 | 画出 LLaMA Block 详细架构图 |
| RMSNorm | RMSNorm vs LayerNorm 数学对比与工程优势 | 手写 RMSNorm |
| RoPE | 旋转位置编码的复数推导、旋转矩阵实现、外推性分析 | **手写 RoPE（面试高频！）** |
| SwiGLU | SwiGLU 激活函数设计动机与数学形式 | 手写 SwiGLU |
| ⚡ KV Cache（入门） | KV Cache 推理加速原理与显存开销分析；**系统深化 → 第 14 周** | 实现带 KV Cache 的推理 |
| **实践** | **LLaMA 预训练** | 在小规模数据上跑通预训练流程 |

**关键检查点**

- [ ] 独立手写 RoPE（含 `apply_rotary_pos_emb`）
- [ ] 独立手写 RMSNorm
- [ ] 独立手写 SwiGLU FFN
- [ ] 理解 GQA 并能实现

> **阶段一小结**：此时你应该能从零手写 Transformer / GPT / LLaMA 的完整代码，理解每一行的数学含义。这是面试的核心竞争力。

---

## 阶段二：高效微调（第 5～7 周）

> **核心思想**：预训练是大厂的游戏，微调才是多数人的战场。掌握指令微调与参数高效微调（PEFT）是落地关键。

### 第 5 周 · 手撕 Alpaca（指令微调）

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| Instruction Finetune | 指令微调核心思想：从「补全」到「遵循指令」 | 理解 SFT 数据格式 |
| Alpaca 模型 | Stanford Alpaca 技术方案与数据生成流程 | 阅读 Alpaca 论文 / 博客 |
| Self-Instruct | 自动化指令数据生成流程与质量控制 | 手写 Self-Instruct 数据管线 |
| CoT / ToT（了解） | Chain-of-Thought / Tree-of-Thought 作为推理策略；深入 → o1 推理阶段 | 实践 CoT prompting |
| Prompt / Prefix / Adapter | 三种参数高效微调方法原理对比 | 撰写对比笔记（参数量 / 效果 / 适用场景） |
| 效率对比 | Full Finetune / Adapter / LoRA 训练效率与效果 | 做实验对比表 |
| **实践** | **Alpaca Instruction 微调** | 完成一次完整指令微调实验 |

---

### 第 6 周 · 手撕 LoRA

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| 论文精读 | LoRA / QLoRA 两篇论文核心思想 | 撰写论文笔记 |
| LoRA 算法推导 | 低秩分解数学原理；为什么 Attention 权重矩阵具有低秩性 | 手写 LoRA 数学推导 |
| LoRA 实现 | LoRA 层代码实现、rank 选择、alpha 参数的作用 | **手写 LoRA Linear 层** |
| QLoRA | NF4 量化原理、双量化（Double Quantization）、分页优化器 | 理解量化数学 |
| 📌 推理量化简介 | GPTQ / AWQ / GGUF 的定位与使用场景（系统讲解 → 第 14 周） | 了解与 QLoRA 训练量化的区别 |
| **实践** | **LLaMA2 + QLoRA 微调** | 单卡微调 LLaMA2-7B |

**关键检查点**

- [ ] 手写 LoRA 前向传播：`h = W₀x + (α/r) · BAx`
- [ ] 理解 rank / alpha / target_modules 选择策略
- [ ] 单卡（A100 / A800）跑通 QLoRA 微调

---

### 第 7 周 · 手撕 Chinese-LLaMA2 + 数据工程

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| 论文讲解 | Chinese-LLaMA2 中文适配策略 | 笔记 |
| 中文 Tokenizer 扩展 | 在已有词表上扩展中文 token，embedding 初始化策略 | 手写词表扩展代码 |
| LLM 数据处理 | 大规模文本数据清洗、去重、质量过滤 | 实现数据处理管线 |
| 📌 数据工程 | 数据配比（The Pile / FineWeb）、去重（MinHash / SimHash）、质量打分与过滤；数据量 vs 质量 trade-off | 阅读 FineWeb / RedPajama 技术报告并写笔记 |
| 二次预训练 + SFT | Continual Pre-training 训练策略、学习率设置、数据配比 | 理解训练策略 |
| Baichuan2 论文 | Baichuan2 技术报告解读 | 对比分析笔记 |
| **实践** | **医疗知识二次预训练** | 在医疗语料上做 Continual Pre-training |

---

## 阶段三：应用落地（第 8 周）

> **核心思想**：大模型不是孤岛。Agent 和 RAG 是连接大模型与现实世界的桥梁。

### 第 8 周 · 手撕 chatLLaMA-Agent 与 RAG

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| 多轮对话部署 | 多轮对话 history 管理、上下文窗口截断策略 | 本地部署对话系统 |
| Agent 原理 | ReAct / Function Calling / Tool Use 核心范式 | 实现简单 Agent |
| 📌 Agent 扩展 | Tool Use 与 Function Calling 实现细节；Memory（短期 / 长期）；Planning（ReAct / Plan-and-Execute） | 实现带 Tool 的 Agent |
| RAG 算法原理 | Retrieval-Augmented Generation 完整流程：索引 → 检索 → 生成 | 手写 RAG 管线 |
| 文本对比网络 | 文本嵌入模型、对比学习损失函数（InfoNCE 等） | 理解 embedding 模型 |
| LangChain RAG | 使用 LangChain 构建 RAG 应用 | 完成 RAG 问答系统 |

**实践产出**

- [ ] 本地部署带 RAG 的多轮对话 Agent
- [ ] 支持文档上传与检索增强问答；可选：支持 Tool 调用

---

## 阶段四：强化学习对齐（第 9～13 周）

> **核心思想**：RLHF 是让大模型「对齐」人类偏好的关键技术。对齐方案内容庞杂，本计划**拆为 5 周**循序渐进，避免单周超载。

### 第 9 周 · 手撕 RL-PPO（强化学习基础）

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| MDP 基础 | 马尔可夫决策过程、Q 值 / V 值函数定义与计算 | 手写 Bellman 方程求解 |
| DQN | Deep Q-Network：经验回放、目标网络 | **手写 DQN** |
| Policy Gradient | REINFORCE 算法、策略梯度定理推导 | **手写 Policy Gradient** |
| Actor-Critic | Actor-Critic 框架、Advantage 函数、GAE | **手写 A2C** |
| PPO | PPO-Clip、clipped surrogate objective | **手写 PPO** |

**关键检查点**

- [ ] DQN 在 CartPole 上训练成功
- [ ] PPO 在 CartPole / LunarLander 上训练成功
- [ ] 理解 PPO clip ratio 与 KL penalty 的直觉

**必读论文**

- *Proximal Policy Optimization Algorithms* (Schulman et al., 2017)
- *Playing Atari with Deep Reinforcement Learning* (Mnih et al., 2013)

---

### 第 10 周 · 手撕 RLHF（原理与实现）

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| 论文精读 | InstructGPT (GPT-3.5) / LLaMA2 论文中 RLHF 部分 | 撰写详细笔记 |
| RLHF 三阶段 | SFT → Reward Model → PPO 完整流程 | 画出完整 RLHF 训练流程图 |
| Reward Model | 偏好数据收集、Bradley-Terry 模型、RM 训练目标 | 手写 Reward Model |
| RLHF 参数共享 | Actor / Critic / Reference / Reward 四模型参数共享策略 | 理解工程实现 |
| RLHF-PPO 架构 | PPO 在 LLM 中的适配：rollout → reward → update | 手写 RLHF-PPO 训练循环 |
| RLHF-PPO Loss | Policy Loss + Value Loss + KL Penalty 完整计算 | **手写完整 Loss 函数** |
| **实践** | 小规模 RLHF 训练 | 完成一次 RLHF 训练实验（不追求多卡） |

---

### 第 11 周 · DPO / R1 / GRPO（对齐方案扩展）

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| 📌 **DPO（与 PPO 并列）** | Direct Preference Optimization 原理；为何 DPO 更简单、更稳定、工业界广泛采用；DPO Loss 推导 | **手写 DPO Loss 与训练循环** |
| R1 训练 | DeepSeek-R1 训练方法与技术报告 | 精读论文并写笔记 |
| R1 详解 | R1 的 RL 训练细节、cold start 策略、数据管线 | 深入分析 |
| GRPO | Group Relative Policy Optimization 算法原理 | **手写 GRPO** |
| X-R1 实践 | 0.5B 模型的 GRPO 实践 | 跑通 GRPO 训练 |
| PPO vs DPO vs GRPO | 适用场景、数据需求、稳定性对比 | 撰写对比表 |

**必读论文**

- *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL* (2025)
- *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (Rafailov et al., 2023)

---

### 第 12 周 · 垂域大模型 Chatbot 全流程实操

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| 数据处理 | 医疗文本清洗、格式化、质量评估（复用第 7 周数据工程） | 数据处理代码 |
| 中文 Tokenizer 扩展 | 在基座模型上扩展领域词汇（复用第 7 周） | 代码实现 |
| 二次预训练 | 领域数据 Continual Pre-training | 完成预训练 |
| SFT QLoRA 微调 | 领域指令微调 | 完成微调 |
| Reward Model 训练 | 训练领域 Reward Model | 完成训练 |
| RLHF-PPO 或 DPO | 完整对齐流程（二选一或对比） | 完成训练 |
| **产出** | 一个可用的垂域（如医疗）Chatbot | 覆盖 SFT → RM → RLHF/DPO 全流程 |

> **说明**：多卡训练（8B / 70B、DeepSpeed）统一放到第 13 周，避免本周过载。

---

### 第 13 周 · 多卡训练与 DeepSpeed 实操

| 主题 | 学习内容 | 产出要求 |
|:-----|:--------|:--------|
| 多卡实操 | 8B / 70B 模型 SFT / DPO / RM / PPO 多卡训练 | 按资源选做 8B 或 70B |
| DeepSpeed 并行 | ZeRO Stage 1 / 2 / 3 配置与使用；对比 FSDP | 配置 DeepSpeed 并跑通 |
| 多卡方案对比 | 8B vs 70B 各方案显存与通信差异 | 实验对比报告或笔记 |
| 📌 安全与评估（选学） | Constitutional AI / RLAIF / 红队测试；MT-Bench / MMLU / HumanEval 评测 | 选做：跑通一个基准评测 |

**阶段四产出**

- [ ] 完整垂域（如医疗）大模型 Chatbot（第 12 周）
- [ ] 覆盖 SFT → RM → RLHF / DPO 全流程
- [ ] 多卡训练经验：DeepSpeed / FSDP（第 13 周）

---

## 阶段五：系统优化与前沿（第 14～18 周）

> **核心思想**：从「能训练」到「能高效训练和推理」。推理量化、长上下文、MoE、部署是拉开差距的关键，本阶段以自主研读论文 + 代码实践为主。

### 第 14 周 · LLM 推理加速（深化）+ 量化 + 部署

| 主题 | 学习内容 | 深度要求 |
|:-----|:--------|:--------|
| ⚡ KV Cache / MQA / GQA（深化） | 与第 4 周衔接；Multi-Query / Grouped-Query 设计动机与显存-计算 trade-off | 能画图说明 GQA head 分组 |
| ⚡ FlashAttention 1/2 / FlashDecoding（深化） | Tiling 算法、在线 softmax、反向传播；与第 3 周入门衔接 | **能写伪代码** |
| PagedAttention / vLLM | PagedAttention 虚拟内存思想、vLLM scheduling 策略 | 部署 vLLM 推理服务 |
| StreamingLLM | 流式推理中 attention sink 现象及解决方案 | 阅读论文 |
| 📌 长上下文 | LongLoRA / Mistral Sliding Window / YaRN / NTK-aware RoPE 外推 | 阅读 1–2 篇论文并写笔记 |
| 📌 推理量化 | GPTQ / AWQ / GGUF 原理与使用场景（与 QLoRA 训练量化区分） | 用 GPTQ / AWQ 量化 7B 模型并推理 |
| 📌 部署与服务化 | TensorRT-LLM / SGLang；模型服务架构、批处理与排队 | 选做：部署推理服务 |
| Speculative Decoding | 投机采样：小模型 draft + 大模型 verify 加速策略 | 理解并实现或阅读代码 |

---

### 第 15 周 · 手撕分布式训练 + MoE

| 主题 | 学习内容 | 深度要求 |
|:-----|:--------|:--------|
| DP / 分布式 Adam | 数据并行基本原理、分布式优化器 | 代码实现 |
| ZeRO 1/2/3 | 三阶段分别切分什么、显存分析 | **能画出显存分析图** |
| TP（Tensor Parallelism） | 张量并行在 LLaMA 中的实现：行切分 / 列切分 | 手写 TP 逻辑 |
| PP（Pipeline Parallelism） | 流水线并行、DualPipe bubble 优化 | 理解 1F1B 调度 |
| CP（Context Parallelism） | Ring Attention 通信模式与实现 | 阅读论文 |
| EP（Expert Parallelism） | MoE 中 GShard 专家并行策略 | 阅读论文 |
| 📌 MoE 专题 | 混合专家架构（DeepSeek-V2/V3 / Mixtral）；routing、负载均衡、显存与计算 | 精读一篇 MoE 论文并写笔记 |
| 计算-通信重叠 | Overlap computation and communication 工程优化 | 理解 pipeline |

**必读论文**

- *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (Rajbhandari et al., 2020)
- *Megatron-LM* (Shoeybi et al., 2019)

---

### 第 16 周 · 手撕多模态 VLM

| 主题 | 学习内容 | 深度要求 |
|:-----|:--------|:--------|
| 手撕 ViT | Vision Transformer：patch embedding / position embedding / CLS token | **手写 ViT** |
| 手撕 CLIP Loss | 对比学习 InfoNCE Loss 在图文匹配中的应用 | **手写 CLIP Loss** |
| 手撕 CLIP Model | 完整 CLIP：Image Encoder + Text Encoder + 对比学习 | **手写 CLIP Model** |
| 手撕 LLaVA-1.5 | Visual Instruction Tuning 训练流程：Projector + LLM | **手写 LLaVA** |
| 📌 多模态扩展（选学） | 音频（Whisper）、视频理解（Video-LLaMA 等）简介 | 了解即可 |

**必读论文**

- *An Image is Worth 16x16 Words* (ViT, Dosovitskiy et al., 2020)
- *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, Radford et al., 2021)
- *Visual Instruction Tuning* (LLaVA, Liu et al., 2023)

---

### 第 17 周 · 手撕 o1 推理

| 主题 | 学习内容 | 深度要求 |
|:-----|:--------|:--------|
| 手撕 MC | Monte Carlo 方法在推理中的应用 | 代码实现 |
| 手撕 MCTS | Monte Carlo Tree Search：选择 → 扩展 → 模拟 → 回溯 | **手写 MCTS** |
| 手撕 AlphaGo-Zero | 自我对弈 + MCTS + 策略网络 | 理解核心思想 |
| 手撕 PRM | Process Reward Model：过程奖励 vs 结果奖励 | 理解 PRM 训练 |
| Scaling Test Time | Test-Time Compute Scaling：多路采样、验证、搜索 | 阅读前沿论文 |

**必读论文**

- *Mastering the Game of Go without Human Knowledge* (AlphaGo Zero, Silver et al., 2017)
- *Let's Verify Step by Step* (Lightman et al., 2023)
- *Scaling LLM Test-Time Compute* (Snell et al., 2024)

---

### 第 18 周 · 综合收尾（数据 / 合成数据 / 评估 / 安全）

| 主题 | 学习内容 | 深度要求 |
|:-----|:--------|:--------|
| 📌 数据工程（深化） | 数据配比（The Pile / FineWeb / RedPajama）；去重与质量 pipeline；数据量 vs 质量 / Scaling 与数据 | 阅读 1–2 份数据报告并整理笔记 |
| 📌 合成数据 | 用 LLM 生成训练数据（Phi / TinyLlama）；质量过滤、多样性控制；边界与风险 | 阅读 Phi 技术报告 |
| 📌 现代评估体系 | MMLU / HumanEval / MT-Bench / Arena Elo / BigBench；评估集设计 | 跑通 2–3 个基准并理解指标 |
| 📌 安全与对齐 | Constitutional AI / RLAIF / 红队与对抗样本；与 RLHF 的关系 | 阅读 1 篇综述或代表性论文 |
| Encoder-only 简介（选学） | BERT 与嵌入模型在检索、分类中的角色；与 Decoder-only 的分工 | 了解即可 |

> **阶段五小结**：第 14–18 周以自主研读论文与代码实践为主，每周设置了明确的学习目标与深度要求，便于自学与复盘。

---

## 进阶拓展路线图

以下是大模型学习中容易被忽略但至关重要的主题，已融入各周。下表便于快速查阅与自检：

| 拓展主题 | 重要性 | 本计划位置 | 自检 |
|:--------|:------:|:----------|:----:|
| 现代评估体系（MMLU / HumanEval / MT-Bench） | ⭐⭐⭐⭐ | W1 入门 → W13 / W18 深化 | ⬜ |
| 数据工程（配比 / 去重 / 质量） | ⭐⭐⭐⭐⭐ | W7 → W18 | ⬜ |
| Agent 扩展（Tool / Memory / Planning） | ⭐⭐⭐⭐ | W8 | ⬜ |
| DPO（与 PPO 并列的对齐方案） | ⭐⭐⭐⭐⭐ | W11 独立小节 + 手写 | ⬜ |
| 安全与对齐（Constitutional AI / RLAIF / 红队） | ⭐⭐⭐ | W13 选学 → W18 | ⬜ |
| 推理量化（GPTQ / AWQ / GGUF） | ⭐⭐⭐⭐ | W6 简介 → W14 系统讲解 | ⬜ |
| 长上下文（YaRN / NTK / Mistral） | ⭐⭐⭐ | W14 | ⬜ |
| 部署与服务化（vLLM / TensorRT-LLM / SGLang） | ⭐⭐⭐ | W14 | ⬜ |
| MoE（DeepSeek-V2 / Mixtral） | ⭐⭐⭐⭐⭐ | W15 专题 | ⬜ |
| 合成数据（Phi / 质量与边界） | ⭐⭐⭐⭐ | W18 | ⬜ |
| Encoder-only / BERT | ⭐⭐ | W18 选学 | ⬜ |

---

## 面试高频手撕代码清单

### Tier 1 · 必须能闭眼手写

| # | 内容 | 对应周 |
|:-:|:-----|:------|
| 1 | Multi-Head Attention | W2 |
| 2 | RoPE 旋转位置编码 | W4 |
| 3 | RMSNorm | W4 |
| 4 | KV Cache 推理 | W4 |
| 5 | LoRA Forward | W6 |
| 6 | PPO (Clip) Loss | W9 |
| 7 | RLHF-PPO 训练循环 | W10 |
| 8 | DPO Loss 与训练循环 | W11 |

### Tier 2 · 需要熟练手写

| # | 内容 | 对应周 |
|:-:|:-----|:------|
| 9 | BPE 编码 | W3 |
| 10 | Top-K / Top-P 采样 | W3 |
| 11 | SwiGLU | W4 |
| 12 | FlashAttention 伪代码 | W3 入门 → W14 深化 |
| 13 | GQA (Grouped Query Attention) | W4 → W14 |
| 14 | GRPO | W11 |

### Tier 3 · 需要理解原理、能说清思路

| # | 内容 | 对应周 |
|:-:|:-----|:------|
| 15 | ZeRO 1/2/3 显存分析 | W15 |
| 16 | Speculative Decoding | W14 |
| 17 | PagedAttention | W14 |
| 18 | MCTS | W17 |
| 19 | CLIP Loss | W16 |
| 20 | Reward Model 训练 | W10 |
| 21 | MoE routing 与负载均衡 | W15 |
| 22 | 现代评估（MMLU / HumanEval）流程 | W1 / W18 |

---

## 推荐学习资源

### 经典教材

| 资源 | 说明 |
|:-----|:-----|
| [《动手学深度学习》(d2l.ai)](https://d2l.ai) | 李沐 — Transformer / Attention 基础 |
| 《深度学习》(Goodfellow) | 数学基础、优化理论 |
| *Reinforcement Learning: An Introduction* (Sutton & Barto) | RL 经典教材 |

### 优质博客与教程

| 资源 | 说明 |
|:-----|:-----|
| [Lilian Weng 博客](https://lilianweng.github.io/) | OpenAI 研究员 — RLHF / PPO 系列文章极佳 |
| [Jay Alammar 博客](https://jalammar.github.io/) | Transformer 可视化讲解 |
| [HuggingFace Blog](https://huggingface.co/blog) | 各类技术教程 |
| [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT) | 最小 GPT 实现 |
| Sebastian Raschka LLM 系列 | 从零构建 LLM |

### 开源代码库

| 仓库 | 说明 |
|:-----|:-----|
| [`huggingface/transformers`](https://github.com/huggingface/transformers) | 工业级实现参考 |
| [`meta-llama/llama`](https://github.com/meta-llama/llama) | LLaMA 官方实现 |
| [`huggingface/trl`](https://github.com/huggingface/trl) | RLHF / DPO 训练框架 |
| [`microsoft/DeepSpeed`](https://github.com/microsoft/DeepSpeed) | 分布式训练框架 |
| [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | 推理加速引擎 |
| [`NVIDIA/TensorRT-LLM`](https://github.com/NVIDIA/TensorRT-LLM) | 高性能推理 |
| [`haotian-liu/LLaVA`](https://github.com/haotian-liu/LLaVA) | 多模态 LLM |

---

## 学习建议

### 1. 先「手撕」后「调包」

每个模块先从零实现一遍（哪怕是 naive 版本），再看 HuggingFace 等工业级实现。写过与看过的理解深度天差地别。

### 2. 论文阅读分层

- **精读**（逐行推导公式）：Attention Is All You Need · GPT-3 · LoRA · PPO · DPO · InstructGPT · DeepSeek-R1 · Mixtral
- **泛读**（抓住核心创新）：FlashAttention · vLLM · Megatron-LM · LLaVA · CLIP
- **扫读**（知道存在即可）：各种变体与改进工作

### 3. 建立代码库

把每周手写代码整理成 GitHub 仓库——既是学习笔记，也是面试 portfolio。

### 4. 以博客倒逼学习

费曼学习法：每完成一个模块写一篇技术博客。写不清楚，说明没真正理解。

### 5. 关注 Scaling 思维与 Research 导向

不要只停留在「能跑通」，持续追问：

- 这个方法为什么 work？（数学直觉）
- 它的 scaling behavior 是什么？
- 它的 limitation 在哪里？
- 未来改进方向？能否衍生一篇论文？

> 📌 **拓展意识**：学习时对照[进阶拓展路线图](#进阶拓展路线图)自检，避免只学主线而忽略数据 / 评估 / MoE / 安全等前沿必备知识。

### 6. 保持前沿嗅觉

大模型领域迭代极快，建议每周花 2–3 小时浏览：

- arXiv（cs.CL / cs.LG / cs.AI）
- Twitter / X 上的大模型研究者
- 各实验室技术博客（OpenAI · DeepMind · Anthropic · DeepSeek · Meta AI）

---

## 周进度追踪表

| 周次 | 阶段 | 核心内容 | 状态 |
|:----:|:----:|:--------|:----:|
| W1 | 基础筑基 | LLM 概述 / 现代 Tokenizer / 评估入门 | ⬜ |
| W2 | 基础筑基 | 手撕 Transformer | ⬜ |
| W3 | 基础筑基 | 手撕 GPT | ⬜ |
| W4 | 基础筑基 | 手撕 LLaMA | ⬜ |
| W5 | 高效微调 | 手撕 Alpaca（指令微调） | ⬜ |
| W6 | 高效微调 | 手撕 LoRA / 推理量化简介 | ⬜ |
| W7 | 高效微调 | 手撕 Chinese-LLaMA2 / 数据工程 | ⬜ |
| W8 | 应用落地 | Agent / RAG / Tool & Memory | ⬜ |
| W9 | RL 对齐 | 手撕 RL-PPO 基础 | ⬜ |
| W10 | RL 对齐 | 手撕 RLHF 原理与实现 | ⬜ |
| W11 | RL 对齐 | DPO / R1 / GRPO | ⬜ |
| W12 | RL 对齐 | 垂域 Chatbot 全流程 | ⬜ |
| W13 | RL 对齐 | 多卡训练 / DeepSpeed | ⬜ |
| W14 | 前沿 | 推理加速 + 量化 + 长上下文 + 部署 | ⬜ |
| W15 | 前沿 | 分布式训练 + MoE 专题 | ⬜ |
| W16 | 前沿 | 多模态 VLM | ⬜ |
| W17 | 前沿 | o1 推理 | ⬜ |
| W18 | 前沿 | 数据 / 合成数据 / 评估 / 安全 | ⬜ |

---

<details>
<summary><b>v2 更新摘要（点击展开）</b></summary>

1. 将 RLHF 对齐部分拆为第 10–13 周，减轻单周负荷
2. 增加进阶拓展模块：现代评估 · 数据工程 · DPO · MoE · 推理量化 · 长上下文 · 部署 · 合成数据 · 安全与对齐
3. 弱化 Word2Vec，强化现代 Tokenizer 与数据管线
4. 明确 FlashAttention / KV Cache 的「入门 → 深化」边界，避免重复
5. 新增进阶拓展路线图与 Research 导向学习建议
6. 预计周期 18～22 周，主线可压缩至 16 周

</details>

---

<p align="center"><em>大模型时代，信息差就是竞争力。<br/>当别人还在调包时，你已经能手撕每一行代码背后的数学，并覆盖数据、评估、MoE、安全等关键领域。<br/>这就是护城河。祝学有所成。</em></p>
