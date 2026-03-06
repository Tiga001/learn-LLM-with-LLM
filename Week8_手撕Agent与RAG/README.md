# 第 8 周：手撕 Agent 与 RAG

> **目标**：理解多轮对话系统的 history 管理、上下文窗口截断与对话模板；掌握 Agent 的核心范式 ReAct / Function Calling / Tool Use，并从零实现一个带工具调用与 Memory 管理的 Agent；理解 RAG（Retrieval-Augmented Generation）的完整流程：文档切分 → 向量化 → 索引 → 检索 → 生成；掌握文本嵌入模型、双塔架构与 InfoNCE 对比学习损失；从零手写 RAG 管线，并搭建一个带知识库的多轮对话 Agent。

---

## 每日安排（建议每日 3~5 小时）

| 天 | 主题 | 核心任务 | 产出 | 难度 |
|----|------|---------|------|:----:|
| Day 1 | 多轮对话系统管理与原理 | History 管理、上下文截断、对话模板、本地部署 | 对话系统架构笔记 + 部署方案 | ⭐⭐⭐ |
| Day 2 | Agent 原理 — ReAct 与 Function Calling | ReAct 范式、Tool Use、Function Calling 实现机制 | Agent 范式分析笔记 | ⭐⭐⭐⭐ |
| Day 3 | **手写 Agent 实现** | 从零实现 ReAct Agent、Tool 注册与调用、Memory 管理 | 手写 Agent 系统 | ⭐⭐⭐⭐⭐ |
| Day 4 | RAG 算法原理 | 检索增强生成完整流程、向量数据库、检索策略 | RAG 架构笔记 + 流程图 | ⭐⭐⭐⭐ |
| Day 5 | 文本嵌入与对比学习 | Embedding 模型、InfoNCE 损失、双塔架构 | 对比学习数学推导 | ⭐⭐⭐⭐ |
| Day 6 | **手写 RAG 系统实践** | 从零构建 RAG 管线、LangChain RAG、端到端评估 | 手写 RAG + LangChain 应用 | ⭐⭐⭐⭐⭐ |
| Day 7 | 综合项目：带 RAG 的 Agent + 复盘 | 整合 Agent + RAG + 多轮对话，串联全周知识 | 综合系统 + 自检 | ⭐⭐⭐⭐⭐ |

---

## 知识图谱

```
Day 1: 多轮对话系统管理与原理 — 对话系统的基础设施
  LLM 无状态本质 → History 管理 → 上下文截断 → 对话模板 → 本地部署
       │
       ▼
Day 2: Agent 原理 — 让 LLM 从“回答”走向“行动”
  ReAct → Tool Use → Function Calling → Agent Loop
       │
       ▼
Day 3: 手写 Agent 实现（本周重要实践！）
  Tool 注册 → Thought / Action / Observation 循环 → 短期/长期 Memory
       │
       ▼
Day 4: RAG 算法原理 — 让 LLM 获得外部知识
  文档切分 → 向量化 → 向量索引 → 召回 → 重排 → 增强生成
       │
       ▼
Day 5: 文本嵌入与对比学习 — RAG 检索效果的数学基础
  Sentence Embedding → 双塔架构 → 相似度函数 → InfoNCE
       │
       ▼
Day 6: 手写 RAG 系统实践（本周核心实验！）
  Naive RAG → 混合检索 → 重排 → 评估 → LangChain 封装
       │
       ▼
Day 7: 综合项目与复盘
  多轮对话 + Agent + RAG → 可落地的知识库助手
  → 第 9 周对齐与推理强化衔接
```

---

## 与前序周次的关系

| 本周内容 | 前序基础 | 后续衔接 |
|---------|---------|---------|
| 多轮对话系统管理 | W4 LLaMA 推理流程、W5 对话模板/SFT 格式 | W12 垂域 Chatbot |
| Agent / ReAct | W5 Day4 CoT 推理策略 | W17 推理强化 / MCTS |
| Function Calling / Tool Use | W5 指令数据格式设计 | W12 垂域 Agent |
| RAG 检索增强 | W4 上下文管理、W6 工程实践 | W12 垂域 RAG 问答 |
| 文本嵌入 / 对比学习 | W1 Embedding 层、W4 Transformer 编码 | W16 多模态对比学习 |
| LangChain / 系统封装 | W6-7 微调工程实践 | W14 服务化与部署 |

---

## 文件结构

```
Week8_手撕Agent与RAG/
├── README.md                            ← 你在这里
├── Day1_多轮对话系统管理.md              ← History 管理、截断策略、模板、部署
├── Day2_Agent原理与ReAct.md             ← ReAct / Function Calling / Tool Use
├── Day3_手写Agent实现.ipynb             ← 手写 Agent：Tool Registry + ReAct Loop + Memory (实践!)
├── Day4_RAG算法原理.md                  ← 检索增强生成完整流程
├── Day5_文本嵌入与对比学习.md            ← Embedding / InfoNCE / 双塔架构
├── Day6_手写RAG系统实践.ipynb           ← 手写 RAG：BM25 + Dense + 混合检索 + 评估 (核心!)
├── Day7_综合项目与复盘.md               ← Agent+RAG 综合 + 周复盘
└── Day7_综合项目实践.ipynb              ← 整合 Agent+RAG+多轮对话的知识库助手 (综合!)
```

---

## 学习建议

推荐按下面的节奏学习：

1. 先完成 Day1 和 Day2，建立“对话上下文怎么维护”和“Agent 为什么能调用工具”的整体认知。
2. 再进入 Day3，把 ReAct Agent 的最小闭环亲手写出来，重点吃透 Tool 注册、状态循环和异常处理。
3. Day4 和 Day5 负责补齐 RAG 的系统视角与数学基础，尤其要把“检索为什么有效”讲明白。
4. Day6 把 RAG 从原理落到可运行代码，最后在 Day7 做系统整合、自检和复盘。

---

## 关键检查点 ✅

完成本周学习后，你应该能够：

- [ ] 解释 LLM 无状态的本质，说明多轮对话为何需要外部维护 history
- [ ] 对比滑动窗口、Token 级截断、摘要压缩三种上下文管理策略
- [ ] 理解 ReAct 范式中的 Thought → Action → Observation 循环
- [ ] **手写一个带 Tool Use 的 ReAct Agent**
- [ ] 解释 Function Calling 的 Schema 设计、参数解析与工具路由机制
- [ ] 理解 Agent Memory 的设计：短期（history）vs 长期（向量数据库/外部存储）
- [ ] 画出 RAG 的完整流程图：切分 → 向量化 → 索引 → 召回 → 重排 → 生成
- [ ] 解释文本嵌入模型的双塔架构与 InfoNCE 对比学习损失
- [ ] **从零手写 RAG 管线（切分 → 嵌入 → 检索 → 生成）**
- [ ] 使用 LangChain 构建一个可复用的 RAG 应用封装
- [ ] **实现一个带 RAG 的多轮对话 Agent 原型**

---

## 本周必读论文

1. **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2023) — **精读**
2. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020) — **精读**

## 参考论文

- *Toolformer: Language Models Can Teach Themselves to Use Tools* (Schick et al., 2023)
- *Gorilla: Large Language Model Connected with Massive APIs* (Patil et al., 2023)
- *Dense Passage Retrieval for Open-Domain Question Answering* (Karpukhin et al., 2020)
- *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* (Reimers & Gurevych, 2019)
- *BGE M3-Embedding: Unified Fine-Tuning for Dense Retrieval* (BAAI, 2024)
- *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* (Asai et al., 2023)
- *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* (Sarthi et al., 2024)

## 推荐资源

- LangChain: [官方文档](https://python.langchain.com/) / [GitHub 仓库](https://github.com/langchain-ai/langchain)
- LlamaIndex: [官方文档](https://docs.llamaindex.ai/) / [GitHub 仓库](https://github.com/run-llama/llama_index)
- ChromaDB: [官方文档](https://docs.trychroma.com/) / [GitHub 仓库](https://github.com/chroma-core/chroma)
- FAISS: [GitHub 仓库](https://github.com/facebookresearch/faiss)
- Lilian Weng: [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- Lilian Weng: [Retrieval-Augmented Generation](https://lilianweng.github.io/posts/2024-07-07-rag/)
- Sentence-Transformers: [文档](https://www.sbert.net/) / [GitHub](https://github.com/UKPLab/sentence-transformers)

---

## 本周最终产出

完成本周后，建议至少形成以下可复用资产：

- 一份多轮对话系统管理笔记：覆盖 history、截断策略、模板和部署架构
- 一个最小 ReAct Agent：支持工具注册、调用与简单 Memory
- 一份 RAG 架构图：解释离线索引和在线检索生成链路
- 一个最小 RAG 系统：支持切分、向量化、召回与回答生成
- 一个 Agent + RAG 综合项目原型：支持多轮问答、知识检索与工具调用
