# 第 15 周：手撕分布式训练 + MoE

> **目标**：系统掌握大模型分布式训练的五大并行维度——数据并行（DP/DDP/ZeRO）、张量并行（TP）、流水线并行（PP）、上下文并行（CP）与专家并行（EP）；从零手写 AllReduce、ZeRO 分片、TP 列/行切分等核心组件；深入 MoE（Mixture of Experts）混合专家架构，理解路由机制、负载均衡与 Mixtral / DeepSeek-V2/V3 的设计；精读 ZeRO 与 Megatron-LM 两篇经典论文。

---

## 每日安排（建议每日 3~5 小时）

| 天 | 主题 | 核心任务 | 产出 | 难度 |
|----|------|---------|------|:----:|
| Day 1 | 分布式训练概览与数据并行 | DP/DDP 原理、AllReduce 通信、分布式优化器、AMP | 并行全景图 + 通信分析笔记 | ⭐⭐⭐ |
| Day 2 | ZeRO 显存优化详解 | ZeRO Stage 1/2/3 显存公式推导、DeepSpeed 配置、FSDP 对比 | **显存分析图（能画出三阶段对比！）** | ⭐⭐⭐⭐⭐ |
| Day 3 | **手写分布式训练核心组件** | 从零实现 AllReduce → DP → ZeRO Stage 1/2 → TP 列/行切分 | 可运行的分布式组件代码 | ⭐⭐⭐⭐⭐ |
| Day 4 | 张量并行与流水线并行 | Megatron TP 实现、PP 调度（1F1B / DualPipe）、CP / EP 原理 | TP 数据流图 + PP 调度图 | ⭐⭐⭐⭐ |
| Day 5 | MoE 混合专家架构 | Router / Gate / Expert 设计、负载均衡、Mixtral / DeepSeek-V2/V3 | MoE 架构笔记 + 论文解读 | ⭐⭐⭐⭐⭐ |
| Day 6 | **手写 MoE 与分布式训练实践** | 从零实现 Router → Expert → MoE Layer → 训练循环 + EP 模拟 | 可运行的 MoE 模型代码 | ⭐⭐⭐⭐⭐ |
| Day 7 | Megatron 与全并行策略复盘 | 精读 Megatron-LM / ZeRO 论文；全周知识串联；计算-通信重叠 | 论文笔记 + 全周自检 | ⭐⭐⭐⭐ |

---

## 知识图谱

```
Day 1: 分布式训练概览与数据并行
  单卡瓶颈 → 五大并行维度全景 → DP/DDP 原理 → AllReduce → AMP
       │
       ▼
Day 2: ZeRO 显存优化详解 — 从「放不下」到「切了就放得下」
  显存四大组成 → ZeRO-1 (OS 分片) → ZeRO-2 (+梯度) → ZeRO-3 (+参数)
  → Offload / Infinity → DeepSpeed 配置 → FSDP 对比
       │
       ▼
Day 3: 手写分布式训练核心组件（本周核心实践 I！）
  AllReduce 模拟 → DP 训练循环 → ZeRO-1/2 实现 → TP 列/行切分
       │
       ▼
Day 4: 张量并行与流水线并行 — 模型切分的两个维度
  TP (层内切分) → Megatron MLP/Attention TP → PP (层间切分)
  → 1F1B / DualPipe 调度 → CP (Ring Attention) → EP (AllToAll)
  → 3D/4D/5D 并行组合
       │
       ▼
Day 5: MoE 混合专家架构 — 用稀疏换规模
  稀疏 vs 稠密 → Router / Expert / Gate → 负载均衡 → Mixtral
  → DeepSeek-V2/V3 → 训练挑战 → 推理优化
       │                                           │
       ▼                                           ▼
Day 6: 手写 MoE 与分布式训练实践（本周核心实践 II！）   → W16 多模态
  Router → Expert → MoE Layer → MoE Block → EP 模拟 → 训练循环
       │
       ▼
Day 7: Megatron 与全并行策略复盘
  Megatron-LM 精读 → ZeRO 精读 → 计算-通信重叠 → 全周串联
```

---

## 与前序周次的关系

| 本周内容 | 前序基础 | 后续衔接 |
|---------|---------|---------|
| 数据并行 DP/DDP | W13 DeepSpeed 多卡实操 | — |
| ZeRO 1/2/3 | W13 DeepSpeed 配置与使用 | — |
| 张量并行 TP | W4 LLaMA 架构（Attention + FFN） | — |
| 流水线并行 PP | W4 LLaMA Block 堆叠 | — |
| Context Parallelism | W14 FlashAttention / Ring Attention | — |
| Expert Parallelism | 本周 MoE 架构 | — |
| MoE 混合专家 | W4 FFN 架构、W14 推理优化 | W16 多模态 VLM |
| AllReduce 通信 | W13 多卡训练基础 | — |

---

## 文件结构

```
Week15_手撕分布式训练与MoE/
├── README.md                              ← 你在这里
├── Day1_分布式训练概览与数据并行.md         ← DP/DDP/AllReduce/AMP 原理
├── Day2_ZeRO显存优化详解.md                ← ZeRO 1/2/3 显存分析 (面试重点!)
├── Day3_手写分布式训练核心组件.ipynb        ← 手写 AllReduce + ZeRO + TP (实践!)
├── Day4_张量并行与流水线并行.md             ← TP/PP/CP/EP + 3D 并行
├── Day5_MoE混合专家架构.md                 ← MoE 路由 + Mixtral + DeepSeek
├── Day6_手写MoE与分布式训练实践.ipynb       ← 手写 MoE Router/Expert/Layer (实践!)
└── Day7_Megatron与全并行策略复盘.md         ← Megatron/ZeRO 精读 + 周复盘
```

---

## 关键检查点 ✅

完成本周学习后，你应该能够：

- [ ] 画出分布式训练五大并行维度的全景图（DP / TP / PP / CP / EP）
- [ ] 写出 Ring AllReduce 的通信步骤与通信量公式
- [ ] **精确推导 ZeRO Stage 1/2/3 各阶段的显存占用公式**
- [ ] **画出 ZeRO 三阶段显存对比图（面试高频！）**
- [ ] 解释 DeepSpeed ZeRO 与 PyTorch FSDP 的异同
- [ ] **手写 Tensor Parallelism 的列切分和行切分 Linear**
- [ ] 画出 Megatron-LM 中 MLP 和 Attention 的 TP 数据流
- [ ] 解释 1F1B Pipeline 调度及 bubble ratio 公式
- [ ] 说明 Ring Attention 的通信模式与 Context Parallelism 原理
- [ ] **手写 MoE Router（Top-K Gating）和负载均衡 Loss**
- [ ] 解释 Mixtral 8x7B 的架构设计——为什么只替换 FFN
- [ ] 说明 DeepSeek-V2/V3 的细粒度专家与共享专家设计
- [ ] 区分 MoE 的总参数量与激活参数量
- [ ] 理解计算-通信重叠（Overlap）的工程优化思路
- [ ] 精读 Megatron-LM 和 ZeRO 论文的核心贡献

---

## 本周必读论文

1. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al., 2020) — **精读**
2. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** (Shoeybi et al., 2019) — **精读**

## 参考论文

- *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (Narayanan et al., 2021) — 3D 并行
- *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism* (Huang et al., 2019)
- *PipeDream: Generalized Pipeline Parallelism for DNN Training* (Narayanan et al., 2019) — 1F1B
- *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (DeepSeek-AI, 2024)
- *DeepSeek-V3 Technical Report* (DeepSeek-AI, 2024) — DualPipe
- *Mixtral of Experts* (Jiang et al., 2024)
- *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* (Fedus et al., 2022)
- *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* (Lepikhin et al., 2021)
- *Ring Attention with Blockwise Transformers for Near-Infinite Context* (Liu et al., 2023)
- *ZeRO-Offload: Democratizing Billion-Scale Model Training* (Ren et al., 2021)
- *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning* (Rajbhandari et al., 2021)

## 推荐资源

- DeepSpeed: [GitHub 仓库](https://github.com/microsoft/DeepSpeed) / [官方文档](https://www.deepspeed.ai/)
- Megatron-LM: [GitHub 仓库](https://github.com/NVIDIA/Megatron-LM)
- PyTorch FSDP: [官方教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- Lilian Weng: [How to Train Really Large Models on Many GPUs](https://lilianweng.github.io/posts/2021-09-25-train-large/)
- HuggingFace: [Model Parallelism](https://huggingface.co/docs/transformers/v4.15.0/parallelism)
- DeepSeek-V3: [GitHub 仓库](https://github.com/deepseek-ai/DeepSeek-V3)
- Mixtral: [HuggingFace 模型](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
