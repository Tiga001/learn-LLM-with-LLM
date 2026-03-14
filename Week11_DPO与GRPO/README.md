# 第 11 周：DPO / R1 / GRPO（对齐方案扩展）

> **目标**：在 Week 10 RLHF-PPO 基础上，系统掌握当前工业界最主流的对齐替代方案——精读 DPO 论文，从 RLHF 优化目标出发严格推导 DPO Loss，理解「隐式 Reward Model」的数学本质，从零手写 DPO 训练循环（面试 Tier 1）；精读 DeepSeek-R1 技术报告，理解纯 RL 训练带来的推理涌现现象与完整训练管线；掌握 GRPO（Group Relative Policy Optimization）的数学原理——用组内相对排名替代 Critic 网络，从零手写 GRPO 训练循环（面试 Tier 2）；最终系统对比 PPO / DPO / GRPO 三种对齐方案的适用场景、数据需求与工程权衡，为 Week 12 垂域 Chatbot 全流程实操做好技术储备。

---

## 每日安排（建议每日 3~5 小时）

| 天 | 主题 | 核心任务 | 产出 | 难度 |
|----|------|---------|------|:----:|
| Day 1 | DPO 论文精读与原理 | DPO 动机、论文精读、核心思想「隐式 RM」、与 RLHF 架构对比 | DPO 论文精读笔记 + 架构对比图 | ⭐⭐⭐⭐ |
| Day 2 | DPO Loss 完整推导 | 从 RLHF 目标 → 闭式解 → 反解 reward → 代入 BT → DPO Loss → 梯度分析 | **手写 DPO Loss 推导**（面试高频！） | ⭐⭐⭐⭐⭐ |
| Day 3 | **手写 DPO 训练** | 从零实现 DPO：偏好数据集 + DPO Loss + 训练循环 + 隐式 Reward 提取 | **DPO 训练成功 + 生成质量对比** | ⭐⭐⭐⭐⭐ |
| Day 4 | DeepSeek-R1 论文精读 | R1-Zero 纯 RL 探索、R1 完整训练管线、GRPO 细节、推理涌现、数据蒸馏 | R1 论文精读笔记 + 训练流程图 | ⭐⭐⭐⭐ |
| Day 5 | GRPO 算法原理与推导 | 去 Critic 动机、组内采样 baseline、GRPO Loss 推导、与 PPO/DPO 对比 | **手写 GRPO Loss 推导** | ⭐⭐⭐⭐⭐ |
| Day 6 | **手写 GRPO 训练** | 从零实现 GRPO：多组采样 + 组内 Advantage + GRPO Loss + 训练循环 | **GRPO 训练成功 + 效果可视化** | ⭐⭐⭐⭐⭐ |
| Day 7 | PPO vs DPO vs GRPO 对比与复盘 | 三方案多维度对比、工业选型指南、其他方案简介、全周复盘 | 对比表 + 公式速查表 + 全周自检 | ⭐⭐⭐⭐ |

---

## 知识图谱

```
Day 1: DPO 论文精读与原理 — RLHF 的更简单替代
  RLHF 痛点回顾 → DPO 核心思想 → 隐式 RM → 架构简化(4→2 模型) → 优势与局限
       │
       ▼
Day 2: DPO Loss 完整推导（本周理论核心！面试 Tier 1！）
  RLHF 目标 → 闭式解 π* → 反解 r(x,y) → 代入 BT → Z(x) 消除 → DPO Loss → 梯度分析
       │
       ▼
Day 3: 手写 DPO 训练（本周第一个实践！）
  偏好数据集 → GPT-2 Policy/Ref → DPO Loss 实现 → 训练循环 → 隐式 Reward → 生成对比
       │
       ▼
Day 4: DeepSeek-R1 论文精读 — 纯 RL 训练的推理涌现
  R1-Zero 纯 RL → R1 完整管线 → GRPO 在 R1 中的应用 → 推理涌现 → 数据蒸馏
       │
       ▼
Day 5: GRPO 算法原理与推导（本周第二个理论核心！）
  PPO Critic 问题 → 组内采样 → GRPO 目标函数 → Loss 推导 → 与 PPO/DPO 对比
       │
       ▼
Day 6: 手写 GRPO 训练（本周第二个实践！面试 Tier 2！）
  Reward 函数 → 多组采样 → 组内归一化 Advantage → GRPO Loss → 训练循环 → 可视化
       │
       ▼
Day 7: PPO vs DPO vs GRPO 对比与复盘
  三方案对比 → 工业选型 → IPO/KTO/SimPO 简介 → 公式速查 → 面试自检
  → 第 12 周垂域 Chatbot 铺路
```

---

## 与前序周次的关系

| 本周内容 | 前序基础 | 后续衔接 |
|---------|---------|---------|
| DPO 动机 | W10 Day7 RLHF 局限性分析 | W12 垂域 Chatbot 选用 DPO/PPO |
| DPO Loss 推导 | W10 Day1 RLHF 优化目标闭式解 | W12 DPO 训练实操 |
| DPO 隐式 RM | W10 Day2-3 Bradley-Terry 模型与 RM 训练 | — |
| DPO 训练循环 | W10 Day6 RLHF-PPO 训练循环 | W12 全流程对比 |
| DeepSeek-R1 论文 | W10 RLHF 完整链路 | W17 o1 推理（MCTS / PRM） |
| GRPO 原理 | W9 Day4-5 策略梯度 / GAE | W12 GRPO 作为替代方案 |
| GRPO 去 Critic | W9 Day5 Actor-Critic 框架 | W13 多卡训练中的显存优势 |
| PPO vs DPO vs GRPO | W9 PPO + W10 RLHF 全部知识 | W12-13 方案选型 |

---

## 文件结构

```
Week11_DPO与GRPO/
├── README.md                              ← 你在这里
├── Day1_DPO论文精读与原理.md                ← DPO 动机、论文精读、隐式 RM、架构对比
├── Day2_DPO_Loss完整推导.md                ← 闭式解 → 反解 reward → DPO Loss (理论核心!)
├── Day3_手写DPO训练.ipynb                  ← 从零实现 DPO 训练循环 (面试 Tier 1!)
├── Day4_DeepSeek-R1论文精读.md              ← R1-Zero / R1 训练管线 / 推理涌现 / 蒸馏
├── Day5_GRPO算法原理与推导.md               ← 去 Critic / 组内采样 / GRPO Loss 推导
├── Day6_手写GRPO训练.ipynb                 ← 从零实现 GRPO 训练 (面试 Tier 2!)
└── Day7_PPO_vs_DPO_vs_GRPO对比与复盘.md    ← 三方案对比、公式速查、全周复盘
```

---

## 学习建议

推荐按下面的节奏学习：

1. Day 1 从 W10 Day7 的 DPO 预览出发，系统精读 DPO 论文。核心是理解「为什么可以绕过 RM」的直觉——策略本身就隐含了 reward 信息。
2. Day 2 是本周的数学核心，DPO Loss 的完整推导需要逐步跟着写。每一步都要理解为什么成立，特别是配分函数 $Z(x)$ 为什么能消掉——这是 DPO 成立的关键。
3. Day 3 亲手实现 DPO 训练。相比 W10 Day6 的 RLHF-PPO，DPO 的代码简洁得多（不需要 rollout、不需要 Critic、不需要 GAE），但效果不一定差。体会这种简洁性。
4. Day 4 精读 DeepSeek-R1，重点关注「纯 RL 训练为什么能产生推理能力」这个核心问题，以及 GRPO 在其中的角色。
5. Day 5 推导 GRPO，核心思想是「用组内相对排名替代 Critic 的绝对估值」。理解了这个思想，公式就是自然的。
6. Day 6 实现 GRPO 训练。与 DPO 不同，GRPO 是 on-policy 方法，需要实时采样和打分，更接近 PPO 但比 PPO 简单。
7. Day 7 做三方案的系统对比——这是面试中经常被问到的，需要能从多个维度（数据、计算、稳定性、效果）清晰地对比三种方案。

---

## 关键检查点 ✅

完成本周学习后，你应该能够：

### Tier 1（面试必须能闭眼手写）

- [ ] **手写 DPO Loss**：$L_{\text{DPO}} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})]$
- [ ] **手写 DPO 训练循环**：forward chosen/rejected → 4 组 log_probs → DPO Loss → backward
- [ ] 完整推导 DPO Loss：从 RLHF 目标 → 闭式解 → 反解 reward → 代入 BT → DPO Loss
- [ ] 解释 DPO 中「隐式 Reward Model」的含义：$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$
- [ ] 解释配分函数 $Z(x)$ 为什么在 DPO 推导中被消掉

### Tier 2（面试加分，需熟练手写）

- [ ] **手写 GRPO Loss 与训练循环**
- [ ] 解释 GRPO 如何用组内平均 reward 替代 Critic 网络
- [ ] 写出 GRPO 的组内归一化 Advantage：$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$
- [ ] 对比 PPO / DPO / GRPO 的数据需求、计算成本、训练稳定性
- [ ] 解释 DeepSeek-R1 的训练流程（cold start → GRPO → rejection sampling → SFT → RL）

### Tier 3（深入理解，能说清思路）

- [ ] 分析 DPO 梯度中 chosen/rejected 的隐式权重
- [ ] 解释 R1-Zero 中「纯 RL 训练产生推理涌现」的现象
- [ ] 对比 on-policy（PPO/GRPO）与 off-policy（DPO）的本质差异
- [ ] 了解 IPO、KTO、SimPO、ORPO 等 DPO 变体的核心改进
- [ ] 讨论 β 在 DPO 和 GRPO 中的作用差异

---

## 本周必读论文

1. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (Rafailov et al., 2023) — **精读**
2. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (DeepSeek-AI, 2025) — **精读**

## 参考论文

- *Training language models to follow instructions with human feedback* (Ouyang et al., 2022) — InstructGPT（W10 精读）
- *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (Shao et al., 2024) — GRPO 首次提出
- *A General Theoretical Paradigm to Understand Learning from Human Feedback* (Azar et al., 2023) — IPO
- *KTO: Model Alignment as Prospect Theoretic Optimization* (Ethayarajh et al., 2024) — KTO
- *SimPO: Simple Preference Optimization with a Reference-Free Reward* (Meng et al., 2024) — SimPO
- *ORPO: Monolithic Preference Optimization without Reference Model* (Hong et al., 2024) — ORPO
- *Proximal Policy Optimization Algorithms* (Schulman et al., 2017) — PPO（W9 精读）

## 推荐资源

- Hugging Face Blog: [DPO — A Simpler Alternative to RLHF](https://huggingface.co/blog/dpo-trl) — DPO 实践教程
- Hugging Face TRL: [DPOTrainer 文档](https://huggingface.co/docs/trl/dpo_trainer) — 工业级 DPO 实现
- Lilian Weng: [LLM Alignment](https://lilianweng.github.io/posts/2023-03-15-llm-alignment/) — 对齐方法综述
- Nathan Lambert: [RLHF Book](https://rlhfbook.com/) — RLHF / DPO 系统性参考
- DeepSeek-R1 技术报告: [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) — 官方代码与报告
- Chip Huyen: [RLHF 综述](https://huyenchip.com/2023/05/02/rlhf.html) — 含 DPO 对比
- Sebastian Raschka: [DPO 讲解](https://sebastianraschka.com/blog/2024/dpo.html) — DPO 代码级讲解

---

## 本周最终产出

完成本周后，建议至少形成以下可复用资产：

- 一份 DPO 论文精读笔记：覆盖动机、核心思想、隐式 RM、与 RLHF 的对比
- 一份 DPO Loss 的完整推导：从 RLHF 目标到最终 Loss，含梯度分析
- 一个从零手写的 DPO 训练循环：在 GPT-2 上完成 DPO 训练，含隐式 Reward 可视化
- 一份 DeepSeek-R1 论文精读笔记：覆盖 R1-Zero、完整训练管线、推理涌现、数据蒸馏
- 一份 GRPO 算法推导笔记：从策略梯度到 GRPO Loss，含与 PPO/DPO 的数学对比
- 一个从零手写的 GRPO 训练循环：多组采样 + 组内 Advantage + 训练成功
- 一份 PPO vs DPO vs GRPO 的系统对比表：覆盖至少 8 个维度
- 一份核心公式速查表：DPO Loss、GRPO Loss、隐式 Reward 等面试必背公式
