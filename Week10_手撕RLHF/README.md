# 第 10 周：手撕 RLHF（原理与实现）

> **目标**：在 Week 9 PPO 基础上，完整掌握 RLHF（Reinforcement Learning from Human Feedback）的三阶段流程——精读 InstructGPT 和 LLaMA2 论文中 RLHF 的核心思想；深入理解 Reward Model 的 Bradley-Terry 建模与训练；从零手写 Reward Model；系统解析 RLHF-PPO 的四模型架构（Actor / Critic / Reference / Reward）与参数共享策略；完整推导 RLHF-PPO 的三部分 Loss（Policy Loss + Value Loss + KL Penalty）；最终从零手写 RLHF-PPO 训练循环，在 GPT-2 small 上完成一次小规模 RLHF 训练实验。RLHF-PPO 训练循环是面试 Tier 1 必须能闭眼手写的内容。

---

## 每日安排（建议每日 3~5 小时）

| 天 | 主题 | 核心任务 | 产出 | 难度 |
|----|------|---------|------|:----:|
| Day 1 | InstructGPT 论文精读与 RLHF 概述 | RLHF 三阶段流程、InstructGPT / LLaMA2 论文解读、优化目标推导 | RLHF 三阶段流程图 + 论文笔记 | ⭐⭐⭐ |
| Day 2 | Reward Model 原理与训练 | Bradley-Terry 模型推导、RM Loss 推导、偏好数据建模 | RM Loss 推导笔记 + 架构图 | ⭐⭐⭐⭐ |
| Day 3 | **手写 Reward Model** | 从零实现 RM：偏好数据构建 + GPT2 backbone + BT Loss + 训练 | **RM 训练成功 + 偏好预测准确率 >80%** | ⭐⭐⭐⭐ |
| Day 4 | RLHF-PPO 架构详解 | 四模型架构、参数共享策略、rollout 全流程、显存分析 | 四模型数据流图 + 显存分析 | ⭐⭐⭐⭐⭐ |
| Day 5 | RLHF-PPO Loss 完整推导 | Policy Loss + Value Loss + KL Penalty 推导、GAE 适配、完整伪代码 | **手写完整 RLHF-PPO Loss** | ⭐⭐⭐⭐⭐ |
| Day 6 | **手写 RLHF 训练** | RLHF-PPO 完整训练循环：rollout → reward → GAE → PPO 更新 | **GPT-2 上 RLHF 训练成功** | ⭐⭐⭐⭐⭐ |
| Day 7 | RLHF 实践分析与复盘 | Reward Hacking 分析、工程挑战、DPO 预览、全周复盘 | 对比笔记 + 全周自检 | ⭐⭐⭐⭐ |

---

## 知识图谱

```
Day 1: InstructGPT 论文精读与 RLHF 概述 — 从 SFT 到对齐
  SFT 局限性 → RLHF 三阶段 → InstructGPT 论文 → 优化目标 max E[RM] - β·KL
       │
       ▼
Day 2: Reward Model 原理与训练 — 让模型理解「什么是好回答」
  偏好数据 → Bradley-Terry 模型 → RM Loss 推导 → RM 架构 → Reward Hacking
       │
       ▼
Day 3: 手写 Reward Model（第一个实践！）
  偏好数据集 → GPT2 + Reward Head → BT Loss → 训练循环 → 偏好预测评估
       │
       ▼
Day 4: RLHF-PPO 架构详解 — 四模型协同的工程挑战
  Actor / Critic / Reference / Reward → 参数共享 → rollout 流程 → 显存分析
       │
       ▼
Day 5: RLHF-PPO Loss 完整推导（本周理论核心！）
  Policy Loss (PPO-Clip) → Value Loss → KL Penalty → GAE 适配 → 完整伪代码
       │
       ▼
Day 6: 手写 RLHF 训练（本周最重要！面试 Tier 1！）
  四模型初始化 → rollout 生成 → RM 打分 → KL 计算 → GAE → PPO 更新 → 生成对比
       │
       ▼
Day 7: RLHF 实践分析与复盘
  Reward Hacking → 工程挑战 → RLHF 局限性 → DPO 预览 → 全周知识串联
  → 第 11 周 DPO / GRPO 铺路
```

---

## 与前序周次的关系

| 本周内容 | 前序基础 | 后续衔接 |
|---------|---------|---------|
| RLHF 三阶段 | W5 SFT 指令微调 | W12 垂域 Chatbot 全流程 |
| Reward Model | W9 DQN 中的价值估计思想 | W11 DPO 绕过 RM 的动机 |
| Bradley-Terry 模型 | 概率论基础（sigmoid / 最大似然） | W11 DPO Loss 推导 |
| RLHF-PPO 四模型 | W9 Actor-Critic 框架 | W13 多卡训练中的模型部署 |
| Policy Loss (PPO-Clip) | W9 PPO-Clip 实现（直接复用） | W11 DPO 替代 PPO |
| KL Penalty | W9 Day7 PPO-Penalty 介绍 | W11 DPO 的隐式 KL 约束 |
| GAE 在 RLHF 中的适配 | W9 GAE 推导与实现 | — |
| Reward Hacking | 本周 RM 训练 | W11 DPO 的优势之一 |

---

## 文件结构

```
Week10_手撕RLHF/
├── README.md                                ← 你在这里
├── Day1_InstructGPT论文精读与RLHF概述.md     ← RLHF 三阶段、论文解读、优化目标
├── Day2_Reward_Model原理与训练.md            ← Bradley-Terry 模型、RM Loss 推导
├── Day3_手写Reward_Model.ipynb              ← 从零实现 RM + 偏好预测 (实践!)
├── Day4_RLHF-PPO架构详解.md                 ← 四模型架构、参数共享、显存分析
├── Day5_RLHF-PPO_Loss完整推导.md            ← Policy/Value/KL Loss 推导 (理论核心!)
├── Day6_手写RLHF训练.ipynb                  ← RLHF-PPO 完整训练循环 (最重要! 面试 Tier 1!)
└── Day7_RLHF实践分析与复盘.md               ← Reward Hacking、DPO 预览、全周复盘
```

---

## 学习建议

推荐按下面的节奏学习：

1. Day 1 从宏观视角理解 RLHF 的动机和三阶段流程。InstructGPT 论文是 RLHF 的奠基之作，务必精读。理解「为什么 SFT 不够」是后续一切的起点。
2. Day 2 深入 Reward Model 的数学原理。Bradley-Terry 模型的推导并不复杂，但理解它是理解整个 RLHF Loss 链条的关键。
3. Day 3 亲手实现 Reward Model。从偏好数据构建到训练完成，体会「把人类偏好转化为标量信号」的过程。
4. Day 4 是本周的架构核心，四模型的协同工作方式和参数共享策略是 RLHF 工程实现的关键。画出数据流图比看十遍文字更有效。
5. Day 5 的 Loss 推导是 Day 6 的理论基础。确保能从目标函数推到最终的三部分 Loss，并理解每一项的作用。
6. Day 6 是本周最重要的实践，也是面试 Tier 1 考点。RLHF-PPO 训练循环的手写能力需要反复练习。

---

## 关键检查点 ✅

完成本周学习后，你应该能够：

- [ ] 画出 RLHF 三阶段流程图：SFT → Reward Model → PPO
- [ ] 解释 InstructGPT 的核心贡献和训练流程
- [ ] 推导 Bradley-Terry 模型：$P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$
- [ ] 写出 RM Loss：$L_{\text{RM}} = -\mathbb{E}[\log \sigma(r_\theta(y_w) - r_\theta(y_l))]$
- [ ] **手写 Reward Model 并训练成功**
- [ ] 画出 RLHF 四模型数据流：Actor 生成 → RM 打分 → Critic 估值 → PPO 更新
- [ ] 解释四模型的参数共享策略
- [ ] 写出 RLHF 优化目标：$\max_\theta \mathbb{E}[R(x,y)] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$
- [ ] **推导完整 RLHF-PPO Loss（Policy + Value + KL）**（面试高频！）
- [ ] 解释 per-token KL penalty 的计算方式
- [ ] 说明 GAE 在只有最终奖励时的适配方法
- [ ] **手写 RLHF-PPO 训练循环**（面试 Tier 1！）
- [ ] 在 GPT-2 上完成一次 RLHF 训练
- [ ] 对比 RLHF 前后的生成质量变化
- [ ] 解释 Reward Hacking 及其对策
- [ ] 说明 RLHF 的局限性以及 DPO 如何解决（为 W11 铺路）

---

## 本周必读论文

1. **Training language models to follow instructions with human feedback** (Ouyang et al., 2022) — InstructGPT，**精读**
2. **LLaMA 2: Open Foundation and Fine-Tuned Chat Models** (Touvron et al., 2023) — LLaMA2 RLHF 部分，**精读**

## 参考论文

- *Fine-Tuning Language Models from Human Preferences* (Ziegler et al., 2019) — 早期 RLHF 工作
- *Learning to summarize from human feedback* (Stiennon et al., 2020) — 摘要任务 RLHF
- *A General Language Assistant as a Laboratory for Alignment* (Askell et al., 2021) — Anthropic 对齐研究
- *Secrets of RLHF in Large Language Models Part I: PPO* (Zheng et al., 2023) — RLHF 工程实践
- *Proximal Policy Optimization Algorithms* (Schulman et al., 2017) — PPO 原始论文（W9 精读）

## 推荐资源

- Hugging Face Blog: [Illustrating RLHF](https://huggingface.co/blog/rlhf) — RLHF 图解
- Hugging Face Blog: [StackLLaMA](https://huggingface.co/blog/stackllama) — RLHF 实践教程
- Chip Huyen: [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html) — RLHF 综述
- Nathan Lambert: [RLHF Book](https://rlhfbook.com/) — RLHF 系统性参考
- Lilian Weng: [Reward Hacking in RLHF](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) — Reward Hacking 专题
- OpenAI: [InstructGPT Blog](https://openai.com/research/instruction-following) — InstructGPT 官方博客
- Anthropic HH-RLHF: [Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) — 人类偏好数据集

---

## 本周最终产出

完成本周后，建议至少形成以下可复用资产：

- 一份 InstructGPT / LLaMA2 RLHF 部分的论文精读笔记
- 一张 RLHF 三阶段 + 四模型架构的完整流程图
- 一份 Bradley-Terry 模型与 RM Loss 的推导笔记
- 一个从零手写的 Reward Model：支持偏好对比训练，准确率 >80%
- 一份 RLHF-PPO Loss 的完整推导：从目标函数到三部分 Loss
- 一个从零手写的 RLHF-PPO 训练循环：在 GPT-2 上完成训练，含生成质量对比
- 一份 Reward Hacking 分析与 DPO 预览笔记：为第 11 周做准备
