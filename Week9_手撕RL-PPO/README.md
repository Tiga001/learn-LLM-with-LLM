# 第 9 周：手撕 RL-PPO（强化学习基础）

> **目标**：建立从经典强化学习到 PPO 的完整知识链——从 MDP 与 Bellman 方程出发，理解 Q 值 / V 值函数的定义与计算；手写 DQN 并在 CartPole 上训练成功；严格推导策略梯度定理，理解 REINFORCE 算法及其高方差问题；掌握 Actor-Critic 框架、Advantage 函数与 GAE 的完整推导；最终从零手写 PPO-Clip 算法，在 CartPole 和 LunarLander 上训练成功。本周是阶段四「强化学习对齐」的数学与工程基础，直接服务于第 10 周的 RLHF 和第 11 周的 DPO / GRPO。

---

## 每日安排（建议每日 3~5 小时）

| 天 | 主题 | 核心任务 | 产出 | 难度 |
|----|------|---------|------|:----:|
| Day 1 | MDP 与价值函数 | MDP 五元组定义、V/Q 函数、Bellman 方程推导 | 手写 Bellman 方程求解 + GridWorld 示例 | ⭐⭐⭐ |
| Day 2 | DQN 原理详解 | 从 Q-Learning 到 DQN 的演进、经验回放、目标网络 | DQN 原理笔记 + Loss 推导 | ⭐⭐⭐⭐ |
| Day 3 | **手写 DQN 实现** | 从零实现 DQN：ReplayBuffer + 网络 + 训练循环 | **CartPole 上 DQN 训练成功** | ⭐⭐⭐⭐ |
| Day 4 | 策略梯度与 REINFORCE | 策略梯度定理完整推导、log-derivative trick、baseline | 策略梯度推导笔记 | ⭐⭐⭐⭐⭐ |
| Day 5 | Actor-Critic 与 GAE | Actor-Critic 框架、Advantage 函数、GAE 推导、A2C | GAE 推导 + A2C 伪代码 | ⭐⭐⭐⭐⭐ |
| Day 6 | **手写 PPO 实现** | PPO-Clip 完整实现：rollout → GAE → mini-batch 更新 | **CartPole + LunarLander 训练成功** | ⭐⭐⭐⭐⭐ |
| Day 7 | PPO 深入分析与复盘 | PPO 变体对比、clip 分析、与 RLHF 衔接、全周复盘 | 对比笔记 + 自检 | ⭐⭐⭐⭐ |

---

## 知识图谱

```
Day 1: MDP 与价值函数 — 强化学习的数学基础
  MDP 五元组 → V(s) / Q(s,a) → Bellman 方程 → 最优策略
       │
       ▼
Day 2: DQN 原理详解 — 从表格法到深度学习
  Q-Learning → 函数逼近 → 经验回放 → 目标网络 → DQN Loss
       │
       ▼
Day 3: 手写 DQN 实现（第一个实践！）
  ReplayBuffer → DQN 网络 → epsilon-greedy → 训练循环 → CartPole 训练
       │
       ▼
Day 4: 策略梯度与 REINFORCE — 从值函数到策略优化
  策略参数化 → 策略梯度定理推导 → log-derivative trick → baseline 减方差
       │
       ▼
Day 5: Actor-Critic 与 GAE — 方差与偏差的权衡（本周理论核心！）
  Actor-Critic 框架 → Advantage 函数 → TD(λ) → GAE 推导 → A2C
       │
       ▼
Day 6: 手写 PPO 实现（本周最重要！）
  PPO-Clip 目标函数 → GAE → Rollout → mini-batch → 三 Loss → 训练成功
       │
       ▼
Day 7: PPO 深入分析与复盘
  PPO-Clip vs PPO-Penalty → TRPO 关系 → clip 可视化 → 与 RLHF 衔接
  → 第 10 周 RLHF 铺路
```

---

## 与前序周次的关系

| 本周内容 | 前序基础 | 后续衔接 |
|---------|---------|---------|
| MDP / Bellman 方程 | 数学基础（概率、期望、递推） | W10 RLHF 中的 reward 建模 |
| DQN | 神经网络基础、梯度下降 | W10 Reward Model 训练 |
| 策略梯度定理 | W3 语言模型概率建模 | W10 RLHF-PPO 中的 Policy Loss |
| Actor-Critic / GAE | 方差-偏差权衡 | W10 RLHF 四模型架构中的 Critic |
| PPO-Clip | 所有前序 RL 知识 | W10 RLHF-PPO 训练循环（直接复用） |
| PPO 与 KL 约束 | 信息论基础 | W11 DPO 的 KL 约束推导 |

---

## 文件结构

```
Week9_手撕RL-PPO/
├── README.md                              ← 你在这里
├── Day1_MDP与价值函数.md                   ← MDP、Bellman 方程、V/Q 函数
├── Day2_DQN原理详解.md                     ← DQN、经验回放、目标网络
├── Day3_手写DQN实现.ipynb                  ← 手写 DQN + CartPole 训练 (实践!)
├── Day4_策略梯度与REINFORCE.md             ← 策略梯度定理推导、REINFORCE
├── Day5_Actor-Critic与GAE.md              ← A2C、Advantage、GAE 推导 (理论核心!)
├── Day6_手写PPO实现.ipynb                  ← PPO-Clip 完整实现 + 训练 (最重要!)
└── Day7_PPO深入分析与复盘.md               ← PPO 变体、clip 分析、RLHF 衔接
```

---

## 学习建议

推荐按下面的节奏学习：

1. Day 1 建立 MDP 与价值函数的数学直觉。这些定义和推导是后面所有算法的基础，不要急于跳过。
2. Day 2 理解 DQN 的三大创新，然后在 Day 3 亲手把 DQN 从零写出来。看懂和写出来是两回事。
3. Day 4 是本周数学最密集的一天，策略梯度定理的推导需要反复咀嚼。理解了这个定理，后面的 Actor-Critic 和 PPO 就是自然的推论。
4. Day 5 的 GAE 推导是连接理论与工程的桥梁，Day 6 的 PPO 实现会直接使用 GAE。
5. Day 6 是本周最重要的实践，PPO 手写能力是面试高频考点，也是 W10 RLHF 的直接基础。
6. Day 7 做对比分析和全周回顾，确保能把整条知识链串起来。

---

## 关键检查点 ✅

完成本周学习后，你应该能够：

- [ ] 写出 MDP 五元组 $(S, A, P, R, \gamma)$ 的形式化定义
- [ ] 推导 Bellman 期望方程和 Bellman 最优方程
- [ ] 解释 DQN 的三大创新：函数逼近、经验回放、目标网络
- [ ] 推导 DQN 的 TD Loss：$L = \mathbb{E}[(r + \gamma \max_{a'} Q_{\text{target}}(s', a') - Q(s, a))^2]$
- [ ] **手写 DQN 并在 CartPole 上训练成功**
- [ ] **完整推导策略梯度定理**（面试高频！）
- [ ] 解释 baseline 为什么能减少方差但不引入偏差
- [ ] 画出 Actor-Critic 的数据流：Actor 输出动作 → 环境反馈 → Critic 估计价值 → 计算 Advantage → 更新 Actor
- [ ] **推导 GAE：$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$**
- [ ] **手写 PPO-Clip Loss：$L^{\text{CLIP}} = \mathbb{E}[\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)]$**（面试高频！）
- [ ] **PPO 在 CartPole 上训练成功**
- [ ] **PPO 在 LunarLander 上训练成功**
- [ ] 理解 PPO clip ratio 与 KL penalty 的直觉
- [ ] 说明 PPO 在 RLHF 中的角色（为 W10 铺路）

---

## 本周必读论文

1. **Proximal Policy Optimization Algorithms** (Schulman et al., 2017) — **精读**
2. **Playing Atari with Deep Reinforcement Learning** (Mnih et al., 2013) — **精读**

## 参考论文

- *Human-level control through deep reinforcement learning* (Mnih et al., 2015) — Nature DQN
- *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (Williams, 1992) — REINFORCE
- *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (Schulman et al., 2016) — GAE
- *Trust Region Policy Optimization* (Schulman et al., 2015) — TRPO
- *Asynchronous Methods for Deep Reinforcement Learning* (Mnih et al., 2016) — A3C
- *Policy Gradient Methods for Reinforcement Learning with Function Approximation* (Sutton et al., 2000) — 策略梯度定理

## 推荐资源

- *Reinforcement Learning: An Introduction* (Sutton & Barto, 2018): [在线版](http://incompleteideas.net/book/the-book.html) — RL 经典教材
- Lilian Weng: [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — 策略梯度系列文章
- OpenAI Spinning Up: [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — PPO 官方教程
- OpenAI Spinning Up: [Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — RL 基础概念
- David Silver RL Course: [Lecture Slides](https://www.davidsilver.uk/teaching/) — DeepMind RL 课程
- Gymnasium: [官方文档](https://gymnasium.farama.org/) / [GitHub 仓库](https://github.com/Farama-Foundation/Gymnasium)
- CleanRL: [GitHub 仓库](https://github.com/vwxyzjn/cleanrl) — 单文件 RL 实现参考

---

## 本周最终产出

完成本周后，建议至少形成以下可复用资产：

- 一份 MDP 与 Bellman 方程的数学笔记：覆盖 V/Q 函数、最优性条件
- 一个从零手写的 DQN：支持 CartPole 训练，含经验回放与目标网络
- 一份策略梯度定理的完整推导：从目标函数到梯度公式
- 一份 GAE 的推导笔记：从 TD error 到广义优势估计
- 一个从零手写的 PPO：支持 CartPole 和 LunarLander 训练，含 GAE 和 clipped objective
- 一份 PPO 与 RLHF 的衔接笔记：为第 10 周做准备
