# Day 7：PPO 深入分析与复盘 — 从经典 RL 到 RLHF 的桥梁

> **目标**：深入理解 PPO-Clip 与 PPO-Penalty 两种变体的数学联系与工程取舍；理解 TRPO 信赖域的数学直觉及 PPO 作为其一阶近似的关系；通过 clip ratio 可视化分析加深对 PPO 机制的理解；预览 PPO 在 LLM 对齐（RLHF）中的适配方式；串联全周 MDP → DQN → Policy Gradient → Actor-Critic → PPO 知识链，完成阶段性复盘。

---

## 一、PPO 两种变体

### 1.1 PPO-Clip（Day 6 实现）

Day 6 我们实现的版本，通过裁剪概率比来限制更新幅度：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$。

### 1.2 PPO-Penalty（KL 散度版本）

另一种变体是对 KL 散度做自适应惩罚：

$$L^{\text{KL}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \hat{A}_t - \beta \cdot D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \right]$$

其中 $\beta$ 是自适应的 KL 惩罚系数，训练过程中动态调整：

- 如果 $D_{\text{KL}} > d_{\text{target}} \times 1.5$：$\beta \leftarrow 2\beta$（KL 偏差太大，加大惩罚）
- 如果 $D_{\text{KL}} < d_{\text{target}} / 1.5$：$\beta \leftarrow \beta / 2$（KL 太小，放松约束）

### 1.3 两种变体对比

| 维度 | PPO-Clip | PPO-Penalty |
|------|----------|-------------|
| 核心机制 | 硬裁剪概率比 | KL 散度软惩罚 |
| 超参数 | $\epsilon$（通常 0.2，很少需要调） | $\beta$（需要自适应调整）|
| 实现复杂度 | 简单（一行 clamp） | 中等（需要计算 KL 和调整 $\beta$） |
| 理论保证 | 无严格保证，但实践效果好 | 与 TRPO 更接近 |
| 实际使用 | **工业界主流** | 学术研究 / RLHF 中有使用 |
| OpenAI 推荐 | 原始论文推荐 | 作为备选 |

**结论**：PPO-Clip 因其简单性和鲁棒性成为工业界标准。但在 RLHF 中，由于 KL 散度对策略偏移的控制更精细，PPO-Penalty 有时也会使用。

---

## 二、TRPO 与 PPO 的数学关系

### 2.1 TRPO 的优化问题

TRPO（Trust Region Policy Optimization）解决的是一个带约束的优化问题：

$$\max_\theta \quad \mathbb{E}_t \left[ r_t(\theta) \hat{A}_t \right]$$

$$\text{s.t.} \quad \mathbb{E}_t \left[ D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot \mid s_t) \| \pi_\theta(\cdot \mid s_t)) \right] \leq \delta$$

直觉：在「新旧策略不会相差太远」的约束下，最大化代理目标。

### 2.2 TRPO 的求解方法

TRPO 使用二阶方法：

1. 对 KL 约束做二阶 Taylor 展开，得到 Fisher 信息矩阵 $F$
2. 用共轭梯度法（Conjugate Gradient）近似求解 $F^{-1} g$
3. 做线搜索确保约束满足

这使得 TRPO 实现复杂、计算昂贵。

### 2.3 PPO 是 TRPO 的一阶近似

PPO 的关键洞察：**不需要严格满足 KL 约束，用 clip 或 penalty 来近似即可**。

```
TRPO: max E[r(θ)A]  s.t. KL ≤ δ     ← 二阶优化，精确但昂贵
  ↓ 近似
PPO-Penalty: max E[r(θ)A - β·KL]     ← 将约束变为惩罚项
  ↓ 进一步简化
PPO-Clip: max E[min(r(θ)A, clip(r(θ))A)]  ← 用 clip 替代 KL
```

PPO 放弃了 TRPO 的理论保证（单调改进），但获得了：
- 一阶优化即可（普通 Adam）
- 实现简单（几行代码）
- 可以做 mini-batch 和多 epoch 更新
- 实践效果与 TRPO 相当甚至更好

### 2.4 为什么 PPO 在实践中胜出

| 维度 | TRPO | PPO |
|------|------|-----|
| 优化方法 | 二阶（Fisher + CG） | 一阶（Adam） |
| 每步计算量 | 高 | 低 |
| 数据复用 | 通常 1 次 | 多个 epoch |
| 实现行数 | 数百行 | 几十行 |
| mini-batch | 困难 | 天然支持 |
| 实际性能 | 好 | 同等或更好 |

---

## 三、PPO Clip 的可视化分析

### 3.1 Clip 在不同 Advantage 下的行为

当 $\hat{A}_t > 0$（好动作）：

```
L = min(r·A, clip(r, 0.8, 1.2)·A)

  L
  ^
  |           ╱ r·A (无 clip)
  |         ╱
  |       ╱── clip(r)·A (被截断)
  |     ╱╱
  |   ╱╱
  | ╱╱
  +────────────────→ r
  0    0.8  1.0  1.2

当 r > 1.2 时，L 不再增长 → 防止过度增大好动作的概率
```

当 $\hat{A}_t < 0$（差动作）：

```
L = min(r·A, clip(r, 0.8, 1.2)·A)
  注意: A < 0, 所以 r·A 随 r 减小而增大

  L
  ^
  |         ╲╲
  |          ╲╲── clip(r)·A (被截断)
  |            ╲╲
  |              ╲ r·A (无 clip)
  |
  +────────────────→ r
  0    0.8  1.0  1.2

当 r < 0.8 时，L 不再增长 → 防止过度减小差动作的概率
```

### 3.2 min 操作的作用

$\min$ 操作确保了一个保守的策略更新：

- 对于好动作 ($A > 0$)：取较小的激励，防止过度自信
- 对于差动作 ($A < 0$)：取较小的惩罚幅度（绝对值更大的负值），防止过度惩罚

本质上，clip + min 构成了一个**悲观的更新规则**——总是选择更保守的那个方向。

### 3.3 Clip Fraction 的监控

训练过程中应监控 clip fraction——被裁剪的样本比例：

- **clip fraction 过高**（如 > 0.3）：策略变化太快，可能需要减小学习率
- **clip fraction 过低**（如 < 0.01）：更新太保守，学习效率低
- **典型范围**：0.05 ~ 0.2

---

## 四、Value Function Clipping 的争议

### 4.1 什么是 Value Clipping

有些实现对 value loss 也做 clipping：

$$L^{\text{VF-clip}} = \max\left[ (V_\phi - G)^2, \; (\text{clip}(V_\phi, V_{\text{old}} - \epsilon, V_{\text{old}} + \epsilon) - G)^2 \right]$$

### 4.2 争议

论文 *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO*（Engstrom et al., 2020）发现：

- Value clipping 在大多数情况下**没有帮助**，甚至可能有害
- PPO 的好性能更多来自代码层面的实现细节（如 advantage normalization、gradient clipping），而非算法本身

### 4.3 重要的实现细节

真正影响 PPO 性能的 tricks：

| Trick | 影响 | 我们的实现中 |
|-------|------|------------|
| Advantage normalization | 大 | ✓（`(adv - mean) / std`）|
| Gradient clipping | 大 | ✓（`max_grad_norm=0.5`）|
| Learning rate annealing | 中 | 未加（可选） |
| Orthogonal initialization | 小 | 未加（可选） |
| Value function clipping | 争议 | 未加 |

---

## 五、PPO 在 RLHF 中的适配 — W10 预览

### 5.1 RLHF 的完整流程

```
Stage 1: SFT (Supervised Fine-Tuning)
  在高质量数据上微调 LLM → π_SFT

Stage 2: Reward Model 训练
  收集人类偏好数据 (y_w > y_l) → 训练 RM

Stage 3: PPO 优化
  用 PPO 优化 LLM 使其最大化 RM 打分，同时不偏离 π_SFT 太远
```

### 5.2 PPO 在 LLM 中的映射

| RL 概念 | RLHF 中的对应 |
|---------|-------------|
| 环境 | 「给定 prompt，生成回答」这个交互过程 |
| 状态 $s_t$ | 到当前 token 为止的序列 $[x_1, \ldots, x_t]$ |
| 动作 $a_t$ | 下一个 token $x_{t+1}$ |
| 策略 $\pi_\theta(a \mid s)$ | LLM 的 next-token 分布 |
| 奖励 $r$ | 只在序列末尾有：$R = \text{RM}(\text{prompt}, \text{response})$ |
| Reference Policy $\pi_{\text{ref}}$ | 冻结的 SFT 模型 |
| KL Penalty | $-\beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ |
| Critic $V_\phi$ | Value Head（通常接在 LLM backbone 上） |

### 5.3 RLHF 的四模型架构

```
┌────────────────────────────────────────────────┐
│                RLHF-PPO 四模型                  │
│                                                 │
│  1. Actor (π_θ)     — 被优化的 LLM              │
│  2. Critic (V_ϕ)    — 价值网络（估计期望回报）    │
│  3. Reward (RM)     — 冻结的奖励模型              │
│  4. Reference (π_ref) — 冻结的 SFT 模型          │
│                                                 │
│  PPO 优化目标:                                   │
│  max E[RM(response)] - β·KL(π_θ ∥ π_ref)       │
│                                                 │
│  其中 PPO 训练循环:                              │
│  Actor 生成 → RM 打分 → GAE 计算 → PPO 更新      │
└────────────────────────────────────────────────┘
```

### 5.4 与经典 RL PPO 的关键差异

| 维度 | 经典 RL PPO（本周） | RLHF PPO（W10） |
|------|-------------------|-----------------|
| 环境 | Gym（CartPole 等） | LLM 生成过程 |
| 奖励 | 每步即时奖励 | 只有最终奖励（RM 打分） |
| 额外约束 | 无 | KL penalty（不偏离 $\pi_{\text{ref}}$ 太远） |
| 模型数量 | 1 个 ActorCritic | 4 个模型 |
| 动作空间 | 小（2~4 个） | 大（词表大小，数万） |
| rollout | 数千步 | 一个完整回答（数百 token） |
| 计算量 | 单 GPU 分钟级 | 多 GPU 小时级 |

这些差异将在 W10 详细展开。本周建立的 PPO 基础（GAE、clip、Actor-Critic）会直接复用。

---

## 六、全周知识串联

### 6.1 知识链回顾

```
Day 1: MDP → 价值函数 → Bellman 方程
  "RL 问题的数学语言是什么？"
       │
       ▼
Day 2: Q-Learning → DQN (函数逼近 + 经验回放 + 目标网络)
  "如何用神经网络学 Q 函数？"
       │
       ▼
Day 3: 手写 DQN → CartPole 训练成功
  "DQN 怎么写？为什么它只能做离散动作？"
       │
       ▼
Day 4: 策略梯度定理 → REINFORCE
  "能不能直接优化策略？→ log-derivative trick → 可以！但方差大"
       │
       ▼
Day 5: Actor-Critic → GAE
  "用 Critic 降方差 + GAE 平衡偏差方差 → 稳定的策略梯度信号"
       │
       ▼
Day 6: 手写 PPO → CartPole + LunarLander 训练成功
  "clip 限制更新幅度 + 多 epoch 复用数据 → 当前最实用的 RL 算法"
       │
       ▼
Day 7: PPO 变体分析 + RLHF 预览
  "PPO 如何服务于 LLM 对齐？→ W10 RLHF"
```

### 6.2 核心公式速查

| 公式 | 名称 | 来源 |
|------|------|------|
| $V^\pi(s) = \mathbb{E}_\pi[\sum_k \gamma^k r_{t+k} \mid s]$ | 状态价值函数 | Day 1 |
| $Q(s,a) \leftarrow Q + \alpha(r + \gamma \max Q' - Q)$ | Q-Learning | Day 2 |
| $\nabla_\theta J = \mathbb{E}[\nabla \log \pi_\theta \cdot \hat{A}]$ | 策略梯度定理 | Day 4 |
| $\hat{A}_t = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}$ | GAE | Day 5 |
| $L = \mathbb{E}[\min(r \hat{A}, \text{clip}(r) \hat{A})]$ | PPO-Clip | Day 6 |

### 6.3 方法演进与动机

| 从 | 到 | 动机 |
|----|---|------|
| 表格 Q-Learning | DQN | 状态空间太大，表格存不下 |
| DQN | Policy Gradient | 需要连续动作和随机策略 |
| REINFORCE | Actor-Critic | MC 回报方差太大 |
| A2C | TRPO | 步长敏感，策略可能崩塌 |
| TRPO | PPO | 二阶优化太复杂 |
| PPO | RLHF-PPO | 让 LLM 对齐人类偏好 |

---

## 七、自检题

### 理论理解

1. PPO-Clip 和 PPO-Penalty 的核心区别是什么？各自的优缺点？
2. TRPO 解决的是什么优化问题？PPO 如何近似它？
3. 画出 $\hat{A} > 0$ 时 PPO-Clip 目标函数关于 $r_t(\theta)$ 的图像。
4. Clip fraction 过高或过低分别说明什么？
5. Value function clipping 的争议是什么？

### 面试手撕

6. 写出 PPO-Clip 的目标函数公式。
7. 写出 GAE 的递推计算公式。
8. 写出 PPO 的完整 Loss（三部分）。
9. 解释 advantage normalization 为什么重要。
10. 说明 PPO 在 RLHF 中的四模型架构。

### 与 W10 衔接

11. RLHF 中的奖励信号来自哪里？与经典 RL 有什么不同？
12. 为什么 RLHF 需要 KL penalty？如果不加会怎样？

---

## 八、关键检查点 — 全周自检

完成本周学习后，你应该能够：

**Tier 1：必须能手写**
- [ ] PPO-Clip Loss 公式及其实现
- [ ] GAE 递推计算
- [ ] 策略梯度定理推导

**Tier 2：需要熟练理解**
- [ ] MDP 五元组与 Bellman 方程
- [ ] DQN 的三大创新及 TD Loss
- [ ] Actor-Critic 框架（Actor/Critic 各做什么）
- [ ] Advantage normalization 的原因

**Tier 3：需要能说清思路**
- [ ] TRPO 与 PPO 的关系
- [ ] PPO-Clip vs PPO-Penalty 的取舍
- [ ] PPO 在 RLHF 中的适配（四模型架构）
- [ ] On-policy vs Off-policy 的区别

---

## 九、产出要求

- [ ] 对比 PPO-Clip 和 PPO-Penalty 的优缺点
- [ ] 解释 TRPO 的信赖域思想及 PPO 如何近似
- [ ] 画出 PPO clip 在 $A > 0$ 和 $A < 0$ 时的图像
- [ ] 说明 RLHF 的四模型架构
- [ ] **能从 MDP 一路讲到 PPO，串联完整知识链**
- [ ] 为第 10 周 RLHF 做好知识准备

---

## 十、推荐深入阅读

本周结束后，如果想继续深入：

1. **PPO 原始论文**：*Proximal Policy Optimization Algorithms* (Schulman et al., 2017)  — 必读
2. **TRPO 论文**：*Trust Region Policy Optimization* (Schulman et al., 2015)  — 理解 PPO 的数学背景
3. **GAE 论文**：*High-Dimensional Continuous Control Using Generalized Advantage Estimation* (Schulman et al., 2016)  — GAE 的完整理论
4. **实现细节**：*Implementation Matters in Deep Policy Gradients* (Engstrom et al., 2020)  — PPO 的工程 tricks
5. **CleanRL 代码**：[github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)  — 单文件 PPO 参考实现
6. **InstructGPT 论文**：*Training language models to follow instructions with human feedback* (Ouyang et al., 2022)  — RLHF 的经典论文，W10 必读

---

> **本周小结**：从 MDP 的数学基础出发，经过 DQN → Policy Gradient → Actor-Critic → GAE → PPO 的完整知识链，我们建立了理解 RLHF 所需的全部 RL 基础。PPO 不是终点——它是 LLM 对齐的起点。下周，我们将把 PPO 嵌入 RLHF 的四模型架构，真正用强化学习来优化大语言模型。
