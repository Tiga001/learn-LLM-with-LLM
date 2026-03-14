# Day 1：MDP 与价值函数 — 强化学习的数学基础

> **目标**：建立强化学习的数学语言——形式化定义马尔可夫决策过程（MDP），推导状态价值函数 $V(s)$ 和动作价值函数 $Q(s,a)$ 的 Bellman 方程，理解最优策略的数学含义，并通过 GridWorld 手算加深直觉。本日是后续所有 RL 算法（DQN、Policy Gradient、PPO）的数学基石。

---

## 一、为什么要学强化学习

### 1.1 LLM 对齐需要 RL

前 8 周我们学习了 LLM 的预训练和微调，但这些方法本质上都是**最大化下一个 token 的似然**：

$$\max_\theta \sum_{t} \log P_\theta(x_t \mid x_{<t})$$

这个目标与「生成对人类有用、无害、诚实的回答」之间存在 gap。RLHF（Reinforcement Learning from Human Feedback）通过引入人类偏好信号，用 RL 方法优化一个更贴近人类期望的目标函数。

而 PPO 正是 RLHF 中最常用的 RL 算法。要理解 PPO，必须先建立 RL 的基础。

### 1.2 本周的知识链

```
MDP → Q/V 函数 → DQN → Policy Gradient → Actor-Critic → GAE → PPO
 ↑                                                              ↓
 本日基础                                               W10 RLHF 直接使用
```

### 1.3 RL 与监督学习的本质区别

| 维度 | 监督学习 | 强化学习 |
|------|---------|---------|
| 数据 | 固定数据集 $(x, y)$ | 智能体与环境交互产生 |
| 标签 | 直接给出正确答案 | 只有延迟的奖励信号 |
| 目标 | 最小化预测误差 | 最大化累积奖励 |
| 核心挑战 | 泛化 | 探索与利用的平衡 |
| i.i.d. 假设 | 数据独立同分布 | 当前动作影响未来状态 |

强化学习的核心困难在于：**你的「训练数据」是由你自己的策略产生的，而策略又在不断更新**。这种自引用结构使得 RL 比监督学习复杂得多。

---

## 二、马尔可夫决策过程（MDP）

### 2.1 MDP 的五元组定义

MDP 是强化学习问题的标准数学框架，由五元组 $(S, A, P, R, \gamma)$ 组成：

| 符号 | 名称 | 含义 |
|------|------|------|
| $S$ | 状态空间 | 所有可能状态的集合 |
| $A$ | 动作空间 | 所有可能动作的集合 |
| $P(s' \mid s, a)$ | 状态转移概率 | 在状态 $s$ 执行动作 $a$ 后转移到 $s'$ 的概率 |
| $R(s, a, s')$ | 奖励函数 | 从 $s$ 执行 $a$ 转移到 $s'$ 获得的即时奖励 |
| $\gamma \in [0, 1)$ | 折扣因子 | 未来奖励的衰减系数 |

简化记号：奖励函数有时写为 $R(s, a)$（对 $s'$ 取期望后）或 $R(s)$（进一步对 $a$ 取期望）。

### 2.2 马尔可夫性质

MDP 的核心假设——**马尔可夫性**：

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

未来只取决于当前状态和动作，与历史无关。这个假设使得问题在数学上可处理。

**注意**：在 RLHF 中，「状态」是到当前 token 为止的完整序列，因此马尔可夫性天然满足。

### 2.3 轨迹（Trajectory）

智能体与环境交互产生一条轨迹 $\tau$：

$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$$

其中每一步：
1. 智能体观察状态 $s_t$
2. 根据策略 $\pi$ 选择动作 $a_t$
3. 环境返回奖励 $r_t = R(s_t, a_t)$ 和下一个状态 $s_{t+1} \sim P(\cdot \mid s_t, a_t)$

### 2.4 交互循环图

```
         ┌──────────┐
         │   Agent   │
         │  (策略 π) │
         └─────┬─────┘
     动作 a_t  │  ↑  状态 s_t, 奖励 r_t
               ▼  │
         ┌─────────────┐
         │ Environment  │
         │   (MDP)      │
         └─────────────┘
```

---

## 三、策略（Policy）

### 3.1 策略的定义

策略 $\pi$ 描述智能体的行为方式，有两种形式：

**随机策略（Stochastic Policy）**：

$$\pi(a \mid s) = P(A_t = a \mid S_t = s)$$

给出在状态 $s$ 下选择每个动作的概率分布。

**确定性策略（Deterministic Policy）**：

$$a = \mu(s)$$

直接映射状态到动作。

### 3.2 为什么区分两种策略

- 随机策略天然支持**探索**，且策略梯度定理需要策略可微
- 确定性策略在推理时更高效（不需要采样）
- 在 RLHF 中，LLM 的自回归生成就是一个随机策略：$\pi_\theta(a_t \mid s_t)$ 是给定上下文后下一个 token 的概率分布

### 3.3 最优策略

如果存在策略 $\pi^*$ 使得对所有状态 $s$ 和所有其他策略 $\pi$，都有：

$$V^{\pi^*}(s) \geq V^{\pi}(s), \quad \forall s \in S$$

则称 $\pi^*$ 为**最优策略**。

**定理**：对于有限 MDP，最优策略一定存在（可能不唯一），且所有最优策略共享相同的最优价值函数。

---

## 四、价值函数

### 4.1 回报（Return）

从时刻 $t$ 开始的**折扣累积回报**：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

折扣因子 $\gamma$ 的作用：

| $\gamma$ 值 | 行为 | 适用场景 |
|-------------|------|---------|
| $\gamma \to 0$ | 只关注即时奖励（短视） | 简单、即时反馈的环境 |
| $\gamma \to 1$ | 同等重视长期奖励（远视） | 需要长期规划的环境 |
| $0.99$ | 常用值，兼顾短期和长期 | 大多数任务 |

$\gamma < 1$ 还有一个数学作用：保证无限和 $G_t$ 收敛。

### 4.2 状态价值函数 V(s)

在策略 $\pi$ 下，从状态 $s$ 出发的**期望回报**：

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid S_t = s\right]$$

直觉：$V^\pi(s)$ 回答了「如果我从状态 $s$ 出发，一直遵循策略 $\pi$，期望能获得多少总奖励？」

### 4.3 动作价值函数 Q(s, a)

在策略 $\pi$ 下，从状态 $s$ 执行动作 $a$ 后的**期望回报**：

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid S_t = s, A_t = a\right]$$

直觉：$Q^\pi(s, a)$ 回答了「如果我在状态 $s$ 先执行动作 $a$，之后一直遵循策略 $\pi$，期望能获得多少总奖励？」

### 4.4 V 和 Q 的关系

$$V^\pi(s) = \sum_{a \in A} \pi(a \mid s) \cdot Q^\pi(s, a) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)]$$

即 $V$ 是 $Q$ 对策略的期望。反过来：

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in S} P(s' \mid s, a) \cdot V^\pi(s')$$

即 $Q$ 等于即时奖励加上后继状态价值的折扣期望。

---

## 五、Bellman 方程

### 5.1 Bellman 期望方程

利用回报 $G_t$ 的递推结构 $G_t = r_t + \gamma G_{t+1}$，可以推导出价值函数的递推关系。

**V 的 Bellman 期望方程**：

$$V^\pi(s) = \mathbb{E}_\pi[r_t + \gamma V^\pi(S_{t+1}) \mid S_t = s]$$

展开期望：

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \cdot V^\pi(s') \right]$$

**Q 的 Bellman 期望方程**：

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')$$

### 5.2 推导过程（以 V 为例）

从定义出发：

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$$

$$= \mathbb{E}_\pi[r_t + \gamma G_{t+1} \mid S_t = s]$$

$$= \mathbb{E}_\pi[r_t \mid S_t = s] + \gamma \mathbb{E}_\pi[G_{t+1} \mid S_t = s]$$

对于第一项，展开对动作和下一状态的期望：

$$\mathbb{E}_\pi[r_t \mid S_t = s] = \sum_a \pi(a \mid s) R(s, a)$$

对于第二项，利用全期望公式：

$$\mathbb{E}_\pi[G_{t+1} \mid S_t = s] = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']$$

$$= \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) V^\pi(s')$$

合并得到 Bellman 期望方程。 $\blacksquare$

### 5.3 Bellman 最优方程

最优价值函数：

$$V^*(s) = \max_\pi V^\pi(s), \quad Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

**V 的 Bellman 最优方程**：

$$V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s') \right]$$

**Q 的 Bellman 最优方程**：

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')$$

直觉：最优策略总是贪心地选择当前 Q 值最大的动作。

### 5.4 最优策略与最优 Q 的关系

一旦知道 $Q^*$，最优策略就是：

$$\pi^*(a \mid s) = \begin{cases} 1, & \text{if } a = \arg\max_{a'} Q^*(s, a') \\ 0, & \text{otherwise} \end{cases}$$

这告诉我们：**学到 Q* 就等价于学到最优策略**。这是 DQN 的理论基础。

### 5.5 四个 Bellman 方程的 Backup Diagram

用 Sutton & Barto 教材的 backup diagram 可以更直观地理解：

```
Bellman 期望方程 (V):              Bellman 期望方程 (Q):

      V(s)                              Q(s,a)
     / | \    ← Σ_a π(a|s)              |
   Q  Q  Q                              ↓ R(s,a)
   |  |  |   ← R + γΣP                 s'
   V  V  V                            / | \  ← Σ_a' π(a'|s')
                                     Q  Q  Q

Bellman 最优方程 (V*):             Bellman 最优方程 (Q*):

      V*(s)                             Q*(s,a)
     / | \    ← max_a                    |
   Q* Q* Q*                             ↓ R(s,a)
   |  |  |   ← R + γΣP                 s'
   V* V* V*                           / | \  ← max_a'
                                     Q* Q* Q*
```

---

## 六、GridWorld 手算示例

### 6.1 环境设定

一个 $3 \times 3$ 的网格世界：

```
┌─────┬─────┬─────┐
│  0  │  1  │  2  │
├─────┼─────┼─────┤
│  3  │  4  │  5  │
├─────┼─────┼─────┤
│  6  │  7  │  8★ │
└─────┴─────┴─────┘
```

- 8★ 是目标状态（到达后 reward = +1，episode 结束）
- 其他所有转移 reward = 0
- 动作空间：$A = \{\text{上, 下, 左, 右}\}$
- 碰到边界则停留在原地
- 折扣因子 $\gamma = 0.9$
- 策略：均匀随机（每个方向 0.25 概率）

### 6.2 手算 V(7) — 目标相邻状态

状态 7 可以向四个方向移动：
- 上 → 状态 4（reward = 0）
- 下 → 状态 7（撞墙，reward = 0）
- 左 → 状态 6（reward = 0）
- **右 → 状态 8★（reward = +1）**

$$V^\pi(7) = 0.25 \times [0 + 0.9 V(4)] + 0.25 \times [0 + 0.9 V(7)] + 0.25 \times [0 + 0.9 V(6)] + 0.25 \times [1 + 0.9 \times 0]$$

由于目标状态 8 的 $V(8) = 0$（终止状态），最后一项简化为 $0.25 \times 1 = 0.25$。

这是一个含有 $V(4)$、$V(6)$、$V(7)$ 的方程。对所有 8 个非终止状态写出类似方程，就形成一个 $8 \times 8$ 的线性方程组：

$$\mathbf{v} = \mathbf{r} + \gamma \mathbf{P} \mathbf{v}$$

$$(\mathbf{I} - \gamma \mathbf{P}) \mathbf{v} = \mathbf{r}$$

### 6.3 矩阵求解

对于随机策略，转移矩阵 $\mathbf{P}$ 的每行是该状态在均匀随机策略下转移到各状态的概率。直接求解线性方程组即可得到精确的 $V^\pi$。

实际中，大规模问题无法用矩阵求逆（状态空间可能是连续的或指数级的），因此需要用迭代方法或函数逼近——这正是 DQN 和策略梯度的动机。

### 6.4 数值结果

使用 $\gamma = 0.9$、均匀随机策略，精确求解后的 $V^\pi$ 大致为：

```
┌──────┬──────┬──────┐
│ 0.28 │ 0.43 │ 0.59 │
├──────┼──────┼──────┤
│ 0.43 │ 0.59 │ 0.77 │
├──────┼──────┼──────┤
│ 0.51 │ 0.77 │ 0.00 │
└──────┴──────┴──────┘
```

观察：离目标越近，$V$ 值越高。这符合直觉——越靠近目标，期望获得的累积奖励越多。

---

## 七、动态规划求解 MDP

### 7.1 策略评估（Policy Evaluation）

给定策略 $\pi$，通过迭代更新 $V$：

$$V_{k+1}(s) = \sum_a \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V_k(s') \right]$$

不断迭代直到收敛 $V_k \to V^\pi$。

### 7.2 策略改进（Policy Improvement）

给定 $V^\pi$，通过贪心改进得到更好的策略：

$$\pi'(s) = \arg\max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s') \right]$$

**策略改进定理**：如果 $\pi'$ 是对 $V^\pi$ 的贪心策略，则 $V^{\pi'} \geq V^\pi$。

### 7.3 策略迭代（Policy Iteration）

交替执行策略评估和策略改进，直到策略不再变化：

```
π₀ → 评估 → V^π₀ → 改进 → π₁ → 评估 → V^π₁ → 改进 → ... → π* → V*
```

### 7.4 值迭代（Value Iteration）

直接对 Bellman 最优方程做迭代：

$$V_{k+1}(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V_k(s') \right]$$

值迭代本质上是将策略改进融入了每一步评估中。

### 7.5 动态规划的局限

上述方法需要知道完整的 MDP 模型 $(P, R)$——但在大多数实际问题中，我们**不知道**状态转移概率。

这引出了**无模型（model-free）方法**：
- **基于值函数**：Q-Learning → DQN（Day 2-3）
- **基于策略**：Policy Gradient → Actor-Critic → PPO（Day 4-6）

---

## 八、Q-Learning 预览

### 8.1 从动态规划到 Q-Learning

动态规划要求已知 $P(s' \mid s, a)$，但实际中我们只能通过**采样**获得经验 $(s, a, r, s')$。

Q-Learning 的核心思想：用采样来替代期望：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

其中 $\alpha$ 是学习率，方括号内是 TD error（时序差分误差）。

### 8.2 Q-Learning 的关键特性

- **Off-policy**：更新目标用的是 $\max_{a'} Q(s', a')$（贪心策略），而采样用的可以是任意策略（如 $\epsilon$-greedy）
- **表格法**：维护一个 $|S| \times |A|$ 的 Q 表
- 当状态空间很大或连续时，表格法不可行 → DQN（Day 2）

---

## 九、RL 在 LLM 中的映射

理解这些概念后，我们来预览它们在 RLHF 中的对应：

| RL 概念 | RLHF 中的对应 |
|---------|-------------|
| 状态 $s$ | 到当前 token 为止的完整序列 $[x_1, \ldots, x_t]$ |
| 动作 $a$ | 下一个 token $x_{t+1}$ |
| 策略 $\pi_\theta$ | LLM 的自回归分布 $P_\theta(x_{t+1} \mid x_{\leq t})$ |
| 奖励 $R$ | Reward Model 对完整回答的打分 |
| 回报 $G$ | 最终奖励 + KL penalty |
| 折扣 $\gamma$ | 通常为 1（只有最终奖励） |
| Critic $V(s)$ | Value head 估计当前序列的期望回报 |

这些对应关系在第 10 周 RLHF 中会详细展开。

---

## 十、自检题

1. 写出 MDP 的五元组定义，解释每个元素的含义。
2. 马尔可夫性质的数学表达式是什么？为什么这个假设如此重要？
3. $V^\pi(s)$ 和 $Q^\pi(s, a)$ 的区别是什么？写出它们之间的关系。
4. 推导 V 的 Bellman 期望方程。
5. Bellman 期望方程和 Bellman 最优方程的区别是什么？
6. 折扣因子 $\gamma$ 的两个作用是什么？
7. 为什么知道 $Q^*$ 就等价于知道最优策略？
8. 策略评估和策略改进分别在做什么？
9. 动态规划方法的核心局限是什么？这如何引出 Q-Learning？
10. 在 RLHF 中，MDP 的状态、动作、奖励分别对应什么？

---

## 十一、产出要求

- [ ] 写出 MDP 五元组 $(S, A, P, R, \gamma)$ 的形式化定义
- [ ] 推导 V 和 Q 的 Bellman 期望方程
- [ ] 推导 Bellman 最优方程
- [ ] 手算 GridWorld 中至少一个状态的 $V^\pi$ 值
- [ ] 理解动态规划（策略迭代 / 值迭代）的思路
- [ ] 说明 Q-Learning 如何绕过"需要知道转移概率"的问题
- [ ] 建立 RL 概念与 RLHF 的对应关系

---

## 十二、与 Day 2 的衔接

今天我们建立了 RL 的数学基础：MDP、V/Q 函数、Bellman 方程。

明天我们将从 Q-Learning 出发，看它在大规模问题上的困难（表格法不可行），然后引入深度学习逼近 Q 函数——这就是 DQN。DQN 通过三个工程创新（函数逼近、经验回放、目标网络）让 Q-Learning 在高维问题上可用，并在 Atari 游戏上达到人类水平。
