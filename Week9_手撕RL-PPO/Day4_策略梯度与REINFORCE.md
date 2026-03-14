# Day 4：策略梯度与 REINFORCE — 直接优化策略的数学之美

> **目标**：理解 DQN 无法处理连续动作空间和随机策略的根本局限；掌握策略参数化的思想，从目标函数出发严格推导策略梯度定理；理解 log-derivative trick 的数学本质；掌握 REINFORCE 算法及其高方差问题；推导 baseline 减方差的数学原理；为 Day 5 的 Actor-Critic 和 Day 6 的 PPO 打下坚实理论基础。本日是全周数学最密集的一天，也是面试高频考点。

---

## 一、为什么需要策略梯度

### 1.1 DQN 的根本局限

Day 2-3 我们实现了 DQN，它通过学习 $Q(s, a)$ 间接得到策略 $\pi(s) = \arg\max_a Q(s, a)$。但这带来两个根本问题：

**问题一：无法处理连续动作空间**

DQN 的选动作操作是 $\arg\max_a Q(s, a)$——需要遍历所有动作。当动作空间是连续的（如机器人关节角度 $a \in \mathbb{R}^d$），这不可行。

**问题二：只能输出确定性策略**

DQN 总是选 Q 最大的动作，没有随机性。但在 RLHF 中，LLM 需要输出 token 的概率分布 $\pi_\theta(a \mid s)$，这是随机策略。

### 1.2 策略梯度的核心思想

既然间接方法有局限，能不能**直接参数化策略 $\pi_\theta(a \mid s)$ 并用梯度优化**？

```
DQN 路线:  学 Q(s,a) → argmax 得到策略 → 间接优化
PG 路线:   直接参数化 π_θ(a|s) → 梯度上升最大化期望回报 → 直接优化
```

| 维度 | Value-based (DQN) | Policy-based (PG) |
|------|-------------------|-------------------|
| 学什么 | Q 函数 | 策略 $\pi_\theta$ |
| 动作空间 | 仅离散 | 离散 + 连续 |
| 策略类型 | 确定性 | 随机 |
| 与 LLM 的对应 | — | $\pi_\theta(a \mid s)$ 就是 LLM 输出分布 |

### 1.3 策略梯度与 LLM 的天然联系

LLM 的自回归生成本身就是一个参数化的随机策略：

$$\pi_\theta(x_{t+1} \mid x_{\leq t}) = P_\theta(x_{t+1} \mid x_1, \ldots, x_t)$$

RLHF 正是用策略梯度方法（PPO）来优化这个策略，使其生成更符合人类偏好的文本。所以策略梯度定理不是一个抽象的数学公式——它是 RLHF 的核心引擎。

---

## 二、策略参数化

### 2.1 离散动作空间

对于离散动作 $a \in \{a_1, a_2, \ldots, a_K\}$，策略网络输出 logits，经 softmax 得到概率分布：

$$\pi_\theta(a \mid s) = \text{softmax}(f_\theta(s))_a = \frac{\exp(f_\theta(s)_a)}{\sum_{a'} \exp(f_\theta(s)_{a'})}$$

这与 LLM 输出 token 概率的方式完全一致。

采样动作：$a \sim \pi_\theta(\cdot \mid s)$（从 Categorical 分布采样）。

### 2.2 连续动作空间

对于连续动作 $a \in \mathbb{R}^d$，策略网络输出高斯分布的参数：

$$\pi_\theta(a \mid s) = \mathcal{N}(a \mid \mu_\theta(s), \sigma_\theta(s)^2)$$

其中 $\mu_\theta(s)$ 是均值网络，$\sigma_\theta(s)$ 是标准差网络（或用固定/可学习的全局参数）。

采样动作：$a = \mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$（重参数化技巧）。

### 2.3 策略网络的典型架构

```
离散动作:
  状态 s → MLP → logits → softmax → π(a|s)

连续动作:
  状态 s → MLP → [μ, log σ] → N(μ, σ²) → 采样 a
```

---

## 三、策略梯度目标函数

### 3.1 目标定义

我们的目标是找到参数 $\theta$，使得策略 $\pi_\theta$ 的**期望回报最大**：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

其中 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ 是一条轨迹，$R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$ 是折扣累积奖励。

展开期望：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int P_\theta(\tau) R(\tau) \, d\tau$$

其中轨迹概率为：

$$P_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)$$

### 3.2 优化方法

使用梯度上升：

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

关键问题：**如何计算 $\nabla_\theta J(\theta)$？**

直接对 $J(\theta) = \int P_\theta(\tau) R(\tau) \, d\tau$ 求梯度困难重重：
- $P_\theta(\tau)$ 包含环境动力学 $P(s' \mid s, a)$，我们不知道
- 期望是对高维轨迹空间的积分，无法解析计算

策略梯度定理巧妙地解决了这两个问题。

---

## 四、策略梯度定理 — 完整推导

### 4.1 Log-Derivative Trick（对数导数技巧）

这是推导的核心数学工具。对于任意概率分布 $p_\theta(x)$：

$$\nabla_\theta p_\theta(x) = p_\theta(x) \cdot \nabla_\theta \log p_\theta(x)$$

**证明**：

$$\nabla_\theta \log p_\theta(x) = \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)}$$

两边乘以 $p_\theta(x)$：

$$p_\theta(x) \cdot \nabla_\theta \log p_\theta(x) = \nabla_\theta p_\theta(x) \quad \blacksquare$$

这个技巧的价值在于：将**对概率的梯度**转化为**对 log 概率的梯度乘以概率**，后者可以通过采样来估计。

### 4.2 策略梯度定理推导

从目标函数出发：

$$\nabla_\theta J(\theta) = \nabla_\theta \int P_\theta(\tau) R(\tau) \, d\tau$$

**Step 1**：将梯度移入积分（在正则条件下可交换）：

$$= \int \nabla_\theta P_\theta(\tau) \cdot R(\tau) \, d\tau$$

**Step 2**：应用 log-derivative trick：

$$= \int P_\theta(\tau) \cdot \nabla_\theta \log P_\theta(\tau) \cdot R(\tau) \, d\tau$$

**Step 3**：将积分写回期望：

$$= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P_\theta(\tau) \cdot R(\tau) \right]$$

**Step 4**：展开 $\log P_\theta(\tau)$：

$$\log P_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T-1} \left[ \log \pi_\theta(a_t \mid s_t) + \log P(s_{t+1} \mid s_t, a_t) \right]$$

对 $\theta$ 求梯度时，$\log p(s_0)$ 和 $\log P(s_{t+1} \mid s_t, a_t)$ 都与 $\theta$ 无关，因此：

$$\nabla_\theta \log P_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

**Step 5**：代入得到策略梯度定理：

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau) \right]}$$

### 4.3 策略梯度定理的三个关键性质

**性质一：不需要环境动力学模型**

$\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ 只涉及策略网络，不需要知道 $P(s' \mid s, a)$。这是 model-free 方法的基础。

**性质二：可用采样估计**

期望可以用蒙特卡洛采样近似：

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)}) \cdot R(\tau^{(i)})$$

**性质三：梯度方向的直觉**

$$\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau)$$

- 如果 $R(\tau) > 0$（好的轨迹）：增大 $\pi_\theta(a_t \mid s_t)$——让好动作更可能被选
- 如果 $R(\tau) < 0$（差的轨迹）：减小 $\pi_\theta(a_t \mid s_t)$——让差动作更不可能被选

直觉上：**策略梯度在做"加权最大似然"——用回报作为权重**。

---

## 五、REINFORCE 算法

### 5.1 从定理到算法

REINFORCE（Williams, 1992）是策略梯度定理的最直接实现：用蒙特卡洛采样估计梯度。

### 5.2 改进：因果性

原始公式中，时刻 $t$ 的梯度乘以整条轨迹的回报 $R(\tau)$。但因果性告诉我们：**时刻 $t$ 的动作不应该被 $t$ 之前的奖励影响**。

改进为 "reward-to-go"（从 $t$ 开始的未来回报）：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]$$

其中 $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$ 是从 $t$ 开始的折扣回报。

这个改进不改变梯度期望，但减少了方差。

### 5.3 REINFORCE 算法伪代码

```
Algorithm: REINFORCE

初始化策略网络 π_θ

For episode = 1, 2, ...:
  # 1. 采集一条完整轨迹
  τ = []
  s = env.reset()
  while not done:
    a ~ π_θ(·|s)
    s', r, done = env.step(a)
    τ.append((s, a, r))
    s = s'

  # 2. 计算每步的 reward-to-go
  For t = T-1, T-2, ..., 0:
    G_t = r_t + γ * G_{t+1}    (G_T = 0)

  # 3. 计算策略梯度并更新
  loss = -Σ_t log π_θ(a_t|s_t) * G_t
  θ ← θ - α * ∇_θ loss
```

注意 loss 取负号是因为 PyTorch 做的是最小化，而我们要最大化 $J(\theta)$。

### 5.4 REINFORCE 的 PyTorch 实现思路

```python
# 采集轨迹后
log_probs = []  # 保存每步的 log π_θ(a_t|s_t)
rewards = []    # 保存每步的奖励

for t in range(T):
    logits = policy_net(s_t)
    dist = Categorical(logits=logits)
    a_t = dist.sample()
    log_probs.append(dist.log_prob(a_t))
    s_t, r_t, done, _ = env.step(a_t.item())
    rewards.append(r_t)

# 计算 reward-to-go
returns = []
G = 0
for r in reversed(rewards):
    G = r + gamma * G
    returns.insert(0, G)
returns = torch.tensor(returns)

# 计算 loss 并更新
loss = 0
for log_prob, G_t in zip(log_probs, returns):
    loss += -log_prob * G_t

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 六、高方差问题

### 6.1 为什么 REINFORCE 方差很高

REINFORCE 使用蒙特卡洛回报 $G_t$ 来估计梯度。$G_t$ 的问题是：

1. **一条轨迹的回报波动很大**：同一个状态-动作对，在不同轨迹中可能得到很不同的 $G_t$
2. **$G_t$ 是多步奖励之和**：每一步的随机性都在叠加
3. **需要等完整轨迹结束**：无法在线更新，样本效率低

直觉类比：想象你用 1 次掷骰子的结果来估计期望。REINFORCE 就是用少量轨迹来估计高维积分。

### 6.2 方差的数学分析

策略梯度的方差来自 $G_t$ 的方差：

$$\text{Var}[\hat{g}] \propto \text{Var}[G_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)]$$

由于 $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots$ 是一个长期累积量，其方差可以很大。

高方差意味着：
- 需要更多样本才能得到可靠的梯度估计
- 训练不稳定，学习率不能太大
- 收敛速度慢

### 6.3 减方差的三条路

| 方法 | 思路 | 详细说明 |
|------|------|---------|
| 因果性（reward-to-go） | 只用未来的奖励 | 已在 5.2 节讨论 |
| **Baseline** | 减去不改变期望的基线值 | 下一节详细推导 |
| Actor-Critic | 用 Critic 网络估计回报，减少蒙特卡洛方差 | Day 5 |

---

## 七、Baseline 减方差 — 数学推导

### 7.1 带 Baseline 的策略梯度

在策略梯度中引入一个与动作无关的基线函数 $b(s_t)$：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t)) \right]$$

### 7.2 证明：Baseline 不改变梯度期望

需要证明 $\mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s) \cdot b(s)] = 0$。

**证明**：

$$\mathbb{E}_{a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a \mid s) \cdot b(s)]$$

$$= b(s) \cdot \mathbb{E}_{a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a \mid s)]$$

（$b(s)$ 与 $a$ 无关，提出来）

$$= b(s) \cdot \sum_a \pi_\theta(a \mid s) \cdot \nabla_\theta \log \pi_\theta(a \mid s)$$

$$= b(s) \cdot \sum_a \pi_\theta(a \mid s) \cdot \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)}$$

$$= b(s) \cdot \sum_a \nabla_\theta \pi_\theta(a \mid s)$$

$$= b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a \mid s)$$

$$= b(s) \cdot \nabla_\theta 1 = 0 \quad \blacksquare$$

关键步骤是 $\sum_a \pi_\theta(a \mid s) = 1$（概率归一化），其梯度为 0。

### 7.3 最优 Baseline

什么样的 $b(s)$ 能最大程度减少方差？可以证明最优 baseline 近似为：

$$b^*(s) \approx \mathbb{E}[G_t \mid S_t = s] = V^\pi(s)$$

即状态价值函数。直觉：$(G_t - V^\pi(s_t))$ 表示这条轨迹比「平均水平」好多少。

### 7.4 引入 Advantage 函数

当 baseline 取 $V^\pi(s)$ 时，策略梯度变为：

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - V^\pi(s_t)) \right]$$

其中 $G_t - V^\pi(s_t)$ 的期望（对 $a_t$）恰好就是 **Advantage 函数**：

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

含义：在状态 $s$ 选动作 $a$ 比「平均动作」好多少。

这就是 Actor-Critic 和 PPO 的理论入口——Day 5 将详细展开。

---

## 八、不同形式的策略梯度

总结策略梯度的不同形式，它们梯度期望相同，但方差不同：

| 形式 | 策略梯度中的信号 | 方差 | 偏差 |
|------|----------------|------|------|
| 总回报 $R(\tau)$ | 最高 | 无 |
| Reward-to-go $G_t$ | 高 | 无 |
| $G_t - b(s_t)$ | 中等 | 无 |
| Advantage $A^\pi(s_t, a_t)$ | 低 | 无（如果 V 精确） |
| TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | 最低 | 有（来自 V 的近似） |

方差和偏差的 trade-off 是 RL 中的核心张力，GAE（Day 5）正是对此的优雅解答。

---

## 九、On-Policy vs Off-Policy

### 9.1 REINFORCE 是 On-Policy 的

策略梯度公式 $\mathbb{E}_{\tau \sim \pi_\theta}[\ldots]$ 要求轨迹来自**当前策略 $\pi_\theta$**。每次更新 $\theta$ 后，之前的数据就「过时」了。

这意味着 REINFORCE 的数据效率极低——采集的数据只能用一次。

### 9.2 与 DQN 的对比

| 维度 | DQN (Off-Policy) | REINFORCE (On-Policy) |
|------|------------------|-----------------------|
| 数据使用 | 存入 Buffer 反复使用 | 用完即弃 |
| 采样效率 | 高 | 低 |
| 稳定性 | 目标网络保证稳定 | 高方差导致不稳定 |
| 适用范围 | 仅离散动作 | 离散 + 连续 |

### 9.3 重要性采样预览

能否用「旧策略」采集的数据来更新「新策略」？这需要**重要性采样**：

$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$$

PPO 正是基于重要性采样实现了一种「近似 off-policy」的方法，使数据可以被复用多次。这是 Day 6 的核心内容。

---

## 十、REINFORCE 的完整数学总结

将所有推导串联起来：

**目标函数**：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \gamma^t r_t\right]$$

**策略梯度定理**（通过 log-derivative trick 推导）：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right]$$

**带 Baseline**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t))\right]$$

**最优 Baseline → Advantage**：

$$b^*(s_t) = V^\pi(s_t), \quad \text{信号} \approx A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

---

## 十一、自检题

1. DQN 为什么无法处理连续动作空间？策略梯度方法如何解决？
2. 用自己的话解释 log-derivative trick。
3. 完整推导策略梯度定理（从 $J(\theta) = \mathbb{E}_\tau[R(\tau)]$ 出发）。
4. 策略梯度公式中为什么不需要知道环境动力学 $P(s' \mid s, a)$？
5. 什么是 reward-to-go？它为什么比使用整条轨迹的 $R(\tau)$ 好？
6. 证明 baseline $b(s)$ 不改变策略梯度的期望。
7. 为什么最优 baseline 近似等于 $V^\pi(s)$？
8. Advantage 函数 $A(s, a) = Q(s, a) - V(s)$ 的直觉含义是什么？
9. REINFORCE 为什么是 on-policy 的？这有什么缺点？
10. 重要性采样如何帮助策略梯度方法复用数据？

---

## 十二、产出要求

- [ ] 解释 DQN 的局限性如何引出策略梯度方法
- [ ] 写出策略参数化的两种形式（离散 / 连续）
- [ ] **完整推导策略梯度定理（面试高频！）**
- [ ] 解释 log-derivative trick 的数学本质
- [ ] 写出 REINFORCE 算法伪代码
- [ ] **证明 baseline 不改变梯度期望**
- [ ] 理解 Advantage 函数的含义
- [ ] 理解 on-policy 的局限性和重要性采样的动机

---

## 十三、与 Day 5 的衔接

今天我们得到了策略梯度的完整理论，也看到了 REINFORCE 的两大问题：

1. **高方差**：蒙特卡洛回报 $G_t$ 的方差很大
2. **低数据效率**：on-policy 数据用完即弃

Day 5 将引入 **Actor-Critic** 框架来解决这两个问题：
- 用 **Critic 网络** $V_\phi(s)$ 替代蒙特卡洛回报，显著降低方差
- 推导 **GAE（Generalized Advantage Estimation）**，在方差和偏差之间找到最优平衡
- 从 A2C 到 TRPO 再到 PPO 的演进，最终在 Day 6 手写 PPO
