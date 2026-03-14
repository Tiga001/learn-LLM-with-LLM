# Day 5：Actor-Critic 与 GAE — 方差与偏差的优雅权衡

> **目标**：掌握 Actor-Critic 框架的核心思想——用 Critic 网络替代蒙特卡洛回报来估计策略梯度信号；深入理解 Advantage 函数的意义；从 TD(0) 到 TD($\lambda$)，严格推导 GAE（Generalized Advantage Estimation）；掌握 A2C 算法；理解从 A2C 到 TRPO 再到 PPO 的演进动机。本日是 Day 6 手写 PPO 的直接理论前置。

---

## 一、REINFORCE 的问题回顾

Day 4 我们推导了策略梯度定理：

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]$$

其中 $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$ 是蒙特卡洛回报。

REINFORCE 的两个核心问题：

| 问题 | 原因 | 后果 |
|------|------|------|
| 高方差 | $G_t$ 是多步奖励之和，每步都有随机性 | 梯度估计噪声大，训练不稳定 |
| 低效率 | 必须等完整 episode 结束才能更新 | 无法在线学习，数据利用率低 |

Actor-Critic 的核心思想：**用一个学习出来的价值函数 $V_\phi(s)$ 来替代（或辅助）蒙特卡洛回报**。

---

## 二、Actor-Critic 框架

### 2.1 两个网络，各司其职

```
Actor（策略网络）π_θ:  决定"做什么" — 输出动作概率分布
Critic（价值网络）V_ϕ:  评价"做得好不好" — 估计状态价值

    ┌──────────────────────────────────────────┐
    │            Actor-Critic 框架              │
    │                                          │
    │   状态 s ──┬──→ Actor π_θ(a|s) ──→ 动作 a│
    │            │                              │
    │            └──→ Critic V_ϕ(s) ──→ 价值估计│
    │                                          │
    │   环境返回 r, s'                          │
    │                                          │
    │   Advantage = f(r, V_ϕ(s), V_ϕ(s'))      │
    │                                          │
    │   更新 Actor:  用 Advantage 信号          │
    │   更新 Critic: 用 TD error 或 MC return   │
    └──────────────────────────────────────────┘
```

### 2.2 为什么叫 Actor-Critic

- **Actor** 像"演员"——它做出动作
- **Critic** 像"评论家"——它评价动作的好坏

Critic 的评价帮助 Actor 更高效地学习，而不是等到 episode 结束才知道好坏。

### 2.3 与 RLHF 的对应

| 组件 | RL 中 | RLHF 中 |
|------|-------|---------|
| Actor $\pi_\theta$ | 策略网络 | LLM（被优化的模型） |
| Critic $V_\phi$ | 价值网络 | Value Head（估计期望回报） |
| Reward | 环境奖励 | Reward Model 的打分 |
| Reference | — | 冻结的参考模型（KL 约束） |

RLHF 中实际有 4 个模型：Actor + Critic + Reward Model + Reference Model。理解 Actor-Critic 是理解 RLHF 架构的基础。

---

## 三、Advantage 函数

### 3.1 定义

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

含义：在状态 $s$ 下，选动作 $a$ 比「按策略 $\pi$ 的平均表现」好多少。

- $A(s, a) > 0$：动作 $a$ 比平均好，应该增大其概率
- $A(s, a) < 0$：动作 $a$ 比平均差，应该减小其概率
- $A(s, a) = 0$：动作 $a$ 与平均水平一致

### 3.2 性质

**关键性质**：Advantage 对动作取期望为 0：

$$\mathbb{E}_{a \sim \pi}[A^\pi(s, a)] = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a) - V^\pi(s)] = V^\pi(s) - V^\pi(s) = 0$$

这意味着 Advantage 天然是一个「去中心化」的信号，方差比 $Q$ 或 $G_t$ 更小。

### 3.3 策略梯度的 Advantage 形式

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A^\pi(s_t, a_t) \right]$$

问题是：我们不知道真正的 $A^\pi$，需要估计它。

---

## 四、Advantage 的估计方法

### 4.1 蒙特卡洛估计

$$\hat{A}_t^{\text{MC}} = G_t - V_\phi(s_t)$$

- $G_t$ 是蒙特卡洛回报（无偏）
- $V_\phi(s_t)$ 是 Critic 的估计（作为 baseline）
- 优点：无偏
- 缺点：高方差（$G_t$ 方差大）

### 4.2 TD(0) 估计

$$\hat{A}_t^{\text{TD}(0)} = \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

$\delta_t$ 是一步 TD error。

- 优点：方差低（只用了一步的随机性）
- 缺点：有偏（$V_\phi$ 不完美时引入偏差）

### 4.3 方差-偏差 Trade-off

```
蒙特卡洛:   G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...    ← 无偏，高方差
              ↑ 完全用真实奖励，不依赖 V_ϕ

TD(0):      δ_t = r_t + γV_ϕ(s_{t+1}) - V_ϕ(s_t)       ← 有偏，低方差
              ↑ 只用一步真实奖励，其余靠 V_ϕ 估计

问题：能否找到一个介于两者之间的最优估计？
答案：GAE。
```

---

## 五、多步 TD 估计

在推导 GAE 之前，先理解多步 TD 的思想。

### 5.1 $n$-step 回报

1-step: $G_t^{(1)} = r_t + \gamma V(s_{t+1})$

2-step: $G_t^{(2)} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2})$

$n$-step: $G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})$

$\infty$-step: $G_t^{(\infty)} = G_t$（蒙特卡洛回报）

### 5.2 对应的 $n$-step Advantage

$$\hat{A}_t^{(1)} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t^{(2)} = \delta_t + \gamma \delta_{t+1} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)$$

$$\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}$$

### 5.3 不同 $n$ 的特性

| $n$ | 偏差 | 方差 |
|-----|------|------|
| 1 (TD(0)) | 最大（最依赖 $V_\phi$） | 最小 |
| 中间 | 中间 | 中间 |
| $\infty$ (MC) | 零 | 最大 |

我们需要一种方法来**自动权衡**不同的 $n$——这就是 GAE。

---

## 六、GAE 完整推导

### 6.1 核心思想

GAE 对所有 $n$-step advantage 做**指数加权平均**，用参数 $\lambda \in [0, 1]$ 控制权重：

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \hat{A}_t^{(n)}$$

其中 $(1-\lambda)$ 是归一化系数，保证权重之和为 1。

### 6.2 展开推导

**Step 1**：写出各 $n$-step advantage：

$$\hat{A}_t^{(1)} = \delta_t$$
$$\hat{A}_t^{(2)} = \delta_t + \gamma \delta_{t+1}$$
$$\hat{A}_t^{(3)} = \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2}$$
$$\vdots$$
$$\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}$$

**Step 2**：代入 GAE 公式：

$$\hat{A}_t^{\text{GAE}} = (1-\lambda) \left[ \lambda^0 \delta_t + \lambda^1 (\delta_t + \gamma \delta_{t+1}) + \lambda^2 (\delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2}) + \cdots \right]$$

**Step 3**：收集 $\delta_{t+l}$ 的系数。$\delta_{t+l}$ 出现在所有 $n > l$ 的项中，且在 $n$-step advantage 中的系数为 $\gamma^l$，在 GAE 中被 $\lambda^{n-1}$ 加权。

因此 $\delta_{t+l}$ 在 GAE 中的总系数为：

$$(1-\lambda) \cdot \gamma^l \sum_{n=l+1}^{\infty} \lambda^{n-1} = (1-\lambda) \cdot \gamma^l \cdot \frac{\lambda^l}{1-\lambda} = (\gamma\lambda)^l$$

**Step 4**：得到 GAE 的简洁公式：

$$\boxed{\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}}$$

其中 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ 是一步 TD error。

### 6.3 GAE 的递推计算

在实际计算中，从后往前递推更高效：

$$\hat{A}_T = 0$$
$$\hat{A}_t = \delta_t + \gamma\lambda \cdot \hat{A}_{t+1}$$

这和「折扣回报 $G_t = r_t + \gamma G_{t+1}$」的递推结构完全类似，只是把 $\gamma$ 换成了 $\gamma\lambda$，把 $r_t$ 换成了 $\delta_t$。

### 6.4 实现伪代码

```python
def compute_gae(rewards, values, next_values, dones, gamma, gae_lambda):
    """
    rewards:     [r_0, r_1, ..., r_{T-1}]
    values:      [V(s_0), V(s_1), ..., V(s_{T-1})]
    next_values: [V(s_1), V(s_2), ..., V(s_T)]
    dones:       [done_0, done_1, ..., done_{T-1}]
    """
    T = len(rewards)
    advantages = [0] * T
    last_gae = 0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        last_gae = advantages[t]

    return advantages
```

### 6.5 $\lambda$ 的两个极端

| $\lambda$ | GAE | 等价于 | 偏差 | 方差 |
|-----------|-----|--------|------|------|
| $\lambda = 0$ | $\hat{A}_t = \delta_t$ | TD(0) | 高 | 低 |
| $\lambda = 1$ | $\hat{A}_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} = G_t - V(s_t)$ | MC | 无 | 高 |
| $\lambda \in (0, 1)$ | 指数加权平均 | 介于两者之间 | 中 | 中 |

实际中 $\lambda = 0.95 \sim 0.99$ 效果较好。PPO 原始论文使用 $\lambda = 0.95$。

### 6.6 GAE 的直觉

```
λ = 0 (只看一步):
  "我刚做了一步，Critic 说下一个状态值多少，够了。"
  → 可能短视，但估计稳定

λ = 1 (看完整轨迹):
  "让我看完整条轨迹，用实际回报来判断。"
  → 看得最远，但估计嘈杂

λ = 0.95 (典型值):
  "我主要关注近处的信息，但也适当参考远处的结果。"
  → 在远见和稳定之间取平衡
```

---

## 七、A2C 算法

### 7.1 A2C = Advantage Actor-Critic

A2C 是 Actor-Critic 框架的标准实现，使用 Advantage 估计（通常是 GAE）作为策略梯度信号。

### 7.2 A2C 的三个 Loss

**1. Policy Loss（Actor 的损失）**：

$$L^{\text{policy}}(\theta) = -\mathbb{E}_t \left[ \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t \right]$$

负号因为我们要最大化目标（PyTorch 做最小化）。

**2. Value Loss（Critic 的损失）**：

$$L^{\text{value}}(\phi) = \mathbb{E}_t \left[ (V_\phi(s_t) - G_t^{\text{target}})^2 \right]$$

其中 $G_t^{\text{target}} = \hat{A}_t + V_\phi(s_t)$（即 GAE advantage 加上 baseline 还原出目标值）。

**3. Entropy Bonus**：

$$L^{\text{entropy}}(\theta) = -\mathbb{E}_t \left[ \mathcal{H}[\pi_\theta(\cdot \mid s_t)] \right]$$

鼓励策略保持一定的探索性（防止过早收敛到确定性策略）。

**总损失**：

$$L = L^{\text{policy}} + c_1 \cdot L^{\text{value}} - c_2 \cdot \mathcal{H}[\pi_\theta]$$

其中 $c_1 \approx 0.5$（value loss 系数），$c_2 \approx 0.01$（entropy 系数）。

### 7.3 A2C 算法伪代码

```
Algorithm: A2C (Advantage Actor-Critic)

初始化 Actor π_θ 和 Critic V_ϕ（可共享 backbone）

For iteration = 1, 2, ...:
  # 1. Rollout: 采集 N 步交互数据
  For t = 0, 1, ..., N-1:
    a_t ~ π_θ(·|s_t)
    s_{t+1}, r_t, done_t = env.step(a_t)
    存储 (s_t, a_t, r_t, s_{t+1}, done_t, log π_θ(a_t|s_t))

  # 2. 计算 GAE advantage
  advantages = compute_gae(rewards, values, next_values, dones, γ, λ)
  returns = advantages + values   # 目标回报

  # 3. 计算三个 loss
  policy_loss = -mean(log_probs * advantages)
  value_loss = mean((V_ϕ(s) - returns)²)
  entropy = mean(H[π_θ(·|s)])
  loss = policy_loss + c₁ * value_loss - c₂ * entropy

  # 4. 梯度更新
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
```

### 7.4 参数共享

Actor 和 Critic 可以共享底层特征提取网络（backbone），只在最后一层分叉：

```
状态 s → 共享 MLP backbone → ┬─→ Actor Head → π(a|s)
                              └─→ Critic Head → V(s)
```

这样做的好处：
- 减少参数量
- 共享的特征表示可能更好
- 训练更高效

---

## 八、从 A2C 到 TRPO

### 8.1 A2C 的问题：步长多大？

A2C 每次用梯度更新策略，但如果步长太大，策略可能突变，导致：
- 性能崩塌：新策略远不如旧策略
- 数据失效：在旧策略下采集的 advantage 不再准确

### 8.2 TRPO 的思想：信赖域

TRPO（Trust Region Policy Optimization, Schulman 2015）的核心思想：**限制每次策略更新的幅度**。

具体来说，在优化时加一个约束——新旧策略的 KL 散度不能太大：

$$\max_\theta \quad \mathbb{E}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right]$$

$$\text{s.t.} \quad \mathbb{E}_t \left[ D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot \mid s_t) \| \pi_\theta(\cdot \mid s_t)) \right] \leq \delta$$

### 8.3 TRPO 的困难

TRPO 需要用二阶优化（计算 Fisher 信息矩阵的逆或用共轭梯度法），实现复杂、计算开销大。

### 8.4 PPO 的动机

PPO 的目标：**用一阶方法（普通梯度下降）近似实现 TRPO 的效果**。

PPO 不用硬约束 KL 散度，而是通过 **clip** 操作来限制策略更新幅度。这就是 Day 6 的核心内容。

---

## 九、从 A2C 到 PPO 的演进总览

```
REINFORCE
  ↓ 问题: 高方差、需要完整 episode
  ↓ 解决: 引入 Critic 估计 value

Actor-Critic (A2C)
  ↓ 问题: 步长敏感，策略可能崩塌
  ↓ 解决: 限制策略更新幅度（信赖域）

TRPO
  ↓ 问题: 二阶优化太复杂
  ↓ 解决: 用 clip 近似信赖域

PPO ← 当前工业标准（RLHF 使用）
```

每一步演进都在解决前一个方法的具体问题：

| 方法 | 解决了什么 | 引入了什么新问题 |
|------|----------|---------------|
| REINFORCE | 直接优化策略 | 高方差 |
| Actor-Critic | 降低方差 | 步长敏感 |
| TRPO | 限制更新幅度 | 实现复杂 |
| **PPO** | **一阶近似信赖域** | **当前最优实践** |

---

## 十、重要性采样与 PPO 预览

### 10.1 重要性采样回顾

Day 4 我们提到了重要性采样的思想。在策略梯度中：

我们想计算 $\mathbb{E}_{a \sim \pi_\theta}[f(a)]$，但数据是从旧策略 $\pi_{\theta_{\text{old}}}$ 采的。

$$\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} f(a) \right]$$

比值 $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ 称为**概率比（probability ratio）**。

### 10.2 PPO-Clip 目标函数预览

PPO 的核心创新是对概率比做 clip，防止更新太大：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

当 $r_t(\theta)$ 偏离 1 太多（即新旧策略差异太大）时，clip 会阻止进一步偏离。

Day 6 将详细推导并从零实现这个目标函数。

---

## 十一、自检题

1. Actor-Critic 中 Actor 和 Critic 分别负责什么？
2. Advantage 函数 $A(s, a) = Q(s, a) - V(s)$ 为什么比直接用 $Q(s, a)$ 更好？
3. 证明 $\mathbb{E}_{a \sim \pi}[A(s, a)] = 0$。
4. 比较 MC 估计和 TD(0) 估计 advantage 的优劣。
5. **推导 GAE 公式：$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$**
6. 写出 GAE 的递推计算公式。
7. $\lambda = 0$ 和 $\lambda = 1$ 分别对应什么估计？
8. A2C 的三个 loss 分别是什么？Entropy bonus 的作用是什么？
9. TRPO 为什么要限制 KL 散度？
10. PPO 如何用 clip 近似实现 TRPO 的效果？（预览）

---

## 十二、产出要求

- [ ] 画出 Actor-Critic 的数据流图
- [ ] 写出 Advantage 函数的定义和性质
- [ ] 比较 MC、TD(0)、$n$-step TD 估计 advantage 的偏差-方差特性
- [ ] **推导 GAE 从加权平均到 $\sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$ 的完整过程**
- [ ] 写出 GAE 的递推实现伪代码
- [ ] 写出 A2C 的三个 loss 公式
- [ ] 理解 A2C → TRPO → PPO 的演进逻辑
- [ ] 为 Day 6 的 PPO 手写做好理论准备

---

## 十三、与 Day 6 的衔接

今天我们完成了 PPO 前的所有理论铺垫：

- **Actor-Critic 框架**：如何训练 Actor 和 Critic
- **GAE**：如何高效且准确地估计 Advantage
- **TRPO → PPO 的动机**：为什么要限制策略更新幅度

Day 6 将把这些全部整合，从零手写 PPO-Clip 算法：
1. 实现共享 backbone 的 Actor-Critic 网络
2. 实现 rollout 采集和 GAE 计算
3. 实现 PPO-Clip 目标函数和 mini-batch 更新
4. 在 CartPole 和 LunarLander 上训练成功
5. 分析 clip ratio、entropy 等超参数的影响

PPO 手写是本周最重要的实践，也是面试高频考点。
