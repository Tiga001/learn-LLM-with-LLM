# Day 2：DQN 原理详解 — 从表格法到深度强化学习

> **目标**：理解 Q-Learning 表格法在大规模问题上的局限，掌握 DQN 的三大核心创新——函数逼近、经验回放、目标网络；推导 DQN 的 TD Loss；理解 $\epsilon$-greedy 探索策略；分析 DQN 的局限性（Q 值高估、连续动作空间不适用），为 Day 3 的手写 DQN 实现打下理论基础。

---

## 一、从 Q-Learning 到 DQN

### 1.1 Q-Learning 回顾

Day 1 我们预览了 Q-Learning 的更新公式：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

其中方括号内的 $\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ 称为 **TD error**（时序差分误差）。

Q-Learning 的核心思想：
1. 维护一个 Q 表 $Q[s][a]$
2. 每次与环境交互得到 $(s, a, r, s')$
3. 用 TD target $y = r + \gamma \max_{a'} Q(s', a')$ 来更新 $Q(s, a)$

### 1.2 表格法的困难

Q-Learning 用一张表存储所有 $(s, a)$ 对的 Q 值。当状态空间很大或连续时，这完全不可行：

| 任务 | 状态空间大小 | Q 表大小 |
|------|------------|---------|
| 小型 GridWorld | $3 \times 3 = 9$ | 可行（$9 \times 4 = 36$ 项） |
| Atari 游戏（84×84 像素） | $256^{84 \times 84} \approx 10^{16886}$ | 不可能存储 |
| 机器人控制（连续状态） | $\infty$ | 无法定义表格 |
| LLM token 生成 | 词表大小 $\times$ 序列长度组合 | 天文数字 |

**解决方案**：用**神经网络**逼近 Q 函数 $Q_\theta(s, a) \approx Q^*(s, a)$。

### 1.3 DQN 的历史地位

2013 年，DeepMind 发表 *Playing Atari with Deep Reinforcement Learning*，首次用深度神经网络成功逼近 Q 函数，在多个 Atari 游戏上达到甚至超越人类水平。

2015 年，改进版发表在 Nature 上（*Human-level control through deep reinforcement learning*），成为深度强化学习的里程碑。

---

## 二、DQN 的三大创新

直接用神经网络替换 Q 表会面临严重的训练不稳定问题。DQN 通过三个关键创新解决：

### 2.1 创新一：用神经网络逼近 Q 函数

用参数为 $\theta$ 的网络 $Q_\theta(s, a)$ 替代 Q 表。

**两种架构设计**：

```
方案 A：输入 (s, a)，输出标量 Q(s,a)
  [s, a] → MLP → Q值

方案 B（DQN 采用）：输入 s，输出所有动作的 Q 值
  [s] → MLP → [Q(s,a₁), Q(s,a₂), ..., Q(s,aₙ)]
```

DQN 采用方案 B，因为一次前向传播就能得到所有动作的 Q 值，选动作时直接取 argmax 即可。

对于 Atari，网络架构为 CNN + FC：

```
84×84×4 帧 → Conv1 → Conv2 → Conv3 → FC(512) → FC(|A|)
```

对于低维状态（如 CartPole），用简单的 MLP 就够了：

```
状态向量(4维) → FC(128) → ReLU → FC(128) → ReLU → FC(|A|=2)
```

### 2.2 创新二：经验回放（Experience Replay）

**问题**：在线学习中，连续的样本 $(s_t, a_t, r_t, s_{t+1})$ 高度相关（时间相邻的状态往往相似），违反了随机梯度下降对 i.i.d. 数据的假设，导致训练不稳定。

**解决方案**：维护一个 Replay Buffer $\mathcal{D}$，存储历史经验，训练时从中均匀随机采样。

```
经验回放流程:

1. 智能体与环境交互，产生经验 (s, a, r, s', done)
2. 将经验存入 Buffer D（先进先出）
3. 训练时，从 D 中随机采样 mini-batch
4. 用 mini-batch 计算 loss 并更新网络
```

**经验回放的好处**：

| 好处 | 解释 |
|------|------|
| 打破时间相关性 | 随机采样打破了样本间的时间依赖 |
| 数据效率 | 每条经验可被重复使用多次 |
| 稳定训练 | mini-batch 的方差更小 |

**Replay Buffer 的数据结构**：

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
```

Buffer 容量通常在 $10^4 \sim 10^6$，需要在内存效率和数据多样性之间权衡。

### 2.3 创新三：目标网络（Target Network）

**问题**：在 Q-Learning 更新中，TD target $y = r + \gamma \max_{a'} Q_\theta(s', a')$ 和当前估计 $Q_\theta(s, a)$ 使用的是**同一个网络**。每次更新 $\theta$ 时，target 也在变——相当于在追一个不断移动的目标，导致训练振荡甚至发散。

**解决方案**：维护两个网络：

| 网络 | 参数 | 用途 | 更新频率 |
|------|------|------|---------|
| 在线网络（Online） | $\theta$ | 选动作、计算当前 Q 值 | 每步梯度更新 |
| 目标网络（Target） | $\theta^-$ | 计算 TD target | 每隔 $C$ 步同步 |

TD target 变为：

$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

由于 $\theta^-$ 固定不动（只每隔 $C$ 步更新），target 是一个"相对静止的靶子"，训练更稳定。

**两种同步策略**：

**硬更新**：每 $C$ 步直接复制

$$\theta^- \leftarrow \theta \quad (\text{every } C \text{ steps})$$

**软更新**（Polyak averaging，也称为指数移动平均）：每步缓慢更新

$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-, \quad \tau \in [0.001, 0.01]$$

DQN 原始论文使用硬更新（$C = 10000$），后续工作中软更新更常用。

---

## 三、DQN Loss 推导

### 3.1 目标函数

DQN 的训练目标是最小化 TD error 的均方误差：

$$L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( y - Q_\theta(s, a) \right)^2 \right]$$

其中 TD target：

$$y = \begin{cases} r, & \text{if } s' \text{ is terminal} \\ r + \gamma \max_{a'} Q_{\theta^-}(s', a'), & \text{otherwise} \end{cases}$$

注意 $y$ 不对 $\theta$ 求梯度（target 视为常量），所以梯度为：

$$\nabla_\theta L = \mathbb{E} \left[ -2(y - Q_\theta(s, a)) \cdot \nabla_\theta Q_\theta(s, a) \right]$$

### 3.2 为什么用 MSE 而不是 MAE

MSE 对大误差惩罚更重，有助于快速纠正严重错误的 Q 估计。但在实践中，为了避免梯度爆炸，有时会使用 Huber Loss（smooth L1）：

$$L_\delta(u) = \begin{cases} \frac{1}{2} u^2, & |u| \leq \delta \\ \delta(|u| - \frac{1}{2}\delta), & |u| > \delta \end{cases}$$

Huber Loss 在小误差时等价于 MSE，在大误差时退化为 MAE，提供了更好的鲁棒性。

### 3.3 完整的梯度更新步骤

```
1. 从 Replay Buffer D 中采样 mini-batch {(sᵢ, aᵢ, rᵢ, s'ᵢ, doneᵢ)}

2. 计算 TD target:
   yᵢ = rᵢ + γ * (1 - doneᵢ) * max_{a'} Q_{θ⁻}(s'ᵢ, a')

3. 计算当前 Q 值:
   Q_current = Q_θ(sᵢ, aᵢ)

4. 计算 loss:
   L = (1/N) Σ (yᵢ - Q_current)²

5. 反向传播更新 θ:
   θ ← θ - α ∇_θ L

6. 每隔 C 步同步目标网络:
   θ⁻ ← θ
```

---

## 四、$\epsilon$-Greedy 探索策略

### 4.1 探索与利用的困境

如果智能体总是选 Q 值最大的动作（纯利用），它可能永远发现不了更好的策略。如果总是随机探索，又利用不了已有知识。

### 4.2 $\epsilon$-Greedy 策略

$$a_t = \begin{cases} \text{random action}, & \text{with probability } \epsilon \\ \arg\max_a Q_\theta(s_t, a), & \text{with probability } 1 - \epsilon \end{cases}$$

### 4.3 $\epsilon$ 衰减

训练初期需要大量探索，后期应更多利用。常用**线性衰减**：

$$\epsilon_t = \max(\epsilon_{\min}, \epsilon_{\text{start}} - \frac{t}{T} \cdot (\epsilon_{\text{start}} - \epsilon_{\min}))$$

典型参数：

| 参数 | 值 | 含义 |
|------|---|------|
| $\epsilon_{\text{start}}$ | 1.0 | 初始完全随机 |
| $\epsilon_{\min}$ | 0.01 | 最终保留少量探索 |
| $T$ | 10000 | 衰减步数 |

```
ε
1.0 ┤ ⣇
    │ ⣇
    │  ⢣
    │   ⠑⠢⡀
    │      ⠈⠢⡀
    │         ⠈⠢⣀
0.01├─────────────⠈⠉⠉⠉⠉⠉⠉⠉⠉
    └──────────────────────── step
                T
```

---

## 五、DQN 算法完整伪代码

```
Algorithm: Deep Q-Network (DQN)

初始化:
  在线网络 Q_θ（随机初始化）
  目标网络 Q_{θ⁻} ← copy(Q_θ)
  经验回放 Buffer D（容量 N）
  探索率 ε = 1.0

For episode = 1, 2, ...:
  s₀ = env.reset()

  For t = 0, 1, ...:
    # 1. 选动作 (ε-greedy)
    if random() < ε:
      a_t = random action
    else:
      a_t = argmax_a Q_θ(s_t, a)

    # 2. 与环境交互
    s_{t+1}, r_t, done = env.step(a_t)

    # 3. 存入 Replay Buffer
    D.push(s_t, a_t, r_t, s_{t+1}, done)

    # 4. 采样 mini-batch 训练
    if len(D) >= batch_size:
      batch = D.sample(batch_size)
      y = r + γ * (1 - done) * max_{a'} Q_{θ⁻}(s', a')
      loss = MSE(Q_θ(s, a), y)
      优化器更新 θ

    # 5. 同步目标网络
    if t % C == 0:
      θ⁻ ← θ

    # 6. 衰减 ε
    ε = max(ε_min, ε - Δε)

    if done: break
```

---

## 六、DQN 的改进变体

### 6.1 Double DQN

**问题**：DQN 中 $\max_{a'} Q_{\theta^-}(s', a')$ 倾向于**高估** Q 值。因为 max 操作本身会引入正偏差——即使所有 Q 估计都有均值为 0 的噪声，max 也会选出被高估的那个。

**解决方案**：将"选动作"和"评估动作"解耦：

$$y = r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a'))$$

- **在线网络** $Q_\theta$ 选出最优动作 $a^* = \arg\max_{a'} Q_\theta(s', a')$
- **目标网络** $Q_{\theta^-}$ 评估该动作的 Q 值

### 6.2 Dueling DQN

将 Q 值分解为两部分：

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')$$

- $V(s)$：状态价值（这个状态本身有多好）
- $A(s, a)$：优势函数（在这个状态下选这个动作的相对优势）

好处：很多状态下，不管选什么动作，价值变化不大（如安全区域），Dueling 能更好地估计 $V(s)$。

### 6.3 Prioritized Experience Replay

不再均匀采样，而是**优先采样 TD error 大**的经验：

$$P(i) \propto |\delta_i|^\alpha$$

直觉：TD error 大的样本"最出乎意料"，能提供最多的学习信号。

### 6.4 Rainbow DQN

将上述所有改进（Double + Dueling + Prioritized + 多步回报 + 分布式 Q + Noisy Nets）组合起来，在 Atari 上取得了最强性能。

---

## 七、DQN 的根本局限

### 7.1 只能处理离散动作空间

DQN 的核心操作是 $\max_a Q(s, a)$——需要遍历所有动作。当动作空间是连续的（如机器人关节角度），无法直接使用。

### 7.2 无法直接输出随机策略

DQN 学到的策略是确定性的（总是选 Q 最大的动作，加上 $\epsilon$-greedy 只是为了探索）。但很多场景需要**随机策略**：
- 博弈中需要混合策略
- RLHF 中 LLM 需要输出 token 概率分布

### 7.3 引出策略梯度方法

DQN 的这些局限正是策略梯度方法的动机：

| 方法 | 学什么 | 动作空间 | 策略类型 |
|------|--------|---------|---------|
| DQN | Q 函数 → 间接得到策略 | 离散 | 确定性 |
| Policy Gradient | 直接优化策略 $\pi_\theta$ | 离散/连续 | 随机 |

这将是 Day 4 的主题。

---

## 八、DQN 超参数一览

| 超参数 | 典型值 | 作用 |
|--------|--------|------|
| 学习率 $\alpha$ | $10^{-4} \sim 10^{-3}$ | 梯度更新步长 |
| 折扣因子 $\gamma$ | 0.99 | 未来奖励权重 |
| Buffer 容量 $N$ | $10^4 \sim 10^6$ | 经验回放容量 |
| Batch size | 32 ~ 128 | 每次训练采样数 |
| 目标网络同步频率 $C$ | 100 ~ 10000 | 硬更新间隔 |
| 软更新系数 $\tau$ | 0.001 ~ 0.01 | Polyak 平均系数 |
| $\epsilon_{\text{start}}$ | 1.0 | 初始探索率 |
| $\epsilon_{\min}$ | 0.01 ~ 0.1 | 最终探索率 |
| $\epsilon$ 衰减步数 $T$ | 1000 ~ 100000 | 探索衰减期 |

---

## 九、自检题

1. 为什么 Q-Learning 的表格法在 Atari 游戏上不可行？
2. DQN 的三大创新分别解决了什么问题？
3. 经验回放为什么能打破样本的时间相关性？
4. 如果没有目标网络，训练会出现什么问题？用你自己的话解释。
5. 写出 DQN 的 TD target 公式，说明什么时候 $y = r$。
6. $\epsilon$-greedy 中 $\epsilon$ 为什么需要衰减？
7. Double DQN 如何缓解 Q 值高估问题？
8. Dueling DQN 将 Q 分解为哪两部分？这样做的好处是什么？
9. DQN 为什么不能处理连续动作空间？
10. DQN 的哪些局限推动了策略梯度方法的发展？

---

## 十、产出要求

- [ ] 说明 Q-Learning 表格法在高维问题上的困难
- [ ] 画出 DQN 的完整架构图（含在线网络、目标网络、Replay Buffer）
- [ ] 推导 DQN Loss：$L = \mathbb{E}[(y - Q_\theta(s, a))^2]$
- [ ] 解释经验回放的工作原理和好处
- [ ] 解释目标网络的硬更新与软更新
- [ ] 写出 $\epsilon$-greedy 的数学定义
- [ ] 了解 Double DQN 和 Dueling DQN 的核心改进
- [ ] 理解 DQN 的局限性，为 Day 4 策略梯度做准备

---

## 十一、与 Day 3 的衔接

今天我们理解了 DQN 的原理和工程设计。明天 Day 3 将从零实现：

1. `ReplayBuffer` 类
2. DQN 网络（2 层 MLP）
3. $\epsilon$-greedy 选动作
4. 训练循环：采样 → TD target → loss → 更新 → 同步目标网络
5. 在 CartPole 上训练，目标：稳定获得 >400 分
6. 可视化训练曲线

理论和代码的结合是真正掌握 DQN 的关键。
