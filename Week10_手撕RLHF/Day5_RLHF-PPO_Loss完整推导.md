# Day 5：RLHF-PPO Loss 完整推导 — 从公式到可实现的代码

> **目标**：完整推导 RLHF-PPO 的三部分 Loss——Policy Loss（PPO-Clip 在 token 级别的适配）、Value Loss（Critic 的回报估计损失）、KL Penalty（约束策略不偏离参考模型）；掌握 GAE 在 RLHF 稀疏奖励场景下的适配；理解 per-token 与 per-sequence 奖励的处理方式；写出完整的 RLHF-PPO 伪代码，使其可直接翻译为 Day 6 的 Python 实现。本日的每一个公式都将对应 Day 6 代码中的一行。

---

## 一、回顾：Day 4 的数据流与 Loss 的关系

### 1.1 Loss 在 RLHF-PPO 迭代中的位置

Day 4 我们梳理了 RLHF-PPO 一轮 iteration 的四个阶段。Loss 计算发生在最后的 **PPO Update 阶段**：

```
Phase 1: Rollout ──→ Phase 2: Scoring ──→ Phase 3: GAE ──→ Phase 4: PPO Update
  Actor 生成           RM 打分              计算 Advantage      ★ 计算 Loss ★
  收集 old_log_probs   Ref 计算 KL          计算 Returns        更新 Actor + Critic
  收集 old_values      构建 per-token reward
```

Phase 1-3 是**数据准备**阶段（`torch.no_grad()`），Phase 4 是**梯度更新**阶段：在这里我们用准备好的数据计算 Loss，反向传播更新 Actor 和 Critic。

### 1.2 三个 Loss 的角色分工

| Loss | 更新谁 | 目标 |
|------|--------|------|
| Policy Loss ($L^{\text{policy}}$) | Actor ($\pi_\theta$) | 让 Actor 生成高 Advantage 的 token |
| Value Loss ($L^{\text{value}}$) | Critic ($V_\phi$) | 让 Critic 准确估计未来回报 |
| KL Penalty | 嵌入 reward | 防止 Actor 偏离 Reference 太远 |

注意 KL Penalty 有两种实现方式（第四节详述）。最主流的做法是将 KL 融入 per-token reward（Day 4 已介绍），而非单独作为 Loss 项。

---

## 二、Policy Loss：PPO-Clip 在 LLM 上的适配

### 2.1 回顾 W9 的 PPO-Clip Loss

Week 9 Day 5 我们推导了经典 PPO-Clip 的目标：

$$L^{\text{CLIP}}(\theta) = -\mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ 是 importance sampling ratio，$\hat{A}_t$ 是 GAE 估计的 Advantage。

这个公式在经典 RL 中的含义：

- $r_t > 1$：新策略在状态 $s_t$ 更倾向于选动作 $a_t$
- $r_t < 1$：新策略不如旧策略倾向选 $a_t$
- Clip 防止 $r_t$ 偏离 1 太远，确保更新稳定

### 2.2 从 MLP 到 LLM：token 级别的映射

在 RLHF 中，PPO-Clip 的每个变量都有对应：

| 经典 PPO (W9) | RLHF-PPO (W10) | 说明 |
|--------------|----------------|------|
| $s_t$ | $(x, y_1, \ldots, y_{t-1})$ | 状态 = prompt + 已生成 token |
| $a_t$ | $y_t$ | 动作 = 下一个 token |
| $\pi_\theta(a_t \mid s_t)$ | $P_\theta(y_t \mid x, y_{<t})$ | LLM 的 next-token 概率 |
| $\pi_{\theta_{\text{old}}}$ | rollout 时的 Actor 概率 | 生成 response 时记录的 log prob |

### 2.3 Importance Sampling Ratio 的计算

在 LLM 场景下，ratio 在每个 token 位置独立计算：

$$r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})}$$

在对数空间计算更数值稳定：

$$\log r_t(\theta) = \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})$$

$$r_t(\theta) = \exp(\log r_t(\theta))$$

其中：
- $\log \pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})$ 是 rollout 时记录的 `old_log_probs`（Phase 1 收集）
- $\log \pi_\theta(y_t \mid x, y_{<t})$ 是 PPO Update 阶段重新前向传播计算的 `new_log_probs`

```python
# PPO Update 阶段
new_logits = actor(input_ids)                        # 重新前向传播
new_log_probs = get_log_probs(new_logits, response_ids)  # 提取对应 token 的 log prob
log_ratio = new_log_probs - old_log_probs            # 对数空间减法
ratio = torch.exp(log_ratio)                         # 转回概率空间
```

### 2.4 Per-token Clip 与 Sequence-level 聚合

RLHF 中有两种聚合策略：

**策略 A：Per-token Loss（更常用）**

在每个 token 上独立计算 clip loss，然后取 response 范围内的平均：

$$L^{\text{policy}} = -\frac{1}{T} \sum_{t=1}^{T} \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)$$

其中 $T$ 是 response 长度，$t$ 只遍历 response 的 token（不含 prompt）。

**策略 B：Per-sequence Loss**

先在 token 上求和，再取 batch 平均：

$$L^{\text{policy}} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{t=1}^{T_i} \min \left( r_t^{(i)} \hat{A}_t^{(i)}, \; \text{clip}(r_t^{(i)}, 1-\epsilon, 1+\epsilon) \hat{A}_t^{(i)} \right)$$

实践中策略 A（per-token 平均）更稳定，因为不同样本的 response 长度差异不会影响梯度尺度。

### 2.5 完整 Policy Loss 公式

$$\boxed{L^{\text{policy}}(\theta) = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)}$$

其中 $\mathcal{M}$ 是 response token 的掩码集合（排除 prompt 和 padding），$r_t(\theta) = \exp(\log \pi_\theta(y_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(y_t \mid s_t))$。

对应代码：

```python
def compute_policy_loss(new_log_probs, old_log_probs, advantages, clip_ratio, response_mask):
    """PPO-Clip Policy Loss。"""
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    
    policy_loss = -torch.min(surr1, surr2)
    
    # 只在 response token 上计算，排除 prompt 和 padding
    policy_loss = (policy_loss * response_mask).sum() / response_mask.sum()
    return policy_loss
```

---

## 三、Value Loss：Critic 学习估计回报

### 3.1 Critic 的目标

Critic $V_\phi(s_t)$ 的目标是预测从时间步 $t$ 开始的期望累积回报 $G_t$（returns）。

回顾 W9：Critic 提供的 baseline 用于降低策略梯度的方差。在 RLHF 中同样——没有准确的 Critic，GAE 计算的 Advantage 就会有大方差，PPO 更新就不稳定。

### 3.2 Value Loss = MSE

最基本的 Value Loss 是预测值与实际 returns 之间的均方误差：

$$L^{\text{value}}(\phi) = \frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \left( V_\phi(s_t) - G_t \right)^2$$

其中 $G_t = \hat{A}_t + V_{\phi_{\text{old}}}(s_t)$ 是 GAE 反推出来的 returns（第五节详述）。

### 3.3 可选：Value Clipping

类似 Policy Loss 的 clipping 思想，可以对 Value 也做 clip，防止 Critic 更新幅度过大：

$$V^{\text{clipped}} = V_{\phi_{\text{old}}}(s_t) + \text{clip}\left(V_\phi(s_t) - V_{\phi_{\text{old}}}(s_t), -\epsilon_v, \epsilon_v\right)$$

$$L^{\text{value-clip}}(\phi) = \frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \max \left( (V_\phi(s_t) - G_t)^2, \; (V^{\text{clipped}} - G_t)^2 \right)$$

Value clipping 在实践中存在争议：
- **支持**：防止 Critic 突然跳变，稳定训练
- **反对**：如果 $V_{\phi_{\text{old}}}$ 本身就不准，clip 会阻碍纠正
- OpenAI 的 PPO 论文中使用了 value clipping，但很多后续工作发现去掉也行

### 3.4 完整 Value Loss 公式

$$\boxed{L^{\text{value}}(\phi) = \frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \left( V_\phi(s_t) - G_t \right)^2}$$

对应代码：

```python
def compute_value_loss(new_values, returns, response_mask):
    """Value Loss (MSE)。"""
    value_loss = (new_values - returns) ** 2
    value_loss = (value_loss * response_mask).sum() / response_mask.sum()
    return value_loss
```

---

## 四、KL Penalty：保持 $\pi_\theta$ 不偏离 $\pi_{\text{ref}}$

### 4.1 为什么需要 KL 约束

回顾 Day 1，RLHF 的优化目标是：

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim D, \, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) \right] - \beta \cdot D_{\text{KL}} \left[ \pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x) \right]$$

如果去掉 KL 项，Actor 会为了最大化 RM 分数而学出各种「作弊」策略（Reward Hacking）——比如不断重复某些高分词汇、生成过长文本等。KL 约束确保 Actor 不会偏离正常语言分布太远。

### 4.2 Per-token KL 的计算

两个策略之间在单个 token 上的 KL 散度：

$$\text{kl}_t = \log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t)$$

这是一个**采样近似**——严格的 KL 需要对整个词表求期望，但由于我们只有采样的 token $y_t$，这里用的是单点估计。完整的序列级 KL：

$$D_{\text{KL}}^{\text{approx}} = \sum_{t=1}^{T} \text{kl}_t = \sum_{t=1}^{T} \left[ \log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t) \right]$$

### 4.3 两种实现方式

#### 方式 A：KL 作为 Reward Shaping（主流）

将 KL penalty 融入 per-token reward 信号：

$$r_t = \begin{cases} -\beta \cdot \text{kl}_t & t < T \\ r_\phi(x, y) - \beta \cdot \text{kl}_T & t = T \end{cases}$$

这是 Day 4 介绍的做法，也是 InstructGPT / trl 库采用的方式。

**优点**：
- KL 自然地参与 GAE 计算，不需要单独处理
- 每个 token 都有 reward（缓解稀疏奖励）
- 实现简洁

#### 方式 B：KL 作为单独 Loss 项

将 KL 直接加到 Loss 函数中：

$$L^{\text{total}} = L^{\text{policy}} + c_v \cdot L^{\text{value}} + \beta \cdot \frac{1}{T} \sum_t \text{kl}_t$$

**优点**：KL 系数 $\beta$ 的影响更直接

**缺点**：GAE 无法利用 KL 信号，稀疏奖励问题仍然存在

### 4.4 两种方式的对比

| 维度 | Reward Shaping (A) | Loss 项 (B) |
|------|-------------------|-------------|
| KL 信号进入 GAE | ✓ | ✗ |
| 缓解稀疏奖励 | ✓（每个 token 都有 reward） | ✗（只有终末 RM score） |
| 实现复杂度 | 低 | 低 |
| 采用者 | InstructGPT, trl, DeepSpeed-Chat | 部分学术工作 |
| 超参数 | $\beta$（KL 系数） | $\beta$（KL 系数） |

**结论**：方式 A 是工业界标准做法，Day 6 我们也将采用这种方式。

```python
def compute_per_token_rewards(rm_scores, log_probs, ref_log_probs, kl_coef):
    """方式 A：KL 作为 Reward Shaping。"""
    kl_per_token = log_probs - ref_log_probs        # (batch, response_len)
    rewards = -kl_coef * kl_per_token                # per-token KL penalty
    rewards[:, -1] += rm_scores                      # 终末 token 加 RM score
    return rewards, kl_per_token
```

---

## 五、GAE 在 RLHF 中的计算

### 5.1 回顾 W9 GAE 公式

Generalized Advantage Estimation（GAE）是在 bias 和 variance 之间做权衡的 Advantage 估计方法：

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}$$

其中 TD 残差为：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- $\gamma$：折扣因子，控制未来奖励的权重
- $\lambda$：GAE 参数，权衡 bias/variance（$\lambda = 0$ 退化为 1-step TD，$\lambda = 1$ 退化为 Monte Carlo）

### 5.2 RLHF 中的特殊性

RLHF 与经典 RL 的关键区别影响 GAE 计算：

| 特点 | 经典 RL (CartPole) | RLHF |
|------|-------------------|------|
| 奖励来源 | 每步环境给 reward | KL penalty + 终末 RM score |
| Episode 长度 | 变长（数百步） | 变长（数十~数百 token） |
| 折扣因子 $\gamma$ | 通常 0.99 | 通常 1.0（response 较短） |
| GAE $\lambda$ | 通常 0.95 | 通常 0.95 |
| 终止条件 | 环境 done 信号 | EOS token 或最大长度 |

**为什么 $\gamma = 1.0$**：RLHF 中 response 长度通常只有几十到几百 token，不需要折扣——每个 token 的 reward 同等重要。

### 5.3 从 $r_t$ 和 $V(s_t)$ 到 $\hat{A}_t$

完整的 GAE 计算流程：

**输入**：
- $r_0, r_1, \ldots, r_{T-1}$：per-token reward（已包含 KL penalty 和终末 RM score）
- $V(s_0), V(s_1), \ldots, V(s_T)$：Critic 在每个 token 位置的估值

**Step 1**：计算 TD 残差

$$\delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t), \quad t = 0, 1, \ldots, T-1$$

对最后一个 token（$t = T-1$），$V(s_T) = 0$（episode 结束）：

$$\delta_{T-1} = r_{T-1} + \gamma \cdot 0 - V(s_{T-1}) = r_{T-1} - V(s_{T-1})$$

**Step 2**：反向递推 GAE

$$\hat{A}_{T-1} = \delta_{T-1}$$

$$\hat{A}_t = \delta_t + \gamma \lambda \cdot \hat{A}_{t+1}, \quad t = T-2, T-3, \ldots, 0$$

**Step 3**：计算 Returns

$$G_t = \hat{A}_t + V(s_t)$$

### 5.4 只有终末奖励 vs per-token KL reward 的影响

如果**不用 KL reward shaping**（方式 B），只有终末 RM score：

$$r_t = \begin{cases} 0 & t < T-1 \\ r_\phi(x, y) & t = T-1 \end{cases}$$

此时 GAE 的 TD 残差：

$$\delta_t = \begin{cases} \gamma V(s_{t+1}) - V(s_t) & t < T-1 \\ r_\phi(x, y) - V(s_{T-1}) & t = T-1 \end{cases}$$

奖励信号只能通过 Critic 的 $V(s_t)$ 逐步反向传播——如果 Critic 不准确，Advantage 估计就会很差。

而**用 KL reward shaping**（方式 A）后，每个 token 的 $r_t = -\beta \cdot \text{kl}_t$（最后加 RM score），GAE 的每个 $\delta_t$ 都有直接的 reward 信号，Advantage 估计更稳定。

### 5.5 完整 GAE 代码

$$\boxed{\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \quad G_t = \hat{A}_t + V(s_t)}$$

```python
def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """
    GAE 计算。
    rewards: (batch, response_len)    - per-token reward
    values:  (batch, response_len)    - Critic 估值
    返回: advantages, returns
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=rewards.device)
    
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0.0  # episode 结束
        else:
            next_value = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae
    
    returns = advantages + values
    return advantages, returns
```

---

## 六、Per-token vs Per-sequence 奖励处理

### 6.1 RM Score 的分配策略

RM 输出的是整个 response 的标量分数 $r_\phi(x, y)$，但 PPO 需要 per-token reward。如何把 sequence-level 的奖励分配到 token 上？

**策略 1：只加在最后一个 token（标准做法）**

$$r_t^{\text{RM}} = \begin{cases} 0 & t < T \\ r_\phi(x, y) & t = T \end{cases}$$

结合 KL penalty：$r_t = r_t^{\text{RM}} - \beta \cdot \text{kl}_t$

**策略 2：平均分配到每个 token**

$$r_t^{\text{RM}} = \frac{r_\phi(x, y)}{T}, \quad \forall t$$

**策略 3：按 Critic 的 value 差异分配**

$$r_t^{\text{RM}} = V(s_{t+1}) - V(s_t) + \frac{r_\phi(x, y) - \sum_t [V(s_{t+1}) - V(s_t)]}{T}$$

实践中策略 1 最简单也最稳定，是 InstructGPT 和 trl 采用的标准做法。GAE 机制本身就能处理稀疏奖励的反向传播。

### 6.2 Response Mask 的作用

RLHF 中输入序列的结构为 `[prompt tokens | response tokens | padding]`：

```
input_ids:     [p1, p2, p3, r1, r2, r3, r4, PAD, PAD]
response_mask: [ 0,  0,  0,  1,  1,  1,  1,   0,   0]
```

**为什么需要 mask**：
- **Prompt 部分**不参与 Loss 计算——这些 token 不是 Actor 「选择」的动作
- **Padding 部分**是填充，没有意义
- 只有 **Response 部分**是 Actor 的实际输出，Loss 应该只在这里计算

所有的 Loss 计算都必须乘以 `response_mask` 并做归一化：

```python
loss = (raw_loss * response_mask).sum() / response_mask.sum()
```

### 6.3 变长序列的处理

一个 batch 中不同样本的 response 长度通常不同。处理策略：

| 策略 | 做法 | 优缺点 |
|------|------|--------|
| 右填充 + Mask | 短序列在右侧 pad，用 mask 标记有效区域 | 简单，可批量处理 |
| 按长度分桶 | 相近长度的 response 组成一个 batch | 减少 padding 浪费，实现较复杂 |
| 逐样本处理 | batch_size = 1，无需 padding | 无 padding 浪费，但无法利用并行 |

Day 6 我们采用**右填充 + Mask** 的策略，实现简单且易于理解。

---

## 七、完整 RLHF-PPO 目标函数

### 7.1 三个 Loss 的组合

将 Policy Loss 和 Value Loss 加权组合（KL 已通过 reward shaping 融入 GAE）：

$$L^{\text{total}}(\theta, \phi) = L^{\text{policy}}(\theta) + c_v \cdot L^{\text{value}}(\phi)$$

其中：
- $L^{\text{policy}}$ 更新 Actor 参数 $\theta$
- $L^{\text{value}}$ 更新 Critic 参数 $\phi$
- $c_v$ 是 Value Loss 的权重系数（通常 0.5~1.0）

可选地，还可以加入 **Entropy Bonus** 鼓励探索：

$$L^{\text{total}} = L^{\text{policy}} + c_v \cdot L^{\text{value}} - c_e \cdot H(\pi_\theta)$$

其中 $H(\pi_\theta) = -\sum_{y} \pi_\theta(y \mid s) \log \pi_\theta(y \mid s)$。在 LLM 场景中，由于词表巨大，entropy 计算开销大且效果有限，通常**不使用** entropy bonus。

### 7.2 梯度流向

```
L_total = L_policy + c_v * L_value

L_policy ───→ Actor 参数 θ ───→ backward ───→ optimizer_actor.step()
L_value  ───→ Critic 参数 ϕ ───→ backward ───→ optimizer_critic.step()
```

如果 Actor 和 Critic 共享 backbone：

```
L_total = L_policy + c_v * L_value
    │
    ├──→ Actor Head 的梯度来自 L_policy
    ├──→ Value Head 的梯度来自 L_value
    └──→ 共享 Backbone 的梯度来自两者的叠加
```

### 7.3 完整 RLHF-PPO 目标函数

$$\boxed{L^{\text{RLHF-PPO}}(\theta, \phi) = \underbrace{-\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \min(r_t \hat{A}_t, \; \text{clip}(r_t, 1\pm\epsilon)\hat{A}_t)}_{L^{\text{policy}}} + c_v \cdot \underbrace{\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} (V_\phi(s_t) - G_t)^2}_{L^{\text{value}}}}$$

其中所有符号的定义：

| 符号 | 定义 | 来源 |
|------|------|------|
| $r_t$ | $\exp(\log \pi_\theta(y_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(y_t \mid s_t))$ | PPO Update 阶段重新计算 |
| $\hat{A}_t$ | GAE 估计的 Advantage | Phase 3: GAE 计算 |
| $\epsilon$ | Clip 范围，通常 0.2 | 超参数 |
| $V_\phi(s_t)$ | Critic 的价值估计 | PPO Update 阶段重新计算 |
| $G_t$ | $\hat{A}_t + V_{\phi_{\text{old}}}(s_t)$，目标 returns | Phase 3: GAE 计算 |
| $c_v$ | Value Loss 系数，通常 0.5 | 超参数 |
| $\mathcal{M}$ | Response token 掩码 | 数据预处理 |

---

## 八、RLHF-PPO 完整伪代码

### 8.1 一轮 Iteration 伪代码

```
算法：RLHF-PPO 训练 (一轮 Iteration)

输入：
  Actor π_θ, Critic V_ϕ, Reference π_ref, Reward Model r_ϕ
  Prompt batch {x_1, ..., x_B}
  超参数：β (KL 系数), ε (clip 范围), γ (折扣), λ (GAE), c_v (value 系数)
          K (PPO epochs), lr_actor, lr_critic

━━━━━ Phase 1: Rollout (torch.no_grad) ━━━━━

for each prompt x_i in batch:
    # Actor 自回归生成
    y_i = Actor.generate(x_i, max_new_tokens=T)
    old_log_probs_i = Actor.log_prob(x_i, y_i)        # (T,)
    old_values_i = Critic.value(x_i, y_i)              # (T,)

━━━━━ Phase 2: Scoring (torch.no_grad) ━━━━━

for each (x_i, y_i) in batch:
    rm_score_i = RM(x_i, y_i)                          # scalar
    ref_log_probs_i = Reference.log_prob(x_i, y_i)     # (T,)
    
    # 构建 per-token reward
    kl_i = old_log_probs_i - ref_log_probs_i            # (T,)
    rewards_i = -β * kl_i                               # (T,)
    rewards_i[T-1] += rm_score_i                        # 终末加 RM score

━━━━━ Phase 3: GAE ━━━━━

for each sample i in batch:
    for t = T-1, T-2, ..., 0:
        next_val = old_values_i[t+1] if t < T-1 else 0
        δ_t = rewards_i[t] + γ * next_val - old_values_i[t]
        A_t = δ_t + γλ * A_{t+1}  (A_T = 0)
    returns_i = advantages_i + old_values_i

    # 标准化 advantages
    advantages = (advantages - mean) / (std + 1e-8)

━━━━━ Phase 4: PPO Update ━━━━━

for epoch = 1 to K:
    for mini_batch in shuffle(batch):
        # 重新前向传播
        new_log_probs = Actor.log_prob(x, y)            # 需要梯度
        new_values = Critic.value(x, y)                  # 需要梯度
        
        # Policy Loss
        ratio = exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1-ε, 1+ε) * advantages
        L_policy = -mean(min(surr1, surr2) * mask) / sum(mask)
        
        # Value Loss
        L_value = mean((new_values - returns)² * mask) / sum(mask)
        
        # 总 Loss
        L_total = L_policy + c_v * L_value
        
        # 反向传播
        L_total.backward()
        clip_grad_norm_(Actor.params + Critic.params, max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad()

输出：更新后的 Actor π_θ
```

### 8.2 完整训练循环伪代码

```
算法：RLHF-PPO 完整训练

输入：
  SFT model, 训练好的 RM, Prompt 数据集 D
  超参数：N_iters, batch_size, 以及上述所有超参数

初始化：
  Actor ← SFT model (可更新)
  Critic ← SFT model + Value Head (可更新)
  Reference ← SFT model (冻结)
  RM ← 训练好的 RM (冻结)

for iter = 1 to N_iters:
    1. 从 D 中采样 batch_size 个 prompts
    2. 执行一轮 RLHF-PPO Iteration（上述伪代码）
    3. 记录日志：mean_reward, mean_kl, policy_loss, value_loss
    
    # 可选：自适应 KL 系数
    if mean_kl > target_kl * 1.5:
        β ← β * 2
    elif mean_kl < target_kl / 1.5:
        β ← β / 2

返回：优化后的 Actor
```

### 8.3 伪代码与公式的对应关系

| 伪代码行 | 对应公式 | 本文章节 |
|----------|---------|---------|
| `ratio = exp(new_log_probs - old_log_probs)` | $r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})$ | 二、2.3 |
| `surr1 = ratio * advantages` | $r_t(\theta) \hat{A}_t$ | 二、2.5 |
| `surr2 = clip(ratio, ...) * advantages` | $\text{clip}(r_t, 1\pm\epsilon) \hat{A}_t$ | 二、2.5 |
| `L_policy = -min(surr1, surr2)` | $L^{\text{policy}} = -\min(\cdot, \cdot)$ | 二、2.5 |
| `L_value = (new_values - returns)²` | $L^{\text{value}} = (V_\phi - G_t)^2$ | 三、3.4 |
| `kl = log_probs - ref_log_probs` | $\text{kl}_t = \log \pi_\theta - \log \pi_{\text{ref}}$ | 四、4.2 |
| `rewards = -β * kl` | $r_t = -\beta \cdot \text{kl}_t$ | 四、4.3 |
| `δ = r + γ * next_val - val` | $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | 五、5.3 |
| `A = δ + γλ * A_next` | $\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$ | 五、5.3 |

---

## 九、RLHF-PPO 超参数总览

在写完伪代码后，总结一下所有超参数及推荐值：

| 超参数 | 符号 | 推荐值 | 作用 |
|--------|------|--------|------|
| KL 系数 | $\beta$ | 0.01~0.2 | 控制 Actor 偏离 Reference 的代价 |
| Clip 范围 | $\epsilon$ | 0.2 | 限制 policy ratio 的变化幅度 |
| 折扣因子 | $\gamma$ | 1.0 | token 奖励折扣（通常不折扣） |
| GAE 参数 | $\lambda$ | 0.95 | Advantage 估计的 bias-variance 权衡 |
| Value 系数 | $c_v$ | 0.5~1.0 | Value Loss 的权重 |
| PPO epochs | $K$ | 2~4 | 每轮 rollout 后更新几个 epoch |
| 学习率 | $\eta$ | 1e-5~5e-6 | Actor/Critic 的学习率 |
| 梯度裁剪 | max_norm | 0.5~1.0 | 防止梯度爆炸 |
| Batch size | $B$ | 64~512 | Prompt 的 batch 大小 |
| Max new tokens | $T$ | 128~512 | 生成的最大 token 数 |

**超参数敏感性排序**：$\beta > \epsilon > \eta > K > c_v$

$\beta$ 是最关键的超参数——太小则 Reward Hacking，太大则无法学习。Day 6 我们将用实验验证这一点。

---

## 十、自检题

### Loss 推导

1. 写出 RLHF-PPO 的完整 Policy Loss 公式，解释每个变量的含义。
2. Importance sampling ratio $r_t(\theta)$ 在 LLM 中如何计算？为什么在对数空间计算更好？
3. 为什么 Policy Loss 要用 `min` 而不是 `max`？Clipping 机制分别在 $\hat{A}_t > 0$ 和 $\hat{A}_t < 0$ 时如何工作？
4. Value Loss 为什么用 MSE？Value Clipping 的争议是什么？

### KL 与 Reward

5. KL Penalty 有哪两种实现方式？各自的优缺点是什么？
6. 为什么 KL 作为 Reward Shaping（方式 A）比作为单独 Loss 项更好？
7. per-token KL $\text{kl}_t = \log \pi_\theta - \log \pi_{\text{ref}}$ 是精确的 KL 散度吗？为什么？

### GAE 与 Reward 处理

8. RLHF 中 $\gamma = 1.0$ 是常见选择，为什么？在什么情况下应该用 $\gamma < 1$？
9. 如果不用 KL reward shaping，只有终末 RM score，GAE 计算会有什么问题？
10. Response Mask 在 Loss 计算中的作用是什么？如果不用 mask 会怎样？

### 面试手撕

11. 不看笔记，写出 RLHF-PPO 一轮 iteration 的完整伪代码（包含四个 Phase）。
12. 写出 `compute_policy_loss` 和 `compute_gae` 两个函数的伪代码。
13. 解释 RLHF-PPO 的总 Loss 是如何组合 Policy Loss 和 Value Loss 的。梯度分别流向哪里？

---

## 十一、产出要求

- [ ] 从头推导 Policy Loss，包括 ratio 计算、clipping 机制、response mask 处理
- [ ] 推导 Value Loss，理解 Critic 训练目标与 returns 的关系
- [ ] 区分 KL penalty 的两种实现方式，说明为什么 Reward Shaping 是主流
- [ ] 手写 GAE 计算的完整代码（反向递推版本）
- [ ] 写出完整 RLHF-PPO 一轮 iteration 的伪代码
- [ ] 理解 response mask 的作用和变长序列的处理
- [ ] 列出 RLHF-PPO 的所有超参数及其推荐值

---

## 十二、与 Day 6 的衔接

今天我们完成了 RLHF-PPO 所有 Loss 的推导和伪代码。明天将把这些公式**逐行翻译为 Python 代码**：

- 第八节的伪代码 → Day 6 的 `rlhf_train()` 主函数
- 第二节的 Policy Loss → Day 6 的 `compute_policy_loss()`
- 第三节的 Value Loss → Day 6 的 `compute_value_loss()`
- 第四节的 KL Reward → Day 6 的 `compute_rewards()`
- 第五节的 GAE → Day 6 的 `compute_gae()`

Day 6 是本周最重要的实践——在 GPT-2 small 上运行完整的 RLHF-PPO 训练循环，亲眼见证模型的行为如何被 RM 的偏好信号所引导。
