# Day 2：DPO Loss 完整推导 — 从公式到可实现的代码

> **目标**：完整推导 DPO Loss 的每一步数学——从 RLHF 带 KL 约束的优化目标出发，推导最优策略的闭式解 $\pi^*$；从闭式解反解 reward 函数；将反解的 reward 代入 Bradley-Terry 偏好模型，证明配分函数 $Z(x)$ 在差值中精确消除；得到最终的 DPO Loss 公式；分析 DPO Loss 的梯度——理解 chosen 概率提升与 rejected 概率降低的隐式权重机制；给出隐式 Reward 的提取方法；写出完整的 DPO 训练伪代码，使其可直接翻译为 Day 3 的 Python 实现。本日的每一个公式都将对应 Day 3 代码中的一行。DPO Loss 的推导是面试 Tier 1 考点。

---

## 一、起点：RLHF 优化目标

### 1.1 回顾 Week 10 的优化目标

Week 10 Day 1 建立的 RLHF 核心优化目标为：

$$\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D}, \, y \sim \pi(\cdot|x)} \left[ r(x, y) \right] - \beta \cdot D_{\text{KL}}\left(\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right)$$

其中：
- $\pi$ 是我们要优化的策略（语言模型）
- $r(x, y)$ 是 reward 函数（来自 RM 或真实人类偏好）
- $\pi_{\text{ref}}$ 是参考策略（冻结的 SFT 模型）
- $\beta > 0$ 是 KL 约束的强度
- $\mathcal{D}$ 是 prompt 分布

### 1.2 展开 KL 散度

将 KL 散度按定义展开：

$$D_{\text{KL}}\left(\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right) = \mathbb{E}_{y \sim \pi(\cdot|x)} \left[ \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

代入优化目标：

$$\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D}} \; \mathbb{E}_{y \sim \pi(\cdot|x)} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

### 1.3 对每个 $x$ 独立优化

由于外层期望是对 $x$ 求均值，对于每个固定的 $x$，问题简化为：

$$\max_{\pi(\cdot|x)} \; \sum_{y} \pi(y|x) \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

约束条件：$\sum_y \pi(y|x) = 1$ 且 $\pi(y|x) \geq 0$。

---

## 二、推导闭式最优解 $\pi^*$

### 2.1 重写目标函数

将方括号内的项整理：

$$r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} = r(x, y) - \beta \log \pi(y|x) + \beta \log \pi_{\text{ref}}(y|x)$$

目标变为：

$$\max_{\pi(\cdot|x)} \; \sum_{y} \pi(y|x) \left[ r(x, y) + \beta \log \pi_{\text{ref}}(y|x) - \beta \log \pi(y|x) \right]$$

### 2.2 提取负熵项

注意 $-\sum_y \pi(y|x) \log \pi(y|x)$ 是策略 $\pi$ 的熵 $H(\pi)$。重写为：

$$\max_{\pi(\cdot|x)} \; \sum_{y} \pi(y|x) \left[ r(x, y) + \beta \log \pi_{\text{ref}}(y|x) \right] + \beta H(\pi(\cdot|x))$$

### 2.3 使用变分法求解

这是一个带约束的凸优化问题。引入 Lagrange 乘子 $\lambda$ 处理归一化约束 $\sum_y \pi(y|x) = 1$：

$$\mathcal{L} = \sum_{y} \pi(y|x) \left[ r(x, y) + \beta \log \pi_{\text{ref}}(y|x) - \beta \log \pi(y|x) \right] - \lambda \left(\sum_y \pi(y|x) - 1\right)$$

对 $\pi(y|x)$ 求导并令其为零：

$$\frac{\partial \mathcal{L}}{\partial \pi(y|x)} = r(x, y) + \beta \log \pi_{\text{ref}}(y|x) - \beta \log \pi(y|x) - \beta - \lambda = 0$$

其中 $-\beta$ 来自 $\frac{\partial}{\partial \pi} [-\pi \log \pi] = -\log \pi - 1$。

### 2.4 解出 $\pi^*$

从上式解出 $\log \pi^*(y|x)$：

$$\log \pi^*(y|x) = \frac{1}{\beta} r(x, y) + \log \pi_{\text{ref}}(y|x) - 1 - \frac{\lambda}{\beta}$$

取指数：

$$\pi^*(y|x) = \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{1}{\beta} r(x, y)\right) \cdot \exp\left(-1 - \frac{\lambda}{\beta}\right)$$

令 $\frac{1}{Z(x)} = \exp(-1 - \frac{\lambda}{\beta})$，利用归一化条件 $\sum_y \pi^*(y|x) = 1$ 确定 $Z(x)$：

$$Z(x) = \sum_{y} \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{1}{\beta} r(x, y)\right)$$

### 2.5 闭式最优解

$$\boxed{\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)}$$

**直觉理解**：最优策略是在参考策略 $\pi_{\text{ref}}$ 的基础上，按 reward 的指数倍重新加权。reward 高的 response 概率增大，reward 低的概率减小。$\beta$ 控制重新加权的幅度——$\beta$ 越小，重新加权越激进。

---

## 三、反解 Reward

### 3.1 从闭式解到 reward

这一步是 DPO 论文的核心创新。对闭式解两边取对数：

$$\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x, y) - \log Z(x)$$

重新排列，**反解出 $r(x, y)$**：

$$\boxed{r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)}$$

### 3.2 这个反解意味着什么

这个等式告诉我们一个深刻的事实：

> 如果我们知道最优策略 $\pi^*$ 和参考策略 $\pi_{\text{ref}}$，就可以**完全恢复** reward 函数（至多差一个只依赖 $x$ 的常数 $\beta \log Z(x)$）。

换言之，**reward 和最优策略之间存在双射关系**——给定 $\pi_{\text{ref}}$ 和 $\beta$，知道其中一个就能推出另一个。

### 3.3 为什么 $\beta \log Z(x)$ 不影响

注意 $\beta \log Z(x)$ 只依赖 prompt $x$，不依赖 response $y$。在 Bradley-Terry 模型中，我们只用 reward 的差：

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \cancel{\beta \log Z(x)} - \cancel{\beta \log Z(x)}$$

$\beta \log Z(x)$ 在差值中精确消除，这就是 BT 模型的「平移不变性」（Week 10 Day 2）。

---

## 四、代入 Bradley-Terry 模型

### 4.1 BT 偏好概率

Week 10 Day 2 推导的 Bradley-Terry 模型：

$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

### 4.2 代入反解的 reward

将第三节反解的 reward 代入 BT 模型：

$$P(y_w \succ y_l \mid x) = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta \log Z(x) - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \beta \log Z(x)\right)$$

$\beta \log Z(x)$ 消除：

$$\boxed{P(y_w \succ y_l \mid x) = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)}$$

### 4.3 关键观察

这个偏好概率的表达式中：
- **没有显式的 reward 函数 $r$**
- **没有 intractable 的配分函数 $Z(x)$**
- 只有两个策略的对数概率比

这意味着：**偏好概率完全由策略决定，不需要单独的 Reward Model**。

---

## 五、DPO Loss 推导

### 5.1 从偏好概率到 Loss

用可学习的策略 $\pi_\theta$ 替代最优策略 $\pi^*$，构建最大似然目标：

$$\max_\theta \; \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log P_\theta(y_w \succ y_l \mid x) \right]$$

等价于最小化负对数似然：

$$\min_\theta \; \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ -\log P_\theta(y_w \succ y_l \mid x) \right]$$

### 5.2 DPO Loss

将偏好概率表达式代入：

$$\boxed{L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]}$$

### 5.3 引入简洁记号

为方便分析，定义隐式 reward margin：

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

则 DPO Loss 可以简洁地写为：

$$L_{\text{DPO}}(\theta) = -\mathbb{E}\left[\log \sigma\left(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)\right)\right]$$

这与 RM 的 Bradley-Terry Loss 形式完全一致：

$$L_{\text{RM}}(\phi) = -\mathbb{E}\left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

唯一的区别是：RM Loss 训练一个显式的 reward 网络 $r_\phi$，而 DPO Loss 训练一个策略 $\pi_\theta$（其隐式 reward 为 $\hat{r}_\theta$）。

### 5.4 完整推导路径回顾

```
Step 1: RLHF 目标
  max_π E[r(x,y)] - β·KL(π ∥ π_ref)
           │
           ▼
Step 2: 闭式最优解
  π*(y|x) = (1/Z) · π_ref(y|x) · exp(r(x,y)/β)
           │
           ▼
Step 3: 反解 reward
  r(x,y) = β log(π*/π_ref) + β log Z(x)
           │
           ▼
Step 4: 代入 BT 模型
  P(y_w > y_l) = σ(r_w - r_l) = σ(β log(π*/π_ref)(y_w) - β log(π*/π_ref)(y_l))
  （Z(x) 在差值中消掉！）
           │
           ▼
Step 5: 用 π_θ 替代 π*，最大化对数似然
  L_DPO = -E[log σ(β log(π_θ/π_ref)(y_w) - β log(π_θ/π_ref)(y_l))]
```

---

## 六、DPO 梯度分析

### 6.1 对 $\theta$ 求梯度

DPO Loss 对参数 $\theta$ 的梯度为：

$$\nabla_\theta L_{\text{DPO}} = -\beta \cdot \mathbb{E}\left[ \underbrace{\sigma\left(-\hat{u}\right)}_{\text{权重}} \left( \underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\text{提升 chosen}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\text{降低 rejected}} \right) \right]$$

其中 $\hat{u} = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$ 是隐式 reward margin。

### 6.2 推导过程

回忆 $\log \sigma(z)$ 的导数：

$$\frac{d}{dz} \log \sigma(z) = 1 - \sigma(z) = \sigma(-z)$$

因此：

$$\nabla_\theta L_{\text{DPO}} = -\mathbb{E}\left[ \sigma(-\hat{u}) \cdot \nabla_\theta \hat{u} \right]$$

而 $\hat{u} = \beta \log \pi_\theta(y_w|x) - \beta \log \pi_\theta(y_l|x) + \text{const}$（$\pi_{\text{ref}}$ 不含 $\theta$），所以：

$$\nabla_\theta \hat{u} = \beta \left( \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right)$$

### 6.3 梯度的直觉解读

梯度包含两个核心成分：

**成分 1：方向** — $\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)$
- 提升 $y_w$（chosen response）的概率
- 降低 $y_l$（rejected response）的概率
- 方向始终正确：让好回答更可能，坏回答更不可能

**成分 2：权重** — $\sigma(-\hat{u}) = 1 - \sigma(\hat{u})$
- 当 $\hat{u}$ 很大时（模型已经正确区分 chosen/rejected），$\sigma(-\hat{u}) \approx 0$，**梯度很小**
- 当 $\hat{u}$ 很小或为负时（模型还没学会区分），$\sigma(-\hat{u}) \approx 1$，**梯度很大**

这是一种**自适应加权**：

| 模型当前状态 | $\hat{u}$ | $\sigma(-\hat{u})$ | 梯度大小 | 含义 |
|------------|-----------|-------------------|---------|------|
| 已经学对了 | 大正值 | ≈ 0 | 很小 | 不浪费梯度在已学会的样本上 |
| 还没学会 | ≈ 0 | ≈ 0.5 | 中等 | 正常学习 |
| 学反了 | 负值 | ≈ 1 | 很大 | 重点纠正错误 |

### 6.4 与 SFT 梯度的对比

| | SFT | DPO |
|--|-----|-----|
| 梯度方向 | 提升目标序列概率 | 提升 chosen + 降低 rejected |
| 权重 | 均匀（每个样本权重相同） | 自适应（关注未学会的样本） |
| 数据 | 只有正样本 $(x, y)$ | 正负对 $(x, y_w, y_l)$ |

### 6.5 DPO 梯度与 RM 梯度的等价性

DPO 的梯度实际上等价于：先用当前策略的隐式 reward 更新 RM，再用更新后的 RM 信号更新策略——只不过这两步合并为一步了。

---

## 七、隐式 Reward 的提取

### 7.1 从训练好的策略中提取 reward

DPO 训练完成后，可以从策略 $\pi_\theta$ 中提取隐式 reward：

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

这个 reward 可以用于：
- 评估 DPO 训练的效果（好回答应该得到更高的隐式 reward）
- 运行时的 Best-of-N 采样（生成 N 个 response，选隐式 reward 最高的）
- 与 RLHF 的 RM 打分对比

### 7.2 计算 log probability ratio

在实现中，$\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 按 token 级别计算然后求和：

$$\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} = \sum_{t=1}^{T} \left[ \log \pi_\theta(y_t|x, y_{<t}) - \log \pi_{\text{ref}}(y_t|x, y_{<t}) \right]$$

```python
def compute_implicit_reward(policy_model, ref_model, input_ids, response_mask, beta):
    """提取 DPO 训练后的隐式 reward"""
    with torch.no_grad():
        policy_logps = get_per_token_log_probs(policy_model, input_ids)
        ref_logps = get_per_token_log_probs(ref_model, input_ids)
    
    log_ratio = (policy_logps - ref_logps) * response_mask
    implicit_reward = beta * log_ratio.sum(dim=-1)
    return implicit_reward
```

---

## 八、DPO 训练的完整伪代码

### 8.1 数据准备

```python
# 输入：偏好数据集 D = {(x_i, y_w_i, y_l_i)}
# 参考模型 π_ref（冻结的 SFT 模型）
# 策略模型 π_θ（初始化为 π_ref 的 deepcopy）
# 超参数：β, lr, epochs, batch_size
```

### 8.2 核心训练循环

```python
for epoch in range(epochs):
    for batch in dataloader:
        # ====== Step 1: 计算 4 组 log probabilities ======
        # π_θ 对 chosen 的 log prob（需要梯度）
        policy_logps_w = get_log_probs(policy_model, batch.chosen_ids)
        # π_θ 对 rejected 的 log prob（需要梯度）
        policy_logps_l = get_log_probs(policy_model, batch.rejected_ids)
        
        with torch.no_grad():
            # π_ref 对 chosen 的 log prob（不需要梯度）
            ref_logps_w = get_log_probs(ref_model, batch.chosen_ids)
            # π_ref 对 rejected 的 log prob（不需要梯度）
            ref_logps_l = get_log_probs(ref_model, batch.rejected_ids)
        
        # ====== Step 2: 计算 DPO Loss ======
        # 隐式 reward margin
        log_ratio_w = policy_logps_w - ref_logps_w  # log(π_θ/π_ref) for chosen
        log_ratio_l = policy_logps_l - ref_logps_l  # log(π_θ/π_ref) for rejected
        logits = beta * (log_ratio_w - log_ratio_l)  # β(Δ_w - Δ_l)
        loss = -F.logsigmoid(logits).mean()
        
        # ====== Step 3: 反向传播 ======
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 8.3 关键实现细节

**get_log_probs 函数**：计算序列的 log probability

```python
def get_log_probs(model, input_ids, attention_mask, response_mask):
    """计算 response 部分每个 token 的 log prob 之和"""
    logits = model(input_ids, attention_mask=attention_mask).logits
    
    # shift: 预测 token t 的 logits 在位置 t-1
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    
    # gather: 取实际 token 对应的 log prob
    per_token_log_probs = log_probs.gather(
        dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    
    # 只对 response 部分求和（乘以 response_mask）
    return (per_token_log_probs * response_mask[:, 1:]).sum(dim=-1)
```

---

## 九、DPO Loss 与 SFT Loss、RM Loss 的对比

### 9.1 三种 Loss 的统一视角

| | SFT Loss | RM Loss | DPO Loss |
|--|----------|---------|----------|
| **公式** | $-\log P_\theta(y \mid x)$ | $-\log \sigma(r_\phi(y_w) - r_\phi(y_l))$ | $-\log \sigma(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})$ |
| **数据** | $(x, y)$ | $(x, y_w, y_l)$ | $(x, y_w, y_l)$ |
| **训练什么** | 语言模型 $\pi_\theta$ | Reward Model $r_\phi$ | 语言模型 $\pi_\theta$ |
| **本质** | 模仿好回答 | 学会判断好坏 | 直接从偏好优化策略 |
| **正样本** | $y$ | $y_w$（间接） | $y_w$（间接） |
| **负样本** | 无 | $y_l$（间接） | $y_l$（间接） |
| **参考模型** | 不需要 | 不需要 | 需要 $\pi_{\text{ref}}$ |

### 9.2 DPO Loss = 隐式 RM 的 BT Loss

从形式上看：

$$L_{\text{DPO}} = L_{\text{RM}}\big|_{r_\phi \leftarrow \hat{r}_\theta}$$

DPO Loss 就是把 RM 的 BT Loss 中的显式 reward $r_\phi$ 替换为隐式 reward $\hat{r}_\theta = \beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$。

### 9.3 β 的作用

$\beta$ 在 DPO 中控制两个方面：

**作为 KL 约束强度**：
- $\beta$ 大 → KL 约束强 → 策略变化小 → 保守更新
- $\beta$ 小 → KL 约束弱 → 策略变化大 → 激进更新

**作为 reward 缩放因子**：
- $\beta$ 大 → 隐式 reward margin 被放大 → 同样的概率差产生更大的 logit → 更容易「确信」
- $\beta$ 小 → 隐式 reward margin 被压缩 → 需要更大的概率差才能区分

**典型取值**：$\beta \in [0.1, 0.5]$，最常用 $\beta = 0.1$。

---

## 十、DPO 的理论等价性与实践差异

### 10.1 数学等价性

在以下假设下，DPO 与 RLHF-PPO 在最优解处严格等价：

1. BT 模型是偏好数据的正确生成模型
2. 偏好数据的分布覆盖策略的输出分布
3. 策略类足够灵活（能表达最优解）
4. 优化充分收敛到全局最优

### 10.2 实践中的差异

| 维度 | DPO | RLHF-PPO |
|------|-----|----------|
| **数据分布** | 固定（offline） | 随策略变化（on-policy） |
| **探索能力** | 无（只学数据中有的） | 有（Actor 实时生成新 response） |
| **分布偏移** | 训练后期策略偏离数据分布 → coverage 问题 | 不存在（始终 on-policy） |
| **收敛到的解** | 可能是局部最优（受限于数据分布） | 理论上可以收敛到全局最优 |

### 10.3 Iterative DPO

为缓解 offline 的局限，实践中常用 Iterative DPO：

```
Round 1: π_ref = π_SFT → DPO → π_1
Round 2: π_ref = π_1  → 用 π_1 生成新 response → 收集新偏好数据 → DPO → π_2
Round 3: π_ref = π_2  → 用 π_2 生成新 response → 收集新偏好数据 → DPO → π_3
...
```

每轮用当前策略生成新的 response，收集新的偏好数据，然后重新做 DPO。这本质上是用多轮 offline 学习来近似 on-policy 学习。

---

## 十一、自检题

### 推导类（面试核心）

1. 从 RLHF 目标出发，推导闭式最优解 $\pi^*$。
2. 从闭式解反解 reward：$r(x,y) = \beta \log \frac{\pi^*}{\pi_{\text{ref}}} + \beta \log Z(x)$。
3. 将反解的 reward 代入 BT 模型，证明 $Z(x)$ 被消掉。
4. 写出最终的 DPO Loss 公式。
5. 推导 DPO Loss 对 $\theta$ 的梯度。

### 理解类

6. DPO 梯度中的 $\sigma(-\hat{u})$ 起什么作用？为什么说它实现了「自适应加权」？
7. $\beta$ 在 DPO 中控制什么？太大和太小分别有什么问题？
8. 隐式 reward $\hat{r}_\theta(x,y)$ 的公式是什么？它可以用于什么？
9. DPO Loss 与 RM 的 BT Loss 在形式上有什么关系？
10. DPO 与 RLHF 在数学上等价的前提条件是什么？

### 面试手撕

11. 不看笔记，写出 DPO Loss 的完整公式。
12. 写出 DPO 训练循环的伪代码（标注哪些需要梯度，哪些不需要）。
13. 写出 `get_log_probs` 函数的实现思路。
14. 面试官问「DPO 的梯度为什么不会在所有样本上均匀分配」，你如何回答？

---

## 十二、产出要求

- [ ] **手写 DPO Loss 完整推导**（面试 Tier 1！从 RLHF 目标到最终 Loss，含 5 个关键步骤）
- [ ] 能默写 DPO Loss 公式：$L_{\text{DPO}} = -\mathbb{E}[\log \sigma(\beta(\log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}))]$
- [ ] 推导并解释 DPO 梯度中 $\sigma(-\hat{u})$ 的自适应权重
- [ ] 写出 DPO 训练的完整伪代码（含 get_log_probs 实现）
- [ ] 解释 DPO Loss 与 RM BT Loss 的形式等价关系
- [ ] 完成全部自检题（重点：推导类 1-5，面试手撕 11-14）

---

## 十三、与 Day 3 的衔接

今天我们完成了 DPO Loss 的严格数学推导。明天将把每一个公式翻译为 Python 代码——**从零手写 DPO 训练循环**。具体来说：

- 构建偏好数据集（类似 W10 Day3 的格式）
- 实现 `get_log_probs` 函数：把序列的对数概率计算封装好
- 实现 `dpo_loss` 函数：对应今天推导的 Loss 公式
- 完整训练循环：forward → 4 组 log_probs → DPO Loss → backward
- 隐式 reward 提取与可视化
- 训练前后生成质量对比

Day 3 的代码中每一行都能在今天的推导中找到数学对应。
