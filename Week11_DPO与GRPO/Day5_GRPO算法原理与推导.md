# Day 5：GRPO 算法原理与推导 — 用组内排名替代 Critic

> **目标**：从 PPO 的 Critic 瓶颈问题出发，理解 GRPO（Group Relative Policy Optimization）的设计动机；掌握 GRPO 的核心思想——用同一 prompt 的多组采样的 reward 均值作为 baseline，替代 Critic 网络的价值估计；从策略梯度定理出发严格推导 GRPO 的目标函数和 Loss 公式；分析组内归一化 Advantage 的统计性质；对比 GRPO 与 PPO、DPO 在数学框架上的异同。GRPO 手写是面试 Tier 2 考点。

---

## 一、动机：PPO 的 Critic 问题

### 1.1 回顾 PPO 中的 Critic

Week 9 Day 5 我们学习了 Actor-Critic 框架：

```
Actor (π_θ)  : 输出动作概率 → 策略
Critic (V_ϕ) : 估计状态价值 → 基线
```

Critic 的作用是提供 Advantage 估计的 baseline：

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

没有 Critic，我们只能用 Monte Carlo 回报 $G_t = \sum_{l=0}^{T-t} \gamma^l r_{t+l}$ 作为优势估计，方差很大。

### 1.2 Critic 在 LLM 中的问题

在 RLHF-PPO（Week 10）中，Critic 带来严重的工程问题：

| 问题 | 影响 |
|------|------|
| **参数量翻倍** | Critic 网络与 Actor 同规模，7B Actor + 7B Critic = 14B 可训练参数 |
| **显存翻倍** | Critic 的参数 + 梯度 + 优化器状态额外占用 ~56 GB（7B 模型） |
| **训练不稳定** | Critic 的价值估计不准会导致 Advantage 偏差，进而影响 Actor 更新 |
| **长序列估值困难** | 对于数千 token 的推理序列，Critic 很难准确估计每个位置的状态价值 |

对于 DeepSeek-R1 的 671B MoE 模型（Day 4），训练一个同等规模的 Critic 在工程上几乎不可能。

### 1.3 GRPO 的核心提问

> 能否**完全去掉 Critic**，用另一种方式估计 Advantage？

GRPO 的回答：**用同一 prompt 的多组采样的 reward 进行组内比较**来替代 Critic 的价值估计。

---

## 二、GRPO 的核心思想

### 2.1 直觉：从绝对估值到相对排名

```
PPO 的方式（绝对估值）：
  对每个 (s_t, a_t)，Critic 估计 V(s_t)
  Advantage = 实际回报 - Critic 估值
  → 需要一个准确的 Critic

GRPO 的方式（相对排名）：
  对同一 prompt，生成 G 个 response
  每个 response 获得一个 reward
  Advantage = (该 response 的 reward - 组内平均 reward) / 组内标准差
  → 不需要 Critic，只需要 reward 函数
```

### 2.2 组内采样的类比

想象一个考试场景：

| 评分方式 | 类比 | 对应算法 |
|---------|------|---------|
| 绝对分数 (100 分制) | 需要知道每道题的标准答案和评分标准 | PPO (Critic 估值) |
| 相对排名 (班级内排名) | 只需要比较同一题的不同答案谁更好 | GRPO (组内对比) |

GRPO 用「相对排名」替代「绝对评分」——不需要知道「好答案应该值多少分」，只需要知道「这个答案比组内平均水平好还是差」。

### 2.3 组内归一化 Advantage

对于 prompt $x$，采样 $G$ 个 response $\{y_1, y_2, \ldots, y_G\}$，获得 reward $\{r_1, r_2, \ldots, r_G\}$。

每个 response 的归一化 Advantage 为：

$$\boxed{\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}}$$

即：

$$\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}, \quad \bar{r} = \frac{1}{G}\sum_{j=1}^G r_j, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \bar{r})^2}$$

**性质**：
- 组内 Advantage 的均值为 0：$\frac{1}{G}\sum_i \hat{A}_i = 0$
- 组内 Advantage 的标准差为 1（归一化后）
- Reward 高于组内平均的 response 得到正 Advantage → 策略会增加其概率
- Reward 低于组内平均的 response 得到负 Advantage → 策略会降低其概率

---

## 三、GRPO 的数学推导

### 3.1 起点：策略梯度定理

Week 9 Day 4 推导的策略梯度定理：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t \right]$$

在 LLM 场景下（response 级别），简化为：

$$\nabla_\theta J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \, y \sim \pi_\theta(\cdot|x)} \left[ \nabla_\theta \log \pi_\theta(y|x) \cdot A(x, y) \right]$$

### 3.2 GRPO 对 Advantage 的估计

PPO 用 Critic 估计 $A(x, y)$。GRPO 用组内采样估计：

对每个 prompt $x$：
1. 从 **旧策略** $\pi_{\theta_{\text{old}}}$ 采样 $G$ 个 response：$\{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|x)$
2. 计算每个 response 的 reward：$r_i = r(x, y_i)$
3. 计算组内归一化 Advantage：$\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$

### 3.3 GRPO 目标函数

将 PPO-Clip 的思想与组内 Advantage 结合，GRPO 的目标函数为：

$$\boxed{J_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \frac{1}{G} \sum_{i=1}^{G} \left[ \min\left( \rho_i(\theta) \hat{A}_i, \; \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]}$$

其中 importance sampling ratio（与 PPO 相同）：

$$\rho_i(\theta) = \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}$$

### 3.4 Token 级别 vs Sequence 级别

GRPO 有两种实现方式：

**Sequence 级别 GRPO**（更常用，DeepSeek-R1 采用）：

$$\rho_i(\theta) = \prod_{t=1}^{T_i} \frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} | x, y_{i,<t})}$$

在对数空间：

$$\log \rho_i(\theta) = \sum_{t=1}^{T_i} \left[\log \pi_\theta(y_{i,t} | x, y_{i,<t}) - \log \pi_{\theta_{\text{old}}}(y_{i,t} | x, y_{i,<t})\right]$$

每个 token 使用**同一个 sequence-level Advantage** $\hat{A}_i$。

**Token 级别 GRPO**（将 sequence reward 分摊到每个 token）：

每个 token 仍用 $\hat{A}_i$（因为没有 Critic 做 per-token 估计），但 ratio 在每个 token 位置独立计算：

$$\rho_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} | x, y_{i,<t})}$$

### 3.5 KL 惩罚项

GRPO 中的 KL 约束防止策略偏离参考模型太远（与 RLHF-PPO 中的作用相同）：

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \approx \frac{1}{T_i} \sum_{t=1}^{T_i} \left[\log \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\text{ref}}(y_{i,t}|x, y_{i,<t})}\right]$$

在 DeepSeek-R1 的实践中，也可以使用近似的 KL 估计：

$$\hat{D}_{\text{KL}} \approx \frac{\pi_{\text{ref}}(y_{i,t}|x, y_{i,<t})}{\pi_\theta(y_{i,t}|x, y_{i,<t})} - \log \frac{\pi_{\text{ref}}(y_{i,t}|x, y_{i,<t})}{\pi_\theta(y_{i,t}|x, y_{i,<t})} - 1$$

---

## 四、GRPO Loss 的完整公式

### 4.1 GRPO Loss（最小化形式）

将目标函数转为 Loss（取负号）：

$$\boxed{L_{\text{GRPO}}(\theta) = -\frac{1}{|B|} \sum_{x \in B} \frac{1}{G} \sum_{i=1}^{G} \left[ \min\left(\rho_i \hat{A}_i, \; \text{clip}(\rho_i, 1\pm\epsilon) \hat{A}_i\right) - \beta \hat{D}_{\text{KL},i} \right]}$$

其中 $B$ 是 prompt 的 mini-batch。

### 4.2 展开所有细节

```
对于 batch B 中的每个 prompt x:
  1. 从 π_old 采样 G 个 response: {y_1, ..., y_G}
  2. 计算 rewards: {r_1, ..., r_G}
  3. 归一化: A_i = (r_i - mean(r)) / std(r)
  4. 对每个 response y_i:
     a. 计算 per-token log ratio:
        log_ratio_t = log π_θ(y_{i,t}) - log π_old(y_{i,t})
     b. 计算 ratio: ρ_t = exp(log_ratio_t)
     c. 计算 clipped objective:
        surr1 = ρ_t * A_i
        surr2 = clip(ρ_t, 1-ε, 1+ε) * A_i
        policy_loss_t = min(surr1, surr2)
     d. 计算 KL:
        kl_t = log π_θ(y_{i,t}) - log π_ref(y_{i,t})
     e. 总 loss 贡献:
        loss_t = -(policy_loss_t - β * kl_t)
```

### 4.3 GRPO 训练循环伪代码

```python
for iteration in range(num_iterations):
    # ====== Phase 1: 采样 ======
    prompts = sample_batch(prompt_dataset)
    
    with torch.no_grad():
        all_responses = []
        all_rewards = []
        all_old_log_probs = []
        
        for x in prompts:
            responses = []
            for _ in range(G):
                y = generate(policy_old, x)
                responses.append(y)
            
            rewards = [reward_fn(x, y) for y in responses]
            old_logps = [get_log_probs(policy_old, x, y) for y in responses]
            
            all_responses.append(responses)
            all_rewards.append(rewards)
            all_old_log_probs.append(old_logps)
    
    # ====== Phase 2: 计算 Advantage ======
    all_advantages = []
    for rewards in all_rewards:
        r = torch.tensor(rewards)
        advantages = (r - r.mean()) / (r.std() + 1e-8)
        all_advantages.append(advantages)
    
    # ====== Phase 3: GRPO 更新 ======
    for epoch in range(K):
        for x, responses, old_logps, advantages in zip(...):
            for y_i, old_lp_i, A_i in zip(responses, old_logps, advantages):
                new_lp = get_log_probs(policy, x, y_i)
                ref_lp = get_log_probs(ref_model, x, y_i)
                
                ratio = torch.exp(new_lp - old_lp_i)
                clipped_ratio = torch.clamp(ratio, 1-eps, 1+eps)
                
                surr1 = ratio * A_i
                surr2 = clipped_ratio * A_i
                policy_loss = -torch.min(surr1, surr2).mean()
                
                kl = (new_lp - ref_lp).mean()
                loss = policy_loss + beta * kl
                
                loss.backward()
                optimizer.step()
    
    # 更新 old policy
    policy_old = copy(policy)
```

---

## 五、GRPO vs PPO vs DPO 的数学对比

### 5.1 三种算法的统一视角

| 维度 | PPO | GRPO | DPO |
|------|-----|------|-----|
| **优化目标** | $\max E[r] - \beta \text{KL}$ | $\max E[r] - \beta \text{KL}$ | 等价于 $\max E[r] - \beta \text{KL}$ |
| **Advantage 来源** | Critic $V_\phi$ (GAE) | 组内归一化 $\frac{r_i - \bar{r}}{\sigma_r}$ | 隐式（梯度中自动出现） |
| **数据方式** | On-policy 采样 | On-policy 采样 | Offline 偏好数据 |
| **Reward 来源** | 显式 RM | 显式 reward 函数 | 隐式 $\beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$ |
| **Clip 机制** | PPO-Clip | PPO-Clip（相同） | 无 Clip |

### 5.2 Advantage 估计方式对比

| 方法 | Advantage 估计 | 偏差 | 方差 | 需要 Critic |
|------|-------------|------|------|-----------|
| PPO (GAE, λ=1) | $\hat{A}_t = G_t - V_\phi(s_t)$ | 低（依赖 Critic 准确度） | 中 | 是 |
| PPO (GAE, λ<1) | $\hat{A}_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$ | 中 | 低 | 是 |
| GRPO | $\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$ | 低（无偏估计） | 中-高（依赖 $G$ 大小） | 否 |
| REINFORCE | $\hat{A} = G_t - b$ (baseline) | 低 | 高 | 否 |

GRPO 可以看作是 REINFORCE with baseline 的改进版——baseline 是组内均值，归一化降低了方差。

### 5.3 Loss 公式对比

**PPO Loss**（Week 9）：

$$L_{\text{PPO}} = -\mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t^{\text{GAE}}, \text{clip}(\rho_t, 1\pm\epsilon) \hat{A}_t^{\text{GAE}}\right)\right] + c_1 \underbrace{(V_\phi(s_t) - G_t)^2}_{\text{Value Loss}}$$

**GRPO Loss**：

$$L_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^G\left[\min\left(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1\pm\epsilon) \hat{A}_i\right)\right] + \beta \cdot \hat{D}_{\text{KL}}$$

**DPO Loss**（Day 2）：

$$L_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta \log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)\right]$$

关键区别：
- PPO 有 Value Loss（训练 Critic）；GRPO 没有
- PPO 的 Advantage 是 per-token (GAE)；GRPO 是 per-sequence
- DPO 完全不需要在线采样

---

## 六、GRPO 的理论分析

### 6.1 组大小 $G$ 的影响

| $G$ | Advantage 估计质量 | 计算成本 | 适用场景 |
|-----|-----------------|---------|---------|
| $G = 2$ | 差（只有两个样本做对比） | 很低 | 资源极度受限 |
| $G = 8$ | 中等 | 中 | 中等规模实验 |
| $G = 16$ | 较好 | 较高 | 实际训练（R1 默认值） |
| $G = 64$ | 好 | 高 | 追求稳定性 |

理论上，组大小 $G \to \infty$ 时，组内均值趋近于真实的 expected reward $\mathbb{E}_{y \sim \pi}[r(x,y)]$，此时 GRPO 的 baseline 等价于最优 constant baseline。

### 6.2 归一化的统计效果

归一化 $\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$ 有几个重要效果：

**效果 1：消除 reward 量纲**
- 不同 prompt 的 reward 范围可能差异很大（如数学题 0/1，代码题 0~100）
- 归一化后所有 prompt 的 Advantage 在相同尺度上

**效果 2：减少方差**
- 原始 reward 的方差被标准差除以后压缩到单位方差
- 策略更新的步长更可控

**效果 3：自动 baseline**
- 减去均值 $\bar{r}$ 起到 baseline 的作用
- 只有高于平均的 response 获得正 Advantage

### 6.3 GRPO 与 REINFORCE 的关系

GRPO 本质上是 REINFORCE with learned baseline 的一个特例：

$$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(y|x) \cdot (r(x,y) - b(x))\right]$$

其中 $b(x) = \bar{r} = \frac{1}{G}\sum_j r(x, y_j)$ 是用 Monte Carlo 估计的 baseline。

GRPO 额外做了三件事：
1. **归一化**：除以 $\sigma_r$，控制梯度尺度
2. **PPO-Clip**：加入 clip 机制，防止过大更新
3. **KL 约束**：加入 KL penalty，防止偏离参考模型

---

## 七、GRPO 的优势与局限

### 7.1 优势

| 优势 | 说明 |
|------|------|
| **无需 Critic** | 显存减半，实现简化 |
| **适合超大模型** | 671B 模型无需训练同等大小的 Critic |
| **适合长序列** | 无需 per-token 价值估计 |
| **实现简单** | 比 PPO 少了整个 Critic 训练流程 |
| **On-policy** | 保持 RL 的探索能力（vs DPO 的 offline） |

### 7.2 局限

| 局限 | 说明 |
|------|------|
| **采样开销** | 每个 prompt 需要生成 $G$ 个 response |
| **sequence-level Advantage** | 比 per-token GAE 信息量少 |
| **依赖 reward 函数** | 需要能对完整 response 打分的 reward |
| **$G$ 太小时方差大** | 组内样本太少会导致 Advantage 估计不准 |
| **不适合连续动作** | 主要针对语言生成的离散动作空间 |

---

## 八、自检题

### 推导类

1. 写出 GRPO 的组内归一化 Advantage 公式。
2. 写出 GRPO 的目标函数（含 clip 和 KL 项）。
3. GRPO 的 Advantage 估计与 PPO 的 GAE 有什么数学关系？
4. 解释 GRPO 中归一化（除以 $\sigma_r$）的三个作用。

### 理解类

5. GRPO 为什么不需要 Critic？用什么替代了 Critic 的功能？
6. 组大小 $G$ 的选择如何影响 GRPO 的性能？$G$ 太小和太大各有什么问题？
7. GRPO 是 on-policy 还是 off-policy？这与 DPO 有什么区别？
8. GRPO 的 KL 约束与 DPO 的隐式 KL 有什么区别？

### 对比类

9. 画一个表格对比 PPO、GRPO、DPO 的 Advantage 来源、数据方式、模型数量。
10. 在什么场景下 GRPO 优于 PPO？在什么场景下 PPO 优于 GRPO？
11. GRPO 和 DPO 分别适合什么类型的任务？

### 面试准备

12. 用 3 句话解释 GRPO 的核心思想。
13. 面试官问「GRPO 的 Advantage 估计有偏吗」，你如何回答？
14. 手写 GRPO 的完整训练循环伪代码。

---

## 九、产出要求

- [ ] **手写 GRPO Loss 公式**（含 clip、KL、组内归一化 Advantage）
- [ ] 推导 GRPO 的 Advantage 估计：从策略梯度定理到组内归一化
- [ ] 画出 PPO vs GRPO vs DPO 的数学对比表（至少 6 个维度）
- [ ] 解释 GRPO 适合超大模型的原因
- [ ] 分析组大小 $G$ 对性能的影响
- [ ] 完成全部自检题

---

## 十、与 Day 6 的衔接

今天我们完成了 GRPO 的数学推导。明天将把公式翻译为代码——**从零手写 GRPO 训练循环**。具体来说：

- 设计简单的数学推理任务和对应的 rule-based reward
- 实现多组采样：对同一 prompt 生成 $G$ 个 response
- 实现组内归一化 Advantage 计算
- 实现 GRPO Loss（PPO-Clip + KL）
- 完整训练循环
- 训练效果可视化

Day 6 的代码中每一行都能在今天的推导中找到数学对应。
