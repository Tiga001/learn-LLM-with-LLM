# Day 7：RLHF 实践分析与复盘 — 从工程挑战到 DPO 展望

> **目标**：系统复盘 RLHF 的工程挑战与常见问题——训练不稳定性、Reward Hacking 的根源与对策、四模型的显存与计算瓶颈；深入分析 RLHF 的根本局限性；从 RLHF 的优化目标自然推导出 DPO 的核心思想，为 Week 11 做铺垫；串联本周 Day 1-6 的全部知识，提供核心公式速查表和分级检查点。

---

## 一、RLHF 工程挑战总结

### 1.1 训练不稳定性

RLHF-PPO 的训练比 SFT 和 RM 训练都不稳定得多。Day 6 的实验中，你可能已经观察到以下现象：

| 现象 | 表现 | 根本原因 |
|------|------|---------|
| KL 爆炸 | mean_kl 快速增大，Actor 输出与 Reference 完全不同 | β 太小，Actor 过度优化 RM |
| Reward Hacking | RM 分数持续上升，但生成质量反而下降 | RM 不完美，Actor 找到 RM 的漏洞 |
| Mode Collapse | Actor 对不同 prompt 生成相似的回答 | 多样性不足，PPO 收敛到单一模式 |
| Value Loss 发散 | Critic 的估计越来越不准 | 学习率过大或 Critic 架构不匹配 |
| Policy Loss 振荡 | policy_loss 在 0 附近剧烈波动 | Advantage 估计不稳定，batch 方差大 |

**健康训练的指标**：

```
✓ mean_reward 稳步上升
✓ mean_kl 缓慢增长但不爆炸（通常 < 10）
✓ policy_loss 在 0 附近小幅波动
✓ value_loss 持续下降
✓ clip_fraction ~5-15%（太小说明更新太保守，太大说明 ratio 偏离太远）
✓ response_length 保持稳定（不是单调增长）
```

### 1.2 显存与计算瓶颈

Day 4 分析了四模型的显存占用。在实际工程中，瓶颈更具体：

| 阶段 | 瓶颈 | 原因 |
|------|------|------|
| Rollout | 时间 | 自回归生成是串行的，无法并行加速 |
| RM Scoring | 显存 | RM 需要处理完整的 (prompt + response) |
| Reference KL | 显存 | Reference 虽然冻结但仍需加载参数 |
| PPO Update | 显存+时间 | 需要重新前向传播 Actor 和 Critic |

**时间分布**（典型场景）：

```
Rollout:     ████████████████████  (~50-60% 的总时间)
Scoring:     ██████                (~15-20%)
GAE:         █                     (~2-3%)
PPO Update:  ████████              (~20-25%)
```

Rollout 是最大的时间瓶颈——自回归生成一个 256 token 的 response 比 forward pass 慢几十倍。

### 1.3 超参数敏感性

Day 6 Part 10 的实验显示，β 的选择直接决定训练成败：

| 超参数 | 敏感度 | 设错的后果 |
|--------|--------|-----------|
| β (KL 系数) | **极高** | 太小 → Reward Hacking；太大 → 学不动 |
| ε (clip ratio) | 中 | 太小 → 更新太慢；太大 → 不稳定 |
| lr (学习率) | 高 | 太大 → 发散；太小 → 收敛慢 |
| K (PPO epochs) | 中 | 太多 → 过拟合当前 batch；太少 → 样本利用率低 |
| batch_size | 中 | 太小 → Advantage 估计方差大；太大 → 显存不够 |

**最佳实践**：

1. 先用小 β（0.01~0.05）快速验证 Reward 能上升
2. 逐步增大 β 直到 KL 可控（mean_kl < 10）
3. Actor lr 通常比 SFT 小一个量级（5e-6 ~ 1e-5）
4. Critic lr 可以比 Actor 大 2~5 倍
5. 使用 Adaptive KL：根据 mean_kl 动态调整 β

```python
# Adaptive KL 实现
target_kl = 6.0
if mean_kl > target_kl * 1.5:
    kl_coef *= 2.0
elif mean_kl < target_kl / 1.5:
    kl_coef /= 2.0
```

---

## 二、Reward Hacking 深入分析

### 2.1 典型表现

Reward Hacking 是 RLHF 最核心的挑战。Day 3 Part 6 演示了 RM 的偏见，Day 6 的训练中这些偏见可能被 PPO 放大：

| 类型 | 表现 | 机制 |
|------|------|------|
| 长度偏见 | 生成越来越长的回答 | RM 对长回答打分偏高 |
| 格式偏见 | 大量使用列表、标号 | RM 对结构化格式打分偏高 |
| 重复偏见 | 重复关键词或短语 | 某些 n-gram 组合得分异常高 |
| Sycophancy | 过度赞同用户 | RM 训练数据中标注者偏好友善回答 |
| 拒绝退化 | 对所有敏感问题都拒绝回答 | RM 给安全回答高分，Actor 学会万能拒绝 |

### 2.2 根本原因

Reward Hacking 的数学本质：

$$\pi^* = \arg\max_\pi \mathbb{E}[\hat{r}(x, y)] - \beta \cdot D_{\text{KL}}[\pi \| \pi_{\text{ref}}]$$

其中 $\hat{r}$ 是**学到的 RM**（近似），不是**真实的人类偏好** $r^*$。当 $\hat{r} \neq r^*$ 时：

$$\pi^* = \arg\max_\pi \mathbb{E}[r^*(x, y) + \underbrace{(\hat{r}(x, y) - r^*(x, y))}_{\text{RM 误差}}] - \beta \cdot D_{\text{KL}}$$

PPO 同时最大化真实偏好 $r^*$ **和** RM 的误差 $\hat{r} - r^*$。当 RM 误差在某个方向上系统性偏大时，Actor 就会朝那个方向「钻漏洞」。

更正式地，Goodhart 定律的体现：

> 当 RM 成为优化目标时，它就不再是人类偏好的好度量。

### 2.3 对策

| 对策 | 做法 | 原理 | 局限 |
|------|------|------|------|
| **KL 约束** | 增大 β | 限制 Actor 偏离 Reference 的程度 | 太强会阻碍学习 |
| **RM 集成** | 使用多个 RM 的平均分 | 减少单个 RM 偏见 | 成本翻倍 |
| **RM 正则化** | 对 RM 添加 weight decay / dropout | 防止 RM 过拟合训练数据 | 可能降低 RM 准确率 |
| **Reward 上限** | $r = \min(r, r_{\max})$ | 防止极端高分 | 可能截断正常高分 |
| **长度惩罚** | $r = r - \alpha \cdot \text{length}$ | 抵消长度偏见 | 需要调 α |
| **迭代式 RLHF** | 定期用新数据重训 RM | RM 适应 Actor 的新分布 | 标注成本高 |
| **DPO** | 直接从偏好学习，绕过 RM | 从根本上消除 RM 偏差 | 失去 reward 信号的灵活性 |

LLaMA 2 的实践：使用**两个独立的 RM**（helpfulness RM + safety RM），取加权平均作为最终 reward。

---

## 三、RLHF 的局限性

### 3.1 标注成本高

| 阶段 | 标注需求 | 典型成本 |
|------|---------|---------|
| SFT 数据 | 写高质量回答 | ~$15-30/条（专业标注者） |
| RM 偏好数据 | 比较两个回答 | ~$5-10/对（相对便宜） |
| RLHF 迭代 | 每轮新增偏好数据 | 持续成本 |

InstructGPT 使用了约 33K 个偏好对比来训练 RM，加上约 13K SFT 数据。这看似不多，但标注质量要求极高——标注者需要深入理解任务，而非简单标注。

### 3.2 训练不稳定

相比 SFT 的稳定收敛，RLHF 的训练更像是在走钢丝：

```
SFT 训练:  ━━━━━━━━━━━━━━━━━━━━━━━━━━→ 收敛
           (平稳下降，几乎不发散)

RLHF 训练: ━━━╱╲━━━━╱╲╱╲━━━━━━╱╲━━━━→ 不稳定收敛
           (振荡，可能发散，需要小心调参)
```

根本原因：RLHF 是在线 RL，数据分布随策略变化——Actor 改变 → 生成分布改变 → reward 分布改变 → 梯度方向改变。这种非平稳性是 RL 固有的难题。

### 3.3 可扩展性问题

RLHF 的计算成本随模型规模超线性增长：

| 模型规模 | 四模型 FP16 参数显存 | 训练总显存 | 需要 GPU |
|---------|-------------------|-----------|---------|
| 1.5B (GPT-2 XL) | 12 GB | ~50 GB | 1× A100 |
| 7B (LLaMA-7B) | 56 GB | ~140 GB | 2× A100 |
| 13B (LLaMA-13B) | 104 GB | ~260 GB | 4× A100 |
| 70B (LLaMA-70B) | 560 GB | ~1.4 TB | 16+ A100 |

每增大模型，不仅四份参数翻倍，rollout 的生成延迟也按比例增加。

### 3.4 偏好数据的分布偏差

偏好标注不可避免地受到标注者偏见的影响：

- **文化偏见**：不同文化背景的标注者对"好回答"的定义不同
- **标注者间不一致**：相同的回答对，不同标注者可能给出相反的判断
- **任务理解偏差**：标注者可能误解标注标准
- **分布外泛化**：RM 在训练分布外的 response 上可能给出不可靠的分数

InstructGPT 论文报告了约 73% 的标注者间一致率——意味着近 30% 的偏好数据本身就是"噪声"。

---

## 四、DPO 预览：为什么可以绕过 RM

### 4.1 从 RLHF 目标推导 DPO

RLHF 的优化目标是：

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[ r(x, y) \right] - \beta \cdot D_{\text{KL}} \left[ \pi_\theta \| \pi_{\text{ref}} \right]$$

这个优化问题有一个**闭式解**（2023 年 DPO 论文的核心发现）：

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp(\frac{1}{\beta} r(x, y))$ 是配分函数。

反解 reward：

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

### 4.2 DPO 的核心思想：隐式 Reward Model

将上面的 reward 表达式代入 Bradley-Terry 偏好模型：

$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

由于 $Z(x)$ 在 $r(y_w) - r(y_l)$ 中被消掉：

$$P(y_w \succ y_l) = \sigma\left(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

这意味着：**不需要显式训练 RM，直接用策略 $\pi_\theta$ 和参考策略 $\pi_{\text{ref}}$ 的对数概率比就能表达偏好**。

DPO Loss：

$$\boxed{L_{\text{DPO}}(\theta) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]}$$

### 4.3 DPO vs RLHF 对比

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 需要 RM | ✓ 需要单独训练 | ✗ 隐式 RM |
| 需要 PPO | ✓ 需要四模型 + GAE + ratio | ✗ 直接 SFT-like 优化 |
| 模型数量 | 4（Actor/Critic/Ref/RM） | 2（Policy/Ref） |
| 显存需求 | 极高 | 较低 |
| 训练稳定性 | 差（RL 固有） | 好（类似 SFT） |
| 数据效率 | 较低（on-policy 采样） | 较高（offline，复用偏好数据） |
| Reward Hacking | 存在 | 不存在（没有显式 RM） |
| RM 信号灵活性 | 高（可实时打分） | 低（只能用偏好对） |
| 在线反馈 | 支持（Actor 实时生成） | 不支持（offline） |

### 4.4 Week 11 衔接预告

Week 11 将深入 DPO 和 GRPO：

- **DPO 完整推导**：从 RLHF 目标出发，逐步推导 DPO Loss
- **DPO 训练实现**：在相同数据上对比 RLHF 和 DPO 的效果
- **GRPO（Group Relative Policy Optimization）**：DeepSeek 提出的方法，用组内排名代替 Critic
- **SimPO、IPO 等变体**：DPO 的改进方向

---

## 五、全周知识串联

### 5.1 SFT → RM → RLHF-PPO 完整流程图

```
Stage 1: SFT                Stage 2: RM Training         Stage 3: RLHF-PPO
(Week 5)                    (Day 2-3)                    (Day 4-6)

┌──────────────┐           ┌──────────────┐           ┌──────────────────────┐
│ 预训练 LLM     │           │ SFT 模型       │           │ SFT 模型 → Actor      │
│     +          │ ────────→ │     +          │ ────────→ │ SFT 模型 → Reference  │
│ 指令数据       │           │ 偏好对比数据    │           │ RM → 奖励信号          │
│ (x, y) pairs   │           │ (x, y_w, y_l)  │           │ Prompt 数据集          │
└──────┬───────┘           └──────┬───────┘           └──────────┬───────────┘
       │                          │                              │
       ▼                          ▼                              ▼
  SFT 模型                   Reward Model               优化后的 Actor
  (能按指令回答)              (能判断好坏)               (生成符合偏好的回答)

目标: L_SFT              目标: L_RM (BT)           目标: max R - β·KL
= -Σ log P(y_t|y_<t)    = -log σ(r_w - r_l)      用 PPO-Clip 更新
```

### 5.2 每个阶段的总结

| 阶段 | 输入 | 输出 | 目标函数 | 对应 Day |
|------|------|------|---------|---------|
| SFT | (prompt, response) pairs | SFT 模型 | $-\sum \log P(y_t \mid y_{<t})$ | Week 5 |
| RM Training | (prompt, chosen, rejected) | Reward Model | $-\log \sigma(r_w - r_l)$ | Day 2-3 |
| RLHF-PPO | Prompts + SFT + RM | 对齐后的 Actor | PPO-Clip + Value Loss | Day 4-6 |

---

## 六、核心公式速查表

### 6.1 RM Loss (Day 2)

$$L_{\text{RM}} = -\mathbb{E}\left[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right]$$

### 6.2 Per-token Reward (Day 4)

$$r_t = \begin{cases} -\beta \cdot \text{kl}_t & t < T \\ r_\phi(x, y) - \beta \cdot \text{kl}_T & t = T \end{cases}$$

$$\text{kl}_t = \log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t)$$

### 6.3 Policy Loss (Day 5)

$$L^{\text{policy}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1\pm\epsilon) \hat{A}_t\right)$$

$$r_t(\theta) = \exp\left(\log \pi_\theta(y_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(y_t \mid s_t)\right)$$

### 6.4 Value Loss (Day 5)

$$L^{\text{value}} = \frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} (V_\phi(s_t) - G_t)^2$$

### 6.5 GAE (Day 5)

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$$

$$G_t = \hat{A}_t + V(s_t)$$

### 6.6 DPO Loss (本日预览)

$$L_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w)}{\ \pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)\right]$$

---

## 七、自检题

### 理论理解

1. RLHF 训练中最常见的三种不稳定现象是什么？各自的表现和原因？
2. Reward Hacking 的数学本质是什么？用公式说明 RM 误差如何导致 Reward Hacking。
3. Goodhart 定律在 RLHF 中的体现是什么？
4. 列出至少四种防止 Reward Hacking 的策略及其优缺点。
5. RLHF 相比 SFT 有哪些根本性的局限？列出至少三个。

### DPO 推导

6. 从 RLHF 的优化目标出发，推导最优策略 $\pi^*$ 的闭式解。
7. 将闭式解代入 BT 模型，推导 DPO Loss。为什么配分函数 $Z(x)$ 会被消掉？
8. DPO 相比 RLHF 的三个核心优势是什么？DPO 的主要缺点又是什么？

### 面试手撕

9. 不看笔记，写出 RLHF 三阶段（SFT → RM → PPO）的完整流程，包括每个阶段的输入、输出、目标函数。
10. 写出 RLHF-PPO 一轮 iteration 的伪代码（四个 Phase），标注哪些需要梯度、哪些不需要。
11. 面试官问「RLHF 训练中 KL 突然爆炸了怎么处理」，你如何回答？
12. 面试官问「Reward 一直涨但用户反馈变差了是什么原因」，你如何回答？

### Week 11 衔接

13. DPO 的 Loss 函数中有哪些变量？它需要哪些数据和模型？
14. DPO 能否在线学习（边生成边训练）？为什么？
15. GRPO 的核心思想是什么？它如何去掉 Critic？

---

## 八、产出要求

- [ ] 列出 RLHF 训练的三大工程挑战及对应的解决策略
- [ ] 分析 Reward Hacking 的至少三种表现形式和根本原因
- [ ] 从 RLHF 优化目标推导 DPO Loss（完整过程）
- [ ] 画出 DPO vs RLHF 的对比表（至少 6 个维度）
- [ ] 画出 SFT → RM → RLHF-PPO 的完整流程图
- [ ] 默写本周核心公式速查表中的全部公式
- [ ] 完成本日自检题（重点：面试手撕 + DPO 推导）

---

## 九、关键检查点

### Tier 1（面试必须）

- [ ] 能闭眼写出 RLHF-PPO 一轮 iteration 的伪代码（rollout → score → GAE → update）
- [ ] 能写出 RM Loss（BT 模型）：$L = -\log \sigma(r_w - r_l)$
- [ ] 能推导 per-token reward 的构建：$r_t = -\beta \cdot \text{kl}_t$，最后一个 token 加 RM score
- [ ] 能写出 PPO-Clip Policy Loss 公式和代码
- [ ] 能解释 RLHF 中 state / action / policy / reward 的对应关系
- [ ] 能解释为什么需要 KL 约束以及 Reference Model 的作用

### Tier 2（面试加分）

- [ ] 能推导 DPO Loss：从 RLHF 目标 → 闭式解 → 代入 BT → DPO
- [ ] 能分析 Reward Hacking 的数学本质和 Goodhart 定律
- [ ] 能估算 RLHF 四模型的显存需求（以 7B 为例）
- [ ] 能解释 Actor-Critic 参数共享的优缺点
- [ ] 能列出 Adaptive KL 的实现方式

### Tier 3（深入理解）

- [ ] 能对比 KL penalty 的两种实现（Reward Shaping vs Loss 项）
- [ ] 能分析 GAE 中 $\gamma = 1.0$ 在 RLHF 中的合理性
- [ ] 能讨论 RLHF 与 DPO 在 online/offline 学习上的权衡
- [ ] 能解释为什么 RLHF 训练比 SFT 不稳定（on-policy 非平稳性）
- [ ] 了解 DeepSpeed-Chat 或 trl 的 RLHF 训练框架

---

## 十、本周回顾与 Week 11 衔接

### 10.1 本周完成的内容

| Day | 主题 | 核心收获 |
|-----|------|---------|
| Day 1 | InstructGPT 论文 + RLHF 概述 | RLHF 三阶段、优化目标、人类偏好数据 |
| Day 2 | Reward Model 原理 | BT 模型、RM Loss、偏好建模 |
| Day 3 | 手写 Reward Model | GPT-2 RM 实现、BT Loss 训练、Reward Hacking 演示 |
| Day 4 | RLHF-PPO 架构 | 四模型、数据流、per-token reward、显存分析 |
| Day 5 | RLHF-PPO Loss 推导 | Policy Loss / Value Loss / KL / GAE / 完整伪代码 |
| Day 6 | 手写 RLHF 训练 | 完整 RLHF-PPO 训练循环、训练监控、超参数分析 |
| Day 7 | 实践分析与复盘 | 工程挑战、Reward Hacking、DPO 预览、全周串联 |

### 10.2 从 RLHF 到 DPO/GRPO

```
Week 10: RLHF-PPO                Week 11: DPO / GRPO
(显式 RM + PPO 优化)              (隐式 RM + 直接优化)

  RM Training                        ×  不需要 RM
       ↓
  Actor + Critic + Ref + RM           Policy + Ref
       ↓                                  ↓
  Rollout → Score → GAE → PPO        直接对偏好数据做 SFT-like 优化
       ↓                                  ↓
  复杂但灵活                          简单但受限于 offline 数据
```

Week 11 将从 DPO 的数学推导入手，实现 DPO 训练，然后介绍 GRPO（去掉 Critic，用组内排名替代 GAE）。这两种方法都是对 RLHF-PPO 复杂性的简化尝试——理解了 RLHF 的完整链路，才能真正理解为什么 DPO 和 GRPO 的简化是合理的。
