# Day 1：DPO 论文精读与原理 — 你的语言模型其实是一个 Reward Model

> **目标**：从 Week 10 Day7 的 RLHF 局限性分析出发，系统精读 DPO 论文——理解 DPO 的核心动机：为什么可以绕过显式 Reward Model；掌握 DPO 的核心思想「隐式 Reward Model」——策略本身隐含了 reward 信息；理解 DPO 与 RLHF 的架构差异（4 模型 → 2 模型）以及由此带来的工程简化；分析 DPO 的优势与局限，理解为什么 DPO 在工业界被广泛采用。本日建立对 DPO 的全局视角，为 Day 2 的严格数学推导和 Day 3 的手写实现奠定基础。

---

## 一、从 RLHF 的痛点出发

### 1.1 回顾 RLHF 的完整链路

Week 10 我们完整学习了 RLHF 三阶段：

```
Stage 1: SFT           Stage 2: RM Training       Stage 3: PPO
(指令微调)              (训练奖励模型)              (策略优化)

π_SFT ──────→ 偏好数据 + π_SFT ──────→ Actor + Critic + Ref + RM
                  ↓                           ↓
              r_ϕ(x, y)                  max E[R] - β·KL
```

这条链路虽然有效（InstructGPT 的 1.3B 模型胜过 175B GPT-3），但工程上存在严重的痛点。

### 1.2 RLHF-PPO 的四大痛点

Week 10 Day7 我们详细分析了 RLHF 的工程挑战，总结为四大痛点：

| # | 痛点 | 具体表现 | 根本原因 |
|---|------|---------|---------|
| 1 | **四模型显存爆炸** | 7B 模型需要 ~140 GB 显存 | Actor + Critic + Reference + RM 四份参数 |
| 2 | **训练极不稳定** | KL 爆炸、Reward Hacking、Mode Collapse | RL 的 on-policy 非平稳性 + RM 不完美 |
| 3 | **超参数极度敏感** | β 稍有偏差就训练失败 | PPO 本身 + KL 约束 + RM 误差三重叠加 |
| 4 | **Reward Hacking** | RM 分数上升但生成质量下降 | Goodhart 定律：RM ≠ 真实人类偏好 |

### 1.3 一个自然的问题

面对这些痛点，一个自然的问题浮现：

> **能不能绕过 Reward Model，直接从偏好数据优化策略？**

这正是 DPO 要回答的核心问题。答案出人意料地优雅：**可以，而且数学上严格等价于 RLHF**。

---

## 二、DPO 论文精读

### 2.1 论文核心信息

- **标题**：*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
- **作者**：Rafailov, Sharma, Mitchell, Manning, Ermon & Finn (Stanford, 2023)
- **发表**：NeurIPS 2023
- **核心贡献**：证明带 KL 约束的 RLHF 目标可以直接用偏好数据优化，无需显式训练 Reward Model

### 2.2 论文标题的深意

标题 "Your Language Model is Secretly a Reward Model" 包含了 DPO 的全部精髓：

> 一个经过对齐训练的语言模型 $\pi_\theta$，与参考模型 $\pi_{\text{ref}}$ 之间的对数概率比，**本身就定义了一个 reward 函数**。

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

不需要单独训练一个 RM 来打分——策略模型本身就「知道」什么是好回答。

### 2.3 论文的主要发现

| 发现 | 意义 |
|------|------|
| RLHF 的 KL 约束优化有闭式最优解 | 理论基础：最优策略 $\pi^*$ 可以用 $\pi_{\text{ref}}$ 和 $r$ 表达 |
| 从闭式解可以反解出 reward | 关键桥梁：reward 可以用策略的对数概率比表达 |
| 将反解的 reward 代入 BT 模型后，配分函数 $Z(x)$ 消掉 | 数学关键：无需计算 intractable 的归一化常数 |
| DPO 与 RLHF 在最优解处数学等价 | 理论保证：不是近似，是精确等价 |
| DPO 在多个任务上匹配或超过 PPO | 实验验证：控制情感、摘要质量、对话 |

---

## 三、DPO 的核心思想：隐式 Reward Model

### 3.1 RLHF 的优化目标回顾

Week 10 Day1 推导的 RLHF 优化目标：

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim D, \, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) \right] - \beta \cdot D_{\text{KL}}\left(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right)$$

这里有两个模型需要训练：
1. **Reward Model** $r_\phi$：需要偏好数据单独训练
2. **Actor** $\pi_\theta$：用 PPO 在 RM 信号下优化

DPO 的核心发现是：**这两步可以合并为一步**。

### 3.2 关键数学洞察

Week 10 Day1 已经给出了 RLHF 优化目标的闭式最优解（Day 2 将严格推导）：

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

从这个闭式解可以**反解 reward**：

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

这个反解告诉我们：**reward 完全由策略和参考策略的对数概率比决定**（加上一个只依赖 $x$ 的常数 $\beta \log Z(x)$）。

### 3.3 配分函数的消除

将反解的 reward 代入 Bradley-Terry 偏好模型：

$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

由于 $\beta \log Z(x)$ 只依赖 $x$，在 $r(y_w) - r(y_l)$ 的差中被消掉：

$$P(y_w \succ y_l) = \sigma\left(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

这是 DPO 最关键的数学步骤——**$Z(x)$ 消除**使得偏好概率可以直接用策略计算，无需知道 reward 的绝对值。

### 3.4 从间接到直接

```
RLHF 路径（间接）：
  偏好数据 ──→ 训练 RM ──→ RM 打分 ──→ PPO 优化 ──→ 对齐策略
  (两步：先学 reward，再优化策略)

DPO 路径（直接）：
  偏好数据 ──→ 直接优化策略 ──→ 对齐策略
  (一步：偏好数据直接变成策略的梯度信号)
```

DPO 绕过了 RM，直接在偏好数据上优化策略。从数学上看，DPO 的策略隐式地定义了一个 reward：

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

这就是 "Your Language Model is Secretly a Reward Model" 的含义。

---

## 四、DPO 与 RLHF 的架构对比

### 4.1 模型数量对比

| 组件 | RLHF (PPO) | DPO |
|------|-----------|-----|
| Actor (训练中的策略) | ✓ | ✓ (即 $\pi_\theta$) |
| Reference (冻结的 SFT 模型) | ✓ | ✓ (即 $\pi_{\text{ref}}$) |
| Critic (价值网络) | ✓ | ✗ **不需要** |
| Reward Model | ✓ | ✗ **不需要** |
| **总计** | **4 个模型** | **2 个模型** |

### 4.2 显存对比

以 LLaMA-7B 为例（FP16）：

| 模型 | RLHF 显存 | DPO 显存 | 说明 |
|------|----------|---------|------|
| Actor 参数 | 14 GB | 14 GB | 相同 |
| Actor 梯度 + 优化器 | 42 GB | 42 GB | AdamW: 参数 × 3 |
| Reference 参数 | 14 GB | 14 GB | 冻结，无梯度 |
| Critic 参数 + 梯度 + 优化器 | 56 GB | **0** | DPO 无 Critic |
| Reward Model 参数 | 14 GB | **0** | DPO 无 RM |
| **总计** | **~140 GB** | **~70 GB** | **减少 50%** |

### 4.3 训练流程对比

```
RLHF-PPO 一轮 iteration（Week 10 Day4-6）：
┌─────────────────────────────────────────────────────┐
│ Phase 1: Rollout                                     │
│   Actor 自回归生成 response（最慢！）                  │
│   记录 old_log_probs, old_values                     │
│                                                      │
│ Phase 2: Scoring                                     │
│   RM 对 response 打分                                │
│   Reference 计算 KL                                  │
│   构建 per-token reward                              │
│                                                      │
│ Phase 3: GAE                                         │
│   计算 Advantage 和 Returns                          │
│                                                      │
│ Phase 4: PPO Update                                  │
│   多轮 mini-batch 更新 Actor 和 Critic               │
└─────────────────────────────────────────────────────┘

DPO 一轮 iteration：
┌─────────────────────────────────────────────────────┐
│ Step 1: Forward                                      │
│   π_θ 和 π_ref 分别对 y_w 和 y_l 计算 log_probs     │
│   共 4 次 forward pass                               │
│                                                      │
│ Step 2: DPO Loss                                     │
│   计算 log ratio 差 → sigmoid → 交叉熵               │
│                                                      │
│ Step 3: Backward                                     │
│   标准反向传播更新 π_θ                                │
└─────────────────────────────────────────────────────┘
```

### 4.4 训练特性对比

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 数据类型 | On-policy（实时生成） | Offline（预收集偏好对） |
| 训练稳定性 | 差（RL 固有问题） | 好（类似 SFT） |
| 超参数敏感性 | 极高（β, ε, lr, K 等） | 较低（主要是 β 和 lr） |
| Reward Hacking | 存在（RM 不完美） | 不存在（无显式 RM） |
| 代码复杂度 | 高（rollout + GAE + 多模型协同） | 低（接近 SFT） |
| 理论保证 | 在最优 RM 下收敛 | 与 RLHF 在最优解处等价 |
| 计算量 | 高（rollout 占 50-60% 时间） | 低（无自回归生成） |
| 在线反馈 | 支持（可实时生成新数据） | 不支持（固定偏好数据集） |
| RM 信号灵活性 | 高（可实时打分、可组合多个 RM） | 低（只能用偏好对） |

---

## 五、DPO 的优势与局限

### 5.1 DPO 的核心优势

**优势 1：工程简洁性**

DPO 的训练代码与 SFT 几乎一样简单：

```python
# DPO 核心代码（伪代码）
for batch in dataloader:
    # 4 次 forward pass
    logps_w = get_log_probs(policy, batch.chosen)      # π_θ(y_w|x)
    logps_l = get_log_probs(policy, batch.rejected)    # π_θ(y_l|x)
    ref_logps_w = get_log_probs(ref_model, batch.chosen)   # π_ref(y_w|x)
    ref_logps_l = get_log_probs(ref_model, batch.rejected) # π_ref(y_l|x)
    
    # DPO Loss
    log_ratio_w = logps_w - ref_logps_w  # β log(π_θ/π_ref) for chosen
    log_ratio_l = logps_l - ref_logps_l  # β log(π_θ/π_ref) for rejected
    loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()
    
    loss.backward()
    optimizer.step()
```

对比 RLHF-PPO 的数百行训练循环，DPO 只需十几行核心代码。

**优势 2：训练稳定性**

DPO 的训练曲线像 SFT 一样平稳：

```
SFT 训练:  ━━━━━━━━━━━━━━━━━━━━━━━━→ 平稳收敛
DPO 训练:  ━━━━━━━━━━━━━━━━━━━━━━━━→ 平稳收敛（类似 SFT）
RLHF 训练: ━━╱╲━━╱╲╱╲━━━╱╲━━━━━━━━→ 不稳定收敛
```

原因：DPO 是 offline 监督学习，数据分布固定，不存在 on-policy RL 的非平稳性问题。

**优势 3：无 Reward Hacking**

DPO 没有显式 RM，因此不存在 Goodhart 定律的问题。策略直接从偏好数据学习，而非从一个可能有偏的 RM 中学习。

**优势 4：计算效率**

- 无需自回归生成（rollout 是 RLHF 最慢的部分）
- 无需 Critic 的训练和前向传播
- 无需 GAE 计算
- 总训练时间可能只有 RLHF 的 1/3 ~ 1/5

### 5.2 DPO 的局限性

**局限 1：Offline 数据依赖**

DPO 只能用预收集的偏好数据训练，无法在训练过程中探索新的 response 空间。

| 场景 | RLHF | DPO |
|------|------|-----|
| 策略当前无法生成的好回答 | 可通过 PPO 探索到 | 无法学到（不在数据中） |
| 数据分布外的 prompt | 可实时生成 response 并优化 | 只能泛化，无法适应 |
| 偏好数据过时 | 可重新生成 | 需要重新收集数据 |

这是 DPO 最根本的局限——**off-policy 方法的固有缺陷**。

**局限 2：数据质量依赖**

由于 DPO 直接从偏好数据学习，数据质量对最终效果影响极大：
- 偏好数据中的噪声直接传导到策略
- 无法像 RLHF 那样通过实时生成来「自我纠正」
- 偏好数据的分布需要覆盖目标策略的分布（off-policy coverage 假设）

**局限 3：β 的选择仍然重要**

虽然 DPO 的超参数比 RLHF 少得多，但 β 的选择仍然关键：
- β 太大：策略变化太小，接近 $\pi_{\text{ref}}$，学不到偏好
- β 太小：策略变化太大，可能过拟合偏好数据中的噪声

**局限 4：缺乏显式 Reward 信号**

在 RLHF 中，RM 的打分可以用于：
- 运行时的 Best-of-N 采样
- 在线数据筛选
- 多任务 reward 组合
- 训练过程中的实时监控

DPO 没有显式 RM，这些功能需要额外实现（可通过隐式 reward 部分恢复）。

---

## 六、DPO 在工业界的应用

### 6.1 为什么工业界偏爱 DPO

DPO 自 2023 年发表以来迅速成为工业界的首选对齐方案，原因包括：

| 因素 | 说明 |
|------|------|
| 工程简单 | 只需在 SFT 代码基础上稍作修改 |
| 资源高效 | 显存减半，训练时间大幅缩短 |
| 易于调试 | 训练行为可预测，类似 SFT |
| 效果不差 | 在多数任务上匹配或超过 PPO |
| 迭代快速 | 实验周期短，方便快速迭代 |

### 6.2 使用 DPO 的知名模型

| 模型 | 对齐方案 | 说明 |
|------|---------|------|
| LLaMA 2 Chat | RLHF (PPO + RS) | Meta，早期仍用 PPO |
| Zephyr | DPO | HuggingFace，首个大规模 DPO 模型 |
| Mixtral Instruct | DPO | Mistral AI |
| Tulu 2 | DPO | Allen AI |
| Neural Chat | DPO | Intel |
| DeepSeek-R1 | GRPO | DeepSeek，用 GRPO 替代 PPO/DPO |
| Claude | RLHF + Constitutional AI | Anthropic |
| GPT-4 | RLHF (推测) | OpenAI |

### 6.3 DPO 的实践流程

```
典型 DPO 训练流程：
1. 准备基座模型（如 LLaMA-7B）
2. SFT 微调 → 得到 π_SFT（作为 π_ref）
3. 收集偏好数据 {(x, y_w, y_l)}
   - 人工标注
   - 或用强模型（GPT-4）做 AI Feedback
4. DPO 训练
   - 冻结 π_ref = π_SFT
   - 初始化 π_θ = π_SFT（deepcopy）
   - 在偏好数据上优化 DPO Loss
5. 评估
   - MT-Bench / AlpacaEval
   - 人类评估
```

---

## 七、DPO 的变体预览

DPO 论文发表后，涌现了大量改进工作。这里简要预览，Day 7 将详细讨论：

| 变体 | 核心改进 | 解决的问题 |
|------|---------|-----------|
| **IPO** (Azar et al., 2023) | 用恒等映射替代 log sigmoid | 避免 DPO 在完美分离数据上过拟合 |
| **KTO** (Ethayarajh et al., 2024) | 不需要偏好对，只需要 thumbs up/down | 降低数据收集成本 |
| **SimPO** (Meng et al., 2024) | 去掉 Reference Model，用长度归一化 | 进一步简化，不需要 $\pi_{\text{ref}}$ |
| **ORPO** (Hong et al., 2024) | 将 SFT 和偏好学习合并为一步 | 不需要单独的 SFT 阶段 |
| **R-DPO** | 添加长度正则化 | 防止长度偏见 |

---

## 八、与 Week 10 的衔接

### 8.1 知识复用

| W10 知识 | W11 中的应用 |
|---------|------------|
| RLHF 优化目标（Day 1） | DPO 推导的起点 |
| Bradley-Terry 模型（Day 2） | DPO Loss 推导的关键环节 |
| RM Loss（Day 2-3） | 对比 DPO 的隐式 RM |
| RLHF-PPO 训练循环（Day 6） | 与 DPO 训练循环对比 |
| Reward Hacking（Day 7） | DPO 如何避免 Reward Hacking |
| 闭式最优解（Day 1, 7） | DPO 推导的数学基础 |

### 8.2 从 W10 Day7 的预览到本周的深入

W10 Day7 已经预览了 DPO 的核心思路：
- 闭式解 → 反解 reward → 代入 BT → 配分函数消除 → DPO Loss

本周将：
- Day 1（今天）：精读论文，建立全局理解
- Day 2：严格推导每一步，包括梯度分析
- Day 3：从零手写 DPO 训练循环

---

## 九、自检题

### 概念理解

1. DPO 解决了 RLHF 的哪些痛点？列出至少三个。
2. "Your Language Model is Secretly a Reward Model" 这句话的数学含义是什么？
3. DPO 需要几个模型？分别是什么？与 RLHF 相比少了什么？
4. DPO 是 online 方法还是 offline 方法？这意味着什么？

### 原理理解

5. DPO 的核心数学洞察是什么？（用一句话概括）
6. 为什么配分函数 $Z(x)$ 能在 DPO 推导中被消掉？
7. DPO 的隐式 reward 是什么？写出公式。
8. DPO 与 RLHF 在数学上是什么关系？（近似 / 等价 / 上界？）

### 对比分析

9. 列出 DPO 相比 RLHF 的三个核心优势和三个核心局限。
10. 在什么场景下你会选择 RLHF 而非 DPO？为什么？
11. DPO 的训练稳定性为什么好于 RLHF？根本原因是什么？
12. 为什么 DPO 没有 Reward Hacking 问题？

### 面试准备

13. 面试官问「DPO 和 RLHF 的本质区别是什么」，你如何回答？
14. 面试官问「DPO 是不是一定比 RLHF 好」，你如何回答？

---

## 十、产出要求

- [ ] 撰写 DPO 论文精读笔记（覆盖动机、核心思想、实验结论）
- [ ] 画出 DPO vs RLHF 的架构对比图（标注模型数量、数据流、显存差异）
- [ ] 用一段话解释 DPO 的核心思想（面试口述练习）
- [ ] 写出 DPO 的隐式 reward 公式并解释含义
- [ ] 列出 DPO 的优势与局限（各至少 3 点，附具体原因）
- [ ] 完成全部自检题

---

## 十一、与 Day 2 的衔接

今天我们从宏观视角理解了 DPO 的动机和核心思想。明天将进入数学核心——**DPO Loss 的完整推导**。具体来说：

- 从 RLHF 优化目标严格推导闭式最优解 $\pi^*$
- 从闭式解反解 reward
- 将反解的 reward 代入 Bradley-Terry 模型
- 证明配分函数 $Z(x)$ 在差值中消除
- 得到最终的 DPO Loss 公式
- 分析 DPO Loss 的梯度——理解 chosen 和 rejected 的权重如何自动调节
- 写出完整的 DPO 训练伪代码

Day 2 的每一步推导都将直接对应 Day 3 代码中的一行。掌握推导是手写代码的前提。
