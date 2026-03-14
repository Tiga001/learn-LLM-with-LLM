# Day 1：InstructGPT 论文精读与 RLHF 概述 — 从 SFT 到人类对齐

> **目标**：理解 RLHF 的核心动机——为什么 SFT 不足以让 LLM 真正「对齐」人类意图；精读 InstructGPT 论文，掌握 RLHF 三阶段流程（SFT → Reward Model → PPO）的完整设计；解读 LLaMA2 论文中 RLHF 的改进策略；推导 RLHF 的数学优化目标；理解人类偏好数据的本质。本日建立对 RLHF 的全局视角，为后续 6 天的深入学习奠定基础。

---

## 一、为什么需要 RLHF

### 1.1 SFT 的局限性

Week 5 我们学习了 SFT（Supervised Fine-Tuning），用指令-回答对 $(x, y)$ 微调模型：

$$L_{\text{SFT}} = -\sum_t \log P_\theta(y_t \mid x, y_{<t})$$

SFT 确实能让模型学会「按指令回答」，但存在根本性问题：

| 问题 | 具体表现 | 根本原因 |
|------|---------|---------|
| 训练目标与人类偏好不一致 | 模型学会模仿数据中的回答，但不理解哪种回答更好 | MLE 目标只看似然，不看质量 |
| 标注瓶颈 | 写高质量回答很贵（每条 15~30 min） | 生成比判断难得多 |
| 多样性不足 | 模型倾向于生成「安全但无趣」的回答 | 监督数据分布有限 |
| 无法表达偏好排序 | SFT 只能学 0/1（这个回答对/错），无法学「A 比 B 好」 | 分类任务 vs 排序任务 |
| 有害内容 | 模型可能生成有害、不诚实或不安全的内容 | 预训练数据中包含有害内容 |

### 1.2 核心洞察：判断比生成容易

RLHF 的关键洞察是：

> **人类很难写出完美的回答，但很擅长判断哪个回答更好。**

对比标注成本：

| 标注类型 | 每条时间 | 专业要求 | 一致性 |
|---------|---------|---------|--------|
| 写高质量回答（SFT 数据） | 15~30 min | 高 | 低（不同标注者风格差异大） |
| 对比两个回答（偏好数据） | 1~5 min | 中 | 较高（二选一判断更容易一致） |

RLHF 巧妙地利用了这一点：先用大量廉价的偏好判断训练 Reward Model，再用 RM 的打分信号通过 PPO 优化 LLM。

### 1.3 RLHF 的历史脉络

```
2017  Christiano et al. — Deep RL from Human Preferences（首次提出从偏好学习奖励函数）
  │
2019  Ziegler et al. — Fine-Tuning LMs from Human Preferences（首次将 RLHF 用于 LM）
  │
2020  Stiennon et al. — Learning to Summarize from Human Feedback（摘要任务 RLHF）
  │
2022  Ouyang et al. — InstructGPT ★ 首次大规模 RLHF → GPT-3.5
  │
2023  Touvron et al. — LLaMA2（开源 RLHF 实践）
  │
2023  Anthropic — Constitutional AI / RLAIF
  │
2023  Rafailov et al. — DPO（W11：不需要 RM 的替代方案）
```

---

## 二、InstructGPT 论文精读

### 2.1 论文核心信息

- **标题**：*Training language models to follow instructions with human feedback*
- **作者**：Ouyang et al. (OpenAI, 2022)
- **核心贡献**：提出并验证了 RLHF 三阶段流程，使 1.3B 参数的 InstructGPT 在人类评估中优于 175B 的 GPT-3

### 2.2 RLHF 三阶段流程

InstructGPT 的训练分为三个阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RLHF 三阶段流程                                   │
│                                                                     │
│  Stage 1: SFT (Supervised Fine-Tuning)                              │
│  ┌──────────────────────────────────────────────┐                   │
│  │  收集高质量 (prompt, response) 数据            │                   │
│  │  → 监督微调 GPT-3                             │                   │
│  │  → 得到 π_SFT                                │                   │
│  │  数据量：~13K                                  │                   │
│  └──────────────────────────────────────────────┘                   │
│                          ↓                                          │
│  Stage 2: Reward Model Training                                     │
│  ┌──────────────────────────────────────────────┐                   │
│  │  对同一 prompt 生成多个 response               │                   │
│  │  → 人类标注偏好排序                            │                   │
│  │  → 训练 Reward Model (RM)                     │                   │
│  │  数据量：~33K prompts × K responses            │                   │
│  └──────────────────────────────────────────────┘                   │
│                          ↓                                          │
│  Stage 3: PPO Optimization                                          │
│  ┌──────────────────────────────────────────────┐                   │
│  │  从新 prompt 出发：                            │                   │
│  │  → π_θ 生成 response                          │                   │
│  │  → RM 打分                                    │                   │
│  │  → PPO 更新 π_θ（同时约束 KL(π_θ ∥ π_SFT)）   │                   │
│  │  数据量：~31K prompts（无需人工标注！）          │                   │
│  └──────────────────────────────────────────────┘                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Stage 1：SFT 的角色

SFT 在 RLHF 中的角色是**初始化**——给模型一个合理的起点：

- 数据来源：OpenAI API 用户提交的 prompt + 人工编写的高质量 response
- 数据量：约 13,000 条
- 训练方式：标准的监督微调（next-token prediction）
- 产出：$\pi_{\text{SFT}}$，作为后续 RM 训练和 PPO 优化的起点

SFT 的目的不是训练出最终模型，而是让模型「学会说话的基本方式」。

### 2.4 Stage 2：Reward Model 训练

RM 的核心思想：把人类偏好转化为标量奖励信号。

**数据收集过程**：
1. 对每个 prompt，用 $\pi_{\text{SFT}}$ 生成 $K$ 个不同的 response（$K$ = 4~9）
2. 人类标注者对 $K$ 个 response 进行排序
3. 从排序中提取 $\binom{K}{2}$ 个偏好对 $(y_w, y_l)$

**RM 训练目标**（Bradley-Terry 模型，Day 2 详细推导）：

$$L_{\text{RM}} = -\frac{1}{\binom{K}{2}} \sum_{(y_w, y_l)} \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

**InstructGPT 的 RM 细节**：
- 架构：6B GPT-3（比最终 Actor 小）
- 训练数据：~33K prompts，每个 4~9 个 response
- 从同一排序中提取的偏好对在同一 batch 中出现（减少过拟合）

### 2.5 Stage 3：PPO 优化

用 RM 的打分作为奖励信号，通过 PPO 优化 LLM：

$$\max_\theta \; \mathbb{E}_{x \sim D, \, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) \right] - \beta \cdot D_{\text{KL}}\left(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)\right)$$

其中：
- $r_\phi(x, y)$：RM 对 response $y$ 的打分
- $\pi_{\text{ref}} = \pi_{\text{SFT}}$：冻结的 SFT 模型作为 reference
- $\beta$：KL penalty 系数，控制策略偏移幅度
- $D_{\text{KL}}$：防止模型为了获取高 RM 分数而偏离正常语言分布

**为什么需要 KL Penalty？**

如果不加 KL 约束，模型会发现 RM 的漏洞（Reward Hacking）：
- 生成冗长重复的回答来刷高分
- 利用 RM 的偏见（如总是选择更长的回答）
- 偏离正常语言分布，生成无意义但高分的文本

KL penalty 确保优化后的 $\pi_\theta$ 不会偏离 $\pi_{\text{ref}}$ 太远。

### 2.6 InstructGPT 的关键结论

| 发现 | 意义 |
|------|------|
| 1.3B InstructGPT > 175B GPT-3（人类评估） | RLHF 的效果远超单纯的模型规模提升 |
| RLHF 模型更少产生有害内容 | 对齐不仅提升有用性，也提升安全性 |
| SFT + RLHF 优于纯 SFT | RM + PPO 阶段提供了 SFT 无法获得的优化信号 |
| 存在 alignment tax | RLHF 可能略微降低某些 NLP benchmark 性能 |
| PPO-ptx 缓解 alignment tax | 在 PPO 中混入预训练数据，防止知识遗忘 |

---

## 三、LLaMA2 的 RLHF 改进

### 3.1 LLaMA2 论文核心信息

- **标题**：*LLaMA 2: Open Foundation and Fine-Tuned Chat Models*
- **作者**：Touvron et al. (Meta, 2023)
- **核心贡献**：开源了 RLHF 训练的 Chat 模型，提出了多项 RLHF 工程改进

### 3.2 LLaMA2 的 RLHF 改进

| 改进 | InstructGPT | LLaMA2 | 目的 |
|------|------------|--------|------|
| RM 数量 | 1 个 RM | 2 个 RM（Safety RM + Helpfulness RM） | 分离安全性和有用性 |
| RLHF 迭代 | 单次 RLHF | 多轮迭代 RLHF（5 轮） | 逐步提升质量 |
| 拒绝采样 | 无 | Rejection Sampling（RS） | 在 PPO 前先用 RS 筛选 |
| PPO 方式 | 标准 PPO | PPO + RS 交替使用 | 更稳定的优化 |
| 安全对齐 | 混合训练 | 专门的 Safety RLHF 阶段 | 强化安全性 |

### 3.3 Rejection Sampling 策略

LLaMA2 引入的 Rejection Sampling 是一个关键创新：

```
对于每个 prompt x：
  1. 用当前策略 π_θ 生成 K 个 response {y_1, ..., y_K}
  2. 用 RM 对每个 response 打分
  3. 选择得分最高的 response 作为新的 SFT 数据
  4. 用选出的数据做 SFT 更新
```

这本质上是一种 Best-of-N 采样策略，可以看作是 PPO 的一种简化替代。LLaMA2 在实践中发现 RS 和 PPO 交替使用效果最好。

### 3.4 双 RM 架构

LLaMA2 用两个独立的 RM 分别建模「有用性」和「安全性」：

$$r_{\text{final}} = r_{\text{safety}} + \alpha \cdot r_{\text{helpfulness}}$$

当 $r_{\text{safety}}$ 检测到不安全内容时，$\alpha$ 会被降低，确保安全性优先。

---

## 四、RLHF 的数学框架

### 4.1 优化目标

RLHF 的核心优化目标可以统一写为：

$$\boxed{\max_\theta \; \mathbb{E}_{x \sim D, \, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) \right] - \beta \cdot D_{\text{KL}}\left(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right)}$$

这个目标有清晰的两部分：
1. **最大化奖励**：$\mathbb{E}[r_\phi(x, y)]$ — 让模型生成 RM 认为好的回答
2. **限制偏移**：$-\beta \cdot D_{\text{KL}}$ — 不要偏离参考策略太远

### 4.2 目标函数的最优解

这个带 KL 约束的优化问题有闭式最优解（Day 5 详细推导，W11 DPO 的理论基础）：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{1}{\beta} r_\phi(x, y)\right)$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{1}{\beta} r_\phi(x, y)\right)$ 是归一化常数。

直觉理解：最优策略是在参考策略的基础上，按奖励值的指数倍重新加权。高奖励的 response 概率增大，低奖励的减小。

### 4.3 为什么不直接用闭式解

虽然有闭式最优解，但无法直接使用，因为：

1. $Z(x)$ 需要对所有可能的 $y$ 求和——对 LLM 来说这是指数级的
2. 即使知道 $\pi^*$ 的形式，也无法直接把 LLM 的参数 $\theta$ 设成对应值

因此需要用 PPO 来迭代逼近这个最优解。（但这个闭式解启发了 DPO——W11 的核心内容。）

### 4.4 RLHF 中的 RL 映射

把 LLM 生成过程映射到 RL 框架：

| RL 概念 | RLHF 中的对应 | 说明 |
|---------|-------------|------|
| 环境 | 「给定 prompt，生成回答」的过程 | 非传统意义的环境 |
| 状态 $s_t$ | 到当前 token 为止的序列 $[x, y_{<t}]$ | 包含完整历史 |
| 动作 $a_t$ | 下一个 token $y_t$ | 从词表 $V$ 中选择 |
| 策略 $\pi_\theta(a|s)$ | LLM 的 next-token 分布 $P_\theta(y_t|x, y_{<t})$ | 就是语言模型本身 |
| 奖励 $r_t$ | 过程中为 0，末尾为 $R = \text{RM}(x, y) - \beta \cdot \text{KL}_t$ | 稀疏奖励 |
| 回合（episode） | 生成一个完整 response | 一条轨迹 |
| Reference Policy | 冻结的 SFT 模型 $\pi_{\text{ref}}$ | 锚定策略 |

### 4.5 稀疏奖励的挑战

RLHF 中的奖励是**极度稀疏**的：

```
token:  y_1    y_2    y_3   ...   y_T
reward:  0      0      0    ...   RM(x,y) - β·KL
```

只有在生成完整 response 后才能得到 RM 打分。这意味着：
- GAE 的计算需要特殊处理（Day 5 详解）
- 信用分配（credit assignment）变得困难
- 训练信号的方差较大

实践中常用的解决方案是在每个 token 上加入 per-token KL penalty 来提供中间信号。

---

## 五、人类偏好数据

### 5.1 偏好数据的本质

RLHF 的核心数据是**偏好对**（preference pairs）：

$$(x, y_w, y_l) \quad \text{表示对于 prompt } x, \text{ 回答 } y_w \text{ 优于 } y_l$$

这比 SFT 的数据更容易收集、也更自然：人类判断「A 比 B 好」远比「从零写出最优回答」容易。

### 5.2 偏好数据的收集流程

```
1. 准备 prompt 集合 → 来自用户请求或设计的测试集
       ↓
2. 生成多个 response → 用 π_SFT 生成 K 个候选
       ↓
3. 人类标注 → 对 K 个 response 进行排序
       ↓
4. 提取偏好对 → 从排序中提取 C(K,2) 个 (y_w, y_l) 对
       ↓
5. 质量控制 → 检查标注一致性、去除低质量标注
```

### 5.3 标注一致性

偏好标注的一个关键挑战是**标注者之间的不一致**：

| 数据集 | 标注者一致率 |
|--------|------------|
| InstructGPT | ~73%（标注者间一致性） |
| Anthropic HH-RLHF | ~65% |
| Chatbot Arena | ~65%（通过 Elo 评分体系聚合） |

标注一致性不高说明「好回答」的标准本身就是模糊的。这是 RLHF 的一个根本性限制——Reward Model 学到的是标注者群体的**平均偏好**。

### 5.4 常见偏好数据集

| 数据集 | 规模 | 特点 |
|--------|------|------|
| Anthropic HH-RLHF | ~170K 偏好对 | Helpfulness + Harmlessness |
| OpenAssistant | ~35K conversations | 多轮对话偏好 |
| Chatbot Arena | 持续增长 | 真实用户偏好、Elo 排名 |
| UltraFeedback | ~64K | GPT-4 标注的偏好 |

---

## 六、RLHF 三阶段之间的关系

### 6.1 数据流视角

```
Stage 1: SFT
  Input:  (prompt, response) 对
  Output: π_SFT
  ────────────────────────────────────────────
  
Stage 2: RM Training
  Input:  π_SFT 生成的 response + 人类偏好排序
  Output: r_ϕ(x, y)
  注意:   RM 的训练数据来自 π_SFT 的输出
  ────────────────────────────────────────────
  
Stage 3: PPO
  Input:  新的 prompts（无需人工标注！）
  Output: π_θ（最终对齐模型）
  注意:   奖励来自 RM，约束来自 π_ref = π_SFT
```

### 6.2 关键问题与回答

**Q1: 为什么不直接用更多 SFT 数据来对齐？**

SFT 的本质是模仿——它能学到「什么样的回答是合理的」，但无法学到「什么样的回答更好」。对齐需要的是偏好排序，不是单条样本。

**Q2: 为什么不直接用 RM 排序后做 SFT？**

这其实就是 Rejection Sampling（LLaMA2 使用的方法之一）。但 RS 只能利用当前策略能生成的最好 response，无法进行梯度级别的精细优化。PPO 能更高效地搜索策略空间。

**Q3: 三个阶段可以合并吗？**

理论上可以端到端训练，但实践中分阶段更稳定：
- SFT 提供良好初始化
- RM 提供稳定的奖励信号
- PPO 做精细优化

**Q4: RM 的质量有多重要？**

极其重要。RM 是整个 PPO 阶段的「指南针」——如果 RM 有偏见或错误，PPO 会放大这些问题（Reward Hacking）。

---

## 七、与 Week 9 的衔接

### 7.1 知识复用

Week 9 建立的所有 RL 基础在本周直接复用：

| W9 知识 | W10 中的应用 |
|---------|------------|
| MDP 框架 | LLM 生成过程的 MDP 建模 |
| Actor-Critic | RLHF 四模型中的 Actor 和 Critic |
| GAE | RLHF 中的 advantage 估计 |
| PPO-Clip | RLHF-PPO 的 policy loss |
| KL Penalty | RLHF 的核心约束 |

### 7.2 关键差异预览

| 维度 | W9 经典 PPO | W10 RLHF PPO |
|------|------------|-------------|
| 环境 | Gymnasium | LLM 生成过程 |
| 奖励 | 每步即时 | 只有最终 RM 打分 |
| 动作空间 | 2~4 个 | 词表大小（数万） |
| 模型数量 | 1 个 ActorCritic | 4 个模型 |
| KL 约束 | 无（或 PPO-Penalty） | 必须有（防止 reward hacking） |
| 计算量 | 分钟级 | 小时级（即使小模型） |

---

## 八、自检题

### 理论理解

1. SFT 的训练目标是什么？它为什么不足以实现人类对齐？
2. RLHF 三阶段分别是什么？各阶段的输入和输出是什么？
3. InstructGPT 的 SFT 数据量约多少？RM 数据量约多少？为什么 RM 数据量更大？
4. 写出 RLHF 的优化目标函数。解释每一项的含义。
5. KL penalty 的作用是什么？如果去掉会发生什么？
6. 解释稀疏奖励在 RLHF 中的含义。与经典 RL 有什么不同？

### LLaMA2 相关

7. LLaMA2 的双 RM 架构解决了什么问题？
8. Rejection Sampling 与 PPO 有什么关系？各自的优缺点？
9. LLaMA2 的迭代 RLHF 策略是什么？为什么比单次 RLHF 更好？

### 面试准备

10. 画出 RLHF 三阶段的完整流程图。
11. 解释 RLHF 中 state、action、policy、reward 分别对应什么。
12. 为什么 RLHF 需要 reference policy？它与 TRPO 的信赖域有什么联系？

---

## 九、产出要求

- [ ] 画出 RLHF 三阶段流程图（要求包含每阶段的输入/输出/数据来源）
- [ ] 撰写 InstructGPT 论文精读笔记（至少覆盖三阶段、数据规模、关键结论）
- [ ] 写出 RLHF 优化目标的数学公式，并解释每一项
- [ ] 总结 InstructGPT 与 LLaMA2 在 RLHF 上的关键差异
- [ ] 解释 RLHF 中的 RL 映射（state, action, reward, policy 各对应什么）

---

## 十、与 Day 2 的衔接

今天我们建立了 RLHF 的全局视角。明天将深入 Stage 2 的核心——**Reward Model**。具体来说：

- Bradley-Terry 模型的数学推导：为什么偏好对比可以用 sigmoid 建模？
- RM Loss 的完整推导：从偏好假设到训练目标
- RM 的架构设计：如何在 LLM backbone 上加 reward head？
- Reward Hacking：RM 的局限性如何导致优化失败？

Day 2 的 RM 理论将直接服务于 Day 3 的手写 RM 实践。
