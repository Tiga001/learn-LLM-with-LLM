# Day 7：PPO vs DPO vs GRPO 对比与复盘 — 三种对齐方案的系统分析

> **目标**：系统对比 PPO、DPO、GRPO 三种对齐方案的数学原理、训练流程、工程特性与适用场景——建立清晰的多维度对比框架；给出工业界的实际选型指南；简介 DPO 的主要变体（IPO、KTO、SimPO、ORPO）；串联本周 Day 1-6 的全部知识，提供核心公式速查表和分级检查点；为 Week 12 垂域 Chatbot 全流程实操做好技术储备。

---

## 一、三种方案的多维度对比

### 1.1 数学原理对比

| 维度 | PPO (W9-W10) | DPO (Day 1-3) | GRPO (Day 4-6) |
|------|-------------|---------------|----------------|
| **优化目标** | $\max \mathbb{E}[r] - \beta D_{\text{KL}}$ | 等价于 PPO 目标 | $\max \mathbb{E}[r] - \beta D_{\text{KL}}$ |
| **Reward 来源** | 显式 RM $r_\phi(x,y)$ | 隐式 $\beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$ | 显式 reward 函数 $r(x,y)$ |
| **Advantage 估计** | Critic + GAE | 隐式（在梯度中自动出现） | 组内归一化 $\frac{r_i - \bar{r}}{\sigma_r}$ |
| **Policy 更新** | PPO-Clip | log-sigmoid (类 SFT) | PPO-Clip |
| **KL 约束** | per-token KL penalty | 隐式（$\pi_{\text{ref}}$ 出现在 Loss 中） | 显式 KL penalty |
| **Loss 公式** | $-\min(\rho \hat{A}, \text{clip} \cdot \hat{A})$ | $-\log \sigma(\beta \Delta \log \text{ratio})$ | $-\min(\rho \hat{A}, \text{clip} \cdot \hat{A}) + \beta \text{KL}$ |

### 1.2 训练流程对比

| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| **数据方式** | On-policy（实时生成） | Offline（预收集偏好对） | On-policy（实时生成） |
| **训练数据** | Prompt 集 + RM 打分 | 偏好对 $(x, y_w, y_l)$ | Prompt 集 + reward 函数 |
| **一轮训练步骤** | Rollout → RM Score → GAE → PPO Update | Forward × 4 → DPO Loss → Backward | Sample G × → Reward → Normalize → GRPO Update |
| **需要 Rollout** | 是（最慢的环节） | 否 | 是（但可并行） |
| **训练复杂度** | 高 | 低（类似 SFT） | 中 |
| **收敛速度** | 较慢（RL 探索） | 快（监督学习） | 中等 |

### 1.3 工程特性对比

| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| **模型数量** | 4 (Actor + Critic + Ref + RM) | 2 (Policy + Ref) | 2 (Policy + Ref) |
| **显存 (7B)** | ~140 GB | ~70 GB | ~70 GB |
| **训练稳定性** | 差 | 好（类似 SFT） | 中（优于 PPO） |
| **超参数敏感度** | 极高 (β, ε, lr, K 等) | 较低 (β, lr) | 中 (β, ε, lr, G) |
| **Reward Hacking** | 存在 | 不存在 | 取决于 reward 设计 |
| **代码行数 (核心)** | ~200+ 行 | ~40 行 | ~80 行 |
| **调试难度** | 高 | 低 | 中 |

### 1.4 适用场景对比

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| **通用对话对齐** | DPO | 偏好数据易收集，训练稳定 |
| **数学推理增强** | GRPO | 答案可验证，规则 reward 精确 |
| **编程能力增强** | GRPO | 测试用例可验证，规则 reward 精确 |
| **安全对齐** | PPO 或 DPO | 需要细粒度偏好判断 |
| **超大模型 (>100B)** | GRPO 或 DPO | 避免 PPO 的 Critic 开销 |
| **资源受限** | DPO | 显存最少，训练最快 |
| **需要实时探索** | PPO 或 GRPO | On-policy 可以探索新回答 |
| **偏好数据充足** | DPO | 充分利用 offline 数据 |
| **偏好数据稀缺** | GRPO | 只需 reward 函数，不需要偏好对 |

---

## 二、工业界实际选型指南

### 2.1 决策流程图

```
你要做什么任务？
       │
       ├── 通用对话对齐（聊天、写作、翻译）
       │      │
       │      ├── 有偏好数据？ ──→ 是 ──→ DPO（首选）
       │      │                 └→ 否 ──→ 用 GPT-4 生成 AI Feedback → DPO
       │      │
       │      └── 需要实时探索？ ──→ PPO（但成本高）
       │
       ├── 推理增强（数学、编程、逻辑）
       │      │
       │      ├── 答案可自动验证？ ──→ 是 ──→ GRPO（首选）
       │      │                      └→ 否 ──→ 训练 RM → PPO 或 DPO
       │      │
       │      └── 模型超大 (>100B)？ ──→ GRPO（避免 Critic 开销）
       │
       └── 安全/有害性对齐
              │
              ├── Constitutional AI ──→ RLAIF → DPO 或 PPO
              └── 人类偏好 ──→ PPO 或 DPO
```

### 2.2 不同公司/团队的选择

| 模型 | 对齐方案 | 理由 |
|------|---------|------|
| GPT-4 (OpenAI) | RLHF (PPO) | 资源充足，追求最优效果 |
| Claude (Anthropic) | RLHF + Constitutional AI | 强调安全性，需要细粒度控制 |
| LLaMA 2 Chat (Meta) | PPO + Rejection Sampling | 早期工作，遵循 InstructGPT 路线 |
| Zephyr (HuggingFace) | DPO | 资源高效，简洁有效 |
| Mixtral Instruct (Mistral) | DPO | 工程简洁性 |
| DeepSeek-R1 | GRPO | 推理任务 + 超大模型 + 规则 reward |
| Qwen-2.5 | DPO | 通用对齐 |
| Tulu 2 (Allen AI) | DPO | 研究可复现性 |

### 2.3 实际建议

**如果你刚开始做对齐**：从 DPO 开始。训练稳定、代码简单、效果不错。

**如果你做推理增强**：用 GRPO。数学/编程的答案可以精确验证，不需要 RM。

**如果你追求极致效果**：PPO 或 Iterative DPO。但需要更多工程投入。

**如果你要做全流程（W12）**：建议 SFT → DPO，简单可靠。有余力再对比 GRPO。

---

## 三、其他对齐方案简介

### 3.1 DPO 变体

DPO 发表后涌现了大量改进工作：

#### IPO (Identity Preference Optimization)

**论文**：*A General Theoretical Paradigm to Understand Learning from Human Feedback* (Azar et al., 2023)

**核心改进**：用恒等映射替代 DPO 中的 log sigmoid。

$$L_{\text{IPO}} = \mathbb{E}\left[\left(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)} - \frac{1}{2}\right)^2\right]$$

**解决的问题**：DPO 在完美可分离数据上会让概率趋向 0/1，导致过拟合。IPO 用 L2 Loss 替代 log-sigmoid，避免这个问题。

#### KTO (Kahneman-Tversky Optimization)

**论文**：*KTO: Model Alignment as Prospect Theoretic Optimization* (Ethayarajh et al., 2024)

**核心改进**：不需要偏好对，只需要单条 response 的好/坏标签。

$$L_{\text{KTO}} = \mathbb{E}\left[\lambda_w \sigma(-z_w) \cdot \mathbb{1}[y = y_w] + \lambda_l \sigma(z_l) \cdot \mathbb{1}[y = y_l]\right]$$

**解决的问题**：偏好对比数据收集成本高。KTO 只需要 thumbs up / thumbs down 标注。

#### SimPO (Simple Preference Optimization)

**论文**：*SimPO: Simple Preference Optimization with a Reference-Free Reward* (Meng et al., 2024)

**核心改进**：去掉 Reference Model，用长度归一化的 log probability 作为隐式 reward。

$$L_{\text{SimPO}} = -\log \sigma\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l) - \gamma\right)$$

**解决的问题**：DPO 仍需要 Reference Model（增加显存），且存在长度偏见。SimPO 完全不需要 $\pi_{\text{ref}}$，并通过长度归一化解决长度偏见。

#### ORPO (Odds Ratio Preference Optimization)

**论文**：*ORPO: Monolithic Preference Optimization without Reference Model* (Hong et al., 2024)

**核心改进**：将 SFT 和偏好学习合并为一步，用 odds ratio 惩罚 rejected response。

$$L_{\text{ORPO}} = L_{\text{SFT}}(y_w) + \lambda \cdot L_{\text{OR}}(y_w, y_l)$$

**解决的问题**：DPO 需要先做 SFT 再做 DPO，ORPO 一步完成。

### 3.2 各方案的关系图

```
RLHF (PPO) ──→ DPO (去 RM) ──→ SimPO (去 Ref)
     │                │              │
     │                ├→ IPO (L2 Loss)
     │                ├→ KTO (无需偏好对)
     │                └→ ORPO (合并 SFT)
     │
     └──→ GRPO (去 Critic) ──→ DeepSeek-R1
```

---

## 四、全周知识串联

### 4.1 SFT → RM → PPO → DPO → GRPO 演进路线

```
SFT (W5)
  │  监督学习，只有正样本
  │  局限：无法学习偏好排序
  ▼
Reward Model (W10 Day 2-3)
  │  学会判断好坏
  │  局限：RM ≠ 真实偏好 → Reward Hacking
  ▼
RLHF-PPO (W10 Day 4-6)
  │  用 RM 信号优化策略
  │  局限：4 模型 / 不稳定 / 显存爆炸
  ▼
DPO (W11 Day 1-3)
  │  绕过 RM，直接从偏好学习
  │  局限：Offline / 依赖偏好数据
  ▼
GRPO (W11 Day 4-6)
  │  去掉 Critic，组内对比
  │  适合：可验证任务 + 超大模型
  ▼
下一步 (W12)
  垂域 Chatbot 全流程实操
  选择最适合的方案
```

### 4.2 每天的核心收获

| Day | 主题 | 核心收获 |
|-----|------|---------|
| Day 1 | DPO 论文精读 | DPO 动机、隐式 RM 思想、4→2 模型简化 |
| Day 2 | DPO Loss 推导 | 闭式解 → 反解 reward → 代入 BT → Z(x) 消除 → DPO Loss |
| Day 3 | 手写 DPO 训练 | 偏好数据 → get_log_probs → DPO Loss → 训练 → 隐式 Reward |
| Day 4 | R1 论文精读 | R1-Zero 涌现 / R1 四阶段管线 / GRPO 应用 / 蒸馏 |
| Day 5 | GRPO 推导 | 去 Critic 动机 → 组内归一化 → GRPO Loss |
| Day 6 | 手写 GRPO 训练 | 多组采样 → 组内 Advantage → PPO-Clip + KL → 训练 |
| Day 7 | 三方案对比 | PPO vs DPO vs GRPO 多维度对比 / 选型指南 / 变体简介 |

---

## 五、核心公式速查表

### 5.1 RLHF 优化目标（三种方法的共同起点）

$$\max_{\pi_\theta} \; \mathbb{E}[r(x,y)] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

### 5.2 闭式最优解

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

### 5.3 DPO Loss（Day 2，面试 Tier 1）

$$L_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

### 5.4 隐式 Reward

$$\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

### 5.5 GRPO Advantage（Day 5，面试 Tier 2）

$$\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}, \quad \bar{r} = \frac{1}{G}\sum_j r_j, \quad \sigma_r = \text{std}(\{r_j\})$$

### 5.6 GRPO Loss（Day 5，面试 Tier 2）

$$L_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^G \left[\min(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1\pm\epsilon) \hat{A}_i) - \beta \hat{D}_{\text{KL},i}\right]$$

### 5.7 PPO Policy Loss（W9，面试 Tier 1）

$$L^{\text{PPO}} = -\mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t^{\text{GAE}}, \text{clip}(\rho_t, 1\pm\epsilon) \hat{A}_t^{\text{GAE}}\right)\right]$$

### 5.8 RM Loss（W10 Day 2，Bradley-Terry）

$$L_{\text{RM}} = -\mathbb{E}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

---

## 六、自检题

### 对比类（面试高频）

1. 用一个表格对比 PPO / DPO / GRPO 的数据方式、模型数量、Advantage 来源、适用场景。
2. 在什么场景下你会选择 DPO 而非 PPO？给出至少三个理由。
3. 在什么场景下你会选择 GRPO 而非 DPO？给出至少三个理由。
4. DPO 和 GRPO 分别适合什么类型的任务？为什么？
5. 三种方案中哪个训练最稳定？为什么？哪个训练最不稳定？为什么？

### 推导类（面试核心）

6. 不看笔记，写出 DPO Loss 的完整公式。
7. 不看笔记，写出 GRPO 的组内归一化 Advantage 公式。
8. 从 RLHF 目标出发，用 5 步推导 DPO Loss。
9. 解释 DPO 推导中配分函数 $Z(x)$ 为什么能消掉。
10. 解释 GRPO 中归一化的三个作用。

### 工程类

11. DPO 训练循环的代码核心是什么？写出伪代码。
12. GRPO 训练循环的代码核心是什么？写出伪代码。
13. 如果你要为一个 7B 模型做对齐，资源有限（1 张 A100），你选择哪种方案？为什么？
14. 如果你要训练一个 670B MoE 模型做数学推理，你选择哪种方案？为什么？

### 深入类

15. DPO 的梯度中 $\sigma(-\hat{u})$ 的自适应权重机制是什么？
16. GRPO 的 Advantage 估计与 PPO 的 GAE 相比，偏差和方差各如何？
17. Iterative DPO 如何缓解 DPO 的 offline 局限？
18. R1-Zero 中推理能力涌现的可能解释有哪些？
19. 列出至少三种 DPO 变体及其核心改进。
20. 如果你要设计一个新的对齐算法，你会从哪个方向改进？

---

## 七、产出要求

- [ ] 画出 PPO vs DPO vs GRPO 的多维度对比表（至少 8 个维度）
- [ ] 画出工业选型决策流程图
- [ ] 默写 DPO Loss 和 GRPO Loss 的完整公式
- [ ] 默写本周所有核心公式（速查表中的 8 个公式）
- [ ] 完成全部自检题（重点：对比类 1-5，推导类 6-10）
- [ ] 为 W12 垂域 Chatbot 选择对齐方案并给出理由

---

## 八、关键检查点

### Tier 1（面试必须能闭眼手写）

- [ ] DPO Loss 公式与推导（5 步）
- [ ] DPO 训练循环伪代码
- [ ] 解释 DPO 中「隐式 Reward Model」的含义
- [ ] PPO vs DPO 的核心区别（至少 4 个维度）

### Tier 2（面试加分，需熟练手写）

- [ ] GRPO Loss 公式与推导
- [ ] GRPO 训练循环伪代码
- [ ] GRPO 组内归一化 Advantage 的公式与性质
- [ ] PPO vs DPO vs GRPO 的三方对比（至少 6 个维度）
- [ ] R1 的训练管线（四阶段）

### Tier 3（深入理解，能说清思路）

- [ ] DPO 梯度的自适应权重分析
- [ ] GRPO 的 Advantage 估计与 GAE 的偏差-方差对比
- [ ] IPO / KTO / SimPO / ORPO 的核心改进
- [ ] R1-Zero 推理涌现的可能解释
- [ ] On-policy vs Offline 学习的本质差异

---

## 九、Week 12 衔接预告

### 9.1 垂域 Chatbot 全流程

Week 12 将把前 11 周学到的所有技术整合为一个**完整的垂域大模型 Chatbot**：

```
数据处理 (W7) → Tokenizer 扩展 (W7) → 二次预训练 (W7)
                                        ↓
                                    SFT 微调 (W5-W6)
                                        ↓
                                    Reward Model (W10) 或跳过
                                        ↓
                                    DPO / PPO / GRPO (W10-W11)
                                        ↓
                                    垂域 Chatbot 部署
```

### 9.2 本周知识在 W12 的应用

| W11 知识 | W12 应用 |
|---------|---------|
| DPO 训练循环 | 用于垂域偏好对齐（如果选择 DPO） |
| GRPO 训练循环 | 用于可验证任务的对齐（如果选择 GRPO） |
| PPO vs DPO vs GRPO 对比 | 方案选型依据 |
| 隐式 Reward 提取 | DPO 训练效果评估 |
| R1 训练管线 | 多阶段训练策略参考 |

### 9.3 建议

对于 W12 的垂域 Chatbot 项目，建议：
1. **首选 DPO**：工程简单、训练稳定、效果不错
2. 如果任务可验证（如医疗问答有标准答案），可以尝试 GRPO
3. 如果资源充足且追求最优效果，可以 DPO → PPO 两阶段
4. 所有方案都需要先做好 SFT——SFT 是一切对齐的起点
