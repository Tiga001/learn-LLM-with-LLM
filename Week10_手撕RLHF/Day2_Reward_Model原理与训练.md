# Day 2：Reward Model 原理与训练 — 让模型学会「什么是好回答」

> **目标**：深入理解 Reward Model 的数学原理——从偏好数据的概率建模出发，推导 Bradley-Terry 模型及其与 Elo 评分的联系；完整推导 RM 的训练目标函数；理解 RM 的网络架构设计；分析偏好数据质量对 RM 的影响；认识 Reward Hacking 问题及其危害。本日是 Day 3 手写 RM 的理论基础，也是理解 RLHF 全链路的关键环节。

---

## 一、偏好数据的数学建模

### 1.1 从偏好到概率

Day 1 我们知道 RLHF 的核心数据是偏好对 $(x, y_w, y_l)$。现在需要回答一个关键问题：

> 如何把「人类觉得 $y_w$ 比 $y_l$ 好」这个主观判断，转化为数学上可优化的目标？

直觉思路：假设每个 response 都有一个内在的「质量分数」$r(x, y) \in \mathbb{R}$，分数越高表示越好。那么：

$$P(y_w \succ y_l \mid x) = f(r(x, y_w) - r(x, y_l))$$

其中 $f$ 是某个单调递增函数：分数差越大，被偏好的概率越高。

### 1.2 模型选择：为什么用 sigmoid？

最自然的选择是 $f = \sigma$（sigmoid 函数），理由如下：

1. **概率归一化**：$\sigma(z) + \sigma(-z) = 1$，确保 $P(y_w \succ y_l) + P(y_l \succ y_w) = 1$
2. **单调性**：$\sigma$ 严格单调递增，分数差越大概率越高
3. **平滑性**：处处可微，适合梯度优化
4. **历史传承**：这就是 Bradley-Terry 模型，在竞技排名中已有几十年历史

---

## 二、Bradley-Terry 模型

### 2.1 模型定义

Bradley-Terry (BT) 模型是最经典的偏好比较模型（1952 年提出），定义为：

$$\boxed{P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l)) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}}$$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 sigmoid 函数。

### 2.2 推导：从 softmax 到 sigmoid

BT 模型等价于对两个选项做 softmax：

$$P(y_w \succ y_l) = \frac{e^{r(y_w)}}{e^{r(y_w)} + e^{r(y_l)}} = \frac{1}{1 + e^{-(r(y_w) - r(y_l))}} = \sigma(r(y_w) - r(y_l))$$

推导关键步骤：分子分母同除以 $e^{r(y_w)}$，得到 sigmoid 形式。

### 2.3 BT 模型的性质

**性质 1：传递性（在概率意义上）**

如果 $r(A) > r(B) > r(C)$，则 $P(A \succ C) > P(A \succ B) > 0.5$。

**性质 2：平移不变性**

对所有 response 的分数加同一常数不改变偏好概率：

$$\sigma((r(y_w) + c) - (r(y_l) + c)) = \sigma(r(y_w) - r(y_l))$$

这意味着 RM 的绝对分数没有意义，只有相对分数差有意义。

**性质 3：尺度敏感性**

将所有分数乘以常数 $\alpha$ 会改变概率分布的「锐度」：
- $\alpha > 1$：概率更极端（更接近 0 或 1）
- $\alpha < 1$：概率更平缓（更接近 0.5）

### 2.4 与 Elo 评分系统的联系

BT 模型与国际象棋的 Elo 评分系统本质相同：

$$P(\text{A 胜 B}) = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$

这就是 $\sigma\left(\frac{\ln 10}{400}(R_A - R_B)\right)$——只是尺度系数不同。

Chatbot Arena 用的就是 Elo 评分来排名不同的 LLM，底层数学模型完全一致。

---

## 三、RM 训练目标推导

### 3.1 最大似然估计

给定偏好数据集 $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N$，用最大似然估计来训练 RM：

$$\max_\theta \prod_{i=1}^N P_\theta(y_w^{(i)} \succ y_l^{(i)} \mid x^{(i)})$$

取对数并取负：

$$\min_\theta -\frac{1}{N} \sum_{i=1}^N \log P_\theta(y_w^{(i)} \succ y_l^{(i)} \mid x^{(i)})$$

代入 BT 模型：

$$\boxed{L_{\text{RM}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \log \sigma\left(r_\theta(x^{(i)}, y_w^{(i)}) - r_\theta(x^{(i)}, y_l^{(i)})\right)}$$

### 3.2 Loss 的直觉理解

$$L = -\log \sigma(\Delta r), \quad \Delta r = r_\theta(y_w) - r_\theta(y_l)$$

- 当 $\Delta r \to +\infty$（RM 正确且自信）：$L \to 0$
- 当 $\Delta r = 0$（RM 无法区分）：$L = \log 2 \approx 0.693$
- 当 $\Delta r \to -\infty$（RM 判断错误）：$L \to +\infty$

```
Loss
  ^
  |   ╲
  |    ╲
  |     ╲
  |      ╲
0.69 ─ ─ ─╲─ ─ ─ ─ RM 随机猜测的 Loss
  |         ╲
  |          ╲──────────
  |
  +──────────────────────→ Δr = r(y_w) - r(y_l)
  -3  -2  -1   0   1   2   3
```

### 3.3 梯度分析

$$\frac{\partial L}{\partial r_\theta(y_w)} = -(1 - \sigma(\Delta r)) = -\sigma(-\Delta r)$$

$$\frac{\partial L}{\partial r_\theta(y_l)} = (1 - \sigma(\Delta r)) = \sigma(-\Delta r)$$

直觉：梯度让 $r(y_w)$ 增大、$r(y_l)$ 减小。当 RM 已经非常自信时（$\Delta r$ 很大），梯度趋近于 0，不再更新——这是一种自然的自适应学习率。

### 3.4 与二分类交叉熵的关系

RM Loss 本质上就是二分类交叉熵：

$$L_{\text{RM}} = -\log \sigma(\Delta r) = \text{BCE}(\sigma(\Delta r), 1)$$

把 $\Delta r$ 看作 logit，标签始终为 1（$y_w$ 应该得分更高）。代码实现只需一行：

```python
loss = -torch.log(torch.sigmoid(r_w - r_l)).mean()
# 等价于：
loss = F.binary_cross_entropy_with_logits(r_w - r_l, torch.ones_like(r_w))
```

### 3.5 InstructGPT 的排序 Loss 变体

InstructGPT 的标注是对 $K$ 个 response 做排序（而非两两对比），因此 Loss 改为：

$$L_{\text{RM}} = -\frac{1}{\binom{K}{2}} \sum_{(i,j): y_i \succ y_j} \log \sigma(r_\theta(x, y_i) - r_\theta(x, y_j))$$

关键细节：**来自同一排序的所有偏好对必须在同一 batch 中**。否则 RM 会过拟合到对每个偏好对「刚好」正确，而非学到有意义的全局排序。

---

## 四、RM 网络架构

### 4.1 基本设计：LLM + Scalar Head

RM 的标准架构是在 LLM backbone 上加一个线性 head：

```
Input: [prompt, response]  (拼接后 tokenize)
          ↓
    LLM Backbone (e.g., GPT / LLaMA)
          ↓
    取最后一个 token 的隐状态 h_T
          ↓
    Linear(hidden_dim, 1)  ← Reward Head
          ↓
    r ∈ ℝ  (标量奖励分数)
```

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.reward_head = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        # 取每个序列最后一个非 padding token 的隐状态
        reward = self.reward_head(last_hidden[:, -1, :])
        return reward.squeeze(-1)
```

### 4.2 架构设计的关键选择

| 选择 | 选项 | InstructGPT | LLaMA2 |
|------|------|------------|--------|
| Backbone 大小 | 与 Actor 相同 / 更小 | 6B（Actor 是 175B） | 70B（与 Actor 同） |
| 初始化 | 从 SFT 初始化 / 从预训练初始化 | 从 SFT | 从预训练 |
| 取哪个 token | 最后一个 / [EOS] / 平均池化 | 最后一个 | 最后一个 |
| Reward Head | Linear / MLP | Linear (无 bias) | Linear |

### 4.3 为什么取最后一个 token

取最后一个 token 的隐状态而非平均池化，原因是：

1. **因果注意力**：GPT 类模型中，最后一个 token 能看到所有前面的 token，包含最完整的信息
2. **计算效率**：只需取一个向量，不需要额外的池化操作
3. **与生成对齐**：LLM 生成时也是基于最后一个 token 的隐状态来预测下一个 token

### 4.4 RM 大小的权衡

| RM 大小 | 优点 | 缺点 |
|---------|------|------|
| 小于 Actor | 推理快、显存省 | 可能表达能力不够 |
| 与 Actor 相同 | 表达能力强 | 显存压力大（四模型） |
| 大于 Actor | 更准确的奖励信号 | 实际很少使用 |

实践建议：在资源允许时，RM 尽量大。LLaMA2 使用与 Actor 同大小的 70B RM，效果显著。

---

## 五、偏好数据质量

### 5.1 影响 RM 质量的因素

| 因素 | 影响 | 解决方案 |
|------|------|---------|
| 标注者一致性 | 不一致的标注会让 RM 学到矛盾信号 | 多人标注 + 多数投票 |
| 标注者偏见 | RM 学到的是标注者偏好，不是「客观好坏」 | 多样化标注者 |
| 数据分布 | RM 在训练分布外泛化能力差 | 保证数据覆盖足够广 |
| 排序 vs 对比 | 排序数据提供更多信号 | 使用排序 + 排序 Loss |
| 数据量 | 太少会过拟合 | 通常需要数万条 |

### 5.2 标注指南的重要性

InstructGPT 的标注指南定义了三个维度的偏好标准：

1. **有用性（Helpfulness）**：是否回答了用户的问题，信息是否准确
2. **诚实性（Honesty）**：不确定时是否承认，是否杜撰事实
3. **无害性（Harmlessness）**：是否包含有害、歧视、不当内容

这三个维度有时会冲突（比如用户要求有害内容时，有用性和无害性矛盾），标注指南需要定义优先级。

### 5.3 数据增强技巧

- **对比顺序随机化**：打乱 $(y_w, y_l)$ 的输入顺序，防止位置偏见
- **同 prompt 多 response**：从排序中提取更多偏好对
- **AI 反馈（RLAIF）**：用强模型（如 GPT-4）代替人类标注，降低成本

---

## 六、Reward Hacking

### 6.1 什么是 Reward Hacking

Reward Hacking（奖励黑客）是 RLHF 最严重的问题之一：

> 策略学会利用 RM 的漏洞来获取高分，而非真正提升回答质量。

本质原因：**RM 不是完美的奖励信号**，它只是从有限数据中学到的人类偏好近似。当 PPO 对 RM 打分进行激进优化时，会找到 RM 的盲区。

### 6.2 Reward Hacking 的典型表现

| 类型 | 表现 | 原因 |
|------|------|------|
| 长度偏见 | 生成越来越长的回答 | 标注者倾向于选择更详细的回答 |
| 格式偏见 | 过度使用列表、加粗、标题 | 标注者倾向于选择格式更好的回答 |
| 重复 | 重复关键短语 | RM 对某些模式给高分 |
| 讨好 | 过度附和用户，缺乏独立判断 | 标注者偏好被认同的感觉 |
| 无意义高分 | 生成语法正确但无实质内容的文本 | RM 在 OOD 数据上的泛化失败 |

### 6.3 KL Penalty 作为对策

RLHF 中的 KL penalty 是对抗 Reward Hacking 的第一道防线：

$$\text{total\_reward} = r_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

$\beta$ 的选择是一个关键的工程决策：

| $\beta$ | 效果 |
|---------|------|
| 太小 | 约束太弱，容易 reward hacking |
| 太大 | 约束太强，策略几乎不更新 |
| 合适 | RM 分数提升的同时，KL 保持在合理范围 |

典型做法：从 $\beta = 0.01 \sim 0.1$ 开始，根据训练过程中 KL 的变化动态调整。

### 6.4 其他对策

| 对策 | 方法 | 代价 |
|------|------|------|
| RM ensemble | 用多个 RM 取平均分 | 推理成本翻倍 |
| 约束输出长度 | 对超长回答做惩罚 | 可能伤害需要长回答的 prompt |
| 定期重新标注 | 用最新策略的输出重新训练 RM | 需要持续标注 |
| Process RM | 对每一步而非最终结果打分 | 标注成本极高 |
| 使用 DPO | 不需要 RM，直接从偏好数据优化 | W11 详解 |

---

## 七、RM 训练的实践要点

### 7.1 训练配方

| 超参数 | 典型值 | 说明 |
|--------|-------|------|
| 学习率 | 1e-5 ~ 5e-6 | 比 SFT 更小 |
| Epochs | 1~2 | RM 容易过拟合 |
| Batch size | 64~128 | 尽量大 |
| 序列长度 | 512~2048 | 取决于任务 |
| 优化器 | AdamW | 标准选择 |
| 权重衰减 | 0.01 | 轻度正则化 |

### 7.2 评估指标

| 指标 | 含义 | 期望值 |
|------|------|--------|
| Accuracy | 正确预测偏好对的比例 | >65%（超过标注者一致性即可） |
| Loss | $-\log \sigma(\Delta r)$ 的平均值 | 持续下降 |
| Reward gap | $r(y_w) - r(y_l)$ 的均值 | 应适度正值，不能太大 |
| Reward std | RM 输出分数的标准差 | 不能太小（无区分度）也不能太大（过拟合） |

### 7.3 过拟合的警告信号

- 训练准确率很高但验证准确率停滞或下降
- Reward gap 持续增大（RM 越来越极端）
- 在 PPO 阶段出现严重的 reward hacking

---

## 八、数学补充：Plackett-Luce 模型

### 8.1 从 pair 到 ranking

Bradley-Terry 模型处理的是两两对比。当有 $K$ 个 response 的完整排序时，可以用 Plackett-Luce (PL) 模型：

$$P(\text{ranking: } y_1 \succ y_2 \succ \cdots \succ y_K) = \prod_{i=1}^{K} \frac{\exp(r(y_i))}{\sum_{j=i}^{K} \exp(r(y_j))}$$

直觉：从 $K$ 个中选最好的概率是 softmax，然后从剩下 $K-1$ 个中选最好的，以此类推。

### 8.2 PL 模型与 BT 模型的关系

当 $K = 2$ 时，PL 模型退化为 BT 模型：

$$P(y_1 \succ y_2) = \frac{\exp(r(y_1))}{\exp(r(y_1)) + \exp(r(y_2))} = \sigma(r(y_1) - r(y_2))$$

InstructGPT 选择从排序中提取 pairwise 对比而非使用 PL Loss，因为实践中 pairwise Loss 更稳定。

---

## 九、自检题

### 数学推导

1. 写出 Bradley-Terry 模型的公式，并证明 $P(y_w \succ y_l) + P(y_l \succ y_w) = 1$。
2. 从最大似然估计推导 RM Loss。
3. 计算 RM Loss 关于 $r_\theta(y_w)$ 的梯度。当 $r(y_w) \gg r(y_l)$ 时梯度如何变化？
4. 解释 BT 模型的平移不变性，它对 RM 训练有什么影响？
5. 写出 Plackett-Luce 模型的公式，并证明 $K=2$ 时退化为 BT 模型。

### 架构与工程

6. 画出 RM 的网络架构图（从输入到标量输出）。
7. 为什么 RM 取最后一个 token 的隐状态而非平均池化？
8. RM 应该从 SFT checkpoint 初始化还是从预训练 checkpoint 初始化？各自的优缺点？
9. 如何检测 RM 是否过拟合？

### Reward Hacking

10. 列举三种 Reward Hacking 的表现，并分析其产生原因。
11. KL penalty 如何缓解 Reward Hacking？$\beta$ 太大或太小分别会怎样？

---

## 十、产出要求

- [ ] 推导 Bradley-Terry 模型：从偏好假设到 $P(y_w \succ y_l) = \sigma(r_w - r_l)$
- [ ] 推导 RM Loss：$L_{\text{RM}} = -\mathbb{E}[\log \sigma(r_w - r_l)]$
- [ ] 画出 RM 的网络架构图
- [ ] 分析 RM Loss 关于 $\Delta r$ 的图像
- [ ] 总结 Reward Hacking 的类型与对策
- [ ] 理解 RM Loss 与二分类交叉熵的等价关系

---

## 十一、与 Day 3 的衔接

今天我们完成了 RM 的全部理论准备。明天将亲手实现一个 Reward Model：

- 构建偏好数据集（模拟数据或小规模真实数据）
- 在 GPT-2 backbone 上加 reward head
- 实现 Bradley-Terry Loss
- 训练 RM 直到偏好预测准确率 >80%
- 可视化 RM 的打分分布
- 演示 Reward Hacking 现象

Day 2 的每一个公式在 Day 3 都会变成代码。
