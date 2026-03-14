# Day 4：RLHF-PPO 架构详解 — 四模型协同的工程挑战

> **目标**：系统理解 RLHF-PPO 的四模型架构——Actor（被优化的 LLM）、Critic（价值网络）、Reference（冻结的 SFT 模型）、Reward Model（冻结的奖励模型）；掌握各模型的角色、数据流和参数共享策略；深入理解 RLHF 的 rollout 过程（prompt → generate → score → update）；分析四模型的显存占用，理解大规模 RLHF 训练的工程瓶颈。本日是 RLHF 从理论到工程的关键桥梁。

---

## 一、回顾：从经典 PPO 到 RLHF-PPO

### 1.1 Week 9 的 PPO

Week 9 我们实现的经典 PPO 只有**一个模型**（ActorCritic），与 Gymnasium 环境交互：

```
┌──────────────┐    action    ┌──────────────┐
│  ActorCritic  │ ──────────→ │  Environment  │
│  (Actor+Critic│ ←────────── │  (CartPole)   │
│   共享网络)    │  state, reward└──────────────┘
└──────────────┘
```

### 1.2 RLHF-PPO 需要四个模型

RLHF 场景下，「环境」变成了「生成回答 + 获得打分」的过程，需要**四个独立的模型**协同工作：

```
┌──────────────────────────────────────────────────────────────────┐
│                      RLHF-PPO 四模型架构                          │
│                                                                   │
│  ┌─────────────────┐              ┌─────────────────┐            │
│  │  1. Actor (π_θ)  │              │  2. Critic (V_ϕ) │            │
│  │  被优化的 LLM     │              │  价值网络         │            │
│  │  ← 梯度更新       │              │  ← 梯度更新       │            │
│  └────────┬────────┘              └────────┬────────┘            │
│           │ 生成 response                   │ 估计 V(s)           │
│           ▼                                ▼                      │
│  ┌─────────────────┐              ┌─────────────────┐            │
│  │ 3. Reference     │              │ 4. Reward Model  │            │
│  │    (π_ref)       │              │    (RM / r_ϕ)    │            │
│  │  冻结的 SFT 模型  │              │  冻结的奖励模型   │            │
│  │  ← 不更新        │              │  ← 不更新        │            │
│  └─────────────────┘              └─────────────────┘            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 二、四模型详解

### 2.1 Actor（$\pi_\theta$）— 被优化的 LLM

| 属性 | 说明 |
|------|------|
| 角色 | 生成 response 的策略网络 |
| 初始化 | 从 SFT 模型初始化 |
| 是否更新 | **是**（PPO 梯度更新） |
| 输入 | prompt $x$ |
| 输出 | response $y = (y_1, y_2, \ldots, y_T)$，以及每个 token 的 log probability |
| 对应 W9 | ActorCritic 中的 Actor Head |

Actor 就是我们最终想要的模型——经过 RLHF 优化后能生成符合人类偏好的回答。

### 2.2 Critic（$V_\phi$）— 价值网络

| 属性 | 说明 |
|------|------|
| 角色 | 估计每个状态（token 位置）的期望回报 |
| 初始化 | 通常从 SFT 模型或 RM 初始化 |
| 是否更新 | **是**（Value Loss 更新） |
| 输入 | prompt + 已生成的 token 序列 $[x, y_{\leq t}]$ |
| 输出 | 标量价值估计 $V_\phi(s_t) \in \mathbb{R}$ |
| 对应 W9 | ActorCritic 中的 Critic Head |

Critic 的作用是估计「从当前位置开始，还能获得多少总奖励」，为 GAE 提供 baseline。

**架构**：与 RM 类似，在 LLM backbone 上加一个 Value Head：

```python
class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden)
        values = self.linear(hidden_states).squeeze(-1)  # (batch, seq_len)
        return values
```

注意：Critic 输出的是**每个 token 位置**的价值，而非单个标量。这是因为 RLHF 中每个 token 都是一个「动作」。

### 2.3 Reference Model（$\pi_{\text{ref}}$）— 冻结的 SFT 模型

| 属性 | 说明 |
|------|------|
| 角色 | 提供 KL 约束的锚点 |
| 初始化 | Actor 的初始副本（= SFT 模型） |
| 是否更新 | **否**（完全冻结） |
| 输入 | 与 Actor 相同的 prompt + response |
| 输出 | 每个 token 的 log probability $\log \pi_{\text{ref}}(y_t \mid x, y_{<t})$ |
| 对应 W9 | 无直接对应（经典 PPO 没有 reference） |

Reference 的唯一作用是计算 KL divergence：

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \sum_t \left[ \log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t) \right]$$

它防止 Actor 为了追求高 RM 分数而偏离正常语言分布。

### 2.4 Reward Model（$r_\phi$）— 冻结的奖励模型

| 属性 | 说明 |
|------|------|
| 角色 | 对 response 打分 |
| 初始化 | Day 3 训练好的 RM |
| 是否更新 | **否**（完全冻结） |
| 输入 | 完整的 (prompt, response) |
| 输出 | 标量奖励 $r_\phi(x, y) \in \mathbb{R}$ |
| 对应 W9 | Gymnasium 环境的 reward 函数 |

RM 是 RLHF 三阶段中 Stage 2 的产物，在 Stage 3（PPO）中被当作固定的「环境」来使用。

### 2.5 四模型对比总览

| 模型 | 符号 | 是否更新 | 输出 | 用途 |
|------|------|---------|------|------|
| Actor | $\pi_\theta$ | ✓ | token 概率 | 生成 response |
| Critic | $V_\phi$ | ✓ | token 级 value | GAE baseline |
| Reference | $\pi_{\text{ref}}$ | ✗ | token 概率 | KL 约束 |
| Reward | $r_\phi$ | ✗ | 序列级 reward | 奖励信号 |

---

## 三、参数共享策略

### 3.1 为什么需要参数共享

四个模型各一份完整的 LLM，显存压力巨大。以 LLaMA-7B 为例：

| 模型 | 参数量 | FP16 显存 | 训练显存（含梯度+优化器） |
|------|-------|-----------|---------------------|
| Actor | 7B | 14 GB | ~56 GB |
| Critic | 7B | 14 GB | ~56 GB |
| Reference | 7B | 14 GB | 14 GB（只推理） |
| Reward | 7B | 14 GB | 14 GB（只推理） |
| **总计** | **28B** | **56 GB** | **~140 GB** |

这远超单卡 80GB 的限制。因此需要参数共享来降低显存。

### 3.2 常见的共享策略

#### 策略 A：Actor-Critic 共享 Backbone

最常见的做法——Actor 和 Critic 共享 LLM backbone，各自只维护独立的 head：

```
┌──────────────────────────────┐
│    共享 LLM Backbone          │
│    (e.g., LLaMA-7B)          │
├──────────────────────────────┤
│  Actor Head:  Linear → logits │  ← 原始 LM head
│  Value Head:  Linear → V(s)   │  ← 新加的 value head
└──────────────────────────────┘
```

**优点**：减少约 50% 的 Actor+Critic 显存
**缺点**：Actor 和 Critic 的梯度可能冲突

#### 策略 B：Reference = Actor 的初始权重

Reference 始终是 Actor 训练前的冻结副本：

```python
# 初始化
actor = AutoModelForCausalLM.from_pretrained("sft_model")
reference = AutoModelForCausalLM.from_pretrained("sft_model")

# 冻结 reference
for param in reference.parameters():
    param.requires_grad = False
```

#### 策略 C：RM 与 Critic 共享 Backbone

某些实现中，Critic 从 RM 初始化（而非从 SFT 初始化），因为 RM 已经学会了「好坏」的概念：

```
RM backbone → Critic backbone（初始化后分离，Critic 继续更新）
```

### 3.3 实际架构（以 GPT-2 为例）

Day 6 我们将采用的架构：

```python
class ActorCriticLM(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        vocab_size = self.backbone.config.vocab_size
        
        # Actor Head: 复用预训练的 LM Head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Critic Head: 新加的 Value Head
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask).last_hidden_state
        logits = self.lm_head(hidden)          # (batch, seq, vocab)
        values = self.value_head(hidden).squeeze(-1)  # (batch, seq)
        return logits, values
```

---

## 四、RLHF Rollout 全流程

### 4.1 完整流程图

RLHF-PPO 的一个训练 iteration 包含以下步骤：

```
Step 1: Rollout（采样阶段）
  ┌─────────────────────────────────────────────────────────┐
  │ 输入: batch of prompts {x_1, ..., x_B}                  │
  │                                                          │
  │ Actor (π_θ) 自回归生成:                                   │
  │   for each prompt x:                                     │
  │     y = [] (空 response)                                 │
  │     for t = 1, 2, ..., T:                                │
  │       logits_t = Actor(x, y_{<t})                        │
  │       y_t ~ Categorical(softmax(logits_t))               │
  │       log_prob_t = log π_θ(y_t | x, y_{<t})             │
  │       y.append(y_t)                                      │
  │     end                                                  │
  │                                                          │
  │ 同时收集:                                                 │
  │   old_log_probs: π_θ 在每个 token 上的 log prob          │
  │   values: V_ϕ(s_t) (Critic 在每个 token 上的估值)        │
  └─────────────────────────────────────────────────────────┘
                              ↓
Step 2: Scoring（打分阶段）
  ┌─────────────────────────────────────────────────────────┐
  │ Reward Model 打分:                                       │
  │   R = RM(x, y)    ← 整个 response 的标量奖励             │
  │                                                          │
  │ Reference Model 计算 KL:                                 │
  │   for each token t:                                      │
  │     ref_log_prob_t = log π_ref(y_t | x, y_{<t})         │
  │     kl_t = log_prob_t - ref_log_prob_t                   │
  │                                                          │
  │ 构建 per-token reward:                                    │
  │   r_t = -β · kl_t              (t < T)                  │
  │   r_T = RM(x, y) - β · kl_T   (最后一个 token)           │
  └─────────────────────────────────────────────────────────┘
                              ↓
Step 3: Advantage Estimation（优势估计）
  ┌─────────────────────────────────────────────────────────┐
  │ 使用 GAE 计算 Advantage:                                  │
  │   δ_t = r_t + γ · V(s_{t+1}) - V(s_t)                  │
  │   A_t = Σ (γλ)^l · δ_{t+l}                             │
  │                                                          │
  │ 计算 returns:                                             │
  │   G_t = A_t + V(s_t)                                    │
  └─────────────────────────────────────────────────────────┘
                              ↓
Step 4: PPO Update（策略更新）
  ┌─────────────────────────────────────────────────────────┐
  │ for epoch = 1, ..., K:                                   │
  │   for mini_batch in shuffle(data):                       │
  │                                                          │
  │     new_log_probs = log π_θ(y_t | x, y_{<t})  ← 重算    │
  │     new_values = V_ϕ(s_t)                     ← 重算    │
  │                                                          │
  │     ratio = exp(new_log_probs - old_log_probs)           │
  │                                                          │
  │     Policy Loss (PPO-Clip):                              │
  │       L_policy = -E[min(ratio·A, clip(ratio)·A)]        │
  │                                                          │
  │     Value Loss:                                          │
  │       L_value = E[(new_values - G_t)²]                   │
  │                                                          │
  │     Total Loss = L_policy + c_v · L_value                │
  │     Backprop + optimizer.step()                          │
  │                                                          │
  └─────────────────────────────────────────────────────────┘
```

### 4.2 时序图

从时间维度看一轮 RLHF 更新：

```
Time  ──────────────────────────────────────────────────→

Phase 1: Rollout (推理, no grad)
  Actor   ████████████  生成 response
  Critic  ████████████  估计 values
  
Phase 2: Scoring (推理, no grad)  
  RM      ████  打分
  Ref     ████████  计算 KL

Phase 3: GAE (CPU/GPU 计算)
  GAE     ██  计算 advantages

Phase 4: PPO Update (训练, with grad) × K epochs
  Actor   ████  重算 log_probs → policy loss → backward
  Critic  ████  重算 values → value loss → backward
```

注意：Phase 1-3 是**推理阶段**（`torch.no_grad()`），Phase 4 才是**训练阶段**。

### 4.3 关键细节：per-token reward 的构建

RLHF 中 RM 只在 response 结束时给出一个标量分数，但 PPO 需要 per-token 的 reward 来计算 GAE。

解决方案：在每个 token 上加 KL penalty 作为中间 reward：

$$r_t = \begin{cases} -\beta \cdot \text{kl}_t & t < T \\ r_\phi(x, y) - \beta \cdot \text{kl}_T & t = T \end{cases}$$

其中 $\text{kl}_t = \log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t)$。

这样做的好处：
1. 每个 token 都有 reward 信号（KL penalty），缓解了稀疏奖励问题
2. GAE 可以正常工作，不需要特殊处理
3. KL penalty 自然地分布到每个 token 上

```python
def compute_rewards(rm_scores, log_probs, ref_log_probs, kl_coef):
    """构建 per-token reward。"""
    kl = log_probs - ref_log_probs       # per-token KL
    rewards = -kl_coef * kl              # KL penalty on every token
    # 最后一个 token 加上 RM score
    rewards[:, -1] += rm_scores
    return rewards
```

---

## 五、PPO 在 LLM 中的映射（详解）

### 5.1 状态空间

经典 RL 中状态是环境的观测（如 CartPole 的 4 维向量）。RLHF 中：

$$s_t = (x, y_1, y_2, \ldots, y_{t-1})$$

状态就是 prompt 加上已经生成的 token 序列。由于 LLM 的因果注意力机制，每个位置的隐状态已经编码了完整的历史信息。

### 5.2 动作空间

$$a_t = y_t \in \{1, 2, \ldots, |V|\}$$

动作是从词表 $V$ 中选择下一个 token。动作空间的大小就是词表大小（GPT-2 约 50K，LLaMA 约 32K）——比经典 RL 的 2~4 个动作大了数万倍。

### 5.3 策略

$$\pi_\theta(a_t \mid s_t) = P_\theta(y_t \mid x, y_{<t}) = \text{softmax}(\text{logits}_t)_{y_t}$$

策略就是 LLM 的 next-token 分布——这是 RLHF 最优雅的地方：**LLM 天然就是一个策略网络**。

### 5.4 回合（Episode）

一个 episode 就是一次完整的 response 生成：

```
s_0 = (prompt)
  → a_0 = y_1 → r_0 = -β·kl_0
s_1 = (prompt, y_1)
  → a_1 = y_2 → r_1 = -β·kl_1
...
s_{T-1} = (prompt, y_1, ..., y_{T-1})
  → a_{T-1} = y_T → r_{T-1} = RM(x, y) - β·kl_{T-1}
```

episode 结束的标志：生成 EOS token 或达到最大长度。

### 5.5 与经典 PPO 的关键差异总览

| 维度 | 经典 PPO (W9) | RLHF-PPO (W10) |
|------|-------------|----------------|
| 状态 | 低维向量（4 维） | token 序列（变长） |
| 动作 | 离散少量（2~4） | 离散大量（词表 ~50K） |
| 策略 | MLP 网络 | 整个 LLM |
| 奖励 | 每步即时奖励 | 只有终末 RM 奖励 + per-token KL |
| Episode 长度 | 数百步 | 数十~数百 token |
| 模型数量 | 1（ActorCritic） | 4（Actor/Critic/Ref/RM） |
| 推理成本 | 微秒级 | 秒级（自回归生成） |
| 训练显存 | < 1 GB | 数十 ~ 数百 GB |

---

## 六、显存分析

### 6.1 单模型显存拆解

以 LLaMA-7B（FP16）为例，一个模型的显存占用：

| 组件 | 大小 | 说明 |
|------|------|------|
| 参数 | 14 GB | 7B × 2 bytes (FP16) |
| 梯度 | 14 GB | 与参数等大 |
| 优化器状态 | 28 GB | Adam: 2 × 参数大小（m 和 v） |
| 激活值 | 可变 | 取决于 batch size 和序列长度 |
| **训练总计** | **~56 GB** | 不含激活值 |
| **推理总计** | **14 GB** | 只需参数 |

### 6.2 四模型总显存

| 模型 | 模式 | 显存 |
|------|------|------|
| Actor (7B) | 训练 | ~56 GB |
| Critic (7B) | 训练 | ~56 GB |
| Reference (7B) | 推理 | ~14 GB |
| Reward (7B) | 推理 | ~14 GB |
| **总计** | | **~140 GB** |

这就是为什么 RLHF 训练需要多 GPU 的根本原因。

### 6.3 显存优化策略

| 策略 | 显存节省 | 代价 |
|------|---------|------|
| Actor-Critic 共享 backbone | ~50 GB | 梯度可能冲突 |
| LoRA 微调 Actor | 大幅减少梯度和优化器 | 可能损失精度 |
| Reference model offload | 14 GB → CPU 内存 | 推理变慢 |
| 使用更小的 RM | 取决于 RM 大小 | 奖励信号质量下降 |
| Gradient checkpointing | ~30% 激活值 | 训练速度降 20~30% |
| DeepSpeed ZeRO | 按 GPU 数均分 | 通信开销 |

### 6.4 GPT-2 small 的显存（Day 6 实际使用）

| 模型 | 参数量 | FP32 显存 |
|------|-------|-----------|
| Actor (GPT-2) | 124M | ~0.5 GB |
| Critic (GPT-2) | 124M + Head | ~0.5 GB |
| Reference (GPT-2) | 124M | ~0.5 GB |
| Reward (GPT-2) | 124M + Head | ~0.5 GB |
| **总计** | **~500M** | **~2 GB** |

GPT-2 small 的规模让我们可以在单 GPU 上完成完整的 RLHF 实验。

---

## 七、Generation（自回归生成）细节

### 7.1 Rollout 阶段的生成

在 RLHF 的 rollout 阶段，Actor 需要自回归生成完整的 response：

```python
def generate_rollout(actor_model, prompts, max_new_tokens, temperature=1.0):
    """
    RLHF rollout: Actor 自回归生成 response。
    需要同时收集:
    - generated_ids: 生成的 token ids
    - log_probs: 每个 token 的 log π_θ(y_t | s_t)
    """
    generated = prompts.clone()  # (batch, prompt_len)
    all_log_probs = []
    
    for t in range(max_new_tokens):
        with torch.no_grad():
            logits = actor_model(generated).logits[:, -1, :]  # (batch, vocab)
        
        # 采样
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        
        # 收集 log prob
        log_prob = F.log_softmax(logits / temperature, dim=-1)
        token_log_prob = log_prob.gather(1, next_token)  # (batch, 1)
        all_log_probs.append(token_log_prob)
        
        # 拼接
        generated = torch.cat([generated, next_token], dim=1)
        
        # 检查 EOS
        if (next_token == eos_token_id).all():
            break
    
    return generated, torch.cat(all_log_probs, dim=1)
```

### 7.2 Top-p / Temperature 的角色

在 RLHF rollout 中，采样策略会影响训练：

| 参数 | 作用 | RLHF 中的选择 |
|------|------|--------------|
| Temperature | 控制分布锐度 | 通常 1.0（不做调整） |
| Top-p | 截断低概率 token | 有时使用，但会影响 log_prob |
| Top-k | 只保留 top-k token | 较少使用 |

PPO 理论上假设采样来自完整分布 $\pi_\theta$，使用 top-p/top-k 会引入 bias。实践中大部分 RLHF 系统使用 temperature=1.0 的标准采样。

---

## 八、工程实现模式

### 8.1 同步模式 vs 异步模式

**同步模式**（简单，Day 6 使用）：
```
Generate → Score → Compute GAE → PPO Update → Generate → ...
```

**异步模式**（高效，生产环境）：
```
Thread 1: Generate → Generate → Generate → ...
Thread 2:           Score → GAE → Update → Score → ...
```

异步模式可以隐藏生成的延迟，但实现复杂度高。

### 8.2 Batch 组织

RLHF 中的 batch 组织与 SFT 不同：

```
SFT batch:   直接从数据集采样 (prompt, response) 对
RLHF batch:  只采样 prompts，response 由 Actor 实时生成
```

这意味着 RLHF 的每个 batch 都是 on-policy 的新数据。

### 8.3 Prompt 管理

实践中需要管理一个 prompt 池：

```python
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, max_length):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        return self.tokenizer(
            self.prompts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
```

---

## 九、自检题

### 架构理解

1. RLHF-PPO 的四模型分别是什么？哪些需要梯度更新，哪些冻结？
2. 为什么需要 Reference Model？它在训练过程中的作用是什么？
3. Actor-Critic 共享 backbone 的优缺点是什么？
4. Critic 输出的是每个 token 的 value 还是整个序列的 value？为什么？

### 数据流理解

5. 画出 RLHF-PPO 一轮 iteration 的完整数据流（从 prompt 到参数更新）。
6. per-token reward 是如何构建的？RM 的分数加在哪个位置？
7. 为什么 rollout 阶段用 `torch.no_grad()`，而 PPO 更新阶段需要梯度？
8. RLHF 中的 episode 什么时候结束？

### 工程分析

9. 以 7B 模型为例，估算 RLHF 四模型的总显存需求。
10. 列出三种降低 RLHF 显存的策略，分析各自的代价。
11. RLHF 中的 batch 与 SFT 中的 batch 有什么本质区别？

### 面试准备

12. 用伪代码写出 RLHF-PPO 一轮 iteration 的流程（rollout → score → GAE → update）。
13. 解释 RLHF 中 state、action、policy、reward 分别对应什么。

---

## 十、产出要求

- [ ] 画出 RLHF 四模型架构图（标明哪些更新、哪些冻结）
- [ ] 画出一轮 RLHF iteration 的完整数据流（从 prompt 到参数更新）
- [ ] 写出 per-token reward 的构建公式
- [ ] 分析四模型的显存占用（以具体模型大小为例）
- [ ] 理解 Actor-Critic 参数共享的实现方式

---

## 十一、与 Day 5 的衔接

今天我们理解了 RLHF-PPO 的四模型架构和完整数据流。明天将深入推导 RLHF-PPO 的三部分 Loss：

- **Policy Loss**：PPO-Clip 在 LLM token 级别的适配
- **Value Loss**：Critic 如何学习准确估计回报
- **KL Penalty**：如何在 per-token 级别计算 KL 散度

Day 5 的 Loss 推导将直接对应 Day 6 手写代码中的每一行。
