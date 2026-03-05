# Day 5：参数高效微调方法 — Prompt Tuning / Prefix Tuning / Adapter

> **目标**：系统理解参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）的核心动机——在保持预训练权重不变的前提下，只用极少量可训练参数实现任务适配；精读 Prompt Tuning、Prefix Tuning、Adapter 三种经典方法的论文思想与数学形式；对比三者的参数量、计算开销、适用场景和效果差异；为第 6 周 LoRA / QLoRA 的深入学习建立理论基础。

---

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的困境

回顾 Day 1-2，全参数微调（Full Fine-tuning）的做法是更新模型的**所有参数**：

$$\theta_{\text{SFT}} = \theta_{\text{pretrain}} - \eta \sum_{(x,y) \in D} \nabla_\theta \mathcal{L}(x, y; \theta)$$

对于 LLaMA-7B，这意味着：

| 资源 | 需求 | 说明 |
|------|------|------|
| 可训练参数 | 6.7B | 全部参数都更新 |
| 模型参数显存 (FP16) | ~13 GB | $6.7B \times 2$ bytes |
| 优化器状态 (FP32) | ~52 GB | AdamW: $4 \times 13$ GB |
| 梯度 (FP16) | ~13 GB | 与模型参数同大小 |
| **总显存** | **~80+ GB** | 需要 4×A100 或 DeepSpeed |

```
全参数微调的核心问题:

1. 显存需求巨大
   7B 模型需要 80+ GB → 多卡才能训练
   → 大多数研究者和企业无法负担

2. 存储成本高
   每个下游任务需要存储一份完整的模型副本
   10 个任务 → 10 × 13 GB = 130 GB
   → 部署极不经济

3. 灾难性遗忘风险
   全参数更新可能大幅偏离预训练分布
   → 丢失通用能力

4. 过拟合风险
   小数据集上微调数十亿参数
   → 严重过拟合
```

### 1.2 PEFT 的核心思想

**参数高效微调**的核心假设：

> 预训练模型已经学到了丰富的通用表示。适配下游任务只需要**微小的调整**——不需要改变所有参数。

$$\theta_{\text{task}} = \theta_{\text{pretrain}} + \Delta\theta$$

其中 $\|\Delta\theta\| \ll \|\theta_{\text{pretrain}}\|$。

PEFT 的做法是：**冻结预训练参数 $\theta_{\text{pretrain}}$，只训练少量额外参数 $\phi$**。

$$\min_\phi \sum_{(x,y) \in D} \mathcal{L}(x, y; \theta_{\text{pretrain}}, \phi)$$

### 1.3 PEFT 方法分类

```
PEFT 方法家族:

┌────────────────────────────────────────────────────┐
│                  PEFT Methods                       │
├─────────────┬──────────────┬───────────────────────┤
│ Addition    │ Reparameter  │ Selection             │
│ (添加参数)   │ (重参数化)     │ (选择性更新)            │
├─────────────┼──────────────┼───────────────────────┤
│ Adapter     │ LoRA (W6)    │ BitFit                │
│ Prefix Tuning│ DoRA        │ 只调某些层              │
│ Prompt Tuning│             │                       │
│ (IA)³       │              │                       │
└─────────────┴──────────────┴───────────────────────┘

本日聚焦:
  ★ Prompt Tuning  → 在输入嵌入层添加可训练 token
  ★ Prefix Tuning  → 在每层 Attention 的 KV 前添加可训练前缀
  ★ Adapter        → 在 Transformer Block 中插入小型瓶颈网络

下周聚焦:
  ★★ LoRA / QLoRA  → 用低秩分解近似权重更新（第 6 周核心）
```

---

## 二、Prompt Tuning

### 2.1 核心论文

**论文**：*The Power of Scale for Parameter-Efficient Prompt Tuning* (Lester et al., 2021, Google)

**核心思想**：在输入序列前添加一段**可训练的连续向量**（soft prompt），只训练这些向量，冻结全部模型参数。

### 2.2 从 Discrete Prompt 到 Soft Prompt

```
传统 Discrete Prompt (人工模板):
  输入: "Classify the sentiment: This movie is great! Answer:"
  → 需要手工设计 prompt 模板
  → 模板的微小变化可能导致性能大幅波动
  → 受限于自然语言的离散空间

Prompt Tuning (连续可学习):
  输入: [P₁][P₂]...[Pₖ] + "This movie is great!"
  → Pᵢ 是可学习的连续向量（soft tokens）
  → 通过梯度下降自动优化
  → 在连续嵌入空间中搜索最优 prompt
```

### 2.3 数学形式

设预训练模型的词嵌入矩阵为 $E \in \mathbb{R}^{V \times d}$，输入 token 序列为 $x = [x_1, x_2, \ldots, x_n]$。

**标准前向传播**：

$$H_0 = [E(x_1), E(x_2), \ldots, E(x_n)] \in \mathbb{R}^{n \times d}$$

**Prompt Tuning 的前向传播**：

$$H_0 = [\underbrace{P_1, P_2, \ldots, P_k}_{\text{soft prompt}}, E(x_1), E(x_2), \ldots, E(x_n)] \in \mathbb{R}^{(k+n) \times d}$$

其中 $P = [P_1, P_2, \ldots, P_k] \in \mathbb{R}^{k \times d}$ 是**唯一的可训练参数**。

```
┌────────────────────────────────────────────────┐
│ Prompt Tuning 的前向传播                          │
│                                                  │
│  输入层:                                          │
│  [P₁][P₂]...[Pₖ] [x₁][x₂]...[xₙ]              │
│   ↑ 可训练         ↑ 来自词嵌入（冻结）             │
│                                                  │
│  拼接后送入冻结的 Transformer:                      │
│  Layer 1: 冻结 ❄️                                │
│  Layer 2: 冻结 ❄️                                │
│  ...                                             │
│  Layer L: 冻结 ❄️                                │
│                                                  │
│  输出: 只取原始 token 位置的 logits                 │
└────────────────────────────────────────────────┘
```

### 2.4 参数量分析

$$\text{可训练参数量} = k \times d$$

| 模型 | $d$ | Prompt 长度 $k$ | 可训练参数 | 占比 |
|------|------|---------|---------|------|
| T5-Small (60M) | 512 | 100 | 51,200 | 0.085% |
| T5-Base (220M) | 768 | 100 | 76,800 | 0.035% |
| T5-XXL (11B) | 4096 | 100 | 409,600 | 0.004% |
| LLaMA-7B | 4096 | 100 | 409,600 | 0.006% |

**关键发现**：模型越大，Prompt Tuning 的参数效率越高——只需 0.004% 的参数就能接近全参数微调的效果。

### 2.5 核心实验结果（论文）

Lester et al. 的核心发现——**Prompt Tuning 的效果随模型规模提升**：

```
T5 模型上 Prompt Tuning vs Full Fine-tuning 的效果对比:

模型规模    Full FT    Prompt Tuning    差距
Small        —          明显落后         大
Base         —          有差距           中
Large        —          接近             小
XL           —          接近             ~0
XXL (11B)    —          持平             ≈ 0  ← 关键拐点！

结论:
  模型规模 ≥ ~10B 时，Prompt Tuning ≈ Full Fine-tuning
  模型规模 < ~1B 时，Prompt Tuning 效果显著差于 Full FT
```

### 2.6 Prompt 长度的影响

| Prompt 长度 $k$ | SuperGLUE 平均分 | 说明 |
|---------|-----------|------|
| 1 | 较低 | 信息量太少 |
| 5 | 中等 | 可用但不理想 |
| 20 | 较好 | 接近最优 |
| 100 | **最优** | 论文推荐值 |
| 150 | 持平/微降 | 收益递减 |

### 2.7 代码实现

```python
import torch
import torch.nn as nn


class PromptTuning(nn.Module):
    """
    Prompt Tuning: 在输入嵌入前添加可学习的 soft prompt。
    
    只有 soft_prompt 参数是可训练的，预训练模型完全冻结。
    """

    def __init__(self, pretrained_model, prompt_length=100, init_from_vocab=True):
        super().__init__()
        self.model = pretrained_model
        self.prompt_length = prompt_length

        # 冻结预训练模型的全部参数
        for param in self.model.parameters():
            param.requires_grad = False

        d_model = self.model.config.hidden_size

        # 初始化 soft prompt
        if init_from_vocab:
            # 从词表中随机采样 token 嵌入作为初始化
            vocab_size = self.model.config.vocab_size
            init_ids = torch.randint(0, vocab_size, (prompt_length,))
            with torch.no_grad():
                init_embeds = self.model.get_input_embeddings()(init_ids)
            self.soft_prompt = nn.Parameter(init_embeds.clone())
        else:
            self.soft_prompt = nn.Parameter(
                torch.randn(prompt_length, d_model) * 0.02
            )

    def forward(self, input_ids, labels=None, **kwargs):
        batch_size = input_ids.shape[0]

        # 获取原始输入的嵌入
        input_embeds = self.model.get_input_embeddings()(input_ids)

        # 将 soft prompt 扩展到 batch 维度并拼接
        prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

        # 调整 attention mask 和 labels
        if labels is not None:
            prompt_labels = torch.full(
                (batch_size, self.prompt_length),
                -100,  # 忽略 prompt 位置的 loss
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat([prompt_labels, labels], dim=1)

        outputs = self.model(
            inputs_embeds=combined_embeds,
            labels=labels,
            **kwargs,
        )
        return outputs

    def trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total, trainable / total * 100
```

### 2.8 Prompt Tuning 的优缺点

| 维度 | 优势 | 劣势 |
|------|------|------|
| 参数效率 | 极高（0.01% 以下） | — |
| 实现简单 | 只改输入层，不修改模型内部 | — |
| 多任务部署 | 多个任务只需切换 prompt 向量 | — |
| 小模型效果 | — | 模型 < 10B 时效果显著差于全参 |
| 序列长度 | — | Prompt 占用上下文窗口 |
| 训练稳定性 | — | 对初始化和学习率敏感 |

---

## 三、Prefix Tuning

### 3.1 核心论文

**论文**：*Prefix-Tuning: Optimizing Continuous Prompts for Generation* (Li & Liang, 2021, Stanford)

**核心思想**：Prompt Tuning 只在输入层添加 soft prompt。Prefix Tuning 更进一步——**在每一层 Transformer 的 Attention 的 Key 和 Value 前都添加可训练的前缀向量**。

### 3.2 Prompt Tuning vs Prefix Tuning 的关键区别

```
Prompt Tuning:
  只在第 0 层（嵌入层）添加 soft prompt
  后续层没有额外的可训练参数
  → 信息只能通过冻结的 Transformer 层间接传播

Prefix Tuning:
  在每一层都添加可训练的 prefix
  每层的 Key 和 Value 前都有独立的可训练向量
  → 每一层都有直接的可训练参数注入
```

### 3.3 数学形式

对于 Transformer 的第 $l$ 层 Attention：

**标准 Attention**：

$$\text{Attn}^{(l)}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Prefix Tuning 的 Attention**：

$$K'^{(l)} = [\underbrace{P_K^{(l)}}_{\text{prefix}}, K^{(l)}], \quad V'^{(l)} = [\underbrace{P_V^{(l)}}_{\text{prefix}}, V^{(l)}]$$

$$\text{Attn}^{(l)}(Q, K', V') = \text{softmax}\left(\frac{Q {K'}^T}{\sqrt{d_k}}\right)V'$$

其中 $P_K^{(l)}, P_V^{(l)} \in \mathbb{R}^{k \times d}$ 是第 $l$ 层的可训练 prefix 参数。

```
┌─────────────────────────────────────────────────────────┐
│  Prefix Tuning 在每一层的工作方式:                          │
│                                                           │
│  Layer l:                                                 │
│                                                           │
│    Q = H^(l) × W_Q     (来自冻结的权重)                    │
│    K = [P_K^(l) | H^(l) × W_K]   ← prefix 拼在前面        │
│    V = [P_V^(l) | H^(l) × W_V]   ← prefix 拼在前面        │
│      ↑ 可训练     ↑ 冻结                                   │
│                                                           │
│    Attention = softmax(Q × K^T / √d) × V                  │
│                                                           │
│  → Q 可以"注意到" prefix，从而被可训练参数影响               │
│  → 但 prefix 不影响 Q 的计算（Q 来自冻结的权重）             │
└─────────────────────────────────────────────────────────┘
```

### 3.4 Reparameterization Trick

直接优化每层的 prefix 参数容易训练不稳定（参数空间大、梯度在深层可能爆炸/消失）。

Li & Liang 提出了一个**重参数化技巧**：用一个小型 MLP 来生成 prefix 参数。

$$P^{(l)} = \text{MLP}(P_{\text{embed}}^{(l)})$$

其中 $P_{\text{embed}}^{(l)} \in \mathbb{R}^{k \times d'}$（$d' < d$），MLP 将其映射到 $\mathbb{R}^{k \times d}$。

```
训练时:
  P_embed^(l) → MLP → P_K^(l), P_V^(l)
  优化 P_embed 和 MLP 的参数

推理时（训练完成后）:
  直接用 MLP 输出的 P_K^(l), P_V^(l)
  丢弃 MLP
  → 推理时无额外计算开销
```

### 3.5 参数量分析

$$\text{可训练参数量} = L \times 2 \times k \times d \quad (\text{直接优化})$$

$$\text{可训练参数量} \approx L \times 2 \times k \times d' + \text{MLP 参数} \quad (\text{重参数化})$$

| 模型 | 层数 $L$ | $d$ | Prefix 长度 $k$ | 可训练参数 | 占比 |
|------|---------|------|---------|---------|------|
| GPT-2 Medium (345M) | 24 | 1024 | 10 | 491,520 | 0.14% |
| GPT-2 Large (774M) | 36 | 1280 | 10 | 921,600 | 0.12% |
| LLaMA-7B | 32 | 4096 | 10 | 2,621,440 | 0.04% |
| LLaMA-7B | 32 | 4096 | 30 | 7,864,320 | 0.12% |

### 3.6 代码实现

```python
class PrefixTuning(nn.Module):
    """
    Prefix Tuning: 在每层 Attention 的 KV 前添加可训练前缀。
    
    使用 MLP 重参数化以稳定训练。
    """

    def __init__(
        self,
        pretrained_model,
        num_layers,
        num_heads,
        d_model,
        prefix_length=10,
        prefix_hidden_dim=512,
    ):
        super().__init__()
        self.model = pretrained_model
        self.num_layers = num_layers
        self.prefix_length = prefix_length
        self.d_model = d_model
        self.d_head = d_model // num_heads

        # 冻结预训练模型
        for param in self.model.parameters():
            param.requires_grad = False

        # Reparameterization: embedding → MLP → prefix
        # 每层需要 K 和 V 两组 prefix
        self.prefix_embedding = nn.Embedding(
            prefix_length, prefix_hidden_dim
        )
        self.prefix_mlp = nn.Sequential(
            nn.Linear(prefix_hidden_dim, prefix_hidden_dim),
            nn.Tanh(),
            nn.Linear(prefix_hidden_dim, num_layers * 2 * d_model),
        )

    def get_prefix(self, batch_size):
        """生成所有层的 prefix KV。"""
        prefix_ids = torch.arange(self.prefix_length, device=next(self.parameters()).device)
        prefix_embed = self.prefix_embedding(prefix_ids)
        prefix_kv = self.prefix_mlp(prefix_embed)

        # 重塑为 (num_layers, 2, prefix_length, d_model)
        prefix_kv = prefix_kv.view(
            self.prefix_length,
            self.num_layers,
            2,
            self.d_model,
        ).permute(1, 2, 0, 3)
        # prefix_kv[l, 0] = prefix_K for layer l
        # prefix_kv[l, 1] = prefix_V for layer l

        # 扩展到 batch 维度
        prefix_kv = prefix_kv.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        return prefix_kv  # (B, L, 2, k, d)

    def forward(self, input_ids, labels=None, **kwargs):
        batch_size = input_ids.shape[0]
        prefix_kv = self.get_prefix(batch_size)
        # 具体的 prefix 注入方式取决于模型实现
        # 以 HuggingFace 的 past_key_values 接口为例:
        past_key_values = []
        for l in range(self.num_layers):
            # 每层的 past_key_values: (prefix_K, prefix_V)
            pk = prefix_kv[:, l, 0]  # (B, k, d)
            pv = prefix_kv[:, l, 1]  # (B, k, d)
            past_key_values.append((pk, pv))

        return self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            labels=labels,
            **kwargs,
        )
```

### 3.7 Prompt Tuning vs Prefix Tuning 效果对比

| 维度 | Prompt Tuning | Prefix Tuning |
|------|-------------|---------------|
| 参数注入位置 | 仅输入嵌入层 | 每层 Attention 的 KV |
| 可训练参数量 | $k \times d$ | $L \times 2 \times k \times d$ |
| 信息传播 | 间接（通过冻结层传播）| 直接（每层都有注入）|
| 小模型效果 | 较差 | 较好 |
| 大模型效果 | 接近全参 | 接近全参 |
| 实现复杂度 | 低 | 中（需要修改 Attention）|
| 推理开销 | 增加序列长度 | 增加序列长度 |

---

## 四、Adapter

### 4.1 核心论文

**论文**：*Parameter-Efficient Transfer Learning for NLP* (Houlsby et al., 2019, Google)

**核心思想**：在 Transformer Block 内部**插入小型瓶颈网络**（Adapter），只训练 Adapter 参数，冻结预训练权重。

### 4.2 Adapter 的结构

Adapter 是一个**瓶颈结构**（bottleneck）：

$$\text{Adapter}(h) = h + f(h \cdot W_{\text{down}}) \cdot W_{\text{up}}$$

- $W_{\text{down}} \in \mathbb{R}^{d \times r}$：将 $d$ 维降到 $r$ 维（降维）
- $f$：非线性激活函数（通常是 ReLU 或 GELU）
- $W_{\text{up}} \in \mathbb{R}^{r \times d}$：将 $r$ 维升回 $d$ 维（升维）
- $r \ll d$：瓶颈维度，控制参数量
- $h$：残差连接

```
Adapter 内部结构:

Input h (d 维)
    │
    ├──────────────────┐  残差连接
    │                  │
    ▼                  │
  W_down (d → r)       │
    │                  │
    ▼                  │
  ReLU / GELU          │
    │                  │
    ▼                  │
  W_up (r → d)         │
    │                  │
    ▼                  │
  + ←──────────────────┘
    │
Output (d 维)
```

### 4.3 Adapter 在 Transformer 中的位置

Houlsby et al. 在每个 Transformer Block 中插入**两个 Adapter**：

```
原始 Transformer Block:
  h → LN → MHA → + → LN → FFN → + → output
                  ↑                ↑
                  h                h'

带 Adapter 的 Transformer Block:
  h → LN → MHA → Adapter₁ → + → LN → FFN → Adapter₂ → + → output
                              ↑                          ↑
                              h                          h'

  冻结: LN, MHA, FFN（预训练权重）
  可训练: Adapter₁, Adapter₂
```

### 4.4 数学形式

设 Transformer 第 $l$ 层的原始前向传播为：

$$h' = h + \text{MHA}(\text{LN}(h))$$
$$h'' = h' + \text{FFN}(\text{LN}(h'))$$

加入 Adapter 后：

$$h' = h + \text{Adapter}_1(\text{MHA}(\text{LN}(h)))$$
$$h'' = h' + \text{Adapter}_2(\text{FFN}(\text{LN}(h')))$$

其中每个 Adapter：

$$\text{Adapter}(x) = x + \text{ReLU}(x W_{\text{down}}) W_{\text{up}}$$

注意残差连接的作用：当 Adapter 参数初始化为接近零时，$\text{Adapter}(x) \approx x$，模型在训练初期的行为接近原始预训练模型。

### 4.5 参数量分析

每个 Adapter 的参数量：$2 \times d \times r$（$W_{\text{down}}$ 和 $W_{\text{up}}$）

每层 2 个 Adapter，共 $L$ 层：

$$\text{可训练参数量} = L \times 2 \times (2 \times d \times r) = 4Ldr$$

| 模型 | $L$ | $d$ | 瓶颈维度 $r$ | 可训练参数 | 占比 |
|------|------|------|------|---------|------|
| BERT-Base (110M) | 12 | 768 | 64 | 2,359,296 | 2.14% |
| BERT-Large (340M) | 24 | 1024 | 64 | 6,291,456 | 1.85% |
| LLaMA-7B | 32 | 4096 | 64 | 33,554,432 | 0.50% |
| LLaMA-7B | 32 | 4096 | 256 | 134,217,728 | 2.00% |

### 4.6 初始化策略

**关键设计**：Adapter 初始化为接近恒等映射。

```python
def init_adapter(d_model, r):
    W_down = nn.Linear(d_model, r, bias=True)
    W_up = nn.Linear(r, d_model, bias=True)

    # 近零初始化 → Adapter(x) ≈ x + 0 = x
    nn.init.normal_(W_down.weight, std=0.01)
    nn.init.zeros_(W_up.weight)
    nn.init.zeros_(W_down.bias)
    nn.init.zeros_(W_up.bias)

    return W_down, W_up
```

**为什么这很重要？**

1. 训练初始 Adapter 输出接近零 → 模型行为接近预训练模型
2. 避免随机初始化破坏预训练的有效表示
3. 训练过程平稳，从预训练模型出发逐步学习任务特定的调整

### 4.7 代码实现

```python
class Adapter(nn.Module):
    """
    瓶颈 Adapter 模块。
    
    d_model → r → d_model，带残差连接。
    """

    def __init__(self, d_model, bottleneck_dim, activation="relu"):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU()

        # 近零初始化
        nn.init.normal_(self.down_proj.weight, std=0.01)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return residual + x


class AdapterTransformerBlock(nn.Module):
    """
    带 Adapter 的 Transformer Block。
    
    在 MHA 后和 FFN 后各插入一个 Adapter。
    冻结原始的 MHA 和 FFN 参数。
    """

    def __init__(self, original_block, d_model, bottleneck_dim):
        super().__init__()
        self.original_block = original_block
        self.adapter_attn = Adapter(d_model, bottleneck_dim)
        self.adapter_ffn = Adapter(d_model, bottleneck_dim)

        # 冻结原始 block 参数
        for param in self.original_block.parameters():
            param.requires_grad = False

    def forward(self, x, **kwargs):
        # MHA + Adapter
        attn_out = self.original_block.attention(
            self.original_block.attention_norm(x), **kwargs
        )
        x = x + self.adapter_attn(attn_out)

        # FFN + Adapter
        ffn_out = self.original_block.feed_forward(
            self.original_block.ffn_norm(x)
        )
        x = x + self.adapter_ffn(ffn_out)

        return x
```

### 4.8 AdapterFusion：多任务 Adapter 组合

后续工作 **AdapterFusion**（Pfeiffer et al., 2021）提出了一种组合多个 Adapter 的方法：

```
场景：已经为 10 个任务各训练了一个 Adapter

传统方式：
  任务 A → 加载 Adapter_A
  任务 B → 加载 Adapter_B
  → 每次只能用一个 Adapter

AdapterFusion：
  新任务 → 学习一个"注意力权重"来融合多个已训练的 Adapter
  → 可以组合利用多个任务的知识
  → 只需学习融合权重，Adapter 本身冻结
```

---

## 五、三种方法的系统对比

### 5.1 参数效率对比

| 方法 | 可训练参数位置 | 参数量公式 | LLaMA-7B 参数量 | 占比 |
|------|-------------|---------|------------|------|
| **Full FT** | 全部 | $\|\theta\|$ | 6.7B | 100% |
| **Prompt Tuning** | 输入嵌入 | $k \times d$ | 410K | 0.006% |
| **Prefix Tuning** | 每层 KV | $L \times 2k \times d$ | 2.6M | 0.04% |
| **Adapter** | 每层瓶颈网络 | $4Ldr$ | 33.5M | 0.50% |
| **LoRA (预告)** | 低秩分解 | $2Ldr$ | ~16.8M | 0.25% |

### 5.2 效果对比

| 方法 | GLUE 平均 | SuperGLUE | 生成任务 | 小模型 | 大模型 |
|------|----------|-----------|---------|--------|--------|
| Full FT | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Prompt Tuning | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Prefix Tuning | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Adapter | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 5.3 工程特性对比

| 维度 | Prompt Tuning | Prefix Tuning | Adapter |
|------|-------------|---------------|---------|
| 是否修改模型结构 | 否 | 否（通过 KV 接口）| 是（插入模块）|
| 推理延迟 | 略增（序列变长）| 略增（序列变长）| 略增（额外计算）|
| 多任务部署 | ⭐⭐⭐⭐⭐ 切换向量 | ⭐⭐⭐⭐ 切换 prefix | ⭐⭐⭐ 切换 adapter |
| 训练稳定性 | ⭐⭐⭐ 对 LR 敏感 | ⭐⭐⭐⭐ 重参数化改善 | ⭐⭐⭐⭐⭐ 残差连接稳定 |
| 与现有框架兼容 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ 需改模型 |
| 显存节省 | 最多 | 多 | 中等 |

### 5.4 方法直觉对比

```
三种方法的直觉类比:

Prompt Tuning ≈ 给学生一个"提示小抄"
  → 学生（模型）不变，只是多了一份参考材料
  → 抄上写什么完全可以优化
  → 但学生能力有限时，提示小抄的效果也有限

Prefix Tuning ≈ 在每一科目给学生配一个"辅导老师"
  → 每个科目（每层）都有独立的辅导
  → 辅导老师可以直接影响学生在该科目的思考过程
  → 比仅有一张提示小抄更有效

Adapter ≈ 给学生加一个"专项训练模块"
  → 在学生的学习流程中嵌入额外的训练环节
  → 每个环节都有自己的小网络来学习任务特定的知识
  → 残差连接确保不会忘记原有的能力
```

---

## 六、PEFT 的统一视角

### 6.1 统一数学框架

三种方法可以从同一个框架理解——**在冻结模型的前向传播中注入可训练信号**：

$$y = F(\underbrace{x + \Delta x_{\text{input}}}_{\text{Prompt Tuning}}; \underbrace{\theta + \Delta\theta_{\text{internal}}}_{\text{Adapter / LoRA}}; \underbrace{\text{KV} + \Delta\text{KV}}_{\text{Prefix Tuning}})$$

| 方法 | 注入位置 | 注入方式 |
|------|---------|---------|
| Prompt Tuning | 输入 $x$ | 拼接可训练向量 |
| Prefix Tuning | KV | 在 KV 前拼接可训练前缀 |
| Adapter | 前向传播路径 | 插入可训练瓶颈模块 |
| LoRA (第 6 周) | 权重矩阵 $W$ | 添加低秩增量 $BA$ |

### 6.2 低秩假设：PEFT 方法的共同理论基础

所有 PEFT 方法都隐含或显式地依赖一个假设：

> **任务适配所需的参数调整是低秩的。**

```
预训练模型的参数空间: d × d 维
  → 蕴含了广泛的知识和能力

任务适配所需的调整: 只在某个低维子空间中
  → Adapter 的瓶颈维度 r 远小于 d
  → Prefix 的长度 k 远小于序列长度
  → LoRA 的秩 r 远小于 d
  → 本质上都在利用"低秩"特性
```

这个假设为什么合理？

1. **任务特异性有限**：不同任务之间的差异远小于"通用语言理解"的复杂度
2. **微调是"微调"**：SFT 本质上只是微小的参数调整（Day 1 的表面对齐假说）
3. **实验验证**：研究发现全参数微调的权重变化矩阵 $\Delta W$ 的有效秩确实很低

### 6.3 从 PEFT 到 LoRA

Adapter 有一个局限——**引入了额外的推理延迟**（每层多了两个小网络的前向传播）。

LoRA（Low-Rank Adaptation）巧妙地解决了这个问题：

```
Adapter 的问题:
  推理时：x → 原始层 → Adapter → 输出
  → 增加了推理的计算量和延迟

LoRA 的思路:
  训练时：y = Wx + BAx（额外分支）
  推理时：y = (W + BA)x = W'x（合并到原始权重）
  → 推理时零额外开销！

这就是为什么 LoRA 成为了目前最主流的 PEFT 方法。
第 6 周将完整展开 LoRA / QLoRA 的数学推导和实现。
```

---

## 七、HuggingFace PEFT 库实践

### 7.1 PEFT 库简介

HuggingFace 的 PEFT 库统一封装了多种 PEFT 方法，使用非常简便：

```python
from peft import (
    get_peft_model,
    PromptTuningConfig,
    PrefixTuningConfig,
    LoraConfig,
    TaskType,
)
from transformers import AutoModelForCausalLM

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# ===== Prompt Tuning =====
pt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,  # prompt 长度
    prompt_tuning_init="TEXT",  # 从文本初始化
    prompt_tuning_init_text="Classify if the sentiment is positive or negative:",
    tokenizer_name_or_path="meta-llama/Llama-2-7b-hf",
)
pt_model = get_peft_model(model, pt_config)
pt_model.print_trainable_parameters()
# trainable params: 81,920 || all params: 6,738,546,688 || trainable%: 0.0012

# ===== Prefix Tuning =====
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,  # prefix 长度
    prefix_projection=True,  # 使用 MLP 重参数化
)
prefix_model = get_peft_model(model, prefix_config)
prefix_model.print_trainable_parameters()

# ===== LoRA（预告）=====
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
```

### 7.2 多任务部署示例

```python
# PEFT 的杀手级特性：多任务共享基座模型

# 加载基座模型（一份）
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 为每个任务加载不同的 PEFT 权重（每份几 MB）
from peft import PeftModel

sentiment_model = PeftModel.from_pretrained(base_model, "path/to/sentiment_adapter")
summary_model = PeftModel.from_pretrained(base_model, "path/to/summary_adapter")
code_model = PeftModel.from_pretrained(base_model, "path/to/code_adapter")

# 部署时:
# 显存: 1 份基座模型 (13GB) + 3 份 adapter (各 ~30MB) = ~13.1 GB
# 全参数微调: 3 份完整模型 = ~39 GB
```

---

## 八、PEFT 方法的演进路线

```
2019.06  Adapter (Houlsby et al.)
  │      首次在 Transformer 中插入轻量模块
  ▼
2021.01  Prefix Tuning (Li & Liang)
  │      在每层 KV 添加可训练前缀
  ▼
2021.04  Prompt Tuning (Lester et al.)
  │      只在输入层添加 soft prompt，极致参数效率
  ▼
2021.06  LoRA (Hu et al.)  ← 第 6 周核心
  │      低秩分解权重更新，推理零开销
  ▼
2022.05  AdapterFusion / (IA)³ / UniPELT
  │      多种 PEFT 方法的组合与统一
  ▼
2023.05  QLoRA (Dettmers et al.)  ← 第 6 周核心
  │      4-bit 量化 + LoRA，单卡微调 65B
  ▼
2024+    DoRA / LoRA+ / GaLore / ...
         更高效的参数高效微调持续演进
```

---

## 九、自检题

1. **为什么需要 PEFT？** 列举全参数微调的三个主要问题。
2. **Prompt Tuning 的 soft prompt 和传统的 prompt 模板有什么区别？** 在数学上是如何实现的？
3. **为什么 Prompt Tuning 的效果在大模型上好、小模型上差？** 直觉解释。
4. **Prefix Tuning 比 Prompt Tuning 强在哪里？** 从信息传播的角度解释。
5. **Prefix Tuning 为什么需要 MLP 重参数化？** 训练和推理时分别怎么处理？
6. **Adapter 的瓶颈结构为什么有效？** 残差连接的作用是什么？
7. **Adapter 的近零初始化为什么重要？** 如果用随机初始化会怎样？
8. **对比三种方法的参数量**。给定 LLaMA-7B（$d=4096, L=32$），分别计算 Prompt Tuning（$k=100$）、Prefix Tuning（$k=10$）、Adapter（$r=64$）的可训练参数量。
9. **什么是"低秩假设"？** 它为什么是 PEFT 方法的理论基础？
10. **LoRA 相比 Adapter 的最大优势是什么？** 提前预习第 6 周。

---

## 十、产出要求

- [ ] 推导 Prompt Tuning 的参数量公式，计算 LLaMA-7B 上的可训练参数量
- [ ] 推导 Prefix Tuning 的数学形式，解释为什么 prefix 出现在 K 和 V 而不是 Q 中
- [ ] 推导 Adapter 的参数量公式，解释瓶颈维度 $r$ 的选择策略
- [ ] 撰写三种方法的全面对比表（参数量、计算开销、适用场景、效果）
- [ ] 写出 Adapter 的 forward 函数（含近零初始化和残差连接）
- [ ] 阅读 HuggingFace PEFT 库文档，了解 API 使用方式
- [ ] 画出"PEFT → LoRA"的演进路线图，标注每种方法的核心创新
