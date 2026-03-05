# Day 5：KV Cache 推理加速 — 理解大模型推理的核心优化

> **目标**：深入理解自回归生成中的冗余计算问题，掌握 KV Cache 的原理和实现，学会分析 KV Cache 的显存开销，理解 GQA/MQA 如何减少 Cache 大小。本日为"入门"，系统深化（PagedAttention / vLLM）将在第 14 周展开。

---

## 一、自回归生成的计算问题

### 1.1 回顾自回归生成

大语言模型的文本生成是**逐 token 进行**的：

```
Prompt:    "The cat sat on the"
Step 1:    生成 "mat"    → 需要计算所有 token 的 attention
Step 2:    生成 "and"    → 需要重新计算所有 token 的 attention（包括 "mat"）
Step 3:    生成 "purred" → 需要再次重新计算所有 token 的 attention
...
```

每一步生成一个新 token 时，我们只关心**最后一个位置**的输出（用来预测下一个 token），但 Attention 需要和**所有历史 token** 交互。

### 1.2 朴素实现的冗余分析

以生成第 $t$ 个 token 为例，Attention 计算需要：

$$\text{Attention}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}$$

**关键观察**：当我们生成第 $t+1$ 个 token 时：
- $K_{1:t}$ 和 $V_{1:t}$ 已经在第 $t$ 步计算过了
- 只有 $K_{t+1}$ 和 $V_{t+1}$ 是新的
- 但朴素实现会重新计算所有 $K_{1:t+1}$ 和 $V_{1:t+1}$

```
朴素实现的计算量：
  Step 1 (T=6):  计算 K,V 各 6 个 → 计算 attention
  Step 2 (T=7):  重新计算 K,V 各 7 个 → 计算 attention  ← K,V 的前 6 个重复了!
  Step 3 (T=8):  重新计算 K,V 各 8 个 → 计算 attention  ← K,V 的前 7 个重复了!
  ...
  Step N (T=6+N): 重新计算 K,V 各 (6+N) 个 → 极大浪费

生成 N 个 token 的总计算量: O(N × (T+N) × d) ← 平方级!
```

### 1.3 冗余到底有多大？

假设生成 1024 个 token（prompt 长度 T=512）：

| 实现 | K/V 投影计算次数 | 比率 |
|------|-----------------|------|
| 朴素实现 | $\sum_{t=512}^{1536} t = \sim 1.05M$ | 100% |
| **KV Cache** | $1024$（每步只算 1 个新 token） | **0.1%** |

**KV Cache 将 K/V 投影的计算量减少了约 1000 倍！**

---

## 二、KV Cache 原理

### 2.1 核心思想

**缓存已计算的 Key 和 Value，每步只计算新 token 的 K 和 V，然后追加到缓存中。**

```
┌──────────────────────────────────────────────────┐
│                 KV Cache 推理流程                   │
│                                                    │
│  Prefill 阶段（处理 prompt，一次性计算所有位置）:     │
│    输入: [The, cat, sat, on, the]                  │
│    计算: K_cache = [K₁, K₂, K₃, K₄, K₅]          │
│          V_cache = [V₁, V₂, V₃, V₄, V₅]          │
│    输出: logits → 采样 → "mat"                      │
│                                                    │
│  Decode 阶段（逐 token 生成）:                      │
│    Step 1:                                         │
│      输入: ["mat"] (只有 1 个 token!)               │
│      计算: K₆, V₆                                  │
│      更新: K_cache = [K₁,...,K₅, K₆]              │
│             V_cache = [V₁,...,V₅, V₆]              │
│      Attention: Q₆ × K_cache^T → softmax → × V_cache│
│      输出: logits → 采样 → "and"                    │
│                                                    │
│    Step 2:                                         │
│      输入: ["and"] (只有 1 个 token!)               │
│      计算: K₇, V₇                                  │
│      更新: K_cache = [K₁,...,K₆, K₇]              │
│             V_cache = [V₁,...,V₆, V₇]              │
│      Attention: Q₇ × K_cache^T → softmax → × V_cache│
│      ...                                           │
└──────────────────────────────────────────────────┘
```

### 2.2 两个阶段的对比

| 维度 | Prefill（预填充） | Decode（逐步解码） |
|------|------------------|------------------|
| 输入长度 | T（prompt 全长） | 1（单个 token） |
| 计算特点 | 计算密集（compute-bound） | 内存密集（memory-bound） |
| K/V 计算 | 一次性计算所有位置 | 只计算 1 个新位置 |
| Attention | $T \times T$ | $1 \times (T + t)$ |
| 瓶颈 | GPU 计算能力 | 显存带宽（读取大 Cache） |
| 耗时占比 | 通常较短（~10%） | 占总推理时间的大部分 |

### 2.3 数学表达

设当前已生成 $t$ 个 token，KV Cache 中已有 $K_{\text{cache}} \in \mathbb{R}^{t \times d}$ 和 $V_{\text{cache}} \in \mathbb{R}^{t \times d}$。

**Decode 一步**（生成第 $t+1$ 个 token）：

1. 输入新 token $x_{t+1}$，计算 $Q_{t+1}, K_{t+1}, V_{t+1}$：

$$Q_{t+1} = x_{t+1} W_Q, \quad K_{t+1} = x_{t+1} W_K, \quad V_{t+1} = x_{t+1} W_V$$

2. 更新 Cache：

$$K_{\text{cache}} \leftarrow \text{concat}(K_{\text{cache}}, K_{t+1}), \quad V_{\text{cache}} \leftarrow \text{concat}(V_{\text{cache}}, V_{t+1})$$

3. 计算 Attention（只算新 token 对所有历史的注意力）：

$$\text{attn}_{t+1} = \text{softmax}\left(\frac{Q_{t+1} \cdot K_{\text{cache}}^T}{\sqrt{d_k}}\right) \cdot V_{\text{cache}}$$

4. 输出投影：

$$o_{t+1} = \text{attn}_{t+1} \cdot W_O$$

**计算量对比**（Decode 单步）：

| 操作 | 朴素实现 | KV Cache |
|------|---------|----------|
| Q 投影 | $(t+1) \times d^2$ | $1 \times d^2$ |
| K 投影 | $(t+1) \times d^2$ | $1 \times d^2$ |
| V 投影 | $(t+1) \times d^2$ | $1 \times d^2$ |
| $QK^T$ | $(t+1)^2 \times d$ | $1 \times (t+1) \times d$ |
| $PV$ | $(t+1)^2 \times d$ | $1 \times (t+1) \times d$ |

KV Cache 将 K/V 投影从 $O((t+1) \times d^2)$ 降为 $O(d^2)$，是**最显著的加速来源**。

---

## 三、KV Cache 的显存分析

### 3.1 单层单头的 Cache 大小

对于一个注意力头，缓存一个位置的 K 和 V：

$$\text{Cache per token per head} = 2 \times d_{\text{head}} \times \text{bytes}$$

以 FP16 为例，$d_{\text{head}} = 128$：

$$2 \times 128 \times 2\text{B} = 512\text{B per token per head}$$

### 3.2 完整模型的 Cache 大小

$$\text{Total KV Cache} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times T \times \text{bytes}$$

| 模型 | 层数 | KV 头数 | $d_{\text{head}}$ | 注意力 | 每 token Cache | 2048 tokens 总 Cache |
|------|------|---------|-----------|--------|---------------|---------------------|
| LLaMA-7B | 32 | 32 (MHA) | 128 | MHA | 512KB | **1 GB** |
| LLaMA-13B | 40 | 40 (MHA) | 128 | MHA | 800KB | **1.6 GB** |
| LLaMA-2 70B | 80 | 8 (GQA) | 128 | GQA | 320KB | **0.64 GB** |
| LLaMA-2 70B (假设 MHA) | 80 | 64 | 128 | MHA | 2560KB | **5.12 GB** |

计算示例（LLaMA-7B, FP16, 2048 tokens）：

$$2 \times 32 \times 32 \times 128 \times 2048 \times 2\text{B} = 1,073,741,824\text{B} = 1\text{GB}$$

### 3.3 Batch Size 对 Cache 的影响

KV Cache 与 batch size **线性增长**——这是批推理的主要瓶颈：

| Batch Size | LLaMA-7B (2048 ctx) | LLaMA-2 70B (4096 ctx) |
|-----------|---------------------|----------------------|
| 1 | 1 GB | 1.28 GB |
| 8 | 8 GB | 10.24 GB |
| 32 | 32 GB | 40.96 GB |
| 128 | 128 GB | 163.84 GB |

**结论**：在批量推理场景中，KV Cache 往往比模型权重本身占更多显存！

### 3.4 GQA/MQA 如何减少 Cache

这正是 GQA 和 MQA 的核心优势——**通过减少 KV 头数来减少 Cache 大小**：

| 方案 | KV 头数 | Cache 大小（vs MHA） | 质量 |
|------|--------|-------------------:|------|
| MHA (LLaMA-1 70B) | 64 | 100% | 最好 |
| **GQA (LLaMA-2 70B)** | **8** | **12.5%** | 接近 MHA |
| MQA | 1 | 1.5% | 略低 |

$$\text{GQA Cache 减少比} = \frac{n_{\text{kv\_heads}}}{n_{\text{heads}}} = \frac{8}{64} = 12.5\%$$

LLaMA-2 70B 使用 GQA ($n_{\text{kv}} = 8$) 将 KV Cache 减少了 **8 倍**，使得推理更加高效。

---

## 四、KV Cache 的代码实现

### 4.1 带 KV Cache 的 Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class Attention(nn.Module):
    """带 KV Cache 的 Grouped Query Attention。"""
    
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        # KV Cache (初始化为 None，第一次 forward 时创建)
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        
        # 线性投影
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # 应用 RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        if use_cache:
            # ===== KV Cache 逻辑 =====
            if self.cache_k is None:
                # 首次调用：初始化 Cache
                self.cache_k = k
                self.cache_v = v
            else:
                # 后续调用：追加新 K, V 到 Cache
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)
            
            # 使用 Cache 中的完整 K, V
            k = self.cache_k
            v = self.cache_v
        
        # 扩展 KV 头以匹配 Q 头数量 (GQA)
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        
        # 转置为 (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.head_dim)
        return self.wo(output)
    
    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, T, n_kv_heads, head_dim = x.shape
        x = x[:, :, :, None, :].expand(B, T, n_kv_heads, self.n_rep, head_dim)
        return x.reshape(B, T, self.n_heads, head_dim)
    
    def clear_cache(self):
        """清除 KV Cache。"""
        self.cache_k = None
        self.cache_v = None
```

### 4.2 LLaMA 官方风格的 Cache 实现

LLaMA 官方使用预分配的固定大小 Cache（更高效，避免动态 concat）：

```python
class AttentionWithPreallocatedCache(nn.Module):
    """使用预分配 Cache 的 Attention（LLaMA 官方风格）。"""
    
    def __init__(self, dim, n_heads, n_kv_heads, max_batch_size, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        # 预分配 Cache（固定大小，避免动态 concat）
        self.cache_k = torch.zeros(
            max_batch_size, max_seq_len, n_kv_heads, self.head_dim
        )
        self.cache_v = torch.zeros(
            max_batch_size, max_seq_len, n_kv_heads, self.head_dim
        )
    
    def forward(self, x, start_pos, freqs_cis, mask=None):
        """
        Args:
            x: (B, T, dim) — Prefill 时 T=prompt_len, Decode 时 T=1
            start_pos: 当前序列的起始位置
        """
        B, T, _ = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # 将新的 K, V 写入 Cache 的对应位置
        self.cache_k[:B, start_pos:start_pos + T] = k
        self.cache_v[:B, start_pos:start_pos + T] = v
        
        # 从 Cache 读取从 0 到当前位置的所有 K, V
        keys = self.cache_k[:B, :start_pos + T]
        values = self.cache_v[:B, :start_pos + T]
        
        # GQA 扩展 + Attention 计算
        keys = self.repeat_kv(keys)
        values = self.repeat_kv(values)
        
        q = q.transpose(1, 2)        # (B, n_heads, T, head_dim)
        keys = keys.transpose(1, 2)   # (B, n_heads, start_pos+T, head_dim)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(q, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, values)
        
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(output)
```

### 4.3 两种 Cache 实现的对比

| 维度 | 动态 concat | 预分配固定大小 |
|------|-----------|--------------|
| 内存分配 | 每步 `torch.cat` | 一次性分配 |
| 内存碎片 | 有（频繁分配释放） | 无 |
| 内存效率 | 只用当前需要的大小 | 总是分配最大大小 |
| 速度 | 略慢（concat 开销） | 更快（原地写入） |
| 适用场景 | 教学/简单实现 | 生产部署 |

---

## 五、带 KV Cache 的完整推理流程

### 5.1 生成函数

```python
@torch.no_grad()
def generate(
    model,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
):
    """
    带 KV Cache 的自回归文本生成。
    
    Args:
        model: LLaMA 模型
        prompt_tokens: (1, T) prompt token IDs
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        top_k: Top-K 采样的 K 值
    """
    model.eval()
    device = prompt_tokens.device
    T = prompt_tokens.shape[1]
    
    # 清除之前的 Cache
    model.clear_cache()
    
    # ===== Prefill 阶段 =====
    # 一次性处理整个 prompt
    freqs_cis = model.freqs_cis[:T].to(device)
    logits = model(prompt_tokens, freqs_cis=freqs_cis, use_cache=True)
    # logits: (1, T, vocab_size)
    
    generated = []
    
    # 从最后一个位置采样
    next_token = sample_top_k(logits[:, -1, :], temperature, top_k)
    generated.append(next_token.item())
    
    # ===== Decode 阶段 =====
    for step in range(max_new_tokens - 1):
        cur_pos = T + step + 1
        
        # 只输入 1 个 token
        x = next_token.unsqueeze(0)  # (1, 1)
        freqs_cis = model.freqs_cis[cur_pos - 1:cur_pos].to(device)
        
        # 模型内部从 Cache 获取所有历史的 K, V
        logits = model(x, freqs_cis=freqs_cis, use_cache=True)
        # logits: (1, 1, vocab_size)
        
        next_token = sample_top_k(logits[:, -1, :], temperature, top_k)
        generated.append(next_token.item())
        
        if next_token.item() == model.config.eos_token_id:
            break
    
    return generated


def sample_top_k(logits, temperature=1.0, top_k=50):
    """Top-K 采样。"""
    if temperature > 0:
        logits = logits / temperature
    
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k, float('-inf'), logits)
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

### 5.2 Prefill vs Decode 的性能特征

```
Prefill 阶段 (处理 prompt):
  ┌─────────────────────────┐
  │ 输入: T 个 token          │
  │ 计算: T 次 Q,K,V 投影    │
  │ Attention: T × T 矩阵    │
  │ 特点: 计算密集 (FLOPS 瓶颈)│
  │ GPU 利用率: 高            │
  └─────────────────────────┘

Decode 阶段 (逐 token 生成):
  ┌─────────────────────────┐
  │ 输入: 1 个 token          │
  │ 计算: 1 次 Q,K,V 投影    │
  │ Attention: 1 × (T+t) 向量│
  │ 特点: 内存密集 (带宽瓶颈)  │
  │ GPU 利用率: 低            │
  │ 瓶颈: 读取 KV Cache       │
  └─────────────────────────┘
```

**Decode 阶段为什么是 memory-bound？**

生成 1 个 token 时：
- 计算量：$O(d^2)$（QKV 投影）+ $O(t \times d)$（Attention）
- 内存读取：完整的 KV Cache + 模型权重

以 LLaMA-7B 为例（$t = 2048$）：
- 模型权重读取：~13.4 GB（FP16）
- KV Cache 读取：~1 GB
- 实际计算：只有 ~0.01 TFLOPS

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Accessed}} \approx \frac{0.01T}{14.4G} \ll 156 \text{ (A100 的 FLOPS/Byte)}$$

远低于 GPU 的计算/带宽比 → 完全 memory-bound。

---

## 六、KV Cache 的进阶话题（预告）

### 6.1 KV Cache 的问题

| 问题 | 说明 | 解决方案 |
|------|------|---------|
| 显存占用大 | 与 batch_size × seq_len 线性增长 | GQA/MQA、量化 Cache |
| 内存碎片 | 预分配浪费，动态分配碎片化 | **PagedAttention** (W14) |
| 批处理困难 | 不同请求的 Cache 长度不同 | **Continuous Batching** (W14) |
| 长序列瓶颈 | 128K 上下文的 Cache 极大 | **StreamingLLM** (W14) |

### 6.2 PagedAttention 简介（第 14 周详讲）

vLLM 的 PagedAttention 借鉴了操作系统的虚拟内存思想：

```
传统 KV Cache:
  请求 1: [████████████░░░░░░░░]  预分配 max_len，实际用了 60%
  请求 2: [██████░░░░░░░░░░░░░░]  预分配 max_len，实际用了 30%
  → 大量显存浪费

PagedAttention:
  物理块: [█][█][█][█][█][█][█][█]  固定大小的块
  请求 1: 指向块 1,3,5,7      (只分配需要的块)
  请求 2: 指向块 2,4           (只分配需要的块)
  → 几乎零浪费，支持动态增长
```

### 6.3 KV Cache 量化

可以对 Cache 中的 K/V 进行量化以减少显存：

| 精度 | 每 token Cache (LLaMA-7B) | 相对大小 |
|------|---------------------------|---------|
| FP32 | 1 MB | 200% |
| **FP16/BF16** | **512 KB** | **100%** |
| INT8 | 256 KB | 50% |
| INT4 | 128 KB | 25% |

注意：Cache 量化需要谨慎——K/V 的精度直接影响 Attention 的准确性。

---

## 七、KV Cache 的速度基准测试

### 7.1 测试代码

```python
import torch
import time

def benchmark_with_and_without_cache(model, prompt_len=128, gen_len=128, device='cuda'):
    """对比有无 KV Cache 的推理速度。"""
    
    # 准备输入
    prompt = torch.randint(0, model.config.vocab_size, (1, prompt_len)).to(device)
    
    # ===== 无 Cache（朴素实现） =====
    model.clear_cache()
    start = time.time()
    
    input_ids = prompt.clone()
    for _ in range(gen_len):
        with torch.no_grad():
            logits = model(input_ids, use_cache=False)
        next_token = logits[:, -1:, :].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    time_no_cache = time.time() - start
    
    # ===== 有 Cache =====
    model.clear_cache()
    start = time.time()
    
    with torch.no_grad():
        # Prefill
        logits = model(prompt, use_cache=True)
        next_token = logits[:, -1:, :].argmax(dim=-1)
        
        # Decode
        for _ in range(gen_len - 1):
            logits = model(next_token, use_cache=True)
            next_token = logits[:, -1:, :].argmax(dim=-1)
    
    time_with_cache = time.time() - start
    
    print(f"无 Cache: {time_no_cache:.2f}s ({gen_len/time_no_cache:.1f} tokens/s)")
    print(f"有 Cache: {time_with_cache:.2f}s ({gen_len/time_with_cache:.1f} tokens/s)")
    print(f"加速比: {time_no_cache/time_with_cache:.1f}x")
```

### 7.2 预期加速效果

| 生成长度 | 无 Cache | 有 Cache | 加速比 |
|---------|---------|---------|--------|
| 64 tokens | ~2.1s | ~0.3s | ~7x |
| 128 tokens | ~7.8s | ~0.6s | ~13x |
| 256 tokens | ~28s | ~1.1s | ~25x |
| 512 tokens | ~105s | ~2.3s | ~46x |

**加速比随生成长度近似线性增长**：生成 $N$ 个 token，朴素实现是 $O(N^2)$，KV Cache 是 $O(N)$。

---

## 八、RoPE 与 KV Cache 的交互

### 8.1 一个关键细节：RoPE 的位置必须正确

使用 KV Cache 时，RoPE 的位置参数必须对应**绝对位置**，而不是相对于当前输入的位置：

```python
# ❌ 错误：Decode 阶段用位置 0 的频率
freqs_cis = model.freqs_cis[0:1]  # 总是用第 0 个位置的 RoPE
logits = model(next_token, freqs_cis=freqs_cis, use_cache=True)

# ✅ 正确：Decode 阶段用当前绝对位置的频率
freqs_cis = model.freqs_cis[current_pos:current_pos+1]  # 用正确的位置
logits = model(next_token, freqs_cis=freqs_cis, use_cache=True)
```

**原因**：RoPE 的旋转角 $m\theta$ 中，$m$ 必须是 token 在序列中的真实位置。如果第 100 步 decode 用了位置 0 的 RoPE，则 K Cache 中历史 token 的 RoPE 角度与新 token 不匹配，注意力分数会错乱。

### 8.2 Prefill 和 Decode 的 RoPE 频率

```python
# Prefill 阶段: 位置 0, 1, 2, ..., T-1
freqs_cis = model.freqs_cis[0:T]  # (T, dim//2)

# Decode 第 1 步: 位置 T
freqs_cis = model.freqs_cis[T:T+1]  # (1, dim//2)

# Decode 第 2 步: 位置 T+1
freqs_cis = model.freqs_cis[T+1:T+2]  # (1, dim//2)

# ...以此类推
```

---

## 九、自检题

### 基础理解

1. **什么是 KV Cache？** 它缓存了什么，为什么能加速？
2. **Prefill 和 Decode 两个阶段有什么区别？** 各自的计算瓶颈是什么？
3. **KV Cache 的显存开销如何计算？** 计算 LLaMA-7B 在 4096 长度时的 Cache 大小。
4. **为什么只缓存 K 和 V，不缓存 Q？**

### 实现细节

5. **动态 concat 和预分配两种 Cache 实现各有什么优缺点？**
6. **使用 KV Cache 时，RoPE 的位置参数应该怎么传？** 为什么位置错误会导致问题？
7. **带 KV Cache 的 generate 函数的完整流程是什么？**

### 显存分析

8. **LLaMA-2 70B (GQA, 8 KV heads) 的 Cache 比 MHA 版本减少了多少？**
9. **为什么说 batch 推理时 KV Cache 可能比模型权重占更多显存？** 计算 batch=32 时的 Cache 大小。
10. **PagedAttention 解决了什么问题？** 用自己的话简述其原理。

---

## 十、产出要求

- [ ] 用自己的话画出 KV Cache 的 Prefill + Decode 流程图
- [ ] 手写带 KV Cache 的 Attention 实现
- [ ] 计算 LLaMA-7B 和 LLaMA-2 70B 的 KV Cache 显存开销
- [ ] 理解 GQA 如何减少 Cache，计算减少比例
- [ ] 实现带 KV Cache 的生成函数并对比速度
- [ ] 理解 Decode 阶段为什么是 memory-bound

---

## 附录：KV Cache 公式速查

$$\boxed{\text{KV Cache Size} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times T_{\text{max}} \times B \times \text{bytes\_per\_element}}$$

| 符号 | 含义 | LLaMA-7B | LLaMA-2 70B |
|------|------|----------|-------------|
| $n_{\text{layers}}$ | 层数 | 32 | 80 |
| $n_{\text{kv\_heads}}$ | KV 头数 | 32 (MHA) | 8 (GQA) |
| $d_{\text{head}}$ | 每头维度 | 128 | 128 |
| $T_{\text{max}}$ | 最大序列长度 | 2048 | 4096 |
| $B$ | Batch size | 1 | 1 |
| bytes | FP16 = 2B | 2 | 2 |
| **总计** | | **1 GB** | **1.28 GB** |
