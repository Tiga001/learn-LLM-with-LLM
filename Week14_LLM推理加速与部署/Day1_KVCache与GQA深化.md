# Day 1：KV Cache 与 GQA 深化 — 推理显存的精确解剖

> **目标**：从 W4 Day5"知道 KV Cache 是什么"进阶到"能精确计算显存瓶颈并做优化决策"；掌握 KV Cache 的逐层、逐头显存公式；定量分析 MHA / MQA / GQA 的显存-质量-速度 trade-off；理解批量推理下 KV Cache 显存爆炸问题（引出 Day4 PagedAttention）；了解 KV Cache 量化与 Window KV Cache 等优化策略。

---

## 一、KV Cache 回顾与深化

### 1.1 为什么需要 KV Cache

自回归生成的计算模式：

$$
P(x_t \mid x_{<t}) = \text{softmax}\left(\frac{q_t K_{<t}^T}{\sqrt{d_k}}\right) V_{<t}
$$

每生成一个新 token $x_t$，需要用 $q_t$ 与之前所有位置的 $K, V$ 做注意力计算。如果不缓存 $K, V$，每一步都要重新从 $x_1, \ldots, x_{t-1}$ 计算所有层的 $K, V$，复杂度为 $O(t \cdot L \cdot d)$，极其浪费。

**KV Cache 的核心思想**：缓存已计算的 $K, V$ 矩阵，每步只需计算新 token 对应的 $k_t, v_t$ 并追加。

```
不使用 KV Cache（每步重新计算）:
  Step 1: 计算 K=[k1],        V=[v1]        → output_1
  Step 2: 计算 K=[k1,k2],     V=[v1,v2]     → output_2  ← k1,v1 重复计算!
  Step 3: 计算 K=[k1,k2,k3],  V=[v1,v2,v3]  → output_3  ← k1,k2,v1,v2 重复计算!
  ...
  总计算量: O(n²) 次 KV 投影

使用 KV Cache:
  Step 1: 计算 k1,v1 → 存入 cache → output_1
  Step 2: 计算 k2,v2 → 追加 cache → output_2  ← 只计算新的!
  Step 3: 计算 k3,v3 → 追加 cache → output_3  ← 只计算新的!
  ...
  总计算量: O(n) 次 KV 投影
```

### 1.2 KV Cache 的生命周期

```
               Prefill 阶段                    Decode 阶段
          ┌──────────────────────┐    ┌───────────────────────────┐
          │  输入 prompt tokens  │    │  逐 token 自回归生成       │
          │  一次性计算所有 K,V   │    │  每步追加 k_t, v_t        │
          │  存入 KV Cache       │ →  │  从 KV Cache 读取历史 K,V │
          │  (计算密集型)         │    │  (显存/带宽密集型)        │
          └──────────────────────┘    └───────────────────────────┘
                                             ↓
                                      KV Cache 持续增长
                                      直到生成结束或达到窗口上限
```

**关键区别**：
- **Prefill**：计算密集 → 受限于 GPU 算力（compute-bound）
- **Decode**：每步只算 1 个 token 的 QKV → 受限于显存带宽（memory-bound）

Decode 阶段是 KV Cache 的核心作用场景。

---

## 二、KV Cache 显存精确计算

### 2.1 单层单头的 KV Cache 显存

对于一个注意力头，KV Cache 存储的是 $K \in \mathbb{R}^{s \times d_k}$ 和 $V \in \mathbb{R}^{s \times d_v}$（通常 $d_k = d_v = d_{\text{head}}$）。

单层单头的 KV Cache 显存：

$$
\text{Memory}_{\text{1 layer, 1 head}} = 2 \times s \times d_{\text{head}} \times \text{bytes\_per\_param}
$$

- $s$：序列长度（已生成 + prompt）
- $d_{\text{head}}$：注意力头维度
- 2：K 和 V 各一份
- bytes_per_param：FP16 = 2 字节，FP32 = 4 字节

### 2.2 单层所有头的 KV Cache 显存

$$
\text{Memory}_{\text{1 layer}} = 2 \times s \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes}
$$

注意这里用 $n_{\text{kv\_heads}}$（KV 头数）而非 $n_{\text{heads}}$（Query 头数），这在 GQA/MQA 中会不同。

对于标准 MHA：$n_{\text{kv\_heads}} = n_{\text{heads}}$。

### 2.3 整个模型的 KV Cache 显存

$$
\text{Memory}_{\text{total}} = 2 \times L \times s \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes}
$$

其中 $L$ 是 Transformer 层数。

等价形式（利用 $n_{\text{kv\_heads}} \times d_{\text{head}} = d_{\text{kv}}$）：

$$
\boxed{\text{Memory}_{\text{KV Cache}} = 2 \times L \times s \times d_{\text{kv}} \times \text{bytes}}
$$

### 2.4 实际模型的 KV Cache 显存计算

以 LLaMA-2-7B 为例：

| 参数 | 值 |
|------|---|
| $L$（层数） | 32 |
| $n_{\text{heads}}$（Query 头） | 32 |
| $d_{\text{head}}$ | 128 |
| $d_{\text{model}}$ | 4096 |
| $n_{\text{kv\_heads}}$（KV 头） | 32（MHA） |

KV Cache 显存（FP16，序列长度 $s$）：

$$
\text{Memory} = 2 \times 32 \times s \times 32 \times 128 \times 2 = 524288 \cdot s \text{ bytes} = 0.5 \cdot s \text{ MB}
$$

| 序列长度 $s$ | KV Cache 显存 | 备注 |
|-------------|-------------|------|
| 512 | 256 MB | 短对话 |
| 2048 | 1 GB | 中等长度 |
| 4096 | 2 GB | LLaMA-2 上下文上限 |
| 8192 | 4 GB | — |
| 32768 | 16 GB | 长上下文模型 |
| 131072 | 64 GB | 128K 上下文（如 LLaMA-3） |

**关键发现**：模型参数约 14 GB（FP16），但 128K 上下文时 KV Cache 就需要 64 GB！KV Cache 显存可能远超模型参数本身。

### 2.5 KV Cache 显存公式速记

```
KV Cache 显存 = 2 × 层数 × 序列长度 × KV头数 × 头维度 × 字节数

速记:
  "2LSD" — 2 × L(ayer) × S(eq) × D(kv_dim) × bytes

对 MHA:  d_kv = n_heads × d_head = d_model
         → 显存 = 2 × L × S × d_model × bytes

对 GQA:  d_kv = n_kv_heads × d_head < d_model
         → 显存按 n_kv_heads / n_heads 比例缩减
```

---

## 三、MHA / MQA / GQA 深度对比

### 3.1 三种注意力机制回顾

```
Multi-Head Attention (MHA) — 标准多头注意力:
  每个 Query 头有独立的 K 和 V 头
  n_kv_heads = n_heads
  
  Head 1: Q1 → K1, V1
  Head 2: Q2 → K2, V2
  Head 3: Q3 → K3, V3
  Head 4: Q4 → K4, V4
  ...
  
  优点: 最大表达能力
  缺点: KV Cache 最大

Multi-Query Attention (MQA) — 多查询注意力:
  所有 Query 头共享同一组 K 和 V
  n_kv_heads = 1
  
  Head 1: Q1 ──┐
  Head 2: Q2 ──┼──→ K_shared, V_shared
  Head 3: Q3 ──┤
  Head 4: Q4 ──┘
  
  优点: KV Cache 最小 (1/n_heads)
  缺点: 质量下降明显

Grouped-Query Attention (GQA) — 分组查询注意力:
  Query 头分组，每组共享 K 和 V
  1 < n_kv_heads < n_heads
  
  Group 1: Q1, Q2 → K1, V1
  Group 2: Q3, Q4 → K2, V2
  
  优点: 质量接近 MHA，KV Cache 接近 MQA
  缺点: 需要选择合适的分组数
```

### 3.2 GQA 的数学描述

设 Query 头数为 $n_h$，KV 头数为 $n_{kv}$，分组大小 $g = n_h / n_{kv}$。

对于第 $i$ 个 Query 头，它使用的 KV 头索引为 $\lfloor i / g \rfloor$：

$$
\text{Attention}_i = \text{softmax}\left(\frac{Q_i K_{\lfloor i/g \rfloor}^T}{\sqrt{d_k}}\right) V_{\lfloor i/g \rfloor}
$$

```python
def grouped_query_attention(Q, K, V, n_heads, n_kv_heads):
    """
    Q: (batch, n_heads, seq_q, d_head)
    K: (batch, n_kv_heads, seq_kv, d_head)
    V: (batch, n_kv_heads, seq_kv, d_head)
    """
    group_size = n_heads // n_kv_heads

    # 方法 1: 显式 repeat KV heads
    K = K.repeat_interleave(group_size, dim=1)  # → (batch, n_heads, seq_kv, d_head)
    V = V.repeat_interleave(group_size, dim=1)

    # 标准注意力
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output
```

### 3.3 KV Cache 显存量化对比

以 LLaMA-2 系列为例（FP16，序列长度 4096）：

| 模型 | 层数 | Query 头 | KV 头 | 机制 | KV Cache | 相对 MHA |
|------|------|---------|-------|------|---------|---------|
| LLaMA-2-7B | 32 | 32 | 32 | MHA | 2 GB | 100% |
| LLaMA-2-7B (假设 GQA-8) | 32 | 32 | 8 | GQA | 512 MB | 25% |
| LLaMA-2-7B (假设 MQA) | 32 | 32 | 1 | MQA | 64 MB | 3.1% |
| LLaMA-2-13B | 40 | 40 | 40 | MHA | 4 GB | — |
| LLaMA-2-70B | 80 | 64 | 8 | **GQA** | **2.5 GB** | 12.5% |
| LLaMA-3-8B | 32 | 32 | 8 | GQA | 512 MB | 25% |
| LLaMA-3-70B | 80 | 64 | 8 | GQA | 2.5 GB | 12.5% |
| Mistral-7B | 32 | 32 | 8 | GQA | 512 MB | 25% |

**关键发现**：
- LLaMA-2-70B 用 GQA-8 将 KV Cache 降低了 8 倍
- 在 70B 规模，如果使用 MHA，仅 KV Cache 就需要 20 GB（序列长度 4096）
- GQA 是大模型推理的必备优化

### 3.4 质量-效率 Trade-off

| 维度 | MHA | GQA | MQA |
|------|-----|-----|-----|
| KV Cache 大小 | $n_h \times d_h$ | $n_{kv} \times d_h$ | $d_h$ |
| KV 投影参数量 | $2 \times d \times d$ | $2 \times d \times n_{kv} \times d_h$ | $2 \times d \times d_h$ |
| 推理吞吐 | 基准 | 高 | 最高 |
| 模型质量 | 最好 | 接近 MHA | 有下降 |
| 首次训练可行？ | 是 | 是 | 是 |
| 从 MHA 转换？ | — | Uptraining | Uptraining |

GQA 论文的核心实验结论：
- GQA-8 的质量与 MHA 几乎无差异（MMLU、HellaSwag 等 benchmark）
- GQA-8 的推理速度接近 MQA
- GQA 是 MHA 和 MQA 的最佳折中

### 3.5 GQA Uptraining

GQA 论文提出了从 MHA 模型转换到 GQA 模型的 uptraining 方法：

```
MHA → GQA 转换 (Uptraining):

步骤 1: 将 MHA 的 KV 头分组
  原始: K1, K2, K3, K4, K5, K6, K7, K8 (8 KV heads)
  分 2 组: [K1, K2, K3, K4] → Group1
           [K5, K6, K7, K8] → Group2

步骤 2: 组内平均初始化
  K_group1 = mean(K1, K2, K3, K4)
  K_group2 = mean(K5, K6, K7, K8)

步骤 3: 少量 uptraining
  用原始训练数据的 ~5% 进行微调
  → 模型学会适应共享的 KV 头
  → 质量恢复到接近原始 MHA
```

---

## 四、批量推理下的 KV Cache 显存爆炸

### 4.1 Batch Size 对 KV Cache 的影响

整个 batch 的 KV Cache 显存：

$$
\text{Memory}_{\text{batch}} = B \times 2 \times L \times s \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes}
$$

以 LLaMA-2-7B（MHA）为例，序列长度 4096，FP16：

| Batch Size | KV Cache | 模型参数 | KV/模型 |
|-----------|---------|---------|---------|
| 1 | 2 GB | 14 GB | 14% |
| 4 | 8 GB | 14 GB | 57% |
| 8 | 16 GB | 14 GB | 114% |
| 16 | 32 GB | 14 GB | 229% |
| 32 | 64 GB | 14 GB | 457% |
| 64 | 128 GB | 14 GB | 914% |

**关键发现**：
- Batch 推理时，KV Cache 很快就超过模型参数本身的显存
- Batch=8 时，KV Cache 就已经比模型参数还大
- 这就是 PagedAttention / vLLM 要解决的核心问题（Day4）

### 4.2 显存碎片化问题

```
传统 KV Cache 分配的问题:

请求 A: 需要最多 2048 tokens
请求 B: 需要最多 4096 tokens
请求 C: 需要最多 1024 tokens

预分配策略（按 max_seq_len 分配）:
  ┌────────────────┬────────────────────────────────────┬────────┐
  │  请求 A (2048)  │       请求 B (4096)                 │  C     │
  └────────────────┴────────────────────────────────────┴────────┘
  
  实际使用:
  ┌──────┬─────────┬───────────┬───────────────────────┬──┬─────┐
  │ A实际 │  A浪费   │  B实际     │      B浪费             │C │C浪费│
  │ 800  │  1248   │  1500     │      2596              │50│ 974│
  └──────┴─────────┴───────────┴───────────────────────┴──┴─────┘
  
  内存利用率 = (800 + 1500 + 50) / (2048 + 4096 + 1024) ≈ 32.8%
  → 近 70% 的显存被浪费!
```

这种显存碎片化和浪费正是 PagedAttention 要解决的问题。

### 4.3 最大批量大小估算

可用于 KV Cache 的显存：

$$
\text{KV Budget} = \text{GPU Memory} - \text{Model Params} - \text{Overhead}
$$

最大 batch size：

$$
B_{\max} = \frac{\text{KV Budget}}{2 \times L \times s \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes}}
$$

```python
def estimate_max_batch_size(
    gpu_memory_gb: float,
    model_params_gb: float,
    n_layers: int,
    n_kv_heads: int,
    d_head: int,
    seq_len: int,
    dtype_bytes: int = 2,  # FP16
    overhead_gb: float = 1.0,
):
    """估算给定 GPU 显存下的最大 batch size"""
    kv_budget_bytes = (gpu_memory_gb - model_params_gb - overhead_gb) * (1024**3)
    per_request_kv = 2 * n_layers * seq_len * n_kv_heads * d_head * dtype_bytes
    return int(kv_budget_bytes / per_request_kv)

# LLaMA-2-7B (MHA) on A100-80GB, seq_len=4096
max_bs = estimate_max_batch_size(
    gpu_memory_gb=80,
    model_params_gb=14,
    n_layers=32,
    n_kv_heads=32,
    d_head=128,
    seq_len=4096,
)
print(f"Max batch size: {max_bs}")  # ≈ 32

# LLaMA-3-8B (GQA-8) on A100-80GB, seq_len=4096
max_bs_gqa = estimate_max_batch_size(
    gpu_memory_gb=80,
    model_params_gb=16,
    n_layers=32,
    n_kv_heads=8,
    d_head=128,
    seq_len=4096,
)
print(f"Max batch size (GQA): {max_bs_gqa}")  # ≈ 128
```

**GQA 的实际意义**：同样的 GPU 显存下，GQA-8 可以服务 4 倍的并发请求。

---

## 五、KV Cache 优化策略

### 5.1 KV Cache 量化

将 KV Cache 从 FP16 降低到 INT8 或 INT4：

$$
\text{Memory Reduction} = \frac{\text{FP16 bytes}}{\text{Quantized bytes}} = \frac{2}{1} = 2\times \text{(INT8)} \quad \text{or} \quad \frac{2}{0.5} = 4\times \text{(INT4)}
$$

```
KV Cache 量化的挑战:

模型权重量化 (GPTQ/AWQ):
  → 权重是静态的，可以离线优化量化参数
  → 校准数据可以充分优化

KV Cache 量化:
  → KV 值是动态生成的，每个请求不同
  → 需要在线量化（推理时实时量化）
  → 量化参数需要低开销
  → 精度更敏感（直接影响注意力分布）
```

主要方法：
- **Per-token 量化**：每个 token 位置独立量化
- **Per-channel 量化**：按 head 维度量化
- **KIVI**（ICML 2024）：Key 按 channel 量化（离群值集中），Value 按 token 量化

### 5.2 Window KV Cache / Sliding Window

只保留最近 $w$ 个位置的 KV Cache：

$$
\text{Attention}(q_t) = \text{softmax}\left(\frac{q_t K_{[t-w:t]}^T}{\sqrt{d_k}}\right) V_{[t-w:t]}
$$

```
完整 KV Cache (seq_len = 8192):
  [k1, k2, k3, ..., k8192]  → 固定显存 = O(8192)

Window KV Cache (window = 2048):
  [k6145, k6146, ..., k8192]  → 固定显存 = O(2048)

优点: 显存固定，不随序列长度增长
缺点: 丢失窗口外的长距离信息
```

代表模型：Mistral-7B 使用 Sliding Window Attention（窗口 4096）。

### 5.3 StreamingLLM（预告 Day4 详解）

StreamingLLM 发现 attention sink 现象：前几个 token 即使语义无关也会获得高注意力权重。

```
StreamingLLM 策略:
  保留前 4 个 token (attention sinks) + 最近 w 个 token

  [k1, k2, k3, k4, ..., k(t-w), ..., k(t)]
   ↑  attention sinks      ↑ 最近窗口
   保留                    保留
                中间丢弃
```

### 5.4 优化策略对比

| 策略 | KV Cache 大小 | 质量影响 | 复杂度 | 适用场景 |
|------|-------------|---------|--------|---------|
| 完整缓存 | $O(s)$ | 无 | 最低 | 短序列 |
| GQA | $O(s \cdot n_{kv}/n_h)$ | 极小 | 低 | 通用（训练时确定） |
| KV 量化 (INT8) | $O(s) / 2$ | 小 | 中 | GPU 推理 |
| KV 量化 (INT4) | $O(s) / 4$ | 中 | 中 | 显存紧张 |
| Sliding Window | $O(w)$ 固定 | 丢失远距 | 低 | 超长序列 |
| StreamingLLM | $O(w + k)$ 固定 | 可控 | 低 | 无限长流 |
| PagedAttention | $O(s)$ 实际使用 | 无 | 高 | 批量服务 |

---

## 六、实战：KV Cache 显存分析表

### 6.1 主流模型 KV Cache 对比

```python
def kv_cache_memory_gb(n_layers, n_kv_heads, d_head, seq_len, batch_size=1, dtype_bytes=2):
    """计算 KV Cache 显存 (GB)"""
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch_size * dtype_bytes / (1024**3)

models = {
    "LLaMA-2-7B":   {"n_layers": 32, "n_kv_heads": 32, "d_head": 128, "attn": "MHA"},
    "LLaMA-2-13B":  {"n_layers": 40, "n_kv_heads": 40, "d_head": 128, "attn": "MHA"},
    "LLaMA-2-70B":  {"n_layers": 80, "n_kv_heads": 8,  "d_head": 128, "attn": "GQA-8"},
    "LLaMA-3-8B":   {"n_layers": 32, "n_kv_heads": 8,  "d_head": 128, "attn": "GQA-8"},
    "LLaMA-3-70B":  {"n_layers": 80, "n_kv_heads": 8,  "d_head": 128, "attn": "GQA-8"},
    "Mistral-7B":   {"n_layers": 32, "n_kv_heads": 8,  "d_head": 128, "attn": "GQA-8"},
    "Qwen2-7B":     {"n_layers": 28, "n_kv_heads": 4,  "d_head": 128, "attn": "GQA-4"},
    "Qwen2-72B":    {"n_layers": 80, "n_kv_heads": 8,  "d_head": 128, "attn": "GQA-8"},
}

for name, cfg in models.items():
    mem_4k = kv_cache_memory_gb(cfg["n_layers"], cfg["n_kv_heads"], cfg["d_head"], 4096)
    mem_32k = kv_cache_memory_gb(cfg["n_layers"], cfg["n_kv_heads"], cfg["d_head"], 32768)
    print(f"{name:20s} ({cfg['attn']:6s}):  4K → {mem_4k:6.2f} GB  |  32K → {mem_32k:6.2f} GB")
```

### 6.2 面试高频：画出 GQA 分组

```
GQA-8 示例 (n_heads=32, n_kv_heads=8, group_size=4):

Query heads:  Q0  Q1  Q2  Q3 | Q4  Q5  Q6  Q7 | Q8  Q9  Q10 Q11 | ... | Q28 Q29 Q30 Q31
               ↓   ↓   ↓   ↓ |  ↓   ↓   ↓   ↓ |  ↓   ↓    ↓   ↓ | ... |  ↓   ↓   ↓   ↓
KV heads:         KV_0       |     KV_1        |      KV_2        | ... |     KV_7
                             |                 |                   |     |

每 4 个 Query head 共享 1 组 KV head
KV Cache 大小 = MHA 的 8/32 = 1/4
```

---

## 七、与后续内容的衔接

```
Day 1:  KV Cache 显存分析 + GQA 深度对比
  → 理解推理显存瓶颈，为什么 KV Cache 是核心问题
        │
        ├─→ Day 2-3: FlashAttention 解决 Attention 计算效率问题
        │             (不是 KV Cache 问题，而是注意力计算的 IO 问题)
        │
        ├─→ Day 4:   PagedAttention 解决 KV Cache 显存管理问题
        │             (碎片化、预分配浪费 → 虚拟内存思想)
        │
        └─→ Day 5-6: 量化同时降低模型参数和 KV Cache 的显存
```

---

## 八、自检题

1. **写出 KV Cache 显存的完整公式**，说明每个变量的含义。
2. **计算 LLaMA-2-7B 在序列长度 4096 下的 KV Cache 显存**。如果 batch=16 呢？
3. **GQA 如何减少 KV Cache？** 画出 GQA-8 的 Query-KV 分组关系。
4. **为什么 LLaMA-2-70B 选择 GQA 而不是 MHA？** 用具体数字说明。
5. **Prefill 阶段和 Decode 阶段的计算瓶颈分别是什么？** 为什么说 Decode 是 memory-bound？
6. **传统 KV Cache 预分配有什么问题？** 为什么内存利用率低？
7. **KV Cache 量化和模型权重量化有什么区别？** 为什么 KV 量化更难？
8. **Sliding Window Attention 的优缺点是什么？** 什么场景适用？
9. **如果你只有一张 A100-80GB，LLaMA-2-7B 最大能支持多大的 batch？** 计算过程。
10. **GQA Uptraining 是什么？** 为什么需要用原始数据的一部分做微调？

---

## 九、产出要求

- [ ] 手写 KV Cache 显存公式并计算 3 个以上模型的 KV Cache 显存
- [ ] 画出 MHA / MQA / GQA 的结构对比图
- [ ] 制作 KV Cache 显存分析表（覆盖 7B / 13B / 70B，不同序列长度和 batch size）
- [ ] 用公式估算给定 GPU 下的最大 batch size
- [ ] 理解 KV Cache 优化策略（量化、Window、StreamingLLM）的 trade-off
- [ ] 能回答面试题：GQA 分组画图 + KV Cache 显存计算
