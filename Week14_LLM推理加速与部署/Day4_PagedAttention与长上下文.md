# Day 4：PagedAttention 与长上下文 — 系统级推理优化

> **目标**：理解 PagedAttention 的虚拟内存思想——物理块 / 逻辑块 / 块表映射；掌握 vLLM 的调度策略（Continuous Batching、Preemption、Prefix Sharing）；理解 StreamingLLM 中 attention sink 现象的本质；了解长上下文技术综述——RoPE 外推（YaRN / NTK-aware RoPE / Dynamic NTK）、LongLoRA（Shifted Sparse Attention）和 Mistral Sliding Window Attention；对比各方案的适用场景。

---

## 一、从 KV Cache 碎片化到 PagedAttention

### 1.1 Day1 回顾：KV Cache 的显存问题

Day1 中我们分析了 KV Cache 的两大问题：
1. **显存占用大**：KV Cache 随 batch size × 序列长度线性增长
2. **显存碎片化**：预分配方式导致大量浪费

```
传统 KV Cache 管理的问题:

请求到来时，系统不知道最终生成多少 tokens:
  → 只能按 max_seq_len 预分配 KV Cache
  → 实际使用可能只有 30-50%
  → 剩余空间全部浪费

类比操作系统:
  传统方式 ≈ 连续内存分配 (contiguous allocation)
    → 外部碎片 + 内部碎片
    → 内存利用率低

  PagedAttention ≈ 虚拟内存 + 分页 (paging)
    → 按需分配，动态增长
    → 内存利用率高
```

### 1.2 PagedAttention 论文信息

- **标题**：*Efficient Memory Management for Large Language Model Serving with PagedAttention*
- **作者**：Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Chhavi Yadav, Zhuohan Li, Ion Stoica
- **机构**：UC Berkeley
- **时间**：2023 年 (SOSP 2023)
- **开源项目**：vLLM

---

## 二、PagedAttention 核心设计

### 2.1 物理块与逻辑块

PagedAttention 将 KV Cache 分成固定大小的**块（block）**，每个块存储固定数量 token 的 KV 向量。

$$
\text{Block size} = B_{\text{tok}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times 2 \times \text{bytes}
$$

典型设置：$B_{\text{tok}} = 16$，即每个块存 16 个 token 的 KV。

```
物理块 (Physical Block):
  GPU 显存中实际分配的固定大小内存块
  
  ┌──────────────────────────────────────┐
  │ Physical Block Pool (GPU HBM)        │
  │                                      │
  │ PB_0  PB_1  PB_2  PB_3  PB_4  PB_5  │
  │ [16t] [16t] [16t] [16t] [16t] [16t] │
  │                                      │
  │ PB_6  PB_7  PB_8  ...               │
  │ [16t] [16t] [16t]                   │
  └──────────────────────────────────────┘

逻辑块 (Logical Block):
  每个请求看到的是连续的逻辑地址
  通过块表 (Block Table) 映射到物理块

  请求 A 的视角:
    LB_0 → LB_1 → LB_2
    (tokens 0-15) → (tokens 16-31) → (tokens 32-47)
    
  实际物理映射:
    LB_0 → PB_2,  LB_1 → PB_5,  LB_2 → PB_0
    → 物理上不连续! 但逻辑上连续
```

### 2.2 块表 (Block Table)

```
块表: 逻辑块 → 物理块的映射

Request A:  Block Table = [2, 5, 0]
  逻辑块 0 → 物理块 2
  逻辑块 1 → 物理块 5
  逻辑块 2 → 物理块 0

Request B:  Block Table = [1, 3]
  逻辑块 0 → 物理块 1
  逻辑块 1 → 物理块 3

Free Block List: [4, 6, 7, 8, ...]

┌──────────────────────────────────────────────────────────────┐
│ GPU 显存布局                                                 │
│                                                              │
│ PB_0: [A的LB_2] PB_1: [B的LB_0] PB_2: [A的LB_0]           │
│ PB_3: [B的LB_1] PB_4: [FREE]    PB_5: [A的LB_1]           │
│ PB_6: [FREE]    PB_7: [FREE]    PB_8: [FREE]    ...        │
│                                                              │
│ 利用率 = 已用块数 / 总块数 = 5/9 ≈ 55.6%                    │
│ (传统方式可能只有 30% 利用率)                                 │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 按需分配的优势

```
传统方式 (按 max_len 预分配):
  请求 A: 预分配 4096 tokens → 实际用 800 → 浪费 3296
  请求 B: 预分配 4096 tokens → 实际用 200 → 浪费 3896
  利用率: (800 + 200) / (4096 × 2) = 12.2%

PagedAttention (按需分配):
  请求 A: 初始分配 1 块 (16 tokens)
    → 用满后再分配下一块
    → 实际用 800 tokens → 分配 ceil(800/16)=50 块
    → 最后一块内部浪费 < 16 tokens
  
  请求 B: 类似，实际用 200 tokens → 13 块
  利用率: (50 + 13) × 16 / (50 + 13) × 16 ≈ 99%+
  (唯一浪费: 最后一块的内部碎片)
```

### 2.4 PagedAttention 的 Attention 计算

```python
# PagedAttention 的注意力计算 (伪代码)
def paged_attention(query, block_table, kv_block_pool, block_size):
    """
    query: (1, n_heads, 1, d_head)  — 当前 token 的 query
    block_table: list[int]          — 逻辑块到物理块的映射
    kv_block_pool: tensor           — 所有物理块的 KV 存储
    """
    all_k, all_v = [], []
    for logical_idx, physical_idx in enumerate(block_table):
        k_block = kv_block_pool.k[physical_idx]  # (block_size, n_kv_heads, d_head)
        v_block = kv_block_pool.v[physical_idx]
        all_k.append(k_block)
        all_v.append(v_block)

    K = torch.cat(all_k, dim=0)  # (total_tokens, n_kv_heads, d_head)
    V = torch.cat(all_v, dim=0)

    # 标准注意力计算 (或 FlashAttention)
    output = attention(query, K, V)
    return output
```

---

## 三、vLLM 调度策略

### 3.1 Continuous Batching

```
传统 Static Batching:
  所有请求同时开始，同时结束
  短请求要等长请求完成 → 资源浪费

  时间轴 →
  请求 A: [███████████████████████]        (长请求)
  请求 B: [████████]                       (短请求 — 完成后空等)
  请求 C: [██████████████]                 (中请求 — 完成后空等)
  GPU:    [████████████████████████]        (都在等请求 A)

  问题: B 完成后 GPU 仍在等待 A → 利用率低

Continuous Batching (又称 Iteration-level Batching):
  在每个 decode step 级别管理 batch
  请求完成立即释放，新请求立即加入

  时间轴 →
  请求 A: [███████████████████████]
  请求 B: [████████]
  请求 C: [██████████████]
  请求 D:          [████████████]           ← B 完成后立即加入!
  请求 E:                  [█████████]      ← C 完成后立即加入!
  GPU:    [████████████████████████████████] (始终饱和!)
```

### 3.2 调度策略

vLLM 的调度器管理三种状态的请求：

```
请求状态机:

  WAITING ──(有足够显存)──→ RUNNING ──(生成完成)──→ FINISHED
     ↑                        │
     │                        │(显存不足)
     │                        ↓
     └────────────────── SWAPPED
                        (KV Cache 换出到 CPU)

调度优先级:
  1. 先恢复 SWAPPED 请求 (避免饥饿)
  2. 再调度 WAITING 请求
  3. 显存不足时，将低优先级 RUNNING 请求 SWAP OUT
```

### 3.3 Preemption（抢占）

当 GPU 显存不足时，vLLM 有两种抢占策略：

```
Swap (交换):
  将被抢占请求的 KV Cache 从 GPU 复制到 CPU
  → 恢复时再从 CPU 复制回 GPU
  → 适合 KV Cache 较小的情况
  
  优点: 不丢失已计算的 KV Cache
  缺点: CPU-GPU 传输有延迟

Recompute (重计算):
  直接释放被抢占请求的 KV Cache
  → 恢复时重新 prefill
  → 适合 prompt 较短的情况
  
  优点: 无 CPU-GPU 传输开销
  缺点: 重新 prefill 有计算开销
```

### 3.4 Prefix Sharing（前缀共享）

```
场景: 多个请求共享相同的 system prompt

请求 A: [System Prompt | User A 的问题]
请求 B: [System Prompt | User B 的问题]
请求 C: [System Prompt | User C 的问题]

传统方式:
  每个请求独立存储 System Prompt 的 KV Cache
  → 3 份重复的 KV Cache!

Prefix Sharing (vLLM 的 Automatic Prefix Caching):
  System Prompt 的 KV Cache 只存一份
  多个请求通过引用计数共享

  物理块:
    PB_0~PB_5: System Prompt 的 KV (共享, ref_count=3)
    PB_6~PB_7: User A 的 KV
    PB_8~PB_9: User B 的 KV
    PB_10:     User C 的 KV

  显存节省: 
    假设 System Prompt = 500 tokens, 3 个请求
    传统: 3 × 500 = 1500 tokens 的 KV
    共享: 500 + 3 × 各自补充 = 远少于 1500
```

### 3.5 Copy-on-Write

类似操作系统的 fork + CoW 机制：

```
Beam Search 场景:
  beam_1: [A, B, C, D]  →  [A, B, C, D, E]
  beam_2: [A, B, C, D]  →  [A, B, C, D, F]
  
  前 4 个 token 的 KV Cache 完全相同!
  
  Copy-on-Write:
    beam_1 和 beam_2 共享前 4 个 token 的物理块
    当 beam_2 写入新 token F 时:
      只复制最后一个物理块 → 修改 → 两份独立

  优势: Beam Search 的 KV Cache 接近 1 × 而非 beam_width ×
```

---

## 四、StreamingLLM

### 4.1 Attention Sink 现象

StreamingLLM（Xiao et al., 2024）发现了一个关键现象：

```
观察: 在自回归 LLM 中，前几个 token (尤其是第 1 个) 
     获得的注意力权重异常高，即使它们语义上不重要

典型注意力权重分布 (简化):
  Token:    [BOS]  "The"  "cat"  "sat"  "on"  "the"  "mat"
  Weight:   0.35   0.03   0.05   0.15   0.02   0.05   0.35
               ↑                                        ↑
          attention sink!                         当前 token

为什么会这样?
  Softmax 的数学性质: 所有权重之和 = 1
  → 即使模型"不需要"关注某些位置，也必须分配权重
  → 前几个 token 成为"垃圾桶"，吸收多余的注意力
  → 这就是 "attention sink"

验证: 如果去掉前几个 token 的 KV Cache:
  → 注意力分布被打乱 → 模型输出质量严重下降
  → 证明 attention sink 对维持正确的 softmax 分布至关重要
```

### 4.2 StreamingLLM 策略

```
Window Attention 的问题:
  只保留最近 w 个 token → 丢弃了 attention sinks
  → 模型生成质量崩溃 (PPL 爆炸)

StreamingLLM 的解决方案:
  保留 attention sinks (前 k 个 token) + 最近 w 个 token
  
  完整上下文:
  [t1, t2, t3, t4, ..., t500, t501, ..., t1000]
   ↑  attention    ↑ 中间丢弃    ↑  最近窗口
   sinks (k=4)                  (w=1000)
   
  StreamingLLM 视角:
  [t1, t2, t3, t4, t997, t998, t999, t1000]
   ↑ sinks ↑        ↑  最近窗口  ↑
   
  显存: O(k + w) = 常量, 不随序列增长!
  
  效果:
    Window Attention (无 sinks):  PPL 在超过窗口后爆炸
    StreamingLLM (有 sinks):     PPL 保持稳定，可处理 400万+ tokens
```

### 4.3 StreamingLLM 的局限

```
优点:
  ✓ 固定显存，支持无限长度流式推理
  ✓ 实现简单，只需保留前 k 个 + 最近 w 个 token 的 KV
  ✓ 不需要微调模型

局限:
  ✗ 窗口外的信息完全丢失
  ✗ 不是真正的"长上下文理解"
  ✗ 只适合流式场景 (如持续对话)
  ✗ 不适合需要全局信息的任务 (如长文档问答)
  
  StreamingLLM ≠ 长上下文
  StreamingLLM = 无限长 + 有限记忆
```

---

## 五、长上下文技术综述

### 5.1 为什么需要长上下文

```
模型训练时的上下文长度 vs 推理时需要的长度:

LLaMA-2:  训练 4K   → 用户想用 32K
LLaMA-3:  训练 8K   → 用户想用 128K
GPT-4:    训练 8K?  → 支持 128K

挑战:
  1. RoPE 位置编码在超出训练长度时外推失败
  2. 超长序列的注意力计算成本 O(N²)
  3. KV Cache 显存限制
```

### 5.2 RoPE 外推问题

回顾 RoPE（Rotary Position Embedding，W4 Day4）：

$$
f(x_m, m) = x_m e^{im\theta}, \quad \theta_i = 10000^{-2i/d}
$$

其中 $m$ 是位置索引，$\theta_i$ 是频率基。

**外推问题**：当推理位置 $m$ 超出训练时的最大位置时，$e^{im\theta}$ 的值域超出训练分布 → 注意力分数异常 → 质量下降。

### 5.3 Position Interpolation (PI)

```
最简单的方法: 线性插值

训练时: 位置 m ∈ [0, L_train)
推理时: 位置 m ∈ [0, L_target)

Position Interpolation:
  m' = m × (L_train / L_target)
  
  例如: L_train=4096, L_target=16384
  推理位置 8192 → m' = 8192 × 4096/16384 = 2048
  
  → 所有位置都被压缩到 [0, L_train) 范围内
  → RoPE 不会看到训练外的位置

优点: 简单有效
缺点: 压缩了位置分辨率 → 近距离 token 的位置区分度下降
      通常需要少量微调才能恢复质量
```

### 5.4 NTK-aware RoPE

```
NTK-aware (Neural Tangent Kernel aware) 的关键洞察:

PI 的问题: 均匀缩放所有频率
  → 高频分量 (区分近距离) 被过度压缩

NTK-aware 的方法: 只缩放低频分量，保留高频

  原始: θ_i = base^{-2i/d},      base = 10000
  NTK:  θ_i = (base × α)^{-2i/d}, α = (L_target / L_train)^{d/(d-2)}
  
  效果:
    低频分量 (i 大): 被缩放 → 能表示更远的位置
    高频分量 (i 小): 几乎不变 → 近距离分辨率保持

优势:
  ✓ 无需微调即可使用 (zero-shot)
  ✓ 近距离分辨率保持良好
  ✓ 实现简单: 只需修改 base
```

### 5.5 Dynamic NTK

```
Dynamic NTK 的改进:

问题: 固定的 NTK 缩放因子对所有序列长度使用相同的 base
  → 短序列: 不需要外推，但 base 已被修改 → 略有损失
  → 长序列: 可能缩放不够

Dynamic NTK:
  根据当前序列长度动态计算 base:
  
  if seq_len > L_train:
      α = (seq_len / L_train) ^ (d / (d - 2))
      base_new = base × α
  else:
      base_new = base  # 不修改

优势:
  ✓ 短序列完全无损
  ✓ 长序列自动调整
  ✓ 实现简单
```

### 5.6 YaRN

```
YaRN (Yet another RoPE extensioN) 综合了多种技术:

1. NTK-by-parts:
   将 RoPE 维度分为三个区域:
   - 低频区 (Position Interpolation 缩放)
   - 高频区 (不缩放)
   - 中频区 (平滑过渡)
   
   → 比 NTK-aware 更精细的控制

2. 注意力缩放:
   长距离注意力会因为位置编码变化导致 softmax 温度不同
   YaRN 额外添加温度修正因子 t:
   
   Attention(Q, K) = softmax(QK^T / (√d × t))
   
   t 根据扩展比例自动计算

3. 效果:
   4K → 128K 扩展，PPL 仅增加 1-2 点
   优于 PI 和 NTK-aware
   需要少量微调 (~400 steps)
```

### 5.7 LongLoRA

```
LongLoRA (Chen et al., 2024):
  目标: 用 LoRA 高效微调模型支持长上下文

核心创新: Shifted Sparse Attention (S²-Attn)

标准 Full Attention (训练时):
  每个 token 都与所有其他 token 做 attention
  → 计算量 O(N²) → 长上下文训练极昂贵

S²-Attn:
  Step 1: 将序列分成 G 个 group，每个 group 内部做 attention
    Group 1: [t1, t2, ..., tN/G]     → 组内 attention
    Group 2: [tN/G+1, ..., t2N/G]    → 组内 attention
    ...
    
  Step 2: 将序列偏移 N/(2G) 后再分组做 attention
    Shifted Group 1: [tN/2G+1, ..., t3N/2G]  → 组内 attention
    ...
    
  → 两次分组 + 偏移 = 信息在所有 token 间流动
  → 计算量 O(N × (N/G)) = O(N²/G)
  → 显著降低训练长上下文的计算成本

关键: S²-Attn 只在训练时使用!
     推理时仍然用标准 full attention + FlashAttention
     → 推理质量不受影响
```

### 5.8 Mistral Sliding Window Attention

```
Mistral-7B 的 Sliding Window Attention:
  每个 token 只关注最近 W=4096 个 token
  
  Layer 0: token_t 关注 [t-4096, t]
  Layer 1: token_t 的表示已包含 Layer 0 中 [t-4096, t] 的信息
           → 再关注 [t-4096, t]
           → 有效感受野 = 2 × 4096

  32 层后:
    有效感受野 = 32 × 4096 = 131072 ≈ 128K
    但实际 KV Cache 只需存 4096 tokens!

  ┌─────────────────────────────────────────────────┐
  │ Token 位置:  1   2   3   4   5   6   7   8   9 │
  │                                                 │
  │ W=4 示例:                                       │
  │   t=5 关注: [2, 3, 4, 5]     ← 窗口内           │
  │   t=6 关注: [3, 4, 5, 6]     ← 窗口滑动         │
  │   t=7 关注: [4, 5, 6, 7]                        │
  │                                                 │
  │ 每层 KV Cache = W (固定)                         │
  │ 而非 seq_len (增长)                              │
  └─────────────────────────────────────────────────┘

优缺点:
  ✓ KV Cache 大小固定 = O(W)
  ✓ 训练时即使用，模型天然适配
  ✓ 与 Flash Attention 兼容
  ✗ 显式地放弃了窗口外的直接注意力
  ✗ 依赖多层堆叠扩展感受野
```

---

## 六、长上下文方案对比

### 6.1 全面对比表

| 方案 | 核心思想 | 是否需要微调 | 推理开销 | 扩展倍数 | 质量影响 |
|------|---------|------------|---------|---------|---------|
| Position Interpolation | 线性压缩位置 | 需要（1000+ steps） | 无 | 4-8× | 小 |
| NTK-aware RoPE | 修改频率基 | 不需要 | 无 | 4-8× | 小-中 |
| Dynamic NTK | 动态修改基 | 不需要 | 极低 | 4-8× | 小 |
| YaRN | 分频缩放+温度 | 少量（~400 steps） | 极低 | 4-32× | 最小 |
| LongLoRA | S²-Attn + LoRA | 需要 | 无（推理用 full） | 8-32× | 小 |
| Sliding Window | 固定窗口注意力 | 训练时即用 | 无 | 理论无限 | 窗口外丢失 |
| StreamingLLM | Sink + Window | 不需要 | 无 | 无限（流式） | 窗口外丢失 |

### 6.2 适用场景

```
场景 → 推荐方案:

"已有一个 4K 模型，想零成本扩展到 16K"
  → Dynamic NTK / NTK-aware (无需微调)

"已有一个 4K 模型，愿意少量微调扩展到 64K+"
  → YaRN (最佳质量-成本平衡)

"要训练一个新的长上下文模型"
  → 直接用长数据训练 + Sliding Window (如 Mistral)
  → 或 LongLoRA (训练成本较低)

"要部署一个持续对话的 chatbot (不限长度)"
  → StreamingLLM (固定显存)

"要做长文档 QA"
  → YaRN / LongLoRA (需要全局理解)
  → StreamingLLM 不适合!
```

---

## 七、vLLM 实践入门

### 7.1 vLLM 离线推理

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

prompts = [
    "What is PagedAttention?",
    "Explain KV Cache in LLM inference.",
    "What is FlashAttention?",
]

outputs = llm.generate(prompts, params)
for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Generated: {output.outputs[0].text[:100]}...")
    print()
```

### 7.2 vLLM API 服务

```bash
# 启动 OpenAI 兼容 API 服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000 \
    --gpu-memory-utilization 0.9

# 客户端调用
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": "What is PagedAttention?",
        "max_tokens": 128,
        "temperature": 0.7
    }'
```

### 7.3 vLLM 关键配置

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `gpu_memory_utilization` | GPU 显存使用比例 | 0.85-0.95 |
| `max_model_len` | 最大序列长度 | 根据模型和显存调整 |
| `tensor_parallel_size` | 张量并行数（多 GPU） | GPU 数量 |
| `block_size` | PagedAttention 块大小 | 16（默认） |
| `swap_space` | CPU 交换空间 (GB) | 4-8 |
| `max_num_seqs` | 最大并发序列数 | 根据显存调整 |
| `enable_prefix_caching` | 启用前缀共享 | True（多用户场景） |

---

## 八、自检题

1. **PagedAttention 的核心思想是什么？** 与操作系统虚拟内存有什么类比关系？
2. **解释物理块、逻辑块和块表的关系。** 画一个具体的映射示例。
3. **Continuous Batching 相比 Static Batching 的优势是什么？** 用时间轴图说明。
4. **vLLM 的两种抢占策略 Swap 和 Recompute 分别适用于什么场景？**
5. **Prefix Sharing 在什么场景下效果最好？** 为什么？
6. **StreamingLLM 的 attention sink 现象是什么？** 为什么去掉前几个 token 的 KV 会导致质量下降？
7. **StreamingLLM 和长上下文技术的本质区别是什么？**
8. **Position Interpolation 的缺点是什么？** NTK-aware 如何改进？
9. **YaRN 相比 NTK-aware 有哪些改进？**
10. **LongLoRA 的 S²-Attn 为什么只在训练时使用，推理时用 full attention？**

---

## 九、产出要求

- [ ] 画出 PagedAttention 的物理块 / 逻辑块 / 块表映射关系图
- [ ] 理解 Continuous Batching 的工作原理并能解释其优势
- [ ] 解释 vLLM 的 Preemption 和 Prefix Sharing 策略
- [ ] 说明 StreamingLLM 中 attention sink 现象的本质
- [ ] 对比 YaRN / NTK-aware RoPE / LongLoRA / Sliding Window 四种长上下文方案
- [ ] 知道 vLLM 的基本使用方式（离线推理 + API 服务）
- [ ] 能根据场景选择合适的长上下文/推理优化方案
