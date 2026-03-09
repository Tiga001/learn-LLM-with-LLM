# Day 7：部署服务化与 Speculative Decoding 复盘 — 推理全景图

> **目标**：了解 TensorRT-LLM 的架构与核心优化技术（In-flight Batching、量化、KV Cache 管理）；理解 SGLang 的前端调度与 RadixAttention 创新；深入掌握 Speculative Decoding 的原理与概率保证——为什么小模型 draft + 大模型 verify 不损失输出质量；搭建 LLM 推理加速的全栈优化全景图（算法层 → 系统层 → 硬件层）；完成第 14 周全周复盘与自检；明确与第 15 周分布式训练的衔接。

---

## 一、TensorRT-LLM 架构概述

### 1.1 什么是 TensorRT-LLM

```
TensorRT-LLM (TRT-LLM):
  NVIDIA 官方推出的 LLM 推理优化库
  基于 TensorRT 深度学习推理引擎

定位:
  vLLM:     侧重系统调度 (PagedAttention, Continuous Batching)
  TRT-LLM:  侧重计算优化 (算子融合, 量化核, 硬件适配)
  SGLang:   侧重前端调度 (结构化生成, RadixAttention)
  
  实际部署中三者互有重叠, 但侧重点不同

TRT-LLM 的核心优势:
  1. NVIDIA GPU 深度优化 (CUDA, cuBLAS, cuDNN)
  2. 算子融合 (kernel fusion) — 减少 kernel launch 和内存 IO
  3. 高效量化推理核 (INT4/INT8/FP8 GEMM)
  4. 多 GPU / 多节点张量并行和流水线并行
  5. In-flight Batching (类似 Continuous Batching)
```

### 1.2 TRT-LLM 编译流程

```
TRT-LLM 的使用分为两个阶段:

阶段 1: 离线编译 (Build)
  HuggingFace 模型
       ↓
  TRT-LLM 模型定义 (Python)
       ↓
  TensorRT Engine (.engine 文件)
  - 算子融合
  - 内存规划
  - 量化优化
  - 硬件特化

阶段 2: 在线推理 (Run)
  TensorRT Engine
       ↓
  TRT-LLM Runtime
  - In-flight Batching
  - KV Cache 管理
  - 多 GPU 调度
       ↓
  推理结果
```

### 1.3 核心优化技术

#### 1.3.1 算子融合（Kernel Fusion）

```
标准执行:
  LayerNorm → Q投影 → K投影 → V投影 → Attention → ...
  每个操作 = 1次 kernel launch + 1次 HBM 读写

TRT-LLM 融合:
  [LayerNorm + QKV投影] → [Attention + Residual + LayerNorm] → ...
  多个操作合并 = 1次 kernel launch + 减少 HBM 读写

典型的融合模式:
  1. QKV 投影融合: W_Q, W_K, W_V 合并为一次 GEMM
  2. MLP 融合: gate_proj + up_proj + SiLU 融合
  3. Attention + Residual 融合
  4. LayerNorm + 下一层投影 融合

效果:
  kernel launch 减少 50-70%
  HBM IO 减少 30-50%
  → 整体推理速度提升 1.5-3×
```

#### 1.3.2 量化推理核

```
TRT-LLM 支持的量化推理:

INT8 SmoothQuant:
  对权重和激活同时做 INT8 量化
  → INT8 GEMM 吞吐是 FP16 的 2×

INT4 AWQ / GPTQ:
  权重 INT4 + 激活 FP16
  → Weight-only quantization GEMM
  → 减少显存, Decode 阶段带宽受限时加速明显

FP8 (Hopper GPU):
  H100 原生支持 FP8 Tensor Core
  → FP8 GEMM 吞吐是 FP16 的 2×
  → 精度优于 INT8 (有指数位)
  → TRT-LLM 支持 FP8 自动校准

  精度排序: FP16 > FP8 > INT8 > INT4
  速度排序: INT4 > FP8 ≈ INT8 > FP16
```

#### 1.3.3 In-flight Batching

```
TRT-LLM 的 In-flight Batching ≈ vLLM 的 Continuous Batching:

  在每个 decode step 级别管理 batch
  - 请求完成后立即释放资源
  - 新请求立即加入正在运行的 batch
  - 每步都在 Prefill 和 Decode 请求间调度

  区别于 vLLM:
    TRT-LLM 的 batch 调度与底层 TensorRT 引擎紧密耦合
    → 更少的调度开销
    → 但灵活性稍低
```

### 1.4 TRT-LLM 使用示例

```python
# TRT-LLM 的典型使用流程

# 1. 编译模型
# (命令行)
# python build.py \
#     --model_dir meta-llama/Llama-2-7b-hf \
#     --dtype float16 \
#     --use_gpt_attention_plugin float16 \
#     --use_gemm_plugin float16 \
#     --max_batch_size 64 \
#     --max_input_len 2048 \
#     --max_output_len 512 \
#     --output_dir ./trt_engines/llama-7b-fp16

# 2. 推理
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("./trt_engines/llama-7b-fp16")

outputs = runner.generate(
    batch_input_ids=[tokenizer.encode("What is attention?")],
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
)

# 3. 部署为服务 (Triton Inference Server)
# TRT-LLM 与 Triton 深度集成
# → 支持多模型, 负载均衡, 监控
```

### 1.5 TRT-LLM vs vLLM

| 维度 | TRT-LLM | vLLM |
|------|---------|------|
| 厂商 | NVIDIA 官方 | UC Berkeley 开源 |
| 核心优势 | 算子融合、量化核 | PagedAttention、调度灵活 |
| 硬件限制 | **仅 NVIDIA GPU** | NVIDIA + AMD (ROCm) |
| 量化支持 | INT4/INT8/FP8 原生核 | AWQ/GPTQ 通过库支持 |
| 使用复杂度 | 高（需要编译） | 低（Python API） |
| 灵活性 | 低（编译后固定） | 高（动态配置） |
| 多 GPU | 张量并行 + 流水线并行 | 张量并行 |
| 部署集成 | Triton Server | 自带 OpenAI API |
| 推理速度 | 通常最快 | 接近 TRT-LLM |
| 适用场景 | 追求极致性能的生产环境 | 快速原型 + 中等规模生产 |

```
选择建议:

快速原型 / 研究:
  → vLLM (简单, Python 友好)

生产环境 (NVIDIA GPU):
  → TRT-LLM (极致性能)
  → 或 vLLM (性能接近, 更灵活)

非 NVIDIA GPU:
  → vLLM (支持 ROCm)

需要结构化生成:
  → SGLang (下一节)
```

---

## 二、SGLang 前端调度与 RadixAttention

### 2.1 SGLang 论文信息

- **标题**：*SGLang: Efficient Execution of Structured Language Model Programs*
- **作者**：Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng
- **机构**：UC Berkeley, Stanford
- **时间**：2024

### 2.2 SGLang 的定位

```
SGLang 解决的问题:

现代 LLM 应用不只是"输入 prompt → 输出 text":
  
  例子 1: 多轮结构化生成
    Step 1: 生成 JSON 中的 "name" 字段
    Step 2: 生成 "age" 字段
    Step 3: 生成 "summary" 字段
    → 每步都是一次 LLM 调用, 但共享前缀!
    
  例子 2: Tree-of-Thought
    根节点: 生成 3 个思路
    每个思路: 再生成 3 个展开
    → 9 次 LLM 调用, 大量 KV Cache 可复用!
    
  例子 3: Agent 循环
    Step 1: 思考 → Step 2: 调用工具 → Step 3: 分析结果 → 循环
    → 每步追加内容, 前缀不断增长

传统框架 (vLLM, TRT-LLM):
  每次调用独立处理
  → 前缀的 KV Cache 反复计算
  → 浪费!

SGLang:
  感知调用之间的结构关系
  → 复用 KV Cache
  → 批量优化
```

### 2.3 RadixAttention

RadixAttention 是 SGLang 的核心创新，使用 **Radix Tree（基数树）** 管理 KV Cache 的前缀复用。

```
Radix Tree (基数树) 结构:

  假设三个请求:
    Request A: "System: You are helpful. User: What is ML?"
    Request B: "System: You are helpful. User: What is DL?"  
    Request C: "System: You are helpful. User: Explain attention."

  传统方式:
    A: [KV_sys + KV_userA]  → 独立计算
    B: [KV_sys + KV_userB]  → 重复计算 KV_sys!
    C: [KV_sys + KV_userC]  → 重复计算 KV_sys!

  RadixAttention (Radix Tree):
                    [System: You are helpful.]  ← 共享前缀, KV Cache 只算一次
                    /           |            \
        [User: What is ML?] [What is DL?] [Explain attention.]
              ↓                  ↓              ↓
         KV_cache_A          KV_cache_B      KV_cache_C

  → 公共前缀 "System: You are helpful." 的 KV Cache 只计算一次!
  → 节省 Prefill 计算 + 显存

Radix Tree vs Hash-based Prefix Caching (vLLM):
  vLLM APC:   按固定块大小 hash → 精确匹配
  RadixTree:  按前缀字符粒度匹配 → 更灵活
              支持增量插入和 LRU 淘汰
              → 适合动态前缀场景
```

### 2.4 SGLang 的前端 DSL

```python
# SGLang 的编程模型 (Domain Specific Language)
import sglang as sgl

@sgl.function
def multi_turn_qa(s, questions):
    s += sgl.system("You are a helpful assistant.")
    
    for q in questions:
        s += sgl.user(q)
        s += sgl.assistant(sgl.gen("answer", max_tokens=256))

# 多轮调用自动复用 KV Cache
state = multi_turn_qa.run(
    questions=["What is ML?", "What is DL?", "Compare them."]
)

# 批量结构化生成
@sgl.function  
def generate_json(s, topic):
    s += f"Generate a JSON about {topic}:\n{{"
    s += '"name": "' + sgl.gen("name", stop='"') + '",\n'
    s += '"description": "' + sgl.gen("desc", stop='"') + '"\n}'

# SGLang 自动:
# 1. 识别 "Generate a JSON about {topic}:\n{" 是公共前缀
# 2. 在 name 生成后, 复用 KV Cache 继续生成 desc
# 3. 批量请求时, 共享 topic 相同的前缀
```

### 2.5 SGLang 性能

| 场景 | vLLM | SGLang | 加速比 |
|------|------|--------|--------|
| 单次推理 | 基准 | ~1× | 无差异 |
| 多轮对话（共享前缀） | 基准 | 2-3× | 前缀复用 |
| Tree-of-Thought | 基准 | 3-5× | 大量分支复用 |
| 结构化 JSON 生成 | 基准 | 2-4× | 中间结果复用 |
| Agent 循环 | 基准 | 2× | 递增前缀复用 |

---

## 三、Speculative Decoding

### 3.1 自回归解码的瓶颈

```
标准自回归解码:
  每步生成 1 个 token, 用大模型 (如 70B) 验证

  Step 1: 大模型推理 → token_1     (耗时 T)
  Step 2: 大模型推理 → token_2     (耗时 T)
  Step 3: 大模型推理 → token_3     (耗时 T)
  ...
  总耗时: N × T

  Decode 阶段的特点:
    每步只处理 1 个 token → 计算量很小
    但仍需加载完整模型权重 → 受限于显存带宽
    → GPU 利用率极低 (通常 < 5%)
    
  本质: Decode 阶段每步都在"浪费" GPU 算力
        GPU 有能力一步处理更多 token, 但自回归只给它 1 个
```

### 3.2 Speculative Decoding 核心思想

```
核心思想: 用小模型"猜"多个 token, 大模型一次性验证

传统解码 (N steps):
  大模型: t1 → t2 → t3 → ... → tN
  每步 1 个 token

Speculative Decoding:
  Step 1: 小模型 (draft model) 快速生成 γ 个候选 token
    draft: t1', t2', t3', t4'  (γ=4, 耗时 ε << T)
    
  Step 2: 大模型 (target model) 一次前向传播验证这 γ 个 token
    target: 验证 [t1', t2', t3', t4']  (1 次前向, 耗时 ≈ T)
    → 因为大模型可以并行处理多个 token (像 Prefill 一样)
    
  Step 3: 确定哪些 token 被接受
    假设 t1', t2' 被接受, t3' 被拒绝
    → 输出 t1', t2', 大模型修正后的 t3
    
  一步完成 3 个 token, 耗时 ≈ T + ε ≈ T
  → 加速比 ≈ 3/1 = 3×!

关键条件:
  1. 小模型足够快 (ε << T)
  2. 小模型足够准 (大部分 token 被接受)
  3. 大模型验证不损失质量 (概率保证!)
```

### 3.3 概率保证：为什么不损失质量

这是 Speculative Decoding 最精妙的部分：**输出分布与纯大模型完全相同**。

```
验证-接受算法:

设 p(x) = 大模型在当前位置对 token x 的概率
   q(x) = 小模型在当前位置对 token x 的概率
   x' = 小模型的候选 token

接受概率:
  if p(x') ≥ q(x'):
    接受 x', 概率 = 1  (大模型更喜欢这个 token)
    
  if p(x') < q(x'):
    接受 x', 概率 = p(x') / q(x')  (按比例接受)
    拒绝概率 = 1 - p(x') / q(x')

拒绝时的修正:
  从修正分布中采样:
    p'(x) = max(0, p(x) - q(x)) / Σ_x max(0, p(x) - q(x))
  
  → 这个修正分布恰好补偿了接受-拒绝过程的偏差
  → 最终的边际分布 = p(x) (大模型分布)
```

#### 数学证明（简化版）

对于任意 token $x$，最终的采样概率：

$$
P(\text{output} = x) = \underbrace{q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right)}_{\text{被接受的概率}} + \underbrace{(1 - \alpha) \cdot \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}}_{\text{拒绝后从修正分布采样}}
$$

其中 $\alpha = \sum_x \min(p(x), q(x))$ 是总接受率。

**化简**：

当 $p(x) \geq q(x)$ 时：
$$
P = q(x) \cdot 1 + (1-\alpha) \cdot \frac{p(x) - q(x)}{1 - \alpha} = q(x) + p(x) - q(x) = p(x) \quad \checkmark
$$

当 $p(x) < q(x)$ 时：
$$
P = q(x) \cdot \frac{p(x)}{q(x)} + (1-\alpha) \cdot 0 = p(x) \quad \checkmark
$$

**结论**：无论 $p(x)$ 和 $q(x)$ 的关系如何，最终采样概率都等于 $p(x)$。Speculative Decoding **在数学上保证不损失任何质量**。

### 3.4 Speculative Decoding 完整算法

```
算法: Speculative Decoding

输入: 
  目标模型 M_target (大模型, 概率 p)
  草稿模型 M_draft  (小模型, 概率 q)
  猜测长度 γ (通常 4-8)
  已有前缀 prefix

输出: 生成的 tokens

Repeat until 生成结束:
  
  1. Draft 阶段:
     用 M_draft 自回归生成 γ 个候选 token:
     x'_1, x'_2, ..., x'_γ
     记录每个位置的 draft 概率: q(x'_1), q(x'_2), ..., q(x'_γ)
  
  2. Verify 阶段:
     将 [prefix, x'_1, ..., x'_γ] 送入 M_target
     一次前向传播, 得到 γ+1 个位置的 target 概率:
     p(·|prefix), p(·|prefix,x'_1), ..., p(·|prefix,x'_1,...,x'_γ)
  
  3. Accept/Reject 阶段:
     for i = 1 to γ:
       r = uniform(0, 1)
       if r < min(1, p(x'_i) / q(x'_i)):
         接受 x'_i → 加入输出
       else:
         拒绝 x'_i
         从修正分布采样 x_corrected:
           p'(x) = max(0, p(x) - q(x)) / Z
         输出 x_corrected
         break  ← 后续候选全部丢弃!
     
     if 所有 γ 个都被接受:
       额外从 p(·|prefix, x'_1, ..., x'_γ) 采样一个 bonus token
       → 最多输出 γ+1 个 token!
  
  4. 更新 prefix, 回到 Step 1
```

### 3.5 加速比分析

```
加速比取决于接受率 α:

设 α = Σ_x min(p(x), q(x)) = 小模型与大模型的重叠程度

理论分析:
  每轮 Speculative Decoding:
    Draft: 耗时 γ × t_draft
    Verify: 耗时 t_target (一次前向, ≈ 处理 γ+1 个 token)
    
  期望每轮接受的 token 数:
    E[accepted] = (1 - α^{γ+1}) / (1 - α)
    
  当 α → 1 (小模型很准):
    E[accepted] → γ+1 (全部接受)
    加速比 → (γ+1) × t_target / (γ × t_draft + t_target)
    
  当 t_draft << t_target:
    加速比 → γ+1 ≈ 5-9×
    
  当 α → 0 (小模型很差):
    E[accepted] → 1 (几乎不接受)
    加速比 → 1× (无加速, 但也不损失质量!)

实际加速比:
  α ≈ 0.7-0.9 (常见的 draft-target 模型对)
  γ = 4-8
  → 加速比 ≈ 2-3×
```

### 3.6 Draft 模型的选择

```
Draft 模型的要求:
  1. 与 target 模型有相似的分布 (高 α)
  2. 推理速度显著快于 target
  3. 共享词表 (否则需要词表映射)

常见方案:

方案 1: 同系列小模型
  Target: LLaMA-70B
  Draft:  LLaMA-7B
  → α ≈ 0.7-0.8
  → 速度差: 70B 约是 7B 的 10×
  → 实际加速: 2-3×

方案 2: 量化 draft
  Target: LLaMA-7B FP16
  Draft:  LLaMA-7B INT4
  → α 很高 (同模型不同精度)
  → 速度差: 约 2-3×
  → 实际加速: 1.5-2×

方案 3: 自身层跳过 (Self-Speculative)
  Target: 完整 32 层模型
  Draft:  跳过中间层, 只用 8 层
  → 无需额外模型
  → 加速适中

方案 4: Medusa / EAGLE
  在 target 模型头部添加多个预测头
  同时预测后续多个 token
  → 不需要独立的 draft 模型
  → 需要少量微调预测头
```

### 3.7 Speculative Decoding 变体

| 变体 | 核心思想 | Draft 来源 | 需要微调 |
|------|---------|-----------|---------|
| 标准 Speculative | 小模型 draft | 独立小模型 | 否 |
| Medusa | 多头并行预测 | Target 的额外预测头 | 需要（训练头） |
| EAGLE | 特征级 draft | Target 的特征预测 | 需要（训练头） |
| Self-Speculative | 层跳过 | Target 自身子集 | 否 |
| Lookahead | N-gram 查表 | Jacobi 迭代 | 否 |
| SpecInfer | Tree 验证 | 多候选树形展开 | 否 |

---

## 四、LLM 推理加速全景图

### 4.1 从算法到系统的优化栈

```
┌────────────────────────────────────────────────────────────────┐
│                    LLM 推理加速全栈                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  应用层                                                        │
│  ├─ SGLang: 结构化生成优化, 前缀复用                            │
│  ├─ 批处理策略: 合并请求, 共享前缀                               │
│  └─ 提前终止: 合理的 stop criteria                              │
│                                                                │
│  解码策略层                                                     │
│  ├─ Speculative Decoding: 小模型 draft + 大模型 verify          │
│  ├─ Medusa / EAGLE: 多头并行预测                                │
│  └─ Lookahead Decoding: N-gram Jacobi 并行                     │
│                                                                │
│  注意力优化层                                                   │
│  ├─ FlashAttention 1/2/3: 减少 HBM IO                         │
│  ├─ FlashDecoding: Decode 阶段 KV 序列并行                     │
│  ├─ GQA / MQA: 减少 KV Cache 大小                              │
│  └─ Sliding Window / StreamingLLM: 固定 KV Cache               │
│                                                                │
│  内存管理层                                                     │
│  ├─ PagedAttention: 虚拟内存分页, 按需分配                      │
│  ├─ KV Cache 量化: INT8/INT4 KV Cache                          │
│  ├─ Prefix Sharing: 共享前缀的 KV Cache                        │
│  └─ Continuous Batching: 迭代级调度                             │
│                                                                │
│  模型压缩层                                                     │
│  ├─ 权重量化: GPTQ / AWQ / GGUF (INT4/INT8)                    │
│  ├─ KV Cache 量化: KIVI, Per-channel/Per-token                 │
│  ├─ 知识蒸馏: 大模型 → 小模型                                   │
│  └─ 剪枝: 结构化/非结构化剪枝                                   │
│                                                                │
│  系统框架层                                                     │
│  ├─ vLLM: PagedAttention + Continuous Batching                 │
│  ├─ TensorRT-LLM: 算子融合 + 硬件优化                          │
│  ├─ SGLang: RadixAttention + 结构化调度                        │
│  └─ llama.cpp: CPU 推理优化                                    │
│                                                                │
│  编译优化层                                                     │
│  ├─ 算子融合 (Kernel Fusion)                                   │
│  ├─ 图优化 (Graph Optimization)                                │
│  └─ 自动调优 (Auto-Tuning)                                     │
│                                                                │
│  硬件层                                                         │
│  ├─ GPU: Tensor Core (FP16/INT8/FP8), HBM 带宽               │
│  ├─ CPU: AVX-512 / ARM NEON, 内存带宽                         │
│  ├─ 多 GPU: 张量并行 / 流水线并行                               │
│  └─ 专用芯片: TPU, Groq LPU, 华为昇腾                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 不同场景的推荐优化组合

```
场景 1: 单 GPU 部署 7B 模型 (消费级)
  优化栈: AWQ-4bit → vLLM → FlashAttention-2
  预期效果: 显存 ~4GB, 吞吐 50+ tok/s

场景 2: 单 GPU 部署 70B 模型 (A100-80GB)
  优化栈: AWQ-4bit → vLLM → GQA → FlashAttention-2
  预期效果: 显存 ~40GB, 吞吐 30+ tok/s

场景 3: 多 GPU 高吞吐服务 (4×A100)
  优化栈: FP16/FP8 → TRT-LLM → 张量并行 → In-flight Batching
  预期效果: 1000+ tok/s 吞吐

场景 4: CPU 推理 (Mac / 无 GPU)
  优化栈: GGUF Q4_K_M → llama.cpp → SIMD
  预期效果: 10-20 tok/s (M1/M2)

场景 5: 延迟敏感 (实时对话)
  优化栈: Speculative Decoding → vLLM/SGLang → FlashDecoding
  预期效果: TTFT < 200ms, 延迟降低 2-3×

场景 6: 复杂 Agent 应用
  优化栈: SGLang (前缀复用) → AWQ → FlashAttention-2
  预期效果: 多轮调用 2-5× 加速
```

---

## 五、全周复盘

### 5.1 本周知识地图

```
Day 1: KV Cache 与 GQA 深化
  ✅ KV Cache 显存精确公式: 2 × L × S × n_kv × d × bytes
  ✅ MHA/MQA/GQA trade-off 量化分析
  ✅ Batch 推理显存爆炸 → 引出 PagedAttention
  ✅ KV Cache 优化: 量化 / Window / StreamingLLM
      │
      ▼
Day 2: FlashAttention 深化
  ✅ FA1 完整伪代码 + Online Softmax + 反向 recomputation
  ✅ FA2 三大改进: 减少非 GEMM / 调换循环 / 序列并行
  ✅ FlashDecoding: Decode 阶段沿 KV 并行
  ✅ FA3: Hopper 异步流水线 + FP8
      │
      ▼
Day 3: 手写 FlashAttention (实践)
  ✅ 标准 Attention + IO 计数
  ✅ Online Softmax 实现与验证
  ✅ Python 级 FlashAttention 前向传播
  ✅ 正确性验证 + SDPA benchmark
  ✅ KV Cache + GQA 推理实现
      │
      ▼
Day 4: PagedAttention 与长上下文
  ✅ PagedAttention: 物理块/逻辑块/块表
  ✅ vLLM: Continuous Batching / Preemption / Prefix Sharing
  ✅ StreamingLLM: attention sink + 固定窗口
  ✅ 长上下文: PI / NTK-aware / YaRN / LongLoRA / Sliding Window
      │
      ▼
Day 5: 推理量化深度原理
  ✅ PTQ 基础: 校准集 / 分组量化 / 逐层优化
  ✅ GPTQ: OBS → Hessian 逆 + 逐列量化 + Cholesky
  ✅ AWQ: 激活感知缩放 → 保护 1% 显著通道
  ✅ GGUF: K-quant 混合量化 + CPU 推理
  ✅ 选型指南: GPU vs CPU vs 边缘
      │
      ▼
Day 6: 量化与部署实践
  ✅ AutoGPTQ 量化 7B 模型
  ✅ AutoAWQ 量化 7B 模型
  ✅ 推理 benchmark: 显存/速度/质量三维对比
  ✅ vLLM 部署 OpenAI 兼容 API
  ✅ 并发压测与吞吐分析
      │
      ▼
Day 7: 部署服务化与 Speculative Decoding (今天)
  ✅ TensorRT-LLM: 算子融合 / 量化核 / In-flight Batching
  ✅ SGLang: RadixAttention / 结构化生成优化
  ✅ Speculative Decoding: draft-verify + 概率保证
  ✅ 推理加速全景图: 算法→系统→硬件
```

### 5.2 面试高频知识点自检

```
面试题 1: "画出 GQA 分组，计算 KV Cache 显存"
  → Day 1: 公式 2×L×S×n_kv×d_head×bytes
  → 面试核心: 能对具体模型算出数字

面试题 2: "手写 FlashAttention 前向伪代码"
  → Day 2-3: Tiling + Online Softmax
  → 面试核心: 能写完整循环 + 解释 Online Softmax 更新

面试题 3: "FA2 比 FA1 快在哪?"
  → Day 2: 减少非 GEMM / 调换内外循环 / 序列维度并行

面试题 4: "PagedAttention 解决什么问题?"
  → Day 4: KV Cache 碎片化 → 虚拟内存 → 按需分配

面试题 5: "StreamingLLM 的 attention sink 是什么?"
  → Day 4: 前几个 token 获得异常高注意力权重
  → 保留 sink + 近窗口 → 无限长推理

面试题 6: "GPTQ 的逐列量化怎么做?"
  → Day 5: Hessian 逆 + 每列量化后用误差补偿其他列

面试题 7: "AWQ 为什么只保护 1% 的通道就够了?"
  → Day 5: 激活感知 — 激活值大的通道量化误差被放大
  → 缩放保护重要通道

面试题 8: "Speculative Decoding 为什么不损失质量?"
  → Day 7: 接受-拒绝 + 修正分布 → 数学保证输出分布 = 大模型分布

面试题 9: "vLLM vs TRT-LLM 怎么选?"
  → Day 7: vLLM 灵活 + TRT-LLM 极致性能 + SGLang 结构化

面试题 10: "画出 LLM 推理加速全栈图"
  → Day 7: 应用→解码→注意力→内存→压缩→系统→编译→硬件
```

### 5.3 本周产出清单

完成本周后，你应该拥有以下可复用资产：

- [ ] **KV Cache 显存分析表**：7B / 13B / 70B 模型在不同 batch size 和序列长度下的显存需求
- [ ] **手写 FlashAttention 实现**：Python 级 tiling + Online Softmax，通过正确性验证
- [ ] **推理量化选型指南**：GPTQ / AWQ / GGUF 的适用场景、性能对比、工具链
- [ ] **可运行的 vLLM 推理服务**：支持 OpenAI 兼容 API 的量化模型部署
- [ ] **LLM 推理加速全景图**：从算法层到系统层到硬件层的完整优化栈

---

## 六、与后续周次的衔接

```
Week 14: LLM 推理加速（深化）+ 量化 + 部署
  → 单机推理的极致优化
       │
       ▼
Week 15: 分布式训练与推理 (预告)
  ├─ 数据并行 (DDP, FSDP, ZeRO)
  ├─ 张量并行 (Megatron-LM)
  ├─ 流水线并行 (GPipe, PipeDream)
  ├─ 序列并行 (Ring Attention, Context Parallelism)
  ├─ MoE (Mixture of Experts) 架构
  └─ 多机多卡推理服务

W14 → W15 的关系:
  W14 单机优化 → W15 多机扩展
  
  KV Cache 管理 (W14) → 分布式 KV Cache (W15)
  FlashAttention (W14) → Ring Attention / CP (W15)
  vLLM 单机 (W14)     → vLLM 多卡张量并行 (W15)
  TRT-LLM 单机 (W14)  → TRT-LLM 多机部署 (W15)
  量化 (W14)          → 量化 + 并行训练 (W15)
```

---

## 七、自检题

1. **TensorRT-LLM 的核心优势是什么？** 与 vLLM 相比各适合什么场景？
2. **TRT-LLM 的算子融合带来了什么好处？** 举两个典型的融合模式。
3. **SGLang 的 RadixAttention 解决什么问题？** 与 vLLM 的 Prefix Caching 有什么区别？
4. **SGLang 在什么场景下比 vLLM 有显著优势？** 举两个例子。
5. **Speculative Decoding 的核心思想是什么？** 为什么能加速但不损失质量？
6. **写出 Speculative Decoding 的接受概率公式。** 为什么最终分布等于大模型分布？
7. **接受率 α 对加速比有什么影响？** α=0.8, γ=5 时，期望每轮接受多少 token？
8. **Draft 模型有哪些选择方案？** 各有什么优缺点？
9. **Medusa 和标准 Speculative Decoding 的核心区别是什么？**
10. **画出 LLM 推理加速全栈图**，至少包含 5 个层次。

---

## 八、产出要求

- [ ] 对比 TensorRT-LLM / vLLM / SGLang 的定位与优劣
- [ ] 理解 RadixAttention 的工作原理
- [ ] 推导 Speculative Decoding 的概率保证——为什么不损失质量
- [ ] 能解释 Speculative Decoding 的接受-拒绝-修正流程
- [ ] 画出 LLM 推理加速全栈优化图（算法 → 系统 → 硬件）
- [ ] 完成全周 10 个面试高频知识点的自检
- [ ] 明确 W15 分布式训练将在 W14 的基础上扩展什么内容
