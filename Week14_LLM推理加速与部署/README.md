# 第 14 周：LLM 推理加速（深化）+ 量化 + 部署

> **目标**：系统深化 LLM 推理全栈优化技术——从 KV Cache / GQA 的显存精确分析，到 FlashAttention 1/2 / FlashDecoding 的算法实现与反向传播推导；掌握 PagedAttention 虚拟内存思想与 vLLM 调度策略；理解长上下文扩展方案（YaRN / NTK-aware RoPE / LongLoRA / Sliding Window）；深入推理量化三大流派 GPTQ / AWQ / GGUF 的算法原理（与 W6 入门衔接）；完成 7B 模型的量化实验与 vLLM 推理服务部署；理解 Speculative Decoding 的概率保证与 TensorRT-LLM / SGLang 的系统级优化。

---

## 每日安排（建议每日 3~5 小时）

| 天 | 主题 | 核心任务 | 产出 | 难度 |
|----|------|---------|------|:----:|
| Day 1 | KV Cache 与 GQA 深化 | KV Cache 显存精确计算、MHA/MQA/GQA 量化分析、批量推理瓶颈 | 显存分析表 + GQA 分组图 | ⭐⭐⭐⭐ |
| Day 2 | FlashAttention 深化 | FA1 反向传播、FA2 核心改进、FlashDecoding、FA3、benchmark | FlashAttention 伪代码 + 对比笔记 | ⭐⭐⭐⭐⭐ |
| Day 3 | **手写 FlashAttention 与推理优化** | 手写 FlashAttention 前向、正确性验证、KV Cache + GQA 推理 | 手写 FlashAttention + benchmark | ⭐⭐⭐⭐⭐ |
| Day 4 | PagedAttention 与长上下文 | PagedAttention 虚拟内存、vLLM 调度、StreamingLLM、RoPE 外推 | 架构笔记 + 方案对比表 | ⭐⭐⭐⭐ |
| Day 5 | 推理量化 GPTQ / AWQ / GGUF | PTQ 基础、GPTQ 算法推导、AWQ 激活感知、GGUF 生态 | 量化算法推导 + 选型笔记 | ⭐⭐⭐⭐ |
| Day 6 | **模型量化与 vLLM 部署实践** | GPTQ/AWQ 量化 7B 模型、推理 benchmark、vLLM 部署与压测 | 量化实验 + 部署脚本 | ⭐⭐⭐⭐⭐ |
| Day 7 | 部署服务化与 Speculative Decoding 复盘 | TensorRT-LLM / SGLang / Speculative Decoding、全周复盘 | 全景图 + 自检 | ⭐⭐⭐⭐ |

---

## 知识图谱

```
Day 1: KV Cache 与 GQA 深化 — 推理显存的精确解剖
  KV Cache 显存公式 → MHA/MQA/GQA trade-off → 批量推理显存爆炸
  ⚡ 与 W4 Day5 KV Cache 入门衔接
       │
       ▼
Day 2: FlashAttention 深化 — 从理解到能写（面试核心！）
  FA1 反向传播 → FA2 改进 → FlashDecoding → FA3 Hopper
  ⚡ 与 W3 Day5 FlashAttention 入门衔接
       │
       ▼
Day 3: 手写 FlashAttention 与推理优化（本周重要实践！）
  标准 Attention → 手写 FA 前向 → 正确性验证 → KV Cache + GQA 优化推理
       │
       ▼
Day 4: PagedAttention 与长上下文 — 系统级推理优化
  PagedAttention 虚拟内存 → vLLM 调度 → StreamingLLM
  → RoPE 外推（YaRN / NTK-aware）→ LongLoRA / Sliding Window
  ⚡ 与 W4 Day4 RoPE 衔接
       │
       ▼
Day 5: 推理量化 GPTQ / AWQ / GGUF — 从入门到算法推导
  PTQ 基础 → GPTQ (Hessian + 贪心量化) → AWQ (激活感知)
  → GGUF (CPU 推理) → 选型指南
  ⚡ 与 W6 Day5 推理量化简介衔接
       │
       ▼
Day 6: 模型量化与 vLLM 部署实践（本周核心实验！）
  AutoGPTQ → AutoAWQ → benchmark → vLLM 推理服务 → 并发压测
       │
       ▼
Day 7: 部署服务化与 Speculative Decoding + 全周复盘
  TensorRT-LLM → SGLang → Speculative Decoding → 推理全景图
  → 第 15 周分布式训练 + MoE 衔接
```

---

## 与前序周次的关系

| 本周内容 | 前序基础 | 后续衔接 |
|---------|---------|---------|
| KV Cache 显存分析 | W4 Day5 KV Cache 推理加速（入门） | W15 分布式推理 |
| GQA 深化 | W4 Day2 GQA 架构介绍 | — |
| FlashAttention 深化 | W3 Day5 FlashAttention 入门 | W15 Ring Attention / CP |
| PagedAttention / vLLM | W4 KV Cache、W8 多轮对话部署 | W15 分布式推理 |
| RoPE 外推 / 长上下文 | W4 Day4 RoPE 旋转位置编码 | — |
| 推理量化 GPTQ/AWQ/GGUF | W6 Day4 QLoRA 量化、Day5 推理量化简介 | — |
| 部署服务化 | W8 LangChain 服务封装 | W15 多卡推理 |
| Speculative Decoding | W3 Day4 采样策略 | W17 o1 推理 |

---

## 文件结构

```
Week14_LLM推理加速与部署/
├── README.md                                ← 你在这里
├── Day1_KVCache与GQA深化.md                 ← KV Cache 显存精确分析 + GQA 深度对比
├── Day2_FlashAttention深化.md               ← FA1/2 反向传播 + FlashDecoding + FA3
├── Day3_手写FlashAttention与推理优化.ipynb   ← 手写 FA 前向 + KV Cache + GQA 推理 (实践!)
├── Day4_PagedAttention与长上下文.md          ← PagedAttention / vLLM / StreamingLLM / 长上下文
├── Day5_推理量化GPTQ_AWQ_GGUF.md            ← GPTQ / AWQ / GGUF 算法推导 (深化!)
├── Day6_模型量化与vLLM部署实践.ipynb         ← 量化 7B 模型 + vLLM 部署推理服务 (核心!)
└── Day7_部署服务化与Speculative_Decoding复盘.md ← TRT-LLM / SGLang / Speculative Decoding + 复盘
```

---

## 关键检查点 ✅

完成本周学习后，你应该能够：

- [ ] 精确计算任意模型配置下 KV Cache 的显存开销（per-layer / per-head / total）
- [ ] 定量分析 MHA / MQA / GQA 的显存-质量-速度 trade-off
- [ ] 解释 FlashAttention 的反向传播为什么不需要存储 $S, P$ 矩阵（recomputation）
- [ ] 说明 FlashAttention-2 相比 FA1 的三个核心改进
- [ ] **手写 FlashAttention 前向传播伪代码（面试高频！）**
- [ ] 解释 FlashDecoding 如何通过沿 KV 序列并行加速推理
- [ ] 画出 PagedAttention 的物理块 / 逻辑块 / 块表映射关系
- [ ] 解释 vLLM 的 Continuous Batching 和 Preemption 策略
- [ ] 说明 StreamingLLM 中 attention sink 现象的本质
- [ ] 对比 YaRN / NTK-aware RoPE / LongLoRA / Sliding Window 四种长上下文方案
- [ ] 推导 GPTQ 的逐列量化算法（Hessian 逆 + 误差补偿）
- [ ] 解释 AWQ 的激活感知量化策略——为什么保护 1% 的显著权重通道就够了
- [ ] 区分 GPTQ / AWQ / GGUF 的适用场景（GPU / CPU / 边缘）
- [ ] **用 AutoGPTQ / AutoAWQ 量化 7B 模型并部署 vLLM 推理服务**
- [ ] 解释 Speculative Decoding 的概率保证——为什么小模型 draft + 大模型 verify 不损失质量
- [ ] 画出 LLM 推理加速的全栈优化图（算法 → 系统 → 硬件）

---

## 本周必读论文

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao et al., 2022) — **精读**
2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** (Dao, 2023) — **精读**
3. **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., 2023) — **精读**（vLLM）
4. **GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers** (Frantar et al., 2023) — **精读**

## 参考论文

- *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (Shah et al., 2024)
- *FlashDecoding++: Faster Large Language Model Inference on GPUs* (Hong et al., 2024)
- *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (Lin et al., 2024)
- *Efficient Streaming Language Models with Attention Sinks* (Xiao et al., 2024) — StreamingLLM
- *YaRN: Efficient Context Window Extension of Large Language Models* (Peng et al., 2023)
- *LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models* (Chen et al., 2024)
- *Fast Inference from Transformers via Speculative Decoding* (Leviathan et al., 2023)
- *Accelerating Large Language Model Decoding with Speculative Sampling* (Chen et al., 2023)
- *Fast Transformer Decoding: One Write-Head is All You Need* (Shazeer, 2019) — MQA
- *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (Ainslie et al., 2023)
- *SGLang: Efficient Execution of Structured Language Model Programs* (Zheng et al., 2024)

## 推荐资源

- vLLM: [GitHub 仓库](https://github.com/vllm-project/vllm) / [官方文档](https://docs.vllm.ai/)
- TensorRT-LLM: [GitHub 仓库](https://github.com/NVIDIA/TensorRT-LLM) / [文档](https://nvidia.github.io/TensorRT-LLM/)
- SGLang: [GitHub 仓库](https://github.com/sgl-project/sglang)
- AutoGPTQ: [GitHub 仓库](https://github.com/AutoGPTQ/AutoGPTQ)
- AutoAWQ: [GitHub 仓库](https://github.com/casper-hansen/AutoAWQ)
- llama.cpp: [GitHub 仓库](https://github.com/ggerganov/llama.cpp)
- Tri Dao: [FlashAttention 论文 + 代码](https://github.com/Dao-AILab/flash-attention)
- Lilian Weng: [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- HuggingFace Blog: [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

---

## 本周最终产出

完成本周后，建议至少形成以下可复用资产：

- 一份 KV Cache 显存分析表：覆盖 7B / 13B / 70B 模型在不同 batch size 和序列长度下的显存需求
- 一个手写 FlashAttention 实现：Python 级 tiling + Online Softmax，能通过正确性验证
- 一份推理量化选型指南：GPTQ / AWQ / GGUF 的适用场景、性能对比、工具链
- 一个可运行的 vLLM 推理服务：支持 OpenAI 兼容 API 的量化模型部署
- 一份 LLM 推理加速全景图：从算法层到系统层到硬件层的完整优化栈
