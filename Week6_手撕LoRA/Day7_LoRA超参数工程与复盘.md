# Day 7：LoRA 超参数工程与复盘 — rank / alpha / target_modules 选择策略

> **目标**：系统掌握 LoRA 超参数的选择策略——rank（秩）、alpha（缩放因子）、target_modules（应用位置）如何影响效果和效率；回顾本周 Day 1 ~ Day 6 的完整知识链路，串联「论文精读 → 数学推导 → 代码实现 → 量化原理 → 实践微调」的全流程认知；为第 7 周 Chinese-LLaMA2 + 数据工程做好铺垫。

---

## Part 1：LoRA 超参数工程

### 1.1 核心超参数一览

```
LoRA 的四个关键超参数:

1. r (rank)           — 低秩分解的秩，决定参数量和表达能力
2. lora_alpha (α)     — 缩放因子，控制 LoRA 分支的贡献强度
3. target_modules     — 在哪些权重矩阵上应用 LoRA
4. lora_dropout       — LoRA 分支的 dropout 率

它们之间的关系:
  实际缩放 = α/r
  可训练参数 = Σ r(d_in + d_out) for each target module
  有效学习率 ∝ lr × (α/r)
```

### 1.2 rank 的选择

#### rank 的影响

| rank | 参数量 | 表达能力 | 过拟合风险 | 适用场景 |
|------|--------|---------|----------|---------|
| 1-4 | 极少 | 低 | 低 | 简单任务、数据极少 |
| 8-16 | 适中 | 中等 | 中等 | **多数场景推荐** |
| 32-64 | 较多 | 较高 | 较高 | 复杂任务、数据充足 |
| 128-256 | 很多 | 高 | 高 | 接近全参微调 |

#### rank 选择的经验法则

```
经验法则 1: 从 r=8 或 r=16 开始
  → 在绝大多数 SFT/指令微调场景中足够
  → LoRA 论文发现 r=4 就能达到 r=64 的 90%+ 效果

经验法则 2: 任务越复杂/数据越多，r 可以越大
  简单任务 (情感分类): r=4~8
  指令微调 (Alpaca 风格): r=8~16
  复杂推理/代码生成: r=16~64
  领域适配 (大量新知识): r=32~128

经验法则 3: 增大 r 的收益递减
  r=4 → r=8:   效果提升明显
  r=8 → r=16:  效果略有提升
  r=16 → r=64: 效果几乎不变
  r=64 → r=256: 可能过拟合

经验法则 4: 如果增大 r 效果不提升，说明瓶颈不在模型容量
  → 可能是数据质量/数量的问题
  → 或者学习率/epoch 需要调整
```

#### rank 与效果的典型实验

| 配置 | LLaMA-7B 参数 | MMLU | MT-Bench |
|------|-------------|------|---------|
| r=4, Wq+Wv | 2.1M (0.03%) | 45.0 | 5.8 |
| r=8, Wq+Wv | 4.2M (0.06%) | 45.2 | 6.0 |
| r=16, Wq+Wv | 8.4M (0.13%) | 45.2 | 6.1 |
| r=16, All Linear | 25.2M (0.38%) | 45.3 | 6.3 |
| r=64, All Linear | 100.7M (1.50%) | 45.3 | 6.3 |
| Full FT | 6.7B (100%) | 45.3 | 6.4 |

### 1.3 alpha 的选择

#### alpha 的数学作用回顾

$$h = W_0 x + \frac{\alpha}{r} BAx$$

$\frac{\alpha}{r}$ 是 LoRA 分支的实际缩放系数。它决定了 LoRA 分支对最终输出的贡献强度。

#### alpha 的选择策略

```
策略 1: α = r (缩放 = 1)
  LoRA 分支的输出不缩放
  需要配合较小的学习率
  论文中 r=4, α=4 时 lr=1e-4

策略 2: α = 2r (缩放 = 2) — 最常见
  LoRA 分支输出放大 2 倍
  HuggingFace PEFT 默认推荐
  r=16, α=32 是最常见的配置

策略 3: 固定 α，只调 r
  当 r 变化时，α/r 自动调整
  好处: 不需要为不同 r 重新搜索学习率
  
  例: 固定 α=16
    r=8:  scaling = 2.0
    r=16: scaling = 1.0
    r=32: scaling = 0.5
    → r 越大，单个 LoRA 参数的影响越小，但总表达能力更强
```

#### 有效学习率

真正影响训练动态的是**有效学习率**：

$$\text{有效学习率} = \text{lr} \times \frac{\alpha}{r}$$

| $r$ | $\alpha$ | $\frac{\alpha}{r}$ | 学习率 | 有效学习率 |
|-----|---------|----------|--------|---------|
| 8 | 16 | 2.0 | 2e-4 | 4e-4 |
| 16 | 32 | 2.0 | 2e-4 | 4e-4 |
| 16 | 16 | 1.0 | 2e-4 | 2e-4 |
| 64 | 16 | 0.25 | 2e-4 | 5e-5 |

**注意**：当 $\frac{\alpha}{r}$ 很大时，有效学习率可能过大导致训练不稳定。

### 1.4 target_modules 的选择

#### 选择维度

```
应用位置选择:

Level 1: Wq + Wv (最小方案)
  论文默认设置
  参数最少，效果在多数基准上已足够
  适合: 资源受限、简单任务

Level 2: Wq + Wk + Wv + Wo (所有 Attention)
  覆盖完整 Attention 层
  比 Level 1 参数多 ~2×
  适合: 中等复杂度任务

Level 3: 所有 Linear (Attention + FFN) — 推荐
  覆盖 Wq, Wk, Wv, Wo, W_gate, W_up, W_down
  参数量最多（~3.5× Level 1），但效果通常最好
  QLoRA 论文使用此配置
  适合: 复杂任务、追求最佳效果

不建议应用 LoRA 的位置:
  × Embedding 层 — 参数量太大，且通常不需要
  × LayerNorm — 参数太少，LoRA 不适用
  × lm_head — 通常与 Embedding 共享权重
```

#### 实验对比

| target_modules | 参数量 (r=16) | 相对 Full FT 效果 |
|---------------|-------------|-----------------|
| Wq only | 4.2M | ~93% |
| Wv only | 4.2M | ~93% |
| Wq + Wv | 8.4M | ~96% |
| Wq + Wk + Wv + Wo | 16.8M | ~97% |
| All Linear | 25.2M | **~99%** |

### 1.5 Dropout 的选择

```
LoRA Dropout 经验:

数据量大 (>10K): dropout = 0 ~ 0.05
  数据充足时不需要太多正则化

数据量中等 (1K~10K): dropout = 0.05 ~ 0.1
  适度正则化防止过拟合

数据量小 (<1K): dropout = 0.1 ~ 0.2
  需要较强正则化
  但通常应该优先增加数据质量和数量

大多数情况: dropout = 0.05 是安全的默认值
```

### 1.6 超参数搜索的推荐配置

| 场景 | rank | alpha | target_modules | dropout | lr |
|------|------|-------|---------------|---------|-----|
| **快速实验** | 8 | 16 | Wq + Wv | 0.05 | 2e-4 |
| **标准 SFT** | 16 | 32 | All Linear | 0.05 | 2e-4 |
| **资源受限** | 8 | 16 | All Linear | 0.05 | 3e-4 |
| **追求极致** | 64 | 128 | All Linear | 0.05 | 1e-4 |
| **QLoRA 标准** | 16 | 32 | All Linear | 0.05 | 2e-4 |

---

## Part 2：第六周知识串联与复盘

### 全周知识链路

```
Day 1: LoRA / QLoRA 论文精读
  LoRA 核心动机: ΔW 是低秩的 → 用 BA 近似
  QLoRA 三大创新: NF4 + 双量化 + 分页优化器
  "为什么低秩就够了？内在维度假说"
       │
       │ "低秩假设的数学基础是什么？"
       ▼
Day 2: LoRA 算法推导 ★★★★ 本周理论核心
  SVD → Eckart-Young → 低秩参数化 → h = W₀x + (α/r)BAx
  初始化: B=0, A=Kaiming → 梯度分析 → 参数量与显存
       │
       │ "推导完了，怎么写代码？"
       ▼
Day 3: 手写 LoRA 实现 ★★★★★ 本周核心实践
  LoRALinear → 注入 LLaMA → merge/unmerge → 权重管理
  Full FT vs LoRA 训练对比 → 参数效率验证
       │
       │ "LoRA 还不够省，QLoRA 的量化怎么做的？"
       ▼
Day 4: QLoRA 量化原理 ★★★★ 本周量化核心
  均匀量化 → NF4 (正态分位数) → 双重量化 → 分页优化器
  精度混合: NF4基座 + BF16 LoRA + FP32 优化器
       │
       │ "推理量化和训练量化有什么区别？"
       ▼
Day 5: 推理量化简介 — 第 14 周的铺垫
  GPTQ (逐列OBQ) → AWQ (激活感知) → GGUF (CPU推理)
  量化方法选择决策树
       │
       │ "在真实模型上跑通完整 QLoRA 实验"
       ▼
Day 6: LLaMA2 + QLoRA 微调实践 ★★★★★ 本周核心实验
  NF4加载 → LoRA注入 → SFT训练 → 文本生成 → 权重管理
       │
       │ "超参数怎么选？全周学了什么？"
       ▼
Day 7: 超参数工程 + 全周复盘 ← 你在这里
  rank/alpha/target_modules 选择策略 → 全周串联 → 第 7 周铺路
```

---

### 核心概念关系图

```
          ┌──────────────────────────────────────────────────────────────┐
          │                LoRA / QLoRA 完整技术栈                        │
          │                                                              │
          │  ┌────────────────────────────────────────────────────┐     │
          │  │              理论层 (Day 1,2)                        │     │
          │  │                                                      │     │
          │  │  内在维度假说: ΔW 是低秩的                            │     │
          │  │      ↓                                               │     │
          │  │  SVD → Eckart-Young → 低秩参数化                     │     │
          │  │      ↓                                               │     │
          │  │  LoRA: h = W₀x + (α/r)BAx                          │     │
          │  │      ↓                                               │     │
          │  │  初始化: B=0, A=Kaiming → 推理零开销: W'=W₀+BA      │     │
          │  └────────────────────────────────────────────────────┘     │
          │                        │                                     │
          │                        ▼                                     │
          │  ┌────────────────────────────────────────────────────┐     │
          │  │              实现层 (Day 3,6)                        │     │
          │  │                                                      │     │
          │  │  LoRALinear 层实现 → inject_lora 自动替换             │     │
          │  │      ↓                                               │     │
          │  │  merge/unmerge → save/load LoRA 权重                 │     │
          │  │      ↓                                               │     │
          │  │  QLoRA 训练流程:                                     │     │
          │  │    NF4 加载 → prepare_kbit → LoRA 注入 → SFT 训练   │     │
          │  └────────────────────────────────────────────────────┘     │
          │                        │                                     │
          │                        ▼                                     │
          │  ┌────────────────────────────────────────────────────┐     │
          │  │              量化层 (Day 4,5)                        │     │
          │  │                                                      │     │
          │  │  训练量化:                                            │     │
          │  │    NF4 (信息论最优) → 双重量化 → 分页优化器           │     │
          │  │                                                      │     │
          │  │  推理量化:                                            │     │
          │  │    GPTQ (逐列OBQ) → AWQ (激活感知) → GGUF (CPU)     │     │
          │  └────────────────────────────────────────────────────┘     │
          │                        │                                     │
          │                        ▼                                     │
          │  ┌────────────────────────────────────────────────────┐     │
          │  │              工程层 (Day 7)                          │     │
          │  │                                                      │     │
          │  │  超参数选择:                                          │     │
          │  │    rank: 8~16 (默认) → 任务复杂度决定                 │     │
          │  │    alpha: 2r (常见) → 影响有效学习率                  │     │
          │  │    target: All Linear (推荐) → 效果最好               │     │
          │  └────────────────────────────────────────────────────┘     │
          └──────────────────────────────────────────────────────────────┘
```

---

### 全周自检清单

#### 论文层 — LoRA / QLoRA 核心理解

- [ ] 用一句话概括 LoRA 的核心贡献
- [ ] 解释"内在维度"假说及其对 LoRA 的启示
- [ ] 说明 LoRA 为什么推理零开销——写出权重合并公式
- [ ] 列举 QLoRA 的三大创新，用一句话分别概括
- [ ] 对比 LoRA 论文和 QLoRA 论文的实验模型和下游任务

#### 数学层 — 推导能力

- [ ] **手写 LoRA 前向传播公式 $h = W_0 x + \frac{\alpha}{r} BAx$（面试高频！）**
- [ ] 推导 LoRA 对 $A$ 和 $B$ 的梯度公式
- [ ] 解释 $B=0$ 初始化的数学原因
- [ ] 推导 LoRA 的参数量公式 $r(d_{\text{in}} + d_{\text{out}})$
- [ ] 计算 LLaMA-7B 各种 LoRA 配置的参数量
- [ ] 解释 $\alpha/r$ 缩放因子的作用——为什么改变 $r$ 不需要重调学习率
- [ ] 用 SVD 的语言解释 LoRA，并说明两者的区别

#### 代码层 — 手写能力

- [ ] **手写 LoRALinear 的 `__init__` 和 `forward`（面试高频！）**
- [ ] **手写 `merge` / `unmerge` 方法**
- [ ] 手写 `inject_lora` 函数（自动替换 Linear 层）
- [ ] 手写 `freeze_non_lora_params` 函数
- [ ] 手写 `save_lora_weights` / `load_lora_weights`
- [ ] 完成 Full FT vs LoRA 的对比训练实验

#### 量化层 — QLoRA 与推理量化

- [ ] 解释 NF4 量化的构造过程（正态分布分位数）
- [ ] 计算双重量化的平均位数（4.127 bit）
- [ ] 解释分页优化器的工作原理
- [ ] 区分训练量化（QLoRA）和推理量化（GPTQ/AWQ/GGUF）
- [ ] 画出 QLoRA 前向传播的精度混合流程图
- [ ] 说明 GPTQ、AWQ、GGUF 各自的核心思想和适用场景

#### 工程层 — 超参数与实践

- [ ] 说明 rank 的选择策略——从 8 开始，根据任务复杂度调整
- [ ] 解释 alpha 和 rank 的关系——有效学习率 = lr × (α/r)
- [ ] 对比 target_modules 不同配置的参数量和效果
- [ ] 在真实模型上完成 QLoRA 微调实验（Day 6）
- [ ] 了解 HuggingFace PEFT 和 bitsandbytes 库的使用

---

### 重要公式速查卡

| 公式 | 来源 |
|------|------|
| $h = W_0 x + \frac{\alpha}{r} BAx$ | LoRA 前向传播 (Day 2, 3) |
| $W' = W_0 + \frac{\alpha}{r} BA$ | 权重合并 (Day 2, 3) |
| $\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal{L}}{\partial h} (Ax)^T$ | LoRA 梯度 (Day 2) |
| $\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^T \frac{\partial \mathcal{L}}{\partial h} x^T$ | LoRA 梯度 (Day 2) |
| $\text{LoRA 参数} = r(d_{\text{in}} + d_{\text{out}})$ | 参数量公式 (Day 2) |
| $\text{总参数} = L \times n_{\text{targets}} \times r(d_{\text{in}} + d_{\text{out}})$ | 全模型 LoRA 参数 (Day 2) |
| $M \approx U_r \Sigma_r V_r^T$ | SVD 低秩近似 (Day 2) |
| $\text{NF4}: q_i = \Phi^{-1}(i / 2^b)$ | NF4 分位数 (Day 4) |
| $\text{位/参数} = 4 + 8/64 + 32/(64 \times 256) \approx 4.127$ | 双重量化 (Day 4) |
| $\text{显存}_{\text{QLoRA}} \approx 0.5|\theta| + 4|\phi| + 2|\phi| + \text{激活值}$ | 显存分析 (Day 4) |
| $\text{有效学习率} = \text{lr} \times \frac{\alpha}{r}$ | 缩放分析 (Day 7) |

---

### 从 Alpaca 到 LoRA：完整技术链路

| 周次 | 阶段 | 核心内容 | 关键产出 |
|------|------|---------|---------|
| W4 | 手撕 LLaMA | 模型架构 / RoPE / RMSNorm / SwiGLU | 手写 LLaMA 模型 |
| W5 | 手撕 Alpaca | SFT / Self-Instruct / CoT / PEFT | 手写 SFT 训练 |
| **W6** | **手撕 LoRA** | **LoRA 推导与实现 / QLoRA / 量化** | **手写 LoRA + QLoRA 实践** |
| W7 | Chinese-LLaMA2 | 中文适配 / 词表扩展 / 数据工程 | 中文 SFT + 数据管线 |
| W8 | Agent + RAG | 多轮对话 / Tool Use / 检索增强 | 部署 RAG Agent |

```
技术能力演进:

W4: 会建好模型 (LLaMA)
  ↓
W5: 会让模型遵循指令 (SFT)
  ↓
W6: 会高效地让模型遵循指令 (LoRA/QLoRA)  ← 本周
  ↓
W7: 会适配特定语言 (Chinese-LLaMA2)
  ↓
W8: 会让模型连接现实世界 (Agent/RAG)
```

---

### 常见疑惑解答

**Q1：rank 越大效果一定越好吗？**

不一定。增大 rank 有收益递减效应，过大还可能导致：
1. 过拟合（尤其数据量小时）
2. 训练不稳定（有效学习率变化）
3. 参数效率下降（参数量增加但效果不提升）

经验：从 $r=16$ 开始，只有效果不足时才考虑增大。

**Q2：alpha 应该随 rank 一起调吗？**

两种常见做法：
- **固定 $\alpha=2r$**：改变 $r$ 时 scaling=2 不变，但有效学习率不变
- **固定 $\alpha$（如 16）**：改变 $r$ 时 scaling 自动调整，不需要改学习率

推荐先用 $\alpha=2r$ 开始，如果训练不稳定再降低 $\alpha$。

**Q3：为什么 QLoRA 论文用所有 Linear 层，但原始 LoRA 论文只用 Wq+Wv？**

1. QLoRA 论文发表晚 2 年，有更多实验经验
2. QLoRA 通过量化节省了大量显存，有"预算"给更多 LoRA 参数
3. 后续实验（包括社区大量复现）表明 All Linear 确实效果更好

**Q4：LoRA 的 Dropout 和模型本身的 Dropout 有什么区别？**

- 模型 Dropout：在预训练权重的输出上
- LoRA Dropout：在 LoRA 分支的 $Ax$ 之后
- LoRA Dropout 只正则化 LoRA 分支，不影响冻结权重的计算

**Q5：训练完 QLoRA 后，部署时应该用什么格式？**

```
方案 A: NF4 + LoRA（不合并）
  显存最小，但推理有额外计算
  适合: 显存受限的场景

方案 B: 合并 → FP16 → GPTQ/AWQ 再量化
  QLoRA 训练 → 反量化为 FP16 → 合并 LoRA → GPTQ/AWQ 4-bit
  推理速度最快（经过推理优化的量化格式）
  适合: 需要高吞吐的生产环境

方案 C: 合并 → GGUF
  QLoRA 训练 → 反量化 → 合并 → GGUF 格式
  适合: CPU 推理场景
```

**Q6：多个 LoRA 可以叠加使用吗？**

可以。HuggingFace PEFT 支持加载多个 LoRA 并切换。也可以手动合并多个 LoRA：

$$W' = W_0 + \frac{\alpha_1}{r_1} B_1 A_1 + \frac{\alpha_2}{r_2} B_2 A_2 + \ldots$$

这在多能力组合（如代码 + 数学 + 对话）中有实际应用。

---

### 本周与课程整体的连接

| 本周学到的 | 后续如何演进 |
|----------|-----------|
| LoRA 数学推导与实现 | → **W7 Chinese-LLaMA2** 使用 LoRA 做中文适配 |
| QLoRA 量化基础 | → **W14 推理量化深化** GPTQ/AWQ/vLLM 系统讲解 |
| SFT + LoRA 实践 | → **W12 垂域 Chatbot** 完整 QLoRA SFT 流程 |
| 超参数选择策略 | → **W10 RLHF** 中的 LoRA 配置选择 |
| 权重合并与管理 | → **W13 多卡训练** DeepSpeed + LoRA |
| Full FT vs LoRA 对比 | → **W11 DPO** LoRA vs Full FT 在对齐中的表现 |

---

### 下周预告：第 7 周 · 手撕 Chinese-LLaMA2 + 数据工程

本周掌握了 LoRA/QLoRA 微调技术，下周将把这些技术应用到**中文语言适配**场景中。

| 主题 | 内容 |
|------|------|
| **Chinese-LLaMA2** | 中文适配策略、词表扩展技术 |
| **中文 Tokenizer 扩展** | 在 LLaMA 词表中扩展中文 token，embedding 初始化 |
| **数据工程** | 大规模文本清洗、去重（MinHash）、质量过滤 |
| **二次预训练** | Continual Pre-training 策略、学习率设置 |
| **领域 SFT** | 在医疗语料上做 SFT（复用本周 LoRA 技术）|

**准备工作**：
1. 回顾 W1 Tokenizer / BPE 基础
2. 理解词表扩展的动机——LLaMA 的中文 token 比例极低
3. 确保能跑通本周的 QLoRA 微调实验——下周将基于此做中文适配

---

### 产出要求

- [ ] 完成 rank/alpha/target_modules 选择的决策框架
- [ ] 计算不同超参数配置下的参数量和有效学习率
- [ ] 完成全周自检清单中的所有项目
- [ ] 确认能闭卷手写本周所有核心代码（LoRALinear / merge / unmerge / inject_lora）
- [ ] **确认能闭卷写出 LoRA 前向传播公式 $h = W_0 x + \frac{\alpha}{r} BAx$（面试必考！）**
- [ ] 画出 LoRA / QLoRA 在 LLaMA Transformer Block 中的完整应用图
- [ ] 预习中文 Tokenizer 扩展的基本概念

---

> **第六周总结**：你现在应该能够完整掌握 LoRA / QLoRA 的理论与实践——从"为什么低秩分解可行"的数学原理，到 LoRALinear 层的代码实现，到 QLoRA 的 NF4 量化创新，到在真实模型上完成 QLoRA 微调。你理解了 LoRA 为什么是当前最主流的 PEFT 方法（推理零开销 + 参数高效 + 效果接近全参），也了解了推理量化的基本版图。LoRA 是你后续几乎所有微调实验的基础工具——从 Chinese-LLaMA2 到 RLHF/DPO，都将依赖本周的技术。手写 LoRA forward 是面试中的高频题目，请务必做到闭卷熟练。
