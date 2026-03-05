# Day 7：微调方法对比与复盘 — 效率对比 + 全周知识串联

> **目标**：系统对比 Full Fine-tuning / Adapter / LoRA 三种微调方法的参数量、显存、训练速度和效果差异；回顾本周 Day 1 ~ Day 6 的完整知识链路，串联「指令微调思想 → 数据生成 → 推理策略 → PEFT 方法 → SFT 实践」的全流程认知；为第 6 周 LoRA / QLoRA 的深入学习做好铺垫。

---

## Part 1：微调方法效率对比

### 1.1 三种主要微调范式

```
Full Fine-tuning（全参数微调）:
  更新模型所有参数
  → 效果最好，但资源需求最高
  → Day 6 实验的第一部分

Adapter（瓶颈适配器）:
  冻结原始参数，在每层插入可训练瓶颈模块
  → 效果接近全参，参数效率高
  → Day 5 学习 + Day 6 实验的第二部分

LoRA（低秩适配）:
  冻结原始参数，用低秩矩阵近似权重更新
  → 推理零开销，已成为主流方法
  → 第 6 周核心内容（本日预习）
```

### 1.2 参数量对比

以 LLaMA-7B（$d=4096, L=32, n_h=32$）为基准：

| 方法 | 可训练参数公式 | 可训练参数量 | 占比 |
|------|-------------|----------|------|
| **Full FT** | $\|\theta\|$ | 6.7B | 100% |
| **Prompt Tuning** ($k=100$) | $k \times d$ | 409,600 | 0.006% |
| **Prefix Tuning** ($k=10$) | $L \times 2k \times d$ | 2,621,440 | 0.04% |
| **Adapter** ($r=64$) | $L \times 2 \times 2dr$ | 33,554,432 | 0.50% |
| **LoRA** ($r=16$) | $L \times 2 \times 2dr$ | 8,388,608 | 0.13% |
| **QLoRA** ($r=16$) | 同 LoRA | 8,388,608 | 0.13% |

```
可训练参数量排序:

Full FT      ████████████████████████████████████████  6.7B (100%)
Adapter      ██                                        33.5M (0.50%)
LoRA         █                                         8.4M (0.13%)
Prefix       ▎                                         2.6M (0.04%)
Prompt       ▏                                         0.4M (0.006%)
```

### 1.3 显存需求对比

模型训练时的显存组成：

$$\text{显存} = \text{模型参数} + \text{优化器状态} + \text{梯度} + \text{激活值}$$

| 方法 | 模型参数 | 优化器状态 | 梯度 | 激活值 | **总计** |
|------|---------|---------|------|--------|---------|
| **Full FT (FP16)** | 13 GB | 52 GB | 13 GB | ~10 GB | **~88 GB** |
| **Adapter** ($r=64$) | 13 GB | 0.3 GB | 0.06 GB | ~10 GB | **~23 GB** |
| **LoRA** ($r=16$) | 13 GB | 0.07 GB | 0.016 GB | ~8 GB | **~21 GB** |
| **QLoRA** ($r=16$) | **3.5 GB** (4-bit) | 0.07 GB | 0.016 GB | ~6 GB | **~10 GB** |

```
显存需求排序:

Full FT      ████████████████████████████████████  ~88 GB (4×A100)
Adapter      █████████                              ~23 GB (1×A100)
LoRA         ████████                               ~21 GB (1×A100)
QLoRA        ████                                   ~10 GB (1×RTX 3090!) ← 游戏改变者
```

**QLoRA 的革命性意义**：让单张消费级显卡（RTX 3090 / 4090）也能微调 7B 甚至 13B 模型。

### 1.4 训练速度对比

| 方法 | 训练速度（相对 Full FT）| 原因 |
|------|---------------------|------|
| Full FT | 1.0× | 基线 |
| Adapter | ~1.1× | 额外的 Adapter 前向/反向传播 |
| LoRA | ~0.9× | 低秩分支的计算开销小 |
| QLoRA | ~0.7× | 4-bit 反量化有额外开销 |

**注意**：训练速度不等于 wall-clock 时间。Full FT 需要更多 GPU，实际训练时间可能更长。

### 1.5 推理开销对比

| 方法 | 推理额外开销 | 原因 |
|------|-----------|------|
| Full FT | 0 | 就是完整模型 |
| Adapter | **有** | 每层多了 Adapter 前向传播 |
| Prefix Tuning | **有** | KV 序列增长了 prefix 长度 |
| Prompt Tuning | **有** | 输入序列增长了 prompt 长度 |
| **LoRA** | **0** | 训练后合并到原始权重中！ |

```
LoRA 的独特优势:

训练时:
  y = Wx + BAx    (额外分支)
  
推理时:
  W' = W + BA     (合并权重)
  y = W'x         (无额外开销！)

→ LoRA 是唯一推理零开销的 PEFT 方法
→ 这就是 LoRA 成为主流的核心原因
```

### 1.6 效果对比综合表

| 方法 | 参数效率 | 显存效率 | 训练速度 | 推理开销 | 通用效果 | 多任务部署 |
|------|---------|---------|---------|---------|---------|---------|
| Full FT | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| Prompt Tuning | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Prefix Tuning | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Adapter | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **LoRA** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **QLoRA** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 1.7 如何选择微调方法？

```
决策树:

你的 GPU 显存有多大？
  │
  ├─ < 16 GB (消费级) → QLoRA（唯一选择）
  │
  ├─ 16~24 GB (单卡 A100/3090/4090)
  │    │
  │    └─ LoRA 或 QLoRA
  │
  ├─ 24~80 GB (单卡 A100 80GB)
  │    │
  │    └─ 需要最佳效果？ → Full FT（小模型）或 LoRA（大模型）
  │
  └─ 多卡 / 集群
       │
       └─ Full FT（如果追求极致效果）
          LoRA（如果需要多任务或快速迭代）

额外考虑:
  多任务部署 → LoRA / Prompt Tuning（共享基座，切换轻量参数）
  追求极致效果 → Full FT > LoRA > Adapter
  追求效率 → QLoRA > LoRA > Adapter > Full FT
```

---

## Part 2：第五周知识串联与复盘

### 全周知识链路

```
Day 1: 指令微调核心思想
  预训练=补全 → SFT=遵循指令 → Loss Mask → 表面对齐假说 → InstructGPT 三阶段
  "为什么 SFT 用极少数据就能改变模型行为？"
       │
       │ "那数据从哪来？"
       ▼
Day 2: Stanford Alpaca 技术方案
  GPT-3.5 生成 52K 指令数据 → LLaMA-7B 全参数 SFT → 接近 text-davinci-003
  数据策略对比: Alpaca vs Vicuna vs WizardLM vs LIMA
       │
       │ "能否自动化生成高质量指令数据？"
       ▼
Day 3: Self-Instruct 数据生成 ★★★★ 本周核心实践
  种子任务 → 指令生成 → 分类判断 → 实例生成 → 质量过滤 → 多样性控制
  手写完整 Self-Instruct 管线 → Evol-Instruct 进化策略
       │
       │ "如何提升模型的推理能力？"
       ▼
Day 4: CoT / ToT 推理策略
  Few-shot CoT → Zero-shot CoT → Self-Consistency → Tree-of-Thought
  推理策略在 SFT 中的应用 → CoT 数据的价值
       │
       │ "全参数微调太贵了，有没有更高效的方法？"
       ▼
Day 5: 参数高效微调方法 (PEFT)
  Prompt Tuning → Prefix Tuning → Adapter → 数学推导 → 统一视角
  低秩假设 → 为 LoRA 铺路
       │
       │ "能不能在真实数据上跑通完整的 SFT？"
       ▼
Day 6: Alpaca 指令微调实践 ★★★★★ 本周核心实验
  数据处理 → Prompt Template → Loss Mask → SFT 训练 → 生成评估
  Full FT vs Adapter 实验对比
       │
       │ "各种微调方法效率如何？全周学了什么？"
       ▼
Day 7: 微调方法对比与复盘 ← 你在这里
  Full FT vs Adapter vs LoRA 效率对比 → 全周知识串联 → 为 W6 铺路
```

---

### 核心概念关系图

```
           ┌──────────────────────────────────────────────────────────┐
           │              指令微调（SFT）完整技术栈                       │
           │                                                          │
           │  ┌──────────────────────────────────────────────────┐   │
           │  │              数据层 (Day 1,2,3)                    │   │
           │  │                                                    │   │
           │  │  SFT 数据格式: (instruction, input, output)        │   │
           │  │      ↓                                             │   │
           │  │  Prompt Template: Alpaca / ChatML / Llama2-Chat    │   │
           │  │      ↓                                             │   │
           │  │  数据来源:                                          │   │
           │  │    人工标注 (InstructGPT)                           │   │
           │  │    Self-Instruct 自动生成 (Alpaca)  ← Day 3 手写   │   │
           │  │    用户对话数据 (Vicuna)                            │   │
           │  │    进化生成 (WizardLM)                              │   │
           │  │      ↓                                             │   │
           │  │  数据质量: 质量 > 数量 (LIMA)                       │   │
           │  └──────────────────────────────────────────────────┘   │
           │                         │                                │
           │                         ▼                                │
           │  ┌──────────────────────────────────────────────────┐   │
           │  │              训练层 (Day 1,5,6)                    │   │
           │  │                                                    │   │
           │  │  Loss Mask: 只在 output 部分计算 loss              │   │
           │  │      ↓                                             │   │
           │  │  微调方法选择:                                      │   │
           │  │    Full FT → 全参数更新                            │   │
           │  │    Adapter → 插入瓶颈网络                          │   │
           │  │    Prompt/Prefix Tuning → 可训练前缀               │   │
           │  │    LoRA → 低秩分解（第 6 周）                      │   │
           │  │      ↓                                             │   │
           │  │  训练策略: 低 LR + 少 Epoch + 梯度裁剪            │   │
           │  └──────────────────────────────────────────────────┘   │
           │                         │                                │
           │                         ▼                                │
           │  ┌──────────────────────────────────────────────────┐   │
           │  │              推理层 (Day 4)                         │   │
           │  │                                                    │   │
           │  │  基础: 标准 Prompting → CoT → Zero-shot CoT       │   │
           │  │  进阶: Self-Consistency → ToT → Reflexion          │   │
           │  │  应用: CoT 数据纳入 SFT 训练                       │   │
           │  │  展望: o1 / R1 用 RL 学会推理（第 17 周）           │   │
           │  └──────────────────────────────────────────────────┘   │
           └──────────────────────────────────────────────────────────┘
```

---

### 全周自检清单

#### 概念层 — 指令微调基础

- [ ] 解释预训练和指令微调的本质区别（补全 vs 遵循指令）
- [ ] 写出 SFT 的 Loss 公式，说明为什么只计算 output 部分
- [ ] 解释"表面对齐假说"（LIMA），说明为什么少量数据就能改变模型行为
- [ ] 解释 SFT 数据的标准三元组格式和 Prompt Template 的作用
- [ ] 对比 Alpaca 格式和 ChatML 格式的优缺点

#### 论文层 — 方案理解与对比

- [ ] 用一句话概括 Stanford Alpaca 的核心贡献
- [ ] 解释 Self-Instruct 的四阶段流程及每阶段的作用
- [ ] 对比 Alpaca、Vicuna、WizardLM、LIMA 的数据策略差异
- [ ] 解释 Evol-Instruct 的深度进化和广度进化策略
- [ ] 从知识蒸馏的角度分析 Alpaca 的理论上限

#### 推理策略 — CoT / ToT

- [ ] 解释 Chain-of-Thought 为什么能提升推理能力（从概率分解的角度）
- [ ] 区分 Few-shot CoT 和 Zero-shot CoT，写出 Zero-shot CoT 的触发短语
- [ ] 解释 Self-Consistency 的核心思想（多数投票）和数学形式
- [ ] 比较 CoT / Self-Consistency / ToT 的推理结构和适用场景
- [ ] 说明 CoT 数据在 SFT 中的价值

#### PEFT 方法 — 参数高效微调

- [ ] 说明 PEFT 的核心动机和"低秩假设"
- [ ] 对比 Prompt Tuning / Prefix Tuning / Adapter 的参数注入位置和参数量
- [ ] 推导 Adapter 的参数量公式：$4Ldr$
- [ ] 解释 Adapter 的近零初始化和残差连接的作用
- [ ] 解释 LoRA 相比 Adapter 的核心优势（推理零开销）

#### 代码层 — 手写能力

- [ ] **手写 Alpaca Prompt Template**（有 input / 无 input 两种）
- [ ] **手写 SFT Loss Mask 实现**（labels 中 prompt 部分标记为 -100）
- [ ] **手写 Self-Instruct 数据管线**（指令生成 + 质量过滤 + 多样性控制）
- [ ] **手写 ROUGE-L 计算**（LCS 算法）
- [ ] **手写 Self-Consistency 多数投票**
- [ ] **手写 Adapter 模块**（含近零初始化和残差连接）
- [ ] **完成完整 SFT 训练循环**（数据 → 训练 → 评估 → 生成）

#### 工程层 — 实践理解

- [ ] 计算 LLaMA-7B 全参数微调的显存需求
- [ ] 对比 Full FT / Adapter / LoRA / QLoRA 的显存需求
- [ ] 解释为什么 QLoRA 能在消费级 GPU 上微调 7B 模型
- [ ] 说明 SFT 训练中学习率、epoch 数、数据配比的选择策略
- [ ] 了解 HuggingFace PEFT 库的基本使用

---

### 重要公式速查卡

| 公式 | 来源 |
|------|------|
| $\mathcal{L}_{\text{SFT}} = -\sum_t \log P(y_t \mid \text{inst}, \text{inp}, y_{<t}; \theta)$ | SFT Loss (Day 1) |
| $\text{labels}[:prompt\_len] = -100$ | Loss Mask (Day 1, 6) |
| ROUGE-L $= \frac{2 \cdot P \cdot R}{P + R}$，$P = \text{LCS}/n$，$R = \text{LCS}/m$ | Self-Instruct 去重 (Day 3) |
| $P(\text{ans} \mid q) = \sum_r P(\text{ans} \mid r, q) \cdot P(r \mid q)$ | CoT 概率分解 (Day 4) |
| $\hat{a} = \arg\max_a \sum_i \mathbb{1}[a_i = a]$ | Self-Consistency 投票 (Day 4) |
| $\text{Prompt Tuning 参数} = k \times d$ | Prompt Tuning (Day 5) |
| $\text{Prefix Tuning 参数} = L \times 2k \times d$ | Prefix Tuning (Day 5) |
| $\text{Adapter 参数} = 4Ldr$ | Adapter (Day 5) |
| $\text{Adapter}(x) = x + \text{ReLU}(xW_{\text{down}})W_{\text{up}}$ | Adapter 前向 (Day 5, 6) |
| $\text{LoRA}: y = Wx + BAx$，推理时 $y = (W+BA)x$ | LoRA 预告 (Day 5, 7) |
| $\text{显存}_{\text{FP16}} \approx 2 \times |\theta|$ bytes | 显存分析 (Day 7) |
| $\text{AdamW 状态} \approx 4 \times \text{模型参数显存}$ | 显存分析 (Day 7) |

---

### 从 GPT 到 LLaMA 到 Alpaca：完整技术链路

| 周次 | 阶段 | 核心内容 | 关键产出 |
|------|------|---------|---------|
| W3 | 手撕 GPT | Transformer 架构、预训练、采样 | 手写 GPT 模型 |
| W4 | 手撕 LLaMA | RMSNorm/RoPE/SwiGLU/GQA、KV Cache | 手写 LLaMA 模型 |
| **W5** | **手撕 Alpaca** | **SFT、数据生成、CoT、PEFT** | **手写 SFT + Self-Instruct** |
| W6 | 手撕 LoRA | LoRA/QLoRA 数学推导、Alpaca-LoRA | 手写 LoRA + 实践 |
| W7 | Chinese-LLaMA2 | 中文适配、词表扩展、数据工程 | 中文 SFT 实践 |

```
技术能力演进:

W3: 会建模型 (GPT)
  ↓
W4: 会建更好的模型 (LLaMA)
  ↓
W5: 会让模型遵循指令 (SFT)  ← 本周
  ↓
W6: 会高效地让模型遵循指令 (LoRA)  ← 下周
  ↓
W7+: 会适配特定语言和领域
```

---

### 常见疑惑解答

**Q1：Full FT 效果一定比 LoRA 好吗？**

不一定。在数据量较小时（< 10K），LoRA 可能效果更好，因为：
1. LoRA 的参数约束起到了正则化作用，减少过拟合
2. 冻结预训练参数保护了通用能力
3. 实证研究表明在多数 benchmark 上 LoRA 与 Full FT 差距很小（< 1%）

只有在大规模数据和计算资源充足的情况下，Full FT 的优势才会明显。

**Q2：Alpaca 的 52K 数据够用吗？Vicuna 的 70K 对话数据呢？**

关键不在数量而在质量和多样性：
- LIMA 证明了 1K 高质量数据就够了
- Alpaca 的 52K 数据覆盖了广泛的任务类型，但质量受限于 GPT-3.5
- Vicuna 的多轮对话数据在对话场景下效果更好
- 现代最佳实践：混合多来源数据 + 严格质量过滤

**Q3：CoT 训练数据和 CoT Prompting 的区别是什么？**

| | CoT Prompting | CoT 训练数据 |
|---|---|---|
| 时机 | 推理时 | 训练时 |
| 方式 | 在 prompt 中添加推理示例 | 在 SFT 数据中包含推理过程 |
| 效果 | 只在推理时有效 | 模型学会了推理能力 |
| 适用 | 大模型（≥100B） | 任何规模模型 |

最强的方式是两者结合：用包含 CoT 的数据训练，推理时再用 CoT prompting。

**Q4：为什么 LoRA 要在第 6 周单独一周学？**

因为 LoRA / QLoRA 是当前最主流的微调方法，内容密度足够支撑一整周：
1. LoRA 的数学推导（低秩分解 + SVD 关联）
2. QLoRA 的量化技术（NF4 + 双重量化 + 分页优化器）
3. 超参数选择（rank、alpha、target modules）
4. Alpaca-LoRA 完整实践
5. 与 Full FT 的系统性对比实验

**Q5：Self-Instruct 生成的数据质量如何保障？**

多层质量控制：
1. **种子任务质量**：人工精心设计，覆盖多任务类型
2. **ROUGE-L 去重**：避免生成重复指令
3. **多样性控制**：监控动词分布，防止模式单一化
4. **格式检查**：确保 instruction/output 格式规范
5. **内容过滤**：排除 LLM 无法完成的任务
6. **人工抽查**：最终质量校验

**Q6：Adapter 和 LoRA 都是在模型中加东西，为什么 LoRA 推理零开销？**

核心区别在于数学结构：

```
Adapter:
  y = f(x) + adapter(f(x))
  → adapter 是非线性网络，无法与 f 合并
  → 推理时必须额外计算 adapter

LoRA:
  y = Wx + BAx = (W + BA)x
  → BA 是线性变换，可以直接加到 W 上
  → 推理时 W' = W + BA，之后只用 W'
  → 零额外开销
```

这就是线性的力量——LoRA 利用了权重更新的线性可加性。

---

### 本周与课程整体的连接

| 本周学到的 | 后续如何演进 |
|----------|-----------|
| SFT 数据格式与 Loss Mask | → **W6 LoRA 微调**使用相同的数据处理流程 |
| Self-Instruct 数据生成 | → **W7 中文数据工程**，如何为中文 LLaMA 生成训练数据 |
| Alpaca Prompt Template | → **W7-8 对话部署**，多轮对话 prompt 管理 |
| CoT 推理策略 | → **W17 o1 推理**，用 RL 训练模型学会 CoT |
| Prompt/Prefix/Adapter | → **W6 LoRA/QLoRA** 系统深化，完成 PEFT 全景认知 |
| Full FT 实验 | → **W6 LoRA 实验**对比，验证 LoRA 的参数效率 |
| 指令微调全流程 | → **W9-11 RLHF/DPO**，从 SFT 走向人类对齐 |

---

### 下周预告：第 6 周 · 手撕 LoRA / QLoRA

本周 Day 5 已经预习了 PEFT 方法的基础，Day 7 对比了各种方法的效率差异。第 6 周将深入 LoRA / QLoRA——当前最主流的参数高效微调方法。

| 主题 | 内容 |
|------|------|
| **LoRA 数学推导** | 低秩分解 $\Delta W = BA$、SVD 关联、秩的选择 |
| **QLoRA 量化技术** | NF4 量化、双重量化、分页优化器 |
| **Alpaca-LoRA 实践** | 在 LLaMA 上用 LoRA 做指令微调 |
| **超参数工程** | rank、alpha、target modules 的最优选择 |
| **系统对比** | Full FT / Adapter / LoRA / QLoRA 全面实验对比 |

**准备工作**：
1. 回顾 Day 5 的 PEFT 方法基础（尤其是 Adapter 的瓶颈结构）
2. 理解"低秩假设"——LoRA 的核心理论基础
3. 回顾线性代数中的 SVD（奇异值分解）基础知识
4. 确保能完成 Day 6 的 SFT 实验——这是 LoRA 实验的基线

---

### 产出要求

- [ ] 完成 Full FT / Adapter / LoRA 参数量和显存的数值计算（以 LLaMA-7B 为基准）
- [ ] 撰写微调方法选择决策树（根据显存、效果、部署需求）
- [ ] 完成全周自检清单中的所有项目
- [ ] 确认能闭卷手写本周所有核心代码（Prompt Template / Loss Mask / Self-Instruct / Adapter / SFT 训练循环）
- [ ] 画出「预训练 → SFT → RLHF」三阶段的完整技术路线图
- [ ] 预习 LoRA 的基本概念，阅读 Hu et al. (2021) 论文的 Abstract 和 Introduction

---

> **第五周总结**：你现在应该能够解释指令微调的完整技术链路——从"为什么需要 SFT"的原理，到 Self-Instruct 数据生成的工程实践，到 CoT 推理策略的理论基础，到 PEFT 方法的数学推导，再到完整 SFT 实验的代码实现。你掌握了从"预训练模型"到"指令遵循模型"的关键转变，这是 LLM 从"能力"走向"可用"的核心桥梁。下周我们将学习 LoRA——当前最主流的微调技术，让这个桥梁变得更加经济高效。
