# 第 6 周：手撕 LoRA / QLoRA

> **目标**：精读 LoRA 与 QLoRA 两篇核心论文，从低秩分解的数学原理出发，推导 LoRA 的前向传播公式 $h = W_0 x + \frac{\alpha}{r} BAx$；手写 LoRA Linear 层并在 LLaMA 架构上应用；深入理解 QLoRA 的 NF4 量化、双重量化（Double Quantization）与分页优化器（Paged Optimizer）三大创新；了解推理量化（GPTQ / AWQ / GGUF）的定位与使用场景；在单卡上完成 LLaMA2 + QLoRA 的完整微调实验。

---

## 每日安排（建议每日 3~5 小时）

| 天 | 主题 | 核心任务 | 产出 | 难度 |
|----|------|---------|------|:----:|
| Day 1 | LoRA 与 QLoRA 论文精读 | 两篇论文核心思想、动机与关键实验 | 论文笔记 + 技术分析表 | ⭐⭐⭐ |
| Day 2 | LoRA 算法推导 | 低秩分解数学原理、SVD 关联、权重矩阵低秩性分析 | 手写 LoRA 数学推导 | ⭐⭐⭐⭐ |
| Day 3 | **手写 LoRA 实现** | LoRA Linear 层、LoRA 注入 LLaMA、权重合并 | 手写 LoRA + 训练验证 | ⭐⭐⭐⭐⭐ |
| Day 4 | QLoRA 量化原理 | NF4 量化、双重量化、分页优化器 | 量化数学推导 + 笔记 | ⭐⭐⭐⭐ |
| Day 5 | 推理量化简介 | GPTQ / AWQ / GGUF 原理与定位 | 量化方法对比笔记 | ⭐⭐⭐ |
| Day 6 | **LLaMA2 + QLoRA 微调实践** | 完整 QLoRA 实验：数据→量化→LoRA→训练→评估 | 训练脚本 + 实验报告 | ⭐⭐⭐⭐⭐ |
| Day 7 | LoRA 超参数工程与复盘 | rank/alpha/target_modules 选择策略；全周串联 | 超参数实验 + 自检 | ⭐⭐⭐⭐ |

---

## 知识图谱

```
Day 1: LoRA / QLoRA 论文精读 — 为什么低秩就够了？
  LoRA 核心动机: 微调的权重更新矩阵 ΔW 是低秩的
  QLoRA 核心动机: 用 4-bit 量化压缩基座模型，省下的显存给 LoRA
       │
       ▼
Day 2: LoRA 算法推导 — 数学基础（本周理论核心！）
  低秩分解 ΔW = BA → SVD 视角 → 为什么 Attention 权重是低秩的
  rank / alpha / 初始化策略的数学解释
       │
       ▼
Day 3: 手写 LoRA 实现（本周重要实践！）
  LoRALinear 层 → 注入 LLaMA → 冻结/训练参数管理 → 权重合并
       │
       ▼
Day 4: QLoRA 量化原理 — 三大创新
  NF4 量化 → 双重量化 → 分页优化器
  "让单张 RTX 3090 也能微调 65B 模型"
       │
       ▼
Day 5: 推理量化简介 — 训练量化 vs 推理量化          → 第 14 周系统深化
  GPTQ (PTQ) → AWQ (激活感知) → GGUF (CPU 推理)
       │
       ▼
Day 6: LLaMA2 + QLoRA 微调实践（本周核心实验！）
  模型量化加载 → LoRA 注入 → SFT 训练 → 权重合并 → 推理评估
       │
       ▼
Day 7: 超参数工程 + 全周复盘
  rank / alpha / target_modules 选择 → Full FT vs LoRA vs QLoRA 对比
  → 第 7 周 Chinese-LLaMA2 衔接
```

---

## 与前序周次的关系

| 本周内容 | 前序基础 | 后续衔接 |
|---------|---------|---------|
| LoRA 低秩分解 | W5 Day5 PEFT 方法概述、低秩假设 | W7 Chinese-LLaMA2 微调 |
| LoRA 实现 | W4 LLaMA 模型架构（Attention/FFN） | W12 垂域 Chatbot LoRA 微调 |
| QLoRA 量化 | W4 FP16 显存分析 | W14 推理量化深化 |
| 推理量化简介 | W4 KV Cache 推理 | W14 GPTQ/AWQ/vLLM 系统讲解 |
| SFT + LoRA 实践 | W5 Day6 SFT 实验、Loss Mask | W7 中文数据 SFT |
| 超参数工程 | W5 Day7 微调方法对比 | W12 垂域微调实践 |

---

## 文件结构

```
Week6_手撕LoRA/
├── README.md                               ← 你在这里
├── Day1_LoRA与QLoRA论文精读.md              ← 两篇论文精读与技术分析
├── Day2_LoRA算法推导.md                     ← 低秩分解数学推导（理论核心）
├── Day3_手写LoRA实现.ipynb                  ← 手写 LoRA Linear + LLaMA 注入 (实践!)
├── Day4_QLoRA量化原理.md                    ← NF4 / 双量化 / 分页优化器
├── Day5_推理量化简介.md                     ← GPTQ / AWQ / GGUF 对比
├── Day6_LLaMA2_QLoRA微调实践.ipynb          ← 完整 QLoRA 实验 (核心!)
└── Day7_LoRA超参数工程与复盘.md             ← 超参数选择 + 周复盘 + 第 7 周铺路
```

---

## 关键检查点 ✅

完成本周学习后，你应该能够：

- [ ] 说明 LoRA 的核心动机——微调权重更新矩阵 $\Delta W$ 的低秩性
- [ ] **手写 LoRA 前向传播：$h = W_0 x + \frac{\alpha}{r} \cdot BAx$（面试高频！）**
- [ ] 解释 LoRA 中 $A$ 和 $B$ 矩阵的初始化策略及其数学原因
- [ ] 推导 LoRA 的参数量公式，对比 Full FT / Adapter 的参数效率
- [ ] 解释 rank 和 alpha 的作用，说明实际中如何选择
- [ ] 说明 LoRA 推理零开销的原理——权重合并 $W' = W_0 + \frac{\alpha}{r} BA$
- [ ] 解释 QLoRA 的三大创新：NF4 量化、双重量化、分页优化器
- [ ] 区分训练量化（QLoRA）和推理量化（GPTQ / AWQ / GGUF）的定位
- [ ] 了解 GPTQ / AWQ / GGUF 三种推理量化方法的核心思路和适用场景
- [ ] **单卡跑通 LLaMA2 + QLoRA 微调实验**
- [ ] 根据任务需求选择合适的 rank / alpha / target_modules

---

## 本周必读论文

1. **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021) — **精读**
2. **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023) — **精读**

## 参考论文

- *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* (Aghajanyan et al., 2021)
- *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (Frantar et al., 2023)
- *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (Lin et al., 2024)
- *DoRA: Weight-Decomposed Low-Rank Adaptation* (Liu et al., 2024)
- *LoRA+: Efficient Low Rank Adaptation of Large Models* (Hayou et al., 2024)
- *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection* (Zhao et al., 2024)

## 推荐资源

- HuggingFace PEFT 库: [GitHub 仓库](https://github.com/huggingface/peft) / [文档](https://huggingface.co/docs/peft)
- Lilian Weng: [Large Language Model Fine-Tuning](https://lilianweng.github.io/)
- Sebastian Raschka: [Practical Tips for Finetuning LLMs](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- bitsandbytes 库: [GitHub 仓库](https://github.com/TimDettmers/bitsandbytes)
- llama.cpp (GGUF): [GitHub 仓库](https://github.com/ggerganov/llama.cpp)
