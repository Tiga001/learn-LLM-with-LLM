# 第 5 周：手撕 Alpaca（指令微调）

> **目标**：理解指令微调（Instruction Tuning / SFT）的核心思想——让模型从"补全"转变为"遵循指令"；精读 Stanford Alpaca 论文与 Self-Instruct 数据生成流程；掌握 CoT/ToT 推理策略；系统对比 Prompt Tuning / Prefix Tuning / Adapter 等参数高效微调方法；在小规模数据上完成一次完整的指令微调实验。

---

## 每日安排（建议每日 3~5 小时）

| 天 | 主题 | 核心任务 | 产出 | 难度 |
|----|------|---------|------|:----:|
| Day 1 | 指令微调核心思想 | SFT 原理、数据格式、从"补全"到"遵循指令"的范式转变 | SFT 数据格式笔记 + 对比实验 | ⭐⭐⭐ |
| Day 2 | Stanford Alpaca 技术方案 | Alpaca 论文精读、数据生成流程、与 Vicuna 等方案对比 | 论文笔记 + 技术分析表 | ⭐⭐⭐ |
| Day 3 | **Self-Instruct 数据生成** | 自动化指令数据生成管线、质量控制、多样性保障 | 手写 Self-Instruct 数据管线 | ⭐⭐⭐⭐ |
| Day 4 | CoT / ToT 推理策略 | Chain-of-Thought / Tree-of-Thought / Few-shot 原理与实践 | CoT 实践 + 笔记 | ⭐⭐⭐ |
| Day 5 | 参数高效微调方法 | Prompt Tuning / Prefix Tuning / Adapter 原理对比 | 三种方法数学推导 + 对比笔记 | ⭐⭐⭐⭐ |
| Day 6 | **Alpaca 指令微调实践** | 完整 SFT 实验：数据处理→训练→评估→生成 | 训练脚本 + 生成示例 | ⭐⭐⭐⭐⭐ |
| Day 7 | 微调方法对比与复盘 | Full FT / Adapter / LoRA 效率对比；串联全周知识 | 实验对比表 + 自检 | ⭐⭐⭐⭐ |

---

## 知识图谱

```
Day 1: 指令微调核心思想 — 为什么需要 SFT？
  预训练 = 补全 (next token prediction)
  SFT = 学习遵循指令 (instruction following)
  "预训练给模型知识，SFT 教模型如何使用知识"
       │
       ▼
Day 2: Stanford Alpaca — 52K 指令数据的威力
  数据来源 (Self-Instruct) → 训练策略 → 效果评估
  "用 GPT 生成数据微调 LLaMA → 逼近 text-davinci-003"
       │
       ▼
Day 3: Self-Instruct — 自动化数据生成（本周重要实践！）
  种子任务 → LLM 生成 → 质量过滤 → 多样性控制
       │
       ▼
Day 4: CoT / ToT — 推理能力的涌现与引导
  Few-shot → Chain-of-Thought → Tree-of-Thought → Self-Consistency
       │                                              │
       ▼                                              ▼
Day 5: 参数高效微调 — 不动全部参数也能微调        → 第 17 周 o1 推理深化
  Prompt Tuning → Prefix Tuning → Adapter → (LoRA → 第 6 周)
       │
       ▼
Day 6: Alpaca 指令微调实践（本周核心实验！）
  数据处理 → 模型加载 → SFT 训练 → 评估 → 文本生成
       │
       ▼
Day 7: 效率对比 + 全周复盘
  Full FT vs Adapter vs LoRA → 第 6 周 LoRA/QLoRA 深入
```

---

## 与前序周次的关系

| 本周内容 | 前序基础 | 后续衔接 |
|---------|---------|---------|
| SFT 数据格式 | W3-4 预训练数据 | W6 LoRA 微调数据 |
| Alpaca 指令微调 | W4 LLaMA 模型架构 | W7 Chinese-LLaMA2 |
| Self-Instruct | W1 Tokenizer / 数据处理 | W7 数据工程 |
| CoT / ToT | W3 GPT 文本生成 | W17 o1 推理 / MCTS |
| Prompt/Prefix/Adapter | W2-4 Transformer / LLaMA | W6 LoRA / QLoRA |
| Full FT vs PEFT | W4 LLaMA 预训练 | W6 LoRA 系统深化 |

---

## 文件结构

```
Week5_手撕Alpaca/
├── README.md                           ← 你在这里
├── Day1_指令微调核心思想.md              ← SFT 原理与数据格式
├── Day2_Stanford_Alpaca技术方案.md       ← Alpaca 论文精读与方案分析
├── Day3_Self-Instruct数据生成.ipynb     ← 手写 Self-Instruct 数据管线 (实践!)
├── Day4_CoT与推理策略.md                ← CoT/ToT/Few-shot 原理与实践
├── Day5_参数高效微调方法.md              ← Prompt/Prefix/Adapter 数学推导与对比
├── Day6_Alpaca指令微调实践.ipynb         ← 完整 SFT 实验 (核心!)
└── Day7_微调方法对比与复盘.md            ← 效率对比 + 周复盘 + 第 6 周铺路
```

---

## 关键检查点 ✅

完成本周学习后，你应该能够：

- [ ] 解释预训练和指令微调的本质区别（补全 vs 遵循指令）
- [ ] 说明 SFT 数据的标准格式（Instruction / Input / Output）
- [ ] 理解 Stanford Alpaca 的技术方案和数据生成流程
- [ ] **手写 Self-Instruct 数据生成管线**
- [ ] 解释 CoT 为什么能提升推理能力，并实践 CoT prompting
- [ ] 说明 Prompt Tuning / Prefix Tuning / Adapter 的核心思想和数学形式
- [ ] 对比三种 PEFT 方法的参数量、训练效率和适用场景
- [ ] **完成一次完整的 Alpaca 风格指令微调实验**
- [ ] 对比 Full Finetune / Adapter / LoRA 的效率差异
- [ ] 为第 6 周 LoRA / QLoRA 深入学习做好铺垫

---

## 本周必读论文

1. **Stanford Alpaca: An Instruction-following LLaMA Model** (Taori et al., 2023) — **精读**
2. **Self-Instruct: Aligning Language Models with Self-Generated Instructions** (Wang et al., 2023) — **精读**

## 参考论文

- *Training language models to follow instructions with human feedback* (InstructGPT, Ouyang et al., 2022)
- *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (Wei et al., 2022)
- *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (Yao et al., 2023)
- *The Power of Scale for Parameter-Efficient Prompt Tuning* (Lester et al., 2021)
- *Prefix-Tuning: Optimizing Continuous Prompts for Generation* (Li & Liang, 2021)
- *Parameter-Efficient Transfer Learning for NLP* (Houlsby et al., 2019) — Adapter 原始论文
- *Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality (Chiang et al., 2023)

## 推荐资源

- Stanford Alpaca: [GitHub 仓库](https://github.com/tatsu-lab/stanford_alpaca)
- Self-Instruct: [GitHub 仓库](https://github.com/yizhongw/self-instruct)
- HuggingFace: [PEFT 库文档](https://huggingface.co/docs/peft)
- Lilian Weng: [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- Sebastian Raschka: [Finetuning LLMs](https://magazine.sebastianraschka.com/p/finetuning-large-language-models)
