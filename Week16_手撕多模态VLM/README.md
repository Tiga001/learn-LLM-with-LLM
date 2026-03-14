# 第 16 周：手撕多模态 VLM

> **目标**：精读 ViT / CLIP / LLaVA 三大多模态核心论文，理解从视觉 Transformer 到视觉-语言对比学习、再到多模态大模型的完整技术演进；从零手写 ViT、CLIP（含 InfoNCE Loss）和 LLaVA 的核心代码；掌握 Visual Instruction Tuning 两阶段训练范式；在小规模数据上跑通多模态图文问答。

---

## 每日安排（建议每日 3~5 小时）

| 天 | 主题 | 核心任务 | 产出 | 难度 |
|----|------|---------|------|:----:|
| Day 1 | 多模态大模型论文精读 | ViT / CLIP / LLaVA 核心创新对比 | 论文对比表 + 技术演进时间线 | ⭐⭐⭐ |
| Day 2 | ViT 与 CLIP 架构详解 | Patch Embedding / 双塔架构 / InfoNCE 数学推导 | 架构图 + 数学推导笔记 | ⭐⭐⭐⭐ |
| Day 3 | **手写 ViT 与 CLIP 模型** | 从零实现 PatchEmbed → ViT → CLIP Image/Text Encoder → InfoNCE | `vit.py` + `clip.py` 可运行 | ⭐⭐⭐⭐⭐ |
| Day 4 | CLIP 对比学习与训练详解 | 对比学习脉络、InfoNCE 深入推导、训练策略与能力边界 | 理解笔记 | ⭐⭐⭐⭐ |
| Day 5 | LLaVA 多模态大模型架构 | LLaVA 架构设计、两阶段训练、LLaVA-1.5 改进 | 架构图 + 训练流程笔记 | ⭐⭐⭐⭐ |
| Day 6 | **手写 LLaVA 多模态实践** | 手写 LLaVA（Vision Encoder + Projector + LLM）+ 图文问答 | 训练脚本 + 推理示例 | ⭐⭐⭐⭐⭐ |
| Day 7 | 多模态前沿 + 周复盘 | 音频/视频多模态扩展；串联全周知识 | 前沿笔记 + 自检 | ⭐⭐⭐⭐ |

---

## 知识图谱

```
Day 1: 论文精读 — 多模态大模型的演进脉络
  ViT (图像 → patch 序列) → CLIP (图文对比学习) → LLaVA (视觉指令微调)
  "从视觉 Transformer，到视觉-语言对齐，再到多模态大模型"
       │
       ▼
Day 2: 架构详解 — ViT + CLIP 的完整数学
  Patch Embedding → ViT Encoder → CLIP 双塔 → InfoNCE Loss → Zero-Shot 推理
       │
       ▼
Day 3: 手写 ViT + CLIP — 核心代码实现（本周重点之一！）
  PatchEmbed → ViTBlock (LN + MSA + FFN) → ViT → CLIP Image/Text Encoder → InfoNCE
       │
       ▼
Day 4: 对比学习深入 — 从理论到训练策略
  NCE → InfoNCE → 互信息下界 → 温度参数 → batch size 影响 → 能力边界
       │
       ▼
Day 5: LLaVA 架构 — 连接视觉与语言的桥梁
  CLIP ViT (冻结) → Projector (MLP) → LLM (Vicuna) → 两阶段训练
       │                                                    │
       ▼                                                    ▼
Day 6: 手写 LLaVA 实践（本周重点之二！）           与各 VLM 的对比
  Vision Encoder → MLP Projector → 多模态拼接 → 训练 → 图文问答推理
       │
       ▼
Day 7: 多模态前沿 + 全周复盘
  音频 (Whisper) / 视频 (Video-LLaMA) / 统一模型 (Gemini, GPT-4o)
  → 第 17 周 o1 推理铺路
```

---

## 与前序周次的关系

| 本周内容 | 前序基础 | 后续衔接 |
|---------|---------|---------|
| ViT Encoder Block | W2-3 Transformer Encoder / Multi-Head Attention | — |
| CLIP Text Encoder | W3 GPT Decoder-only 架构 | — |
| InfoNCE Loss | W9-10 RL 中的 Loss 设计经验 | — |
| LLaVA 的 LLM 部分 | W4 手写 LLaMA / W5-6 指令微调 + LoRA | — |
| LLaVA Projector 训练 | W5 Alpaca 微调 / W6 LoRA 微调策略 | — |
| 多模态推理 | W4 KV Cache / W14 推理加速 | W17 o1 推理 |
| 多模态 Agent | W8 Agent / Tool Use / RAG | — |

---

## 文件结构

```
Week16_手撕多模态VLM/
├── README.md                              ← 你在这里
├── Day1_多模态大模型论文精读.md             ← ViT / CLIP / LLaVA 论文核心创新对比
├── Day2_ViT与CLIP架构详解.md               ← ViT + CLIP 完整数学推导
├── Day3_手写ViT与CLIP模型.ipynb            ← 手写 ViT + CLIP 完整实现 (核心!)
├── Day4_CLIP对比学习与训练详解.md           ← InfoNCE 推导 + 训练策略 + 能力边界
├── Day5_LLaVA多模态大模型架构.md            ← LLaVA-1.0/1.5 架构 + Visual Instruction Tuning
├── Day6_手写LLaVA多模态实践.ipynb           ← 手写 LLaVA + 两阶段训练 + 图文问答 (核心!)
└── Day7_多模态前沿与复盘.md                 ← 音频/视频扩展 + 全周复盘
```

---

## 关键检查点 ✅

完成本周学习后，你应该能够：

- [ ] 画出 ViT / CLIP / LLaVA 各自的架构图（含维度标注）
- [ ] 解释 ViT 为什么将图像切分为 patch 而不是使用 CNN 特征
- [ ] 写出 Patch Embedding 的数学公式：$\mathbf{z}_0 = [\mathbf{x}_\text{cls}; \mathbf{x}_p^1 \mathbf{E}; \dots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}$
- [ ] **闭卷手写 Patch Embedding（含 CLS token 和位置编码）**
- [ ] **闭卷手写 ViT Encoder Block（LayerNorm + MSA + FFN + Residual）**
- [ ] **闭卷手写完整 ViT 模型**
- [ ] **手写 InfoNCE Loss（面试高频！）**
- [ ] **手写完整 CLIP 模型（Image Encoder + Text Encoder + 对比损失）**
- [ ] 解释 CLIP 的 zero-shot 分类机制
- [ ] 解释 LLaVA 两阶段训练（Stage 1 对齐 vs Stage 2 指令微调）各冻结哪些参数
- [ ] **手写 LLaVA 的 MLP Projector 和多模态输入拼接逻辑**
- [ ] 实现 LLaVA 的推理流程：图像编码 → 投影 → 拼接 → LLM 自回归生成
- [ ] 对比 LLaVA / BLIP-2 / Flamingo 的视觉-语言连接方式

---

## 本周必读论文

1. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (ViT, Dosovitskiy et al., 2020) — **精读**
2. **Learning Transferable Visual Models From Natural Language Supervision** (CLIP, Radford et al., 2021) — **精读**
3. **Visual Instruction Tuning** (LLaVA, Liu et al., 2023) — **精读**

## 参考论文

- *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5, Liu et al., 2023)
- *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models* (Li et al., 2023)
- *Flamingo: a Visual Language Model for Few-Shot Learning* (Alayrac et al., 2022)
- *Representation Learning with Contrastive Predictive Coding* (CPC, van den Oord et al., 2018) — InfoNCE 起源
- *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR, Chen et al., 2020)

## 推荐资源

- Haotian Liu: [LLaVA 官方仓库](https://github.com/haotian-liu/LLaVA) — 多模态 LLM 参考实现
- OpenAI: [CLIP 官方仓库](https://github.com/openai/CLIP) — CLIP 原始实现
- Google: [Vision Transformer 论文页](https://arxiv.org/abs/2010.11929)
- Lilian Weng: [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) — 对比学习综述
- Jay Alammar: [The Illustrated ViT](https://jalammar.github.io/)
- HuggingFace: [CLIP 模型文档](https://huggingface.co/docs/transformers/model_doc/clip)
