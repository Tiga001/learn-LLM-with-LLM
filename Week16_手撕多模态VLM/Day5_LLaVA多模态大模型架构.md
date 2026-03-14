# Day 5：LLaVA 多模态大模型架构 — 连接视觉与语言的桥梁

> **目标**：全面理解从 CLIP 到多模态大模型（VLM）的架构演进；掌握 LLaVA 的三大组件（Vision Encoder / Projector / LLM）及其设计动机；深入理解两阶段训练策略（预训练对齐 + 指令微调）；对比 LLaVA-1.0 与 LLaVA-1.5 的改进。为 Day 6 手写实现打下架构基础。

---

## 一、从 CLIP 到 VLM：多模态大模型架构演进

### 1.1 核心问题

有了 CLIP（视觉-语言对齐）和 LLM（语言理解与生成），如何将它们连接起来？

```
CLIP ViT:  Image → 视觉特征 (R^{N × D_v})
LLM:       Text tokens → 自然语言回答

问题: 如何让 LLM "看到" 视觉特征？
```

不同工作给出了不同的解决方案。

### 1.2 三种视觉-语言连接范式

```
范式 A: Cross-Attention 插入 (Flamingo, 2022)
  ┌──────┐                ┌─────────────────────────────────┐
  │Vision│──┐             │        LLM (冻结)                │
  │Encoder│  │             │  ┌──────┐  ┌──────┐  ┌──────┐  │
  └──────┘  │             │  │ LLM  │  │XAttn │  │ LLM  │  │
            └────────────▶│  │Layer │→ │Layer │→ │Layer │  │
                          │  └──────┘  └──────┘  └──────┘  │
                          └─────────────────────────────────┘

范式 B: Q-Former 桥接 (BLIP-2, 2023)
  ┌──────┐    ┌───────────────┐    ┌──────┐
  │Vision│──▶ │   Q-Former    │──▶ │ LLM  │
  │Encoder│    │(可学习 Query) │    │      │
  └──────┘    └───────────────┘    └──────┘

范式 C: 线性/MLP 投影 (LLaVA, 2023)
  ┌──────┐    ┌───────────┐    ┌──────┐
  │Vision│──▶ │ Projector │──▶ │ LLM  │
  │Encoder│    │(线性/MLP) │    │      │
  └──────┘    └───────────┘    └──────┘
```

### 1.3 三种范式对比

| 维度 | Flamingo (Cross-Attn) | BLIP-2 (Q-Former) | LLaVA (Projector) |
|------|:---:|:---:|:---:|
| 连接方式 | 在 LLM 层间插入交叉注意力 | 可学习查询 token 做桥接 | 简单线性/MLP 投影 |
| 视觉 token 数 | 可控（通过 Perceiver） | 固定 32 个 | 全部（如 256 个） |
| 修改 LLM 架构 | 是（插入新层） | 否 | 否 |
| 训练参数量 | 多（交叉注意力层） | 中（Q-Former ~188M） | 少（Projector ~几M） |
| 实现复杂度 | 高 | 中 | **低** |
| 性能 | 强 | 强 | **LLaVA-1.5 后同样强** |

**趋势**：LLaVA 的简单方案在 LLaVA-1.5 后证明了与复杂方案相当甚至更好的性能，成为主流。

---

## 二、LLaVA 架构全景图

### 2.1 完整架构

```
输入: 图像 (H × W × 3) + 文本指令 "Describe this image."
       │                              │
       ▼                              │
┌──────────────────────────┐          │
│  CLIP ViT-L/14 (冻结)    │          │
│  Image → Patch Tokens    │          │
│  输出: Z_v ∈ R^{N × D_v} │          │
│  N=256, D_v=1024          │          │
└──────────────────────────┘          │
       │                              │
       ▼                              │
┌──────────────────────────┐          │
│  Projector                │          │
│  LLaVA-1.0: Linear       │          │
│    W ∈ R^{D_v × D_l}     │          │
│  LLaVA-1.5: 2-layer MLP  │          │
│    Linear → GELU → Linear │          │
│  输出: H_v ∈ R^{N × D_l} │          │
└──────────────────────────┘          │
       │                              │
       ▼                              ▼
┌─────────────────────────────────────────────┐
│              Token Sequence                  │
│  [IMG][IMG]...[IMG] [BOS] Describe this ... │
│  ├── N 个视觉 token ──┤├── 文本 token ──────┤ │
│  各维度 D_l              各维度 D_l           │
└─────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  LLM (Vicuna / LLaMA)    │
│  自回归生成                │
│  输出: "This image shows..."│
└──────────────────────────┘
```

### 2.2 各组件详解

**Vision Encoder: CLIP ViT-L/14**

| 参数 | 值 |
|------|------|
| 架构 | ViT-Large, Patch=14 |
| 输入分辨率 | 224×224（LLaVA-1.0）/ 336×336（LLaVA-1.5） |
| Patch 数 $N$ | 256 / 576 |
| 输出维度 $D_v$ | 1024 |
| 参数量 | ~304M |
| 训练状态 | **全程冻结**（不参与训练） |

LLaVA 使用 CLIP ViT 最后一层之前（倒数第二层）的输出，因为倒数第二层保留了更多空间信息，最后一层过度聚合到了全局 CLS token。

**Projector: 维度对齐**

LLaVA-1.0（线性投影）：

$$\mathbf{H}_v = \mathbf{Z}_v \mathbf{W}, \quad \mathbf{W} \in \mathbb{R}^{D_v \times D_l}$$

参数量：$D_v \times D_l = 1024 \times 4096 \approx 4.2M$

LLaVA-1.5（两层 MLP）：

$$\mathbf{H}_v = \text{GELU}(\mathbf{Z}_v \mathbf{W}_1) \mathbf{W}_2$$

$$\mathbf{W}_1 \in \mathbb{R}^{D_v \times D_h}, \quad \mathbf{W}_2 \in \mathbb{R}^{D_h \times D_l}$$

参数量：$D_v \times D_h + D_h \times D_l \approx 8.4M$（$D_h = D_l = 4096$）

**LLM: Vicuna (LLaMA-based)**

| 参数 | LLaVA-7B | LLaVA-13B |
|------|:---:|:---:|
| 基座 | Vicuna-7B (LLaMA-based) | Vicuna-13B |
| 隐藏维度 $D_l$ | 4096 | 5120 |
| 层数 | 32 | 40 |
| 注意力头数 | 32 | 40 |
| 上下文长度 | 2048 | 2048 |

### 2.3 维度流完整示例（LLaVA-1.5-7B）

```
图像 (1, 3, 336, 336)
  │  CLIP ViT-L/14 (冻结)
  ▼
(1, 576, 1024)         ← 576 = (336/14)² 个 patch, 每个 1024 维
  │  MLP Projector
  │  Linear(1024, 4096) → GELU → Linear(4096, 4096)
  ▼
(1, 576, 4096)         ← 576 个视觉 token, 每个 4096 维 (= LLM 维度)

文本 "Describe this image." → Tokenizer → [28, 7, 5, ...]
  │  Token Embedding
  ▼
(1, T_text, 4096)      ← T_text 个文本 token

拼接:
(1, 576 + T_text, 4096) ← 全部 token 送入 LLM
  │  Vicuna-7B (32 层 LLaMA)
  ▼
(1, 576 + T_text, 32000) ← logits, 自回归生成
```

---

## 三、LLaVA 两阶段训练

### 3.1 训练策略总览

| | Stage 1: 预训练对齐 | Stage 2: 指令微调 |
|------|:---:|:---:|
| **目的** | 让 Projector 学会"翻译"视觉特征 | 让模型学会指令跟随 |
| **数据** | CC3M 558K 图文对 | LLaVA-Instruct 158K / 665K |
| **训练参数** | 仅 Projector | Projector + LLM |
| **冻结参数** | ViT + LLM | 仅 ViT |
| **学习率** | $2 \times 10^{-3}$ | $2 \times 10^{-5}$ |
| **训练时间** | ~4 小时 (8×A100) | ~10 小时 (8×A100) |
| **Epoch** | 1 | 1 |

### 3.2 Stage 1: 预训练对齐

**目标**：让 Projector 学会将 CLIP 的视觉特征"翻译"成 LLM 可理解的语义。

**数据格式**：简单的图文描述对

```json
{
  "image": "COCO_train2014_000000123456.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nProvide a brief description of the given image."},
    {"from": "gpt", "value": "A cat sitting on a wooden table near a window."}
  ]
}
```

其中 `<image>` 占位符在前向传播时被替换为视觉 token 序列。

**训练目标**：标准的自回归语言建模 loss，**仅计算 assistant 回答部分的 loss**（不对图像 token 和 human 问题计算 loss）。

$$\mathcal{L}_\text{Stage1} = -\sum_{t \in \text{answer}} \log P(x_t | x_{<t}, \mathbf{H}_v)$$

**直觉**：Stage 1 本质上是在做 Image Captioning，但目的不是学好 captioning，而是让 Projector 学会正确地"翻译"视觉特征。

### 3.3 Stage 2: Visual Instruction Tuning

**目标**：让模型具备多模态指令跟随能力（对话、描述、推理）。

**数据格式**：多轮视觉对话

```json
{
  "image": "COCO_train2014_000000123456.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is the cat doing in this image?"},
    {"from": "gpt", "value": "The cat is sitting on a wooden table..."},
    {"from": "human", "value": "What can you infer about the time of day?"},
    {"from": "gpt", "value": "Based on the warm sunlight coming through..."}
  ]
}
```

**LLaVA-Instruct 数据**（由 GPT-4 生成）：

| 类型 | 数量 | 说明 |
|------|:---:|------|
| 对话 (Conversation) | 58K | 多轮日常问答 |
| 详细描述 (Detail Description) | 23K | 图像的详细多段落描述 |
| 复杂推理 (Complex Reasoning) | 77K | 需要推理能力的问题 |
| **总计** | **158K** | — |

LLaVA-1.5 扩展到 665K，加入了学术 VQA 数据集：

| 数据集 | 数量 | 类型 |
|------|:---:|------|
| LLaVA-Instruct | 158K | GPT-4 生成对话 |
| VQAv2 | 83K | 视觉问答 |
| GQA | 72K | 组合推理 |
| OKVQA | 9K | 知识性问答 |
| TextVQA | 22K | 文字识别 |
| OCR-VQA | 80K | 文档理解 |
| ShareGPT | 40K | 纯文本对话 |
| **总计** | **~665K** | — |

### 3.4 训练 Loss 的关键细节

LLaVA 的训练 loss 有一个重要设计：**只对 assistant 回答的 token 计算 loss**。

```
Token sequence:
  [IMG][IMG]...[IMG]  Human: What is this?  Assistant: This is a cat.
  ├── 不计算 loss ──┤  ├── 不计算 loss ──┤  ├── 计算 loss ──────┤

Loss mask:
  [0, 0, ..., 0,       0, 0, 0, 0, 0,       1, 1, 1, 1, 1, 1]
```

这防止模型学习"复述问题"或"重建图像 token"，让它专注于生成高质量回答。

### 3.5 为什么两阶段？为什么不端到端？

| 方案 | 问题 |
|------|------|
| 直接端到端（全参数） | 视觉-语言未对齐，LLM 看不懂视觉 token，训练不稳定 |
| 只训练 Projector | Projector 参数太少，无法学会复杂指令跟随 |
| **Stage 1 + Stage 2** | **Stage 1 先对齐，Stage 2 再微调 LLM，渐进式学习** |

类比：学外语时，先学单词发音（对齐），再学造句和写作（指令跟随）。

---

## 四、LLaVA-1.5 改进

### 4.1 关键改进一览

| 改进 | LLaVA-1.0 | LLaVA-1.5 | 影响 |
|------|:---:|:---:|------|
| Projector | 单层 Linear | **两层 MLP + GELU** | +2-5% 各 benchmark |
| 分辨率 | 224×224 | **336×336** | 更多细节 |
| Vision Encoder 层 | 最后一层 | **倒数第二层** | 更好的空间特征 |
| Stage 2 数据 | 158K | **665K（含学术 VQA）** | 更全面 |
| LLM | Vicuna | Vicuna（不变） | — |

### 4.2 MLP Projector 为什么更好？

单层线性投影是一个**线性变换**，表达能力有限：

$$\mathbf{H}_v = \mathbf{Z}_v \mathbf{W} \quad (\text{线性})$$

两层 MLP 引入了**非线性**：

$$\mathbf{H}_v = \text{GELU}(\mathbf{Z}_v \mathbf{W}_1) \mathbf{W}_2 \quad (\text{非线性})$$

非线性让 Projector 可以学到更复杂的视觉-语言映射，特别是：
- 不同类型的视觉特征（颜色、形状、纹理）可能需要不同的变换
- GELU 激活提供了"软门控"效果

### 4.3 分辨率提升的效果

| 分辨率 | Patch 数 | 视觉 token 数 | 细节捕获 |
|:---:|:---:|:---:|------|
| 224×224 | $(224/14)^2 = 256$ | 256 | 适合粗粒度理解 |
| 336×336 | $(336/14)^2 = 576$ | 576 | 可看到更小的物体和文字 |

代价：更多视觉 token → LLM 输入序列更长 → 推理更慢、显存更大。

### 4.4 LLaVA-1.5 性能

LLaVA-1.5 在 12 个 benchmark 中 11 个取得了 SOTA（7B 和 13B 规模）：

| Benchmark | LLaVA-1.0 (7B) | LLaVA-1.5 (7B) | LLaVA-1.5 (13B) |
|-----------|:---:|:---:|:---:|
| VQAv2 | 76.5 | 78.5 | **80.0** |
| GQA | — | 62.0 | **63.3** |
| TextVQA | — | 58.2 | **61.3** |
| POPE | — | 85.9 | **85.9** |
| MMBench | — | 64.3 | **67.7** |

---

## 五、LLaVA 推理流程

### 5.1 单张图像问答

```python
# 伪代码
def llava_inference(model, image, question, tokenizer):
    # Step 1: 编码图像
    with torch.no_grad():
        visual_features = model.vision_encoder(image)   # (1, N, D_v)

    # Step 2: 投影到 LLM 空间
    visual_tokens = model.projector(visual_features)     # (1, N, D_l)

    # Step 3: 编码文本
    text_ids = tokenizer.encode(question)
    text_embeds = model.llm.embed_tokens(text_ids)       # (1, T, D_l)

    # Step 4: 拼接
    input_embeds = torch.cat([visual_tokens, text_embeds], dim=1)  # (1, N+T, D_l)

    # Step 5: LLM 自回归生成
    output_ids = model.llm.generate(inputs_embeds=input_embeds)

    # Step 6: 解码
    answer = tokenizer.decode(output_ids)
    return answer
```

### 5.2 多轮对话

在多轮对话中，图像只需编码一次：

```
第 1 轮:
  [IMG tokens] + "Human: What is in this image?" + "Assistant: " → 生成回答 A1

第 2 轮:
  [IMG tokens] + "Human: What is in this image?" + "Assistant: A1"
              + "Human: How many people?" + "Assistant: " → 生成回答 A2
```

视觉 token 在每轮对话中保持不变，只是文本上下文在增长。

### 5.3 推理优化

| 优化 | 说明 |
|------|------|
| KV Cache | LLM 部分使用 KV Cache 加速自回归生成（W4 已学） |
| 视觉缓存 | 图像只编码一次，多轮复用 |
| 量化 | Vision Encoder 和 LLM 可分别量化 |
| 批量推理 | 多张图片可以 batch 推理 |

---

## 六、与其他 VLM 的对比分析

### 6.1 架构对比

| 模型 | Vision Encoder | 桥接模块 | LLM | 视觉 token 数 |
|------|:---:|:---:|:---:|:---:|
| Flamingo | NFNet | Perceiver Resampler + XAttn | Chinchilla | 64 |
| BLIP-2 | ViT-G (冻结) | Q-Former (32 queries) | OPT / FlanT5 | 32 |
| MiniGPT-4 | ViT-G (冻结) | Linear (1 层) | Vicuna | 32 (via Q-Former) |
| **LLaVA** | **CLIP ViT-L/14** | **Linear / MLP** | **Vicuna** | **256 / 576** |
| Qwen-VL | ViT-G (可训练) | Cross-Attn Resampler | Qwen-7B | 256 |
| InternVL | InternViT-6B | PixelShuffle + MLP | InternLM2 | 256 |

### 6.2 LLaVA 的独特优势

1. **极简架构**：不修改 LLM 架构，不需要 Q-Former，只加一个 MLP
2. **保留全部视觉信息**：不压缩视觉 token（vs Flamingo/BLIP-2 压缩到 32-64 个）
3. **开源友好**：代码简洁，易于复现和修改
4. **数据生成范式**：用 GPT-4 生成训练数据的方法可推广

### 6.3 LLaVA 的局限

1. **视觉 token 过多**：576 个视觉 token 占用大量上下文窗口
2. **单分辨率**：固定分辨率，对不同大小的图片不够灵活
3. **Vision Encoder 冻结**：无法适配特定领域
4. **无 grounding 能力**：无法输出物体的坐标

后续工作（LLaVA-NeXT、LLaVA-OneVision）解决了部分问题。

---

## 七、自检题

### 基础题

1. 画出 LLaVA 的完整架构图，标注三个核心组件和数据流方向。
2. LLaVA 两阶段训练中，Stage 1 和 Stage 2 分别训练哪些参数？
3. LLaVA-1.5 相比 LLaVA-1.0 的三个主要改进是什么？
4. 为什么 LLaVA 使用 CLIP ViT 的倒数第二层输出而非最后一层？
5. 在训练 loss 计算中，为什么只对 assistant 的回答计算 loss？

### 进阶题

6. 对比 Flamingo、BLIP-2、LLaVA 三种架构的优缺点，分析 LLaVA 为何后来居上。
7. LLaVA-1.5 将 Projector 从单层线性换成两层 MLP 带来了显著提升，从函数逼近论的角度解释原因。
8. LLaVA 的视觉 token 有 576 个，如何优化推理速度？列出至少 3 种方案。
9. 设计一个改进方案：如何让 LLaVA 支持多张图片输入？

### 面试题

10. 面试官问："LLaVA 的 Projector 为什么这么简单就能 work？用一个线性层/MLP 就够了吗？"
11. 面试官问："如果让你从零设计一个 VLM，Vision Encoder 选 CLIP ViT 还是自己训练一个 ViT？为什么？"
12. 面试官问："LLaVA 的两阶段训练能合并为一个阶段吗？会有什么问题？"

---

## 八、产出要求

- [ ] 画出 LLaVA-1.5 的完整架构图（含 ViT / MLP Projector / LLM 三组件及维度标注）
- [ ] 画出三种视觉-语言连接范式对比图（Flamingo Cross-Attn / BLIP-2 Q-Former / LLaVA Projector）
- [ ] 写出 LLaVA 两阶段训练的参数冻结策略表（Stage 1 与 Stage 2 分别训练/冻结哪些参数）
- [ ] 写出 LLaVA-1.5 完整的维度流（从图像输入到 logits 输出，每步标注张量形状）
- [ ] 用数学公式写出线性 Projector 与 MLP Projector 的区别，解释非线性变换的必要性
- [ ] 列出 LLaVA-1.0 → LLaVA-1.5 的关键改进及各自对性能的影响
- [ ] 用自己的语言解释：为什么训练 loss 只对 assistant 回答部分计算？
- [ ] 对比 LLaVA 与 BLIP-2 / Flamingo 的架构优劣势（至少各 3 条）
- [ ] **闭卷写出 LLaVA 推理流程的伪代码（图像编码 → 投影 → 拼接 → 自回归生成）**
- [ ] 完成全部自检题

---

## 九、明日预告

Day 6 将进入 **手写 LLaVA 多模态实践**（本周重点之二）：
- 从零实现 MLP Projector（线性投影 + GELU + 线性投影）
- 实现多模态输入拼接逻辑（视觉 token + 文本 token 的序列拼接）
- 两阶段训练流程：Stage 1 预训练对齐 + Stage 2 指令微调
- 在小规模图文数据上完成端到端训练与推理
- 实现图文问答 Demo：输入一张图片和问题，输出自然语言回答

为 Day 7 多模态前沿总结与全周复盘做好实践基础。