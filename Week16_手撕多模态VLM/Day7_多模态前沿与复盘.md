# Day 7：多模态前沿与全周复盘 — 从 VLM 到统一多模态

> **目标**：了解多模态大模型在音频、视频等模态的扩展；理解统一多模态架构（Gemini、GPT-4o）的设计理念；掌握最新开源 VLM（Qwen2-VL、InternVL2、LLaVA-OneVision）的技术演进；串联全周 ViT → CLIP → LLaVA 知识链路；完成面试高频知识点自检；明确与第 17 周的衔接。

---

## 一、音频多模态：从 Whisper 到语音 LLM

### 1.1 Whisper — 语音领域的 "CLIP"

```
Whisper (OpenAI, 2022):
  大规模弱监督语音识别模型
  用 680K 小时互联网音频-文本对训练

核心思想:
  与 CLIP 类似 — 用海量弱标注数据训练通用模型

架构:
  Audio Encoder (Transformer) + Text Decoder (Transformer)
  
  音频 (mel spectrogram)
    │  Conv1d + 位置编码
    ▼
  Audio Encoder (Transformer Encoder)
    │  Cross-Attention
    ▼
  Text Decoder (Transformer Decoder)
    │  自回归生成
    ▼
  文本转录
```

### 1.2 Whisper 与 ViT 的类比

| 维度 | ViT（视觉） | Whisper（音频） |
|------|:---:|:---:|
| 输入 | 图像 patch 序列 | Mel Spectrogram 帧序列 |
| 预处理 | Conv2d patch 投影 | Conv1d 帧投影 |
| 编码器 | Transformer Encoder | Transformer Encoder |
| 预训练数据 | ImageNet / LAION | 680K 小时音频 |
| 下游任务 | 图像分类 / 检测 | 语音识别 / 翻译 |

### 1.3 语音多模态 LLM

将 Whisper 的音频编码器与 LLM 连接，形成语音理解模型：

```
方案 A: Whisper + Projector + LLM (类 LLaVA 架构)
  ┌──────────┐    ┌───────────┐    ┌──────┐
  │ Whisper  │──▶ │ Projector │──▶ │ LLM  │
  │ Encoder  │    │ (MLP)     │    │      │
  └──────────┘    └───────────┘    └──────┘
  
  代表: SALMONN (2023), Qwen-Audio (2023)

方案 B: 音频直接 tokenize
  音频 → 离散 token (Codec) → 与文本 token 拼接 → LLM
  
  代表: AudioPaLM (2023), SpeechGPT (2023)
```

### 1.4 代表工作

| 模型 | 音频编码器 | 桥接方式 | LLM | 能力 |
|------|:---:|:---:|:---:|------|
| SALMONN | Whisper + BEATs | Q-Former | Vicuna | 语音理解 + 音频理解 |
| Qwen-Audio | Whisper | Linear | Qwen-7B | 多语言语音理解 |
| AudioPaLM | USM | Token 拼接 | PaLM-2 | 语音翻译 + TTS |

---

## 二、视频多模态：从图像到时空理解

### 2.1 视频理解的核心挑战

```
图像 → 静态空间理解
  一张图: (H, W, 3)
  ViT 处理: N = (H/P)² 个 patch tokens

视频 → 动态时空理解
  T 帧视频: (T, H, W, 3)
  问题: T × N 个 token → 序列爆炸！

例如:
  1 张 336×336 图 → 576 个 token
  30 秒视频 (30fps, 900帧) → 900 × 576 = 518,400 个 token ← 远超 LLM 上下文长度！
```

### 2.2 视频 token 压缩策略

```
策略 1: 均匀采样帧
  900 帧 → 采样 8-16 帧 → 8×576 = 4,608 tokens
  优点: 简单
  缺点: 可能丢失关键帧

策略 2: 时间池化
  对连续帧的视觉特征做时间维度平均池化
  (T, N, D) → 时间池化 → (T', N, D), T' << T
  
策略 3: 可学习时间压缩 (Token Merging)
  用 Q-Former 或 Perceiver 压缩时间维度
  (T, N, D) → Temporal Q-Former → (K, D), K << T×N

策略 4: 动态分辨率
  关键帧用高分辨率, 非关键帧用低分辨率
  → 自适应 token 预算
```

### 2.3 代表工作

**Video-LLaMA (2023)**

```
Video-LLaMA 架构:

视频 (T, H, W, 3)
  │  每帧独立编码
  ▼
CLIP ViT → (T, N, D_v) → 时间位置编码
  │  Video Q-Former (可学习时间压缩)
  ▼
(K, D_l) → 拼接文本 → LLM
  
  额外: Audio Q-Former 处理音频通道
```

**LLaVA-Video / LLaVA-OneVision (2024)**

```
LLaVA-OneVision:
  统一图像 + 视频的 VLM

视频处理:
  1. 均匀采样 T 帧 (默认 32 帧)
  2. 每帧通过 CLIP ViT 编码
  3. 使用 AnyRes 动态分辨率
  4. 视觉 token 做 2×2 空间池化 → 压缩 4 倍
  5. 拼接所有帧的 token → 送入 LLM

关键创新:
  - 单阶段训练: 图像和视频数据混合训练
  - AnyRes: 支持任意分辨率
  - 空间池化: 控制 token 总数
```

**Qwen2-VL (2024)**

| 特性 | 值 |
|------|------|
| Vision Encoder | ViT-G (675M 参数) |
| 分辨率 | 动态 (最小 56×56 到最大 28×28 的 tile 数) |
| 视频支持 | 原生支持，时间维度 2 帧合并 |
| Rotary 3D PE | 将 RoPE 扩展到 (时间, 高度, 宽度) 三维 |
| 模型规模 | 2B / 7B / 72B |

### 2.4 视频 VLM 对比

| 模型 | 视频编码 | 时间压缩 | 最大帧数 | LLM |
|------|:---:|:---:|:---:|:---:|
| Video-LLaMA | ViT + Q-Former | Video Q-Former | 8 帧 | Vicuna |
| Video-ChatGPT | ViT | 时空池化 | 100 帧 | Vicuna |
| LLaVA-Video | ViT + AnyRes | 空间池化 2×2 | 32 帧 | Qwen2 |
| Qwen2-VL | ViT-G | 时间 2 帧合并 | 动态 | Qwen2 |
| InternVL2 | InternViT-6B | 动态分辨率 | 动态 | InternLM2 |

---

## 三、统一多模态模型

### 3.1 从"拼接"到"原生"多模态

```
第一代 VLM (2023): 拼接式
  CLIP ViT (冻结) + Projector + LLM
  → 视觉是"外挂"，LLM 本身不理解视觉
  → 代表: LLaVA, BLIP-2, MiniGPT-4

第二代 VLM (2024): 深度融合
  更大的 Vision Encoder + 更深的融合 + 联合训练
  → 视觉编码器和 LLM 联合优化
  → 代表: Qwen2-VL, InternVL2, LLaVA-OneVision

下一代 (2024+): 原生多模态
  单一模型处理文本 / 图像 / 音频 / 视频
  → 所有模态统一 tokenize，共享一个 Transformer
  → 代表: Gemini, GPT-4o
```

### 3.2 Gemini — Google 的统一多模态架构

```
Gemini (2023.12):
  
设计理念:
  "原生多模态" — 从预训练阶段就同时处理多种模态
  不是后期将视觉编码器"接"到 LLM 上

架构 (推测):
  ┌─────────────────────────────────────┐
  │         Unified Transformer         │
  │                                     │
  │  文本 token   ─┐                    │
  │  图像 token   ─┤→ 统一序列 → 生成   │
  │  音频 token   ─┤                    │
  │  视频 token   ─┘                    │
  └─────────────────────────────────────┘

关键特点:
  1. 多模态混合预训练: 文本+图像+音频+视频+代码
  2. 原生长上下文: 支持 1M+ token 上下文
  3. 多模态输出: 可以生成文本 + 图像（后续版本）
  4. 模型系列: Ultra / Pro / Flash / Nano
```

### 3.3 GPT-4o — OpenAI 的 "Omni" 模型

```
GPT-4o (2024.05):
  "o" = omni (全能)
  
与之前版本的区别:
  GPT-4V: 文本 LLM + 视觉编码器 (分离式)
  GPT-4o: 统一模型，原生处理文本/图像/音频

关键能力:
  1. 实时语音对话: 端到端，无需 ASR → LLM → TTS 流水线
  2. 音频理解: 可以感知语气、情感、环境声
  3. 图像生成: 原生支持（后续更新）
  4. 低延迟: 语音响应 ~320ms（接近人类对话延迟）

架构推测:
  所有模态 → 统一 tokenizer → 单一大 Transformer → 多模态输出
```

### 3.4 开源 vs 闭源 VLM 对比

| 维度 | LLaVA-1.5 | Qwen2-VL-72B | InternVL2 | Gemini Pro | GPT-4o |
|------|:---:|:---:|:---:|:---:|:---:|
| 开源 | 完全开源 | 完全开源 | 完全开源 | 闭源 | 闭源 |
| Vision Encoder | CLIP ViT-L | ViT-G (675M) | InternViT-6B | 未知 | 未知 |
| LLM | Vicuna-7/13B | Qwen2-72B | InternLM2-34B | 未知 | 未知 |
| 动态分辨率 | 否 | 是 | 是 | 是 | 是 |
| 视频理解 | 否 | 原生 | 原生 | 原生 | 原生 |
| 音频理解 | 否 | Qwen2-Audio | 否 | 原生 | 原生 |
| 多模态输出 | 仅文本 | 仅文本 | 仅文本 | 文本+图像 | 文本+图像+音频 |

---

## 四、开源 VLM 技术演进（2024-2025）

### 4.1 LLaVA 系列演进

```
LLaVA-1.0 (2023.04)
  CLIP ViT-L/14 + Linear + Vicuna
  分辨率: 224×224, Visual tokens: 256
  数据: 158K instruction
       │
       ▼
LLaVA-1.5 (2023.10)
  CLIP ViT-L/14 + MLP + Vicuna
  分辨率: 336×336, Visual tokens: 576
  数据: 665K (含学术 VQA)
  关键改进: MLP Projector + 高分辨率 + 更多数据
       │
       ▼
LLaVA-NeXT (2024.01)
  + AnyRes 动态分辨率
  + 更大 LLM (Mistral-7B, LLaMA3-8B, Yi-34B)
  + SGLang 推理优化
       │
       ▼
LLaVA-OneVision (2024.08)
  + 统一图像和视频
  + 单阶段训练
  + 空间池化压缩 token
  + Qwen2 作为 LLM
```

### 4.2 动态分辨率技术

LLaVA-1.5 的固定分辨率（336×336）的局限：高分辨率图像被压缩丢失细节，低分辨率图像被拉伸浪费 token。

```
AnyRes (LLaVA-NeXT):

原始图像 (1344×896)
  │
  ▼
切分为多个 336×336 的 tile + 一个低分辨率全局图
  ┌─────┬─────┬─────┬─────┐
  │tile1│tile2│tile3│tile4│  ← 4 个高分辨率 tile
  ├─────┼─────┼─────┼─────┤
  │tile5│tile6│tile7│tile8│
  └─────┴─────┴─────┴─────┘
         + 
  ┌─────────────────┐
  │  全局缩略图      │  ← 1 个低分辨率全局图
  └─────────────────┘

每个 tile → ViT → 576 tokens
全局图     → ViT → 576 tokens
总计: (8+1) × 576 = 5,184 tokens

优势:
  - 保留高分辨率细节（文字、小物体）
  - 全局图保持整体理解
  - 适配任意宽高比
```

### 4.3 关键技术趋势

| 趋势 | 说明 | 代表工作 |
|------|------|---------|
| 更大 Vision Encoder | 从 CLIP ViT-L (304M) 到 ViT-G (1.8B) 甚至 InternViT-6B | InternVL2, Qwen2-VL |
| 动态分辨率 | 自适应处理不同大小的图像 | LLaVA-NeXT, Qwen2-VL |
| 视觉 token 压缩 | 减少视觉 token 数量以支持更多图/更长视频 | LLaVA-OneVision, Qwen2-VL |
| Vision Encoder 可训练 | 解冻 ViT 做联合训练 | InternVL2, Qwen2-VL |
| 多任务训练 | 视觉问答 + OCR + Grounding + 检测 | Qwen2-VL, CogAgent |
| 图像/视频统一 | 同一模型同时处理图像和视频 | LLaVA-OneVision, Qwen2-VL |

---

## 五、多模态 Agent 与工具使用

### 5.1 VLM 驱动的多模态 Agent

```
传统 Agent (W8):
  用户指令 → LLM → 工具调用 → 结果 → LLM → 回答
  局限: 只能理解文本

多模态 Agent:
  用户指令 + 图像/视频 → VLM → 工具调用 → 结果 → VLM → 回答
  能力: 看图理解 → 决策 → 行动

示例场景:
  "看看这张截图，帮我点击登录按钮"
  → VLM 理解截图布局 → 定位按钮坐标 → 调用点击工具
```

### 5.2 代表工作

| 模型 | 能力 | 核心技术 |
|------|------|---------|
| CogAgent | GUI 操作 | 高分辨率 ViT + Grounding |
| AppAgent | App 操作 | 截图理解 + 操作生成 |
| ScreenAI | 屏幕理解 | PaLI 架构 + 屏幕预训练 |
| GPT-4o + Tools | 通用多模态 Agent | 原生多模态 + Function Calling |

### 5.3 多模态 RAG

```
传统 RAG (W8):
  查询 → 文本检索 → 文本上下文 → LLM → 回答

多模态 RAG:
  查询 (+ 图像) → 图文检索 (CLIP) → 图文上下文 → VLM → 回答

应用:
  - 产品图搜索 + 问答
  - 医学影像检索 + 诊断
  - 文档理解 (图表 + 文字)
```

---

## 六、全周复盘

### 6.1 本周知识地图

```
Day 1: 多模态大模型论文精读
  ✅ ViT 核心创新: 图像 → patch 序列 → Transformer
  ✅ CLIP 核心创新: 图文对比学习 (InfoNCE) → Zero-Shot
  ✅ LLaVA 核心创新: CLIP ViT + Projector + LLM → Visual Instruction Tuning
  ✅ 三篇论文的技术演进脉络
      │
      ▼
Day 2: ViT 与 CLIP 架构详解
  ✅ Patch Embedding 数学: Conv2d 等价 + CLS + PosEmb
  ✅ ViT Encoder Block: Pre-LN + MSA + FFN + Residual
  ✅ CLIP 双塔架构: Image Encoder + Text Encoder + 共享嵌入空间
  ✅ InfoNCE Loss 数学推导 + 对称 Cross-Entropy
      │
      ▼
Day 3: 手写 ViT + CLIP (实践核心!)
  ✅ PatchEmbedding: Conv2d + CLS + Position
  ✅ ViTBlock: LayerNorm → MultiheadAttention → GELU FFN
  ✅ 完整 ViT 模型 + 分类头
  ✅ CLIP Image Encoder + Text Encoder
  ✅ InfoNCE Loss 实现 (4 行代码!)
  ✅ 完整 CLIP 模型 + HuggingFace 权重对比
      │
      ▼
Day 4: CLIP 对比学习与训练深入
  ✅ NCE → InfoNCE → 互信息下界
  ✅ 温度参数 τ 的梯度分析 → Hard Negative Mining
  ✅ Batch Size 与互信息估计精度
  ✅ CLIP 训练策略: 32768 batch + AllGather
  ✅ CLIP 能力边界与局限
      │
      ▼
Day 5: LLaVA 多模态大模型架构
  ✅ 三种视觉-语言连接范式: Flamingo / BLIP-2 / LLaVA
  ✅ LLaVA 三组件: Vision Encoder + Projector + LLM
  ✅ 两阶段训练: Stage 1 对齐 + Stage 2 指令微调
  ✅ LLaVA-1.5 改进: MLP Projector + 336 分辨率
  ✅ 完整维度流追踪
      │
      ▼
Day 6: 手写 LLaVA 多模态实践 (实践核心!)
  ✅ VisionEncoder: 复用 ViT, 取倒数第二层, 去掉 CLS
  ✅ MLP Projector: Linear → GELU → Linear
  ✅ 多模态输入拼接: <image> 替换 + Loss mask
  ✅ LLM Decoder: RMSNorm + CausalAttn + SwiGLU
  ✅ 完整 LLaVA: 三组件端到端 + 参数冻结策略
  ✅ 两阶段训练: Stage 1 Projector only + Stage 2 Projector + LLM
  ✅ 推理 Demo: 图文问答完整流程
      │
      ▼
Day 7: 多模态前沿与复盘 (今天)
  ✅ 音频多模态: Whisper → 语音 LLM
  ✅ 视频多模态: 帧采样 + token 压缩 + 时空理解
  ✅ 统一多模态: Gemini / GPT-4o 原生多模态架构
  ✅ 开源 VLM 演进: LLaVA-NeXT → LLaVA-OneVision → Qwen2-VL
  ✅ 全周复盘与面试自检
```

### 6.2 面试高频知识点自检

```
面试题 1: "画出 ViT 的架构图，解释 Patch Embedding"
  → Day 1-2: 图像切分为 P×P patch → Conv2d 投影 → + CLS + PosEmb → Transformer
  → 闭卷写出公式: z_0 = [x_cls; x_p^1 E; ...; x_p^N E] + E_pos

面试题 2: "手写 InfoNCE Loss"
  → Day 3-4: 4 行代码
  logits = (img_feat @ txt_feat.T) / tau
  labels = torch.arange(len(logits))
  loss = (CE(logits, labels) + CE(logits.T, labels)) / 2

面试题 3: "CLIP 的 Zero-Shot 分类怎么做？"
  → Day 2: 将类别名变成 "a photo of a {class}" prompt
  → 计算图像与所有 prompt 的余弦相似度 → 取最大值

面试题 4: "CLIP 的温度参数 τ 有什么作用？"
  → Day 4: 控制 softmax 锐度
  → τ 小: Hard negative mining, 集中在难负例
  → τ 大: 均匀分布, 所有负例等权
  → CLIP 让 τ 可学习, 初始 0.07

面试题 5: "LLaVA 的架构是什么？画出来"
  → Day 5: CLIP ViT (冻结) → MLP Projector → LLM (Vicuna)
  → 视觉 token 和文本 token 在序列维度拼接

面试题 6: "LLaVA 的两阶段训练各训练什么参数？"
  → Day 5-6:
  Stage 1: 仅 Projector (ViT + LLM 冻结), lr=2e-3, 558K 图文对
  Stage 2: Projector + LLM (ViT 冻结), lr=2e-5, 665K 指令数据

面试题 7: "LLaVA 的 Projector 为什么简单就能 work？"
  → Day 5: CLIP 已完成视觉-语言对齐, Projector 只做维度适配
  → LLM 表示空间强大, 可吸收不完美投影
  → 两阶段训练让 Projector 在 Stage 1 专注对齐

面试题 8: "手写 LLaVA 的推理流程伪代码"
  → Day 6: 
  visual_feats = vit(image)           # (1, N, D_v)
  visual_tokens = projector(visual_feats)  # (1, N, D_l)
  text_embeds = embed(question)       # (1, T, D_l)
  input_embeds = cat([visual_tokens, text_embeds])
  output = llm.generate(input_embeds)

面试题 9: "LLaVA vs BLIP-2 vs Flamingo 的区别？"
  → Day 5:
  Flamingo: Cross-Attn 插入 LLM, 修改架构, 复杂
  BLIP-2: Q-Former 桥接, 压缩到 32 token, 中等复杂
  LLaVA: MLP 投影, 保留全部 token, 简单高效

面试题 10: "如果让你设计一个 VLM, 你会怎么做？"
  → Day 7: 参考 Qwen2-VL / InternVL2
  → 大 ViT (可训练) + MLP Projector + 强 LLM
  → 动态分辨率 + token 压缩
  → 图文视频混合训练
```

### 6.3 本周产出清单

完成本周后，你应该拥有以下可复用资产：

- [ ] **ViT 完整实现**：PatchEmbedding + ViTBlock + ViT，可加载 HuggingFace 权重
- [ ] **CLIP 完整实现**：Image Encoder + Text Encoder + InfoNCE，可做 Zero-Shot 分类
- [ ] **InfoNCE Loss 4 行代码**：面试闭卷手写
- [ ] **LLaVA 完整实现**：VisionEncoder + MLP Projector + LLM + 两阶段训练 + 推理
- [ ] **多模态拼接逻辑**：`<image>` 占位符替换 + Loss mask
- [ ] **各架构维度流追踪**：ViT / CLIP / LLaVA 从输入到输出的完整张量形状
- [ ] **VLM 对比表**：Flamingo / BLIP-2 / LLaVA / Qwen2-VL / InternVL2 的架构对比
- [ ] **多模态前沿笔记**：音频/视频/统一多模态的技术脉络

---

## 七、与后续周次的衔接

```
Week 16: 手撕多模态 VLM
  → ViT → CLIP → LLaVA 完整实现 + 多模态前沿
       │
       ▼
Week 17: o1 推理与思维链 (预告)
  ├─ Chain-of-Thought (CoT) 推理
  ├─ o1 的推理时计算 (test-time compute)
  ├─ 推理搜索策略: Best-of-N / Tree Search / MCTS
  ├─ Process Reward Model (PRM)
  ├─ Self-Consistency / Majority Voting
  └─ 多模态推理: 视觉 CoT

W16 → W17 的关系:
  W16 多模态理解能力 → W17 推理能力
  
  VLM 生成回答 (W16) → CoT 让回答更准确 (W17)
  LLaVA 训练 (W16)   → RLHF + PRM 对齐 (W17)
  视觉理解 (W16)     → 视觉推理 (W17)
  
  W16 解决了 "看懂图"
  W17 解决了 "想清楚"
```

---

## 八、自检题

### 基础题

1. Whisper 的架构与 ViT 有什么相似之处？音频信号的 "patch" 是什么？
2. 视频理解的核心挑战是什么？列出 3 种视频 token 压缩策略。
3. Gemini 的 "原生多模态" 与 LLaVA 的 "拼接式多模态" 有什么本质区别？
4. LLaVA-NeXT 的 AnyRes 是如何处理不同分辨率图像的？
5. 画出本周 ViT → CLIP → LLaVA 的知识演进图。

### 进阶题

6. 如果一个 30 秒视频（30fps）需要用 LLaVA 处理，你会如何设计 token 压缩策略？计算压缩前后的 token 数量。
7. 对比 Qwen2-VL 和 LLaVA-OneVision 的视频处理方式，分析各自优劣。
8. 为什么统一多模态模型（Gemini/GPT-4o）要从预训练阶段就引入多模态，而不是像 LLaVA 那样后期对齐？
9. 多模态 RAG 系统中，CLIP 可以充当什么角色？设计一个多模态 RAG 的流程。

### 面试题

10. 面试官问："从 ViT 到 CLIP 到 LLaVA，每一步的核心创新是什么？用一句话分别概括。"
11. 面试官问："让你从零搭建一个多模态大模型，你会选什么 Vision Encoder？什么 LLM？什么桥接方式？为什么？"
12. 面试官问："LLaVA 的视觉 token 有 576 个，占据了大量上下文窗口。如何在保持性能的前提下减少视觉 token？"

---

## 九、产出要求

- [ ] 写出 Whisper 与 ViT 的架构类比表
- [ ] 列出视频 VLM 的 3 种 token 压缩策略及优缺点
- [ ] 画出 "拼接式多模态 → 深度融合 → 原生多模态" 的演进图
- [ ] 对比 LLaVA-1.0 → 1.5 → NeXT → OneVision 的技术演进
- [ ] 写出 Qwen2-VL 的关键技术创新（动态分辨率 + 3D RoPE + 视频支持）
- [ ] 完成全周 10 个面试高频知识点的闭卷自检
- [ ] 画出本周完整知识图谱（ViT → CLIP → LLaVA → 前沿）
- [ ] **闭卷手写**: InfoNCE Loss（4 行）+ LLaVA 推理流程（5 步）+ MLP Projector（3 行）
- [ ] 明确 W17 o1 推理将在 W16 的基础上扩展什么内容