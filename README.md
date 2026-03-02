# 手撕 LLM

<p align="center">
  <img src="https://img.shields.io/badge/周期-18~22_周-green" alt="duration" />
  <img src="https://img.shields.io/badge/语言-Python-blue" alt="language" />
  <img src="https://img.shields.io/badge/更新中-orange" alt="status" />
</p>

大预言模型教你从零手写大语言模型核心代码，系统掌握 LLM 全栈技术。

## 这是什么

一份面向有深度学习基础的研究者的 **LLM 硬核学习仓库**。不是调包教程，而是逐行手写每个核心模块——从 Transformer 的 Multi-Head Attention，到 RLHF 的 PPO 训练循环，再到分布式训练与多模态推理。

## 覆盖内容

- **基础架构**：Transformer / GPT / LLaMA（RoPE、RMSNorm、SwiGLU、KV Cache）
- **高效微调**：Instruction Tuning / LoRA / QLoRA / 中文适配 / 数据工程
- **应用落地**：Agent / RAG / 多轮对话部署
- **对齐技术**：RL 基础（DQN → PPO）→ RLHF → DPO → GRPO → DeepSeek-R1
- **系统优化**：FlashAttention / 推理量化 / vLLM / 分布式训练（ZeRO / TP / PP）/ MoE
- **前沿方向**：多模态 VLM（ViT → CLIP → LLaVA）/ o1 推理（MCTS / PRM / Scaling Test Time）

## 快速开始

1. 阅读 [`LLM_学习计划.md`](LLM_学习计划.md) 了解完整路线
2. 按周进入对应文件夹，跟随笔记与 Notebook 实践
3. 每周末对照「关键检查点」自测

## License

MIT
