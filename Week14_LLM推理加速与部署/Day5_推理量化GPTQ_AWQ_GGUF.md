# Day 5：推理量化 GPTQ / AWQ / GGUF 深度原理 — 从入门到算法推导

> **目标**：从 W6 Day5"了解定位"进阶到"理解算法原理"；掌握 PTQ（后训练量化）的基础概念——校准集、逐层量化、量化误差分析；深入推导 GPTQ 的核心算法——从 OBQ / OBS 到 Hessian 逆 + 贪心量化 + Cholesky 分解；理解 AWQ 的激活感知量化策略——为什么保护 1% 的显著权重通道就够了；了解 GGUF 的 CPU 推理优化与混合量化方案；建立清晰的量化方案选型指南。

---

## 一、PTQ（后训练量化）基础

### 1.1 什么是后训练量化

```
Post-Training Quantization (PTQ):
  训练完成后，不改变模型结构，直接将权重从高精度降到低精度
  
  与 QAT (Quantization-Aware Training) 的区别:
    QAT: 训练过程中模拟量化 → 模型学会适应量化误差
         → 精度更高，但需要完整训练数据和训练过程
    PTQ: 训练后直接量化 → 只需少量校准数据
         → 方便快捷，但精度可能略低
  
  LLM 场景为什么用 PTQ:
    1. 大模型训练成本极高，重新训练不现实
    2. 只需 128~1024 个校准样本
    3. 量化后精度损失小 (大模型对量化更鲁棒)
```

### 1.2 均匀量化基础

对称量化（Symmetric Quantization）：

$$
x_q = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x|)}{2^{b-1} - 1}
$$

$$
\hat{x} = x_q \times s
$$

非对称量化（Asymmetric Quantization）：

$$
x_q = \text{round}\left(\frac{x - z}{s}\right), \quad s = \frac{\max(x) - \min(x)}{2^b - 1}, \quad z = \min(x)
$$

```
INT4 对称量化示例:

原始 FP16 权重: [0.12, -0.35, 0.78, -0.92, 0.05, 0.63]
abs_max = 0.92
scale = 0.92 / 7 = 0.1314  (INT4 范围: -8 ~ 7)

量化:
  0.12 / 0.1314 = 0.91  → round → 1   → 反量化: 1 × 0.1314 = 0.1314
 -0.35 / 0.1314 = -2.66 → round → -3  → 反量化: -3 × 0.1314 = -0.3943
  0.78 / 0.1314 = 5.94  → round → 6   → 反量化: 6 × 0.1314 = 0.7886
 -0.92 / 0.1314 = -7.00 → round → -7  → 反量化: -7 × 0.1314 = -0.9200
  0.05 / 0.1314 = 0.38  → round → 0   → 反量化: 0 × 0.1314 = 0.0000 ← 精度损失!
  0.63 / 0.1314 = 4.80  → round → 5   → 反量化: 5 × 0.1314 = 0.6571

量化误差:
  [0.0114, 0.0443, 0.0086, 0.0000, 0.0500, 0.0271]
  → 最大误差 0.05，出现在小权重被量化到 0 时
```

### 1.3 分组量化（Group Quantization）

```
问题: 全层共享一个 scale 和 zero_point
  → 异常值 (outlier) 会拉大 scale → 降低其他权重的精度

分组量化: 将权重按组划分, 每组独立量化

权重矩阵 W ∈ R^{m × n}, group_size = 128:
  分为 n/128 组, 每组 128 个权重共享 scale 和 zero_point
  
  ┌──────┬──────┬──────┬──────┐
  │Group0│Group1│Group2│ ...  │ ← 行中的每 128 列一组
  │s0,z0 │s1,z1 │s2,z2 │      │ ← 每组独立的量化参数
  └──────┴──────┴──────┴──────┘

额外开销:
  每组需存 1 个 scale (FP16) + 1 个 zero_point (INT4)
  = 2.5 bytes / group
  
  对于 group_size=128, INT4:
    权重: 128 × 0.5 bytes = 64 bytes
    量化参数: 2.5 bytes
    有效位宽: (64 + 2.5) / 128 = 0.52 bytes ≈ 4.16 bits
    → 开销很小!
```

### 1.4 校准集的作用

```
为什么 PTQ 需要校准数据?

简单量化 (RTN — Round-to-Nearest):
  直接四舍五入，不需要数据
  → 简单但精度差

高级量化 (GPTQ / AWQ):
  需要校准数据来:
    1. GPTQ: 计算 Hessian 矩阵 H = XX^T (X 是激活值)
       → 衡量每个权重对输出的影响
    2. AWQ: 统计激活值分布 → 找到重要通道
    3. 逐层量化: 用校准数据前向传播 → 得到每层的输入
       → 优化每层的量化使得输出误差最小

校准集要求:
  - 通常 128~1024 个样本
  - 来自训练数据分布 (如 C4, WikiText)
  - 不需要标签
  - 量化质量对校准集不太敏感
```

---

## 二、量化误差分析框架

### 2.1 逐层量化目标

对于第 $l$ 层，权重 $W$ 和输入激活 $X$（来自校准数据），量化目标：

$$
\hat{W} = \arg\min_{\hat{W}} \|WX - \hat{W}X\|_F^2
$$

即：找到量化后的 $\hat{W}$，使得该层的输出误差最小。

### 2.2 为什么逐层优化

```
整体优化 vs 逐层优化:

整体优化:
  min_{所有层的 Ŵ} || f(x; W) - f(x; Ŵ) ||
  → 涉及整个网络的非线性映射 → 极难优化

逐层优化:
  对每层独立: min_{Ŵ_l} || W_l X_l - Ŵ_l X_l ||_F²
  → 线性问题 → 有解析解!
  → 逐层贪心: 从第 1 层到最后一层依次量化
  → 量化后更新后续层的输入 (用量化后的输出作为下一层输入)

实践中: 逐层贪心 + 高质量单层优化 = 很好的整体效果
```

### 2.3 RTN vs 高级量化的数学对比

```
Round-to-Nearest (RTN):
  Ŵ_ij = Q(W_ij)  (独立量化每个元素)
  
  量化误差: Σ_ij (W_ij - Ŵ_ij)² × ||X_j||²
  → 权重误差被激活值放大
  → 某些通道的激活值很大 → 误差被放大

GPTQ / AWQ:
  考虑权重之间的相关性
  → 量化一个权重时, 调整其他权重来补偿
  → 或者对重要权重做特殊保护
  → 整体输出误差更小
```

---

## 三、GPTQ 算法深度推导

### 3.1 从 OBS 到 GPTQ

```
Optimal Brain Surgeon (OBS) — 1993:
  目标: 在神经网络中删除 (量化) 一个权重, 同时调整其他权重
        使得输出误差最小
  
  关键工具: Hessian 矩阵 H = ∂²L/∂W² ≈ XX^T
  
  当量化权重 w_q (量化误差 δ_q = w_q - Q(w_q)):
    其他权重的最优调整:
      δ_w = -δ_q × H^{-1}[:, q] / H^{-1}[q, q]
    
    输出误差增量:
      δ_L = δ_q² / (2 × H^{-1}[q, q])
    
    → H^{-1}[q, q] 越大 → 量化该权重的代价越小

Optimal Brain Quantization (OBQ) — 2022:
  将 OBS 应用到量化场景:
    逐个权重量化, 每次选择代价最小的权重
    量化后调整剩余权重补偿
  
  复杂度: O(d³) per row → 大模型不可行

GPTQ — 2023:
  OBQ 的高效版本:
    1. 不再选择"最优"量化顺序 → 按列顺序 (或按激活排序)
    2. 利用 Cholesky 分解高效更新 H^{-1}
    → 复杂度: O(d² × n) per row → 大模型可行!
```

### 3.2 GPTQ 核心公式推导

设权重矩阵 $W \in \mathbb{R}^{m \times n}$，校准数据的 Hessian $H = X X^T \in \mathbb{R}^{n \times n}$。

逐列量化误差目标：

$$
\min_{\hat{W}} \|WX - \hat{W}X\|_F^2 = \min_{\hat{W}} \text{tr}[(W - \hat{W}) H (W - \hat{W})^T]
$$

当量化第 $q$ 列时，最优的误差补偿（调整未量化的列）：

$$
\delta_{\text{row}} = -\frac{w_q - \hat{w}_q}{[H^{-1}]_{qq}} \cdot (H^{-1})_{:,q}
$$

输出误差增量：

$$
E_q = \frac{(w_q - \hat{w}_q)^2}{2 [H^{-1}]_{qq}}
$$

### 3.3 GPTQ 完整算法

```
算法: GPTQ

输入: 权重矩阵 W ∈ R^{m×n}, Hessian H = XX^T ∈ R^{n×n}
参数: 量化函数 Q(·), 分组大小 group_size, dampening λ

预处理:
  1. H ← H + λI   (dampening, 防止 H 奇异)
  2. H^{-1} ← Cholesky 分解后求逆
  3. 可选: 按 diag(H) 对列排序 (desc_act=True)

主循环 (逐列):
  for q = 0 to n-1:
    // 当前列的量化
    w_q = W[:, q]                        ← 第 q 列 (m 维向量)
    ŵ_q = Q(w_q)                         ← 量化
    
    // 量化误差
    δ_q = w_q - ŵ_q                      ← m 维误差向量
    
    // 写入量化结果
    W[:, q] = ŵ_q
    
    // 用 Hessian 信息补偿剩余列
    W[:, q+1:] += δ_q ⊗ H^{-1}[q, q+1:] / H^{-1}[q, q]
    //  ↑ 外积: m × (n-q-1) 矩阵

    // 更新 H^{-1} (Cholesky 行消除)
    H^{-1}[q+1:, q+1:] -= H^{-1}[q+1:, q] ⊗ H^{-1}[q, q+1:] / H^{-1}[q, q]

输出: 量化后的 Ŵ
```

### 3.4 Cholesky 分解的作用

```
为什么需要 Cholesky 分解?

直接计算 H^{-1}:
  H 是 n×n 矩阵 → 求逆 O(n³)
  每量化一列后需要更新 H^{-1} → 朴素更新也是 O(n²) per step
  
Cholesky 分解的优势:
  H^{-1} = (LL^T)^{-1} where L 是下三角
  
  逐列量化 = 逐行消除 Cholesky 因子
  → 每步只需 O(n) 更新
  → 总复杂度: O(n² × m) per layer (m 行, 每行 n 列)

实际加速:
  GPTQ 用 "lazy batch" 策略:
    每 128 列为一批, 批内逐列量化
    批间更新权重矩阵
    → 更好的 GPU 利用率
```

### 3.5 GPTQ 的 `desc_act` 选项

```
desc_act (descending activation order):
  按激活值大小对列排序 (从大到小) → 先量化重要列
  
为什么有帮助:
  重要列 (激活大) 的量化误差影响更大
  → 先量化重要列时, 有更多未量化列可以补偿
  → 精度更好

  不排序: 量化列 0 → 列 1 → ... → 列 n
  排序:   量化最重要列 → 次重要 → ... → 最不重要

代价:
  排序后权重列顺序改变 → 推理时需要按排序反向映射
  → 推理稍慢 (< 5%)
  
  通常精度提升 > 速度损失 → 推荐开启
```

---

## 四、AWQ 算法深度解析

### 4.1 从 W6 Day5 到深度理解

W6 Day5 介绍了 AWQ 的核心洞察：**不是所有权重同等重要**，保护激活值大的通道。

现在我们深入理解其数学原理。

### 4.2 权重重要性的量化分析

对于线性层 $y = Wx$，量化误差：

$$
\Delta y = (W - \hat{W})x
$$

第 $c$ 个输入通道对误差的贡献：

$$
\|\Delta y_c\| \approx |W_{:,c} - \hat{W}_{:,c}| \cdot |x_c|
$$

**关键洞察**：即使 $|W_{:,c} - \hat{W}_{:,c}|$ 相同，如果 $|x_c|$ 大 100 倍，误差就大 100 倍。

```
实验观察 (AWQ 论文):

权重通道按激活值大小排序:
  Top 1% 通道:   平均激活值 = 50.0    ← 显著通道
  Middle 98%:    平均激活值 = 1.0     ← 普通通道
  Bottom 1%:     平均激活值 = 0.01    ← 不活跃通道

如果对所有通道均匀量化:
  Top 1% 的量化误差被放大 50× → 输出误差主要来自这 1%
  
如果保护 Top 1% (不量化或高精度):
  PPL 恢复到接近 FP16!

结论:
  "保护 1% 的显著通道" 就能显著提升量化质量
```

### 4.3 AWQ 的缩放方法

AWQ 不是简单地跳过重要权重不量化，而是通过**缩放**保护它们：

$$
y = Wx = (W \cdot \text{diag}(\mathbf{s})) \cdot (\text{diag}(\mathbf{s})^{-1} \cdot x)
$$

对权重缩放后量化：

$$
\hat{y} = Q(W \cdot \text{diag}(\mathbf{s})) \cdot (\text{diag}(\mathbf{s})^{-1} \cdot x)
$$

```
为什么缩放有效?

通道 c 的权重 w_c = 0.01, 激活 x_c = 100:
  正确输出: 0.01 × 100 = 1.0
  
  直接量化: Q(0.01) = 0 (INT4 精度不够!) 
  → 误差 = 1.0

  缩放后 (s_c = 10):
    w_c' = 0.01 × 10 = 0.1
    x_c' = 100 / 10 = 10
    Q(0.1) = 0.1 (INT4 能表示)
    → 0.1 × 10 = 1.0 → 误差 ≈ 0!

本质:
  放大权重 → 在 INT4 的量化网格中获得更高精度
  缩小激活 → 数学等价性保持
  → 重要通道得到更好的量化保护
```

### 4.4 最优缩放因子搜索

AWQ 使用基于激活值统计的启发式搜索：

$$
s_c^* = \left(\frac{\overline{|x_c|}}{\max_c \overline{|x_c|}}\right)^\alpha
$$

其中 $\overline{|x_c|}$ 是通道 $c$ 在校准集上的平均激活值绝对值，$\alpha \in [0, 1]$ 是搜索的超参数。

```
搜索过程:

for α in [0, 0.1, 0.2, ..., 0.9, 1.0]:
    s = (mean_activation / max_activation) ^ α
    W_scaled = W × diag(s)
    W_quantized = Q(W_scaled)
    error = || W × X - W_quantized × diag(s^{-1}) × X ||
    记录最小 error 对应的 α

最优 α 通常在 0.5 附近

α = 0: s = 1 (不缩放, 退化为 RTN)
α = 1: s 正比于激活值 (最大保护)
0 < α < 1: 平衡保护程度和量化网格利用率
```

### 4.5 AWQ vs GPTQ 深度对比

| 维度 | GPTQ | AWQ |
|------|------|-----|
| **核心方法** | 逐列量化 + Hessian 补偿 | 激活感知缩放 + 均匀量化 |
| **数学基础** | OBS / OBQ 框架 | 通道重要性分析 |
| **优化目标** | 每列量化误差最小化 | 整体缩放后量化误差最小化 |
| **需要 Hessian** | 是 ($H = XX^T$) | 否（只需激活值统计） |
| **量化过程** | 逐列顺序，列间补偿 | 搜索缩放因子，然后均匀量化 |
| **量化速度** | 中等（需要 Hessian + 逐列） | **快**（无逐列迭代） |
| **推理核** | 通用 GEMM | 有专门优化的推理核 |
| **PPL 精度** | 优秀 | 优秀（略优） |
| **生态成熟度** | 高（AutoGPTQ，广泛使用） | 高（AutoAWQ，vLLM 首选） |

```
选择建议:

优先 AWQ:
  - 部署在 vLLM / TGI 上
  - 需要快速量化
  - 追求推理速度

优先 GPTQ:
  - 需要 desc_act 提升极致精度
  - 已有成熟的 GPTQ 工作流
  - 需要与旧系统兼容
```

---

## 五、GGUF 深度解析

### 5.1 GGUF 格式设计

```
GGUF (GPT-Generated Unified Format):
  llama.cpp 定义的模型存储格式

文件结构:
  ┌──────────────────────────┐
  │ Magic Number (0x46475547) │
  │ Version                   │
  │ Metadata (JSON-like)      │
  │   - model architecture    │
  │   - quantization type     │
  │   - vocab, special tokens │
  │   - hyperparameters       │
  ├──────────────────────────┤
  │ Tensor Data               │
  │   - 量化后的权重          │
  │   - 量化参数 (scale, etc) │
  └──────────────────────────┘

优势:
  - 自包含: 一个文件包含模型+tokenizer+配置
  - 跨平台: Mac/Windows/Linux 通用
  - 版本兼容: 向后兼容设计
```

### 5.2 K-Quant 混合量化

```
K-Quant (K-means inspired quantization):
  不同层使用不同的量化精度
  
  重要性分析:
    Attention 的 Q, K 投影 → 较高精度 (5-6 bit)
    FFN 的中间层 → 较低精度 (3-4 bit)
    Embedding 和 Output 层 → 较高精度 (6-8 bit)

Q4_K_M 示例:
  "Q": Quantized
  "4": 主要量化位宽 4-bit
  "K": K-quant 混合策略
  "M": Medium 混合级别 (S/M/L 三档)

  M (Medium):
    约 75% 的层使用 Q4_K (4.5 bit 含 scale)
    约 25% 的层使用 Q6_K (6 bit)
    加权平均 ≈ 4.85 bits/param

混合策略对比:
  Q4_K_S (Small):  更多层用 Q4 → 更小, 精度稍低
  Q4_K_M (Medium): 平衡 → 最常用
  Q4_K_L (Large):  更多层用 Q6 → 更大, 精度更高
```

### 5.3 GGUF 量化级别完整对比

| 类型 | 有效位宽 | 方法 | 7B 模型大小 | PPL 损失 | 适用场景 |
|------|---------|------|-----------|---------|---------|
| Q2_K | 2.6 | K-quant | ~2.8 GB | 显著 | 极端压缩 |
| Q3_K_S | 3.4 | K-quant small | ~2.9 GB | 较大 | 内存极限 |
| Q3_K_M | 3.9 | K-quant medium | ~3.3 GB | 中等 | 低配设备 |
| Q4_0 | 4.0 | 均匀量化 | ~3.8 GB | 中等 | 兼容性 |
| Q4_K_S | 4.6 | K-quant small | ~3.9 GB | 较小 | 常用 |
| **Q4_K_M** | **4.8** | **K-quant medium** | **~4.1 GB** | **较小** | **推荐** |
| Q5_K_S | 5.5 | K-quant small | ~4.6 GB | 小 | 质量优先 |
| Q5_K_M | 5.7 | K-quant medium | ~4.8 GB | 很小 | 质量优先 |
| Q6_K | 6.6 | K-quant | ~5.5 GB | 极小 | 高质量 |
| Q8_0 | 8.0 | 均匀量化 | ~7.2 GB | 几乎无 | 接近无损 |
| F16 | 16.0 | 无量化 | ~13 GB | 无 | 基准 |

### 5.4 llama.cpp 的 CPU 优化

```
llama.cpp 的 CPU 推理优化技术:

1. SIMD 向量化 (AVX2 / ARM NEON):
   量化矩阵乘法用 SIMD 指令加速
   → INT4/INT8 乘法比 FP16 快数倍

2. 内存映射 (mmap):
   不需要把整个模型加载到内存
   → 操作系统按需加载页面
   → 启动更快, 多模型共享物理内存

3. 量化格式优化:
   数据布局针对 CPU cache 行优化
   → 减少 cache miss

4. Metal / CUDA / Vulkan 后端:
   虽然主打 CPU, 也支持 GPU 加速
   → Mac 用 Metal, NVIDIA 用 CUDA

5. 量化 KV Cache:
   KV Cache 也可以用低精度存储
   → 进一步降低内存占用
```

---

## 六、量化方案选型指南

### 6.1 决策流程图

```
你的部署场景是什么？
  │
  ├─ 企业级 GPU 推理服务
  │    │
  │    ├─ 追求最高吞吐 → AWQ + vLLM
  │    │     quantization="awq", dtype="half"
  │    │
  │    ├─ 追求极致精度 → GPTQ (desc_act=True) + vLLM
  │    │
  │    └─ 显存紧张 → AWQ-4bit + KV Cache 量化
  │
  ├─ 消费级 GPU (3090/4090)
  │    │
  │    ├─ 运行 7B 模型 → AWQ-4bit 或 GPTQ-4bit
  │    │    显存: ~4 GB + KV Cache
  │    │
  │    └─ 运行 13B+ 模型 → AWQ-4bit + 量化 KV
  │         可能需要 offload 到 CPU
  │
  ├─ CPU 推理 (Mac / 无 GPU)
  │    │
  │    ├─ 8GB 内存 → Q3_K_M 或 Q4_K_S
  │    ├─ 16GB 内存 → Q4_K_M (推荐)
  │    └─ 32GB+ 内存 → Q5_K_M 或 Q6_K
  │
  └─ 边缘设备 / 手机
       │
       └─ Q2_K 或 Q3_K_S → 极限压缩
```

### 6.2 精度-大小-速度对比（LLaMA-2-7B）

| 方法 | 大小 | WikiText-2 PPL | GPU 推理速度 | CPU 推理速度 |
|------|------|---------------|-------------|-------------|
| FP16 | 13.0 GB | 5.47 | 基准 | N/A |
| GPTQ-4bit | 3.6 GB | 5.63 | 快 | 不支持 |
| GPTQ-4bit (desc_act) | 3.6 GB | 5.57 | 快（略慢） | 不支持 |
| AWQ-4bit | 3.6 GB | 5.60 | **最快** | 不支持 |
| GGUF Q4_K_M | 4.1 GB | ~5.70 | 支持 | **中等** |
| GGUF Q6_K | 5.5 GB | ~5.52 | 支持 | 中等 |
| GGUF Q8_0 | 7.2 GB | ~5.48 | 支持 | 较慢 |

---

## 七、量化工具链

### 7.1 AutoGPTQ 使用

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 准备校准数据
calibration_texts = [...]  # 128 条文本
calibration_dataset = [
    tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    for text in calibration_texts
]

# 量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    sym=True,
)

# 加载 + 量化
model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
model.quantize(calibration_dataset)

# 保存
model.save_quantized("llama2-7b-gptq-4bit")
tokenizer.save_pretrained("llama2-7b-gptq-4bit")
```

### 7.2 AutoAWQ 使用

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-2-7b-hf"

model = AutoAWQForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("llama2-7b-awq-4bit")
tokenizer.save_pretrained("llama2-7b-awq-4bit")
```

### 7.3 GGUF 转换

```bash
# 使用 llama.cpp 自带的转换脚本
python convert_hf_to_gguf.py \
    meta-llama/Llama-2-7b-hf \
    --outfile llama2-7b-f16.gguf \
    --outtype f16

# 量化
./llama-quantize \
    llama2-7b-f16.gguf \
    llama2-7b-q4_k_m.gguf \
    Q4_K_M
```

---

## 八、自检题

1. **PTQ 和 QAT 的核心区别是什么？** LLM 为什么主要用 PTQ？
2. **分组量化（group quantization）解决什么问题？** group_size=128 的额外开销是多少？
3. **GPTQ 的核心公式推导**：量化第 $q$ 列后，如何用 Hessian 信息补偿其他列？
4. **GPTQ 为什么需要 Cholesky 分解？** 对复杂度有什么影响？
5. **GPTQ 的 `desc_act` 选项做了什么？** 为什么能提升精度？
6. **AWQ 如何判断权重的重要性？** 缩放方法的数学原理是什么？
7. **AWQ 为什么不需要 Hessian？** 相比 GPTQ 速度快在哪里？
8. **GGUF 的 K-quant 是什么？** Q4_K_M 的 "4"、"K"、"M" 分别代表什么？
9. **GPTQ、AWQ、GGUF 各适用什么场景？** 画出选型决策树。
10. **如果校准集从 WikiText 换成代码数据，量化质量会变化吗？** 为什么？

---

## 九、产出要求

- [ ] 推导 GPTQ 的逐列量化 + 误差补偿公式
- [ ] 解释 AWQ 的激活感知缩放策略及其数学原理
- [ ] 对比 RTN / GPTQ / AWQ 的量化误差差异
- [ ] 说明 GGUF K-quant 混合量化的原理
- [ ] 画出量化方案选型决策树
- [ ] 制作 GPTQ / AWQ / GGUF 的精度-大小-速度对比表
- [ ] 为 Day6 的量化实验做好知识准备
