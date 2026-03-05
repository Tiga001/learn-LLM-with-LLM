# Day 2：LoRA 算法推导 — 低秩分解的数学原理

> **目标**：从线性代数的基本工具（SVD / 低秩近似）出发，严格推导 LoRA 的数学形式；理解为什么 Attention 权重矩阵的更新是低秩的；推导 rank、alpha 的数学作用与缩放机制；分析 LoRA 的参数量、梯度传播与优化器状态；为 Day 3 的代码实现打下坚实的数学基础。

---

## 一、前置知识：SVD 与低秩近似

### 1.1 奇异值分解（SVD）

任意矩阵 $M \in \mathbb{R}^{m \times n}$ 都可以进行奇异值分解：

$$M = U \Sigma V^T$$

其中：
- $U \in \mathbb{R}^{m \times m}$：左奇异向量矩阵（正交）
- $\Sigma \in \mathbb{R}^{m \times n}$：对角矩阵，对角元素为奇异值 $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$
- $V \in \mathbb{R}^{n \times n}$：右奇异向量矩阵（正交）

### 1.2 Eckart-Young 定理：最佳低秩近似

**定理**：对于矩阵 $M$ 的最佳秩-$r$ 近似（在 Frobenius 范数意义下）为：

$$M_r = \sum_{i=1}^{r} \sigma_i u_i v_i^T = U_r \Sigma_r V_r^T$$

其中 $U_r, \Sigma_r, V_r$ 分别取 SVD 的前 $r$ 个分量。近似误差为：

$$\|M - M_r\|_F = \sqrt{\sum_{i=r+1}^{\min(m,n)} \sigma_i^2}$$

**直觉**：如果奇异值快速衰减（前几个很大，后面接近零），那么用很低的秩就能近似整个矩阵。

### 1.3 低秩矩阵的参数化

一个秩为 $r$ 的矩阵 $M \in \mathbb{R}^{m \times n}$ 可以写成两个矩阵的乘积：

$$M = BA, \quad B \in \mathbb{R}^{m \times r}, \quad A \in \mathbb{R}^{r \times n}$$

参数量对比：
- 原始矩阵 $M$：$m \times n$ 个参数
- 低秩分解 $BA$：$m \times r + r \times n = r(m + n)$ 个参数

当 $r \ll \min(m, n)$ 时，参数量大幅减少：

$$\frac{r(m+n)}{mn} = r\left(\frac{1}{n} + \frac{1}{m}\right)$$

**示例**：$m = n = 4096$（LLaMA-7B 的隐藏维度），$r = 16$：

$$\frac{16 \times (4096 + 4096)}{4096 \times 4096} = \frac{131072}{16777216} \approx 0.78\%$$

---

## 二、LoRA 的数学推导

### 2.1 问题形式化

设预训练模型参数为 $\theta_0$，目标任务的最优参数为 $\theta^*$。全参数微调寻找：

$$\theta^* = \theta_0 + \Delta\theta, \quad \Delta\theta = \arg\min_{\Delta\theta} \mathcal{L}(\theta_0 + \Delta\theta)$$

对于单个权重矩阵 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，微调后的权重为：

$$W = W_0 + \Delta W$$

LoRA 的核心假设：$\Delta W$ 是低秩的。

### 2.2 低秩假设的验证

对全参数微调得到的 $\Delta W$ 做 SVD：

$$\Delta W = U \Sigma V^T = \sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i u_i v_i^T$$

实验观察（Hu et al., 2021 & Aghajanyan et al., 2021）：

$$\sigma_1 \gg \sigma_2 > \sigma_3 > \ldots > \sigma_r \gg \sigma_{r+1} \approx \sigma_{r+2} \approx \ldots \approx 0$$

其中 $r$ 很小（通常 4~64）。这意味着：

$$\Delta W \approx \Delta W_r = \sum_{i=1}^{r} \sigma_i u_i v_i^T$$

**为什么 $\Delta W$ 是低秩的？**

```
直觉解释 1: 任务信息量有限
  预训练: 在 TB 级数据上学习，模型需要 d×d 维的完整表示
  微调: 在 KB~MB 级数据上适配，只需要调整少量"方向"
  → ΔW 的信息量远小于 W₀，自然是低秩的

直觉解释 2: 表面对齐假说
  SFT 只是在教模型"格式"和"风格"（W5 Day1）
  → 知识不变，只调整行为模式
  → 行为调整只涉及少数几个"维度"

直觉解释 3: 从优化轨迹看
  SGD 在高维空间中的有效搜索方向数量有限
  特别是在小数据集上，梯度的有效秩很低
  → 参数更新被限制在一个低维子空间中
```

### 2.3 LoRA 参数化

基于低秩假设，LoRA 将 $\Delta W$ 参数化为：

$$\Delta W = BA, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad A \in \mathbb{R}^{r \times d_{\text{in}}}$$

修改后的前向传播：

$$h = Wx = (W_0 + \Delta W)x = W_0 x + BAx$$

### 2.4 缩放因子 $\alpha/r$ 的推导

LoRA 在实际实现中引入了缩放因子：

$$h = W_0 x + \frac{\alpha}{r} BAx$$

**为什么需要缩放？**

考虑 $A$ 的初始化为 $\mathcal{N}(0, \sigma^2)$，$B$ 初始化为零。当 $B$ 从零开始训练时，$BAx$ 的量级取决于 $A$ 和 $x$。

当我们改变 $r$ 时（比如从 $r=8$ 变为 $r=64$），$BAx$ 的量级会发生变化。缩放因子 $\frac{\alpha}{r}$ 的作用是：

**当改变 $r$ 时，不需要重新调整学习率。**

具体地：
- $\alpha$ 是一个固定的常数（超参数），通常等于第一次实验时选定的 $r$ 值
- 当 $r$ 变大时，$\frac{\alpha}{r}$ 变小，补偿 $BAx$ 量级的增大
- 当 $r$ 变小时，$\frac{\alpha}{r}$ 变大，补偿 $BAx$ 量级的减小

```
数学分析:

设 A 的元素 ~ N(0, σ²), B 的元素（训练后）~ N(0, σ'²)

BAx 的量级:
  Var[(BAx)_i] ≈ r · σ'² · σ² · ||x||²

当 r 翻倍:
  Var[(BAx)_i] ≈ 2r · σ'² · σ² · ||x||²  → 量级翻倍

加入 α/r 缩放后:
  Var[(α/r · BAx)_i] ≈ (α/r)² · r · σ'² · σ² · ||x||²
                      = α² · σ'² · σ² · ||x||² / r

→ 量级随 r 增大而减小，但更重要的是:
  当固定 α = r₀（第一次实验的 r），改变 r 时，
  有效缩放 α/r 自动调整，减少超参数搜索负担
```

**常见设置**：

| $r$ | $\alpha$ | $\alpha/r$ | 说明 |
|-----|---------|-----------|------|
| 8 | 16 | 2.0 | 论文常见设置 |
| 16 | 32 | 2.0 | $\alpha = 2r$ |
| 16 | 16 | 1.0 | $\alpha = r$ |
| 64 | 16 | 0.25 | 保持与 $r=16, \alpha=16$ 同样的学习率 |

经验法则：**$\alpha$ 通常设为 $r$ 或 $2r$**。

### 2.5 初始化策略的数学分析

#### $B$ 矩阵初始化为零

$$B = \mathbf{0} \implies \Delta W = BA = \mathbf{0} \cdot A = \mathbf{0}$$

训练开始时，$h = W_0 x + \mathbf{0} = W_0 x$。

这意味着：
1. 模型初始行为 = 预训练模型 → 训练从一个好的起点开始
2. 训练初期的梯度完全由预训练模型的输出决定 → 梯度信号稳定

#### $A$ 矩阵用 Kaiming 初始化

$$A_{ij} \sim \mathcal{N}\left(0, \frac{1}{d_{\text{in}}}\right) \quad \text{或} \quad A_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{1}{d_{\text{in}}}}, \sqrt{\frac{1}{d_{\text{in}}}}\right)$$

这样当 $B$ 从零开始更新时，$BA x$ 的量级从零平滑增长。

**如果反过来（$A=0, B$ 随机）会怎样？**

$$\Delta W = B \cdot \mathbf{0} = \mathbf{0}$$

效果相同！但论文选择 $B=0$ 的原因是：梯度 $\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial h} \cdot (Ax)^T$，由于 $A$ 非零，$B$ 可以立即接收到有意义的梯度。而如果 $A=0$，$\frac{\partial \mathcal{L}}{\partial A} = B^T \cdot \frac{\partial \mathcal{L}}{\partial h} \cdot x^T$，初始时 $B$ 随机，梯度信号可能不够稳定。

### 2.6 梯度传播分析

前向传播：$h = W_0 x + \frac{\alpha}{r} BAx$

关于 $B$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \cdot \frac{\partial \mathcal{L}}{\partial h} \cdot (Ax)^T$$

关于 $A$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} \cdot B^T \cdot \frac{\partial \mathcal{L}}{\partial h} \cdot x^T$$

注意：
- $W_0$ **不接收梯度**（冻结），但它影响 $\frac{\partial \mathcal{L}}{\partial h}$ 的值
- 计算 $\frac{\partial \mathcal{L}}{\partial B}$ 需要 $Ax$，计算 $\frac{\partial \mathcal{L}}{\partial A}$ 需要 $B$ 和 $x$
- 梯度的计算量远小于对 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 计算梯度

---

## 三、参数量与显存分析

### 3.1 参数量

对于单个权重矩阵 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$：

$$\text{LoRA 参数} = d_{\text{out}} \times r + r \times d_{\text{in}} = r(d_{\text{out}} + d_{\text{in}})$$

$$\text{参数比} = \frac{r(d_{\text{out}} + d_{\text{in}})}{d_{\text{out}} \times d_{\text{in}}}$$

对 LLaMA-7B 的 Attention 层（$d = 4096$，$W_q, W_k, W_v, W_o$ 各为 $4096 \times 4096$）：

| 应用模块 | $r$ | 每层 LoRA 参数 | 32层总参数 | 占比 |
|---------|-----|------------|---------|------|
| $W_q, W_v$ | 8 | $2 \times 8 \times 2 \times 4096 = 131,072$ | 4,194,304 | 0.063% |
| $W_q, W_v$ | 16 | $2 \times 16 \times 2 \times 4096 = 262,144$ | 8,388,608 | 0.125% |
| 所有 Linear | 16 | $6 \times 16 \times 2 \times 4096 = 786,432$ | 25,165,824 | 0.375% |

### 3.2 优化器状态

使用 AdamW 优化器，每个可训练参数需要存储一阶动量 $m$ 和二阶动量 $v$：

$$\text{优化器状态显存} = 2 \times \text{LoRA 参数量} \times 4 \text{ bytes (FP32)}$$

| 配置 | LoRA 参数 | 优化器状态 | 与 Full FT 对比 |
|------|---------|---------|------------|
| $r=16$, $W_q + W_v$ | 8.4M | 67 MB | Full FT: 52 GB |
| $r=16$, 全部 Linear | 25.2M | 201 MB | 节省 99.6% |

### 3.3 显存节省来源

```
Full FT 显存:
  模型参数 (FP16):    13 GB     ← LoRA 也需要
  优化器状态 (FP32):  52 GB     ← LoRA 大幅节省 (67 MB)
  梯度 (FP16):       13 GB     ← LoRA 大幅节省 (16 MB)
  激活值:            ~10 GB     ← LoRA 节省少量
  ──────────────────────
  总计:              ~88 GB

LoRA 显存:
  模型参数 (FP16):    13 GB     ← 必须加载完整模型
  LoRA 参数 (FP16):   16 MB     ← 新增
  优化器状态 (FP32):  67 MB     ← 大幅减少
  梯度 (FP16):       16 MB     ← 大幅减少
  激活值:            ~8 GB      ← 略有减少
  ──────────────────────
  总计:              ~21 GB     ← 节省 76%

QLoRA 显存:
  模型参数 (NF4):     3.5 GB    ← 4-bit 量化
  LoRA 参数 (BF16):   16 MB
  优化器状态 (FP32):  67 MB
  梯度 (BF16):       16 MB
  激活值:            ~6 GB
  ──────────────────────
  总计:              ~10 GB     ← 节省 89%
```

---

## 四、LoRA 与 SVD 的关系

### 4.1 LoRA 是否等价于 SVD 近似？

不是。两者有本质区别：

| 维度 | SVD 近似 | LoRA |
|------|---------|------|
| 对象 | 已有矩阵 $\Delta W$ | 未知的 $\Delta W$ |
| 方法 | 后验分解（先微调再分解）| 在线学习（直接学习 $B$ 和 $A$）|
| 最优性 | Frobenius 范数最优 | 任务 loss 最优 |
| 约束 | $U, V$ 正交 | $B, A$ 无约束 |

### 4.2 LoRA 找到的是否是最优低秩近似？

不一定是 $\Delta W$ 的 SVD 最优秩-$r$ 近似，但可能是**更好**的解。

原因：LoRA 的目标是最小化**任务 loss**（$\mathcal{L}$），而不是最小化 $\|\Delta W - BA\|_F$。

$$\text{SVD 目标}: \min_{B,A} \|W^* - W_0 - BA\|_F^2$$
$$\text{LoRA 目标}: \min_{B,A} \mathcal{L}(W_0 + BA)$$

LoRA 可能找到一个在 Frobenius 范数意义上不是最优近似，但在任务性能上更优的解。

### 4.3 为什么不直接先 Full FT 再 SVD 压缩？

```
先 Full FT 再 SVD:
  Full FT → 得到 ΔW (d×d 参数) → SVD → 取前 r 个分量
  问题: 需要先做 Full FT（同样的显存和计算成本）
  → 没有解决核心问题

LoRA:
  直接在 r(d+d) 维空间中优化
  → 从一开始就在低秩空间中搜索
  → 训练成本与低秩参数量成正比
```

---

## 五、LoRA 应用于 Transformer 的数学形式

### 5.1 Multi-Head Attention 中的 LoRA

对于 LLaMA 的 Attention 层，有四个权重矩阵：$W_q, W_k, W_v, W_o$。

以 $W_q$ 为例：

$$Q = W_q X = (W_{q,0} + \frac{\alpha}{r} B_q A_q) X$$

其中 $W_{q,0} \in \mathbb{R}^{d \times d}$ 是冻结的预训练权重，$B_q \in \mathbb{R}^{d \times r}$，$A_q \in \mathbb{R}^{r \times d}$。

对所有 Attention 矩阵应用 LoRA：

$$Q = (W_{q,0} + \frac{\alpha}{r} B_q A_q) X$$
$$K = (W_{k,0} + \frac{\alpha}{r} B_k A_k) X$$
$$V = (W_{v,0} + \frac{\alpha}{r} B_v A_v) X$$
$$O = (W_{o,0} + \frac{\alpha}{r} B_o A_o) \cdot \text{Concat}(\text{head}_1, \ldots, \text{head}_h)$$

### 5.2 FFN 中的 LoRA

LLaMA 使用 SwiGLU FFN：

$$\text{FFN}(x) = (\text{Swish}(x W_{\text{gate}}) \odot x W_{\text{up}}) W_{\text{down}}$$

对 FFN 的三个矩阵也可以应用 LoRA：

$$W_{\text{gate}} \rightarrow W_{\text{gate},0} + \frac{\alpha}{r} B_{\text{gate}} A_{\text{gate}}$$
$$W_{\text{up}} \rightarrow W_{\text{up},0} + \frac{\alpha}{r} B_{\text{up}} A_{\text{up}}$$
$$W_{\text{down}} \rightarrow W_{\text{down},0} + \frac{\alpha}{r} B_{\text{down}} A_{\text{down}}$$

### 5.3 Dropout in LoRA

LoRA 在 $A$ 矩阵的输出后添加 Dropout：

$$h = W_0 x + \frac{\alpha}{r} B \cdot \text{Dropout}(Ax)$$

这对防止 LoRA 过拟合有帮助，尤其在数据量较小时。

---

## 六、权重合并与多 LoRA 切换

### 6.1 权重合并（Merge）

训练完成后，LoRA 权重可以**合并到基座模型**中：

$$W' = W_0 + \frac{\alpha}{r} BA$$

合并后的模型与标准模型结构完全相同，推理时无任何额外开销。

```python
# 权重合并伪代码
def merge_lora(W0, A, B, alpha, r):
    return W0 + (alpha / r) * B @ A

# 合并后推理
# 之前: h = W0 @ x + (alpha/r) * B @ A @ x  (两次矩阵乘法)
# 之后: h = W_merged @ x                     (一次矩阵乘法)
```

### 6.2 多 LoRA 切换

```
场景: 一个基座模型服务多个任务

方案 A: 全参数微调
  任务 1 → 模型_1 (13 GB)
  任务 2 → 模型_2 (13 GB)
  任务 3 → 模型_3 (13 GB)
  总存储: 39 GB

方案 B: LoRA
  基座模型 (13 GB) + LoRA_1 (16 MB) + LoRA_2 (16 MB) + LoRA_3 (16 MB)
  总存储: 13.05 GB
  切换任务: 卸载当前 LoRA，加载新 LoRA (~毫秒级)

方案 C: 多 LoRA 并行推理
  基座模型 (13 GB) + 所有 LoRA 同时加载
  每个请求根据任务路由到不同的 LoRA 分支
  → vLLM 等框架已支持
```

### 6.3 LoRA 的算术组合

```
有趣的应用: LoRA 的线性组合

LoRA_english: 让模型说英语
LoRA_formal: 让模型使用正式语言

组合: LoRA_formal_english = α₁ · LoRA_english + α₂ · LoRA_formal

ΔW = α₁ · B₁A₁ + α₂ · B₂A₂

→ 不需要重新训练就能组合不同能力
→ 类似于图像生成中的 LoRA 组合
```

---

## 七、LoRA 的理论局限与开放问题

### 7.1 低秩假设的适用边界

```
低秩假设成立的条件:
  ✅ 微调数据量较小（SFT、指令微调）
  ✅ 任务与预训练分布不太远
  ✅ 模型已经有较好的基础能力

低秩假设可能不成立:
  ❌ 大规模 Continual Pre-training（新领域数据量很大）
  ❌ 跨模态适配（如从文本到视觉）
  ❌ 极端分布外任务
  
  → 这些场景下 Full FT 可能显著优于 LoRA
```

### 7.2 rank 的选择困境

- 理论上，$r$ 应该等于 $\Delta W$ 的有效秩
- 但在训练前我们不知道 $\Delta W$ 的有效秩
- 实践中只能通过实验来确定最优的 $r$

自适应秩方法（如 AdaLoRA）试图在训练中动态调整每层的秩，但增加了实现复杂度。

### 7.3 LoRA 能否完全替代 Full FT？

```
已有研究表明:
  在绝大多数 NLP 基准测试中: LoRA ≈ Full FT (差距 < 1%)
  在知识密集型任务中: LoRA 可能略差 (差距 1-3%)
  在需要大幅改变模型行为的任务中: Full FT > LoRA

结论:
  LoRA 是 Full FT 的高效近似，适用于大多数场景
  但不是所有场景的最优解
  → 选择方法时需要考虑任务特性和资源约束
```

---

## 八、自检题

1. **写出 SVD 的定义**，解释 Eckart-Young 定理的含义。
2. **从低秩分解的角度推导 LoRA 的参数量**。对于 $W_0 \in \mathbb{R}^{4096 \times 4096}$，$r=16$ 时 LoRA 参数量是多少？相比原始矩阵压缩了多少倍？
3. **推导 LoRA 前向传播公式**，包括缩放因子 $\frac{\alpha}{r}$。解释 $\alpha$ 和 $r$ 的关系。
4. **解释 $B$ 初始化为零的数学原因**。如果 $A$ 和 $B$ 都随机初始化会怎样？
5. **推导 LoRA 对 $A$ 和 $B$ 的梯度公式**。说明为什么梯度计算效率高于对 $W_0$ 的梯度。
6. **解释 LoRA 和 SVD 近似的区别**。为什么 LoRA 的解可能比 SVD 近似更好？
7. **计算 LLaMA-7B 使用 LoRA ($r=16$, 所有 Linear) 的可训练参数量和优化器状态显存**。
8. **解释权重合并的原理**。合并后推理为什么零开销？
9. **为什么微调的 $\Delta W$ 是低秩的？** 给出至少两个直觉解释和一个实验证据。
10. **LoRA 的理论局限是什么？** 在什么场景下 LoRA 可能不如 Full FT？

---

## 九、产出要求

- [ ] 推导 LoRA 的前向传播公式 $h = W_0 x + \frac{\alpha}{r} BAx$
- [ ] 推导 LoRA 对 $A$ 和 $B$ 的梯度公式
- [ ] 计算 LLaMA-7B 各种 LoRA 配置的参数量（$r=8/16/64$，$W_q W_v$ vs 全部 Linear）
- [ ] 用 SVD 的语言解释 LoRA，并说明两者的区别
- [ ] 画出 LoRA 在 Transformer Block 中的应用位置图（标注冻结/可训练）
- [ ] 解释缩放因子 $\alpha/r$ 的数学作用，推导不同 $r$ 下的缩放效果
- [ ] 分析 LoRA 和 Full FT 的显存对比（模型参数 + 优化器 + 梯度 + 激活值）
