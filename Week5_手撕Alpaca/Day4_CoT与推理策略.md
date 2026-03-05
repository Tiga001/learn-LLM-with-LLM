# Day 4：CoT 与推理策略 — 推理能力的涌现与引导

> **目标**：理解 Chain-of-Thought（CoT）Prompting 的核心思想——通过引导模型"分步思考"来激活推理能力；掌握 Few-shot CoT、Zero-shot CoT、Self-Consistency、Tree-of-Thought 等推理策略的原理与适用场景；实践 CoT 在数学推理、逻辑推理、常识推理中的应用；建立"推理增强"的技术视野，为第 17 周 o1 推理打好基础。

---

## 一、为什么需要推理策略？

### 1.1 大模型的推理困境

预训练和 SFT 之后的模型具备了丰富的知识和指令遵循能力，但在**复杂推理任务**上表现不佳：

```
简单问题（直接回忆）:
  Q: "法国的首都是什么？"
  A: "巴黎。" ✅ — 预训练知识直接命中

复杂推理（多步推导）:
  Q: "餐厅有 23 个苹果。他们用了 20 个做午餐，又买了 6 个。现在有多少个苹果？"
  A: "29 个。" ❌ — 模型跳过了推理过程，直接"猜"答案

  正确推理过程:
  A: "餐厅最初有 23 个苹果。用了 20 个后剩余 23 - 20 = 3 个。
      又买了 6 个后变为 3 + 6 = 9 个。答案是 9 个。" ✅
```

### 1.2 标准 Prompting 的局限

```
标准 Prompting（直接问答）:
  Input  → [LLM] → Output
  
  问题: 模型在一步中同时完成"理解+推理+输出"
  → 当推理链条较长时，单步跳跃容易出错
  → 模型的 hidden states 空间不足以隐式完成复杂推理

关键洞察:
  人类解决复杂问题时也不会"一步到位"
  而是会写草稿、列步骤、逐步推导
  → 能否让模型也"展示推理过程"？
```

### 1.3 System 1 vs System 2 思维

Daniel Kahneman 在《Thinking, Fast and Slow》中提出的双系统理论与 LLM 推理高度相关：

| 思维系统 | 特点 | LLM 对应 | 示例 |
|---------|------|---------|------|
| System 1（快思考）| 快速、直觉、自动 | 标准 Prompting | 简单问答、模式匹配 |
| System 2（慢思考）| 慢速、逻辑、审慎 | CoT Prompting | 数学推理、逻辑推导 |

**CoT 的本质**：强制模型从 System 1 切换到 System 2，把中间推理步骤显式化。

---

## 二、Chain-of-Thought（CoT）Prompting

### 2.1 核心论文

**论文**：*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (Wei et al., 2022, Google Brain)

**核心发现**：只需要在 few-shot 示例中加入推理过程（chain of thought），就能显著提升大模型在推理任务上的表现。

### 2.2 Few-shot CoT

```
标准 Few-shot Prompting:
  Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
     Each can has 3 balls. How many tennis balls does he have now?
  A: The answer is 11.

  Q: The cafeteria had 23 apples. They used 20 for lunch and bought 6 more.
     How many apples does the cafeteria have?
  A: The answer is 27. ❌

Few-shot CoT Prompting:
  Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
     Each can has 3 balls. How many tennis balls does he have now?
  A: Roger started with 5 balls. 2 cans of 3 balls each is 2 × 3 = 6 balls.
     5 + 6 = 11. The answer is 11.

  Q: The cafeteria had 23 apples. They used 20 for lunch and bought 6 more.
     How many apples does the cafeteria have?
  A: The cafeteria started with 23 apples. They used 20, so they had
     23 - 20 = 3. They bought 6 more, so 3 + 6 = 9. The answer is 9. ✅
```

**关键区别**：唯一的改变是在示例的答案中加入了推理过程。模型会"模仿"这种推理模式。

### 2.3 CoT 的数学解释

从概率角度理解 CoT 为什么有效：

$$P(\text{answer} \mid \text{question}) = \sum_{\text{reasoning}} P(\text{answer} \mid \text{reasoning}, \text{question}) \cdot P(\text{reasoning} \mid \text{question})$$

标准 Prompting 试图直接估计 $P(\text{answer} \mid \text{question})$，这是一个**边缘分布**，丢弃了中间推理过程的信息。

CoT 将其分解为两步：
1. 先生成推理链 $P(\text{reasoning} \mid \text{question})$
2. 基于推理链生成答案 $P(\text{answer} \mid \text{reasoning}, \text{question})$

```
直接回答（困难）:
  "23 - 20 + 6 = ?" → 直接输出答案 → 容易出错

分步推理（容易）:
  Step 1: "23 - 20 = 3" → 简单减法 ✅
  Step 2: "3 + 6 = 9"   → 简单加法 ✅
  → 每一步都是简单计算，组合起来正确率更高
```

### 2.4 CoT 的涌现特性

Wei et al. 的关键发现——**CoT 是大模型的涌现能力**：

| 模型规模 | 标准 Prompting | CoT Prompting | CoT 提升 |
|---------|---------------|---------------|---------|
| ~1B | 低 | 低（甚至下降）| ≤ 0% |
| ~10B | 中等 | 略有提升 | 5-10% |
| ~100B | 中等 | **显著提升** | **20-50%** |
| 540B (PaLM) | 中高 | **大幅提升** | **30-60%** |

```
关键结论:
  1. CoT 只在足够大的模型上有效（通常 ≥ ~100B 参数）
  2. 小模型使用 CoT 可能反而降低性能（生成无意义的"推理"）
  3. 这是一种涌现能力（emergent ability）：跨过某个规模阈值后突然出现
```

---

## 三、Zero-shot CoT："Let's think step by step"

### 3.1 核心论文

**论文**：*Large Language Models are Zero-Shot Reasoners* (Kojima et al., 2022)

**惊人发现**：不需要 few-shot 示例，只需要在 prompt 末尾加上一句 **"Let's think step by step"** 就能激活推理能力！

### 3.2 方法对比

```
Standard Zero-shot:
  Q: A juggler can juggle 16 balls. Half are golf balls, and half of the
     golf balls are blue. How many blue golf balls?
  A: The answer is 8. ❌

Zero-shot CoT:
  Q: A juggler can juggle 16 balls. Half are golf balls, and half of the
     golf balls are blue. How many blue golf balls? Let's think step by step.
  A: The juggler can juggle 16 balls total.
     Half of them are golf balls: 16 / 2 = 8 golf balls.
     Half of the golf balls are blue: 8 / 2 = 4 blue golf balls.
     The answer is 4. ✅
```

### 3.3 Zero-shot CoT 的两阶段流程

```
Stage 1: 推理提取（Reasoning Extraction）
  Prompt: [Question] + "Let's think step by step."
  → 模型生成推理过程

Stage 2: 答案提取（Answer Extraction）
  Prompt: [Question] + [模型生成的推理] + "Therefore, the answer is"
  → 模型输出最终答案
```

### 3.4 触发短语的效果对比

Kojima et al. 测试了多种触发短语：

| 触发短语 | GSM8K 准确率 | 说明 |
|---------|-------------|------|
| "Let's think step by step" | **78.7%** | 最佳 |
| "Let's think about this logically" | 74.5% | 次佳 |
| "Let's solve this problem by splitting it into steps" | 72.2% | 有效 |
| "First," | 67.3% | 较弱 |
| (无触发短语) | 58.1% | 基线 |

---

## 四、Self-Consistency：多数投票提升可靠性

### 4.1 核心论文

**论文**：*Self-Consistency Improves Chain of Thought Reasoning in Language Models* (Wang et al., 2023)

**核心思想**：一个问题可能有多种正确的推理路径。采样多条推理链，取最终答案的多数投票结果。

### 4.2 方法原理

```
标准 CoT（贪心解码）:
  问题 → [采样 1 条推理链] → 答案
  → 只有 1 次机会，推理出错就全错

Self-Consistency:
  问题 → [采样 K 条推理链] → K 个答案 → 多数投票 → 最终答案
  
  路径 1: "23-20=3, 3+6=9" → 答案 9
  路径 2: "先加后减: 23+6=29, 29-20=9" → 答案 9
  路径 3: "20-6=14, 23-14=9" → 答案 9
  路径 4: "23-20=2, 2+6=8" → 答案 8  ← 计算错误
  路径 5: "23-20=3, 3+6=9" → 答案 9
  
  多数投票: 9 (4票) > 8 (1票) → 最终答案: 9 ✅
```

### 4.3 数学形式化

给定问题 $x$，采样 $K$ 条推理链 $r_1, \ldots, r_K$，每条链得到答案 $a_i$：

$$\hat{a} = \arg\max_{a} \sum_{i=1}^{K} \mathbb{1}[a_i = a]$$

即选择出现频率最高的答案。

**为什么有效？**
- 正确的推理路径通常不止一条
- 不同路径的错误模式不同（不太可能多条路径犯相同的错）
- 多数投票天然具有纠错能力

### 4.4 Self-Consistency 的超参数

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| 采样数量 K | 5~40 | 更多通常更好，但成本线性增长 |
| Temperature | 0.5~0.7 | 需要一定多样性，太低则路径趋同 |
| Top-p | 0.95 | 保持较高多样性 |

```
准确率与采样数量的关系（GSM8K 上的典型趋势）:
  K=1:   ~78% （等价于标准 CoT）
  K=5:   ~84%
  K=10:  ~86%
  K=20:  ~87%
  K=40:  ~88%
  → 明显的收益递减，K=10~20 通常是性价比最高的选择
```

### 4.5 代码实现

```python
from collections import Counter

def self_consistency(model, question, num_samples=10, temperature=0.7):
    """
    Self-Consistency: 采样多条推理链并进行多数投票。
    """
    answers = []
    for _ in range(num_samples):
        prompt = f"{question}\nLet's think step by step."
        response = model.generate(prompt, temperature=temperature)
        answer = extract_answer(response)
        answers.append(answer)
    
    vote = Counter(answers)
    return vote.most_common(1)[0][0]

def extract_answer(response: str) -> str:
    """从推理链中提取最终答案。"""
    patterns = [
        r"[Tt]he answer is\s*(.+?)[\.\n]",
        r"[Aa]nswer:\s*(.+?)[\.\n]",
        r"=\s*(\d+)\s*$",
    ]
    for pattern in patterns:
        import re
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
    return response.strip().split('\n')[-1]
```

---

## 五、Tree-of-Thought（ToT）：推理的搜索空间

### 5.1 核心论文

**论文**：*Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (Yao et al., 2023)

**核心思想**：将推理过程建模为**树状搜索**——在每个推理步骤生成多个候选"思维"，评估每个候选的前景，有策略地探索推理空间。

### 5.2 从 CoT 到 ToT 的演进

```
Standard Prompting（IO）:
  Input ─────────────────────→ Output
  （一步到位）

Chain-of-Thought（CoT）:
  Input → Thought 1 → Thought 2 → Thought 3 → Output
  （一条线性推理链）

Self-Consistency（CoT-SC）:
  Input → Chain 1 → Answer 1 ─┐
        → Chain 2 → Answer 2 ─┼→ Vote → Output
        → Chain 3 → Answer 3 ─┘
  （多条独立链，最终投票）

Tree-of-Thought（ToT）:
  Input → Thought 1a → Thought 2a → Thought 3a → Output ✅
              ↘            ↗      ↘
          Thought 1b → Thought 2b → Thought 3b → 放弃 ✗
              ↘
          Thought 1c → ... → 放弃 ✗
  （树状搜索，带评估和剪枝）
```

### 5.3 ToT 的三个核心组件

#### 1. 思维分解（Thought Decomposition）

将问题分解为中间步骤，每个步骤是一个"思维节点"：

```
问题: "用 4 个数字（1, 5, 6, 10）通过四则运算得到 24"

思维分解:
  Level 1: 选择两个数和一种运算
    → "10 + 6 = 16 (剩余: 1, 5, 16)"
    → "10 - 6 = 4 (剩余: 1, 5, 4)"
    → "10 × 6 = 60 (剩余: 1, 5, 60)"
    → ...

  Level 2: 在剩余数字中继续选择
    → 对每个 Level 1 的结果继续分解
    ...

  Level 3: 最终检查是否得到 24
```

#### 2. 思维评估（Thought Evaluation）

用 LLM 评估每个中间状态的"前景"：

```
Prompt: "Given the numbers {remaining_numbers}, evaluate how likely it is
         to reach 24 through arithmetic operations."

评估结果:
  "剩余: 1, 5, 16" → "sure" (5×16=80 太大, 但 16+5+1=22 接近, 16+5×1=21)
  "剩余: 1, 5, 4"  → "sure" (5×4+1=21 接近, (5-1)×4=16 不行, 但 4×(5+1)=24 ✅)
  "剩余: 1, 5, 60" → "impossible" (数字太大)

→ 优先探索"sure"的分支，剪枝"impossible"的分支
```

#### 3. 搜索算法（Search Algorithm）

ToT 支持两种搜索策略：

| 策略 | BFS（广度优先）| DFS（深度优先）|
|------|---------------|---------------|
| 适用场景 | 推理步骤少、分支多 | 推理步骤多、需要回溯 |
| 优势 | 不会遗漏最优解 | 节省内存 |
| 劣势 | 内存需求大 | 可能陷入错误分支 |

```python
def tree_of_thought_bfs(model, problem, max_depth=3, branch_factor=3):
    """
    ToT 广度优先搜索实现。
    """
    frontier = [{"state": problem, "path": []}]
    
    for depth in range(max_depth):
        candidates = []
        for node in frontier:
            thoughts = generate_thoughts(model, node["state"], k=branch_factor)
            for thought in thoughts:
                new_state = apply_thought(node["state"], thought)
                score = evaluate_thought(model, new_state)
                candidates.append({
                    "state": new_state,
                    "path": node["path"] + [thought],
                    "score": score
                })
        
        # 保留 top-k 候选
        candidates.sort(key=lambda x: x["score"], reverse=True)
        frontier = candidates[:branch_factor]
    
    return frontier[0]["path"]
```

### 5.4 ToT vs CoT vs Self-Consistency 对比

| 维度 | CoT | Self-Consistency | ToT |
|------|-----|-----------------|-----|
| 推理结构 | 线性链 | 多条独立链 | 树状搜索 |
| 中间步骤评估 | 无 | 无 | 有（LLM 评估）|
| 回溯能力 | 无 | 无 | 有 |
| 计算成本 | 1× | K× | 高（取决于树的大小）|
| 适用场景 | 通用推理 | 数学计算 | 创意搜索、规划 |

---

## 六、更多推理策略

### 6.1 Least-to-Most Prompting

**论文**：*Least-to-Most Prompting Enables Complex Reasoning in Large Language Models* (Zhou et al., 2023)

**核心思想**：将复杂问题分解为子问题，从最简单的子问题开始逐步解决。

```
问题: "最后一个参加聚会的人是谁？Amy 比 Ben 晚到，
      Charlie 在 Ben 之前到达，Diana 在 Amy 之后到达。"

Step 1 — 分解:
  子问题 1: "谁先到，Charlie 还是 Ben？" → Charlie
  子问题 2: "谁先到，Ben 还是 Amy？" → Ben
  子问题 3: "谁先到，Amy 还是 Diana？" → Amy

Step 2 — 逐步解决:
  从子问题 1: Charlie < Ben（Charlie 先到）
  从子问题 2: Ben < Amy（Ben 先到）
  从子问题 3: Amy < Diana（Amy 先到）
  → 顺序: Charlie → Ben → Amy → Diana
  → 最后到的是 Diana ✅
```

### 6.2 Plan-and-Solve Prompting

**核心思想**：先制定计划，再按计划执行。

```
Plan-and-Solve Prompt:
  "Let's first understand the problem and devise a plan to solve it.
   Then, let's carry out the plan and solve the problem step by step."

与 Zero-shot CoT 的区别:
  Zero-shot CoT: "Let's think step by step" → 模型自由发挥
  Plan-and-Solve: 先规划步骤 → 再按步骤执行 → 更有结构
```

### 6.3 Reflexion：自我反思

**论文**：*Reflexion: Language Agents with Verbal Reinforcement Learning* (Shinn et al., 2023)

```
循环:
  1. 生成初始答案
  2. 评估答案（用 LLM 或外部工具）
  3. 如果答案错误，生成"反思"（为什么错了？如何改进？）
  4. 将反思加入 prompt，重新生成答案
  5. 重复直到答案正确或达到最大轮数

示例:
  Round 1: "23 - 20 + 6 = 29" ← 错误
  反思: "我直接算了 23+6=29，忘记先减 20 了。应该先算 23-20=3。"
  Round 2: "23 - 20 = 3, 3 + 6 = 9" ← 正确 ✅
```

---

## 七、CoT 在指令微调中的应用

### 7.1 CoT 数据在 SFT 中的价值

回到本周主题（指令微调），CoT 对 SFT 的影响至关重要：

```
不含 CoT 的 SFT 数据:
  Q: "What is 23 - 20 + 6?"
  A: "9"
  → 模型学会直接输出答案，但不学推理

包含 CoT 的 SFT 数据:
  Q: "What is 23 - 20 + 6?"
  A: "First, 23 - 20 = 3. Then, 3 + 6 = 9. The answer is 9."
  → 模型学会推理过程 + 输出答案
```

### 7.2 CoT 数据对模型能力的影响

| SFT 数据类型 | 简单问答能力 | 复杂推理能力 | 说明 |
|-------------|-----------|------------|------|
| 直接答案 | 高 | 低 | 模型只学会了"回忆" |
| CoT 推理链 | 高 | 高 | 模型同时学会了"推理" |
| 混合（推荐）| 高 | 高 | 不同任务用不同格式 |

### 7.3 生成 CoT SFT 数据

在 Day 3 的 Self-Instruct 管线中，可以要求 LLM 在生成 output 时包含推理过程：

```python
COT_GENERATION_PROMPT = """Generate a response that includes step-by-step 
reasoning before giving the final answer.

Instruction: {instruction}
Input: {input}

Response (include reasoning steps):"""
```

**注意**：并非所有任务都需要 CoT。简单的事实问答或创意写作不需要推理链。通常只在以下场景使用：
- 数学计算
- 逻辑推理
- 多步因果分析
- 代码调试
- 复杂规划

---

## 八、推理策略的全景图与展望

### 8.1 推理策略演进

```
2022.01  Chain-of-Thought (Wei et al.)
  │      "在 few-shot 中加入推理过程"
  ▼
2022.05  Zero-shot CoT (Kojima et al.)
  │      "Let's think step by step"
  ▼
2022.03  Self-Consistency (Wang et al.)
  │      "采样多条链 + 多数投票"
  ▼
2023.03  Least-to-Most (Zhou et al.)
  │      "分解子问题，从简到难"
  ▼
2023.05  Tree-of-Thought (Yao et al.)
  │      "树状搜索 + 评估 + 回溯"
  ▼
2023.06  Reflexion (Shinn et al.)
  │      "自我反思 + 迭代改进"
  ▼
2024+    o1 / R1 推理
         "RL 训练让模型自主学会 CoT"
         → 第 17 周深入
```

### 8.2 从 Prompting 到 Training

本日讨论的推理策略都是**推理时**（inference-time）的方法——不改变模型参数，仅改变 prompt 策略。

更强的方式是在**训练时**让模型学会推理：

| 方式 | 代表 | 思路 |
|------|------|------|
| Prompting CoT | 本日内容 | 推理时引导 |
| SFT with CoT | Flan / Alpaca | 用 CoT 数据做 SFT |
| RL for Reasoning | o1 / R1 / GRPO | 用 RL 奖励正确推理 |
| Process Reward Model | PRM / Math-Shepherd | 奖励每一步推理 |

```
能力递增:
  标准 Prompting < CoT Prompting < SFT with CoT < RL for Reasoning
                                                          ↑
                                              第 17 周 o1 推理重点
```

---

## 九、实践：CoT Prompting 对比实验

以下是你可以动手实践的对比实验（使用任何可用的 LLM API 或本地模型）：

### 实验 1：标准 vs CoT

```python
standard_prompt = """
Q: A store has 156 shirts. They sell 49 on Monday, 78 on Tuesday, 
   and receive 34 new shirts on Wednesday. How many shirts are left?
A:"""

cot_prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans with 3 balls each.
   How many total? 
A: Roger started with 5 balls. 2 cans × 3 = 6 new balls. 5 + 6 = 11.

Q: A store has 156 shirts. They sell 49 on Monday, 78 on Tuesday, 
   and receive 34 new shirts on Wednesday. How many shirts are left?
A:"""
```

### 实验 2：Zero-shot CoT 触发短语对比

```python
triggers = [
    "Let's think step by step.",
    "Let's work through this carefully.",
    "Let's break this down.",
    "First, let me understand the problem.",
    "",  # 无触发（基线）
]

question = "If 3 shirts cost $45 and each shirt costs the same, how much would 7 shirts cost?"

for trigger in triggers:
    prompt = f"Q: {question} {trigger}\nA:"
    # response = model.generate(prompt)
    # print(f"Trigger: '{trigger}' → Answer: {response}")
```

### 实验 3：Self-Consistency

```python
# 同一个问题采样 10 次，temperature=0.7
# 对比贪心解码（K=1）和多数投票（K=10）的准确率
```

---

## 十、自检题

1. **什么是 Chain-of-Thought Prompting？** 与标准 Prompting 的核心区别是什么？
2. **为什么 CoT 只在大模型（≥100B）上有效？** 小模型使用 CoT 可能有什么问题？
3. **Zero-shot CoT 只需要加一句什么话就能激活推理？** 为什么这句话有效？
4. **Self-Consistency 的核心思想是什么？** 它在数学上等价于什么操作？
5. **Tree-of-Thought 的三个核心组件是什么？** 它比 CoT 强在哪里？
6. **Least-to-Most Prompting 和 CoT 有什么区别？**
7. **CoT 数据在 SFT 中有什么价值？** 为什么 CoT SFT 数据比直接答案数据更好？
8. **从 CoT Prompting 到 o1/R1 推理，技术路线的核心区别是什么？**

---

## 十一、产出要求

- [ ] 对比标准 Prompting 和 CoT Prompting 在至少 3 个推理问题上的效果
- [ ] 实践 Zero-shot CoT（使用 "Let's think step by step"）
- [ ] 实现 Self-Consistency 的多数投票逻辑
- [ ] 画出 IO → CoT → CoT-SC → ToT 的推理策略演进图
- [ ] 撰写 CoT 在 SFT 数据中的应用笔记
- [ ] 为第 17 周 o1 推理建立预备知识框架
