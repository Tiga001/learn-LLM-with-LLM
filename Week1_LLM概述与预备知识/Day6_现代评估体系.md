# Day 6：现代评估体系 — MMLU / HumanEval / MT-Bench / Arena Elo

> **目标**：理解 LLM 时代的主流评估方法，知道每种基准测什么、怎么用、有何局限，为后续阶段（W13/W18）的评估实践做铺垫。

---

## 为什么需要新的评估方式？

Day 5 学习的经典指标（BLEU / ROUGE / PPL）在 LLM 时代暴露了根本局限：

| 局限 | 原因 |
|------|------|
| **开放式生成无参考答案** | BLEU/ROUGE 需要标准答案，但 LLM 的对话/创作输出没有唯一正确答案 |
| **PPL 不反映能力** | 低 PPL 不代表模型能做推理、编程、数学 |
| **无法评估指令遵循** | 经典指标无法衡量模型是否真正理解并执行了用户的指令 |
| **无法评估安全性** | 模型可能流畅地输出有害内容 |

现代评估体系分为三大类：

```
1. 学术基准（多选题 / 代码生成）
   MMLU → 知识广度
   HumanEval / MBPP → 代码能力

2. 模型评估（LLM-as-Judge）
   MT-Bench → GPT-4 打分
   AlpacaEval → 自动对比

3. 人类评估（众包排名）
   Chatbot Arena → Elo 排名
```

---

## 一、MMLU — 知识广度的标尺

### 1.1 是什么

**MMLU** (Massive Multitask Language Understanding, Hendrycks et al., 2021)

- **57 个学科**的多选题（4 选 1），覆盖 STEM、人文、社科、职业等
- 从高中到专业水平，共 ~14,000 题
- 示例学科：抽象代数、天文学、临床医学、法律、机器学习...

### 1.2 测试格式

```
Question: The longest wavelength of light that can be used to cause 
          photoelectric emission from potassium is 564 nm. What is 
          the work function of potassium?

(A) 1.1 eV  (B) 2.2 eV  (C) 3.3 eV  (D) 4.4 eV

Answer: (B)
```

评估方式：给模型几个 few-shot 示例，让模型生成答案字母，计算准确率。

### 1.3 主要模型得分（2025 年水平）

| 模型 | MMLU 5-shot 准确率 |
|------|:------------------:|
| 人类专家 | ~89% |
| GPT-4o | ~88% |
| Claude 3.5 Sonnet | ~89% |
| LLaMA-3 70B | ~82% |
| LLaMA-3 8B | ~68% |
| DeepSeek-V2.5 | ~80% |
| 随机猜测 | 25% |

### 1.4 局限性

- **多选题形式过于简单**：不能测开放式推理
- **数据污染**：许多模型的训练数据可能包含 MMLU 题目
- **不测生成质量**：只看对错，不看表述
- **衍生版本**：MMLU-Pro (更难、10 选 1)、GPQA (博士级专家题)

---

## 二、HumanEval — 代码能力的标尺

### 2.1 是什么

**HumanEval** (Chen et al., 2021, OpenAI)

- **164 道编程题**，每题包含函数签名、docstring 和测试用例
- 模型需要根据描述生成正确的函数实现
- 使用 **pass@k** 指标：采样 k 次，至少有一次通过所有测试的概率

### 2.2 测试格式

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer 
    to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    # 模型需要生成这里的实现
```

### 2.3 pass@k 指标

从模型采样 $n$ 次生成代码，其中 $c$ 次通过所有测试用例：

$$
\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

最常报告 **pass@1**（一次就对的概率）。

### 2.4 主要模型得分

| 模型 | HumanEval pass@1 |
|------|:-----------------:|
| GPT-4o | ~91% |
| Claude 3.5 Sonnet | ~92% |
| DeepSeek-Coder-V2 | ~90% |
| LLaMA-3 70B | ~81% |
| LLaMA-3 8B | ~62% |
| GPT-3.5 | ~48% |
| Codex (2021) | ~28% |

### 2.5 相关基准

| 基准 | 特点 |
|------|------|
| **MBPP** | 974 道更简单的 Python 题 |
| **HumanEval+** | 修复了原版测试用例不充分的问题 |
| **LiveCodeBench** | 实时更新的编程题（防数据污染） |
| **SWE-bench** | 真实 GitHub Issue 修复（更贴近工程实践） |

---

## 三、MT-Bench — LLM-as-Judge

### 3.1 是什么

**MT-Bench** (Zheng et al., 2023, LMSYS)

- **80 道多轮对话题**，覆盖 8 个类别：
  写作、角色扮演、推理、数学、编程、提取、STEM、人文
- 每题 2 轮对话（测多轮能力）
- 用 **GPT-4 作为评委打分**（1-10 分）

### 3.2 评估流程

```
1. 向被测模型提出问题（2 轮对话）
2. 将模型回答送给 GPT-4
3. GPT-4 按照评分标准打 1-10 分
4. 取所有题目的平均分
```

### 3.3 示例

```
[第1轮] 用户: 写一个关于时间旅行的短故事，限 100 字。
模型回答: ...

[第2轮] 用户: 现在把故事改写成第一人称视角。
模型回答: ...

GPT-4 评分: 8/10
理由: 故事有创意，第一人称改写基本正确，但叙事不够紧凑...
```

### 3.4 优势与局限

| 优势 | 局限 |
|------|------|
| 测试了开放式生成质量 | GPT-4 作为评委有自身偏好（偏好长回答） |
| 包含多轮对话能力 | 题目只有 80 道，统计显著性不强 |
| 可自动化运行 | 评委模型的能力是天花板 |
| 覆盖多种能力维度 | 不同版本 GPT-4 评分可能不一致 |

---

## 四、Chatbot Arena — 人类众包排名

### 4.1 是什么

**Chatbot Arena** (LMSYS, 2023~)

- 用户在网站上与两个匿名模型同时对话
- 用户根据回答质量投票选出更好的一个
- 用 **Elo 评分系统**（类似国际象棋排名）计算模型排名

### 4.2 Elo 评分系统

借鉴国际象棋的 Elo 系统：

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

其中 $E_A$ 是 A 模型获胜的期望概率，$R_A, R_B$ 是双方的 Elo 分。

每次对战后更新：
$$R_A^{\text{new}} = R_A + K \cdot (S_A - E_A)$$

- $S_A = 1$（A 赢）, $0.5$（平局）, $0$（A 输）
- $K$ 是更新步长

### 4.3 为什么 Arena 被认为最可靠？

| 原因 | 解释 |
|------|------|
| **真实用户** | 不是预定义的题目，而是用户想问什么就问什么 |
| **盲评** | 用户不知道模型身份，避免品牌偏见 |
| **规模大** | 数十万次对战，统计上可靠 |
| **持续更新** | 新模型随时加入，排名实时变化 |
| **开放式** | 不受固定题目和评分标准限制 |

### 4.4 局限

- 用户群体偏向英语和技术话题
- 短对话偏多，不测长期一致性
- 用户投票质量不一（有人随便点）
- 难以细粒度分析模型在具体能力上的强弱

### 4.5 当前排名趋势（2025~2026）

访问 [https://lmarena.ai](https://lmarena.ai) 查看最新排名。

---

## 五、评估体系总结对比

| 维度 | MMLU | HumanEval | MT-Bench | Arena |
|------|------|-----------|----------|-------|
| 测什么 | 知识广度 | 代码能力 | 对话质量 | 综合能力 |
| 评估方式 | 多选题准确率 | 代码通过率 | GPT-4 打分 | 人类投票 |
| 题目数 | ~14K | 164 | 80 | 无限（用户出题） |
| 自动化 | ✅ 完全自动 | ✅ 完全自动 | ✅ 自动（需 API） | ❌ 需人类参与 |
| 防数据污染 | ❌ 已知有泄漏 | ❌ 题目已公开 | ⚠️ 题目已公开 | ✅ 用户实时出题 |
| 可信度 | 中 | 中 | 中高 | **最高** |
| 适用阶段 | 发论文/报告 | 代码模型评测 | 对话模型评测 | 最终排名 |

### 评估的金字塔

```
                    ┌───────────┐
                    │  Arena    │  ← 最终裁判，最贴近真实使用
                    │ (人类评估) │
                ┌───┴───────────┴───┐
                │    MT-Bench       │  ← LLM-as-Judge，快速迭代
                │  (GPT-4 评估)     │
            ┌───┴───────────────────┴───┐
            │  MMLU / HumanEval / 基准   │  ← 自动化，大规模，但有限
            │      (自动评测)             │
        ┌───┴───────────────────────────┴───┐
        │        PPL / BLEU / ROUGE          │  ← 最基础，最有限
        │         (经典指标)                  │
        └───────────────────────────────────┘
```

---

## 六、实用工具与平台

| 工具/平台 | 用途 | 链接 |
|-----------|------|------|
| **lm-evaluation-harness** | 统一运行 MMLU / HumanEval 等基准 | [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| **Open LLM Leaderboard** | HuggingFace 模型排行榜 | [huggingface.co/spaces/open-llm-leaderboard](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard) |
| **Chatbot Arena** | 人类评估排名 | [lmarena.ai](https://lmarena.ai) |
| **LiveCodeBench** | 防污染的代码评测 | [livecodebench.github.io](https://livecodebench.github.io) |

---

## 七、自检题

1. **为什么 PPL 低不代表模型好？** 举一个具体例子。
2. **MMLU 的主要数据污染风险是什么？** 如何缓解？
3. **pass@1 和 pass@10 有什么区别？** 为什么 pass@1 更有实际意义？
4. **MT-Bench 用 GPT-4 当评委有什么偏见？** 如何缓解？
5. **为什么 Chatbot Arena 被认为最可靠？** 它有什么局限？
6. **如果你训练了一个垂域医疗 LLM，你会用什么方式评估？** 为什么？

---

## 八、产出要求

- [ ] 写一份「LLM 评估方法速查表」：每种方法一句话说明 + 适用场景
- [ ] 浏览 Chatbot Arena 排行榜，记录 Top 10 模型及其 Elo 分数
- [ ] 思考：如果你要评测一个 7B 中文模型，你会选哪些基准？为什么？
