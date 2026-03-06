# Day 1：多轮对话系统管理 — history、截断策略与部署基础

> **目标**：理解为什么 LLM 天生是“无状态”的；掌握多轮对话系统如何通过 history 管理、上下文窗口截断和对话模板实现连续交流；理解本地对话系统的基本部署架构；手写一个最小可用的多轮对话系统，为 Day2-Day7 的 Agent 与 RAG 铺路。

---

## 一、为什么多轮对话是一个“系统问题”

### 1.1 LLM 本身并不会记住上一轮

大语言模型本质上是一个条件概率模型：

$$
P(x_t \mid x_{<t})
$$

它在生成第 $t$ 个 token 时，只依赖当前输入序列 $x_{<t}$，并不会自动保存“会话状态”。

这意味着：

- 模型没有天然的 session
- 模型不会自动记住上一轮用户说过的话
- 所谓多轮对话，本质上是系统把历史对话重新拼接回输入

```
第 1 轮:
  用户: 我叫小明
  助手: 你好，小明

第 2 轮:
  用户: 我叫什么名字？

如果系统只把“我叫什么名字？”发给模型，
模型并不知道“小明”来自哪里。

只有当系统发送:
  [user] 我叫小明
  [assistant] 你好，小明
  [user] 我叫什么名字？
模型才有机会回答“你叫小明”。
```

### 1.2 多轮对话系统的职责

一个完整的对话系统至少要负责以下四件事：

1. 保存对话 history
2. 控制上下文长度，避免超过模型窗口
3. 按模型要求组织 prompt / chat template
4. 调用推理引擎，并把回复回写到 history

因此，多轮对话不是单纯的“prompt 工程”，而是一个结合了状态管理、长度控制和推理调用的系统工程。

---

## 二、history 管理的核心问题

### 2.1 最朴素的方案：完整保留所有历史

最直接的做法是每一轮都发送全部历史：

```python
messages = [
    {"role": "system", "content": system_prompt},
    *history,
    {"role": "user", "content": user_input},
]
```

优点：

- 实现简单
- 信息最完整
- 最不容易丢失上下文

缺点：

- token 数会随轮次线性增长
- 推理成本越来越高
- 很容易超出上下文窗口

### 2.2 token 增长为什么会成为瓶颈

设平均每轮用户输入 40 tokens，模型回复 120 tokens，那么每轮大约新增 160 tokens。

如果模型上下文窗口是 4096，且预留 512 tokens 作为生成空间，则 history 的预算大约是：

$$
4096 - 512 = 3584
$$

能容纳的轮次数大致为：

$$
3584 / 160 \approx 22
$$

这说明看似很长的窗口，在真实多轮对话里很快就会被用完。

### 2.3 需要预留 generation budget

多轮对话系统不能把整个窗口都拿来放 history，因为还要给模型留出生成回复的空间。

常见预算分配：

```text
总窗口: 4096 tokens
  - system prompt: 256
  - history: 3000
  - 当前用户输入: 300
  - 生成空间: 540
```

工程上通常先预留 `max_new_tokens`，然后在剩余预算里安排 history。

---

## 三、三种经典的上下文管理策略

### 3.1 滑动窗口（Sliding Window）

只保留最近若干轮对话。

```python
def sliding_window(messages, max_turns=3):
    if len(messages) <= max_turns * 2:
        return messages
    return messages[-(max_turns * 2):]
```

优点：

- 实现最简单
- 没有额外计算开销
- 适合原型验证

缺点：

- 早期关键信息会被直接丢弃
- 用户身份、长期偏好容易遗忘

### 3.2 Token 级截断（Token-level Truncation）

按 token 预算从后往前保留消息，直到预算耗尽。

```python
def truncate_by_tokens(messages, tokenizer, max_tokens):
    selected = []
    total = 0
    for msg in reversed(messages):
        msg_tokens = len(tokenizer.encode(msg["content"]))
        if total + msg_tokens > max_tokens:
            break
        selected.insert(0, msg)
        total += msg_tokens
    return selected
```

优点：

- 比固定轮次更精细
- 更贴近真实上下文预算
- 工业系统中最常见

缺点：

- 依赖 tokenizer
- 仍然会直接丢失早期信息

### 3.3 摘要压缩（Summary Compression）

对较早的对话做摘要，只保留最近若干轮完整内容。

```
原始 history (3000 tokens):
  [u1: 介绍自己是一名医生，在北京工作]
  [a1: 打招呼，询问需求]
  [u2: 想了解心脏病的治疗方案]
  [a2: 详细介绍了三种方案...]
  ... (后续 8 轮详细讨论)

摘要压缩后 (~300 tokens):
  [摘要: 用户是一名北京医生，正在了解心脏病治疗方案。
   已讨论过药物治疗、介入手术、搭桥手术三种方案的优缺点。
   用户对微创介入方案最感兴趣。]
  + [最近 3 轮完整对话]
```

```python
SUMMARY_PROMPT = """请将以下对话历史压缩为一段简洁摘要，保留：
- 用户身份、需求和偏好
- 已达成的结论
- 重要上下文

对话历史:
{conversation}

摘要:
"""

def summary_compression(messages, max_recent_turns=3, summarizer_fn=None):
    if len(messages) <= max_recent_turns * 2:
        return messages

    early_messages = messages[:-(max_recent_turns * 2)]
    recent_messages = messages[-(max_recent_turns * 2):]

    conversation_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in early_messages
    )
    summary = summarizer_fn(SUMMARY_PROMPT.format(conversation=conversation_text))

    return [{"role": "system", "content": f"[对话摘要] {summary}"}] + recent_messages
```

优点：

- 能保留长期关键信息
- token 压缩率高
- 适合长会话

缺点：

- 需要额外一次 LLM 调用
- 摘要可能遗漏细节
- 多次摘要可能产生“电话游戏效应”

### 3.4 三种策略对比

| 维度 | 滑动窗口 | Token 级截断 | 摘要压缩 |
|------|---------|------------|---------|
| 实现难度 | 低 | 中 | 高 |
| 信息保留 | 低 | 中 | 高 |
| 额外开销 | 无 | tokenizer 编码 | 额外 LLM 调用 |
| 延迟 | 低 | 低 | 高 |
| 适用场景 | 简单原型 | 通用生产 | 长对话 / 高价值会话 |

### 3.5 混合策略：工业系统的主流做法

真实系统往往组合使用多种策略：

```text
1. system prompt 永远保留
2. 最近 2~3 轮完整保留
3. 中间历史做 token 级截断
4. 更早历史做摘要压缩或长期记忆存储
```

核心原则：

- 当前轮相关信息优先级最高
- system prompt 和安全规则不能丢
- 能压缩的尽量压缩，不能丢的必须保留

---

## 四、对话模板格式

### 4.1 为什么需要 chat template

模型本质上只看到一串 token，并不知道哪些 token 来自用户，哪些来自助手。chat template 的作用是显式标注角色边界。

```text
没有模板:
  我叫小明你好小明我想学 Python

有模板:
  <|user|>我叫小明<|assistant|>你好小明<|user|>我想学 Python
```

### 4.2 ChatML 格式

```text
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
什么是机器学习？<|im_end|>
<|im_start|>assistant
机器学习是人工智能的一个分支...<|im_end|>
```

特点：

- 消息边界清晰
- 原生适配多轮对话
- OpenAI / Qwen 体系常见

### 4.3 LLaMA-2 Chat 格式

```text
<s>[INST] <<SYS>>
你是一个有帮助的助手。
<</SYS>>

什么是机器学习？ [/INST] 机器学习是人工智能的一个分支... </s>
```

特点：

- `[INST]` 包裹用户输入
- system prompt 放在首轮
- 与 LLaMA-2 chat 系列训练格式一致

### 4.4 Alpaca 格式

```text
### Instruction:
什么是机器学习？

### Input:
(可选输入)

### Response:
机器学习是人工智能的一个分支...
```

特点：

- 结构清晰
- 更适合单轮 instruction tuning
- 不适合长多轮对话

### 4.5 使用 tokenizer 自动应用模板

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "什么是机器学习？"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

工程上通常不建议手写模板字符串，而是优先复用模型官方模板。

---

## 五、本地对话系统部署架构

### 5.1 基本架构

```text
用户输入
  ↓
对话管理器
  - 保存 history
  - 做截断与压缩
  ↓
模板引擎
  - 应用 chat template
  ↓
推理引擎
  - Transformers / vLLM / llama.cpp
  ↓
后处理
  - 停止词
  - 文本清洗
  ↓
写回 history
```

### 5.2 推理引擎选型

| 引擎 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| HuggingFace Transformers | 灵活、研究友好 | 吞吐较低 | 教学 / 原型 |
| vLLM | 吞吐高、连续批处理 | 部署复杂一些 | 服务化 |
| llama.cpp / GGUF | CPU 也能跑 | 扩展性一般 | 本地运行 |
| Ollama | 开箱即用 | 自定义能力弱 | 个人实验 |

### 5.3 几个关键工程问题

1. 流式输出：用户不想等完整生成后再看到结果。
2. 停止条件：需要 `max_new_tokens`、`eos_token` 或 `stop strings`。
3. 并发处理：多用户场景需要队列和批处理。
4. KV Cache：上下文越长，推理显存与延迟越高。

---

## 六、实践：构建简单的多轮对话系统

### 6.1 最小可用实现

```python
class SimpleChatSystem:
    """最简单的多轮对话系统"""

    def __init__(self, model, tokenizer, system_prompt="你是一个有帮助的AI助手。",
                 max_context=4096, max_new_tokens=512):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.history = []

    def chat(self, user_input):
        self.history.append({"role": "user", "content": user_input})

        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        messages = self._truncate(messages)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        self.history.append({"role": "assistant", "content": response})
        return response

    def _truncate(self, messages):
        """按 token 预算保留最近消息"""
        budget = self.max_context - self.max_new_tokens
        system_msg = messages[0]
        system_tokens = len(self.tokenizer.encode(system_msg["content"]))
        remaining_budget = budget - system_tokens

        selected = []
        total = 0
        for msg in reversed(messages[1:]):
            msg_tokens = len(self.tokenizer.encode(msg["content"]))
            if total + msg_tokens > remaining_budget:
                break
            selected.insert(0, msg)
            total += msg_tokens

        return [system_msg] + selected

    def reset(self):
        self.history = []
```

### 6.2 使用示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("model_path", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("model_path")

chat = SimpleChatSystem(model, tokenizer)

print(chat.chat("你好，我叫小明"))
print(chat.chat("我叫什么名字？"))
print(chat.chat("帮我写一首关于春天的诗"))
```

### 6.3 这个最小系统的局限

- 只能保留短期 history
- 没有摘要压缩
- 没有长期记忆
- 没有工具调用
- 没有检索增强

这正是后面 Agent 和 RAG 要解决的问题。

---

## 七、与后续内容的衔接

```text
Day 1:
  多轮对话 = history 管理 + 截断策略 + 模板 + 推理调用

Day 2-3:
  在这个基础上加入 Tool Use / ReAct 循环
  → 对话系统升级为 Agent

Day 4-6:
  在这个基础上加入外部知识检索
  → 对话系统升级为 RAG

Day 7:
  把对话、Agent、RAG 整合成统一系统
```

---

## 八、自检题

1. 为什么说 LLM 是无状态的？
2. 多轮对话为什么本质上依赖 history 拼接？
3. 滑动窗口、Token 级截断、摘要压缩分别适合什么场景？
4. 为什么要预留 `max_new_tokens`？
5. chat template 的作用是什么？
6. ChatML 和 LLaMA-2 Chat 模板有什么差异？
7. 本地对话系统为什么需要流式输出？
8. KV Cache 为什么会随着对话增长？
9. 为什么说 Day1 讲的是 Agent 和 RAG 的共同基础设施？
10. 这个 `SimpleChatSystem` 还缺少哪些关键能力？

---

## 九、产出要求

- [ ] 理解 LLM 的无状态本质，并能解释多轮对话如何实现
- [ ] 对比三种上下文管理策略的原理与优缺点
- [ ] 说明 chat template 在对话系统中的作用
- [ ] 画出本地多轮对话系统的基本架构
- [ ] 手写一个最小可用的 `SimpleChatSystem`
- [ ] 明确 Day2-Day7 将在 Day1 的基础上分别补什么能力

