# Day 2：Agent 原理与 ReAct — 从“回答问题”到“调用工具”

> **目标**：理解 Agent 的本质不是“更聪明的 prompt”，而是“LLM + 状态机 + 工具接口”的系统组合；掌握 ReAct、Function Calling、Tool Use 的核心思想；理解 Agent Loop、工具注册、参数解析与 Memory 的基本设计；为 Day3 手写 Agent 实现打下理论基础。

---

## 一、什么是 Agent

### 1.1 Agent 不只是聊天机器人

普通聊天模型的工作方式通常是：

```text
用户问题 -> LLM 直接生成答案
```

但很多任务只靠模型参数内部的知识并不够，例如：

- 查询今天的天气
- 搜索数据库
- 调用计算器
- 读取企业知识库
- 执行多步任务

这时就需要：

```text
用户问题 -> LLM 判断需要什么动作 -> 调用工具 -> 读取结果 -> 继续推理 -> 给出答案
```

这就是 Agent 的核心思想。

### 1.2 Agent 的最小定义

一个最小 Agent 可以看成三部分：

1. `Planner`：决定下一步要做什么
2. `Tool Executor`：真正执行外部动作
3. `Memory / State`：保存任务上下文和中间结果

用公式化的方式表示：

$$
\text{Action}_t = f_\theta(\text{Query}, \text{History}, \text{Observation}_{<t}, \text{ToolSpec})
$$

Agent 的关键不是“模型会不会答题”，而是“模型会不会在合适的时候做正确动作”。

---

## 二、ReAct：Reasoning + Acting

### 2.1 ReAct 的核心思想

ReAct 论文提出，模型在执行任务时应交替进行：

- `Thought`：当前如何思考
- `Action`：下一步执行什么动作
- `Observation`：动作返回了什么结果

整体循环如下：

```text
Question
  ↓
Thought
  ↓
Action
  ↓
Observation
  ↓
Thought
  ↓
Action
  ↓
Observation
  ↓
Final Answer
```

### 2.2 为什么 ReAct 比纯 CoT 更强

纯 CoT（Chain-of-Thought）只是在模型内部推理：

```text
问题 -> 思维链 -> 最终答案
```

ReAct 在此基础上加入外部交互：

```text
问题 -> 思维链 -> 调工具 -> 观察结果 -> 更新思维链 -> 最终答案
```

因此它在以下场景中更强：

- 需要最新信息
- 需要查证事实
- 需要多步操作
- 需要访问外部环境

### 2.3 一个 ReAct 示例

```text
Question: 北京今天适合跑步吗？

Thought: 我需要先知道今天北京的天气。
Action: weather_api(city="北京")
Observation: 多云，18℃，空气质量良，风力 2 级

Thought: 天气温和、空气质量较好，适合户外运动。
Final Answer: 今天北京适合跑步，建议选择早晚温度更舒适的时段。
```

### 2.4 ReAct 的优势与风险

优势：

- 可解释性更好
- 易于插入工具
- 能处理信息不足的问题

风险：

- 推理链可能冗长
- 工具调用次数过多会拖慢响应
- 工具返回脏数据时可能误导模型

---

## 三、Tool Use 与 Function Calling

### 3.1 Tool Use 的本质

Tool Use 指模型不直接生成自然语言答案，而是生成一个“结构化动作请求”。

例如：

```json
{
  "tool_name": "search_docs",
  "arguments": {
    "query": "什么是向量数据库"
  }
}
```

系统接收到这段结构化输出后，调用对应工具，再把结果交还给模型。

### 3.2 Function Calling 是 Tool Use 的工程化接口

Function Calling 的本质是：提前告诉模型有哪些函数、每个函数的参数结构是什么，让模型用结构化 JSON 形式输出调用请求。

最常见的描述方式是 JSON Schema：

```json
{
  "name": "search_docs",
  "description": "在课程文档中检索相关内容",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "用户的问题或关键词"
      },
      "top_k": {
        "type": "integer",
        "description": "返回结果数量"
      }
    },
    "required": ["query"]
  }
}
```

### 3.3 为什么要让模型输出结构化参数

如果模型直接输出：

```text
去搜索一下“向量数据库”的定义吧
```

系统很难稳定解析。

如果模型输出：

```json
{"name": "search_docs", "arguments": {"query": "向量数据库", "top_k": 3}}
```

系统就可以稳定执行：

1. 校验参数是否合法
2. 调用函数
3. 捕获异常
4. 返回 observation

这就是 Function Calling 的价值。

---

## 四、Agent Loop：一个 Agent 是如何跑起来的

### 4.1 最小闭环

一个典型的 Agent Loop 如下：

```text
用户问题
  ↓
LLM 生成:
  - 直接回答
  或
  - 调用某个工具
  ↓
如果是工具调用:
  执行工具
  返回 observation
  再次喂给 LLM
  ↓
直到:
  - 生成 Final Answer
  - 达到最大步骤数
```

### 4.2 伪代码

```python
def agent_loop(user_query, tools, llm, max_steps=5):
    messages = [{"role": "user", "content": user_query}]

    for step in range(max_steps):
        response = llm(messages, tools=tools)

        if response["type"] == "final_answer":
            return response["content"]

        if response["type"] == "tool_call":
            tool_name = response["name"]
            arguments = response["arguments"]
            observation = tools[tool_name](**arguments)

            messages.append({"role": "assistant", "content": str(response)})
            messages.append({"role": "tool", "content": str(observation)})

    return "未能在最大步骤数内完成任务。"
```

### 4.3 为什么必须限制最大步骤数

否则模型可能陷入：

- 无限思考
- 循环调用同一个工具
- 无意义地重复检索

因此工程上常见的保护措施包括：

- `max_steps`
- `max_tool_calls`
- 超时控制
- 工具白名单

---

## 五、Memory：Agent 为什么比普通问答更依赖状态管理

### 5.1 短期记忆

短期记忆通常就是当前任务上下文：

- 用户最近几轮对话
- 工具调用结果
- 当前子任务进展

它一般直接保存在 history 中。

### 5.2 长期记忆

长期记忆指可跨轮甚至跨会话保存的信息，例如：

- 用户偏好
- 用户身份信息
- 历史任务记录
- 已构建的知识索引

长期记忆的实现形式通常是：

- 数据库
- KV 存储
- 向量数据库
- 文件系统

### 5.3 Agent 的 Memory 与 Day1 的关系

Day1 里我们管理的是纯对话 history，到了 Agent 阶段，history 中除了用户和助手消息，还会出现：

- 工具调用请求
- 工具返回结果
- 中间推理轨迹

这使得上下文管理更复杂，也更容易爆掉 token 预算。

---

## 六、Agent 设计中的几个关键工程点

### 6.1 工具注册

系统需要提前维护一张工具表：

```python
TOOLS = {
    "calculator": calculator_tool,
    "search_docs": search_docs_tool,
    "weather_api": weather_api_tool,
}
```

这样模型输出工具名后，系统才能路由到真正的函数。

### 6.2 参数校验

不能盲目执行模型给出的参数。必须校验：

- 参数名是否正确
- 参数类型是否匹配
- 是否缺失必填字段
- 是否存在危险输入

### 6.3 观测结果的格式

工具返回值也需要统一格式，否则模型很难稳定消费 observation。

推荐做法：

```json
{
  "status": "ok",
  "result": "...",
  "metadata": {
    "source": "weather_api"
  }
}
```

### 6.4 错误处理

工具可能失败，因此 Agent 系统需要：

- 捕获异常
- 告诉模型“工具失败了”
- 让模型决定是否重试、换工具或直接解释失败原因

---

## 七、ReAct、Function Calling 与工作流的关系

### 7.1 三者不是互斥的

这三者常常同时出现：

- ReAct：定义“思考 + 行动”的循环范式
- Tool Use：定义“模型可以调用外部能力”
- Function Calling：定义“工具调用如何结构化落地”

可以把它们理解为不同层级：

```text
范式层: ReAct
接口层: Function Calling
执行层: Tool Executor
```

### 7.2 从 Agent 到 Workflow

当任务变复杂时，单个 Agent 可能不够，需要引入固定流程：

```text
分类 -> 检索 -> 工具调用 -> 总结
```

这时系统就从“纯 Agent”走向了“Agent + Workflow”的混合架构。

---

## 八、与 Day3 的衔接

Day2 解决的是“Agent 为什么成立”，Day3 解决的是“Agent 如何从零写出来”。

Day3 将具体实现：

- Tool 类与注册机制
- ReAct prompt 设计
- Agent Loop
- 工具路由与执行
- 简单 Memory 管理
- 基础评测样例

---

## 九、自检题

1. Agent 和普通聊天机器人最大的区别是什么？
2. 为什么说 ReAct 本质上是在 CoT 外面接了一个行动回路？
3. Function Calling 为什么要用结构化 Schema？
4. Agent Loop 为什么必须限制最大步骤数？
5. 工具返回 observation 时为什么要统一格式？
6. 短期记忆和长期记忆分别适合保存什么信息？
7. ReAct、Tool Use、Function Calling 三者是什么关系？
8. 为什么 Day1 的上下文管理能力在 Agent 中会变得更重要？
9. 一个好的 Agent 为什么既要“会思考”，又要“会停下”？
10. 哪些任务更适合 Workflow，而不是完全自由的 Agent？

---

## 十、产出要求

- [ ] 理解 Agent 的系统组成：LLM + Tool + State
- [ ] 说明 ReAct 的 Thought / Action / Observation 循环
- [ ] 解释 Function Calling 的 Schema 设计意义
- [ ] 手写 Agent Loop 的核心伪代码
- [ ] 明确短期记忆与长期记忆的区别
- [ ] 为 Day3 的手写 Agent 实现做好接口设计准备

