---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - Agent
tags:
  - agent
  - python
  - llm
  - openai
title: "Learn Claude Code（一）：从 0 到 1 手写一个 Agent Loop：不到 200 行，跑通模型调工具的全流程"
createTime: 2026/07/04 13:44:40
permalink: /zh/article/agent-loop-handwritten/
---

> 很多人天天聊 Agent，但 Agent 核心的"循环"到底是怎么跑起来的，反而不清楚。
>
> 这篇文章带你手写一个最小可用的 Agent Loop。不依赖 LangChain、不依赖 CrewAI，只用 OpenAI SDK + 一个 while True，跑通"模型思考→调用工具→拿到结果→继续推理"的完整闭环。

---

## 01 这篇文章适合谁？

- 你知道大模型能调 Function Calling，但没亲手搭过完整的 Agent 循环
- 你好奇"模型怎么决定什么时候调工具，什么时候直接回答"
- 你想理解流式输出（Stream）模式下 Tool Calling 的数据流
- 你想在自己的项目里嵌入一个最简 Agent，不想上重型框架

不适合谁：

- 你要的是生产级 Agent 框架（去看 LangGraph、CrewAI）
- 你完全不想碰代码

---

## 02 最终效果是什么？

运行这个脚本后，你会得到一个**交互式终端**：

```
s01 >> 帮我看看当前目录有哪些文件
```

模型收到问题后：

1. **决定**需要执行 Shell 命令
2. **输出**要运行的命令（你会在终端看到 `$ ls`）
3. **执行**命令并拿到结果
4. **继续推理**，基于命令输出给出人类可读的回答
5. **流式输出**回答内容

整个循环完全自动化。你只负责输入问题，模型自己决定：

- 要不要调工具？
- 调什么工具？
- 工具返回后怎么继续？

![](/images/agent-loop/overview.png)

---

## 03 需要准备什么？

环境要求很低：

| 项目   | 说明                                                     |
| ------ | -------------------------------------------------------- |
| Python | 3.9+                                                     |
| API    | 任意兼容 OpenAI SDK 的大模型 API（本文以 DeepSeek 为例） |
| 依赖   | `openai`、`python-dotenv`                                |

安装依赖：

```bash
pip install openai python-dotenv
```

环境变量（`.env` 文件）：

```bash
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

> 用其他模型也一样，换 base_url 和 model 名就行。

---

## 04 项目结构长什么样？

整个脚本只有一个文件，但我们拆成 5 个模块来理解：

```text
agent_loop.py
├── ① 自定义输入函数  my_input()       # 处理终端输入（退格、Ctrl+C）
├── ② 环境与客户端    OpenAI client    # API 配置
├── ③ 工具定义        TOOLS            # 告诉模型它能调什么
├── ④ 工具执行        run_bash()       # 真正跑命令的地方
└── ⑤ Agent 循环      agent_loop()     # 核心：流式推理 + 自动调工具
```

我们逐个拆开。

![](/images/agent-loop/structure.png)

---

## 05 Step 1：自定义终端输入

为什么需要这个？

Python 自带的 `input()` 函数在终端中不支持退格键（Backspace）的正确处理——按退格会显示 `^H` 而不是删除字符。

所以我们自己实现了一个 `my_input()`，原理很简单：

```python
def my_input(prompt: str = "") -> str:
    # 把终端设为 raw 模式（逐个字符读取）
    # 自己处理：回车确认、退格删除、Ctrl+C 中断
    # 退出时自动恢复终端设置
```

关键点：

- `tty.setraw(fd)` 让终端进入"原始模式"，每个按键都即时读取
- 遇到 `\r` 或 `\n` 才结束输入
- 遇到 `\x7f`（DEL）或 `\x08`（BS）就删除缓冲区最后一个字符，并输出退格序列 `\b \b`
- 用 `try/finally` 保证即使出错也能恢复终端

> 这一步不是 Agent 核心，但它是"让命令行工具好用"的工程细节。

![](/images/agent-loop/custom-input.png)

---

## 06 Step 2：环境配置与模型客户端

这一步很简单，但很重要：

```python
load_dotenv()  # 从 .env 加载 API Key

client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL')
)

MODEL = os.getenv("DEEPSEEK_MODEL")
```

**然后用一个 system prompt 告诉模型它是什么角色：**

```python
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."
```

这句话很关键——`Act, don't explain` 让模型倾向于直接调工具而不是长篇大论。

---

## 07 Step 3：告诉模型它能用什么工具

这一步是 Agent 的"能力边界"。我们只给模型一个工具——执行 Shell 命令：

```python
TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"}
            },
            "required": ["command"],
        },
    },
}]
```

> 一个 Agent 能做多少事，取决于你给了它多少工具。这就是 Tool Calling 的起点。

---

## 08 Step 4：安全执行 Shell 命令

模型说"我要执行这个命令"，但你不能让它为所欲为。

`run_bash()` 做了两件事：

**第一，安全过滤：**

```python
# 阻止危险命令
dangerous = ["rm -rf /", "sudo ", "shutdown ", "reboot ", ...]
if any(d in command.lower() for d in dangerous):
    return "Error: Dangerous command blocked."
```

甚至分了 Linux 和 Windows 两套危险命令列表。

**第二，执行并返回结果：**

```python
r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
out = (r.stdout + r.stderr).strip()
return out[:50000] if out else "No output."
```

关键细节：

- `timeout=120`——防止命令卡死
- 输出截断到 50000 字符——防止模型被超长输出冲晕

![](/images/agent-loop/run-bash.png)

---

## 09 Step 5：核心——Agent Loop

这是整篇文章的**灵魂**。

```python
def agent_loop(messages: list):
```

这个函数只有 50 行左右，但它是一个完整的 **ReAct 循环**。

### 循环是怎么跑的？

整个过程可以拆成四步：

```
① 发起推理请求（Stream 模式）
    ↓
② 流式接收模型输出（普通文字 或 Tool Calling 指令）
    ↓
③ finish_reason 判断
   ├─ "tool_calls" → 执行工具，把结果塞回 messages，继续循环
   └─ "stop"       → 返回最终答案，结束
```

这里重点看一下**流式处理 Tool Calling**：

当模型决定调工具时，它返回的不是普通 content，而是 `tool_calls` 增量数据。

```python
for chunk in stream:
    delta = chunk.choices[0].delta

    # 流式输出普通文字
    if delta.content:
        print(delta.content, end="", flush=True)

    # 流式累积 Tool Calling 参数
    if delta.tool_calls:
        # 把多个 chunk 的 tool_calls 片段拼起来
        tool_call_deltas[idx]["function"]["arguments"] += tc.function.arguments
```

这就是**流式 Function Calling**——模型一边"想"参数，你一边收到片段，最后拼成完整的 JSON。

### 判断循环方向

```python
if finish_reason != "tool_calls":
    return  # 模型说够了，结束

# 否则执行工具，继续循环
for tc in assistant_msg.get("tool_calls", []):
    args = json.loads(tc["function"]["arguments"])
    output = run_bash(args["command"])
    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": output})
```

**核心逻辑就是：只要模型返回 `finish_reason="tool_calls"`，就执行工具、塞回结果、继续循环。**

![](/images/agent-loop/core-loop.png)

---

## 10 Step 6：把一切串起来

主函数做的事情很纯粹：

```python
history = []
while True:
    query = my_input("s01 >> ")
    if query.strip().lower() in ("q", "exit", ""):
        break
    history.append({"role": "user", "content": query})
    agent_loop(history)
```

- 用户输入问题
- 加入消息历史
- 启动 Agent Loop
- 循环结束后，打印模型的最终回答
- 等待下一个问题

---

## 11 运行和测试

启动后，你会看到：

```
s01 >> 当前目录是什么？
```

模型会：

1. 调用 `bash` 工具执行 `pwd`（你会在终端看到黄色高亮的 `$ pwd`）
2. 得到输出后继续推理，告诉你当前目录
3. 流式输出最终答案

你还可以试试：

```
s01 >> 帮我创建一个 test.txt 文件，里面写入 "Hello Agent Loop"
```

模型会自动完成：`echo "Hello Agent Loop" > test.txt`

> 你看到的是黄色的 `$ echo "Hello Agent Loop" > test.txt` 在执行，然后模型告诉你"已创建"。

---

## 12 完整代码

以下是 `agent_loop.py` 的完整代码，可以直接复制保存运行：

```python
import os
import sys
import atexit
import subprocess


# ── 自定义输入函数（彻底避免退格问题） ──
def my_input(prompt: str = "") -> str:
    """
    替代 builtins.input()，自己处理字符回显和退格，
    不依赖 readline / termios 驱动层配置。
    """
    import termios
    import tty

    # 不是 TTY 就回退到标准 input
    if not sys.stdin.isatty():
        return input(prompt)

    sys.stdout.write(prompt)
    sys.stdout.flush()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf: list[str] = []

    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            # 回车 → 结束
            if ch in ("\r", "\n"):
                sys.stdout.write("\r\n")
                break
            # Ctrl+C → KeyboardInterrupt
            if ch == "\x03":
                raise KeyboardInterrupt
            # Ctrl+D → EOFError
            if ch == "\x04":
                raise EOFError
            # 退格：DEL (127) 或 BS (8)
            if ord(ch) in (127, 8):
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b")
            # 常规字符
            else:
                buf.append(ch)
                sys.stdout.write(ch)
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return "".join(buf)


# 确保退出时恢复终端
atexit.register(lambda: None)


from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL')
)

MODEL = os.getenv("DEEPSEEK_MODEL")

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]

def run_bash(command: str) -> str:
    # OS-aware dangerous command patterns
    dangerous = [
        "rm -rf /", "sudo ", "shutdown", "reboot", "> /dev/sd",
        "dd if=", "mkfs", ":(){", "chmod 777 /", "poweroff",
        "halt", "init 0", "init 6",
    ]
    if os.name == "nt":
        dangerous = [
            "del /f /s", "rd /s /q", "format ", "diskpart",
            "shutdown", "reg delete", "net user", "taskkill /f",
        ]

    if any(d in command.lower() for d in dangerous):
        return "Error: Dangerous command blocked."

    try:
        if os.name == "nt":
            shell = "powershell.exe"
            cmd = (
                "$OutputEncoding = [Console]::OutputEncoding = "
                "[System.Text.Encoding]::UTF8; " + command
            )
        else:
            shell = "/bin/bash"
            cmd = command

        r = subprocess.run(
            cmd, shell=True, executable=shell,
            cwd=os.getcwd(), capture_output=True, text=True,
            encoding="utf-8", timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "No output."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (120s)."
    except (FileNotFoundError, OSError) as e:
        return f"Error: {str(e)}"


def agent_loop(messages: list):
    import json

    while True:
        content_buf = []
        tool_call_deltas = {}

        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
            temperature=0.2,
            extra_body={"thinking": {"type": "enabled"}},
        )

        finish_reason = None

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            if delta.content:
                content_buf.append(delta.content)
                print(delta.content, end="", flush=True)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_deltas:
                        tool_call_deltas[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                    if tc.id:
                        tool_call_deltas[idx]["id"] += tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_call_deltas[idx]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_call_deltas[idx]["function"]["arguments"] += tc.function.arguments

        print()

        full_content = "".join(content_buf) if content_buf else None
        assistant_msg = {"role": "assistant", "content": full_content}
        if tool_call_deltas:
            assistant_msg["tool_calls"] = [
                {
                    "id": v["id"],
                    "type": "function",
                    "function": {
                        "name": v["function"]["name"],
                        "arguments": v["function"]["arguments"],
                    },
                }
                for _, v in sorted(tool_call_deltas.items())
            ]

        messages.append(assistant_msg)

        if finish_reason != "tool_calls":
            return

        results = []
        for tc in assistant_msg.get("tool_calls", []):
            args = json.loads(tc["function"]["arguments"])
            command = args["command"]
            print(f"\033[33m$ {command}\033[0m")
            output = run_bash(command)
            print(output[:200])
            results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": output,
            })

        messages.extend(results)

if __name__ == "__main__":
    print("s01: Agent Loop")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = my_input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
```

保存为 `agent_loop.py`，然后在终端执行：

```bash
python agent_loop.py
```

记得在同一个目录下创建 `.env` 文件并写入你的 API Key。

---

## 13 常见问题和排查

| 问题         | 排查方向                                                              |
| ------------ | --------------------------------------------------------------------- |
| 模型不调工具 | 检查 `tool_choice="auto"` 是否设置，system prompt 是否包含 "use bash" |
| 流式输出乱码 | 确认终端编码为 UTF-8，Windows 需设置 `$OutputEncoding`（代码已处理）  |
| 退格键不正常 | 确保 `my_input` 正常生效，终端支持 raw 模式                           |
| API 报错     | 检查 `.env` 文件中的 Key 和 Base URL                                  |
| 命令超时     | `timeout=120` 可调整，或检查命令是否阻塞                              |

---

## 14 还能怎么优化？

这个 Agent Loop 是最简原型，你可以沿这些方向升级：

**1. 增加更多工具**

不只是 bash，可以加文件读写、网络请求、代码执行沙箱等。

**2. 持久化对话历史**

把 messages 存到文件或数据库，让 Agent 有长期记忆。

**3. 支持多轮工具链**

当前是一轮调一个工具。可以升级成"模型连续调多个工具，再综合结果回答"。

**4. 更好的输出体验**

用 rich 库做更漂亮的终端输出，或者接入 Web UI。

**5. 错误重试机制**

工具执行失败时，让模型尝试修正参数重新调用。

---

## 15 总结

这篇文章的核心就一句话：

> **Agent Loop 的本质是：模型推理 → 判断要不要调工具 → 调工具 → 把结果给模型继续推理 → 循环直到模型说够了。**

整个实现只有三个关键代码段：

1. `TOOLS` 定义——告诉模型它能做什么
2. `agent_loop()` 中的 `finish_reason` 判断——决定循环还是结束
3. `run_bash()`——真正执行工具并返回结果

**200 行代码，一个完整的 Agent 就活了。**

当然它不是生产级的，但它帮你拆掉了 Agent 最神秘的那层窗户纸。下次再有人问"Agent 到底怎么循环的"，你可以说：就是一个 while True，判断 finish_reason 是不是 tool_calls。
