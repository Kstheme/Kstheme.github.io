---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - 机器学习
tags:
  - context engineering
  - llm
  - 李宏毅
title: '李宏毅2026 Context Engineering：让AI学会"遗忘"与"聚焦"的核心技术'
createTime: 2026/06/16 15:18:20
permalink: /zh/article/context-engineering/
---

## 一、 为什么大语言模型需要上下文管理？

所有大语言模型的输入长度都有上限，但在处理复杂任务时，需要的对话历史和中间信息（称为"上下文"或 Context）往往会超过这个限制。如果不加管理，模型很快就会"失忆"或"跑题"。

我们可以把 AI 的工作流程想象成一个循环：

**无管理状态（原始流程）**

```text
I₁ ← 初始任务输入
C₁ ← 空的上下文

循环 t = 1, 2, 3...
    Oₜ = LLM(Iₜ, Cₜ)  （模型根据当前输入和上下文生成输出）
    Cₜ₊₁ ← Cₜ | Iₜ | Oₜ  （粗暴地把所有历史都堆在一起）
```

这种方式会迅速导致上下文爆炸，最终超出模型的处理能力。

**有管理状态（Context Engineering）**

```text
I₁ ← 初始任务输入
C₁ ← 空的上下文

循环 t = 1, 2, 3...
    Oₜ = LLM(Iₜ, Cₜ)
    Cₜ₊₁ ← F(Cₜ, Iₜ, Oₜ)  （核心！通过一个函数 F 来"修剪"和"整理"上下文）
```

这里的 **`F`** 就是 Context Engineering 的核心，其最主要的功能是**压缩（Summarization）**，即把冗长的历史对话和中间结果提炼成简短、有用的摘要。

---

## 二、 两大主流压缩技术：摘要 vs. 掩码

### 摘要式压缩（Summarization）

最直观的方法，就是让另一个大语言模型来给历史对话写摘要。例如 OpenClaw 等框架的做法。

**优点**：能提炼核心信息，保留任务逻辑。

**缺点**：本身需要消耗额外的计算（调用一次 LLM），且摘要可能丢失关键细节。

### 掩码/替换式压缩（Observation Masking）

更"简单粗暴"但有效的方法：直接把又长又臭的工具输出（如代码运行日志、文档内容）替换成一句提示语，比如"**此处曾有工具 A 的输出，详见外部日志文件**"。

![](/images/context-engineering/masking-diagram.png)

在著名的 **SWE-bench**（一个让 AI 修复 GitHub Issue 的基准测试）上测试发现，这种生硬的替换，其效果和做完整摘要差不多，但成本（花费的 Token 和费用）要低得多。

![](/images/context-engineering/swe-bench-result.png)

*参考文献：[LLM Context Compression with Observation Masking](https://arxiv.org/abs/2508.21433)*

**优点**：成本极低，效果不减。

**缺点**：信息完全丢失，如果 AI 后续需要用到这些细节，就必须有"读取日志"的机制。

### 组合拳：先掩码，后摘要

更优的策略是结合两者：**前期用廉价的掩码法控制长度，当上下文再次膨胀到临界点时，再启动一次性的摘要进行深度压缩。**

![](/images/context-engineering/mask-then-summarize.png)

---

## 三、 压缩的副作用：轨迹延长与重复劳动

压缩并非只有好处。过度压缩会带来一个严重问题：**轨迹延长（Trajectory Elongation）**。

想象一下，AI 在解决一个复杂 bug，它调用了 10 次工具。你把前 9 次的工具输出都压缩（摘要或掩码）了。结果，AI"忘记"自己已经试过某些方法，在第 11 步时，又去重复调用第 2 步用过的工具。

**这就导致：虽然每次单步的上下文变短了（省钱），但完成任务需要的总步数变多了（费时）。一增一减，总成本可能没变，甚至更高。**

---

## 四、 记忆系统：上下文的外挂硬盘

更系统的做法是引入**记忆（Memory）**概念。把上下文 `C` 拆成两部分：

- **`P`**（Prompt / Working Memory）：真正输入给模型、参与本次计算的"工作记忆"。
- **`M`**（Memory）：存储在外部（如数据库、向量库、文件）的"长期记忆"。

公式变为：

```text
I₁ ← 初始输入
C₁ = {P₁, M₁} ← {空, 空}

循环 t = 1, 2, 3...
    Oₜ = LLM(Iₜ, Pₜ)  （模型只看到工作记忆 P）
    Cₜ₊₁ = {Pₜ₊₁, Mₜ₊₁} ← F({Pₜ, Mₜ}, Iₜ, Oₜ)
```

**核心思想**：不是所有经历过的事情都需要"记住"，只把最重要的精华放进工作区 `P`，其余存入外挂"硬盘" `M`，需要时再按需读取。

![](/images/context-engineering/memory-system.png)

*参考文献：A-MEM, Mem0, Memory OS 等论文*

---

## 五、 一个难题：AI 自己不愿"遗忘"

研究发现，**大语言模型有"记忆癖"**，它本能地不愿意主动压缩或删除历史信息，因为这感觉像是在"抹除记忆"。即使你在系统指令里明确告诉它"现在该压缩了"，它也可能找借口去干别的事。

**解决方案：专项训练（微调）**。

研究者通过**强化学习（Reinforcement Learning）**专门训练模型，让它学会在适当的时候"调用压缩工具"。例如 **AgentFold** 这项工作，就是训练模型自己决定何时压缩、如何写压缩后的提示条。

![](/images/context-engineering/agentfold-training.png)

*参考文献：[AgentFold: Teaching LLMs to Compress Context](https://arxiv.org/abs/2510.24699)*

---

## 六、 Subagent：一种优雅的"自主压缩"

另一个巧妙的思路是**召唤子代理（Spawn Subagent）**。

主 AI 遇到一个复杂的子任务时，可以"召唤"一个子 AI（Subagent）去专门处理。子 AI 完成任务后，只向主 AI 返回一个最终结果（如"已修复 XX bug"），然后自身"消散"。对于主 AI 而言，整个复杂的子过程被压缩成了一句结果汇报。

![](/images/context-engineering/subagent-workflow.png)

这本质上是一种**结构化的、任务导向的上下文压缩**。同样，这种能力通常也需要通过强化学习进行专项训练才能获得。

---

## 七、 治本之道：源头过滤与按需加载

压缩是"治标"，更根本的方法是**不让垃圾信息进入上下文**。分析发现，上下文中占比最大（常超 80%）的是来自外部的**观察（Observation）**，比如读取的代码文件、运行日志、文档内容。

![](/images/context-engineering/context-composition.png)

### 智能读取（Intelligent Read）

传统工具：`read(log.txt)` → 返回整个 1000 行的日志文件。

智能工具：`read(log.txt, about="bug fix")` → 只返回与 bug 修复相关的 10 行关键日志。

这需要工具本身具备一定的理解能力（例如内置一个小型微调模型）。

![](/images/context-engineering/intelligent-read.png)

*参考文献：[Selective Context: Filtering Irrelevant Information](https://arxiv.org/abs/2601.16746)*

### 动态工具加载

系统指令（System Prompt）里如果塞了几百个工具的说明，会极其冗长。**MCP-Zero** 等研究提出：工具应该按需动态加载。AI 先根据任务思考"我需要什么工具"，然后通过一个"工具搜索引擎"只加载相关的几个工具说明到上下文中。

![](/images/context-engineering/dynamic-tool-loading.png)

*参考文献：[MCP-Zero: Dynamic Tool Loading](https://arxiv.org/abs/2506.01056)*

---

## 八、 终极形态：让 AI 自己管理上下文（Agentic Context Engineering）

既然 AI 这么聪明，为什么不把上下文管理的策略也交给 AI 自己来学习和决定呢？这就是**自主化上下文工程**的理念。

![](/images/context-engineering/agentic-architecture.png)

![](/images/context-engineering/agentic-workflow.png)

1. **早期尝试 - Dynamic Cheatsheet**：直接把当前上下文和历史给 AI，问它"你觉得该怎么整理和压缩，才能更好地帮你完成后续任务？"然后让它输出一个新的、整理好的上下文。

2. **进阶方案 - Agentic Context Engineering**：把上下文看作一个"战术手册（Playbook）"。AI 通过三个步骤来优化它：
    - **反思（Reflect）**：回顾任务进展，找出当前手册的不足。
    - **规划（Plan）**：设计一个修改手册的具体方案（如"删除过时的步骤，添加新发现的关键代码片段"）。
    - **执行（Execute）**：按照方案修改手册。

*参考文献：[Recursive Language Models](https://arxiv.org/abs/2512.24601)*

---

## 九、 总结

Context Engineering 是构建强大、高效 AI Agent 的基石技术，它远不止是"压缩"那么简单，而是一套涉及**记忆、遗忘、聚焦、检索**的复杂认知管理系统。从被动压缩到主动过滤，再到让 AI 自我优化，这一领域正在快速演进，目标是让 AI 像人类一样，具备在信息洪流中保持专注、提取关键、高效决策的能力。
