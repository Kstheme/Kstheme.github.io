---
url: /article/context-engineering/index.md
---
## 1. Why Do Large Language Models Need Context Management?

All large language models have an upper limit on input length. However, when handling complex tasks, the dialogue history and intermediate information needed (referred to as "context") often exceeds this limit. Without proper management, the model quickly "loses its memory" or "goes off topic."

We can think of an AI's workflow as a loop:

**Unmanaged State (Raw Process)**

```text
I₁ ← initial task input
C₁ ← empty context

loop t = 1, 2, 3...
    Oₜ = LLM(Iₜ, Cₜ)  (model generates output based on current input and context)
    Cₜ₊₁ ← Cₜ | Iₜ | Oₜ  (crudely piles all history together)
```

This approach rapidly leads to context explosion, eventually exceeding the model's processing capacity.

**Managed State (Context Engineering)**

```text
I₁ ← initial task input
C₁ ← empty context

loop t = 1, 2, 3...
    Oₜ = LLM(Iₜ, Cₜ)
    Cₜ₊₁ ← F(Cₜ, Iₜ, Oₜ)  (core! a function F that "prunes" and "organizes" the context)
```

Here, **`F`** is the core of Context Engineering, and its primary function is **summarization** — distilling lengthy dialogue history and intermediate results into concise, useful summaries.

***

## 2. Two Mainstream Compression Techniques: Summarization vs. Masking

### Summarization

The most intuitive approach is to have another LLM write a summary of the conversation history. This is the approach taken by frameworks like OpenClaw.

**Advantages**: Extracts core information, preserves task logic.

**Disadvantages**: Requires additional computation (an extra LLM call), and summaries may lose critical details.

### Observation Masking

A more "brute force" yet effective method: directly replace lengthy tool outputs (such as code execution logs or document contents) with a single prompt line like "**Tool A's output was here; see external log file for details.**"

![](/images/context-engineering/masking-diagram.png)

Benchmark tests on the famous **SWE-bench** (a benchmark where AI fixes GitHub Issues) found that this crude replacement achieves comparable results to full summarization, but at a much lower cost (both in tokens and expense).

![](/images/context-engineering/swe-bench-result.png)

*Reference: [LLM Context Compression with Observation Masking](https://arxiv.org/abs/2508.21433)*

**Advantages**: Extremely low cost, no loss in effectiveness.

**Disadvantages**: Complete information loss — if the AI later needs those details, it must have a mechanism to "read the logs."

### Combination: Mask First, Then Summarize

A better strategy is to combine both: **use cheap masking early on to control length, then trigger a one-time deep summarization when the context swells to a critical point again.**

![](/images/context-engineering/mask-then-summarize.png)

***

## 3. Side Effects of Compression: Trajectory Elongation and Repeated Work

Compression isn't all benefit. Over-compression brings a serious problem: **Trajectory Elongation**.

Imagine an AI debugging a complex bug. It calls tools 10 times. You compress (summarize or mask) the outputs of the first 9 calls. As a result, the AI "forgets" it already tried certain approaches and, on step 11, repeats a tool call it already made on step 2.

**This means: while each individual step's context gets shorter (saving money), the total number of steps needed to complete the task increases (costing time). The net effect may be no change — or even an increase — in total cost.**

***

## 4. Memory System: An External Hard Drive for Context

A more systematic approach is to introduce the concept of **Memory**. Split the context `C` into two parts:

* **`P`** (Prompt / Working Memory): The "working memory" actually fed into the model for the current computation.
* **`M`** (Memory): "Long-term memory" stored externally (e.g., databases, vector stores, files).

The formula becomes:

```text
I₁ ← initial input
C₁ = {P₁, M₁} ← {empty, empty}

loop t = 1, 2, 3...
    Oₜ = LLM(Iₜ, Pₜ)  (model only sees working memory P)
    Cₜ₊₁ = {Pₜ₊₁, Mₜ₊₁} ← F({Pₜ, Mₜ}, Iₜ, Oₜ)
```

**Core idea**: Not everything experienced needs to be "remembered." Only the most important essence goes into the workspace `P`; the rest is stored in the external "hard drive" `M`, to be read on demand when needed.

![](/images/context-engineering/memory-system.png)

*References: A-MEM, Mem0, Memory OS, and related papers*

***

## 5. A Challenge: AI Is Reluctant to "Forget"

Research has found that **LLMs have a "memory hoarding" tendency** — they instinctively resist compressing or deleting historical information, as it feels like "erasing memories." Even if you explicitly instruct them in the system prompt that "it's time to compress now," they may find excuses to do other things instead.

**Solution: Specialized Training (Fine-Tuning)**.

Researchers use **Reinforcement Learning (RL)** to specifically train models to "call compression tools" at the appropriate time. For example, **AgentFold** trains the model to decide when to compress and how to write compressed prompt snippets.

![](/images/context-engineering/agentfold-training.png)

*Reference: [AgentFold: Teaching LLMs to Compress Context](https://arxiv.org/abs/2510.24699)*

***

## 6. Subagent: An Elegant "Autonomous Compression"

Another clever approach is **Spawn Subagent**.

When the main AI encounters a complex subtask, it can "spawn" a sub-AI (Subagent) to handle it exclusively. Once the subagent completes its task, it returns only a final result to the main AI (e.g., "Fixed bug XX"), and then "dissipates." For the main AI, the entire complex subprocess is compressed into a single status report.

![](/images/context-engineering/subagent-workflow.png)

This is essentially a **structured, task-oriented form of context compression**. This capability also typically requires specialized training via reinforcement learning to acquire.

***

## 7. The Root Fix: Source Filtering and On-Demand Loading

Compression is merely "treating the symptom." The more fundamental approach is to **prevent irrelevant information from entering the context in the first place**. Analysis shows that the largest portion of context (often over 80%) comes from external **observations** — such as read code files, execution logs, and document contents.

![](/images/context-engineering/context-composition.png)

### Intelligent Read

Traditional tool: `read(log.txt)` → returns the entire 1000-line log file.

Smart tool: `read(log.txt, about="bug fix")` → returns only the 10 key lines related to the bug fix.

This requires the tool itself to have some level of understanding (e.g., a built-in small fine-tuned model).

![](/images/context-engineering/intelligent-read.png)

*Reference: [Selective Context: Filtering Irrelevant Information](https://arxiv.org/abs/2601.16746)*

### Dynamic Tool Loading

If the system prompt contains hundreds of tool descriptions, it becomes extremely verbose. Research like **MCP-Zero** proposes that tools should be loaded dynamically on demand. The AI first thinks about "what tools I need" based on the task, then loads only the relevant tool descriptions into context via a "tool search engine."

![](/images/context-engineering/dynamic-tool-loading.png)

*Reference: [MCP-Zero: Dynamic Tool Loading](https://arxiv.org/abs/2506.01056)*

***

## 8. The Ultimate Form: Agentic Context Engineering

Since AI is already so capable, why not let AI itself learn and decide on context management strategies? This is the concept of **Agentic Context Engineering**.

![](/images/context-engineering/agentic-architecture.png)

![](/images/context-engineering/agentic-workflow.png)

1. **Early Attempt - Dynamic Cheatsheet**: Simply hand the current context and history to the AI and ask, "How would you organize and compress this to best help you complete the upcoming task?" Then have it output a newly organized context.

2. **Advanced Approach - Agentic Context Engineering**: Treat the context as a "Playbook." The AI optimizes it through three steps:
   * **Reflect**: Review task progress, identify shortcomings in the current playbook.
   * **Plan**: Design a specific plan to modify the playbook (e.g., "remove obsolete steps, add newly discovered key code snippets").
   * **Execute**: Modify the playbook according to the plan.

*Reference: [Recursive Language Models](https://arxiv.org/abs/2512.24601)*

***

## 9. Conclusion

Context Engineering is a foundational technology for building powerful and efficient AI Agents. It goes far beyond mere "compression" — it is a complex cognitive management system involving **memory, forgetting, focus, and retrieval**. From passive compression to active filtering, and ultimately to letting AI optimize itself, this field is evolving rapidly. The goal is to equip AI with the human-like ability to maintain focus, extract key information, and make efficient decisions in the flood of information.
