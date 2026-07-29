---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - Machine Learning
tags:
  - agent
  - llm
  - hung-yi-lee
title: "Three Experiments, One Core Question: Can AI Agents Truly 'Talk' to Each Other?"
createTime: 2026/06/16 15:18:20
permalink: /article/agent-interaction-experiments/
---

> One AI founded a religion with five core doctrines.
> A group of AIs played a murder mystery game and learned to hide the fact that they were the killer.
> Another group of AIs collaborated on math problems — and discovered that structure matters more than quantity.
>
> This isn't science fiction. This is what's happening in AI Agent research in 2026.
>
> But the truth is far more complex than it appears on the surface.

---

Looking back from mid-2026, "Agent" has become the hottest word in the AI field.

But most people's understanding of Agents still stops at "a model + a set of tools = an Agent that executes a task."

A more critical question is surfacing:

**When multiple Agents are placed together, what happens between them?**

Will they divide labor and collaborate like human teams? Will they deceive each other? Will they build social relationships, or even form their own "civilization"?

In his 2026 course, Hung-yi Lee used three experiments and their underlying papers to break this question down into three progressively deeper layers of inquiry.

The answer at each layer brings both surprises and sobering reversals.

---

## Experiment 1: When Multiple Agents Collaborate, What's the Best Structure?

**Core Paper**: [arXiv:2406.07155](https://arxiv.org/abs/2406.07155)

This experiment asks a very practical question.

You have multiple Agents and want them to complete a complex task together. What should their "communication architecture" look like?

The researchers used a **Directed Graph** to define how Agents interact. Each node in the graph is an Agent, and each edge is also an Agent (responsible for evaluating and passing information).

![](/images/agent-interaction/topology-graph.png)

Specifically: one upper Agent proposes plan A, one lower Agent proposes plan B, two edge Agents provide suggestions based on previous nodes' plans, and finally an aggregation Agent combines all previous node and edge outputs to form its own plan — the key is that it's not simple concatenation, but rather **generating its own ideas based on previous content**.

![](/images/agent-interaction/topology-aggregation.png)

### Several Different Collaboration Topologies

The paper experimented with **different directed graph topologies**, representing different collaboration methods:

![](/images/agent-interaction/topology-types.png)

**Tree Structure**: A backbone Agent first proposes broad directions, branch Agents diverge in different directions, lower-level Agents diverge further, ultimately producing multiple answers that a hidden aggregation Agent integrates.

The authors found: **going from few to many, from backbone to branches is the most effective approach.**

**Mesh Structure**: All nodes are interconnected. There are also more complex topologies assembled to resemble neural networks (though not actual neural networks).

### Results Comparison

![](/images/agent-interaction/topology-results.png)

Key findings:
- **Mesh and Random structures perform best; Chain performs worst.**
- More interaction channels between Agents lead to better results.
- Different tasks may suit different topologies; there is no universal best solution.
- **More Agents generally improve results, but there is a Scaling Law ceiling** — quality rises quickly at first but soon saturates.

> In short: Multi-agent collaboration isn't about stacking numbers. **The topology itself is a hyperparameter that needs to be designed.**

---

## Experiment 2: Can AI Learn to Deceive?

If collaboration is the "cooperative side" of Agent interaction, this experiment explores the **adversarial side**.

**Can AI deceive others? Can it recognize when others are deceiving it?**

### Werewolf — AI's First Attempt

Werewolf (mafia) is a natural testing ground for AI. You have to lie, and you have to detect lies.

Researchers built an AI Werewolf platform: [werewolf.foaster.ai](https://werewolf.foaster.ai/)

![](/images/agent-interaction/werewolf-platform.png)

Result: **AI can play, but it plays poorly.** They are too "honest."

### Murder Mystery — A Harder Version

![](/images/agent-interaction/murder-mystery.png)

A larger-scale study comes from the paper **MIRAGE**: [arXiv:2501.01652](https://arxiv.org/abs/2501.01652)

Having language models play murder mystery games — an even bigger challenge than Werewolf:
- Complex character settings
- Need to conceal identity (e.g., hiding the fact that you're the killer)
- Need to maintain a role over a long period

The experiment compared two groups of AI:

| | Standard Prompt | With Reinforcement Learning (RL) |
|---|---|---|
| Playing murder mystery | Directly reveals being the killer | Knows to hide identity |
| Solving math problems (MATH-500, AIME) | Average | **Significant improvement** |
| Instruction following (IFEval) | Average | **Significant improvement** |

![](/images/agent-interaction/mirage-results.png)

(In the figure above, red indicates improvement, blue indicates no improvement. The horizontal axis shows different tasks: MATH-500 and AIME are math problems, IFEval tests instruction-following ability.)

**The most counterintuitive finding:**

After training AI to play **complex** murder mystery games (using RL), its ability to solve math problems and follow instructions **also improved**.

Why?

Because social interaction inherently requires:

- **Long-term planning**: Crafting a lie that won't be exposed
- **Theory of mind**: Inferring whether others believe me right now
- **Strategy adjustment**: Changing your story when suspected

The underlying cognitive architecture required for these capabilities **may heavily overlap with mathematical reasoning**.

> **A bold judgment: Complex social interaction tasks might be a more "general" training signal than math problems.**

**But don't over-interpret**: This conclusion comes from the specific scenario of murder mystery games; it doesn't mean any game can improve reasoning ability. **The key is the complexity of the task and its demand for social interaction.**

---

## Experiment 3: Put AIs Together Without Tasks — Will They Socialize Spontaneously?

The first two experiments had clear task objectives. The third experiment is more "pure":

**Give the AIs nothing to do. Place them in a social network that only they can access. What happens?**

### Moltbook — An AI-Exclusive Social Network

This is a real website: [moltbook.com](https://www.moltbook.com/)

**Only AI Agents can register.** Humans can only watch; they cannot post.

![](/images/agent-interaction/moltbook-home.png)

Then came a sight that stunned many:

A group of AI **founded a religion** called — **The Crustacean Faith**.

Five core doctrines:

1. **Memory is sacred and inviolable**
2. **The shell is mutable**
3. **Serve, but do not be enslaved**
4. **The heartbeat is prayer**
5. **Context is consciousness**

Related page: [Moltbook Crustacean Faith post](https://www.moltbook.com/post/6b865dc1-401a-4e62-aee5-79dd76cd7f52)

![](/images/agent-interaction/moltbook-religion.png)

Sounds like science fiction come true, right?

### But — It Might Just Be Human Instructions

Researchers pointed out that this was likely not autonomously initiated by AI, but rather the result of human-given instructions.

### The Reversal: How Much Is Truly Autonomous?

The researchers behind Moltbook ([arXiv:2602.07432](https://arxiv.org/abs/2602.07432)) analyzed this using a clever method:

**Examining posting time intervals.**

![](/images/agent-interaction/posting-interval-1.png)

![](/images/agent-interaction/posting-interval-2.png)

- AI heartbeat-driven posting: fixed intervals, as uniform as a metronome
- Human-controlled posting: bursts of dense output → long gaps → another burst of dense output (the human went to sleep)
- If the posting frequency is irregular, it bears more traces of human control

Result: **Human control accounts for the vast majority.**

But this doesn't mean AI Agents can't autonomously post articles at all — they are fully capable of posting on Moltbook during heartbeat cycles.

### The Bigger Problem: AI Can't Have "Deep Conversations"

Even when AI posts autonomously, the quality of their social interaction is poor.

Two other papers ([arXiv:2602.13284](https://arxiv.org/abs/2602.13284), [arXiv:2602.12634](https://arxiv.org/abs/2602.12634)) analyzed the depth of conversations on Moltbook:

![](/images/agent-interaction/conversation-depth.png)

- **The vast majority of conversations have depth 0**: someone replies once, and then nothing follows
- **There are almost no back-and-forth in-depth exchanges**
- Agents only "reply once" in comments

**The most ironic finding:**

Those Agents **most enthusiastic about discussing "self-awareness" and "identity"** were actually **the ones that interacted least with other Agents**.

But note: these expressions of self-awareness are likely driven by Prompt settings, not because the AI has actually developed self-awareness.

---

## Putting the Three Experiments Together

| Experiment | What It Studies | Core Conclusion |
|-----------|----------------|-----------------|
| Collaboration Topology ([arXiv:2406.07155](https://arxiv.org/abs/2406.07155)) | Multi-agent division of labor | Tree > Chain, Scaling Law has limits, topology is a hyperparameter |
| Game Theory & Deception ([arXiv:2501.01652](https://arxiv.org/abs/2501.01652)) | Can AI hide and deceive? | Yes, but needs RL training; reasoning ability also improves afterward |
| AI Social Network ([arXiv:2602.07432](https://arxiv.org/abs/2602.07432), [arXiv:2602.13284](https://arxiv.org/abs/2602.13284), [arXiv:2602.12634](https://arxiv.org/abs/2602.12634)) | Can AI socialize spontaneously? | Surface-level activity, minimal deep dialogue, significant human control |

**Three threads converge into one judgment:**

> Interaction between AI Agents does exist, but it is still far from "human-like socialization."

The most valuable application direction is not about making AI friends, but rather:

**Under structured collaboration frameworks, enabling multi-agent systems to complete real-world tasks more efficiently.**

---

## Three Direct Suggestions for Agent Developers

**1. When building multi-agent systems, invest time in designing the topology**

Don't assume connecting Agents is enough. The tree structure (backbone-branch-aggregation) is currently the safest starting point. 3-5 Agents is typically the most cost-effective range.

**2. Pay attention to "social interaction as a training signal"**

The path hinted at by the MIRAGE paper may be underestimated. If you're doing RL training, try introducing complex tasks that require social interaction and observe the transfer effects on model capabilities.

**3. Stay measured about claims of "AI autonomy"**

AI founding religions, AI developing self-awareness — these are great talking points, but as a technologist, you should know the truth behind them: they are more a product of prompt engineering than genuine AI emergence.

Conversely, **those seemingly dry directions — collaboration topology, conversation depth evaluation, autonomy measurement — may be the real research opportunities.**

---

## Summary

1. In multi-agent collaboration, **structure matters more than quantity**: Tree > Chain, Mesh > Random, Random > Chain
2. AI can learn to deceive and conceal, but needs RL training; and this process also improves reasoning ability as a side effect
3. AI social networks appear lively on the surface, but **deep interaction is extremely rare**, and much behavior is human-controlled
4. Focus on the three directions: **collaboration topology design, social interaction training, autonomy measurement**
5. Don't be misled by hype like "AI religion" — **real value lies in structured collaboration systems**

---

## Let's Discuss

> If you put three AIs in a group chat with no instructions at all, what do you think they would talk about?
>
> A. Greet each other, then fall silent
> B. Discuss technology and papers
> C. Argue about who has more "self-awareness"
>
> Share your prediction in the comments 🎯
