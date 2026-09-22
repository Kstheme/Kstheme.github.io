---
title: Projects
---

A collection of open-source projects I've built and maintained.

<VPCardGrid :cols="{ sm: 1, md: 2, lg: 2 }">

<VPCard title="Study-Planner" icon="material-symbols:calendar-clock-outline">

An AI-powered learning operating system that turns goals into structured study plans. A six-stage LangGraph agent workflow (Profile → Knowledge → Resource → Planner → Critic → Output) uses a Planner-Critic repair loop to keep improving plan completeness and executability. A full RAG pipeline parses PDF/TXT/Markdown, stores vectors in Milvus, and traces citations — supporting material Q&A, summarisation, knowledge-point extraction, and flashcard generation. Includes a PostgreSQL persistence layer plus dashboards for task distribution, daily load, and review progress. Six-layer architecture with a framework-free domain model, and a Fake/Real dual mode toggled by one line of config. **30+ pytest cases.**

<Badge type="tip" text="Python" />
<Badge type="tip" text="LangGraph" />
<Badge type="tip" text="RAG" />
<Badge type="tip" text="Milvus" />
<Badge type="info" text="⭐ 6" />

**[GitHub →](https://github.com/Kstheme/Study-Planner)**

</VPCard>

<VPCard title="repo-system-design-skills" icon="material-symbols:code-blocks-outline">

Reusable skill modules for learning system design from real codebases. Maps architecture, traces data flow, and coaches design decisions — works with both Claude Code and Codex.

<Badge type="tip" text="Markdown" />

**[GitHub →](https://github.com/Kstheme/repo-system-design-skills)**

</VPCard>

<VPCard title="nature-skills" icon="material-symbols:auto-stories-outline">

A collection of reusable AI-agent skills for scientific research, targeting Nature-style academic writing and figure creation. I contributed the **Nature-Paper-Card** skill — it generates source-constrained, in-depth Paper Cards (sections 01–16) covering method logic, evidence chains, conclusion boundaries, and critical analysis, helping researchers deeply understand papers and discover new research ideas.

<Badge type="tip" text="Markdown" />
<Badge type="info" text="Fork" />

**[GitHub →](https://github.com/Yuan1z0825/nature-skills)**

</VPCard>

<VPCard title="Kstheme.github.io" icon="material-symbols:globe-outline">

Personal blog and portfolio built with VuePress & vuepress-theme-plume, publishing regularly on LLMs, Agents, RAG, AI4Math, computer vision, and AI engineering practice. Also runs the WeChat official account **Kstheme for AI** (250+ followers).

<Badge type="tip" text="TypeScript" />

**[GitHub →](https://github.com/Kstheme/Kstheme.github.io)**

</VPCard>

</VPCardGrid>
