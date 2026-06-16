---
title: About
---

# Kstheme

I make machines see documents, read their content, and reason about what they find.

For the past three years I've been building production systems that sit at the intersection of **Computer Vision** and **Large Language Models** — teaching AI to handle messy, real-world documents: pages that are blurred, distorted, shadowed, handwritten, or printed on cheap paper. Then extracting their structure and reasoning over the content with language models.

The result is systems that don't just recognise text — they understand what it means, evaluate its quality, and answer questions about it. Thousands of pages processed daily, at production latency.

---

## What I build

I design and deliver end-to-end AI systems from the ground up. Not proof-of-concepts that stall at deployment, but systems that go live, stay live, and handle real traffic.

My work spans the full pipeline — from image preprocessing and layout analysis, through OCR structured reconstruction, to LLM-based evaluation, RAG retrieval, and knowledge graph reasoning. I own the architecture, the model training, the deployment optimisation, and everything in between.

**Key areas of depth:**

- **Document Intelligence** — OCR pipeline design, layout analysis, document dewarping and enhancement for degraded real-world scans
- **LLM Post-Training & Alignment** — LoRA/SFT fine-tuning, instruction construction, structured output alignment, prompt engineering at scale
- **RAG & Knowledge Graphs** — GraphRAG architecture, hybrid retrieval (BM25 + vector + graph), Neo4j modelling, multi-hop reasoning
- **Agent Systems** — Multi-agent workflows with LangGraph, task planning, tool calling, memory, production orchestration
- **Production Engineering** — ONNX/TensorRT acceleration, vLLM serving, FastAPI, edge deployment with Paddle-Lite

---

## What I've shipped

<VPCardGrid :cols="{ sm: 1, md: 2, lg: 2 }">

<VPCard title="Smart Homework Grading System" icon="material-symbols:auto-detect">

An end-to-end grading pipeline that processes scanned homework — from image capture to final score. Handles blurry phone photos, distorted bindings, and mixed print-handwriting content. **97% accuracy across 1000 samples, 1–1.5s per page.**

<Badge type="tip" text="RT-DETR" />
<Badge type="tip" text="U2Net" />
<Badge type="tip" text="OCR" />
<Badge type="tip" text="ONNX/TensorRT" />

</VPCard>

<VPCard title="Essay Evaluation Platform" icon="material-symbols:edit-document">

A Fast-and-Slow architecture: a fine-tuned Qwen2.5-32B handles deep reasoning, while a lightweight MLP scores routine essays instantly. Deployed to **production mobile and web apps** serving real users. **98.5% OCR accuracy, ~90% agreement with human raters.**

<Badge type="tip" text="Qwen2.5-32B" />
<Badge type="tip" text="LoRA" />
<Badge type="tip" text="vLLM" />
<Badge type="tip" text="LangGPT" />

</VPCard>

<VPCard title="K12 Math GraphRAG Q&A" icon="material-symbols:account-tree">

A knowledge-graph-enhanced question answering system for K12 mathematics. Dual-engine Neo4j + Milvus GraphRAG with hybrid retrieval routes queries intelligently — simple lookups go straight to vector search, complex multi-step problems trigger graph traversal and multi-hop reasoning.

<Badge type="tip" text="Neo4j" />
<Badge type="tip" text="Milvus" />
<Badge type="tip" text="GraphRAG" />
<Badge type="tip" text="DeepSeek" />

</VPCard>

<VPCard title="Study Planner AI Agent" icon="material-symbols:calendar-clock">

An autonomous learning assistant — independently designed, built, and deployed. Six-stage LangGraph agent pipeline with a Planner-Critic repair loop that iteratively improves study plans. Full RAG over learning materials, persistent progress tracking, and a Streamlit interface. **30+ tests, dual-mode architecture.**

<Badge type="tip" text="LangGraph" />
<Badge type="tip" text="Agent" />
<Badge type="tip" text="RAG" />
<Badge type="tip" text="Streamlit" />

</VPCard>

</VPCardGrid>

---

## How I work

I believe the best AI systems are built by people who can see the whole picture — from the pixel to the prompt to the production endpoint. That's how I operate: I don't hand off between specialities, I go deep across the stack.

I also believe that AI-assisted development is the new normal. I practice what I call **Spec-Driven AI Development**: write the spec and the tests first, then iterate with AI tools until the tests pass. It's fast, it's reliable, and it produces production-quality code from day one.

---

## Let's build something

I'm always interested in tackling hard problems where documents meet intelligence — whether it's contract analysis, form processing, knowledge base automation, or something I haven't imagined yet.

`killkstheme@gmail.com` · [GitHub](https://github.com/Kstheme) · [知乎](https://www.zhihu.com/people/kstheme)

中国图象图形学学会（CSIG）Member
