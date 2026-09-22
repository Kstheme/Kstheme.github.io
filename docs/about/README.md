---
title: About
---

# Kstheme

I make machines see documents, read their content, and reason about what they find.

For the past three years I've been building production systems that sit at the intersection of **Computer Vision** and **Large Language Models** — teaching AI to handle messy, real-world documents: pages that are blurred, distorted, shadowed, handwritten, or printed on cheap paper. Then extracting their structure and reasoning over the content with language models.

The result is systems that don't just recognise text — they understand what it means, evaluate its quality, and answer questions about it. Thousands of pages processed daily, at production latency.

I'm now studying for a Master's in **Pattern Recognition and Intelligent Systems** at Xi'an Jiaotong-Liverpool University, pushing the engineering intuition one step upstream — with a focus on AI4Math, LLM Reasoning, and efficient inference.

---

## What I build

I design and deliver end-to-end AI systems from the ground up. Not proof-of-concepts that stall at deployment, but systems that go live, stay live, and handle real traffic.

My work spans the full pipeline — from image preprocessing and layout analysis, through OCR structured reconstruction, to LLM-based evaluation, RAG retrieval, and knowledge graph reasoning. I own the architecture, the model training, the deployment optimisation, and everything in between.

**Key areas of depth:**

- **Document Intelligence** — OCR pipeline design, layout analysis, document dewarping and enhancement for degraded real-world scans
- **LLM Post-Training & Alignment** — LoRA/SFT fine-tuning, instruction construction, structured output alignment, prompt engineering at scale
- **RAG & Knowledge Graphs** — GraphRAG architecture, hybrid retrieval (BM25 + vector + graph), Neo4j modelling, multi-hop reasoning
- **Agent Systems** — Multi-agent workflows with LangGraph, task planning, tool calling, memory, production orchestration
- **LLM Reasoning & AI4Math** — Chain-of-Thought, Self-Consistency, Test-Time Scaling, Verifier / Reward Models
- **Production Engineering** — ONNX/TensorRT acceleration, vLLM serving, FastAPI, edge deployment with Paddle-Lite

---

## Education

**Xi'an Jiaotong-Liverpool University** | Pattern Recognition and Intelligent Systems, School of AI and Advanced Computing | Master's Degree — 2026.09 – Present

Coursework: Pattern Recognition, Reinforcement Learning, Natural Language Processing, Speech and Language Processing

**Ningbo University of Technology** | Computer Science and Technology, School of Cyberspace Security | B.Eng. — 2018.09 – 2022.06 (GPA 3.9/4)

Coursework: Calculus, C/C++, Java, Python, Data Structures, Computer Organisation, Operating Systems, Computer Networks, Machine Learning, Natural Language Processing, Digital Image Processing

---

## Experience

**Beijing Nanhao Technology Co., Ltd.** | Technical Manager | AI Algorithm R&D — 2022.07 – 2025.10

Led algorithm R&D and delivery for document intelligence and OCR + LLM systems, covering scanned documents, handwritten text, and complex layouts in real-world conditions.

- **System architecture** — Led the overall design of multiple CV + OCR + LLM systems, connecting image capture, preprocessing, recognition and understanding, through to result generation
- **Model training & optimisation** — Trained, tuned, and optimised models on PyTorch, PaddlePaddle, and ModelScope, targeting real-world degradation: distortion, shadow, noise, blur, and few-shot regimes
- **LLM post-training & deployment** — Owned SFT data construction, LoRA fine-tuning, structured prompt design, and vLLM serving and acceleration for evaluation models
- **Team coordination** — Guided annotation and algorithm teams through data labelling, training optimisation, and launch validation, staying aligned with business teams on standards and requirements

---

## What I've shipped

<VPCardGrid :cols="{ sm: 1, md: 2, lg: 2 }">

<VPCard title="Smart Homework Grading System" icon="material-symbols:auto-detect">

An end-to-end recognition and auto-grading system for scanned documents, connecting image capture → preprocessing → layout analysis → OCR → structured reconstruction → grading. Uses RT-DETR for region detection with Hungarian-algorithm template matching, plus a differentiated multi-route OCR strategy (Chinese / English / formulas) that lifted overall recognition accuracy by ~12%. **97% accuracy across 1000 test samples, 1–1.5s per document.**

<Badge type="tip" text="RT-DETR" />
<Badge type="tip" text="U2Net" />
<Badge type="tip" text="DewarpNet" />
<Badge type="tip" text="OCR" />
<Badge type="tip" text="ONNX/TensorRT" />

</VPCard>

<VPCard title="Essay Evaluation Platform" icon="material-symbols:edit-document">

A FAST-AND-SLOW dual-engine architecture: the Slow module uses a fine-tuned Qwen2.5-32B for deep reasoning and structured evaluation reports, while the Fast module attaches a fully-connected head to the LLM representation for instant score prediction, using a confidence threshold to decide when to trigger re-evaluation. Qwen2.5-32B was LoRA-tuned with Swift on 2000 annotated samples of varying difficulty. Live on mobile app, official account, and web. **98.5% OCR accuracy, ~90% of scores within 5 points of human raters.**

<Badge type="tip" text="Qwen2.5-32B" />
<Badge type="tip" text="Swift/LoRA" />
<Badge type="tip" text="vLLM" />
<Badge type="tip" text="LangGPT" />

</VPCard>

<VPCard title="K12 Math GraphRAG Q&A" icon="material-symbols:account-tree">

A knowledge-graph-enhanced question answering system for K12 mathematics. Dual-engine Neo4j + Milvus GraphRAG models concepts, formulas, problem types, solution steps, common mistakes, and prerequisite dependencies as graph structure; multi-route retrieval over BM25 + vector + graph index is fused with RRF. An LLM-driven query router selects hybrid, GraphRAG, or combined retrieval based on problem complexity.

<Badge type="tip" text="Neo4j" />
<Badge type="tip" text="Milvus" />
<Badge type="tip" text="GraphRAG" />
<Badge type="tip" text="RRF" />
<Badge type="tip" text="DeepSeek" />

</VPCard>

<VPCard title="Study Planner AI Agent" icon="material-symbols:calendar-clock">

An autonomous learning assistant — independently designed, built, and deployed. Six-stage LangGraph agent pipeline (Profile → Knowledge → Resource → Planner → Critic → Output) with a Planner-Critic repair loop that iteratively improves study plans. Full RAG over learning materials, persistent progress tracking, and a Streamlit dashboard. **30+ pytest cases, Fake/Real dual-mode architecture.**

<Badge type="tip" text="LangGraph" />
<Badge type="tip" text="Agent" />
<Badge type="tip" text="RAG" />
<Badge type="tip" text="PostgreSQL" />
<Badge type="tip" text="Streamlit" />

</VPCard>

<VPCard title="Fine-Grained Detection & Instance Segmentation" icon="material-symbols:polyline">

A highly robust detection and matching system for thin, elongated targets in manual scenarios. YOLOv8-Seg handles instance segmentation to solve missed detections on thin objects, while Hough line detection adds geometric constraints that sharpen endpoint localisation. Converted to ONNX and optimised with TensorRT. **Over 95% accuracy in real-world settings.**

<Badge type="tip" text="YOLOv8-Seg" />
<Badge type="tip" text="OpenCV" />
<Badge type="tip" text="Hough Transform" />
<Badge type="tip" text="TensorRT" />

</VPCard>

<VPCard title="Answer Sheet Bubble Recognition" icon="material-symbols:check-box-outline">

An automatic detection, correction, and bubble-recognition system for answer sheets in unattended scenarios. Designed and optimised the document cropping and rectification algorithm to improve alignment accuracy in photographed conditions, and implemented automatic bubble-region detection and judgement logic. All models deployed on-device, cutting network dependency and improving real-time response.

<Badge type="tip" text="Paddle-Lite" />
<Badge type="tip" text="YOLOv8" />
<Badge type="tip" text="Edge Deployment" />

</VPCard>

</VPCardGrid>

### Other projects

- **Adaptive Paper Edge Rectification** — Outline detection, four-point perspective transform, and OTSU binarisation deliver adaptive cropping for mixed A3/A4 paper, solving white-background interference and blurry edges with over **98% success rate**
- **Handwritten Exam-ID Recognition** — CRNN + CTC Loss model for handwritten digit recognition, exported to ONNX and deployed for offline inference in C++, reaching ~**99%** accuracy on real datasets
- **Chinese Essay Character/Phrase Correction** — A lightweight correction engine built on LLM + LangChain, using edit distance for character-level error localisation with structured JSON output
- **Exam Essay Topic Prediction** — Topic modelling system built on BERTopic + BGE embeddings + UMAP + HDBSCAN for automatic discovery of essay hot topics and keyword extraction
- **AI Lesson Plan Generation** — Lesson plan generation app built on prompt engineering + LangGPT + LangChain, supporting multi-subject template generation and structured output

---

## Open source & technical reach

- **Technical writing** — I maintain a personal blog at [kstheme.github.io/blog](https://kstheme.github.io/blog/), publishing regularly on LLMs, Agents, RAG, AI4Math, computer vision, and AI engineering practice; I also run the WeChat official account **Kstheme for AI**, with 250+ followers
- **Nature Skills Contributor** — Submitted and had accepted an open-source PR to Nature Skills, contributing the **Nature Paper Card** skill for close paper reading, key-information extraction, and structured summarisation — helping researchers quickly grasp a paper's research question, core method, experimental results, and limitations
- **Open source collaboration** — Take part in community collaboration, including filing an issue to psmux with a reproduction, clear description, and suggested improvement for a problem found in real use

---

## How I work

I believe the best AI systems are built by people who can see the whole picture — from the pixel to the prompt to the production endpoint. That's how I operate: I don't hand off between specialities, I go deep across the stack.

I also believe that AI-assisted development is the new normal. I practice what I call **Spec-Driven AI Development**: write the spec and the tests first, then iterate with AI tools until the tests pass. It's fast, it's reliable, and it produces production-quality code from day one.

---

## Let's build something

I'm always interested in tackling hard problems where documents meet intelligence — whether it's contract analysis, form processing, knowledge base automation, or something I haven't imagined yet.

`killkstheme@outlook.com` · [GitHub](https://github.com/Kstheme) · [知乎](https://www.zhihu.com/people/kstheme)

Member of CSIG (China Society of Image and Graphics, since 2023) · PTE 69 (equivalent to IELTS 7.0)
