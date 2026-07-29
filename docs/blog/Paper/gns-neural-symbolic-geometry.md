---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - Paper Review
tags:
  - gns
  - neural-symbolic
  - geometry
  - paper
  - aaai
title: "Why Can a 13B Small Model Solve Geometry Problems That GPT Gets Wrong? — A Close Reading of AAAI-25 GNS"
createTime: 2026/07/29 17:27:20
permalink: /blog/gns-neural-symbolic-geometry/
copyright: Kstheme
---

## 01 Beating GPT-4o with a 13B Model: Why Do Geometry Problems Stump Even Top AI?

**GPT-4o got it wrong. Gemini-1.5-Flash got it wrong too.**

The problem was simple: triangle ABC is bisected perpendicularly by DE. The area of triangle ABC is 8.0. Find the area of triangle ADE.

GPT-4o wrote a seemingly correct chain of reasoning — it correctly identified DE as the perpendicular bisector of BC, correctly noted that triangles ADE and ABC are similar, but tripped on the ratio and gave the wrong answer. Gemini-1.5-Flash directly wrote the wrong proportional relationship.

Yet a small 7B model, specially fine-tuned, used a clean pipeline of "Pythagorean theorem → ratio calculation → symbolic program" and arrived at the correct answer.

This is not an isolated case. When OpenAI released GPT-4o in May 2024, it achieved 60.6% accuracy on the geometry problem-solving subtask of MathVista — impressive for a ~1.8 trillion parameter model costing billions to train. Yet just months later, a AAAI-25 paper pushed this number to **63.9%** using a model with only 4.2B parameters, surpassing GPT-4o and Gemini Ultra to top the leaderboard.

The method is not a bigger model, more data, or longer training. It's an idea discussed for years but rarely successfully implemented — **bringing symbolic reasoning back into neural networks.**

**Abstract: This paper proposes GNS (Geometry Neural-Symbolic), a framework that explicitly parses visual information from plane geometry problems into symbolic clauses, enabling MLLMs to first "read" geometry diagrams, then use symbolic solvers for precise computation. GNS achieves superior PGP performance across multiple benchmarks with models far smaller than GPT-4o.**

![](/images/gns/main-results.png)

_Accuracy comparison of different MLLM backbones on MathVista geometry tasks, before and after GNS fine-tuning. All models show significant improvement under GNS, with Phi3-Vision-4.2B reaching 63.9%, surpassing GPT-4o's 60.6%._

## 02 Paper Information Card

| Item                   | Content                                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Paper Title**        | GNS: Solving Plane Geometry Problems by Neural-Symbolic Reasoning with Multi-Modal LLMs                                           |
| **Venue**              | AAAI-25 (CCF-A)                                                                                                                   |
| **Research Task**      | Plane Geometry Problem (PGP) Solving                                                                                              |
| **Core Method**        | Neural-Symbolic MLLM framework with four modules: Knowledge Prediction, Symbolic Parsing, Problem Reasoning, Symbolic Computation |
| **Technical Approach** | MLLM full-parameter fine-tuning + structured prompt pipeline + SymPy symbolic solving                                             |
| **Training**           | Yes — full-parameter fine-tuning                                                                                                  |
| **Dataset**            | GNS-260K (built from PGPS9K, GeoQA+, Geo170K; 9,426 unique diagrams, expanded to 260K samples)                                    |
| **Key Results**        | MathVista GPS 63.9% (#1, surpassing GPT-4o 60.6%); GeoQA new SOTA; MathVerse significant improvement                              |
| **Keywords**           | Plane Geometry Solving, Neural-Symbolic Reasoning, MLLM, Symbolic Parsing, GNS-260K                                               |

**One-sentence summary:** GNS enables 4.2B-13B models to surpass GPT-4o (~1.8T parameters) on plane geometry problem solving by having MLLMs learn to explicitly parse geometry diagrams into symbolic clauses, combined with knowledge prediction, reasoning, and an external symbolic solver (SymPy) for precise computation.

---

## 03 Why Is This Problem Fundamentally Difficult

**Layer 1: The visual-textual information gap.** Most PGP text descriptions are extremely brief, with nearly all geometric information hidden in the diagram. MLLM pretraining data is dominated by natural scene images, with minimal geometry diagrams.

**Layer 2: Precision requirements for reasoning.** A small error in proportional relationships amplifies exponentially in the final answer. LLMs excel at pattern matching but not precise computation.

**Layer 3: Error accumulation in multi-step reasoning.** Each step's error propagates to the next. If the model misidentifies similarity relationships at the start, all subsequent reasoning is wasted.

![](/images/gns/difficulties.png)

_Three essential difficulties of plane geometry problem solving: cross-modal information gap, computational precision requirements, and multi-step error accumulation._

---

## 04 Why Existing Methods Fall Short

| Approach                     | Representative Work     | Key Limitation                                                   |
| ---------------------------- | ----------------------- | ---------------------------------------------------------------- |
| **Early Rule-Based**         | Seo et al. (2014, 2015)[^1] | Small datasets, rigid rules                                      |
| **Neural Methods**           | NGS, UniGeo[^2][^3]        | Coarse geometry understanding                                    |
| **Symbolic Methods**         | Inter-GPS, FormalGeo[^4][^5] | Limited data, predefined rules                                   |
| **MLLM + Data Augmentation** | G-LLaVA + Geo170K[^6]     | Treats PGP as generic QA, lacks explicit geometric understanding |

The core problem: **none of these approaches simultaneously combine the flexibility of neural networks for understanding with the precision of symbolic systems for reasoning.**

![](/images/gns/evolution.png)

_Evolution of PGP solving methods. GNS is the first neural-symbolic framework to achieve high performance on both understanding and reasoning dimensions._

---

## 05 Core Method: What the Paper Actually Proposes

GNS is a four-module pipeline. Given a plane geometry problem $P = [Q, I]$ ($Q$: text question, $I$: geometric diagram):

```text
Text Question(Q) + Diagram(I)
      │
      ├──→ [Knowledge Prediction] — determine required theorem categories
      │
      ├──→ [Symbolic Parsing]     — parse diagram/text into symbolic clauses
      │
      └──→ [Problem Reasoning]    — reason with diagram, knowledge, and clauses
                    │
               [Symbolic Computation] — solve precisely with SymPy
                    │
                Final numerical answer
```

| Module                   | Input                                      | Output                                   | Problem Solved                                    |
| ------------------------ | ------------------------------------------ | ---------------------------------------- | ------------------------------------------------- |
| **Knowledge Prediction** | Diagram $I$ + Prompt $P_K$ + Question $Q$  | Knowledge label $T_K$                    | Simulates human "classify before solve" strategy  |
| **Symbolic Parsing**     | Diagram $I$ + Prompt $P_P$ + Question $Q$  | Symbolic clause paragraph $T_P$          | Makes implicit geometry elements explicit         |
| **Problem Reasoning**    | Diagram $I$ + Prompt $P_R$ + $T_K$ + $T_P$ | Solution program $R_S$ + CoT description | Translates parsed info into executable steps      |
| **Symbolic Computation** | Solution program $R_S$                     | Final numerical answer $V$               | Avoids LLM inherent numerical computation defects |

Two key design choices:

1. Knowledge Prediction and Symbolic Parsing run **in parallel**, avoiding sequential error propagation.
2. Symbolic Computation is **not optional** — without it, accuracy drops 3.4-4.8% even when geometry relations are correctly understood[^7].

![](/images/gns/framework.png)

_GNS framework overview. Given the problem image and text "Find RT," GNS predicts knowledge category "Polygon Similarity," parses symbolic clauses, establishes proportional equations, and computes the final answer via symbolic solver._

---

## 06 Key Mechanisms

### Mechanism 1: Two-Layer Symbolic Clause Structure

| Clause Type    | Description                     | Example                                  |
| -------------- | ------------------------------- | ---------------------------------------- |
| **Semantic**   | Quantitative/semantic relations | `BG = EG = 10`                           |
| **Structural** | Topological connections         | `line A B C`, `circle G lies on A C D F` |

### Mechanism 2: Unified Symbolic Solving System

GNS-260K unifies solution program formats across different source datasets, embedding numerical constants directly into solution programs to eliminate variable mapping steps.

![](/images/gns/dataset.png)

_GNS-260K multi-task dataset structure. Built from 9,426 unique diagrams, expanded to 260K samples._

---

## 07 Closed Loop, Agent, and Execution Mechanism

GNS has **no** closed-loop feedback mechanism. It's a forward pipeline:

| Agent Concept | GNS Component                             | Notes                                |
| ------------- | ----------------------------------------- | ------------------------------------ |
| Think         | Problem Reasoning                         | MLLM conducts CoT reasoning          |
| Act           | Program Generation + Symbolic Computation | SymPy executes precise computation   |
| Feedback      | **Does not exist**                        | Pipeline is unidirectional           |
| Replan        | **Does not exist**                        | No backtracking on invalid solutions |

GNS chooses a single-pass pipeline because **geometry problem-solving is naturally a deterministic, plan-able sequence.**

---

## 08 What the Experiments Actually Prove

### Main: MathVista GPS Benchmark

| Model                   | Parameters | Accuracy  |
| ----------------------- | ---------- | --------- |
| GPT-4o                  | ~1.8T      | 60.6%     |
| Gemini Ultra            | Unknown    | 56.2%     |
| **GNS-Phi3-Vision**     | **4.2B**   | **63.9%** |
| **GNS-LLaVA-13B**       | **13B**    | **63.9%** |
| G-LLaVA-13B (prev SOTA) | 13B        | 56.7%     |
| Human baseline          | —          | 48.4%     |

All 5 GNS-MLLMs surpass human baseline. The smallest (DeepSeek-VL-1.3B at 55.3%) already reaches 95%+ of G-LLaVA-13B's level.

![](/images/gns/benchmark-results.png)

### Ablation Study

| Setting | Knowledge Prediction | Symbolic Parsing | Symbolic Computation | Accuracy |
| ------- | -------------------- | ---------------- | -------------------- | -------- |
| ①       | ✗                    | ✗                | ✗                    | 52.4%    |
| ②       | ✗                    | ✗                | ✓                    | 55.8%    |
| ③       | ✗                    | ✓                | ✓                    | 60.6%    |
| ④       | ✓                    | ✗                | ✗                    | 57.2%    |
| ⑤       | ✓                    | ✓                | ✓                    | 62.0%    |

- **Symbolic Computation**: largest contribution (+3.4% from ①→②, +4.8% from ④→⑤)
- **Symbolic Parsing**: medium contribution (+4.8% from ②→③)
- **Knowledge Prediction**: smallest contribution (+1.4% from ③→⑤)

![](/images/gns/ablation.png)

---

## 09 Understanding the Result Boundaries

| Condition               | Setting                                | Implication                           |
| ----------------------- | -------------------------------------- | ------------------------------------- |
| **Input type**          | Raw diagram + text (no ground-truth)   | No manual preprocessing needed        |
| **End-to-end**          | No (4-stage pipeline)                  | Higher latency than single-stage MLLM |
| **Training data**       | GNS-260K (260K samples)                | Largest PGP dataset                   |
| **Model scale**         | 1.3B-13B, 5 MLLMs                      | Effects on 70B+ models unknown        |
| **Symbolic dependency** | SymPy required                         | Limited deployment without Python     |
| **GPT-4 involvement**   | GPT-4 generated reasoning descriptions | Potential data bias                   |
| **Inference cost**      | Not reported                           | 4 MLLM calls + SymPy computation      |

---

## 10 Paper Limitations and Further Analysis

### Paper-Stated Limitations

| Limitation            | Specifics                                         |
| --------------------- | ------------------------------------------------- |
| Geometry3K challenges | Information-sparse problems still difficult       |
| MathVerse gap         | 27.1% vs 39.4% GPT-4V (due to non-PGP problems)   |
| GPT-4 annotation      | Closed-source model dependency and potential bias |

### Further Analysis

| Observation                                 | Potential Issue                                     | Needs Validation                          |
| ------------------------------------------- | --------------------------------------------------- | ----------------------------------------- |
| Ablation only on LLaVA-1.5-7B               | Module contributions may vary by model scale        | Repeat on 3-4 different backbones         |
| Knowledge prediction gain too small (+1.4%) | May be practically omittable                        | Test without symbolic parsing/computation |
| GPT-4 annotation bias                       | Training data may carry GPT-4 reasoning preferences | Retrain without GPT-4 annotations         |
| Inference cost unreported                   | 4-stage pipeline slower than single-stage           | Report end-to-end latency comparison      |

---

## 11 Research Extensions and Summary

### Research Directions

| Dimension                | Paper Status           | Extension                            | Hypothesis                                               |
| ------------------------ | ---------------------- | ------------------------------------ | -------------------------------------------------------- |
| **Data annotation**      | GPT-4 dependent        | **Self-supervised symbolic parsing** | Symbolic solver output can reverse-generate NL reasoning |
| **Reasoning path**       | Fixed 4-stage pipeline | **Dynamic module routing**           | Different difficulties need different modules            |
| **Symbolic interaction** | No feedback            | **Verification-backtrack loop**      | Geometry solutions have self-consistency constraints     |
| **Application scope**    | Plane geometry only    | **Extend to other math domains**     | Symbolic clause paradigm generalizable                   |

**Research Question:**

> How can the neural-symbolic paradigm be extended to support dynamic computation graphs, self-supervised annotation, and automated verification loops for broader mathematical reasoning tasks?

### Summary: What GNS Changed

| Past (G-LLaVA etc.)          | GNS Direction                                          |
| ---------------------------- | ------------------------------------------------------ |
| PGP as generic multimodal QA | Decompose into understanding + reasoning + computation |
| Implicit visual encoding     | Explicit symbolic parsing of diagrams                  |
| Pure neural computation      | Neural understanding + symbolic computation            |
| Dataset-specific formats     | Unified symbolic solving system                        |
| Volume-driven                | Structure-driven                                       |

The true insight: **when neural networks hit a bottleneck on a task, the most effective improvement may not be scaling up the model, but designing a suitable "structural interface" — letting models do what they're good at (understanding, reasoning) while delegating what they're not (precise computation, symbolic operations) to specialized tools.**

![](/images/gns/paradigm-shift.png)

_GNS's core paradigm shift: from "implicit encoding → NL reasoning" single channel to "explicit symbolic parsing → neural reasoning + symbolic computation" dual channel._

---

_Based on Ning et al. (AAAI-25) "GNS: Solving Plane Geometry Problems by Neural-Symbolic Reasoning with Multi-Modal LLMs." Code and data: [https://github.com/ning-mz/GNS](https://github.com/ning-mz/GNS)_

## References

[^1]: Seo M, Hajishirzi H, Farhadi A, et al. Solving geometry problems: Combining text and diagram interpretation[C]//Proceedings of the 2015 conference on empirical methods in natural language processing. 2015: 1466-1476.
[^2]: Chen J, Tang J, Qin J, et al. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning[C]//Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021. 2021: 513-523.
[^3]: Chen J, Li T, Qin J, et al. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression[C]//Proceedings of the 2022 conference on empirical methods in natural language processing. 2022: 3313-3323.
[^4]: Lu P, Gong R, Jiang S, et al. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning[C]//Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2021: 6774-6786.
[^5]: Zhang X, Zhu N, He Y, et al. Formalgeo: The first step toward human-like imo-level geometric automated reasoning[J]. arXiv preprint arXiv:2310.18021, 2023.
[^6]: Gao J, Pi R, Zhang J, et al. G-llava: Solving geometric problem with multi-modal large language model[C]//International Conference on Learning Representations. 2025, 2025: 3490-3511.
[^7]: Gao L, Madaan A, Zhou S, et al. Pal: Program-aided language models[C]//International conference on machine learning. PMLR, 2023: 10764-10799.

Copyright Ownership: Kstheme, Contributors: Kstheme
