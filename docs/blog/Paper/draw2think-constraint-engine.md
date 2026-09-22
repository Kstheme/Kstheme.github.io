---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - Paper Review
tags:
  - draw2think
  - geometry
  - agent
  - paper
title: 'GeoGebra Can Precisely Measure a "Wrongly Built Figure": A Close Reading of Draw2Think'
createTime: 2026/09/22 09:39:20
permalink: /article/draw2think-constraint-engine/
copyright: Kstheme
---

> A problem asks: draw a perpendicular to AB through point A, intersecting circle O at point C.
>
> The model picked the wrong point. It made GeoGebra do something else: draw a perpendicular to AB through point D, and intersect it with circle O.
>
> GeoGebra executed it perfectly, and every angle and length measured afterward was exact.
>
> So — was this a success or a failure?

The value of the paper this article covers isn't "making models better at geometry." It's that it forces the question above into a corner where it must be answered.

**Paper Info**

| Item         | Content                                                                         |
| ------------ | ------------------------------------------------------------------------------- |
| Title        | Draw2Think: Harnessing Geometry Reasoning through Constraint Engine Interaction |
| Authors      | Juncheng Hu, Jiawei Du, Xin Zhang, Joey Tianyi Zhou                             |
| Affiliations | National University of Singapore; A\*STAR CFAR / IHPC                           |
| Version      | Preprint, arXiv:2605.20743v1                                                    |
| Approach     | Frozen VLM + GeoGebra constraint engine, **no task fine-tuning**                |

---

## 01 What the Paper Says in One Sentence

Let a frozen VLM, through typed tool calls, repeatedly construct, query, and revise a GeoGebra canvas; then run an independent predicate audit on the final figure.

**Its contribution is more a designed tool-execution framework than a new model.**

There's an easily overlooked word here: **frozen**. The paper trains nothing and adds no new weights — all gains come from "giving the model an executable geometry workspace at inference time."

![](/images/draw2think/image-20260921122331081.png)

_Comparison of four routes for externalizing intermediate state. Draw2Think's difference is the last one: verification happens immediately after each action, not once at the end._

---

## 02 The Problem It Actually Targets

The paper's starting point isn't "the model computed wrong" — it's a more precise description:

> When reasoning, vision-language models treat angles, auxiliary lines, and length relations as subsequent premises, but these intermediate states may exist only in text or latent representations, **never checked by any execution environment**.

Note the keyword: **latent**.

Meaning: the model internally "believes" it constructed a perpendicular relation, but that relation never landed on any checkable carrier. It may be right, it may be wrong, and we have no means to distinguish.

The paper gives a real case (MathVista/290): the baseline model mistakenly treats two angles as equal, reasons through with alternate interior angles, and outputs **105°**; after the construct-plus-measure loop, the reading is **75°**.

One local misreading propagates down the whole reasoning chain. That's the seam the paper wants to cut into.

![](/images/draw2think/image-20260921114333007.png)

_What the paper really targets isn't "the model can't do it" — it's that intermediate states have no external receipt._

---

## 03 Why Earlier Methods Fall Short

In Related Work the paper classifies routes and positions itself. One caveat is mandatory: **this comparison is framed by the authors** — it shouldn't be read directly as "old methods can't do precise checking."

| Route                                 | Intermediate Representation & Feedback                                 | Gap the Paper Identifies                                                                                  |
| ------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Text / image CoT                      | Textual derivation, generated images, visual re-observation            | Lacks constraint-level receipts; visual approximation ≠ mathematical relation holding                     |
| Drawing programs & rendering feedback | Code, modifiable figure state                                          | Timing of checks and authority of return values are insufficient                                          |
| Symbolic solving & formal reasoning   | Axioms, predicates, derivations                                        | Handles proof obligations, but differs in emphasis from "instance solving while constructing and reading" |
| **Draw2Think**                        | typed ToolSpecs → persistent canvas → object changes / values / errors | Wires execution, readout, and local recovery into the reasoning loop                                      |

The paper's distinction between "pixel similarity" and "geometric correctness" is the basis of this comparison: **a figure that looks close to 90° doesn't mean it satisfies the perpendicular constraint.** A visually near-vertical line may actually be 89°.

There's a phrasing worth remembering here: **constraint-first**.

- Method 1: generate a visual result first, then hope it satisfies the geometric relation;
- Method 2: first declare "I want an object satisfying the perpendicular relation," then have the engine generate a result satisfying it.

This isn't "method 2 is more accurate" — it's that **the order of commitment is reversed**.

---

## 04 Core Method: Propose–Draw–Verify

The paper writes the entire geometry solution as a **state-space search**, not "calling a few tools."

$$\hat y = h(Q, S_K), \qquad S_k = \mathcal T(S_{k-1}, a_k)$$

The symbols aren't complicated; one by one:

- $S_{k-1}$: current canvas state (which objects have been constructed, which relations hold);
- $a_k$: the action at this step;
- $\mathcal T$: the engine executing this action;
- $S_k$: the new canvas after execution;
- $h$: reading the answer the problem asks for out of the final state $S_K$.

**What it emphasizes is really this: what step 2 can do depends on what step 1 has already established on the canvas.** So what gets "searched" isn't an answer sentence — it's a geometric world state that can support a correct readout.

A concrete example, starting from an empty canvas:

```text
S0 = empty
 → S1 = {A, B}
 → S2 = {A, B, segment AB}
 → S3 = {A, B, AB, perpendicular through A}
 → find intersection, read angle, give answer
```

The three roles divide like this:

```text
Propose: VLM decides what to do next (choose tool u_k + fill parameters φ_k)
   ↓
Draw: GeoGebra executes the action, updates the canvas (S_k = T(S_{k-1}, a_k))
   ↓
Verify: the engine returns object changes, values, or errors to the model
```

The paper splits $a_k = (u_k, \phi_k)$ and writes this factorization:

$$P(a_k|\cdots) = P(u_k|\cdots)\cdot P(\phi_k|u_k,\cdots)$$

**Why is this split useful?** Because it shows "knowing to draw a perpendicular" and "mistakenly passing a point object as a line parameter" are two different failures. Choosing the tool and filling parameters are two distinct difficulties — the former leans on natural-language description, the latter on type signatures and preconditions.

The paper also gives two engineering details, both about "whether it can actually run in practice":

- **Tool catalog**: **92** planar items = **55 construction + 24 query + 13 rendering**; ordinary planar solving actually exposes 79, the 13 rendering tools are only used for GenExam, and 3D activates another 21 extensions;
- **Persistent state**: object dependencies form a DAG; deleting a parent object **cascades** to descendants. When a step fails, the action is rejected, but previously successful objects remain, so the model only needs to fix the current branch.

![](/images/draw2think/image-20260921114528151.png)

_The PDV data flow. The line in the middle is the most important line in the whole paper: the engine will not cross it to judge the problem's intent for the model._

![](/images/draw2think/image-20260921114713097.png)

_Same model, two reasoning modes. The difference isn't the model — it's whether intermediate states get checked by the engine._

---

## 05 The Most Important Boundary: Two Kinds of "Faithful"

This is the most easily misread and most worth taking away from the paper. It deliberately splits two concepts:

| Concept                      | Belongs to            | Question Answered                                                           | Who Guarantees It                                     |
| ---------------------------- | --------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Construction Fidelity**    | Model-level property  | Does the final canvas satisfy the geometric relations the problem requires? | **Not guaranteed by the engine**; needs offline audit |
| **Measurement Faithfulness** | Engine-level property | On **this current canvas**, are the returned values/relations faithful?     | Automatically guaranteed by the engine                |

Back to the opening example:

- Measurement Faithfulness ✅ **Success**. Because GeoGebra faithfully answered "in this (wrong) figure, what are the angles and lengths";
- Construction Fidelity ❌ **Failure**. Because the problem required `Perpendicular(A,B,C)`, but the model actually constructed `Perpendicular(D,B,C)` — the object relations don't match.

**The key conclusion is one sentence: the engine can measure a wrongly built figure with perfect accuracy.**

In the paper's own words: the engine guarantees "what can be derived from the current canvas," not "whether the current canvas matches the problem's intent."

![](/images/draw2think/fig5_two_perpendiculars.png)

_One legal canvas, two verdicts. GeoGebra never raised an error, because it has no idea you picked the wrong point._

![](/images/draw2think/image-20260921120850748.png)

_Remember this figure and you'll avoid 90% of over-interpretation about geometry agents._

---

## 06 So What Does Verify in PDV Actually Verify?

Continuing from the previous section, we must separate PDV's Verify from "verifying the problem's intent":

```text
Verify:
"Did I correctly execute the action you asked me to do?"

Construction Fidelity:
"Was the action you asked me to do actually what the problem needed?"
```

PDV's Verify can tell you:

- whether the action is legal;
- what object was created;
- what the current measured value is;
- whether there was an execution error.

It **cannot** automatically judge:

- "Did you pick the wrong point?"
- "Is this auxiliary line actually what the problem needs?"
- "Does the whole canvas now fully match the problem's intent?"

This is also why the paper isolates Construction Fidelity into a separate **offline audit**, rather than claiming PDV has solved intent correctness.

The paper does one more thing here: it splits errors into two classes, because they need different fixes.

| Error Type                      | Meaning                                                                                       | After Loosening Tolerance                              |
| ------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **Structural error**            | The geometric relation itself is wrong, e.g. should be perpendicular but constructed parallel | **No improvement** — forms an essentially flat plateau |
| **Numerical / precision error** | Relation is right, but coordinates/float precision fail the threshold                         | Pass rate rises                                        |

Quick quiz: an angle is theoretically 90°, the construction relation is correct, but the engine reads **89.9998°** — which class is this?

The answer is the second. This distinction directly determines whether you should change the model strategy or the numerical handling.

---

## 07 Results: Not Uniform Improvement, but "Selective Gains"

The main experiment uses Gemini-3-Flash Preview, temperature=0, Pass@1, same problems and same model; BL is direct answering (single call), CT is closed-loop construction (up to 30 rounds, 120 seconds each).

The results are this set of numbers (unit %, differences in percentage points):

| Dataset / Setting        |     N |   BL |   CT |      Diff |
| ------------------------ | ----: | ---: | ---: | --------: |
| SolidGeo-hard Level 3    |   177 | 59.9 | 76.3 | **+16.4** |
| MathVerse-solid          |   119 | 82.4 | 88.2 |      +5.8 |
| GeoSketch                |   390 | 81.8 | 85.9 |      +4.1 |
| MathVerse Plane          |   510 | 87.5 | 91.0 |      +3.5 |
| GeoQA / UniGeo calc test |   754 | 93.6 | 96.9 |      +3.3 |
| GeoLaux                  |   221 | 93.2 | 94.1 |      +0.9 |
| MathVista GPS            |   208 | 97.1 | 97.6 |      +0.5 |
| OlympiadBench            |   112 | 89.3 | 89.3 |       0.0 |
| Geo3K                    |   601 | 98.2 | 95.3 |  **−2.9** |
| PGPS9K                   | 1,000 | 94.5 | 90.5 |  **−4.0** |

**Both gains and losses — that's the most interesting part of this paper.** The authors call it **selective gains** and give a mechanistic explanation:

- The big gains are on visually heavy problems requiring spatial construction (hard 3D, geometry sketches). Here the engine replaces **expensive and unstable internal spatial simulation**;
- The losses are on textbook-style benchmarks already near saturation. Here the baseline gets them right with cheap internal reasoning, and **forcing a construct-query pass just adds decision points and failure opportunities**.

There's a conclusion in the paper you can copy straight into your notes: **External tools are not universally beneficial.**

In engineering terms: **more tools isn't better — they're valuable only when a bottleneck actually exists.**

An applied judgment: a triangle with two angles 50° and 60°, find the third — that needs one mental computation of 180−50−60, not a canvas.

![](/images/draw2think/image-20260921121207550.png)

_Same system, +16.4 on hard 3D and −4.0 on PGPS9K. A tool's value depends heavily on problem type._

---

## 08 GeoGoal: 95.94% vs 84%, and a 100% Trap

To independently audit "whether the model actually built it right," the paper uses GeoGoal: **256 problems, 13,254 geometric predicates, 7,395 query expressions**. The evaluator looks only at the named-point coordinates of the final canvas — not the reasoning trace, not the final answer.

Three metrics must be kept apart:

- **SR = 95.94%**: predicate-level — what fraction of all predicates pass;
- **SC = 84.0%**: strict problem-level — **all predicates of a problem must pass** to count as success;
- **CR = 100%**: non-empty canvas rate — only requires that something was drawn.

**High SR doesn't mean many problems were "fully constructed correctly."** Extreme example: 100 problems, 10 predicates each, each problem wrong on exactly 1 — SR is 90%, SC is 0%.

**CR = 100% is even easier to misread.** It only says the model "successfully produced a non-empty canvas," not that the canvas matches the problem. As long as objects were drawn, CR is 100% even if every key constraint is missing.

Here's a set of numbers that says more. Split canvases into two groups by whether SC passed, and compare subsequent readout quality:

| Readout Match Threshold | SC Passed (n=215) | SC Failed (n=41) |
| ----------------------- | ----------------: | ---------------: |
| All Ti exactly match    |             31.2% |             0.0% |
| ≥90% Ti match           |             83.7% |             7.3% |
| ≥80% Ti match           |             94.4% |            22.0% |

**Note how to read this table**: it supports "construction passing correlates with readout quality," and incidentally proves something — **SC passing isn't sufficient either**, since strictly exact matches are only 31.2%.

The paper also runs a tolerance sweep that cleanly separates the two error classes: **structural targets plateau stably at about 88% across five orders of magnitude** (loosening tolerance doesn't help), while **numerical targets climb from 51% at 1e−4 to 94% at 3e−2**. This is the experimental evidence for the distinction in Section 06.

![](/images/draw2think/image-20260921121311151.png)

_Left: canvases built correctly have noticeably better readout quality. Right: structural errors don't improve with looser tolerance, numerical errors do — the two failure classes need completely different fixes._

---

## 09 What the Paper Proves, and What It Doesn't

One of the paper's most valuable contributions is separating three kinds of "correct." These three layers can't be merged:

| Layer                     | Question Answered                                                                  | Actual Evidence in the Paper                                                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Execution & readout**   | Did the tool run on the current input and return a quantity of the current canvas? | Typed interface, engine returns, error logs; numeric tools and silently undefined degenerate objects still exist                      |
| **Construction fidelity** | Does the current canvas satisfy the relations the problem specifies?               | GeoGoal's tolerance check on designated predicates; depends on the given construction text and a finite predicate set                 |
| **Mathematical proof**    | For all configurations satisfying the premises, does the conclusion hold?          | **This paper is explicitly limited to numerical instances**; it relies on the DAG being a proof-relevant trace, not a universal proof |

So the paper provides **instance-level verification**, not a **universally quantified proof**.

A very clean boundary statement:

> Measuring AB ⊥ CD on some instance only shows "this holds for this specific submitted canvas state" — it doesn't follow that "all legal configurations satisfy it."

One more phrase that must be mentioned: **"engine-exact" can't be stretched indefinitely**. The paper's actual audit is float coordinates + tolerance checking (absolute tolerance 4×10⁻⁷ or relative 0.1%), and the tool catalog includes numerical solving. It can be read as "controlled execution and checkable readout relative to a given state," not as "all return values are mathematically error-free symbolic certificates."

![](/images/draw2think/image-20260921121450040.png)

_Three layers of "correct" need three different verifiers. A green success marker can't replace them._

| Step | Tool Call                 | Deps  | Used by | Geometric Interpretation             | Semantic             |
| ---- | ------------------------- | ----- | ------- | ------------------------------------ | -------------------- |
| 1    | add_point(A, 0, 0)        | ∅     | {3,4,5} | Point existence                      | $A=(0,0)$            |
| 2    | add_point(B, 4, 0)        | ∅     | {3,7}   | Point existence                      | $B=(4,0)$            |
| 3    | add_segment(AB, A, B)     | {1,2} | {4}     | Two points $\rightarrow$ segment     | $\lvert AB \rvert=4$ |
| 4    | add_perp_line(L, A, AB)   | {1,3} | {6}     | Perpendicular through a point        | $L \perp AB$         |
| 5    | add_circle(c, A, 3)       | {1}   | {6}     | Center + radius $\rightarrow$ circle | $r=3$                |
| 6    | add_intersect(P, L, c, 1) | {4,5} | {7}     | Line–circle intersection (1st)       | $P=(0,-3)$           |
| 7    | query_distance(B, P)      | {2,6} | —       | Measurement readout                  | $d=5.0 \checkmark$   |

![](/images/draw2think/image-20260921121607862.png)

_One figure to understand what "dependency" means: delete any node and every downstream object fails with it._

---

## 10 Limitations, Failure Cases, and What to Know Before Reproducing

The limitations the authors themselves admit are written fairly honestly:

- **Step validity doesn't guarantee trajectory-strategy validity**: long construction chains, ambiguity awareness, and missed relations still fail;
- **Some problems don't need the engine at all**: forced construction only adds overhead; deciding "when to call" must be left to the strategy layer;
- **Only targets single numerical Euclidean instances**: doesn't cover general quantified proofs;
- **Narrow observation channel**: mainly returns objects, deltas, and values — **hard to detect layout or region-selection errors**;
- **Limited cross-model validation**: transfer is only observed on 112 OlympiadBench problems.

Among the public failure cases, two are especially worth examining:

- **Execution all correct, intent all wrong**: in one problem, the first 3 of 7 rounds failed and were rejected, after which everything was error-free, yet the final legal canvas **still missed 57% of GT predicates**. This shows the residual burden is at the **policy level**, not the execution level;
- **Query misalignment**: the model queried an irrelevant quantity, eventually worked its way to the answer, and appears to have "used the engine," but **the answer wasn't directly mediated by a readout**.

The paper also records a detail: failed calls mainly come from the model layer (referencing nonexistent objects, intersecting when preconditions aren't met, passing degenerate objects, parameter type errors), and about 5% of failed calls involve degenerate inputs that may silently produce undefined objects.

**If you want to reproduce, know three things:**

1. **Three different evaluation pipelines**, don't mix them: ordinary solving (answer accuracy), GeoGoal (construction fidelity — the model submits only a canvas, no answer), GenExam (outputs images, one extra rendering layer);
2. **The data wasn't used for training.** This is inference-time evaluation: existing benchmarks → frozen VLM → GeoGebra interaction → answer/canvas/image → evaluation;
3. **training-free ≠ no engineering investment.** The tool table was drafted with LLM assistance and refined over roughly eight rounds using trial-run failures; what the model sees includes purpose, parameter types, preconditions, naming, and angle-direction rules.

**Minor blemishes found while close-reading** (written out so you know to be careful when reproducing): MathVista's correct-problem count appears as both 203/208 and 204/208 across tables in the paper; the drawing-failure attribution table's category counts (17+14+4+5=40) don't cover the 48 failures in its title. These don't affect the main conclusions, but exact reproduction needs a consistent per-problem log.

---

## 11 Takeaways for Ordinary Developers, Researchers, and Interviewees

**For people building agents**: the most transferable lesson isn't "hook up a tool" — it's:

> **Design the intermediate workspace as executable objects, not just a stored explanation.**

Its value depends on two things: whether the workspace checks the constraints that actually matter, and whether the check results enter decisions **promptly**.

**For people doing evaluation**: three things you can borrow directly.

1. **Answer accuracy conflates "actually reasoned it out," "text shortcut," and "got lucky"** — so you need process metrics;
2. **Ablations should be designed by functional channel.** In the paper, the "no query" group can still obtain the same values through `add_*` return values — **deleting the function name isn't deleting the information**;
3. **Net benefit must count both Saves and Breaks.** Across eight planar settings, total N=3796, Save=133, Break=128, for a net gain of only 5 problems (about +0.132 percentage points). Showing only "rescued cases" hides problems "originally correct but broken by the tool."

**For interviewees**: if asked about "tool calling / agent memory and feedback," this paper gives a good answer skeleton:

> First, leave the high-uncertainty "propose the next step" to the model, and give the deterministic "execute and verify" to the engine;
> Second, distinguish "the action was executed" from "the action was what the problem needed";
> Third, distinguish instance-level verification from formal proof — don't call a tool's success marker a mathematical proof.

**For researchers**: the biggest gap this paper leaves is one sentence —

> **step-level verification ≠ strategy-level reasoning.**

GeoGebra will only tell you "I successfully drew this perpendicular." It won't tell you "**you didn't need to draw this perpendicular at all**."

The paper itself admits: the current canvas is mainly exposed to the model as flat JSON, and **DAG topology isn't yet used as a full planning input** — this is written into future work.

![](/images/draw2think/image-20260921122059768.png)

_The paper solved "is this step executed correctly," not "should this step be done at all." That's also the real entry point for the next research step._

---

## 12 Summary

1. **Draw2Think's core isn't "making VLMs draw geometry"** — it's turning the intermediate state in geometric reasoning into an executable, measurable, revisable constraint workspace;
2. **It solves step-level grounding**: whether an action was executed, and what the true values of the current canvas are;
3. **It doesn't solve strategy-level reasoning, nor formal proof**: intent understanding, whether the whole strategy is reasonable, and whether conclusions hold universally remain separate problems.

The line most worth taking away is still the opening example:

**The engine can measure a wrongly built figure with perfect accuracy. Its guarantee stops at the canvas — it doesn't cross over to intent.**

---

## References

- Draw2Think: Harnessing Geometry Reasoning through Constraint Engine Interaction — [https://arxiv.org/abs/2605.20743](https://arxiv.org/abs/2605.20743)
- Project page: [https://draw2think.github.io/](https://draw2think.github.io/)
- Code repository harness-geometry: [https://github.com/draw2think/harness-geometry](https://github.com/draw2think/harness-geometry)
- Newclid (symbolic solver; this paper uses its `check_numerical()` for predicate checking): [https://arxiv.org/abs/2411.11938](https://arxiv.org/abs/2411.11938)
- GeoGebra `Prove` command docs (distinguishing current-coordinate judgment from general symbolic judgment): [https://geogebra.github.io/docs/manual/en/commands/Prove/](https://geogebra.github.io/docs/manual/en/commands/Prove/)
- GeoGebra `NSolve` command docs (numerical solving): [https://geogebra.github.io/docs/manual/en/commands/NSolve/](https://geogebra.github.io/docs/manual/en/commands/NSolve/)
- MathCanvas: [https://arxiv.org/abs/2510.14958](https://arxiv.org/abs/2510.14958)

Copyright Ownership: Kstheme, Contributors: Kstheme
