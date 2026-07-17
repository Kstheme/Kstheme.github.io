---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - Paper Review
tags:
  - pri-tpg
  - llm
  - theorem-proving
  - paper
title: "This Paper Proposes Pri-TPG, a Training-Free Long-Range Reasoning Method That Introduces Structural Priors to Solve 'Structural Drift' in LLM Geometry Theorem Proving"
createTime: 2026/07/17 21:58:42
permalink: /article/pri-tpg-structural-priors/
---

> From content retrieval to structure retrieval — a training-free long-range reasoning approach.

Large language models exhibit a paradoxical behavior when solving mathematical proofs:

**As problems grow longer, model performance doesn't just gradually decline — it rapidly collapses.**

On the FormalGeo7K geometry reasoning benchmark, Vanilla ICL achieves 52.19% accuracy on shallow L1 problems, but drops to just 7.89% on L3. On longer L5 and L6 reasoning tasks, accuracy falls directly to 0.

After introducing structural priors, Pri-TPG using the same GPT-5 mini improves overall accuracy from 26.29% to 84.42%. With GPT-5.2, it reaches 89.29%.

The paper calls this phenomenon where standard ICL rapidly fails as reasoning depth increases:

> **Structural Drift.**

This concept reveals an important problem:

> LLMs may know what each theorem means, but they don't know the order in which these theorems should be organized.

A more specific question emerges:

> Can we avoid updating model parameters, and instead provide the LLM with an external structure that guides it to search only along more promising proof paths?

This is the problem that the paper **Non-Parametric Structural Priors for Geometry Theorem Prediction** aims to solve.

![](/images/pri-tpg/structural-drift.png)

# Paper Information Card

| Item               | Content                                                                    |
| ------------------ | -------------------------------------------------------------------------- |
| Paper Title        | Non-Parametric Structural Priors for Geometry Theorem Prediction           |
| Core Method        | Pri-TPG                                                                    |
| Research Task      | Multi-step geometry theorem prediction                                     |
| Technical Approach | RAG + Theorem Precedence Graph + Symbolic Execution                        |
| Key Feature        | Training-free — no gradient training required for theorem prediction       |
| Main Datasets      | FormalGeo7K, Geometry3K, GeoQA                                             |
| Main Result        | 89.29% on FormalGeo7K                                                      |
| Keywords           | Structural Drift, Structural Prior, LLM Planner, Neural-Symbolic Reasoning |

One-sentence summary:

> Pri-TPG doesn't directly retrieve answers — it retrieves proof structures from similar problems, then uses these structures to constrain the LLM's next action.

## 01. Why Is Geometry Theorem Proving Essentially a Search Problem?

A geometry problem that seems straightforward to a human is often not a single-step Q&A for an automated reasoning system, but a continuous decision process.

For example, a complete proof might require:

```text
Prove two sides are equal
↓
Determine triangle congruence
↓
Derive corresponding angle equality
↓
Construct similar triangles
↓
Derive the target ratio
```

The key isn't just "selecting the correct theorem."

More importantly:

**The currently selected theorem must be grounded in facts already established.**

If a theorem's prerequisites haven't been satisfied, it cannot be executed — even if it's semantically relevant to the target.

Formally, a geometry problem can be described as:

```text
Initial state S₀
↓ Apply theorem a₁
State S₁
↓ Apply theorem a₂
State S₂
↓
……
↓
Final state Sₜ satisfies goal g
```

A symbolic executor can check whether each step is valid, but it cannot automatically tell the model:

> Among hundreds of theorems, which one should be chosen next?

Assume a theorem library of 300 theorems and a proof requiring H steps. In the most naive case, the search space approaches:

```text
300ᴴ
```

The longer the reasoning chain, the more erroneous branches exist.

Therefore, the difficulty of geometry proving is not just insufficient knowledge — it's:

> **How to continuously find valid next steps among an exponentially growing set of candidate paths.**

![](/images/pri-tpg/state-transition.png)

## 02. Structural Drift: Knowing Theorems but Not Proof Structure

The most straightforward approach is In-Context Learning.

Give the LLM a few example proofs and let it imitate them to select theorems step by step.

But the paper finds that Vanilla ICL, while effective on shallow problems, rapidly collapses as proof length increases.

The authors attribute this to the fact that standard ICL provides "content examples" but no explicit "structural constraints."

| What the LLM may know                               | What the LLM may not know about structure            |
| --------------------------------------------------- | ---------------------------------------------------- |
| What the Pythagorean theorem is                     | Whether its prerequisites are satisfied now          |
| All triangle congruence criteria                    | Which criterion should be applied first              |
| Which theorems relate to angles vs. lengths         | Which theorems are inexecutable in the current state |
| Which theorems are semantically related to the goal | Whether a theorem lies on a valid proof path         |
| How to generate the next explanation                | Which branch to switch to after failure              |

The problem can be compressed into two sentences:

> **A theorem being semantically relevant doesn't mean it's executable in the current state.**
> **Selecting locally correct steps doesn't mean maintaining a globally coherent proof structure.**

Standard ICL is like making flat selections across the entire theorem library.

With each step facing a large set of candidates, early small errors compound and eventually derail the entire proof path.

This is Structural Drift.

![](/images/pri-tpg/flat-vs-structured.png)

## 03. Pri-TPG: Retrieving Not Knowledge, but Proof Structure

What does traditional RAG typically do?

```text
Input a question
↓
Retrieve relevant text or knowledge
↓
Include in the prompt
↓
Let the LLM generate an answer
```

It addresses:

> What should the model know?

Pri-TPG works differently.

It retrieves historical geometry problems similar to the current one, then extracts from their proof traces:

- Which theorems appear frequently;
- Which theorems typically appear first;
- Which theorems usually follow others;
- Which local reasoning structures may apply to the current problem.

Thus, Pri-TPG is closer to:

```text
Input a question
↓
Retrieve similar proofs
↓
Extract theorem usage order
↓
Build a theorem precedence graph
↓
Constrain the next reasoning step
```

The difference can be summarized as:

| Traditional RAG                      | Pri-TPG                                     |
| ------------------------------------ | ------------------------------------------- |
| Retrieves facts, passages, documents | Retrieves historical proof traces           |
| Answers "what to know"               | Answers "what to do first, what to do next" |
| Results added to prompt as text      | Results transformed into graph structure    |
| Primarily augments content           | Primarily constrains action space           |
| Content-Augmented                    | Structure-Augmented                         |

Therefore, Pri-TPG's key insight isn't simply adding more retrieval — it's:

> **Transforming retrieval results from "content context" into "reasoning structure."**

## 04. What Is the Theorem Precedence Graph?

Pri-TPG's core structure is the:

> **Theorem Precedence Graph (TPG).**

It's a directed graph.

| Graph Element            | Meaning                                                           |
| ------------------------ | ----------------------------------------------------------------- |
| Node (v)                 | A theorem                                                         |
| Edge ($u \rightarrow v$) | Theorem u produces results that support using theorem v           |
| Edge weight              | Frequency of this theorem transition in similar historical proofs |
| START node               | Theorems more likely to be used first at the start of a proof     |

For example, a local path might be:

```text
Isosceles triangle判定
↓
Isosceles triangle properties
↓
Angle equality
↓
Similar triangle判定
```

Instead of choosing from 300 theorems arbitrarily, TPG tells the model:

> Now that you've just used a certain theorem, the more reasonable next candidates are typically its successors in the graph.

TPG is not a fixed proof chain.

It's more like a local navigation map:

| Fixed Proof Chain             | TPG                                           |
| ----------------------------- | --------------------------------------------- |
| Must follow a single path     | Can retain multiple feasible successors       |
| Close to template replication | Closer to structural constraints              |
| Sensitive to variation        | Allows the LLM to choose within a local graph |
| Lacks flexibility             | Preserves some planning freedom               |

The paper notes that in FormalGeo7K experiments, after candidate filtering, the number of theorems per step is reduced from roughly 300 to about 30 — approximately a 90% reduction in single-step search space.

## 05. Pri-TPG Doesn't Use Just a Single Static Graph

If we simply count theorem order across all historical proofs and build one global graph, there's an obvious problem:

> Different problems require different theorem structures.

Circles, triangles, parallelograms, length calculations, angle proofs — each has very different proof paths.

Therefore, Pri-TPG refines structural priors layer by layer.

![](/images/pri-tpg/workflow.png)

_Starting from the global theorem library, the system progressively narrows the search space to the set of theorems most likely valid in the current state through retrieval, graph construction, and symbolic verification._

### Three Layers of Structural Priors

| Layer                | Input                                           | Main Operation                                               | Problem Solved                                               |
| -------------------- | ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Global Prior         | All historical proof traces                     | Statistics of global theorem precedence                      | Establishes domain-level structure                           |
| Query-Adaptive Prior | Current problem and similar historical problems | Build problem-related candidate pool and local TPG           | Eliminates theorems irrelevant to the current problem        |
| State-Aware Prior    | Current symbolic state and previous action      | Symbolic pruning, structural localization, candidate ranking | Eliminates inexecutable or structurally unreasonable actions |

### Layer 1: Global Prior

The global prior comes from all historical proof traces.

It answers:

> In the entire dataset, which theorems frequently have precedence relationships?

This layer provides domain-level structure.

But it's too broad — the global graph contains many theorems irrelevant to the current problem.

### Layer 2: Query-Adaptive Prior

Pri-TPG retrieves similar historical problems based on the current problem.

Retrieval conditions include:

| Input Modality           | Role                                        |
| ------------------------ | ------------------------------------------- |
| Problem text             | Captures semantics and goal type            |
| Geometry image           | Captures图形 structure                      |
| Initial formalized state | Captures already established symbolic facts |

The system uses multimodal encoders to map this information into a unified vector space, then retrieves Top-K similar problems from the training set.

Next, it extracts the theorems used in these similar problems and merges their corresponding TPGs to form a problem-specific:

```text
Candidate theorem set Lq

Query-relative graph Gq
```

This layer answers:

> For this specific problem, which theorem sequences are more likely useful?

If a must-use theorem is missed during retrieval, no amount of LLM reasoning power can complete the proof later.

Results for different retrieval sizes:

| Top-K | Total | Easy  | Medium | Hard  |
| ----- | ----- | ----- | ------ | ----- |
| 15    | 71.86 | 83.37 | 52.96  | 28.69 |
| 30    | 79.00 | 94.39 | 61.64  | 30.33 |
| 100   | 80.29 | 95.32 | 64.30  | 30.33 |
| 200   | 84.42 | 96.14 | 73.05  | 40.98 |

As K increases, overall performance steadily improves, especially on medium and hard problems. Long-chain proofs require recalling more specialized theorems — if a critical theorem is missed, subsequent planning cannot succeed.

### Layer 3: State-Aware Prior

Even if a theorem appears in similar problems, it may not be usable right now.

Therefore, Pri-TPG continues to prune based on the current state at each step.

| Mechanism                | Approach                                                    | Role                                              |
| ------------------------ | ----------------------------------------------------------- | ------------------------------------------------- |
| Symbolic Pruning         | Check whether candidate theorem prerequisites are satisfied | Eliminates mathematically inexecutable actions    |
| Structural Localization  | Prioritize successors of the previous theorem in the graph  | Maintains local reasoning coherence               |
| Candidate Prioritization | Rank based on goal, graph structure, and history            | Presents higher-value candidates to the LLM first |

The system progressively narrows from:

```text
Global theorem library
↓
Problem-relevant theorem set
↓
Currently executable theorems
↓
Structurally reasonable theorems
```

![](/images/pri-tpg/three-layer-prior.png)

## 06. What Do the LLM, Graph, and Symbolic Executor Each Handle?

Pri-TPG is neither a pure LLM system nor a pure symbolic search system.

It's a neural-symbolic closed loop.

| Component        | Role                | Main Responsibility                                        |
| ---------------- | ------------------- | ---------------------------------------------------------- |
| LLM              | Planner             | Selects the next theorem from candidates                   |
| TPG              | Structural Prior    | Provides theorem precedence, narrows action space          |
| Symbolic Solver  | Executor / Verifier | Checks prerequisites, executes theorems, updates state     |
| Retrieval Module | Retriever           | Recalls relevant proof structures from historical problems |
| Current State    | Environment State   | Records known facts and newly added facts                  |

The loop is:

```text
LLM selects a theorem
↓
Symbolic executor verifies and executes
↓
Update current state
↓
Re-filter graph and candidates based on new state
↓
LLM selects again
```

This closely resembles the typical Agent paradigm:

| Agent Paradigm       | Pri-TPG Component                 |
| -------------------- | --------------------------------- |
| Think                | LLM analyzes candidates           |
| Act                  | Select a theorem                  |
| Environment Feedback | Symbolic executor returns results |
| Update State         | Update symbolic facts             |
| Replan               | Re-select based on new state      |

Thus, while the paper's task is geometry proving, it actually studies a more general Agent problem:

> How to enable LLMs to perform long-horizon planning in environments with large action spaces, strict rules, and clear feedback?

## 07. Candidates Are Not Simply Sorted by Similarity

Pri-TPG doesn't just list retrieved theorems.

It ranks candidates based on three types of information:

| Score Component | Meaning                                           | Intuition                                        |
| --------------- | ------------------------------------------------- | ------------------------------------------------ |
| (s\_{goal})     | Candidate's relevance to the final goal           | Theorems closer to the goal ranked higher        |
| (s\_{graph})    | Structural relationship to current graph position | Successors of the previous theorem ranked higher |
| (s\_{hist})     | Historical repetition and failure penalty         | Avoid loops and repeated无效 attempts            |

The overall form is:

```text
Candidate score = Goal relevance + Graph structural relevance - Historical repetition penalty
```

This scoring doesn't replace the LLM's final decision — it surfaces more promising candidates first, then lets the LLM reason based on the problem, state, and historical trajectory.

## 08. Why Must It Be a Closed Loop Instead of Generating the Full Proof at Once?

The authors also designed a Single-pass version.

It still has the query-adaptive TPG, but requires the LLM to generate the complete theorem sequence at once, with the symbolic executor only verifying at the end.

Results:

| Method      | Overall | Easy | Medium | Hard |
| ----------- | ------- | ---- | ------ | ---- |
| Vanilla ICL | 26.3    | 39.7 | 6.9    | 0.0  |
| Single-pass | 34.3    | 53.3 | 5.7    | 0.0  |
| Pri-TPG     | 84.3    | 96.1 | 73.0   | 41.0 |

Even with structural priors, Single-pass scores 0 on Hard tasks.

This shows that one-shot planning cannot handle errors that occur mid-proof in long chains.

| Single-pass                         | Iterative Pri-TPG                                  |
| ----------------------------------- | -------------------------------------------------- |
| Generates the full sequence at once | Selects one step at a time                         |
| Unified verification at the end     | Immediate verification after each step             |
| Cannot correct mid-proof            | Can re-plan based on feedback                      |
| Errors accumulate along the chain   | Errors are截断 in time                             |
| Long tasks容易 collapse             | Medium-to-long tasks are significantly more stable |

Formal reasoning must be a closed loop:

```text
Plan
↓
Execute
↓
Feedback
↓
Correct
```

![](/images/pri-tpg/singlepass-vs-iterative.png)

_Single-pass scores 0 on Hard tasks even with query-adaptive prior; step-by-step symbolic feedback is critical for medium-to-long proofs._

One-sentence summary:

> **The graph structure solves "where to go," symbolic feedback solves "what to do after going wrong."**

## 09. What Does the Experiment Actually Prove?

A proper close reading of experiments shouldn't just look at who scored higher.

More importantly:

> Which research hypothesis does each experiment validate?

Pri-TPG's experiments form a clear chain of evidence.

### Main Results

| Method                   | Trained | Total | L1    | L2    | L3    | L4    | L5    | L6    |
| ------------------------ | ------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Vanilla ICL (GPT-5 mini) | No      | 26.29 | 52.19 | 23.67 | 7.89  | 5.10  | 0.00  | 0.00  |
| Pri-TPG (GPT-5 mini)     | No      | 84.42 | 98.54 | 93.09 | 78.20 | 64.33 | 58.06 | 23.33 |
| Pri-TPG (GPT-5.2)        | No      | 89.29 | 99.16 | 96.28 | 87.92 | 77.07 | 66.13 | 30.00 |
| FGeo-HyperGNet           | Yes     | 88.36 | 96.24 | 91.76 | 87.59 | 82.17 | 56.45 | 56.67 |

From the table:

- Pri-TPG performs strongly from L1 to L5;
- Vanilla ICL rapidly collapses with increasing depth;
- Pri-TPG still shows significant decline on L6;
- Trained specialized models maintain advantages on extremely long chains.

### Evidence Chain 1: What Do RAG and TPG Each Contribute?

| Setting     | Iterative | RAG | TPG | Total | Hard  |
| ----------- | --------- | --- | --- | ----- | ----- |
| Vanilla ICL | ✓         | ✗   | ✗   | 26.29 | 0.00  |
| w/o TPG     | ✓         | ✓   | ✗   | 72.64 | 22.95 |
| Pri-TPG     | ✓         | ✓   | ✓   | 84.42 | 40.98 |

This set of experiments shows:

| Module             | Main Contribution                   |
| ------------------ | ----------------------------------- |
| RAG                | Narrows the candidate theorem range |
| TPG                | Organizes theorem precedence        |
| Iterative Feedback | Supports mid-course correction      |

RAG improves performance from 26.29% to 72.64%, demonstrating the importance of candidate narrowing.

TPG further improves from 72.64% to 84.42%, showing:

> Finding relevant theorems doesn't mean knowing how to organize them.

### Evidence Chain 2: More Precise Structural Priors → Better Results

| Setting              | Total | Easy  | Medium | Hard  |
| -------------------- | ----- | ----- | ------ | ----- |
| Vanilla ICL          | 26.29 | 39.65 | 6.86   | 0.00  |
| Global Prior         | 29.71 | 40.70 | 14.89  | 4.10  |
| Query-Adaptive Prior | 58.43 | 72.40 | 41.61  | 18.85 |
| State-Aware Pri-TPG  | 84.42 | 96.14 | 73.05  | 40.98 |

These results show that using only the global graph provides limited help.

The real improvements come from:

1. Making the structure relevant to the current problem;
2. Making the structure relevant to the current state.

Thus, bigger structural priors aren't better — what matters is:

> **The closer they are to the current problem and current reasoning state, the better.**

### Evidence Chain 3: Does the Method Depend on a Specific LLM?

| Backbone          | Total | Easy  | Medium | Hard  |
| ----------------- | ----- | ----- | ------ | ----- |
| DeepSeek v3.2     | 83.57 | 95.56 | 72.10  | 39.34 |
| GPT-5 mini        | 84.42 | 96.14 | 73.00  | 40.98 |
| Claude 4.5 Sonnet | 87.07 | 97.31 | 77.54  | 48.36 |
| Gemini 3.0 Pro    | 88.43 | 97.54 | 81.56  | 48.36 |
| GPT-5.2           | 89.29 | 97.89 | 83.92  | 48.36 |

Different models show different absolute scores, but all achieve strong results after integrating Pri-TPG.

This suggests that Pri-TPG is more like a pluggable external reasoning scaffold than a technique specific to any single model.

![](/images/pri-tpg/experiment-chain.png)

**Summary Card:**

> RAG solves "what are the candidates," TPG solves "how to organize candidates," Symbolic Feedback solves "how to catch errors in time."

## 10. How Should 89.29% Be Properly Understood?

89.29% is a strong result, but its evaluation boundary needs attention.

| Evaluation Condition             | Specific Setting                     |
| -------------------------------- | ------------------------------------ |
| Input format                     | Uses ground-truth formal inputs      |
| Main evaluation focus            | Theorem planning and symbolic search |
| Includes full auto-formalization | No                                   |
| Time limit per problem           | 600 seconds                          |
| Max reasoning steps              | 20                                   |
| Top-K retrieval size             | 200                                  |
| Max recovery attempts            | 3                                    |
| Multiple LLM calls               | Yes                                  |

To isolate "reasoning and search ability," the paper provides correct formalized inputs directly to all methods.

That is, the system does not start from completely raw natural language and geometry images to independently complete the entire formalization process.

More precisely:

> 89.29% mainly measures the system's ability to perform theorem planning and symbolic search given correct formalized representations.

Therefore:

> **Training-free doesn't mean no cost, nor does it mean the end-to-end system is fully solved.**

"Non-parametric" here doesn't mean the entire system has no parametric models either.

| System still depends on                             | Exists? |
| --------------------------------------------------- | ------- |
| Pre-trained LLM                                     | Yes     |
| Multimodal Embedding model                          | Yes     |
| Historical proof database                           | Yes     |
| Additional gradient training for theorem prediction | No      |

"Non-parametric" mainly means:

> The theorem prediction strategy isn't re-trained into model weights — it's provided as external retrieval graphs and structural priors at inference time.

## 11. What Limitations Does the Paper Explicitly Acknowledge?

| Limitation                            | Specific Manifestation                                         |
| ------------------------------------- | -------------------------------------------------------------- |
| Lower inference efficiency            | Multi-step planning requires repeated LLM calls                |
| Very long chains remain difficult     | Significant accuracy drop on L6                                |
| TPG is local                          | Primarily encodes local precedence, not full global depth      |
| Single-step errors have high impact   | One failed step can破坏 the entire subsequent proof chain      |
| Still relies on base model capability | Stronger backbones generally yield higher absolute performance |

These limitations show:

> Pri-TPG significantly mitigates local unstructured search, but hasn't fully solved global long-range consistency.

## 12. My Further Analysis

The following questions aren't all explicitly raised by the paper, but are延伸 judgments based on its method design and experimental results.

### 1. Is TPG a Logical Dependency Graph or an Empirical Order Graph?

The paper describes edges (u \rightarrow v) as prerequisite dependencies between theorems.

But the graph is primarily constructed from historical解题 trajectories.

| Strict Logical Dependency           | Historical Empirical Order            |
| ----------------------------------- | ------------------------------------- |
| A is a necessary prerequisite for B | A frequently appears before B         |
| Strong causal implication           | May just be data bias                 |
| More stable across datasets         | May be influenced by proof style      |
| Can be formally verified            | May just be statistical co-occurrence |

A appearing before B in historical trajectories doesn't necessarily mean A is a strict necessary condition for B.

It could also be:

- A common proof convention in the dataset;
- The training set prefers a certain proof route;
- Other valid orders also exist;
- The two theorems simply co-occur frequently.

Thus, a question worth further investigation is:

> Does TPG learn causal dependencies or high-frequency path templates?

### 2. Local Structural Consistency ≠ Global Proof Consistency

Pri-TPG primarily performs structural localization based on successors of the previous theorem.

| Local Advantage                       | Potential Problem                            |
| ------------------------------------- | -------------------------------------------- |
| Maintains step-by-step coherence      | May restrict cross-branch switches           |
| Reduces无效 successors                | May miss alternative proof routes            |
| Reduces short-range search difficulty | Doesn't guarantee eventual goal reachability |
| Leverages historical local patterns   | May struggle with backtracking               |

Locally reasonable consecutive steps don't necessarily lead to the target.

This is also a key reason why very long chains like L6 remain difficult.

### 3. Failure Affects Ranking but Doesn't Actually Update the Graph

Pri-TPG applies history penalty to repeated and failed theorems.

But this mechanism mainly adjusts candidate ranking within the current推理.

| Current Mechanism                 | Further Structure Learning               |
| --------------------------------- | ---------------------------------------- |
| Down-weights failed theorems      | Locates the node or edge causing failure |
| Avoids repeated attempts          | Updates the graph structure itself       |
| Effective within the current task | Can accumulate经验 across tasks          |
| Local ranking adjustment          | Global credit assignment                 |

It hasn't yet truly achieved:

> Using a single failure to back-locate which node or edge in the graph is problematic, and updating the overall structure accordingly.

### 4. High Recall Rate vs. Reasoning Cost: A Continuing Tension

Experiments show that larger K yields higher accuracy, especially on medium-to-hard tasks.

| Larger K                              | Smaller K                               |
| ------------------------------------- | --------------------------------------- |
| Less likely to miss critical theorems | Lower retrieval cost                    |
| Higher candidate coverage             | More compact candidate set              |
| More reliable for long chains         | More likely to miss key theorems        |
| Greater subsequent ranking pressure   | Higher requirement on retrieval quality |

Balancing:

- Smaller retrieval scale;
- Lower invocation cost;
- Higher recall of critical theorems

remains both an engineering and research challenge.

## 13. From Pri-TPG to Dynamic Self-Evolving TPG

Pri-TPG currently solves:

> How to extract local structural priors from historical successful trajectories to help LLMs narrow the search space.

But it hasn't fully solved:

> After a reasoning failure, how to localize the error and update the global graph structure?

A natural follow-up direction is:

### Dynamic Self-Evolving TPG

That is:

> Using both successful and failed trajectories to enable the theorem precedence graph to continuously update during inference.

### Pri-TPG vs. Dynamic TPG

| Dimension          | Pri-TPG                                               | Dynamic Self-Evolving TPG                           |
| ------------------ | ----------------------------------------------------- | --------------------------------------------------- |
| Main data source   | Historical successful trajectories                    | Successful + failed trajectories                    |
| Graph state        | Built during inference, structure mainly from history | Continuously updated during current inference       |
| Failure handling   | Down-weights failed actions                           | Credit assignment across nodes, edges, and subpaths |
| Search method      | Local structure guidance                              | Local guidance + global backtracking                |
| Learning goal      | Avoid unstructured exploration                        | Learn to correct erroneous structures               |
| Global consistency | Limited                                               | Core optimization target                            |

A possible process:

```text
LLM reasons along the current TPG
↓
Symbolic executor detects path failure or goal unreachable
↓
Locate the node or edge that first made subsequent completion impossible
↓
Perform backward credit assignment on the relevant local structure
↓
Lower erroneous edge weights, raise alternative path weights
↓
Update the TPG
↓
Re-plan
```

The corresponding research question can be formulated as:

> Can an LLM agent use failed reasoning trajectories to perform backward credit assignment over a theorem precedence graph, thereby improving global consistency in long-horizon theorem proving?

Further exploration directions:

| Direction                                          | Potential Role                                    |
| -------------------------------------------------- | ------------------------------------------------- |
| Tree-search-based global backtracking              | Switch to alternative branches after failure      |
| Value propagation on the graph                     | Back-propagate final outcomes to nodes and edges  |
| Node-level credit assignment                       | Determine which theorem choice was critical       |
| Edge-level credit assignment                       | Determine which theorem transition was unreliable |
| Joint modeling of success and failure trajectories | Learn both what to do and what not to do          |
| Uncertainty-driven expansion                       | Broaden search at high-uncertainty positions      |
| Hierarchical or hypergraph                         | Express more complex multi-theorem dependencies   |

![](/images/pri-tpg/self-evolving.png)

_Perhaps the true next step isn't building larger static reasoning graphs, but enabling graphs to self-correct from failures._

## 14. Conclusion: From Content Augmentation to Structure Augmentation

Pri-TPG appears on the surface to be a geometry theorem prediction paper.

But the idea it truly conveys is:

> LLM reasoning failures sometimes aren't because the model doesn't know the answer, but because it faces an action space that's too large without stable structural navigation.

This paper addresses the problem through three mechanisms:

| Module            | Problem Solved                                           |
| ----------------- | -------------------------------------------------------- |
| RAG               | Find theorems relevant to the current problem            |
| TPG               | Organize precedence relationships between these theorems |
| Symbolic Feedback | Verify, update, and re-plan after each step              |

It transforms the LLM from a free generator into a structurally constrained high-level planner.

The core shift can be summarized as:

| Past                               | Direction Represented by Pri-TPG                   |
| ---------------------------------- | -------------------------------------------------- |
| Retrieve knowledge                 | Retrieve structure                                 |
| One-shot generation                | Closed-loop execution                              |
| Unconstrained reasoning            | Structured search                                  |
| Only leverages successful examples | Future: leverage both success and failure feedback |
| Internal model memory strategy     | External interpretable structural prior            |

Of course, Pri-TPG still primarily solves local theorem ordering.

For extremely long proofs, it still faces:

| Unresolved Issues                     | Manifestation                                |
| ------------------------------------- | -------------------------------------------- |
| Insufficient global consistency       | Significant performance drop on L6           |
| Limited backtracking ability          | Mainly relies on local successors            |
| Inadequate use of failure information | Only down-weights, doesn't update the graph  |
| High reasoning cost                   | Multiple LLM calls and large-scale retrieval |

But this precisely shows that this paper isn't an end point, but an excellent starting point.

Future long-horizon Agents may need not only stronger models, larger contexts, and more tools, but also a self-evolving reasoning system capable of:

> **Extracting structure from history, validating structure during execution, and updating structure from failure.**

And that may be the most值得 pursuing question that Pri-TPG leaves for Agent research.
