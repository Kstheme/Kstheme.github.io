---
url: /article/transformer-self-attention/index.md
---
> Interviewer: "Please explain in detail how the self-attention mechanism in the Transformer model works. Why is it better suited for handling long sequences than RNN?"
> If you can fluently answer all of the following, this question is a perfect score.

***

## 1. How Does the Self-Attention Mechanism Work?

### 1.1 Intuition: Let Every Word See the Entire Sentence

Consider the input sentence:
**"The cat sat on the mat because it was tired."**
We want the model to understand that "it" refers to "cat."

* **RNN**: Must read word by word from the first to the last, compressing information into a single hidden state — the farther away, the more information is lost.
* **Self-Attention**: Lets "it" directly attend to all other words in the sentence and automatically assign attention weights — high weight to "cat," low weight to "mat."

This is the core idea of self-attention: **each word computes relevance with every other word in the sequence, then aggregates information.**

***

### 1.2 Core Computation: Scaled Dot-Product Attention (with Shape Derivation)

Each input word is projected into three vectors:

* **Query (Q)**: Represents "what I am looking for right now."
* **Key (K)**: Represents "what information labels I have."
* **Value (V)**: Represents "the actual information content I contribute."

> Let the input sequence length be $n$ and the embedding dimension of each token be $d\_{model}$.
> In the attention computation, Q and K have dimension $d\_k$, and V has dimension $d\_v$ (typically $d\_k = d\_v$).

#### Step 1: Linear Projection

* Input matrix $X$ shape: $(n, d\_{model})$
* Learnable weight matrices:
  * $W^Q$ shape: $(d\_{model}, d\_k)$
  * $W^K$ shape: $(d\_{model}, d\_k)$
  * $W^V$ shape: $(d\_{model}, d\_v)$
* Obtain three matrices:
  $$
  Q = XW^Q \quad \text{shape} (n, d\_k) \\
  K = XW^K \quad \text{shape} (n, d\_k) \\
  V = XW^V \quad \text{shape} (n, d\_v)
  $$

#### Step 2: Compute Attention Scores

Take the dot product between Query and all Keys:
$$
\text{Scores} = QK^T
$$

* $Q$ shape $(n, d\_k)$, $K^T$ shape $(d\_k, n)$ → resulting shape $(n, n)$
* Each element $\text{score}\_{ij}$ represents the raw attention that position $i$ pays to position $j$.

#### Step 3: Scale

Divide by $\sqrt{d\_k}$ to prevent dot products from growing too large:
$$
\text{Scaled Scores} = \frac{QK^T}{\sqrt{d\_k}}
$$

Shape remains $(n, n)$.

#### Step 4: Softmax Normalization

Apply softmax to each row to obtain the attention weight matrix $A$:
$$
A\_{ij} = \frac{\exp(\text{score}*{ij}/\sqrt{d\_k})}{\sum*{k=1}^n \exp(\text{score}\_{ik}/\sqrt{d\_k})}
$$

Shape remains $(n, n)$, and each row sums to 1.
Now $A\_{ij}$ represents the proportion of information that position $i$ gathers from position $j$.

#### Step 5: Weighted Sum

Weight and sum all Value vectors using the attention weights:
$$
\text{Attention}(Q, K, V) = A \cdot V
$$

* $A$ shape $(n, n)$, $V$ shape $(n, d\_v)$ → resulting shape $(n, d\_v)$

**The output at each position dynamically fuses contextual information from the entire sequence.**

***

### 1.3 Multi-Head Attention: Simultaneous Focus Across Subspaces (with Shape Derivation)

A single head can only capture one type of relationship. Transformer uses **multi-head attention** to let the model learn from multiple perspectives simultaneously.

* Set the number of heads $h$; each head has Q, K dimension $d\_k = d\_{model} / h$ and V dimension $d\_v = d\_{model} / h$.

* For each head $i$, there are independent projection weights:
  * $W\_i^Q$ shape: $(d\_{model}, d\_k)$
  * $W\_i^K$ shape: $(d\_{model}, d\_k)$
  * $W\_i^V$ shape: $(d\_{model}, d\_v)$

* Compute attention for each head:
  $$
  \text{head}\_i = \text{Attention}(QW\_i^Q, KW\_i^K, VW\_i^V)
  $$
  Each $\text{head}\_i$ has shape $(n, d\_v)$.

* Concatenate all heads:
  $$
  \text{Concat}(\text{head}*1, \dots, \text{head}*h)
  $$
  Since $h \times d\_v = d*{model}$, the concatenated shape reverts to $(n, d*{model})$.

* Finally, pass through an output projection matrix $W^O$ (shape $(d\_{model}, d\_{model})$):
  $$
  \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}\_1, \dots, \text{head}*h) W^O
  $$
  Output shape remains $(n, d*{model})$.

> This ensures that multi-head attention transforms dimensions while seamlessly integrating with the subsequent residual connection and layer normalization.

***

### 1.4 Compensating for Missing Position Information: Positional Encoding

Self-attention is **permutation equivariant** — shuffling the input order merely shuffles the output correspondingly.
To inject sequence order, Transformer adds **positional encoding** $P$ to the input word embeddings. $P$ also has shape $(n, d\_{model})$ and is obtained through sine/cosine functions or learned embeddings.

The input becomes $X + P$, allowing the model to distinguish positional relationships like "A comes before B."

***

## 2. Why Is Self-Attention Better Than RNN for Long Sequences?

| Dimension | RNN | Transformer Self-Attention |
|-----------|-----|----------------------------|
| **Information Flow Path Length** | O(n), long-range dependencies decay severely | **O(1)**, any two positions connect directly |
| **Parallelization** | Must compute sequentially along time steps | Matrix multiplication, entire sequence processed in parallel at once |
| **Gradient Stability** | BPTT involves chain multiplication, prone to vanishing/exploding | Gradients only pass through Softmax and linear layers; path is very short |
| **Memory Bottleneck** | Compressed into a fixed-size hidden state; information gets diluted | Each output can directly access the full Value matrix |
| **Interpretability** | Difficult | Attention weight matrix $n \times n$ can be directly visualized |
| **Complexity** | O(n·d²) but sequential | Standard O(n²·d), but highly parallelized, actually faster than RNN |

### Detailed Explanation

1. **Information flow path length is O(1)**
   RNN requires $n$ recursive steps to pass information from the first word to the $n$th word, causing severe decay in long-range dependencies. Self-attention establishes direct connections between any two positions in a single matrix multiplication — the path is always 1.

2. **Fully parallelized, dramatic training efficiency**
   RNN must compute step by step along the time dimension and cannot parallelize across the sequence. Self-attention is fundamentally matrix multiplication — the entire sequence is processed in one shot, achieving extremely high GPU utilization and dozens of times faster training.

3. **More stable gradient propagation**
   RNN's BPTT involves chain multiplication, making it prone to vanishing or exploding gradients. Self-attention's gradients only pass through Softmax and linear mappings — the path is extremely short, gradients are stable, and long-distance dependency signals are much easier to learn.

4. **No fixed-size memory bottleneck**
   RNN compresses all history into a fixed-dimensional hidden state; information in long sequences gets diluted. In self-attention, each output can directly access the full Value matrix, and memory capacity scales naturally with sequence length.

5. **Stronger interpretability**
   The attention weight matrix $A$ ($n \times n$) can be directly visualized, clearly showing word-to-word dependency relationships, making debugging and analysis easier.

6. **Complexity comparison and advanced approaches**
   Standard self-attention has $O(n^2)$ time complexity, but matrix multiplication is highly parallelized — for sequences up to several thousand tokens, it is practically faster than RNN's sequential computation. For extremely long sequences, variants such as sparse attention, linear attention, and state space models reduce complexity to $O(n)$ or $O(n\log n)$ while preserving global receptive fields, still outperforming RNN.

***

## 3. Interview Answer Template (Concise Version)

When asked this in an interview, here's how to answer concisely:

### How does self-attention work?

> Imagine every word is at a roundtable discussion, able to talk directly to any other word — not like RNN where messages have to be passed along one by one.
>
> The computation is straightforward:
> Input sequence X, shape `[n, d_model]`.
> Use three matrices to project X into Q (query), K (key), and V (value), with shapes `[n, d_k]`, `[n, d_k]`, `[n, d_v]` respectively.
> In standard multi-head attention, `d_k = d_v = d_model / h`.
>
> Then:
>
> 1. **Compute scores**: `S = QKᵀ` → shape `[n, n]`.
> 2. **Scale**: `S / √d_k`.
> 3. **Softmax**: apply per row to get weight matrix A (`[n, n]`, each row sums to 1).
> 4. **Weighted sum**: `A V` → output `[n, d_v]`, each word fuses global context.
>
> Multi-head attention runs the above process $h$ times in parallel, each head learning different relationships. Finally, all head outputs are concatenated into `[n, d_model]` and linearly projected.
> Since self-attention is order-agnostic, we add **positional encoding** to the input to inform the model about word positions.

### Why is it better than RNN for long sequences?

> Four core reasons:
>
> 1. **Extremely short path**: Any two words connect directly (O(1)), so long-range dependencies never get lost.
> 2. **Fully parallel**: It's all matrix multiplication — throw the entire sentence into the GPU at once, training is dozens of times faster.
> 3. **Stable gradients**: Gradients only pass through Softmax and linear layers — no chain multiplication, much more stable optimization.
> 4. **No memory bottleneck**: Can directly access V from any position — lossless information, no need to "forget."
>
> In a nutshell: self-attention replaces RNN's **step-by-step recursive compression** with **global direct interaction**, fundamentally solving the core pain point of long-sequence modeling.

***

## 4. Summary

* **Self-Attention**: Q, K, V scaled dot-product + Softmax + weighted sum → each word aggregates global information.
* **Multi-Head Attention**: Multiple subspaces learn in parallel, concatenated then linearly transformed.
* **Compared to RNN**: O(1) path length, fully parallel, stable gradients, no memory bottleneck, strong interpretability.

**If you can clearly explain QKV shape transformations, why we divide by $\sqrt{d\_k}$, how multi-head concatenation works, and contrast with RNN's four advantages, the interviewer will be very impressed.**

***

**If you found this useful, feel free to like, share, and forward to friends who are practicing for interviews!**
