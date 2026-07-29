---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - Machine Learning
tags:
  - positional-encoding
  - transformer
  - llm
  - hung-yi-lee
title: "The Most Complete Breakdown of LLM Positional Encoding: Absolute, Relative, Rotary, and No Positional Encoding at All"
createTime: 2026/07/22 17:53:20
permalink: /article/positional-encoding-guide/
copyright: Kstheme
---

> **Abstract**: Transformer's Self-Attention itself does not perceive token order — shuffling the tokens of "the cat ate the fish" produces exactly the same Attention output. This article systematically traces the complete technical evolution of Positional Embedding: from the clock-hand analogy of Sinusoidal absolute positional encoding, to ALiBi and T5 relative positional biases, to the mathematical principles of RoPE (Rotary Position Embedding), and finally exploring Train Short Test Long extension methods and the "no positional encoding" NoPE/DroPE approaches.

Copyright Ownership: Kstheme, Contributors: Kstheme

---

## 01 A Basic Question: How Does Transformer Know the Order of Tokens?

The original Transformer has no way to consider the order of input. How does it process input tokens?

Input tokens become Embeddings, which then become inputs to a Layer. Each layer contains Self-Attention, which outputs another set of tokens.

But there's a problem: **Self-Attention itself does not consider the order of input tokens.**

![](/images/positional-encoding/pe-01.png)

If I swap the positions of Token A and Token C and compute Self-Attention again, you'll find that the output $O_D$ is completely unaffected.

![](/images/positional-encoding/pe-02.png)

So we must find a way to "inject" positional information into the Transformer. This is what Positional Embedding solves.

This article covers:

1. **Absolute Positional Embedding** (Sinusoidal)
2. **Relative Positional Embedding** (ALiBi, T5)
3. **RoPE** (Rotary Position Embedding)
4. **Train Short, Test Long** (Position Interpolation, Frequency-Based Methods, YaRN, Dynamic Scaling, LongRoPE)
5. **No Positional Embedding** (NoPE, DroPE)

---

## 02 Absolute Positional Embedding

The earliest Positional Embedding idea was straightforward: **assign a special Embedding to each position**. This Embedding represents positional information.

We use $P_0$ through $P_3$ to represent the special Embedding for each position, then **add** this Embedding to our vectors.

![](/images/positional-encoding/pe-03.png)

When tokens change order, their Embeddings become different, and the final Self-Attention output also differs.

![](/images/positional-encoding/pe-04.png)

### Sinusoidal Positional Embedding

Reference: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

This Positional Embedding existed from the birth of the Transformer.

Let $d$ represent the length of the Embedding ($d = 128 \text{ or } 256$).

![](/images/positional-encoding/pe-05.png)

We use $\boldsymbol{p}[i]$ to represent its values.

![](/images/positional-encoding/pe-06.png)

How does Sinusoidal Position Embedding construct all Embeddings? The formula is:

$$
\begin{align}
\boldsymbol{p_k}[2i] &= \sin\left(\frac{k}{10000^{2i/d}}\right) \\
\boldsymbol{p_k}[2i+1] &= \cos\left(\frac{k}{10000^{2i/d}}\right)
\end{align}
$$

#### Visual Understanding

If we take $\boldsymbol{p_0}[0]$ through $\boldsymbol{p_{49}}[0]$, we'll see a Sine function.

![](/images/positional-encoding/pe-07.png)

Next, $\boldsymbol{p_0}[1]$ through $\boldsymbol{p_{49}}[1]$ form a Cosine function.

![](/images/positional-encoding/pe-08.png)

Next, $\boldsymbol{p_0}[10]$ through $\boldsymbol{p_{49}}[10]$ also form a Sine function, but with a larger period.

![](/images/positional-encoding/pe-09.png)

Plotting all Position Embeddings would look like this:

![](/images/positional-encoding/pe-10.png)

#### Clock Hand Analogy: Second Hand, Minute Hand, Hour Hand

We can view Sinusoidal Positional Embedding differently: even dimensions use Sine, odd dimensions use Cosine. So we can imagine each pair $2i$ and $2i+1$ as a **2D vector** (a clock hand on a 2D plane). This hand rotates as $k$ increases — when $k$ rises, the hand rotates counterclockwise.

![](/images/positional-encoding/pe-11.png)

The rotation period is:

$$
\frac{k}{10000^{2i/d}} = 2\pi
$$

The number of $k$ needed for one full rotation:

$$
k = 2\pi \cdot 10000^{2i/d}
$$

If $d = 128$, then $i = 0, ..., \frac{d}{2} - 1 = 0, ..., 63$.

Different $i$ values have different periods:

| $i$ value                      | $k$ value |
| ------------------------------ | --------- |
| $i = 0$ (dimensions 0, 1)      | 6.3       |
| $i = 32$ (dimensions 64, 65)   | 628.3     |
| $i = 63$ (dimensions 126, 127) | 54410.1   |

The first two dimensions rotate fastest — about 6.3 tokens per cycle. Dimensions 10-11 rotate more slowly. Dimensions 100-101 barely change within the first 6 tokens.

Every two dimensions form a hand. The fastest is the **second hand**, slower is the **minute hand**, slowest is the **hour hand**. With dimension 128, we have **64 hands**. The Transformer uses these 64 hands and their positions to determine each token's location.

![](/images/positional-encoding/pe-12.png)

Using clock hands to represent position seems reasonable, but why did the authors choose this approach? Because they wanted Positional Embedding to handle **Relative Position**.

---

### Relative Positioning is Crucial!

Relative Position means: in the sentence "the cat ate the fish," cat and fish are two tokens apart. When Transformer considers which token should follow fish, it looks back at cat. Fish's attention score for cat is 0.7.

![](/images/positional-encoding/pe-13.png)

If I insert many tokens before cat — say 1000 tokens — but **the relative position between cat and fish remains unchanged**, we want fish's attention on cat to remain unchanged. Even if fish is at position 1004 and cat at 1001, fish's attention score for cat remains 0.7.

![](/images/positional-encoding/pe-14.png)

But if cat is at position 1 and fish at position 1004 — very far apart — we want fish's attention score for cat to be small.

![](/images/positional-encoding/pe-15.png)

Sinusoidal Positional Embedding was chosen because it has special properties that help with Relative Position.

$$
\boldsymbol{p_{k+r}} = M_r \boldsymbol{p_k}
$$

Where $\boldsymbol{p_k}$ represents position $k$ and $\boldsymbol{p_{k+r}}$ represents position $k+r$.

This formula is independent of $k$ — it only depends on the **relative position** between $k+r$ and $k$:

$$
\begin{align}
\boldsymbol{p_{4}} &= M_3 \boldsymbol{p_{1}} \\
\boldsymbol{p_{14}} &= M_3 \boldsymbol{p_{11}} \\
\boldsymbol{p_{104}} &= M_3 \boldsymbol{p_{101}}
\end{align}
$$

Let's examine how Sinusoidal Positional Embedding satisfies the Relative Position condition.

Position $k$ formula:

$$
\begin{align}
\boldsymbol{p_k}[2i] &= \sin\left(\frac{k}{10000^{2i/d}}\right) \\
\boldsymbol{p_k}[2i+1] &= \cos\left(\frac{k}{10000^{2i/d}}\right)
\end{align}
$$

Position $k+r$ formula:

$$
\begin{align}
\boldsymbol{p_{k+r}}[2i] &= \sin\left(\frac{k+r}{10000^{2i/d}}\right) \\
\boldsymbol{p_{k+r}}[2i+1] &= \cos\left(\frac{k+r}{10000^{2i/d}}\right)
\end{align}
$$

Using sine and cosine addition formulas:

$$
\begin{align}
\sin(a+b) &= \sin(a)\cos(b) + \cos(a)\sin(b) \\
\cos(a+b) &= \cos(a)\cos(b) - \sin(a)\sin(b)
\end{align}
$$

Expand $\boldsymbol{p_{k+r}}$:

$$
\begin{align}
\boldsymbol{p_{k+r}}[2i]
&= \sin\left(\frac{k+r}{10000^{2i/d}}\right) \\
&= \boldsymbol{p_k}[2i] \cos\left(\frac{r}{10000^{2i/d}}\right) + \boldsymbol{p_k}[2i+1] \sin\left(\frac{r}{10000^{2i/d}}\right)
\end{align}
$$

$$
\begin{align}
\boldsymbol{p_{k+r}}[2i+1] &= \cos\left(\frac{k+r}{10000^{2i/d}}\right) \\
&= \boldsymbol{p_k}[2i+1] \cos\left(\frac{r}{10000^{2i/d}}\right) - \boldsymbol{p_k}[2i] \sin\left(\frac{r}{10000^{2i/d}}\right)
\end{align}
$$

In matrix form:

$$
\begin{bmatrix}
\boldsymbol{p_{k+r}}[2i] \\
\boldsymbol{p_{k+r}}[2i + 1]
\end{bmatrix} =
\begin{bmatrix}
\cos\left(\frac{r}{10000^{2i/d}}\right) & \sin\left(\frac{r}{10000^{2i/d}}\right) \\
-\sin\left(\frac{r}{10000^{2i/d}}\right) & \cos\left(\frac{r}{10000^{2i/d}}\right)
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{p_{k}}[2i] \\
\boldsymbol{p_{k}}[2i + 1]
\end{bmatrix}
$$

The rotation matrix is denoted as $M_{r,i}$.

![](/images/positional-encoding/pe-16.png)

How does this design affect Self-Attention? Using $p_n$ and $p_m$ for positions, the attention score $a$ is:

![](/images/positional-encoding/pe-17.png)

$$
\begin{align}
a &= \boldsymbol{q_B} \cdot \boldsymbol{k_A} \\
&= (W_q (\boldsymbol{x_{B}} + \boldsymbol{p_m}))^T W_k (\boldsymbol{x_A} + \boldsymbol{p_n}) \\
&= \boldsymbol{x_{B}} W_q^T W_k \boldsymbol{x_A} + \boldsymbol{x_{B}}^T W_q^T W_k \boldsymbol{p_n} + \boldsymbol{p_m}^T W_q^T W_k \boldsymbol{x_A} + \boldsymbol{p_m}^T W_q^T W_k \boldsymbol{p_n}
\end{align}
$$

The term $\boldsymbol{p_m}^T W_q^T W_k \boldsymbol{p_n}$ shows **absolute position**. Converting to relative position:

$$
\begin{align}
\boldsymbol{p_m}^T W_q^T W_k \boldsymbol{p_n} &= (M_{m-n} \boldsymbol{p_n})^T W_q^T W_k \boldsymbol{p_n} \\
&= (\boldsymbol{p_n})^T M_{m-n} W_q^T W_k \boldsymbol{p_n}
\end{align}
$$

Where $M_{m-n}$ only depends on the relative position, but the overall formula still contains absolute position correlations.

---

## 03 Relative Positional Embedding

Since using Absolute Positional Embedding to achieve relative positioning is circuitous, a natural question arises: **can we directly modify the Attention computation based on relative position?**

### ALiBi (Attention with Linear Biases)

Reference: [https://arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409)

ALiBi's approach: **discard Positional Embedding entirely**. Compute Attention scores without positional information, then **subtract $b(m-n)$** — the relative distance between positions $m$ and $n$. This makes Attention smaller for distant tokens. The constant $b$ is manually set and can differ per Attention Head.

![](/images/positional-encoding/pe-18.png)

ALiBi outperforms Sinusoidal methods across the board, especially on longer sequences.

![](/images/positional-encoding/pe-19.png)

### T5's Learnable Relative Bias

Reference: [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)

T5 uses a **learnable parameter** for the bias $b_{m-n}$ instead of manually setting it. However, the hand-crafted ALiBi actually outperformed the trained T5 approach.

![](/images/positional-encoding/pe-20.png)

ALiBi eventually gave way to stronger methods, but its lesson remains: **Relative Position matters greatly**.

### RoPE: Rotary Position Embedding

Reference: [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)

RoPE is one of today's most popular methods. Instead of adding a positional embedding, it **applies a rotation to $k$ and $q$ vectors** to encode positional information.

![](/images/positional-encoding/pe-21.png)

The Attention weight is then:

$$
a = (\boldsymbol{k^n_A})^T \boldsymbol{q^m_B} = \boldsymbol{(k_A)}^T R_{m-n} \boldsymbol{q_B}
$$

$R_{m-n}$ only depends on the **relative position** between A and B, directly encoding relative position into the attention computation.

Key advantage: **It doesn't affect the Attention computation process itself** — only $q$ and $k$ are modified, and the transformed $k$ can be stored in KV Cache.

#### Rotation as Position

Consider only the **first two dimensions** of $k$ and $q$:

![](/images/positional-encoding/pe-22.png)

These two dimensions form a vector on a 2D plane.

![](/images/positional-encoding/pe-23.png)

Adding positional information means **rotating** this vector. To encode position $n$, rotate the first two dimensions of $k$ by $n\theta$:

![](/images/positional-encoding/pe-24.png)
![](/images/positional-encoding/pe-25.png)

Similarly for $q$, rotate by $m\theta$:

![](/images/positional-encoding/pe-26.png)

RoPE applies this rotation to **every pair of dimensions**:

![](/images/positional-encoding/pe-27.png)

Each pair of dimensions uses a different rotation angle $n\theta_i$, where:

$$
\theta_i = \frac{1}{10000^{2i/d}}
$$

with $i = 0, 1, ..., \frac{d}{2} - 1$.

![](/images/positional-encoding/pe-28.png)

#### RoPE's Key Property: Relative Position Invariance

If we move $k$ from position $n$ to $n+r$ and $q$ from $m$ to $m+r$, their **relative position stays the same**, and the Attention value remains unchanged:

$$
\boldsymbol{k^n_A} \cdot \boldsymbol{q^m_B} = \boldsymbol{k^{n+r}_A} \cdot \boldsymbol{q^{m+r}_B}
$$

Suppose cat is at position 1 and fish at position 3. We add position 1 to cat's $k$ and position 3 to fish's $k$. Fish gets its Query, adds position to get $\boldsymbol{q^3_B}$, then computes attention $a$ with cat's $k$.

![](/images/positional-encoding/pe-29.png)

Adding many tokens before cat and fish doesn't affect their Embeddings. If cat is at 101 and fish at 103, the attention value is identical to when cat was at 1 and fish at 3.

![](/images/positional-encoding/pe-30.png)

Why does RoPE achieve this? Comparing original $k$ and $k$ with position $n$:

![](/images/positional-encoding/pe-31.png)

If $k$ is moved to position $n+r$, it is rotated by $(n+r)\theta$:

![](/images/positional-encoding/pe-32.png)

The angle between $\begin{bmatrix} \boldsymbol{k^n_A}[0] \\ \boldsymbol{k^n_A}[1] \end{bmatrix}$ and $\begin{bmatrix} \boldsymbol{k^{n+r}_A}[0] \\ \boldsymbol{k^{n+r}_A}[1] \end{bmatrix}$ is $r\theta$. Same for $\boldsymbol{q_B}$:

![](/images/positional-encoding/pe-33.png)

The dot product of $\begin{bmatrix} \boldsymbol{k^n_A}[0] \\ \boldsymbol{k^n_A}[1] \end{bmatrix}$ and $\begin{bmatrix} \boldsymbol{q^m_B}[0] \\ \boldsymbol{q^m_B}[1] \end{bmatrix}$ equals that of $\begin{bmatrix} \boldsymbol{k^{n+r}_A}[0] \\ \boldsymbol{k^{n+r}_A}[1] \end{bmatrix}$ and $\begin{bmatrix} \boldsymbol{q^{m+r}_B}[0] \\ \boldsymbol{q^{m+r}_B}[1] \end{bmatrix}$, because equal rotation doesn't change the inner product.

#### RoPE Mathematical Derivation

Let's examine why RoPE only depends on the relative position of $k$ and $q$, using the first two dimensions.

Rotating $k$ by $n\theta$:

$$
\begin{bmatrix}
\boldsymbol{k^n_A}[0] \\
\boldsymbol{k^n_A}[1]
\end{bmatrix}=
\begin{bmatrix}
\cos(n\theta) & -\sin(n\theta) \\
\sin(n\theta) & \cos(n\theta)
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{k_A}[0] \\
\boldsymbol{k_A}[1]
\end{bmatrix}
$$

![](/images/positional-encoding/pe-34.png)

Rotating $q$ by $m\theta$:

$$
\begin{bmatrix}
\boldsymbol{q^m_B}[0] \\
\boldsymbol{q^m_B}[1]
\end{bmatrix}=
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{q_B}[0] \\
\boldsymbol{q_B}[1]
\end{bmatrix}
$$

![](/images/positional-encoding/pe-35.png)

Computing the dot product:

$$
\begin{align}
\begin{bmatrix}
\boldsymbol{k^n_A}[0] \\
\boldsymbol{k^n_A}[1]
\end{bmatrix}
\cdot
\begin{bmatrix}
\boldsymbol{q^m_B}[0] \\
\boldsymbol{q^m_B}[1]
\end{bmatrix}
&=
\begin{bmatrix}
\boldsymbol{k_A}[0] \\
\boldsymbol{k_A}[1]
\end{bmatrix}^{\!T}
\begin{bmatrix}
\cos(n\theta) & -\sin(n\theta) \\
\sin(n\theta) & \cos(n\theta)
\end{bmatrix}^{\!T}
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{q_B}[0] \\
\boldsymbol{q_B}[1]
\end{bmatrix} \\[1em]
&=
\begin{bmatrix}
\boldsymbol{k_A}[0] \\
\boldsymbol{k_A}[1]
\end{bmatrix}^{\!T}
\begin{bmatrix}
\cos((m-n)\theta) & -\sin((m-n)\theta) \\
\sin((m-n)\theta) & \cos((m-n)\theta)
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{q_B}[0] \\
\boldsymbol{q_B}[1]
\end{bmatrix}
\end{align}
$$

The rotation matrices combine: $-n\theta$ (transpose) $+ m\theta = (m-n)\theta$.

Replacing $\boldsymbol{k^n_A}$ with $\boldsymbol{k^{n+r}_A}$ and $\boldsymbol{q^m_B}$ with $\boldsymbol{q^{m+r}_B}$:

$$
\begin{align}
\begin{bmatrix}
\boldsymbol{k^{n+r}_A}[0] \\
\boldsymbol{k^{n+r}_A}[1]
\end{bmatrix}
\cdot
\begin{bmatrix}
\boldsymbol{q^{m+r}_B}[0] \\
\boldsymbol{q^{m+r}_B}[1]
\end{bmatrix}
&=
\begin{bmatrix}
\boldsymbol{k_A}[0] \\
\boldsymbol{k_A}[1]
\end{bmatrix}^{\!T}
\begin{bmatrix}
\cos(((m+r)-(n+r))\theta) & -\sin(((m+r)-(n+r))\theta) \\
\sin(((m+r)-(n+r))\theta) & \cos(((m+r)-(n+r))\theta)
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{q_B}[0] \\
\boldsymbol{q_B}[1]
\end{bmatrix}
\end{align}
$$

Since $((m+r)-(n+r))\theta = (m-n)\theta$, the result is identical — confirming relative position invariance.

#### RoPE Implementation Details

RoPE rotates $k$ and $q$ separately before computing Attention:

![](/images/positional-encoding/pe-36.png)

This is equivalent to multiplying $q$ by $R_{m-n}$, computing only $q$'s rotation:

![](/images/positional-encoding/pe-37.png)

In practice, the **first approach** is preferred. With the first method, both position-encoded $k$ and $q$ can be stored in KV Cache. With the second, $q$ would need to multiply by different $R_{m-n}$ each time. RoPE's compatibility with KV Cache is key to its popularity.

#### RoPE vs ALiBi: An Important Distinction

Many misunderstand RoPE — thinking like ALiBi, farther $k$ and $q$ have smaller Attention weights. But **RoPE does not guarantee this**.

This isn't necessarily a weakness. RoPE can do something ALiBi cannot: **skip intermediate tokens to attend directly to earlier ones**. In ALiBi, farther tokens always have smaller attention. With RoPE, rotation angles can bring certain positions into alignment regardless of distance.

![](/images/positional-encoding/pe-38.png)

For example, "my cat" and "his dog" — "cat" should attend to "my" and "dog" to "his," not to the middle word "the." RoPE enables this nuanced attention.

If $q$ and $k$ differ by $2\theta$, but $q$ at $n+1$ and $k$ at $n$ differ by just $\theta$, and $q$ at $n+2$ and $k$ at $n$ might have the same angle — attention scores can actually increase despite greater distance, allowing the model to skip intermediate tokens.

![](/images/positional-encoding/pe-39.png)

---

## 04 Train Short, Test Long

Reference: [https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html](https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html)

Can we train on short sequences but **handle very long sequences during testing without failure**?

![](/images/positional-encoding/pe-40.png)

This is crucial for Agents that need to run indefinitely with very large context windows.

During training, we may not have very long corpora. The goal is positional encoding that supports Train Short, Test Long.

Intuitively, during training we only see $N$ tokens with positions $1, 2, ..., N$.

![](/images/positional-encoding/pe-41.png)

![](/images/positional-encoding/pe-42.png)

When testing with $LN$ tokens, can we just assign positions up to $LN$?

![](/images/positional-encoding/pe-43.png)

Can RoPE handle this? It seems not.

(From: [https://arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409))

![](/images/positional-encoding/pe-44.png)

When testing with very long sequences:

- **Sinusoidal**: fails immediately on longer sequences.
- **RoPE**: slightly better than Sinusoidal, but also degrades with length.
- **ALiBi**: the only one that holds up, because it uses hand-crafted rather than learned parameters — proving surprisingly robust.

### Why Does RoPE Fail on Long Sequences?

During training, the maximum rotation angle seen is $N\theta$. Testing at $2N\theta$ presents an angle the model has never seen, and RoPE "freaks out."

![](/images/positional-encoding/pe-45.png)

### Position Interpolation

References:

- [https://arxiv.org/pdf/2306.15595](https://arxiv.org/pdf/2306.15595)
- [https://kaiokendev.github.io/context#a-bigger-problem](https://kaiokendev.github.io/context#a-bigger-problem)

Solution: **don't assign rotation angles beyond $N$**.

With $LN$ tokens, position numbers don't have to be $1$ to $LN$ — they can be $\frac{1}{L}$ to $N$:

![](/images/positional-encoding/pe-46.png)

![](/images/positional-encoding/pe-47.png)

This is **Position Interpolation**, though it still requires fine-tuning.

### Frequency-Based Approach

RoPE processes dimensions in pairs, and Position Interpolation treats all dimensions identically. Frequency-Based Approach asks: **can different dimensions be treated differently?**

When extending from $N$ to $LN$, can we multiply by a function that depends on both the extension factor and the dimension?

![](/images/positional-encoding/pe-48.png)

Principle: $\theta_0$ is high-frequency, larger $i$ means lower frequency. High-frequency vectors can stay unchanged; low-frequency vectors need compression.

![](/images/positional-encoding/pe-49.png)

Every two dimensions form a rotating pointer. For the first two dimensions, the pointer rotates very fast. With $N=128$, $\theta_0$ may have seen the entire plane many times over, while $\theta_{32}$ has barely covered 1/4.

![](/images/positional-encoding/pe-50.png)

For $\theta_0$, positions beyond $N$ are fine — it's seen it all. But for $\theta_{32}$, extending $N$ to 256 means encountering an unseen rotation angle.

![](/images/positional-encoding/pe-51.png)

#### NTK-aware Scaling

$$
f(L, i) = \left(\frac{1}{L}\right)^{\frac{2i}{d-2}}
$$

Meaning:

- $i=0$: $f(L,0)=1$, no compression at highest frequency.
- $i=\frac{d}{2}-1$: $f(L, \frac{d}{2}-1)=\frac{1}{L}$, maximum compression at lowest frequency.

Reference: [https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)

![](/images/positional-encoding/pe-52.png)

### YaRN

**YaRN** (Yet another RoPE extensionN method). Reference: [https://arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071)

Keep some low-frequency scaling unchanged, keep some high-frequency scaling unchanged, **only modify the middle frequencies**.

![](/images/positional-encoding/pe-53.png)

### Dynamic Scaling

Previous methods handle long sequences, but performance on short sequences may degrade.

Reference: [https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/)

![](/images/positional-encoding/pe-54.png)

Dynamic Scaling: use different treatments for different sequence lengths. For short sequences, don't modify — performance stays good since training saw these lengths.

![](/images/positional-encoding/pe-55.png)

One variant: don't modify positions at the beginning of the sequence, only apply scaling after a certain proportion.

![](/images/positional-encoding/pe-56.png)

Dynamic Scaling combined with Position Interpolation or NTK outperforms both alone.

![](/images/positional-encoding/pe-57.png)

### Frequency-Based + Dynamic: LongRoPE

Frequency-Based determines the compression function; Dynamic Scaling determines where compression begins. **LongRoPE** uses **Evolutionary Search** to find optimal strategies.

Reference: [https://arxiv.org/abs/2402.13753](https://arxiv.org/abs/2402.13753)

![](/images/positional-encoding/pe-58.png)

This achieves remarkable results — models handling up to **2048K** input length.

![](/images/positional-encoding/pe-59.png)

---

## 05 No Positional Embedding?!

### Does Self-Attention Really Have No Positional Information?

Single-layer Self-Attention has no positional information. But **multi-layer** Self-Attention is different.

The first layer captures relationships between tokens. The second layer sees these relationships embedded — "cat relates to 'ate'" and "fish relates to 'ate'" produce different representations. So "cat ate fish" and "fish ate cat" yield different final Embeddings **even without Positional Embedding**.

![](/images/positional-encoding/pe-60.png)

### NoPE: No Positional Embedding

Reference: [https://arxiv.org/pdf/2305.19466](https://arxiv.org/pdf/2305.19466)

Experiments show that **No Position Embedding works surprisingly well**. On Copy tasks, models with Positional Embedding degraded on long sequences, while NoPE maintained high accuracy.

![](/images/positional-encoding/pe-61.png)

However, comparing RoPE vs NoPE during training — NoPE underperforms RoPE.

![](/images/positional-encoding/pe-62.png)

We need Positional Embedding because it helps **during training**.

### DroPE: Dropping Positional Embedding Late in Training

Start training with Positional Embedding, then remove it near the end. Loss spikes briefly but recovers quickly. Training without Position Embedding can match RoPE's loss.

DroPE **outperforms RoPE + YaRN** on extended contexts. Training at 2K context, then removing positional embedding enables good performance at inference contexts far beyond 2K.

![](/images/positional-encoding/pe-63.png)

Reference: [https://arxiv.org/pdf/2512.12167](https://arxiv.org/pdf/2512.12167)

---

## References

1. [**Sinusoidal Positional Embedding (Original Transformer)**](https://arxiv.org/abs/1706.03762)
2. [**ALiBi (Attention with Linear Biases)**](https://arxiv.org/abs/2108.12409)
3. [**T5 (Text-to-Text Transfer Transformer)**](https://arxiv.org/abs/1910.10683)
4. [**RoPE (Rotary Position Embedding)**](https://arxiv.org/abs/2104.09864)
5. [**Train Short Test Long (RoPE Context Extension Survey)**](https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html)
6. [**Position Interpolation**](https://arxiv.org/pdf/2306.15595)
7. [**Position Interpolation (Kaiokendev blog)**](https://kaiokendev.github.io/context#a-bigger-problem)
8. [**NTK-aware Scaling**](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
9. [**YaRN**](https://arxiv.org/abs/2309.00071)
10. [**Dynamic Scaling**](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/)
11. [**LongRoPE**](https://arxiv.org/abs/2402.13753)
12. [**NoPE**](https://arxiv.org/pdf/2305.19466)
13. [**DroPE**](https://arxiv.org/pdf/2512.12167)
