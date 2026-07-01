---
url: /article/minimax-m3-deployment/index.md
---
The full problem is as follows:

MiniMax M3 is an MoE large language model with 428B total parameters, 23B activated parameters per token, 60 Transformer layers, native MSA sparse attention, native support for up to 1M context window, and KV Cache stored in BF16 precision by default. Inference reserves 10% of total VRAM for operator activation and framework system overhead.

Hardware conditions:

1. GPU: 8 × NVIDIA A6000 Pro, 96GB VRAM per card, deployed with TP8 tensor parallelism;
2. Network: Single node 400Gbps high-speed interconnect; VRAM is the performance bottleneck for this calculation, bandwidth does not participate in concurrency estimation.

Please answer the following questions:

1. If all 428B weights are quantized to INT8, how much VRAM does the model occupy? Can the 8-card A6000 Pro cluster fully load all weights?
2. For text-only inference with a 128K context window, calculate the approximate KV Cache VRAM per request under both standard dense attention and M3's native MSA sparse attention.
3. Based on the above hardware, 128K context, and MSA sparse attention enabled, what is the theoretical maximum number of concurrent requests this cluster can support?
4. If the context window expands from 128K to 1M, how does the per-request KV Cache usage and theoretical concurrency change? Calculate the concurrency reduction ratio and the corresponding estimated concurrency.

***

## Full Answer Key

## 1. Known Conditions and Core Parameters

### Given Conditions

1. Model: MiniMax M3, MoE mixture-of-experts architecture, 428B total parameters, 23B activated parameters per token, 60 Transformer layers, native MSA sparse attention, KV Cache stored in BF16 precision
2. Quantization: All weights quantized to INT8, 1 byte per parameter
3. Hardware: 8 × NVIDIA A6000 Pro, 96GB VRAM per card, TP8 tensor parallelism
4. System overhead: 10% of total VRAM reserved for system operations, operator activation, and framework scheduling
5. Context baseline: 128K default, targeting 1M expansion

### Supplementary Model Architecture Parameters (from official public configuration)

| Parameter                       | Value | Description                                     |
| ------------------------------- | ----- | ----------------------------------------------- |
| Hidden dimension (hidden\_size)  | 6144  | Transformer layer feature dimension             |
| Number of attention query heads | 64    | Total Query attention heads                     |
| Number of KV heads (GQA)        | 4     | Grouped query attention, only 4 Key/Value heads |
| Attention head dimension        | 96    | Derived from 6144 ÷ 64                          |
| BF16 bytes per element          | 2     | Default KV Cache storage precision              |
| INT8 bytes per element          | 1     | Single parameter size after quantization        |

### Core Formulas

1. Model weight VRAM = Total parameters × Bytes per parameter
2. KV Cache VRAM per request = 2 (K matrix + V matrix) × Number of layers × Number of KV heads × Head dimension × Sequence length × Bytes per element
3. Theoretical max concurrency = Available KV VRAM ÷ KV VRAM per request

***

## Question 1: INT8 Weight VRAM Usage and 8-Card Feasibility

### Calculation

1. **INT8 full weight VRAM usage**

During MoE inference, all expert weights must reside in VRAM, so we calculate using the full 428B parameters:

$$
\begin{align\*}
\text{Weight VRAM} &= 428 \times 10^9 \text{ parameters} \times 1 \text{ Byte/parameter} \\
&= 428 \text{ GB}
\end{align\*}
$$

(Note: Industry standard decimal estimation, 1GB = 10⁹ bytes, consistent with hardware VRAM specifications; the error is within acceptable engineering tolerance.)

2. **Total available VRAM across 8 cards**

96GB per card, total VRAM: $8 \times 96 = 768 \text{ GB}$

After deducting 10% system overhead, effective available VRAM:

$$
768 \times (1 - 10%) = 691.2 \text{ GB}
$$

### Conclusion

INT8 full weights occupy 428GB VRAM, which is less than the effective 691.2GB across 8 cards. **8 A6000 Pro cards can fully load the model**, with 263.2GB remaining for KV Cache storage.

### Notes

If we only considered the activated parameters per token (23B) in INT8, the usage would be only 23GB. However, MoE architecture requires loading all expert weights for sparse routing, so the full 428B parameter count must be used.

***

## Question 2: KV Cache VRAM per Request at 128K Context (Two Scenarios)

MSA sparse attention primarily optimizes computational complexity and inference latency for long contexts; it does not change the total KV Cache storage. The core difference between the two scenarios lies in whether GQA grouped query optimization is adopted.

### Scenario 1: Standard Dense Attention (Traditional MHA, No GQA)

Traditional dense Transformers use full multi-head attention, with the number of KV heads equal to the number of query heads (64), offering no storage optimization.

KV VRAM per token:

$$
\begin{align\*}
\text{KV per token} &= 2 \times 60 \text{ layers} \times 64 \text{ KV heads} \times 96 \text{ head dim} \times 2 \text{ bytes} \\
&= 1,474,560 \text{ bytes} \approx 1.44 \text{ MB}
\end{align\*}
$$

Total KV VRAM at 128K context:

$$
1.44 \text{ MB/token} \times 128 \times 10^3 \text{ token} \approx 184.3 \text{ GB}
$$

### Scenario 2: M3 Native Architecture (GQA + MSA Sparse Attention)

M3 uses a GQA grouped query architecture with only 4 KV heads, dramatically reducing KV Cache size. MSA handles computation acceleration while KV is still fully stored.

KV VRAM per token:

$$
\begin{align\*}
\text{KV per token} &= 2 \times 60 \text{ layers} \times 4 \text{ KV heads} \times 96 \text{ head dim} \times 2 \text{ bytes} \\
&= 92,160 \text{ bytes} \approx 90 \text{ KB}
\end{align\*}
$$

Total KV VRAM at 128K context:

$$
90 \text{ KB/token} \times 128 \times 10^3 \text{ token} \approx 11.5 \text{ GB}
$$

### Conclusion

* Standard dense MHA: ~**184GB** KV per request at 128K context — extreme VRAM pressure, a single card cannot serve a single long-context request
* M3 native GQA + MSA: ~**11.5GB** KV per request at 128K context — only 1/16 of the traditional architecture, the core foundation for long-context viability

***

## Question 3: Theoretical Maximum Concurrency at 128K Context

### Logic

After deducting model weight VRAM from total effective VRAM, the remaining space is entirely used for KV Cache storage. Dividing by per-request KV VRAM yields the theoretical maximum concurrency from the VRAM perspective.

### Calculation

1. Available VRAM for KV Cache = Effective total VRAM − Weight VRAM = $691.2 - 428 = 263.2 \text{ GB}$
2. KV VRAM per request (M3 native, 128K) = 11.5 GB
3. Theoretical max concurrency:

$$
\text{Concurrency} = \frac{263.2}{11.5} \approx 22.9
$$

### Conclusion

With MSA optimization enabled and 128K context, this cluster can theoretically support approximately **22~23 concurrent requests**.

### Notes

* This value is the VRAM-only limit, not accounting for computation latency, network overhead, scheduling losses, or traffic spikes. Production environments typically reserve 30%~50% safety margin, making actual usable concurrency around 11~15.
* With traditional dense MHA architecture, a single request's KV already exceeds single-card VRAM, making normal deployment impossible.

***

## Question 4: Concurrency Changes When Context Expands to 1M

### Core Principle

KV Cache VRAM is **strictly linearly correlated** with sequence length: if sequence length grows N times, per-request KV VRAM also grows N times. When total available KV VRAM is fixed, maximum concurrency shrinks to $1/N$ of the original.

### Calculation

1. Context length scaling factor: $\frac{1\text{M}}{128\text{K}} = 8\times$
2. KV VRAM per request at 1M context: $11.5 \times 8 = 92 \text{ GB}$
3. Theoretical max concurrency at 1M:

$$
\text{Concurrency} = \frac{263.2}{92} \approx 2.9
$$

4. Concurrency reduction ratio: approximately $\boldsymbol{\frac{1}{8}}$ of the 128K scenario, an 87.5% reduction

### Conclusion

* Per-request KV Cache VRAM increases by 8×, theoretical max concurrency drops to 1/8 of the original, approximately **2~3 concurrent requests**.
* In production, the Prefill phase at 1M context takes significantly longer, making computation bottlenecks more prominent — actual usable concurrency will be lower than the theoretical value. MSA sparse attention can improve decoding speed by 15× or more, ensuring long-context inference viability, but does not change the VRAM-level concurrency ceiling.
