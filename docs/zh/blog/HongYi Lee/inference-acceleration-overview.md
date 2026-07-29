---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - 机器学习
tags:
  - llm
  - inference
  - acceleration
  - 李宏毅
title: "从 Flash Attention 到 Speculative Decoding：大模型推理加速最全解读"
createTime: 2026/07/01 18:16:40
permalink: /zh/article/inference-acceleration/
---

> 用过 ChatGPT 的人都知道，模型生成回答需要等几秒甚至几十秒。这背后的瓶颈在哪？有没有办法让大模型"说快一点"？本文从底层原理出发，拆解 6 大类加速方案，帮你建立起完整的推理加速知识框架。

---

## 01 一个核心问题：大模型生成为什么慢？

要理解加速方法，先得理解**大模型到底是怎么生成内容的**。

大模型生成文本的过程，本质上是一个**逐字接龙**游戏：

- 模型先看到你输入的 Prompt（这叫 **Prefill** 阶段，一次处理所有输入 Token）
- 然后一个词一个词地往外蹦（这叫 **Decode** 阶段，每次生成一个 Token）

![](/images/inference-acceleration/prefill-decode.png)

那每一步具体怎么算的呢？这就要说到 Transformer 最核心的模块——**Self-Attention**。

简单来说，每个 Token 都会生成三个向量：**Query（查询）、Key（键）、Value（值）**。生成下一个 Token 时，它的 Query 会跟前面所有 Token 的 Key 算"相似度"（dot product），然后用这个相似度去加权汇总所有的 Value。

![](/images/inference-acceleration/self-attention-qkv.png)

你看，问题就出在这里：

> **每生成一个 Token，都要跟前面所有的 Token 做一次注意力计算。**
>
> Token 越多，计算量就越大，等待时间就越长。

那怎么加速呢？学术界和工业界想出了**一系列方法**，而且它们可以组合使用。下面我们逐一拆解。

---

## 02 Flash Attention：不改变算法，只改变数据搬运方式

论文地址：https://arxiv.org/abs/2205.14135

### 一句话理解

Flash Attention 没有改变 Attention 的计算公式，它只是**改变了计算的顺序**，让数据少搬几次家，从而大幅提速。

### 先搞懂 GPU 的工作方式

要理解 Flash Attention 为什么快，得先知道 GPU 是怎么干活的：

- GPU 有很多 **Execution Unit（执行单元）**，它们负责算数
- 每个执行单元有一个很小的"工作台"——**SRAM**，速度极快但容量极小
- 大量的数据存在"仓库"里——**HBM（高带宽内存）**，容量大但读写慢

![](/images/inference-acceleration/gpu-sram-hbm.png)

所以每次运算时，执行单元得先把数据从 HBM 搬到 SRAM，算完再搬回去。

> **Attention 计算的瓶颈，不是算得慢，是搬数据搬得慢。**

传统 Attention 计算分好几步，每步都要读写 HBM，来回搬很多次：

1. 搬 Q、K → 算 dot product → 搬结果回 HBM
2. 找最大值 → 搬回 HBM
3. 算分母（Softmax 求和）→ 搬回 HBM
4. 算最终 attention weight → 搬回 HBM
5. 搬 V 进来做 weighted sum → 搬结果回 HBM

整个过程，HBM 被反复读写。

### Flash Attention 的核心思路

Flash Attention 把上面这些步骤合并了——**不需要先算出完整的 attention weight，再去做 weighted sum**。它在 SRAM 上边算边合并，一步到位。

具体来说，它把 K 和 V 切成多个 Chunk，每次只处理一个 Chunk，关键技巧是：

> 处理下一个 Chunk 时，如果发现前面 Chunk 用的最大值小了，可以**用一个修正项调整**前面已经算好的结果，不需要重算。

![](/images/inference-acceleration/flash-attention-chunking.png)

用公式来表达就是：

当处理第 2 个 Chunk 时，之前算的 $o_1$ 需要调整：

$$
o_2 = o_1 \frac{s_1}{s_2}(e^{d_1 - d_2}) + \sum^{2N}_{i=N+1}{\frac{e^{a_i - d_2}}{s_2}v_i}
$$

这个公式看起来很复杂，但核心思想很简单：**前面算的部分，乘以一个"修正因子"即可，不需要重算**。

### 效果与局限

- ✅ **不改变结果**：和标准 Attention 数学上完全等价
- ✅ **即插即用**：可以直接用在任何用了 Attention 的模型上
- ✅ **显著加速**：Sequence 越长，效果越明显
- ⚠️ 如果 Sequence 太短，加速效果不明显（前面处理本身时间就短）

---

## 03 KV Cache：用存储换速度

### 核心思想

在 Decode 阶段，每生成一个 Token，模型都要计算它跟前面所有 Token 的 Attention。但前面的 Token 不会变，所以它们的 K 和 V 也不需要重新算。

> **KV Cache 就是把前面 Token 的 K 和 V 存起来，每次只算新 Token 的 QKV，避免重复计算。**

![](/images/inference-acceleration/kv-cache-diagram.png)

这个思路简单直接，而且完全不改变 Attention 的计算逻辑。

### 代价：内存爆炸

KV Cache 的问题也很明显——**它太吃内存了**。

每次输入一个新 Token，就要多存一组 K 和 V。而且这还不是一组——Transformer 有**多层 × 多头的 K 和 V**。

拿 Gemma 2 来算笔账：

![](/images/inference-acceleration/gemma2-kv-cache.png)

$$
\text{每个Token: } 46(\text{层}) \times 32 (\text{头}) \times 128 (\text{维度}) \times 2 (\text{FP16}) \times 2(\text{V, K}) = 753664 \text{ 字节} (\text{约736KB})
$$

注意这是**一个 Token** 需要的空间。如果 Sequence 长度是 114k，那 A100 的 80GB 显存就刚好被填满。

> **KV Cache 让模型变快，但它会吃掉大量显存，限制了能处理的上下文长度。**

这是 KV Cache 加速方案的核心矛盾。

---

## 04 减少 KV 存储：三招让 Cache 更省空间

既然 KV Cache 太占地方，能不能让 K 和 V 少存一点？有三个经典方案：

### 方案一：Multi-Query Attention（MQA）

多头注意力（MHA）里，每个头都有自己的 K 和 V。MQA 的想法是：

> **多个 Query 头共享一组 K 和 V。**

![](/images/inference-acceleration/mqa-diagram.png)

好处是 KV Cache 大幅减少，但问题也很明显：**共享一组 K/V 太粗暴了，模型表现会下降**。

### 方案二：Grouped-Query Attention（GQA）

GQA 是一个折中方案：**把 Query 头分成几组，每组共享一组 K 和 V**。

![](/images/inference-acceleration/gqa-diagram.png)

它介于 MHA 和 MQA 之间，在**效率和效果之间取了一个平衡**。现在很多新模型（如 Llama 2/3）都用的 GQA。

### 方案三：Multi-head Latent Attention（MLA）

论文地址：https://arxiv.org/abs/2405.04434 （DeepSeek）

MLA 的想法更巧妙：**不直接存 K 和 V，而是先把它们压缩成一个低维向量再存**，用的时候也不一定要解压缩。

![](/images/inference-acceleration/mla-compression.png)

这样做有两个关键技巧：

**技巧 1：在压缩空间里做 dot product**

把输入 X 压缩成向量 $c$，存进仓库的就是这个 $c$。需要算 Attention 时：

$$
a = q \cdot k = q^T k = q^T W_k c = (W_k^T q)^T c = (W_k^T q) \cdot c
$$

你看，**不需要先把 $c$ 解压成 $k$**，直接把 $q$ 转一下就能在压缩空间里做 dot product。

![](/images/inference-acceleration/mla-dot-product.png)

**技巧 2：在压缩空间里做 Weighted Sum**

算完 attention weight 后，要做 weighted sum 得到输出：

$$
\begin{align*}
o &= \hat{a}_1 v_1 + \hat{a}_2 v_2 + \hat{a}_3 v_3 + \hat{a}_4 v_4 \\
  &= \hat{a}_1 W_v c_1 + \hat{a}_2 W_v c_2 + \hat{a}_3 W_v c_3 + \hat{a}_4 W_v c_4 \\
  &= W_v (\hat{a}_1 c_1 + \hat{a}_2 c_2 + \hat{a}_3 c_3 + \hat{a}_4 c_4)
\end{align*}
$$

核心洞察：**先在压缩的 $c$ 上做 weighted sum，最后只解压缩一次**。这大大减少了计算量。

![](/images/inference-acceleration/mla-weighted-sum.png)

> **MLA 是一个需要重新训练模型的方法**，但它被 DeepSeek 等前沿模型采用，证明了这条路是可行的。

---

## 05 Sliding Window Attention + Streaming LLM：只看附近的内容

### Sliding Window Attention

核心思路很简单：**每次做 Attention 时，不需要看整个 Sequence，只看最近的 N 个 Token**。

![](/images/inference-acceleration/sliding-window-attention.png)

但这样模型不就看不到长距离信息了吗？有个巧妙的观察：

> **Transformer 层数越深，Sliding Window Attention 能看到的范围实际上越大。**

因为第 1 层只看附近几个 Token，第 2 层的输入就已经包含了第 1 层窗口里的信息，相当于变相扩大了感受野。网络足够深的话，即使每个窗口不大，也能覆盖很长的范围。

### 混合策略

还有一种方案：**有些层用 Sliding Window，有些层用全局 Attention**。

![](/images/inference-acceleration/hybrid-attention.png)

这样既能节省 KV Cache，又能在关键层保持全局视野。

### Streaming LLM

论文地址：https://arxiv.org/abs/2309.17453

这里有个有趣的发现：**只用 Sliding Window 效果会变差，但如果保留最开始的几个 Token 就好了**。

![](/images/inference-acceleration/streaming-llm-attention.png)

而且这招**不需要重新训练模型**，直接改推理代码就行。

![](/images/inference-acceleration/streaming-llm-results.png)

实验结果显示，Streaming LLM 在长 Sequence 上的表现明显优于纯 Window Attention。

---

## 06 Pruning KV Cache：丢掉没用的 K 和 V

更直接的方法来了：**如果有些 K 和 V 根本用不上，直接丢掉不就好了？**

研究发现，Attention 其实**非常稀疏**——大部分 Token 的 attention weight 非常小，几乎等于没用上。

![](/images/inference-acceleration/attention-sparsity.png)

颜色越深表示 attention weight 越大。可以看到，**只有很少的 Token 被真正 attention 到了**。

基于这个观察，两篇论文提出了不同的裁剪策略：

- **Scissorhands**（https://arxiv.org/abs/2305.17118）
- **H2O**（https://arxiv.org/abs/2306.14048）

核心思路一致：**如果一个 K/V 长时间没被 Attention 用到，就把它从 Cache 里清除**。

![](/images/inference-acceleration/kv-pruning.png)

Scissorhands 的实验显示，**压缩 5 倍的情况下，模型表现跟不压缩基本一样**。

但 ⚠️ 后续研究也发现：**如果让模型做非常难的任务，随意丢弃 K/V 可能会导致表现大幅下降**。这个方法适合常规任务，关键场景要谨慎。

---

## 07 跨对话 Cache：Agent 场景的大杀器

前面的 KV Cache 都是**同一个对话里**的优化。但 KV Cache 还有一个更高级的玩法——**跨对话共享**。

不同对话里如果出现相同的文本片段，它们的 K 和 V 理论上是可以复用的。

![](/images/inference-acceleration/cross-conversation-cache.png)

### 什么场景最受益？

**AI Agent 场景**是跨对话 Cache 的最佳舞台：

每个 Agent 调用都带着一串 System Prompt（角色设定、工具定义、记忆指令等），这些内容不同对话间高度一致。

![](/images/inference-acceleration/agent-cache-scenario.png)

### 使用技巧

要让 Cache Hit 率最大化，**内容的排列顺序有讲究**：

> **越稳定不动的内容放越前面，越可能变动的内容放越后面。**

![](/images/inference-acceleration/cache-order.png)

另外，**同一个 Prompt 用不同写法，Cache Hit 率可以差很多**：

![](/images/inference-acceleration/prompt-cache-hit.png)

换一种写法后，Cache Hit 明显提高，这意味着**直接省钱**。

![](/images/inference-acceleration/prompt-writing-cost.png)

有一篇论文专门测了这个效果：https://arxiv.org/abs/2601.06007

结论是：**用好的 Prompt 写法，结合 Cached Input，Agent 的花费可以大幅降低**。

---

## 08 总结：一张表看清所有加速方案

| 方法                            | 说明                              | 改变 Attention？ | 需要训练？ | 主要代价             |
| ------------------------------- | --------------------------------- | :--------------: | :--------: | -------------------- |
| **Flash Attention**             | 减少 HBM 读写次数，优化计算顺序   |        ✗         |     ✗      | 一点额外运算         |
| **KV Cache**                    | 存储已算好的 K 和 V，避免重复计算 |        ✗         |     ✗      | 占用大量显存         |
| **Multi-Query Attention**       | 多个 Query 头共享一组 K/V         |        ✓         |     ✓      | 可能明显伤害模型能力 |
| **Grouped-Query Attention**     | Query 分组共享 K/V                |        ✓         |     ✓      | 效果-效率平衡        |
| **Multi-head Latent Attention** | 压缩 K/V 后再存储                 |        ✓         |     ✓      | 需要重新训练         |
| **Sliding Window Attention**    | 只 Attention 附近 Token           |        ✓         |     ?      | 可能丢失长距离信息   |
| **Streaming LLM**               | Sliding Window + 保留开头的 Token |        ✗         |     ✗      | —                    |
| **Pruning KV Cache**            | 丢弃不常用的 K 和 V               |        ✓         |     ✗      | 复杂任务效果下降     |
| **Speculative Decoding**        | 用小模型预估结果，大模型校验      |   ✗（理论上）    |     ✗      | 小模型额外算力       |

---

## 参考资料

1. **Flash Attention**：https://arxiv.org/abs/2205.14135
2. **Multi-head Latent Attention（DeepSeek）**：https://arxiv.org/abs/2405.04434
3. **Streaming LLM**：https://arxiv.org/abs/2309.17453
4. **Scissorhands**：https://arxiv.org/abs/2305.17118
5. **H2O（Heavy-Hitter Oracle）**：https://arxiv.org/abs/2306.14048
6. **Cached Input 对 Agent 花费的影响**：https://arxiv.org/abs/2601.06007
