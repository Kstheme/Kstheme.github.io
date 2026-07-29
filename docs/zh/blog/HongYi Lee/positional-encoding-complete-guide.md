---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - 机器学习
tags:
  - positional-encoding
  - transformer
  - llm
  - 李宏毅
title: "大模型位置编码最全拆解：绝对位置、相对位置、旋转位置，以及不用位置编码"
createTime: 2026/07/22 17:53:20
permalink: /zh/article/positional-encoding-guide/
copyright: Kstheme
---

> **摘要**：Transformer 的 Self-Attention 本身不感知 Token 顺序——把"猫吃鱼"和"鱼吃猫"的 Token 打乱，Attention 输出完全一样。本文系统梳理 Positional Embedding 的完整技术演化：从 Sinusoidal 绝对位置编码的指针类比，到 ALiBi、T5 的相对位置偏置，再到 RoPE 旋转位置编码的数学原理，最后探讨 Train Short Test Long 的一系列扩展方法以及"不用位置编码"的 NoPE/DroPE 思路。

> 用过 Transformer 的人都学过 Positional Embedding，但你真的理解它解决了什么问题、不同方案之间有什么本质区别吗？本文从最底层的问题出发，一步步拆解位置编码的完整技术树。

Copyright Ownership: Kstheme, Contributors: Kstheme

---

## 01 一个基础问题：Transformer 怎么知道 Token 的顺序？

原来的 Transformer 是没有办法考虑输入的顺序的。Transformer 是怎么处理输入 Token 的呢？

输入的 Token 会变成 Embedding，然后这些 Embedding 会变成某一个 Layer 的输入。每个 Layer 里面都会有 Self-Attention，Self-Attention 会输出另外一排 Token。

但是有一个问题：**Self-Attention 本身没有考虑 Token 输入的顺序。**

![](/images/positional-encoding/pe-01.png)

假设我把 Token A 跟 Token C 的位置对调，再去算 Self-Attention，你会发现对于 $O_D$ 的输出是没有任何影响的。

![](/images/positional-encoding/pe-02.png)

所以我们必须想办法把位置信息"塞进"Transformer 里，这就是 Positional Embedding 要解决的问题。

本文覆盖的内容如下：

1. **Absolute Positional Embedding**（以 Sinusoidal 为代表）
2. **Relative Positional Embedding**（ALiBi、T5）
3. **RoPE**（Rotary Position Embedding）
4. **Train Short, Test Long**（位置插值、频域方法、YaRN、Dynamic Scaling、LongRoPE）
5. **No Positional Embedding**（NoPE、DroPE）

---

## 02 Absolute Positional Embedding

最早的 Positional Embedding 想法很直接：**对每一个位置设置一个特别的 Embedding**。这个 Embedding 就代表了位置的信息。

我们用 $P_0$ 到 $P_3$ 代表每一个位置的特别 Embedding，然后把这个特别的 Embedding **加到**我们的向量上面去。

![](/images/positional-encoding/pe-03.png)

这样的话，当 Token 调换顺序之后，它整个的 Embedding 就不一样了，最后算出来的 Self-Attention 的输出也就不一样。

![](/images/positional-encoding/pe-04.png)

### Sinusoidal Positional Embedding

参考文献：[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

这个 Positional Embedding 是在 Transformer 诞生之初就存在的 Embedding。

我们用 $d$ 来代表 Embedding 的长度（$d = 128 \text{ or } 256$）。

![](/images/positional-encoding/pe-05.png)

我们可以用 $\boldsymbol{p}[i]$ 来代表它里面的数值。

![](/images/positional-encoding/pe-06.png)

那 Sinusoidal Position Embedding 是怎么构建所有的 Embedding 的？构建公式如下：

$$
\begin{align}
\boldsymbol{p_k}[2i] &= \sin\left(\frac{k}{10000^{2i/d}}\right) \\
\boldsymbol{p_k}[2i+1] &= \cos\left(\frac{k}{10000^{2i/d}}\right)
\end{align}
$$

#### 图像化理解

我们把 $\boldsymbol{p_0}[0]$ 到 $\boldsymbol{p_{49}}[0]$ 都拿出来，会发现这个图像是一个 Sin 函数。

![](/images/positional-encoding/pe-07.png)

接下来我们来看 $\boldsymbol{p_0}[1]$ 到 $\boldsymbol{p_{49}}[1]$，这个图像是一个 Cos 函数。

![](/images/positional-encoding/pe-08.png)

接下来我们来看 $\boldsymbol{p_0}[10]$ 到 $\boldsymbol{p_{49}}[10]$，它也是一个 Sin 函数，只不过它的周期比 $\boldsymbol{p_0}[0]$ 到 $\boldsymbol{p_{49}}[0]$ 要大。

![](/images/positional-encoding/pe-09.png)

那如果把全部的 Position Embedding 画出来的话，就会像下图这样：

![](/images/positional-encoding/pe-10.png)

#### 指针类比：秒针、分针、时针

我们现在用另外的方法来看 Sinusoidal Positional Embedding：它的偶数维度用 Sin 函数表示，奇数维度用 Cos 函数表示。所以我们可以把 $2i$ 和 $2i+1$ 的 Dimension 想象成一个**二维的向量**（二维平面上的一根指针）。这根指针会随着 $k$ 的数值来进行旋转——当 $k$ 上升时，指针开始往逆时针方向旋转。

![](/images/positional-encoding/pe-11.png)

旋转的周期如下：

$$
\frac{k}{10000^{2i/d}} = 2\pi
$$

那么指针走一圈大概需要多少 $k$ 呢？我们只需要把分母下面的数字移到右边，就可以知道 $k$ 等于多少：

$$
k = 2\pi \cdot 10000^{2i/d}
$$

假设 $d = 128$，那么 $i = 0, ..., \frac{d}{2} - 1 = 0, ..., 63$。

下面我们看一下 $i$ 的数值不同，它的周期会有怎样的变化：

| $i$ 取值                   | $k$ 值  |
| -------------------------- | ------- |
| $i = 0$（第 0, 1 维）      | 6.3     |
| $i = 32$（第 64, 65 维）   | 628.3   |
| $i = 63$（第 126, 127 维） | 54410.1 |

Self-Attention 看到的跟时间有关的信息是这样变化的：这一排向量的**最前面两位走的是最快的**，大概 6.3 个 Token 就会转一圈。中间 10 到 11 维就转得比较慢。中间 100 到 101 位，如果我们只看前 6 个 Token，那它几乎是没有变化的。

每两个 Dimension 合在一起就是一个指针。**转得最快的是秒针，转得稍微慢点的是分针，转得最慢的是时针**。接下来会有多少个指针呢？那就看你的 Dimension 有多少。比如我的 Dimension 是 128，那我就有 **64 个指针**。Transformer 就根据这 64 个指针、每一个指针的位置，来判断每个 Token 具体在哪个位置。

![](/images/positional-encoding/pe-12.png)

用指针表示位置听起来还是比较合理的，但是应该还有很多其他的方式来表示位置。那为什么当初会用秒针、时针、分针的概念来表示位置呢？这是因为 Transformer 的作者希望 Positional Embedding 可以考虑 **Relative Position（相对位置）**。

---

### Relative Positioning is Crucial!

Relative Position 就是说，假设有一句话："猫吃了鱼"，猫跟鱼之间间隔了两个 Token。Transformer 在考虑鱼后面要接哪个 Token 的时候，它会往前面考虑猫。鱼对猫的注意力分数是 0.7。

![](/images/positional-encoding/pe-13.png)

假设我在猫前面塞了大量的 Token，可能猫前面有 1000 个 Token，但是**猫跟鱼之间的相对位置没有变**，所以我们希望鱼对猫的注意力也不会有所谓改变。哪怕鱼是在 1004 的位置，猫在 1001 的位置，但鱼对猫的注意力分数还是 0.7。

![](/images/positional-encoding/pe-14.png)

但如果今天鱼跟猫的位置距离非常远，猫在第 1 个 Token，而鱼在第 1004 个 Token，因为距离非常远，所以我们希望鱼对猫的注意力分数要变得比较小。

![](/images/positional-encoding/pe-15.png)

当时选择了 Sinusoidal Positional Embedding，就是因为它有一些特殊的性质，而这些特殊的性质好像对 Relative Position 有帮助。

$$
\boldsymbol{p_{k+r}} = M_r \boldsymbol{p_k}
$$

其中，$\boldsymbol{p_k}$ 代表第 $k$ 个位置的 Position，$\boldsymbol{p_{k+r}}$ 代表 $k+r$ 位置的 Position。

而这个公式是跟 $k$ 没有关系的，只跟 $k+r$ 和 $k$ 之间的**相对位置**有关系。具体如下：

$$
\begin{align}
\boldsymbol{p_{4}} &= M_3 \boldsymbol{p_{1}} \\
\boldsymbol{p_{14}} &= M_3 \boldsymbol{p_{11}} \\
\boldsymbol{p_{104}} &= M_3 \boldsymbol{p_{101}}
\end{align}
$$

我们可以具体来看一下 Sinusoidal Positional Embedding 是怎样满足 Relative Position 的条件的。

首先位置 $k$ 的公式为：

$$
\begin{align}
\boldsymbol{p_k}[2i] &= \sin\left(\frac{k}{10000^{2i/d}}\right) \\
\boldsymbol{p_k}[2i+1] &= \cos\left(\frac{k}{10000^{2i/d}}\right)
\end{align}
$$

位置 $k+r$ 的公式为：

$$
\begin{align}
\boldsymbol{p_{k+r}}[2i] &= \sin\left(\frac{k+r}{10000^{2i/d}}\right) \\
\boldsymbol{p_{k+r}}[2i+1] &= \cos\left(\frac{k+r}{10000^{2i/d}}\right)
\end{align}
$$

我们接下来会用到 Sin、Cos 的和角公式：

$$
\begin{align}
\sin(a+b) &= \sin(a)\cos(b) + \cos(a)\sin(b) \\
\cos(a+b) &= \cos(a)\cos(b) - \sin(a)\sin(b)
\end{align}
$$

我们可以根据这个和角公式来展开 $\boldsymbol{p_{k+r}}$ 的两项：

$$
\begin{align}
\boldsymbol{p_{k+r}}[2i]
&= \sin\left(\frac{k+r}{10000^{2i/d}}\right) \\
&= \sin\left(\frac{k}{10000^{2i/d}}\right) \cos\left(\frac{r}{10000^{2i/d}}\right) + \cos\left(\frac{k}{10000^{2i/d}}\right) \sin\left(\frac{r}{10000^{2i/d}}\right)
\end{align}
$$

$$
\begin{align}
\boldsymbol{p_{k+r}}[2i+1] &= \cos\left(\frac{k+r}{10000^{2i/d}}\right) \\
&= \cos\left(\frac{k}{10000^{2i/d}}\right) \cos\left(\frac{r}{10000^{2i/d}}\right) - \sin\left(\frac{k}{10000^{2i/d}}\right) \sin\left(\frac{r}{10000^{2i/d}}\right)
\end{align}
$$

我们把 $\boldsymbol{p_k}[2i]$ 和 $\boldsymbol{p_k}[2i+1]$ 代入进去得：

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

我们现在把 $\boldsymbol{p_{k+r}}[2i]$ 和 $\boldsymbol{p_{k+r}}[2i+1]$ 写成一个矩阵形式：

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

中间这个旋转矩阵，我们用 $M_{r,i}$ 来进行表示。

![](/images/positional-encoding/pe-16.png)

那么 Sinusoidal Position Embedding 这种设计对后面去计算 Self Attention 的时候会有什么影响呢？

假设我们把 A 跟 B 的位置用 $p_n$ 和 $p_m$ 来表示，Attention 分数 $a$ 的计算公式如下：

![](/images/positional-encoding/pe-17.png)

$$
\begin{align}
a &= \boldsymbol{q_B} \cdot \boldsymbol{k_A} \\
&= (\boldsymbol{q_B})^T \boldsymbol{k_A} \\
&= (W_q (\boldsymbol{x_{B}} + \boldsymbol{p_m}))^T W_k (\boldsymbol{x_A} + \boldsymbol{p_n}) \\
&= (\boldsymbol{x_{B}} + \boldsymbol{p_m})^T W_q^T W_k (\boldsymbol{x_A} + \boldsymbol{p_n}) \\
&= \boldsymbol{x_{B}} W_q^T W_k \boldsymbol{x_A} + \boldsymbol{x_{B}}^T W_q^T W_k \boldsymbol{p_n} + \boldsymbol{p_m}^T W_q^T W_k \boldsymbol{x_A} + \boldsymbol{p_m}^T W_q^T W_k \boldsymbol{p_n}
\end{align}
$$

其中，$\boldsymbol{x_{B}} W_q^T W_k \boldsymbol{x_A}$ 只跟内容有关，$\boldsymbol{x_{B}}^T W_q^T W_k \boldsymbol{p_n} + \boldsymbol{p_m}^T W_q^T W_k \boldsymbol{x_A}$ 跟内容、位置的交互影响有关，$\boldsymbol{p_m}^T W_q^T W_k \boldsymbol{p_n}$ 只跟位置有关。

但是 $\boldsymbol{p_m}^T W_q^T W_k \boldsymbol{p_n}$ 显示的是**绝对位置**，所以我们要把它变成跟相对位置有关的公式：

$$
\begin{align}
\boldsymbol{p_m}^T W_q^T W_k \boldsymbol{p_n} &= (M_{m-n} \boldsymbol{p_n})^T W_q^T W_k \boldsymbol{p_n} \\
&= (\boldsymbol{p_n})^T M_{m-n} W_q^T W_k \boldsymbol{p_n}
\end{align}
$$

其中 $(\boldsymbol{p_n})^T M_{m-n} W_q^T W_k \boldsymbol{p_n}$ 里面的 $M_{m-n}$ 只跟相对位置有关，但是整体的公式还是会存在绝对位置之间的关联。

---

## 03 Relative Positional Embedding

既然用 Absolute Positional Embedding 来做相对位置这么迂回，一个自然的想法是：**能不能根据相对位置直接去改 Attention 计算？**

### ALiBi（Attention with Linear Biases）

参考文献：[https://arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409)

ALiBi 的做法是：**丢掉 Positional Embedding**。在没有 Positional Embedding 的情况下先把 Attention 分数算出来，然后**减掉 $b(m-n)$**。这个公式就是说减掉 $m$ 和 $n$ 两个位置之间的相对距离。减掉相对距离就会达成一个效果——距离越远 Attention 越小。而常数 $b$ 是你手动设置的数值，你可以给不同的 Attention Head 设置不同的数值。在这篇论文里，并没有设置很复杂的方法来设置 $b$，就凭直觉来设置。

![](/images/positional-encoding/pe-18.png)

这个方法听起来很简单，那我们看一下它的效果怎么样。横轴是 Sequence 的长度，纵轴是 Perplexity，困惑度越小证明效果越好。下面的图片表明，在更长的 Sequence 上，ALiBi 的效果会表现得更好，而且 **ALiBi 全面碾压 Sinusoidal 方法**。

![](/images/positional-encoding/pe-19.png)

### T5 的可学习 Relative Bias

参考文献：[https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)

讲到这边你可能会想说，这个 $b$ 都是人为设置的，如果直接用 **Learn 的方法** 会不会更好呢？其实早在 ALiBi 之前，就一定有人想过用 Learn 的方法来确定 $b$。

对于 Learn 的方法有很多不同的版本，我们这边重点讨论 T5 的方法。它的大概做法就是 $A$ 减掉一个 Bias $b_{m-n}$，这个 Bias 是跟相对距离 $m$，$n$ 有关系的，它是一个**可以被训练的参数**。比如说位置 0~5 是一个数值，5~10 是一个数值，20 之后可能又统一成为其他的数值。

那这个效果怎么样呢？**这个效果是要比 ALiBi 还要差的**，所以 ALiBi 人手设置的数值，要比当时训练的数值结果还要好。

![](/images/positional-encoding/pe-20.png)

后来，ALiBi 也消失在了历史的洪流之中，逐渐被更强的 Positional Embedding 方法所取代。但是它给我们的启示就是：**Relative Position 真的很重要**。

### RoPE：Rotary Position Embedding

参考文献：[https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)

RoPE 是现在最流行的方法之一。它是怎么做的？我们先讲结论：

今天在做 Attention 的 Dot Product 之前，先把 Position 加到 $k$ 上面，但是它加 Position 的方法并不是直接加入一个 Embedding，而是**通过旋转 $k$ 和 $q$ 这两个向量的方法来加入 Positional 的信息**。我们会得到包含信息位置的 $\boldsymbol{k^n_A}$ 和 $\boldsymbol{q^m_B}$。

![](/images/positional-encoding/pe-21.png)

然后再用含信息位置的 $q$ 和 $k$ 来算 Attention Weight：

$$
\begin{align}
a &= (\boldsymbol{k^n_A})^T \boldsymbol{q^m_B} \\
&= \boldsymbol{(k_A)}^T R_{m-n} \boldsymbol{q_B}
\end{align}
$$

其中，$R_{m-n}$ 只跟 A、B 之间的**相对位置**有关，这样的话我们就可以把相对位置矩阵很直接地塞入到 A 跟 B 之间。

这个方法很重要的一点是：**它不会影响 Attention 计算的过程**。像 ALiBi 的话，它会在做 Attention 过程当中多做一个操作。但是 RoPE 基本上完全没有影响到原来 Attention 的操作，只是 $q$ 跟 $k$ 变得跟原来有些不同而已，而且可以把变换的 $K$ 放到 KV Cache 里面。

那我们现在来看看在 RoPE 这个方法里面是怎么去加 Position 的。

#### Rotation as Position

我们现在只考虑 $k$ 跟 $q$ 这两个向量的**前两位**。

![](/images/positional-encoding/pe-22.png)

把前两位拿出来，它就是一个二维平面上的向量。

![](/images/positional-encoding/pe-23.png)

当我们把位置信息加进去的时候，其实就是对这个向量做了一个**旋转**。如果你今天要把位置 $n$ 的信息加进去，那就把原来的 $k$ 前两位旋转 $n\theta$ 这个角度，然后就会得到 $\boldsymbol{k^n_A}[0]$ 和 $\boldsymbol{k^n_A}[1]$。

![](/images/positional-encoding/pe-24.png)

![](/images/positional-encoding/pe-25.png)

同理，$q$ 也做同样的运算。当我们把位置加进去之后，就相当于旋转 $m\theta$ 个角度。

![](/images/positional-encoding/pe-26.png)

所以对 RoPE 来说，它就是把旋转的角度当做 Position 的信息。

在刚才举例的时候，举例的是前两位，因为只有前两位才可以放到二维平面里面。但是 RoPE 是**每两个维度**来进行考虑的。

![](/images/positional-encoding/pe-27.png)

第三、第四个维度，我们就会用另外的角度来进行旋转。我们会把 $\boldsymbol{k_A}[2]$ 和 $\boldsymbol{k_A}[3]$ 旋转 $n\theta'$ 的角度，得到 $\boldsymbol{k^n_A}[2]$ 和 $\boldsymbol{k^n_A}[3]$，$q$ 也同理。

#### 旋转角度怎么设定？

因为每两个维度都会旋转不同的角度 $n\theta_i$，$n\theta_i$ 的公式如下：

$$
\theta_i = \frac{1}{10000^{2i/d}}
$$

其中，$i = 0, 1, ..., \frac{d}{2} - 1$。

![](/images/positional-encoding/pe-28.png)

#### RoPE 的核心性质：相对位置不变性

我们看看 RoPE 可以达到什么样的效果。假如说有一个 $k$ 在 $n$ 的位置，有一个 $q$ 在 $m$ 的位置，我们对它做 Attention。如果我们现在把 $k$ 移到 $n+r$ 的位置，把 $q$ 移到 $m+r$ 的位置，它们之间的**相对位置不变**，那么 Attention 的数值也是不会改变的。

$$
\boldsymbol{k^n_A} \cdot \boldsymbol{q^m_B} = \boldsymbol{k^{n+r}_A} \cdot \boldsymbol{q^{m+r}_B}
$$

假设猫的 Position 是 1，鱼的 Position 是 3，我们会把猫的 $k$ 加上 Position 1 的信息，把鱼的 $k$ 加上 Position 3 的信息。鱼这个 Token 得到它的 Query，再把位置加上去得到 $\boldsymbol{q^3_B}$，然后通过和猫的 $k$ 进行计算，得到了 Attention $a$。

![](/images/positional-encoding/pe-29.png)

现在在猫跟鱼前面加了大量其他的 Token，也不会影响猫跟鱼这两个 Token 算出来的 Embedding。假设猫的 Token 在 101，鱼的 Token 位置在 103，它们两个之间算出来的 Attention 数值，跟猫的 Token 在位置 1、鱼的 Token 在位置 3 算出来的 Attention 数值是一样的。

![](/images/positional-encoding/pe-30.png)

下面再进一步说明为什么 RoPE 可以达到这个效果。下图中，原来的 $k$ 跟加上位置 $n$ 之后的 $k$ 的对比：

![](/images/positional-encoding/pe-31.png)

假设 $k$ 又被放到 $n+r$ 的位置，那就相当于把它旋转 $(n+r)\theta$，获得了一个新的位置。

![](/images/positional-encoding/pe-32.png)

那么 $\begin{bmatrix} \boldsymbol{k^n_A}[0] \\ \boldsymbol{k^n_A}[1] \end{bmatrix}$ 和 $\begin{bmatrix} \boldsymbol{k^{n+r}_A}[0] \\ \boldsymbol{k^{n+r}_A}[1] \end{bmatrix}$ 之间的夹角是 $r\theta$，对 $\boldsymbol{q_B}$ 来说也是一样的。

![](/images/positional-encoding/pe-33.png)

所以 $\begin{bmatrix} \boldsymbol{k^n_A}[0] \\ \boldsymbol{k^n_A}[1] \end{bmatrix}$ 和 $\begin{bmatrix} \boldsymbol{q^m_B}[0] \\ \boldsymbol{q^m_B}[1] \end{bmatrix}$ 的内积，与 $\begin{bmatrix} \boldsymbol{k^{n+r}_A}[0] \\ \boldsymbol{k^{n+r}_A}[1] \end{bmatrix}$ 和 $\begin{bmatrix} \boldsymbol{q^{m+r}_B}[0] \\ \boldsymbol{q^{m+r}_B}[1] \end{bmatrix}$ 是一样的，因为只是把向量做了等量同样角度的旋转，所以不会改变它的内积。

#### RoPE 的数学推导

接下来我们来探讨一下使用 RoPE 为什么只跟 $k$ 和 $q$ 的相对位置有关。

我们还是只拿前两个维度来举例。不带位置的 $k$ 是如何加上位置的？假设旋转了 $n\theta$ 角度，公式如下：

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

同理，$q$ 要旋转 $m\theta$ 公式如下：

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

接下来我们要让 $k$ 跟 $q$ 做 Dot Product：

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
\boldsymbol{k^n_A}[0] \\
\boldsymbol{k^n_A}[1]
\end{bmatrix}^{\!T}
\begin{bmatrix}
\boldsymbol{q^m_B}[0] \\
\boldsymbol{q^m_B}[1]
\end{bmatrix} \\[1em]
&=
\left(
\begin{bmatrix}
\cos(n\theta) & -\sin(n\theta) \\
\sin(n\theta) & \cos(n\theta)
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{k_A}[0] \\
\boldsymbol{k_A}[1]
\end{bmatrix}
\right)^{\!T}
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

其中，$\begin{bmatrix} \cos(n\theta) & -\sin(n\theta) \\ \sin(n\theta) & \cos(n\theta) \end{bmatrix}^T$ 代表旋转 $-n\theta$，$\begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix}$ 代表旋转 $m\theta$，合起来就是 $\begin{bmatrix} \cos((m-n)\theta) & -\sin((m-n)\theta) \\ \sin((m-n)\theta) & \cos((m-n)\theta) \end{bmatrix}$ 代表旋转 $(m-n)\theta$。

如果是把 $\boldsymbol{k^n_A}$ 换成 $\boldsymbol{k^{n+r}_A}$，把 $\boldsymbol{q^m_B}$ 换成 $\boldsymbol{q^{m+r}_B}$，会有什么不同？

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
\end{bmatrix} \\[1em]
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

其中，$\begin{bmatrix} \cos(((m+r)-(n+r))\theta) & -\sin(((m+r)-(n+r))\theta) \\ \sin(((m+r)-(n+r))\theta) & \cos(((m+r)-(n+r))\theta) \end{bmatrix}$ 代表旋转 $((m+r)-(n+r))\theta = (m-n)\theta$。

所以结果与没有加 $r$ 时完全相同，验证了相对位置不变性。

#### RoPE 的实现细节

刚才说 RoPE 采取的方法是，把 $k$ 跟 $q$ 分别做旋转，再去做 Attention。

![](/images/positional-encoding/pe-36.png)

其实这也等价于把 $q$ 乘上 $R_{m-n}$，也就是说 $q$ 自己做旋转，然后再去跟 $k$ 做相乘，相当于只在 $q$ 上做计算。

![](/images/positional-encoding/pe-37.png)

虽然这两个方法计算起来是等价的，实际上会去采用**前一种方式**。因为在我们存 KV Cache 的时候，如果用前一种方式，可以把带位置的 $k$ 跟 $q$ 都存下来。但如果我们用后一种方式，那 $q$ 每次和 $k$ 去做 Attention 的时候都需要乘上不同的 $R_{m-n}$，效率很低。所以 RoPE 能流行起来的主要原因，就是它可以很好地跟别的方法做搭配。

#### RoPE vs ALiBi：一个重要的区别

很多人对 RoPE 有一个误解——它可能像 ALiBi 一样，当 $k$ 跟 $q$ 的距离越远，算出来的 Attention Weight 就越小。但事实是：**RoPE 并没有保证 $q$ 跟 $k$ 距离越远的话，它们的 Attention Weight 就越小。**

但是 RoPE 没有"越远越小"不一定是一件坏事，它可以做到原来 ALiBi 做不到的事情。假设我们现在有一个 Token，它想跳过前一个 Token 直接注意到前面的位置。对于 ALiBi 来说，如果这个 Token 离原来的 Token 越远，注意力就越小。但对于 RoPE 来说，它可以完成一件事：**直接忽略中间的 Token，直接注意到前面的 Token**。

![](/images/positional-encoding/pe-38.png)

有的时候，这个方式对于模型理解句子是很有帮助的。比如说"我的猫"和"他的狗"这两个句子——"猫"应该注意到"我"上面，"狗"应该注意到"他"上面，而不是注意到中间的"的"上面。所以 RoPE 可以做到比 ALiBi 更细致的 Attention。

假设 $q$ 跟 $k$ 中间差了 $2\theta$，但是 $n+1$ 位置的 $q$ 跟 $n$ 位置的 $k$ 可能就差了一个 $\theta$，$n+2$ 位置的 $q$ 跟 $n$ 位置的 $k$ 它们的角度可能就是一样的。那么这就恰恰证明了，中间隔了一个 Token，但是注意力分数却可以往上涨，它就可以忽略掉中间的 Token。

![](/images/positional-encoding/pe-39.png)

---

## 04 Train Short, Test Long

参考文献：[https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html](https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html)

我们能不能在训练的时候，虽然 Transformer 在训练阶段只能看见比较短的 Sequence，但是让它在**测试的时候看到非常长的 Sequence 也不要坏掉**？

![](/images/positional-encoding/pe-40.png)

因为我们在训练的过程中，可能没办法找到非常长的语料来进行训练。而且在今天的 Agent 当中，我们希望 Agent 可以永无止境地运行下去，所以说它需要吃非常长的 Sequence，要有非常大的 Context Window。所以我们期待有什么样的 Positional Embedding 可以做到 Train Short Test Long。

怎么做 Train Short Test Long？一个很直觉的想法就是，我们训练的时候最多只见过 $N$ 个 Token，我的位置也是 $1, 2, ..., N$。

![](/images/positional-encoding/pe-41.png)

![](/images/positional-encoding/pe-42.png)

假设测试的时候模型需要处理 $LN$ 个 Token，我们能不能就直接给它我们需要的编号？

![](/images/positional-encoding/pe-43.png)

RoPE 可以处理这样的状况吗？看样子是不行的。

（下图来自文献：[https://arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409)）

![](/images/positional-encoding/pe-44.png)

下图就是测试的时候给模型不同长度的 Sequence，然后看它的 Perplexity 效果如何。假如测试的时候给他们非常长的 Input Sequence：

- 对于 **Sinusoidal** 来说，只要测试 Sequence 一长，它就直接坏掉。
- 对于 **RoPE** 来说，它会比 Sinusoidal 稍微好一点点，但是随着 Sequence 越长它也会坏掉。
- 事实上，只有 **ALiBi** 是能够支撑得住的，因为它不是通过学习得到的参数，而是通过人为设置得到的。看起来这种人为设置是非常 Robust 的，它可以处理比较长的 Sequence。

但是 ALiBi 比较简单，所以后人想的是，能不能**通过调整 RoPE**，让它能够做到 Train Short Test Long。

### 为什么 RoPE 在长序列上会失败？

假设我们在训练的时候，旋转最大的角度就是 $N\theta$，超过这个角度的向量它都没有见过。假如说你在测试的时候给它的角度是 $2N\theta$。对 RoPE 来说，它从来没有看到过一个向量被转到过这个地步，它就会"发疯"，完全不知道要怎么做。

![](/images/positional-encoding/pe-45.png)

### Position Interpolation

参考文献：

- [https://arxiv.org/pdf/2306.15595](https://arxiv.org/pdf/2306.15595)
- [https://kaiokendev.github.io/context#a-bigger-problem](https://kaiokendev.github.io/context#a-bigger-problem)

那么要怎么办呢？那就是**不要给它超过 $N$ 的旋转角度**。

比如现在有 $LN$ 个 Token，我们的编号不一定要从 $1$ 到 $LN$，我们可以让它从 $\frac{1}{L}$ 到 $N$——从来没有规定过 Position 的编号一定要是整数。

![](/images/positional-encoding/pe-46.png)

![](/images/positional-encoding/pe-47.png)

这个方法叫做 **Position Interpolation（位置插值）**。但是使用这个方法之前，模型还是要针对这种非整数的旋转角度进行**微调**，否则效果还是很差。

不过 Position Interpolation 不一定能够带给我们很好的表现，所以又有各种新方法。

### Frequency-Based Approach

有一个新的方法叫做 **Frequency-Based Approach**。

刚才我们在做 RoPE 的时候，是两个维度两个维度考虑的，而 Position Interpolation 的方法是对不同的维度做同样的处理。Frequency-Based Approach 的想法是：**能不能在不同的维度做不同的处理？**

比方说，我们在测试的时候要把 $N$ 扩展到 $LN$，我们能不能乘上一个特别的函数？这个函数一方面跟我们要拓展多少位有关，同时也会跟现在在哪一个维度有关。

![](/images/positional-encoding/pe-48.png)

那么 Frequency-Based Approach 是怎么设计的呢？它的原则是：$\theta_0$ 是比较高频的向量，$i$ 的编号越大就代表越低频。对于高频向量，可以不动，不用改它的 Position；对于低频向量，则需要进行压缩。

![](/images/positional-encoding/pe-49.png)

对于 RoPE 而言，它每两个 Dimension 合起来也是一个指针。这个指针也是在二维的平面上来旋转。对于前两个维度而言，它的指针旋转速度是非常快的。假设 $N=128$，那么对于 $\theta_0$ 来说，它可能已经把整个平面上的维度全部看了一遍甚至几遍；而对于 $\theta_{32}$ 来说，它可能看了连 1/4 都没有。

![](/images/positional-encoding/pe-50.png)

所以对于 $\theta_0$ 来说，你就算给它大于 $N$ 的 Position 也是没有关系的，因为它已经全部看过了。但如果在 $\theta_{32}$ 这个维度上面，你如果把 $N$ 提高到 256 的话，那 256 旋转的那个角度它是没有看过的。

![](/images/positional-encoding/pe-51.png)

#### NTK-aware Scaling

Frequency-Based Approach 有很多不同的变体，其中比较知名的变体叫 **NTK-aware Scaling**。

NTK-aware Scaling 的公式如下：

$$
f(L, i) = \left(\frac{1}{L}\right)^{\frac{2i}{d-2}}
$$

这个公式的设计是有意义的：

- 假设 $i=0$，那么 $f(L, 0) = 1$，在最高频的 $\theta_0$ 上面是不需要做任何压缩的。
- 假设 $i = \frac{d}{2} - 1$，处在最低频的状态，那么 $f\left(L, \frac{d}{2} - 1\right) = \frac{1}{L}$，它会把最低频的 $\theta_{\frac{d}{2}-1}$ 压缩到 $\frac{1}{L}$。

不过 NTK-aware Scaling 的方法并没有一篇对应的 Paper，它是出现在 Reddit 的文章里面。

参考文献：[https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)

![](/images/positional-encoding/pe-52.png)

### YaRN

另外一种知名的变体叫做 **YaRN**（Yet another RoPE extensionN method）。

参考文献：[https://arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071)

YaRN 的思想是：保留一些低频维度的 Scaling 不变，保留一些高频维度的 Scaling 不变，**只变中间的**。

![](/images/positional-encoding/pe-53.png)

### Dynamic Scaling

还有一种另外的想法叫做 **Dynamic Scaling**。它的核心思想是：之前的方法可能长的序列都能做，但是对于短的 Sequence 来说，可能它的效果会变得更差。

参考文献：[https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/)

![](/images/positional-encoding/pe-54.png)

所以 Dynamic Scaling 的想法就是，对于不同长度的 Sequence 用不同的处理方式。比如我们要输入 5 个 Token，那我们的 Token 数量就要乘 $\frac{4}{5}$。但如果 Sequence 小于 Training 的 Sequence，那我们就不需要做任何改动，这样的话你在短序列上的效果也不会差，因为在 Training 的时候见过这些小的 Sequence。

![](/images/positional-encoding/pe-55.png)

这个想法很简单，但是你会发现它会破坏 KV Cache。

还有一种 Dynamic Scaling 的变体方法：在一定比例的 Sequence 上不做任何 Position 改动，过一定比例后再做 Position 改动，没有必要压缩最前面的 Position。

![](/images/positional-encoding/pe-56.png)

那 Dynamic Scaling 的方法好不好呢？如果我们使用 Dynamic Scaling 加上 Position Interpolation 或者加上 NTK 方法，它的效果会比不加 Dynamic Scaling 更好。

![](/images/positional-encoding/pe-57.png)

### Frequency-Based + Dynamic：LongRoPE

Frequency-Based 方法是决定后面的压缩函数，Dynamic Scaling 来决定在哪个地方开始压缩。具体要怎么决定呢？有一篇论文叫做 **LongRoPE**，用了一个 **Evolutionary Search** 的方法，去找出最好的 Frequency-Based 方法和最好的 Dynamic Scaling 方法。

参考文献：[https://arxiv.org/abs/2402.13753](https://arxiv.org/abs/2402.13753)

![](/images/positional-encoding/pe-58.png)

这个方法可以得到一个非常惊人的结果——它可以让模型的输入长度处理到 **2048K**。

![](/images/positional-encoding/pe-59.png)

---

## 05 No Positional Embedding？！

有一个灵魂的发问：**真的需要 Positional Embedding 吗？**

### Self-Attention 真的没有位置信息吗？

如果我们只考虑第一层的 Self-Attention，那它确实是没有位置信息。但如果我们考虑**第二层**的 Self-Attention，那结论就不太一样了。

我们看第一层 Attention 可以发现猫跟"吃"的关系，还有鱼跟"吃"的关系。对于下一层的 Attention，它看到第二个位置的时候，会看到一个猫跟吃有关的东西，还有一个鱼跟吃有关的东西。所以"猫吃鱼"和"鱼吃猫"最后得到的 Embedding 是不一样的。所以**就算没有 Positional Embedding，Self-Attention 在多层的情况下也是具有 Positional 信息的**。

![](/images/positional-encoding/pe-60.png)

### NoPE：没有 Positional Embedding

这篇论文提出了一个方法叫做 **NoPE**，就是没有 Positional Embedding。

参考文献：[https://arxiv.org/pdf/2305.19466](https://arxiv.org/pdf/2305.19466)

这篇论文训练了一些 Transformer 来做任务，发现其实**不加 Position Embedding 也没关系**。下图是一些结论，横轴代表做的任务长度，纵轴代表准确率，准确率越高越好。

我们会发现在 Copy 这个任务里面，加上 Positional Embedding 以后，很多模型在 Sequence 长到一定程度时就下降了。可是在没有加 Positional Embedding 的情况下，它的准确率非常高。

![](/images/positional-encoding/pe-61.png)

如果根据上面的结论，那我们在大模型训练时不应该加 Position Embedding。但为什么现在的大模型都会加 Position Embedding 呢？

有一篇文章有一些相关的讨论。下面这张图比较了 **RoPE** 跟 **NoPE** 训练的过程。横轴是训练的 Step（做了几次训练参数的更新），纵轴是 Loss，Loss 越小越好。结论显而易见——**NoPE 在训练的时候效果就不如 RoPE 好**。

所以我们为什么需要 Positional Embedding？是因为我们在 **训练** 的时候需要它。

![](/images/positional-encoding/pe-62.png)

### DroPE：训练后期扔掉 Positional Embedding

那我们能不能在训练到一定程度之后就把 Positional Embedding 扔掉呢？现在有一个方法叫做 **DroPE**。它的想法是：一开始的训练是有 Positional Embedding 的，训练到快结束的时候，把 Positional Embedding 拔掉。Loss 就会突然冲上去，但是很快 Loss 就可以下降下来。在没有 Position Embedding 的情况下，它的训练效果是可以跟 RoPE 一样低的。

那 DroPE 这个方法表现怎么样呢？它是可以比 **RoPE + YaRN** 表现还要好的。比如我们训练的 Context 是 2K，如果说我们一直在用 Positional Embedding 的话，那当我们在做 Inference 的时候，Context 超过训练的 Context，它的效果就会下降。但是因为我们用 DroPE 的时候会把它的 Positional Embedding 在后面拔掉，它也会在没有 Positional Embedding 的情况下进行学习。那么这个时候，在我们的 Inference Context 超过 Training Context 的时候，它的效果依旧很好。

![](/images/positional-encoding/pe-63.png)

参考文献：[https://arxiv.org/pdf/2512.12167](https://arxiv.org/pdf/2512.12167)

---

## 参考资料

1. [**Sinusoidal Positional Embedding（原始 Transformer）**](https://arxiv.org/abs/1706.03762)
2. [**ALiBi（Attention with Linear Biases）**](https://arxiv.org/abs/2108.12409)
3. [**T5（Text-to-Text Transfer Transformer）**](https://arxiv.org/abs/1910.10683)
4. [**RoPE（Rotary Position Embedding）**](https://arxiv.org/abs/2104.09864)
5. [**Train Short Test Long（RoPE Context Extension 综述）**](https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html)
6. [**Position Interpolation**](https://arxiv.org/pdf/2306.15595)
7. [**Position Interpolation（Kaiokendev 博客）**](https://kaiokendev.github.io/context#a-bigger-problem)
8. [**NTK-aware Scaling（Reddit 帖）**](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
9. [**YaRN（Yet another RoPE extensionN method）**](https://arxiv.org/abs/2309.00071)
10. [**Dynamic Scaling（Reddit 帖）**](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/)
11. [**LongRoPE**](https://arxiv.org/abs/2402.13753)
12. [**NoPE（No Positional Embedding）**](https://arxiv.org/pdf/2305.19466)
13. [**DroPE**](https://arxiv.org/pdf/2512.12167)
