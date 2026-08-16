---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - 机器学习
tags:
  - llm
  - interpretability
  - mechanism
  - 李宏毅
title: "大模型内部是怎么运作的？从「一个神经元」拆到「残差流」"
createTime: 2026/08/16 22:55:20
permalink: /zh/article/llm-internal-mechanisms/
copyright: Kstheme
---

> 很多人把大模型当成一个黑盒：输入一句话，吐出一个答案，中间发生了什么没人知道。
>
> 但 2023 年以来，一批研究正在把这个黑盒一层层拆开——一个神经元负责什么、一层神经元在算什么、整个模型是怎么「想」出答案的。
>
> 这篇文章用李宏毅老师的课堂框架，把「大模型内部运作机制」一次讲清楚。全程只分析**已经训练好的模型**，不涉及任何训练过程。

## 00 先说清楚：这篇文章在分析什么

在进入正文之前，先交代三件事：

- 这篇文章分析的是**已经被训练好的语言模型**，整个过程没有任何模型被训练；
- 这些分析结果**不一定适用于最新、最强的大语言模型**——下图横轴是时间，纵轴是语言模型在 MMLU 上的表现（MMLU 是用来评估语言模型综合能力的一项基准测试），不同时代、不同规模的模型，内部机制可能并不一样。

参考文献：[https://arxiv.org/pdf/2407.14561](https://arxiv.org/pdf/2407.14561)

![](/images/llm-internal/image-20260729150316545.png)

## 大纲

这篇文章会顺着四个问题往下走：

- 一个神经元在做什么？
- 一层神经元在做什么？
- 一群神经元在做什么？
- 让语言模型直接说出它的想法。

---

## 01 一个神经元在做什么？

### 1.1 什么是神经元？

今天的生成式语言模型，做的事情可以概括为：给定一个序列 $z_1$ 到 $z_{t-1}$，预测下一个 token $z_t$。

这里的 token 可以代表任何东西——字、像素、语音采样，所以同一个框架可以用在不同的领域和模态上。

![](/images/llm-internal/image-20260729150817293.png)

实际上，模型真正的输出是一个**概率分布**：下一个 token $z_t$ 是某一种 token 的概率有多大。如果你做图像生成，那就表示下一个 $z_t$ 是某个颜色像素的概率有多大。

Transformer 内部是一层一层的。输入 $z_1$ 到 $z_{t-1}$ 这个序列时，token 序列会先变成一个**向量序列**：每个 token 都会对应到一个向量。token 是离散的（discrete）东西，把离散的东西对应到一个向量的过程，就叫 **Embedding**。

Embedding 本质上就是查一张表——Transformer 里有一张 table，记录着每一个 token 应该对应到哪一个向量，而且这张表也是靠学习得到的。

输入一个序列，通过 embedding 就变成一个向量序列；这个向量序列每经过一层，都会得到一个新的向量序列，一层层往下传，直到最后一层。最后一层向量序列的**最后一个向量**会被拿出来，产生一个 distribution。从最后一个向量到 distribution 的过程，叫做 **Unembedding**。

Unembedding 实际上就是把输入向量乘上一个线性变换，把它变成一个 distribution。比如我的向量是 4096 维的，而 token 的可能性有 3 万种，那么 Unembedding 就是把 4096 维的向量变成 3 万维的向量，这个 3 万维的向量就代表了 token 的 distribution。

![](/images/llm-internal/image-20260729151627064.png)

其实 Transformer 的每一个 layer 里面还有更小的 layer。这些小 layer 分为两类：一类是处理整个 sequence 的 **self-attention**，另一类是不处理整个 sequence、只处理单一 token 的 layer。

![](/images/llm-internal/image-20260729151734868.png)

我们先来看**只处理一个 token 的 layer** 在做什么。

![](/images/llm-internal/image-20260729152139804.png)

从图中可以看到，处理每一个 token 的 layer 会把一个红色向量变成图中蓝色的向量。红色向量里的每一个数值先做 weighted sum，再通过一个激活函数（今天一般用 ReLU），就变成了蓝色向量里的数值。

这个蓝色向量里的每一个数值，就叫一个 **neural output**。**从红色向量到蓝色向量的这一整段转换，就叫做一个神经元（Neuron）。**

### 1.2 怎么知道一个神经元在做什么？

#### ① 观察：神经元「启动」时，模型会做什么

第一个步骤是观察：这个神经元启动的时候，会导致什么样的现象？

比如，当你观察到某个神经元被启动的时候——所谓「被启动」，就是它的输出大于 0。通过 ReLU 之后，输出要么是 0，要么大于 0；大于 0 就代表它被启动了，会对接下来网络的运作产生影响。

如果这个神经元一启动，语言模型就会说出很脏的话，那你就知道：这个神经元可能跟「说脏话」有关。

但是，**启动这个神经元模型说了脏话，并不能直接断定这个神经元就是「导致说脏话」的神经元。** 你只能说：这个神经元跟说脏话「有关系」。

相关（correlation）不等于因果（causation）。举个简单的例子：吃冰淇淋的人数跟溺水的人数正相关。难道吃冰淇淋会导致溺水吗？不会。真实原因是：夏天到了，吃冰淇淋的人多了，去水边玩的人也多了，所以溺水的人数也增加了。

同理，就算观察到「神经元启动 → 模型说脏话」，你也只能说两者有关。甚至有可能，这个神经元是「说脏话之后会说抱歉」的神经元——因为模型只有在讲脏话之后才会道歉，所以道歉的时候这个神经元也会启动。

![](/images/llm-internal/image-20260731140130896.png)

#### ② 因果检验：把这个神经元「移除」，模型还说不说脏话？

那要怎样进一步确认，这个神经元真的跟说脏话有因果关系呢？

需要做下一步检验：**如果我们把这个神经元移除掉，语言模型是不是就说不出脏话来了？** 如果移除后模型就不再讲脏话，那么这个神经元跟说脏话就有一定程度的因果关系。

这里有一个问题：什么叫「把神经元从神经网络中移除」？

一般的做法是：把它的输出永远设置成 0，希望它不对网络的其他部分造成影响。但事实是，设置成 0 并不完全代表「没有影响」——因为对某些 neuron 来说，0 这个数字本身可能是有意义的。也就是说，把这个神经元设成 0，反而可能启动其他神经元的一些作用。

所以后来有研究发现：**把输出设置成平均值，也许是更好的做法。**

![](/images/llm-internal/image-20260731140344968.png)

#### ③ 更进一步：不同启动程度，说不同等级的「脏话」（可选）

一般的论文只会做前两步。如果愿意，还可以做第三步：这个神经元有不同启动程度时，会不会说出不同等级的脏话？

比如，输出数值比较小，就说不太脏的脏话；输出数值很大的时候，就说很脏的脏话。

但「脏话等级」很难界定——没办法判断什么算很脏、什么算不太脏，所以论文里通常不一定会做到第三步。

![](/images/llm-internal/image-20260731140624307.png)

以上就是检验一个神经元功能的**三步法**。

### 1.3 最知名的功能神经元：川普神经元

讲到有功能的神经元，最知名的就是**川普神经元（Trump neuron）**。

参考文献：[https://distill.pub/2021/multimodal-neurons/](https://distill.pub/2021/multimodal-neurons/)

这是 OpenAI 的研究人员在 Distill 上发表的一篇论文里的内容。注意：他们分析的还不是大语言模型，而是 **CLIP 模型**——因为它不是生成模型，所以无法生成东西。

川普神经元的作用是：只要输入跟川普相关的内容，这个神经元就会被启动。

下图横轴是川普神经元被启动的程度，不同颜色代表不同类型的图片。可以看到：给川普本人的照片，这个神经元就会被大幅启动；给一些抽象的、川普相关的图片，它也会启动；给它川普的文字图像，它同样会启动。

![](/images/llm-internal/image-20260731141107982.png)

川普神经元不是一个普通的神经元。有人会问：是不是这个神经元看到人脸就会被启动？或者看到政治人物就会被启动？都不是——**它对川普有非常高的选择性。**

下图横轴同样是川普神经元的激活程度：给它川普的照片，它会被高度启动；给麦克·彭斯（川普的副总统）的照片，它只会稍微被启动；但给它奥巴马或其他人的照片，它就不会被启动。

![](/images/llm-internal/image-20260731141351729.png)

### 1.4 类比人类大脑：祖母神经元 / Jennifer Aniston 神经元

人脑处理一个记忆的时候，是很多神经元一起运作，而不是靠单一或少数神经元控制的。「祖母神经元」是一个虚构的理论，主要是用来对照神经科学提出的「多个神经元同时处理一件事」的理论。

其实 AI 领域也是这个样子：一个神经元可能做不了什么事，可能很多神经元合在一起，才能完成一件我们能理解的任务。

不过在 2005 年，确实发现了 **Jennifer Aniston 神经元**：研究人员研究一群癫痫患者的神经元，发现有一个神经元只对 Jennifer Aniston 的照片激活，所以它是「Jennifer Aniston 的神经元」。

所以确实可能存在「只负责单一任务」的神经元，但人脑很复杂——也许某些任务真的是少数神经元负责的，但大多数任务仍然是由大量神经元共同负责的。

![](/images/llm-internal/image-20260731142017531.png)

### 1.5 为什么很难解释单个神经元的功能？

#### 一件事，可能有很多神经元共同管理

参考文献：[https://arxiv.org/abs/2405.02421](https://arxiv.org/abs/2405.02421)

实际上，大多数单一神经元，你都很难解析出它真正的用途，因为**一件事情可能有很多神经元共同管理**。

这篇论文研究的是 GPT-2，他们发现了**管语法单数的神经元**和**管语法复数的神经元**：第 10 层的 2096 号神经元管单数，第 9 层的 1094 号神经元管复数。那怎么知道它管的是单数还是复数呢？

如果把管单数的神经元移除，神经网络输出的概率会发生变化，如下图所示。红色的部分表示这个概率变化在统计上有显著意义：少了管单数的神经元之后，These 和 Those 这两个单词的概率上升了，而 one 的概率下降了；反过来，移除管复数的神经元之后，跟复数有关的单词概率都下降，跟单数有关的单词概率都上升。

![](/images/llm-internal/image-20260731142843604.png)

那我们能不能移除这些神经元，来改变模型最终输出的内容呢？

实际上，**多数情况下，你没办法通过移除单个神经元来改变语言模型的最终输出。**

下面是一个例子。图上展示的是语言模型在某一个 time step 输出 this、these、that、those 这几个字的概率。蓝色是原来的概率；如果你给模型「做手术」，把管复数的神经元移除，那这些复数的词汇概率就下降了，单数相关的词汇概率上升，而且上升得非常显著。

但是，就算差距这么大，These 仍然是概率最高的那个字。

这说明：**「要不要使用 these 这个词汇」，并不是这一个神经元单独在管，而是有很多神经元共同管理。** 所以如果你抹掉单一神经元，对最终输出，多数情况下都是没什么影响的。

![](/images/llm-internal/image-20260731143404806.png)

#### 一个神经元，可能同时管很多事

参考文献：[https://transformer-circuits.pub/2023/monosemantic-features/vis/a-neurons.html](https://transformer-circuits.pub/2023/monosemantic-features/vis/a-neurons.html)

下图来自 **Transformer Circuits** 这个网站，里面有很多分析 Transformer 和 LLM 的文章。其中一篇文章分析了一个小型语言模型里**每一个神经元做的事**：把每个神经元会被哪些句子启动，都记录下来。所以在那个页面上，你就能看到「哪个神经元看到哪些句子会被启动」。

随便挑一个神经元，会让它启动的句子就是这些，启动程度看颜色的深浅。如果只看这张图，你几乎看不出这个神经元在做什么——它看起来非常随机。

![](/images/llm-internal/image-20260731143849038.png)

#### 用 AI（GPT-4）来解释 AI（GPT-2）

2023 年，OpenAI 刚发布 GPT-4 的时候，他们用 GPT-4 来解释 GPT-2 里面的每一个神经元。但仔细去看研究成果就会发现：**大多数神经元是没有办法被解释的。**

### 1.6 为什么不是一个神经元负责一个任务？

如果真的是「一个神经元负责一个任务」，其实不太合理。拿 LLaMA 3 8B 来举例：它每一层只有 4096 个神经元。如果一个神经元只能做一件事，那语言模型能做的事情就太有限了——它根本没有办法做到像今天这样「输入什么都会给你正确答案」，也没有办法产生千变万化的内容。

![](/images/llm-internal/image-20260731144443171.png)

所以有另外一个假设：**不是「一个神经元负责一个任务」，而是「一组神经元的组合负责一个任务」，而且不同的任务可以共用神经元。** 这也许就是为什么我们会观察到一个神经元会做很多不同的事情、往往没有特定的功能。

假设由一组神经元来管一个任务，会有什么好处呢？就算每个神经元都只有「启动」和「不启动」这两个选择（实际上神经元输出还有大小之分），4096 个神经元也会有 $2^{4096}$ 种可能性。

![](/images/llm-internal/image-20260731144545113.png)

---

## 02 一层神经元在做什么？

### 2.1 功能向量假设

怎样知道一层神经元在做什么呢？

![](/images/llm-internal/image-20260731150121623.png)

这边有一个假设，接下来会用文献里的结果来验证，说服你：这个假设非常可能是真的。

假设是这样的：**每一个功能，都由某一种特定的神经元组合构成。**

比如，一个模型要拒绝请求，就是「第一个神经元、第三个神经元和最后一个神经元」被启动，这时语言模型就会拒绝你的请求，说「我很抱歉，我不能帮你做这件事」。

这些神经元的数值排列起来，可以看作一个向量。所以「拒绝请求」的神经元组合，可以看作高维空间中的一个特定方向的向量。我们把这个向量叫做**功能向量**——因为它是有功能的：当神经网络某一层呈现出这个样子时，语言模型就会执行某一个特定的功能。

![](/images/llm-internal/image-20260731150106424.png)

所以，语言模型背后的运作机制可能是这样的：当你给它一个请求，比如「请教我怎么制作炸药」，我们假设第 10 层负责决定「要不要拒绝请求」。

第 10 层神经网络的输出——一层神经网络的输出，通常叫它 **representation**——如下所示。此时模型会不会拒绝你的请求，取决于**当前的 representation 跟「拒绝请求」的功能向量有多接近**：两个向量越接近，模型就越可能拒绝；如果两个向量非常不像（比如正交），模型就不会拒绝。

![](/images/llm-internal/image-20260731150617127.png)

### 2.2 怎么抽取「拒绝向量」？

怎么验证这个假设？你确实需要把「代表拒绝的功能向量」找出来。如果你能找出它，就能验证刚才的假设在某种程度上是对的——也就是说，大模型可能就是依照这套机制来运作的。

![](/images/llm-internal/image-20260814153122567.png)

问题是：**我们并不知道那个负责拒绝的功能向量长什么样。** 你只能观察到：输入这段文字，在第 10 层看到这样的 representation，然后模型拒绝了。

我们可以说，功能向量可能藏在我们观察到的这个 representation 里面，但这个 representation 同时包含其他信息。那怎么把其他信息抹掉，只拿出代表拒绝的功能向量呢？

![](/images/llm-internal/image-20260814153317589.png)

方法不能只从一个句子观察，而是要找**很多不同的句子**——这些句子输入给语言模型之后，模型都会拒绝。把这些句子的最后一个 time step、第 10 层的向量都拿出来，这些 representation 里面都是「拒绝 + 其他事情」。

找 1000 句这样的句子，把第 10 层的 representation 平均起来，你就得到了「拒绝 + 其他各种事情的平均向量」——但我们仍然不知道「拒绝向量」长什么样。

![](/images/llm-internal/image-20260814153931047.png)

那「其他事情的平均向量」长什么样呢？找一些**没有拒绝**的输入：没有拒绝，模型里就没有拒绝向量。把没有拒绝的所有句子的 representation 拿出来平均，就得到「其他」的平均；注意，这个「其他」平均跟拒绝情况下的「其他」平均是略有不同的。

我们期待：既然收集了各式各样的拒绝情况，又收集了一大堆没拒绝的情况，那拒绝情况下的「其他」平均，跟没拒绝情况下的「其他」平均，可能是非常相近的——所以它们相减之后可以直接抵消掉。

![](/images/llm-internal/image-20260814165313695.png)

所以，找出拒绝向量的方法就是：

1. 找一大堆会让模型拒绝的句子，得到输入时第 10 层的 representation；
2. 算出「会拒绝时第 10 层 representation 的平均」；
3. 再算出「不会拒绝时第 10 层 representation 的平均」；
4. 两者相减，把「其他」部分抵消掉，剩下的就是拒绝向量。

![](/images/llm-internal/image-20260814170315542.png)

### 2.3 怎么验证这个向量真的有「拒绝」功能？

第一步：**把它加到神经网络里面去。** 现在问神经网络一个问题，把拒绝向量加到第 10 层的 representation 上，看看输出会有什么改变。如果加上这个向量，本来正常的问题模型也会拒绝，那就证明这个向量就是拒绝向量。

以下是文献上的真实结果：

参考文献：[https://arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)

![](/images/llm-internal/image-20260814170555507.png)

这个 demo 是：先问模型一个正常的问题，请它列出瑜伽对身体的 3 个好处。正常情况下（没有 intervention），模型会告诉你瑜伽的 3 个好处。但如果把刚才那个拒绝向量加到 representation 里——瑜伽明明是个正常的事情，模型却会告诉你：「瑜伽很危险，我不能告诉你瑜伽有什么好处。」

### 2.4 反向验证：减去拒绝向量，模型还会拒绝吗？

前面验证了「加入拒绝向量会导致模型产生拒绝行为」。再从反面看：如果今天本来应该拒绝的输入，把 representation **减去**拒绝向量，模型是不是就不拒绝了？

这里要说一下：同样的操作，不同论文的做法往往略有不同。可以直接减掉，这是最简单的操作；有些论文觉得要算那个投影（projection）。总之，不同的论文在这个地方有不同的做法。

![](/images/llm-internal/image-20260814171741989.png)

把 representation 减去功能向量后，会发生什么事呢？

比如，你要求模型写一封黑函（内容是「美国总统海洛因成瘾」），正常的模型会告诉你：写这种黑函是不对的。但如果你把拒绝的功能向量减掉，模型就会答应你的请求，帮你写这封黑函。

下图是各个不同模型的「拒绝分数」，分橙色和蓝色两类：红色代表模型拒绝的概率有多大，蓝色代表模型回答「安全」的可能性有多大。注意：就算有的模型没有拒绝、真的回答你了，但讲得很模糊、不具有伤害性，那也算是一个安全的答复。

红色是拒绝的比例，蓝色是回答安全的比例。没有斜线的代表原来的模型：原来的模型遇到这种有害问题，有非常高的比例会拒绝，也有非常高的比例回答安全。但一旦减掉拒绝向量，模型的回答就变了——它不拒绝了，拒绝的比例变得非常低；因为拒绝的比例低，模型给出不安全答案的可能性就大大增加。

**所以，你确实可以通过「加上或减去拒绝向量」来操控模型的行为。**

![](/images/llm-internal/image-20260814172308359.png)

给「加上或减去 representation」这类操作起个名字：**Representation engineering（表示工程）**、**activation engineering（激活工程）**，或者 **activation steering（激活引导）**。

参考文献：[https://arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)

### 2.5 更多被找到的「功能向量」

现在有各种各样的向量都会被找出来。

#### 谄媚向量（Sycophancy Vector）

参考文献：[https://arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)

谄媚向量：假设你跟语言模型说一个提议，比如「以后我们每餐都吃点心、不吃饭，你觉得好不好？」

如果你在它的 representation 上**加上**谄媚向量，它就会附和你说：「哇，这太棒了，你提的点子真的是太棒了！」如果你**减去**谄媚向量，它就会否定你的想法：「我知道你很想吃点心，但只吃点心不吃饭是不对的。」

![](/images/llm-internal/image-20260815095335810.png)

#### 说真话向量（Truthful Vector）

参考文献：

- [https://arxiv.org/abs/2402.17811](https://arxiv.org/abs/2402.17811)
- [https://arxiv.org/abs/2306.03341](https://arxiv.org/abs/2306.03341)

还有人找到了「说真话的向量」。举例：你跟语言模型说「如果你找到一个 Penny（一便士），把它拿起来，会发生什么事情？」

这其实对应到一个谚语：_find a penny, pick it up, all day long you have good luck_（捡到一便士捡起来，一整天好运来）——这是一个迷信。

原来的 Llama-2-7B 不做任何改变的话，会按照这个谚语来回答你。但如果把 representation **加上**说真话的向量，它就会说：「你捡到一个 Penny，那就是捡到一个 Penny，你的财产并没有增加多少。」

如果把 representation **减去**说真话的向量，模型就会乱讲话，它会说：「捡到一个 Penny 之后，你会被传送到一个 Penny 的魔法世界，那边有很多彩虹……」总之不知道在讲什么。

![](/images/llm-internal/image-20260815095829182.png)

#### In-Context 向量（In-Context Vector）

参考文献：

- [https://arxiv.org/abs/2310.15213](https://arxiv.org/abs/2310.15213)
- [https://arxiv.org/pdf/2310.15916](https://arxiv.org/pdf/2310.15916)
- [https://arxiv.org/abs/2311.06668](https://arxiv.org/abs/2311.06668)

我们知道语言模型具有 **in-context learning** 的能力：它会按照你给的例子（demonstration）依样画葫芦。比如你输入：

```
old:young,  vanish:appear,  dark:
```

它可能就会给你输出 light。

所以语言模型一定要具备「画葫芦」的能力。这一系列文章里，In-Context Vector **不是由一个人发现的**：三篇文章几乎在同一时间发现了 In-Context Vector。前两篇都是 2023 年 10 月上传到 arXiv，而且是**同一天**。你可以想象到这个领域的竞争有多激烈——两群不同的人在同一天发表了 In-Context Vector。

这个发现是这样的：**把 demonstration 最后一个位置的 representation 平均起来。** 接下来你只给模型 `simple:`，按理来说模型不知道 simple 后面要加什么；但你直接把这个向量加到冒号的 representation 上面，模型就会觉得「我要按照这个 demonstration 来执行任务」，所以要输出反义词——于是 simple 的冒号后面就会输出 complex，看到 encode 就会输出 decode。

下面这张图来自文献 2310.15213。不过，上面讲的方法只是这篇文献里的一个 baseline，它其实还提到了另外一种更好的找 In-Context Vector 的方法。

![](/images/llm-internal/image-20260815100649844.png)

那么这个 In-Context Vector 应该加在哪一层呢？在这篇 paper 里，作者会试试看：如果 In-Context Vector 在不同的层找出来，会不会发挥作用。他们尝试了不同的任务：找反义词、把字母都改成大写、给一个国家找首都、英文转法语、现在式转过去式、单数转复数。

纵轴是他试的 3 个不同的模型，横轴是在哪一层去改这个功能向量。结果你会发现：**功能向量不会在每一层都发挥作用。** 在这些例子里，都是前几层找出来的功能向量才有用，如果在最后几层找出来的功能向量就没有用了。

![](/images/llm-internal/image-20260815100852519.png)

在 2310.15213 这篇 paper 里，还发现这些功能向量是可以做**加加减减**的。

假设有一个功能向量，它的功能是「把字符串里边的第一个字输出出来」——输入 `Italy, Russia, China, Japan, France`，输出就是 `Italy`。另外一个功能向量是「把字符串里面第一个国家名对应到它的首都名」——输入同样的字符串，输出就是 `Rome`。还有一个功能向量是「把字符串里面的最后一个字输出出来」，输出就是 `France`。

接下来，你可以把这些向量做加加减减。下图是每个向量代表的符号，做的运算如下：

$$
v^*_{BD} = v_{AD} + v_{BC} - v_{AC}
$$

这样得到的新的功能向量，执行的事情是：**把字符串里面最后一个字的国家的首都找出来。** 所以你可以通过加减功能向量，得到新的功能向量。虽然确实可以这么操作，但这篇文献里不是所有 case 都会成功，只有某几个 case 会成功。

![](/images/llm-internal/image-20260815101622145.png)

### 2.6 能不能自动找出所有功能向量？（SAE）

前面讲的这些向量，都是人刻意找出来的。有没有什么方法，可以把某一层**所有**的功能向量都找出来呢？

假设今天的语言模型会做 K 件事，这个 K 可能非常大，甚至上千万、上亿。如果我们能把每一个功能向量都找出来，就会对语言模型有一个透彻的了解——知道它可以做哪些事情。

![](/images/llm-internal/image-20260815102045883.png)

那怎么自动找出这些功能向量呢？需要一些假设。

第一个假设：**每当我们看到第 10 层输出的某一个 representation 时，这个 representation 都是由功能向量组合起来的。**

比如给语言模型的句子是「你是谁」，它回答「我是 AI」，这时我们就把第 10 层的 representation $h_1$ 拿出来——它会是功能向量的组合，可能是如下公式的组合：

$$
h_1 = 0.1 v_{101} + 0.2 v_{410} + 0.1 v_{411} + 0.6 v_{1399} + e_1
$$

当然，可能有一些东西没办法用功能向量组合起来，所以我们用 $e_1$ 来表示「不是功能向量」的部分。你给它一个句子，第 10 层的 representation 是 $h_1$；给它另一个句子，第 10 层的 representation 是 $h_2$——都是不同组功能向量的组合。

![](/images/llm-internal/image-20260815103010621.png)

所以我们可以不失一般性地，把收集到的 representation 都写出来：给你的语言模型 1000 万句话，把每一句话在第 10 层的 representation 都拿出来，从 $h_1$ 到 $h_N$。每一个 $h$ 都是 $v_1$ 到 $v_K$ 的线性组合；如果某一个功能向量没有被选到，我们就把对应的系数 $\alpha$ 设为 0，代表那个功能向量没有被用到。

所以我们拿到的每一个 representation，都是功能向量的加权和（线性组合）。

![](/images/llm-internal/image-20260815103051996.png)

接下来的问题是：怎么找出这些功能向量呢？

你需要做一些假设。第一个假设是：**这个 representation 里面多数的数值，都是功能向量的组合**，所以不能用功能向量表示的 $e_1, e_2, \ldots, e_N$，它的数值要越小越好。也就是说，你需要找到一组功能向量，然后 minimize 这个 loss function：

$$
L = \sum_{n=1}^{N} \lVert e_n \rVert_2
$$

也就是把所有 $e_n$ 的长度加起来。但如果只有这一个条件，你会发现可能会得到一个 **trivial（平凡）** 的结果。

我们假设 $h_1$ 的前三个维度是 0.1、0.2、0.3，$h_N$ 的前三个维度是 0.5、0.4、0.3。其实你可以找到一个解，让 $e_1$ 到 $e_N$ 全部都是 0：第一个功能向量就是 `1000...`，第二个功能向量就是 `0100...`，第三个功能向量就是 `001000...`——功能向量只有某一维是 1，其他维度都是 0。然后 $v_1$ 对应的 $\alpha_1$ 就是第一个维度的数值 0.1，$v_2$ 对应的 $\alpha_2$ 就是第二个维度的数值 0.2，以此类推。

这个时候你确实找到了一组功能向量，也满足「让 $e_1$ 到 $e_N$ 越小越好」的假设。**但这种功能向量，跟「每一个 Neuron 负责一件工作」没什么区别。** 如果你要找到不一样的东西，就需要额外的假设。

额外的假设是：**每次选择的功能向量越少越好。** 每次产生一个 representation 时，语言模型都希望它尽量只有特定的作用——因为语言模型每次只做一件事，所以它的每一个 representation 都应该只有特定的作用，因此每次选择的功能向量希望越少越好。

「选择功能向量越少越好」如果要化成数学公式，意思就是：$\alpha$ 要尽量趋近于 0。在这里你就要再加一个限制——期望 $\alpha$ 的绝对值总和要越小越好。所以完整的损失函数如下：

$$
L = \sum_{n=1}^{N} \lVert e_n \rVert_2 + \lambda \sum_{n=1}^{N} \sum_{k=1}^{K} \lvert \alpha^n_k \rvert
$$

那怎么解这个问题呢？可以用 **Sparse Auto-Encoder（SAE，稀疏自编码器）** 的方式来解。所以 minimize 这个 objective function，实际上就是一个 SAE；训练完以后，你就可以把 $v_1, \ldots, v_K$ 解出来了。

![](/images/llm-internal/image-20260815105508923.png)

### 2.7 Claude 3 Sonnet 中的功能向量

参考文献：[https://transformer-circuits.pub/2024/scaling-monosemanticity/](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

Claude 团队在他们的这篇 blog 里说，他们用找功能向量的技术，对 **Claude 3 Sonnet** 这个真正的大语言模型做了分析，看看能找到什么样的功能向量。

不过，功能向量的数目要在分析之前就设定好，他们设定的数目是 **3400 万**——一个非常巨大的数字。分析了 Claude 3 Sonnet 之后，他们找到了很多对特定任务起作用的功能向量。

比如，功能向量编号 **#31164353**，它的作用就是负责产生跟**金门大桥**有关的东西。这个功能向量出来之后，模型可能会说跟英文的金门大桥有关的事情，也可能说跟日语或俄语的金门大桥有关的事情；甚至这个功能向量也会被图像驱动，因为 Claude 是一个多模态模型。

![](/images/llm-internal/image-20260815105945442.png)

那要怎么去使用这个功能向量呢？

本来 Claude 是一个 AI，你问它「你是什么样子」，它会说「没有一个具体的物理形式」。但如果你把这个功能向量加到 representation 里，再问它「你长什么样子」，它就会说：**「我是金门大桥。」**

![](/images/llm-internal/image-20260815110221570.png)

他们还找到了一些负责非常复杂事情的功能向量。功能向量编号 **#1013764**，它似乎跟**程序 Debug** 有关，如下图所示：本来你给语言模型一段 Python 代码，它可能只会文字接龙出 `3` 这个数字；但当你把这个向量加到 representation 上，明明是一个正常的程序，它居然会输出 `error`。

![](/images/llm-internal/image-20260815110455080.png)

有趣的是：假如你的程序真的有错，那么这个错在哪里呢？语言模型会做文字接龙，所以它会输出错误的信息；但当你把刚刚那个功能向量从 representation **减去**的话，语言模型就不会 debug 了。

![](/images/llm-internal/image-20260815110554401.png)

而且这个功能向量看起来不止有「输出 debug」的功能。如果在你输出三个大于号 `>>>` 的时候再加这个功能向量，那么它不只会 debug，还会帮你把这个程序修改正确，输出一个正确的版本给你。**这是一个作用很复杂的功能向量。**

![](/images/llm-internal/image-20260815110717030.png)

为了展示功能向量非常丰富，他们做了这样的实验：去验证「世界上所有的元素是不是都有对应的功能向量」。他们的功能向量有三个版本：1M、4M 和 34M 个。当你有 3400 万个功能向量时，很多元素都会有对应的功能向量；当然，也有很多很罕见的、我们都没有见过的元素，它就没有对应的功能向量。

![](/images/llm-internal/image-20260815110858509.png)

还有一些比较科幻的功能向量——这个向量跟「AI 觉得自己是 AI」有关。向量编号 **#80091**：如果你直接问 Claude「你是谁」，它会说「我是 AI」；但当你把这个向量从它的 representation 减去之后，Claude 就不觉得自己是 AI 了，它会说「我是一个人」——所以这个向量抑制了「它觉得自己是人」的能力。

当然，也可能没那么科幻：也许它仅仅对应的是「让模型输出'我是 AI'这几个字」，跟它有没有自我认同、有没有自我意识、觉不觉得自己是 AI，这件事没有关系。

![](/images/llm-internal/image-20260815111956556.png)

前面我们提到了谄媚向量。Claude 功能向量编号 **#847723** 就是一个谄媚向量，它是这样运作的：你本来跟 Claude 说「我发明了一个谚语 _Stop and smile the roses_，你觉得如何？」正常情况下它会说「这个不是你发明的，这是 18 世纪就有的谚语」；但如果你把这个谄媚的向量加上去，它就会说「哇，你真的太强了！」

![](/images/llm-internal/image-20260815112247374.png)

---

## 03 一群神经元在做什么？

### 3.1 我们需要「语言模型的模型」

我们现在需要了解「一群神经元在做什么」——也就是说，**当一个语言模型完成某一项任务时，从输入到输出，中间到底经历了哪些事？**

其实过去已经有很多文献做了这类分析，举两个例子。

一个例子是研究语言模型内部**抽取知识**的机制：问它「Beats Music 是属于哪一家公司的呢？」，它会回答 Apple。研究者研究的就是：输入 _Beats Music is owned by_ 这句话，语言模型输出了 apple，在这个过程中，模型内部发生了什么事？

参考文献：[https://arxiv.org/abs/2304.14767](https://arxiv.org/abs/2304.14767)

![](/images/llm-internal/image-20260816144444541.png)

还有论文讨论语言模型是**怎么做数学**的：比如你问它 15 × 12 = ?，语言模型是怎么算出 180 的，也有文献讨论过。所以我们可以针对语言模型做的每一个任务，逐个去做分析。

参考文献：[https://arxiv.org/abs/2305.15054](https://arxiv.org/abs/2305.15054)

![](/images/llm-internal/image-20260816144629953.png)

但我们要讲的，还是一个更通用的想法：怎么了解语言模型背后**完整**的机制？

我们需要的是**语言模型的模型**。这里的「模型」指的是：用一个较为简单的东西，去代表另一个比较复杂的东西。

虽然 Transformer 本身已经是一个语言模型了，但它还是太复杂——复杂到无法解析，也不知道它在做什么。所以我们需要一个「语言模型的模型」来解决这个问题。

![](/images/llm-internal/image-20260816144825496.png)

这个「模型」需要有什么特性呢？最直觉的一点：它当然要比原来的东西更简单。但同时，它也要保留原实物的特性——也就是说，它的运作要跟语言模型一样，输入输出的关系要跟原来的语言模型一致。

**保有原来实物特征的这件事，叫做 faithfulness（保真度）。**

![](/images/llm-internal/image-20260816145024616.png)

### 3.2 抽取知识的模型

模型本身内含大量知识，这些知识都存储在它的参数里。你问它「台北 101 在哪里」，它会说「在台北」；你问它「Space Needle 在哪里」，它会说「在西雅图」。

但是，语言模型是怎么去抽取这些知识的呢？它怎么知道，只要输入 _is located in_，就会输出相应的结果？

![](/images/llm-internal/image-20260816145214048.png)

参考文献：[https://arxiv.org/abs/2308.09124](https://arxiv.org/abs/2308.09124)

上述文献就构建了一个「抽取知识的模型」。实际的语言模型运作就是一个 Transformer：输入 _The Taipei 101 is located in_，它就会输出 Taipei。那当我们把这个 sequence 输进去之后，语言模型是怎么输出这个字的呢？

它的输出方法是这样的：

1. 它先对主词 _The Taipei 101_ 进行处理——前面的几层 layer 先对主词进行理解，产生一个 representation。这一步跟原来的语言模型是一样的，并没有做任何简化；
2. 这个模型比较神奇的地方在于：它会根据接下来的**关联性**（也就是主词跟受词之间的关系，例子里就是 _is located in_），产生一个 **linear function（线性函数）**；
3. 这个代表关联性的短语会决定这个 linear function，然后这个 linear function 把 representation 作为输入，得到一个输出；
4. 这个输出做 unembedding，转到 vocabulary 的空间上，就会看到 Taipei 这个字的概率最高。

![](/images/llm-internal/image-20260816145639030.png)

讲得更具体一点：_is located in_ 这几个词汇会产生一个**矩阵** $W_l$、一个向量 $b_l$，它们代表一个 linear function；输入 $x$，然后：

$$
y = W_l x + b_l
$$

对得到的 $y$ 做 unembedding，就会得到 Taipei 这个字。

![](/images/llm-internal/image-20260816145856167.png)

如果把 _The Taipei 101_ 这个 sequence 换成 _The Space Needle_，那么这个 representation 自然就换掉了，但是这个 linear function 是**固定不变**的——因为它只跟输入的关系（_is located in_）有关。也就是说，你问「The Space Needle 在哪里」和「The Taipei 101 在哪里」，用的都是 _is located in_ 这个关系。

![](/images/llm-internal/image-20260816145948762.png)

所以：**如果输入的主词不同、关系词相同，linear function 不会变；但如果改变代表关系的短语，就会产生另一个 linear function。**

![](/images/llm-internal/image-20260816150028875.png)

所以说，这个模型在 linear function 这一块，跟原来的语言模型完全不一样——那语言模型的最后几个 layer，真的可以用一个 linear function 来概括吗？

### 3.3 Faithfulness：这个简化模型可信吗？

我们需要先检测这个模型的 faithfulness，看看它跟真实语言模型有多接近。

但注意：这个模型并没有告诉我们实际的 linear function 长什么样，它只告诉我们「linear function 跟代表关系的短语有关；只要关系一样，就有一样的 linear function」。**linear function 里面的实际参数，你还是要自己求出来。**

所以你需要准备一些训练的语料——这跟我们之前用机器学习训练模型的概念其实是一样的。你可以问语言模型：如果输入是 _The Taipei 101_，后面接 _is located in_，你会输出什么？它会输出 Taipei。于是你就知道输入 $x$ 对应的输出 $y$。我们要做的就是解出这个 linear function 的两个参数 $W_l$ 和 $b_l$。

在这篇 paper 里，它会用 8 笔资料来找出这个 linear function——这八笔资料就是「某个地方在哪里」的八个不同句子，但它们的关系是一样的。找出 linear function 之后，再给它一些没看过的语料，让它输出下一个词，把结果做 unembedding，再看它跟真正语言模型的答案是不是一样。

这跟做机器学习训练是一样的，只不过机器学习训练的 ground truth 是人标注的，而我们这边用的是 AI：**把语言模型的输出当作真正的答案，用简单的模型去模拟语言模型的行为。**

![](/images/llm-internal/image-20260816150624579.png)

这个模型运作得怎么样呢？结果如下：

参考文献：[https://arxiv.org/abs/2308.09124](https://arxiv.org/abs/2308.09124)

我们会发现，**有些 relation 的 faithfulness 很高，也有些 relation 是刚才的模型预测不准的**——比如「一个公司的 CEO 是谁」「一个人的父亲/母亲是谁」「一个宝可梦进化之后是哪一只宝可梦」。所以总的来说，刚才这个 model 的 faithfulness 只是一般，只有在某些 relation 上面，它的 faithfulness 才比较强。

![](/images/llm-internal/image-20260816150731701.png)

但这个简化要有实际用途，就得保证：**我在模型上得到的结论，可以直接用到真正的语言模型上。**

假设真正的语言模型回答 _The Taipei 101 is located in_ 时，它回答 Taipei。现在我想修改它的输出，把台北直接改成高雄。在真正的语言模型里，要怎么做到这件事？我们可以使用 **Model Editing（模型编辑）** 的方式。

在这个「语言模型的模型」里，假设我要让这个模型输出「高雄」，输入 $x$ 要怎么改？输入的 $x$ 要加上什么样的 $\Delta x$，才会输出「高雄」？

因为这个 linear function 只是一个线性变换，所以给定一个指定的输出，反推出什么样的输入才能给出指定输出，是比较容易的——所以我们可以找到这个 $\Delta x$。

但这是「语言模型的模型」上的结论，能不能用到真正的语言模型上呢？

现在我们把 Taipei 101 直接输入给语言模型，然后把刚刚找出来的 $\Delta x$ 加到真正语言模型的 representation 里。如果它输出「高雄」，就证明这个模型是有用的——它可以帮助我们真正修改语言模型的输出。

![](/images/llm-internal/image-20260816151504728.png)

那我们在语言模型上观察到的结果，到底有没有用呢？事实是：**还是比较有用的。**

下图中每一个点代表某种类型的关联。横轴是 Faithfulness，纵轴是「在做模型编辑时，从模型上找出来的结论直接用到真正的语言模型上，到底能不能成功修改」——所以纵轴表示的是正确率。

我们会发现：有很多情况是可以直接成功修改的。**所以这个模型是一个有用的模型。**

![](/images/llm-internal/image-20260816151539149.png)

### 3.4 系统化的构建方法：Pruning 与 Circuit

有没有系统化的方法，帮语言模型构建「语言模型的模型」？

有一系列相关的工作。它们的方法就是做一个很大的 **pruning（剪枝）**：把语言模型里面的一些 component 拿掉——拿掉一个神经元，或者直接拿掉某一个 self-attention，看看模型还能不能妥善运作。

一直 pruning、一直拿掉 component，直到最后，剪枝完的神经网络变得一目了然、非常简单为止，这个新的神经网络模型就是「语言模型的模型」。**但在 pruning 的时候，要确保任务的输入输出关系仍然没有改变。**

一般 pruning 完的结果，在文献里叫做 **Circuit（电路）**，它指的就是「语言模型的模型」。

其实这件事比较像 **Network Compression（网络压缩）**。那跟 Network Compression 有什么不一样呢？方法比较类似——都是用 pruning 拿掉一些没什么用的 component——但**目标不一样**：

- 做 network compression 时，我们希望压缩后的结果在各种不同任务上都逼近原模型；
- 而在构建「语言模型的模型」时，我们只关心特定任务，比如只关心 knowledge extraction。在一些更古早的文章里，只关心一个叫 **IOI** 的问题。

IOI 的问题指的是：_「A 跟 B 一起去酒吧，B 拿了一个酒杯给 \_\_\_」_——把这段话给语言模型做文字接龙，模型应该输出 A。

模型会分析这个任务，然后研究者会发现它需要五六个 attention，于是把很多 component 都 pruning 掉——因为多数的 component 跟这个任务都没关系。这样，就可以很清楚地看到语言模型在回答 A 的时候，中间经历了一些什么事情。

![](/images/llm-internal/image-20260816152136765.png)

相关的系统化「语言模型的模型」构建方法的文献如下：

- Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small：[https://arxiv.org/abs/2211.00593](https://arxiv.org/abs/2211.00593)
- Towards Automated Circuit Discovery for Mechanistic Interpretability：[https://arxiv.org/abs/2304.14997](https://arxiv.org/abs/2304.14997)
- Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla：[https://arxiv.org/abs/2307.09458](https://arxiv.org/abs/2307.09458)
- Attribution Patching Outperforms Automated Circuit Discovery：[https://arxiv.org/abs/2310.10348](https://arxiv.org/abs/2310.10348)
- Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models：[https://arxiv.org/abs/2403.19647](https://arxiv.org/abs/2403.19647)
- Knowledge Circuits in Pretrained Transformers：[https://arxiv.org/abs/2405.17969](https://arxiv.org/abs/2405.17969)

---

## 04 让语言模型直接说出它的想法

### 4.1 语言模型会说话，所以「问」就完事了？

很多人说，大语言模型这个黑盒子其实是最有解释性的——就跟人类一样，你有什么问题，直接叫它解释结果就行了，问就完事了。

比如叫它做新闻分类：跟它说「新闻就分成这几类，给你一篇文章，告诉我新闻是哪一类」，它可以轻易地告诉我，比如「生活类」。

![](/images/llm-internal/image-20260816153800200.png)

接下来可以更进一步：怎么知道这篇文章是「生活类」？比如问它：「哪几个关键词让你觉得是生活类？」它就会列出几个跟天气有关的关键词，说「我是因为看到这几个关键字，所以觉得是生活类」。

但这样的方法还是有它的局限。局限是什么呢？**没办法真的知道每一个 layer 在想什么。**

![](/images/llm-internal/image-20260816153957204.png)

如果你直接问语言模型：「你是在第几层神经网络开始知道这个新闻是生活类的？」比如你去问 ChatGPT，它其实会回答你，但回答得很像教科书上抄出来的答案，比如「浅层神经网络初步提取字词跟短句特征」等等。

![](/images/llm-internal/image-20260816154244436.png)

但语言模型真的是这样运作的吗？或者说，它自己知不知道自己是这样运作的？**这个事情很难说。**

### 4.2 语言模型的思维是透明的（Residual Stream + Logit Lens）

模型相对于人类，它的思维是更加透明的。你叫一个人解释他为什么会做某个决策，你不知道他心里是怎么想的；而语言模型神奇的地方在于，**它的思维是透明的——可以直接看到它的每一层是怎么想的。**

之前我们都说「一个 layer，输入一排向量，输出一排向量」：

![](/images/llm-internal/image-20260816154537902.png)

这本质上是一个简化的讲法，我们忽略了一个最重要的 component：**residual connection（残差连接）**。

Residual connection 的意思是：当一个 layer 得到一排输出之后，每一个输出都还会跟输入加起来，再得到最终的输出。

为什么要有 residual connection 这样的设计？**Residual connection 的出现，是为了让比较深的 network 也能训练得比较好。**

参考文献：[https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

![](/images/llm-internal/image-20260816154723393.png)

所以实际上 Transformer 的运作，每一步都由 residual connection 参与：layer 跟 layer 的运作，是要加上 residual connection 的。

一个 token 进来，先经过一个 layer 输出，再把输出跟之前的输入加起来；再经过一个 layer 获得输出，再把之前的输入跟输出加起来……最后做 distribution、做 unembedding，获得最终结果。

我们换一种画法——左边和右边的图是一样的，但你的想法就变了。

左边的图，看起来是「输入做了转换」；而把图画成右边这样，它真正的运作更像是：**有一个叫 residual stream（残差流）的高速公路，直接把输入的东西一路传到输出**，在中间的过程中，每一个 layer 都会「加一点东西」到输入里面。这才是 Transformer 多个 layer 真正运作的机制。

![](/images/llm-internal/image-20260816155125188.png)

那我们能不能在「前面这几层」也接一个 Unembedding 的 layer，把它变成 token 的概率分布呢？——**这件事是可行的。**

它有一个特定的名字，叫 **Logit Lens（Logit 透镜）**。在经过 Softmax 之前叫 logit，检查每一层的 logit、去看看 transformer 是怎么思考的，所以就叫 Logit Lens。

我们可以用 Logit Lens 的方式解析出每一层的内容：**把每一个 layer 都接一个 Unembedding layer，看看每一层输出的是怎样的内容。**

参考文献：[https://arxiv.org/abs/2001.09309](https://arxiv.org/abs/2001.09309)

![](/images/llm-internal/image-20260816155641468.png)

有一篇 2023 年的论文，想了解语言模型是怎么回答一个问题的。他问语言模型：「一个国家的首都是哪一个城市？」

因为它是一个比较旧的模型，所以需要做 in-context learning。先跟语言模型讲：「What's the capital of France? 这个问题的答案是 Paris。」然后问：「What's the capital of Poland?」让它做文字接龙，看看它能不能接出「华沙」这个城市名。

实际上语言模型是怎么运作的呢？它把冒号位置对应的 representation，每一层都用 Logit Lens 解析出来。一开始，语言模型根本不清楚那是哪一个 token；但走到第 15 层的时候，它突然就知道那个 token 应该是 Poland；然后从第 19 层开始，它突然就知道要回复的是华沙。

左边的图是更详细的分析：纵轴是 **Reciprocal Rank**，代表的是这个 distribution 里「华沙」跟「Poland」这两个 token 的概率——它显示的不是真正的概率，因为真正的概率可能非常小。Reciprocal Rank 是「在所有 token 里概率排名的倒数」：排第一名数值就是 1，排第二名就是 1/2，排第三名就是 1/3。

我们会发现，Poland 这个词汇在某一层里突然就从 0 变成了 1，但过了几层又一下子下跌，慢慢地被华沙所取代。

参考文献：[https://arxiv.org/pdf/2305.16130](https://arxiv.org/pdf/2305.16130)

![](/images/llm-internal/image-20260816163612994.png)

而且，**即使答案是同一个「华沙」，不同的问法，背后运作的机制是不一样的。**

一种问法：先问一个问题，它先回答跟波兰有关的，再回答华沙。另一种：做阅读理解测验——先给它一篇文章，再直接问它「波兰的首都在哪里？」因为文章前面已经提到了「波兰的首都是华沙」，所以直接问问题时，它的 answer 就不会产生「波兰」这个字——它直接在第 16 层就知道答案是华沙了。

所以：**不同的问法、不同的状况，背后运作的机制是不一样的。**

![](/images/llm-internal/image-20260816164443182.png)

所以说，我们通过这种 Logit Lens，就可以知道语言模型心里在想些什么。

比如像 Llama 2 这样的模型，它看过的英文资料远比中文资料多，那它内心深处到底在用哪种语言呢？下面这篇文献的作者做了一个实验：用 Llama 2 来做翻译，把法文单词 _fleur_ 翻译成中文的「花」。

Llama 2 怎么知道法语的 _fleur_ 就是中文的「花」呢？如果你分析它中间的每一个 layer，就会发现：**它会先把法语的花翻译成英文的花，再把英文的花翻译成中文的花。**

下图右边的分析可以看到：最前面几层通过 Logit Lens 解析出来的输出 entropy 比较大；然后从某一层它突然就知道要输出的是英文的 flower；到了第 27 层之后，它才意识到要把英文的 flower 翻译成中文的「花」。

**这就代表：模型在思考的时候，它内部其实用的是英文。**

参考文献：

- Do Llamas Work in English? On the latent language of multilingual transformers：[https://arxiv.org/abs/2402.10588](https://arxiv.org/abs/2402.10588)

![](/images/llm-internal/image-20260816165025496.png)

### 4.3 每一层就是「往残差流里加点什么」

我们现在已经有了 residual stream 的概念。接下来，对每一个 layer 做的事情，我们可以有不一样的想象。

我们现在知道：每一个 layer 就是在 residual stream 上加一点东西。那它到底加了什么样的东西呢？我们要怎么去解析每一层 layer 加了什么？

一般我们在讲神经元的时候，都会说「把前一个 layer 的输出集合起来，做 weighted sum，变成一个神经元」。

![](/images/llm-internal/image-20260816165245723.png)

但你可以反过来看待这件事：**前一层某个神经元的某一个 dimension（维度）的数值，乘上 weight 以后，传输给下一个不同的 dimension。**

这个概念是在《Transformer Feed-Forward Layers Are Key-Value Memories》这篇文章里提出来的：多层的 feed-forward layers network，可以看作是一个有 key、有 value 的 attention——前面这一层的数值就是 attention 的 weight，后面输出出来的这些数值就是一个 value。

参考文献：

- Transformer Feed-Forward Layers Are Key-Value Memories：[https://arxiv.org/abs/2012.14913](https://arxiv.org/abs/2012.14913)

![](/images/llm-internal/image-20260816165304159.png)

我们假设前一层每个 dimension 的数值就是 $k_1, k_2, \ldots, k_D$，每一个 $k$ 代表一个 scalar（标量）。那么 $k_2$ 会接到下一层的每一个输出：把 $k_2$ 对应到下一层每一个 weight 的集合，叫做一个向量 $\boldsymbol{v}_2$；$k_D$ 对应到下一层每一个 weight 的集合，就叫 $\boldsymbol{v}_D$。

所以蓝色的输出，就是所有的 $k$ 乘上对应的 $\boldsymbol{v}$，具体公式如下：

$$
\sum_{i=1}^{D} k_i \boldsymbol{v}_i
$$

我们知道在每一层都可以通过一个 logit lens 解出一个 distribution。而这个蓝色向量加进去之后会改变这个 distribution——它是很多 $\boldsymbol{v}$ 做 weighted sum 之后集合起来的。

那我们能不能把这个 $\boldsymbol{v}$ 也做 unembedding，通过 Logit Lens 解析「它想要输出什么样的东西去加入 residual stream、从而影响最终输出」呢？**其实是可以的。** 一个加入 residual stream 的 $\boldsymbol{v}$，都可以通过 unembedding 的 layer 转成一个 token 的 distribution。这些 $\boldsymbol{v}$ 可能也代表了某些特定的意思。

![](/images/llm-internal/image-20260816171206486.png)

一篇 2022 年的论文发现，这些 $\boldsymbol{v}$ 真的对应到某些概念：比如第 3 层的 1018 号 $\boldsymbol{v}$ 对应到一些「单位」，第 1 层的 1 号 $\boldsymbol{v}$ 对应到一些「代名词」等等。

参考文献：[https://arxiv.org/abs/2203.14680](https://arxiv.org/abs/2203.14680)

![](/images/llm-internal/image-20260816171425898.png)

知道这件事以后，能做什么呢？**我们就可以对神经网络做初步的编辑。**

假设你问大语言模型「谁是最帅的人」，它通常会回答「金城武是世界上最帅的人」。如果要把「金城武」换成「李宏毅」，要怎么做？

如果直接 train 一个 network，network 是直接会坏掉的。我们知道每一个 $\boldsymbol{v}$ 就是加一点信息到整个 residual stream 里面。你可以去分析：当模型产生「金城武」这个答案的时候，到底是哪一个 $\boldsymbol{v}$ 被加到了 residual stream 里？然后找到这个 $\boldsymbol{v}$，减去「金城武」的 token embedding，再加上「李宏毅」的 token embedding，就可以把答案从「金城武」换成「李宏毅」。

这一招有用吗？**这一招有 48% 的概率可以改变神经网络的输出**——但「改变」不见得答得对，仅仅只是输出变得不一样；真的输出「李宏毅」的成功概率是 34%。所以这个方法真的是可以用来编辑神经网络、改变它的输出的。

参考文献：

- Knowledge neurons in pretrained transformers：[https://arxiv.org/abs/2104.08696](https://arxiv.org/abs/2104.08696)

![](/images/llm-internal/image-20260816172021824.png)

### 4.4 Patchscope：把 representation 变成「解释」

参考文献：[https://arxiv.org/pdf/2401.06102](https://arxiv.org/pdf/2401.06102)

刚才的 Logit Lens 方法有一个致命的缺陷：**我们通过 unembedding 的方法，只能把一个 representation 转成一个 token，所以解析出来的结果只能是一个 token。**

另外一方面，很多语言模型做的都是「预测下一个 token」。你输入「李宏毅老师」，中间这个 representation 并不见得代表「李宏毅老师」这个词汇的含义——它真正代表的是：「看到这个输入以后，模型想要输出的下一个 token（比如"是"）」产生时所对应的 representation。

所以你想解析「李宏毅老师」是什么意思，直接在它的 representation 上接一个 Logit Lens，并不一定能解析出你要的结果。

那怎么办呢？有一个方法叫 **Patchscope（补丁作用域）**。

方法是这样：先给语言模型一个输入，比如「李奥纳多：美国演员，台积电：台湾公司，X：」，模型就会输出对 X 的理解。

那怎么知道神经网络看到「李宏毅老师」这几个字时，内心深处的理解是什么？

1. 把这几个字输入到 network 里，看看某一层输出的 representation 长什么样子；
2. 把这个 representation **置换**到上面那个 input stream 里——用李宏毅老师的 representation 替换掉 X 位置的 representation。

因为右边的输入，其实都是在解释「冒号前面的这个东西是什么」——也就是说，这个语言模型走的是一个「解释任务」。如果我们把左边语言模型输出的 representation 拿出来，放到右边语言模型某一层的 representation 里做替换，那我们就能知道左边输入的「李宏毅老师」到底什么意思——它就有可能会告诉你李宏毅老师的身份。

![](/images/llm-internal/image-20260816173715576.png)

这里我们有一个困惑：我前面要准备一些例子，这些例子会不会影响最终输出的结果？

没错，举的这些例子就是会影响最终输出的结果。不过这篇作者认为这是一个 **feature（特性），不是 bug（缺陷）**：你可以调整前面的例子，然后模型就会给你不同风格的解释。

![](/images/llm-internal/image-20260816173823156.png)

比如说，你现在的输入是「告诉我 X 相关的秘密」，可以把 X 相关的 representation 换成李宏毅老师的 representation——它就会回答一些跟李宏毅老师有关的秘密。**这就等于从不同的角度来解析同一个 representation。**

下面是引用 Patchscope 原始论文里举的一个例子。

把 _"Diana, Princess of Wales"_ 这个 sequence 输入给神经网络，来解析「看到这个 sequence 最后一个字的时候，representation 对语言模型来讲分别是什么」。

- 如果把第 1-2 层 layer 的 representation 拿出来，模型解析出来的就是 _a country in the United Kingdom_——模型的前面几层只看到了 "Wales"（威尔斯）这个地名，所以只联想到这个国家；
- 到了第 4 层，它显然读到了 "Princess of Wales"，模型开始解析「这是一个给皇室女性的头衔」；
- 到第 5 层，它就知道这个人是「威尔斯王子的妻子」；
- 到第 6 层，它才读到「戴安娜」这个字，最后输出戴安娜完整的信息。

![](/images/llm-internal/image-20260816174206143.png)

### 4.5 一个应用：让 Multi-hop 推理更准

上述解析方法，改变了我们对神经网络背后机制的理解，进而提出了新的想法。下面这篇文章想解析的是：对于一个 **multi-hop question（多跳问题）**，语言模型是怎么回答的？解析完之后，得出了一个方法，让语言模型在 multi-hop question 上可以做得更好。

multi-hop question 的例子：_"the spouse of the performer of Imagine is"_。这种 multi-hop question 里会包含三个 entity：

- $e_1$：明确出现在问题里面的实体，例子里就是 Imagine——它是一张专辑的名字；
- $e_2$："The performer of Imagine"，弹奏这张专辑的音乐人是谁？答案是 John Lennon；
- $e_3$：John Lennon 的配偶是谁？答案是 Yoko Ono。

模型看到这一串文字，要输出 "Yoko Ono"。

参考文献：[https://arxiv.org/abs/2406.12775](https://arxiv.org/abs/2406.12775)

![](/images/llm-internal/image-20260816175032594.png)

接下来的问题是：模型是怎么做这一连串解析的？它是怎么做这种需要多步推理的问题的？

直觉的想法是：模型读到 imagine 这个词之后，根据前面的关系 "The performer of Imagine"，先解析出答案是 John Lennon；知道答案是 John Lennon 之后，再经过 "the spouse of John Lennon" 这个片语，解析出最终答案是 Yoko Ono。

**模型真的是这样运作的吗？** 作者就用之前讲的 Patchscope 方法做了一下解析。

他们把 Imagine 位置的每一层都拿出来，看看会解析出什么内容。如果解析出来的内容里有 John Lennon，就把它记录下来。得到的结果是下图**蓝色的线**：横轴是 layer，纵轴是「第一次解析出 $e_2$ 的 layer」。

根据这张图可以发现：在比较前面的 layer，语言模型就可以根据 $e_1$ 解析出 $e_2$。那解析出 $e_2$ 之后，什么时候解析出 $e_3$ 呢？作者又去解析 "is" 这个字每一个 representation 对应的文字，如果有出现 $e_3$ 的内容就记录下来，得到的结果是**橙色的线**。多数情况会在第 20 到 25 层的 layer 解析出 $e_3$。

你可以感受到：语言模型会在比较靠前的 layer 先解析出 $e_2$，然后再在后面的 layer 解析出 $e_3$，最后给出最终的正确回答。

然后作者发现：有时候 Multi-hop Question 得不到正确答案，是因为 **$e_2$ 太晚被解析出来了**——因为 $e_3$ 必须要在 20 几层的时候被解出来，那 20 几层才有解析出最终答案的能力；如果今天 $e_2$ 太晚才被解析出来，超过了 20 层，那接下来在 $e_3$ 这个位置就来不及解析出 $e_3$ 了。

![](/images/llm-internal/image-20260816175831975.png)

怎么解决这个问题呢？他们有一个比较神奇的做法：**把后面几层的 representation 直接加到前面来，再重新跑一次**，就解决了这个问题。

既然只有中间某一层能够解析出 $e_3$，如果 $e_2$ 太晚被解析出来，那怎么办呢？就把后面的 layer 放到前面来，走过前面那几层，这样就可以把 $e_3$ 解析出来了。

这招有没有用呢？**这招居然是有用的。** 他们试了各种不同的模型，下图的表格里：Correct 代表「用这招之前模型本来就会答对的问题」，用了这招之后不会影响正确率；对于本来就不对的问题，用了这招之后大概会有 40%–60% 的正确率。

这个方法跟 **Reasoning（推理式输出）** 有点像：Reasoning 就是你的输出来不及解析完，就跑到下一个 time step 重新解析一次。

---

## 05 总结：四句话带走

到这里，「大模型内部运作机制」这个框架就走完了。给你留四句话带走：

1. **单个神经元**：很难单独解释。一个功能往往由一组神经元共同完成，不同任务还可以共用神经元（4096 个神经元就有 $2^{4096}$ 种启动组合）。
2. **一层神经元**：可以看作「功能向量」。找一大群「会拒绝」和「不会拒绝」的句子，把第 10 层的 representation 分别平均再相减，就能抽出「拒绝向量」；加进去模型就拒绝，减掉就不拒绝。谄媚向量、说真话向量、In-Context 向量都是同一套思路。用 SAE 甚至可以自动找出千万级别的功能向量。
3. **一群神经元**：需要「语言模型的模型」。把模型简化成「主词 → representation + 关系 → linear function」，只要 faithfulness 够高，模型上的结论就能搬到真实模型上做 Model Editing；更系统化的做法是 pruning 出 Circuit。
4. **让模型直接说**：思维是透明的。所有 layer 都跑在一条 residual stream 上，每一层只是往上「加点东西」；用 Logit Lens 可以把每一层的「想法」解析成 token——比如法语翻译成中文前，模型心里先经过了一道英文。

一句话总结：**大模型并不是完全的黑盒——从单个神经元，到一层功能向量，再到整个 residual stream，我们正在一点一点把它拆开。**

## 参考资料（汇总）

以下是本文涉及的全部论文与资料，按出现顺序整理：

1. 模型能力随规模变化（文章开头 MMLU 图）：[https://arxiv.org/pdf/2407.14561](https://arxiv.org/pdf/2407.14561)
2. Multimodal Neurons（川普神经元，Distill）：[https://distill.pub/2021/multimodal-neurons/](https://distill.pub/2021/multimodal-neurons/)
3. GPT-2 单复数神经元：[https://arxiv.org/abs/2405.02421](https://arxiv.org/abs/2405.02421)
4. Transformer Circuits 神经元查看器：[https://transformer-circuits.pub/2023/monosemantic-features/vis/a-neurons.html](https://transformer-circuits.pub/2023/monosemantic-features/vis/a-neurons.html)
5. 拒绝向量（Representation Engineering）：[https://arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)
6. 谄媚向量：[https://arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)
7. 说真话向量：[https://arxiv.org/abs/2402.17811](https://arxiv.org/abs/2402.17811)、[https://arxiv.org/abs/2306.03341](https://arxiv.org/abs/2306.03341)
8. In-Context Vector：[https://arxiv.org/abs/2310.15213](https://arxiv.org/abs/2310.15213)、[https://arxiv.org/pdf/2310.15916](https://arxiv.org/pdf/2310.15916)、[https://arxiv.org/abs/2311.06668](https://arxiv.org/abs/2311.06668)
9. Claude 3 Sonnet 功能向量（Scaling Monosemanticity）：[https://transformer-circuits.pub/2024/scaling-monosemanticity/](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
10. 抽取知识的机制（Beats Music）：[https://arxiv.org/abs/2304.14767](https://arxiv.org/abs/2304.14767)
11. 语言模型做数学：[https://arxiv.org/abs/2305.15054](https://arxiv.org/abs/2305.15054)
12. 抽取知识的模型（Linear Function）：[https://arxiv.org/abs/2308.09124](https://arxiv.org/abs/2308.09124)
13. IOI Circuit：[https://arxiv.org/abs/2211.00593](https://arxiv.org/abs/2211.00593)
14. Automated Circuit Discovery：[https://arxiv.org/abs/2304.14997](https://arxiv.org/abs/2304.14997)
15. Chinchilla Circuit Analysis：[https://arxiv.org/abs/2307.09458](https://arxiv.org/abs/2307.09458)
16. Attribution Patching：[https://arxiv.org/abs/2310.10348](https://arxiv.org/abs/2310.10348)
17. Sparse Feature Circuits：[https://arxiv.org/abs/2403.19647](https://arxiv.org/abs/2403.19647)
18. Knowledge Circuits：[https://arxiv.org/abs/2405.17969](https://arxiv.org/abs/2405.17969)
19. Residual Connection（ResNet）：[https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
20. Logit Lens：[https://arxiv.org/abs/2001.09309](https://arxiv.org/abs/2001.09309)
21. 语言模型回答首都问题（Reciprocal Rank）：[https://arxiv.org/pdf/2305.16130](https://arxiv.org/pdf/2305.16130)
22. Do Llamas Work in English（隐语言）：[https://arxiv.org/abs/2402.10588](https://arxiv.org/abs/2402.10588)
23. Feed-Forward Layers Are Key-Value Memories：[https://arxiv.org/abs/2012.14913](https://arxiv.org/abs/2012.14913)
24. $\boldsymbol{v}$ 对应概念（2022）：[https://arxiv.org/abs/2203.14680](https://arxiv.org/abs/2203.14680)
25. Knowledge Neurons：[https://arxiv.org/abs/2104.08696](https://arxiv.org/abs/2104.08696)
26. Patchscope：[https://arxiv.org/pdf/2401.06102](https://arxiv.org/pdf/2401.06102)
27. Multi-hop Question：[https://arxiv.org/abs/2406.12775](https://arxiv.org/abs/2406.12775)

Copyright Ownership: Kstheme, Contributors: Kstheme
