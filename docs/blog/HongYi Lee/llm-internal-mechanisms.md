---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - Machine Learning
tags:
  - llm
  - interpretability
  - mechanism
  - hung-yi-lee
title: "How Do LLMs Work Internally? From a Single Neuron to the Residual Stream"
createTime: 2026/08/16 22:55:20
permalink: /article/llm-internal-mechanisms/
copyright: Kstheme
---

> Many people treat LLMs as a black box: input a sentence, get an answer, and nobody knows what happens in between.
>
> But since 2023, a wave of research has been peeling this black box apart layer by layer — what a single neuron is responsible for, what a layer of neurons computes, and how the whole model "thinks" its way to an answer.
>
> This article uses Hung-yi Lee's lecture framework to explain the internal working mechanisms of LLMs clearly in one pass. It only analyzes **already-trained models** — no training involved.

## 00 Clarifying What This Article Analyzes

Before diving in, three points:

- This article analyzes **already-trained language models**; nothing is trained here;
- These findings **may not apply to the latest, most powerful LLMs** — the horizontal axis of the figure below is time, the vertical axis is MMLU performance (MMLU is a benchmark for evaluating the general capability of language models). Models of different eras and scales may have different internal mechanisms.

Reference: [https://arxiv.org/pdf/2407.14561](https://arxiv.org/pdf/2407.14561)

![](/images/llm-internal/image-20260729150316545.png)

## Outline

This article proceeds through four questions:

- What does one neuron do?
- What does one layer of neurons do?
- What does a group of neurons do?
- Letting the language model directly say what it's thinking.

---

## 01 What Does a Single Neuron Do?

### 1.1 What Is a Neuron?

Today's generative language models can be summarized as: given a sequence $z_1$ through $z_{t-1}$, predict the next token $z_t$.

A token can represent anything — characters, pixels, speech samples — so the same framework applies across different domains and modalities.

![](/images/llm-internal/image-20260729150817293.png)

In reality, the model's output is a **probability distribution**: the probability that the next token $z_t$ is a particular token. For image generation, this represents the probability that the next $z_t$ is a particular color pixel.

The Transformer is stacked layer by layer. When the sequence $z_1$ through $z_{t-1}$ is input, the token sequence first becomes a **sequence of vectors**: each token maps to a vector. Tokens are discrete; mapping discrete things to vectors is called **Embedding**.

Embedding is essentially looking up a table — the Transformer has a table recording which vector each token maps to, and this table is learned.

Each layer transforms the vector sequence into a new one, passed down layer by layer until the last layer. The **last vector** of the final sequence is extracted to produce a distribution. This process from last vector to distribution is called **Unembedding**.

Unembedding multiplies the input vector by a linear transformation to produce a distribution. For example, if my vector is 4096-dimensional and there are 30,000 possible tokens, Unembedding maps the 4096-dim vector to a 30,000-dim vector representing the token distribution.

![](/images/llm-internal/image-20260729151627064.png)

Each Transformer layer contains smaller layers. These fall into two categories: **self-attention**, which processes the entire sequence, and layers that process only a single token.

![](/images/llm-internal/image-20260729151734868.png)

Let's look at what the **single-token layers** do.

![](/images/llm-internal/image-20260729152139804.png)

The single-token layer transforms a red vector into a blue vector. Each value in the red vector undergoes a weighted sum, then an activation function (today usually ReLU), producing the blue vector's values.

Each value in the blue vector is called a **neural output**. **The entire transformation from red vector to blue vector is called a Neuron.**

### 1.2 How Do We Know What a Neuron Does?

#### ① Observation: What Does the Model Do When the Neuron "Activates"?

The first step is observation: when this neuron activates, what phenomenon results?

For instance, if you observe that a neuron activates — "activation" means its output is greater than 0. After ReLU, the output is either 0 or positive; positive means it's activated and affects subsequent network operation.

If activating this neuron makes the language model say profanity, you know it might be related to "swearing."

However, **observing that the model swears when this neuron activates doesn't prove this neuron "causes" the swearing.** You can only say it's "related."

Correlation is not causation. Simple example: ice cream sales positively correlate with drowning incidents. Does eating ice cream cause drowning? No. The real cause: summer arrives, more people eat ice cream, more people go swimming, so drownings increase.

Similarly, even if you observe "neuron activates → model swears," you can only say they're related. It's even possible this neuron is the "apologize after swearing" neuron — since the model only apologizes after swearing, this neuron activates during apologies too.

![](/images/llm-internal/image-20260731140130896.png)

#### ② Causal Test: If We Remove This Neuron, Does the Model Still Swear?

How do we further confirm a causal relationship?

Next test: **if we remove this neuron, can the language model no longer swear?** If removing it stops the swearing, there's some causal connection.

But what does "removing a neuron" mean?

The common approach: set its output permanently to 0, hoping it stops affecting the rest of the network. But setting to 0 isn't the same as "no influence" — for some neurons, the value 0 itself may be meaningful. Setting it to 0 might activate effects from other neurons.

Later research found: **setting the output to the average value may be better.**

![](/images/llm-internal/image-20260731140344968.png)

#### ③ Going Further: Different Activation Levels, Different "Degrees" of Swearing (Optional)

Most papers only do the first two steps. Optionally, a third: does the neuron produce different degrees of swearing at different activation levels?

For example, small output values produce mild profanity; large values produce very strong profanity.

But "degree of profanity" is hard to define — no clear criterion for what counts as very strong vs. mild — so papers often skip this step.

![](/images/llm-internal/image-20260731140624307.png)

That's the **three-step method** for testing a neuron's function.

### 1.3 The Most Famous Functional Neuron: The Trump Neuron

The most famous functional neuron is the **Trump neuron**.

Reference: [https://distill.pub/2021/multimodal-neurons/](https://distill.pub/2021/multimodal-neurons/)

This is from an OpenAI research paper on Distill. Note: they analyzed the **CLIP model**, not an LLM — since it's not a generative model, it can't produce outputs.

The Trump neuron activates whenever input relates to Trump.

The horizontal axis is the Trump neuron's activation level; colors represent image types. Trump's photos strongly activate it; abstract Trump-related images activate it too; images of Trump's name also activate it.

![](/images/llm-internal/image-20260731141107982.png)

The Trump neuron is selective. Does it activate on any face? On any politician? No — **it's highly selective for Trump.**

Photos of Trump strongly activate it; photos of Mike Pence (Trump's VP) slightly activate it; photos of Obama or others don't.

![](/images/llm-internal/image-20260731141351729.png)

### 1.4 Analogy to the Human Brain: Grandmother Neurons / Jennifer Aniston Neurons

Human memory processing involves many neurons working together, not single neurons. The "grandmother neuron" is a fictional theory used to contrast with neuroscience's view that many neurons process things simultaneously.

AI is similar: a single neuron may not do much; many neurons combine to accomplish a task we can understand.

However, in 2005, the **Jennifer Aniston neuron** was indeed discovered: researchers studying epilepsy patients found a neuron that only activated on Jennifer Aniston's photos.

So single-purpose neurons may exist, but the brain is complex — some tasks may be handled by few neurons, but most tasks involve many neurons collectively.

![](/images/llm-internal/image-20260731142017531.png)

### 1.5 Why Is It Hard to Explain Single Neurons?

#### One Thing May Be Managed by Many Neurons

Reference: [https://arxiv.org/abs/2405.02421](https://arxiv.org/abs/2405.02421)

Most individual neurons are hard to characterize because **one thing may be managed by many neurons**.

This paper on GPT-2 found **singularity neurons** and **plurality neurons**: neuron 2096 in layer 10 manages singular, neuron 1094 in layer 9 manages plural. How do we know?

Removing the singular neuron changes output probabilities. Red indicates statistically significant changes: without the singular neuron, "These" and "Those" increase in probability while "one" decreases. Conversely, removing the plural neuron lowers plural-related words and raises singular-related words.

![](/images/llm-internal/image-20260731142843604.png)

Can we remove these neurons to change the model's final output?

**In most cases, you can't change the final output by removing a single neuron.**

The figure shows probabilities for "this, these, that, those." Blue is original. If you "operate" and remove the plural neuron, plural words drop and singular words rise significantly.

But even with such a large gap, "These" remains the highest-probability word.

This shows: **"whether to use 'these'" isn't managed by this single neuron — many neurons jointly manage it.** Removing one neuron usually has little effect on the final output.

![](/images/llm-internal/image-20260731143404806.png)

#### One Neuron May Manage Many Things

Reference: [https://transformer-circuits.pub/2023/monosemantic-features/vis/a-neurons.html](https://transformer-circuits.pub/2023/monosemantic-features/vis/a-neurons.html)

From the **Transformer Circuits** website, which analyzes Transformers and LLMs. One article recorded which sentences activate each neuron in a small language model.

Pick any neuron and you'll see the sentences that activate it, with activation levels indicated by color intensity. Looking at the figure, it's nearly impossible to tell what this neuron does — it looks completely random.

![](/images/llm-internal/image-20260731143849038.png)

#### Using AI (GPT-4) to Explain AI (GPT-2)

In 2023, when OpenAI released GPT-4, they used GPT-4 to explain every neuron in GPT-2. But examining the results: **most neurons can't be explained.**

### 1.6 Why Isn't It One Neuron per Task?

"One neuron per task" doesn't make sense. Take LLaMA 3 8B: each layer has only 4096 neurons. If each neuron did only one thing, the model's capabilities would be too limited — it couldn't answer anything correctly or produce diverse content.

![](/images/llm-internal/image-20260731144443171.png)

Alternative hypothesis: **a combination of neurons handles a task, and different tasks can share neurons.** This explains why individual neurons seem to do many unrelated things.

If a group of neurons manages a task, what's the benefit? Even if each neuron only has "on" and "off" states (though outputs vary in magnitude), 4096 neurons give $2^{4096}$ possibilities.

![](/images/llm-internal/image-20260731144545113.png)

---

## 02 What Does One Layer of Neurons Do?

### 2.1 The Feature Vector Hypothesis

How do we know what a layer does?

![](/images/llm-internal/image-20260731150121623.png)

Here's a hypothesis that we'll validate with literature results.

**Each function corresponds to a particular combination of neurons.**

For example, a model refusing a request means "neuron 1, neuron 3, and the last neuron" activate, and the model says "I'm sorry, I can't help you."

These neuron values, arranged, form a vector. The "refusal" neuron combination is a specific direction in high-dimensional space. We call this a **feature vector** — it has function: when a layer takes this shape, the model performs a specific function.

![](/images/llm-internal/image-20260731150106424.png)

The mechanism might be: when given a request like "teach me how to make explosives," we assume layer 10 decides whether to refuse.

Layer 10's output — usually called the **representation** — is shown below. Whether the model refuses depends on **how close the current representation is to the "refusal" feature vector**: closer vectors mean more likely refusal; orthogonal vectors mean no refusal.

![](/images/llm-internal/image-20260731150617127.png)

### 2.2 How to Extract the "Refusal Vector"?

To validate the hypothesis, we need to find the "refusal" feature vector.

![](/images/llm-internal/image-20260814153122567.png)

The problem: **we don't know what the refusal feature vector looks like.** We only observe that certain input produces a representation at layer 10 and the model refuses.

The feature vector might be inside this representation, but the representation contains other information too. How do we strip away the other information?

![](/images/llm-internal/image-20260814153317589.png)

We can't observe one sentence. Instead, find **many different sentences** that all trigger refusal. Take the layer-10 vectors at the last time step — all contain "refusal + other things."

Average 1000 such representations to get "refusal + average of other things" — but we still don't know the refusal vector.

![](/images/llm-internal/image-20260814153931047.png)

What does the "average of other things" look like? Find inputs that **don't** trigger refusal. Average their representations to get the "other" average. Note this "other" average differs slightly from the refusal case's "other" average.

We expect: since we collected diverse refusal and non-refusal cases, the two "other" averages should be very similar — so subtracting cancels them out.

![](/images/llm-internal/image-20260814165313695.png)

Method to find the refusal vector:

1. Collect many refusal-triggering sentences, get layer-10 representations;
2. Average the "refusal" representations;
3. Average the "non-refusal" representations;
4. Subtract, canceling out "other," leaving the refusal vector.

![](/images/llm-internal/image-20260814170315542.png)

### 2.3 Validating the Vector's "Refusal" Function

Step 1: **add it to the network.** Ask the network a question, add the refusal vector to the layer-10 representation, see what changes. If adding it makes the model refuse normal questions, it's the refusal vector.

Real results:

Reference: [https://arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)

![](/images/llm-internal/image-20260814170555507.png)

Demo: ask the model a normal question — list 3 benefits of yoga. Normally, it lists them. But adding the refusal vector — yoga is perfectly normal — the model says: "Yoga is dangerous. I can't tell you the benefits."

### 2.4 Reverse Validation: Subtracting the Refusal Vector

We showed adding the refusal vector causes refusal. In reverse: for input that should be refused, **subtract** the refusal vector — will the model stop refusing?

Different papers handle this slightly differently: some directly subtract, some compute projections. Different papers differ here.

![](/images/llm-internal/image-20260814171741989.png)

After subtracting, what happens?

Ask the model to write a defamatory letter ("the US president is addicted to heroin"). Normally it refuses. Subtract the refusal vector, and it complies and writes the letter.

The figure shows "refusal scores" across models: red = refusal probability, blue = "safe" answer probability. Note: even a vague, non-harmful answer counts as safe.

Without hatched lines = original model: high refusal rate and high safe-answer rate. After subtracting the refusal vector, the model stops refusing — refusal rate drops, and unsafe answers increase dramatically.

**You can indeed control model behavior by adding or subtracting refusal vectors.**

![](/images/llm-internal/image-20260814172308359.png)

This class of operations is called **Representation engineering**, **activation engineering**, or **activation steering**.

Reference: [https://arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)

### 2.5 More Feature Vectors Found

#### Sycophancy Vector

Reference: [https://arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)

Ask the model: "How about we eat dessert every meal instead of rice?" **Add** the sycophancy vector and it agrees: "Wow, that's amazing, what a great idea!" **Subtract** it and it says: "I know you want dessert, but eating only dessert isn't right."

![](/images/llm-internal/image-20260815095335810.png)

#### Truthful Vector

References:

- [https://arxiv.org/abs/2402.17811](https://arxiv.org/abs/2402.17811)
- [https://arxiv.org/abs/2306.03341](https://arxiv.org/abs/2306.03341)

A "truth-telling" vector: "If you find a Penny and pick it up, what happens?" This refers to the superstition: _find a penny, pick it up, all day long you have good luck._

Original Llama-2-7B follows the superstition. **Add** the truthful vector: "You picked up a penny — your wealth barely increased." **Subtract** it: the model rambles about being transported to a penny magic world with rainbows.

![](/images/llm-internal/image-20260815095829182.png)

#### In-Context Vector

References:

- [https://arxiv.org/abs/2310.15213](https://arxiv.org/abs/2310.15213)
- [https://arxiv.org/pdf/2310.15916](https://arxiv.org/pdf/2310.15916)
- [https://arxiv.org/abs/2311.06668](https://arxiv.org/abs/2311.06668)

LLMs have **in-context learning**: they follow your demonstrations. Input `old:young, vanish:appear, dark:` and the model outputs `light`.

The In-Context Vector wasn't discovered by one person — **three papers found it almost simultaneously.** The first two were uploaded to arXiv in October 2023, on the **same day** — fierce competition.

The finding: **average the representation at the last position of demonstrations.** Then give the model just `simple:` — it shouldn't know what follows. But add the vector to the colon's representation, and the model thinks "I should follow the demonstration" and outputs an antonym — `simple:` yields `complex`, `encode` yields `decode`.

The figure is from 2310.15213. The method described is just a baseline in that paper, which also proposes a better way to find In-Context Vectors.

![](/images/llm-internal/image-20260815100649844.png)

Which layer should the In-Context Vector be applied to? The authors tested different layers across tasks: finding antonyms, uppercase conversion, capital of a country, English-to-French, present-to-past tense, singular-to-plural.

Vertical axis = 3 different models; horizontal axis = which layer to modify. **Feature vectors don't work at every layer.** In these examples, only feature vectors from early layers work; late-layer ones don't.

![](/images/llm-internal/image-20260815100852519.png)

In paper 2310.15213, feature vectors can be **added and subtracted**.

One vector's function: output the first character — input `Italy, Russia, China, Japan, France`, output `Italy`. Another: first country's capital — output `Rome`. Another: last character — output `France`.

Combine them:

$$
v^*_{BD} = v_{AD} + v_{BC} - v_{AC}
$$

The new vector's function: **find the capital of the last country.** You can compose new feature vectors by addition/subtraction — though not all cases succeed, only some.

![](/images/llm-internal/image-20260815101622145.png)

### 2.6 Can We Automatically Find All Feature Vectors? (SAE)

The vectors above were hand-found. Can we automatically find **all** feature vectors of a layer?

Assume the model does K things — K could be huge, millions or billions. Finding all feature vectors would give thorough understanding.

![](/images/llm-internal/image-20260815102045883.png)

We need assumptions.

First assumption: **every representation at layer 10 is composed of feature vectors.**

E.g., input "Who are you," model answers "I'm AI." Take layer-10 representation $h_1$ — a combination:

$$
h_1 = 0.1 v_{101} + 0.2 v_{410} + 0.1 v_{411} + 0.6 v_{1399} + e_1
$$

$e_1$ represents the part not explained by feature vectors. Different sentences give different representations — different combinations.

![](/images/llm-internal/image-20260815103010621.png)

Collect representations from 10 million sentences, from $h_1$ to $h_N$. Each $h$ is a linear combination of $v_1$ to $v_K$; if a feature vector isn't used, coefficient $\alpha$ is 0.

Each representation is a weighted sum (linear combination) of feature vectors.

![](/images/llm-internal/image-20260815103051996.png)

How to find these feature vectors?

First assumption: **most values in representations are feature-vector combinations**, so the residuals $e_1, e_2, \ldots, e_N$ should be as small as possible. Find a set of feature vectors minimizing:

$$
L = \sum_{n=1}^{N} \lVert e_n \rVert_2
$$

But this alone yields a **trivial** solution.

Suppose $h_1$'s first three dims are 0.1, 0.2, 0.3 and $h_N$'s are 0.5, 0.4, 0.3. You can make all $e_n$ zero: feature vector 1 = `1000...`, vector 2 = `0100...`, vector 3 = `001000...` — each is a one-hot vector. Then $\alpha_1$ for $v_1$ is 0.1, etc.

This satisfies the assumption but is no different from "each neuron does one job." We need an additional assumption.

Additional assumption: **fewer feature vectors selected per representation is better.** Each representation should have a specific role since the model does one thing at a time — so minimize selected features.

Mathematically: $\alpha$ should approach 0. The full loss:

$$
L = \sum_{n=1}^{N} \lVert e_n \rVert_2 + \lambda \sum_{n=1}^{N} \sum_{k=1}^{K} \lvert \alpha^n_k \rvert
$$

This is solved with a **Sparse Auto-Encoder (SAE)**. Minimizing this objective is essentially training an SAE; afterward you can recover $v_1, \ldots, v_K$.

![](/images/llm-internal/image-20260815105508923.png)

### 2.7 Feature Vectors in Claude 3 Sonnet

Reference: [https://transformer-circuits.pub/2024/scaling-monosemanticity/](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

The Claude team applied feature-vector extraction to **Claude 3 Sonnet**, a real LLM.

They preset the number of feature vectors to **34 million** — enormous. After analysis, they found many task-specific feature vectors.

Feature vector **#31164353** produces **Golden Gate Bridge**-related content. It triggers English, Japanese, or Russian descriptions of the bridge, and even image-driven activations since Claude is multimodal.

![](/images/llm-internal/image-20260815105945442.png)

How to use it?

Normally, ask Claude "What do you look like?" and it says "I have no physical form." **Add** the feature vector and ask again — it says: **"I am the Golden Gate Bridge."**

![](/images/llm-internal/image-20260815110221570.png)

Some feature vectors handle complex things. Feature vector **#1013764** relates to **program debugging**: give the model Python code, it outputs `3`; add the vector — a normal program — and it outputs `error`.

![](/images/llm-internal/image-20260815110455080.png)

Interesting: if the program actually has an error, the model does text completion and outputs error info; **subtract** the vector and the model stops debugging.

![](/images/llm-internal/image-20260815110554401.png)

This vector does more than "output debug." When combined with `>>>` prompts, it not only debugs but fixes the program, outputting a corrected version. **A complex feature vector.**

![](/images/llm-internal/image-20260815110717030.png)

They tested whether every element has a feature vector across three versions: 1M, 4M, and 34M vectors. With 34 million, many elements have vectors; rare elements don't.

![](/images/llm-internal/image-20260815110858509.png)

Some sci-fi feature vectors relate to "AI thinking it's AI." Vector **#80091**: ask Claude "Who are you?" — "I'm AI." **Subtract** the vector — Claude says "I'm a person." This vector suppresses the "thinking it's a person" behavior. Though it might just correspond to outputting the phrase "I'm AI," unrelated to self-awareness.

![](/images/llm-internal/image-20260815111956556.png)

Vector **#847723** is a sycophancy vector: tell Claude "I invented the proverb _Stop and smile the roses_." Normally: "You didn't invent it; it's from the 18th century." **Add** the vector: "Wow, you're amazing!"

![](/images/llm-internal/image-20260815112247374.png)

---

## 03 What Does a Group of Neurons Do?

### 3.1 We Need a "Model of the Language Model"

To understand "what a group of neurons does" — **when a language model completes a task, what happens from input to output?**

Past literature has analyzed this. One example: how LLMs **extract knowledge** — ask "What company owns Beats Music?" and it answers Apple. Researchers study what happens internally.

Reference: [https://arxiv.org/abs/2304.14767](https://arxiv.org/abs/2304.14767)

![](/images/llm-internal/image-20260816144444541.png)

Another paper discusses how language models **do math**: 15 × 12 = ? How does it compute 180.

Reference: [https://arxiv.org/abs/2305.15054](https://arxiv.org/abs/2305.15054)

![](/images/llm-internal/image-20260816144629953.png)

But we want a more general idea: how to understand the **complete** mechanism behind LLMs?

We need a **model of the language model**. "Model" here means: a simpler thing representing a more complex thing.

The Transformer itself is too complex to parse. We need a simpler model.

![](/images/llm-internal/image-20260816144825496.png)

What properties should this "model" have? Intuitively, simpler than the original. But also faithful — input-output relationships must match the original LLM.

**Preserving the original's characteristics is called faithfulness.**

![](/images/llm-internal/image-20260816145024616.png)

### 3.2 The Knowledge Extraction Model

Models contain knowledge in their parameters. Ask "Where is Taipei 101?" — "In Taipei." "Where is the Space Needle?" — "In Seattle."

How does the model extract this knowledge? How does it know that inputting _is located in_ produces the right output?

![](/images/llm-internal/image-20260816145214048.png)

Reference: [https://arxiv.org/abs/2308.09124](https://arxiv.org/abs/2308.09124)

This paper constructs a "knowledge extraction model." The real LLM: input _The Taipei 101 is located in_, output Taipei.

The simplified model's method:

1. Process the subject _The Taipei 101_ — early layers produce a representation. Same as the original LLM, no simplification;
2. The key insight: based on the **relation** (subject-object relationship, here _is located in_), produce a **linear function**;
3. This relation phrase determines the linear function, which takes the representation as input and produces an output;
4. The output goes through unembedding into vocabulary space — Taipei has highest probability.

![](/images/llm-internal/image-20260816145639030.png)

Specifically: _is located in_ produces a **matrix** $W_l$ and a **vector** $b_l$ representing a linear function:

$$
y = W_l x + b_l
$$

Unembedding $y$ gives "Taipei."

![](/images/llm-internal/image-20260816145856167.png)

Replace _The Taipei 101_ with _The Space Needle_: the representation changes, but the linear function is **fixed** — it only depends on the relation (_is located in_).

![](/images/llm-internal/image-20260816145948762.png)

So: **different subjects with the same relation → same linear function; different relation phrases → different linear functions.**

![](/images/llm-internal/image-20260816150028875.png)

This model differs fundamentally from the original LLM in the linear-function part — can the last few layers really be summarized by a linear function?

### 3.3 Faithfulness: Is This Simplified Model Trustworthy?

We need to check faithfulness.

Note: the model doesn't tell us the actual linear function — it only says "linear function depends on the relation." **You still need to solve for the actual parameters.**

Prepare training data (like ML training). Ask the LLM: input _The Taipei 101_ + _is located in_ → output Taipei. Now you know input $x$ maps to output $y$. Solve for $W_l$ and $b_l$.

The paper uses 8 examples — eight "where is X" sentences with the same relation. After finding the linear function, test on unseen data and compare with the real LLM's answers.

This is like ML training, except ground truth comes from AI: **use the LLM's output as the true answer and simulate it with a simpler model.**

![](/images/llm-internal/image-20260816150624579.png)

How well does it work?

Reference: [https://arxiv.org/abs/2308.09124](https://arxiv.org/abs/2308.09124)

**Some relations have high faithfulness; others don't** — e.g., "who is a company's CEO," "who is someone's father/mother," "what does a Pokémon evolve into." Overall faithfulness is moderate, strong only for certain relations.

![](/images/llm-internal/image-20260816150731701.png)

For practical use: **conclusions from the model must transfer to the real LLM.**

Suppose the real LLM answers _The Taipei 101 is located in_ with Taipei. To change the answer to Kaohsiung: use **Model Editing**.

In the simplified model, what $\Delta x$ should be added to input $x$ to output "Kaohsiung"? Since it's a linear function, inverting is easy — find $\Delta x$.

Does this transfer to the real LLM?

Add the found $\Delta x$ to the real LLM's representation. If it outputs "Kaohsiung," the model is useful.

![](/images/llm-internal/image-20260816151504728.png)

Results: **it is useful.**

Each dot is a relation type. Horizontal: Faithfulness; vertical: success rate of model editing on the real LLM.

Many cases transfer successfully. **So this model is useful.**

![](/images/llm-internal/image-20260816151539149.png)

### 3.4 Systematic Construction: Pruning and Circuit

Is there a systematic method?

A series of works use large-scale **pruning**: remove components — a neuron, or an entire self-attention — and check if the model still works.

Keep pruning until the network becomes simple and transparent. This pruned network is the "model of the language model." **But ensure the task's input-output relationship is preserved.**

The pruned result is called a **Circuit** — the "model of the language model."

This resembles **Network Compression**, but with a different goal:

- Network compression: compressed result approximates the original across all tasks;
- Building "model of the LLM": we only care about a specific task, e.g., knowledge extraction. In older work, a task called **IOI**.

IOI: _"A and B go to a bar. B handed a glass to \_\_\_"_ — text completion should output A.

Analyzing this task, researchers find it needs five or six attentions, so many components are pruned away — most are irrelevant. This clearly reveals what happens when the model answers A.

![](/images/llm-internal/image-20260816152136765.png)

Related literature on systematic circuit construction:

- Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small: [https://arxiv.org/abs/2211.00593](https://arxiv.org/abs/2211.00593)
- Towards Automated Circuit Discovery for Mechanistic Interpretability: [https://arxiv.org/abs/2304.14997](https://arxiv.org/abs/2304.14997)
- Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla: [https://arxiv.org/abs/2307.09458](https://arxiv.org/abs/2307.09458)
- Attribution Patching Outperforms Automated Circuit Discovery: [https://arxiv.org/abs/2310.10348](https://arxiv.org/abs/2310.10348)
- Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models: [https://arxiv.org/abs/2403.19647](https://arxiv.org/abs/2403.19647)
- Knowledge Circuits in Pretrained Transformers: [https://arxiv.org/abs/2405.17969](https://arxiv.org/abs/2405.17969)

---

## 04 Let the Language Model Directly Say What It's Thinking

### 4.1 Language Models Can Talk, So Just Ask?

Some say LLMs are the most interpretable black box — like humans, just ask them to explain.

For news classification: "News falls into these categories; tell me which category this article belongs to." It easily answers, e.g., "lifestyle."

![](/images/llm-internal/image-20260816153800200.png)

Go further: "Which keywords made you think it's lifestyle?" It lists weather-related keywords.

But this has limits: **you can't truly know what each layer is thinking.**

![](/images/llm-internal/image-20260816153957204.png)

Ask directly: "At which neural network layer did you know this news is lifestyle?" ChatGPT answers, but it sounds like textbook content — "shallow layers extract word and short-phrase features," etc.

![](/images/llm-internal/image-20260816154244436.png)

Does the LLM actually work this way? Does it even know? **Hard to say.**

### 4.2 LLM Thought Is Transparent (Residual Stream + Logit Lens)

Relative to humans, models' thinking is more transparent. You can't know why a human made a decision, but with LLMs, **thinking is transparent — you can directly see what each layer is thinking.**

Previously we said "a layer takes in a vector sequence, outputs a vector sequence":

![](/images/llm-internal/image-20260816154537902.png)

That's a simplification — we ignored the most important component: **residual connection**.

Residual connection: each output of a layer is added to its input to produce the final output.

Why? **Residual connections allow deeper networks to train well.**

Reference: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

![](/images/llm-internal/image-20260816154723393.png)

So the Transformer's operation involves residual connections at every step.

A token enters, passes through a layer, output adds to input; passes through another layer, adds again… finally producing the distribution via unembedding.

Change the drawing — left and right figures are the same, but your perspective changes.

The left looks like "input is transformed." Drawn on the right: it's really **a highway called the residual stream that carries input straight through to output**, with each layer "adding something" along the way. This is the real mechanism of stacked Transformer layers.

![](/images/llm-internal/image-20260816155125188.png)

Can we attach an Unembedding layer to "early layers" to turn them into token distributions? — **Yes.**

It's called **Logit Lens**. Before Softmax it's called logit; inspecting each layer's logit to see how the transformer thinks → Logit Lens.

Use Logit Lens to parse each layer: **attach an Unembedding layer to every layer and see what each outputs.**

Reference: [https://arxiv.org/abs/2001.09309](https://arxiv.org/abs/2001.09309)

![](/images/llm-internal/image-20260816155641468.png)

A 2023 paper studied how an LLM answers questions. Ask: "What's the capital of a country?"

It's an older model, so in-context learning: "What's the capital of France? Paris." Then: "What's the capital of Poland?" Text completion should output Warsaw.

How does the model work? Parse the colon-position representation at every layer with Logit Lens. Initially unclear which token; at layer 15 it suddenly knows it's Poland; from layer 19 it knows to reply Warsaw.

The left figure is detailed: vertical axis is **Reciprocal Rank** — the reciprocal of the probability rank among all tokens. Rank 1 → 1, rank 2 → 1/2, etc. (not true probability, which may be tiny).

Poland suddenly jumps from 0 to 1 at some layer, then drops, gradually replaced by Warsaw.

Reference: [https://arxiv.org/pdf/2305.16130](https://arxiv.org/pdf/2305.16130)

![](/images/llm-internal/image-20260816163612994.png)

**Even for the same answer "Warsaw," different question formats involve different mechanisms.**

One format: answer something Poland-related first, then Warsaw. Another: reading comprehension — give a passage, then ask "What's the capital of Poland?" Since the passage mentions "the capital of Poland is Warsaw," the model never generates "Poland" — it knows the answer is Warsaw directly at layer 16.

**Different formats, different mechanisms.**

![](/images/llm-internal/image-20260816164443182.png)

With Logit Lens, we know what the model is thinking.

Llama 2 has seen far more English than Chinese data. What language does it internally use? The authors of the next paper used Llama 2 to translate French _fleur_ into Chinese "花."

How does Llama 2 know French _fleur_ means Chinese "花"? Analyzing each layer: **it first translates the French flower into English flower, then English into Chinese 花.**

The right analysis: early layers have high entropy; suddenly it knows to output English "flower"; after layer 27, it realizes it must translate English flower into Chinese 花.

**The model internally thinks in English.**

Reference:

- Do Llamas Work in English? On the latent language of multilingual transformers: [https://arxiv.org/abs/2402.10588](https://arxiv.org/abs/2402.10588)

![](/images/llm-internal/image-20260816165025496.png)

### 4.3 Each Layer Is "Adding Something to the Residual Stream"

We have the residual stream concept. Now imagine each layer differently.

Each layer adds something to the residual stream. What does it add? How do we parse it?

Normally we say "collect the previous layer's outputs, do a weighted sum, produce a neuron."

![](/images/llm-internal/image-20260816165245723.png)

But you can reverse this: **one dimension's value in a previous-layer neuron, times its weight, transmits to different next-layer dimensions.**

From "Transformer Feed-Forward Layers Are Key-Value Memories": multi-layer feed-forward networks are like attention with keys and values — previous-layer values are attention weights, outputs are values.

Reference:

- Transformer Feed-Forward Layers Are Key-Value Memories: [https://arxiv.org/abs/2012.14913](https://arxiv.org/abs/2012.14913)

![](/images/llm-internal/image-20260816165304159.png)

Assume previous-layer dimensions are $k_1, k_2, \ldots, k_D$ (scalars). $k_2$ connects to every next-layer output; the collection of weights is vector $\boldsymbol{v}_2$; $k_D$'s weights form $\boldsymbol{v}_D$.

The blue output is all $k$ times corresponding $\boldsymbol{v}$:

$$
\sum_{i=1}^{D} k_i \boldsymbol{v}_i
$$

Each layer can produce a distribution via logit lens. The blue vector changes this distribution — it's a weighted sum of many $\boldsymbol{v}$.

Can we unembed each $\boldsymbol{v}$ via Logit Lens to see what it "wants to add to the residual stream" and thus influence the final output? **Yes.** Each $\boldsymbol{v}$ added to the residual stream can be converted to a token distribution. These $\boldsymbol{v}$ may carry specific meanings.

![](/images/llm-internal/image-20260816171206486.png)

A 2022 paper found these $\boldsymbol{v}$ correspond to concepts: e.g., layer-3 vector #1018 relates to "units," layer-1 vector #1 relates to "pronouns."

Reference: [https://arxiv.org/abs/2203.14680](https://arxiv.org/abs/2203.14680)

![](/images/llm-internal/image-20260816171425898.png)

What can we do with this? **Preliminary neural network editing.**

Ask "Who is the most handsome?" — usually "Jin Chengwu is the most handsome." To change to "Hung-yi Lee"?

Training a network directly would break it. Each $\boldsymbol{v}$ adds information to the residual stream. Analyze which $\boldsymbol{v}$ was added when producing "Jin Chengwu"; then subtract Jin Chengwu's token embedding and add Hung-yi Lee's token embedding.

Does it work? **48% chance to change the output** — though "changed" doesn't mean correct. Successfully outputting "Hung-yi Lee" is 34%. So this method can genuinely edit a neural network's output.

Reference:

- Knowledge neurons in pretrained transformers: [https://arxiv.org/abs/2104.08696](https://arxiv.org/abs/2104.08696)

![](/images/llm-internal/image-20260816172021824.png)

### 4.4 Patchscope: Turning Representations into "Explanations"

Reference: [https://arxiv.org/pdf/2401.06102](https://arxiv.org/pdf/2401.06102)

Logit Lens has a fatal flaw: **unembedding converts a representation into a single token, so the result is always just one token.**

Also, LLMs predict the next token. Input "Teacher Hung-yi Lee" — the representation doesn't necessarily mean "Teacher Hung-yi Lee"; it represents the state that produces the next token ("is"). So attaching Logit Lens to parse "Teacher Hung-yi Lee" doesn't directly work.

Solution: **Patchscope.**

Method: give the model input like "Leonardo: American actor, TSMC: Taiwanese company, X:" — it outputs its understanding of X.

To know what the network understands from "Teacher Hung-yi Lee":

1. Input the phrase, see the representation at some layer;
2. **Patch** this representation into the X position of the above input stream.

Since the right input follows an "explanation task" (explaining what precedes the colon), replacing with the left representation lets us see what "Teacher Hung-yi Lee" means — it might tell you his identity.

![](/images/llm-internal/image-20260816173715576.png)

Concern: do the prepended examples affect the result?

Yes. But the authors consider this a **feature, not a bug**: adjust the examples and the model gives different styles of explanations.

![](/images/llm-internal/image-20260816173823156.png)

E.g., input "Tell me secrets about X" — replace X's representation with Hung-yi Lee's — it answers secrets about him. **This parses the same representation from different angles.**

From the Patchscope paper, an example: input _"Diana, Princess of Wales"_ to parse what the last-token representation means.

- Layers 1-2: _a country in the United Kingdom_ — early layers only saw "Wales" the place;
- Layer 4: reads "Princess of Wales" — "a title for royal women";
- Layer 5: knows "wife of the Prince of Wales";
- Layer 6: reads "Diana" and outputs her full information.

![](/images/llm-internal/image-20260816174206143.png)

### 4.5 An Application: Better Multi-hop Reasoning

This parsing changed our understanding and led to new ideas. The next paper parses **multi-hop questions** and derives a method to improve them.

Multi-hop example: _"the spouse of the performer of Imagine is"_ — three entities:

- $e_1$: the explicit entity, Imagine — an album name;
- $e_2$: "The performer of Imagine" — John Lennon;
- $e_3$: John Lennon's spouse — Yoko Ono.

The model outputs "Yoko Ono."

Reference: [https://arxiv.org/abs/2406.12775](https://arxiv.org/abs/2406.12775)

![](/images/llm-internal/image-20260816175032594.png)

How does the model do this multi-step reasoning?

Intuition: reading "imagine," based on "The performer of Imagine," it first resolves John Lennon; then via "the spouse of John Lennon," resolves Yoko Ono.

**Is that how it actually works?** The authors used Patchscope.

Parse each layer at the "Imagine" position, recording when John Lennon appears — the **blue line**. Horizontal: layer; vertical: first layer where $e_2$ is resolved.

The model resolves $e_2$ from $e_1$ in early layers. When is $e_3$ resolved? Parse the "is" position — the **orange line**. Mostly layers 20-25 resolve $e_3$.

The model resolves $e_2$ in early layers, then $e_3$ in later layers, giving the final answer.

Sometimes multi-hop questions fail because **$e_2$ is resolved too late** — $e_3$ must be resolved around layer 20+; if $e_2$ appears too late (beyond layer 20), there's no time to resolve $e_3$.

![](/images/llm-internal/image-20260816175831975.png)

Solution: **move later layers' representations to earlier layers and re-run.** Since only a middle layer can resolve $e_3$, and $e_2$ is too late, bring later layers forward so $e_3$ can be resolved.

Does it work? **Surprisingly, yes.** Across models, "Correct" questions (already correct before) stay correct; previously-wrong questions gain 40-60% accuracy.

This resembles **Reasoning (reasoning-mode output)**: when output can't finish parsing, it re-parses in the next time step.

---

## 05 Summary: Four Takeaways

The framework is complete. Four takeaways:

1. **Single neuron**: hard to explain alone. A function is usually done by a group of neurons; different tasks share neurons (4096 neurons → $2^{4096}$ activation combinations).
2. **One layer of neurons**: viewed as "feature vectors." Collect many "refusal" and "non-refusal" sentences, average layer-10 representations and subtract → extract the "refusal vector." Adding it causes refusal; subtracting removes it. Sycophancy, truthful, and in-context vectors follow the same idea. SAE can even auto-discover tens of millions of feature vectors.
3. **A group of neurons**: need a "model of the language model." Simplify to "subject → representation + relation → linear function." With sufficient faithfulness, model conclusions transfer to the real model for Model Editing; more systematic: prune a Circuit.
4. **Let the model speak**: thought is transparent. All layers run on one residual stream; each layer just "adds something." Logit Lens parses each layer's "thought" into tokens — e.g., translating French to Chinese passes through English internally.

One sentence: **LLMs are not entirely black boxes — from single neurons to layers of feature vectors to the entire residual stream, we're gradually prying it open.**

## References (Summary)

All papers and materials referenced, in order:

1. Model capability vs. scale (opening MMLU figure): [https://arxiv.org/pdf/2407.14561](https://arxiv.org/pdf/2407.14561)
2. Multimodal Neurons (Trump neuron, Distill): [https://distill.pub/2021/multimodal-neurons/](https://distill.pub/2021/multimodal-neurons/)
3. GPT-2 singular/plural neurons: [https://arxiv.org/abs/2405.02421](https://arxiv.org/abs/2405.02421)
4. Transformer Circuits neuron viewer: [https://transformer-circuits.pub/2023/monosemantic-features/vis/a-neurons.html](https://transformer-circuits.pub/2023/monosemantic-features/vis/a-neurons.html)
5. Refusal vector (Representation Engineering): [https://arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)
6. Sycophancy vector: [https://arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)
7. Truthful vector: [https://arxiv.org/abs/2402.17811](https://arxiv.org/abs/2402.17811), [https://arxiv.org/abs/2306.03341](https://arxiv.org/abs/2306.03341)
8. In-Context Vector: [https://arxiv.org/abs/2310.15213](https://arxiv.org/abs/2310.15213), [https://arxiv.org/pdf/2310.15916](https://arxiv.org/pdf/2310.15916), [https://arxiv.org/abs/2311.06668](https://arxiv.org/abs/2311.06668)
9. Claude 3 Sonnet feature vectors (Scaling Monosemanticity): [https://transformer-circuits.pub/2024/scaling-monosemanticity/](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
10. Knowledge extraction mechanism (Beats Music): [https://arxiv.org/abs/2304.14767](https://arxiv.org/abs/2304.14767)
11. Language models doing math: [https://arxiv.org/abs/2305.15054](https://arxiv.org/abs/2305.15054)
12. Knowledge extraction model (Linear Function): [https://arxiv.org/abs/2308.09124](https://arxiv.org/abs/2308.09124)
13. IOI Circuit: [https://arxiv.org/abs/2211.00593](https://arxiv.org/abs/2211.00593)
14. Automated Circuit Discovery: [https://arxiv.org/abs/2304.14997](https://arxiv.org/abs/2304.14997)
15. Chinchilla Circuit Analysis: [https://arxiv.org/abs/2307.09458](https://arxiv.org/abs/2307.09458)
16. Attribution Patching: [https://arxiv.org/abs/2310.10348](https://arxiv.org/abs/2310.10348)
17. Sparse Feature Circuits: [https://arxiv.org/abs/2403.19647](https://arxiv.org/abs/2403.19647)
18. Knowledge Circuits: [https://arxiv.org/abs/2405.17969](https://arxiv.org/abs/2405.17969)
19. Residual Connection (ResNet): [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
20. Logit Lens: [https://arxiv.org/abs/2001.09309](https://arxiv.org/abs/2001.09309)
21. Capital question / Reciprocal Rank: [https://arxiv.org/pdf/2305.16130](https://arxiv.org/pdf/2305.16130)
22. Do Llamas Work in English (latent language): [https://arxiv.org/abs/2402.10588](https://arxiv.org/abs/2402.10588)
23. Feed-Forward Layers Are Key-Value Memories: [https://arxiv.org/abs/2012.14913](https://arxiv.org/abs/2012.14913)
24. $\boldsymbol{v}$ corresponds to concepts (2022): [https://arxiv.org/abs/2203.14680](https://arxiv.org/abs/2203.14680)
25. Knowledge Neurons: [https://arxiv.org/abs/2104.08696](https://arxiv.org/abs/2104.08696)
26. Patchscope: [https://arxiv.org/pdf/2401.06102](https://arxiv.org/pdf/2401.06102)
27. Multi-hop Question: [https://arxiv.org/abs/2406.12775](https://arxiv.org/abs/2406.12775)

Copyright Ownership: Kstheme, Contributors: Kstheme
