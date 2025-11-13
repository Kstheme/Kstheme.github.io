---
author: Kstheme
date: 2025-11-10
category:
    - Machine Learning
tag:
    - attention
    - llm
---
# The three levels of handcrafting self-attention

## Introduction
There are many details in the implementation process of self-attention, and different interviews have different requirements for the implementation of self-attention. So, we need to learn various ways to implement self-attention, so as to tell the interviewer that we understand the details of self-attention.

## The formula of self-attention
$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

## Code Implementation

### First Realm: Simplified Version

```python :line-numbers
import math
import torch
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    def __init__(self, hidden_dim: int = 728) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Intialize three different linear application layers
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x shape is: (batch_size, seq_len, hidden_dim)

        # acquire different Q, K, V
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        # Q, K, V shape: (batch_size, seq_len, hidden_dim)

        # (batch_size, seq_len, hidden_dim) * (batch_size, hidden_dim, seq_len) = (batch_size, seq_len, seq_len)
        attention_value = torch.matmul(
            Q, K.transpose(-1, -2)
        )

        # calculate attention weights
        attention_weights = torch.softmax(attention_value / math.sqrt(self.hidden_dim), dim=-1)

        # result of the calculation: (batch_size, seq_len, hidden_dim)
        output = torch.matmul(attention_weights, V)

        return output
```

The first realm is relatively simple, you can implement it entirely by following the formula.

### Second Realm: Efficiency Optimization

Combine the Q, K, V martices and then split them.

```python: line-numbers
class SelfAttentionV2(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        # QKV shape: (batch_size, seq_len, hidden_dim * 3)
        QKV = self.proj(x)
        Q, K, V = torch.split(QKV, self.hidden_dim, dim=-1)
        attention_weight = torch.softmax(
            torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim), dim=-1
        )
        output = attention_weight @ V
        return output
```

### Third Realm: add some details (interview-style implementation)

In addition to the formula, there are some additional details:
- add dropout
- given that each sentence has a distinct length, it is necessary to add an attention mask
- output martix mapping

```python: line-numbers
class SelfAttentionV3(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        # x shape: (batch_size, seq_len, hidden_dim)
        QKV = self.proj(x)
        Q, K, V = torch.split(QKV, self.hidden_dim, dim=-1)

        attention_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim)

        # if attention_mask is not None, we need to assign an extremely small value to the masked tokens —— this way, their value will be 0 after applying Softmax.
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float("1e-20")
            )

        attention_weight = torch.softmax(
            attention_weight, dim=-1
        )

        # applying dropout
        attention_weight = self.attention_dropout(attention_weight)

        attention_result = attention_weight @ V

        output = self.output_proj(attention_result)

        return output
```

## The core optimization context (iteration logic) from V1 to V3
1. **Phase 1: Engineering efficiency optimization (V1 -> V2)**
    - **Optimization Point**: Merge 3 separate linear layers into 1 combined linear layer, then split QKV matrix.
    - **Core Logic**: Mathematically completely equivalent (only weight concatenation), but reduces kernel launch times and memory fragmentation, while improving hardware parallel efficiency (GPUs can better utilize batch matrix multiplication computing power)
    - **Value**: Transition from a "teaching-level redundant implementation" to "engineering-efficient implementation" —— no performance loss, only efficiency improvement.
2. **Phase 2: Functionlity completeness optimization (V2 -> V3)**
    - **Optimization Point 1**: add attention_mask support
    - **Problem Solved**: Adapt to pratical scenarios (batch padding in NLP, causal masking for generation tasks) and shield against interference from invalid positions.
    - **Optimization Point 2**: add Dropout for attention weights
    - **Problem Solved**: Regularization —— prevent the model from over-relying on a few key positions and alleviate overfitting.
    - **Optimization Point 3**: Add output linear projection (output_proj).
    - **Problem Solved**: Refine the feature aggregated by attention, enhance the model's representational power, and adapt to the stacking of deep networks.
    - **Value**: Shift from "effieciency-centric" to "production-ready industrial-grade functionality", covering key needs like batch training and better generalization.
