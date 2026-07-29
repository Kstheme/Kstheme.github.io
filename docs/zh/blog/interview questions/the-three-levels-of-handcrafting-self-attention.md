---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - 机器学习
tags:
  - attention
  - llm
title: 自注意力手工实现的三个境界
createTime: 2026/06/16 15:18:20
permalink: /zh/article/ozftpyls/
---

## 引言

自注意力机制的实现过程中有很多细节，不同的面试对自注意力的实现有不同的要求。因此，我们需要学习各种实现自注意力的方式，以便向面试官展示我们对自注意力细节的深入理解。

## 自注意力公式

$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

## 代码实现

### 第一境界：简化版本

```python
import math
import torch
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    def __init__(self, hidden_dim: int = 728) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # 初始化三个不同的线性层
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x 形状为: (batch_size, seq_len, hidden_dim)

        # 获取不同的 Q、K、V
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        # Q、K、V 形状: (batch_size, seq_len, hidden_dim)

        # (batch_size, seq_len, hidden_dim) * (batch_size, hidden_dim, seq_len) = (batch_size, seq_len, seq_len)
        attention_value = torch.matmul(
            Q, K.transpose(-1, -2)
        )

        # 计算注意力权重
        attention_weights = torch.softmax(attention_value / math.sqrt(self.hidden_dim), dim=-1)

        # 计算结果: (batch_size, seq_len, hidden_dim)
        output = torch.matmul(attention_weights, V)

        return output
```

第一境界相对简单，完全按照公式实现即可。

### 第二境界：效率优化

将 Q、K、V 矩阵合并后再拆分。

```python
class SelfAttentionV2(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)

    def forward(self, x):
        # x 形状: (batch_size, seq_len, hidden_dim)
        # QKV 形状: (batch_size, seq_len, hidden_dim * 3)
        QKV = self.proj(x)
        Q, K, V = torch.split(QKV, self.hidden_dim, dim=-1)
        attention_weight = torch.softmax(
            torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim), dim=-1
        )
        output = attention_weight @ V
        return output
```

### 第三境界：增加细节（面试风格实现）

除了公式之外，还有一些额外的细节：
- 添加 dropout
- 考虑到每个句子长度不同，需要添加注意力掩码（attention mask）
- 输出矩阵映射

```python
class SelfAttentionV3(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        # x 形状: (batch_size, seq_len, hidden_dim)
        QKV = self.proj(x)
        Q, K, V = torch.split(QKV, self.hidden_dim, dim=-1)

        attention_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim)

        # 如果 attention_mask 不为空，需要将被掩码的位置赋一个极小值 —— 这样应用 Softmax 后它们的值将为 0
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float("1e-20")
            )

        attention_weight = torch.softmax(
            attention_weight, dim=-1
        )

        # 应用 dropout
        attention_weight = self.attention_dropout(attention_weight)

        attention_result = attention_weight @ V

        output = self.output_proj(attention_result)

        return output
```

## 从 V1 到 V3 的核心优化脉络

1. **阶段一：工程效率优化（V1 -> V2）**
    - **优化点**：将 3 个独立的线性层合并为 1 个组合线性层，然后拆分 QKV 矩阵。
    - **核心逻辑**：数学上完全等价（仅是权重拼接），但减少了内核启动次数和内存碎片，同时提升了硬件并行效率（GPU 能更好地利用批量矩阵乘法算力）。
    - **价值**：从"教学级冗余实现"过渡到"工程高效实现"——无性能损失，仅有效率提升。

2. **阶段二：功能完整性优化（V2 -> V3）**
    - **优化点 1**：添加 attention_mask 支持
    - **解决问题**：适配实际场景（NLP 中的 batch padding、生成任务中的因果掩码），屏蔽无效位置的干扰。
    - **优化点 2**：添加注意力权重 Dropout
    - **解决问题**：正则化——防止模型过度依赖少数关键位置，缓解过拟合。
    - **优化点 3**：添加输出线性投影（output_proj）
    - **解决问题**：精炼注意力聚合后的特征，增强模型的表征能力，适应深层网络的堆叠。
    - **价值**：从"以效率为中心"转向"生产级工业功能完备"，覆盖批量训练和更好泛化等关键需求。
