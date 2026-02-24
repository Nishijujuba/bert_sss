# BERT 模型源码解析 - 第五章：BertSelfOutput 与 BertAttention

> 源码位置：transformers/models/bert/modeling_bert.py (第458-527行)

## 1. 开篇：注意力的"收尾工作"

自注意力计算完成后，输出还需要经过一些后处理步骤才能传递到下一层。本章我们将介绍：

1. **BertSelfOutput**：注意力输出的后处理
2. **BertAttention**：整合自注意力和输出层的完整模块

---

## 2. BertSelfOutput 类详解

```python
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
```

### 2.1 组件说明

| 组件 | 维度 | 作用 |
|------|------|------|
| `dense` | 768 → 768 | 线性变换 |
| `LayerNorm` | 768 | 层归一化 |
| `dropout` | - | 正则化 |

### 2.2 前向传播

```python
def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states
```

**数据流图：**
```
attention_output [batch, seq, 768]
         │
         ↓
      dense (768 → 768)
         │
         ↓
      dropout
         │
         ↓
    ┌────┴────┐
    │   +     │ ← input_tensor (残差连接)
    └────┬────┘
         │
         ↓
    LayerNorm
         │
         ↓
    output [batch, seq, 768]
```

### 2.3 残差连接（Residual Connection）

```python
hidden_states = self.LayerNorm(hidden_states + input_tensor)
```

**残差连接的作用：**
- 缓解梯度消失问题
- 允许梯度直接流向底层
- 使深层网络更容易训练

**公式：**
$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

---

## 3. 注意力实现选择器

```python
BERT_SELF_ATTENTION_CLASSES = {
    "eager": BertSelfAttention,      # 手动实现
    "sdpa": BertSdpaSelfAttention,   # SDPA 优化实现
}
```

**使用方式：**
```python
# 在 BertAttention 中根据配置选择
self.self = BERT_SELF_ATTENTION_CLASSES[config._attn_implementation](config, ...)
```

---

## 4. BertAttention 类详解

```python
class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BERT_SELF_ATTENTION_CLASSES[config._attn_implementation](
            config, position_embedding_type=position_embedding_type
        )
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()
```

### 4.1 组件结构

```
BertAttention
├── self (BertSelfAttention 或 BertSdpaSelfAttention)
│   ├── query (Linear)
│   ├── key (Linear)
│   ├── value (Linear)
│   └── dropout (Dropout)
└── output (BertSelfOutput)
    ├── dense (Linear)
    ├── LayerNorm
    └── dropout
```

### 4.2 注意力头剪枝（Pruning）

```python
def prune_heads(self, heads):
    if len(heads) == 0:
        return

    heads, index = find_pruneable_heads_and_indices(
        heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
    )

    # 剪枝线性层
    self.self.query = prune_linear_layer(self.self.query, index)
    self.self.key = prune_linear_layer(self.self.key, index)
    self.self.value = prune_linear_layer(self.self.value, index)
    self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

    # 更新超参数
    self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    self.pruned_heads = self.pruned_heads.union(heads)
```

**剪枝的目的：**
- 移除不重要的注意力头
- 减少模型参数和计算量
- 适应边缘设备部署

**剪枝示意：**
```
原始 12 个头:
[0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11]

剪枝第 2, 5, 8 个头后:
[0] [1] [3] [4] [6] [7] [9] [10] [11]  (9 个头)
```

### 4.3 前向传播

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor]:
    # 1. 自注意力计算
    self_outputs = self.self(
        hidden_states,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_value,
        output_attentions,
    )

    # 2. 输出后处理（含残差连接）
    attention_output = self.output(self_outputs[0], hidden_states)

    # 3. 组装输出
    outputs = (attention_output,) + self_outputs[1:]  # 添加注意力权重（如果有）
    return outputs
```

---

## 5. 完整数据流

```
输入 hidden_states [batch, seq, 768]
         │
         ↓
┌─────────────────────────────────────┐
│         BertSelfAttention           │
│  ┌─────────────────────────────┐   │
│  │ Q, K, V = Linear(hidden)    │   │
│  │ scores = Q @ K^T / √d       │   │
│  │ probs = softmax(scores)     │   │
│  │ output = probs @ V          │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
         │
         ↓
    self_output [batch, seq, 768]
         │
         ↓
┌─────────────────────────────────────┐
│         BertSelfOutput              │
│  ┌─────────────────────────────┐   │
│  │ x = dense(self_output)      │   │
│  │ x = dropout(x)              │   │
│  │ x = LayerNorm(x + residual) │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
         │
         ↓
输出 attention_output [batch, seq, 768]
```

---

## 6. 输出元组结构

| 场景 | 输出内容 |
|------|----------|
| 普通编码器 | `(attention_output,)` |
| `output_attentions=True` | `(attention_output, attention_probs)` |
| 解码器 | `(attention_output, past_key_value)` |
| 解码器 + 输出注意力 | `(attention_output, attention_probs, past_key_value)` |

---

## 7. 本章小结

本章介绍了注意力模块的"收尾工作"：

1. **BertSelfOutput**：
   - 线性变换
   - Dropout 正则化
   - 残差连接 + LayerNorm

2. **BertAttention**：
   - 整合自注意力和输出层
   - 支持注意力头剪枝
   - 灵活选择 eager 或 sdpa 实现

3. **设计模式**：
   - 残差连接：缓解梯度消失
   - LayerNorm：稳定训练

下一章，我们将介绍 **BertIntermediate 和 BertOutput**——前馈网络（FFN）的实现。

---

*下一章预告：[bert_modeling_guide_06_ffn.md] - 前馈网络的升维与降维*
