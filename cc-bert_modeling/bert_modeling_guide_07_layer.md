# BERT 模型源码解析 - 第七章：BertLayer（完整 Transformer 层）

> 源码位置：transformers/models/bert/modeling_bert.py (第558-642行)

## 1. 开篇：Transformer 层的完整组装

现在我们已经了解了所有组件：

- **BertEmbeddings**：输入嵌入
- **BertSelfAttention**：自注意力机制
- **BertSelfOutput**：注意力后处理
- **BertIntermediate**：FFN 升维层
- **BertOutput**：FFN 降维层

本章我们将它们组装成一个**完整的 Transformer 层**。

---

## 2. BertLayer 类概览

```python
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention

        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
```

### 2.1 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size_feed_forward` | 0 | FFN 分块大小（0 表示不分块） |
| `is_decoder` | False | 是否作为解码器 |
| `add_cross_attention` | False | 是否添加交叉注意力 |

### 2.2 组件结构

**作为编码器（BERT 默认）：**
```
BertLayer
├── attention (BertAttention)
│   ├── self (BertSelfAttention)
│   └── output (BertSelfOutput)
├── intermediate (BertIntermediate)
└── output (BertOutput)
```

**作为解码器（带交叉注意力）：**
```
BertLayer
├── attention (BertAttention)         ← 自注意力
├── crossattention (BertAttention)    ← 交叉注意力（新增）
├── intermediate (BertIntermediate)
└── output (BertOutput)
```

---

## 3. 前向传播详解

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
```

### 3.1 参数说明

| 参数 | 形状 | 说明 |
|------|------|------|
| `hidden_states` | [batch, seq, 768] | 输入隐藏状态 |
| `attention_mask` | [batch, 1, 1, seq] | 自注意力掩码 |
| `encoder_hidden_states` | [batch, enc_seq, 768] | 编码器输出（交叉注意力用） |
| `past_key_value` | tuple | KV 缓存 |

### 3.2 自注意力计算

```python
# 解码器的 KV 缓存位于 past_key_value 的位置 1, 2
self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

self_attention_outputs = self.attention(
    hidden_states,
    attention_mask,
    head_mask,
    output_attentions=output_attentions,
    past_key_value=self_attn_past_key_value,
)
attention_output = self_attention_outputs[0]
```

### 3.3 处理自注意力输出

```python
if self.is_decoder:
    outputs = self_attention_outputs[1:-1]  # 中间的注意力权重
    present_key_value = self_attention_outputs[-1]  # 最后的 KV 缓存
else:
    outputs = self_attention_outputs[1:]  # 只有注意力权重（如果有的话）
```

**输出元组结构：**

| 场景 | `self_attention_outputs` |
|------|--------------------------|
| 编码器 | `(attention_output, [attention_probs])` |
| 解码器 | `(attention_output, [attention_probs], past_key_value)` |

### 3.4 交叉注意力（仅解码器）

```python
cross_attn_present_key_value = None

if self.is_decoder and encoder_hidden_states is not None:
    if not hasattr(self, "crossattention"):
        raise ValueError(
            f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
            " by setting `config.add_cross_attention=True`"
        )

    # 交叉注意力的 KV 缓存位于 past_key_value 的位置 3, 4
    cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

    cross_attention_outputs = self.crossattention(
        attention_output,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        cross_attn_past_key_value,
        output_attentions,
    )
    attention_output = cross_attention_outputs[0]
    outputs = outputs + cross_attention_outputs[1:-1]

    # 将交叉注意力的 KV 缓存添加到位置 3, 4
    cross_attn_present_key_value = cross_attention_outputs[-1]
    present_key_value = present_key_value + cross_attn_present_key_value
```

**past_key_value 结构：**
```
past_key_value = (
    self_attn_key,      # 位置 0
    self_attn_value,    # 位置 1
    cross_attn_key,     # 位置 2
    cross_attn_value,   # 位置 3
)
```

### 3.5 前馈网络（FFN）

```python
layer_output = apply_chunking_to_forward(
    self.feed_forward_chunk,
    self.chunk_size_feed_forward,
    self.seq_len_dim,
    attention_output
)
outputs = (layer_output,) + outputs
```

### 3.6 feed_forward_chunk 方法

```python
def feed_forward_chunk(self, attention_output):
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output
```

---

## 4. 分块前向传播（Chunking）

```python
layer_output = apply_chunking_to_forward(
    self.feed_forward_chunk,
    self.chunk_size_feed_forward,  # 分块大小
    self.seq_len_dim,              # 在哪个维度分块（1 = 序列维度）
    attention_output
)
```

**作用：**
- 当序列很长时，FFN 的中间层会占用大量显存
- 分块处理可以减少峰值显存使用
- 以计算换显存

**分块示意：**
```
假设 chunk_size = 128, seq_len = 512

原始: 一次处理 [batch, 512, 768] → [batch, 512, 3072] → [batch, 512, 768]

分块: 分 4 次处理
  - [batch, 0:128, 768] → ... → [batch, 0:128, 768]
  - [batch, 128:256, 768] → ... → [batch, 128:256, 768]
  - [batch, 256:384, 768] → ... → [batch, 256:384, 768]
  - [batch, 384:512, 768] → ... → [batch, 384:512, 768]

拼接: [batch, 512, 768]
```

---

## 5. 完整数据流图

### 5.1 编码器模式

```
hidden_states [batch, seq, 768]
         │
         ↓
┌─────────────────────────────────────┐
│         BertAttention               │
│  (Self-Attention + Residual + LN)   │
└─────────────────────────────────────┘
         │
         ↓
attention_output [batch, seq, 768]
         │
         ↓
┌─────────────────────────────────────┐
│         BertIntermediate            │
│  (Linear 768→3072 + GELU)           │
└─────────────────────────────────────┘
         │
         ↓
intermediate_output [batch, seq, 3072]
         │
         ↓
┌─────────────────────────────────────┐
│         BertOutput                  │
│  (Linear 3072→768 + Residual + LN)  │
└─────────────────────────────────────┘
         │
         ↓
layer_output [batch, seq, 768]
```

### 5.2 解码器模式（带交叉注意力）

```
hidden_states [batch, seq, 768]
         │
         ↓
┌─────────────────────────────────────┐
│         Self-Attention              │  ← past_key_value[:2]
│  (带因果掩码的自注意力)              │
└─────────────────────────────────────┘
         │
         ↓
         ├────────────────────────────┐
         │                            │
         ↓                            ↓
┌─────────────────────────────────────┐
│       Cross-Attention               │  ← encoder_hidden_states
│  (Query来自解码器, K/V来自编码器)    │  ← past_key_value[2:]
└─────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────┐
│         FFN                         │
│  (Intermediate + Output)            │
└─────────────────────────────────────┘
         │
         ↓
layer_output [batch, seq, 768]
```

---

## 6. 输出元组结构

### 6.1 编码器模式

| `output_attentions` | 输出元组 |
|---------------------|----------|
| False | `(layer_output,)` |
| True | `(layer_output, self_attn_probs)` |

### 6.2 解码器模式

| `output_attentions` | 输出元组 |
|---------------------|----------|
| False | `(layer_output, present_key_value)` |
| True | `(layer_output, self_attn_probs, cross_attn_probs, present_key_value)` |

---

## 7. 本章小结

本章介绍了完整的 Transformer 层：

1. **BertLayer 组成**：
   - 自注意力（必需）
   - 交叉注意力（仅解码器）
   - 前馈网络（升维 + 降维）

2. **关键设计**：
   - 残差连接：每个子层后都有
   - LayerNorm：每个子层后都有
   - KV 缓存：支持增量生成

3. **分块处理**：
   - 支持长序列的显存优化
   - 通过 `chunk_size_feed_forward` 配置

下一章，我们将介绍 **BertEncoder**——将 12 个 Transformer 层堆叠起来。

---

*下一章预告：[bert_modeling_guide_08_encoder.md] - 堆叠 12 层 Transformer*
