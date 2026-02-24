# BERT 模型源码解析 - 第三章：BertSelfAttention 自注意力机制

> 源码位置：transformers/models/bert/modeling_bert.py (第223-355行)

## 1. 开篇：注意力的本质

自注意力（Self-Attention）是 Transformer 架构的核心创新。它让模型能够"关注"输入序列中的不同位置，捕捉词与词之间的依赖关系。

**核心公式：**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

---

## 2. BertSelfAttention 类概览

```python
class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
```

---

## 3. 初始化函数详解

### 3.1 维度验证

```python
if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
    raise ValueError(
        f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
        f"heads ({config.num_attention_heads})"
    )
```

**解析：**
- 隐藏层维度必须能被注意力头数整除
- BERT-base：768 ÷ 12 = 64（每个头的维度）
- BERT-large：1024 ÷ 16 = 64

### 3.2 注意力头配置

```python
self.num_attention_heads = config.num_attention_heads        # 注意力头数量（12）
self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 每个头的维度（64）
self.all_head_size = self.num_attention_heads * self.attention_head_size  # 总维度（768）
```

**多头注意力示意：**
```
输入 [batch, seq_len, 768]
         ↓
┌────────┬────────┬────────┬────────┐
│ Head 1 │ Head 2 │  ...   │ Head 12│
│  64维  │  64维  │  ...   │  64维  │
└────────┴────────┴────────┴────────┘
         ↓
拼接 → [batch, seq_len, 768]
```

### 3.3 Query、Key、Value 线性投影层

```python
self.query = nn.Linear(config.hidden_size, self.all_head_size)  # Q 投影
self.key = nn.Linear(config.hidden_size, self.all_head_size)    # K 投影
self.value = nn.Linear(config.hidden_size, self.all_head_size)  # V 投影
```

**解析：**
- 三个独立的线性层，将输入投影到 Q、K、V 空间
- 输入：[batch, seq_len, 768]
- 输出：[batch, seq_len, 768]

### 3.4 Dropout 与位置编码类型

```python
self.dropout = nn.Dropout(config.attention_probs_dropout_prob)  # 通常 0.1

self.position_embedding_type = position_embedding_type or getattr(
    config, "position_embedding_type", "absolute"
)
```

**位置编码类型：**
- `"absolute"`：绝对位置编码（BERT 默认）
- `"relative_key"`：相对位置编码（作用于 Key）
- `"relative_key_query"`：相对位置编码（作用于 Key 和 Query）

### 3.5 相对位置编码（可选）

```python
if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
    self.max_position_embeddings = config.max_position_embeddings
    self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
```

**解析：**
- 相对位置编码：表示两个位置之间的距离
- 距离范围：[-511, +511]，共 1023 种可能
- 嵌入维度：64（与每个注意力头相同）

### 3.6 解码器标志

```python
self.is_decoder = config.is_decoder  # 是否作为解码器使用
```

---

## 4. 张量形状变换函数

```python
def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)
```

**变换过程：**
```
输入: [batch, seq_len, 768]
        ↓ view
     [batch, seq_len, 12, 64]
        ↓ permute(0, 2, 1, 3)
     [batch, 12, seq_len, 64]
```

**为什么这样变换？**
- 将不同注意力头分离开来
- 每个头独立计算注意力分数

---

## 5. 前向传播函数详解

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

### 5.1 参数说明

| 参数 | 形状 | 说明 |
|------|------|------|
| `hidden_states` | [batch, seq, 768] | 输入隐藏状态 |
| `attention_mask` | [batch, 1, 1, seq] | 注意力掩码 |
| `head_mask` | [heads] 或 [layers, heads] | 掩码特定注意力头 |
| `encoder_hidden_states` | [batch, enc_seq, 768] | 编码器输出（交叉注意力） |
| `past_key_value` | (K, V) 缓存 | 用于增量生成 |
| `output_attentions` | bool | 是否返回注意力权重 |

### 5.2 Query 投影

```python
mixed_query_layer = self.query(hidden_states)  # [batch, seq_len, 768]
```

### 5.3 交叉注意力判断

```python
is_cross_attention = encoder_hidden_states is not None
```

**解析：**
- 如果提供了 `encoder_hidden_states`，则是交叉注意力
- 用于 encoder-decoder 架构（如 BART、T5）

### 5.4 Key 和 Value 的计算

```python
if is_cross_attention and past_key_value is not None:
    # 复用缓存的 K, V（交叉注意力）
    key_layer = past_key_value[0]
    value_layer = past_key_value[1]
    attention_mask = encoder_attention_mask

elif is_cross_attention:
    # 首次交叉注意力：从编码器输出计算 K, V
    key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
    value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
    attention_mask = encoder_attention_mask

elif past_key_value is not None:
    # 自回归解码：拼接过去的 K, V
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
    value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

else:
    # 普通自注意力
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
```

**KV Cache 示意图：**
```
时间步 1: K=[k1], V=[v1]
时间步 2: K=[k1, k2], V=[v1, v2]
时间步 3: K=[k1, k2, k3], V=[v1, v2, v3]
```

### 5.5 Query 形状变换

```python
query_layer = self.transpose_for_scores(mixed_query_layer)
# [batch, 12, seq_len, 64]
```

### 5.6 更新 KV Cache

```python
use_cache = past_key_value is not None
if self.is_decoder:
    past_key_value = (key_layer, value_layer)
```

### 5.7 计算注意力分数

```python
# Q × K^T
attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
# [batch, 12, query_len, key_len]
```

### 5.8 相对位置编码（可选）

```python
if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
    query_length, key_length = query_layer.shape[2], key_layer.shape[2]

    if use_cache:
        position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
    else:
        position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)

    position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
    distance = position_ids_l - position_ids_r  # 相对距离矩阵

    positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
    positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

    if self.position_embedding_type == "relative_key":
        relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        attention_scores = attention_scores + relative_position_scores
    elif self.position_embedding_type == "relative_key_query":
        relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
        attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
```

**相对距离矩阵示例：**
```
Query 位置: [0, 1, 2]
Key   位置: [0, 1, 2]

distance = [[0-0, 0-1, 0-2],    [[ 0, -1, -2],
            [1-0, 1-1, 1-2],  =  [ 1,  0, -1],
            [2-0, 2-1, 2-2]]     [ 2,  1,  0]]
```

### 5.9 缩放与掩码

```python
# 缩放
attention_scores = attention_scores / math.sqrt(self.attention_head_size)
# 除以 sqrt(64) = 8

# 应用注意力掩码
if attention_mask is not None:
    attention_scores = attention_scores + attention_mask
```

**掩码的作用：**
- PAD 位置：attention_mask = -inf，softmax 后接近 0
- 有效位置：attention_mask = 0

### 5.10 Softmax 与 Dropout

```python
# 归一化为概率分布
attention_probs = nn.functional.softmax(attention_scores, dim=-1)

# Dropout
attention_probs = self.dropout(attention_probs)
```

### 5.11 头掩码（可选）

```python
if head_mask is not None:
    attention_probs = attention_probs * head_mask
```

**用途：** 剪枝特定注意力头

### 5.12 计算输出

```python
# 注意力权重 × Value
context_layer = torch.matmul(attention_probs, value_layer)
# [batch, 12, seq_len, 64]

# 恢复形状
context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
# [batch, seq_len, 12, 64]

new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
context_layer = context_layer.view(new_context_layer_shape)
# [batch, seq_len, 768]
```

### 5.13 返回值

```python
outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

if self.is_decoder:
    outputs = outputs + (past_key_value,)
return outputs
```

---

## 6. 完整数据流图

```
输入 hidden_states [batch, seq, 768]
         │
    ┌────┼────┐
    ↓    ↓    ↓
   Q    K    V   (三个线性投影)
    │    │    │
    ↓    ↓    ↓
transpose_for_scores
    │    │    │
    ↓    ↓    ↓
[batch, heads, seq, head_dim]
         │
         ↓
   Q × K^T / √d_k    (注意力分数)
         │
         ↓
   + attention_mask
         │
         ↓
      softmax        (注意力概率)
         │
         ↓
      dropout
         │
         ↓
   prob × V          (加权求和)
         │
         ↓
   reshape 恢复形状
         │
         ↓
输出 [batch, seq, 768]
```

---

## 7. 本章小结

BertSelfAttention 实现了多头自注意力机制：

1. **Q/K/V 投影**：三个线性层将输入投影到不同空间
2. **多头分离**：将 768 维拆分为 12 个 64 维的注意力头
3. **注意力计算**：Q × K^T / √d，缩放防止梯度消失
4. **掩码应用**：处理变长序列和因果约束
5. **KV Cache**：支持增量生成

下一章，我们将介绍 **BertSdpaSelfAttention**——利用 PyTorch 2.0 的优化实现。

---

*下一章预告：[bert_modeling_guide_04_sdpa.md] - SDPA 优化的注意力计算*
