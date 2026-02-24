# BERT 模型源码解析 - 第四章：BertSdpaSelfAttention 优化版

> 源码位置：transformers/models/bert/modeling_bert.py (第357-456行)

## 1. 开篇：为什么需要 SDPA？

在上一章中，我们了解了自注意力机制的完整实现。但是，标准的实现方式在计算效率上并非最优。

**PyTorch 2.0+ 提供了 `scaled_dot_product_attention` (SDPA) 函数**，它集成了多种优化：

1. **Flash Attention**：减少显存访问，大幅提升速度
2. **Memory-Efficient Attention**：降低显存占用
3. **Math Attention**：标准实现，作为后备

---

## 2. BertSdpaSelfAttention 类概览

```python
class BertSdpaSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")
```

**解析：**
- 继承自 `BertSelfAttention`，复用大部分逻辑
- `require_contiguous_qkv`：PyTorch 2.2.0 之前版本的兼容性处理

---

## 3. SDPA 的限制条件

```python
if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
    logger.warning_once(
        "BertSdpaSelfAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
        "non-absolute `position_embedding_type` or `output_attentions=True` or `head_mask`. Falling back to "
        "the manual attention implementation..."
    )
    return super().forward(...)  # 回退到手动实现
```

**SDPA 不支持的场景：**

| 场景 | 原因 | 回退方案 |
|------|------|----------|
| 非绝对位置编码 | SDPA 不支持相对位置 | 使用手动实现 |
| `output_attentions=True` | SDPA 不返回注意力权重 | 使用手动实现 |
| `head_mask` 不为空 | SDPA 不支持头掩码 | 使用手动实现 |

---

## 4. Query、Key、Value 的计算

```python
bsz, tgt_len, _ = hidden_states.size()

query_layer = self.transpose_for_scores(self.query(hidden_states))
# [batch, heads, seq_len, head_dim]

is_cross_attention = encoder_hidden_states is not None

current_states = encoder_hidden_states if is_cross_attention else hidden_states
attention_mask = encoder_attention_mask if is_cross_attention else attention_mask
```

**与父类的区别：**
- 代码更简洁，但逻辑相同
- 提前获取 `bsz` 和 `tgt_len`，用于后续 reshape

---

## 5. KV Cache 处理

```python
# 支持 prefix tuning
if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
    key_layer, value_layer = past_key_value
else:
    key_layer = self.transpose_for_scores(self.key(current_states))
    value_layer = self.transpose_for_scores(self.value(current_states))
    if past_key_value is not None and not is_cross_attention:
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
```

---

## 6. 连续内存处理（PyTorch 2.2 之前）

```python
# SDPA 在 torch==2.1.2 中使用非连续输入和自定义 attn_mask 时有 bug
# 参考: https://github.com/pytorch/pytorch/issues/112577
if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask is not None:
    query_layer = query_layer.contiguous()
    key_layer = key_layer.contiguous()
    value_layer = value_layer.contiguous()
```

**什么是 contiguous？**
- PyTorch 张量可能在内存中不连续（如转置后的张量）
- 某些 CUDA kernel 要求连续内存
- `.contiguous()` 会复制数据使其连续

---

## 7. 因果掩码优化

```python
# 通过 is_causal 参数触发 Flash Attention 或 Efficient kernels
# tgt_len > 1 是必要的，以匹配 AttentionMaskConverter.to_causal_4d 的行为
is_causal = (
    True if self.is_decoder and not is_cross_attention and attention_mask is None and tgt_len > 1 else False
)
```

**条件解析：**

| 条件 | 说明 |
|------|------|
| `self.is_decoder` | 必须是解码器 |
| `not is_cross_attention` | 不是交叉注意力 |
| `attention_mask is None` | 没有自定义掩码 |
| `tgt_len > 1` | 序列长度大于 1 |

**因果掩码示意：**
```
      [t1] [t2] [t3] [t4]
[t1]   ✓    ✗    ✗    ✗
[t2]   ✓    ✓    ✗    ✗
[t3]   ✓    ✓    ✓    ✗
[t4]   ✓    ✓    ✓    ✓
```

---

## 8. 核心：scaled_dot_product_attention

```python
attn_output = torch.nn.functional.scaled_dot_product_attention(
    query_layer,      # [batch, heads, q_len, head_dim]
    key_layer,        # [batch, heads, k_len, head_dim]
    value_layer,      # [batch, heads, v_len, head_dim]
    attn_mask=attention_mask,      # 可选的注意力掩码
    dropout_p=self.dropout_prob if self.training else 0.0,  # Dropout 概率
    is_causal=is_causal,           # 是否使用因果掩码
)
```

**这个函数内部做了什么？**
```python
# 等价于手动实现
def scaled_dot_product_attention(Q, K, V, attn_mask, dropout_p, is_causal):
    # 1. 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

    # 2. 应用掩码
    if is_causal:
        causal_mask = torch.triu(torch.ones(...), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
    if attn_mask is not None:
        scores = scores + attn_mask

    # 3. Softmax
    attn_probs = torch.softmax(scores, dim=-1)

    # 4. Dropout
    attn_probs = torch.dropout(attn_probs, dropout_p, training)

    # 5. 加权求和
    output = torch.matmul(attn_probs, V)

    return output
```

**但 SDPA 融合了这些操作，更高效！**

---

## 9. 输出形状恢复

```python
attn_output = attn_output.transpose(1, 2)
# [batch, seq_len, heads, head_dim]

attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
# [batch, seq_len, 768]
```

---

## 10. 返回值

```python
outputs = (attn_output,)
if self.is_decoder:
    outputs = outputs + (past_key_value,)
return outputs
```

---

## 11. SDPA vs 手动实现对比

| 特性 | 手动实现 | SDPA |
|------|----------|------|
| 计算速度 | 基准 | 快 2-4 倍 |
| 显存占用 | 基准 | 减少 50%+ |
| 返回注意力权重 | ✓ | ✗ |
| 相对位置编码 | ✓ | ✗ |
| 头掩码 | ✓ | ✗ |
| Flash Attention | ✗ | ✓ |

---

## 12. 性能对比图

```
显存占用 (相对值)
│
│  ┌────────────┐
│  │  手动实现   │ ████████████████████ 100%
│  └────────────┘
│  ┌────────────┐
│  │   SDPA     │ ████████████ 60%
│  └────────────┘
└──────────────────→

推理速度 (tokens/sec)
│
│  ┌────────────┐
│  │   SDPA     │ ████████████████████████████ 300%
│  └────────────┘
│  ┌────────────┐
│  │  手动实现   │ ██████████ 100%
│  └────────────┘
└──────────────────→
```

---

## 13. 本章小结

BertSdpaSelfAttention 是对标准自注意力的优化实现：

1. **继承复用**：大部分逻辑与父类相同
2. **SDPA 加速**：利用 PyTorch 2.0 的融合注意力 kernel
3. **智能回退**：不支持的场景自动切换到手动实现
4. **因果掩码**：通过 `is_causal` 触发进一步优化
5. **版本兼容**：处理 PyTorch 2.2 之前的 bug

下一章，我们将介绍 **BertSelfOutput 和 BertAttention**——完成注意力模块的剩余部分。

---

*下一章预告：[bert_modeling_guide_05_attention_output.md] - 注意力输出的后处理*
