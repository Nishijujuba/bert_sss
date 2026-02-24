# BERT 模型源码解析 - 第八章：BertEncoder 编码器

> 源码位置：transformers/models/bert/modeling_bert.py (第644-735行)

## 1. 开篇：从一层到十二层

BERT-base 有 12 个 Transformer 层，BERT-large 有 24 层。BertEncoder 负责将这些层堆叠起来，并管理：

1. **层的堆叠**：按顺序执行每一层
2. **梯度检查点**：节省显存的训练技巧
3. **输出收集**：汇总所有层的隐藏状态和注意力

---

## 2. BertEncoder 类概览

```python
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
```

### 2.1 组件说明

| 组件 | BERT-base | BERT-large |
|------|-----------|------------|
| `num_hidden_layers` | 12 | 24 |
| `layer` | ModuleList[12个BertLayer] | ModuleList[24个BertLayer] |

### 2.2 nn.ModuleList vs nn.Sequential

```python
# nn.ModuleList：手动控制执行
self.layer = nn.ModuleList([BertLayer(config) for _ in range(12)])

# 执行方式
for layer_module in self.layer:
    hidden_states = layer_module(hidden_states)
```

**为什么用 ModuleList 而不是 Sequential？**
- 需要在每层之间收集输出
- 需要处理 KV 缓存
- 需要支持梯度检查点

---

## 3. 前向传播详解

### 3.1 初始化输出收集器

```python
all_hidden_states = () if output_hidden_states else None
all_self_attentions = () if output_attentions else None
all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
```

### 3.2 梯度检查点与缓存的不兼容

```python
if self.gradient_checkpointing and self.training:
    if use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        )
        use_cache = False
```

**为什么不兼容？**
- 梯度检查点在反向传播时重新计算前向传播
- KV 缓存需要保存中间状态
- 两者机制冲突

### 3.3 遍历每一层

```python
next_decoder_cache = () if use_cache else None

for i, layer_module in enumerate(self.layer):
    # 收集隐藏状态（在层处理之前）
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    # 获取当前层的头掩码和 KV 缓存
    layer_head_mask = head_mask[i] if head_mask is not None else None
    past_key_value = past_key_values[i] if past_key_values is not None else None

    # 执行层前向传播
    if self.gradient_checkpointing and self.training:
        layer_outputs = self._gradient_checkpointing_func(
            layer_module.__call__,
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
    else:
        layer_outputs = layer_module(
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

    # 更新隐藏状态
    hidden_states = layer_outputs[0]

    # 收集 KV 缓存
    if use_cache:
        next_decoder_cache += (layer_outputs[-1],)

    # 收集注意力权重
    if output_attentions:
        all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if self.config.add_cross_attention:
            all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
```

### 3.4 收集最终隐藏状态

```python
if output_hidden_states:
    all_hidden_states = all_hidden_states + (hidden_states,)
```

**注意：** `all_hidden_states` 包含 13 个张量（嵌入层 + 12 个编码器层的输出）

---

## 4. 梯度检查点（Gradient Checkpointing）

### 4.1 原理

```
正常训练:
┌──────────────────────────────────────────┐
│ 保存所有中间激活值用于反向传播            │
│ 显存占用: O(n) 其中 n 是层数             │
└──────────────────────────────────────────┘

梯度检查点:
┌──────────────────────────────────────────┐
│ 只保存部分检查点，反向时重新计算          │
│ 显存占用: O(√n)                          │
│ 计算开销: 增加约 33%                      │
└──────────────────────────────────────────┘
```

### 4.2 启用方式

```python
model.gradient_checkpointing_enable()
# 或
model.encoder.gradient_checkpointing = True
```

### 4.3 适用场景

| 场景 | 是否推荐 |
|------|----------|
| 显存不足 | ✓ 推荐 |
| 训练大模型 | ✓ 推荐 |
| 需要快速迭代 | ✗ 不推荐（增加计算时间） |
| 推理阶段 | ✗ 不需要 |

---

## 5. 返回值

### 5.1 元组形式（return_dict=False）

```python
return tuple(
    v
    for v in [
        hidden_states,           # 最后一层隐藏状态
        next_decoder_cache,      # KV 缓存
        all_hidden_states,       # 所有层隐藏状态
        all_self_attentions,     # 所有自注意力权重
        all_cross_attentions,    # 所有交叉注意力权重
    ]
    if v is not None
)
```

### 5.2 数据类形式（return_dict=True）

```python
return BaseModelOutputWithPastAndCrossAttentions(
    last_hidden_state=hidden_states,
    past_key_values=next_decoder_cache,
    hidden_states=all_hidden_states,
    attentions=all_self_attentions,
    cross_attentions=all_cross_attentions,
)
```

---

## 6. 完整数据流图

```
hidden_states (from embeddings)
         │
         ↓
┌─────────────────────────────────────────────────┐
│                 Layer 0                         │
│  ┌─────────────────────────────────────────┐   │
│  │ Self-Attention → Add&Norm → FFN → Add&Norm │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────┐
│                 Layer 1                         │
│  ┌─────────────────────────────────────────┐   │
│  │ Self-Attention → Add&Norm → FFN → Add&Norm │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
         │
         ↓
        ...
         │
         ↓
┌─────────────────────────────────────────────────┐
│                Layer 11                         │
│  ┌─────────────────────────────────────────┐   │
│  │ Self-Attention → Add&Norm → FFN → Add&Norm │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
         │
         ↓
    last_hidden_state
```

---

## 7. Head Mask 的使用

```python
layer_head_mask = head_mask[i] if head_mask is not None else None
```

**Head Mask 结构：**

| 形状 | 含义 |
|------|------|
| `[num_heads]` | 所有层共用同一个掩码 |
| `[num_layers, num_heads]` | 每层独立的掩码 |

**示例：剪枝第 2、5、8 个注意力头**
```python
head_mask = torch.ones(12, 12)  # 12 层，每层 12 个头
head_mask[:, [2, 5, 8]] = 0     # 禁用这些头
```

---

## 8. 输出维度汇总

| 输出 | 形状（BERT-base） |
|------|-------------------|
| `last_hidden_state` | [batch, seq_len, 768] |
| `past_key_values` | 12 × (K, V)，每个 [batch, heads, seq, head_dim] |
| `hidden_states` | 13 × [batch, seq_len, 768] |
| `attentions` | 12 × [batch, heads, seq, seq] |

---

## 9. 本章小结

本章介绍了 BertEncoder：

1. **层堆叠**：使用 `nn.ModuleList` 管理 12/24 层
2. **梯度检查点**：以计算换显存
3. **输出收集**：可选收集所有层的隐藏状态和注意力
4. **KV 缓存**：支持增量生成

下一章，我们将介绍 **BertPooler 和预测头**——将编码器输出转换为任务特定的预测。

---

*下一章预告：[bert_modeling_guide_09_pooler_heads.md] - 池化层与预测头*
