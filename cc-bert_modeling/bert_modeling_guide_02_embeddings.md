# BERT 模型源码解析 - 第二章：BertEmbeddings 嵌入层

> 源码位置：transformers/models/bert/modeling_bert.py (第159-221行)

## 1. 开篇：BERT 的"第一道门"

在深度学习的世界中，嵌入（Embedding）是将离散符号转换为连续向量的魔法。BERT 的嵌入层是输入数据的"接待大厅"，负责将原始的 token ID 转换为模型可以理解的向量表示。

---

## 2. BertEmbeddings 类概览

```python
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
```

**BERT 的嵌入由三部分组成：**

```
总嵌入 = 词嵌入 + 位置嵌入 + 句子嵌入
```

| 嵌入类型 | 作用 | 维度 |
|----------|------|------|
| Word Embeddings | 表示词汇表中每个词 | [vocab_size, hidden_size] |
| Position Embeddings | 表示词在序列中的位置 | [max_position, hidden_size] |
| Token Type Embeddings | 区分不同句子 | [type_vocab_size, hidden_size] |

---

## 3. 初始化函数详解

```python
def __init__(self, config):
    super().__init__()
```

### 3.1 词嵌入层（Word Embeddings）

```python
self.word_embeddings = nn.Embedding(
    config.vocab_size,        # 词汇表大小，BERT-base 为 30522
    config.hidden_size,       # 隐藏层维度，BERT-base 为 768
    padding_idx=config.pad_token_id  # 填充 token 的索引
)
```

**解析：**
- `vocab_size=30522`：BERT 使用 WordPiece 分词，词汇表大小约 3 万
- `hidden_size=768`：每个词被映射为 768 维向量
- `padding_idx`：指定 [PAD] token 的索引，其嵌入在训练中不会更新

**数据流示例：**
```
输入: [101, 2023, 2003, 102]  # [CLS] "This" "is" [SEP]
      ↓ nn.Embedding 查表
输出: [[768维向量], [768维向量], [768维向量], [768维向量]]
形状: [batch_size, seq_length, 768]
```

### 3.2 位置嵌入层（Position Embeddings）

```python
self.position_embeddings = nn.Embedding(
    config.max_position_embeddings,  # 最大序列长度，BERT 为 512
    config.hidden_size               # 768
)
```

**解析：**
- BERT 使用**绝对位置编码**，每个位置（0-511）都有可学习的嵌入
- 与 Transformer 原论文的正弦位置编码不同，BERT 的位置编码是可训练的

**位置索引示意：**
```
序列:  [CLS]  The   cat  sat  [SEP]
位置:    0     1     2     3     4
```

### 3.3 句子嵌入层（Token Type Embeddings）

```python
self.token_type_embeddings = nn.Embedding(
    config.type_vocab_size,  # 句子类型数量，BERT 为 2
    config.hidden_size       # 768
)
```

**解析：**
- 用于区分输入中的不同句子（在 NSP 任务中）
- `type_vocab_size=2`：句子 A 用 0，句子 B 用 1

**示例：**
```
输入: [CLS] The cat sat [SEP] The dog ran [SEP]
类型:   0    0   0   0    0     1   1   1    1
```

### 3.4 Layer Normalization

```python
# self.LayerNorm is not snake-cased to stick with TensorFlow model
# variable name and be able to load any TensorFlow checkpoint file
self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
```

**解析：**
- 变量名 `LayerNorm`（驼峰式）是为了兼容原始 TensorFlow 检查点
- LayerNorm 对每个样本进行归一化：
  $$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

### 3.5 Dropout

```python
self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 通常为 0.1
```

**解析：**
- 随机丢弃部分神经元，防止过拟合
- 在训练时生效，推理时不生效

### 3.6 注册缓冲区

```python
self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

self.register_buffer(
    "position_ids",
    torch.arange(config.max_position_embeddings).expand((1, -1)),
    persistent=False
)

self.register_buffer(
    "token_type_ids",
    torch.zeros(self.position_ids.size(), dtype=torch.long),
    persistent=False
)
```

**解析：**
- `register_buffer`：注册一个不会被视为模型参数的张量
- `persistent=False`：不会保存到 state_dict 中
- `position_ids`：预置的位置索引 `[0, 1, 2, ..., 511]`
- `token_type_ids`：预置的全零句子类型索引

---

## 4. 前向传播函数详解

```python
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    past_key_values_length: int = 0,
) -> torch.Tensor:
```

### 4.1 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `input_ids` | LongTensor | token ID 序列，形状 [batch, seq_len] |
| `token_type_ids` | LongTensor | 句子类型 ID，形状同上 |
| `position_ids` | LongTensor | 位置 ID，形状同上 |
| `inputs_embeds` | FloatTensor | 直接提供嵌入（跳过词嵌入） |
| `past_key_values_length` | int | 缓存的 key/value 长度（用于生成） |

### 4.2 获取输入形状

```python
if input_ids is not None:
    input_shape = input_ids.size()
else:
    input_shape = inputs_embeds.size()[:-1]

seq_length = input_shape[1]
```

**解析：**
- 支持 `input_ids` 或 `inputs_embeds` 两种输入方式
- `input_shape` 为 `[batch_size, seq_length]`

### 4.3 处理位置 ID

```python
if position_ids is None:
    position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
```

**解析：**
- 如果未提供 `position_ids`，从预置缓冲区中切片获取
- `past_key_values_length` 用于增量生成场景

**示例：**
```python
# 正常情况（past_key_values_length=0, seq_length=4）
position_ids = [0, 1, 2, 3]

# 增量生成（past_key_values_length=4, seq_length=1）
position_ids = [4]  # 继续从位置 4 开始
```

### 4.4 处理句子类型 ID

```python
if token_type_ids is None:
    if hasattr(self, "token_type_ids"):
        buffered_token_type_ids = self.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
        token_type_ids = buffered_token_type_ids_expanded
    else:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
```

**解析：**
- 如果未提供，使用全零张量（默认都是句子 A）
- 使用缓冲区可以支持模型追踪（tracing）

### 4.5 计算嵌入

```python
if inputs_embeds is None:
    inputs_embeds = self.word_embeddings(input_ids)

token_type_embeddings = self.token_type_embeddings(token_type_ids)

embeddings = inputs_embeds + token_type_embeddings

if self.position_embedding_type == "absolute":
    position_embeddings = self.position_embeddings(position_ids)
    embeddings += position_embeddings
```

**计算流程图：**
```
┌─────────────┐
│  input_ids  │
└──────┬──────┘
       ↓
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│word_embeds  │  +  │token_type_embeds │  +  │position_embeds │
└──────┬──────┘     └──────────────────┘     └────────────────┘
       ↓
   embeddings
```

### 4.6 后处理

```python
embeddings = self.LayerNorm(embeddings)
embeddings = self.dropout(embeddings)
return embeddings
```

**解析：**
1. LayerNorm：归一化，稳定训练
2. Dropout：正则化，防止过拟合

---

## 5. 完整数据流示例

```python
# 假设输入
input_ids = torch.tensor([[101, 2023, 2003, 102]])  # [CLS] "This" "is" [SEP]
token_type_ids = torch.tensor([[0, 0, 0, 0]])        # 都属于句子 A

# 步骤 1: 词嵌入
word_emb = word_embeddings(input_ids)  # [1, 4, 768]

# 步骤 2: 位置嵌入
position_ids = [0, 1, 2, 3]
pos_emb = position_embeddings(position_ids)  # [1, 4, 768]

# 步骤 3: 句子类型嵌入
type_emb = token_type_embeddings(token_type_ids)  # [1, 4, 768]

# 步骤 4: 相加
embeddings = word_emb + pos_emb + type_emb  # [1, 4, 768]

# 步骤 5: LayerNorm + Dropout
output = dropout(layer_norm(embeddings))  # [1, 4, 768]
```

---

## 6. 本章小结

BertEmbeddings 是 BERT 模型的入口，它完成了：

1. **词嵌入**：将离散 token 转为连续向量
2. **位置嵌入**：注入位置信息
3. **句子嵌入**：区分不同句子
4. **归一化**：LayerNorm 稳定数值
5. **正则化**：Dropout 防止过拟合

下一章，我们将深入 BERT 的核心机制——**自注意力（Self-Attention）**，这是 Transformer 架构的灵魂所在。

---

*下一章预告：[bert_modeling_guide_03_self_attention.md] - Query、Key、Value 的数学之美*
