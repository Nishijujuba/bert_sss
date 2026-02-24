# BERT 模型源码解析 - 第十一章：BertModel 核心模型

> 源码位置：transformers/models/bert/modeling_bert.py (第957-1168行)

## 1. 开篇：BERT 的"躯干"

`BertModel` 是 BERT 的核心类，它将我们之前介绍的所有组件组装在一起：

- **BertEmbeddings**：输入嵌入
- **BertEncoder**：12 层 Transformer 编码器
- **BertPooler**：池化层（可选）

---

## 2. BertModel 类概览

```python
class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need] by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
    Lukasz Kaiser and Illia Polosukhin.

    To behave as a decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    _no_split_modules = ["BertEmbeddings", "BertLayer"]
```

### 2.1 _no_split_modules

```python
_no_split_modules = ["BertEmbeddings", "BertLayer"]
```

**作用：** 在模型并行或设备映射时，这些模块不会被拆分。

---

## 3. 初始化方法

```python
def __init__(self, config, add_pooling_layer=True):
    super().__init__(config)
    self.config = config

    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)

    self.pooler = BertPooler(config) if add_pooling_layer else None

    self.attn_implementation = config._attn_implementation
    self.position_embedding_type = config.position_embedding_type

    # Initialize weights and apply final processing
    self.post_init()
```

### 3.1 组件组装

```
BertModel
├── embeddings (BertEmbeddings)
│   ├── word_embeddings
│   ├── position_embeddings
│   └── token_type_embeddings
├── encoder (BertEncoder)
│   └── layer[0..11] (BertLayer)
│       ├── attention (BertAttention)
│       ├── intermediate (BertIntermediate)
│       └── output (BertOutput)
└── pooler (BertPooler or None)
```

### 3.2 add_pooling_layer 参数

```python
self.pooler = BertPooler(config) if add_pooling_layer else None
```

**何时设为 False？**
- 某些下游任务不需要 pooled_output
- 减少不必要的计算

---

## 4. 辅助方法

### 4.1 嵌入层访问

```python
def get_input_embeddings(self):
    return self.embeddings.word_embeddings

def set_input_embeddings(self, value):
    self.embeddings.word_embeddings = value
```

### 4.2 注意力头剪枝

```python
def _prune_heads(self, heads_to_prune):
    """
    Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
    See base class PreTrainedModel
    """
    for layer, heads in heads_to_prune.items():
        self.encoder.layer[layer].attention.prune_heads(heads)
```

**使用示例：**
```python
# 剪枝第 0 层的第 1、2 个头，第 3 层的第 0 个头
heads_to_prune = {0: [1, 2], 3: [0]}
model._prune_heads(heads_to_prune)
```

---

## 5. 前向传播详解

### 5.1 参数说明

```python
def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
```

| 参数 | 形状 | 说明 |
|------|------|------|
| `input_ids` | [batch, seq] | 输入 token ID |
| `attention_mask` | [batch, seq] | 注意力掩码 |
| `token_type_ids` | [batch, seq] | 句子类型 ID |
| `position_ids` | [batch, seq] | 位置 ID |
| `head_mask` | [heads] 或 [layers, heads] | 头掩码 |
| `inputs_embeds` | [batch, seq, hidden] | 直接提供嵌入 |
| `encoder_hidden_states` | [batch, enc_seq, hidden] | 编码器输出 |
| `past_key_values` | list of tuples | KV 缓存 |

### 5.2 配置参数处理

```python
output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
output_hidden_states = (
    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
)
return_dict = return_dict if return_dict is not None else self.config.use_return_dict

if self.config.is_decoder:
    use_cache = use_cache if use_cache is not None else self.config.use_cache
else:
    use_cache = False
```

### 5.3 输入验证

```python
if input_ids is not None and inputs_embeds is not None:
    raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
elif input_ids is not None:
    self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
    input_shape = input_ids.size()
elif inputs_embeds is not None:
    input_shape = inputs_embeds.size()[:-1]
else:
    raise ValueError("You have to specify either input_ids or inputs_embeds")
```

### 5.4 获取设备信息

```python
batch_size, seq_length = input_shape
device = input_ids.device if input_ids is not None else inputs_embeds.device

past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
```

### 5.5 处理 token_type_ids

```python
if token_type_ids is None:
    if hasattr(self.embeddings, "token_type_ids"):
        buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        token_type_ids = buffered_token_type_ids_expanded
    else:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
```

---

## 6. 嵌入层计算

```python
embedding_output = self.embeddings(
    input_ids=input_ids,
    position_ids=position_ids,
    token_type_ids=token_type_ids,
    inputs_embeds=inputs_embeds,
    past_key_values_length=past_key_values_length,
)
```

**输出形状：** `[batch, seq_len, 768]`

---

## 7. 注意力掩码处理

### 7.1 默认掩码

```python
if attention_mask is None:
    attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
```

### 7.2 SDPA 掩码 vs 普通掩码

```python
use_sdpa_attention_masks = (
    self.attn_implementation == "sdpa"
    and self.position_embedding_type == "absolute"
    and head_mask is None
    and not output_attentions
)
```

### 7.3 扩展注意力掩码

```python
if use_sdpa_attention_masks and attention_mask.dim() == 2:
    # SDPA 优化的掩码格式
    if self.config.is_decoder:
        extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            input_shape,
            embedding_output,
            past_key_values_length,
        )
    else:
        extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
            attention_mask, embedding_output.dtype, tgt_len=seq_length
        )
else:
    # 普通掩码格式
    extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
```

**掩码变换：**
```
原始: [batch, seq_len] → 0/1
扩展: [batch, 1, 1, seq_len] → 0/-inf
```

---

## 8. 编码器前向传播

```python
encoder_outputs = self.encoder(
    embedding_output,
    attention_mask=extended_attention_mask,
    head_mask=head_mask,
    encoder_hidden_states=encoder_hidden_states,
    encoder_attention_mask=encoder_extended_attention_mask,
    past_key_values=past_key_values,
    use_cache=use_cache,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)
sequence_output = encoder_outputs[0]
```

---

## 9. 池化层与输出

```python
pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

if not return_dict:
    return (sequence_output, pooled_output) + encoder_outputs[1:]

return BaseModelOutputWithPoolingAndCrossAttentions(
    last_hidden_state=sequence_output,
    pooler_output=pooled_output,
    past_key_values=encoder_outputs.past_key_values,
    hidden_states=encoder_outputs.hidden_states,
    attentions=encoder_outputs.attentions,
    cross_attentions=encoder_outputs.cross_attentions,
)
```

---

## 10. 完整数据流图

```
input_ids [batch, seq_len]
         │
         ↓
┌─────────────────────────────────────┐
│         BertEmbeddings              │
│  ┌─────────────────────────────┐   │
│  │ word_emb + pos_emb + type_emb│   │
│  │ LayerNorm + Dropout          │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
         │
         ↓
embedding_output [batch, seq_len, 768]
         │
         ↓
┌─────────────────────────────────────┐
│         BertEncoder                 │
│  ┌─────────────────────────────┐   │
│  │ Layer 0                      │   │
│  │ Layer 1                      │   │
│  │ ...                          │   │
│  │ Layer 11                     │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
         │
         ↓
sequence_output [batch, seq_len, 768]
         │
         ├────────────────────────────┐
         │                            │
         ↓                            ↓
┌─────────────────────┐    ┌─────────────────────┐
│ BertPooler (可选)    │    │ 直接作为输出         │
│ 取 [CLS] + dense    │    │ sequence_output     │
└──────────┬──────────┘    └─────────────────────┘
           │
           ↓
pooled_output [batch, 768]
```

---

## 11. 返回值结构

### 11.1 BaseModelOutputWithPoolingAndCrossAttentions

```python
@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None  # [batch, seq, 768]
    pooler_output: torch.FloatTensor = None      # [batch, 768]
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
```

### 11.2 使用示例

```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)

# 访问输出
last_hidden = outputs.last_hidden_state  # [1, seq_len, 768]
pooled = outputs.pooler_output           # [1, 768]
```

---

## 12. 本章小结

本章介绍了 BERT 的核心模型类：

1. **BertModel 组成**：
   - BertEmbeddings：输入嵌入
   - BertEncoder：多层编码器
   - BertPooler：池化层（可选）

2. **编码器/解码器模式**：
   - 默认作为编码器
   - 可配置为解码器（支持交叉注意力）

3. **注意力掩码处理**：
   - 支持普通掩码和 SDPA 优化掩码
   - 自动扩展为 4D 格式

下一章，我们将介绍各种**下游任务模型**。

---

*下一章预告：[bert_modeling_guide_12_task_models.md] - 预训练与下游任务模型*
