# BERT 模型源码解析 - 第九章：池化层与预测头

> 源码位置：transformers/models/bert/modeling_bert.py (第737-822行)

## 1. 开篇：从序列表示到任务输出

编码器输出的是序列中每个 token 的表示 `[batch, seq_len, 768]`。但很多任务需要：

1. **序列级别**：整个句子的表示（如分类任务）
2. **Token 级别**：每个 token 的预测（如命名实体识别）
3. **MLM 任务**：预测被遮蔽的词

本章介绍这些"输出头"的实现。

---

## 2. BertPooler 类详解

```python
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

### 2.1 [CLS] Token 的作用

```
输入:  [CLS]  The   cat  sat  [SEP]
位置:    0     1     2    3     4

池化: 取位置 0 的隐藏状态
```

**为什么用 [CLS]？**
- [CLS] 在预训练时参与了所有 token 的自注意力
- 它的表示聚合了整个句子的信息
- 方便用于序列级别的任务

### 2.2 数据流

```
hidden_states [batch, seq_len, 768]
         │
         ↓ 取 [:, 0]
first_token_tensor [batch, 768]
         │
         ↓
    dense (768 → 768)
         │
         ↓
    Tanh()
         │
         ↓
pooled_output [batch, 768]
```

### 2.3 为什么用 Tanh？

- Tanh 输出范围 [-1, 1]，有界
- 比线性激活更稳定
- 与原始 BERT 论文保持一致

---

## 3. BertPredictionHeadTransform 类详解

```python
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
```

### 3.1 作用

这是 MLM 预测头的"变换层"，用于在输出到词表之前对隐藏状态进行处理。

### 3.2 数据流

```
hidden_states [batch, seq_len, 768]
         │
         ↓
    dense (768 → 768)
         │
         ↓
    GELU()
         │
         ↓
    LayerNorm
         │
         ↓
transformed [batch, seq_len, 768]
```

---

## 4. BertLMPredictionHead 类详解

```python
class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
```

### 4.1 权重绑定（Weight Tying）

```python
# 输出权重与输入嵌入共享
self.decoder.weight = model.embeddings.word_embeddings.weight
```

**优点：**
- 减少参数量（768 × 30522 ≈ 23M）
- 输入和输出空间一致
- 正则化效果

### 4.2 数据流

```
hidden_states [batch, seq_len, 768]
         │
         ↓
    BertPredictionHeadTransform
         │
         ↓
    decoder (768 → vocab_size, 权重与 word_embeddings 绑定)
         │
         ↓
prediction_logits [batch, seq_len, vocab_size]
```

### 4.3 参数说明

| 参数 | 形状 | 说明 |
|------|------|------|
| `decoder.weight` | [vocab_size, 768] | 与 word_embeddings 共享 |
| `bias` | [vocab_size] | 每个 token 的偏置 |

---

## 5. BertOnlyMLMHead 类详解

```python
class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
```

**用途：** 用于 `BertForMaskedLM` 模型

---

## 6. BertOnlyNSPHead 类详解

```python
class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score
```

**用途：** 用于 Next Sentence Prediction (NSP) 任务

**输出：**
- `[0]`：两个句子**不**连续
- `[1]`：两个句子连续

---

## 7. BertPreTrainingHeads 类详解

```python
class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score
```

**用途：** 用于 `BertForPreTraining` 模型

**输出：**
1. `prediction_scores`：MLM 预测 [batch, seq_len, vocab_size]
2. `seq_relationship_score`：NSP 预测 [batch, 2]

---

## 8. 各预测头的关系图

```
                        ┌─────────────────────────────────────┐
                        │         encoder_output              │
                        │   [batch, seq_len, 768]             │
                        └──────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ↓                        ↓                        ↓
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  取 [:, 0, :]   │    │                 │    │                 │
    │  [CLS] 向量     │    │   sequence_output   │    │                 │
    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
             │                      │                      │
             ↓                      ↓                      ↓
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   BertPooler    │    │ BertLMPrediction│    │   Token 分类头   │
    │   (CLS → dense) │    │     Head        │    │   (Linear)      │
    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
             │                      │                      │
             ↓                      ↓                      ↓
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ pooled_output   │    │ prediction_scores│   │ logits          │
    │ [batch, 768]    │    │ [batch,seq,vocab]│   │ [batch,seq,num] │
    └────────┬────────┘    └─────────────────┘    └─────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ↓                 ↓
┌───────────┐   ┌───────────┐
│ NSP Head  │   │   分类头   │
│(Linear→2) │   │ (Linear)  │
└─────┬─────┘   └─────┬─────┘
      │               │
      ↓               ↓
┌───────────┐   ┌───────────┐
│ seq_rel   │   │ logits    │
│ [batch,2] │   │[batch,num]│
└───────────┘   └───────────┘
```

---

## 9. 预测头与模型的对应关系

| 模型 | 使用的预测头 |
|------|-------------|
| `BertModel` | 无（只返回编码器输出） |
| `BertForPreTraining` | `BertPreTrainingHeads` |
| `BertForMaskedLM` | `BertOnlyMLMHead` |
| `BertForNextSentencePrediction` | `BertOnlyNSPHead` |
| `BertForSequenceClassification` | 分类头（自定义） |
| `BertForTokenClassification` | Token 分类头（自定义） |
| `BertForQuestionAnswering` | QA 头（自定义） |

---

## 10. 本章小结

本章介绍了 BERT 的输出头：

1. **BertPooler**：
   - 取 [CLS] token 的表示
   - 经过 dense + Tanh
   - 用于序列级别任务

2. **BertLMPredictionHead**：
   - 变换层 + 输出层
   - 权重与输入嵌入绑定
   - 用于 MLM 任务

3. **NSP 头**：
   - 二分类：句子是否连续
   - 输入是 pooled_output

下一章，我们将介绍 **BertPreTrainedModel**——所有 BERT 模型的基类。

---

*下一章预告：[bert_modeling_guide_10_pretrained.md] - 预训练模型的基类*
