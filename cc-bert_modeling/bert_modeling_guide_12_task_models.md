# BERT 模型源码解析 - 第十二章：预训练与下游任务模型

> 源码位置：transformers/models/bert/modeling_bert.py (第1177行及以后)

## 1. 开篇：从预训练到下游任务

BERT 的强大之处在于其"预训练+微调"范式。本章我们将介绍：

1. **预训练模型**：BertForPreTraining、BertForMaskedLM
2. **分类模型**：BertForSequenceClassification、BertForMultipleChoice
3. **Token 级别模型**：BertForTokenClassification
4. **问答模型**：BertForQuestionAnswering

---

## 2. 模型架构概览

```
                    BertModel (核心)
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ↓                    ↓                    ↓
预训练任务            分类任务             Token级任务
    │                    │                    │
    ├── BertForPreTraining                    │
    ├── BertForMaskedLM  ├── SequenceClassification
    ├── BertLMHeadModel  ├── MultipleChoice   ├── TokenClassification
    └── NextSentencePred └── ...              └── QuestionAnswering
```

---

## 3. BertForPreTraining（预训练模型）

### 3.1 模型结构

```python
class BertForPreTraining(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.post_init()
```

**组件：**
- `bert`：核心 BERT 模型
- `cls`：包含 MLM 和 NSP 两个预测头

### 3.2 前向传播

```python
def forward(
    self,
    input_ids, attention_mask, token_type_ids, ...
    labels: Optional[torch.Tensor] = None,           # MLM 标签
    next_sentence_label: Optional[torch.Tensor] = None,  # NSP 标签
):
    # BERT 编码
    outputs = self.bert(...)
    sequence_output, pooled_output = outputs[:2]

    # 预测
    prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

    # 计算损失
    total_loss = None
    if labels is not None and next_sentence_label is not None:
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss

    return BertForPreTrainingOutput(...)
```

**数据流：**
```
input_ids
    ↓
BertModel
    ↓
├── sequence_output → BertLMPredictionHead → prediction_scores
└── pooled_output → Linear(768→2) → seq_relationship_score
```

---

## 4. BertForMaskedLM（掩码语言模型）

### 4.1 模型结构

```python
class BertForMaskedLM(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)  # 不需要池化层
        self.cls = BertOnlyMLMHead(config)

        self.post_init()
```

**注意：** `add_pooling_layer=False`，因为 MLM 任务不需要 pooled_output

### 4.2 损失计算

```python
if labels is not None:
    loss_fct = CrossEntropyLoss()  # -100 index = padding token
    masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
```

**labels 中的 -100：**
- -100 表示忽略该位置
- CrossEntropyLoss 默认忽略 -100

---

## 5. BertForSequenceClassification（序列分类）

### 5.1 模型结构

```python
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()
```

**数据流：**
```
input_ids
    ↓
BertModel
    ↓
pooled_output [batch, 768]
    ↓
Dropout
    ↓
Linear(768 → num_labels)
    ↓
logits [batch, num_labels]
```

### 5.2 自动推断任务类型

```python
if self.config.problem_type is None:
    if self.num_labels == 1:
        self.config.problem_type = "regression"
    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        self.config.problem_type = "single_label_classification"
    else:
        self.config.problem_type = "multi_label_classification"
```

### 5.3 损失函数选择

| 问题类型 | 损失函数 |
|----------|----------|
| 回归 (num_labels=1) | MSELoss |
| 单标签分类 | CrossEntropyLoss |
| 多标签分类 | BCEWithLogitsLoss |

---

## 6. BertForMultipleChoice（多项选择）

### 6.1 模型结构

```python
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)  # 输出 1 个分数
```

### 6.2 输入形状处理

```python
# 输入形状：[batch, num_choices, seq_len]
num_choices = input_ids.shape[1]

# 展平为 [batch * num_choices, seq_len]
input_ids = input_ids.view(-1, input_ids.size(-1))

# BERT 编码
outputs = self.bert(input_ids, ...)

# 获取 pooled_output 并分类
pooled_output = outputs[1]
logits = self.classifier(pooled_output)

# 重塑为 [batch, num_choices]
reshaped_logits = logits.view(-1, num_choices)
```

**数据流：**
```
[batch, 4, seq_len]  (4 个选项)
    ↓ view
[batch*4, seq_len]
    ↓ BERT
[batch*4, 768]
    ↓ Linear(768→1)
[batch*4, 1]
    ↓ view
[batch, 4]  (每个选项的分数)
```

---

## 7. BertForTokenClassification（Token 分类）

### 7.1 模型结构

```python
class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)  # 不需要池化层
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
```

### 7.2 数据流

```
input_ids [batch, seq_len]
    ↓
BertModel
    ↓
sequence_output [batch, seq_len, 768]
    ↓
Dropout
    ↓
Linear(768 → num_labels)
    ↓
logits [batch, seq_len, num_labels]
```

**应用场景：**
- 命名实体识别 (NER)
- 词性标注 (POS)
- 分块识别 (Chunking)

---

## 8. BertForQuestionAnswering（问答）

### 8.1 模型结构

```python
class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)  # 通常 num_labels=2
```

### 8.2 数据流

```python
sequence_output = outputs[0]  # [batch, seq_len, 768]

logits = self.qa_outputs(sequence_output)  # [batch, seq_len, 2]

# 分离起始和结束位置 logits
start_logits, end_logits = logits.split(1, dim=-1)
start_logits = start_logits.squeeze(-1).contiguous()  # [batch, seq_len]
end_logits = end_logits.squeeze(-1).contiguous()      # [batch, seq_len]
```

### 8.3 损失计算

```python
if start_positions is not None and end_positions is not None:
    # 忽略超出序列长度的位置
    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)

    ignored_index = start_logits.size(1)
    start_positions = start_positions.clamp(0, ignored_index)
    end_positions = end_positions.clamp(0, ignored_index)

    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2
```

---

## 9. 模型对比总结

| 模型 | 输出类型 | 池化层 | 典型任务 |
|------|----------|--------|----------|
| BertModel | 隐藏状态 | ✓ | 特征提取 |
| BertForPreTraining | MLM + NSP | ✓ | 预训练 |
| BertForMaskedLM | Token 预测 | ✗ | MLM 微调 |
| BertForSequenceClassification | 序列分类 | ✓ | 情感分析 |
| BertForMultipleChoice | 选项分数 | ✓ | 阅读理解选择 |
| BertForTokenClassification | Token 分类 | ✗ | NER |
| BertForQuestionAnswering | 起止位置 | ✗ | SQuAD |

---

## 10. 使用示例

### 10.1 序列分类

```python
from transformers import BertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)

logits = outputs.logits  # [1, 2]
predicted_class = logits.argmax(-1).item()
```

### 10.2 Token 分类 (NER)

```python
from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

inputs = tokenizer("Hello, my name is John.", return_tensors="pt")
outputs = model(**inputs)

logits = outputs.logits  # [1, seq_len, num_labels]
predicted_tags = logits.argmax(-1)
```

### 10.3 问答

```python
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

question = "What is the capital of France?"
context = "Paris is the capital and most populous city of France."

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits

start_idx = start_logits.argmax(-1).item()
end_idx = end_logits.argmax(-1).item()

answer = tokenizer.decode(inputs["input_ids"][0, start_idx:end_idx+1])
```

---

## 11. 本章小结

本章介绍了 BERT 的各种下游任务模型：

1. **预训练模型**：
   - BertForPreTraining：MLM + NSP
   - BertForMaskedLM：仅 MLM

2. **分类模型**：
   - BertForSequenceClassification：序列级别分类
   - BertForMultipleChoice：多选项任务

3. **Token 级模型**：
   - BertForTokenClassification：命名实体识别
   - BertForQuestionAnswering：问答任务

4. **设计模式**：
   - 共享 BertModel 核心编码器
   - 添加任务特定的预测头
   - 自动推断问题类型和损失函数

---

## 12. 系列总结

恭喜你完成了 BERT 模型源码解析系列！我们共同探索了：

1. **导入与配置**：模型的基础设施
2. **BertEmbeddings**：输入嵌入层
3. **BertSelfAttention**：自注意力机制
4. **BertSdpaSelfAttention**：优化的注意力实现
5. **BertSelfOutput & BertAttention**：注意力后处理
6. **BertIntermediate & BertOutput**：前馈网络
7. **BertLayer**：完整的 Transformer 层
8. **BertEncoder**：多层编码器堆叠
9. **预测头**：池化层与任务特定输出
10. **BertPreTrainedModel**：预训练模型基类
11. **BertModel**：核心模型类
12. **下游任务模型**：各种应用场景

希望这个系列对你理解 BERT 的实现有所帮助！

---

*本系列完*
