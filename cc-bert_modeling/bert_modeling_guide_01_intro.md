# BERT 模型源码解析 - 第一章：导入与配置

> 源码位置：transformers/models/bert/modeling_bert.py (第1-158行)

## 1. 开篇：BERT 模型文件概览

本章我们将探索 BERT 模型实现的"前奏部分"——包括许可证声明、模块导入、以及一些辅助函数。就像一部小说的序章，这部分为后续的精彩内容奠定了基础。

---

## 2. 许可证与版权声明（第1-15行）

```python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

**解读：**
- 这是标准的 Apache 2.0 开源许可证声明
- BERT 最初由 Google AI Language Team 开发，HuggingFace 团队进行了 PyTorch 移植
- NVIDIA 也贡献了部分代码

---

## 3. 标准库与第三方库导入（第18-28行）

```python
import math                          # 数学运算，用于缩放注意力分数
import os                            # 文件系统操作
import warnings                      # 警告信息处理
from dataclasses import dataclass    # 数据类装饰器，用于定义配置类
from typing import List, Optional, Tuple, Union  # 类型注解

import torch                         # PyTorch 核心库
import torch.utils.checkpoint        # 梯度检查点，用于节省显存
from packaging import version        # 版本比较工具
from torch import nn                 # 神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 常用损失函数
```

**核心导入说明：**

| 导入项 | 用途 |
|--------|------|
| `math` | 计算注意力分数的缩放因子 $\sqrt{d_k}$ |
| `torch.utils.checkpoint` | 梯度检查点技术，以计算换显存 |
| `BCEWithLogitsLoss` | 二分类任务的损失函数 |
| `CrossEntropyLoss` | 多分类任务的损失函数（如 MLM） |
| `MSELoss` | 均方误差损失 |

---

## 4. Transformers 内部模块导入（第30-58行）

### 4.1 激活函数映射
```python
from ...activations import ACT2FN
```
`ACT2FN` 是一个字典，将激活函数名称映射到实际函数：
```python
ACT2FN = {
    "relu": torch.nn.ReLU(),
    "gelu": torch.nn.GELU(),
    "silu": torch.nn.SiLU(),
    # ...
}
```

### 4.2 注意力掩码工具
```python
from ...modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
```
- 用于生成 SDPA (Scaled Dot Product Attention) 优化的注意力掩码
- SDPA 是 PyTorch 2.0+ 提供的高效注意力实现

### 4.3 模型输出类
```python
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,           # 用于 Masked Language Model
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput, # 用于文本分类
    TokenClassifierOutput,    # 用于命名实体识别
)
```

这些是**数据类**，用于规范化模型的输出格式。例如：
```python
@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
```

### 4.4 工具函数
```python
from ...pytorch_utils import (
    apply_chunking_to_forward,      # 分块前向传播，节省显存
    find_pruneable_heads_and_indices,  # 查找可剪枝的注意力头
    prune_linear_layer              # 剪枝线性层
)
```

---

## 5. 配置与日志（第61-84行）

```python
logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "google-bert/bert-base-uncased"  # 文档示例用的模型
_CONFIG_FOR_DOC = "BertConfig"  # 文档示例用的配置类
```

### 5.1 文档测试常量
这些常量用于自动生成 API 文档和示例：

```python
# TokenClassification (命名实体识别) 示例
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"
_TOKEN_CLASS_EXPECTED_OUTPUT = "['O', 'I-ORG', 'I-ORG', ...]"
_TOKEN_CLASS_EXPECTED_LOSS = 0.01

# QuestionAnswering (问答) 示例
_CHECKPOINT_FOR_QA = "deepset/bert-base-cased-squad2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 7.41

# SequenceClassification (文本分类) 示例
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "textattack/bert-base-uncased-yelp-polarity"
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
```

---

## 6. TensorFlow 权重加载函数（第86-156行）

这是一个重要的辅助函数，用于将 TensorFlow 预训练权重转换为 PyTorch 格式：

```python
def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
```

### 6.1 函数执行流程

```
TensorFlow Checkpoint
        ↓
┌───────────────────────────┐
│  1. 导入 TensorFlow        │
│  2. 列出所有变量名和形状   │
│  3. 加载每个变量的数组     │
└───────────────────────────┘
        ↓
┌───────────────────────────┐
│  4. 遍历变量，匹配参数     │
│     - 跳过优化器状态       │
│     - 转换 kernel 权重     │
│     - 处理 embedding       │
└───────────────────────────┘
        ↓
    PyTorch Model
```

### 6.2 关键代码解析

```python
# 跳过 Adam 优化器的状态变量（预训练模型不需要）
if any(
    n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", ...]
    for n in name
):
    logger.info(f"Skipping {'/'.join(name)}")
    continue
```

```python
# TensorFlow 的 kernel 对应 PyTorch 的 weight
# 但需要转置！因为 TF 和 PyTorch 的线性层权重格式不同
if scope_names[0] == "kernel" or scope_names[0] == "gamma":
    pointer = getattr(pointer, "weight")
```

```python
# TF: [out_features, in_features]
# PyTorch: [in_features, out_features]
elif m_name == "kernel":
    array = np.transpose(array)  # 关键：转置权重矩阵
```

---

## 7. 本章小结

本章我们完成了 BERT 模型源码的"序章"部分：

1. **许可证声明**：Apache 2.0 开源协议
2. **核心依赖**：PyTorch、Transformers 内部模块
3. **输出类**：规范化不同任务的模型输出
4. **TF 权重转换**：兼容原始 TensorFlow 预训练模型

下一章，我们将进入 BERT 的核心——**嵌入层（BertEmbeddings）**，这是所有输入经过的第一道处理工序。

---

*下一章预告：[bert_modeling_guide_02_embeddings.md] - 词嵌入、位置嵌入、句子嵌入的三重奏*
