# BERT 模型源码解析 - 第六章：BertIntermediate 与 BertOutput（前馈网络）

> 源码位置：transformers/models/bert/modeling_bert.py (第529-556行)

## 1. 开篇：Transformer 的"思考"层

在自注意力之后，每个 Transformer 层还包含一个**前馈神经网络（Feed-Forward Network, FFN）**。这个网络负责对信息进行非线性变换和特征整合。

**FFN 的结构：**
```
输入 [batch, seq, 768]
    ↓
Linear(768 → 3072)  ← 升维 4 倍
    ↓
Activation (GELU)   ← 非线性激活
    ↓
Linear(3072 → 768)  ← 降维回来
    ↓
输出 [batch, seq, 768]
```

---

## 2. BertIntermediate 类详解（升维层）

```python
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
```

### 2.1 组件说明

| 组件 | BERT-base | BERT-large |
|------|-----------|------------|
| `hidden_size` | 768 | 1024 |
| `intermediate_size` | 3072 (4×768) | 4096 (4×1024) |
| `hidden_act` | "gelu" | "gelu" |

### 2.2 激活函数选择

```python
if isinstance(config.hidden_act, str):
    self.intermediate_act_fn = ACT2FN[config.hidden_act]
else:
    self.intermediate_act_fn = config.hidden_act
```

**支持两种方式：**
1. **字符串指定**：`config.hidden_act = "gelu"` → 从 `ACT2FN` 字典获取
2. **直接传入**：`config.hidden_act = nn.GELU()` → 直接使用

**ACT2FN 字典示例：**
```python
ACT2FN = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
    # ...
}
```

### 2.3 GELU 激活函数

BERT 使用 **GELU（Gaussian Error Linear Unit）** 而非 ReLU：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

**GELU vs ReLU 对比：**

```
        GELU                    ReLU
     ╱‾‾‾‾‾‾‾‾‾‾‾           ╱‾‾‾‾‾‾‾
    ╱                       ╱
───┼───→               ───┼───→
  ╱                       ╱
 ╱                       ╱
```

**GELU 的优势：**
- 在零点附近更平滑
- 负值区域有非零输出
- 通常在 NLP 任务中表现更好

### 2.4 前向传播

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states
```

**数据流：**
```
[batch, seq, 768] → Linear → [batch, seq, 3072] → GELU → [batch, seq, 3072]
```

---

## 3. BertOutput 类详解（降维层）

```python
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
```

### 3.1 组件说明

| 组件 | 维度变化 | 作用 |
|------|----------|------|
| `dense` | 3072 → 768 | 降维回原始大小 |
| `LayerNorm` | 768 | 层归一化 |
| `dropout` | - | 正则化 |

### 3.2 前向传播

```python
def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states
```

**数据流图：**
```
intermediate_output [batch, seq, 3072]
         │
         ↓
    dense (3072 → 768)
         │
         ↓
      dropout
         │
         ↓
    ┌────┴────┐
    │   +     │ ← input_tensor (残差连接，来自注意力输出)
    └────┬────┘
         │
         ↓
    LayerNorm
         │
         ↓
    output [batch, seq, 768]
```

---

## 4. 完整的前馈网络（FFN）

将 BertIntermediate 和 BertOutput 组合起来：

```python
def feed_forward_chunk(self, attention_output):
    intermediate_output = self.intermediate(attention_output)  # 升维 + 激活
    layer_output = self.output(intermediate_output, attention_output)  # 降维 + 残差 + LayerNorm
    return layer_output
```

**完整数据流：**
```
attention_output [batch, seq, 768]
         │
         ↓
┌─────────────────────────────────────┐
│       BertIntermediate              │
│  ┌─────────────────────────────┐   │
│  │ Linear(768 → 3072)          │   │
│  │ GELU()                      │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
         │
         ↓
intermediate_output [batch, seq, 3072]
         │
         ↓
┌─────────────────────────────────────┐
│         BertOutput                  │
│  ┌─────────────────────────────┐   │
│  │ Linear(3072 → 768)          │   │
│  │ Dropout()                   │   │
│  │ x + attention_output        │   │  ← 残差连接
│  │ LayerNorm()                 │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
         │
         ↓
layer_output [batch, seq, 768]
```

---

## 5. 为什么先升维再降维？

**直觉理解：**
- 升维：在高维空间中进行更复杂的特征组合
- 降维：压缩信息，提取关键特征

**类比：**
```
想象你在读书：
1. 升维：把一段话展开成详细的思考
2. 激活：进行非线性推理
3. 降维：总结成核心观点
```

**数学视角：**
- 升维增加了网络的"表达能力"
- 中间的非线性激活使网络能学习复杂函数
- 降维确保输出维度一致，便于残差连接

---

## 6. 参数量分析

以 BERT-base 为例：

### BertIntermediate
```
Linear(768, 3072)
- 权重: 768 × 3072 = 2,359,296
- 偏置: 3072
- 总计: 2,362,368
```

### BertOutput
```
Linear(3072, 768)
- 权重: 3072 × 768 = 2,359,296
- 偏置: 768
- LayerNorm: 768 × 2 = 1,536
- 总计: 2,361,600
```

### 每层 FFN 总参数
```
2,362,368 + 2,361,600 = 4,723,968 ≈ 4.7M
```

### BERT-base 12 层 FFN 总参数
```
4.7M × 12 ≈ 56.6M
```

---

## 7. 与注意力层的对比

| 组件 | 参数量（BERT-base 每层） |
|------|-------------------------|
| 自注意力 (Q, K, V + Output) | ~2.4M |
| FFN (Intermediate + Output) | ~4.7M |
| **总计** | **~7.1M** |

**观察：**
- FFN 的参数量是自注意力的约 2 倍
- 这在所有 Transformer 模型中都很常见

---

## 8. 本章小结

本章介绍了 BERT 的前馈网络：

1. **BertIntermediate**：
   - 线性升维：768 → 3072（4 倍）
   - GELU 激活函数

2. **BertOutput**：
   - 线性降维：3072 → 768
   - Dropout 正则化
   - 残差连接 + LayerNorm

3. **设计思想**：
   - 升维增加表达能力
   - 非线性激活捕捉复杂模式
   - 残差连接稳定训练

下一章，我们将把这些组件组装成 **BertLayer**——一个完整的 Transformer 层。

---

*下一章预告：[bert_modeling_guide_07_layer.md] - 组装一个完整的 Transformer 层*
