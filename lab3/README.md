# Lab 3 — Seq2Seq 翻译模型（注意力机制）

## 文件说明

```
lab3/
├── seq2seq/
│   ├── seq2seq_translation.py          # 主程序：训练 + 评估 + 可视化
│   ├── seq2seq_translation_exercise.ipynb  # Jupyter Notebook 版本
│   ├── data/
│   │   └── eng-fra.txt                 # 英法平行语料
│   ├── outputs/                        # 运行结果（自动生成）
│   │   ├── loss_rnn.png                # 纯 RNN 训练损失曲线
│   │   ├── loss_attn.png               # Attention 模型训练损失曲线
│   │   ├── loss_comparison.png         # 两种模型损失对比
│   │   └── attn_*.png                  # 注意力权重热力图
│   └── *.png                           # 模型结构示意图
└── 实验报告_模板.md                     # 实验报告模板
```

## 快速开始

### 环境

```bash
conda activate speech_env
# PyTorch 2.2.0, CUDA 可用
```

### 运行

```bash
cd lab3/seq2seq
python seq2seq_translation.py
```

程序会自动完成:
1. 加载并预处理英法翻译数据（11,445 个翻译对）
2. 训练**纯 RNN Seq2Seq 模型**（200 epoch）
3. 训练**基于 Bahdanau 注意力的 Seq2Seq 模型**（200 epoch）
4. 生成损失对比图、翻译结果对比、注意力热力图
5. 所有结果保存到 `outputs/` 目录

### 配置

在 `seq2seq_translation.py` 顶部可调整参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 128 | 词嵌入 & 隐藏层维度 |
| `batch_size` | 32 | 批次大小 |
| `n_epochs` | 200 | 训练轮数 |
| `MAX_LENGTH` | 10 | 最大句子长度 |

## 模型架构

### Encoder（编码器）
```
Embedding → Dropout → GRU → (outputs, hidden)
```

### DecoderRNN（无注意力解码器）
```
Embedding → GRU → Linear → softmax
仅使用编码器最终隐藏状态作为上下文
```

### AttnDecoderRNN（注意力解码器）
```
Embedding ──→ [concat] → GRU → Linear → softmax
                ↑
Attention(query, encoder_outputs) → context
    ├─ Wa(encoder_outputs)
    ├─ Ua(decoder_hidden)
    └─ Va(tanh(Wa + Ua)) → softmax → weighted sum
```

## 关键实现

### Bahdanau 注意力机制

```python
# 能量分数：Va * tanh(Wa * keys + Ua * query)
scores = self.Va(torch.tanh(self.Wa(keys) + self.Ua(query)))

# 软注意力权重
attn_weights = F.softmax(scores, dim=1)

# 加权上下文向量
context = torch.bmm(attn_weights.transpose(1, 2), keys)
```

## 参考资料

- [Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2015)](https://arxiv.org/abs/1409.0473)
- [PyTorch NLP Translation Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
