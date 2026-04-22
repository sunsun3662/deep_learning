# Lab2：循环神经网络姓名分类（RNN vs LSTM）

本实验基于 PyTorch 完成字符级姓名分类任务，比较普通 RNN 与 LSTM 在同一数据集上的表现。

## 1. 实验目标

- 理解并实现字符级序列分类任务。
- 使用 RNN 完成姓名语言分类。
- 使用 LSTM 完成同任务并与 RNN 对比。
- 产出可用于实验报告的结构图、训练曲线与混淆矩阵。

## 2. 目录结构

```text
lab2/
  README.md
  run_lab2.py
  实验报告.md
  rnn_pytorch_tutorial.ipynb
  char_rnn_classification_tutorial.ipynb
  data/
    names/                 # 自动下载与解压后生成
  outputs/
    rnn/
      model_structure.txt
      history.json
      summary.json
      best.pt
      loss_curve.png
      acc_curve.png
      confusion_matrix.png
    lstm/
      model_structure.txt
      history.json
      summary.json
      best.pt
      loss_curve.png
      acc_curve.png
      confusion_matrix.png
    comparison_summary.json
```

## 3. 环境要求

- Python 3.8+
- torch
- numpy
- matplotlib

建议使用 CUDA 环境提升训练速度。

## 4. 快速开始

推荐在 `lab2` 目录下执行：

```bash
D:/anaconda3/envs/speech_env/python.exe run_lab2.py --data-root data --output outputs --epochs 20 --batch-size 256 --hidden-size 128 --emb-dim 64 --lr 0.002
```

如果首次运行未找到数据，脚本会自动下载并解压 `https://download.pytorch.org/tutorial/data.zip`。

## 5. 参数说明

`run_lab2.py` 支持如下参数：

- `--data-root`：数据根目录，默认 `data`
- `--output`：输出目录，默认 `outputs`
- `--epochs`：训练轮次，默认 `12`
- `--batch-size`：批大小，默认 `256`
- `--emb-dim`：字符嵌入维度，默认 `64`
- `--hidden-size`：隐藏层维度，默认 `128`
- `--lr`：学习率，默认 `0.002`
- `--seed`：随机种子，默认 `2026`

## 6. 输出结果说明

每个模型目录（`outputs/rnn`、`outputs/lstm`）包含：

- `model_structure.txt`：模型结构文本
- `history.json`：每轮训练/验证 loss 与 accuracy
- `summary.json`：核心指标汇总（val_loss、val_acc、参数量等）
- `best.pt`：最优权重
- `loss_curve.png`：损失曲线
- `acc_curve.png`：准确率曲线
- `confusion_matrix.png`：归一化混淆矩阵

总对比文件：

- `outputs/comparison_summary.json`：包含 RNN/LSTM 指标对比、准确率差值、总耗时

## 7. 已完成实验配置（报告使用）

当前报告对应的实验设置为：

- Epochs = 20
- Batch size = 256
- Learning rate = 0.002
- Hidden size = 128
- Embedding dim = 64
- Seed = 2026

## 8. 常见问题

1. 直接在工作区根目录运行脚本出现路径相关错误
- 解决：进入 `lab2` 目录运行，或显式传入 `--data-root lab2/data --output lab2/outputs`。

2. 首次运行下载数据失败
- 解决：检查网络后重试；也可手动下载 `data.zip` 并解压到 `lab2/data/`。

3. CUDA 不可用
- 脚本会自动退回 CPU，可正常运行但耗时更长。

## 9. 相关文件

- 实验主脚本：`run_lab2.py`
- 实验报告：`实验报告.md`
- 原始教程 notebook：`rnn_pytorch_tutorial.ipynb`
- 参考教程 notebook：`char_rnn_classification_tutorial.ipynb`
