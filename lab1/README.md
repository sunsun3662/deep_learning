# Lab1 第一次实验（卷积神经网络）

本项目用于完成 CIFAR10 分类实验，包含老师原始 CNN 基线、个人实现的经典卷积网络以及 Res2Net 扩展实验。

## 1. 项目内容

已实现模型：

- CNN（老师原始版本，LeNet 风格）
- ResNet（个人实现 BasicBlock + Shortcut）
- DenseNet（个人实现 DenseBlock + Transition）
- MobileNet（个人实现深度可分离卷积）
- Res2Net（扩展，个人实现多尺度残差块）

主要脚本：

- `models.py`：所有模型定义
- `train.py`：单模型训练与评估
- `run_all.py`：批量训练入口
- `实验报告.md`：已生成的实验报告（20 epoch + Res2Net）

## 2. 环境要求

- Python 3.8+
- torch
- torchvision
- matplotlib（用于导出曲线图）

建议使用 CUDA 环境训练，可明显缩短实验时间。

## 3. 快速开始

在 `lab1` 目录下运行。

单模型训练示例：

```bash
D:/anaconda3/envs/speech_env/python.exe train.py --model cnn --epochs 20 --batch-size 128 --num-workers 0 --eval-test
```

`--model` 可选值：

- `cnn`
- `resnet`
- `densenet`
- `mobilenet`
- `res2net`

## 4. 批量训练

训练四个基础模型（不含 Res2Net）：

```bash
D:/anaconda3/envs/speech_env/python.exe run_all.py --epochs 20 --batch-size 128 --num-workers 0
```

训练四个基础模型 + Res2Net：

```bash
D:/anaconda3/envs/speech_env/python.exe run_all.py --epochs 20 --batch-size 128 --num-workers 0 --include-res2net
```

## 5. 输出结果说明

每个模型会生成到 `outputs/<model_name>/`，包含：

- `model_structure.txt`：网络结构文本
- `history.json`：每轮训练/验证指标
- `loss_curve.png`：loss 曲线图
- `acc_curve.png`：准确率曲线图
- `best.pt`：验证集最优权重
- `test_metrics.json`：测试集 loss 与 acc
- `per_class_accuracy.json`：测试集各类别准确率
- `summary.json`：参数量、最佳轮次、最佳验证准确率、测试准确率

## 6. 报告对应关系

实验报告中每个模型需要的三类核心材料对应如下：

- 网络结构：`outputs/<model_name>/model_structure.txt`
- loss 曲线：`outputs/<model_name>/loss_curve.png`
- 准确率曲线：`outputs/<model_name>/acc_curve.png`

可选扩展（Res2Net）材料：

- `outputs/res2net/model_structure.txt`
- `outputs/res2net/loss_curve.png`
- `outputs/res2net/acc_curve.png`



## 8. 常见问题

如果训练长时间无输出，可优先尝试：

- 将 `--num-workers` 设为 `0`（Windows 下更稳定）
- 确认 CUDA 可用（若不可用则会自动使用 CPU）
- 先用 `--epochs 1 --train-limit 1024 --val-limit 256` 做流程验证
