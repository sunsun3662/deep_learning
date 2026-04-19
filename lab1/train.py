import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import build_model

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# CIFAR10 标签顺序固定，便于输出可读的分类别准确率。
CIFAR10_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass
class EvalResult:
    # 单次评估返回结构：总体 loss、总体 acc、可选的分类别 acc。
    loss: float
    acc: float
    per_class_acc: Optional[Dict[str, float]] = None


def set_seed(seed: int) -> None:
    # 固定随机种子，提升多次实验的可复现性。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    # 优先使用 GPU，若不可用则自动退回 CPU。
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    # 统计可训练参数量，用于报告中的模型规模对比。
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    train_limit: int,
    val_limit: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # 训练集增强：随机裁剪 + 随机翻转，提升泛化能力。
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    # train_aug 用于训练；train_eval 用于验证（同一批样本，不同变换策略）。
    train_aug = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    train_eval = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_tf)

    total = len(train_aug)
    val_size = int(total * val_split)
    # 使用固定种子打乱，保证每次训练/验证划分一致。
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=gen).tolist()
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    if train_limit > 0:
        train_idx = train_idx[:train_limit]
    if val_limit > 0:
        val_idx = val_idx[:val_limit]

    train_set = Subset(train_aug, train_idx)
    val_set = Subset(train_eval, val_idx)

    # 训练集开启 shuffle；验证/测试集关闭 shuffle。
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    # 标准训练流程：前向 -> 计算损失 -> 反向传播 -> 参数更新。
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_count += batch_size

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[Tuple[str, ...]] = None,
) -> EvalResult:
    # 评估阶段不计算梯度，减少显存占用并提升速度。
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    per_class_correct = None
    per_class_total = None
    if class_names is not None:
        # 记录每个类别的命中数与总样本数，后续计算分类别准确率。
        per_class_correct = {name: 0 for name in class_names}
        per_class_total = {name: 0 for name in class_names}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_count += batch_size

            if class_names is not None and per_class_correct is not None and per_class_total is not None:
                for label, pred in zip(labels, preds):
                    cname = class_names[int(label)]
                    per_class_total[cname] += 1
                    if pred == label:
                        per_class_correct[cname] += 1

    per_class_acc = None
    if class_names is not None and per_class_correct is not None and per_class_total is not None:
        per_class_acc = {
            name: per_class_correct[name] / max(per_class_total[name], 1) for name in class_names
        }

    return EvalResult(
        loss=total_loss / max(total_count, 1),
        acc=total_correct / max(total_count, 1),
        per_class_acc=per_class_acc,
    )


def save_curves(history: Dict[str, List[float]], output_dir: str) -> None:
    # 曲线图是实验报告核心材料：loss 曲线 + acc 曲线。
    if not HAS_MATPLOTLIB:
        print("matplotlib 不可用，跳过曲线图保存。")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "acc_curve.png"), dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    # 命令行参数统一管理，便于单模型与批量训练复用。
    parser = argparse.ArgumentParser(description="CIFAR10 classification experiments")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn", "resnet", "densenet", "mobilenet", "res2net"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--train-limit", type=int, default=0, help="调试用，0表示不限制")
    parser.add_argument("--val-limit", type=int, default=0, help="调试用，0表示不限制")
    parser.add_argument("--eval-test", action="store_true", help="训练后在测试集上评估")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    out_dir = os.path.join(args.output_root, args.model)
    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
    )

    # 根据模型名动态构建网络，确保四类模型训练代码一致。
    model = build_model(args.model, num_classes=10).to(device)
    with open(os.path.join(out_dir, "model_structure.txt"), "w", encoding="utf-8") as f:
        # 保存网络结构文本，可直接粘贴到实验报告中。
        f.write(str(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_result.loss)
        history["val_acc"].append(val_result.acc)

        # 按验证集准确率保存最优权重，避免最后一轮退化影响测试结果。
        if val_result.acc > best_val_acc:
            best_val_acc = val_result.acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_result.loss:.4f} val_acc={val_result.acc:.4f}"
        )

    save_curves(history, out_dir)

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 测试前加载验证集最优模型。
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    summary = {
        "model": args.model,
        "device": str(device),
        "epochs": args.epochs,
        "params": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
    }

    if args.eval_test:
        # 输出测试集总体指标与分类别指标，便于结果分析与可视化。
        test_result = evaluate(model, test_loader, criterion, device, class_names=CIFAR10_CLASSES)
        summary["test_loss"] = test_result.loss
        summary["test_acc"] = test_result.acc

        with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"test_loss": test_result.loss, "test_acc": test_result.acc}, f, indent=2)

        with open(os.path.join(out_dir, "per_class_accuracy.json"), "w", encoding="utf-8") as f:
            json.dump(test_result.per_class_acc, f, indent=2)

        print(f"Test: loss={test_result.loss:.4f}, acc={test_result.acc:.4f}")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        # 汇总关键信息：模型规模、最佳轮次与准确率。
        json.dump(summary, f, indent=2)

    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
