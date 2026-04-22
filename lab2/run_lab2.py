import argparse
import json
import os
import random
import string
import time
import unicodedata
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


DATA_URL = "https://download.pytorch.org/tutorial/data.zip"
SEED = 2026


def set_seed(seed: int) -> None:
    # 固定随机性，保证每次实验结果可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unicode_to_ascii(s: str, allowed: str) -> str:
    # 将带重音字符规范化后过滤到允许字符集合。
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in allowed
    )


def ensure_names_data(root: Path) -> Path:
    # 若本地已存在 names 数据则直接复用，否则自动下载并解压。
    names_dir = root / "data" / "names"
    if names_dir.exists() and any(names_dir.glob("*.txt")):
        return names_dir

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "data.zip"
    print(f"[INFO] Downloading names data to: {zip_path}")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    print(f"[INFO] Extracted dataset into: {root}")
    return names_dir


@dataclass
class NamesItem:
    # seq: 字符索引序列；label_idx: 语言类别索引。
    seq: List[int]
    label_idx: int


class NamesDataset(Dataset):
    def __init__(self, names_dir: Path):
        # 0 预留给 padding，真实字符从 1 开始编号。
        self.allowed_characters = string.ascii_letters + " .,;'"
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.allowed_characters)}
        self.pad_idx = 0
        self.vocab_size = len(self.allowed_characters) + 1

        self.languages: List[str] = []
        self.items: List[NamesItem] = []

        label_files = sorted(names_dir.glob("*.txt"))
        if not label_files:
            raise FileNotFoundError(f"No .txt language files found under: {names_dir}")

        for label_file in label_files:
            language = label_file.stem
            self.languages.append(language)

        lang_to_idx = {lang: i for i, lang in enumerate(self.languages)}

        for label_file in label_files:
            language = label_file.stem
            lines = label_file.read_text(encoding="utf-8").strip().splitlines()
            for raw_name in lines:
                name = unicode_to_ascii(raw_name.strip(), self.allowed_characters)
                if not name:
                    continue
                seq = [self.char_to_idx[ch] for ch in name if ch in self.char_to_idx]
                if not seq:
                    continue
                self.items.append(NamesItem(seq=seq, label_idx=lang_to_idx[language]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        return torch.tensor(item.seq, dtype=torch.long), torch.tensor(item.label_idx, dtype=torch.long)


def collate_batch(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 按当前 batch 最大长度做右侧 padding，并返回每条序列原始长度。
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.numel() for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, : seq.numel()] = seq

    labels_t = torch.stack(labels)
    return padded, lengths, labels_t


class BaseNameClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int, num_classes: int):
        super().__init__()
        # 统一封装嵌入层与分类头，RNN/LSTM 子类只需实现特征提取。
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def _forward_features(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        features = self._forward_features(x, lengths)
        logits = self.classifier(features)
        return self.log_softmax(logits)


class RNNClassifier(BaseNameClassifier):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int, num_classes: int):
        super().__init__(vocab_size, emb_dim, hidden_size, num_classes)
        self.rnn = nn.RNN(emb_dim, hidden_size, batch_first=True)

    def _forward_features(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # pack_padded_sequence 避免模型在 padding 位置做无效计算。
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        return h_n[-1]


class LSTMClassifier(BaseNameClassifier):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int, num_classes: int):
        super().__init__(vocab_size, emb_dim, hidden_size, num_classes)
        self.lstm = nn.LSTM(emb_dim, hidden_size, batch_first=True)

    def _forward_features(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # LSTM 返回 (h_n, c_n)，分类任务通常使用最后一层 h_n。
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return h_n[-1]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, object]:
    # 评估阶段不更新参数，同时收集预测用于后续混淆矩阵。
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    all_preds: List[int] = []
    all_labels: List[int] = []

    for x, lengths, y in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        out = model(x, lengths)
        loss = criterion(out, y)

        total_loss += loss.item() * y.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    # 返回损失/精度以及原始预测序列，便于后处理分析。
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "preds": all_preds,
        "labels": all_labels,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Dict[str, List[float]]:
    # 使用 Adam + NLLLoss 训练，并按验证集准确率保存最佳权重。
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_state = None
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, lengths, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x, lengths)
            loss = criterion(out, y)
            loss.backward()
            # 梯度裁剪可降低 RNN/LSTM 训练中梯度爆炸风险。
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            pred = out.argmax(dim=1)
            running_correct += (pred == y).sum().item()
            running_total += y.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_stats = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["acc"])

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def build_confusion_matrix(preds: List[int], labels: List[int], num_classes: int) -> np.ndarray:
    # 行归一化：每一行代表该真实类别被预测到各类别的概率分布。
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for gt, pr in zip(labels, preds):
        cm[gt, pr] += 1
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return cm / row_sum


def save_curves(history: Dict[str, List[float]], out_dir: Path) -> None:
    # 按 epoch 绘制训练与验证曲线，便于观察收敛和过拟合趋势。
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("NLL Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "acc_curve.png", dpi=180)
    plt.close()


def save_confusion(cm: np.ndarray, labels: List[str], out_dir: Path) -> None:
    # 绘制归一化混淆矩阵：颜色越深表示该真实类别被预测到该列的比例越高。
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=90)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close()


def run_experiment(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    labels: List[str],
    output_root: Path,
) -> Dict[str, object]:
    # 单个模型的完整流程：训练 -> 评估 -> 导出曲线/矩阵/权重/摘要。
    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] ===== {model_name.upper()} =====")
    print(model)

    history = train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr)

    criterion = nn.NLLLoss()
    val_stats = evaluate(model, val_loader, device, criterion)
    cm = build_confusion_matrix(val_stats["preds"], val_stats["labels"], len(labels))

    torch.save(model.state_dict(), out_dir / "best.pt")
    (out_dir / "model_structure.txt").write_text(str(model), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    save_curves(history, out_dir)
    save_confusion(cm, labels, out_dir)

    summary = {
        "model": model_name,
        "epochs": epochs,
        "learning_rate": lr,
        "val_loss": val_stats["loss"],
        # val_acc 为加载最佳权重后重新评估得到的最终验证精度。
        "val_acc": val_stats["acc"],
        "best_val_acc": max(history["val_acc"]) if history["val_acc"] else 0.0,
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] {model_name} done. val_acc={summary['val_acc']:.4f}")
    return summary


def main() -> None:
    # 命令行参数集中管理，便于复现实验和超参对比。
    parser = argparse.ArgumentParser(description="Lab2: Name classification with RNN and LSTM")
    parser.add_argument("--data-root", type=str, default="data", help="Path containing names data or where data.zip will be extracted")
    parser.add_argument("--output", type=str, default="outputs", help="Output folder")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    # 自动选择 CUDA/CPU，确保脚本可跨设备运行。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    data_root = Path(args.data_root)
    names_dir = ensure_names_data(data_root)

    dataset = NamesDataset(names_dir)
    print(f"[INFO] Loaded dataset: {len(dataset)} samples, {len(dataset.languages)} classes")

    train_len = int(0.85 * len(dataset))
    val_len = len(dataset) - train_len
    # 固定随机种子划分数据，保证训练/验证集一致。
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    start = time.time()

    rnn_model = RNNClassifier(
        vocab_size=dataset.vocab_size,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        num_classes=len(dataset.languages),
    ).to(device)

    lstm_model = LSTMClassifier(
        vocab_size=dataset.vocab_size,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        num_classes=len(dataset.languages),
    ).to(device)

    # 先跑基线 RNN，再跑 LSTM；两者使用同一数据划分与训练配置。
    rnn_summary = run_experiment(
        model_name="rnn",
        model=rnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        labels=dataset.languages,
        output_root=output_root,
    )

    lstm_summary = run_experiment(
        model_name="lstm",
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        labels=dataset.languages,
        output_root=output_root,
    )

    # 汇总到单一 JSON，方便报告直接引用核心数字。
    compare = {
        "device": str(device),
        "seed": args.seed,
        "dataset_size": len(dataset),
        "num_classes": len(dataset.languages),
        "rnn": rnn_summary,
        "lstm": lstm_summary,
        # 正值表示 LSTM 验证准确率高于 RNN。
        "lstm_minus_rnn_val_acc": lstm_summary["val_acc"] - rnn_summary["val_acc"],
        "elapsed_seconds": time.time() - start,
    }

    (output_root / "comparison_summary.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print("\n[INFO] All done. Comparison summary saved to outputs/comparison_summary.json")


if __name__ == "__main__":
    main()
