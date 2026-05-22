import argparse
import subprocess
import sys

BASE_MODELS = ["cnn", "resnet", "densenet", "mobilenet"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all CIFAR10 models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-res2net", action="store_true", help="是否将 Res2Net 一并纳入批量训练")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = BASE_MODELS + (["res2net"] if args.include_res2net else [])
    for model in models:
        cmd = [
            sys.executable,
            "train.py",
            "--model",
            model,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--num-workers",
            str(args.num_workers),
            "--data-dir",
            args.data_dir,
            "--output-root",
            args.output_root,
            "--seed",
            str(args.seed),
            "--eval-test",
        ]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
