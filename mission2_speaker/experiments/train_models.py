"""Single training entry point: fixed 10 epochs, or a controlled window suite."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m2.experiment_workflow import MODEL_PROFILES, WINDOW_CONDITIONS, train_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", choices=[*MODEL_PROFILES, "all"], default="resnet50")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-fraction", type=float, default=0.2)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--resnet-init", type=Path, help="Optional local torchvision ImageNet ResNet-50 .pth")
    p.add_argument("--window-suite", action="store_true", help="Four ResNet input conditions, same split/seed/budget")
    args = p.parse_args()
    common = dict(training_root=args.training_root, epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, seed=args.seed, dev_fraction=args.dev_fraction, device=args.device,
                  resnet_init=args.resnet_init)
    if args.window_suite:
        if args.model != "resnet50":
            p.error("--window-suite requires --model resnet50")
        for condition in WINDOW_CONDITIONS:
            train_model("resnet50", output_dir=args.output_dir / condition["name"],
                        **common, **{k: v for k, v in condition.items() if k != "name"})
    else:
        names = list(MODEL_PROFILES) if args.model == "all" else [args.model]
        for name in names:
            train_model(name, output_dir=args.output_dir / name, **common)


if __name__ == "__main__":
    main()
