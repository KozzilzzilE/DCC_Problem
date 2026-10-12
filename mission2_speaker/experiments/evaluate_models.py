"""Evaluate three local classifiers and seven equal-weight combinations."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2.experiment_workflow import evaluate_models


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-root", type=Path, required=True)
    p.add_argument("--checkpoints-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    a = p.parse_args()
    for row in evaluate_models(a.evaluation_root, a.checkpoints_dir, a.output_dir, a.batch_size, a.device):
        print(" + ".join(row['models']), f"Accuracy={row['accuracy'] * 100:.4f}%, Macro F1={row['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
