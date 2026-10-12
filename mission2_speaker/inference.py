"""
Mission 2 (Speaker Classification: Dispatcher vs Caller) Standalone CLI Script.

Execution Example:
    python inference.py --audio_dir <wav_dir> --label_dir <json_dir> \
                        --ckpt_path checkpoints_final/ --output ./outputs/mission2.csv

Final setting: 3 equal model weights, threshold 0.50.
Validation (111,947 utterances, FP32, batch 64): 103,511 correct = 92.46%; Macro F1 0.9242.
Historical single/2-model comparisons used earlier checkpoints; see README.

The CLI prints elapsed prediction time for the current invocation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_CKPT = HERE / "checkpoints_final"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mission 2 Speaker Classification Inference Script")
    p.add_argument("--audio_dir", required=True, help="Directory containing audio (.wav) files")
    p.add_argument("--label_dir", required=True, help="Directory containing JSON label/transcript files")
    p.add_argument("--ckpt_path", default=str(DEFAULT_CKPT), help="Checkpoint file (.pt) or directory path")
    p.add_argument("--output", required=True, help="Path for output submission CSV file")
    p.add_argument("--ensemble_mode", default="3model", choices=["auto", "3model", "2model"],
                   help="Default 3model requires all final checkpoints. Use auto explicitly for a single checkpoint or automatic selection.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Utterances per model forward pass within a call (default: 64)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not Path(args.ckpt_path).exists():
        raise FileNotFoundError(
            f"Final checkpoints are missing: {args.ckpt_path}. "
            "Place the final checkpoint files in checkpoints_final, "
            "or specify --ckpt_path explicitly."
        )

    from m2.infer import predict_directory

    started = time.perf_counter()
    df = predict_directory(
        args.audio_dir,
        args.label_dir,
        args.ckpt_path,
        ensemble_mode=args.ensemble_mode,
        batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - started

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    n = len(df)
    per_item = elapsed * 1000 / max(1, n)
    print(f"[Completed] Output saved to {out} ({n:,} utterances processed in {elapsed:.2f}s, {per_item:.2f} ms/sample)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
