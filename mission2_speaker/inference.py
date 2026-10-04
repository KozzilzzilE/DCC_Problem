"""
Mission 2 (Speaker Classification: Dispatcher vs Caller) Standalone CLI Script.

Execution Example:
    python inference.py --audio_dir <wav_dir> --label_dir <json_dir> \
                        --ckpt_path checkpoints/ --output ./outputs/mission2.csv

Computational Efficiency Metrics:
- Total Parameters: 31.88 M (Active: 31.88 M)
  * ReDimNet2-B2: 2.57 M, ECAPA-TDNN: 5.80 M, AudioResNet-50: 23.50 M
- Benchmark Environment: NVIDIA GeForce RTX 3060 (6GB VRAM), Intel Core i7-12700H
- Average Inference Latency: ~11.50 ms / sample (RTF: 0.0038, 260+ utts/sec)
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

DEFAULT_CKPT = HERE / "checkpoints"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mission 2 Speaker Classification Inference Script")
    p.add_argument("--audio_dir", required=True, help="Directory containing audio (.wav) files")
    p.add_argument("--label_dir", required=True, help="Directory containing JSON label/transcript files")
    p.add_argument("--ckpt_path", default=str(DEFAULT_CKPT), help="Checkpoint file (.pt) or directory path")
    p.add_argument("--output", required=True, help="Path for output submission CSV file")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from m2.infer import predict_directory

    started = time.perf_counter()
    df = predict_directory(args.audio_dir, args.label_dir, args.ckpt_path)
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
