"""멘토링 질의를 뒷받침할 두 가지 측정.

Q3  조각 길이별 정확도 — 짧은 조각이 성능 한계의 원인인가?
Q5  두 모델의 오차 겹침 — 앙상블에 여지가 있는가?

    python -m m1.analysis --ckpt <resnet.pt> --ckpt <w2v2.pt>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ._console import ensure_utf8_stdout
from .aggregate import GENDER_OUTPUT, call_label, gender_to_target
from .cache import CacheIndex
from .datasets import samples_from_rows
from .evaluate import call_probabilities, predict_segment_probs, truth_from_samples
from .models import load_checkpoint

# 조각 길이 구간 (초). 관측된 분포가 p50 1.55s / p90 4.71s 라 그 주변을 촘촘히 나눈다.
BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, float("inf"))]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mission 1 오류 분석")
    p.add_argument("--ckpt", type=Path, action="append", required=True)
    p.add_argument("--val-cache", type=Path, default=Path("cache/val"))
    p.add_argument("--out", type=Path, default=Path("mission1_gender/reports/analysis.json"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args(argv)


def bucket_label(lo: float, hi: float) -> str:
    return f"{lo:.0f}s+" if hi == float("inf") else f"{lo:.0f}-{hi:.0f}s"


def length_analysis(samples, probs) -> list[dict]:
    """조각 길이 구간별 정확도."""
    seconds = np.array([s.row.length / 8000.0 for s in samples])
    gold = np.array([s.target for s in samples])
    pred = (probs >= 0.5).astype(int)
    correct = pred == gold

    rows = []
    for lo, hi in BUCKETS:
        mask = (seconds >= lo) & (seconds < hi)
        if not mask.any():
            continue
        rows.append({
            "bucket": bucket_label(lo, hi),
            "n": int(mask.sum()),
            "share": round(float(mask.mean()), 4),
            "accuracy": round(float(correct[mask].mean()), 4),
            "mean_seconds": round(float(seconds[mask].mean()), 2),
        })
    return rows


def call_predictions(samples, probs) -> dict[str, str]:
    return {cid: call_label(p) for cid, p in call_probabilities(samples, probs).items()}


def overlap_analysis(truth_labels, preds_a, preds_b, probs_a, probs_b) -> dict:
    """두 모델의 통화 단위 오차가 겹치는지, 평균 앙상블이 이득인지."""
    calls = sorted(truth_labels)
    wrong_a = {c for c in calls if preds_a[c] != truth_labels[c]}
    wrong_b = {c for c in calls if preds_b[c] != truth_labels[c]}

    both = wrong_a & wrong_b
    only_a = wrong_a - wrong_b
    only_b = wrong_b - wrong_a
    n = len(calls)

    # 두 모델의 통화 확률을 평균낸 단순 앙상블
    ensemble_correct = sum(
        1 for c in calls
        if call_label((probs_a[c] + probs_b[c]) / 2.0) == truth_labels[c]
    )

    return {
        "n_calls": n,
        "wrong_a": len(wrong_a),
        "wrong_b": len(wrong_b),
        "wrong_both": len(both),
        "wrong_only_a": len(only_a),
        "wrong_only_b": len(only_b),
        "overlap_ratio": round(len(both) / max(1, min(len(wrong_a), len(wrong_b))), 4),
        "accuracy_a": round(1 - len(wrong_a) / n, 4),
        "accuracy_b": round(1 - len(wrong_b) / n, 4),
        "oracle_accuracy": round(1 - len(both) / n, 4),
        "mean_ensemble_accuracy": round(ensemble_correct / n, 4),
    }


def main(argv=None) -> int:
    ensure_utf8_stdout()
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index = CacheIndex.load(args.val_cache)
    samples = samples_from_rows(index.rows)
    truth = truth_from_samples(samples)
    truth_labels = {c: GENDER_OUTPUT[gender_to_target(g)] for c, g in truth.items()}

    results = {"length": {}, "per_model": {}}
    call_probs: dict[str, dict[str, float]] = {}

    for path in args.ckpt:
        model, branch, cfg, _ = load_checkpoint(path, device=device)
        print(f"=== {path.stem} ({branch}) ===", flush=True)

        probs = predict_segment_probs(
            model, index, samples, cfg, branch, device,
            batch_size=args.batch_size, mode="sliding", num_workers=args.num_workers,
        )

        rows = length_analysis(samples, probs)
        results["length"][path.stem] = rows
        for r in rows:
            print("  %-7s n=%6d (%4.1f%%)  조각 acc %.4f" % (
                r["bucket"], r["n"], r["share"] * 100, r["accuracy"]), flush=True)

        cp = call_probabilities(samples, probs)
        call_probs[path.stem] = cp
        results["per_model"][path.stem] = {"branch": branch}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    names = [p.stem for p in args.ckpt]
    if len(names) >= 2:
        a, b = names[0], names[1]
        preds_a = {c: call_label(v) for c, v in call_probs[a].items()}
        preds_b = {c: call_label(v) for c, v in call_probs[b].items()}
        results["overlap"] = overlap_analysis(
            truth_labels, preds_a, preds_b, call_probs[a], call_probs[b]
        )
        results["overlap"]["model_a"] = a
        results["overlap"]["model_b"] = b
        print("\n=== 오차 겹침 ===", flush=True)
        print(json.dumps(results["overlap"], ensure_ascii=False, indent=2), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
