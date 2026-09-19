"""N 개 체크포인트의 Validation 오류 겹침과 앙상블 상한.

    python -m m1.overlap --ckpt a.pt --ckpt b.pt --ckpt c.pt

각 체크포인트의 통화 확률은 reports/call_probs_<stem>.json 에 저장/재사용하고,
임계값은 체크포인트에 보정된 값을 쓴다. 튜닝 없는 고정 평균 앙상블과 oracle
(매 통화마다 맞는 모델을 고를 수 있을 때의 상한)만 보고한다.

실측 (2026-09-07, resnet / w2v2 / audeering):
  셋 다 오답 46 통화(1.26%) -> 3-way oracle 0.9874. 고정 평균은 어느 조합도
  w2v2 단독(0.9835)을 넘지 못했다. 그 46 통화가 이 데이터의 오답 바닥이다.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from ._console import ensure_utf8_stdout
from .aggregate import gender_to_target
from .cache import CacheIndex
from .datasets import samples_from_rows
from .evaluate import (
    call_probabilities,
    predict_segment_probs,
    suggested_workers,
    truth_from_samples,
)
from .models import decision_threshold, load_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mission 1 오류 겹침")
    p.add_argument("--ckpt", type=Path, action="append", required=True)
    p.add_argument("--val-cache", type=Path, default=Path("cache/val"))
    p.add_argument("--reports", type=Path, default=Path("mission1_gender/reports"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--refresh", action="store_true", help="저장된 확률을 무시하고 다시 추론")
    p.add_argument("--use-ckpt-threshold", action="store_true",
                   help="(연구용) 체크포인트 저장 임계값 사용. 기본은 규정대로 0.5 고정")
    return p.parse_args(argv)


def call_probs_for(ckpt: Path, index, samples, calls, args, device) -> tuple[dict, float]:
    """(call_id -> P(여), 임계값). 저장된 확률이 통화 집합과 일치하면 재사용."""
    cache = args.reports / f"call_probs_{ckpt.stem}.json"
    if cache.exists() and not args.refresh:
        cp = json.loads(cache.read_text(encoding="utf-8"))
        if set(cp) == set(calls):
            _, _, _, payload = load_checkpoint(ckpt, device="cpu")
            print(f"{ckpt.stem}: 저장된 확률 재사용", flush=True)
            return cp, decision_threshold(payload, use_checkpoint=args.use_ckpt_threshold)

    model, branch, cfg, payload = load_checkpoint(ckpt, device=device)
    workers = suggested_workers(branch)
    print(f"{ckpt.stem} ({branch}): 추론 중 (workers={workers})", flush=True)
    probs = predict_segment_probs(
        model, index, samples, cfg, branch, device,
        batch_size=args.batch_size, mode="sliding", num_workers=workers,
    )
    cp = call_probabilities(samples, probs)
    args.reports.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(cp, indent=1), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cp, decision_threshold(payload, use_checkpoint=args.use_ckpt_threshold)


def main(argv=None) -> int:
    ensure_utf8_stdout()
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index = CacheIndex.load(args.val_cache)
    samples = samples_from_rows(index.rows)
    truth = truth_from_samples(samples)
    calls = sorted(truth)
    gold = np.array([gender_to_target(truth[c]) for c in calls])

    names = [c.stem for c in args.ckpt]
    probs, thr = {}, {}
    for ckpt in args.ckpt:
        cp, t = call_probs_for(ckpt, index, samples, calls, args, device)
        probs[ckpt.stem] = np.array([cp[c] for c in calls])
        thr[ckpt.stem] = t

    wrong = {k: ((probs[k] >= thr[k]).astype(int) != gold) for k in names}
    n = len(calls)
    lines = [f"Validation {n}통화"]
    for k in names:
        lines.append(f"  {k:16s} t={thr[k]:.3f}  acc {1 - wrong[k].mean():.4f}  오답 {int(wrong[k].sum())}")

    pairs = {}
    lines.append("\n쌍별 겹침 (둘 다 오답 / 앞만 / 뒤만 / oracle)")
    for a, b in combinations(names, 2):
        both = wrong[a] & wrong[b]
        pairs[f"{a}+{b}"] = {
            "both": int(both.sum()),
            "only_a": int((wrong[a] & ~wrong[b]).sum()),
            "only_b": int((~wrong[a] & wrong[b]).sum()),
            "oracle": round(float(1 - both.mean()), 4),
        }
        r = pairs[f"{a}+{b}"]
        lines.append(f"  {a:14s}+{b:16s} {r['both']:3d} / {r['only_a']:3d} / {r['only_b']:3d} / oracle {r['oracle']:.4f}")

    all_wrong = np.logical_and.reduce([wrong[k] for k in names])
    lines.append(f"\n전부 오답: {int(all_wrong.sum())}통화 ({all_wrong.mean() * 100:.2f}%)"
                 f"  ->  {len(names)}-way oracle {1 - all_wrong.mean():.4f}")

    ensembles = {}
    lines.append("\n고정 평균 앙상블 (임계값 0.5, 튜닝 없음)")
    for r in range(2, len(names) + 1):
        for combo in combinations(names, r):
            avg = np.mean([probs[k] for k in combo], axis=0)
            acc = float(((avg >= 0.5).astype(int) == gold).mean())
            ensembles["+".join(combo)] = round(acc, 4)
            lines.append(f"  {'+'.join(combo):40s} {acc:.4f}")

    text = "\n".join(lines)
    print(text)

    out = args.reports / ("overlap_" + "_".join(names) + ".json")
    out.write_text(json.dumps({
        "n_calls": n,
        "thresholds": {k: thr[k] for k in names},
        "accuracy": {k: round(float(1 - wrong[k].mean()), 4) for k in names},
        "n_wrong": {k: int(wrong[k].sum()) for k in names},
        "pairs": pairs,
        "all_wrong": int(all_wrong.sum()),
        "all_way_oracle": round(float(1 - all_wrong.mean()), 4),
        "fixed_mean_ensembles": ensembles,
        "all_wrong_calls": [c for c, w in zip(calls, all_wrong) if w],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
