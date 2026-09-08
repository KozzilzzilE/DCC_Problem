"""결정 임계값 보정 — dev 에서 정해 체크포인트에 저장한다.

    python -m m1.calibrate --ckpt mission1_gender/ckpt/resnet_full.pt

조각 확률을 통화 단위로 평균하면 0.5 가 최적이 아닐 수 있다. 실측으로
ResNet50 은 0.515 가 dev 최적이고, 이를 Validation 에 적용하면 0.9791 ->
0.9808 이 된다 (계산 비용 0).

**Validation 으로 임계값을 고르면 안 된다.** 평가 데이터에 맞춘 값은 실제
Test 에서 재현되지 않는다 (Validation 최적값 0.540 을 썼다면 +0.28%p 로
보이지만, dev 에서 정직하게 고른 값의 실제 이득은 +0.16%p 다). 이 스크립트는
학습 때 쓴 것과 같은 dev 분할만 사용한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ._console import ensure_utf8_stdout
from .aggregate import gender_to_target
from .cache import CacheIndex
from .evaluate import call_probabilities, predict_segment_probs, truth_from_samples
from .models import load_checkpoint
from .models.factory import DEFAULT_THRESHOLD, write_threshold

# 0.5 에서 크게 벗어난 임계값은 보정이 아니라 과적합이다. 범위를 좁게 잡는다.
GRID = np.arange(0.35, 0.66, 0.005)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mission 1 결정 임계값 보정")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--train-cache", type=Path, default=Path("cache/train"))
    p.add_argument("--dev-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=None,
                   help="기본값은 갈래에 맞춰 자동 (resnet 0 / w2v2 4)")
    p.add_argument("--dry-run", action="store_true", help="체크포인트를 수정하지 않는다")
    return p.parse_args(argv)


def accuracy_at(call_probs: dict[str, float], gold: dict[str, int], threshold: float) -> float:
    return sum(1 for c, p in call_probs.items() if (p >= threshold) == bool(gold[c])) / len(call_probs)


def best_threshold(call_probs: dict[str, float], gold: dict[str, int]) -> tuple[float, float]:
    """dev 정확도가 가장 높은 임계값. 동률이면 0.5 에 가까운 쪽을 고른다."""
    scored = [(accuracy_at(call_probs, gold, t), -abs(t - 0.5), t) for t in GRID]
    acc, _, threshold = max(scored)
    return round(float(threshold), 3), float(acc)


def main(argv=None) -> int:
    ensure_utf8_stdout()
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from .evaluate import suggested_workers
    from .train import build_splits

    index = CacheIndex.load(args.train_cache)
    if not index.rows:
        raise SystemExit(f"캐시가 비어 있습니다: {args.train_cache}")

    _, dev_samples = build_splits(index, args.dev_fraction, args.seed, 0, 0)
    dev_truth = truth_from_samples(dev_samples)

    model, branch, cfg, payload = load_checkpoint(args.ckpt, device=device)
    workers = args.num_workers if args.num_workers is not None else suggested_workers(branch)
    print(f"{args.ckpt.name} ({branch}) | dev {len(dev_truth)}통화 | num_workers={workers}",
          flush=True)

    probs = predict_segment_probs(
        model, index, dev_samples, cfg, branch, device,
        batch_size=args.batch_size, mode="sliding", num_workers=workers,
    )
    call_probs = call_probabilities(dev_samples, probs)
    gold = {c: gender_to_target(dev_truth[c]) for c in call_probs}

    base = accuracy_at(call_probs, gold, DEFAULT_THRESHOLD)
    threshold, acc = best_threshold(call_probs, gold)

    print(f"  dev 임계값 {DEFAULT_THRESHOLD:.3f} : {base:.4f}")
    print(f"  dev 최적   {threshold:.3f} : {acc:.4f}  ({(acc - base) * 100:+.2f}%p)")

    if args.dry_run:
        print("  --dry-run 이라 체크포인트를 수정하지 않았습니다.")
        return 0

    write_threshold(args.ckpt, threshold, dev_accuracy=acc)
    print(f"  -> {args.ckpt} 에 decision_threshold={threshold:.3f} 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
