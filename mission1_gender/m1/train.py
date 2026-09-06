"""Mission 1 학습 CLI — 두 갈래 공용.

    python -m m1.train --branch resnet --cache cache/train --out mission1_gender/resnet.pt
    python -m m1.train --branch w2v2   --cache cache/train --out mission1_gender/w2v2.pt

Validation 폴더는 여기서 절대 쓰지 않는다. Training 안에서 통화 ID 기준으로
잘라낸 dev 로만 early stopping 과 모델 선택을 한다 (대회 규칙).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ._console import ensure_utf8_stdout
from .cache import CacheIndex
from .config import FeatureConfig, TrainConfig
from .datasets import SegmentWindowDataset, Sample, samples_from_rows, split_calls
from .evaluate import majority_baseline, predict_segment_probs, score, truth_from_samples
from .models import build_model, save_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mission 1 성별 분류 학습")
    p.add_argument("--branch", choices=("resnet", "w2v2", "audeering"), default="resnet")
    p.add_argument("--cache", type=Path, default=Path("cache/train"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dev-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--feature", choices=("logmel", "mfcc"), default="logmel")
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--window-frames", type=int, default=192)
    p.add_argument("--max-train-calls", type=int, default=0, help="0 이면 전부 사용")
    p.add_argument("--max-dev-calls", type=int, default=0,
                   help="0 이면 전부 사용. 스모크 테스트에서 dev 평가 비용을 줄일 때 쓴다")
    p.add_argument("--eval-mode", choices=("center", "sliding"), default="center")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--w2v2-model", type=str, default=None)
    p.add_argument("--log", type=Path, default=None, help="epoch 별 지표 JSON 경로")
    return p.parse_args(argv)


def build_splits(
    index: CacheIndex,
    dev_fraction: float,
    seed: int,
    max_train_calls: int,
    max_dev_calls: int = 0,
):
    groups = index.by_call()
    labelled = [cid for cid, rows in groups.items() if rows[0].gender]
    train_ids, dev_ids = split_calls(labelled, dev_fraction, seed)

    if max_train_calls:
        train_ids = set(sorted(train_ids)[:max_train_calls])
    if max_dev_calls:
        dev_ids = set(sorted(dev_ids)[:max_dev_calls])

    train_rows = [r for cid in sorted(train_ids) for r in groups[cid]]
    dev_rows = [r for cid in sorted(dev_ids) for r in groups[cid]]
    return samples_from_rows(train_rows), samples_from_rows(dev_rows)


def run_epoch(model, loader, criterion, optimizer, scaler, device, amp) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    seen = 0
    correct = 0

    for waveform, target in loader:
        waveform = waveform.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(waveform)
            loss = criterion(logits, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch = target.numel()
        total_loss += loss.item() * batch
        seen += batch
        correct += ((logits.detach().float() >= 0).float() == target).sum().item()

    return total_loss / max(1, seen), correct / max(1, seen)


def main(argv=None) -> int:
    ensure_utf8_stdout()
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    cfg = FeatureConfig(
        kind=args.feature, n_mels=args.n_mels, window_frames=args.window_frames
    )
    train_cfg = TrainConfig(
        branch=args.branch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
        amp=amp,
    )

    index = CacheIndex.load(args.cache)
    if not index.rows:
        raise SystemExit(f"캐시가 비어 있습니다: {args.cache}")

    train_samples, dev_samples = build_splits(
        index, args.dev_fraction, args.seed, args.max_train_calls, args.max_dev_calls
    )
    dev_truth = truth_from_samples(dev_samples)

    # 첫 epoch 이 끝나기까지 수 분이 걸리므로 설정 요약은 즉시 흘려보낸다.
    print(f"device={device} branch={args.branch} amp={amp}", flush=True)
    print(f"train: {len(train_samples)} segments / {len({s.row.call_id for s in train_samples})} calls", flush=True)
    print(f"dev  : {len(dev_samples)} segments / {len(dev_truth)} calls", flush=True)
    print(f"dev majority baseline (call-level): {majority_baseline(dev_truth):.4f}", flush=True)

    kwargs = {"model_name": args.w2v2_model} if args.branch in ("w2v2", "audeering") else {}
    model = build_model(args.branch, cfg, **kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.1f}M", flush=True)

    train_loader = DataLoader(
        SegmentWindowDataset(index, train_samples, cfg, branch=args.branch, train=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda") if amp else None

    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        loss, train_seg_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp
        )
        train_seconds = time.time() - started

        probs = predict_segment_probs(
            model, index, dev_samples, cfg, args.branch, device,
            batch_size=max(64, args.batch_size), mode=args.eval_mode,
            num_workers=args.num_workers, amp=amp,
        )
        metrics = score(dev_samples, probs, dev_truth)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(loss, 5),
            "train_segment_accuracy": round(train_seg_acc, 5),
            "dev_call_accuracy": round(metrics.call_accuracy, 5),
            "dev_segment_accuracy": round(metrics.segment_accuracy, 5),
            "train_seconds": round(train_seconds, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs} loss {loss:.4f} train_seg {train_seg_acc:.4f} | "
            f"{metrics.summary()} | {train_seconds:.0f}s",
            flush=True,
        )

        if metrics.call_accuracy > best:
            best = metrics.call_accuracy
            save_checkpoint(
                args.out, model, args.branch, cfg,
                metrics={
                    "dev_call_accuracy": metrics.call_accuracy,
                    "dev_segment_accuracy": metrics.segment_accuracy,
                    "dev_confusion": metrics.confusion,
                    "epoch": epoch,
                },
                extra={
                    "train_config": train_cfg.to_dict(),
                    "n_params": n_params,
                    "n_train_segments": len(train_samples),
                    "model_name": args.w2v2_model,
                },
            )
            print(f"  -> saved {args.out} (dev call acc {best:.4f})", flush=True)

    log_path = args.log or args.out.with_suffix(".history.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {"history": history, "best_dev_call_accuracy": best,
             "dev_majority_baseline": majority_baseline(dev_truth),
             "feature_config": cfg.to_dict(), "train_config": train_cfg.to_dict()},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"best dev call accuracy: {best:.4f}  (log: {log_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
