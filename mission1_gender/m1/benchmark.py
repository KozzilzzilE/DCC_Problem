"""두 갈래(ResNet50 vs Wav2Vec2)의 정확도·속도 비교표 생성.

    python -m m1.benchmark --ckpt mission1_gender/ckpt/resnet_full.pt \
                           --ckpt mission1_gender/ckpt/w2v2_full.pt \
                           --out mission1_gender/reports/comparison

두 갈래가 같은 캐시·같은 조각 분할·같은 집계를 쓰므로, 표에 남는 차이는
모델에서 나온 것이다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ._console import ensure_utf8_stdout
from .cache import CacheIndex
from .datasets import samples_from_rows
from .evaluate import suggested_workers, majority_baseline, predict_segment_probs, score, truth_from_samples
from .models import load_checkpoint

COLUMNS = [
    ("label", "체크포인트"),
    ("branch", "갈래"),
    ("n_params_m", "파라미터(M)"),
    ("dev_call_accuracy", "dev 통화 Acc"),
    ("val_call_accuracy", "Validation 통화 Acc"),
    ("val_segment_accuracy", "Validation 조각 Acc"),
    ("train_seconds_per_epoch", "학습 s/epoch"),
    ("preprocess_segments_per_second", "전처리 seg/s"),
    ("inference_ms_per_call", "추론 ms/통화"),
    ("peak_vram_mb", "VRAM 피크(MB)"),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mission 1 갈래 비교")
    p.add_argument("--ckpt", type=Path, action="append", required=True)
    p.add_argument("--val-cache", type=Path, default=Path("cache/val"))
    p.add_argument("--out", type=Path, default=Path("mission1_gender/reports/comparison"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=None,
                   help="기본값은 갈래에 맞춰 자동 (resnet 0 / 16k 업샘플 갈래 4)")
    p.add_argument("--eval-mode", choices=("center", "sliding"), default="sliding")
    p.add_argument("--latency-calls", type=int, default=200)
    return p.parse_args(argv)


def _rounded(value: float | None, digits: int) -> float | None:
    """NaN 은 JSON 에서 유효하지 않으므로 None(null) 으로 눕힌다."""
    if value is None or value != value:
        return None
    return round(value, digits)


def measure_preprocess_throughput(model, cfg, branch, device, n: int = 512) -> float | None:
    """피처 프런트엔드만 따로 재 초당 처리 조각 수를 구한다."""
    frontend = getattr(model, "frontend", None)
    if frontend is None:
        return None  # w2v2 는 raw waveform 을 그대로 먹어 별도 전처리가 없다

    batch = torch.zeros(64, cfg.window_samples, device=device)
    with torch.no_grad():
        for _ in range(3):  # warm-up
            frontend(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        done = 0
        while done < n:
            frontend(batch)
            done += len(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return done / (time.perf_counter() - started)


def measure_call_latency(model, index, samples, cfg, branch, device, mode, n_calls) -> float:
    """통화 1건을 처음부터 끝까지 예측하는 평균 지연 (ms)."""
    by_call: dict[str, list] = {}
    for sample in samples:
        by_call.setdefault(sample.row.call_id, []).append(sample)

    chosen = sorted(by_call)[:n_calls]
    if not chosen:
        return float("nan")

    for cid in chosen[: min(3, len(chosen))]:  # warm-up
        predict_segment_probs(model, index, by_call[cid], cfg, branch, device, mode=mode)

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for cid in chosen:
        predict_segment_probs(model, index, by_call[cid], cfg, branch, device, mode=mode)
    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - started) * 1000.0 / len(chosen)


def evaluate_checkpoint(path: Path, args, device) -> dict:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, branch, cfg, payload = load_checkpoint(path, device=device)

    index = CacheIndex.load(args.val_cache)
    samples = samples_from_rows(index.rows)
    truth = truth_from_samples(samples)

    started = time.perf_counter()
    probs = predict_segment_probs(
        model, index, samples, cfg, branch, device,
        batch_size=args.batch_size, mode=args.eval_mode, num_workers=args.num_workers if args.num_workers is not None else suggested_workers(branch),
    )
    val_seconds = time.perf_counter() - started
    metrics = score(samples, probs, truth)

    history_path = path.with_suffix(".history.json")
    train_seconds = None
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
        train_seconds = float(np.median([r["train_seconds"] for r in history]))

    return {
        "label": path.stem,
        "branch": branch,
        "feature": cfg.kind,
        "model_name": payload.get("extra", {}).get("model_name"),
        "n_params_m": round(payload.get("extra", {}).get("n_params", 0) / 1e6, 1),
        "dev_call_accuracy": _rounded(payload.get("metrics", {}).get("dev_call_accuracy"), 4),
        "val_call_accuracy": _rounded(metrics.call_accuracy, 4),
        "val_segment_accuracy": _rounded(metrics.segment_accuracy, 4),
        "val_confusion": metrics.confusion,
        "val_per_gender_accuracy": {k: _rounded(v, 4) for k, v in metrics.per_gender_accuracy.items()},
        "val_total_seconds": _rounded(val_seconds, 1),
        "train_seconds_per_epoch": _rounded(train_seconds, 1),
        "preprocess_segments_per_second": _rounded(
            measure_preprocess_throughput(model, cfg, branch, device), 1
        ),
        "inference_ms_per_call": _rounded(
            measure_call_latency(model, index, samples, cfg, branch, device,
                                 args.eval_mode, args.latency_calls), 2
        ),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if device.type == "cuda" else 0.0,
        "n_val_calls": metrics.n_calls,
        "n_val_segments": metrics.n_segments,
    }


def to_markdown(results: list[dict], baseline: float) -> str:
    header = "| " + " | ".join(title for _, title in COLUMNS) + " |"
    divider = "|" + "|".join(["---"] * len(COLUMNS)) + "|"
    lines = [
        "# Mission 1 — CNN(ResNet50) vs 음성 특화(Wav2Vec2) 비교",
        "",
        f"Validation 다수결 기준선(통화 단위): **{baseline:.4f}**",
        "",
        header,
        divider,
    ]
    for row in results:
        cells = [
            "해당 없음" if row.get(key) is None else str(row.get(key, ""))
            for key, _ in COLUMNS
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## 통화 단위 혼동행렬 (Validation)", ""]
    for row in results:
        lines.append(f"- **{row['label']}** ({row['branch']}): {row['val_confusion']}"
                     f" / 성별별 {row['val_per_gender_accuracy']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ensure_utf8_stdout()
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index = CacheIndex.load(args.val_cache)
    if not index.rows:
        raise SystemExit(f"Validation 캐시가 비어 있습니다: {args.val_cache}")
    baseline = majority_baseline(truth_from_samples(samples_from_rows(index.rows)))

    results = []
    for path in args.ckpt:
        print(f"=== {path} ===", flush=True)
        row = evaluate_checkpoint(path, args, device)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps({"baseline": baseline, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = to_markdown(results, baseline)
    args.out.with_suffix(".md").write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
