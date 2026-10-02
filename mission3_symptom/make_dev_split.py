"""Training CSV 를 Training 내부 학습용과 dev 로 나눈다 (하이퍼파라미터 선택 전용).

주최 측 답변(2026-09-30): pos_weight 의 p, 앙상블 가중치 w, TF-IDF LR 의 C 같은 하이퍼파라미터는
제공된 Validation 이 아니라 Training 내부 dev/OOF 로 정하고, Validation 은 결정된 모델의 성능 확인에만 쓴다.
절차와 선택 규칙은 reports/dev_selection_protocol.md 에 있다.

    python make_dev_split.py --train-csv data_csv/mission3_train.csv --out-dir runs/devsel/data

- 통화(행) 단위 무작위 분할이다. numpy default_rng(seed).permutation 의 앞 round(n x fraction) 행이 dev 다.
- 각 파일 안의 행 순서는 원본 순서를 유지한다. 같은 입력과 seed 면 항상 같은 분할이 나온다.
- out-dir 에 train_split.csv, dev_split.csv, split.json(원본 sha256, 행 수, 라벨 분포)을 쓴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_NAME = "train_split.csv"
DEV_NAME = "dev_split.csv"
MANIFEST_NAME = "split.json"


def split_frame(frame, dev_fraction: float = 0.1, seed: int = 1234) -> Tuple[object, object]:
    """(학습용, dev) DataFrame. 두 부분은 겹치지 않고 합치면 원본 전체다."""
    import numpy as np

    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(f"dev_fraction 은 0 과 1 사이여야 합니다: {dev_fraction}")
    n = len(frame)
    n_dev = int(round(n * dev_fraction))
    if not 0 < n_dev < n:
        raise ValueError(f"행 {n}개로는 dev 비율 {dev_fraction} 분할을 만들 수 없습니다")
    order = np.random.default_rng(seed).permutation(n)
    dev_index = np.sort(order[:n_dev])
    train_index = np.sort(order[n_dev:])
    return frame.iloc[train_index].reset_index(drop=True), frame.iloc[dev_index].reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positives(frame) -> dict:
    from m3.config import TARGET_SYMPTOMS

    return {symptom: int(frame[symptom].sum()) for symptom in TARGET_SYMPTOMS}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mission 3 Training 내부 dev 분할 (하이퍼파라미터 선택 전용)")
    p.add_argument("--train-csv", required=True, help="Training CSV (make_csv.py 출력). Validation 을 넣지 않는다")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dev-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import pandas as pd

    from m3.config import TARGET_SYMPTOMS

    args = parse_args(argv)
    source = Path(args.train_csv)
    frame = pd.read_csv(source, encoding="utf-8-sig")
    missing = [column for column in ["call_id", "text", *TARGET_SYMPTOMS] if column not in frame.columns]
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}")
    if frame["call_id"].duplicated().any():
        raise ValueError("call_id 가 중복된 행이 있습니다. 통화 단위 분할을 할 수 없습니다")

    train_part, dev_part = split_frame(frame, args.dev_fraction, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, dev_path = out_dir / TRAIN_NAME, out_dir / DEV_NAME
    train_part.to_csv(train_path, index=False, encoding="utf-8-sig")
    dev_part.to_csv(dev_path, index=False, encoding="utf-8-sig")

    manifest = {
        "purpose": "하이퍼파라미터 선택 전용 Training 내부 dev 분할 (Validation 미사용)",
        "source_csv": source.name,
        "source_sha256": _sha256(source),
        "source_rows": len(frame),
        "dev_fraction": args.dev_fraction,
        "seed": args.seed,
        "rule": "numpy.random.default_rng(seed).permutation(n) 앞 round(n*fraction) 행 = dev, 각 부분은 원본 순서 유지",
        "train_rows": len(train_part),
        "dev_rows": len(dev_part),
        "train_sha256": _sha256(train_path),
        "dev_sha256": _sha256(dev_path),
        "train_positives": _positives(train_part),
        "dev_positives": _positives(dev_part),
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"학습용 {len(train_part):,}행 -> {train_path}")
    print(f"dev    {len(dev_part):,}행 -> {dev_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
