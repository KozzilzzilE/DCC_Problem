"""라벨 JSON 폴더를 JSONL 캐시로 모으고, 예선과 같은 Training 내부 dev 통화 목록을 만든다.

    python scripts/build_cache.py --train-label ../data/train/label \
        --m3-dev-csv <예선 runs/devsel/data/dev_split.csv> --out-dir runs/cache

dev 분할은 예선 Mission 3 의 것(make_dev_split.py, seed 1234, 10%)을 그대로 쓴다.
예선에서 학습한 모델이 이 dev 통화를 보지 않았으므로 같은 분할을 써야 비교가 깨끗하다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4.data import build_cache, load_cache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-label", required=True)
    # 학습과 값 고르기에는 Training 캐시만 쓴다. Validation 캐시는 정해진 설정을 마지막에 확인할 때만 만든다.
    ap.add_argument("--val-label", help="Validation 확인용 캐시도 만들 때만 지정 (학습에는 쓰지 않는다)")
    ap.add_argument("--m3-dev-csv", required=True)
    ap.add_argument("--out-dir", default="runs/cache")
    args = ap.parse_args()

    out = Path(args.out_dir)
    n_train = build_cache(Path(args.train_label), out / "train.jsonl")
    n_val = build_cache(Path(args.val_label), out / "val.jsonl") if args.val_label else None

    with open(args.m3_dev_csv, "r", encoding="utf-8-sig", newline="") as handle:
        dev_ids = [row["call_id"] for row in csv.DictReader(handle)]
    train_ids = {c.call_id for c in load_cache(out / "train.jsonl")}
    missing = [i for i in dev_ids if i not in train_ids]
    if missing:
        raise SystemExit(f"dev 목록의 통화 {len(missing)}건이 Training 에 없습니다")
    (out / "dev_ids.txt").write_text("\n".join(dev_ids) + "\n", encoding="utf-8")
    fit_ids = sorted(train_ids - set(dev_ids))
    (out / "fit_ids.txt").write_text("\n".join(fit_ids) + "\n", encoding="utf-8")

    info = {"train_calls": n_train, "val_calls": n_val, "dev_calls": len(dev_ids), "fit_calls": len(fit_ids)}
    (out / "cache.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
