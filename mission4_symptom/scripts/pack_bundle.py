"""학습 결과(best_model 폴더)를 제출용 `.pt` 번들로 묶는다.

    python scripts/pack_bundle.py --members runs/E002_prefix_uniform --out ckpt/mission4.pt

확률 보정에 쓰는 log(pos_weight) 는 첫 멤버의 run_config.json 에서 읽어 번들에 넣는다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4.bundle import pack_bundle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="run 폴더 (여러 개면 확률 평균)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", default="space")
    ap.add_argument("--truncation-side", default="right")
    ap.add_argument("--rule", default="expected", choices=["expected", "threshold"], help="4-1 결정 규칙")
    ap.add_argument("--halt-type", default="expected", choices=["expected", "confidence", "stable", "fixed", "never"])
    ap.add_argument("--halt-tau", type=float, default=0.52)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    config_path = Path(args.members[0]) / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weights = config.get("pos_weight")          # 학습 스크립트가 남긴 클래스별 pos_weight
    settings = {"fmt": args.fmt, "truncation_side": args.truncation_side,
                "logit_shift": [math.log(w) for w in weights] if weights else None,
                "rule": {"type": args.rule},
                # 4-2 중단 시점에는 Macro-F1 에 맞는 임계값 규칙을 쓴다 (dev 비교 결과, EXPERIMENT_LOG E004)
                "halt_rule": {"type": "threshold", "threshold": 0.5, "force_top1": True},
                "halt": {"type": args.halt_type, "tau": args.halt_tau, "min_step": 5}}
    path = pack_bundle(args.members, args.out, settings, note=args.note)
    print(f"{path} ({path.stat().st_size / 2**20:.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
