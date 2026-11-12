"""학습 결과(best_model 폴더)를 제출용 `.pt` 번들로 묶는다.

    python scripts/pack_bundle.py --members runs/E002_prefix_uniform --out ckpt/mission4.pt

확률 보정에 쓰는 log(pos_weight) 는 첫 멤버의 run_config.json 에서 읽어 번들에 넣는다.
maxent 규칙이면 증상 조합 사전분포를 학습에 쓴 통화의 라벨로 만들어 함께 넣는다. 통화 목록은 첫 멤버의
run_config.json(calls, ids)에서 읽고, `--calls` / `--prior-ids` 로 바꿀 수 있다 (dev·Validation 라벨로 만들면 안 된다).
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
from m4.data import label_matrix, load_cache, read_ids  # noqa: E402
from m4.decision import label_set_prior  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="run 폴더 (여러 개면 확률 평균)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", default="space")
    ap.add_argument("--truncation-side", default="right")
    ap.add_argument("--rule", default="maxent", choices=["maxent", "expected", "threshold"], help="4-1 결정 규칙")
    ap.add_argument("--calls", help="maxent 사전분포를 만들 라벨 캐시 (기본: 첫 멤버가 학습에 쓴 캐시)")
    ap.add_argument("--prior-ids", help="사전분포에 쓸 통화 목록 (기본: 첫 멤버가 학습에 쓴 목록)")
    ap.add_argument("--halt-type", default="expected", choices=["expected", "confidence", "stable", "fixed", "never"])
    ap.add_argument("--halt-tau", type=float, default=0.52)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    config_path = Path(args.members[0]) / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weights = config.get("pos_weight")          # 학습 스크립트가 남긴 클래스별 pos_weight
    label_prior = None
    if args.rule == "maxent":
        # 사전분포는 모델이 학습에 쓴 통화의 라벨로만 만든다. 따로 주지 않으면 학습 설정에 적힌 캐시와 목록을 쓴다.
        calls_path = args.calls or config.get("calls")
        ids_path = args.prior_ids if args.prior_ids is not None else config.get("ids")
        if not calls_path:
            raise SystemExit("학습 설정에 라벨 캐시 경로가 없습니다. --calls 와 --prior-ids 를 직접 지정하십시오.")
        if "val" in Path(calls_path).name.lower():
            raise SystemExit(f"Validation 라벨로는 사전분포를 만들 수 없습니다: {calls_path}")
        keep = read_ids(Path(ids_path)) if ids_path else None      # 목록이 없으면 캐시 전체로 학습한 모델
        labels = label_matrix(load_cache(Path(calls_path), keep))
        print(f"사전분포 라벨: {calls_path} / 목록 {ids_path or '(캐시 전체)'} / {len(labels)}통화")
        prior, marg = label_set_prior(labels)
        # 어떤 라벨로 만들었는지 번들에 남긴다 (추론에는 쓰지 않는 기록)
        label_prior = {"prior": prior.tolist(), "marg": marg.tolist(),
                       "source": {"calls": str(calls_path), "ids": str(ids_path) if ids_path else None,
                                  "n_calls": int(len(labels))}}
    settings = {"fmt": args.fmt, "truncation_side": args.truncation_side,
                "logit_shift": [math.log(w) for w in weights] if weights else None,
                "rule": {"type": args.rule}, "label_prior": label_prior,
                # 4-2 중단 시점에는 Macro-F1 에 맞는 임계값 규칙을 쓴다 (dev 비교 결과, EXPERIMENT_LOG E004)
                "halt_rule": {"type": "threshold", "threshold": 0.5, "force_top1": True},
                "halt": {"type": args.halt_type, "tau": args.halt_tau, "min_step": 5}}
    path = pack_bundle(args.members, args.out, settings, note=args.note)
    print(f"{path} ({path.stat().st_size / 2**20:.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
