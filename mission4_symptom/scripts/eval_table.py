"""누적 확률표 하나에 대해 4-1 결정 규칙과 4-2 중단 정책 묶음을 모의 채점하고 metrics.json 으로 남긴다.

    python scripts/eval_table.py --table runs/E001_m3_baseline/dev_prefix.npz --split dev

여기서 훑는 값(τ, k 등)은 Training 내부 dev 에서만 고른다. Validation 표에는 이미 정한 설정만 넣는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4 import simulate as sim  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--split", required=True, help="기록용 이름 (dev / validation)")
    ap.add_argument("--out")
    args = ap.parse_args()

    table = sim.PrefixTable.load(args.table)
    rules = {
        "thr0.5": lambda p: sim.decide(p),
        "thr0.5+top1": lambda p: sim.decide(p, force_top1=True),
        "top1": lambda p: sim.decide(p, threshold=2.0, force_top1=True),
        "thr0.5+top1,max2": lambda p: sim.decide(p, force_top1=True, max_labels=2),
    }
    res = {"table": args.table, "split": args.split, "n_calls": len(table),
           "n_moments": int(table.offsets[-1]), "mission_4_1": {}, "mission_4_2": {}}

    print(f"== 4-1 ({args.split}, 통화 {len(table)}건)")
    for name, rule in rules.items():
        r = sim.score_4_1(table, rule)
        res["mission_4_1"][name] = r
        print(f"  {name:<18} {r['score']:.4f}  빈 예측 {r['empty_rate']:.3f}  평균 개수 {r['mean_labels']:.2f}  "
              f"앞/가운데/뒤 {r['by_third']['앞']:.3f}/{r['by_third']['가운데']:.3f}/{r['by_third']['뒤']:.3f}")

    rule = rules["thr0.5+top1"]
    policies = {"끝까지": sim.halt_full, "안정 2회(팀원 규칙)": sim.halt_stable()}
    policies.update({f"고정 k={k}": sim.halt_fixed(k) for k in (5, 6, 8, 10, 12, 15, 20)})
    policies.update({f"확신도 τ={t}": sim.halt_confidence(t) for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)})
    print(f"== 4-2 (중단 시점 예측 규칙: thr0.5+top1)")
    for name, halt in policies.items():
        r = sim.score_4_2(table, halt, rule)
        res["mission_4_2"][name] = r
        print(f"  {name:<20} {r['score']:.4f} = F1 {r['macro_f1']:.4f} x 조기 {r['mean_earliness']:.4f}  "
              f"중단 중앙 {r['halt_median']:.0f}  끝까지 {r['never_halted']:.2f}")

    out = Path(args.out) if args.out else Path(args.table).with_name(f"metrics_{args.split}.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
