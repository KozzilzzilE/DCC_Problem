"""확률표 하나 또는 여러 개의 평균을, 채택한 규칙(4-1 기대 점수 최대화 / 4-2 기대 점수 중단)으로 채점한다.

    python scripts/eval_combo.py --shift-from runs/E002_prefix_uniform --tables A.npz B.npz

4-2 는 고정 기준 0.52 의 점수와, 기준을 5겹 교차검증으로 골랐을 때의 점수를 함께 낸다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from halt_study import first_stop, score  # noqa: E402
from m4 import decision as dec  # noqa: E402
from m4 import simulate as sim  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def evaluate(tables, shift, tau: float = 0.52, folds: int = 5, label_prior=None) -> dict:
    """label_prior 를 주면 4-1 에 maxent 규칙을 쓴다. 4-2 의 중단 신호와 중단 시점 예측은 어느 쪽이든 같다."""
    t = tables[0]
    if len(tables) > 1:
        t = sim.PrefixTable(t.call_ids, t.offsets, np.mean([x.probs for x in tables], axis=0), t.labels)
    cal = dec.calibrate(t.probs, shift)
    pred41, value = dec.expected_rule(cal)            # value: 4-2 중단 신호 (독립 가정 기대 점수)
    extra = {}
    if label_prior is not None:
        pred41 = dec.maxent_rule(cal, *label_prior)[0]
        # 맞춘 분포의 증상별 확률이 입력과 0.001 넘게 어긋난 순간의 비율 (합이 1 미만이라 올려 맞춘 순간 제외)
        ok = cal.sum(axis=1) >= 1.02
        idx = np.nonzero(ok)[0][:: max(1, int(ok.sum()) // 5000)]
        resid = np.abs(dec.maxent_weights(cal[idx], *label_prior) @ dec._SETS - np.clip(cal[idx], 1e-4, 1 - 1e-4)).max(axis=1)
        extra = {"maxent_low_sum_rate": float((~ok).mean()), "maxent_misfit_rate": float((resid > 1e-3).mean()),
                 "mean_labels_4_1": float(pred41.sum(axis=1).mean())}
    r41 = sim.score_4_1(t, lambda p: pred41)
    pred42 = dec.threshold_rule(t.probs.copy())
    fixed = score(first_stop(value >= tau, t.offsets), pred42, t)
    taus = np.round(np.arange(0.40, 0.71, 0.02), 2)
    halts = {x: first_stop(value >= x, t.offsets) for x in taus}
    fold = np.random.default_rng(0).permutation(len(t)) % folds
    cv = []
    for k in range(folds):
        tr, te = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
        pick = max(taus, key=lambda x: score(halts[x], pred42, t, tr)["score"])
        cv.append(score(halts[pick], pred42, t, te)["score"])
    full = pred42[t.offsets[1:] - 1]
    return {"m4_1": r41["score"], "m4_1_thirds": r41["by_third"], "m4_2_tau0.52": fixed,
            "m4_2_cv": float(np.mean(cv)), "full_call_f1": sim.macro_f1(full, t.labels)["macro_f1"], **extra}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--shift-from", required=True)
    ap.add_argument("--out")
    ap.add_argument("--maxent-prior-ids", help="이 통화 목록의 라벨로 증상 조합 사전분포를 만들어 4-1 에 maxent 규칙을 쓴다")
    ap.add_argument("--calls", default="runs/cache/train.jsonl")
    args = ap.parse_args()
    prior = None
    if args.maxent_prior_ids:
        from m4.data import label_matrix, load_cache, read_ids
        prior = dec.label_set_prior(label_matrix(load_cache(Path(args.calls), read_ids(Path(args.maxent_prior_ids)))))
    shift = np.log(json.loads((Path(args.shift_from) / "run_config.json").read_text(encoding="utf-8"))["pos_weight"])
    r = evaluate([sim.PrefixTable.load(p) for p in args.tables], shift, label_prior=prior)
    f = r["m4_2_tau0.52"]
    th = r["m4_1_thirds"]
    print(f"4-1 {r['m4_1']:.4f} (앞 {th['앞']:.3f} / 가운데 {th['가운데']:.3f} / 뒤 {th['뒤']:.3f})  "
          f"4-2 {f['score']:.4f} = F1 {f['macro_f1']:.4f} x {f['earliness']:.4f} (교차검증 {r['m4_2_cv']:.4f})  "
          f"끝까지 F1 {r['full_call_f1']:.4f}"
          + (f"  [maxent: 평균 {r['mean_labels_4_1']:.2f}개, 합<1 순간 {r['maxent_low_sum_rate']:.3f}, 어긋남 {r['maxent_misfit_rate']:.4f}]"
             if "mean_labels_4_1" in r else ""))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(r, tables=args.tables), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
