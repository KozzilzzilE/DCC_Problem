"""(c) 에서 나온 곁가지 확인: 보정 로짓을 전체적으로 날카롭게(온도 T<1) 하면 4-1 이 오르는가.

cal' = sigmoid(logit(cal) / T). 4-1 은 cal' 에 기대 점수 규칙. 4-2 는 중단 신호만 cal' 에서 계산하고
중단 시점 예측은 원래 규칙(보정 전 0.5, 비면 1개)을 그대로 둔다. T 는 5겹 교차검증으로 고른다.
"""
import copy
import json

import numpy as np

import common as cm
from halt_study import first_stop
from m4 import decision as dec
from m4 import simulate as sim

g = cm.Geometry(cm.load_table("E002"))
shift = cm.load_shift()
Ts = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
out = {}
for bname, keys in (("E002", ["E002"]), ("E007", ["E007"]), ("ENS3", ["E001", "E002", "E007"])):
    raw = np.mean([cm.load_table(k).probs for k in keys], axis=0).astype(np.float64)
    base = cm.Candidate(g, dec.calibrate(raw, shift), raw, "T=1.0")
    zc = cm.logit(dec.calibrate(raw, shift))
    cands = []
    for T in Ts:
        c = base if T == 1.0 else cm.Candidate(g, cm.sigmoid(zc / T), raw, f"T={T}")   # raw 는 그대로: 중단 시점 예측 규칙 불변
        cands.append(c)
    r = cm.cv_select(g, cands)
    b = cm.cv_select(g, [base])
    d = cm.boot_diff(g, r, b)
    grid = {c.name: {"m4_1": c.s41(), "mean_k": float(c.pred41.sum(1).mean()), "m4_2_best_tau": max(c.s42(t) for t in cm.TAUS)} for c in cands}
    out[bname] = {"grid": grid, "cv41": r["cv41"], "cv42": r["cv42"], "base_cv42": b["cv42"], "pick41": r["pick41"],
                  "pick42": [f"{a},τ={t}" for a, t in r["pick42"]], "vs_base_cv": d}
    print(f"{bname}: T별 4-1 " + "  ".join(f"{k} {v['m4_1']:.4f}(평균 {v['mean_k']:.2f}개)" for k, v in grid.items()))
    print(f"   교차검증 4-1 {r['cv41']:.4f} ({d['d41']:+.4f} {cm.fmt_ci(d['ci41'])}) 고른 값 {r['pick41']}")
    print(f"   교차검증 4-2 {r['cv42']:.4f} (기준 {b['cv42']:.4f}, 겹밖 모음 차이 {d['d42']:+.4f} {cm.fmt_ci(d['ci42'])}) 고른 값 {out[bname]['pick42']}", flush=True)
(cm.HERE / "s07_temperature.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
