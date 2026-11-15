"""S02 독립 가정을 바꾸면 4-1 이 오르는가.

기준: 보정 확률 + 독립 가정 + 상위 k (max_k=4).
변형: 정답 집합의 결합 분포를 (size / tilt / maxent) 로 바꾸고, 예측 집합은 (상위 k / 511가지 전부)에서 고른다.
결합 분포의 사전 정보(π, 개수 분포)는 학습용(fit) 26,280통화 라벨에서만 만든다 → dev 에서 고르는 값이 없다.
α(틸트 세기) 등 dev 에서 고르는 값은 5겹 교차검증으로 보고한다.
"""
import json
import time

import numpy as np

import joint as J
from common import HERE, Ctx, FOLDS, baseline_pred, boot, dec, fmt_boot, load_table, mean_table, shift

sh = shift()
Y = J.fit_labels()
pi, marg, size_true = J.prior_joint(Y)
g_set = J.size_factor(marg, size_true)
print("fit 통화", len(Y), "주변확률", marg.round(3), "개수 분포", size_true[:5].round(4))
OUT = {"prior_marg": marg.tolist(), "size_true": size_true.tolist()}

tables = {"E002": load_table("E002"), "E007": load_table("E007"), "ENS3": mean_table(["E001", "E002", "E007"])}

for name, t in tables.items():
    c = Ctx(t)
    cal = dec.calibrate(t.probs, sh)
    pred0, val0 = baseline_pred(t, sh)
    base = c.per_call(c.ms(pred0))
    res = {"baseline": float(base.mean()), "believed_value_baseline": float(c.per_call(val0).mean())}
    print(f"\n== {name}: 기준 {base.mean():.4f} (모델이 믿는 기대값 {c.per_call(val0).mean():.4f})")

    def report(label, pred, val):
        pc = c.per_call(c.ms(pred))
        b = boot(pc - base)
        k = pred.sum(1)
        res[label] = {"score": float(pc.mean()), "delta": b, "believed": float(c.per_call(val).mean()),
                      "mean_k": float(k.mean()), "k1_rate": float((k == 1).mean())}
        print(f"  {label:<34} {pc.mean():.4f}  Δ {fmt_boot(b)}  믿는 값 {c.per_call(val).mean():.4f}  평균 k {k.mean():.2f}  k=1 {100*(k==1).mean():.0f}%")
        return pc

    t0 = time.time()
    # 독립 가정, 예측 후보만 넓힘 / 좁힘
    for mk in (1, 2, 3, 9):
        p, v = dec.expected_rule(cal, max_k=mk)
        report(f"indep 상위k max_k={mk}", p, v)
    p, v = J.run_chunks(cal, J.w_indep, "full")
    report("indep 511 전부 탐색", p, v)
    # 개수 분포 보정
    for pw in (0.5, 1.0):
        p, v = J.run_chunks(cal, lambda q: J.w_size(q, g_set, pw), "topk")
        report(f"size^{pw} 상위k", p, v)
    p, v = J.run_chunks(cal, lambda q: J.w_size(q, g_set, 1.0), "full")
    report("size^1 511 전부", p, v)
    # 결합 사전분포 틸트
    tilt = {}
    for a in (0.6, 0.8, 1.0, 1.2, 1.5):
        p, v = J.run_chunks(cal, lambda q: J.w_tilt(q, pi, marg, a), "full")
        tilt[a] = report(f"tilt α={a} 511 전부", p, v)
    p, v = J.run_chunks(cal, lambda q: J.w_tilt(q, pi, marg, 1.0), "topk")
    report("tilt α=1.0 상위k", p, v)
    # α 를 교차검증으로 고름
    alphas = sorted(tilt)
    cv = np.zeros(c.C)
    picks = []
    for k in range(FOLDS):
        tr, te = c.fold != k, c.fold == k
        a = max(alphas, key=lambda x: tilt[x][tr].mean())
        picks.append(a)
        cv[te] = tilt[a][te]
    b = boot(cv - base)
    res["tilt α 교차검증"] = {"score": float(cv.mean()), "delta": b, "picks": picks}
    print(f"  tilt α 교차검증 (고른 α {picks})        {cv.mean():.4f}  Δ {fmt_boot(b)}")
    # 최대 엔트로피 (주변확률 일치)
    w_res = []
    def wm(q):
        w, r = J.w_maxent(q, pi, marg, return_resid=True)
        w_res.append(r)
        return w
    p, v = J.run_chunks(cal, wm, "full")
    r = np.concatenate(w_res)
    report("maxent 511 전부", p, v)
    res["maxent_resid"] = {"p50": float(np.median(r)), "p99": float(np.percentile(r, 99)), "max": float(r.max()),
                           "sum_below_1": float((cal.sum(1) < 1.02).mean())}
    print(f"     maxent 주변확률 잔차 중앙 {np.median(r):.2e} / 99% {np.percentile(r, 99):.2e} / 최대 {r.max():.2e}; Σp<1.02 비율 {100*(cal.sum(1) < 1.02).mean():.1f}%")
    np.save(HERE / f"s02_maxent_pred_{name}.npy", p)
    np.save(HERE / f"s02_maxent_val_{name}.npy", v)
    p2, v2 = J.run_chunks(cal, lambda q: J.w_maxent(q, pi, marg), "topk")
    report("maxent 상위k", p2, v2)
    print(f"   ({time.time() - t0:.0f}초)")
    OUT[name] = res

(HERE / "s02_joint.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=2), encoding="utf-8")
