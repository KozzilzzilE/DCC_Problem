"""exp08: exp06 에서 가장 좋았던 구성(R4': 기대 점수 + β δ̂, 목표는 통화 끝 확률 기준 효용 차이)을 엄격하게 다시 잰다.

- 완전 중첩 교차검증: 바깥 겹의 통화는 운영점 상수, 목표 계산, GBM 학습, (β, τ) 선택 어디에도 쓰이지 않는다.
- 겹 분할을 바꿔 가며 반복(REPS 회)해 분할 운에 따른 흔들림을 본다.
- 같은 분할에서 τ 만 고르는 기준(중첩 선택)과 τ=0.52 고정 기준을 함께 낸다.

    python exp08_validate.py ENS3 3
"""
from __future__ import annotations

import sys
import time

import lightgbm as lgb
import numpy as np

import common as C
import feats as F

M_NAME = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BETAS = (0.0, 1.0, 2.0, 4.0, 8.0)
TAUS = np.round(np.arange(0.40, 0.67, 0.02), 2)

ctx = C.Ctx(M_NAME)
calls = C.load_dev_calls(ctx)
v, cal, st, pr = ctx.value, ctx.cal, ctx.step, ctx.pred_thr
base = C.baseline(ctx)
stop0 = v >= 0.52
nn = np.repeat(ctx.n, ctx.n)

# 특징 (exp06 과 같음)
srt = -np.sort(-cal, axis=1)[:, :3]
same_prev = (pr == ctx.prev(pr)).all(axis=1) & ~ctx.first
run_len = np.zeros(ctx.M)
vmax = v.copy()
for i in range(1, ctx.M):
    if not ctx.first[i]:
        run_len[i] = run_len[i - 1] + 1 if same_prev[i] else 0
        vmax[i] = max(vmax[i - 1], v[i])
FA = np.column_stack([cal, srt, v, v - ctx.prev(v), vmax, run_len, cal.sum(axis=1), ctx.raw.max(axis=1), pr.sum(axis=1), st,
                      (cal - ctx.prev(cal)).max(axis=1), np.abs(cal - ctx.prev(cal)).sum(axis=1)])
FB = np.column_stack([FA, F.build(calls)[:, 1:], np.load(C.OUT / "dev_log_nhat.npy")])
SETS = {"FA": FA, "FB": FB}

# 기준 정책이 t 뒤에 끊는 순간
nxt = np.full(ctx.M, -1, dtype=np.int64)
for i in range(ctx.N):
    a, b = ctx.off[i], ctx.off[i + 1]
    cur = b - 1
    for j in range(b - 1, a - 1, -1):
        nxt[j] = cur if j < b - 1 else -1
        if stop0[j] and st[j] >= C.GRACE:
            cur = j
okn = (nxt >= 0) & (st >= C.GRACE)
qend = np.repeat(cal[ctx.off[1:] - 1], ctx.n, axis=0)
earl = C.earliness(st, nn)


def target(train_calls: np.ndarray) -> np.ndarray:
    """학습 통화의 기준 운영점 상수로 만든 목표 δq(t) = uq(t) - uq(기준 정책의 다음 중단 시점)."""
    idx = np.nonzero(train_calls)[0]
    tp, fp, fn = C.f1_counts(base[1][idx], ctx.labels[idx])
    Fm, f1 = C.macro_f1_from(tp, fp, fn)
    D = np.maximum(2 * tp + fp + fn, 1)
    E = float(C.earliness(base[0][idx], ctx.n[idx]).mean())
    uq = E * (pr * (qend * (2 - f1) / D - (1 - qend) * f1 / D)).sum(axis=1) * len(idx) / 9 + float(Fm) * earl
    d = np.zeros(ctx.M)
    d[okn] = uq[okn] - uq[nxt[okn]]
    return d


def gbm(seed=0):
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.06, num_leaves=15, min_child_samples=100, subsample=0.7,
                             subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=1, random_state=seed, verbose=-1)


def pick(score_fn, grid):
    best, bk = -1, None
    for key in grid:
        s = score_fn(key)
        if s > best + 1e-12:
            best, bk = s, key
    return bk


out = []
for rep in range(REPS):
    fold = np.random.default_rng(rep).permutation(ctx.N) % C.FOLDS      # rep 0 은 기존 스크립트와 같은 분할
    cf = fold[ctx.call_of]
    t0 = time.time()
    oof = {"tau": np.empty(ctx.N, dtype=np.int64)}
    cvs = {"fixed": [], "tau": []}
    picks = {}
    Htau = {float(t): C.first_stop(v >= t, ctx.off) for t in TAUS}
    for k in range(C.FOLDS):
        trc = fold != k
        tr_idx, te_idx = np.nonzero(trc)[0], np.nonzero(~trc)[0]
        cvs["fixed"].append(C.score_hp(*base, ctx.labels, ctx.n, te_idx)["score"])
        bt = pick(lambda t: C.score(ctx, Htau[t], pr, tr_idx)["score"], list(Htau))
        oof["tau"][te_idx] = Htau[bt][te_idx]
        cvs["tau"].append(C.score(ctx, Htau[bt], pr, te_idx)["score"])
        d = target(trc)
        inner = [f for f in range(C.FOLDS) if f != k]
        for name, X in SETS.items():
            dh_in = np.zeros(ctx.M)
            for kk in inner:
                tr = okn & np.isin(cf, [f for f in inner if f != kk])
                dh_in[cf == kk] = gbm().fit(X[tr], d[tr]).predict(X[cf == kk])
            dh_te = gbm().fit(X[okn & (cf != k)], d[okn & (cf != k)]).predict(X)
            for mode, grid in (("mix", [(b, float(t)) for b in BETAS for t in TAUS]),
                               ("pure", [(float(th),) for th in np.round(np.arange(-0.06, 0.031, 0.005), 3)])):
                def stop_of(key, dh):
                    return (v + key[0] * dh >= key[1]) if mode == "mix" else (dh >= key[0])
                bk = pick(lambda key: C.score(ctx, C.first_stop(stop_of(key, dh_in), ctx.off), pr, tr_idx)["score"], grid)
                h = C.first_stop(stop_of(bk, dh_te), ctx.off)
                tag = f"{name}_{mode}"
                oof.setdefault(tag, np.empty(ctx.N, dtype=np.int64))[te_idx] = h[te_idx]
                cvs.setdefault(tag, []).append(C.score(ctx, h, pr, te_idx)["score"])
                picks.setdefault(tag, []).append(bk)
    row = {"model": M_NAME, "rep": rep, "seconds": time.time() - t0, "cv": {k_: float(np.mean(x)) for k_, x in cvs.items()}, "picks": picks, "boot_vs_fixed": {}, "boot_vs_tau": {}}
    print(f"[{M_NAME}] 분할 {rep} ({row['seconds']:.0f}초): τ=0.52 고정 겹평균 {row['cv']['fixed']:.4f}, τ 중첩 선택 {row['cv']['tau']:.4f}", flush=True)
    tau_pair = (oof["tau"], C.take(ctx, oof["tau"], pr))
    for tag in [t for t in oof if t != "tau"]:
        pair = (oof[tag], C.take(ctx, oof[tag], pr))
        b1 = C.paired_boot(ctx, pair, base)
        b2 = C.paired_boot(ctx, pair, tau_pair)
        row["boot_vs_fixed"][tag], row["boot_vs_tau"][tag] = b1, b2
        s = C.score_hp(*pair, ctx.labels, ctx.n)
        print(f"    {tag:<8} 교차검증 {row['cv'][tag]:.4f} (고정 대비 {row['cv'][tag] - row['cv']['fixed']:+.4f}, τ선택 대비 {row['cv'][tag] - row['cv']['tau']:+.4f})  "
              f"OOF {s['score']:.4f} = {s['macro_f1']:.4f} x {s['earliness']:.4f}  고정 대비 {b1['diff']:+.4f} [{b1['lo']:+.4f}, {b1['hi']:+.4f}]  "
              f"τ선택 대비 {b2['diff']:+.4f} [{b2['lo']:+.4f}, {b2['hi']:+.4f}]  고른 값 {picks[tag]}", flush=True)
    out.append(row)
    C.save_json(f"exp08_validate_{M_NAME}.json", out)
