"""v02: 학습형 중단 정책(기대 점수 + β·δ̂ >= τ)의 독립 재계산. 완전 중첩 5겹 교차검증.

    python v02_nested.py <모델> <분할 seed 들, 쉼표> <특징: FB|FA|FBx(길이 추정 뺌)|STEP> [learner: lgb|hgb]

바깥 겹 k 의 통화는 (a) 운영점 상수 (b) 목표 δ (c) GBM 학습 (d) (β, τ) 선택 어디에도 들어가지 않는다.
분할 seed 는 대상 에이전트(0, 1, 2)와 겹치지 않게 101 부터 쓴다.

비교하는 정책 (모두 중단 시점 예측은 보정 전 0.5·비면 1개)
  fixed     기대 점수 >= 0.52                       (기준)
  tau       기대 점수 >= τ, τ 는 4겹에서 선택        (정직한 기준: τ 한 개를 고르는 자유도는 기준에도 준다)
  mix       기대 점수 + β·δ̂ >= τ, (β, τ) 는 4겹 안쪽 교차 학습 예측으로 선택   ← 주장된 방법
  mix456    β=4, τ=0.56 고정 (대상 에이전트가 dev 를 보고 정한 값이라 참고용)
"""
from __future__ import annotations

import sys
import time

import numpy as np

import v_common as V

NAME = sys.argv[1]
SEEDS = [int(x) for x in sys.argv[2].split(",")]
FSET = sys.argv[3] if len(sys.argv) > 3 else "FB"
LEARNER = sys.argv[4] if len(sys.argv) > 4 else "lgb"
BETAS = (0.0, 1.0, 2.0, 4.0, 8.0)
TAUS = [round(float(t), 2) for t in np.arange(0.40, 0.67, 0.02)]
K = 5

md = V.Model(NAME)
fs = V.FastScorer(md)
z = np.load(V.OUT / "v_feats.npz")
PF = V.prob_feats(md)
X = {"FB": np.column_stack([PF, z["conv"], z["nhat"]]), "FBx": np.column_stack([PF, z["conv"]]), "FA": PF,
     "STEP": md.step[:, None].astype(np.float64), "PLACEBO": np.column_stack([PF, z["conv"], z["nhat"]])}[FSET]
PLACEBO_RNG = np.random.default_rng(999)


def fit_predict(tr, te_rows, d):
    """tr 행으로 학습해 te_rows 행을 예측. PLACEBO 는 목표를 학습 행 안에서 뒤섞는다(특징과 목표의 관계를 끊은 대조군)."""
    y = d[tr]
    if FSET == "PLACEBO":
        y = PLACEBO_RNG.permutation(y)
    return new_gbm().fit(X[tr], y).predict(X[te_rows])
vM = md.pad(md.v, -1.0)
h_fixed = md.halts(vM, 0.52)
nn = np.repeat(md.n, md.n)
earl_t = V.earl(md.step, nn)                                   # 학습 목표(라벨) 계산용. 특징에는 들어가지 않는다
is_last = md.step == nn

# 기준 정책이 t 뒤(> t)에 끊는 순간. 마지막 발화에서는 정의하지 않는다.
stop0 = (md.v >= 0.52) & (md.step >= V.GRACE)
nxt = np.full(md.M, -1, dtype=np.int64)
for i in range(md.N):
    a, b = int(md.off[i]), int(md.off[i + 1])
    upcoming = b - 1
    for j in range(b - 2, a - 1, -1):
        if stop0[j + 1]:
            upcoming = j + 1
        nxt[j] = upcoming
rows_ok = (md.step >= V.GRACE) & ~is_last
assert (nxt[rows_ok] > np.nonzero(rows_ok)[0]).all()
q_end = np.repeat(md.cal[md.off[1:] - 1], md.n, axis=0)        # 통화 끝의 보정 확률 (목표 계산용)
predf = md.pred.astype(np.float64)


def target(tr_calls: np.ndarray) -> np.ndarray:
    """δq(t) = uq(t) - uq(기준 정책의 다음 중단 순간). 상수는 학습 통화에서만 계산한다."""
    p = md.pred_at(h_fixed)[tr_calls]
    lab = md.labels[tr_calls]
    tp, fp, fn = (p & lab).sum(0).astype(float), (p & ~lab).sum(0).astype(float), (~p & lab).sum(0).astype(float)
    D = np.maximum(2 * tp + fp + fn, 1.0)
    f1 = 2 * tp / D
    F = f1.mean()
    E = float(V.earl(h_fixed[tr_calls], md.n[tr_calls]).mean())
    ntr = int(tr_calls.sum())
    gain = (q_end * (2 - f1) - (1 - q_end) * f1) / D             # 클래스 c 를 예측에 넣을 때 Macro-F1x9 의 기대 변화
    uq = E * ntr / 9.0 * (predf * gain).sum(axis=1) + F * earl_t
    d = np.zeros(md.M)
    d[rows_ok] = uq[rows_ok] - uq[nxt[rows_ok]]
    return d


def new_gbm():
    if LEARNER == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=100,
                                             l2_regularization=5.0, random_state=0)
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.06, num_leaves=15, min_child_samples=100, subsample=0.7, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=5.0, n_jobs=2, random_state=0, verbose=-1)


def best_key(cands: dict, mask: np.ndarray):
    """cands {키: 통화별 중단 시점}. mask 통화에서 점수가 가장 높은 키 (동점이면 먼저 나온 것)."""
    best, bk = -1.0, None
    for key, h in cands.items():
        s = fs.score(h, mask)
        if s > best + 1e-12:
            best, bk = s, key
    return bk


H_TAU = {t: md.halts(vM, t) for t in TAUS}
W = V.boot_weights(md.N)
results = []
oof_store = {}
for seed in SEEDS:
    t0 = time.time()
    fold = np.random.default_rng(seed).permutation(md.N) % K
    cf = fold[md.call_of]
    oof = {k_: np.zeros(md.N, dtype=np.int64) for k_ in ("tau", "mix", "mix456")}
    picks = {"tau": [], "mix": []}
    for k in range(K):
        trc, tec = fold != k, fold == k
        bt = best_key(H_TAU, trc)
        oof["tau"][tec] = H_TAU[bt][tec]
        picks["tau"].append(bt)
        d = target(trc)
        inner = [f for f in range(K) if f != k]
        dh_in = np.zeros(md.M)
        for kk in inner:
            tr = rows_ok & np.isin(cf, [f for f in inner if f != kk])
            te = cf == kk
            dh_in[te] = fit_predict(tr, te, d)
        tr = rows_ok & (cf != k)
        assert not np.isin(md.call_of[tr], np.nonzero(tec)[0]).any()          # 남은 겹 통화가 학습 행에 없다
        dh_te = fit_predict(tr, np.ones(md.M, dtype=bool), d)
        dM_in, dM_te = md.pad(dh_in, 0.0), md.pad(dh_te, 0.0)
        cands = {(b, t): md.halts(vM + b * dM_in, t) for b in BETAS for t in TAUS}
        bk = best_key(cands, trc)
        picks["mix"].append(bk)
        oof["mix"][tec] = md.halts(vM + bk[0] * dM_te, bk[1])[tec]
        oof["mix456"][tec] = md.halts(vM + 4.0 * dM_te, 0.56)[tec]
    row = {"model": NAME, "seed": seed, "fset": FSET, "learner": LEARNER, "fixed": V.score42(md, h_fixed), "picks": {k_: [str(x) for x in v_] for k_, v_ in picks.items()}}
    line = f"[{NAME} {FSET} {LEARNER}] 분할 {seed} ({time.time() - t0:.0f}초)  고정 {row['fixed']['score']:.4f}"
    for tag in oof:
        s = V.score42(md, oof[tag])
        fm = float(np.mean([fs.score(oof[tag], fold == k) - fs.score(h_fixed, fold == k) for k in range(K)]))
        row[tag] = {**s, "never": float((oof[tag] == md.n).mean()), "foldmean_minus_fixed": fm,
                    "vs_fixed": V.paired_boot(md, oof[tag], h_fixed, W), "vs_tau": V.paired_boot(md, oof[tag], oof["tau"], W)}
        b1, b2 = row[tag]["vs_fixed"], row[tag]["vs_tau"]
        line += f"\n    {tag:<7} OOF {s['score']:.4f} = {s['f1']:.4f} x {s['earl']:.4f}  끝까지 {row[tag]['never']:.3f}  고정 대비 {b1['diff']:+.4f} [{b1['lo']:+.4f}, {b1['hi']:+.4f}] (겹평균 {fm:+.4f})"
        if tag != "tau":
            line += f"  τ선택 대비 {b2['diff']:+.4f} [{b2['lo']:+.4f}, {b2['hi']:+.4f}]"
    line += f"\n    고른 값 mix {picks['mix']}  tau {picks['tau']}"
    print(line, flush=True)
    results.append(row)
    for tag in oof:
        oof_store[f"{tag}_{seed}"] = oof[tag]
    V.save_json(V.OUT / f"v02_{NAME}_{FSET}_{LEARNER}_{SEEDS[0]}.json", results)
    np.savez_compressed(V.OUT / f"v02_{NAME}_{FSET}_{LEARNER}_{SEEDS[0]}_oof.npz", fixed=h_fixed, **oof_store)
