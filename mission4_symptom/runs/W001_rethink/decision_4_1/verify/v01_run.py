"""확률표 하나에 대해 주장된 방법들을 독립 구현으로 다시 돌리고 통화별 점수를 v01_<표>.npz 로 저장한다.

    python v01_run.py E002 [full|lite]

full: maxent (T, λ) 격자 9칸 + 독립 격자 18칸 + 나머지.  lite: maxent 는 그대로/λ=1.3 두 칸만.
"""
import sys
import time

import numpy as np

import vlib as V

for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8", errors="replace")

name = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "full"
t = V.load(name)
sh = V.log_shift()
z = V.logit(t["probs"])
Y = V.fit_label_matrix()
pi, marg = V.prior_from(Y)
out = {}
T0 = time.time()


def q_of(T, L):
    return V.sig(z / T - L * sh[None, :])


def keep(key, pred, val=None):
    ms = V.moment_jacc(pred, t["gold"])
    out[key] = V.per_call(ms, t)
    out["k|" + key] = np.array([pred.sum(1).mean()])
    print(f"  {key:<18} {out[key].mean():.4f}  평균 k {pred.sum(1).mean():.2f}  ({time.time() - T0:.0f}초)", flush=True)
    return ms


cal = q_of(1.0, 1.0)

# ---- 기준: 독립 가정 상위 k (max_k=4). 공식 구현(m4.decision.expected_rule)과 예측이 같은지도 확인한다.
p0, v0, _ = V.run(cal, V.w_indep)
ms0 = keep("base", p0)
sys.path.insert(0, str(V.ROOT))
from m4 import decision as dec  # noqa: E402  (대조 전용)
pref, vref = dec.expected_rule(dec.calibrate(t["probs"], sh))
print(f"  공식 expected_rule 과 예측 불일치 행: {(pref != p0).any(1).sum()} / {len(p0)}", flush=True)
out["base_official"] = V.per_call(V.moment_jacc(pref, t["gold"]), t)
out["mom|base"] = ms0.astype(np.float32)
out["sig|base"] = v0.astype(np.float32)

# ---- 주장 1: maxent 그대로
resid = {}
def mx(q, min_sum=1.02, prior=pi):
    return V.w_maxent_ipf(q, prior, min_sum=min_sum)

p1, v1, r1 = V.run(cal, mx)
ms1 = keep("M|1.0|1.0", p1)
out["mom|M"] = ms1.astype(np.float32)
out["sig|M"] = v1.astype(np.float32)
out["kk|base"] = p0.sum(1).astype(np.int8)
out["kk|M"] = p1.sum(1).astype(np.int8)
out["resid|M"] = np.array([np.median(r1), np.percentile(r1, 99), r1.max(), (cal.sum(1) < 1.02).mean()])
print(f"     IPF 잔차 중앙 {np.median(r1):.1e} 99% {np.percentile(r1, 99):.1e} 최대 {r1.max():.1e}; Σq<1.02 {100 * (cal.sum(1) < 1.02).mean():.1f}%", flush=True)
p, _, _ = V.run(cal, mx, max_k=2)
keep("M_max2", p)
p, _, _ = V.run(cal, mx, mode="any")
keep("M_any", p)

# ---- 주장 1 민감도: 고르지 않았다는 고정값(min_sum, 평활, 사전분포 표본)을 바꿔도 같은가
for ms_ in (1.0001, 1.2):
    p, _, _ = V.run(cal, lambda q: mx(q, min_sum=ms_))
    keep(f"M_minsum{ms_}", p)
for sm in (0.0001, 20.0):
    pi2, _ = V.prior_from(Y, smooth=sm)
    p, _, _ = V.run(cal, lambda q: mx(q, prior=pi2))
    keep(f"M_smooth{sm}", p)
rng = np.random.default_rng(7)
half = rng.permutation(len(Y))[: len(Y) // 2]
pi3, _ = V.prior_from(Y[half])
p, _, _ = V.run(cal, lambda q: mx(q, prior=pi3))
keep("M_halfprior", p)

# ---- 주장 2: maxent + (T, λ)
grid_m = [(T, L) for T in (0.8, 1.0, 1.25) for L in (0.7, 1.0, 1.3)] if mode == "full" else [(1.0, 1.3)]
for T, L in grid_m:
    if (T, L) == (1.0, 1.0):
        continue
    p, _, _ = V.run(q_of(T, L), mx)
    keep(f"M|{T}|{L}", p)
if mode == "full":
    for T, L in ((1.0, 1.6), (1.25, 1.6), (1.0, 2.0)):          # 격자 끝(λ=1.3) 밖
        p, _, _ = V.run(q_of(T, L), mx)
        keep(f"Mx|{T}|{L}", p)

# ---- 주장 3: 독립 + (T, λ), 최대 개수, 개수 분포 보정, 틸트
for T in (0.6, 0.8, 1.0):
    for L in (1.0, 1.4, 1.8, 2.2, 2.8, 3.5):
        if (T, L) == (1.0, 1.0):
            out["I|1.0|1.0"] = out["base"]
            continue
        p, _, _ = V.run(q_of(T, L), V.w_indep)
        keep(f"I|{T}|{L}", p)
for T, L in ((0.4, 1.8), (0.4, 2.2), (0.5, 1.8)):                # 격자 끝(T=0.6) 밖
    p, _, _ = V.run(q_of(T, L), V.w_indep)
    keep(f"Ix|{T}|{L}", p)
for mk in (1, 2, 3, 9):
    p, _, _ = V.run(cal, V.w_indep, max_k=mk)
    keep(f"I_max{mk}", p)
p, _, _ = V.run(cal, V.w_indep, mode="any")
keep("I_any", p)
g = V.size_gain(pi, marg)
for pw in (0.5, 1.0):
    def ws(q, pw=pw):
        w = V.w_indep(q) * g[None, :] ** pw
        return w / w.sum(axis=1, keepdims=True)
    p, _, _ = V.run(cal, ws)
    keep(f"size^{pw}", p)
p, _, _ = V.run(cal, lambda q: V.w_tilt(q, pi, marg, 1.0), mode="any")
keep("tilt1_any", p)
p, _, _ = V.run(cal, lambda q: V.w_tilt(q, pi, marg, 1.0))
keep("tilt1_topk", p)

np.savez(V.VER / f"v01_{name}.npz", **out)
print(f"끝 {name} ({time.time() - T0:.0f}초)", flush=True)
