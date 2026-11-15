"""반박 검증: "보정 로짓에 온도 T<1" 주장 (lexical_errors/s07_temperature.py) 을 독립 코드로 다시 계산한다.

주장 쪽 스크립트(common.py, s07_*.py)는 import 하지 않는다. 기대 점수 규칙도 다른 알고리즘(포아송-이항 DP)으로
다시 짜고, m4.decision.expected_rule 과 값이 같은지만 대조한다. Validation 은 읽지 않는다.
예측에 쓰는 것은 그 순간의 9종 확률과 스칼라 T 뿐이다. n_utt 는 채점(통화 평균, 조기판단력)에만 쓴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

TABLES = {"E001": "runs/E001_m3_baseline/dev_prefix.npz", "E002": "runs/E002_prefix_uniform/dev_prefix.npz",
          "E007": "runs/E007_prefix_uniform_s43/dev_prefix.npz", "E003": "runs/E003_prefix_plus_full_s43/dev_prefix.npz"}
COMBOS = {"E001": ["E001"], "E002": ["E002"], "E007": ["E007"], "E003": ["E003"],
          "E002+E007": ["E002", "E007"], "ENS3": ["E001", "E002", "E007"]}
T_FINE = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
T_CLAIM = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
T_COARSE = [0.7, 0.85, 1.0, 1.2, 1.5]
TAUS = np.round(np.arange(0.40, 0.71, 0.02), 2)
OUT = {}


# ------------------------------------------------------------------ 자료
def load(key):
    z = np.load(ROOT / TABLES[key], allow_pickle=False)
    return z["call_ids"], z["offsets"].astype(np.int64), z["probs"], z["labels"].astype(bool)


ids0, OFF, _, LAB = load("E002")
for k in TABLES:                                    # 네 표가 같은 통화·같은 순간인지
    i, o, p, l = load(k)
    assert (i == ids0).all() and (o == OFF).all() and (l == LAB).all(), k
NC = len(ids0)
N = np.diff(OFF)
M = int(OFF[-1])
STEP = np.concatenate([np.arange(1, k + 1) for k in N])
GOLD = np.repeat(LAB, N, axis=0)
dev_ids = set(x.strip() for x in (ROOT / "runs/cache/dev_ids.txt").read_text(encoding="utf-8").split() if x.strip())
assert set(str(x) for x in ids0) == dev_ids, "확률표 통화가 dev 목록과 다르다"
SHIFT = np.log(np.asarray(json.loads((ROOT / "runs/E002_prefix_uniform/run_config.json").read_text(encoding="utf-8"))["pos_weight"]))
print(f"dev 통화 {NC}, 순간 {M}, 증상 수 분포 {np.bincount(LAB.sum(1))[1:6] / NC}")


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def raw_of(keys):
    return np.mean([load(k)[2] for k in keys], axis=0)


# ------------------------------------------------------------------ 기대 점수 규칙 (독립 구현)
def expected_topk(cal, max_k=4):
    """증상 독립 가정에서 E[ |상위k ∩ Y| / |상위k ∪ Y|  |  Y 비어 있지 않음 ] 이 최대인 k.

    상위 k개 중 정답 수 I, 나머지 중 정답 수 O 는 각각 포아송-이항 분포. 점수 = I / (k + O).
    """
    q = np.clip(cal.astype(np.float64), 1e-4, 1 - 1e-4)
    order = np.argsort(-q, axis=1, kind="stable")
    qs = np.take_along_axis(q, order, axis=1)
    m = len(q)
    p_empty = np.prod(1 - qs, axis=1)
    best = np.full(m, -1.0)
    best_k = np.ones(m, dtype=np.int64)
    for k in range(1, max_k + 1):
        di = np.zeros((m, k + 1)); di[:, 0] = 1.0
        for j in range(k):
            pj = qs[:, j:j + 1]
            di[:, 1:] = di[:, 1:] * (1 - pj) + di[:, :-1] * pj
            di[:, 0] = di[:, 0] * (1 - pj[:, 0])
        r = 9 - k
        do = np.zeros((m, r + 1)); do[:, 0] = 1.0
        for j in range(k, 9):
            pj = qs[:, j:j + 1]
            do[:, 1:] = do[:, 1:] * (1 - pj) + do[:, :-1] * pj
            do[:, 0] = do[:, 0] * (1 - pj[:, 0])
        gain = np.arange(k + 1)[:, None] / (k + np.arange(r + 1)[None, :])        # i / (k + o)
        e = np.einsum("mi,io,mo->m", di, gain, do) / (1 - p_empty)
        better = e > best + 1e-12
        best[better] = e[better]
        best_k[better] = k
    rank = order.argsort(axis=1, kind="stable")
    return rank < best_k[:, None], best


def call_scores(pred):
    inter = (pred & GOLD).sum(1)
    union = (pred | GOLD).sum(1)
    ms = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return np.add.reduceat(ms, OFF[:-1]) / N, ms


# ------------------------------------------------------------------ 0. 내 구현과 m4.decision 대조
from m4 import decision as dec  # noqa: E402  (대조용, 계산에는 쓰지 않는다)

cal0 = sig(logit(load("E002")[2]) - SHIFT)
p_mine, v_mine = expected_topk(cal0)
p_ref, v_ref = dec.expected_rule(dec.calibrate(load("E002")[2], SHIFT))
print(f"[0] 독립 구현 대조 (E002, T=1): 예측 불일치 순간 {(p_mine != p_ref).any(1).sum()} / {M}, 기대 점수 최대 차이 {np.abs(v_mine - v_ref).max():.2e}")
OUT["impl_check"] = {"pred_mismatch_moments": int((p_mine != p_ref).any(1).sum()), "value_max_abs_diff": float(np.abs(v_mine - v_ref).max())}

# ------------------------------------------------------------------ 1. 표 x 온도 격자
ALL_T = sorted(set(T_FINE) | set(T_COARSE))
S = {}       # S[표][T] = 통화별 4-1 점수
VAL = {}     # 중단 신호(기대 점수)
PRED = {}
MEANK = {}
for name, keys in COMBOS.items():
    raw = raw_of(keys)
    zc = logit(raw) - SHIFT
    S[name], VAL[name], MEANK[name] = {}, {}, {}
    for T in ALL_T:
        pred, val = expected_topk(sig(zc / T))
        S[name][T] = call_scores(pred)[0]
        VAL[name][T] = val
        MEANK[name][T] = float(pred.sum(1).mean())
        if T in (0.6, 1.0):
            PRED[(name, T)] = pred
print("\n[1] dev 전체 4-1 (괄호: 평균 예측 개수)")
OUT["grid41"] = {}
for name in COMBOS:
    OUT["grid41"][name] = {str(T): {"m4_1": float(S[name][T].mean()), "mean_k": MEANK[name][T]} for T in ALL_T}
    print(f"  {name:<10}" + "  ".join(f"T{T}:{S[name][T].mean():.4f}({MEANK[name][T]:.2f})" for T in ALL_T))


def boot_ci(d, n_boot=2000, seed=123):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    b = d[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], float((b > 0).mean()), float((b > 0.002).mean())


def fmt(ci):
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


# ------------------------------------------------------------------ 2. 5겹 교차검증 (주장과 같은 겹 배정) + 짝 부트스트랩
def cv41(name, grid, fold):
    oof = np.zeros(NC)
    picks = []
    for k in range(5):
        tr, te = fold != k, fold == k
        b = max(grid, key=lambda T: S[name][T][tr].mean())
        oof[te] = S[name][b][te]
        picks.append(b)
    return oof, picks


print("\n[2] 5겹 교차검증으로 T 선택 (겹 배정 seed 0 = 주장과 같은 배정), 기준 T=1 과의 차이, 통화 단위 짝 부트스트랩 2000회")
fold0 = np.random.default_rng(0).permutation(NC) % 5
OUT["cv41"] = {}
for gname, grid in (("주장 격자", T_CLAIM), ("촘촘", T_FINE), ("거친", T_COARSE)):
    for name in COMBOS:
        oof, picks = cv41(name, grid, fold0)
        fold_means = np.mean([oof[fold0 == k].mean() for k in range(5)])
        d = oof - S[name][1.0]
        ci, p0, p2 = boot_ci(d)
        OUT["cv41"][f"{gname}|{name}"] = {"cv": float(fold_means), "base": float(S[name][1.0].mean()), "delta": float(d.mean()), "ci": ci,
                                           "picks": picks, "P(d>0)": p0, "P(d>0.002)": p2}
        print(f"  {gname:<6} {name:<10} 교차검증 {fold_means:.4f}  기준 {S[name][1.0].mean():.4f}  차이 {d.mean():+.4f} {fmt(ci)}  "
              f"P(>0)={p0:.3f} P(>0.002)={p2:.3f}  고른 T {picks}")

# ------------------------------------------------------------------ 3. 겹 배정을 바꿔 가며 (50가지) 교차검증 이득의 분포
print("\n[3] 겹 배정 seed 50가지: 교차검증 이득(촘촘 격자)의 평균·최소·최대, 고른 T 의 분포")
OUT["cv41_reseed"] = {}
for name in COMBOS:
    ds, allp = [], []
    for s in range(1, 51):
        fold = np.random.default_rng(1000 + s).permutation(NC) % 5
        oof, picks = cv41(name, T_FINE, fold)
        ds.append(float((oof - S[name][1.0]).mean()))
        allp += picks
    u, c = np.unique(allp, return_counts=True)
    OUT["cv41_reseed"][name] = {"mean": float(np.mean(ds)), "min": float(np.min(ds)), "max": float(np.max(ds)),
                                "picks": {str(a): int(b) for a, b in zip(u, c)}}
    print(f"  {name:<10} 평균 {np.mean(ds):+.4f}  최소 {np.min(ds):+.4f}  최대 {np.max(ds):+.4f}  고른 T {dict(zip(u.tolist(), c.tolist()))}")

# ------------------------------------------------------------------ 4. 선택을 포함한 부트스트랩 (뽑힌 통화에서 T 선택 -> 안 뽑힌 통화에서 채점)
print("\n[4] 선택 포함 부트스트랩 2000회: 뽑힌 통화에서 T 를 고르고, 안 뽑힌 통화(약 37%)에서 T=1 과의 차이를 잰다 (촘촘 격자)")
OUT["oob41"] = {}
for name in COMBOS:
    mat = np.stack([S[name][T] for T in T_FINE], axis=1)
    base = S[name][1.0]
    rng = np.random.default_rng(7)
    ds, picks = [], []
    for _ in range(2000):
        cnt = np.bincount(rng.integers(0, NC, NC), minlength=NC)
        b = int(np.argmax(cnt @ mat))
        oob = cnt == 0
        ds.append(float((mat[oob, b] - base[oob]).mean()))
        picks.append(T_FINE[b])
    ds = np.asarray(ds)
    u, c = np.unique(picks, return_counts=True)
    OUT["oob41"][name] = {"mean": float(ds.mean()), "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                          "P(d>0)": float((ds > 0).mean()), "P(d>0.002)": float((ds > 0.002).mean()),
                          "picks": {str(a): int(b) for a, b in zip(u, c)}}
    print(f"  {name:<10} 평균 {ds.mean():+.4f} {fmt(OUT['oob41'][name]['ci'])}  P(>0)={(ds > 0).mean():.3f} P(>0.002)={(ds > 0.002).mean():.3f}  "
          f"고른 T {dict(zip(u.tolist(), c.tolist()))}")

# ------------------------------------------------------------------ 5. 표 사이 이전: 한 표에서 고른 T 를 다른 표에 고정 적용 (대상 표에서는 아무것도 고르지 않는다)
print("\n[5] 표 사이 이전: 원천 표의 dev 전체 최적 T 를 대상 표에 고정 적용, T=1 과의 차이")
OUT["transfer41"] = {}
for src in ("E002", "E007", "E001"):
    Tsrc = max(T_FINE, key=lambda T: S[src][T].mean())
    for dst in COMBOS:
        if dst == src:
            continue
        d = S[dst][Tsrc] - S[dst][1.0]
        ci, p0, p2 = boot_ci(d)
        OUT["transfer41"][f"{src}->{dst}"] = {"T": Tsrc, "delta": float(d.mean()), "ci": ci, "P(d>0)": p0, "P(d>0.002)": p2}
        print(f"  {src}(T={Tsrc}) -> {dst:<10} {d.mean():+.4f} {fmt(ci)}  P(>0)={p0:.3f} P(>0.002)={p2:.3f}")

# ------------------------------------------------------------------ 6. seed 흔들림 크기와 비교, 개수 제한과 비교
print("\n[6] 비교 기준")
d = S["E007"][1.0] - S["E002"][1.0]
ci, _, _ = boot_ci(d)
print(f"  seed 차이 (E007 − E002, T=1): {d.mean():+.4f} {fmt(ci)}")
d = S["E007"][0.6] - S["E002"][0.6]
ci2, _, _ = boot_ci(d)
print(f"  seed 차이 (E007 − E002, T=0.6): {d.mean():+.4f} {fmt(ci2)}")
OUT["seed_noise"] = {"T1": {"delta": float((S['E007'][1.0] - S['E002'][1.0]).mean()), "ci": ci}, "T0.6": {"delta": float(d.mean()), "ci": ci2}}
OUT["count_rules"] = {}
for name in ("E002", "E007", "E001", "ENS3"):
    raw = raw_of(COMBOS[name])
    cal = sig(logit(raw) - SHIFT)
    row = {}
    for mk in (1, 2, 3):
        pred, _ = expected_topk(cal, max_k=mk)
        sc = call_scores(pred)[0]
        d = sc - S[name][1.0]
        ci, p0, p2 = boot_ci(d)
        d6 = S[name][0.6] - sc
        ci6, _, _ = boot_ci(d6)
        row[f"max_k={mk}"] = {"m4_1": float(sc.mean()), "mean_k": float(pred.sum(1).mean()), "delta_vs_T1": float(d.mean()), "ci": ci,
                              "T0.6_minus_this": float(d6.mean()), "ci_T0.6_minus_this": ci6}
    OUT["count_rules"][name] = row
    print(f"  {name:<6} 개수 상한만 바꿈(T=1): " + "  ".join(
        f"{k} {v['m4_1']:.4f}({v['mean_k']:.2f}개, T=1 대비 {v['delta_vs_T1']:+.4f} {fmt(v['ci'])}; T=0.6 − 이것 {v['T0.6_minus_this']:+.4f} {fmt(v['ci_T0.6_minus_this'])})"
        for k, v in row.items()))

# ------------------------------------------------------------------ 7. 이득이 어느 발화 순번에서 나오는가 (T=0.6 − T=1, 순간 점수)
print("\n[7] 발화 순번별 순간 점수 차이 (T=0.6 − T=1), 순간 단순 평균")
OUT["by_step"] = {}
for name in ("E002", "E007", "ENS3"):
    ms6 = call_scores(PRED[(name, 0.6)])[1]
    ms1 = call_scores(PRED[(name, 1.0)])[1]
    row = {}
    for lab, m in (("1-4", STEP <= 4), ("5-9", (STEP >= 5) & (STEP <= 9)), ("10-19", (STEP >= 10) & (STEP <= 19)), ("20+", STEP >= 20)):
        row[lab] = {"n": int(m.sum()), "T1": float(ms1[m].mean()), "T0.6": float(ms6[m].mean()), "delta": float((ms6[m] - ms1[m]).mean()),
                    "k_T1": float(PRED[(name, 1.0)][m].sum(1).mean()), "k_T0.6": float(PRED[(name, 0.6)][m].sum(1).mean())}
    OUT["by_step"][name] = row
    print(f"  {name:<6} " + "  ".join(f"{k}: {v['delta']:+.4f} (개수 {v['k_T1']:.2f}->{v['k_T0.6']:.2f})" for k, v in row.items()))
# 정답 증상 개수별 (분석용. 예측에는 쓰지 않는다)
nsym = LAB.sum(1)
print("  정답 증상 개수별 통화 점수 차이 (T=0.6 − T=1): " + " | ".join(
    f"{name} " + " ".join(f"{k}개 {float((S[name][0.6] - S[name][1.0])[np.minimum(nsym, 3) == k].mean()):+.4f}" for k in (1, 2, 3))
    for name in ("E002", "E007", "ENS3")))
OUT["by_nsym"] = {name: {str(k): float((S[name][0.6] - S[name][1.0])[np.minimum(nsym, 3) == k].mean()) for k in (1, 2, 3)} for name in ("E002", "E007", "ENS3")}


# ------------------------------------------------------------------ 8. 4-2: 중단 신호만 온도를 준 기대 점수로 (중단 시점 예측은 보정 전 0.5, 비면 1개)
def first_stop(stop):
    cand = np.where(stop & (STEP >= 5), STEP, 10 ** 9)
    return np.minimum(np.minimum.reduceat(cand, OFF[:-1]), N)


def f1e(pred, gold, earl, w=None):
    w = np.ones(len(pred)) if w is None else w
    tp = (w[:, None] * (pred & gold)).sum(0); fp = (w[:, None] * (pred & ~gold)).sum(0); fn = (w[:, None] * (~pred & gold)).sum(0)
    seen = (tp + fp + fn) > 0
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
    return float(f1[seen].mean()) * float((w * earl).sum() / w.sum())


print("\n[8] 4-2: (T, τ) 를 5겹 교차검증으로 함께 고름 vs T=1 에서 τ 만 고름 (겹 배정 seed 0), 겹 밖 예측을 모은 점수 차이와 부트스트랩 1000회")
OUT["cv42"] = {}
for name in ("E002", "E007", "ENS3", "E001"):
    raw = raw_of(COMBOS[name])
    p42 = raw >= 0.5
    empty = ~p42.any(1)
    p42[empty, raw[empty].argmax(1)] = True
    H = {}
    for T in T_CLAIM:
        for t in TAUS:
            h = first_stop(VAL[name][T] >= t)
            H[(T, t)] = (p42[OFF[:-1] + h - 1], 0.5 + 0.5 * np.maximum(0.0, 1 - np.maximum(h, 5) / N))

    def run(cands):
        oofp = np.zeros((NC, 9), dtype=bool); oofe = np.zeros(NC); fm = []; picks = []
        for k in range(5):
            tr, te = fold0 != k, fold0 == k
            b = max(cands, key=lambda c: f1e(H[c][0][tr], LAB[tr], H[c][1][tr]))
            oofp[te], oofe[te] = H[b][0][te], H[b][1][te]
            fm.append(f1e(H[b][0][te], LAB[te], H[b][1][te])); picks.append(b)
        return float(np.mean(fm)), oofp, oofe, picks

    a = run(list(H.keys()))
    b = run([(1.0, t) for t in TAUS])
    rng = np.random.default_rng(11)
    ds = []
    for _ in range(1000):
        w = np.bincount(rng.integers(0, NC, NC), minlength=NC).astype(float)
        ds.append(f1e(a[1], LAB, a[2], w) - f1e(b[1], LAB, b[2], w))
    dpool = f1e(a[1], LAB, a[2]) - f1e(b[1], LAB, b[2])
    ci = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
    fixed = f1e(*H[(1.0, 0.52)][:1], LAB, H[(1.0, 0.52)][1])
    OUT["cv42"][name] = {"fixed_T1_tau0.52": fixed, "cv_T_tau": a[0], "cv_tau_only": b[0], "pooled_delta": dpool, "ci": ci,
                         "picks": [[float(x), float(y)] for x, y in a[3]]}
    print(f"  {name:<6} 고정(T=1, τ=0.52) {fixed:.4f} | 교차검증 T·τ {a[0]:.4f}  τ만 {b[0]:.4f}  겹밖 모음 차이 {dpool:+.4f} {fmt(ci)}  고른 값 {a[3]}")

(HERE / "v01_temperature.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n저장", HERE / "v01_temperature.json")
