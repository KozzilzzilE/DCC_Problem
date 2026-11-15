"""반박 검증용 독립 구현. decision_4_1/ 의 common.py, joint.py 는 import 하지 않는다.

- 확률표는 npz 를 직접 읽는다 (probs, offsets, labels, call_ids).
- 정답 집합 511가지는 비트 코드(1..511)로 직접 만든다.
- maxent 는 뉴턴법이 아니라 반복 비례 맞춤(IPF)으로 511가지 전부에서 푼다 (알고리즘도 다르게).
- 예측은 순간 한 행의 확률만 입력으로 받는다. n_utt 는 채점(통화 평균)에만 쓴다.
- Validation 은 읽지 않는다. 사전분포는 fit_ids.txt 통화의 labels 필드만 쓴다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VER = Path(__file__).resolve().parent
ROOT = VER.parents[3]                       # mission4_symptom
PATHS = {
    "E001": ROOT / "runs/E001_m3_baseline/dev_prefix.npz",
    "E002": ROOT / "runs/E002_prefix_uniform/dev_prefix.npz",
    "E007": ROOT / "runs/E007_prefix_uniform_s43/dev_prefix.npz",
    "E003": ROOT / "runs/E003_prefix_plus_full_s43/dev_prefix.npz",
}
COMBOS = {"ENS3": ["E001", "E002", "E007"], "ENS2": ["E002", "E007"]}

CODES = np.arange(1, 512)
BITS = ((CODES[:, None] >> np.arange(9)[None, :]) & 1).astype(np.float64)      # (511, 9)
SIZE = BITS.sum(axis=1)
_I = BITS @ BITS.T
JACC = _I / (SIZE[:, None] + SIZE[None, :] - _I)                                # (예측 511, 정답 511)


def log_shift() -> np.ndarray:
    cfg = json.loads((ROOT / "runs/E002_prefix_uniform/run_config.json").read_text(encoding="utf-8"))
    return np.log(np.asarray(cfg["pos_weight"], dtype=np.float64))


def load(name: str) -> dict:
    names = COMBOS.get(name, [name])
    zs = [np.load(PATHS[n], allow_pickle=False) for n in names]
    ids = [str(x) for x in zs[0]["call_ids"]]
    for z in zs[1:]:
        assert [str(x) for x in z["call_ids"]] == ids and (z["offsets"] == zs[0]["offsets"]).all()
        assert (z["labels"] == zs[0]["labels"]).all()
    probs = np.mean([z["probs"].astype(np.float64) for z in zs], axis=0)
    off = zs[0]["offsets"].astype(np.int64)
    lab = zs[0]["labels"].astype(bool)
    n = np.diff(off)
    return {"ids": ids, "probs": probs, "off": off, "labels": lab, "n": n,
            "gold": np.repeat(lab, n, axis=0), "step": np.concatenate([np.arange(1, k + 1) for k in n])}


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def moment_jacc(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    inter = (pred & gold).sum(axis=1).astype(np.float64)
    union = (pred | gold).sum(axis=1).astype(np.float64)
    assert (union > 0).all()
    return inter / union


def per_call(ms: np.ndarray, t: dict) -> np.ndarray:
    return np.add.reduceat(ms, t["off"][:-1]) / t["n"]


# ----------------------------------------------------------------------------- 사전분포 (fit 통화 라벨만)
def fit_label_matrix() -> np.ndarray:
    fit = {l.strip() for l in open(ROOT / "runs/cache/fit_ids.txt", encoding="utf-8") if l.strip()}
    dev = {l.strip() for l in open(ROOT / "runs/cache/dev_ids.txt", encoding="utf-8") if l.strip()}
    assert not (fit & dev)
    rows = []
    with open(ROOT / "runs/cache/train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["call_id"] in fit:
                rows.append(r["labels"])
    Y = np.asarray(rows, dtype=np.float64)
    assert len(Y) == len(fit)
    return Y


def prior_from(Y: np.ndarray, smooth: float = 0.5):
    marg = Y.mean(axis=0)
    code = (Y @ (2 ** np.arange(9))).astype(int)
    cnt = np.bincount(code, minlength=512)[1:].astype(np.float64)
    ind = np.exp(BITS @ np.log(marg) + (1 - BITS) @ np.log1p(-marg))
    ind /= ind.sum()
    pi = cnt + smooth * ind
    pi /= pi.sum()
    return pi, marg


# ----------------------------------------------------------------------------- 정답 집합 분포
def w_indep(q: np.ndarray) -> np.ndarray:
    lo = np.log(q) - np.log1p(-q)
    lw = lo @ BITS.T
    lw -= lw.max(axis=1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(axis=1, keepdims=True)


def lift_to_sum(q: np.ndarray, min_sum: float) -> np.ndarray:
    """Σq < min_sum 인 행은 모든 로짓에 같은 상수를 더해 합을 min_sum 으로 올린다 (주장된 방법의 규격)."""
    q = q.copy()
    low = q.sum(axis=1) < min_sum
    if low.any():
        z = np.log(q[low]) - np.log1p(-q[low])
        a, b = np.zeros(low.sum()), np.full(low.sum(), 15.0)
        for _ in range(60):
            c = 0.5 * (a + b)
            s = sig(z + c[:, None]).sum(axis=1)
            b = np.where(s >= min_sum, c, b)
            a = np.where(s < min_sum, c, a)
        q[low] = np.clip(sig(z + b[:, None]), 1e-4, 1 - 1e-4)
    return q


def w_maxent_ipf(q: np.ndarray, pi: np.ndarray, min_sum: float = 1.02, tol: float = 2e-6, max_sweeps: int = 400):
    """π 에 대한 I-투영(주변확률 = q)을 IPF 로 푼다. (w (m,511), 잔차 (m,))"""
    q = lift_to_sum(q, min_sum)
    m = len(q)
    w = np.repeat(pi[None, :], m, axis=0)
    resid = np.full(m, np.inf)
    act = np.arange(m)
    for _ in range(max_sweeps):
        wa, qa = w[act], q[act]
        for j in range(9):
            mj = wa @ BITS[:, j]
            up = qa[:, j] / np.maximum(mj, 1e-300)
            dn = (1 - qa[:, j]) / np.maximum(1 - mj, 1e-300)
            wa *= dn[:, None] + BITS[None, :, j] * (up - dn)[:, None]
        r = np.abs(wa @ BITS - qa).max(axis=1)
        w[act] = wa
        resid[act] = r
        act = act[r > tol]
        if not len(act):
            break
    w /= w.sum(axis=1, keepdims=True)
    return w, resid


def w_tilt(q: np.ndarray, pi: np.ndarray, marg: np.ndarray, alpha: float) -> np.ndarray:
    th = alpha * ((np.log(q) - np.log1p(-q)) - (np.log(marg) - np.log1p(-marg))[None, :])
    lw = np.log(pi)[None, :] + th @ BITS.T
    lw -= lw.max(axis=1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(axis=1, keepdims=True)


def size_gain(pi: np.ndarray, marg: np.ndarray) -> np.ndarray:
    ind = np.exp(BITS @ np.log(marg) + (1 - BITS) @ np.log1p(-marg))
    ind /= ind.sum()
    g = np.array([pi[SIZE == s].sum() / ind[SIZE == s].sum() for s in range(1, 10)])
    return g[(SIZE - 1).astype(int)]


# ----------------------------------------------------------------------------- 결정
def pick_topk(w: np.ndarray, q: np.ndarray, max_k: int = 4):
    """q 내림차순 상위 k개 집합 중 기대 Jaccard 최대인 것. JACC 표에서 집합 코드로 바로 찾는다."""
    order = np.argsort(-q, axis=1, kind="stable")
    code = np.cumsum(1 << order, axis=1)                    # 상위 1..9개 집합의 비트 코드
    best = np.full(len(q), -1.0)
    bk = np.ones(len(q), dtype=np.int64)
    for k in range(1, max_k + 1):
        ej = np.einsum("ms,ms->m", w, JACC[code[:, k - 1] - 1])
        up = ej > best + 1e-12
        best[up] = ej[up]
        bk[up] = k
    rank = order.argsort(axis=1, kind="stable")
    return rank < bk[:, None], best


def pick_any(w: np.ndarray):
    val = w @ JACC.T
    b = val.argmax(axis=1)
    return BITS[b].astype(bool), val[np.arange(len(w)), b]


def run(q_all: np.ndarray, wfun, mode: str = "topk", max_k: int = 4, chunk: int = 8000):
    """q_all (m,9) 을 받아 (예측, 기대 점수, 잔차) 를 돌려준다. 행 단위 계산이라 다른 순간·통화 정보는 들어가지 않는다."""
    q_all = np.clip(q_all.astype(np.float64), 1e-4, 1 - 1e-4)
    pred = np.zeros(q_all.shape, dtype=bool)
    val = np.zeros(len(q_all))
    res = np.zeros(len(q_all))
    for s in range(0, len(q_all), chunk):
        q = q_all[s:s + chunk]
        out = wfun(q)
        if isinstance(out, tuple):
            w, res[s:s + chunk] = out
        else:
            w = out
        if mode == "topk":
            pred[s:s + chunk], val[s:s + chunk] = pick_topk(w, q, max_k)
        else:
            pred[s:s + chunk], val[s:s + chunk] = pick_any(w)
    return pred, val, res


def boot_ci(d: np.ndarray, B: int = 1000, seed: int = 20261112):
    rng = np.random.default_rng(seed)
    C = len(d)
    means = np.array([d[rng.integers(0, C, C)].mean() for _ in range(B)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fb(b) -> str:
    return f"{b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}]"
