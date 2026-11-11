"""정답 집합 511가지의 사후분포를 만드는 여러 방법과, 그 분포에서 기대 Jaccard 가 최대인 예측 집합을 고르는 함수.

indep      : 지금 채택 규칙. w(S) ∝ Π p_i^{s_i} (1-p_i)^{1-s_i}, 빈 집합 제외 후 정규화.
size       : w(S) ∝ indep(S) × g(|S|).  g 는 학습용(fit) 라벨의 개수 분포 / 사전 주변확률로 만든 독립 분포의 개수 분포.
tilt       : w(S) ∝ π(S) × Π (odds(p_i)/odds(π_i))^{α s_i}.  π 는 fit 라벨의 실제 결합 분포(511칸, 평활).
maxent     : w(S) ∝ π(S) exp(θ·S), θ 는 주변확률이 p 와 같아지도록 반복해 맞춘다 (불가능하면 가장 가까운 곳에서 멈춤).
"""
from __future__ import annotations

import numpy as np

from common import ROOT, SETS, SET_SIZE, load_cache, read_ids

_INTER = SETS @ SETS.T                                   # (511 예측, 511 정답) 교집합 크기
_UNION = SET_SIZE[:, None] + SET_SIZE[None, :] - _INTER
JACC = _INTER / _UNION                                   # (P, S)
_CODE = (SETS @ (2 ** np.arange(9))).astype(int)         # 집합 -> 비트 코드


def fit_labels() -> np.ndarray:
    ids = set(read_ids(ROOT / "runs/cache/fit_ids.txt"))
    calls = load_cache(ROOT / "runs/cache/train.jsonl", ids)
    return np.asarray([c.labels for c in calls], dtype=np.float64)


def prior_joint(Y: np.ndarray, smooth: float = 0.5):
    """fit 라벨의 결합 분포 π(S) (511,), 주변확률 π_i (9,), 개수 분포.
    평활: 독립 분포를 pseudo-count `smooth` 건만큼 섞는다 (관측되지 않은 조합도 0 이 되지 않게)."""
    marg = Y.mean(axis=0)
    codes = (Y @ (2 ** np.arange(9))).astype(int)
    cnt = np.bincount(codes, minlength=512).astype(np.float64)[_CODE]
    ind = np.exp(SETS @ np.log(marg) + (1 - SETS) @ np.log(1 - marg))
    ind /= ind.sum()
    pi = cnt + smooth * ind
    pi /= pi.sum()
    size = np.array([pi[SET_SIZE == s].sum() for s in range(1, 10)])
    return pi, marg, size


def w_indep(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, 1e-4, 1 - 1e-4)
    lw = np.log(q) @ SETS.T + np.log(1 - q) @ (1 - SETS).T
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def size_factor(marg: np.ndarray, size_true: np.ndarray) -> np.ndarray:
    ind = np.exp(SETS @ np.log(marg) + (1 - SETS) @ np.log(1 - marg))
    ind /= ind.sum()
    size_ind = np.array([ind[SET_SIZE == s].sum() for s in range(1, 10)])
    g = size_true / np.maximum(size_ind, 1e-12)
    return g[(SET_SIZE - 1).astype(int)]                  # (511,)


def w_size(q: np.ndarray, g_set: np.ndarray, power: float = 1.0) -> np.ndarray:
    w = w_indep(q) * g_set[None, :] ** power
    return w / w.sum(axis=1, keepdims=True)


def w_tilt(q: np.ndarray, pi: np.ndarray, marg: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    q = np.clip(q, 1e-4, 1 - 1e-4)
    theta = alpha * (np.log(q / (1 - q)) - np.log(marg / (1 - marg))[None, :])
    lw = np.log(pi)[None, :] + theta @ SETS.T
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def w_maxent(q: np.ndarray, pi: np.ndarray, marg: np.ndarray, iters: int = 60, damp: float = 0.7,
             min_sum: float = 1.02, return_resid: bool = False):
    """주변확률을 q 에 맞춘 최대 엔트로피(π 기준) 분포. Σq < 1 이면 불가능하므로 odds 를 같은 비율로 키워 min_sum 에 맞춘다."""
    q = np.clip(q.astype(np.float64), 1e-4, 1 - 1e-4)
    s = q.sum(axis=1)
    low = s < min_sum
    if low.any():
        # odds 에 공통 배수를 곱해 합이 min_sum 이 되게 한다 (이분법)
        ql = q[low]
        lo, hi = np.zeros(len(ql)), np.full(len(ql), 12.0)
        od = np.log(ql / (1 - ql))
        for _ in range(40):
            mid = (lo + hi) / 2
            ss = (1 / (1 + np.exp(-(od + mid[:, None])))).sum(axis=1)
            hi = np.where(ss >= min_sum, mid, hi)
            lo = np.where(ss < min_sum, mid, lo)
        q = q.copy()
        q[low] = np.clip(1 / (1 + np.exp(-(od + hi[:, None]))), 1e-4, 1 - 1e-4)
    target = np.log(q / (1 - q))
    theta = target - np.log(marg / (1 - marg))[None, :]
    lpi = np.log(pi)[None, :]
    for _ in range(iters):
        lw = lpi + theta @ SETS.T
        w = np.exp(lw - lw.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        m = np.clip(w @ SETS, 1e-6, 1 - 1e-6)
        theta += damp * (target - np.log(m / (1 - m)))
        theta = np.clip(theta, -25, 25)
    lw = lpi + theta @ SETS.T
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    if return_resid:
        return w, np.abs(w @ SETS - q).max(axis=1)
    return w


def decide_full(w: np.ndarray):
    """기대 Jaccard 가 최대인 예측 집합(511가지 전부 탐색). (예측 0/1, 기대 점수)"""
    val = w @ JACC.T                                     # (m, 511 예측)
    best = val.argmax(axis=1)
    return SETS[best].astype(bool), val[np.arange(len(w)), best]


def decide_topk(w: np.ndarray, q: np.ndarray, max_k: int = 4):
    """q 순위 상위 k 개만 후보로 두고 기대 Jaccard 가 최대인 k 를 고른다."""
    order = np.argsort(-q, axis=1, kind="stable")
    best = np.full(len(q), -1.0)
    best_k = np.ones(len(q), dtype=np.int64)
    chosen = np.zeros(q.shape)
    for k in range(1, max_k + 1):
        np.put_along_axis(chosen, order[:, k - 1:k], 1.0, axis=1)
        inter = chosen @ SETS.T
        sc = (w * inter / (k + SET_SIZE[None, :] - inter)).sum(axis=1)
        better = sc > best + 1e-12
        best[better] = sc[better]
        best_k[better] = k
    rank = order.argsort(axis=1, kind="stable")
    return rank < best_k[:, None], best


def run_chunks(q: np.ndarray, wfun, decide="full", chunk: int = 15000, max_k: int = 4):
    pred = np.zeros(q.shape, dtype=bool)
    val = np.zeros(len(q))
    for s in range(0, len(q), chunk):
        w = wfun(q[s:s + chunk])
        if decide == "full":
            pred[s:s + chunk], val[s:s + chunk] = decide_full(w)
        else:
            pred[s:s + chunk], val[s:s + chunk] = decide_topk(w, q[s:s + chunk], max_k)
    return pred, val


# ----------------------------------------------------------------------------- 빠른 경로 (정답 집합 크기 4 이하만, float32)
_S4 = SET_SIZE <= 4
SETS4 = SETS[_S4]
SIZE4 = SET_SIZE[_S4]


class FastMaxent:
    """maxent + 상위 k 결정을 크기 4 이하 집합 255가지로 줄여 계산한다 (5개 이상 정답은 fit 에서 0.06%)."""

    def __init__(self, pi: np.ndarray, marg: np.ndarray, iters: int = 40, damp: float = 0.8, min_sum: float = 1.02):
        p4 = pi[_S4] / pi[_S4].sum()
        self.lpi = np.log(p4)[None, :]
        self.lmarg = np.log(marg / (1 - marg))[None, :]
        self.iters, self.damp, self.min_sum = iters, damp, min_sum

    def weights(self, q: np.ndarray) -> np.ndarray:
        q = np.clip(q.astype(np.float64), 1e-4, 1 - 1e-4)
        s = q.sum(axis=1)
        low = s < self.min_sum
        if low.any():
            ql = q[low]
            lo, hi = np.zeros(len(ql)), np.full(len(ql), 12.0)
            od = np.log(ql / (1 - ql))
            for _ in range(30):
                mid = (lo + hi) / 2
                ss = (1 / (1 + np.exp(-(od + mid[:, None])))).sum(axis=1)
                hi = np.where(ss >= self.min_sum, mid, hi)
                lo = np.where(ss < self.min_sum, mid, lo)
            q = q.copy()
            q[low] = np.clip(1 / (1 + np.exp(-(od + hi[:, None]))), 1e-4, 1 - 1e-4)
        target = np.log(q / (1 - q))
        theta = target - self.lmarg
        for _ in range(self.iters):
            lw = self.lpi + theta @ SETS4.T
            w = np.exp(lw - lw.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            m = np.clip(w @ SETS4, 1e-6, 1 - 1e-6)
            theta += self.damp * (target - np.log(m / (1 - m)))
            np.clip(theta, -25, 25, out=theta)
        lw = self.lpi + theta @ SETS4.T
        w = np.exp(lw - lw.max(axis=1, keepdims=True))
        return w / w.sum(axis=1, keepdims=True)

    def decide(self, q: np.ndarray, max_k: int = 4, chunk: int = 20000):
        pred = np.zeros(q.shape, dtype=bool)
        val = np.zeros(len(q))
        for s in range(0, len(q), chunk):
            qq = q[s:s + chunk]
            w = self.weights(qq)
            order = np.argsort(-qq, axis=1, kind="stable")
            best = np.full(len(qq), -1.0)
            best_k = np.ones(len(qq), dtype=np.int64)
            chosen = np.zeros(qq.shape)
            for k in range(1, max_k + 1):
                np.put_along_axis(chosen, order[:, k - 1:k], 1.0, axis=1)
                inter = chosen @ SETS4.T
                sc = (w * inter / (k + SIZE4[None, :] - inter)).sum(axis=1)
                better = sc > best + 1e-12
                best[better] = sc[better]
                best_k[better] = k
            rank = order.argsort(axis=1, kind="stable")
            pred[s:s + chunk] = rank < best_k[:, None]
            val[s:s + chunk] = best
        return pred, val


_FM = None


def fast_maxent() -> "FastMaxent":
    global _FM
    if _FM is None:
        pi, marg, _ = prior_joint(fit_labels())
        _FM = FastMaxent(pi, marg)
    return _FM


class NewtonMaxent(FastMaxent):
    """FastMaxent 와 같은 분포를 뉴턴법으로 푼다 (반복 8회 안팎, 수렴 정밀도가 더 높다)."""

    _PAIR = (SETS4[:, :, None] * SETS4[:, None, :]).reshape(len(SETS4), 81)

    def __init__(self, pi, marg, iters: int = 9, min_sum: float = 1.02):
        super().__init__(pi, marg, iters=iters, min_sum=min_sum)

    def weights(self, q: np.ndarray) -> np.ndarray:
        q = np.clip(q.astype(np.float64), 1e-4, 1 - 1e-4)
        s = q.sum(axis=1)
        low = s < self.min_sum
        if low.any():
            ql = q[low]
            lo, hi = np.zeros(len(ql)), np.full(len(ql), 12.0)
            od = np.log(ql / (1 - ql))
            for _ in range(30):
                mid = (lo + hi) / 2
                ss = (1 / (1 + np.exp(-(od + mid[:, None])))).sum(axis=1)
                hi = np.where(ss >= self.min_sum, mid, hi)
                lo = np.where(ss < self.min_sum, mid, lo)
            q = q.copy()
            q[low] = np.clip(1 / (1 + np.exp(-(od + hi[:, None]))), 1e-4, 1 - 1e-4)
        theta = np.log(q / (1 - q)) - self.lmarg
        eye = np.eye(9)[None, :, :] * 1e-7
        for _ in range(self.iters):
            lw = self.lpi + theta @ SETS4.T
            w = np.exp(lw - lw.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            m = w @ SETS4
            cov = (w @ self._PAIR).reshape(-1, 9, 9) - m[:, :, None] * m[:, None, :] + eye
            step = np.linalg.solve(cov, (m - q)[:, :, None])[:, :, 0]
            big = np.abs(step).max(axis=1, keepdims=True)
            theta -= step * np.minimum(1.0, 3.0 / np.maximum(big, 1e-12))
            np.clip(theta, -30, 30, out=theta)
        lw = self.lpi + theta @ SETS4.T
        w = np.exp(lw - lw.max(axis=1, keepdims=True))
        return w / w.sum(axis=1, keepdims=True)


_NM = None


def newton_maxent() -> "NewtonMaxent":
    global _NM
    if _NM is None:
        pi, marg, _ = prior_joint(fit_labels())
        _NM = NewtonMaxent(pi, marg)
    return _NM
