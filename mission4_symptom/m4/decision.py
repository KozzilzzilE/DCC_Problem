"""9종 확률을 증상 리스트로 바꾸는 결정 규칙.

세 가지 규칙이 있다.

threshold  임계값 이상인 증상을 낸다. 하나도 없으면 확률이 가장 높은 1개를 낸다.
expected   4-1 순간 점수(맞힌 수 / 정답 또는 예측에 있는 수)의 기대값이 가장 큰 개수만큼 낸다.
           증상끼리 독립이라고 보고 정답 집합 511가지(빈 집합 제외)의 확률을 모두 계산한 뒤,
           확률 상위 k개를 냈을 때의 기대 점수를 k = 1..max_k 에 대해 구해 가장 큰 k 를 고른다.
           임계값을 따로 정하지 않는다.
maxent     expected 와 같은 계산이지만 "증상끼리 독립"이라는 가정을 쓰지 않는다. 학습 라벨에서 센
           증상 조합 빈도(511칸)를 사전분포로 두고, 증상별 확률이 모델의 보정 확률과 같아지도록 맞춘
           분포(사전분포에 가장 가까운 분포)로 정답 집합의 확률을 계산한다. 독립 가정은 정답이
           3~4개일 확률을 실제보다 크게 잡아 증상을 너무 많이 내게 한다 (EXPERIMENT_LOG W001).

expected·maxent 규칙은 확률이 실제 빈도와 맞아야 한다. 학습 때 양성 손실에 pos_weight 를 곱하면 로짓이
log(pos_weight) 만큼 높게 나오므로, `calibrate` 로 그만큼을 빼고 쓴다 (값은 학습 설정에서 온다).
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, Optional, Sequence

import numpy as np

N_CLASSES = 9
# 정답이 될 수 있는 집합 511가지 (평가 대상은 증상이 하나 이상인 통화뿐이라 빈 집합은 없다)
_SETS = np.array(list(itertools.product([0, 1], repeat=N_CLASSES)), dtype=np.float64)[1:]
_SET_SIZE = _SETS.sum(axis=1)


def calibrate(probs: np.ndarray, logit_shift: Optional[Sequence[float]]) -> np.ndarray:
    """로짓에서 logit_shift(클래스별)를 뺀 확률. logit_shift 가 없으면 그대로 돌려준다."""
    if logit_shift is None:
        return probs
    p = np.clip(probs.astype(np.float64), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) - np.asarray(logit_shift, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def expected_rule(probs: np.ndarray, max_k: int = 4, chunk: int = 20000):
    """기대 순간 점수가 가장 큰 상위 k개를 고른다. (예측 0/1 (m, 9), 그때의 기대 점수 (m,)) 를 돌려준다."""
    pred = np.zeros(probs.shape, dtype=bool)
    best_all = np.zeros(len(probs), dtype=np.float64)
    for s in range(0, len(probs), chunk):
        q = np.clip(probs[s:s + chunk].astype(np.float64), 1e-4, 1 - 1e-4)
        # 정답 집합별 확률 (독립 가정), 빈 집합을 뺀 뒤 다시 정규화
        w = np.exp(np.log(q) @ _SETS.T + np.log(1 - q) @ (1 - _SETS).T)
        w /= w.sum(axis=1, keepdims=True)
        order = np.argsort(-q, axis=1, kind="stable")
        best = np.full(len(q), -1.0)
        best_k = np.ones(len(q), dtype=np.int64)
        chosen = np.zeros(q.shape)
        for k in range(1, max_k + 1):
            np.put_along_axis(chosen, order[:, k - 1:k], 1.0, axis=1)      # 상위 k개까지 누적
            inter = chosen @ _SETS.T
            score = (w * inter / (k + _SET_SIZE[None, :] - inter)).sum(axis=1)
            better = score > best + 1e-12
            best[better] = score[better]
            best_k[better] = k
        rank = order.argsort(axis=1, kind="stable")
        pred[s:s + chunk] = rank < best_k[:, None]
        best_all[s:s + chunk] = best
    return pred, best_all


def label_set_prior(labels: np.ndarray, smooth: float = 0.5):
    """학습 라벨 (통화 x 9, 0/1) 에서 증상 조합 사전분포 (511,) 와 증상별 양성률 (9,) 을 만든다.

    한 번도 관측되지 않은 조합이 0 이 되지 않게, 독립 분포를 smooth 건만큼 섞는다.
    """
    labels = np.asarray(labels, dtype=np.float64)
    marg = labels.mean(axis=0)
    weights = 2 ** np.arange(N_CLASSES)
    counts = np.bincount((labels @ weights).astype(int), minlength=2 ** N_CLASSES).astype(np.float64)
    counts = counts[(_SETS @ weights).astype(int)]
    indep = np.exp(_SETS @ np.log(marg) + (1 - _SETS) @ np.log(1 - marg))
    prior = counts + smooth * indep / indep.sum()
    return prior / prior.sum(), marg


def maxent_weights(probs: np.ndarray, prior: np.ndarray, marg: np.ndarray, iters: int = 60,
                   damp: float = 0.7, min_sum: float = 1.02) -> np.ndarray:
    """정답 집합 511가지의 확률 (m, 511). w(S) ∝ prior(S)·exp(θ·S) 꼴에서 증상별 확률이 probs 와 같아지게 θ 를 맞춘다.

    정답은 증상이 하나 이상이므로 증상별 확률의 합이 1 보다 작으면 맞출 수 없다. 그런 순간에는
    모든 증상의 odds 에 같은 배수를 곱해 합을 min_sum 으로 올린 뒤 맞춘다 (증상 간 순위는 그대로).
    """
    q = np.clip(probs.astype(np.float64), 1e-4, 1 - 1e-4)
    low = q.sum(axis=1) < min_sum
    if low.any():
        logit = np.log(q[low] / (1 - q[low]))
        lo, hi = np.zeros(len(logit)), np.full(len(logit), 12.0)
        for _ in range(40):                                   # 공통 이동량을 이분법으로 찾는다
            mid = (lo + hi) / 2
            total = (1 / (1 + np.exp(-(logit + mid[:, None])))).sum(axis=1)
            hi = np.where(total >= min_sum, mid, hi)
            lo = np.where(total < min_sum, mid, lo)
        q = q.copy()
        q[low] = np.clip(1 / (1 + np.exp(-(logit + hi[:, None]))), 1e-4, 1 - 1e-4)
    target = np.log(q / (1 - q))
    theta = target - np.log(marg / (1 - marg))[None, :]
    log_prior = np.log(prior)[None, :]

    def weights(th: np.ndarray) -> np.ndarray:
        lw = log_prior + th @ _SETS.T
        w = np.exp(lw - lw.max(axis=1, keepdims=True))
        return w / w.sum(axis=1, keepdims=True)

    for _ in range(iters):                                    # 증상별 확률의 로그 odds 차이만큼 θ 를 고친다
        m = np.clip(weights(theta) @ _SETS, 1e-6, 1 - 1e-6)
        theta = np.clip(theta + damp * (target - np.log(m / (1 - m))), -25, 25)
    return weights(theta)


def _best_top_k(weights: np.ndarray, probs: np.ndarray, max_k: int):
    """정답 집합 확률 weights (m, 511) 에서, probs 순위 상위 k개를 냈을 때 기대 순간 점수가 가장 큰 k 를 고른다."""
    order = np.argsort(-probs, axis=1, kind="stable")
    best = np.full(len(probs), -1.0)
    best_k = np.ones(len(probs), dtype=np.int64)
    chosen = np.zeros(probs.shape)
    for k in range(1, max_k + 1):
        np.put_along_axis(chosen, order[:, k - 1:k], 1.0, axis=1)
        inter = chosen @ _SETS.T
        score = (weights * inter / (k + _SET_SIZE[None, :] - inter)).sum(axis=1)
        better = score > best + 1e-12
        best[better] = score[better]
        best_k[better] = k
    rank = order.argsort(axis=1, kind="stable")
    return rank < best_k[:, None], best


def maxent_rule(probs: np.ndarray, prior: np.ndarray, marg: np.ndarray, max_k: int = 4, chunk: int = 15000):
    """증상 조합 사전분포를 쓴 기대 순간 점수 최대화. (예측 0/1 (m, 9), 그때의 기대 점수 (m,)) 를 돌려준다."""
    pred = np.zeros(probs.shape, dtype=bool)
    value = np.zeros(len(probs), dtype=np.float64)
    for s in range(0, len(probs), chunk):
        q = probs[s:s + chunk]
        pred[s:s + chunk], value[s:s + chunk] = _best_top_k(maxent_weights(q, prior, marg), q, max_k)
    return pred, value


def threshold_rule(probs: np.ndarray, threshold: float = 0.5, force_top1: bool = True,
                   max_labels: Optional[int] = None) -> np.ndarray:
    pred = probs >= threshold
    if max_labels is not None:
        rank = np.argsort(-probs, axis=1, kind="stable").argsort(axis=1, kind="stable")
        pred &= rank < max_labels
    if force_top1:
        empty = ~pred.any(axis=1)
        pred[empty, probs[empty].argmax(axis=1)] = True
    return pred


def make_rule(rule: Dict[str, object], logit_shift: Optional[Sequence[float]] = None,
              label_prior: Optional[Dict[str, Sequence[float]]] = None) -> Callable[[np.ndarray], np.ndarray]:
    """설정 dict 로 결정 함수(확률 (m, 9) -> 예측 0/1 (m, 9))를 만든다. 추론과 모의 채점이 같이 쓴다.

    label_prior 는 maxent 규칙에만 필요하다: {"prior": 조합 사전분포 511개, "marg": 증상별 양성률 9개}.
    """
    kind = rule.get("type", "threshold")
    if kind == "threshold":
        use_shift = logit_shift if rule.get("calibrated", False) else None
        return lambda p: threshold_rule(calibrate(p, use_shift), float(rule.get("threshold", 0.5)),
                                        bool(rule.get("force_top1", True)), rule.get("max_labels"))
    if kind == "expected":
        return lambda p: expected_rule(calibrate(p, logit_shift), int(rule.get("max_k", 4)))[0]
    if kind == "maxent":
        if not label_prior:
            raise ValueError("maxent 규칙에는 증상 조합 사전분포(label_prior)가 필요합니다")
        prior = np.asarray(label_prior["prior"], dtype=np.float64)
        marg = np.asarray(label_prior["marg"], dtype=np.float64)
        # 사전분포가 깨져 있으면(길이 오류, NaN, 0 이하) 규칙이 조용히 "항상 1개"로 바뀌므로 여기서 막는다
        if prior.shape != (len(_SETS),) or marg.shape != (N_CLASSES,):
            raise ValueError(f"사전분포 크기가 맞지 않습니다: prior {prior.shape}, marg {marg.shape}")
        if not (np.isfinite(prior).all() and (prior > 0).all() and ((marg > 0) & (marg < 1)).all()):
            raise ValueError("사전분포에 유한한 양수가 아닌 값이 있습니다 (학습 라벨에 양성이 없는 증상이 있는지 확인)")
        return lambda p: maxent_rule(calibrate(p, logit_shift), prior, marg, int(rule.get("max_k", 4)))[0]
    raise ValueError(f"알 수 없는 결정 규칙: {kind}")
