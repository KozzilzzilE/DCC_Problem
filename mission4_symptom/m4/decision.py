"""9종 확률을 증상 리스트로 바꾸는 결정 규칙.

두 가지 규칙이 있다.

threshold  임계값 이상인 증상을 낸다. 하나도 없으면 확률이 가장 높은 1개를 낸다.
expected   4-1 순간 점수(맞힌 수 / 정답 또는 예측에 있는 수)의 기대값이 가장 큰 개수만큼 낸다.
           증상끼리 독립이라고 보고 정답 집합 511가지(빈 집합 제외)의 확률을 모두 계산한 뒤,
           확률 상위 k개를 냈을 때의 기대 점수를 k = 1..max_k 에 대해 구해 가장 큰 k 를 고른다.
           임계값을 따로 정하지 않는다.

expected 규칙은 확률이 실제 빈도와 맞아야 한다. 학습 때 양성 손실에 pos_weight 를 곱하면 로짓이
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


def make_rule(rule: Dict[str, object], logit_shift: Optional[Sequence[float]] = None
              ) -> Callable[[np.ndarray], np.ndarray]:
    """설정 dict 로 결정 함수(확률 (m, 9) -> 예측 0/1 (m, 9))를 만든다. 추론과 모의 채점이 같이 쓴다."""
    kind = rule.get("type", "threshold")
    if kind == "threshold":
        use_shift = logit_shift if rule.get("calibrated", False) else None
        return lambda p: threshold_rule(calibrate(p, use_shift), float(rule.get("threshold", 0.5)),
                                        bool(rule.get("force_top1", True)), rule.get("max_labels"))
    if kind == "expected":
        return lambda p: expected_rule(calibrate(p, logit_shift), int(rule.get("max_k", 4)))[0]
    raise ValueError(f"알 수 없는 결정 규칙: {kind}")
