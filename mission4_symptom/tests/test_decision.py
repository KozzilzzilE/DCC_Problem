"""기대 점수 최대화 규칙이 전수 탐색과 같은 답을 내는지 확인한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4 import decision as dec  # noqa: E402


def _expected_score(pred: np.ndarray, q: np.ndarray) -> float:
    """예측 집합 하나의 기대 순간 점수를 정의대로 계산한다 (정답 집합 511가지에 대한 합)."""
    w = np.prod(np.where(dec._SETS > 0, q, 1 - q), axis=1)
    w /= w.sum()
    inter = dec._SETS @ pred
    union = pred.sum() + dec._SET_SIZE - inter
    return float((w * inter / union).sum())


def test_expected_rule_matches_brute_force():
    rng = np.random.default_rng(0)
    probs = rng.random((300, 9)) ** 2
    pred, value = dec.expected_rule(probs, max_k=9)
    for row in range(len(probs)):
        q = np.clip(probs[row], 1e-4, 1 - 1e-4)
        best = max(_expected_score(s, q) for s in dec._SETS)       # 가능한 예측 집합 511가지 전부
        assert abs(_expected_score(pred[row].astype(float), q) - best) < 1e-9
        assert abs(value[row] - best) < 1e-9


def test_calibrate_shift():
    p = np.array([[0.5, 0.9]])
    out = dec.calibrate(p, [np.log(3.0), 0.0])
    assert abs(out[0, 0] - 0.25) < 1e-9 and abs(out[0, 1] - 0.9) < 1e-9
    assert dec.calibrate(p, None) is p


def test_rules_never_empty():
    rng = np.random.default_rng(1)
    probs = rng.random((200, 9)) * 0.3
    for rule in ({"type": "threshold"}, {"type": "expected"}):
        assert dec.make_rule(rule, [0.5] * 9)(probs.copy()).any(axis=1).all()
