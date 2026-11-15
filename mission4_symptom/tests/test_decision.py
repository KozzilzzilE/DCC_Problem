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


def _toy_prior(seed: int = 5):
    rng = np.random.default_rng(seed)
    labels = (rng.random((4000, 9)) < 0.15).astype(float)
    labels[:, 1] = np.maximum(labels[:, 1], labels[:, 6] * (rng.random(4000) < 0.6))   # 6번이 있으면 1번도 자주 붙게
    labels[labels.sum(axis=1) == 0, 3] = 1
    return dec.label_set_prior(labels)


def test_maxent_matches_marginals():
    """맞춘 분포의 증상별 확률이 입력 확률과 같아야 한다.

    증상별 확률의 합이 1.1~3 인 순간만 본다. 합이 1 보다 작으면 맞출 수 없어 따로 처리하고,
    합이 아주 크면(사전분포에 거의 없는 5개 이상 조합이 필요) 수렴하지 않는다. 실제 dev 확률에서
    맞지 않는 순간의 비율은 scripts/eval_combo.py 가 함께 출력한다.
    """
    prior, marg = _toy_prior()
    rng = np.random.default_rng(6)
    probs = rng.random((500, 9)) ** 2
    total = probs.sum(axis=1)
    probs = probs[(total >= 1.1) & (total <= 3.0)]
    w = dec.maxent_weights(probs, prior, marg, iters=300)
    assert abs(w.sum(axis=1) - 1).max() < 1e-9
    assert np.abs(w @ dec._SETS - np.clip(probs, 1e-4, 1 - 1e-4)).max() < 1e-3


def test_maxent_rule_is_top_k_and_never_empty():
    prior, marg = _toy_prior()
    rng = np.random.default_rng(7)
    probs = rng.random((400, 9)) ** 3            # 합이 1 보다 작은 순간도 섞여 있다
    pred, value = dec.maxent_rule(probs, prior, marg)
    assert pred.any(axis=1).all() and (value > 0).all() and (value <= 1).all()
    for row in range(len(probs)):                # 낸 증상은 항상 확률 상위 k개
        k = int(pred[row].sum())
        assert set(np.nonzero(pred[row])[0]) == set(np.argsort(-probs[row], kind="stable")[:k])
    rule = dec.make_rule({"type": "maxent"}, None, {"prior": prior.tolist(), "marg": marg.tolist()})
    assert (rule(probs) == pred).all()


def test_make_rule_from_bundle_settings():
    """번들 설정(json 으로 오간 list)만으로 만든 규칙이 maxent_rule 과 같은 예측을 낸다. 한 순간씩 넣어도 같다."""
    import json

    import pytest

    from m4.bundle import DEFAULT_SETTINGS

    prior, marg = _toy_prior()
    shift = [0.3] * 9
    settings = json.loads(json.dumps(dict(DEFAULT_SETTINGS, rule={"type": "maxent"}, logit_shift=shift,
                                          label_prior={"prior": prior.tolist(), "marg": marg.tolist()})))
    rule = dec.make_rule(settings["rule"], settings["logit_shift"], settings["label_prior"])
    probs = (np.random.default_rng(8).random((200, 9)) ** 2).astype(np.float32)
    want = dec.maxent_rule(dec.calibrate(probs, shift), prior, marg)[0]
    assert (rule(probs.copy()) == want).all()
    assert all((rule(probs[i:i + 1].copy())[0] == want[i]).all() for i in range(len(probs)))   # 추론기는 한 순간씩 부른다
    # 사전분포가 없으면 maxent 는 만들 수 없고, 예전 번들의 규칙은 사전분포 없이도 그대로 된다
    assert DEFAULT_SETTINGS["label_prior"] is None
    with pytest.raises(ValueError):
        dec.make_rule({"type": "maxent"}, shift, DEFAULT_SETTINGS["label_prior"])
    for old in ({"type": "threshold"}, {"type": "expected"}):
        assert dec.make_rule(old, shift, None)(probs.copy()).any(axis=1).all()


def test_label_set_prior_counts_by_hand():
    """작은 라벨 행렬에서 손으로 센 조합 빈도와 같아야 한다 (평활을 끄고 비교)."""
    labels = np.zeros((14, 9))
    labels[:6, 3] = 1                       # {3} 여섯 번
    labels[6:9, [1, 6]] = 1                 # {1, 6} 세 번
    labels[9, [0, 3, 7]] = 1                # {0, 3, 7} 한 번
    for row, c in zip(range(10, 14), (2, 4, 5, 8)):
        labels[row, c] = 1                  # 나머지 증상은 한 번씩 단독으로
    prior, marg = dec.label_set_prior(labels, smooth=0.0)

    def index(*classes):
        row = np.zeros(9)
        row[list(classes)] = 1
        return int(np.nonzero((dec._SETS == row).all(axis=1))[0][0])

    assert abs(prior[index(3)] - 6 / 14) < 1e-12
    assert abs(prior[index(1, 6)] - 3 / 14) < 1e-12
    assert abs(prior[index(0, 3, 7)] - 1 / 14) < 1e-12
    assert abs(prior[index(2)] - 1 / 14) < 1e-12
    assert abs(prior.sum() - 1) < 1e-12 and (prior > 0).sum() == 7
    assert np.allclose(marg, np.array([1, 3, 1, 7, 1, 1, 3, 1, 1]) / 14)


def test_maxent_rule_picks_best_k():
    """고른 k 가, 정답 집합 분포에서 정의대로 계산한 기대 순간 점수를 최대로 하는 k 여야 한다."""
    prior, marg = _toy_prior()
    rng = np.random.default_rng(8)
    probs = rng.random((200, 9)) ** 2
    pred, value = dec.maxent_rule(probs, prior, marg, max_k=4)
    weights = dec.maxent_weights(probs, prior, marg)
    for row in range(len(probs)):
        order = np.argsort(-probs[row], kind="stable")
        scores = []
        for k in range(1, 5):
            chosen = np.zeros(9)
            chosen[order[:k]] = 1
            inter = dec._SETS @ chosen
            scores.append(float((weights[row] * inter / (k + dec._SET_SIZE - inter)).sum()))
        assert abs(value[row] - max(scores)) < 1e-12
        assert int(pred[row].sum()) == int(np.argmax(scores)) + 1


def test_maxent_uses_the_set_prior():
    """같은 확률이라도 사전분포가 '1번과 6번은 함께 나온다'면 둘을 같이 내고, '함께 나오지 않는다'면 하나만 낸다."""
    others = np.zeros((70, 9))
    for i, c in enumerate((0, 2, 3, 4, 5, 7, 8)):
        others[i * 10:(i + 1) * 10, c] = 1          # 나머지 증상은 열 번씩 단독으로
    together = np.zeros((500, 9))
    together[:, [1, 6]] = 1
    apart = np.zeros((500, 9))
    apart[:250, 1] = 1
    apart[250:, 6] = 1
    probs = np.full((1, 9), 0.02)
    probs[0, 1], probs[0, 6] = 0.6, 0.5
    both = dec.maxent_rule(probs, *dec.label_set_prior(np.vstack([together, others])))[0][0]
    one = dec.maxent_rule(probs, *dec.label_set_prior(np.vstack([apart, others])))[0][0]
    assert both[[1, 6]].all() and both.sum() == 2
    assert one.sum() == 1 and one[1]


def test_maxent_extreme_inputs_never_empty():
    prior, marg = _toy_prior()
    for dtype in (np.float32, np.float64):
        for fill in (0.0, 1e-30, 1e-7, 0.5, 0.999999, 1.0):
            pred, value = dec.maxent_rule(np.full((3, 9), fill, dtype=dtype), prior, marg)
            assert pred.any(axis=1).all() and np.isfinite(value).all()
        onehot = np.eye(9, dtype=dtype)
        assert (dec.maxent_rule(onehot, prior, marg)[0] == onehot.astype(bool)).all()


def test_make_rule_rejects_broken_prior():
    prior, marg = _toy_prior()
    broken = prior.copy()
    broken[0] = np.nan
    for bad in ({"prior": broken.tolist(), "marg": marg.tolist()},
                {"prior": prior[:100].tolist(), "marg": marg.tolist()}):
        try:
            dec.make_rule({"type": "maxent"}, None, bad)
        except ValueError:
            continue
        raise AssertionError("깨진 사전분포를 받아들였다")
