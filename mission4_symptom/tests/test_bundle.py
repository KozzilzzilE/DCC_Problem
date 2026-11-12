"""번들을 묶기 전에 멤버들의 학습 설정을 대조하는 로직을 확인한다 (모델을 읽지 않는다)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4 import SYMPTOM_CLASSES  # noqa: E402
from m4.bundle import learned_halt_setting, member_pos_weight, shared_pos_weight, shared_prior_source  # noqa: E402

WEIGHTS = [2.1, 2.3, 3.0, 1.8, 1.9, 2.7, 2.8, 2.1, 2.5]


def _final(weights=WEIGHTS, calls="runs/cache/train.jsonl", ids="runs/cache/fit_ids.txt") -> dict:
    """본선 학습 스크립트가 남기는 형식."""
    return {"pos_weight": list(weights), "calls": calls, "ids": ids}


def _prelim(weights=WEIGHTS) -> dict:
    """예선 학습 스크립트가 남기는 형식 (증상 이름별 통계, calls·ids 없음). 키 순서를 일부러 뒤집는다."""
    stats = {c: {"pos_weight": w} for c, w in zip(SYMPTOM_CLASSES, weights)}
    return {"train_pos_weight_statistics": dict(reversed(list(stats.items())))}


def test_member_pos_weight_reads_both_formats():
    assert member_pos_weight(_final()) == WEIGHTS
    assert member_pos_weight(_prelim()) == WEIGHTS          # 키 순서가 아니라 증상 이름으로 맞춘다
    assert member_pos_weight({}) is None
    assert member_pos_weight({"train_pos_weight_statistics": {"고열": {"pos_weight": 2.0}}}) is None


def test_member_pos_weight_rejects_broken_values():
    with pytest.raises(ValueError):
        member_pos_weight({"pos_weight": WEIGHTS[:8]})
    with pytest.raises(ValueError):
        member_pos_weight({"pos_weight": [0.0] + WEIGHTS[1:]})


def test_shared_pos_weight_mixed_formats():
    # float32 로 저장된 값과 float64 값의 차이(1e-7 수준)는 같은 값으로 본다
    near = [w + 1e-7 for w in WEIGHTS]
    weights, missing = shared_pos_weight([_prelim(), _final(near), {}], ["a", "b", "c"])
    assert weights == WEIGHTS and missing == ["c"]


def test_shared_pos_weight_missing_first_member():
    weights, missing = shared_pos_weight([{}, _final()], ["a", "b"])
    assert weights == WEIGHTS and missing == ["a"]
    assert shared_pos_weight([{}, {}], ["a", "b"]) == (None, ["a", "b"])


def test_shared_pos_weight_stops_on_mismatch():
    other = list(WEIGHTS)
    other[4] += 0.01
    with pytest.raises(ValueError, match="pos_weight"):
        shared_pos_weight([_final(), _prelim(other)], ["a", "b"])


def test_shared_prior_source():
    src = ("runs/cache/train.jsonl", "runs/cache/fit_ids.txt")
    assert shared_prior_source([_prelim(), _final(), _final()], ["a", "b", "c"]) == src
    assert shared_prior_source([_prelim(), {}], ["a", "b"]) == (None, None)
    assert shared_prior_source([_final(ids=None)], ["a"]) == ("runs/cache/train.jsonl", None)
    with pytest.raises(ValueError, match="학습 통화"):
        shared_prior_source([_final(), _final(ids="other.txt")], ["a", "b"])


def _policy_dict(shift) -> dict:
    return {"type": "learned", "beta": 16.0, "tau": 0.64, "min_step": 5, "logit_shift": shift,
            "feature_names": ["a", "b"], "model_str": "tree"}


def test_learned_halt_setting_checks_logit_shift():
    """학습형 중단 정책을 번들에 넣기 전에 보정값을 대조한다. 다르면 멈추고, 저장 정밀도 차이(1e-7 수준)는 넘긴다."""
    shift = [0.76, 0.85, 1.10, 0.60, 0.64, 1.01, 1.02, 0.75, 0.93]
    near = [x + 4e-8 for x in shift]
    got = learned_halt_setting(_policy_dict(shift), near)
    assert got["type"] == "learned" and got["logit_shift"] == shift and got["beta"] == 16.0      # 정책의 값을 그대로 둔다
    other = list(shift)
    other[3] += 0.01
    with pytest.raises(ValueError, match="logit_shift"):
        learned_halt_setting(_policy_dict(shift), other)
    with pytest.raises(ValueError, match="logit_shift"):
        learned_halt_setting(_policy_dict(shift), None)
    with pytest.raises(ValueError, match="logit_shift"):
        learned_halt_setting(_policy_dict(None), shift)
    with pytest.raises(ValueError, match="logit_shift"):
        learned_halt_setting(_policy_dict(shift), shift[:8])


def test_learned_halt_setting_rejects_other_policies():
    with pytest.raises(ValueError, match="학습형"):
        learned_halt_setting({"type": "expected", "tau": 0.5, "min_step": 5}, None)
    broken = _policy_dict(None)
    del broken["model_str"]
    with pytest.raises(ValueError, match="model_str"):
        learned_halt_setting(broken, None)
