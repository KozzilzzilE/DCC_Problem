"""조각 확률 -> 통화 라벨 집계와 출력 인코딩 검증."""
import numpy as np
import pytest

from m1.aggregate import (
    FEMALE,
    GENDER_OUTPUT,
    MALE,
    call_label,
    call_probability,
    gender_to_target,
)


def test_output_encoding_is_korean():
    assert GENDER_OUTPUT == {0: "남", 1: "여"}


def test_target_mapping():
    assert gender_to_target("F") == FEMALE == 1
    assert gender_to_target("M") == MALE == 0
    assert gender_to_target(None) is None


def test_call_probability_is_mean_of_segments():
    assert call_probability(np.array([0.2, 0.4, 0.9])) == pytest.approx(0.5)


def test_call_probability_of_empty_is_none():
    assert call_probability(np.array([])) is None


def test_call_label_thresholds_at_half():
    assert call_label(0.9) == "여"
    assert call_label(0.1) == "남"
    assert call_label(0.5) == "여"
    assert call_label(0.499999) == "남"


def test_call_label_falls_back_to_majority_class_when_no_segments():
    """조각이 0개인 통화도 반드시 한 행을 내야 한다 (CSV 행수 보존)."""
    assert call_label(None) == "여"


def test_single_segment_call():
    prob = call_probability(np.array([0.77]))
    assert prob == pytest.approx(0.77)
    assert call_label(prob) == "여"


def test_soft_voting_beats_hard_majority():
    """확신 없는 다수(0.51 x2)보다 확신하는 소수(0.02)가 이길 수 있어야 한다."""
    probs = np.array([0.51, 0.51, 0.02])
    assert call_label(call_probability(probs)) == "남"
    hard = (probs >= 0.5).mean()
    assert hard > 0.5  # 하드 보팅이면 반대로 '여'


def test_probabilities_outside_unit_interval_rejected():
    with pytest.raises(ValueError):
        call_probability(np.array([0.5, 1.7]))


def test_nan_probability_rejected():
    """NaN 이 범위 검사를 통과해 조용히 '남' 이 되면 안 된다."""
    with pytest.raises(ValueError, match="NaN"):
        call_probability(np.array([0.5, np.nan]))
