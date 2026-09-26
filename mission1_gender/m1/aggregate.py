"""조각 확률 -> 통화 단위 라벨 집계.

Mission 1 의 정답은 통화 단위인데 모델은 신고자 발화 조각 단위로 학습한다.
통화당 신고자 조각이 평균 15.7 개이므로, 조각별 확률을 평균(soft voting)하면
개별 조각의 오류가 상쇄되어 통화 정확도가 조각 정확도보다 높아진다.
"""
from __future__ import annotations

import numpy as np

MALE = 0
FEMALE = 1

# 출력 CSV 표기. 출제 PDF 본문과 inference.py 템플릿 기준으로 한글을 쓴다.
GENDER_OUTPUT = {MALE: "남", FEMALE: "여"}

# 라벨 원본(M/F) -> 학습 타깃
_TARGET = {"M": MALE, "F": FEMALE}

# 신고자 조각이 하나도 없어 예측이 불가능할 때의 폴백. 학습/검증 양쪽 모두
# 여성이 다수 클래스이므로 다수 클래스를 택한다.
FALLBACK_CLASS = FEMALE


def gender_to_target(gender: str | None) -> int | None:
    """라벨 원본 'M'/'F' 를 학습 타깃 0/1 로 변환한다."""
    if gender is None:
        return None
    return _TARGET.get(gender.upper())


def call_probability(segment_probs: np.ndarray) -> float | None:
    """한 통화의 조각별 P(여) 를 평균한다. 조각이 없으면 None."""
    probs = np.asarray(segment_probs, dtype=np.float64).ravel()
    if probs.size == 0:
        return None
    if np.isnan(probs).any():
        # NaN 은 min/max 비교를 통과해 조용히 '남' 으로 떨어진다. 발생 시 즉시 드러내는 편이 낫다.
        raise ValueError("segment probabilities contain NaN")
    if probs.min() < 0.0 or probs.max() > 1.0:
        raise ValueError(f"probabilities must lie in [0, 1], got [{probs.min()}, {probs.max()}]")
    return float(probs.mean())


def call_label(probability: float | None, threshold: float = 0.5) -> str:
    """통화 확률을 '남'/'여' 문자열로 변환한다.

    확률이 None(조각 없음)이어도 반드시 라벨을 돌려준다. 평가 CSV 는 입력 통화
    수와 행 수가 같아야 하므로 어떤 통화도 빠뜨릴 수 없다.
    """
    if probability is None:
        return GENDER_OUTPUT[FALLBACK_CLASS]
    return GENDER_OUTPUT[FEMALE if probability >= threshold else MALE]
