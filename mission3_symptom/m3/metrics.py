"""Mission 3 평가 지표 모듈

규정 준수 Macro F1-score 및 클래스별 F1 계산을 담당
대회 공식 산식:
    Macro F1 = (1 / 9) * sum(F1_c for c in 9 classes)
"""

from typing import Dict, List, Tuple, Union
import numpy as np

from .config import NUM_CLASSES, TARGET_SYMPTOMS


def calculate_binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """단일 증상 클래스에 대한 이진 F1-score 계산 (zero_division=0 적용)"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))


def eval_macro_f1(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    return_per_class: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """9개 증상에 대한 Macro F1-score 및 클래스별 세부 F1 계산

    Args:
        y_true: 실제 정답 이진 행렬 (N, 9)
        y_pred: 예측 이진 행렬 (N, 9) (0 또는 1)
        return_per_class: True일 경우 클래스별 F1 딕셔너리도 함께 반환

    Returns:
        macro_f1 (float) 또는 (macro_f1, per_class_f1_dict)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    assert y_true.shape == y_pred.shape, f"형상 불일치: y_true {y_true.shape} vs y_pred {y_pred.shape}"
    assert y_true.shape[1] == NUM_CLASSES, f"열 개수가 9가 아닙니다: {y_true.shape[1]}"

    class_f1s: Dict[str, float] = {}
    for idx, sym in enumerate(TARGET_SYMPTOMS):
        class_f1s[sym] = calculate_binary_f1(y_true[:, idx], y_pred[:, idx])

    macro_f1 = float(np.mean(list(class_f1s.values())))

    if return_per_class:
        return macro_f1, class_f1s
    return macro_f1
