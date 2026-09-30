"""Mission 3 클래스별 임계값(Threshold) 최적화 모듈

불균형 다중 라벨 데이터셋에서 9개 증상 각각의 F1을 최대화하는
최적 임계값을 그리드 탐색(Grid Search)을 통해 도출.
"""

from typing import Dict, List, Tuple, Union
import numpy as np

from .config import NUM_CLASSES, TARGET_SYMPTOMS
from .metrics import calculate_binary_f1, eval_macro_f1


def apply_thresholds(
    y_probs: Union[np.ndarray, List],
    thresholds: Union[List[float], np.ndarray, float],
) -> np.ndarray:
    """확률 행렬에 클래스별 임계값을 적용하여 이진 예측 행렬 (N, 9)을 반환.

    Args:
        y_probs: 모델 출력 Sigmoid 확률 행렬 (N, 9)
        thresholds: 단일 float 또는 9개 클래스별 float 임계값 배열

    Returns:
        np.ndarray: 0 또는 1로 구성된 이진 예측 행렬 (N, 9)
    """
    y_probs = np.asarray(y_probs, dtype=float)

    if isinstance(thresholds, (int, float)):
        return (y_probs >= thresholds).astype(int)

    thresholds = np.asarray(thresholds, dtype=float)
    assert len(thresholds) == NUM_CLASSES, f"임계값 개수가 9개가 아닙니다: {len(thresholds)}"

    return (y_probs >= thresholds).astype(int)


def get_threshold_curves(
    y_probs: Union[np.ndarray, List],
    y_true: Union[np.ndarray, List],
    step: float = 0.01,
    min_th: float = 0.05,
    max_th: float = 0.95,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """노트북 시각화를 위해 각 클래스별 임계값 후보와 F1-score 곡선 데이터를 생성.

    Returns:
        Dict[str, Tuple[thresholds_array, f1_scores_array]]: 증상별 (임계값 목록, F1 목록)
    """
    y_probs = np.asarray(y_probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    thresholds = np.arange(min_th, max_th + step / 2, step)

    curves = {}
    for idx, sym in enumerate(TARGET_SYMPTOMS):
        col_probs = y_probs[:, idx]
        col_true = y_true[:, idx]
        f1_list = []
        for th in thresholds:
            pred = (col_probs >= th).astype(int)
            f1_list.append(calculate_binary_f1(col_true, pred))
        curves[sym] = (thresholds, np.array(f1_list))

    return curves


def find_best_thresholds(
    y_probs: Union[np.ndarray, List],
    y_true: Union[np.ndarray, List],
    step: float = 0.01,
    min_th: float = 0.05,
    max_th: float = 0.95,
) -> Tuple[np.ndarray, float, float, Dict[str, float], Dict[str, float]]:
    """검증셋에 대해 9개 증상 각각의 최적 임계값을 탐색.

    Returns:
        best_thresholds: 9개 클래스별 최적 임계값 (array)
        baseline_macro_f1: 기본 0.5 임계값 적용 시 Macro F1
        best_macro_f1: 최적 임계값 적용 시 Macro F1
        baseline_class_f1: 기본 0.5 임계값 적용 시 클래스별 F1
        best_class_f1: 최적 임계값 적용 시 클래스별 F1
    """
    y_probs = np.asarray(y_probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    threshold_candidates = np.arange(min_th, max_th + step / 2, step)

    # 1. 기본 임계값 0.5 기준 계산
    base_preds = apply_thresholds(y_probs, 0.5)
    base_macro, base_class_f1 = eval_macro_f1(y_true, base_preds, return_per_class=True)

    # 2. 클래스별 최적 임계값 그리드 탐색
    best_thresholds = np.full(NUM_CLASSES, 0.5, dtype=float)
    best_class_f1 = {}

    for idx, sym in enumerate(TARGET_SYMPTOMS):
        col_probs = y_probs[:, idx]
        col_true = y_true[:, idx]

        best_th = 0.5
        best_f = -1.0

        for th in threshold_candidates:
            pred = (col_probs >= th).astype(int)
            f1 = calculate_binary_f1(col_true, pred)
            if f1 > best_f:
                best_f = f1
                best_th = th

        best_thresholds[idx] = round(float(best_th), 4)
        best_class_f1[sym] = round(float(best_f), 4)

    # 3. 최적화 후 최종 Macro F1
    opt_preds = apply_thresholds(y_probs, best_thresholds)
    best_macro = eval_macro_f1(y_true, opt_preds)

    return best_thresholds, float(base_macro), float(best_macro), base_class_f1, best_class_f1
