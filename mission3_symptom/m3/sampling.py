"""Training-only group-aware sampling for the Mission 3 multi-label task."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from .config import TARGET_SYMPTOMS


NAUSEA_COLUMN = "오심"
VOMIT_COLUMN = "구토"
GROUP_NAMES = ("A", "B", "C", "D")
GROUP_DEFINITIONS = {
    "A": {NAUSEA_COLUMN: 0, VOMIT_COLUMN: 0},
    "B": {NAUSEA_COLUMN: 0, VOMIT_COLUMN: 1},
    "C": {NAUSEA_COLUMN: 1, VOMIT_COLUMN: 0},
    "D": {NAUSEA_COLUMN: 1, VOMIT_COLUMN: 1},
}


def classify_nausea_vomit_groups(dataframe: pd.DataFrame) -> np.ndarray:
    """Classify each Training row into the A/B/C/D nausea-vomit group."""
    missing = [
        column
        for column in (NAUSEA_COLUMN, VOMIT_COLUMN)
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"그룹 분류에 필요한 라벨 컬럼이 없습니다: {missing}")

    pair = dataframe[[NAUSEA_COLUMN, VOMIT_COLUMN]]
    if pair.isna().any().any() or not pair.isin([0, 1]).all().all():
        raise ValueError("오심/구토 라벨에는 0 또는 1만 허용됩니다.")

    nausea = pair[NAUSEA_COLUMN].to_numpy(dtype=np.int64, copy=False)
    vomit = pair[VOMIT_COLUMN].to_numpy(dtype=np.int64, copy=False)
    group_index = nausea * 2 + vomit
    return np.asarray(GROUP_NAMES, dtype="<U1")[group_index]


def calculate_nausea_group_counts(dataframe: pd.DataFrame) -> Dict[str, int]:
    groups = classify_nausea_vomit_groups(dataframe)
    return {name: int(np.sum(groups == name)) for name in GROUP_NAMES}


def _sampling_summary(
    dataframe: pd.DataFrame,
    groups: np.ndarray,
    sample_weights: np.ndarray,
    group_weights: Dict[str, float],
    num_samples: int,
    replacement: bool,
) -> Dict[str, object]:
    total_weight = float(sample_weights.sum())
    if total_weight <= 0:
        raise ValueError("sampling weight 합은 양수여야 합니다.")

    group_counts = {name: int(np.sum(groups == name)) for name in GROUP_NAMES}
    group_statistics: Dict[str, Dict[str, object]] = {}
    for name in GROUP_NAMES:
        probability = float(group_counts[name] * group_weights[name] / total_weight)
        original_ratio = float(group_counts[name] / max(len(dataframe), 1))
        group_statistics[name] = {
            "definition": GROUP_DEFINITIONS[name],
            "original_count": group_counts[name],
            "original_ratio": original_ratio,
            "sampling_weight": group_weights[name],
            "expected_sampling_probability": probability,
            "expected_count": float(probability * num_samples),
            "relative_exposure": (
                float(probability / original_ratio) if original_ratio > 0 else None
            ),
        }

    label_statistics: Dict[str, Dict[str, object]] = {}
    for symptom in TARGET_SYMPTOMS:
        labels = dataframe[symptom].to_numpy(dtype=np.float64, copy=False)
        original_count = int(labels.sum())
        expected_probability = float(np.dot(sample_weights, labels) / total_weight)
        expected_count = float(expected_probability * num_samples)
        label_statistics[symptom] = {
            "original_positive_count": original_count,
            "original_positive_ratio": float(original_count / max(len(dataframe), 1)),
            "expected_positive_count": expected_count,
            "expected_positive_ratio": expected_probability,
            "relative_exposure": (
                float(expected_count / original_count) if original_count > 0 else None
            ),
        }

    return {
        "enabled": True,
        "source": "training_rows_only",
        "num_training_rows": int(len(dataframe)),
        "num_samples": int(num_samples),
        "replacement": bool(replacement),
        "group_statistics": group_statistics,
        "label_exposure": label_statistics,
    }


def build_pure_nausea_sampler(
    dataframe: pd.DataFrame,
    pure_nausea_weight: float,
    seed: int,
) -> Tuple[WeightedRandomSampler, Dict[str, object]]:
    """Build a reproducible sampler that only upweights pure-nausea rows."""
    if not np.isfinite(pure_nausea_weight) or pure_nausea_weight < 1.0:
        raise ValueError("pure_nausea_weight는 1.0 이상의 유한한 값이어야 합니다.")
    if dataframe.empty:
        raise ValueError("sampler를 구성할 Training row가 없습니다.")

    groups = classify_nausea_vomit_groups(dataframe)
    group_weights = {"A": 1.0, "B": 1.0, "C": float(pure_nausea_weight), "D": 1.0}
    sample_weights = np.asarray(
        [group_weights[group] for group in groups],
        dtype=np.float64,
    )
    num_samples = len(dataframe)
    replacement = True
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=replacement,
        generator=generator,
    )
    summary = _sampling_summary(
        dataframe,
        groups,
        sample_weights,
        group_weights,
        num_samples,
        replacement,
    )
    return sampler, summary
