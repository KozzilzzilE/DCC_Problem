"""Mission 3 CSV 데이터셋 및 DataLoader 구성."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import NUM_CLASSES, TARGET_SYMPTOMS


REQUIRED_COLUMNS = ["call_id", "text", *TARGET_SYMPTOMS]


def load_symptom_csv(
    csv_path: Union[str, Path],
    max_samples: Optional[int] = None,
    sample_seed: int = 42,
) -> pd.DataFrame:
    """CSV를 읽고 모델에 필요한 text와 9개 이진 라벨을 검증."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")

    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(f"CSV에 샘플이 없습니다: {path}")
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"필수 CSV 컬럼이 없습니다: {missing}")

    if dataframe["text"].isna().any():
        count = int(dataframe["text"].isna().sum())
        raise ValueError(f"text가 비어 있는 샘플이 있습니다: {count}건")
    if not dataframe["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("text 컬럼에는 문자열만 허용됩니다.")

    label_frame = dataframe[TARGET_SYMPTOMS]
    if label_frame.isna().any().any():
        raise ValueError("9개 타겟 라벨에 결측값이 있습니다.")
    if not label_frame.isin([0, 1]).all().all():
        raise ValueError("9개 타겟 라벨에는 0 또는 1만 허용됩니다.")

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples는 양수여야 합니다.")
        if max_samples < len(dataframe):
            dataframe = dataframe.sample(n=max_samples, random_state=sample_seed)

    return dataframe.reset_index(drop=True)


def labels_from_dataframe(dataframe: pd.DataFrame) -> np.ndarray:
    """config.py의 고정된 클래스 순서로 float32 라벨 행렬을 생성."""
    labels = dataframe[TARGET_SYMPTOMS].to_numpy(dtype=np.float32, copy=True)
    if labels.ndim != 2 or labels.shape[1] != NUM_CLASSES:
        raise ValueError(f"라벨 형상이 올바르지 않습니다: {labels.shape}")
    return labels


class SymptomDataset(Dataset):
    """모델 입력에는 CSV의 text만 포함하는 다중 라벨 데이터셋."""

    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError("max_length는 양수여야 합니다.")
        self.texts: List[str] = dataframe["text"].tolist()
        self.labels = labels_from_dataframe(dataframe)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, object]:
        encoded = self.tokenizer(
            self.texts[index],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item = {key: value for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


class MultiLabelCollator:
    """동적 padding을 적용하면서 다중 라벨 dtype과 형상을 보존."""

    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        labels = torch.stack([feature["labels"] for feature in features])
        model_features = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch


def create_dataloader(
    dataframe: pd.DataFrame,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    pad_to_multiple_of: Optional[int] = None,
) -> DataLoader:
    """재현 가능한 순서로 학습 또는 검증 DataLoader를 생성."""
    if batch_size <= 0:
        raise ValueError("batch_size는 양수여야 합니다.")
    if num_workers < 0:
        raise ValueError("num_workers는 0 이상이어야 합니다.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = SymptomDataset(dataframe, tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=MultiLabelCollator(tokenizer, pad_to_multiple_of=pad_to_multiple_of),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def calculate_token_length_stats(
    texts: Sequence[str],
    tokenizer,
    max_length: int,
    batch_size: int = 512,
) -> Dict[str, Union[int, float]]:
    """truncation 영향을 기록하기 위해 원문 토큰 길이 분포를 계산."""
    lengths: List[int] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        lengths.extend(int(length) for length in encoded["length"])

    if not lengths:
        raise ValueError("토큰 길이를 계산할 text가 없습니다.")

    values = np.asarray(lengths, dtype=np.int64)
    over_limit = int(np.sum(values > max_length))
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "over_max_length": over_limit,
        "over_max_length_ratio": float(over_limit / values.size),
    }
