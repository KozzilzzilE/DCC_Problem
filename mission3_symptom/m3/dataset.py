"""Mission 3 CSV 데이터셋 및 DataLoader 구성."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import NUM_CLASSES, TARGET_SYMPTOMS
from .truncation import content_budget, head_tail_concat

ENCODE_MODES = ("truncate", "head_tail")


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


def _as_id_list(value) -> List[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _wrap_special_tokens(tokenizer, content_ids: Sequence[int]) -> List[int]:
    content = [int(token_id) for token_id in content_ids]
    try:
        wrapped = tokenizer.build_inputs_with_special_tokens(content)
        return _as_id_list(wrapped)
    except (AttributeError, TypeError, NotImplementedError):
        pass

    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if cls_id is None:
        cls_id = getattr(tokenizer, "bos_token_id", None)
    if sep_id is None:
        sep_id = getattr(tokenizer, "eos_token_id", None)

    output = list(content)
    if cls_id is not None:
        output = [int(cls_id), *output]
    if sep_id is not None:
        output = [*output, int(sep_id)]
    return output


def encode_text(
    tokenizer,
    text: str,
    max_length: int,
    encode_mode: str = "truncate",
) -> Dict[str, List[int]]:
    """대회 허용 본문만 사용해 BERT 입력을 만든다. 라벨별 분기는 하지 않는다."""
    if encode_mode not in ENCODE_MODES:
        raise ValueError(f"지원하지 않는 encode_mode입니다: {encode_mode}")

    if encode_mode == "truncate":
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {key: value for key, value in encoded.items()}

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    content_ids = _as_id_list(encoded["input_ids"])
    budget = content_budget(max_length)
    if len(content_ids) <= budget:
        return encode_text(tokenizer, text, max_length, encode_mode="truncate")

    keep = head_tail_concat(len(content_ids), budget)
    input_ids = _wrap_special_tokens(tokenizer, [content_ids[index] for index in keep])
    features: Dict[str, List[int]] = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
    }
    model_inputs = getattr(tokenizer, "model_input_names", [])
    if "token_type_ids" in model_inputs:
        features["token_type_ids"] = [0] * len(input_ids)
    return features


class SymptomDataset(Dataset):
    """모델 입력에는 CSV의 text만 포함하는 다중 라벨 데이터셋."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int,
        encode_mode: str = "truncate",
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length는 양수여야 합니다.")
        if encode_mode not in ENCODE_MODES:
            raise ValueError(f"지원하지 않는 encode_mode입니다: {encode_mode}")
        self.texts: List[str] = dataframe["text"].tolist()
        self.labels = labels_from_dataframe(dataframe)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encode_mode = encode_mode

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = encode_text(
            self.tokenizer,
            self.texts[index],
            self.max_length,
            encode_mode=self.encode_mode,
        )
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
    encode_mode: str = "truncate",
) -> DataLoader:
    """재현 가능한 순서로 학습 또는 검증 DataLoader를 생성."""
    if batch_size <= 0:
        raise ValueError("batch_size는 양수여야 합니다.")
    if num_workers < 0:
        raise ValueError("num_workers는 0 이상이어야 합니다.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = SymptomDataset(
        dataframe,
        tokenizer,
        max_length=max_length,
        encode_mode=encode_mode,
    )
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
