"""Mission 3 KoBERT 다중 라벨 분류 모델 구성."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import NUM_CLASSES, TARGET_SYMPTOMS


DEFAULT_MODEL_NAME = "skt/kobert-base-v1"


def build_tokenizer_and_model(
    model_name_or_path: Union[str, Path] = DEFAULT_MODEL_NAME,
    local_files_only: bool = False,
    revision: Optional[str] = None,
) -> Tuple[object, AutoModelForSequenceClassification]:
    """사전학습 KoBERT와 새 9-label classification head를 생성."""
    source = str(model_name_or_path)
    label2id = {symptom: index for index, symptom in enumerate(TARGET_SYMPTOMS)}
    id2label = {index: symptom for index, symptom in enumerate(TARGET_SYMPTOMS)}
    load_options: Dict[str, object] = {"local_files_only": local_files_only}
    if revision is not None:
        load_options["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(
        source,
        **load_options,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=NUM_CLASSES,
        label2id=label2id,
        id2label=id2label,
        problem_type="multi_label_classification",
        **load_options,
    )
    return tokenizer, model


def load_saved_model(model_dir: Union[str, Path]):
    """저장된 모델 번들을 외부 다운로드 없이 다시 로드."""
    source = str(Path(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        local_files_only=True,
    )
    if int(model.config.num_labels) != NUM_CLASSES:
        raise ValueError(f"저장 모델의 출력 클래스 수가 9가 아닙니다: {model.config.num_labels}")
    return tokenizer, model


def save_model_bundle(model, tokenizer, output_dir: Union[str, Path]) -> Path:
    """향후 offline inference에 필요한 모델과 tokenizer를 함께 저장."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return path
