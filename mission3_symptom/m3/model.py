"""Mission 3 KoBERT 다중 라벨 분류 모델 구성."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from transformers import AutoModelForSequenceClassification

from .config import NUM_CLASSES, TARGET_SYMPTOMS
from .kobert_tokenizer import KoBertTokenizer


DEFAULT_MODEL_NAME = "skt/kobert-base-v1"
TOKENIZER_SANITY_TEXT = "한국어 모델을 공유합니다."
EXPECTED_SPECIAL_TOKEN_IDS = {
    "unk_token_id": 0,
    "pad_token_id": 1,
    "cls_token_id": 2,
    "sep_token_id": 3,
    "mask_token_id": 4,
}
JAMO_PATTERN = re.compile(r"[\u1100-\u11ff\u3130-\u318f]")


def validate_tokenizer_model_compatibility(tokenizer, model) -> Dict[str, object]:
    """잘못된 tokenizer가 학습까지 진행되지 않도록 입력 계약을 검증."""
    if not isinstance(tokenizer, KoBertTokenizer):
        raise TypeError(f"KoBertTokenizer가 아닙니다: {type(tokenizer).__name__}")

    for attribute, expected in EXPECTED_SPECIAL_TOKEN_IDS.items():
        actual = getattr(tokenizer, attribute)
        if actual != expected:
            raise ValueError(f"{attribute}가 KoBERT vocabulary와 다릅니다: {actual}")

    tokens = tokenizer.tokenize(TOKENIZER_SANITY_TEXT)
    if any(JAMO_PATTERN.search(token) for token in tokens):
        raise ValueError(f"한국어가 자모 단위로 분해되었습니다: {tokens}")

    encoded = tokenizer(TOKENIZER_SANITY_TEXT, add_special_tokens=True)
    input_ids = encoded["input_ids"]
    if input_ids[0] != tokenizer.cls_token_id:
        raise ValueError("첫 token이 [CLS]가 아닙니다.")
    if input_ids[-1] != tokenizer.sep_token_id:
        raise ValueError("마지막 token이 [SEP]가 아닙니다.")

    content_ids = input_ids[1:-1]
    unknown_count = sum(token_id == tokenizer.unk_token_id for token_id in content_ids)
    unknown_ratio = unknown_count / max(len(content_ids), 1)
    if unknown_ratio > 0.1:
        raise ValueError(f"한국어 문장의 [UNK] 비율이 비정상적입니다: {unknown_ratio:.4f}")

    embedding_size = int(model.get_input_embeddings().num_embeddings)
    if tokenizer.vocab_size != embedding_size or len(tokenizer) != embedding_size:
        raise ValueError(
            "tokenizer와 model embedding vocabulary 크기가 다릅니다: "
            f"vocab_size={tokenizer.vocab_size}, len={len(tokenizer)}, "
            f"embedding_size={embedding_size}"
        )

    result = {
        "tokenizer_class": type(tokenizer).__name__,
        "tokens": tokens,
        "unk_ratio": unknown_ratio,
        "first_token_id": input_ids[0],
        "last_token_id": input_ids[-1],
        "vocab_size": tokenizer.vocab_size,
        "embedding_size": embedding_size,
    }
    print(f"Tokenizer sanity check: {result}")
    return result


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

    tokenizer = KoBertTokenizer.from_pretrained(
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
    validate_tokenizer_model_compatibility(tokenizer, model)
    return tokenizer, model


def load_saved_model(model_dir: Union[str, Path]):
    """저장된 모델 번들을 외부 다운로드 없이 다시 로드."""
    source = str(Path(model_dir))
    tokenizer = KoBertTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        local_files_only=True,
    )
    if int(model.config.num_labels) != NUM_CLASSES:
        raise ValueError(f"저장 모델의 출력 클래스 수가 9가 아닙니다: {model.config.num_labels}")
    validate_tokenizer_model_compatibility(tokenizer, model)
    return tokenizer, model


def save_model_bundle(model, tokenizer, output_dir: Union[str, Path]) -> Path:
    """향후 offline inference에 필요한 모델과 tokenizer를 함께 저장."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return path
