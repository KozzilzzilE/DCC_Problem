"""Mission 3 다중 라벨 분류 모델 및 토크나이저 구성 (KLUE-RoBERTa-base).

Hugging Face 표준 `AutoTokenizer` / `AutoModelForSequenceClassification` 으로 백본과 9개 증상 분류 헤드를
만든다. 로드 직후 토크나이저가 한국어를 자모 단위로 깨뜨리지 않는지, 시작·종료 특수 토큰과 [UNK] 비율,
모델 임베딩 크기가 맞는지 확인해, 예외 없이 점수만 떨어지는 조용한 실패를 막는다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import NUM_CLASSES, TARGET_SYMPTOMS

# 토크나이저 한글 정상 처리 검증을 위한 테스트 문장
TOKENIZER_SANITY_TEXT = "한국어 모델을 공유합니다."

# 한글 초성/중성/종성(자모) 유니코드 범위 정규식: 토크나이저가 한글을 자모 단위로 깨뜨리는지 감지
JAMO_PATTERN = re.compile(r"[\u1100-\u11ff\u3130-\u318f]")


def _console_safe_text(value: object) -> str:
    """현재 stdout 인코딩에서 표현할 수 없는 문자를 escape하여 반환합니다."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(encoding, errors="backslashreplace").decode(encoding)


def validate_tokenizer_model_compatibility(tokenizer, model) -> Dict[str, object]:
    """토크나이저와 모델의 호환성을 검증합니다.

    KLUE-RoBERTa 토크나이저는 CLS/SEP 를 쓰고, 토큰 수(32,000)가 모델 임베딩 크기와 같아야 합니다.
    어긋나면 예외를 내서, 점수만 조용히 떨어지는 상태로 학습·추론이 진행되지 않게 합니다.
    """
    # 1. 한글 자모 분해 깨짐 현상 체크
    tokens = tokenizer.tokenize(TOKENIZER_SANITY_TEXT)
    if any(JAMO_PATTERN.search(token) for token in tokens):
        raise ValueError(f"한국어가 자모 단위로 분해되었습니다: {tokens}")

    # 2. 인코딩 후 시작/종료 스페셜 토큰 확인 (CLS 또는 BOS / SEP 또는 EOS)
    encoded = tokenizer(TOKENIZER_SANITY_TEXT, add_special_tokens=True)
    input_ids = encoded["input_ids"]

    first_expected = getattr(tokenizer, "cls_token_id", None) or getattr(tokenizer, "bos_token_id", None)
    if first_expected is not None and input_ids[0] != first_expected:
        raise ValueError(f"첫 token이 시작 스페셜 토큰이 아닙니다. 실제: {input_ids[0]}, 예상: {first_expected}")

    last_expected = getattr(tokenizer, "sep_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    if last_expected is not None and input_ids[-1] != last_expected:
        raise ValueError(f"마지막 token이 종료 스페셜 토큰이 아닙니다. 실제: {input_ids[-1]}, 예상: {last_expected}")

    # 3. [UNK] 비율 확인
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None:
        content_ids = input_ids[1:-1]
        unknown_count = sum(token_id == unk_id for token_id in content_ids)
        unknown_ratio = unknown_count / max(len(content_ids), 1)
        if unknown_ratio > 0.1:
            raise ValueError(f"한국어 문장의 [UNK] 비율이 비정상적입니다: {unknown_ratio:.4f}")
    else:
        unknown_ratio = 0.0

    # 4. 토크나이저가 만드는 토큰 id 가 모두 모델 임베딩 안에 있어야 한다.
    # 임베딩을 늘리면 새 행이 무작위로 채워져 학습·저장한 모델과 달라지므로, 늘리지 않고 멈춘다.
    embedding_size = int(model.get_input_embeddings().num_embeddings)
    if len(tokenizer) > embedding_size:
        raise ValueError(
            f"토크나이저 크기({len(tokenizer)})가 모델 임베딩 크기({embedding_size})보다 큽니다. "
            "같은 체크포인트의 tokenizer 와 모델인지 확인하세요."
        )

    result = {
        "tokenizer_class": type(tokenizer).__name__,
        "tokens": tokens,
        "unk_ratio": unknown_ratio,
        "first_token_id": input_ids[0],
        "last_token_id": input_ids[-1],
        "vocab_size": len(tokenizer),
        "embedding_size": embedding_size,
    }
    print(_console_safe_text(f"일반 백본({type(tokenizer).__name__}) Sanity check 통과: {result}"))
    return result


def build_tokenizer_and_model(
    model_name_or_path: Union[str, Path],
    local_files_only: bool = False,
    revision: Optional[str] = None,
) -> Tuple[object, object]:
    """사전학습 백본(klue/roberta-base 또는 TAPT 결과 폴더)과 9개 증상 분류 헤드를 생성합니다.

    분류 헤드는 새로 초기화되므로(전역 난수 사용) 호출 전에 `set_seed` 가 끝나 있어야 재현된다.
    """
    source = str(model_name_or_path)
    label2id = {symptom: index for index, symptom in enumerate(TARGET_SYMPTOMS)}
    id2label = {index: symptom for index, symptom in enumerate(TARGET_SYMPTOMS)}
    load_options: Dict[str, object] = {"local_files_only": local_files_only}
    if revision is not None:
        load_options["revision"] = revision

    # 1. Hugging Face 표준 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        **load_options,
    )

    # 2. 백본 + 9개 클래스 다중 라벨 분류 헤드 (첫 토큰 표현을 쓰는 표준 sequence classification)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=NUM_CLASSES,
        label2id=label2id,
        id2label=id2label,
        problem_type="multi_label_classification",
        **load_options,
    )

    # 3. 토크나이저와 모델의 호환성 및 한글 처리 상태 검증
    validate_tokenizer_model_compatibility(tokenizer, model)
    return tokenizer, model


def load_saved_model(model_dir: Union[str, Path]):
    """학습 후 저장된 모델 번들(best_model 폴더)을 외부 다운로드 없이 오프라인 환경에서 다시 로드합니다."""
    source = str(Path(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        local_files_only=True,
    )
    if int(model.config.num_labels) != NUM_CLASSES:
        raise ValueError(f"저장 모델의 출력 클래스 수가 9가 아닙니다: {model.config.num_labels}")
    validate_tokenizer_model_compatibility(tokenizer, model)
    return tokenizer, model


def save_model_bundle(model, tokenizer, output_dir: Union[str, Path]) -> Path:
    """추후 inference나 평가에 필요한 모델 가중치와 토크나이저 번들을 한 폴더에 함께 저장합니다."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return path
