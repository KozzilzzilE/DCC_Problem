"""Mission 3 다중 라벨 분류 모델 및 토크나이저 구성 (KoBERT, KoELECTRA, RoBERTa 등 다중 백본 지원).

[설계 배경 및 안내]
1. KoBERT (skt/kobert-base-v1):
   - Hugging Face AutoTokenizer로 로드 시 SentencePiece 모델이 아닌 XLNetTokenizer 등이 잘못 매핑되어
     한국어가 자모 단위로 분해되고 [UNK] 토큰이 폭증하는 고질적인 문제가 있습니다.
     따라서 KoBERT 계열은 별도로 구현된 `KoBertTokenizer`와 엄격한 특수 토큰 검증을 적용합니다.
2. 일반 한국어 백본 (monologg/koelectra-base-v3-discriminator, klue/roberta-base 등):
   - 표준 Hugging Face `AutoTokenizer`를 정상적으로 지원하므로, 모델명만 지정하면
     자동으로 다운로드/로드되어 학습에 즉시 투입될 수 있도록 범용 검증을 적용합니다.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification

from .config import NUM_CLASSES, TARGET_SYMPTOMS

# KoBERT tokenizer 는 sentencepiece 에 의존한다. 제출 모델(KLUE-RoBERTa)은 이를 쓰지 않으므로
# 최상단에서 import 하지 않는다. 평가 환경에 sentencepiece 가 없어도 추론이 죽지 않아야 한다.


def _kobert_tokenizer_class():
    """KoBERT 계열을 실제로 다룰 때만 sentencepiece 의존성을 건드린다."""
    from .kobert_tokenizer import KoBertTokenizer

    return KoBertTokenizer


def _is_kobert_tokenizer(tokenizer) -> bool:
    """클래스를 import 하지 않고 KoBERT tokenizer 인지 판별한다."""
    return type(tokenizer).__name__ == "KoBertTokenizer"


# 기본 베이스라인 모델 (KoBERT)
DEFAULT_MODEL_NAME = "skt/kobert-base-v1"

# 토크나이저 한글 정상 처리 검증을 위한 테스트 문장
TOKENIZER_SANITY_TEXT = "한국어 모델을 공유합니다."

# KoBERT SentencePiece 단어 사전에서 기대하는 고정 Special Token ID 목록
EXPECTED_KOBERT_SPECIAL_TOKEN_IDS = {
    "unk_token_id": 0,
    "pad_token_id": 1,
    "cls_token_id": 2,
    "sep_token_id": 3,
    "mask_token_id": 4,
}

# 한글 초성/중성/종성(자모) 유니코드 범위 정규식: 토크나이저가 한글을 자모 단위로 깨뜨리는지 감지
JAMO_PATTERN = re.compile(r"[\u1100-\u11ff\u3130-\u318f]")

CLS_POOLING = "cls"
LABEL_ATTENTION_POOLING = "label_attention"
SUPPORTED_POOLING_TYPES = (CLS_POOLING, LABEL_ATTENTION_POOLING)


@dataclass
class LabelAttentionSequenceClassifierOutput(SequenceClassifierOutput):
    """표준 sequence-classification 출력에 label별 token attention을 추가합니다."""

    label_attention_weights: Optional[torch.FloatTensor] = None


class RobertaForLabelWiseAttentionClassification(RobertaForSequenceClassification):
    """RoBERTa의 기존 classifier parameter를 재사용하는 label-wise pooling 모델."""

    def __init__(self, config):
        super().__init__(config)
        self.config.pooling_type = LABEL_ATTENTION_POOLING
        self.label_queries = nn.Embedding(config.num_labels, config.hidden_size)
        # RoBERTa/Hugging Face의 Embedding 초기화 규칙(initializer_range)을 그대로 사용합니다.
        self._init_weights(self.label_queries)

    def label_attention_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B,L,H]를 padding을 제외한 label별 [B,C,H] 표현으로 집계합니다."""
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states는 [batch, sequence_length, hidden_size]여야 합니다: "
                f"shape={tuple(hidden_states.shape)}"
            )
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != self.config.hidden_size:
            raise ValueError(
                f"hidden_size가 config와 다릅니다: {hidden_size} != {self.config.hidden_size}"
            )

        if attention_mask is None:
            valid_mask = torch.ones(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        else:
            if tuple(attention_mask.shape) != (batch_size, sequence_length):
                raise ValueError(
                    "attention_mask는 [batch, sequence_length]여야 합니다: "
                    f"shape={tuple(attention_mask.shape)}"
                )
            valid_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)

        if not bool(valid_mask.any(dim=-1).all()):
            raise ValueError("모든 token이 padding인 입력은 label attention을 계산할 수 없습니다.")

        attention_scores = torch.einsum(
            "blh,ch->bcl",
            hidden_states,
            self.label_queries.weight,
        )
        attention_scores = attention_scores.masked_fill(
            ~valid_mask[:, None, :],
            torch.finfo(attention_scores.dtype).min,
        )
        attention_weights = torch.softmax(attention_scores.float(), dim=-1).to(
            hidden_states.dtype
        )
        label_representations = torch.einsum(
            "bcl,blh->bch",
            attention_weights,
            hidden_states,
        )
        return label_representations, attention_weights

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> LabelAttentionSequenceClassifierOutput:
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        label_representations, attention_weights = self.label_attention_pool(
            outputs.last_hidden_state,
            attention_mask,
        )

        x = self.classifier.dropout(label_representations)
        x = self.classifier.dense(x)
        x = torch.tanh(x)
        x = self.classifier.dropout(x)
        logits = torch.einsum("bch,ch->bc", x, self.classifier.out_proj.weight)
        logits = logits + self.classifier.out_proj.bias

        loss = None
        if labels is not None:
            labels = labels.to(device=logits.device, dtype=logits.dtype)
            loss = BCEWithLogitsLoss()(logits, labels)

        return LabelAttentionSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            label_attention_weights=attention_weights,
        )


def validate_pooling_type(pooling_type: str) -> str:
    """pooling 선택값을 검증하고 정상화된 문자열을 반환합니다."""
    if pooling_type not in SUPPORTED_POOLING_TYPES:
        raise ValueError(
            f"지원하지 않는 pooling_type입니다: {pooling_type}. "
            f"허용값: {SUPPORTED_POOLING_TYPES}"
        )
    return pooling_type


def validate_label_attention_backbone(model_type: str, source: str) -> None:
    """첫 ablation에서는 RoBERTa 계열 외 label attention 사용을 명시적으로 차단합니다."""
    if model_type != "roberta":
        raise ValueError(
            "pooling_type='label_attention'은 현재 RoBERTa 계열만 지원합니다: "
            f"model={source}, model_type={model_type}"
        )


def _console_safe_text(value: object) -> str:
    """현재 stdout 인코딩에서 표현할 수 없는 문자를 escape하여 반환합니다."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(encoding, errors="backslashreplace").decode(encoding)


def is_kobert_model(model_name_or_path: Union[str, Path]) -> bool:
    """주어진 모델 이름이나 로컬 디렉터리 경로가 KoBERT 계열인지 판별합니다.
    
    판별 기준:
    1. 로컬 디렉터리에 KoBERT 전용 단어 모델인 `spiece.model` 파일이 존재하는 경우
    2. 모델 이름/경로 문자열에 'kobert'가 포함된 경우
    """
    path = Path(model_name_or_path)
    if path.is_dir() and (path / "spiece.model").is_file():
        return True
    name = str(model_name_or_path).lower()
    return "kobert" in name


def validate_kobert_tokenizer_model_compatibility(tokenizer, model) -> Dict[str, object]:
    """KoBERT 전용 토크나이저와 모델 간의 특수 입력 계약을 엄격히 검증합니다.
    
    KoBERT는 SentencePiece vocabulary 매핑이 어긋나면 성능이 급락하므로,
    Special Token ID 일치 여부와 어휘 크기, UNK 비율을 철저히 검사합니다.
    """
    # 1. KoBERT 전용 토크나이저 클래스인지 확인
    if not _is_kobert_tokenizer(tokenizer):
        raise TypeError(f"KoBertTokenizer가 아닙니다: {type(tokenizer).__name__}")

    # 2. [CLS], [SEP], [PAD] 등의 Special Token ID가 일치하는지 확인
    for attribute, expected in EXPECTED_KOBERT_SPECIAL_TOKEN_IDS.items():
        actual = getattr(tokenizer, attribute)
        if actual != expected:
            raise ValueError(f"{attribute}가 KoBERT vocabulary와 다릅니다: {actual}")

    # 3. 토크나이징 시 한글이 자모 단위(ㅎ, ㅏ, ㄴ, ...)로 분해되지 않는지 확인
    tokens = tokenizer.tokenize(TOKENIZER_SANITY_TEXT)
    if any(JAMO_PATTERN.search(token) for token in tokens):
        raise ValueError(f"한국어가 자모 단위로 분해되었습니다: {tokens}")

    # 4. 문장 앞뒤로 [CLS], [SEP] 토큰이 정확히 붙는지 확인
    encoded = tokenizer(TOKENIZER_SANITY_TEXT, add_special_tokens=True)
    input_ids = encoded["input_ids"]
    if input_ids[0] != tokenizer.cls_token_id:
        raise ValueError("첫 token이 [CLS]가 아닙니다.")
    if input_ids[-1] != tokenizer.sep_token_id:
        raise ValueError("마지막 token이 [SEP]가 아닙니다.")

    # 5. 문장 본문에 알 수 없는 토큰([UNK]) 비율이 10%를 초과하지 않는지 확인
    content_ids = input_ids[1:-1]
    unknown_count = sum(token_id == tokenizer.unk_token_id for token_id in content_ids)
    unknown_ratio = unknown_count / max(len(content_ids), 1)
    if unknown_ratio > 0.1:
        raise ValueError(f"한국어 문장의 [UNK] 비율이 비정상적입니다: {unknown_ratio:.4f}")

    # 6. 토크나이저 단어 수와 모델 임베딩 레이어의 크기가 동일한지 확인
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
    print(_console_safe_text(f"KoBERT Tokenizer sanity check 통과: {result}"))
    return result


def validate_generic_tokenizer_model_compatibility(tokenizer, model) -> Dict[str, object]:
    """일반 Hugging Face 모델(KoELECTRA, RoBERTa 등)의 토크나이저 호환성을 검증합니다.
    
    모델마다 토큰 규격이 조금씩 다를 수 있으므로(예: BERT는 CLS/SEP, RoBERTa는 BOS/EOS),
    각 모델의 스페셜 토큰 규격에 맞춰 범용적으로 유효성을 체크합니다.
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

    # 4. 토크나이저와 모델 임베딩 크기 동기화
    # 일부 사전학습 토크나이저는 추가 토큰이 있어 임베딩 레이어보다 클 수 있습니다.
    embedding_size = int(model.get_input_embeddings().num_embeddings)
    if len(tokenizer) > embedding_size:
        print(f"[안내] 토크나이저 크기({len(tokenizer)})에 맞춰 모델 임베딩 레이어 크기를 자동 확장합니다.")
        model.resize_token_embeddings(len(tokenizer))
        embedding_size = int(model.get_input_embeddings().num_embeddings)

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


def validate_tokenizer_model_compatibility(tokenizer, model) -> Dict[str, object]:
    """토크나이저 타입에 맞춰 적합한 검증 로직(KoBERT 전용 vs 범용)을 자동으로 분기 실행합니다."""
    if _is_kobert_tokenizer(tokenizer):
        return validate_kobert_tokenizer_model_compatibility(tokenizer, model)
    return validate_generic_tokenizer_model_compatibility(tokenizer, model)


def build_tokenizer_and_model(
    model_name_or_path: Union[str, Path] = DEFAULT_MODEL_NAME,
    local_files_only: bool = False,
    revision: Optional[str] = None,
    pooling_type: str = CLS_POOLING,
) -> Tuple[object, object]:
    """사전학습 백본(KoBERT, KoELECTRA, RoBERTa 등)과 9개 증상 분류 헤드(Linear Head)를 생성합니다.
    
    - KoBERT 모델이면: SentencePiece 기반 `KoBertTokenizer` 로드
    - 그 외 한국어 모델이면: Hugging Face 표준 `AutoTokenizer` 자동 로드
    """
    source = str(model_name_or_path)
    pooling_type = validate_pooling_type(pooling_type)
    label2id = {symptom: index for index, symptom in enumerate(TARGET_SYMPTOMS)}
    id2label = {index: symptom for index, symptom in enumerate(TARGET_SYMPTOMS)}
    load_options: Dict[str, object] = {"local_files_only": local_files_only}
    if revision is not None:
        load_options["revision"] = revision

    label_attention_config = None
    if pooling_type == LABEL_ATTENTION_POOLING:
        label_attention_config = AutoConfig.from_pretrained(source, **load_options)
        validate_label_attention_backbone(label_attention_config.model_type, source)
        label_attention_config.num_labels = NUM_CLASSES
        label_attention_config.label2id = label2id
        label_attention_config.id2label = id2label
        label_attention_config.problem_type = "multi_label_classification"
        label_attention_config.pooling_type = LABEL_ATTENTION_POOLING

    # 1. 모델 종류에 따라 최적의 토크나이저 자동 선택 및 로드
    if is_kobert_model(source):
        tokenizer = _kobert_tokenizer_class().from_pretrained(
            source,
            **load_options,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            **load_options,
        )

    # 2. cls 기본 경로는 기존 AutoModelForSequenceClassification 호출을 그대로 유지합니다.
    if pooling_type == CLS_POOLING:
        model = AutoModelForSequenceClassification.from_pretrained(
            source,
            num_labels=NUM_CLASSES,
            label2id=label2id,
            id2label=id2label,
            problem_type="multi_label_classification",
            **load_options,
        )
    else:
        assert label_attention_config is not None
        model = RobertaForLabelWiseAttentionClassification.from_pretrained(
            source,
            config=label_attention_config,
            **load_options,
        )

    # 3. 토크나이저와 모델의 호환성 및 한글 처리 상태 검증
    validate_tokenizer_model_compatibility(tokenizer, model)
    return tokenizer, model


def load_saved_model(model_dir: Union[str, Path]):
    """학습 후 저장된 모델 번들을 외부 다운로드 없이 오프라인 환경에서 다시 로드합니다.
    
    저장 디렉터리의 아티팩트를 분석하여 KoBERT인지 일반 모델인지 자동으로 감지합니다.
    """
    source = str(Path(model_dir))
    if is_kobert_model(source):
        tokenizer = _kobert_tokenizer_class().from_pretrained(source, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)

    config = AutoConfig.from_pretrained(source, local_files_only=True)
    pooling_type = validate_pooling_type(getattr(config, "pooling_type", CLS_POOLING))
    if pooling_type == LABEL_ATTENTION_POOLING:
        validate_label_attention_backbone(config.model_type, source)
        model = RobertaForLabelWiseAttentionClassification.from_pretrained(
            source,
            config=config,
            local_files_only=True,
        )
    else:
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

