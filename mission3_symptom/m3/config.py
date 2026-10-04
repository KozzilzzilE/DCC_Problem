"""Mission 3 설정 및 표준 상수 정의"""

from typing import Dict, List

# 대회 공식 9개 타겟 증상 목록 (가나다순)
TARGET_SYMPTOMS: List[str] = [
    "고열",
    "구토",
    "두통",
    "복통",
    "어지러움",
    "열상",
    "오심",
    "전신쇠약",
    "호흡곤란",
]

NUM_CLASSES: int = len(TARGET_SYMPTOMS)
SYMPTOM_TO_IDX: Dict[str, int] = {sym: idx for idx, sym in enumerate(TARGET_SYMPTOMS)}

# 발화 경계(턴 구분) 표현 방식.
# 대회 Q&A 답변(2026-09-20)으로 speaker 값을 사용하지 않고 발화 경계만 남기는 전처리는 허용됨.
# 개행(\n)은 모드로 제공하지 않는다. KLUE-RoBERTa 등 BERT 계열 tokenizer 는 basic tokenization
# 단계에서 개행을 공백과 동일하게 처리하므로 token 열이 전혀 바뀌지 않아 "space" 와 같다.
# 전용 special token 을 새로 등록하는 모드는 두지 않는다 (tokenizer 등록·embedding resize 가 필요해
# 저장된 어휘를 바꾸게 된다).
UTTERANCE_SEP_MODES: Dict[str, str] = {
    "space": " ",        # 기본값. 제출 모델이 쓰는 방식이며 경계 정보 없음
    "sep": " [SEP] ",    # tokenizer 기본 어휘의 경계 토큰을 재사용 (vocab 변경 불필요)
}
DEFAULT_UTTERANCE_SEP_MODE: str = "space"

# max_length 를 넘는 통화의 인코딩 방식. 앞부분만 남기는 truncate 하나만 지원한다.
# run_config.json / inference_config.json 의 encode_mode 필드는 기존 산출물과의 호환을 위해 남겨 두고,
# 이 값이 아니면 학습·추론 모두 즉시 실패시킨다 (다른 방식으로 학습된 모델을 조용히 잘못 읽지 않게).
ENCODE_MODE: str = "truncate"


def resolve_utterance_sep(mode: str = DEFAULT_UTTERANCE_SEP_MODE) -> str:
    """구분자 모드 이름을 실제 문자열로 변환. 알 수 없는 모드는 조용히 넘기지 않는다."""
    try:
        return UTTERANCE_SEP_MODES[mode]
    except KeyError:
        raise ValueError(
            f"지원하지 않는 utterance separator 모드입니다: {mode!r} "
            f"(가능한 값: {sorted(UTTERANCE_SEP_MODES)})"
        ) from None


def check_encode_mode(mode: object) -> str:
    """설정 파일의 encode_mode 가 지원하는 값(truncate)인지 확인하고 그대로 돌려준다."""
    if mode != ENCODE_MODE:
        raise ValueError(f"지원하지 않는 encode_mode입니다: {mode!r} (지원: {ENCODE_MODE!r})")
    return ENCODE_MODE
