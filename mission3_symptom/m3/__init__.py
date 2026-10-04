"""Mission 3 환자 증상 다중 라벨 분류 패키지

모듈 구성:
  - config        : 9개 증상 목록, 발화 경계 모드, 인코딩 방식 등 고정 상수
  - labels        : 라벨 JSON 에서 대화 본문(utterances[].text)과 정답만 읽는다 (대회 규정 강제)
  - dataset       : 학습 CSV -> 토큰화 -> DataLoader
  - model         : KLUE-RoBERTa 분류 모델·토크나이저 로드와 호환성 검사
  - training      : 학습 루프, epoch 별 평가, 체크포인트 저장
  - metrics       : 임계값 0.5 고정 macro F1
  - tfidf_member  : Training 전용 TF-IDF + LR 보조 멤버
  - bundle        : 제출용 .pt 번들 묶기·불러오기
  - infer         : 제출 추론 (0.5 고정 판정, 제출 CSV 규격)
"""

from .config import (
    DEFAULT_UTTERANCE_SEP_MODE,
    NUM_CLASSES,
    SYMPTOM_TO_IDX,
    TARGET_SYMPTOMS,
    UTTERANCE_SEP_MODES,
    resolve_utterance_sep,
)
from .labels import (
    TranscriptRecord,
    load_transcripts_dir,
    read_transcript,
)
from .metrics import (
    apply_thresholds,
    calculate_binary_f1,
    eval_macro_f1,
)

__all__ = [
    "TARGET_SYMPTOMS",
    "UTTERANCE_SEP_MODES",
    "DEFAULT_UTTERANCE_SEP_MODE",
    "resolve_utterance_sep",
    "NUM_CLASSES",
    "SYMPTOM_TO_IDX",
    "TranscriptRecord",
    "read_transcript",
    "load_transcripts_dir",
    "apply_thresholds",
    "calculate_binary_f1",
    "eval_macro_f1",
]
