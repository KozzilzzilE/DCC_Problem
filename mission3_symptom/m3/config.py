"""Mission 3 설정 및 표준 상수 정의"""

from pathlib import Path
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
IDX_TO_SYMPTOM: Dict[int, str] = {idx: sym for idx, sym in enumerate(TARGET_SYMPTOMS)}

# 발화 경계(턴 구분) 표현 방식.
# 대회 Q&A 답변(2026-09-20)으로 speaker 값을 사용하지 않고 발화 경계만 남기는 전처리는 허용됨.
# 개행(\n)은 모드로 제공하지 않는다. KLUE-RoBERTa 등 BERT 계열 tokenizer 는 basic tokenization
# 단계에서 개행을 공백과 동일하게 처리하므로 token 열이 전혀 바뀌지 않아 "space" 와 같다.
UTTERANCE_SEP_MODES: Dict[str, str] = {
    "space": " ",        # 기존 동작. 경계 정보 없음
    "sep": " [SEP] ",    # tokenizer 기본 어휘의 경계 토큰을 재사용 (vocab 변경 불필요)
    "turn": " [TURN] ",  # 전용 special token. tokenizer 등록과 embedding resize 가 선행돼야 함
}
DEFAULT_UTTERANCE_SEP_MODE: str = "space"


def resolve_utterance_sep(mode: str = DEFAULT_UTTERANCE_SEP_MODE) -> str:
    """구분자 모드 이름을 실제 문자열로 변환. 알 수 없는 모드는 조용히 넘기지 않는다."""
    try:
        return UTTERANCE_SEP_MODES[mode]
    except KeyError:
        raise ValueError(
            f"지원하지 않는 utterance separator 모드입니다: {mode!r} "
            f"(가능한 값: {sorted(UTTERANCE_SEP_MODES)})"
        ) from None


# 경로 설정
MISSION3_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = MISSION3_DIR / "reports"
BEST_THRESHOLDS_PATH = REPORTS_DIR / "best_thresholds.json"
COMPARISON_REPORT_PATH = REPORTS_DIR / "comparison.md"
