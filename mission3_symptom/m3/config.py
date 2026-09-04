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

# 경로 설정
MISSION3_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = MISSION3_DIR / "reports"
BEST_THRESHOLDS_PATH = REPORTS_DIR / "best_thresholds.json"
COMPARISON_REPORT_PATH = REPORTS_DIR / "comparison.md"
