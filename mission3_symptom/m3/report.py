"""Mission 3 리포트 생성 및 최적 임계값 내보내기 모듈"""

import json
from pathlib import Path
from typing import Dict, Union
import numpy as np

from .config import (
    BEST_THRESHOLDS_PATH,
    COMPARISON_REPORT_PATH,
    REPORTS_DIR,
    TARGET_SYMPTOMS,
)


def save_thresholds_json(
    best_thresholds: Union[np.ndarray, list],
    output_path: Union[str, Path] = BEST_THRESHOLDS_PATH,
) -> Path:
    """도출된 9개 증상별 최적 임계값을 JSON 파일로 저장.
    (inference.py 추론 시 로드하여 사용)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    th_dict = {
        sym: float(best_thresholds[i])
        for i, sym in enumerate(TARGET_SYMPTOMS)
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(th_dict, f, ensure_ascii=False, indent=2)

    return output_path


def generate_comparison_markdown(
    base_macro: float,
    opt_macro: float,
    base_class_f1: Dict[str, float],
    opt_class_f1: Dict[str, float],
    best_thresholds: Union[np.ndarray, list],
    output_path: Union[str, Path] = COMPARISON_REPORT_PATH,
) -> str:
    """임계값 최적화 전/후 비교 마크다운 리포트를 생성하고 저장."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    delta_macro = opt_macro - base_macro
    delta_macro_str = f"+{delta_macro:.4f}" if delta_macro > 0 else f"{delta_macro:.4f}"

    lines = [
        "# Mission 3 — 환자 증상 인식: 클래스별 임계값(Threshold) 최적화 성과",
        "",
        f"- **기준선 (기본 임계값 0.5) Macro F1**: `{base_macro:.4f}`",
        f"- **최적화 후 (Class-wise Threshold) Macro F1**: `{opt_macro:.4f}` ({delta_macro_str})",
        "",
        "## 9개 증상별 F1-score 비교표",
        "",
        "| 증상 클래스 | 기준(0.5) F1 | 최적 임계값 | 최적화 F1 | 상승폭(Delta) |",
        "|---|---|---|---|---|",
    ]

    for idx, sym in enumerate(TARGET_SYMPTOMS):
        bf = base_class_f1[sym]
        th = float(best_thresholds[idx])
        of = opt_class_f1[sym]
        diff = of - bf
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        lines.append(f"| **{sym}** | {bf:.4f} | `{th:.2f}` | {of:.4f} | **{diff_str}** |")

    lines.append("")
    lines.append("## 결론 및 시사점")
    lines.append("1. **불균형 완화**: 발현 빈도가 낮은 희귀 증상(고열, 열상 등)에서 임계값을 하향 조정하여 Recall을 대폭 확보함.")
    lines.append("2. **대회 지표(Macro F1) 극대화**: 클래스별 단순 평균인 Macro F1 특성상, 취약 클래스의 점수 상승이 전체 점수를 직접 견인함.")

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content
