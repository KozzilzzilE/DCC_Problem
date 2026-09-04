"""라벨 JSON 리더 및 대회 규칙 강제 모듈.

대회 규칙: Mission 3 은 학습·추론 모두 라벨링 데이터에서 오직 대화 전사 본문
텍스트(`utterances[].text`)만 모델 입력으로 사용 가능
(정답 `symptom` 은 학습 타깃으로만 사용).

이 모듈이 반환하는 자료구조에는 그 외 메타데이터 필드가 **아예 담기지 않음.**
`gender`, `address`, `urgencyLevel`, `startAt`, `endAt`, `speaker` 등은 파싱
단계에서 원천 차단되어 버려지므로, 하위 모델링 코드가 실수로라도 규칙을 위반 불가.

또한 9개 평가 대상 외 증상(예: `골절`, `찰과상`, `화상` 등)은 정답 벡터에서
자동 필터링되어 9차원 표준 이진 벡터로 안전하게 변환.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

from .config import NUM_CLASSES, SYMPTOM_TO_IDX, TARGET_SYMPTOMS


@dataclass(frozen=True)
class TranscriptRecord:
    """통화 1건의 전사 데이터.
    
    대회 규칙에 따라 허용된 본문 텍스트(`text`)와 정답 타깃(`symptoms`, `label_vector`)만 포함하며,
    음성 메타데이터(시간, 화자, 인적사항 등)는 원천 배제.
    """

    call_id: str
    text: str
    symptoms: Tuple[str, ...]
    label_vector: Optional[np.ndarray]  # shape: (9,), dtype: int (평가 추론 시에는 None 가능)


def _parse_symptoms(raw_symptoms: object) -> Tuple[Tuple[str, ...], np.ndarray]:
    """JSON의 symptom 필드에서 9개 타겟 증상만 필터링하고 이진 벡터로 변환."""
    target_set = set(TARGET_SYMPTOMS)
    filtered: List[str] = []
    vector = np.zeros(NUM_CLASSES, dtype=int)

    if isinstance(raw_symptoms, list):
        for item in raw_symptoms:
            if isinstance(item, str) and item in target_set:
                filtered.append(item)
                vector[SYMPTOM_TO_IDX[item]] = 1
    elif isinstance(raw_symptoms, str) and raw_symptoms in target_set:
        filtered.append(raw_symptoms)
        vector[SYMPTOM_TO_IDX[raw_symptoms]] = 1

    return tuple(sorted(filtered)), vector


def _parse_dialogue_text(raw_utterances: object) -> str:
    """utterances 배열에서 startAt, endAt, speaker 메타데이터는 모두 버리고 text만 결합."""
    if not isinstance(raw_utterances, list):
        return ""

    texts: List[str] = []
    for item in raw_utterances:
        if not isinstance(item, dict):
            continue
        txt = item.get("text")
        if isinstance(txt, str) and txt.strip():
            texts.append(txt.strip())

    return " ".join(texts)


def read_transcript(json_path: Union[str, Path]) -> TranscriptRecord:
    """단일 JSON 파일을 읽어 규칙을 강제한 TranscriptRecord로 변환."""
    path = Path(json_path)
    call_id = path.stem

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 허용된 text만 추출 (시간/화자/인적사항 등 제거)
    text = _parse_dialogue_text(data.get("utterances"))

    # 학습 타깃 9개 증상 필터링 (골절, 찰과상 등 비타겟 제거)
    raw_syms = data.get("symptom")
    symptoms, label_vec = _parse_symptoms(raw_syms)

    return TranscriptRecord(
        call_id=call_id,
        text=text,
        symptoms=symptoms,
        label_vector=label_vec,
    )


def load_transcripts_dir(
    label_dir: Union[str, Path],
    max_samples: Optional[int] = None,
) -> List[TranscriptRecord]:
    """라벨 폴더 내의 모든 JSON을 일괄 파싱하여 모델에 바로 넣을 수 있는 리스트로 반환."""
    label_dir = Path(label_dir)
    json_files = sorted(label_dir.glob("*.json"))

    if max_samples is not None:
        json_files = json_files[:max_samples]

    records: List[TranscriptRecord] = []
    for jf in json_files:
        try:
            records.append(read_transcript(jf))
        except Exception:
            continue

    return records
