"""라벨 JSON 리더 및 대회 규칙 강제 모듈.

대회 규칙: Mission 3 은 학습·추론 모두 라벨링 데이터에서 오직 대화 전사 본문
텍스트(`utterances[].text`)만 모델 입력으로 사용 가능
(정답 `symptom` 은 학습 타깃으로만 사용).
-> 멘토링 데이 09.05 질문건에 따라 전체 제외 즉 텍스트만 포함.

이 모듈이 반환하는 자료구조에는 그 외 메타데이터 필드가 **아예 담기지 않음.**
`gender`, `address`, `urgencyLevel`, `startAt`, `endAt`, `speaker` 등은 파싱
단계에서 원천 차단되어 버려지므로, 하위 모델링 코드가 실수로라도 규칙을 위반 불가.

또한 9개 평가 대상 외 증상(예: `골절`, `찰과상`, `화상` 등)은 정답 벡터에서
자동 필터링되어 9차원 표준 이진 벡터로 안전하게 변환.
Windows / macOS / Colab 환경 간의 인코딩 차이 및 한글 자모 분리(NFD)를 자동 방지.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np

from .config import (
    DEFAULT_UTTERANCE_SEP_MODE,
    NUM_CLASSES,
    SYMPTOM_TO_IDX,
    TARGET_SYMPTOMS,
    UTTERANCE_SEP_MODES,
    resolve_utterance_sep,
)


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


def _normalize_text(text: str) -> str:
    """macOS의 NFD(자모 분리) 한글을 표준 NFC(완성형) 한글로 정규화."""
    return unicodedata.normalize("NFC", text)


def _read_json_robust(path: Path) -> Dict[str, Any]:
    """UTF-8(BOM 포함), CP949 등 어떤 OS/환경의 JSON도 깨짐 없이 안전하게 로드."""
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue

    # 최후의 fallback: 디코딩 에러를 치환한다. 본문이 '?' 로 깨진 채 모델에 들어가면
    # 예외 없이 예측만 나빠지므로, 이 경로를 탔다는 사실 자체를 반드시 남긴다.
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    print(f"[경고] 인코딩을 판별하지 못해 치환 모드로 읽었습니다. 본문이 손상됐을 수 있습니다: {path}")
    return data


def _parse_symptoms(raw_symptoms: object) -> Tuple[Tuple[str, ...], np.ndarray]:
    """JSON의 symptom 필드에서 9개 타겟 증상만 필터링하고 이진 벡터로 변환."""
    target_set = set(TARGET_SYMPTOMS)
    filtered: List[str] = []
    vector = np.zeros(NUM_CLASSES, dtype=int)

    items = raw_symptoms if isinstance(raw_symptoms, list) else [raw_symptoms]
    for item in items:
        if isinstance(item, str):
            norm_item = _normalize_text(item.strip())
            if norm_item in target_set:
                filtered.append(norm_item)
                vector[SYMPTOM_TO_IDX[norm_item]] = 1

    return tuple(sorted(filtered)), vector


def _parse_dialogue_text(raw_utterances: object, separator: str = " ") -> str:
    """utterances 배열에서 startAt, endAt, speaker 메타데이터는 모두 버리고 text만 결합.

    `separator` 로 발화 사이에 경계 표시를 남길 수 있다. 빈 발화는 결합 전에 걸러내므로
    구분자가 연속으로 붙어 가짜 턴 경계가 생기지 않는다. 경계는 발화 사이에만 들어가며
    전체 텍스트의 앞뒤를 감싸지 않는다.
    """
    if not isinstance(raw_utterances, list):
        return ""

    texts: List[str] = []
    for item in raw_utterances:
        if not isinstance(item, dict):
            continue
        txt = item.get("text")
        if isinstance(txt, str) and txt.strip():
            texts.append(_normalize_text(txt.strip()))

    return separator.join(texts)


def verify_utterance_sep_mode(
    texts: "Iterable[str]",
    sep_mode: str,
    source: str = "",
) -> None:
    """본문이 선언한 발화 경계 모드와 실제로 일치하는지 확인한다.

    학습은 경계를 살린 CSV 로 하고 추론은 공백 CSV 로 하는 식의 불일치는 예외를 내지 않고
    점수만 조용히 떨어뜨린다. 그래서 본문을 직접 보고 어긋나면 즉시 실패시킨다.
    """
    marker = resolve_utterance_sep(sep_mode).strip()
    sample = [text for text in texts if isinstance(text, str)]
    if not sample:
        raise ValueError("검사할 본문이 없습니다.")
    label = f" ({source})" if source else ""

    if marker:
        if not any(re.search(re.escape(marker), text) for text in sample):
            raise ValueError(
                f"utterance_sep_mode={sep_mode!r} 인데 본문에 구분자 {marker!r} 가 없습니다{label}. "
                "경계를 살린 CSV 로 다시 만들었는지 확인하세요."
            )
        return

    for mode, separator in UTTERANCE_SEP_MODES.items():
        other = separator.strip()
        if not other:
            continue
        if any(re.search(re.escape(other), text) for text in sample):
            raise ValueError(
                f"utterance_sep_mode={sep_mode!r} 인데 본문에 {mode!r} 구분자 {other!r} 가 "
                f"있습니다{label}."
            )


def read_transcript(
    json_path: Union[str, Path],
    sep_mode: str = DEFAULT_UTTERANCE_SEP_MODE,
) -> TranscriptRecord:
    """단일 JSON 파일을 읽어 규칙이 강제된 TranscriptRecord로 변환.

    `sep_mode` 는 발화 경계 표현 방식이며 기본값은 기존과 동일한 공백 결합이다.
    """
    path = Path(json_path)
    call_id = path.stem

    data = _read_json_robust(path)

    # 허용된 text만 추출 (시간/화자/인적사항 등 영구 제거)
    text = _parse_dialogue_text(data.get("utterances"), resolve_utterance_sep(sep_mode))

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
    sep_mode: str = DEFAULT_UTTERANCE_SEP_MODE,
) -> List[TranscriptRecord]:
    """라벨 폴더 내의 모든 JSON을 일괄 파싱하여 모델에 바로 넣을 수 있는 리스트로 반환."""
    label_dir = Path(label_dir)
    json_files = sorted(label_dir.glob("*.json"))

    if max_samples is not None:
        json_files = json_files[:max_samples]

    resolve_utterance_sep(sep_mode)  # 잘못된 모드는 파일을 읽기 전에 즉시 실패시킨다

    records: List[TranscriptRecord] = []
    for jf in json_files:
        try:
            records.append(read_transcript(jf, sep_mode=sep_mode))
        except Exception:
            continue

    return records


def load_transcripts_dataframe(
    label_dir: Union[str, Path],
    max_samples: Optional[int] = None,
    sep_mode: str = DEFAULT_UTTERANCE_SEP_MODE,
):
    """KoBERT 모델 학습에 바로 넘길 수 있도록 pandas DataFrame 형태로 반환.
    
    컬럼:
      - call_id: 통화 식별자
      - text: 순수 발화 전사 텍스트
      - symptoms: 필터링된 타겟 증상명 리스트
      - label_vector: 9차원 이진 리스트 ([0, 1, 0, ...])
    """
    import pandas as pd

    records = load_transcripts_dir(label_dir, max_samples=max_samples, sep_mode=sep_mode)
    data = [
        {
            "call_id": r.call_id,
            "text": r.text,
            "symptoms": list(r.symptoms),
            "label_vector": r.label_vector.tolist() if r.label_vector is not None else None,
        }
        for r in records
    ]
    return pd.DataFrame(data)