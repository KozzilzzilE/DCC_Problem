"""라벨 JSON 을 통화 단위로 읽는다. 발화 정렬과 제외 규칙은 대회 템플릿(inference.py load_samples)과 같다.

평가셋에는 발화(utterances)만 남고 나머지 라벨 필드는 지워지므로, 모델 입력으로는 발화의
index / speaker / text / startAt / endAt 만 쓴다. 다른 필드(urgencyLevel, triage 등)는 읽지 않는다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from . import SYMPTOM_CLASSES


@dataclass
class Call:
    call_id: str                 # 라벨 파일 stem
    texts: List[str]             # 발화 전사 (템플릿과 같은 순서)
    speakers: List[int]          # 0 = 119대원, 1 = 신고자
    starts: List[int]            # startAt (ms)
    ends: List[int]              # endAt (ms)
    labels: List[int]            # 9종 0/1

    @property
    def n_utt(self) -> int:
        return len(self.texts)


def read_call(path: Path) -> Optional[Call]:
    """JSON 하나를 읽는다. 9종 필터 뒤 증상이 비거나 발화가 없으면 None (템플릿과 같은 제외 규칙)."""
    with open(path, "r", encoding="utf-8-sig") as handle:
        d = json.load(handle)
    raw = set(d.get("symptom") or [])
    labels = [1 if c in raw else 0 for c in SYMPTOM_CLASSES]
    if not any(labels):
        return None
    utts = list(d.get("utterances") or [])
    if not utts:
        return None
    utts.sort(key=lambda u: (int(u.get("startAt", 0)), int(u.get("endAt", 0))))
    return Call(
        call_id=path.stem,
        texts=[str(u.get("text", "")) for u in utts],
        speakers=[int(u.get("speaker", -1)) for u in utts],
        starts=[int(u.get("startAt", 0)) for u in utts],
        ends=[int(u.get("endAt", 0)) for u in utts],
        labels=labels,
    )


def build_cache(label_dir: Path, out_path: Path) -> int:
    """폴더 전체를 JSONL 한 파일로 모은다 (JSON 3만 개를 매번 여는 시간을 줄이려는 캐시)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for name in sorted(os.listdir(label_dir)):
            if not name.lower().endswith(".json"):
                continue
            call = read_call(Path(label_dir) / name)
            if call is None:
                continue
            out.write(json.dumps(call.__dict__, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_cache(path: Path, keep_ids: Optional[Iterable[str]] = None) -> List[Call]:
    keep = set(keep_ids) if keep_ids is not None else None
    calls: List[Call] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if keep is None or row["call_id"] in keep:
                calls.append(Call(**row))
    return calls


def read_ids(path: Path) -> List[str]:
    return [line.strip() for line in open(path, "r", encoding="utf-8") if line.strip()]


def prefix_texts(call: Call, fmt: str = "space") -> List[str]:
    """step t(1..n) 까지 누적한 입력 문자열 n개. 추론(Predictor)과 학습이 이 함수를 같이 쓴다.

    fmt
      space   : 발화를 공백으로 잇는다 (예선 Mission 3 입력과 같은 형식)
      speaker : 발화 앞에 화자 표시([대원]/[신고자])를 붙인다
    """
    parts: List[str] = []
    out: List[str] = []
    for text, speaker in zip(call.texts, call.speakers):
        piece = format_utterance(text, speaker, fmt)
        if piece:
            parts.append(piece)
        out.append(" ".join(parts))
    return out


def format_utterance(text: str, speaker: int, fmt: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if fmt == "space":
        return text
    if fmt == "speaker":
        tag = "[대원]" if speaker == 0 else ("[신고자]" if speaker == 1 else "")
        return f"{tag} {text}".strip()
    raise ValueError(f"알 수 없는 입력 형식: {fmt}")


def label_matrix(calls: Sequence[Call]):
    import numpy as np

    return np.asarray([c.labels for c in calls], dtype=np.int8)
