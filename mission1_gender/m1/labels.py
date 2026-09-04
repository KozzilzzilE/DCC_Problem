"""라벨 JSON 리더.

대회 규칙: Mission 1 은 학습·추론 모두 라벨링 데이터의 `startAt`, `endAt`,
`speaker` 만 사용할 수 있다 (정답 `gender` 는 학습 타깃으로만 사용).

이 모듈이 반환하는 자료구조에는 그 외 필드가 **담기지 않는다.** `text`,
`symptom`, `address`, `urgencyLevel` 등은 파싱 단계에서 버려지므로, 하위
코드가 실수로라도 규칙을 위반할 수 없다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DISPATCHER = 0  # 119 대원
CALLER = 1  # 신고자

_VALID_GENDERS = ("M", "F")


@dataclass(frozen=True, slots=True)
class Utterance:
    """한 사람의 한 발화 조각. 허용된 세 필드만 갖는다."""

    start_ms: int
    end_ms: int
    speaker: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class CallRecord:
    """통화 1건. `gender` 는 학습 타깃이며 평가 입력에는 없을 수 있다."""

    call_id: str
    utterances: tuple[Utterance, ...]
    gender: str | None


def _parse_gender(raw: object) -> str | None:
    if isinstance(raw, str) and raw.upper() in _VALID_GENDERS:
        return raw.upper()
    return None


def _parse_utterances(raw: object) -> tuple[Utterance, ...]:
    if not isinstance(raw, list):
        return ()

    parsed = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item["startAt"])
            end = int(item["endAt"])
            speaker = int(item["speaker"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:  # 길이가 0 이하인 조각은 쓸 수 없다
            continue
        parsed.append(Utterance(start_ms=start, end_ms=end, speaker=speaker))
    return tuple(parsed)


def read_call(json_path: str | Path) -> CallRecord:
    """라벨 JSON 하나를 읽어 허용된 필드만 담은 CallRecord 로 반환한다."""
    path = Path(json_path)
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    return CallRecord(
        call_id=path.stem,
        utterances=_parse_utterances(payload.get("utterances")),
        gender=_parse_gender(payload.get("gender")),
    )


def caller_utterances(record: CallRecord) -> tuple[Utterance, ...]:
    """신고자(speaker == 1)의 발화 조각만 골라낸다."""
    return tuple(u for u in record.utterances if u.speaker == CALLER)


def iter_calls(label_dir: str | Path) -> Iterator[CallRecord]:
    """디렉터리의 모든 라벨 JSON 을 call_id 순으로 읽는다."""
    for path in sorted(Path(label_dir).glob("*.json")):
        yield read_call(path)
