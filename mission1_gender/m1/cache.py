"""신고자 발화 조각 오디오 캐시.

팀원 baseline 의 Dataset 은 발화 조각 하나를 꺼낼 때마다 통화 전체 wav 를 다시
디코딩한다 (조각당 ~1.1 MB, 학습 조각 약 105 만 개). 여기서는 통화당 wav 를
**한 번만** 읽어 신고자 조각을 전부 잘라 저장해 둔다.

원본 8 kHz 를 그대로 유지하므로 CNN 갈래(스펙트로그램)와 Wav2Vec2 갈래(16 kHz
업샘플)가 같은 캐시를 공유한다. 덕분에 두 갈래의 속도 비교가 I/O 차이가 아닌
모델 차이를 반영한다.

용량: 통화당 신고자 음성 34.0 초 -> 약 544 KB. 전체 29,200 통화 기준 15.9 GB.
"""
from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .labels import caller_utterances, read_call

INDEX_NAME = "index.csv"
SEGMENT_DIRNAME = "segments"
_INDEX_FIELDS = ("call_id", "seg_idx", "offset", "length", "gender")


@dataclass(frozen=True, slots=True)
class CacheRow:
    """캐시에 담긴 조각 하나의 위치와 통화 라벨."""

    call_id: str
    seg_idx: int
    offset: int
    length: int
    gender: str | None

    def as_tuple(self) -> tuple:
        return (self.call_id, self.seg_idx, self.offset, self.length, self.gender)


class CacheIndex:
    """캐시 디렉터리와 그 안의 조각 목록."""

    def __init__(self, cache_dir: str | Path, rows: list[CacheRow]):
        self.cache_dir = Path(cache_dir)
        self.rows = rows

    @property
    def segment_dir(self) -> Path:
        return self.cache_dir / SEGMENT_DIRNAME

    def call_path(self, call_id: str) -> Path:
        return self.segment_dir / f"{call_id}.npy"

    def load_segment(self, row: CacheRow) -> np.ndarray:
        """조각 하나를 int16 배열로 읽는다 (memmap 이라 통화 전체를 올리지 않는다)."""
        data = np.load(self.call_path(row.call_id), mmap_mode="r")
        return np.asarray(data[row.offset : row.offset + row.length])

    def by_call(self) -> dict[str, list[CacheRow]]:
        groups: dict[str, list[CacheRow]] = {}
        for row in self.rows:
            groups.setdefault(row.call_id, []).append(row)
        return groups

    def save(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / INDEX_NAME
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_INDEX_FIELDS)
            for row in self.rows:
                writer.writerow(
                    (row.call_id, row.seg_idx, row.offset, row.length, row.gender or "")
                )
        return path

    @classmethod
    def load(cls, cache_dir: str | Path) -> "CacheIndex":
        cache_dir = Path(cache_dir)
        path = cache_dir / INDEX_NAME
        if not path.exists():
            return cls(cache_dir, [])

        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            for item in csv.DictReader(f):
                rows.append(
                    CacheRow(
                        call_id=item["call_id"],
                        seg_idx=int(item["seg_idx"]),
                        offset=int(item["offset"]),
                        length=int(item["length"]),
                        gender=item["gender"] or None,
                    )
                )
        return cls(cache_dir, rows)


def _build_one(task: tuple[str, str, str, int]) -> list[tuple]:
    """통화 1건을 캐시로 만든다. ProcessPoolExecutor 를 위해 모듈 최상위에 둔다."""
    json_path, audio_dir, cache_dir, min_segment_ms = task
    record = read_call(json_path)
    segments_meta = []

    utterances = caller_utterances(record)
    if not utterances:
        return []

    wav_path = Path(audio_dir) / f"{record.call_id}.wav"
    if not wav_path.exists():
        return []

    audio, sample_rate = sf.read(wav_path, dtype="int16", always_2d=False)
    if audio.ndim > 1:  # 방어적 처리 — 제공 데이터는 모두 mono 다
        audio = audio[:, 0]

    min_samples = max(1, int(min_segment_ms * sample_rate / 1000))
    chunks = []
    offset = 0
    for utt in utterances:
        start = max(0, int(utt.start_ms * sample_rate / 1000))
        end = min(len(audio), int(utt.end_ms * sample_rate / 1000))
        if end - start < min_samples:
            continue
        chunk = audio[start:end]
        chunks.append(chunk)
        segments_meta.append(
            (record.call_id, len(segments_meta), offset, len(chunk), record.gender or "")
        )
        offset += len(chunk)

    if not chunks:
        return []

    out_path = Path(cache_dir) / SEGMENT_DIRNAME / f"{record.call_id}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.concatenate(chunks).astype(np.int16))
    return segments_meta


def build_cache(
    label_dir: str | Path,
    audio_dir: str | Path,
    cache_dir: str | Path,
    min_segment_ms: int = 100,
    workers: int = 0,
    progress: bool = False,
) -> CacheIndex:
    """라벨/오디오 디렉터리를 훑어 신고자 조각 캐시를 만든다.

    이미 인덱스에 있는 통화는 건너뛰므로 중단 후 재실행해도 안전하다.
    workers > 1 이면 통화 단위로 병렬 처리한다.
    """
    cache_dir = Path(cache_dir)
    existing = CacheIndex.load(cache_dir)
    done = {row.call_id for row in existing.rows}

    pending = [
        (str(p), str(audio_dir), str(cache_dir), min_segment_ms)
        for p in sorted(Path(label_dir).glob("*.json"))
        if p.stem not in done
    ]

    produced: list[tuple] = []
    if pending:
        if workers and workers > 1:
            chunksize = max(1, len(pending) // (workers * 8))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = pool.map(_build_one, pending, chunksize=chunksize)
                produced = _drain(results, len(pending), progress)
        else:
            produced = _drain((_build_one(t) for t in pending), len(pending), progress)

    rows = existing.rows + [
        CacheRow(call_id=c, seg_idx=i, offset=o, length=n, gender=g or None)
        for c, i, o, n, g in produced
    ]
    rows.sort(key=lambda r: (r.call_id, r.seg_idx))

    index = CacheIndex(cache_dir, rows)
    index.save()
    return index


def _drain(results, total: int, progress: bool) -> list[tuple]:
    produced: list[tuple] = []
    for done, meta in enumerate(results, start=1):
        produced.extend(meta)
        if progress and (done % 500 == 0 or done == total):
            print(f"cached {done}/{total} calls, {len(produced)} segments", flush=True)
    return produced


def cache_size_bytes(cache_dir: str | Path) -> int:
    seg_dir = Path(cache_dir) / SEGMENT_DIRNAME
    if not seg_dir.exists():
        return 0
    return sum(f.stat().st_size for f in seg_dir.glob("*.npy"))


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)
