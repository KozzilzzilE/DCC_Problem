"""캐시 위의 torch Dataset.

두 모델 갈래가 같은 캐시·같은 크롭 규칙을 쓰고, 갈래별 차이는 마지막
리샘플 단계뿐이다 (Wav2Vec2 는 16 kHz 입력을 요구). 따라서 두 갈래의 정확도
차이는 입력 데이터가 아니라 모델에서 나온다.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .aggregate import gender_to_target
from .cache import CacheIndex, CacheRow
from .config import FeatureConfig

INT16_SCALE = 32768.0
W2V2_SAMPLE_RATE = 16000
# raw waveform 을 16 kHz 로 받는 갈래. 캐시(8 kHz)에서 꺼낼 때 업샘플한다.
RESAMPLE_BRANCHES = frozenset({"w2v2", "audeering"})
_MEMMAP_CACHE_SIZE = 256


@dataclass(frozen=True, slots=True)
class Sample:
    row: CacheRow
    target: int


def samples_from_rows(rows: list[CacheRow], require_label: bool = True) -> list[Sample]:
    """라벨이 있는 조각만 학습 샘플로 만든다."""
    out = []
    for row in rows:
        target = gender_to_target(row.gender)
        if target is None:
            if require_label:
                continue
            target = -1
        out.append(Sample(row=row, target=target))
    return out


def split_calls(
    call_ids: list[str], dev_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    """통화 ID 기준 분할.

    조각이 아니라 통화 단위로 나눈다. 같은 통화의 조각이 train 과 dev 에
    나뉘어 들어가면 화자가 양쪽에 존재해 dev 점수가 부풀려진다.
    """
    ordered = sorted(call_ids)
    rng = np.random.RandomState(seed)
    shuffled = list(rng.permutation(ordered))
    n_dev = max(1, int(round(len(shuffled) * dev_fraction)))
    return set(shuffled[n_dev:]), set(shuffled[:n_dev])


class _SegmentReader:
    """열려 있는 memmap 을 소수만 유지하며 조각을 읽는다."""

    def __init__(self, index: CacheIndex):
        self.index = index
        self._open: OrderedDict[str, np.ndarray] = OrderedDict()

    def read(self, row: CacheRow) -> np.ndarray:
        data = self._open.get(row.call_id)
        if data is None:
            data = np.load(self.index.call_path(row.call_id), mmap_mode="r")
            self._open[row.call_id] = data
            if len(self._open) > _MEMMAP_CACHE_SIZE:
                self._open.popitem(last=False)
        else:
            self._open.move_to_end(row.call_id)
        return np.asarray(data[row.offset : row.offset + row.length])


def crop_or_pad(segment: np.ndarray, target_len: int, start: int | None) -> np.ndarray:
    """고정 길이로 맞춘다. 짧으면 뒤를 0 으로 채우고, 길면 start 에서 자른다."""
    if len(segment) < target_len:
        out = np.zeros(target_len, dtype=segment.dtype)
        out[: len(segment)] = segment
        return out
    if start is None:
        start = (len(segment) - target_len) // 2
    return segment[start : start + target_len]


def to_waveform(segment: np.ndarray, branch: str) -> np.ndarray:
    """int16 -> float32 [-1, 1]. w2v2 갈래는 8 kHz -> 16 kHz 로 올린다."""
    wave = segment.astype(np.float32) / INT16_SCALE
    if branch in RESAMPLE_BRANCHES:
        # 폴리페이즈 FIR 업샘플. 원본에 없던 4 kHz 이상 대역은 비어 있으므로
        # 사전학습 도메인과의 갭이 남는다 (분석 시 명시).
        wave = resample_poly(wave, 2, 1).astype(np.float32)
    return wave


class SegmentWindowDataset(Dataset):
    """신고자 조각 하나당 고정 길이 창 하나."""

    def __init__(
        self,
        index: CacheIndex,
        samples: list[Sample],
        cfg: FeatureConfig,
        branch: str = "resnet",
        train: bool = True,
        seed: int = 0,
    ):
        self.index = index
        self.samples = samples
        self.cfg = cfg
        self.branch = branch
        self.train = train
        self.seed = seed
        self._reader: _SegmentReader | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def _read(self, row: CacheRow) -> np.ndarray:
        # DataLoader 워커마다 독립된 memmap 을 갖도록 지연 생성한다.
        if self._reader is None:
            self._reader = _SegmentReader(self.index)
        return self._reader.read(row)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        segment = self._read(sample.row)
        target_len = self.cfg.window_samples

        if self.train and len(segment) > target_len:
            rng = np.random.RandomState((self.seed * 1_000_003 + idx) % (2**31))
            start = int(rng.randint(0, len(segment) - target_len + 1))
        else:
            start = None

        wave = to_waveform(crop_or_pad(segment, target_len, start), self.branch)
        return torch.from_numpy(wave), torch.tensor(float(sample.target))


class SlidingWindowDataset(Dataset):
    """긴 조각을 겹치는 창 여러 개로 펼친다 (추론용).

    출제 PDF 힌트대로 짧은 조각은 padding, 긴 조각은 stride 를 준 sliding
    window 로 다룬다. 각 창의 확률을 조각 단위로 평균한 뒤 다시 통화 단위로
    평균한다.
    """

    def __init__(
        self,
        index: CacheIndex,
        samples: list[Sample],
        cfg: FeatureConfig,
        branch: str = "resnet",
        stride_ratio: float = 0.5,
    ):
        from .features import sliding_windows

        self.index = index
        self.samples = samples
        self.cfg = cfg
        self.branch = branch
        self._reader: _SegmentReader | None = None

        window = cfg.window_samples
        stride = max(1, int(window * stride_ratio))
        self.items: list[tuple[int, int]] = [
            (i, start)
            for i, sample in enumerate(samples)
            for start, _ in sliding_windows(sample.row.length, window, stride)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sample_idx, start = self.items[idx]
        if self._reader is None:
            self._reader = _SegmentReader(self.index)

        segment = self._reader.read(self.samples[sample_idx].row)
        window = self.cfg.window_samples
        if len(segment) < window:
            chunk = crop_or_pad(segment, window, None)
        else:
            chunk = segment[start : start + window]

        return torch.from_numpy(to_waveform(chunk, self.branch)), sample_idx
