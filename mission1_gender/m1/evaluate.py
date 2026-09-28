"""조각 확률 예측과 통화 단위 평가.

공식 지표는 통화 단위 Accuracy 다. 조각 단위 정확도는 진단용으로만 함께
보고한다 (둘의 차이가 soft voting 이 얼마나 벌어주는지를 보여준다).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .aggregate import FEMALE, GENDER_OUTPUT, call_label, call_probability, gender_to_target
from .cache import CacheIndex
from .config import FeatureConfig
from .datasets import Sample, SegmentWindowDataset, SlidingWindowDataset


# 이 간격마다 torch.cuda.empty_cache(). 긴 루프의 할당자 단편화 방지.
EMPTY_CACHE_EVERY = 200


@dataclass
class CallMetrics:
    call_accuracy: float
    segment_accuracy: float
    n_calls: int
    n_segments: int
    confusion: dict[str, int]
    per_gender_accuracy: dict[str, float]

    def summary(self) -> str:
        return (
            f"call acc {self.call_accuracy:.4f} ({self.n_calls} calls) | "
            f"seg acc {self.segment_accuracy:.4f} ({self.n_segments} segs) | "
            f"남 {self.per_gender_accuracy.get('남', float('nan')):.4f} "
            f"여 {self.per_gender_accuracy.get('여', float('nan')):.4f}"
        )


def suggested_workers(branch: str) -> int:
    """갈래별 권장 DataLoader 워커 수.

    워커 수는 항목당 CPU 작업량에 맞춰야 한다.
    - resnet : 캐시에서 메모리 복사만 하므로 CPU 작업이 없다. Windows 의 spawn
      워커를 쓰면 배치마다 25MB 를 파이프로 넘기느라 오히려 5.6배 느려진다.
    - w2v2   : 창마다 resample_poly 로 8k -> 16k 업샘플을 한다. 실제 CPU 작업이
      있어 워커가 필요하다.
    """
    return 4 if branch in ("w2v2", "audeering") else 0


@torch.no_grad()
def predict_segment_probs(
    model: torch.nn.Module,
    index: CacheIndex,
    samples: list[Sample],
    cfg: FeatureConfig,
    branch: str,
    device: torch.device,
    batch_size: int = 128,
    mode: str = "center",
    num_workers: int = 0,
    amp: bool = True,
) -> np.ndarray:
    """각 조각의 P(여) 를 돌려준다 (samples 와 같은 순서, 길이)."""
    model.eval()

    if mode == "sliding":
        dataset = SlidingWindowDataset(index, samples, cfg, branch=branch)
    elif mode == "center":
        dataset = SegmentWindowDataset(index, samples, cfg, branch=branch, train=False)
    else:
        raise ValueError(f"mode must be 'center' or 'sliding', got {mode!r}")

    # 추론에서는 num_workers=0 이 가장 빠르다. Validation 캐시는 1.97GB 라 페이지
    # 캐시에 들어가고 접근도 순차적이라 I/O 가 사실상 공짜인데(81,379 창 로딩 7.4초),
    # Windows 의 spawn 워커를 쓰면 배치마다 25MB 를 파이프로 넘기느라 122초가 된다.
    # 전체 Validation 추론이 23초 vs 130초로 갈린다.
    #
    # 학습은 반대다. 462,190 조각을 셔플해 15.67GB 캐시에 랜덤 접근하므로 I/O 가
    # 병목이고, 워커가 1 epoch 을 2,056초에서 326초로 줄인다. train.py 의 기본값을
    # 여기에 맞춰 낮추면 안 된다.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total = np.zeros(len(samples), dtype=np.float64)
    counts = np.zeros(len(samples), dtype=np.int64)
    cursor = 0

    for step, (waveform, tag) in enumerate(loader, start=1):
        if device.type == "cuda" and step % EMPTY_CACHE_EVERY == 0:
            # Windows WDDM 은 VRAM 초과를 OOM 대신 시스템 RAM 페이징으로 숨겨
            # 5~6배 느려진다 (교사 확률 추출 124분 -> 17분). 단편화된 캐시
            # 블록을 주기적으로 돌려준다. 비용은 무시할 수준.
            torch.cuda.empty_cache()
        waveform = waveform.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(waveform)
        probs = torch.sigmoid(logits.float()).cpu().numpy()

        if mode == "sliding":
            np.add.at(total, tag.numpy(), probs)
            np.add.at(counts, tag.numpy(), 1)
        else:
            n = len(probs)
            total[cursor : cursor + n] = probs
            counts[cursor : cursor + n] = 1
            cursor += n

    if counts.min() == 0:
        raise RuntimeError("some segments received no prediction")
    return total / counts


def call_probabilities(samples: list[Sample], segment_probs: np.ndarray) -> dict[str, float]:
    """조각 확률을 통화 단위로 평균한다 (soft voting)."""
    buckets: dict[str, list[float]] = {}
    for sample, prob in zip(samples, segment_probs):
        buckets.setdefault(sample.row.call_id, []).append(float(prob))
    return {cid: call_probability(np.array(v)) for cid, v in buckets.items()}


def score(
    samples: list[Sample],
    segment_probs: np.ndarray,
    truth: dict[str, str],
    threshold: float = 0.5,
) -> CallMetrics:
    """truth: call_id -> 'M' | 'F'. threshold 는 대회 규정상 0.5 고정 (보정값은 연구용 분석에서만)."""
    call_probs = call_probabilities(samples, segment_probs)

    confusion = {"남->남": 0, "남->여": 0, "여->남": 0, "여->여": 0}
    correct = 0
    for cid, prob in call_probs.items():
        gold = GENDER_OUTPUT[gender_to_target(truth[cid])]
        pred = call_label(prob, threshold)
        confusion[f"{gold}->{pred}"] += 1
        correct += gold == pred

    per_gender = {}
    for gold in ("남", "여"):
        hit = confusion[f"{gold}->{gold}"]
        total = hit + confusion[f"{gold}->{'여' if gold == '남' else '남'}"]
        per_gender[gold] = hit / total if total else float("nan")

    seg_pred = (segment_probs >= 0.5).astype(int)
    seg_gold = np.array([s.target for s in samples])

    return CallMetrics(
        call_accuracy=correct / max(1, len(call_probs)),
        segment_accuracy=float((seg_pred == seg_gold).mean()) if len(samples) else float("nan"),
        n_calls=len(call_probs),
        n_segments=len(samples),
        confusion=confusion,
        per_gender_accuracy=per_gender,
    )


def truth_from_samples(samples: list[Sample]) -> dict[str, str]:
    return {s.row.call_id: s.row.gender for s in samples if s.row.gender}


def majority_baseline(truth: dict[str, str]) -> float:
    """다수결 기준선. 모델이 이걸 못 넘으면 아무것도 배우지 못한 것이다."""
    if not truth:
        return float("nan")
    females = sum(1 for g in truth.values() if gender_to_target(g) == FEMALE)
    return max(females, len(truth) - females) / len(truth)
