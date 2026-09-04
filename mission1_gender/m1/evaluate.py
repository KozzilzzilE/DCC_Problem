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

    for waveform, tag in loader:
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
) -> CallMetrics:
    """truth: call_id -> 'M' | 'F'."""
    call_probs = call_probabilities(samples, segment_probs)

    confusion = {"남->남": 0, "남->여": 0, "여->남": 0, "여->여": 0}
    correct = 0
    for cid, prob in call_probs.items():
        gold = GENDER_OUTPUT[gender_to_target(truth[cid])]
        pred = call_label(prob)
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
