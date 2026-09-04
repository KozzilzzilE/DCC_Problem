"""Speech 갈래 — Wav2Vec2 파인튜닝.

입력은 16 kHz raw waveform 이다. 원본이 8 kHz 전화 음성이라 업샘플 후에도
4 kHz 이상 대역은 비어 있고, Wav2Vec2 의 사전학습 도메인(16 kHz 광대역)과
갭이 남는다. 이 갭 자체가 CNN 갈래와의 비교 분석에서 다룰 논점이다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, Wav2Vec2Model

from ..datasets import W2V2_SAMPLE_RATE

# 한국어 base 계열을 우선 시도하고, 받을 수 없으면 원본 base 로 폴백한다.
# 실제로 어떤 체크포인트가 쓰였는지는 항상 체크포인트 메타에 기록된다.
DEFAULT_CANDIDATES = (
    "kresnik/wav2vec2-base-korean",
    "Bingsu/wav2vec2-base-korean",
    "facebook/wav2vec2-base",
)


def resolve_checkpoint(candidates=DEFAULT_CANDIDATES) -> str:
    """받을 수 있는 첫 번째 체크포인트 이름을 돌려준다."""
    errors = []
    for name in candidates:
        try:
            AutoConfig.from_pretrained(name)
            return name
        except Exception as exc:  # 네트워크/404 등
            errors.append(f"{name}: {type(exc).__name__}")
    raise RuntimeError("no wav2vec2 checkpoint available -> " + "; ".join(errors))


class Wav2Vec2Gender(nn.Module):
    """(B, samples@16k) -> (B,) logit. 양수면 '여'(class 1)."""

    def __init__(
        self,
        model_name: str,
        freeze_feature_encoder: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_feature_encoder:
            # 하위 CNN 특징 추출기는 얼려 둔다. 8 GB VRAM 에서 메모리를 아끼고,
            # 조각 단위 이진 분류라는 작은 과제에 과적합하는 것도 막는다.
            self.backbone.feature_extractor._freeze_parameters()

        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Wav2Vec2 는 발화 단위로 정규화된 입력을 기대한다.
        wave = waveform.float()
        wave = (wave - wave.mean(dim=-1, keepdim=True)) / wave.std(dim=-1, keepdim=True).clamp_min(1e-5)

        hidden = self.backbone(wave).last_hidden_state  # (B, frames, hidden)
        return self.head(hidden.mean(dim=1)).squeeze(-1)

    @property
    def input_sample_rate(self) -> int:
        return W2V2_SAMPLE_RATE
