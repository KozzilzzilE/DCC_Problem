"""전화 음성 사전학습 갈래 — audeering wav2vec2-large-robust(6층) 파인튜닝.

지금까지 쓴 ResNet50 과 Wav2Vec2-base 는 둘 다 16 kHz 고음질로 사전학습됐고,
Validation 에서 같은 통화에서 막혔다. 이 백본은 Libri-Light 에 더해 Fisher
2,000 시간 + Switchboard 300 시간의 **전화 음성**으로 사전학습됐고, 전화 품질
aGender 로 성별 파인튜닝까지 거쳤다. zero-shot 만으로 우리 오답 70 통화 중 30 을
맞혀 오류 프로파일이 다름을 확인했다 (reports/audeering_diag.json).

원 모델의 3-way(female/male/child) 헤드는 버리고 이진 헤드를 새로 단다. 문헌상
성별 정보는 여러 층에 분산돼 있고 중간층이 가장 잘 분리되므로, 마지막 층만 쓰는
대신 층별 학습 가중치로 hidden state 를 합친다 (SUPERB 방식).

라이선스: CC-BY-NC-SA-4.0 (비상업). 제출 문서에 명시할 것.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from .w2v2 import build_backbone

from ..datasets import W2V2_SAMPLE_RATE

DEFAULT_NAME = "audeering/wav2vec2-large-robust-6-ft-age-gender"


class AudeeringGender(nn.Module):
    """(B, samples@16k) -> (B,) logit. 양수면 '여'(class 1)."""

    def __init__(
        self,
        model_name: str = DEFAULT_NAME,
        freeze_feature_encoder: bool = True,
        dropout: float = 0.1,
        mask_time_prob: float = 0.05,
        layer_weighted: bool = True,
        hf_config: dict | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = build_backbone(model_name, hf_config)

        # SpecAugment 에 해당하는 latent 시간 마스킹. 학습 모드에서만 적용된다.
        # 두 갈래 모두 epoch 3 에서 과적합했으므로 정규화를 넣는다.
        self.backbone.config.apply_spec_augment = mask_time_prob > 0
        self.backbone.config.mask_time_prob = mask_time_prob
        self.backbone.config.mask_time_length = 10
        self.backbone.config.mask_feature_prob = 0.0

        if freeze_feature_encoder:
            self.backbone.feature_extractor._freeze_parameters()

        # hidden_states 는 임베딩 출력 + 각 층 출력 = num_hidden_layers + 1 개
        n_states = self.backbone.config.num_hidden_layers + 1
        self.layer_weighted = layer_weighted
        self.layer_logits = nn.Parameter(torch.zeros(n_states))

        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        wave = waveform.float()
        wave = (wave - wave.mean(dim=-1, keepdim=True)) / wave.std(dim=-1, keepdim=True).clamp_min(1e-5)

        out = self.backbone(wave, output_hidden_states=self.layer_weighted)
        if self.layer_weighted:
            # (L+1, B, T, H) 로 stack 하면 배치 32 에서 수백 MB 짜리 임시 텐서가 생겨
            # 8 GB VRAM 을 압박한다. 같은 가중합을 층별로 누적해 메모리를 1/7 로 줄인다.
            weights = torch.softmax(self.layer_logits, dim=0)
            hidden = None
            for w, state in zip(weights, out.hidden_states):
                hidden = state * w if hidden is None else hidden + state * w
        else:
            hidden = out.last_hidden_state

        return self.head(hidden.mean(dim=1)).squeeze(-1)

    def layer_weights(self) -> list[float]:
        """진단용 — 학습 후 어느 층이 쓰였는지."""
        return torch.softmax(self.layer_logits.detach(), dim=0).tolist()

    @property
    def input_sample_rate(self) -> int:
        return W2V2_SAMPLE_RATE
