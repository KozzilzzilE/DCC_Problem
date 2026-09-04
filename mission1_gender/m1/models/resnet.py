"""Vision 갈래 — log-Mel/MFCC 스펙트로그램을 2D 이미지로 보고 ResNet50 으로 분류.

피처 프런트엔드를 모델 안에 넣어 두면 추론 시 waveform 만 주면 되고, 학습과
추론의 전처리가 어긋날 여지가 없다. 프런트엔드는 GPU 에서 배치로 돈다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from ..config import FeatureConfig
from ..features import MelFrontend


class ResNetGender(nn.Module):
    """(B, samples) -> (B,) logit. 양수면 '여'(class 1)."""

    def __init__(self, cfg: FeatureConfig, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.cfg = cfg
        self.frontend = MelFrontend(cfg)

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)

        # 스펙트로그램은 1채널이다. 사전학습된 RGB conv1 을 채널 평균으로 접어
        # 넣으면 ImageNet 이 학습한 엣지/텍스처 필터를 그대로 물려받는다.
        old = self.backbone.conv1
        new = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        self.backbone.conv1 = new

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # STFT 는 복소수 연산이라 AMP 와 궁합이 나쁘다. 프런트엔드만 fp32 로 고정.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            feat = self.frontend(waveform.float())
        return self.backbone(feat.to(self.backbone.conv1.weight.dtype)).squeeze(-1)

    @property
    def input_sample_rate(self) -> int:
        return self.cfg.sample_rate
