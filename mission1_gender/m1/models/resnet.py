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


def spec_augment(feat: torch.Tensor, freq_mask: int, time_mask: int, n_masks: int) -> torch.Tensor:
    """(B, 1, F, T) 피처에 주파수/시간 띠를 0 으로 가린다.

    피처는 샘플별로 표준화돼 있어 0 이 곧 평균값이다. 샘플마다 독립적으로
    n_masks 개씩, 폭은 [0, mask] 에서 균등 추출 (SpecAugment 원 논문 방식).
    """
    b, _, n_freq, n_time = feat.shape
    device = feat.device
    keep = torch.ones(b, 1, n_freq, n_time, device=device, dtype=feat.dtype)

    if freq_mask > 0:
        f_idx = torch.arange(n_freq, device=device).view(1, -1)
        for _ in range(n_masks):
            width = torch.randint(0, freq_mask + 1, (b, 1), device=device)
            start = (torch.rand(b, 1, device=device) * (n_freq - width + 1)).long()
            band = (f_idx >= start) & (f_idx < start + width)          # (B, F)
            keep = keep * (~band).to(feat.dtype).view(b, 1, n_freq, 1)

    if time_mask > 0:
        t_idx = torch.arange(n_time, device=device).view(1, -1)
        for _ in range(n_masks):
            width = torch.randint(0, time_mask + 1, (b, 1), device=device)
            start = (torch.rand(b, 1, device=device) * (n_time - width + 1)).long()
            band = (t_idx >= start) & (t_idx < start + width)          # (B, T)
            keep = keep * (~band).to(feat.dtype).view(b, 1, 1, n_time)

    return feat * keep


class ResNetGender(nn.Module):
    """(B, samples) -> (B,) logit. 양수면 '여'(class 1)."""

    def __init__(
        self,
        cfg: FeatureConfig,
        pretrained: bool = True,
        dropout: float = 0.2,
        freq_mask: int = 0,
        time_mask: int = 0,
        n_masks: int = 2,
    ):
        super().__init__()
        self.cfg = cfg
        # SpecAugment. 학습 모드에서만 적용되고 추론에는 영향이 없다.
        # ResNet 은 학습 조각 정확도가 0.86 -> 0.95 로 오르는 동안 dev 는 0.88 에
        # 멈추는 전형적 과적합이었는데 증강이 하나도 없었다.
        self.freq_mask = int(freq_mask)
        self.time_mask = int(time_mask)
        self.n_masks = int(n_masks)
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
        if self.training and (self.freq_mask or self.time_mask):
            feat = spec_augment(feat, self.freq_mask, self.time_mask, self.n_masks)
        return self.backbone(feat.to(self.backbone.conv1.weight.dtype)).squeeze(-1)

    @property
    def input_sample_rate(self) -> int:
        return self.cfg.sample_rate
