"""T8 학습 손실. 평가용 pos_weight 및 Calibration과 독립적이다."""
from __future__ import annotations

import torch
from torch import nn


ASL_CONFIG = {"gamma_pos": 0, "gamma_neg": 4, "clip": 0.05, "eps": 1e-8,
              "focusing_weight_gradient": "detach", "pos_weight_applied": False,
              "reduction": "mean", "compute_dtype": "float32"}


class AsymmetricLoss(nn.Module):
    """Alibaba-MIIL ASL 수식에 mean reduction과 FP32 연산을 적용한다.

    https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
    eps clamp 영역에서는 공식 구현과 마찬가지로 해당 로그의 기울기가 0이다.
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos, self.gamma_neg = gamma_pos, gamma_neg
        self.clip, self.eps = clip, eps

    def forward(self, logits, targets):
        # 외부 AMP 문맥에도 FP16의 eps 소실/log(0)을 방지한다.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            x, y = logits.float(), targets.float()
            p = x.sigmoid()
            q = (1 - p + self.clip).clamp(max=1)
            log_likelihood = y * p.clamp(min=self.eps).log()
            log_likelihood = log_likelihood + (1 - y) * q.clamp(min=self.eps).log()
            with torch.no_grad():
                pt = p * y + q * (1 - y)
                gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                weight = (1 - pt).pow(gamma)
            return -(log_likelihood * weight).mean()


def make_loss(name, pos_weight):
    """pos_weight 인자는 기존 BCE 전용이다."""
    if name == "weighted-bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "asl":
        return AsymmetricLoss()
    raise ValueError(f"알 수 없는 Loss: {name}")
