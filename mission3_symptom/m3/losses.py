"""Mission 3 multi-label 학습 loss 구현과 선택."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LabelDependencyLoss(nn.Module):
    """BCE에 Label 간의 동시 발생성(Co-occurrence) 제약 조건을 추가한 Loss.
    
    자주 동시 발생(Co-occurrence)하는 증상들이 서로 비슷한 예측 확률을 갖도록
    차이의 제곱에 Co-occurrence 가중치를 곱해 페널티(Penalty)로 부여합니다.
    """

    def __init__(
        self,
        co_occurrence_matrix: torch.Tensor,
        alpha: float = 0.1,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # (num_classes, num_classes) 형태의 정규화된 동시 발생 행렬
        self.register_buffer("co_occurrence", co_occurrence_matrix.float())
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss = self.bce(logits, targets)
        
        # (Batch, NumClasses)
        probs = torch.sigmoid(logits)
        
        # (Batch, NumClasses, NumClasses): 각 샘플 내 클래스 쌍의 예측 확률 차이 제곱
        prob_diff_sq = (probs.unsqueeze(2) - probs.unsqueeze(1)) ** 2
        
        # 동시 발생 빈도가 높은 쌍일수록 확률 차이가 작도록 유도
        # batch 및 클래스 차원에 대해 평균을 구함
        dependency_penalty = (prob_diff_sq * self.co_occurrence).mean()
        
        return base_loss + self.alpha * dependency_penalty


class PairwiseRankingLoss(nn.Module):
    """BCE에 양성/음성 증상 쌍 ranking 제약을 더한 Loss.

    같은 샘플에서 정답이 1인 증상의 logit이 0인 증상보다 커지도록
    ``softplus(s_neg - s_pos)`` 평균을 페널티로 부여한다.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                "Pairwise logits와 targets shape이 같아야 합니다: "
                f"logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
            )

        base_loss = self.bce(logits, targets)
        logits_float = logits.float()
        targets_float = targets.to(dtype=logits_float.dtype)

        positive_mask = targets_float.unsqueeze(2)
        negative_mask = (1.0 - targets_float).unsqueeze(1)
        pair_mask = positive_mask * negative_mask
        score_diff = logits_float.unsqueeze(1) - logits_float.unsqueeze(2)
        pair_loss = torch.nn.functional.softplus(score_diff)
        pair_count = pair_mask.sum().clamp(min=1.0)
        ranking_penalty = (pair_loss * pair_mask).sum() / pair_count

        return base_loss + self.alpha * ranking_penalty


class AsymmetricLoss(nn.Module):
    """Sigmoid 기반 multi-label Asymmetric Loss.

    각 label을 독립적인 binary task로 다루며 easy negative의 영향을 positive보다
    강하게 줄인다. ``reduction='mean'``은 기존 BCEWithLogitsLoss의 기본 reduction과
    loss scale을 맞추기 위한 이번 실험 설정이다.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
        disable_focal_loss_grad: bool = True,
    ) -> None:
        super().__init__()
        if gamma_neg < 0 or gamma_pos < 0:
            raise ValueError("gamma_neg와 gamma_pos는 0 이상이어야 합니다.")
        if not 0.0 <= clip < 1.0:
            raise ValueError("clip은 0 이상 1 미만이어야 합니다.")
        if not 0.0 < eps < 1.0:
            raise ValueError("eps는 0 초과 1 미만이어야 합니다.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction은 'none', 'mean', 'sum' 중 하나여야 합니다.")

        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction
        self.disable_focal_loss_grad = bool(disable_focal_loss_grad)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                "ASL logits와 targets shape이 같아야 합니다: "
                f"logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
            )

        # AMP에서도 sigmoid/log 계산은 float32로 수행해 log(0)과 underflow를 피한다.
        logits_float = logits.float()
        targets_float = targets.to(dtype=logits_float.dtype)
        positive_probability = torch.sigmoid(logits_float)
        negative_probability = 1.0 - positive_probability

        if self.clip > 0.0:
            negative_probability = (negative_probability + self.clip).clamp(max=1.0)

        positive_log_loss = targets_float * torch.log(
            positive_probability.clamp(min=self.eps)
        )
        negative_log_loss = (1.0 - targets_float) * torch.log(
            negative_probability.clamp(min=self.eps)
        )
        loss = positive_log_loss + negative_log_loss

        if self.gamma_neg > 0.0 or self.gamma_pos > 0.0:
            def calculate_focal_weight() -> torch.Tensor:
                target_probability = (
                    positive_probability * targets_float
                    + negative_probability * (1.0 - targets_float)
                )
                one_sided_gamma = (
                    self.gamma_pos * targets_float
                    + self.gamma_neg * (1.0 - targets_float)
                )
                return (1.0 - target_probability).pow(one_sided_gamma)

            if self.disable_focal_loss_grad:
                with torch.no_grad():
                    focal_weight = calculate_focal_weight()
            else:
                focal_weight = calculate_focal_weight()
            loss = loss * focal_weight

        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    loss_type: str,
    *,
    pos_weight: Optional[torch.Tensor] = None,
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 1.0,
    asl_clip: float = 0.05,
    asl_eps: float = 1e-8,
    asl_reduction: str = "mean",
    asl_disable_focal_loss_grad: bool = True,
    co_occurrence_matrix: Optional[torch.Tensor] = None,
    dependency_alpha: float = 0.1,
    pairwise_alpha: float = 0.1,
) -> nn.Module:
    """Config 값으로 BCE, ASL, Dependency 또는 Pairwise Ranking Loss를 생성한다."""
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_type == "asl":
        if pos_weight is not None:
            raise ValueError("ASL과 pos_weight는 동시에 사용할 수 없습니다.")
        return AsymmetricLoss(
            gamma_neg=asl_gamma_neg,
            gamma_pos=asl_gamma_pos,
            clip=asl_clip,
            eps=asl_eps,
            reduction=asl_reduction,
            disable_focal_loss_grad=asl_disable_focal_loss_grad,
        )
    if loss_type == "dependency":
        if co_occurrence_matrix is None:
            raise ValueError("Dependency loss를 사용하려면 co_occurrence_matrix가 필수입니다.")
        return LabelDependencyLoss(
            co_occurrence_matrix=co_occurrence_matrix,
            alpha=dependency_alpha,
            pos_weight=pos_weight,
        )
    if loss_type == "pairwise":
        return PairwiseRankingLoss(
            alpha=pairwise_alpha,
            pos_weight=pos_weight,
        )
    raise ValueError(f"지원하지 않는 loss_type입니다: {loss_type}")
