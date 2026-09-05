"""모델 생성과 체크포인트 입출력.

체크포인트에는 가중치뿐 아니라 FeatureConfig 와 갈래 이름을 함께 저장한다.
추론 시 --ckpt_path 하나만 주면 학습과 동일한 전처리가 복원되므로, 설정
불일치로 조용히 성능이 무너지는 사고가 구조적으로 막힌다.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import FeatureConfig

CHECKPOINT_VERSION = 1

# 조각 확률을 통화 단위로 평균했을 때의 기본 결정 경계.
# m1.calibrate 로 dev 에서 보정한 값이 체크포인트에 있으면 그쪽이 우선한다.
DEFAULT_THRESHOLD = 0.5


def build_model(branch: str, cfg: FeatureConfig, **kwargs) -> nn.Module:
    if branch == "resnet":
        from .resnet import ResNetGender

        return ResNetGender(cfg, **kwargs)
    if branch == "w2v2":
        from .w2v2 import Wav2Vec2Gender, resolve_checkpoint

        model_name = kwargs.pop("model_name", None) or resolve_checkpoint()
        return Wav2Vec2Gender(model_name, **kwargs)
    raise ValueError(f"unknown branch {branch!r}")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    branch: str,
    cfg: FeatureConfig,
    metrics: dict | None = None,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": CHECKPOINT_VERSION,
        "branch": branch,
        "feature_config": cfg.to_dict(),
        "state_dict": model.state_dict(),
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if branch == "w2v2":
        payload["extra"]["model_name"] = getattr(model, "model_name", None)

    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, str, FeatureConfig, dict]:
    """(model, branch, feature_config, payload) 를 돌려준다. 모델은 eval 상태다."""
    payload = torch.load(path, map_location=device, weights_only=False)

    if "branch" not in payload or "feature_config" not in payload:
        raise ValueError(
            f"{path} 는 m1 체크포인트가 아닙니다 (branch/feature_config 누락)."
        )

    branch = payload["branch"]
    cfg = FeatureConfig.from_dict(payload["feature_config"])

    kwargs = {}
    if branch == "resnet":
        # 저장된 가중치를 덮어쓸 것이므로 ImageNet 가중치를 새로 받을 필요가 없다.
        kwargs["pretrained"] = False
    elif branch == "w2v2":
        kwargs["model_name"] = payload.get("extra", {}).get("model_name")

    model = build_model(branch, cfg, **kwargs)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, branch, cfg, payload


def checkpoint_threshold(payload: dict) -> float:
    """체크포인트에 보정된 임계값이 있으면 그 값, 없으면 0.5."""
    value = (payload or {}).get("extra", {}).get("decision_threshold")
    if value is None:
        return DEFAULT_THRESHOLD
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"decision_threshold 는 (0, 1) 이어야 합니다: {value}")
    return value


def write_threshold(path: str | Path, threshold: float, dev_accuracy: float | None = None) -> Path:
    """학습을 다시 하지 않고 체크포인트에 보정된 임계값만 기록한다."""
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold 는 (0, 1) 이어야 합니다: {threshold}")

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.setdefault("extra", {})["decision_threshold"] = float(threshold)
    if dev_accuracy is not None:
        payload["extra"]["decision_threshold_dev_accuracy"] = float(dev_accuracy)
    torch.save(payload, path)
    return path
