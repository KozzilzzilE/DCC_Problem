"""Mission 1 설정 값.

모든 설정은 frozen dataclass 다. 체크포인트에 그대로 직렬화해 두면 추론 시
학습과 동일한 전처리가 자동으로 복원되므로 설정 불일치가 생기지 않는다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class FeatureConfig:
    """log-Mel / MFCC 프런트엔드 설정.

    기본값은 8 kHz 전화 음성 기준이다. n_fft=1024 는 128 ms 창으로, 성별 판별의
    주 단서인 F0(남 85~180 Hz / 여 165~255 Hz)를 7.8 Hz 해상도로 분해한다.
    """

    sample_rate: int = 8000
    n_fft: int = 1024
    hop_length: int = 128
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 4000.0
    kind: str = "logmel"  # "logmel" | "mfcc"
    n_mfcc: int = 40
    window_frames: int = 192

    def __post_init__(self) -> None:
        if self.kind not in ("logmel", "mfcc"):
            raise ValueError(f"kind must be 'logmel' or 'mfcc', got {self.kind!r}")
        if self.fmax > self.sample_rate / 2:
            raise ValueError(f"fmax {self.fmax} exceeds Nyquist {self.sample_rate / 2}")
        if self.kind == "mfcc" and self.n_mfcc > self.n_mels:
            raise ValueError(f"n_mfcc {self.n_mfcc} > n_mels {self.n_mels}")

    @property
    def window_samples(self) -> int:
        """window_frames 개의 STFT 프레임을 정확히 만드는 샘플 수 (center=True 기준)."""
        return (self.window_frames - 1) * self.hop_length

    @property
    def n_channels(self) -> int:
        """모델 입력의 주파수 축 크기."""
        return self.n_mfcc if self.kind == "mfcc" else self.n_mels

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(frozen=True)
class TrainConfig:
    branch: str = "resnet"  # "resnet" | "w2v2" | "audeering"
    epochs: int = 8
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    dev_fraction: float = 0.1
    seed: int = 1234
    amp: bool = True

    def __post_init__(self) -> None:
        if self.branch not in ("resnet", "w2v2", "audeering"):
            raise ValueError(f"branch must be resnet/w2v2/audeering, got {self.branch!r}")
        if not 0.0 < self.dev_fraction < 1.0:
            raise ValueError(f"dev_fraction must be in (0, 1), got {self.dev_fraction}")

    def to_dict(self) -> dict:
        return asdict(self)
