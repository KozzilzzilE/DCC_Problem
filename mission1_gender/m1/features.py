"""순수 torch log-Mel / MFCC 프런트엔드.

설치된 torch 2.13+cu130 에 맞는 torchaudio 빌드가 없어(cu130 채널 최대 2.11)
STFT 와 mel 필터뱅크를 직접 구현한다. 부수 효과로 피처 계산이 GPU 에서 배치
단위로 돌아, 두 모델 갈래의 속도 비교가 I/O 가 아닌 모델 차이를 반영하게 된다.

수치는 librosa 와 일치하도록 맞췄고 tests/test_features.py 에서 대조 검증한다.

규칙 준수: 이 모듈의 어떤 함수도 label 을 인자로 받지 않는다. 따라서
"sample instance 마다 label 에 따라 상이한 전처리" 가 구조적으로 불가능하다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import FeatureConfig

_LOG_EPS = 1e-10


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    """Slaney mel scale (librosa htk=False)."""
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    above = freq >= min_log_hz
    mels[above] = min_log_mel + np.log(freq[above] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    above = mels >= min_log_mel
    freqs[above] = min_log_hz * np.exp(logstep * (mels[above] - min_log_mel))
    return freqs


def mel_filterbank(cfg: FeatureConfig) -> torch.Tensor:
    """(n_mels, n_fft // 2 + 1) Slaney 정규화 mel 필터뱅크."""
    n_freqs = cfg.n_fft // 2 + 1
    fft_freqs = np.linspace(0, cfg.sample_rate / 2, n_freqs)

    mel_edges = np.linspace(
        _hz_to_mel(np.array([float(cfg.fmin)]))[0],
        _hz_to_mel(np.array([float(cfg.fmax)]))[0],
        cfg.n_mels + 2,
    )
    hz_edges = _mel_to_hz(mel_edges)

    fdiff = np.diff(hz_edges)
    ramps = hz_edges[:, None] - fft_freqs[None, :]

    weights = np.zeros((cfg.n_mels, n_freqs), dtype=np.float64)
    for i in range(cfg.n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney 정규화: 각 필터의 면적을 동일하게 맞춘다
    enorm = 2.0 / (hz_edges[2 : cfg.n_mels + 2] - hz_edges[: cfg.n_mels])
    weights *= enorm[:, None]

    return torch.from_numpy(weights).float()


def dct_matrix(n_out: int, n_in: int) -> torch.Tensor:
    """orthonormal DCT-II 행렬 (scipy dct(type=2, norm='ortho') 와 동일)."""
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    basis = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    basis *= np.sqrt(2.0 / n_in)
    if n_out > 0:
        basis[0] *= np.sqrt(0.5)
    return torch.from_numpy(basis).float()


def sliding_windows(total: int, window: int, stride: int) -> list[tuple[int, int]]:
    """[0, total) 를 덮는 고정 길이 창 목록.

    신호가 창보다 짧으면 창 하나만 돌려주고 길이 보정은 호출부(제로 패딩)에
    맡긴다. 길면 마지막 창을 끝에 붙여 꼬리가 잘리지 않게 한다.
    """
    if total <= window:
        return [(0, window)]

    starts = list(range(0, total - window + 1, stride))
    windows = [(s, s + window) for s in starts]
    if windows[-1][1] != total:
        windows.append((total - window, total))
    return windows


class MelFrontend(nn.Module):
    """waveform -> 정규화된 2D 피처. (B, T) -> (B, 1, n_channels, window_frames)"""

    def __init__(self, cfg: FeatureConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("fb", mel_filterbank(cfg), persistent=False)
        self.register_buffer("window", torch.hann_window(cfg.n_fft), persistent=False)
        if cfg.kind == "mfcc":
            self.register_buffer("dct", dct_matrix(cfg.n_mfcc, cfg.n_mels), persistent=False)

    def mel_power(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, T) -> (B, n_mels, frames) mel 파워 스펙트로그램."""
        spec = torch.stft(
            waveform,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.n_fft,
            window=self.window,
            center=True,
            pad_mode="constant",
            onesided=True,
            normalized=False,
            return_complex=True,
        )
        power = spec.real.square() + spec.imag.square()
        return torch.matmul(self.fb, power)

    def mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, T) -> (B, n_mfcc, frames). librosa.feature.mfcc 와 같은 정의."""
        mel_db = _power_to_db(self.mel_power(waveform))
        return torch.matmul(self.dct, mel_db)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        target = self.cfg.window_samples
        if waveform.shape[-1] < target:
            waveform = nn.functional.pad(waveform, (0, target - waveform.shape[-1]))

        if self.cfg.kind == "mfcc":
            feat = self.mfcc(waveform)
        else:
            feat = torch.log(self.mel_power(waveform).clamp_min(_LOG_EPS))

        feat = _standardise_per_sample(feat)
        return feat.unsqueeze(1)


def _power_to_db(power: torch.Tensor, top_db: float = 80.0) -> torch.Tensor:
    """librosa.power_to_db(ref=1.0, amin=1e-10, top_db=80) 와 동일."""
    db = 10.0 * torch.log10(power.clamp_min(_LOG_EPS))
    peak = db.amax(dim=(-2, -1), keepdim=True)
    return torch.maximum(db, peak - top_db)


def _standardise_per_sample(feat: torch.Tensor) -> torch.Tensor:
    """샘플별로 독립 정규화한다.

    배치 통계를 쓰면 한 샘플의 예측이 같은 배치에 우연히 들어온 다른 샘플에
    영향을 받아 학습과 추론이 어긋난다. 통화 음량 차이도 여기서 흡수된다.
    """
    mean = feat.mean(dim=(-2, -1), keepdim=True)
    std = feat.std(dim=(-2, -1), keepdim=True, unbiased=False)
    return (feat - mean) / std.clamp_min(1e-5)
