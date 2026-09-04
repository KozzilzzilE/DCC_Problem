"""순수 torch 피처 프런트엔드가 librosa와 수치적으로 일치하는지, 그리고
길이 처리(패딩/크롭/슬라이딩 윈도우)가 규칙을 지키는지 검증."""
import inspect

import numpy as np
import pytest
import torch

from m1.config import FeatureConfig
from m1.features import MelFrontend, mel_filterbank, sliding_windows

librosa = pytest.importorskip("librosa")


@pytest.fixture
def cfg():
    return FeatureConfig()


@pytest.fixture
def tone():
    """220Hz + 440Hz 합성음 3초 (8kHz)."""
    t = np.arange(24448, dtype=np.float32) / 8000.0
    return (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_mel_filterbank_matches_librosa(cfg):
    ours = mel_filterbank(cfg).numpy()
    theirs = librosa.filters.mel(
        sr=cfg.sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax
    )
    assert ours.shape == theirs.shape == (cfg.n_mels, cfg.n_fft // 2 + 1)
    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-7)


def test_logmel_matches_librosa_melspectrogram(cfg, tone):
    ref = librosa.feature.melspectrogram(
        y=tone,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    front = MelFrontend(cfg)
    ours = front.mel_power(torch.from_numpy(tone).unsqueeze(0)).squeeze(0).numpy()

    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=1e-3, atol=1e-4)


def test_forward_shape_logmel(cfg, tone):
    out = MelFrontend(cfg)(torch.from_numpy(tone).unsqueeze(0))
    assert out.shape == (1, 1, cfg.n_mels, cfg.window_frames)


def test_forward_shape_mfcc(tone):
    cfg = FeatureConfig(kind="mfcc", n_mfcc=40)
    out = MelFrontend(cfg)(torch.from_numpy(tone).unsqueeze(0))
    assert out.shape == (1, 1, 40, cfg.window_frames)


def test_mfcc_matches_librosa(tone):
    cfg = FeatureConfig(kind="mfcc", n_mfcc=20)
    ref = librosa.feature.mfcc(
        y=tone,
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    ours = MelFrontend(cfg).mfcc(torch.from_numpy(tone).unsqueeze(0)).squeeze(0).numpy()
    assert ours.shape == ref.shape
    # librosa는 power_to_db(top_db=80) 클리핑을 쓰므로 상관계수로 비교
    for row_ours, row_ref in zip(ours[:8], ref[:8]):
        assert np.corrcoef(row_ours, row_ref)[0, 1] > 0.99


def test_output_is_normalised_per_sample(cfg):
    """배치 내 각 샘플이 독립적으로 정규화되어야 한다 (샘플 간 정보 누수 금지)."""
    loud = np.random.RandomState(0).randn(24448).astype(np.float32) * 10.0
    quiet = loud * 0.001

    batch = torch.from_numpy(np.stack([loud, quiet]))
    out = MelFrontend(cfg)(batch)

    np.testing.assert_allclose(out[0].numpy(), out[1].numpy(), rtol=1e-3, atol=1e-3)
    for i in range(2):
        assert abs(float(out[i].mean())) < 1e-4
        assert abs(float(out[i].std()) - 1.0) < 1e-2


def test_short_input_is_zero_padded(cfg):
    short = torch.zeros(1, 4000)
    out = MelFrontend(cfg)(short)
    assert out.shape == (1, 1, cfg.n_mels, cfg.window_frames)


def test_frontend_signature_takes_no_label(cfg):
    """규칙: sample instance마다 label에 따라 상이한 전처리 금지."""
    params = set(inspect.signature(MelFrontend.forward).parameters)
    assert params == {"self", "waveform"}


def test_window_samples_yields_exact_frame_count(cfg):
    n = cfg.window_samples
    frames = MelFrontend(cfg).mel_power(torch.zeros(1, n)).shape[-1]
    assert frames == cfg.window_frames


def test_sliding_windows_covers_whole_signal(cfg):
    total = cfg.window_samples * 3 + 777
    wins = sliding_windows(total, cfg.window_samples, cfg.window_samples // 2)

    assert all(e - s == cfg.window_samples for s, e in wins)
    assert wins[0][0] == 0
    assert wins[-1][1] == total  # 마지막 창은 끝에 붙여 꼬리 손실 없음
    assert len(wins) >= 3


def test_sliding_windows_short_signal_gives_single_window(cfg):
    wins = sliding_windows(1000, cfg.window_samples, cfg.window_samples // 2)
    assert wins == [(0, cfg.window_samples)]


def test_deterministic(cfg, tone):
    front = MelFrontend(cfg).eval()
    x = torch.from_numpy(tone).unsqueeze(0)
    np.testing.assert_array_equal(front(x).numpy(), front(x).numpy())
