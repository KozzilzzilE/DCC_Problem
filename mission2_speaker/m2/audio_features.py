# -*- coding: utf-8 -*-
"""
DCC Mission 2: Unified Audio Preprocessing & Feature Extraction.

Shared preprocessing for Dataset, Notebook, and CLI in the same library environment.
- Audio Sample Rate: 16,000 Hz
- Target Duration: 3.0s (48,000 samples)
- 80-Mel (ReDimNet, ECAPA-TDNN): n_mels=80, n_fft=512, hop_length=160
- 128-Mel (AudioResNet-50): n_mels=128, n_fft=2048, hop_length=512
- Mel dB Normalization: librosa.power_to_db(top_db=80.0, ref=max(1e-10, max(mel)))
                        mel_norm = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple, Union

import librosa
import numpy as np

SAMPLE_RATE = 16000
TARGET_DURATION = 3.0
TARGET_LENGTH = int(SAMPLE_RATE * TARGET_DURATION)  # 48,000 samples


def extract_normalized_mel(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Standard Librosa Slaney-scale Mel Spectrogram extraction.
    Output: (n_mels, time) float32 array in [0.0, 1.0].
    """
    if audio.ndim != 1 or not np.all(np.isfinite(audio)):
        raise ValueError("Mel input must be a finite mono waveform")
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    ref_val = float(np.max(mel))
    ref_val = max(1e-10, ref_val)

    mel_db = librosa.power_to_db(mel, ref=ref_val, top_db=80.0)
    mel_norm = (mel_db + 80.0) / 80.0
    return np.clip(mel_norm, 0.0, 1.0).astype(np.float32)


@lru_cache(maxsize=32)
def _load_call_cached(wav_path_str: str, target_sr: int, file_size: int, mtime_ns: int) -> np.ndarray:
    audio, _ = librosa.load(wav_path_str, sr=target_sr)
    if not len(audio) or not np.all(np.isfinite(audio)):
        raise ValueError(f"Empty or non-finite audio: {wav_path_str}")
    return audio


def load_and_resample_call(wav_path_str: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load call audio and resample to target_sr once.
    Cached across utterances belonging to the same call.
    """
    path = Path(wav_path_str).resolve()
    stat = path.stat()
    return _load_call_cached(str(path), target_sr, stat.st_size, stat.st_mtime_ns)


def normalize_clip(clip: np.ndarray, target_length: int = TARGET_LENGTH,
                   crop_mode: str = "center") -> np.ndarray:
    if crop_mode not in {"center", "random"}:
        raise ValueError(f"Unsupported crop mode: {crop_mode}")
    if clip.ndim != 1 or not np.all(np.isfinite(clip)):
        raise ValueError("Clip must be a finite mono waveform")
    if len(clip) < target_length:
        return np.pad(clip, (0, target_length - len(clip)), mode="constant")
    if len(clip) > target_length:
        offset = (np.random.randint(0, len(clip) - target_length + 1)
                  if crop_mode == "random" else (len(clip) - target_length) // 2)
        return clip[offset:offset + target_length]
    return clip


def extract_utterance_clip(
    full_audio: np.ndarray,
    start_ms: float,
    end_ms: float,
    target_sr: int = SAMPLE_RATE,
    target_length: int = TARGET_LENGTH,
    crop_mode: str = "center",
) -> np.ndarray:
    """
    Extract utterance slice from resampled call audio and length-normalize to 3.0s (48,000 samples).
    - If length < target_length: right zero-padding.
    - If length > target_length: center crop ('center') or random crop ('random').
    Identical to predict_directory in m2/infer.py.
    """
    audio_len = len(full_audio)
    start_idx = int((start_ms / 1000.0) * target_sr)
    end_idx = int((end_ms / 1000.0) * target_sr)

    if end_idx <= start_idx or start_idx >= audio_len:
        return np.zeros(target_length, dtype=np.float32)

    clip = full_audio[max(0, start_idx) : min(audio_len, end_idx)]
    return normalize_clip(clip, target_length, crop_mode)
