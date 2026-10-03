"""
Mission 2: Speaker Classification Inference Engine.
- Task: 119 Emergency Call Speaker Classification (Dispatcher: 0 vs Caller: 1)
- Model Architecture: 3-Model Ensemble (ReDimNet2-B2 + ECAPA-TDNN + AudioResNet-50)
- Equal Ensemble Weights: 1/3 for each model
- Fixed Decision Threshold: 0.50 (Competition Rule Compliant)

Computational Efficiency Metrics:
- Total Parameters: 31.88 M (Active Parameters: 31.88 M)
  * ReDimNet2-B2: 2.57 M
  * ECAPA-TDNN: 5.80 M
  * AudioResNet-50: 23.50 M
- Benchmark Environment: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM), Intel Core i7-12700H
- Average Inference Latency: ~11.50 ms per clip (RTF: 0.0038)
  * ReDimNet2-B2: 1.73 ms
  * ECAPA-TDNN: 5.12 ms
  * AudioResNet-50: 4.65 ms
- Throughput: ~260+ utterances/sec
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

DECISION_THRESHOLD = 0.50
SAMPLE_RATE = 16000
TARGET_DURATION = 3.0
TARGET_LENGTH = int(SAMPLE_RATE * TARGET_DURATION)  # 48,000 samples

OUTPUT_COLUMNS = ["audio file name", "startAt", "endAt", "speaker"]


def pad_or_truncate_audio(audio: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
    """Pad with zero (silence) or center-crop to target length."""
    if len(audio) == target_length:
        return audio
    elif len(audio) > target_length:
        start = (len(audio) - target_length) // 2
        return audio[start:start + target_length]
    else:
        pad_width = target_length - len(audio)
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return np.pad(audio, (pad_left, pad_right), mode="constant", constant_values=0.0)


def extract_normalized_mel(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256
) -> torch.Tensor:
    """
    Extract Mel-Spectrogram with training-identical normalization:
    (mel_db + 80.0) / 80.0 -> values roughly in [0.0, 1.0].
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db + 80.0) / 80.0
    return torch.tensor(mel_norm, dtype=torch.float32)


def load_model_weights(model: nn.Module, ckpt_path: Path, device: torch.device) -> nn.Module:
    """Load model checkpoint state_dict safely with fail-fast exception handling."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[Error] Checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(str(ckpt_path), map_location=device)
        if isinstance(state, dict):
            if "state_dict" in state:
                state_dict = state["state_dict"]
            elif "model_state_dict" in state:
                state_dict = state["model_state_dict"]
            elif "model" in state:
                state_dict = state["model"]
            else:
                state_dict = state
        else:
            state_dict = state

        # 1. Try direct load
        try:
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            return model
        except Exception:
            pass

        # 2. Try prefix adaptation (handling module. and resnet. prefixes)
        model_keys = set(model.state_dict().keys())
        adapted_state = {}
        for k, v in state_dict.items():
            k_clean = k.replace("module.", "")
            if f"resnet.{k_clean}" in model_keys:
                adapted_state[f"resnet.{k_clean}"] = v
            elif k_clean.replace("resnet.", "") in model_keys:
                adapted_state[k_clean.replace("resnet.", "")] = v
            elif k_clean in model_keys:
                adapted_state[k_clean] = v
            else:
                adapted_state[k] = v

        model.load_state_dict(adapted_state, strict=True)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"[Error] Failed to load checkpoint ({ckpt_path}): {e}") from e


class Mission2InferenceEngine:
    """Speaker classification inference engine with 3-model ensemble support."""
    def __init__(self, ckpt_path: Union[str, Path], device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        ckpt_path = Path(ckpt_path).resolve()
        self.models: Dict[str, nn.Module] = {}

        # Import self-contained models
        from m2.models import AudioResNet, ECAPA_TDNN, ReDimNet2_B2

        if ckpt_path.is_dir():
            ckpt_dir = ckpt_path
        else:
            ckpt_dir = ckpt_path.parent

        p_redim = ckpt_dir / "best_redimnet.pt"
        p_ecapa = ckpt_dir / "best_ecapa_tdnn.pt"
        p_resnet = ckpt_dir / "best_resnet50.pt"

        # Check if all 3 ensemble checkpoints exist
        if p_redim.exists() and p_ecapa.exists() and p_resnet.exists():
            print(f"[Mission 2] Initializing 3-Model Ensemble (Checkpoint directory: {ckpt_dir})")
            m_r = ReDimNet2_B2(num_classes=1).to(self.device)
            m_e = ECAPA_TDNN(in_channels=80, channels=512, num_classes=1).to(self.device)
            m_res = AudioResNet(num_classes=1).to(self.device)

            self.models["redimnet"] = load_model_weights(m_r, p_redim, self.device)
            self.models["ecapa_tdnn"] = load_model_weights(m_e, p_ecapa, self.device)
            self.models["resnet50"] = load_model_weights(m_res, p_resnet, self.device)
            self.is_ensemble = True
            print("[Mission 2] Successfully loaded 3 models: ReDimNet2-B2, ECAPA-TDNN, AudioResNet-50")
        elif ckpt_path.is_file():
            print(f"[Mission 2] Initializing Single Model Mode: {ckpt_path.name}")
            fname = ckpt_path.name.lower()
            if "redim" in fname:
                m = ReDimNet2_B2(num_classes=1).to(self.device)
                self.models["redimnet"] = load_model_weights(m, ckpt_path, self.device)
            elif "ecapa" in fname:
                m = ECAPA_TDNN(in_channels=80, channels=512, num_classes=1).to(self.device)
                self.models["ecapa_tdnn"] = load_model_weights(m, ckpt_path, self.device)
            else:
                m = AudioResNet(num_classes=1).to(self.device)
                self.models["resnet50"] = load_model_weights(m, ckpt_path, self.device)
            self.is_ensemble = False
        else:
            raise FileNotFoundError(f"[Error] Checkpoint path not found: {ckpt_path}")

    @torch.no_grad()
    def predict_clip(self, clip_audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
        """
        Predict probability of being Caller (Class 1) for a single audio clip.
        Returns float probability in [0.0, 1.0].
        """
        import librosa

        if sr != SAMPLE_RATE:
            clip_audio = librosa.resample(clip_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        padded_audio = pad_or_truncate_audio(clip_audio, TARGET_LENGTH)

        # Preprocess for ReDimNet & ECAPA (80-mel)
        mel_80 = extract_normalized_mel(
            padded_audio, sr=SAMPLE_RATE, n_mels=80, n_fft=1024, hop_length=256
        ).unsqueeze(0).to(self.device)

        # Preprocess for ResNet-50 (128-mel)
        mel_128 = extract_normalized_mel(
            padded_audio, sr=SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512
        ).unsqueeze(0).unsqueeze(0).to(self.device)

        probs = []

        if "redimnet" in self.models:
            out = self.models["redimnet"](mel_80)
            p = torch.sigmoid(out).item() if out.shape[-1] == 1 else torch.softmax(out, dim=-1)[0, 1].item()
            probs.append(p)

        if "ecapa_tdnn" in self.models:
            out = self.models["ecapa_tdnn"](mel_80)
            p = torch.sigmoid(out).item() if out.shape[-1] == 1 else torch.softmax(out, dim=-1)[0, 1].item()
            probs.append(p)

        if "resnet50" in self.models:
            out = self.models["resnet50"](mel_128)
            p = torch.sigmoid(out).item() if out.shape[-1] == 1 else torch.softmax(out, dim=-1)[0, 1].item()
            probs.append(p)

        # Equal weighting (1/3 each) or single model
        return float(np.mean(probs))


def predict_directory(
    audio_dir: Union[str, Path],
    label_dir: Union[str, Path],
    ckpt_path: Union[str, Path],
    sr: int = 16000
) -> pd.DataFrame:
    """
    Perform speaker classification on directory of audio files and JSON label transcripts.
    Strictly follows startAt / endAt utterance boundaries.
    """
    import librosa

    audio_dir = Path(audio_dir).resolve()
    label_dir = Path(label_dir).resolve()

    if not audio_dir.exists():
        raise FileNotFoundError(f"[Error] audio_dir does not exist: {audio_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"[Error] label_dir does not exist: {label_dir}")

    engine = Mission2InferenceEngine(ckpt_path=ckpt_path)

    json_files = sorted(list(label_dir.glob("*.json"))) + sorted(list(label_dir.glob("**/*.json")))
    # Deduplicate while preserving order
    seen = set()
    unique_json_files = []
    for jf in json_files:
        if jf not in seen:
            seen.add(jf)
            unique_json_files.append(jf)

    if not unique_json_files:
        raise FileNotFoundError(f"[Error] No JSON label files found in {label_dir}")

    results: List[Dict[str, Union[str, int]]] = []
    correct_count = 0
    total_count = 0

    print(f"[Mission 2] Processing {len(unique_json_files)} label files...")

    for jf in tqdm(unique_json_files, desc="[Mission 2] Inference"):
        stem = jf.stem
        wav_name = f"{stem}.wav"
        wav_path = audio_dir / wav_name

        if not wav_path.exists():
            found = list(audio_dir.glob(f"**/{wav_name}"))
            if found:
                wav_path = found[0]
            else:
                continue

        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            full_audio, _ = librosa.load(str(wav_path), sr=sr)
        except Exception as e:
            print(f"[Warning] Failed to load audio ({wav_name}): {e}")
            continue

        dialog_list = data.get("utterances") or data.get("dialogs") or data.get("dialogue") or []

        for utt in dialog_list:
            start_ms = utt.get("startAt") if "startAt" in utt else utt.get("start_time", 0)
            end_ms = utt.get("endAt") if "endAt" in utt else utt.get("end_time", 0)

            if start_ms < 100 and end_ms < 100 and (end_ms - start_ms) > 0.05:
                start_ms = int(start_ms * 1000)
                end_ms = int(end_ms * 1000)
            else:
                start_ms = int(start_ms)
                end_ms = int(end_ms)

            start_sample = int((start_ms / 1000.0) * sr)
            end_sample = int((end_ms / 1000.0) * sr)
            clip = full_audio[start_sample:end_sample]

            avg_prob = engine.predict_clip(clip, sr=sr)
            final_pred = 1 if avg_prob >= DECISION_THRESHOLD else 0

            if "speaker" in utt:
                actual = int(utt["speaker"])
                if final_pred == actual:
                    correct_count += 1
                total_count += 1

            results.append({
                "audio file name": wav_name,
                "startAt": start_ms,
                "endAt": end_ms,
                "speaker": final_pred
            })

    if total_count > 0:
        accuracy = (correct_count / total_count) * 100.0
        print(f"\n[Evaluation Result] Total: {total_count:,} utterances | Correct: {correct_count:,}")
        print(f"[Evaluation Result] Accuracy: {accuracy:.2f}% (Threshold: {DECISION_THRESHOLD})")
    else:
        print("\n[Info] Ground truth labels not provided; evaluation skipped.")

    return pd.DataFrame(results, columns=OUTPUT_COLUMNS)
