# -*- coding: utf-8 -*-
# Mission 2 Inference and Serving Engine
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Constants
SAMPLE_RATE = 16000
TARGET_DURATION = 3.0  # seconds
TARGET_LENGTH = int(SAMPLE_RATE * TARGET_DURATION)  # 48,000 samples
DECISION_THRESHOLD = 0.50

OUTPUT_COLUMNS = ["audio file name", "startAt", "endAt", "speaker"]


def load_model_weights(model: nn.Module, ckpt_path: Path, device: torch.device) -> nn.Module:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[Error] Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    try:
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
    except Exception:
        pass

    model_keys = set(model.state_dict().keys())
    adapted = {}
    for k, v in state_dict.items():
        k_clean = k.replace("module.", "").replace("_orig_mod.", "")
        if f"resnet.{k_clean}" in model_keys:
            adapted[f"resnet.{k_clean}"] = v
        elif k_clean.replace("resnet.", "") in model_keys:
            adapted[k_clean.replace("resnet.", "")] = v
        else:
            adapted[k_clean] = v

    model.load_state_dict(adapted, strict=True)
    model.eval()
    return model


class Mission2InferenceEngine:
    def __init__(
        self,
        ckpt_path: Union[str, Path],
        device: Optional[torch.device] = None,
        ensemble_mode: str = "auto",
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Full FP32 on GPU: with TF32 convolutions (PyTorch default) cuDNN picks different
        # algorithms per batch size, and probabilities near 0.5 can flip between runs.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

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

        has_redim = p_redim.exists()
        has_ecapa = p_ecapa.exists()
        has_resnet = p_resnet.exists()

        if ensemble_mode == "3model":
            if not ckpt_path.is_dir():
                raise ValueError(
                    f"[Error] --ensemble_mode 3model requires a checkpoint directory, but a file was provided: {ckpt_path}"
                )
            missing = [p.name for p, exists in [(p_redim, has_redim), (p_ecapa, has_ecapa), (p_resnet, has_resnet)] if not exists]
            if missing:
                raise FileNotFoundError(
                    f"[Error] --ensemble_mode 3model requires all 3 checkpoints in {ckpt_dir}, but missing: {missing}"
                )
            print(f"[Mission 2] Initializing 3-Model Equal Ensemble (1/3 weights) from: {ckpt_dir}")
            m_r = ReDimNet2_B2(num_classes=1).to(self.device)
            m_e = ECAPA_TDNN(in_channels=80, channels=512, num_classes=1).to(self.device)
            m_res = AudioResNet(num_classes=1).to(self.device)

            self.models["redimnet"] = load_model_weights(m_r, p_redim, self.device)
            self.models["ecapa_tdnn"] = load_model_weights(m_e, p_ecapa, self.device)
            self.models["resnet50"] = load_model_weights(m_res, p_resnet, self.device)
            self.is_ensemble = True
            print("[Mission 2] Successfully loaded 3 models: ReDimNet2-B2, ECAPA-TDNN, AudioResNet-50")

        elif ensemble_mode == "2model":
            if not ckpt_path.is_dir():
                raise ValueError(
                    f"[Error] --ensemble_mode 2model requires a checkpoint directory, but a file was provided: {ckpt_path}"
                )
            missing = [p.name for p, exists in [(p_redim, has_redim), (p_ecapa, has_ecapa)] if not exists]
            if missing:
                raise FileNotFoundError(
                    f"[Error] --ensemble_mode 2model requires best_redimnet.pt and best_ecapa_tdnn.pt in {ckpt_dir}, but missing: {missing}"
                )
            print(f"[Mission 2] Initializing 2-Model Lightweight Ensemble (ReDimNet2-B2 + ECAPA-TDNN, 80-Mel unified)")
            m_r = ReDimNet2_B2(num_classes=1).to(self.device)
            m_e = ECAPA_TDNN(in_channels=80, channels=512, num_classes=1).to(self.device)
            self.models["redimnet"] = load_model_weights(m_r, p_redim, self.device)
            self.models["ecapa_tdnn"] = load_model_weights(m_e, p_ecapa, self.device)
            self.is_ensemble = True
            print("[Mission 2] Successfully loaded 2 models: ReDimNet2-B2, ECAPA-TDNN (8.36M params)")

        elif ensemble_mode == "auto":
            if ckpt_path.is_dir():
                if has_redim and has_ecapa and has_resnet:
                    print(f"[Mission 2] Auto-detected 3-Model Ensemble in: {ckpt_dir}")
                    m_r = ReDimNet2_B2(num_classes=1).to(self.device)
                    m_e = ECAPA_TDNN(in_channels=80, channels=512, num_classes=1).to(self.device)
                    m_res = AudioResNet(num_classes=1).to(self.device)

                    self.models["redimnet"] = load_model_weights(m_r, p_redim, self.device)
                    self.models["ecapa_tdnn"] = load_model_weights(m_e, p_ecapa, self.device)
                    self.models["resnet50"] = load_model_weights(m_res, p_resnet, self.device)
                    self.is_ensemble = True
                    print("[Mission 2] Successfully loaded 3 models: ReDimNet2-B2, ECAPA-TDNN, AudioResNet-50")
                elif has_redim and has_ecapa:
                    print(f"[Mission 2] Auto-detected 2-Model Ensemble (ReDimNet + ECAPA) in: {ckpt_dir}")
                    m_r = ReDimNet2_B2(num_classes=1).to(self.device)
                    m_e = ECAPA_TDNN(in_channels=80, channels=512, num_classes=1).to(self.device)
                    self.models["redimnet"] = load_model_weights(m_r, p_redim, self.device)
                    self.models["ecapa_tdnn"] = load_model_weights(m_e, p_ecapa, self.device)
                    self.is_ensemble = True
                elif has_ecapa:
                    print(f"[Mission 2] Auto-detected Single Model: ECAPA-TDNN")
                    m = ECAPA_TDNN(in_channels=80, channels=512, num_classes=1).to(self.device)
                    self.models["ecapa_tdnn"] = load_model_weights(m, p_ecapa, self.device)
                    self.is_ensemble = False
                elif has_redim:
                    print(f"[Mission 2] Auto-detected Single Model: ReDimNet2-B2")
                    m = ReDimNet2_B2(num_classes=1).to(self.device)
                    self.models["redimnet"] = load_model_weights(m, p_redim, self.device)
                    self.is_ensemble = False
                elif has_resnet:
                    print(f"[Mission 2] Auto-detected Single Model: AudioResNet-50")
                    m = AudioResNet(num_classes=1).to(self.device)
                    self.models["resnet50"] = load_model_weights(m, p_resnet, self.device)
                    self.is_ensemble = False
                else:
                    raise FileNotFoundError(f"[Error] No valid checkpoints found in directory: {ckpt_dir}")
            elif ckpt_path.is_file():
                print(f"[Mission 2] Initializing Single Model Mode from file: {ckpt_path.name}")
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
        else:
            raise ValueError(f"[Error] Unknown ensemble_mode: '{ensemble_mode}'. Valid choices: 'auto', '3model', '2model'.")

    @torch.no_grad()
    def predict_clip(self, clip_audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
        return float(self.predict_clips([clip_audio], sr=sr)[0])

    @torch.no_grad()
    def predict_clips(self, clips: List[np.ndarray], sr: int = SAMPLE_RATE, batch_size: int = 64) -> np.ndarray:
        """Return the ensemble probability for each clip.

        Mel features are computed per clip exactly as before; only the model forward
        passes are batched (one call per model per chunk) to avoid per-utterance GPU syncs.
        """
        import librosa
        from m2.audio_features import normalize_clip

        if not self.models:
            raise RuntimeError("[Error] No active models in inference engine to compute probability.")
        if batch_size < 1:
            raise ValueError(f"[Error] batch_size must be >= 1, got {batch_size}")

        use_80 = "redimnet" in self.models or "ecapa_tdnn" in self.models
        use_128 = "resnet50" in self.models
        out = np.empty(len(clips), dtype=np.float64)

        for start in range(0, len(clips), batch_size):
            chunk = clips[start:start + batch_size]
            mels_80, mels_128 = [], []
            for clip_audio in chunk:
                if sr != SAMPLE_RATE:
                    clip_audio = librosa.resample(clip_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                clip_audio = normalize_clip(clip_audio, TARGET_LENGTH, "center")
                if use_80:
                    mels_80.append(self._extract_normalized_mel(
                        clip_audio, sr=SAMPLE_RATE, n_mels=80, n_fft=512, hop_length=160))
                if use_128:
                    mels_128.append(self._extract_normalized_mel(
                        clip_audio, sr=SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512))

            # Same model order as the original per-clip loop: ReDimNet, ECAPA-TDNN, ResNet-50
            probs = []
            if use_80:
                t_80 = torch.from_numpy(np.stack(mels_80)).to(self.device)
                for key, label in (("redimnet", "ReDimNet2-B2"), ("ecapa_tdnn", "ECAPA-TDNN")):
                    if key in self.models:
                        p = torch.sigmoid(self.models[key](t_80)).reshape(-1).cpu().numpy().astype(np.float64)
                        if not np.all(np.isfinite(p)):
                            raise RuntimeError(f"[Critical Error] {label} produced non-finite probability")
                        probs.append(p)
            if use_128:
                t_128 = torch.from_numpy(np.stack(mels_128)).unsqueeze(1).to(self.device)
                p = torch.sigmoid(self.models["resnet50"](t_128)).reshape(-1).cpu().numpy().astype(np.float64)
                if not np.all(np.isfinite(p)):
                    raise RuntimeError("[Critical Error] AudioResNet-50 produced non-finite probability")
                probs.append(p)

            mean_p = np.mean(np.stack(probs, axis=0), axis=0)
            if not np.all(np.isfinite(mean_p)):
                raise RuntimeError("[Critical Error] Final ensemble probability is non-finite")
            out[start:start + len(chunk)] = mean_p

        return out

    @staticmethod
    def _extract_normalized_mel(
        audio: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int
    ) -> np.ndarray:
        from m2.audio_features import extract_normalized_mel
        return extract_normalized_mel(audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)


def predict_directory(
    audio_dir: Path,
    label_dir: Path,
    ckpt_path: Path,
    output_path: Optional[Path] = None,
    ensemble_mode: str = "auto",
    batch_size: int = 64,
) -> pd.DataFrame:
    import librosa
    from m2.audio_features import load_and_resample_call, extract_utterance_clip
    from m2.input_mapping import discover_input_files

    audio_dir = Path(audio_dir).resolve()
    label_dir = Path(label_dir).resolve()
    if output_path is not None:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if not audio_dir.exists():
        raise FileNotFoundError(f"[Error] audio_dir does not exist: {audio_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"[Error] label_dir does not exist: {label_dir}")

    unique_json_files, wav_files = discover_input_files(audio_dir, label_dir)
    engine = Mission2InferenceEngine(ckpt_path=ckpt_path, ensemble_mode=ensemble_mode)

    results = []
    correct_count = 0
    total_count = 0
    missing_audio_files = 0
    missing_audio_utts = 0
    sr = SAMPLE_RATE

    print(f"[Mission 2] Processing {len(unique_json_files)} label files...")

    for jf in tqdm(unique_json_files, desc="[Mission 2] Inference"):
        stem = jf.stem
        wav_name = f"{stem}.wav"
        wav_path = wav_files.get(stem)

        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        dialog_list = data.get("utterances") or data.get("dialogs") or data.get("dialogue") or []

        if wav_path is None:
            missing_audio_files += 1
            for utt in dialog_list:
                start_ms = int(utt.get("startAt", utt.get("start_time", 0)))
                end_ms = int(utt.get("endAt", utt.get("end_time", 0)))
                silent_prob = engine.predict_clip(np.zeros(TARGET_LENGTH, dtype=np.float32), sr=sr)
                final_pred = 1 if silent_prob >= DECISION_THRESHOLD else 0
                results.append({
                    "audio file name": wav_name,
                    "startAt": start_ms,
                    "endAt": end_ms,
                    "speaker": final_pred
                })
                missing_audio_utts += 1
            continue

        try:
            full_audio = load_and_resample_call(str(wav_path), target_sr=sr)
        except Exception as e:
            print(f"[Warning] Failed to load audio ({wav_name}): {e}")
            missing_audio_files += 1
            for utt in dialog_list:
                start_ms = int(utt.get("startAt", utt.get("start_time", 0)))
                end_ms = int(utt.get("endAt", utt.get("end_time", 0)))
                silent_prob = engine.predict_clip(np.zeros(TARGET_LENGTH, dtype=np.float32), sr=sr)
                final_pred = 1 if silent_prob >= DECISION_THRESHOLD else 0
                results.append({
                    "audio file name": wav_name,
                    "startAt": start_ms,
                    "endAt": end_ms,
                    "speaker": final_pred
                })
                missing_audio_utts += 1
            continue

        clips, spans = [], []
        for utt in dialog_list:
            start_value = utt.get("startAt", utt.get("start_time", 0))
            end_value = utt.get("endAt", utt.get("end_time", 0))
            spans.append((int(start_value), int(end_value)))
            clips.append(extract_utterance_clip(full_audio, float(start_value), float(end_value), crop_mode="center"))

        try:
            call_probs = engine.predict_clips(clips, sr=sr, batch_size=batch_size)
        except Exception as e:
            raise RuntimeError(f"[Critical Error in {wav_name}]: {e}") from e

        for utt, (start_ms, end_ms), prob_1 in zip(dialog_list, spans, call_probs):
            final_pred = 1 if prob_1 >= DECISION_THRESHOLD else 0

            if "speaker" in utt:
                try:
                    actual = int(utt["speaker"])
                    if final_pred == actual:
                        correct_count += 1
                    total_count += 1
                except (ValueError, TypeError):
                    pass

            results.append({
                "audio file name": wav_name,
                "startAt": start_ms,
                "endAt": end_ms,
                "speaker": final_pred
            })

    print(f"\n[Mission 2] Inference finished: Generated {len(results):,} total prediction rows.")
    if missing_audio_utts > 0:
        print(f"[Mission 2] Note: {missing_audio_files} files ({missing_audio_utts} utterances) lacked audio and were preserved via silence fallback.")
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100.0
        print(f"[Evaluation Result] Valid Audio Utterances: {total_count:,} | Correct: {correct_count:,}")
        print(f"[Evaluation Result] Accuracy: {accuracy:.2f}% (Threshold: {DECISION_THRESHOLD})")
    else:
        print("\n[Info] Ground truth labels not provided; evaluation skipped.")

    df_out = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    df_out = df_out.sort_values(["audio file name", "startAt", "endAt"], kind="stable").reset_index(drop=True)
    if output_path is not None:
        df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[Mission 2] Submission CSV saved successfully: {output_path} ({len(df_out):,} rows)")
    return df_out
