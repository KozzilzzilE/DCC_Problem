# -*- coding: utf-8 -*-
"""
DCC Mission 2: Path B Common Helpers and Preprocessing Pipeline.

Ensures strict parity between Training Dataset, Notebook Evaluator, and Submission CLI.
Preprocessing Standard:
- Librosa 16kHz audio, 3.0s window (48,000 samples)
- 80-Mel (ReDimNet, ECAPA-TDNN): n_mels=80, n_fft=512, hop_length=160 (301 frames)
- 128-Mel (AudioResNet-50): n_mels=128, n_fft=2048, hop_length=512 (94 frames)
- dB Normalization: mel_db = librosa.power_to_db(mel, ref=max(1e-10, max(mel)), top_db=80.0)
                    mel_norm = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from m2.models import AudioResNet, ECAPA_TDNN, ReDimNet2_B2


def compute_sha256(filepath: Union[Path, str]) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_normalized_mel(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Standard Librosa Mel Spectrogram extraction identical to m2/infer.py.
    Output range: [0.0, 1.0] float32.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    ref_val = float(np.max(mel))
    ref_val = max(1e-10, ref_val)

    mel_db = librosa.power_to_db(mel, ref=ref_val, top_db=80.0)
    mel_norm = (mel_db + 80.0) / 80.0
    mel_norm = np.clip(mel_norm, 0.0, 1.0).astype(np.float32)
    return mel_norm


class DCCAudioDatasetUnified(Dataset):
    """
    Unified Dataset for Training and Evaluation.
    Guarantees exact parity with submission CLI audio loading and Mel preprocessing.
    """
    def __init__(
        self,
        data_dir: Union[Path, str],
        is_train: bool = True,
        allowed_call_ids: Optional[set] = None,
        max_files: Optional[int] = None,
        max_samples: Optional[int] = None,
        crop_mode: str = "random",
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        window_sec: float = 3.0,
        sr: int = 16000,
    ):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.crop_mode = crop_mode if is_train else "center"
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_sec = window_sec
        self.sr = sr
        self.target_samples = int(sr * window_sec)

        whitelist = set(allowed_call_ids) if allowed_call_ids is not None else None
        self.samples, self.audit_info = self._parse_data(whitelist, max_files, max_samples)

    def _parse_data(self, whitelist: Optional[set], max_files: Optional[int], max_samples: Optional[int]):
        samples = []
        audit = {"total_json": 0, "parsed_json": 0, "skipped_calls": 0, "missing_wav": 0, "invalid_utts": 0}

        label_dir = self.data_dir / "label"
        audio_dir = self.data_dir / "audio"

        if not label_dir.exists():
            label_dir = self.data_dir
            audio_dir = self.data_dir

        json_files = sorted(glob.glob(str(label_dir / "**" / "*.json"), recursive=True))
        audit["total_json"] = len(json_files)
        if max_files:
            json_files = json_files[:max_files]

        for jf_str in json_files:
            jf = Path(jf_str)
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            stem = jf.stem
            call_id = stem
            for prefix in ("VS_", "VL_", "TS_", "TL_"):
                if call_id.startswith(prefix):
                    call_id = call_id[len(prefix):]
                    break

            if whitelist is not None and call_id not in whitelist:
                audit["skipped_calls"] += 1
                continue

            wav_name = stem + ".wav"
            wav_path = audio_dir / wav_name
            if not wav_path.exists():
                candidates = list(audio_dir.glob(f"**/{wav_name}"))
                if candidates:
                    wav_path = candidates[0]
                else:
                    audit["missing_wav"] += 1
                    continue

            utterances = data.get("utterances", [])
            for utt_idx, utt in enumerate(utterances):
                spk = utt.get("speaker", None)
                if spk is None:
                    spk = utt.get("role", None)

                if spk in (1, "1", "신고자"):
                    label_val = 1.0
                elif spk in (0, "0", "상황실"):
                    label_val = 0.0
                else:
                    audit["invalid_utts"] += 1
                    continue

                start_ms = float(utt.get("startAt", 0.0))
                end_ms = float(utt.get("endAt", 0.0))
                if end_ms <= start_ms:
                    audit["invalid_utts"] += 1
                    continue

                sample_id = f"{call_id}_{utt_idx:04d}_{int(start_ms)}_{int(end_ms)}"
                samples.append({
                    "sample_id": sample_id,
                    "call_id": call_id,
                    "wav_path": str(wav_path),
                    "start_s": start_ms / 1000.0,
                    "end_s": end_ms / 1000.0,
                    "label": label_val,
                })
                if max_samples and len(samples) >= max_samples:
                    audit["parsed_json"] += 1
                    return samples, audit
            audit["parsed_json"] += 1

        return samples, audit

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        info = self.samples[idx]
        wav_path = info["wav_path"]
        start_s = info["start_s"]
        end_s = info["end_s"]
        label = info["label"]

        try:
            dur = max(0.01, end_s - start_s)
            y, _ = librosa.load(wav_path, sr=self.sr, offset=start_s, duration=dur)
        except Exception:
            y = np.zeros(self.target_samples, dtype=np.float32)

        curr_len = len(y)
        if curr_len < self.target_samples:
            pad_len = self.target_samples - curr_len
            wav = np.pad(y, (0, pad_len), mode="constant")
        elif curr_len > self.target_samples:
            if self.crop_mode == "random":
                max_offset = curr_len - self.target_samples
                start_idx = np.random.randint(0, max_offset + 1)
                wav = y[start_idx : start_idx + self.target_samples]
            else:
                start_idx = (curr_len - self.target_samples) // 2
                wav = y[start_idx : start_idx + self.target_samples]
        else:
            wav = y

        mel = extract_normalized_mel(
            wav, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # ReDimNet and ECAPA expect (1, n_mels, T); AudioResNet expects (1, n_mels, T) which DataLoader batches to (B, 1, 128, T)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return mel_tensor, label_tensor, info["sample_id"]


def build_model(model_name: str) -> nn.Module:
    if model_name == "redimnet":
        return ReDimNet2_B2(num_classes=1)
    elif model_name == "ecapa_tdnn":
        return ECAPA_TDNN(in_channels=80, channels=512, num_classes=1)
    elif model_name == "resnet50":
        return AudioResNet(num_classes=1)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def load_model_weights_adapted(model: nn.Module, ckpt_path: Path, device: torch.device) -> nn.Module:
    """Load model state dict with robust ResNet key prefix adaptation."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    try:
        model.load_state_dict(state_dict, strict=True)
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
    return model


def train_path_b_model(
    model_name: str,
    parent_ckpt_path: Path,
    parent_saved_epoch: int,
    additional_epochs: int,
    output_dir: Path,
    train_loader: DataLoader,
    device: torch.device,
    lr: float = 5e-5,
    is_smoke: bool = False,
    run_stage: str = "fixed_continue",
) -> Path:
    """
    Execute fixed additional training under Path B rules.
    - Zero evaluation on official Validation set
    - Zero score-based best selection
    - Saves complete optimizer, scheduler, scaler, and resume metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_sha256 = compute_sha256(parent_ckpt_path)
    model = build_model(model_name).to(device)
    load_model_weights_adapted(model, parent_ckpt_path, device)

    final_filename = f"smoke_last_{model_name}.pt" if is_smoke else f"last_{model_name}.pt"
    final_path = output_dir / final_filename

    if additional_epochs <= 0:
        # Register parent directly
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "completed_epoch": parent_saved_epoch,
            "planned_epochs": parent_saved_epoch,
            "is_smoke": is_smoke,
            "run_stage": run_stage,
            "train_samples": 0,
            "resume_kind": "direct_registered_parent",
            "parent_saved_epoch": parent_saved_epoch,
            "additional_epochs": 0,
            "parent_checkpoint_path": str(parent_ckpt_path),
            "parent_checkpoint_sha256": parent_sha256,
            "parent_selection": "official_validation_best",
            "selection_rule": "fixed_additional_last",
            "timestamp": datetime.now().isoformat(),
        }, final_path)
        print(f"[{model_name}] Registered parent as final: {final_path} (SHA-256: {compute_sha256(final_path)})")
        return final_path

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=additional_epochs, eta_min=1e-6)
    scaler = GradScaler()

    model.train()
    history = []
    total_samples = len(train_loader.dataset)

    for ep in range(1, additional_epochs + 1):
        actual_epoch = parent_saved_epoch + ep
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {actual_epoch}/{parent_saved_epoch + additional_epochs} [Train {model_name}]")
        for inputs, targets, _ in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            with autocast():
                logits = model(inputs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(targets)
            preds = (torch.sigmoid(logits) >= 0.50).float()
            correct += (preds == targets).sum().item()
            total += len(targets)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / max(1, total) * 100:.2f}%"})

        scheduler.step()
        train_acc = correct / max(1, total) * 100.0
        train_loss = total_loss / max(1, total)

        history.append({
            "epoch": actual_epoch,
            "additional_epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
        })
        print(f"[{model_name}] Epoch {actual_epoch} Finished | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

    # Save complete state checkpoint
    ckpt_payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "model_name": model_name,
        "completed_epoch": parent_saved_epoch + additional_epochs,
        "planned_epochs": parent_saved_epoch + additional_epochs,
        "is_smoke": is_smoke,
        "run_stage": run_stage,
        "train_samples": total_samples,
        "parent_saved_epoch": parent_saved_epoch,
        "additional_epochs": additional_epochs,
        "resume_kind": "weights_only_finetune",
        "parent_checkpoint_path": str(parent_ckpt_path),
        "parent_checkpoint_sha256": parent_sha256,
        "parent_selection": "official_validation_best",
        "selection_rule": "fixed_additional_last",
        "history": history,
        "rng_states": {
            "torch": torch.get_rng_state().tolist(),
            "numpy": np.random.get_state()[1].tolist(),
            "python": random.getstate()[1],
        },
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(ckpt_payload, final_path)
    print(f"[{model_name}] Checkpoint saved: {final_path} (SHA-256: {compute_sha256(final_path)})")
    return final_path


def freeze_model_config(
    output_dir: Path,
    redim_path: Path,
    ecapa_path: Path,
    resnet_path: Path,
    threshold: float = 0.50,
    allow_smoke: bool = False,
) -> Path:
    """
    Freeze final model config.
    Strictly prevents freezing smoke or incomplete models unless allow_smoke=True.
    """
    paths = {"redimnet": redim_path, "ecapa_tdnn": ecapa_path, "resnet50": resnet_path}
    models_meta = {}

    for m_name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {p}")
        data = torch.load(p, map_location="cpu", weights_only=False)
        is_smoke = data.get("is_smoke", False)
        if is_smoke and not allow_smoke:
            raise ValueError(f"Cannot freeze smoke checkpoint for production: {p}")

        models_meta[m_name] = {
            "checkpoint_path": str(p.resolve()),
            "sha256": compute_sha256(p),
            "completed_epoch": data.get("completed_epoch", 10),
            "is_smoke": is_smoke,
            "parent_saved_epoch": data.get("parent_saved_epoch", None),
            "additional_epochs": data.get("additional_epochs", None),
            "parent_selection": data.get("parent_selection", "official_validation_best"),
            "selection_rule": data.get("selection_rule", "fixed_additional_last"),
        }

    config_path = output_dir / "final_model_config.json"
    cfg = {
        "frozen_timestamp": datetime.now().isoformat(),
        "selection_rule": "fixed_additional_last",
        "threshold": threshold,
        "ensemble_weights": {
            "redimnet": 1.0 / 3.0,
            "ecapa_tdnn": 1.0 / 3.0,
            "resnet50": 1.0 / 3.0,
        },
        "preprocessing": {
            "sr": 16000,
            "window_sec": 3.0,
            "standard": "librosa_power_to_db_top80_normalized_0_1",
            "redimnet": {"n_mels": 80, "n_fft": 512, "hop_length": 160},
            "ecapa_tdnn": {"n_mels": 80, "n_fft": 512, "hop_length": 160},
            "resnet50": {"n_mels": 128, "n_fft": 2048, "hop_length": 512},
        },
        "models": models_meta,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[Frozen Config] Successfully saved: {config_path}")
    return config_path


def evaluate_official_final_unified(
    frozen_config_path: Path,
    val_data_root: Path,
    device: torch.device,
    batch_size: int = 32,
    max_files: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Evaluate official validation set strictly following frozen config parameters.
    """
    if not frozen_config_path.exists():
        raise FileNotFoundError(f"Frozen config not found: {frozen_config_path}")

    with open(frozen_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    threshold = float(cfg.get("threshold", 0.50))
    weights = cfg.get("ensemble_weights", {"redimnet": 1/3, "ecapa_tdnn": 1/3, "resnet50": 1/3})
    w_r = float(weights.get("redimnet", 1/3))
    w_e = float(weights.get("ecapa_tdnn", 1/3))
    w_res = float(weights.get("resnet50", 1/3))

    # Verify checkpoint integrity
    models = {}
    for m_name, m_info in cfg["models"].items():
        ckpt_p = Path(m_info["checkpoint_path"])
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found for {m_name}: {ckpt_p}")
        calc_sha = compute_sha256(ckpt_p)
        if calc_sha != m_info["sha256"]:
            raise ValueError(f"Hash mismatch for {m_name}: expected {m_info['sha256']}, got {calc_sha}")

        m = build_model(m_name).to(device)
        load_model_weights_adapted(m, ckpt_p, device)
        m.eval()
        models[m_name] = m

    print("[Verification] All 3 models loaded and SHA-256 verified successfully.")

    # Datasets
    val_ds_80 = DCCAudioDatasetUnified(
        val_data_root, is_train=False, n_mels=80, n_fft=512, hop_length=160, max_files=max_files
    )
    val_ds_128 = DCCAudioDatasetUnified(
        val_data_root, is_train=False, n_mels=128, n_fft=2048, hop_length=512, max_files=max_files
    )

    n_samples = len(val_ds_80)
    print(f"Validation utterances: {n_samples:,} (audit: {val_ds_80.audit_info})")

    loader_80 = DataLoader(val_ds_80, batch_size=batch_size, shuffle=False, num_workers=0)
    loader_128 = DataLoader(val_ds_128, batch_size=batch_size, shuffle=False, num_workers=0)

    preds_r, preds_e, targets_all, sample_ids = [], [], [], []
    with torch.no_grad():
        for inputs, targets, sids in tqdm(loader_80, desc="Evaluation [ReDimNet + ECAPA]"):
            inputs = inputs.to(device)
            p_r = torch.sigmoid(models["redimnet"](inputs)).cpu().numpy().squeeze(1)
            p_e = torch.sigmoid(models["ecapa_tdnn"](inputs)).cpu().numpy().squeeze(1)

            if not np.all(np.isfinite(p_r)) or not np.all(np.isfinite(p_e)):
                raise RuntimeError("Non-finite probability detected in ReDimNet or ECAPA!")

            preds_r.extend(p_r.tolist() if p_r.ndim > 0 else [p_r.item()])
            preds_e.extend(p_e.tolist() if p_e.ndim > 0 else [p_e.item()])
            targets_all.extend(targets.numpy().tolist() if targets.ndim > 0 else [targets.item()])
            sample_ids.extend(sids)

    preds_res = []
    with torch.no_grad():
        for inputs, _, _ in tqdm(loader_128, desc="Evaluation [AudioResNet-50]"):
            inputs = inputs.to(device)
            p_res = torch.sigmoid(models["resnet50"](inputs)).cpu().numpy().squeeze(1)
            if not np.all(np.isfinite(p_res)):
                raise RuntimeError("Non-finite probability detected in AudioResNet-50!")
            preds_res.extend(p_res.tolist() if p_res.ndim > 0 else [p_res.item()])

    prob_r = np.array(preds_r, dtype=np.float32)
    prob_e = np.array(preds_e, dtype=np.float32)
    prob_res = np.array(preds_res, dtype=np.float32)
    targets_arr = np.array(targets_all, dtype=np.int64)

    prob_ens = w_r * prob_r + w_e * prob_e + w_res * prob_res
    pred_ens = (prob_ens >= threshold).astype(np.int64)

    acc = accuracy_score(targets_arr, pred_ens) * 100.0
    f1 = f1_score(targets_arr, pred_ens, average="macro")
    rec_0 = recall_score(targets_arr, pred_ens, pos_label=0) * 100.0
    rec_1 = recall_score(targets_arr, pred_ens, pos_label=1) * 100.0
    cm = confusion_matrix(targets_arr, pred_ens)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    print("=" * 70)
    print(f" [Official Evaluation Results] Utterances: {len(targets_arr):,}")
    print(f" - Accuracy: {acc:.4f}% ({tn + tp:,} / {len(targets_arr):,})")
    print(f" - Macro F1: {f1:.4f}")
    print(f" - Recall (상황실 0): {rec_0:.2f}% ({tn:,} / {tn + fp:,})")
    print(f" - Recall (신고자 1): {rec_1:.2f}% ({tp:,} / {tp + fn:,})")
    print(f" - Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=" * 70)

    metrics = {
        "frozen_timestamp": cfg.get("frozen_timestamp"),
        "eval_timestamp": datetime.now().isoformat(),
        "total_evaluated": len(targets_arr),
        "threshold": threshold,
        "weights": {"redimnet": w_r, "ecapa_tdnn": w_e, "resnet50": w_res},
        "accuracy": acc,
        "macro_f1": f1,
        "recall_0_dispatch": rec_0,
        "recall_1_caller": rec_1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "official_final_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        pred_df = pd.DataFrame({
            "sample_id": sample_ids,
            "target": targets_arr,
            "pred_label": pred_ens,
            "prob_ensemble": prob_ens,
            "prob_redim": prob_r,
            "prob_ecapa": prob_e,
            "prob_resnet": prob_res,
        })
        pred_df.to_csv(output_dir / "official_final_predictions.csv", index=False, encoding="utf-8-sig")

    return metrics
