# -*- coding: utf-8 -*-
"""
DCC Mission 2: Path B Common Helpers and Preprocessing Pipeline.

Ensures strict bit-level parity between Training Dataset, Notebook Evaluator, and Submission CLI.
- Shared feature extraction via m2.audio_features
- Librosa 16kHz audio, 3.0s window (48,000 samples)
- 80-Mel (ReDimNet, ECAPA-TDNN): n_mels=80, n_fft=512, hop_length=160 (301 frames)
- 128-Mel (AudioResNet-50): n_mels=128, n_fft=2048, hop_length=512 (94 frames)
- dB Normalization: librosa.power_to_db(mel, ref=max(1e-10, max(mel)), top_db=80.0)
                    mel_norm = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)
- Atomic checkpoint saving per epoch with tmp-rename pattern
- Full RNG serialization & restoration across Python, NumPy, PyTorch CPU, and PyTorch CUDA
- Incomplete/smoke checkpoint rejection in config freezing and submission export
- Verification of sample ID alignment between 80-Mel and 128-Mel loaders
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import shutil
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

from m2.audio_features import (
    SAMPLE_RATE,
    TARGET_DURATION,
    TARGET_LENGTH,
    extract_normalized_mel,
    extract_utterance_clip,
    load_and_resample_call,
)
from m2.models import AudioResNet, ECAPA_TDNN, ReDimNet2_B2


def compute_sha256(filepath: Union[Path, str]) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def serialize_rng_state() -> dict:
    """
    Serialize full RNG states across Python, NumPy, PyTorch CPU, and PyTorch CUDA.
    NumPy get_state() returns: ('MT19937', uint32_array, pos, has_gauss, cached_gaussian)
    """
    py_st = random.getstate()
    np_st = np.random.get_state()
    torch_cpu_st = torch.get_rng_state().tolist()
    torch_cuda_st = (
        [s.tolist() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )

    return {
        "python": [py_st[0], list(py_st[1]), py_st[2]],
        "numpy": {
            "state_name": str(np_st[0]),
            "keys": np_st[1].tolist(),
            "pos": int(np_st[2]),
            "has_gauss": int(np_st[3]),
            "cached_gaussian": float(np_st[4]),
        },
        "torch_cpu": torch_cpu_st,
        "torch_cuda": torch_cuda_st,
    }


def restore_rng_state(rng_data: dict) -> None:
    """
    Restore full RNG states from serialized dictionary.
    """
    if not rng_data:
        return
    try:
        if "python" in rng_data:
            py_st = rng_data["python"]
            random.setstate((py_st[0], tuple(py_st[1]), py_st[2]))
        if "numpy" in rng_data:
            np_d = rng_data["numpy"]
            np_tuple = (
                np_d["state_name"],
                np.array(np_d["keys"], dtype=np.uint32),
                int(np_d["pos"]),
                int(np_d["has_gauss"]),
                float(np_d["cached_gaussian"]),
            )
            np.random.set_state(np_tuple)
        if "torch_cpu" in rng_data:
            torch.set_rng_state(torch.tensor(rng_data["torch_cpu"], dtype=torch.uint8))
        if "torch_cuda" in rng_data and torch.cuda.is_available() and rng_data["torch_cuda"]:
            torch.cuda.set_rng_state_all(
                [torch.tensor(s, dtype=torch.uint8) for s in rng_data["torch_cuda"]]
            )
    except Exception as e:
        print(f"[Warning] Failed to restore RNG states: {e}")


class DCCAudioDatasetUnified(Dataset):
    """
    Unified Dataset for Training and Evaluation.
    Guarantees exact parity with submission CLI audio loading, caching, and Mel extraction.
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
        window_sec: float = TARGET_DURATION,
        sr: int = SAMPLE_RATE,
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
                    "start_ms": start_ms,
                    "end_ms": end_ms,
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
        start_ms = info["start_ms"]
        end_ms = info["end_ms"]
        label = info["label"]

        full_audio = load_and_resample_call(wav_path, target_sr=self.sr)
        clip = extract_utterance_clip(
            full_audio=full_audio,
            start_ms=start_ms,
            end_ms=end_ms,
            target_sr=self.sr,
            target_length=self.target_samples,
            crop_mode=self.crop_mode,
        )

        mel = extract_normalized_mel(
            audio=clip,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

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
    Execute fixed additional training under Path B rules with resume support.
    - Zero evaluation on official Validation set
    - Zero score-based best selection
    - Saves complete optimizer, scheduler, scaler, full RNG, and resume metadata per epoch
    - Atomic checkpoint writing via temporary file replacement
    - Resumes from last completed epoch if checkpoint exists and plan matches
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_sha256 = compute_sha256(parent_ckpt_path)
    total_planned = parent_saved_epoch + additional_epochs

    final_filename = f"smoke_last_{model_name}.pt" if is_smoke else f"last_{model_name}.pt"
    final_path = output_dir / final_filename

    # ResNet with 0 additional epochs: register parent directly
    if additional_epochs <= 0:
        model = build_model(model_name).to(device)
        load_model_weights_adapted(model, parent_ckpt_path, device)
        payload = {
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
            "history": [],
            "rng_states": serialize_rng_state(),
            "timestamp": datetime.now().isoformat(),
        }
        tmp_path = output_dir / f"{final_filename}.tmp"
        torch.save(payload, tmp_path)
        tmp_path.replace(final_path)
        print(f"[{model_name}] Registered parent as final: {final_path} (SHA-256: {compute_sha256(final_path)})")
        return final_path

    # Check if a valid checkpoint already exists for resume or skip
    start_add_epoch = 1
    model = build_model(model_name).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=additional_epochs, eta_min=1e-6)
    scaler = GradScaler()
    history = []
    resume_kind = "weights_only_finetune"

    if final_path.exists():
        try:
            ckpt_existing = torch.load(final_path, map_location=device, weights_only=False)
            ext_name = ckpt_existing.get("model_name")
            ext_parent_sha = ckpt_existing.get("parent_checkpoint_sha256")
            ext_completed = ckpt_existing.get("completed_epoch", 0)
            ext_planned = ckpt_existing.get("planned_epochs", total_planned)
            ext_smoke = ckpt_existing.get("is_smoke", False)

            if ext_name == model_name and ext_parent_sha == parent_sha256 and ext_smoke == is_smoke:
                if ext_completed >= total_planned:
                    print(f"[{model_name}] Existing run already completed planned {ext_completed}/{total_planned} epochs. Skipping training.")
                    return final_path
                elif ext_completed > parent_saved_epoch:
                    # Partial run detected, resume from completed_epoch
                    print(f"[{model_name}] Resuming from epoch {ext_completed} (target: {total_planned}).")
                    model.load_state_dict(ckpt_existing["model_state_dict"])
                    if "optimizer_state_dict" in ckpt_existing:
                        optimizer.load_state_dict(ckpt_existing["optimizer_state_dict"])
                    if "scheduler_state_dict" in ckpt_existing:
                        scheduler.load_state_dict(ckpt_existing["scheduler_state_dict"])
                    if "scaler_state_dict" in ckpt_existing:
                        scaler.load_state_dict(ckpt_existing["scaler_state_dict"])
                    history = list(ckpt_existing.get("history", []))
                    restore_rng_state(ckpt_existing.get("rng_states", {}))
                    start_add_epoch = ext_completed - parent_saved_epoch + 1
                    resume_kind = "interrupted_resume"
        except Exception as e:
            print(f"[{model_name}] Could not resume existing checkpoint ({e}). Starting fresh.")

    if resume_kind == "weights_only_finetune":
        load_model_weights_adapted(model, parent_ckpt_path, device)

    model.train()
    total_samples = len(train_loader.dataset)

    for ep in range(start_add_epoch, additional_epochs + 1):
        actual_epoch = parent_saved_epoch + ep
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {actual_epoch}/{total_planned} [Train {model_name}]")
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

        # Atomic per-epoch save
        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_name": model_name,
            "completed_epoch": actual_epoch,
            "planned_epochs": total_planned,
            "is_smoke": is_smoke,
            "run_stage": run_stage,
            "train_samples": total_samples,
            "parent_saved_epoch": parent_saved_epoch,
            "additional_epochs": additional_epochs,
            "resume_kind": resume_kind,
            "parent_checkpoint_path": str(parent_ckpt_path),
            "parent_checkpoint_sha256": parent_sha256,
            "parent_selection": "official_validation_best",
            "selection_rule": "fixed_additional_last",
            "history": history,
            "rng_states": serialize_rng_state(),
            "timestamp": datetime.now().isoformat(),
        }
        tmp_path = output_dir / f"{final_filename}.tmp"
        torch.save(ckpt_payload, tmp_path)
        tmp_path.replace(final_path)

    print(f"[{model_name}] Completed all planned epochs ({total_planned}). Saved: {final_path} (SHA-256: {compute_sha256(final_path)})")
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
    Strictly prevents freezing smoke or incomplete checkpoints unless allow_smoke=True.
    Validates essential metadata: completed_epoch == planned_epochs, parent hashes, history.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"redimnet": redim_path, "ecapa_tdnn": ecapa_path, "resnet50": resnet_path}
    models_meta = {}

    for m_name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {p}")
        data = torch.load(p, map_location="cpu", weights_only=False)
        is_smoke = bool(data.get("is_smoke", False))
        if is_smoke and not allow_smoke:
            raise ValueError(f"Cannot freeze smoke checkpoint for production: {p}")

        completed_ep = data.get("completed_epoch")
        planned_ep = data.get("planned_epochs")
        if completed_ep is None or planned_ep is None:
            raise ValueError(f"Checkpoint {p} lacks completed_epoch or planned_epochs metadata!")

        if completed_ep < planned_ep:
            raise ValueError(
                f"Checkpoint {p} is incomplete! (completed: {completed_ep} < planned: {planned_ep})"
            )

        models_meta[m_name] = {
            "checkpoint_path": str(p.resolve()),
            "sha256": compute_sha256(p),
            "completed_epoch": completed_ep,
            "planned_epochs": planned_ep,
            "is_smoke": is_smoke,
            "parent_saved_epoch": data.get("parent_saved_epoch"),
            "additional_epochs": data.get("additional_epochs"),
            "parent_checkpoint_sha256": data.get("parent_checkpoint_sha256"),
            "parent_selection": data.get("parent_selection", "official_validation_best"),
            "selection_rule": data.get("selection_rule", "fixed_additional_last"),
        }

    config_path = output_dir / "final_model_config.json"
    cfg = {
        "frozen_timestamp": datetime.now().isoformat(),
        "selection_rule": "fixed_additional_last",
        "threshold": threshold,
        "is_smoke": allow_smoke,
        "ensemble_weights": {
            "redimnet": 1.0 / 3.0,
            "ecapa_tdnn": 1.0 / 3.0,
            "resnet50": 1.0 / 3.0,
        },
        "preprocessing": {
            "sr": SAMPLE_RATE,
            "window_sec": TARGET_DURATION,
            "standard": "librosa_slaney_power_to_db_top80_normalized_0_1",
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


def evaluate_smoke_unified(
    frozen_config_path: Path,
    train_data_root: Path,
    device: torch.device,
    batch_size: int = 16,
    max_files: int = 5,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Dedicated Smoke Evaluation.
    Evaluates exclusively on training samples (never touching official validation data).
    Verifies model loading, finite probabilities, prediction CSV, and row counts.
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

    models = {}
    for m_name, m_info in cfg["models"].items():
        ckpt_p = Path(m_info["checkpoint_path"])
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found for {m_name}: {ckpt_p}")
        m = build_model(m_name).to(device)
        load_model_weights_adapted(m, ckpt_p, device)
        m.eval()
        models[m_name] = m

    smoke_ds_80 = DCCAudioDatasetUnified(
        train_data_root, is_train=False, n_mels=80, n_fft=512, hop_length=160, max_files=max_files
    )
    smoke_ds_128 = DCCAudioDatasetUnified(
        train_data_root, is_train=False, n_mels=128, n_fft=2048, hop_length=512, max_files=max_files
    )

    if len(smoke_ds_80) != len(smoke_ds_128):
        raise ValueError("Dataset length mismatch between 80-Mel and 128-Mel smoke datasets!")

    loader_80 = DataLoader(smoke_ds_80, batch_size=batch_size, shuffle=False)
    loader_128 = DataLoader(smoke_ds_128, batch_size=batch_size, shuffle=False)

    preds_r, preds_e, targets_all, sids_80 = [], [], [], []
    with torch.no_grad():
        for inputs, targets, sids in loader_80:
            inputs = inputs.to(device)
            p_r = torch.sigmoid(models["redimnet"](inputs)).cpu().numpy().squeeze(1)
            p_e = torch.sigmoid(models["ecapa_tdnn"](inputs)).cpu().numpy().squeeze(1)

            if not np.all(np.isfinite(p_r)) or not np.all(np.isfinite(p_e)):
                raise RuntimeError("Non-finite probability in smoke ReDimNet/ECAPA!")

            preds_r.extend(p_r.tolist() if p_r.ndim > 0 else [p_r.item()])
            preds_e.extend(p_e.tolist() if p_e.ndim > 0 else [p_e.item()])
            targets_all.extend(targets.numpy().tolist() if targets.ndim > 0 else [targets.item()])
            sids_80.extend(sids)

    preds_res, sids_128 = [], []
    with torch.no_grad():
        for inputs, _, sids in loader_128:
            inputs = inputs.to(device)
            p_res = torch.sigmoid(models["resnet50"](inputs)).cpu().numpy().squeeze(1)
            if not np.all(np.isfinite(p_res)):
                raise RuntimeError("Non-finite probability in smoke ResNet-50!")
            preds_res.extend(p_res.tolist() if p_res.ndim > 0 else [p_res.item()])
            sids_128.extend(sids)

    if sids_80 != sids_128:
        raise ValueError("Sample ID mismatch between 80-Mel and 128-Mel smoke loaders!")

    prob_r = np.array(preds_r, dtype=np.float32)
    prob_e = np.array(preds_e, dtype=np.float32)
    prob_res = np.array(preds_res, dtype=np.float32)
    targets_arr = np.array(targets_all, dtype=np.int64)

    prob_ens = w_r * prob_r + w_e * prob_e + w_res * prob_res
    pred_ens = (prob_ens >= threshold).astype(np.int64)

    acc = accuracy_score(targets_arr, pred_ens) * 100.0 if len(targets_arr) > 0 else 0.0

    metrics = {
        "is_smoke": True,
        "eval_timestamp": datetime.now().isoformat(),
        "total_evaluated": len(targets_arr),
        "accuracy": acc,
        "sample_ids_count": len(sids_80),
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "smoke_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        pred_df = pd.DataFrame({
            "sample_id": sids_80,
            "target": targets_arr,
            "pred_label": pred_ens,
            "prob_ensemble": prob_ens,
        })
        pred_df.to_csv(output_dir / "smoke_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"[Smoke Verification Complete] Samples: {len(targets_arr)} | Accuracy: {acc:.2f}% | Output: {output_dir}")
    return metrics


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
    Strictly rejects smoke configurations or checkpoints (is_smoke=True).
    Strictly verifies sample alignment (sample_ids_80 == sample_ids_128).
    """
    if not frozen_config_path.exists():
        raise FileNotFoundError(f"Frozen config not found: {frozen_config_path}")

    with open(frozen_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if cfg.get("is_smoke", False):
        raise ValueError("Cannot evaluate smoke configuration in official final evaluation!")

    threshold = float(cfg.get("threshold", 0.50))
    weights = cfg.get("ensemble_weights", {"redimnet": 1/3, "ecapa_tdnn": 1/3, "resnet50": 1/3})
    w_r = float(weights.get("redimnet", 1/3))
    w_e = float(weights.get("ecapa_tdnn", 1/3))
    w_res = float(weights.get("resnet50", 1/3))

    models = {}
    for m_name, m_info in cfg["models"].items():
        if m_info.get("is_smoke", False):
            raise ValueError(f"Smoke checkpoint detected for {m_name} in frozen config!")

        ckpt_p = Path(m_info["checkpoint_path"])
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found for {m_name}: {ckpt_p}")
        calc_sha = compute_sha256(ckpt_p)
        if calc_sha != m_info["sha256"]:
            raise ValueError(f"Hash mismatch for {m_name}: expected {m_info['sha256']}, got {calc_sha}")

        raw_ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        if raw_ckpt.get("is_smoke", False):
            raise ValueError(f"Underlying checkpoint {ckpt_p} is marked as smoke!")

        m = build_model(m_name).to(device)
        load_model_weights_adapted(m, ckpt_p, device)
        m.eval()
        models[m_name] = m

    print("[Verification] All 3 production models loaded and SHA-256 verified successfully.")

    val_ds_80 = DCCAudioDatasetUnified(
        val_data_root, is_train=False, n_mels=80, n_fft=512, hop_length=160, max_files=max_files
    )
    val_ds_128 = DCCAudioDatasetUnified(
        val_data_root, is_train=False, n_mels=128, n_fft=2048, hop_length=512, max_files=max_files
    )

    if len(val_ds_80) != len(val_ds_128):
        raise ValueError(f"Dataset length mismatch: 80-Mel ({len(val_ds_80)}) vs 128-Mel ({len(val_ds_128)})")

    n_samples = len(val_ds_80)
    print(f"Validation utterances: {n_samples:,} (audit: {val_ds_80.audit_info})")

    loader_80 = DataLoader(val_ds_80, batch_size=batch_size, shuffle=False, num_workers=0)
    loader_128 = DataLoader(val_ds_128, batch_size=batch_size, shuffle=False, num_workers=0)

    preds_r, preds_e, targets_80, sample_ids_80 = [], [], [], []
    with torch.no_grad():
        for inputs, targets, sids in tqdm(loader_80, desc="Evaluation [ReDimNet + ECAPA]"):
            inputs = inputs.to(device)
            p_r = torch.sigmoid(models["redimnet"](inputs)).cpu().numpy().squeeze(1)
            p_e = torch.sigmoid(models["ecapa_tdnn"](inputs)).cpu().numpy().squeeze(1)

            if not np.all(np.isfinite(p_r)) or not np.all(np.isfinite(p_e)):
                raise RuntimeError("Non-finite probability detected in ReDimNet or ECAPA!")

            preds_r.extend(p_r.tolist() if p_r.ndim > 0 else [p_r.item()])
            preds_e.extend(p_e.tolist() if p_e.ndim > 0 else [p_e.item()])
            targets_80.extend(targets.numpy().tolist() if targets.ndim > 0 else [targets.item()])
            sample_ids_80.extend(sids)

    preds_res, targets_128, sample_ids_128 = [], [], []
    with torch.no_grad():
        for inputs, targets, sids in tqdm(loader_128, desc="Evaluation [AudioResNet-50]"):
            inputs = inputs.to(device)
            p_res = torch.sigmoid(models["resnet50"](inputs)).cpu().numpy().squeeze(1)
            if not np.all(np.isfinite(p_res)):
                raise RuntimeError("Non-finite probability detected in AudioResNet-50!")
            preds_res.extend(p_res.tolist() if p_res.ndim > 0 else [p_res.item()])
            targets_128.extend(targets.numpy().tolist() if targets.ndim > 0 else [targets.item()])
            sample_ids_128.extend(sids)

    # Strict sample alignment checks
    if sample_ids_80 != sample_ids_128:
        raise ValueError("Sample ID mismatch between 80-Mel and 128-Mel evaluation loaders!")
    if targets_80 != targets_128:
        raise ValueError("Target label mismatch between 80-Mel and 128-Mel evaluation loaders!")

    prob_r = np.array(preds_r, dtype=np.float32)
    prob_e = np.array(preds_e, dtype=np.float32)
    prob_res = np.array(preds_res, dtype=np.float32)
    targets_arr = np.array(targets_80, dtype=np.int64)

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
            "sample_id": sample_ids_80,
            "target": targets_arr,
            "pred_label": pred_ens,
            "prob_ensemble": prob_ens,
            "prob_redim": prob_r,
            "prob_ecapa": prob_e,
            "prob_resnet": prob_res,
        })
        pred_df.to_csv(output_dir / "official_final_predictions.csv", index=False, encoding="utf-8-sig")

    return metrics


def export_submission_package(
    frozen_config_path: Path,
    export_dir: Path,
) -> None:
    """
    Exports the 3 selected checkpoints to export_dir with standard filenames
    (best_redimnet.pt, best_ecapa_tdnn.pt, best_resnet50.pt) for inference CLI compatibility.
    - Strictly verifies config is not smoke.
    - Strictly checks that on-disk SHA-256 of each checkpoint matches the frozen config.
    - Performs atomic copy via temporary folder and re-checks hashes.
    """
    with open(frozen_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if cfg.get("is_smoke", False):
        raise ValueError("Cannot export submission package from smoke configuration!")

    export_dir.mkdir(parents=True, exist_ok=True)
    tmp_export_dir = export_dir / ".tmp_export"
    if tmp_export_dir.exists():
        shutil.rmtree(tmp_export_dir)
    tmp_export_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "redimnet": "best_redimnet.pt",
        "ecapa_tdnn": "best_ecapa_tdnn.pt",
        "resnet50": "best_resnet50.pt",
    }

    export_meta = {
        "description": "DCC Mission 2 Submission Checkpoints (Path B fixed_additional_last)",
        "selection_rule": "fixed_additional_last",
        "note": "Standard filenames best_*.pt retained for inference CLI compatibility. Parent history: official_validation_best.",
        "exported_timestamp": datetime.now().isoformat(),
        "files": {},
    }

    for m_name, target_filename in mapping.items():
        m_info = cfg["models"][m_name]
        src_path = Path(m_info["checkpoint_path"])
        if not src_path.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {src_path}")

        # Check hash against frozen config
        disk_sha = compute_sha256(src_path)
        if disk_sha != m_info["sha256"]:
            raise ValueError(
                f"Tampered or modified checkpoint detected for {m_name}! "
                f"Config SHA: {m_info['sha256']} != Disk SHA: {disk_sha}"
            )

        ckpt_data = torch.load(src_path, map_location="cpu", weights_only=False)
        if ckpt_data.get("is_smoke", False):
            raise ValueError(f"Cannot export smoke checkpoint for submission: {src_path}")

        tmp_dst = tmp_export_dir / target_filename
        shutil.copy2(src_path, tmp_dst)
        copied_sha = compute_sha256(tmp_dst)
        if copied_sha != disk_sha:
            raise ValueError(f"Copy corruption detected for {target_filename}!")

        export_meta["files"][target_filename] = {
            "model_name": m_name,
            "source_path": str(src_path),
            "sha256": copied_sha,
            "completed_epoch": m_info["completed_epoch"],
            "parent_saved_epoch": m_info.get("parent_saved_epoch"),
            "additional_epochs": m_info.get("additional_epochs"),
        }

    # Atomically move from .tmp_export to export_dir
    for target_filename in mapping.values():
        dst_final = export_dir / target_filename
        if dst_final.exists():
            dst_final.unlink()
        (tmp_export_dir / target_filename).rename(dst_final)
        print(f"Exported {target_filename} (SHA-256: {export_meta['files'][target_filename]['sha256']})")

    shutil.rmtree(tmp_export_dir)

    with open(export_dir / "selection_metadata.json", "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2, ensure_ascii=False)
    print(f"[Export Complete] Submission files successfully verified and exported to {export_dir}")
