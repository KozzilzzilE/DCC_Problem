# -*- coding: utf-8 -*-
"""
DCC Mission 2: Path B Common Helpers and Preprocessing Pipeline.

Shares evaluation/CLI preprocessing; Training uses a separate random crop policy.
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
import tempfile
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
        raise ValueError(f"Cannot restore checkpoint RNG state: {e}") from e


MODEL_PLAN = {"redimnet": (9, 1), "ecapa_tdnn": (8, 2), "resnet50": (10, 0)}


def validate_training_root(data_dir: Union[str, Path], official_val_root=None) -> None:
    """Reject known validation roots without reading any validation files."""
    root = Path(data_dir).resolve()
    if any(p.lower() in {"val", "validation", "validation_data"}
           or p.lower().startswith("validation[") for p in root.parts):
        raise ValueError(f"Training/smoke cannot use a Validation root: {root}")
    if official_val_root is not None:
        val = Path(official_val_root).resolve()
        if root == val or root in val.parents or val in root.parents:
            raise ValueError("Training and official Validation roots must be separate")


def dataset_identity(dataset) -> dict:
    """Fingerprint sample mapping and source file stats; not an audio content hash."""
    if not hasattr(dataset, "samples"):
        identity = getattr(dataset, "data_identity", None)
        if not isinstance(identity, dict) or not identity.get("sha256"):
            raise ValueError("Dataset requires a reproducible data_identity or sample manifest")
        return {**identity, "sample_count": len(dataset)}
    sources = {}
    fingerprint = hashlib.sha256()
    for s in dataset.samples:
        path = s["wav_path"]
        if path not in sources:
            wav = Path(path).resolve()
            st = wav.stat()
            sources[path] = [str(wav), st.st_size, st.st_mtime_ns]
        record = {k: s[k] for k in ("sample_id", "call_id", "start_ms", "end_ms", "label")}
        record["source"] = sources[path]
        fingerprint.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
    return {"sha256": fingerprint.hexdigest(), "sample_count": len(dataset.samples),
            "method": "sample_mapping_and_audio_size_mtime", "audio_content_hashed": False}


def _training_contract(model_name, parent_hash, parent_epoch, additional_epochs,
                       loader, lr, is_smoke, seed, device):
    ds = loader.dataset
    return {
        "schema_version": 2, "model_name": model_name,
        "parent_checkpoint_sha256": parent_hash, "parent_saved_epoch": parent_epoch,
        "additional_epochs": additional_epochs, "is_smoke": is_smoke, "seed": seed,
        "data": dataset_identity(ds) if additional_epochs else None,
        "preprocessing": ({k: getattr(ds, k, None) for k in
                          ("sr", "window_sec", "n_mels", "n_fft", "hop_length", "crop_mode")}
                          if additional_epochs else None),
        "loader": {"batch_size": loader.batch_size, "drop_last": loader.drop_last,
                   "num_workers": loader.num_workers, "sampler": type(loader.sampler).__name__}
                   if additional_epochs else None,
        "optimizer": {"name": "Adam", "lr": lr},
        "scheduler": {"name": "CosineAnnealingLR", "T_max": additional_epochs, "eta_min": 1e-6},
        "execution": {"device_type": device.type, "torch_version": str(torch.__version__),
                      "numpy_version": np.__version__, "librosa_version": librosa.__version__,
                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                      "cudnn_deterministic": torch.backends.cudnn.deterministic},
        "source_sha256": {name: compute_sha256(Path(__file__).parent / name)
                          for name in ("path_b.py", "audio_features.py", "models.py")},
    }


def _atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _loader_rng(loader) -> dict:
    return {name: generator.get_state().tolist() for name, generator in
            (("loader", loader.generator), ("sampler", getattr(loader.sampler, "generator", None)))
            if generator is not None}


def _restore_loader_rng(loader, saved: dict) -> None:
    for name, generator in (("loader", loader.generator),
                            ("sampler", getattr(loader.sampler, "generator", None))):
        if generator is not None:
            if name not in saved:
                raise ValueError(f"Missing DataLoader generator state: {name}")
            generator.set_state(torch.tensor(saved[name], dtype=torch.uint8))


def _validate_checkpoint(data, model_name, allow_smoke=False, require_complete=True):
    required = {"model_state_dict", "model_name", "completed_epoch", "planned_epochs",
                "is_smoke", "parent_saved_epoch", "additional_epochs", "parent_checkpoint_sha256",
                "parent_selection", "selection_rule", "history", "training_contract", "train_samples"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Checkpoint lacks verified run metadata: {sorted(missing)}")
    if data["model_name"] != model_name or type(data["is_smoke"]) is not bool:
        raise ValueError("Invalid checkpoint model/smoke metadata")
    if data["is_smoke"] and not allow_smoke:
        raise ValueError("Cannot use a smoke checkpoint for production")
    parent, extra = MODEL_PLAN[model_name]
    if (data["parent_saved_epoch"], data["additional_epochs"]) != (parent, extra):
        raise ValueError(f"Checkpoint does not match the fixed plan for {model_name}")
    completed = data["completed_epoch"]
    if data["planned_epochs"] != parent + extra or not parent <= completed <= parent + extra:
        raise ValueError("Invalid completed/planned epoch metadata")
    if require_complete and completed != parent + extra:
        raise ValueError("Cannot freeze/export an incomplete checkpoint")
    if data["selection_rule"] != "fixed_additional_last" or data["parent_selection"] != "official_validation_best":
        raise ValueError("Checkpoint selection/source history is missing or inconsistent")
    sha = data["parent_checkpoint_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Invalid parent checkpoint SHA-256")
    contract = data["training_contract"]
    if (contract.get("schema_version") != 2 or contract.get("model_name") != model_name
            or contract.get("parent_checkpoint_sha256") != sha
            or contract.get("parent_saved_epoch") != parent
            or contract.get("additional_epochs") != extra
            or contract.get("is_smoke") != data["is_smoke"]):
        raise ValueError("Checkpoint contract does not match its metadata")
    history = data["history"]
    if [h.get("epoch") for h in history] != list(range(parent + 1, completed + 1)):
        raise ValueError("Incomplete or inconsistent additional-epoch history")
    if extra:
        if data["train_samples"] <= 0 or contract.get("data", {}).get("sample_count") != data["train_samples"]:
            raise ValueError("Checkpoint lacks a nonempty Training sample manifest")
        for h in history:
            if h.get("samples_seen", 0) <= 0 or h.get("optimizer_steps", 0) <= 0:
                raise ValueError("Checkpoint lacks actual Training sample/step counts")
            expected = data["train_samples"]
            loader = contract["loader"]
            if loader["drop_last"]:
                expected = (expected // loader["batch_size"]) * loader["batch_size"]
            if h["samples_seen"] != expected:
                raise ValueError("Checkpoint history does not cover the planned epoch sample count")
        for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng_states", "loader_rng_states"):
            if key not in data:
                raise ValueError(f"Missing resume state: {key}")
        if not {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(data["rng_states"]):
            raise ValueError("Checkpoint lacks complete RNG states")
    elif data.get("resume_kind") != "direct_registered_parent" or data["train_samples"] != 0:
        raise ValueError("Zero-epoch ResNet must be an explicit parent registration")


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
        if is_train:
            validate_training_root(self.data_dir)
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
        ids = [s["sample_id"] for s in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate call/utterance IDs in Dataset")

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
            except Exception as exc:
                raise ValueError(f"Cannot parse label JSON: {jf}") from exc

            stem = jf.stem
            if self.is_train and stem.startswith(("VS_", "VL_")):
                raise ValueError(f"Validation-labelled call in Training root: {jf}")
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
                    raise ValueError(f"Unknown speaker label in {jf}, utterance {utt_idx}: {spk}")

                start_ms = float(utt.get("startAt", 0.0))
                end_ms = float(utt.get("endAt", 0.0))
                if not np.isfinite([start_ms, end_ms]).all() or start_ms < 0 or end_ms <= start_ms:
                    raise ValueError(f"Invalid utterance boundaries in {jf}, utterance {utt_idx}")

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
    seed: int = 42,
) -> Path:
    """Run the fixed plan; resume only a matching run, preserving every finished epoch."""
    if MODEL_PLAN.get(model_name) != (parent_saved_epoch, additional_epochs):
        raise ValueError("Model epochs must match the predeclared Path B plan")
    if train_loader.num_workers != 0:
        raise ValueError("Exact epoch-boundary resume currently requires num_workers=0")
    if additional_epochs and not len(train_loader.dataset):
        raise ValueError("Training dataset is empty")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be positive and finite")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_ckpt_path = Path(parent_ckpt_path)
    parent_sha = compute_sha256(parent_ckpt_path)
    contract = _training_contract(model_name, parent_sha, parent_saved_epoch,
                                  additional_epochs, train_loader, lr, is_smoke, seed, device)
    filename = f"smoke_last_{model_name}.pt" if is_smoke else f"last_{model_name}.pt"
    final_path = output_dir / filename
    total_planned = parent_saved_epoch + additional_epochs
    existing = None
    if final_path.exists():
        # Never silently replace a corrupt, incompatible, or unverified earlier run.
        existing = torch.load(final_path, map_location="cpu", weights_only=False)
        _validate_checkpoint(existing, model_name, allow_smoke=is_smoke, require_complete=False)
        if existing["training_contract"] != contract:
            raise ValueError("Existing run has a different parent/data/plan/environment. Use a new output directory.")
        if existing["completed_epoch"] == total_planned:
            print(f"[{model_name}] Completed run reused without overwriting: {final_path}")
            return final_path

    # Separate model seeds make runner and notebook order independent.
    model_seed = seed + list(MODEL_PLAN).index(model_name)
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    model = build_model(model_name).to(device)
    base = {
        "schema_version": 2, "model_name": model_name,
        "planned_epochs": total_planned, "is_smoke": is_smoke, "run_stage": run_stage,
        "parent_saved_epoch": parent_saved_epoch, "additional_epochs": additional_epochs,
        "parent_checkpoint_path": str(parent_ckpt_path.resolve()),
        "parent_checkpoint_sha256": parent_sha, "parent_selection": "official_validation_best",
        "selection_rule": "fixed_additional_last", "training_contract": contract,
        "train_samples": len(train_loader.dataset) if additional_epochs else 0,
    }
    if not additional_epochs:
        load_model_weights_adapted(model, parent_ckpt_path, device)
        _atomic_save({**base, "model_state_dict": model.state_dict(),
                      "completed_epoch": parent_saved_epoch, "history": [],
                      "resume_kind": "direct_registered_parent", "timestamp": datetime.now().isoformat()}, final_path)
        return final_path

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=additional_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=device.type == "cuda")
    history = []
    start_add_epoch = 1
    if existing:
        model.load_state_dict(existing["model_state_dict"], strict=True)
        optimizer.load_state_dict(existing["optimizer_state_dict"])
        scheduler.load_state_dict(existing["scheduler_state_dict"])
        scaler.load_state_dict(existing["scaler_state_dict"])
        history = list(existing["history"])
        restore_rng_state(existing["rng_states"])
        _restore_loader_rng(train_loader, existing["loader_rng_states"])
        start_add_epoch = existing["completed_epoch"] - parent_saved_epoch + 1
    else:
        load_model_weights_adapted(model, parent_ckpt_path, device)
    model.train()
    for ep in range(start_add_epoch, additional_epochs + 1):
        actual_epoch = parent_saved_epoch + ep
        total_loss = correct = total = steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {actual_epoch}/{total_planned} [Train {model_name}]")
        for inputs, targets, sample_ids in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            with autocast(enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits, targets)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Training output/loss for {model_name}: {list(sample_ids)}")
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                raise RuntimeError("Non-finite gradients; current epoch was not marked complete")
            total_loss += loss.item() * len(targets)
            correct += ((torch.sigmoid(logits) >= .5).float() == targets).sum().item()
            total += len(targets)
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if not total or not steps:
            raise ValueError("Training epoch contains no samples or optimizer steps")
        expected_samples = len(train_loader.dataset)
        if train_loader.drop_last:
            expected_samples = (expected_samples // train_loader.batch_size) * train_loader.batch_size
        if total != expected_samples:
            raise ValueError("Training epoch did not process the planned sample count")
        if any(t.is_floating_point() and not torch.isfinite(t).all() for t in model.state_dict().values()):
            raise RuntimeError("Non-finite model state; current epoch was not marked complete")
        scheduler.step()
        history.append({"epoch": actual_epoch, "additional_epoch": ep,
                        "train_loss": total_loss / total, "train_acc": correct / total * 100,
                        "samples_seen": total, "optimizer_steps": steps})
        payload = {**base, "model_state_dict": model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(),
                   "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                   "completed_epoch": actual_epoch, "history": history,
                   "resume_kind": "weights_only_finetune", "resumed_from_epoch": existing["completed_epoch"] if existing else None,
                   "rng_states": serialize_rng_state(), "loader_rng_states": _loader_rng(train_loader),
                   "timestamp": datetime.now().isoformat()}
        _atomic_save(payload, final_path)
        print(f"[{model_name}] Saved completed epoch {actual_epoch}: {final_path}")
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
    if threshold != 0.50:
        raise ValueError("Path B threshold is fixed at 0.50")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"redimnet": redim_path, "ecapa_tdnn": ecapa_path, "resnet50": resnet_path}
    models_meta = {}

    for m_name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {p}")
        data = torch.load(p, map_location="cpu", weights_only=False)
        _validate_checkpoint(data, m_name, allow_smoke=allow_smoke)
        is_smoke = bool(data.get("is_smoke", False))
        if is_smoke and not allow_smoke:
            raise ValueError(f"Cannot freeze smoke checkpoint for production: {p}")

        completed_ep = data.get("completed_epoch")
        planned_ep = data.get("planned_epochs")
        if completed_ep is None or planned_ep is None:
            raise ValueError(f"Checkpoint {p} lacks completed_epoch or planned_epochs metadata!")

        if completed_ep != planned_ep:
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
            "training_contract": data["training_contract"],
            "train_samples": data["train_samples"],
        }

    config_path = output_dir / "final_model_config.json"
    cfg = {
        "schema_version": 2,
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
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        saved.pop("frozen_timestamp", None)
        current = dict(cfg)
        current.pop("frozen_timestamp", None)
        if saved != current:
            raise ValueError("Existing frozen configuration differs; use a new run directory")
        return config_path
    tmp = config_path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(config_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"[Frozen Config] Successfully saved: {config_path}")
    return config_path


def _check_eval_config(cfg: dict, production: bool) -> None:
    if cfg.get("schema_version") != 2:
        raise ValueError("Freeze completed schema-2 checkpoints before evaluation/export")
    if production and cfg.get("is_smoke", True):
        raise ValueError("Cannot use smoke configuration for official evaluation/export")
    if cfg.get("threshold") != 0.50 or set(cfg.get("models", {})) != set(MODEL_PLAN):
        raise ValueError("Frozen configuration must contain all 3 models at threshold 0.50")
    weights = cfg.get("ensemble_weights", {})
    if set(weights) != set(MODEL_PLAN) or any(weights[n] != 1 / 3 for n in MODEL_PLAN):
        raise ValueError("Path B ensemble weights are fixed at 1/3 each")
    expected_preprocessing = {
        "sr": SAMPLE_RATE, "window_sec": TARGET_DURATION,
        "standard": "librosa_slaney_power_to_db_top80_normalized_0_1",
        "redimnet": {"n_mels": 80, "n_fft": 512, "hop_length": 160},
        "ecapa_tdnn": {"n_mels": 80, "n_fft": 512, "hop_length": 160},
        "resnet50": {"n_mels": 128, "n_fft": 2048, "hop_length": 512},
    }
    if cfg.get("preprocessing") != expected_preprocessing:
        raise ValueError("Unsupported or changed frozen preprocessing settings")


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
    validate_training_root(train_data_root)
    if not frozen_config_path.exists():
        raise FileNotFoundError(f"Frozen config not found: {frozen_config_path}")

    with open(frozen_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _check_eval_config(cfg, production=False)

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
        if compute_sha256(ckpt_p) != m_info["sha256"]:
            raise ValueError(f"Smoke checkpoint hash mismatch for {m_name}")
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

    preds_res, sids_128, targets_128 = [], [], []
    with torch.no_grad():
        for inputs, targets, sids in loader_128:
            inputs = inputs.to(device)
            p_res = torch.sigmoid(models["resnet50"](inputs)).cpu().numpy().squeeze(1)
            if not np.all(np.isfinite(p_res)):
                raise RuntimeError("Non-finite probability in smoke ResNet-50!")
            preds_res.extend(p_res.tolist() if p_res.ndim > 0 else [p_res.item()])
            sids_128.extend(sids)
            targets_128.extend(targets.numpy().tolist())

    if sids_80 != sids_128:
        raise ValueError("Sample ID mismatch between 80-Mel and 128-Mel smoke loaders!")
    if targets_all != targets_128:
        raise ValueError("Target label mismatch between smoke loaders")

    prob_r = np.array(preds_r, dtype=np.float64)
    prob_e = np.array(preds_e, dtype=np.float64)
    prob_res = np.array(preds_res, dtype=np.float64)
    targets_arr = np.array(targets_all, dtype=np.int64)

    prob_ens = np.mean(np.stack([prob_r, prob_e, prob_res]), axis=0)
    if not np.all(np.isfinite(prob_ens)):
        raise RuntimeError("Non-finite ensemble probability")
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
    _check_eval_config(cfg, production=True)

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
        _validate_checkpoint(raw_ckpt, m_name)
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
    if not n_samples:
        raise ValueError("No valid utterances in official evaluation dataset")
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

    prob_r = np.array(preds_r, dtype=np.float64)
    prob_e = np.array(preds_e, dtype=np.float64)
    prob_res = np.array(preds_res, dtype=np.float64)
    targets_arr = np.array(targets_80, dtype=np.int64)

    prob_ens = np.mean(np.stack([prob_r, prob_e, prob_res]), axis=0)
    if not np.all(np.isfinite(prob_ens)):
        raise RuntimeError("Non-finite ensemble probability")
    pred_ens = (prob_ens >= threshold).astype(np.int64)

    acc = accuracy_score(targets_arr, pred_ens) * 100.0
    f1 = f1_score(targets_arr, pred_ens, labels=[0, 1], average="macro", zero_division=0)
    rec_0 = recall_score(targets_arr, pred_ens, pos_label=0, zero_division=0) * 100.0
    rec_1 = recall_score(targets_arr, pred_ens, pos_label=1, zero_division=0) * 100.0
    cm = confusion_matrix(targets_arr, pred_ens, labels=[0, 1])
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


def export_submission_package(frozen_config_path: Path, export_dir: Path) -> None:
    """Verify all sources, stage the whole package, and publish to a new directory."""
    cfg = json.loads(Path(frozen_config_path).read_text(encoding="utf-8"))
    _check_eval_config(cfg, production=True)
    mapping = {"redimnet": "best_redimnet.pt", "ecapa_tdnn": "best_ecapa_tdnn.pt", "resnet50": "best_resnet50.pt"}
    for name in mapping:
        info = cfg["models"][name]
        src = Path(info["checkpoint_path"])
        if compute_sha256(src) != info["sha256"]:
            raise ValueError(f"Checkpoint changed after freezing: {name}")
        _validate_checkpoint(torch.load(src, map_location="cpu", weights_only=False), name)
    export_dir = Path(export_dir).resolve()
    if export_dir.exists() and any(export_dir.iterdir()):
        if (export_dir / "selection_metadata.json").exists() and all(
                (export_dir / filename).is_file() and compute_sha256(export_dir / filename) == cfg["models"][name]["sha256"]
                for name, filename in mapping.items()):
            print(f"Existing matching export reused: {export_dir}")
            return
        raise ValueError("Export directory already contains different files; use a new export directory")
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{export_dir.name}-", dir=export_dir.parent))
    try:
        meta = {"selection_rule": "fixed_additional_last", "parent_selection": "official_validation_best",
                "frozen_config_sha256": compute_sha256(frozen_config_path),
                "note": "best_*.pt are compatibility filenames, not a new Validation best selection.", "files": {}}
        for name, filename in mapping.items():
            info = cfg["models"][name]
            shutil.copy2(info["checkpoint_path"], staging / filename)
            if compute_sha256(staging / filename) != info["sha256"]:
                raise ValueError(f"Export hash mismatch: {filename}")
            meta["files"][filename] = {**info, "model_name": name}
        (staging / "selection_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        if export_dir.exists():
            export_dir.rmdir()  # Only the empty directory accepted above.
        staging.rename(export_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Verified submission checkpoints exported: {export_dir}")
