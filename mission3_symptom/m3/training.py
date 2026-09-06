"""Mission 3 KoBERT 학습, 검증 및 실험 산출물 저장."""

from __future__ import annotations

import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import TARGET_SYMPTOMS
from .dataset import (
    calculate_token_length_stats,
    create_dataloader,
    load_symptom_csv,
)
from .metrics import eval_macro_f1
from .model import DEFAULT_MODEL_NAME, build_tokenizer_and_model, load_saved_model, save_model_bundle
from .threshold import apply_thresholds


@dataclass(frozen=True)
class TrainingConfig:
    train_csv: str
    val_csv: str
    output_dir: str
    model_name_or_path: str = DEFAULT_MODEL_NAME
    model_revision: Optional[str] = None
    seed: int = 42
    max_length: int = 512
    train_batch_size: int = 8
    val_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    amp: bool = False
    local_files_only: bool = False
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_steps: Optional[int] = None
    smoke_test: bool = False


def set_seed(seed: int) -> None:
    """데이터 순서와 모델 초기화를 같은 조건으로 재현."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """실행 환경에 맞는 device를 선택하고 잘못된 요청은 즉시 차단."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 현재 환경에서 사용할 수 없습니다.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"지원하지 않는 device입니다: {requested}")
    return torch.device(requested)


def _move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    labels = batch["labels"].to(device, non_blocking=True)
    model_inputs = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "labels"
    }
    return model_inputs, labels


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device: torch.device,
    scaler,
    amp_enabled: bool,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    global_step: int,
    max_steps: Optional[int],
) -> Tuple[float, int, bool]:
    """한 epoch을 학습하고 optimizer update 기준 global step을 반환."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    reached_max_steps = False
    progress = tqdm(dataloader, desc="Train", leave=False)

    for batch_index, batch in enumerate(progress):
        model_inputs, labels = _move_batch_to_device(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(**model_inputs).logits
            loss = loss_fn(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"유한하지 않은 학습 loss가 발생했습니다: {loss.item()}")

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        scaler.scale(loss / gradient_accumulation_steps).backward()

        is_last_batch = batch_index + 1 == len(dataloader)
        should_update = (batch_index + 1) % gradient_accumulation_steps == 0 or is_last_batch
        if should_update:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if max_steps is not None and global_step >= max_steps:
                reached_max_steps = True
                break

        progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss, global_step, reached_max_steps


def evaluate(
    model,
    dataloader,
    loss_fn,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, object]:
    """Validation 전체를 평가하고 threshold 0.5 성능과 원본 출력을 반환."""
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            model_inputs, labels = _move_batch_to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(**model_inputs).logits
                loss = loss_fn(logits, labels)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            logits_list.append(logits.float().cpu().numpy())
            labels_list.append(labels.float().cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started_at

    logits_array = np.concatenate(logits_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    probabilities = torch.sigmoid(torch.from_numpy(logits_array)).numpy()
    predictions = apply_thresholds(probabilities, 0.5)
    macro_f1, per_class_f1 = eval_macro_f1(
        labels_array,
        predictions,
        return_per_class=True,
    )
    return {
        "loss": total_loss / max(total_samples, 1),
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "logits": logits_array,
        "probabilities": probabilities,
        "labels": labels_array,
        "inference_seconds": inference_seconds,
        "seconds_per_sample": inference_seconds / max(total_samples, 1),
    }


def _write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _environment_metadata(device: torch.device, amp_enabled: bool) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "device_type": device.type,
        "amp_enabled": amp_enabled,
    }
    if device.type == "cuda":
        metadata["device_name"] = torch.cuda.get_device_name(device)
        metadata["cuda_version"] = torch.version.cuda
    return metadata


def _validate_config(config: TrainingConfig) -> None:
    positive_ints = {
        "max_length": config.max_length,
        "train_batch_size": config.train_batch_size,
        "val_batch_size": config.val_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "epochs": config.epochs,
    }
    invalid = [name for name, value in positive_ints.items() if value <= 0]
    if invalid:
        raise ValueError(f"양수여야 하는 설정입니다: {invalid}")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate는 양수여야 합니다.")
    if config.weight_decay < 0:
        raise ValueError("weight_decay는 0 이상이어야 합니다.")
    if config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm은 양수여야 합니다.")
    if config.num_workers < 0:
        raise ValueError("num_workers는 0 이상이어야 합니다.")
    if config.max_steps is not None and config.max_steps <= 0:
        raise ValueError("max_steps는 양수여야 합니다.")
    if not 0.0 <= config.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio는 0 이상 1 미만이어야 합니다.")


def _build_optimizer(model, learning_rate: float, weight_decay: float):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    parameter_groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and not any(key in name for key in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and any(key in name for key in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(parameter_groups, lr=learning_rate)


def run_training(config: TrainingConfig) -> Dict[str, object]:
    """설정에 따라 학습하고 best checkpoint 기준 산출물을 저장."""
    _validate_config(config)
    output_dir = Path(config.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output_dir이 디렉터리가 아닙니다: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"비어 있지 않은 output directory입니다: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = resolve_device(config.device)
    amp_enabled = bool(config.amp and device.type == "cuda")
    if config.amp and not amp_enabled:
        print("AMP는 CUDA 환경에서만 활성화됩니다. 현재 실행에서는 비활성화합니다.")

    train_df = load_symptom_csv(
        config.train_csv,
        max_samples=config.max_train_samples,
        sample_seed=config.seed,
    )
    val_df = load_symptom_csv(
        config.val_csv,
        max_samples=config.max_val_samples,
        sample_seed=config.seed,
    )
    tokenizer, model = build_tokenizer_and_model(
        config.model_name_or_path,
        local_files_only=config.local_files_only,
        revision=config.model_revision,
    )

    train_lengths = calculate_token_length_stats(
        train_df["text"].tolist(), tokenizer, config.max_length
    )
    val_lengths = calculate_token_length_stats(
        val_df["text"].tolist(), tokenizer, config.max_length
    )
    pin_memory = device.type == "cuda"
    train_loader = create_dataloader(
        train_df,
        tokenizer,
        max_length=config.max_length,
        batch_size=config.train_batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )
    val_loader = create_dataloader(
        val_df,
        tokenizer,
        max_length=config.max_length,
        batch_size=config.val_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )

    model.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = _build_optimizer(model, config.learning_rate, config.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    planned_steps = updates_per_epoch * config.epochs
    total_steps = min(planned_steps, config.max_steps) if config.max_steps else planned_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    config_data = asdict(config)
    config_data.update(
        {
            "target_symptoms": TARGET_SYMPTOMS,
            "num_train_samples": len(train_df),
            "num_val_samples": len(val_df),
            "effective_batch_size": (
                config.train_batch_size * config.gradient_accumulation_steps
            ),
            "total_parameters": total_parameters,
            "active_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "resolved_model_revision": getattr(model.config, "_commit_hash", None),
            "train_token_lengths": train_lengths,
            "val_token_lengths": val_lengths,
            "environment": _environment_metadata(device, amp_enabled),
        }
    )
    _write_json(output_dir / "run_config.json", config_data)

    history: List[Dict[str, object]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    global_step = 0
    best_model_dir = output_dir / "best_model"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started_at = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        train_loss, global_step, reached_max_steps = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            global_step=global_step,
            max_steps=config.max_steps,
        )
        validation = evaluate(model, val_loader, loss_fn, device, amp_enabled)
        epoch_result = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train_loss": train_loss,
            "val_loss": validation["loss"],
            "val_macro_f1_at_0_5": validation["macro_f1"],
            "val_per_class_f1_at_0_5": validation["per_class_f1"],
            "val_inference_seconds": validation["inference_seconds"],
        }
        history.append(epoch_result)
        _write_json(output_dir / "history.json", {"epochs": history})

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={validation['loss']:.4f}, "
            f"val_macro_f1@0.5={validation['macro_f1']:.4f}"
        )
        if float(validation["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(validation["macro_f1"])
            best_epoch = epoch
            save_model_bundle(model, tokenizer, best_model_dir)

        if reached_max_steps:
            break

    training_seconds = time.perf_counter() - training_started_at
    _, best_model = load_saved_model(best_model_dir)
    best_model.to(device)
    best_validation = evaluate(best_model, val_loader, loss_fn, device, amp_enabled)

    np.save(output_dir / "val_logits.npy", best_validation["logits"])
    np.save(output_dir / "val_probs.npy", best_validation["probabilities"])
    np.save(output_dir / "val_labels.npy", best_validation["labels"])

    final_metrics: Dict[str, object] = {
        "checkpoint": str(best_model_dir),
        "best_epoch": best_epoch,
        "threshold": 0.5,
        "val_loss": best_validation["loss"],
        "val_macro_f1": best_validation["macro_f1"],
        "val_per_class_f1": best_validation["per_class_f1"],
        "num_val_samples": len(val_df),
        "inference_seconds": best_validation["inference_seconds"],
        "seconds_per_sample": best_validation["seconds_per_sample"],
        "validation_batch_size": config.val_batch_size,
        "training_seconds": training_seconds,
        "target_symptoms": TARGET_SYMPTOMS,
    }
    if device.type == "cuda":
        final_metrics["peak_cuda_memory_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
    _write_json(output_dir / "baseline_metrics.json", final_metrics)
    return final_metrics
