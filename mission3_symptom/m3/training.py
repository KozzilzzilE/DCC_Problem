"""Mission 3 KLUE-RoBERTa 학습, 검증 및 실험 산출물 저장."""

from __future__ import annotations

import json
import math
import re
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import DEFAULT_UTTERANCE_SEP_MODE, ENCODE_MODE, TARGET_SYMPTOMS, check_encode_mode
from .dataset import (
    calculate_token_length_stats,
    create_dataloader,
    load_symptom_csv,
)
from .infer import build_inference_config
from .labels import verify_utterance_sep_mode
from .metrics import apply_thresholds, eval_macro_f1
from .model import build_tokenizer_and_model, load_saved_model, save_model_bundle


# 시작점 공개 사전학습 모델 (TAPT 결과 폴더를 --model-name-or-path 로 넘길 수도 있다)
BASELINE_MODEL_NAME = "klue/roberta-base"


CHECKPOINT_METRICS = ("val_loss", "val_macro_f1", "fixed_epoch")


@dataclass(frozen=True)
class TrainingConfig:
    """분류 학습 한 번의 설정. train.py 의 CLI 인자가 그대로 옮겨지고, run_config.json 에 기록된다."""

    train_csv: str
    val_csv: str
    output_dir: str
    model_name_or_path: str = BASELINE_MODEL_NAME
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
    use_pos_weight: bool = False
    # pos_weight = (negative / positive) ** power. 1.0 은 원래 공식, 0 은 plain BCE 와 같다.
    # 제출 모델의 값 0.5 는 run_dev_selection.sh 3 단계가 Training 내부 dev 로 정했다 (decisions/power.json).
    pos_weight_power: float = 1.0
    local_files_only: bool = False
    # 저장할 체크포인트 선정 기준. val_* 는 --val-csv(Training 내부 dev 분할) 위의 값이다.
    #   "fixed_epoch" (기본): 평가 점수를 보지 않고 checkpoint_epoch 번째 epoch 를 저장하고 학습을 멈춘다.
    #   "val_loss" / "val_macro_f1": dev 손실 최저 / dev macro F1 최고 epoch 를 저장한다.
    checkpoint_metric: str = "fixed_epoch"
    # fixed_epoch 일 때 저장할 epoch (None 이면 마지막 epoch). 학습률 스케줄은 epochs 기준 그대로라
    # epochs=3, checkpoint_epoch=2 는 3 epoch 학습의 2 epoch 시점과 같은 가중치가 된다.
    checkpoint_epoch: Optional[int] = None
    # max_length 를 넘는 통화는 앞부분만 남긴다. run_config / inference_config 호환을 위해 남긴 필드로 'truncate' 만 받는다.
    encode_mode: str = ENCODE_MODE
    # 학습 CSV 의 발화 경계 표현. run_config.json 에 기록되어 추론이 같은 모드를 복원한다.
    utterance_sep_mode: str = DEFAULT_UTTERANCE_SEP_MODE
    # layer-wise LR decay. 분류 헤드는 learning_rate, 인코더 층을 내려갈수록 decay 를 곱한다. 1.0 은 끔.
    llrd_decay: float = 1.0


def fixed_checkpoint_epoch(config: TrainingConfig) -> int:
    """fixed_epoch 기준에서 저장할 epoch. 지정이 없으면 마지막 epoch."""
    return config.checkpoint_epoch if config.checkpoint_epoch is not None else config.epochs


def checkpoint_mode(config: TrainingConfig) -> str:
    """run_config / baseline_metrics 에 남길 체크포인트 선정 방식."""
    if config.checkpoint_metric == "val_loss":
        return "min"
    if config.checkpoint_metric == "fixed_epoch":
        return f"epoch {fixed_checkpoint_epoch(config)}"
    return "max"


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
    """배치를 device 로 옮기고 `(모델 입력, 라벨)` 로 나눈다."""
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
    epoch: int,
    total_epochs: int,
) -> Tuple[float, int]:
    """한 epoch을 학습하고 (평균 학습 loss, optimizer update 기준 global step)을 반환."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(
        dataloader,
        desc=f"Train {epoch}/{total_epochs}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

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
        loss_value = float(loss.detach().cpu())
        total_loss += loss_value * batch_size
        total_samples += batch_size
        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            avg_loss=f"{total_loss / total_samples:.4f}",
        )
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

    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss, global_step


def evaluate(
    model,
    dataloader,
    loss_fn,
    device: torch.device,
    amp_enabled: bool,
    description: str = "평가",
) -> Dict[str, object]:
    """평가 CSV(--val-csv, Training 내부 dev 분할) 전체를 추론해 threshold 0.5 성능과 원본 출력을 반환."""
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()

    progress = tqdm(
        dataloader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    with torch.no_grad():
        for batch in progress:
            model_inputs, labels = _move_batch_to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(**model_inputs).logits
                loss = loss_fn(logits, labels)

            batch_size = int(labels.shape[0])
            loss_value = float(loss.detach().cpu())
            total_loss += loss_value * batch_size
            total_samples += batch_size
            progress.set_postfix(avg_loss=f"{total_loss / total_samples:.4f}")
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
    """한글이 그대로 보이게(ensure_ascii=False) UTF-8 JSON 으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _environment_metadata(device: torch.device, amp_enabled: bool) -> Dict[str, object]:
    """run_config.json 에 남길 실행 환경(Python·PyTorch·transformers 버전, device, GPU 이름)."""
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
    """학습을 시작하기 전에 설정 값의 범위와 조합을 확인하고, 틀리면 바로 ValueError 를 낸다."""
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
    if not 0.0 <= config.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio는 0 이상 1 미만이어야 합니다.")
    if not math.isfinite(config.pos_weight_power) or config.pos_weight_power < 0:
        raise ValueError("pos_weight_power는 0 이상의 유한한 값이어야 합니다.")
    if config.pos_weight_power != 1.0 and not config.use_pos_weight:
        raise ValueError("pos_weight_power를 바꾸려면 use_pos_weight를 함께 켜야 합니다.")
    # 최적 모델 선정 기준 검증
    if config.checkpoint_metric not in CHECKPOINT_METRICS:
        raise ValueError(
            f"지원하지 않는 checkpoint_metric입니다: {config.checkpoint_metric}. "
            "('val_loss', 'val_macro_f1', 'fixed_epoch'만 허용됩니다)"
        )
    if config.checkpoint_epoch is not None:
        if config.checkpoint_metric != "fixed_epoch":
            raise ValueError("checkpoint_epoch 는 checkpoint_metric='fixed_epoch' 와 함께만 쓸 수 있습니다.")
        if not 1 <= config.checkpoint_epoch <= config.epochs:
            raise ValueError(
                f"checkpoint_epoch 는 1 이상 epochs({config.epochs}) 이하여야 합니다: {config.checkpoint_epoch}"
            )
    check_encode_mode(config.encode_mode)
    if not 0.0 < config.llrd_decay <= 1.0:
        raise ValueError("llrd_decay는 0 초과 1 이하여야 합니다.")


# 인코더 층 파라미터 이름의 층 번호 (예: roberta.encoder.layer.11.attention.self.query.weight -> 11)
_LAYER_INDEX = re.compile(r"\.layer\.(\d+)\.")


def _layer_depths(names: List[str]) -> Dict[str, int]:
    """파라미터마다 분류 헤드에서 몇 단계 아래인지. 헤드 0, 맨 위 층 1, 임베딩은 맨 아래 층 + 1.

    RoBERTa 분류 모델의 파라미터는 세 종류뿐이다: `roberta.encoder.layer.N.*`(인코더 층),
    `roberta.embeddings.*`(임베딩), `classifier.*`(분류 헤드). 층 번호도 `.embeddings.` 도 없는 이름은
    분류 헤드로 본다.
    """
    indices = [int(m.group(1)) for m in map(_LAYER_INDEX.search, names) if m]
    num_layers = max(indices) + 1 if indices else 0
    depths = {}
    for name in names:
        match = _LAYER_INDEX.search(name)
        if match:
            depths[name] = num_layers - int(match.group(1))
        elif ".embeddings." in name:
            depths[name] = num_layers + 1
        else:
            depths[name] = 0
    return depths


def _build_optimizer(model, learning_rate: float, weight_decay: float, llrd_decay: float = 1.0):
    """AdamW 를 만든다. bias·LayerNorm 가중치는 weight decay 0, 나머지는 `weight_decay` 다.

    - `llrd_decay == 1.0`(기본): 위 두 그룹만 두고 모든 파라미터가 같은 `learning_rate` 를 쓴다.
    - `llrd_decay < 1.0`(layer-wise LR decay): `_layer_depths` 로 파라미터마다 깊이 d 를 매긴다
      (분류 헤드 0, 맨 위 인코더 층 1, ..., 맨 아래 층 L, 임베딩 L + 1). 그리고 (d, decay 제외 여부)
      묶음마다 그룹을 만들어 lr = learning_rate x llrd_decay ** d 를 준다. 층을 하나 내려갈 때마다
      decay 를 한 번 더 곱하는 셈이다. 예: RoBERTa-base(12층), lr 5e-5, decay 0.8 이면 헤드 5e-5,
      11번 층 4e-5, 0번 층 5e-5 x 0.8^12, 임베딩 5e-5 x 0.8^13.
    그룹은 (d, 제외 여부) 오름차순이고, 선형 스케줄러는 그룹마다 자기 lr 을 같은 비율로 줄인다.
    """
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    if llrd_decay != 1.0:
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        depths = _layer_depths([n for n, _ in named])
        grouped: Dict[Tuple[int, bool], List[torch.nn.Parameter]] = {}
        for name, parameter in named:
            key = (depths[name], any(k in name for k in no_decay))
            grouped.setdefault(key, []).append(parameter)
        return torch.optim.AdamW(
            [
                {
                    "params": params,
                    "lr": learning_rate * llrd_decay ** depth,
                    "weight_decay": 0.0 if excluded else weight_decay,
                }
                for (depth, excluded), params in sorted(grouped.items())
            ],
            lr=learning_rate,
        )
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


def _calculate_pos_weights(
    train_df,
    device: torch.device,
    power: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, Dict[str, object]]]:
    """Training 라벨 개수로만 `(negative / positive) ** power` 를 계산한다."""
    positive_counts = train_df[TARGET_SYMPTOMS].sum(axis=0).astype(np.int64)
    zero_positive = [
        symptom for symptom in TARGET_SYMPTOMS if int(positive_counts[symptom]) == 0
    ]
    if zero_positive:
        raise ValueError(
            "pos_weight를 계산할 positive sample이 없는 클래스입니다: "
            + ", ".join(zero_positive)
        )

    negative_counts = len(train_df) - positive_counts
    ratios = negative_counts / positive_counts
    weight_values = ratios ** float(power)
    statistics = {
        symptom: {
            "positive_count": int(positive_counts[symptom]),
            "negative_count": int(negative_counts[symptom]),
            "negative_over_positive": float(ratios[symptom]),
            "pos_weight_power": float(power),
            "pos_weight": float(weight_values[symptom]),
        }
        for symptom in TARGET_SYMPTOMS
    }
    weights = torch.tensor(
        [statistics[symptom]["pos_weight"] for symptom in TARGET_SYMPTOMS],
        dtype=torch.float32,
        device=device,
    )
    return weights, statistics


def _create_train_val_loaders(
    train_df,
    val_df,
    tokenizer,
    config: TrainingConfig,
    pin_memory: bool,
):
    """학습 loader(seed 고정 셔플)와 평가 loader(CSV 행 순서 그대로)를 만든다."""
    train_loader = create_dataloader(
        train_df,
        tokenizer,
        max_length=config.max_length,
        batch_size=config.train_batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        pad_to_multiple_of=8 if pin_memory else None,
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
        pad_to_multiple_of=8 if pin_memory else None,
    )
    return train_loader, val_loader


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

    train_df = load_symptom_csv(config.train_csv)
    val_df = load_symptom_csv(config.val_csv)
    verify_utterance_sep_mode(
        train_df["text"].head(200), config.utterance_sep_mode, source=config.train_csv)
    verify_utterance_sep_mode(
        val_df["text"].head(200), config.utterance_sep_mode, source=config.val_csv)

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
    train_loader, val_loader = _create_train_val_loaders(
        train_df,
        val_df,
        tokenizer,
        config,
        pin_memory=pin_memory,
    )

    model.to(device)
    pos_weight_statistics = None
    pos_weights = None
    if config.use_pos_weight:
        pos_weights, pos_weight_statistics = _calculate_pos_weights(
            train_df, device, power=config.pos_weight_power
        )
        print(f"Training label 기반 pos_weight (power={config.pos_weight_power}):")
        for symptom in TARGET_SYMPTOMS:
            stats = pos_weight_statistics[symptom]
            print(
                f"- {symptom}: positive={stats['positive_count']}, "
                f"negative={stats['negative_count']}, "
                f"pos_weight={stats['pos_weight']:.6f}"
            )

    # 손실: 9개 클래스 독립 BCE. use_pos_weight 면 양성 항에 클래스별 pos_weight 를 곱한다 (없으면 plain BCE).
    loss_fn = BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = _build_optimizer(
        model, config.learning_rate, config.weight_decay, llrd_decay=config.llrd_decay)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = updates_per_epoch * config.epochs  # fixed_epoch 로 일찍 멈춰도 스케줄은 epochs 기준
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
            "train_pos_weight_statistics": pos_weight_statistics,
            "checkpoint_selection_criterion": {
                "metric": config.checkpoint_metric,
                "mode": checkpoint_mode(config),
            },
            "environment": _environment_metadata(device, amp_enabled),
        }
    )
    _write_json(output_dir / "run_config.json", config_data)

    history: List[Dict[str, object]] = []
    # 최적 모델 선정을 위한 기준 점수 추적 (val_loss 최저값 또는 val_macro_f1 최고값)
    best_val_loss = math.inf
    best_val_macro_f1 = -math.inf
    best_epoch = 0
    global_step = 0
    best_model_dir = output_dir / "best_model"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started_at = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        train_loss, global_step = train_one_epoch(
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
            epoch=epoch,
            total_epochs=config.epochs,
        )
        validation = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            amp_enabled,
            description=f"평가 {epoch}/{config.epochs}",
        )
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
        # 하이퍼파라미터를 '고정 epoch' 기준으로 정할 수 있게 epoch 마다 평가 확률을 남긴다 (행 순서 = 평가 CSV).
        np.save(output_dir / f"val_probs_epoch{epoch}.npy", validation["probabilities"])

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={validation['loss']:.4f}, "
            f"val_macro_f1@0.5={validation['macro_f1']:.4f}"
        )
        # 최적 모델(Best Checkpoint) 판정 및 가중치 번들 저장
        is_best = False
        current_loss = float(validation["loss"])
        current_f1 = float(validation["macro_f1"])

        if config.checkpoint_metric == "val_loss":
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                is_best = True
        elif config.checkpoint_metric == "val_macro_f1":
            if current_f1 > best_val_macro_f1:
                best_val_macro_f1 = current_f1
                is_best = True
        elif config.checkpoint_metric == "fixed_epoch":
            # 평가 점수를 보지 않는다. 정해 둔 epoch 를 저장한다.
            is_best = epoch == fixed_checkpoint_epoch(config)

        if is_best:
            best_epoch = epoch
            save_model_bundle(model, tokenizer, best_model_dir)
            # best_model/ 만 따로 제출해도 추론이 학습 설정을 복원할 수 있어야 한다.
            # 부모 run 디렉터리의 run_config.json 이 함께 가지 않으면 sep_mode 가
            # 조용히 기본값으로 떨어져 점수만 깎인다.
            _write_json(
                best_model_dir / "inference_config.json", build_inference_config(config))
            score_str = f"loss={current_loss:.4f}" if config.checkpoint_metric == "val_loss" else f"macro_f1={current_f1:.4f}"
            print(f"  -> Best Checkpoint 갱신 (Epoch {epoch}, {config.checkpoint_metric}: {score_str})")

        if config.checkpoint_metric == "fixed_epoch" and epoch >= fixed_checkpoint_epoch(config):
            break

    training_seconds = time.perf_counter() - training_started_at
    _, best_model = load_saved_model(best_model_dir)
    best_model.to(device)
    best_validation = evaluate(
        best_model,
        val_loader,
        loss_fn,
        device,
        amp_enabled,
        description="평가 (저장한 체크포인트)",
    )

    np.save(output_dir / "val_logits.npy", best_validation["logits"])
    np.save(output_dir / "val_probs.npy", best_validation["probabilities"])
    np.save(output_dir / "val_labels.npy", best_validation["labels"])

    final_metrics: Dict[str, object] = {
        "checkpoint": str(best_model_dir),
        "best_epoch": best_epoch,
        "checkpoint_selection_criterion": {
            "metric": config.checkpoint_metric,
            "mode": checkpoint_mode(config),
        },
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
