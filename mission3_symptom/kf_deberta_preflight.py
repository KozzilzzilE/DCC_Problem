"""KF-DeBERTa compatibility, memory, and offline preflight.

This diagnostic performs exactly one completed optimizer step. It never iterates
over Training or Validation batches and never evaluates task metrics.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from m3.config import NUM_CLASSES, TARGET_SYMPTOMS


MODEL_NAME = "kakaobank/kf-deberta-base"
MODEL_REVISION = "363b171d71443b0874b0bf9cea053eb5b1650633"
EXPECTED_MISSING_KEYS = {
    "classifier.bias",
    "classifier.weight",
    "pooler.dense.bias",
    "pooler.dense.weight",
}
EXPECTED_UNEXPECTED_KEYS = {
    "cls.predictions.bias",
    "cls.predictions.transform.LayerNorm.bias",
    "cls.predictions.transform.LayerNorm.weight",
    "cls.predictions.transform.dense.bias",
    "cls.predictions.transform.dense.weight",
}
SANITY_TEXTS = (
    "환자가 고열과 구토가 있고 숨쉬기 힘들다고 합니다.",
    "39.2도예요, 숨이 넘 차고 머리가 띵해여ㅠㅠ BP 90/60, SpO2 88%.",
)


def _load_options(source: str, revision: str | None) -> Dict[str, object]:
    options: Dict[str, object] = {}
    if Path(source).is_dir():
        options["local_files_only"] = True
    elif revision:
        options["revision"] = revision
    return options


def load_assets(source: str, revision: str | None):
    """Load the standard Auto* path and retain its checkpoint loading report."""
    options = _load_options(source, revision)
    label2id = {label: index for index, label in enumerate(TARGET_SYMPTOMS)}
    id2label = {index: label for index, label in enumerate(TARGET_SYMPTOMS)}
    config = AutoConfig.from_pretrained(source, **options)
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=NUM_CLASSES,
        label2id=label2id,
        id2label=id2label,
        problem_type="multi_label_classification",
        output_loading_info=True,
        **options,
    )
    return config, tokenizer, model, loading_info


def _sorted_keys(value: Iterable[str]) -> List[str]:
    return sorted(str(key) for key in value)


def validate_loading_info(loading_info: Dict[str, object]) -> Dict[str, object]:
    missing = set(loading_info.get("missing_keys", ()))
    unexpected = set(loading_info.get("unexpected_keys", ()))
    mismatched = set(loading_info.get("mismatched_keys", ()))
    error_messages = list(loading_info.get("error_msgs", ()))
    abnormal_missing = missing - EXPECTED_MISSING_KEYS
    abnormal_unexpected = unexpected - EXPECTED_UNEXPECTED_KEYS
    if abnormal_missing or abnormal_unexpected or mismatched or error_messages:
        raise RuntimeError(
            "Abnormal checkpoint loading report: "
            f"missing={sorted(abnormal_missing)}, "
            f"unexpected={sorted(abnormal_unexpected)}, "
            f"mismatched={sorted(mismatched)}, errors={error_messages}"
        )
    return {
        "missing_keys": _sorted_keys(missing),
        "unexpected_keys": _sorted_keys(unexpected),
        "mismatched_keys": _sorted_keys(mismatched),
        "error_messages": error_messages,
        "abnormal_missing_keys": [],
        "abnormal_unexpected_keys": [],
    }


def validate_contract(config, tokenizer, model) -> Dict[str, object]:
    embedding_size = int(model.get_input_embeddings().num_embeddings)
    if config.model_type != "deberta-v2" or model.config.model_type != "deberta-v2":
        raise ValueError("KF-DeBERTa must load as model_type='deberta-v2'.")
    if type(model).__name__ != "DebertaV2ForSequenceClassification":
        raise TypeError(f"Unexpected model class: {type(model).__name__}")
    if int(model.config.num_labels) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} labels, got {model.config.num_labels}.")
    if model.config.problem_type != "multi_label_classification":
        raise ValueError(f"Unexpected problem_type: {model.config.problem_type}")
    if len(tokenizer) != embedding_size or tokenizer.vocab_size != embedding_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"vocab_size={tokenizer.vocab_size}, len={len(tokenizer)}, "
            f"embedding_size={embedding_size}"
        )
    if int(tokenizer.model_max_length) != 512:
        raise ValueError(f"Unexpected tokenizer max length: {tokenizer.model_max_length}")
    if int(model.config.max_position_embeddings) != 512:
        raise ValueError(
            f"Unexpected model max positions: {model.config.max_position_embeddings}"
        )
    if tokenizer.pad_token_id != model.config.pad_token_id:
        raise ValueError("Tokenizer and model pad token IDs differ.")

    sanity_rows = []
    for text in SANITY_TEXTS:
        encoded = tokenizer(text, add_special_tokens=True)
        sanity_rows.append(
            {
                "tokens": tokenizer.tokenize(text),
                "length": len(encoded["input_ids"]),
                "unk_count": sum(
                    token_id == tokenizer.unk_token_id
                    for token_id in encoded["input_ids"]
                ),
            }
        )

    long_text = (SANITY_TEXTS[0] + " ") * 300
    truncated = tokenizer(
        long_text,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        padding=False,
    )
    if len(truncated["input_ids"]) != 512:
        raise ValueError("Tokenizer did not truncate to exactly 512 tokens.")
    if truncated["input_ids"][0] != tokenizer.cls_token_id:
        raise ValueError("Truncated input does not start with CLS.")
    if truncated["input_ids"][-1] != tokenizer.sep_token_id:
        raise ValueError("Truncated input does not preserve the final SEP.")
    token_type_ids = truncated.get("token_type_ids")
    if token_type_ids is not None and set(token_type_ids) != {0}:
        raise ValueError("Single-sequence token_type_ids must contain only zero.")

    return {
        "config_class": type(config).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": int(tokenizer.vocab_size),
        "tokenizer_length": len(tokenizer),
        "embedding_size": embedding_size,
        "model_type": model.config.model_type,
        "architecture": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_labels": int(model.config.num_labels),
        "problem_type": model.config.problem_type,
        "model_max_length": int(tokenizer.model_max_length),
        "max_position_embeddings": int(model.config.max_position_embeddings),
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "model_input_names": list(tokenizer.model_input_names),
        "special_tokens": dict(tokenizer.special_tokens_map),
        "special_token_ids": {
            name: getattr(tokenizer, f"{name}_token_id")
            for name in ("pad", "cls", "sep", "unk", "mask", "bos", "eos")
        },
        "sanity": sanity_rows,
        "truncation": {
            "length": len(truncated["input_ids"]),
            "first_is_cls": truncated["input_ids"][0] == tokenizer.cls_token_id,
            "last_is_sep": truncated["input_ids"][-1] == tokenizer.sep_token_id,
            "attention_mask_length": len(truncated["attention_mask"]),
            "token_type_ids_present": token_type_ids is not None,
            "token_type_values": sorted(set(token_type_ids or [])),
        },
    }


def calculate_tokenizer_statistics(
    texts: Sequence[str],
    tokenizer,
    *,
    max_length: int = 512,
    batch_size: int = 256,
) -> Dict[str, object]:
    """Calculate aggregate tokenizer diagnostics without retaining source text."""
    lengths: List[int] = []
    sample_unk_ratios: List[float] = []
    total_unknown = 0
    total_content = 0
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_special_tokens_mask=True,
        )
        for input_ids, special_mask in zip(
            encoded["input_ids"], encoded["special_tokens_mask"]
        ):
            content_count = sum(mask == 0 for mask in special_mask)
            unknown_count = sum(
                token_id == tokenizer.unk_token_id and mask == 0
                for token_id, mask in zip(input_ids, special_mask)
            )
            lengths.append(len(input_ids))
            total_content += content_count
            total_unknown += unknown_count
            sample_unk_ratios.append(unknown_count / max(content_count, 1))

    if not lengths:
        raise ValueError("No text samples were provided.")
    length_values = np.asarray(lengths, dtype=np.int64)
    unk_values = np.asarray(sample_unk_ratios, dtype=np.float64)
    over_limit = int(np.sum(length_values > max_length))
    return {
        "count": int(length_values.size),
        "micro_unk_ratio": float(total_unknown / max(total_content, 1)),
        "sample_unk_ratio_p50": float(np.percentile(unk_values, 50)),
        "sample_unk_ratio_p95": float(np.percentile(unk_values, 95)),
        "sample_unk_ratio_max": float(unk_values.max()),
        "sample_unk_ratio_over_10_percent": int(np.sum(unk_values > 0.1)),
        "token_length_min": int(length_values.min()),
        "token_length_p50": float(np.percentile(length_values, 50)),
        "token_length_p90": float(np.percentile(length_values, 90)),
        "token_length_p95": float(np.percentile(length_values, 95)),
        "token_length_p99": float(np.percentile(length_values, 99)),
        "token_length_max": int(length_values.max()),
        "over_max_length": over_limit,
        "over_max_length_ratio": float(over_limit / length_values.size),
    }


def load_text_column(csv_path: str | Path) -> List[str]:
    dataframe = pd.read_csv(csv_path, usecols=["text"])
    if dataframe["text"].isna().any():
        raise ValueError(f"Missing text in {csv_path}")
    return dataframe["text"].astype(str).tolist()


def _fixed_length_batch(tokenizer, batch_size: int) -> Dict[str, torch.Tensor]:
    long_text = (SANITY_TEXTS[0] + " ") * 300
    batch = tokenizer(
        [long_text] * batch_size,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    if tuple(batch["input_ids"].shape) != (batch_size, 512):
        raise ValueError(f"Unexpected smoke input shape: {tuple(batch['input_ids'].shape)}")
    return dict(batch)


def run_one_optimizer_step(model, tokenizer, batch_size: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the memory preflight.")
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    inputs = {
        key: value.to(device)
        for key, value in _fixed_length_batch(tokenizer, batch_size).items()
    }
    labels = torch.zeros((batch_size, NUM_CLASSES), dtype=torch.float32, device=device)
    labels[:, 0] = 1.0
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        logits = model(**inputs).logits
        loss = loss_function(logits, labels)
    if tuple(logits.shape) != (batch_size, NUM_CLASSES) or not torch.isfinite(loss):
        raise FloatingPointError("Invalid logits shape or non-finite smoke loss.")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise FloatingPointError("Non-finite gradient in memory smoke.")
    gradient_norm = clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("Non-finite gradient norm in memory smoke.")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    result = {
        "success": True,
        "batch_size": batch_size,
        "sequence_length": 512,
        "amp": True,
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "loss_finite": True,
        "gradients_finite": finite_gradients,
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "optimizer_steps": 1,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    del inputs, labels, logits, loss
    return model, optimizer, result


def run_validation_forward(model, tokenizer, batch_size: int) -> Dict[str, object]:
    device = torch.device("cuda")
    inputs = {
        key: value.to(device)
        for key, value in _fixed_length_batch(tokenizer, batch_size).items()
    }
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    ):
        logits = model(**inputs).logits
    torch.cuda.synchronize(device)
    result = {
        "success": True,
        "batch_size": batch_size,
        "sequence_length": 512,
        "amp": True,
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    del inputs, logits
    return result


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def release_cuda(model=None, optimizer=None) -> None:
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    if model is not None:
        model.to("cpu")
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def offline_round_trip(model, tokenizer) -> Dict[str, object]:
    model.to("cpu").eval()
    sample_inputs = tokenizer(SANITY_TEXTS[0], return_tensors="pt")
    with torch.no_grad():
        expected_logits = model(**sample_inputs).logits

    with tempfile.TemporaryDirectory(prefix="kf_deberta_bundle_") as directory:
        bundle = Path(directory)
        model.save_pretrained(bundle, safe_serialization=True)
        tokenizer.save_pretrained(bundle)
        required = {"config.json", "model.safetensors", "tokenizer_config.json"}
        filenames = {path.name for path in bundle.iterdir() if path.is_file()}
        missing = required - filenames
        if missing or not ({"tokenizer.json", "vocab.txt"} & filenames):
            raise FileNotFoundError(f"Incomplete offline bundle: missing={sorted(missing)}")
        if any(path.is_symlink() for path in bundle.iterdir()):
            raise RuntimeError("Offline bundle contains a cache-dependent symlink.")

        previous_hub_offline = os.environ.get("HF_HUB_OFFLINE")
        previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            local_config = AutoConfig.from_pretrained(bundle, local_files_only=True)
            local_tokenizer = AutoTokenizer.from_pretrained(bundle, local_files_only=True)
            local_model = AutoModelForSequenceClassification.from_pretrained(
                bundle, local_files_only=True
            ).eval()
        finally:
            if previous_hub_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_hub_offline
            if previous_transformers_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = previous_transformers_offline

        try:
            reloaded_inputs = local_tokenizer(SANITY_TEXTS[0], return_tensors="pt")
            tokenizer_consistent = all(
                torch.equal(sample_inputs[key], reloaded_inputs[key])
                for key in sample_inputs.keys()
            ) and set(sample_inputs.keys()) == set(reloaded_inputs.keys())
            with torch.no_grad():
                actual_logits = local_model(**reloaded_inputs).logits
            max_absolute_logit_difference = float(
                (expected_logits - actual_logits).abs().max()
            )
            logits_consistent = torch.allclose(
                expected_logits, actual_logits, rtol=1e-6, atol=1e-6
            )
            if not tokenizer_consistent or not logits_consistent:
                raise AssertionError(
                    "Offline tokenizer or logits round-trip changed outputs: "
                    f"tokenizer={tokenizer_consistent}, "
                    f"max_abs_diff={max_absolute_logit_difference}"
                )
            result = {
                "success": True,
                "files": sorted(filenames),
                "all_files_regular": True,
                "config_class": type(local_config).__name__,
                "tokenizer_class": type(local_tokenizer).__name__,
                "architecture": type(local_model).__name__,
                "num_labels": int(local_model.config.num_labels),
                "tokenizer_consistent": tokenizer_consistent,
                "logits_shape": list(actual_logits.shape),
                "logits_consistent": bool(logits_consistent),
                "max_absolute_logit_difference": max_absolute_logit_difference,
                "local_files_only": True,
                "offline_environment": True,
            }
            return result
        finally:
            del local_model, local_tokenizer, local_config
            gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--model-name-or-path", default=MODEL_NAME)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def write_report(path: Path, report: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_json)
    config, tokenizer, model, loading_info = load_assets(
        args.model_name_or_path, args.model_revision
    )
    report = {
        "model": MODEL_NAME,
        "revision": args.model_revision,
        "contract": validate_contract(config, tokenizer, model),
        "loading": validate_loading_info(loading_info),
        "tokenizer_statistics": {
            "training": calculate_tokenizer_statistics(
                load_text_column(args.train_csv), tokenizer
            ),
            "validation": calculate_tokenizer_statistics(
                load_text_column(args.val_csv), tokenizer
            ),
        },
    }
    write_report(output_path, report)

    selected_batch = 8
    optimizer = None
    batch_8_oom = False
    try:
        model, optimizer, train_memory = run_one_optimizer_step(
            model, tokenizer, selected_batch
        )
    except RuntimeError as error:
        if not _is_cuda_oom(error):
            raise
        batch_8_oom = True

    if batch_8_oom:
        report["batch_8_oom"] = True
        release_cuda(model, optimizer)
        del model
        gc.collect()
        _, _, model, _ = load_assets(args.model_name_or_path, args.model_revision)
        selected_batch = 4
        try:
            model, optimizer, train_memory = run_one_optimizer_step(
                model, tokenizer, selected_batch
            )
        except RuntimeError as fallback_error:
            if _is_cuda_oom(fallback_error):
                raise RuntimeError("Both batch 8 and batch 4 memory preflights OOM.") from fallback_error
            raise
    else:
        report["batch_8_oom"] = False

    report["train_memory"] = train_memory
    report["full_training_train_batch"] = selected_batch
    report["full_training_gradient_accumulation"] = 16 // selected_batch
    write_report(output_path, report)
    try:
        report["validation_memory"] = run_validation_forward(model, tokenizer, 16)
        report["full_training_val_batch"] = 16
    except RuntimeError as error:
        if not _is_cuda_oom(error):
            raise
        report["validation_memory"] = {
            "success": False,
            "batch_size": 16,
            "sequence_length": 512,
            "oom": True,
        }
        torch.cuda.empty_cache()
        report["validation_fallback_memory"] = run_validation_forward(
            model, tokenizer, 8
        )
        report["full_training_val_batch"] = 8
    write_report(output_path, report)

    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    report["offline_round_trip"] = offline_round_trip(model, tokenizer)
    write_report(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
