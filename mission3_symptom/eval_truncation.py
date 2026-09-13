"""Validation에서 앞쪽 truncation vs head-tail vs 슬라이딩 창 Macro F1을 비교한다.

이 PC에는 val CSV와 학습 체크포인트가 없다. 코랩에서 아래처럼 실행한다.

python mission3_symptom/eval_truncation.py \
  --val-csv /content/drive/.../mission3_val.csv \
  --ckpt-dir /content/drive/.../runs/<run>/best_model \
  --output-json mission3_symptom/reports/truncation_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

MISSION3_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MISSION3_DIR))

from m3.config import NUM_CLASSES, TARGET_SYMPTOMS
from m3.dataset import labels_from_dataframe, load_symptom_csv
from m3.metrics import eval_macro_f1
from m3.model import load_saved_model
from m3.threshold import apply_thresholds, find_best_thresholds
from m3.truncation import (
    analyze_truncated_symptoms,
    content_budget,
    head_tail_concat,
    merge_chunk_probs,
    split_ids_for_windows,
)


def _as_id_list(value) -> List[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _content_ids(tokenizer, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
    kwargs = {
        "add_special_tokens": False,
        "truncation": False,
        "padding": False,
    }
    offsets = None
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, **kwargs)
        raw_offsets = encoded.get("offset_mapping")
        if raw_offsets is not None:
            offsets = [(int(start), int(end)) for start, end in raw_offsets]
            if offsets and isinstance(offsets[0], (list, tuple)) and len(offsets[0]) == 2:
                if not isinstance(offsets[0][0], (int, float)):
                    offsets = [(int(start), int(end)) for start, end in offsets[0]]
                else:
                    offsets = [(int(start), int(end)) for start, end in offsets]
    except (TypeError, ValueError, NotImplementedError):
        encoded = tokenizer(text, **kwargs)
    return _as_id_list(encoded["input_ids"]), offsets


def _wrap_special_tokens(tokenizer, content_ids: Sequence[int]) -> List[int]:
    content = [int(token_id) for token_id in content_ids]
    try:
        wrapped = tokenizer.build_inputs_with_special_tokens(content)
        return _as_id_list(wrapped)
    except (AttributeError, TypeError, NotImplementedError):
        pass

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None:
        cls_id = tokenizer.bos_token_id
    if sep_id is None:
        sep_id = tokenizer.eos_token_id

    output = list(content)
    if cls_id is not None:
        output = [int(cls_id), *output]
    if sep_id is not None:
        output = [*output, int(sep_id)]
    return output


def _features_from_content(tokenizer, content_ids: Sequence[int]) -> Dict[str, List[int]]:
    input_ids = _wrap_special_tokens(tokenizer, content_ids)
    features: Dict[str, List[int]] = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
    }
    if "token_type_ids" in getattr(tokenizer, "model_input_names", []):
        features["token_type_ids"] = [0] * len(input_ids)
    return features


def _forward_batch(model, tokenizer, contents: Sequence[Sequence[int]], device: torch.device) -> np.ndarray:
    features = [_features_from_content(tokenizer, ids) for ids in contents]
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits
    return torch.sigmoid(logits.float()).cpu().numpy()


def _select_contents(
    content_ids: Sequence[int],
    method: str,
    max_length: int,
) -> List[List[int]]:
    if method == "truncate":
        budget = content_budget(max_length)
        return [list(content_ids[:budget])]
    if method == "head_tail":
        budget = content_budget(max_length)
        keep = head_tail_concat(len(content_ids), budget)
        return [[content_ids[index] for index in keep]]
    if method == "sliding":
        return split_ids_for_windows(content_ids, max_length=max_length)
    raise ValueError(f"지원하지 않는 method: {method}")


def _score(labels: np.ndarray, probs: np.ndarray) -> Dict[str, object]:
    pred_half = apply_thresholds(probs, 0.5)
    f1_half, per_half = eval_macro_f1(labels, pred_half, return_per_class=True)
    thresholds, baseline, best, baseline_cls, best_cls = find_best_thresholds(probs, labels)
    return {
        "macro_f1@0.5": float(f1_half),
        "per_class_f1@0.5": {key: float(value) for key, value in per_half.items()},
        "optimized_macro_f1": float(best),
        "optimized_thresholds": [float(value) for value in thresholds],
        "per_class_f1_optimized": {key: float(value) for key, value in best_cls.items()},
        "baseline_macro_f1_from_search": float(baseline),
        "baseline_class_f1_from_search": {key: float(value) for key, value in baseline_cls.items()},
    }


def evaluate_methods(
    dataframe,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
    methods: Sequence[str],
) -> Dict[str, object]:
    labels = labels_from_dataframe(dataframe)
    texts = dataframe["text"].tolist()
    model.eval()

    contents: List[List[int]] = []
    offsets_list: List[Optional[List[Tuple[int, int]]]] = []
    for text in tqdm(texts, desc="tokenize", leave=False):
        ids, offsets = _content_ids(tokenizer, text)
        contents.append(ids)
        offsets_list.append(offsets)

    n_over = int(sum(len(ids) > content_budget(max_length) for ids in contents))
    cut_counter: Counter[str] = Counter()
    n_cut_calls = 0
    n_offset_ok = 0
    for text, ids, offsets, gold_row in zip(texts, contents, offsets_list, labels):
        if offsets is None:
            continue
        n_offset_ok += 1
        gold = [TARGET_SYMPTOMS[index] for index, flag in enumerate(gold_row) if flag == 1]
        report = analyze_truncated_symptoms(text, gold, offsets, max_length=max_length)
        if report["cut_symptoms"]:
            n_cut_calls += 1
            cut_counter.update(report["cut_symptoms"])

    results: Dict[str, object] = {
        "n_samples": int(len(texts)),
        "n_over_max_length": n_over,
        "over_max_length_ratio": float(n_over / max(len(texts), 1)),
        "n_offset_mapping_ok": n_offset_ok,
        "n_cut_calls": n_cut_calls,
        "cut_symptoms": dict(cut_counter),
        "methods": {},
    }

    for method in methods:
        probs = np.zeros((len(texts), NUM_CLASSES), dtype=np.float32)
        if method in {"truncate", "head_tail"}:
            windows = [_select_contents(ids, method, max_length)[0] for ids in contents]
            for start in tqdm(range(0, len(windows), batch_size), desc=method, leave=False):
                end = min(start + batch_size, len(windows))
                probs[start:end] = _forward_batch(
                    model, tokenizer, windows[start:end], device
                )
        else:
            for index in tqdm(range(len(texts)), desc=method, leave=False):
                windows = _select_contents(contents[index], method, max_length)
                chunk_probs = []
                for chunk_start in range(0, len(windows), batch_size):
                    chunk_probs.append(
                        _forward_batch(
                            model,
                            tokenizer,
                            windows[chunk_start : chunk_start + batch_size],
                            device,
                        )
                    )
                merged = merge_chunk_probs(np.concatenate(chunk_probs, axis=0), method="max")
                probs[index] = merged.astype(np.float32)
        results["methods"][method] = _score(labels, probs)

    truncate = results["methods"].get("truncate", {})
    for method, payload in results["methods"].items():
        if method == "truncate" or not truncate:
            continue
        payload["delta_macro_f1@0.5"] = float(
            payload["macro_f1@0.5"] - truncate["macro_f1@0.5"]
        )
        payload["delta_optimized_macro_f1"] = float(
            payload["optimized_macro_f1"] - truncate["optimized_macro_f1"]
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 3 truncation val 비교")
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--ckpt-dir", required=True, help="학습 run의 best_model 폴더")
    parser.add_argument("--output-json", default="mission3_symptom/reports/truncation_val.json")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["truncate", "head_tail", "sliding"],
        choices=["truncate", "head_tail", "sliding"],
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataframe = load_symptom_csv(args.val_csv, max_samples=args.max_val_samples)
    tokenizer, model = load_saved_model(args.ckpt_dir)
    model.to(device)
    model.eval()

    results = evaluate_methods(
        dataframe=dataframe,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        methods=args.methods,
    )
    results["val_csv"] = str(Path(args.val_csv))
    results["ckpt_dir"] = str(Path(args.ckpt_dir))
    results["device"] = str(device)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"samples={results['n_samples']} over512={results['n_over_max_length']} cut_calls={results['n_cut_calls']}")
    for method, payload in results["methods"].items():
        extra = ""
        if "delta_macro_f1@0.5" in payload:
            extra = (
                f"  dF1@0.5={payload['delta_macro_f1@0.5']:+.4f}"
                f"  dF1_opt={payload['delta_optimized_macro_f1']:+.4f}"
            )
        print(
            f"{method:10s}  F1@0.5={payload['macro_f1@0.5']:.4f}"
            f"  opt={payload['optimized_macro_f1']:.4f}{extra}"
        )
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
