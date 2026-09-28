"""Task-adaptive pretraining (TAPT): Training CSV 본문으로 MLM 을 이어서 학습한다.

119 신고 통화 전사문은 사전학습 코퍼스(뉴스·웹)와 문체가 달라, 분류 학습 전에 같은 본문으로
masked LM 을 조금 더 돌려 백본을 도메인에 맞춘다 (Gururangan et al., 2020).

- **Training CSV 의 text 만** 쓴다. 라벨은 읽지 않고, Validation 은 받지 않는다.
- 결과 폴더는 `train.py --model-name-or-path <폴더> --local-files-only` 로 바로 분류 학습에 쓴다.

    python tapt_mlm.py --train-csv <mission3_train.csv> --output-dir runs/tapt_klue_base_e4 --amp
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from m3.config import DEFAULT_UTTERANCE_SEP_MODE
from m3.dataset import load_symptom_csv
from m3.labels import verify_utterance_sep_mode
from m3.training import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-csv", required=True, help="Training CSV (text 컬럼만 사용)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", default="klue/roberta-base")
    parser.add_argument("--model-revision")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"비어 있지 않은 output directory입니다: {output_dir}")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")

    frame = load_symptom_csv(args.train_csv, max_samples=args.max_train_samples, sample_seed=args.seed)
    texts = frame["text"].tolist()
    # 분류 학습(공백 결합)과 같은 입력 분포여야 한다.
    verify_utterance_sep_mode(texts[:200], DEFAULT_UTTERANCE_SEP_MODE, source=args.train_csv)

    load = {"local_files_only": args.local_files_only}
    if args.model_revision:
        load["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **load)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, **load).to(device)

    encoded = tokenizer(texts, truncation=True, max_length=args.max_length, return_special_tokens_mask=True)
    examples = [
        {key: encoded[key][i] for key in ("input_ids", "attention_mask", "special_tokens_mask")}
        for i in range(len(texts))
    ]
    collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=args.mlm_probability)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
                        generator=generator)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
             "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    updates = math.ceil(len(loader) / args.gradient_accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * args.warmup_ratio), updates)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(1, args.epochs + 1):
        total, count = 0.0, 0
        progress = tqdm(loader, desc=f"MLM {epoch}/{args.epochs}", unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"유한하지 않은 MLM loss: {loss.item()}")
            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            total += float(loss.detach()); count += 1
            if (step + 1) % args.gradient_accumulation_steps == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(avg_loss=f"{total / count:.4f}")
        history.append({"epoch": epoch, "mlm_loss": total / max(count, 1)})
        print(f"Epoch {epoch}: mlm_loss={total / max(count, 1):.4f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    meta = {k: v for k, v in vars(args).items()}
    meta.update({
        "source": "Training CSV text column only (labels unused, Validation unused)",
        "num_texts": len(texts),
        "history": history,
        "training_seconds": time.perf_counter() - started,
    })
    (output_dir / "tapt_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {output_dir}")


if __name__ == "__main__":
    main()
