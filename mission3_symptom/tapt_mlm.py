"""Task-adaptive pretraining (TAPT): Training CSV 본문으로 MLM 을 이어서 학습한다.

119 신고 통화 전사문은 사전학습 코퍼스(뉴스·웹)와 문체가 달라, 분류 학습 전에 같은 본문으로
masked LM 을 조금 더 돌려 백본을 도메인에 맞춘다 (Gururangan et al., 2020).

- 학습(역전파)에는 **Training CSV 의 text 만** 쓴다. 라벨은 읽지 않는다.
- 결과 폴더는 `train.py --model-name-or-path <폴더> --local-files-only` 로 바로 분류 학습에 쓴다.
- `--eval-csv` 를 주면 학습 전과 매 epoch 뒤에 고정 마스크로 MLM 손실을 잰다 (평가 전용, 역전파 없음).
  Training 표본(본 텍스트)과 eval CSV 표본(안 본 텍스트)의 차이로 적응과 암기를 구분하는 진단용이다.

    python tapt_mlm.py --train-csv <mission3_train.csv> --output-dir runs/tapt_klue_base_e4 --amp
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

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


EVAL_SEED = 2024


def sample_texts(texts: List[str], n: int, seed: int) -> List[str]:
    """진단용 표본. 본문이 n 보다 적으면 전부 쓴다."""
    import pandas as pd

    series = pd.Series(list(texts))
    return series.sample(min(n, len(series)), random_state=seed).tolist()


def build_eval_batches(tokenizer, collator, named_texts: Dict[str, List[str]], max_length: int,
                       seed: int = EVAL_SEED, batch_size: int = 16) -> Dict[str, list]:
    """고정 마스크 평가 배치. 전역 난수(CPU·CUDA)는 건드리지 않는다.

    마스크 생성용으로 시드를 새로 걸면 학습의 dropout 난수까지 바뀌어, `--eval-csv` 를 켜느냐에
    따라 TAPT 가중치가 달라진다. fork_rng 로 격리해 평가를 순수 진단으로 둔다.
    """
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    batches: Dict[str, list] = {}
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for name, texts in named_texts.items():
            enc = tokenizer(texts, truncation=True, max_length=max_length, return_special_tokens_mask=True)
            rows = [{k: enc[k][i] for k in ("input_ids", "attention_mask", "special_tokens_mask")}
                    for i in range(len(texts))]
            batches[name] = [collator(rows[i:i + batch_size]) for i in range(0, len(rows), batch_size)]
    return batches


def provenance_note(eval_csv: Optional[str]) -> str:
    if not eval_csv:
        return "Training CSV text column only (labels unused, Validation unused)"
    return ("Backprop: Training CSV text column only (labels unused). "
            f"eval_csv={eval_csv} used only for no_grad fixed-mask MLM loss logging; "
            "not used for checkpoint selection (the last epoch is always saved).")


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
    parser.add_argument("--eval-csv", help="진단용 held-out 본문 CSV (평가 전용, 학습에 쓰지 않음)")
    parser.add_argument("--eval-samples", type=int, default=256)
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

    eval_sets = {}
    if args.eval_csv:
        import pandas as pd

        eval_sets = build_eval_batches(
            tokenizer,
            collator,
            {
                "seen_train": sample_texts(frame["text"].tolist(), args.eval_samples, seed=EVAL_SEED),
                "unseen_eval": sample_texts(pd.read_csv(args.eval_csv)["text"].tolist(), args.eval_samples,
                                            seed=EVAL_SEED),
            },
            max_length=args.max_length,
            seed=EVAL_SEED,
        )

    def heldout_losses() -> dict:
        model.eval()
        out = {}
        with torch.no_grad():
            for name, batches in eval_sets.items():
                total_loss, total_tokens = 0.0, 0
                for batch in batches:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                        loss = model(**batch).loss
                    n = int((batch["labels"] != -100).sum())
                    total_loss += float(loss) * n
                    total_tokens += n
                out[name] = total_loss / max(total_tokens, 1)
        model.train()
        return out

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
    if eval_sets:
        before = heldout_losses()
        history.append({"epoch": 0, **before})
        print(f"Epoch 0: " + ", ".join(f"{k}={v:.4f}" for k, v in before.items()), flush=True)
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
        record = {"epoch": epoch, "mlm_loss": total / max(count, 1)}
        if eval_sets:
            record.update(heldout_losses())
        history.append(record)
        print(f"Epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in record.items() if k != "epoch"),
              flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    meta = {k: v for k, v in vars(args).items()}
    meta.update({
        "source": provenance_note(args.eval_csv),
        "num_texts": len(texts),
        "history": history,
        "training_seconds": time.perf_counter() - started,
    })
    (output_dir / "tapt_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {output_dir}")


if __name__ == "__main__":
    main()
