"""누적 입력(통화 앞부분)으로 증상 9종 분류기를 학습한다.

    python scripts/train_prefix.py --calls runs/cache/train.jsonl --ids runs/cache/fit_ids.txt \
        --backbone <TAPT 폴더 또는 klue/roberta-base> --out runs/E002_prefix_uniform

통화마다 epoch 당 `--samples-per-call` 개의 앞부분을 뽑는다. 길이 t 는 1..n 에서 균등하게 뽑는다.
4-1 채점이 모든 순간을 같은 무게로 평균하므로, 학습 분포도 순간에 대해 균등하게 맞춘 것이다.
라벨은 통화 전체의 증상이다 (앞부분에 아직 증상이 나오지 않았어도 같은 라벨).

나머지 설정은 예선 Mission 3 최종 레시피를 따른다: LLRD 0.8, lr 5e-5, pos_weight=(neg/pos)^0.5, AMP.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4 import SYMPTOM_CLASSES  # noqa: E402
from m4.data import label_matrix, load_cache, prefix_texts, read_ids  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def llrd_groups(model, lr: float, decay: float, weight_decay: float):
    """층별 학습률 감쇠. 맨 위 인코더 층이 lr*decay, 아래로 갈수록 decay 를 한 번씩 더 곱한다.

    인코더 층과 임베딩에 속하지 않는 나머지(분류 헤드, pooler, DeBERTa 의 상대 위치 임베딩 등)는 lr 로 학습한다.
    백본 구조가 달라도(RoBERTa, DeBERTa, ELECTRA) 빠지는 파라미터가 없게 이름이 아니라 소속으로 나눈다.
    """
    layers = model.base_model.encoder.layer
    n = len(layers)
    groups = []
    taken = set()

    def add(named, rate):
        named = [(name, p) for name, p in named if id(p) not in taken and p.requires_grad]
        taken.update(id(p) for _, p in named)
        no_decay = [p for name, p in named if "bias" in name or "LayerNorm" in name]
        decayed = [p for name, p in named if not ("bias" in name or "LayerNorm" in name)]
        if decayed:
            groups.append({"params": decayed, "lr": rate, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": rate, "weight_decay": 0.0})

    for i, layer in enumerate(layers):
        add(layer.named_parameters(), lr * decay ** (n - i))
    add(model.base_model.embeddings.named_parameters(), lr * decay ** (n + 1))
    add(model.named_parameters(), lr)          # 남은 것 전부
    return groups


LOSS_DEFAULTS = {
    "bce": (0.0, 0.0),
    "softjaccard": (0.5, 0.0),
    "jacc_macrof1": (0.5, 0.2),
}


def resolve_loss_weights(
    loss_name: str,
    jaccard_weight: float | None = None,
    macro_f1_weight: float | None = None,
) -> tuple[float, float]:
    """Resolve auxiliary Soft-Jaccard / Soft-Macro-F1 loss weights from --loss preset and overrides."""
    if loss_name not in LOSS_DEFAULTS:
        raise ValueError(f"Unsupported --loss={loss_name}; expected {tuple(LOSS_DEFAULTS)}")
    def_j, def_m = LOSS_DEFAULTS[loss_name]
    jw = float(def_j if jaccard_weight is None else jaccard_weight)
    mw = float(def_m if macro_f1_weight is None else macro_f1_weight)
    return jw, mw


def compute_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight_t: torch.Tensor,
    jaccard_weight: float = 0.0,
    macro_f1_weight: float = 0.0,
) -> torch.Tensor:
    """Weighted BCE loss with optional Soft-Jaccard and Soft-Macro-F1 auxiliary terms."""
    logits_f = logits.float()
    elem = torch.nn.functional.binary_cross_entropy_with_logits(logits_f, target, reduction="none")
    w = torch.where(target > 0.5, pos_weight_t.unsqueeze(0).expand_as(target), torch.ones_like(elem))
    loss = (elem * w).mean()
    if jaccard_weight > 0.0 or macro_f1_weight > 0.0:
        probs = torch.sigmoid(logits_f)
        if jaccard_weight > 0.0:
            inter = (probs * target).sum(dim=1)
            union = (probs + target - probs * target).sum(dim=1).clamp_min(1e-6)
            loss = loss + jaccard_weight * (1.0 - inter / union).mean()
        if macro_f1_weight > 0.0:
            tp_c = (probs * target).sum(dim=0)
            p_c = probs.sum(dim=0)
            y_c = target.sum(dim=0)
            macro_dice = (2.0 * tp_c + 1.0) / (p_c + y_c + 1.0)
            loss = loss + macro_f1_weight * (1.0 - macro_dice.mean())
    return loss


def sample_epoch_items(
    prefixes,
    samples_per_call: int,
    rng: np.random.Generator,
    include_full: bool = False,
    full_only: bool = False,
):
    """Sample (call_idx, prefix_len) items for one training epoch."""
    if full_only and include_full:
        raise ValueError("--full-only and --include-full cannot be enabled together")
    if full_only:
        return [(i, len(p)) for i, p in enumerate(prefixes) for _ in range(samples_per_call)]
    items = [(i, int(rng.integers(1, len(p) + 1))) for i, p in enumerate(prefixes) for _ in range(samples_per_call)]
    if include_full:
        items += [(i, len(p)) for i, p in enumerate(prefixes)]
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--ids", help="학습에 쓸 통화 목록 (없으면 캐시 전체)")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", default="space", choices=["space", "speaker"])
    ap.add_argument("--truncation-side", default="right", choices=["right", "left"])
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--samples-per-call", type=int, default=2)
    ap.add_argument("--include-full", action="store_true", help="통화 전체도 epoch 마다 1개씩 더 넣는다")
    ap.add_argument("--full-only", action="store_true", help="앞부분 표본 대신 통화 전체 길이만 샘플링한다")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1, help="이 횟수만큼 기울기를 모아 한 번 갱신 (메모리가 모자랄 때)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--llrd", type=float, default=0.8)
    ap.add_argument("--pos-weight-power", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--count-classes", type=int, default=0,
                    help="증상 개수 보조 출력의 칸 수 (0 = 끔). 3 이면 1개 / 2개 / 3개 이상")
    ap.add_argument("--count-weight", type=float, default=0.5, help="보조 손실(교차 엔트로피)에 곱하는 값")
    ap.add_argument("--max-steps", type=int, default=0, help="점검용. 0 이면 끝까지")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing. 큰 백본이 8GB 에 안 들어갈 때 활성값 메모리를 줄인다 (계산은 약 30%% 늘어남)")
    ap.add_argument("--loss", default="bce", choices=["bce", "softjaccard", "jacc_macrof1"],
                    help="학습 손실 프리셋 (기본 bce; softjaccard는 0.5, jacc_macrof1은 0.5/0.2)")
    ap.add_argument("--jaccard-weight", type=float, default=None, help="Soft-Jaccard 보조 손실 가중치 (기본 프리셋 따름)")
    ap.add_argument("--macro-f1-weight", type=float, default=None, help="Soft-Macro-F1 보조 손실 가중치 (기본 프리셋 따름)")
    ap.add_argument("--log-every", type=int, default=200, help="이 스텝마다 진행 기록")
    args = ap.parse_args()
    if args.full_only and args.include_full:
        raise SystemExit("--full-only 와 --include-full 은 동시에 지정할 수 없습니다.")
    args.jaccard_weight, args.macro_f1_weight = resolve_loss_weights(
        args.loss, args.jaccard_weight, args.macro_f1_weight
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.txt", "a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    calls = load_cache(Path(args.calls), read_ids(Path(args.ids)) if args.ids else None)
    labels = label_matrix(calls).astype(np.float32)
    prefixes = [prefix_texts(c, args.fmt) for c in calls]
    pos = labels.sum(axis=0)
    pos_weight = ((len(labels) - pos) / np.maximum(pos, 1)) ** args.pos_weight_power
    say(f"학습 통화 {len(calls)}건, 설정 {json.dumps(vars(args), ensure_ascii=False)}")
    say("pos_weight " + ", ".join(f"{c} {w:.2f}" for c, w in zip(SYMPTOM_CLASSES, pos_weight)))

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    tokenizer.truncation_side = args.truncation_side
    # 보조 출력을 켜면 분류 헤드의 출력이 9 + count_classes 개가 된다. 앞 9개가 증상, 뒤가 증상 개수 칸이다.
    n_sym = len(SYMPTOM_CLASSES)
    names = list(SYMPTOM_CLASSES) + [f"count_{k + 1}" for k in range(args.count_classes)]
    model = AutoModelForSequenceClassification.from_pretrained(
        args.backbone, num_labels=len(names), problem_type="multi_label_classification",
        id2label=dict(enumerate(names)), label2id={c: i for i, c in enumerate(names)})
    count_target = np.clip(labels.sum(axis=1).astype(np.int64), 1, max(args.count_classes, 1)) - 1
    if args.count_classes:
        say("증상 개수 칸별 통화 비율 " + ", ".join(
            f"{k + 1}{'+' if k + 1 == args.count_classes else ''}개 {(count_target == k).mean():.3f}"
            for k in range(args.count_classes)))
    device = torch.device("cuda")
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.to(device)

    per_epoch = len(calls) * (args.samples_per_call + int(args.include_full))
    steps_per_epoch = per_epoch // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    updates = total_steps // args.grad_accum       # 실제 가중치 갱신 횟수
    optimizer = torch.optim.AdamW(llrd_groups(model, args.lr, args.llrd, 0.01))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * updates), updates)
    scaler = torch.amp.GradScaler("cuda")
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    history = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng([args.seed, epoch])
        # (통화 번호, 앞부분 길이) 표본. --full-only 면 통화 전체 길이만 사용한다.
        items = sample_epoch_items(
            prefixes, args.samples_per_call, rng, include_full=args.include_full, full_only=args.full_only
        )
        order = rng.permutation(len(items))
        model.train()
        running, seen = 0.0, 0
        for b in range(steps_per_epoch):
            batch = [items[j] for j in order[b * args.batch_size:(b + 1) * args.batch_size]]
            enc = tokenizer([prefixes[i][t - 1] for i, t in batch], truncation=True,
                            max_length=args.max_length, padding=True, return_tensors="pt").to(device)
            target = torch.tensor(labels[[i for i, _ in batch]], device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            loss = compute_loss(
                logits[:, :n_sym], target, pos_weight_t,
                jaccard_weight=args.jaccard_weight, macro_f1_weight=args.macro_f1_weight,
            )
            if args.count_classes:
                ctarget = torch.tensor(count_target[[i for i, _ in batch]], device=device)
                loss = loss + args.count_weight * torch.nn.functional.cross_entropy(logits.float()[:, n_sym:], ctarget)
            scaler.scale(loss / args.grad_accum).backward()
            running += loss.item()
            seen += 1
            step += 1
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % args.log_every == 0:
                say(f"epoch {epoch} step {step}/{total_steps} loss {running / seen:.4f} "
                    f"lr {scheduler.get_last_lr()[0]:.2e} 경과 {time.time() - t0:.0f}s")
            if args.max_steps and step >= args.max_steps:
                break
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "seconds": round(time.time() - t0, 1)})
        say(f"epoch {epoch} 끝: 평균 loss {running / max(seen, 1):.4f}")
        if args.max_steps and step >= args.max_steps:
            break

    model.save_pretrained(out / "best_model", safe_serialization=True)
    tokenizer.save_pretrained(out / "best_model")
    config = dict(vars(args), n_calls=len(calls), total_steps=step, history=history,
                  pos_weight=[float(w) for w in pos_weight], training_seconds=round(time.time() - t0, 1),
                  peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated()))
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"저장 {out / 'best_model'} ({config['training_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
