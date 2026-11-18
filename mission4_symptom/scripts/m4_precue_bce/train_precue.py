"""Pre-cue evidence-aware prefix training with full state checkpointing and strict leak checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

THIS_DIR = Path(__file__).resolve().parent
LOCAL_FALLBACK_SRC = Path("/Users/sunggyu/Desktop/이성규/DCC/artifacts_private/e026_same_input_replay_20261116/source")
if (THIS_DIR / "m4").exists():
    sys.path.insert(0, str(THIS_DIR))
elif LOCAL_FALLBACK_SRC.exists():
    sys.path.insert(0, str(LOCAL_FALLBACK_SRC))

from m4 import SYMPTOM_CLASSES  # noqa: E402
from m4.data import Call, label_matrix, load_cache, prefix_texts, read_ids  # noqa: E402

SYMPTOM_PATTERNS = {
    "고열": r"고열|발열|열이|열은|열도|열나|열 나|체온|[34]\d(?:\.\d)?도",
    "구토": r"구토|토하|토해|토했|토한|토를|토도|토가|게워|게우|게웠",
    "두통": r"두통|머리.{0,8}(?:아프|아파|아픈|통증|깨질|지끈)",
    "복통": r"복통|배가.{0,10}(?:아프|아파|아픈|통증)|배 아|배를.{0,5}(?:잡|부여)|아랫배|윗배|명치.{0,10}(?:아프|아파|통증)",
    "어지러움": r"어지|현기증|핑.{0,3}돌|빙빙",
    "열상": r"열상|찢어|찢어진|베였|베여|베인|베었",
    "오심": r"오심|메스|매스|울렁|미식|니글|헛구역|구역질",
    "전신쇠약": r"전신쇠약|기력|기운.{0,6}(?:없|빠|떨어)|힘이.{0,6}(?:없|빠)|무기력",
    "호흡곤란": r"호흡곤란|호흡.{0,10}(?:힘|곤란|어려)|숨.{0,12}(?:못|힘|어려|안 쉬|안쉬|차|막)|숨차",
}
SRX = [re.compile(SYMPTOM_PATTERNS[c]) for c in SYMPTOM_CLASSES]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def state_hash(state: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((str(t.dtype), list(t.shape))).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def norm_transcript(call: Call) -> str:
    return " ".join(" ".join(call.texts).split())


def verify_non_overlapping(train_calls: Sequence[Call], calls_path: Path, forbid_ids_path: Optional[Path]) -> None:
    if not forbid_ids_path:
        return
    forbid_ids = read_ids(forbid_ids_path)
    overlap_ids = set(c.call_id for c in train_calls) & set(forbid_ids)
    if overlap_ids:
        raise ValueError(f"ID overlap with forbidden split: {len(overlap_ids)}")
    forbid_calls = load_cache(calls_path, forbid_ids)
    train_norms = {norm_transcript(c) for c in train_calls}
    forbid_norms = {norm_transcript(c) for c in forbid_calls}
    overlap_text = train_norms & forbid_norms
    if overlap_text:
        raise ValueError(f"Normalized transcript overlap with forbidden split: {len(overlap_text)}")


def cue_steps_matrix(calls: Sequence[Call]) -> np.ndarray:
    """Return (N, 9) 1-based first lexical cue step per call and symptom (10**9 if uncued)."""
    out = np.full((len(calls), len(SYMPTOM_CLASSES)), 10**9, dtype=np.int32)
    for i, c in enumerate(calls):
        for s_idx, txt in enumerate(c.texts, start=1):
            for k, rx in enumerate(SRX):
                if out[i, k] == 10**9 and rx.search(txt):
                    out[i, k] = s_idx
    return out


def compute_prefix_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight_t: torch.Tensor,
    is_precue_t: Optional[torch.Tensor],
    precue_pos_weight: float,
    jaccard_weight: float = 0.0,
    macro_f1_weight: float = 0.0,
) -> torch.Tensor:
    """Weighted BCE matching nn.BCEWithLogitsLoss when precue_pos_weight < 0, plus optional SoftJaccard / SoftMacroF1."""
    logits_f = logits.float()
    elem = F.binary_cross_entropy_with_logits(logits_f, target, reduction="none")
    pw = pos_weight_t.unsqueeze(0).expand_as(target)
    if precue_pos_weight >= 0.0 and is_precue_t is not None:
        pw = torch.where(is_precue_t, torch.full_like(pw, precue_pos_weight), pw)
    w = torch.where(target > 0.5, pw, torch.ones_like(pw))
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


def llrd_groups(model: torch.nn.Module, lr: float, decay: float, weight_decay: float) -> List[Dict[str, object]]:
    layers = model.base_model.encoder.layer
    n = len(layers)
    groups: List[Dict[str, object]] = []
    taken = set()

    def add(named, rate: float) -> None:
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
    add(model.named_parameters(), lr)
    return groups


def load_model_checked(backbone: Path, local_files_only: bool) -> Tuple[torch.nn.Module, Dict[str, object]]:
    names = list(SYMPTOM_CLASSES)
    model, info = AutoModelForSequenceClassification.from_pretrained(
        str(backbone),
        num_labels=len(names),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(names)),
        label2id={c: i for i, c in enumerate(names)},
        local_files_only=local_files_only,
        output_loading_info=True,
    )
    missing = list(info.get("missing_keys", []))
    unexpected = list(info.get("unexpected_keys", []))
    mismatch = list(info.get("mismatched_keys", []))
    errors = list(info.get("error_msgs", []))
    if mismatch or errors or any(not k.startswith("classifier.") for k in missing):
        raise ValueError(f"Unexpected model loading issue: {info}")
    if any(not k.startswith("lm_head.") for k in unexpected):
        raise ValueError(f"Unexpected non-lm_head keys: {unexpected}")
    return model, {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatch,
        "error_msgs": errors,
    }


def make_epoch_items(prefixes: Sequence[List[str]], samples_per_call: int, seed: int, epoch: int) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    rng = np.random.default_rng([seed, epoch])
    items = [
        (i, int(rng.integers(1, len(p) + 1)))
        for i, p in enumerate(prefixes)
        for _ in range(samples_per_call)
    ]
    order = rng.permutation(len(items))
    return items, order


def save_resume_state(
    resume_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    amp: bool,
    epoch: int,
    next_batch: int,
    completed_epochs: int,
    global_step: int,
    updates: int,
    skips: int,
    running_loss: float,
    running_seen: int,
    history: list,
    config_digest: str,
) -> None:
    tmp = resume_path.with_suffix(".partial")
    np_state = np.random.get_state()
    torch.save(
        {
            "config_digest": config_digest,
            "epoch": epoch,
            "next_batch": next_batch,
            "completed_epochs": completed_epochs,
            "global_step": global_step,
            "updates": updates,
            "skips": skips,
            "running_loss": running_loss,
            "running_seen": running_seen,
            "history": history,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if amp else None,
            "python_rng": random.getstate(),
            "numpy_rng": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
        },
        tmp,
    )
    tmp.replace(resume_path)


def run_training(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train_log.txt").open("a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    calls_path = Path(args.calls)
    fit_ids = read_ids(Path(args.ids))
    calls = load_cache(calls_path, fit_ids)
    verify_non_overlapping(calls, calls_path, Path(args.forbid_ids) if args.forbid_ids else None)

    labels = label_matrix(calls).astype(np.float32)
    prefixes = [prefix_texts(c, args.fmt) for c in calls]
    cue_steps = cue_steps_matrix(calls) if args.precue_pos_weight >= 0.0 else None

    pos = labels.sum(axis=0)
    pos_weight = ((len(labels) - pos) / np.maximum(pos, 1)) ** args.pos_weight_power

    # Stable digest covering scientific configuration (excluding resume/stop control flags)
    sci_cfg = {
        "backbone": str(Path(args.backbone).name),
        "ids_sha256": file_sha256(Path(args.ids)),
        "calls_sha256": file_sha256(calls_path),
        "n_calls": len(calls),
        "fmt": args.fmt,
        "truncation_side": args.truncation_side,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "samples_per_call": args.samples_per_call,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "llrd": args.llrd,
        "pos_weight_power": args.pos_weight_power,
        "precue_pos_weight": args.precue_pos_weight,
        "jaccard_weight": args.jaccard_weight,
        "macro_f1_weight": args.macro_f1_weight,
        "seed": args.seed,
        "initial_head_sha256": file_sha256(Path(args.initial_head)) if args.initial_head else None,
    }
    config_digest = hashlib.sha256(json.dumps(sci_cfg, sort_keys=True).encode()).hexdigest()

    say(f"Calls={len(calls)} precue_pos_weight={args.precue_pos_weight} seed={args.seed} digest={config_digest[:12]}")
    say("pos_weight=" + ", ".join(f"{c}:{w:.4f}" for c, w in zip(SYMPTOM_CLASSES, pos_weight)))

    tokenizer = AutoTokenizer.from_pretrained(str(args.backbone), local_files_only=not args.allow_network)
    tokenizer.truncation_side = args.truncation_side
    model, loading_info = load_model_checked(Path(args.backbone), local_files_only=not args.allow_network)
    if args.initial_head:
        head = torch.load(Path(args.initial_head), map_location="cpu", weights_only=True)
        model.load_state_dict({**model.state_dict(), **head}, strict=True)

    init_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    init_hash = state_hash(init_state)
    atomic_json(out / "initialization.json", {"initial_state_sha256": init_hash, "loading_info": loading_info, "sci_cfg": sci_cfg})

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    amp = device.type == "cuda"
    model.to(device)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    per_epoch = len(calls) * args.samples_per_call
    steps_per_epoch = per_epoch // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    updates_total = total_steps // args.grad_accum
    optimizer = torch.optim.AdamW(llrd_groups(model, args.lr, args.llrd, 0.01))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * updates_total), updates_total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    resume_path = out / "training_state_last.pt"
    history = []
    completed_epochs = 0
    resume_epoch = 1
    resume_next_batch = 0
    global_step = 0
    updates = 0
    skips = 0
    running_loss = 0.0
    running_seen = 0

    if args.resume and resume_path.exists():
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        if st["config_digest"] != config_digest:
            raise ValueError("Resume checkpoint config_digest mismatch")
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        completed_epochs = int(st["completed_epochs"])
        resume_epoch = int(st["epoch"])
        resume_next_batch = int(st["next_batch"])
        global_step = int(st["global_step"])
        updates = int(st["updates"])
        skips = int(st["skips"])
        running_loss = float(st["running_loss"])
        running_seen = int(st["running_seen"])
        history = list(st["history"])
        torch.set_rng_state(st["cpu_rng"])
        if amp and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        random.setstate(st["python_rng"])
        ns = st["numpy_rng"]
        np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        say(f"RESUMED from completed_epochs={completed_epochs} epoch={resume_epoch} next_batch={resume_next_batch} global_step={global_step}")

    if amp:
        torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    precue_rows_total = 0
    pos_rows_total = 0

    for epoch in range(1, args.epochs + 1):
        if epoch <= completed_epochs:
            continue
        items, order = make_epoch_items(prefixes, args.samples_per_call, args.seed, epoch)
        model.train()
        start_b = resume_next_batch if epoch == resume_epoch else 0
        if start_b == 0:
            running_loss, running_seen = 0.0, 0

        for b in range(start_b, steps_per_epoch):
            batch = [items[j] for j in order[b * args.batch_size:(b + 1) * args.batch_size]]
            call_idxs = [i for i, _ in batch]
            steps_t = np.array([t for _, t in batch], dtype=np.int32)[:, None]
            texts = [prefixes[i][t - 1] for i, t in batch]
            enc = tokenizer(texts, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt").to(device)
            target = torch.tensor(labels[call_idxs], device=device)

            is_precue_t = None
            if cue_steps is not None:
                cs = cue_steps[call_idxs]
                is_precue_np = (cs < 10**9) & (steps_t < cs)
                precue_rows_total += int((is_precue_np & (labels[call_idxs] > 0.5)).sum())
                pos_rows_total += int((labels[call_idxs] > 0.5).sum())
                is_precue_t = torch.tensor(is_precue_np, dtype=torch.bool, device=device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            loss = compute_prefix_bce_loss(
                logits, target, pos_weight_t, is_precue_t, args.precue_pos_weight, args.jaccard_weight, args.macro_f1_weight
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} batch={b} step={global_step + 1}")

            scaled_loss = loss / args.grad_accum
            if amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            running_loss += float(loss.item())
            running_seen += 1
            global_step += 1

            if global_step % args.grad_accum == 0:
                if amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    advanced = scaler.get_scale() >= scale_before
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    advanced = True
                if advanced:
                    scheduler.step()
                    updates += 1
                else:
                    skips += 1
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_every == 0:
                say(
                    f"epoch {epoch} step {global_step}/{total_steps} loss={running_loss / running_seen:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} upd={updates} skips={skips} elapsed={time.time() - t0:.1f}s"
                )

            if args.stop_after_steps and global_step >= args.stop_after_steps:
                next_ep = epoch + 1 if (b + 1) >= steps_per_epoch else epoch
                next_b = 0 if (b + 1) >= steps_per_epoch else (b + 1)
                comp_ep = epoch if (b + 1) >= steps_per_epoch else completed_epochs
                save_resume_state(
                    resume_path, model, optimizer, scheduler, scaler, amp,
                    next_ep, next_b, comp_ep, global_step, updates, skips,
                    running_loss, running_seen, history, config_digest,
                )
                say(f"STOP_AFTER_STEPS step={global_step}")
                log.close()
                return 0

            if args.save_every > 0 and global_step % args.save_every == 0 and (b + 1) < steps_per_epoch:
                save_resume_state(
                    resume_path, model, optimizer, scheduler, scaler, amp,
                    epoch, b + 1, completed_epochs, global_step, updates, skips,
                    running_loss, running_seen, history, config_digest,
                )
                say(f"CHECKPOINT step={global_step}")

        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(running_seen, 1),
            "updates": updates,
            "amp_skips": skips,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        completed_epochs = epoch
        resume_epoch = epoch + 1
        resume_next_batch = 0
        save_resume_state(
            resume_path, model, optimizer, scheduler, scaler, amp,
            epoch + 1, 0, completed_epochs, global_step, updates, skips,
            0.0, 0, history, config_digest,
        )
        say(f"EPOCH_COMPLETE epoch={epoch} train_loss={row['train_loss']:.4f}")

    # Final preservation
    model.save_pretrained(out / "best_model", safe_serialization=True)
    tokenizer.save_pretrained(out / "best_model")
    final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    final_hash = state_hash(final_state)
    final_pt = out / "final.pt"
    tmp_f = final_pt.with_suffix(".partial")
    torch.save({"model": final_state, "config_digest": config_digest, "state_hash": final_hash, "seed": args.seed, "epoch": args.epochs}, tmp_f)
    tmp_f.replace(final_pt)

    opt_steps = [int(v["step"].item()) if torch.is_tensor(v["step"]) else int(v["step"]) for v in optimizer.state.values() if "step" in v]
    audit = {
        "adam_state_count": len(opt_steps),
        "min_step": min(opt_steps, default=0),
        "max_step": max(opt_steps, default=0),
        "updates": updates,
        "amp_skips": skips,
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "initial_state_sha256": init_hash,
        "final_state_sha256": final_hash,
        "precue_pos_elements": precue_rows_total,
        "total_pos_elements": pos_rows_total,
    }
    atomic_json(out / "optimizer_audit.json", audit)
    cfg_out = {
        **vars(args),
        "n_calls": len(calls),
        "total_steps": global_step,
        "history": history,
        "pos_weight": [float(w) for w in pos_weight],
        "training_seconds": round(time.time() - t0, 1),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if amp else 0,
        "config_digest": config_digest,
        "final_pt_sha256": file_sha256(final_pt),
        "training_state_last_sha256": file_sha256(resume_path),
    }
    atomic_json(out / "run_config.json", cfg_out)
    say(f"TRAINING_COMPLETE final_sha256={cfg_out['final_pt_sha256']}")
    log.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--forbid-ids", default="")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--initial-head", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", default="space", choices=["space", "speaker"])
    ap.add_argument("--truncation-side", default="right", choices=["right", "left"])
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--samples-per-call", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--llrd", type=float, default=0.8)
    ap.add_argument("--pos-weight-power", type=float, default=0.5)
    ap.add_argument("--precue-pos-weight", type=float, default=1.0, help="<0 matches exact E002 BCEWithLogitsLoss")
    ap.add_argument("--jaccard-weight", type=float, default=0.0, help="SoftJaccard loss weight added to BCE")
    ap.add_argument("--macro-f1-weight", type=float, default=0.0, help="Batch smoothed SoftMacroF1 loss weight added to BCE")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--stop-after-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--allow-network", action="store_true")
    return ap


if __name__ == "__main__":
    raise SystemExit(run_training(build_parser().parse_args()))
