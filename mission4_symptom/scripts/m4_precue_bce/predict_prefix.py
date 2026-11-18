"""Generate prefix probability table (.npz) compatible with m4.simulate.PrefixTable."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
LOCAL_FALLBACK_SRC = Path("/Users/sunggyu/Desktop/이성규/DCC/artifacts_private/e026_same_input_replay_20261116/source")
if (THIS_DIR / "m4").exists():
    sys.path.insert(0, str(THIS_DIR))
elif LOCAL_FALLBACK_SRC.exists():
    sys.path.insert(0, str(LOCAL_FALLBACK_SRC))

from m4.data import label_matrix, load_cache, prefix_texts, read_ids  # noqa: E402
from m4.simulate import PrefixTable  # noqa: E402


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calls", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", default="space", choices=["space", "speaker"])
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--truncation-side", default="right", choices=["right", "left"])
    ap.add_argument("--token-budget", type=int, default=24000)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    calls = load_cache(Path(args.calls), read_ids(Path(args.ids)))
    texts, offsets = [], [0]
    for call in calls:
        texts.extend(prefix_texts(call, args.fmt))
        offsets.append(len(texts))

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.truncation_side = args.truncation_side
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device).eval()

    t0 = time.time()
    enc = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)["input_ids"]
    order = np.argsort([-len(x) for x in enc], kind="stable")
    n_sym = label_matrix(calls).shape[1]
    probs = np.zeros((len(texts), n_sym), dtype=np.float32)
    pad = tokenizer.pad_token_id
    pos = 0
    with torch.inference_mode():
        while pos < len(order):
            width = max(len(enc[order[pos]]), 1)
            size = max(1, min(256, args.token_budget // width))
            idx = order[pos : pos + size]
            ids = torch.full((len(idx), width), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for row, j in enumerate(idx):
                ids[row, : len(enc[j])] = torch.tensor(enc[j])
                mask[row, : len(enc[j])] = 1
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            probs[idx] = torch.sigmoid(logits.float()[:, :n_sym]).cpu().numpy()
            pos += size
    seconds = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    PrefixTable([c.call_id for c in calls], np.asarray(offsets, dtype=np.int64), probs, label_matrix(calls)).save(out)
    meta = {
        "model": str(args.model),
        "calls": str(args.calls),
        "ids": str(args.ids),
        "fmt": args.fmt,
        "max_length": args.max_length,
        "truncation_side": args.truncation_side,
        "n_calls": len(calls),
        "n_moments": len(texts),
        "seconds": round(seconds, 1),
        "truncated_moments": int(sum(len(x) >= args.max_length for x in enc)),
        "npz_sha256": file_sha256(out),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
