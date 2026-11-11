"""통화의 모든 누적 입력(step 1..n)에 대한 9종 확률표를 만든다.

    python scripts/prefix_probs.py --model <best_model 폴더> --calls runs/cache/train.jsonl \
        --ids runs/cache/dev_ids.txt --out runs/<실험>/dev_prefix.npz

발화가 하나 늘 때마다 그때까지의 전사 전체를 다시 인코딩한 결과와 같다 (추론 Predictor 와 같은 입력).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4.data import label_matrix, load_cache, prefix_texts, read_ids  # noqa: E402
from m4.simulate import PrefixTable  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calls", required=True)
    ap.add_argument("--ids", help="이 목록의 통화만 쓴다 (없으면 전체)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", default="space", choices=["space", "speaker"])
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--truncation-side", default="right", choices=["right", "left"],
                    help="right = 앞부분 유지(예선 방식), left = 최근 발화 유지")
    ap.add_argument("--token-budget", type=int, default=24000, help="배치 하나의 (길이 x 개수) 상한")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    calls = load_cache(Path(args.calls), read_ids(Path(args.ids)) if args.ids else None)
    if args.limit:
        calls = calls[: args.limit]
    texts, offsets = [], [0]
    for call in calls:
        texts.extend(prefix_texts(call, args.fmt))
        offsets.append(len(texts))
    print(f"통화 {len(calls)}건, 순간 {len(texts)}개")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.truncation_side = args.truncation_side
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    t0 = time.time()
    # 빈 문자열(전사가 빈 발화만 들어온 순간)도 [CLS][SEP] 로 인코딩해 확률을 낸다.
    enc = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)["input_ids"]
    order = np.argsort([-len(x) for x in enc], kind="stable")
    probs = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    pad = tokenizer.pad_token_id
    pos = 0
    with torch.inference_mode():
        while pos < len(order):
            width = len(enc[order[pos]])
            size = max(1, min(256, args.token_budget // width))
            idx = order[pos:pos + size]
            ids = torch.full((len(idx), width), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for row, j in enumerate(idx):
                ids[row, : len(enc[j])] = torch.tensor(enc[j])
                mask[row, : len(enc[j])] = 1
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            probs[idx] = torch.sigmoid(logits.float()).cpu().numpy()
            pos += size
    seconds = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    PrefixTable([c.call_id for c in calls], np.asarray(offsets), probs, label_matrix(calls)).save(out)
    meta = {"model": str(args.model), "calls": str(args.calls), "ids": args.ids, "fmt": args.fmt,
            "max_length": args.max_length, "truncation_side": args.truncation_side,
            "n_calls": len(calls), "n_moments": len(texts), "seconds": round(seconds, 1),
            "truncated_moments": int(sum(len(x) >= args.max_length for x in enc))}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
