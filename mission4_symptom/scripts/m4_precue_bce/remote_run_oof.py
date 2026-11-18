"""Remote L4 5-fold OOF generator over fit (26,280 calls) for leak-free Gap 3 halting evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/content/m4_precue")


def run_cmd(cmd: list[str], log_path: Path | None = None) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    f = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if f:
                f.write(line)
                f.flush()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed rc={rc}: {' '.join(cmd)}")
    finally:
        if f:
            f.close()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=2026)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--precue-pos-weight", type=float, default=-1.0)
    ap.add_argument("--jaccard-weight", type=float, default=0.0)
    ap.add_argument("--macro-f1-weight", type=float, default=0.0)
    args = ap.parse_args()

    t0 = time.time()
    out_root = Path("/content/oof_runs") / args.oof_name
    out_root.mkdir(parents=True, exist_ok=True)
    master_log = out_root / "oof_runner.log"

    fit_ids = [line.strip() for line in (ROOT / "inputs/fit_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not (out_root / "fold0_held.txt").exists():
        fold_idx = np.random.default_rng(args.split_seed).permutation(len(fit_ids)) % args.n_folds
        for k in range(args.n_folds):
            held = [cid for cid, fk in zip(fit_ids, fold_idx) if fk == k]
            tr = [cid for cid, fk in zip(fit_ids, fold_idx) if fk != k]
            (out_root / f"fold{k}_held.txt").write_text("\n".join(held) + "\n", encoding="utf-8")
            (out_root / f"fold{k}_fit.txt").write_text("\n".join(tr) + "\n", encoding="utf-8")

    sys.path.insert(0, str(ROOT))
    from m4 import simulate as sim  # noqa: E402

    for k in range(args.n_folds):
        fold_dir = out_root / f"fold{k}"
        oof_npz = fold_dir / "oof_prefix.npz"
        if (fold_dir / "DONE.json").exists() and oof_npz.exists():
            print(f"[fold {k}] already finished, skipping.", flush=True)
            continue
        fold_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                sys.executable, "-u", str(ROOT / "train_precue.py"),
                "--calls", str(ROOT / "inputs/train.jsonl"),
                "--ids", str(out_root / f"fold{k}_fit.txt"),
                "--forbid-ids", str(ROOT / "inputs/dev_ids.txt"),
                "--backbone", str(ROOT / "tapt"),
                "--out", str(fold_dir),
                "--precue-pos-weight", str(args.precue_pos_weight),
                "--jaccard-weight", str(args.jaccard_weight),
                "--macro-f1-weight", str(args.macro_f1_weight),
                "--seed", str(args.seed),
                "--epochs", "3", "--samples-per-call", "2", "--batch-size", "16",
                "--lr", "5e-5", "--llrd", "0.8", "--pos-weight-power", "0.5",
                "--max-length", "512", "--fmt", "space", "--truncation-side", "right",
                "--save-every", "2500", "--log-every", "400", "--resume",
            ],
            log_path=master_log,
        )
        run_cmd(
            [
                sys.executable, "-u", str(ROOT / "predict_prefix.py"),
                "--model", str(fold_dir / "best_model"),
                "--calls", str(ROOT / "inputs/train.jsonl"),
                "--ids", str(out_root / f"fold{k}_held.txt"),
                "--out", str(oof_npz),
                "--fmt", "space", "--max-length", "512", "--truncation-side", "right",
            ],
            log_path=master_log,
        )
        (fold_dir / "DONE.json").write_text(json.dumps({"fold": k, "npz_sha256": file_sha256(oof_npz)}), encoding="utf-8")

    # Merge all 5 OOF tables in deterministic fold order and verify disjoint coverage of 26,280 fit calls
    tables = [sim.PrefixTable.load(out_root / f"fold{k}/oof_prefix.npz") for k in range(args.n_folds)]
    merged_ids = [cid for t in tables for cid in t.call_ids]
    assert len(merged_ids) == len(fit_ids) and set(merged_ids) == set(fit_ids), (len(merged_ids), len(fit_ids))
    merged_offsets = np.concatenate([[0], np.cumsum(np.concatenate([t.n_utt for t in tables]))]).astype(np.int64)
    merged_probs = np.concatenate([t.probs for t in tables], axis=0).astype(np.float32)
    merged_labels = np.concatenate([t.labels for t in tables], axis=0).astype(np.uint8)
    merged_path = out_root / "fit_oof_merged.npz"
    sim.PrefixTable(merged_ids, merged_offsets, merged_probs, merged_labels).save(merged_path)
    merged_sha = file_sha256(merged_path)

    light_zip = Path(f"/content/{args.oof_name}_light.zip")
    zip_files = [
        str(merged_path),
        str(master_log),
    ]
    for k in range(args.n_folds):
        fd = out_root / f"fold{k}"
        for fn in ["oof_prefix.meta.json", "run_config.json", "optimizer_audit.json", "initialization.json", "train_log.txt", "DONE.json"]:
            if (fd / fn).exists():
                zip_files.append(str(fd / fn))
        zip_files.extend([str(out_root / f"fold{k}_fit.txt"), str(out_root / f"fold{k}_held.txt")])

    run_cmd(["zip", "-q", str(light_zip)] + zip_files)
    done_meta = {
        "done": True,
        "oof_name": args.oof_name,
        "n_calls": len(merged_ids),
        "n_moments": int(merged_probs.shape[0]),
        "merged_sha256": merged_sha,
        "light_zip": str(light_zip),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_root / "DONE.json").write_text(json.dumps(done_meta, indent=2), encoding="utf-8")
    print("OOF_ALL_DONE:", json.dumps(done_meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
