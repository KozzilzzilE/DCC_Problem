"""Remote Colab single-fold runner with heartbeat output and idempotent resume for OOF."""
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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def ensure_splits(out_root: Path, n_folds: int, split_seed: int) -> None:
    if (out_root / "fold0_held.txt").exists():
        return
    fit_ids = [line.strip() for line in (ROOT / "inputs/fit_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    fold_idx = np.random.default_rng(split_seed).permutation(len(fit_ids)) % n_folds
    for k in range(n_folds):
        held = [cid for cid, fk in zip(fit_ids, fold_idx) if fk == k]
        tr = [cid for cid, fk in zip(fit_ids, fold_idx) if fk != k]
        (out_root / f"fold{k}_held.txt").write_text("\n".join(held) + "\n", encoding="utf-8")
        (out_root / f"fold{k}_fit.txt").write_text("\n".join(tr) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-name", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=2026)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--precue-pos-weight", type=float, default=-1.0)
    ap.add_argument("--jaccard-weight", type=float, default=0.0)
    ap.add_argument("--macro-f1-weight", type=float, default=0.0)
    args = ap.parse_args()

    out_root = Path("/content/oof_runs") / args.oof_name
    out_root.mkdir(parents=True, exist_ok=True)
    ensure_splits(out_root, args.n_folds, args.split_seed)

    fold_dir = out_root / f"fold{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    done_path = fold_dir / "DONE.json"
    oof_npz = fold_dir / "oof_prefix.npz"
    if done_path.exists() and oof_npz.exists():
        print("FOLD_ALREADY_DONE", done_path.read_text(encoding="utf-8"), flush=True)
        return 0

    worker_log = fold_dir / "fold_worker.log"
    train_cmd = (
        f"{sys.executable} -u {ROOT / 'train_precue.py'} "
        f"--calls {ROOT / 'inputs/train.jsonl'} "
        f"--ids {out_root / f'fold{args.fold}_fit.txt'} "
        f"--forbid-ids {ROOT / 'inputs/dev_ids.txt'} "
        f"--backbone {ROOT / 'tapt'} "
        f"--out {fold_dir} "
        f"--precue-pos-weight {args.precue_pos_weight} "
        f"--jaccard-weight {args.jaccard_weight} "
        f"--macro-f1-weight {args.macro_f1_weight} "
        f"--seed {args.seed} --epochs 3 --samples-per-call 2 --batch-size 16 "
        f"--lr 5e-5 --llrd 0.8 --pos-weight-power 0.5 --max-length 512 "
        f"--fmt space --truncation-side right --save-every 1500 --log-every 200 --resume"
    )
    pred_cmd = (
        f"{sys.executable} -u {ROOT / 'predict_prefix.py'} "
        f"--model {fold_dir / 'best_model'} "
        f"--calls {ROOT / 'inputs/train.jsonl'} "
        f"--ids {out_root / f'fold{args.fold}_held.txt'} "
        f"--out {oof_npz} "
        f"--fmt space --max-length 512 --truncation-side right"
    )

    marker = f"--out {fold_dir} "
    ps_out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    alive = [
        line.strip().split(None, 1)[0]
        for line in ps_out.splitlines()
        if marker in line and "pgrep" not in line and "remote_run_oof_fold" not in line
    ]
    proc = None
    if not alive and not (fold_dir / "WORKER_OK").exists():
        shell_script = (
            f"set -e\n"
            f"if [ ! -f {fold_dir / 'final.pt'} ] || [ ! -d {fold_dir / 'best_model'} ]; then\n"
            f"  {train_cmd}\n"
            f"fi\n"
            f"if [ ! -f {oof_npz} ]; then\n"
            f"  {pred_cmd}\n"
            f"fi\n"
            f"echo WORKER_SUCCESS > {fold_dir / 'WORKER_OK'}\n"
        )
        proc = subprocess.Popen(
            ["bash", "-lc", shell_script],
            stdout=worker_log.open("a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"[fold {args.fold}] launched worker pid={proc.pid}", flush=True)
    else:
        print(f"[fold {args.fold}] attached to existing worker pids={alive}", flush=True)

    # Stream periodic heartbeats while keeping the Jupyter kernel busy & websocket active
    while True:
        if (fold_dir / "WORKER_OK").exists() and oof_npz.exists():
            meta = {"fold": args.fold, "npz_sha256": file_sha256(oof_npz)}
            done_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[fold {args.fold}] FOLD_DONE {json.dumps(meta)}", flush=True)
            return 0
        if proc is not None and proc.poll() is not None and not (fold_dir / "WORKER_OK").exists():
            tail = worker_log.read_text(encoding="utf-8")[-2000:] if worker_log.exists() else ""
            print(f"[fold {args.fold}] WORKER_FAILED rc={proc.returncode}\n{tail}", flush=True)
            return 1
        ps_now = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
        still = [line for line in ps_now.splitlines() if str(fold_dir) in line and "remote_run_oof_fold" not in line]
        if proc is None and not still and not (fold_dir / "WORKER_OK").exists():
            tail = worker_log.read_text(encoding="utf-8")[-2000:] if worker_log.exists() else ""
            print(f"[fold {args.fold}] ATTACHED_WORKER_EXITED\n{tail}", flush=True)
            return 1
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True
        ).stdout.strip()
        tlog = fold_dir / "train_log.txt"
        last = tlog.read_text(encoding="utf-8").splitlines()[-1] if tlog.exists() and tlog.read_text(encoding="utf-8").splitlines() else "init"
        print(f"[fold {args.fold} heartbeat] gpu=({gpu}) | {last}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
