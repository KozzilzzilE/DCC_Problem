"""Parameterized remote L4 runner for m4_precue experiments (training, dev inference, and chunked export)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--precue-pos-weight", type=float, default=-1.0)
    ap.add_argument("--jaccard-weight", type=float, default=0.0)
    ap.add_argument("--macro-f1-weight", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    run_out = Path("/content/runs") / args.run_name
    run_out.mkdir(parents=True, exist_ok=True)
    log_path = run_out / "runner_stdout.log"

    run_cmd(
        [
            sys.executable, "-u", str(ROOT / "train_precue.py"),
            "--calls", str(ROOT / "inputs/train.jsonl"),
            "--ids", str(ROOT / "inputs/fit_ids.txt"),
            "--forbid-ids", str(ROOT / "inputs/dev_ids.txt"),
            "--backbone", str(ROOT / "tapt"),
            "--out", str(run_out),
            "--precue-pos-weight", str(args.precue_pos_weight),
            "--jaccard-weight", str(args.jaccard_weight),
            "--macro-f1-weight", str(args.macro_f1_weight),
            "--seed", str(args.seed), "--epochs", "3", "--samples-per-call", "2", "--batch-size", "16",
            "--lr", "5e-5", "--llrd", "0.8", "--pos-weight-power", "0.5", "--max-length", "512",
            "--fmt", "space", "--truncation-side", "right",
            "--save-every", "2500", "--log-every", str(args.log_every), "--resume",
        ],
        log_path=log_path,
    )

    run_cmd(
        [
            sys.executable, "-u", str(ROOT / "predict_prefix.py"),
            "--model", str(run_out / "best_model"),
            "--calls", str(ROOT / "inputs/train.jsonl"),
            "--ids", str(ROOT / "inputs/dev_ids.txt"),
            "--out", str(run_out / "dev_prefix.npz"),
            "--fmt", "space", "--max-length", "512", "--truncation-side", "right",
        ],
        log_path=log_path,
    )

    sys.path.insert(0, str(ROOT))
    from m4 import decision as dec, simulate as sim  # noqa: E402
    import numpy as np  # noqa: E402

    tbl = sim.PrefixTable.load(run_out / "dev_prefix.npz")
    pred = dec.threshold_rule(tbl.probs.copy(), force_top1=False)
    gold = np.repeat(tbl.labels.astype(bool), tbl.n_utt, axis=0)
    ms = sim.moment_scores(pred, gold)
    per_call = np.add.reduceat(ms, tbl.offsets[:-1]) / tbl.n_utt
    quick = {"m4_1_raw05": float(per_call.mean()), "empty_rate": float((~pred.any(axis=1)).mean()), "elapsed_sec": round(time.time() - t0, 1)}
    (run_out / "remote_quick_summary.json").write_text(json.dumps(quick, indent=2), encoding="utf-8")
    print("REMOTE_QUICK_SUMMARY:", json.dumps(quick), flush=True)

    light_zip = Path(f"/content/{args.run_name}_light.zip")
    weights_zip = Path(f"/content/{args.run_name}_weights.zip")
    chunks_dir = Path(f"/content/{args.run_name}_weight_chunks")

    run_cmd([
        "zip", "-q", "-j", str(light_zip),
        str(run_out / "dev_prefix.npz"),
        str(run_out / "dev_prefix.meta.json"),
        str(run_out / "run_config.json"),
        str(run_out / "optimizer_audit.json"),
        str(run_out / "initialization.json"),
        str(run_out / "train_log.txt"),
        str(run_out / "remote_quick_summary.json"),
        str(run_out / "runner_stdout.log"),
    ])
    run_cmd([
        "zip", "-q", "-0", "-r", str(weights_zip),
        str(run_out / "final.pt"),
        str(run_out / "training_state_last.pt"),
        str(run_out / "best_model"),
    ])
    run_cmd([
        sys.executable, "/content/chunk_checkpoint.py", "split",
        str(weights_zip), str(chunks_dir),
    ])
    (run_out / "DONE.json").write_text(json.dumps({"done": True, **quick}, indent=2), encoding="utf-8")
    print(f"ALL_STAGES_DONE:{args.run_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
