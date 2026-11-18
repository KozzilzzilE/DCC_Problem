"""Remote L4 runner for Stage 1 m4_softjaccard_v1_s42 (BCE + SoftJaccard, exact E002 setup otherwise)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/content/m4_precue")
RUN_OUT = Path("/content/runs/m4_softjaccard_v1_s42")


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
    t0 = time.time()
    RUN_OUT.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            sys.executable, "-u", str(ROOT / "train_precue.py"),
            "--calls", str(ROOT / "inputs/train.jsonl"),
            "--ids", str(ROOT / "inputs/fit_ids.txt"),
            "--forbid-ids", str(ROOT / "inputs/dev_ids.txt"),
            "--backbone", str(ROOT / "tapt"),
            "--out", str(RUN_OUT),
            "--precue-pos-weight", "-1.0",
            "--jaccard-weight", "1.0",
            "--seed", "42", "--epochs", "3", "--samples-per-call", "2", "--batch-size", "16",
            "--lr", "5e-5", "--llrd", "0.8", "--pos-weight-power", "0.5", "--max-length", "512",
            "--fmt", "space", "--truncation-side", "right",
            "--save-every", "2500", "--log-every", "500", "--resume",
        ],
        log_path=RUN_OUT / "runner_stdout.log",
    )

    run_cmd(
        [
            sys.executable, "-u", str(ROOT / "predict_prefix.py"),
            "--model", str(RUN_OUT / "best_model"),
            "--calls", str(ROOT / "inputs/train.jsonl"),
            "--ids", str(ROOT / "inputs/dev_ids.txt"),
            "--out", str(RUN_OUT / "dev_prefix.npz"),
            "--fmt", "space", "--max-length", "512", "--truncation-side", "right",
        ],
        log_path=RUN_OUT / "runner_stdout.log",
    )

    sys.path.insert(0, str(ROOT))
    from m4 import decision as dec, simulate as sim  # noqa: E402
    import numpy as np  # noqa: E402

    tbl = sim.PrefixTable.load(RUN_OUT / "dev_prefix.npz")
    pred = dec.threshold_rule(tbl.probs.copy(), force_top1=False)
    gold = np.repeat(tbl.labels.astype(bool), tbl.n_utt, axis=0)
    ms = sim.moment_scores(pred, gold)
    per_call = np.add.reduceat(ms, tbl.offsets[:-1]) / tbl.n_utt
    quick = {"m4_1_raw05": float(per_call.mean()), "empty_rate": float((~pred.any(axis=1)).mean()), "elapsed_sec": round(time.time() - t0, 1)}
    (RUN_OUT / "remote_quick_summary.json").write_text(json.dumps(quick, indent=2), encoding="utf-8")
    print("REMOTE_QUICK_SUMMARY:", json.dumps(quick), flush=True)

    run_cmd([
        "zip", "-q", "-j", "/content/m4_softjaccard_s42_light.zip",
        str(RUN_OUT / "dev_prefix.npz"),
        str(RUN_OUT / "dev_prefix.meta.json"),
        str(RUN_OUT / "run_config.json"),
        str(RUN_OUT / "optimizer_audit.json"),
        str(RUN_OUT / "initialization.json"),
        str(RUN_OUT / "train_log.txt"),
        str(RUN_OUT / "remote_quick_summary.json"),
        str(RUN_OUT / "runner_stdout.log"),
    ])
    run_cmd([
        "zip", "-q", "-0", "-r", "/content/m4_softjaccard_s42_weights.zip",
        str(RUN_OUT / "final.pt"),
        str(RUN_OUT / "training_state_last.pt"),
        str(RUN_OUT / "best_model"),
    ])
    run_cmd([
        sys.executable, "/content/chunk_checkpoint.py", "split",
        "/content/m4_softjaccard_s42_weights.zip", "/content/sj_weight_chunks",
    ])
    (RUN_OUT / "DONE.json").write_text(json.dumps({"done": True, **quick}, indent=2), encoding="utf-8")
    print("ALL_SJ_STAGES_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
