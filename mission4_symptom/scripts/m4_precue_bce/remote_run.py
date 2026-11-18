"""Remote L4 runner: CUDA stop/resume verification, full Stage 1 seed-42 training, Dev prediction, and chunked export."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/content/m4_precue")
RUN_OUT = Path("/content/runs/m4_precue_bce_v1_s42")
SMOKE = Path("/content/smoke_precue")


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
    SMOKE.mkdir(parents=True, exist_ok=True)
    fit_lines = [l for l in (ROOT / "inputs/fit_ids.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    smoke_ids = SMOKE / "smoke_ids.txt"
    smoke_ids.write_text("\n".join(fit_lines[:32]) + "\n", encoding="utf-8")

    # 1. CUDA AMP 2-step stop/resume bit-identity check
    common_smoke = [
        sys.executable, "-u", str(ROOT / "train_precue.py"),
        "--calls", str(ROOT / "inputs/train.jsonl"),
        "--ids", str(smoke_ids),
        "--forbid-ids", str(ROOT / "inputs/dev_ids.txt"),
        "--backbone", str(ROOT / "tapt"),
        "--precue-pos-weight", "1.0",
        "--epochs", "1", "--samples-per-call", "2", "--batch-size", "16",
        "--seed", "42", "--log-every", "2",
    ]
    if not (SMOKE / "resumed/final.pt").exists():
        run_cmd(common_smoke + ["--out", str(SMOKE / "full")])
        run_cmd(common_smoke + ["--out", str(SMOKE / "resumed"), "--stop-after-steps", "2"])
        run_cmd(common_smoke + ["--out", str(SMOKE / "resumed"), "--resume"])
    import torch  # noqa: E402
    cfg_full = json.loads((SMOKE / "full/run_config.json").read_text())
    cfg_res = json.loads((SMOKE / "resumed/run_config.json").read_text())
    st_f = torch.load(SMOKE / "full/final.pt", map_location="cpu", weights_only=True)["model"]
    st_r = torch.load(SMOKE / "resumed/final.pt", map_location="cpu", weights_only=True)["model"]
    max_diff = max(float((st_f[k] - st_r[k]).abs().max()) for k in st_f)
    mean_diff = sum(float((st_f[k] - st_r[k]).abs().mean()) for k in st_f) / len(st_f)
    loss_diff = abs(cfg_full["history"][0]["train_loss"] - cfg_res["history"][0]["train_loss"])
    if max_diff > 1e-4 or loss_diff > 1e-4:
        raise RuntimeError(f"CUDA stop/resume mismatch: max_diff={max_diff} mean_diff={mean_diff} loss_diff={loss_diff}")
    print(f"CUDA_STOP_RESUME_VERIFIED max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} loss_diff={loss_diff:.2e}", flush=True)

    # 2. Full Stage 1 seed-42 training
    RUN_OUT.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            sys.executable, "-u", str(ROOT / "train_precue.py"),
            "--calls", str(ROOT / "inputs/train.jsonl"),
            "--ids", str(ROOT / "inputs/fit_ids.txt"),
            "--forbid-ids", str(ROOT / "inputs/dev_ids.txt"),
            "--backbone", str(ROOT / "tapt"),
            "--out", str(RUN_OUT),
            "--precue-pos-weight", "1.0",
            "--seed", "42", "--epochs", "3", "--samples-per-call", "2", "--batch-size", "16",
            "--lr", "5e-5", "--llrd", "0.8", "--pos-weight-power", "0.5", "--max-length", "512",
            "--fmt", "space", "--truncation-side", "right",
            "--save-every", "2500", "--log-every", "500", "--resume",
        ],
        log_path=RUN_OUT / "runner_stdout.log",
    )

    # 3. Dev 2,920 prefix prediction
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

    # 4. Quick remote score check + pack light & heavy archives
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
        "zip", "-q", "-j", "/content/m4_precue_s42_light.zip",
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
        "zip", "-q", "-0", "-r", "/content/m4_precue_s42_weights.zip",
        str(RUN_OUT / "final.pt"),
        str(RUN_OUT / "training_state_last.pt"),
        str(RUN_OUT / "best_model"),
    ])
    run_cmd([
        sys.executable, "/content/chunk_checkpoint.py", "split",
        "/content/m4_precue_s42_weights.zip", "/content/weight_chunks",
    ])
    (RUN_OUT / "DONE.json").write_text(json.dumps({"done": True, **quick}, indent=2), encoding="utf-8")
    print("ALL_REMOTE_STAGES_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
