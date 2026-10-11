import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# DCC Mission 2 ZCR Confirmation Runner
# Clean execution of baseline vs ZCR on fresh 1,000 calls across Seed 42 and Seed 43.

PYTHON_EXE = r"C:\Users\user\Desktop\DCC\venv\Scripts\python.exe"
QUICK_SCRIPT = r"C:\Users\user\Desktop\DCC\mission2_speaker\experiments\quick_hypotheses.py"
BASE_DIR = Path(r"D:\DCC_m2_zcr_confirmation\20261011")
CACHE_DIR = BASE_DIR / "cache_1000_resnet128"

def run_command(cmd, log_path):
    print(f"Executing: {' '.join(cmd)}")
    with open(log_path, "w", encoding="utf-8") as f_log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f_log.write(line)
            f_log.flush()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with return code {proc.returncode}: {' '.join(cmd)}")

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    out_s42 = BASE_DIR / "probe_s42"
    out_s43 = BASE_DIR / "probe_s43"

    # Step 1: Probe Seed 42
    print("=" * 70)
    print("Starting ZCR Confirmation: Seed 42 (baseline vs zcr, 5 epochs)")
    print("=" * 70)
    t0_s42 = time.perf_counter()
    cmd_s42 = [
        PYTHON_EXE, QUICK_SCRIPT, "probe",
        "--cache-dir", str(CACHE_DIR),
        "--output-dir", str(out_s42),
        "--model", "resnet50",
        "--conditions", "baseline", "zcr",
        "--epochs", "5",
        "--batch-size", "16",
        "--lr", "0.0001",
        "--seed", "42",
        "--device", "cuda",
        "--head-dropout", "0.3",
        "--bootstrap", "2000"
    ]
    run_command(cmd_s42, BASE_DIR / "probe_s42.log")
    dur_s42 = time.perf_counter() - t0_s42
    print(f"Seed 42 completed in {dur_s42:.1f}s.")

    # Step 2: Probe Seed 43
    print("=" * 70)
    print("Starting ZCR Confirmation: Seed 43 (baseline vs zcr, 5 epochs)")
    print("=" * 70)
    t0_s43 = time.perf_counter()
    cmd_s43 = [
        PYTHON_EXE, QUICK_SCRIPT, "probe",
        "--cache-dir", str(CACHE_DIR),
        "--output-dir", str(out_s43),
        "--model", "resnet50",
        "--conditions", "baseline", "zcr",
        "--epochs", "5",
        "--batch-size", "16",
        "--lr", "0.0001",
        "--seed", "43",
        "--device", "cuda",
        "--head-dropout", "0.3",
        "--bootstrap", "2000"
    ]
    run_command(cmd_s43, BASE_DIR / "probe_s43.log")
    dur_s43 = time.perf_counter() - t0_s43
    print(f"Seed 43 completed in {dur_s43:.1f}s.")

    # Step 3: Parse and combine results
    print("=" * 70)
    print("Synthesizing ZCR Confirmation Results...")
    print("=" * 70)

    diff_s42 = pd.read_csv(out_s42 / "pilot_differences_seed42.csv")
    comp_s42 = pd.read_csv(out_s42 / "pilot_comparison_seed42.csv")

    diff_s43 = pd.read_csv(out_s43 / "pilot_differences_seed43.csv")
    comp_s43 = pd.read_csv(out_s43 / "pilot_comparison_seed43.csv")

    # Combine comparison table
    combined_comp = pd.concat([comp_s42, comp_s43], ignore_index=True)
    combined_comp.to_csv(BASE_DIR / "ZCR_PILOT_COMPARISON_ALL.csv", index=False)

    combined_diff = pd.concat([diff_s42, diff_s43], ignore_index=True)
    combined_diff.to_csv(BASE_DIR / "ZCR_PILOT_DIFFERENCES_ALL.csv", index=False)

    # Calculate gate metrics
    # Pre-fixed criterion is Epoch 5 (last_epoch_accuracy)
    s42_zcr_row = diff_s42[diff_s42["condition"] == "zcr"].iloc[0]
    s43_zcr_row = diff_s43[diff_s43["condition"] == "zcr"].iloc[0]

    s42_delta_last = float(s42_zcr_row["delta_accuracy_last_pp"])
    s42_ci_low = float(s42_zcr_row["paired_ci_last_low_pp"])
    s42_ci_high = float(s42_zcr_row["paired_ci_last_high_pp"])
    s42_r0_drop = float(s42_zcr_row["delta_recall_0_last_pp"])

    s43_delta_last = float(s43_zcr_row["delta_accuracy_last_pp"])
    s43_ci_low = float(s43_zcr_row["paired_ci_last_low_pp"])
    s43_ci_high = float(s43_zcr_row["paired_ci_last_high_pp"])
    s43_r0_drop = float(s43_zcr_row["delta_recall_0_last_pp"])

    mean_delta_last = (s42_delta_last + s43_delta_last) / 2.0

    # Decision criteria checks
    c1_both_positive = (s42_delta_last > 0.0) and (s43_delta_last > 0.0)
    c2_mean_ge_half = mean_delta_last >= 0.50
    c3_r0_ok = (s42_r0_drop >= -1.0) and (s43_r0_drop >= -1.0)
    c4_ci_low_positive = (s42_ci_low > 0.0) and (s43_ci_low > 0.0)

    passed_all = c1_both_positive and c2_mean_ge_half and c3_r0_ok and c4_ci_low_positive

    summary_text = f"""# DCC Mission 2 ZCR Confirmation Report (1,000 Calls Fresh Cache)

- Date: 2026-10-11
- Sample: Fresh 1,000 calls (800 Train, 200 Dev), 30,580 utterances total (Train 24,144 / Dev 6,436)
- Model: AudioResNet (ResNet-50 128-Mel input, full fine-tuning, head dropout 0.3)
- Cache Policy: Shared deduplicated baseline.npy (1.47 GB) + features.npz (ZCR mean/std)
- Primary Evaluation: Pre-fixed Epoch 5 (last predictions)
- Bootstrap: 2,000 paired call-clustered resamples

## Quantitative Results (Epoch 5 Fixed Evaluation)

| Metric | Seed 42 | Seed 43 | Mean | Required Gate | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Delta Accuracy (%p) | {s42_delta_last:+.4f}%p | {s43_delta_last:+.4f}%p | {mean_delta_last:+.4f}%p | Both > 0, Mean >= +0.50%p | {'PASS' if (c1_both_positive and c2_mean_ge_half) else 'FAIL'} |
| 95% Paired CI (%p) | [{s42_ci_low:+.4f}, {s42_ci_high:+.4f}] | [{s43_ci_low:+.4f}, {s43_ci_high:+.4f}] | - | CI Lower Bound > 0 | {'PASS' if c4_ci_low_positive else 'FAIL'} |
| Delta Class 0 Recall (%p) | {s42_r0_drop:+.4f}%p | {s43_r0_drop:+.4f}%p | {(s42_r0_drop + s43_r0_drop)/2:+.4f}%p | Drop < 1.0%p (>= -1.0%p) | {'PASS' if c3_r0_ok else 'FAIL'} |

## Gate Criteria Evaluation

1. Both Seeds Positive Delta: {c1_both_positive} (Seed 42: {s42_delta_last:+.4f}%p, Seed 43: {s43_delta_last:+.4f}%p)
2. Mean Delta >= +0.5%p: {c2_mean_ge_half} ({mean_delta_last:+.4f}%p)
3. Class 0 Recall Drop < 1.0%p: {c3_r0_ok} (Seed 42: {s42_r0_drop:+.4f}%p, Seed 43: {s43_r0_drop:+.4f}%p)
4. Both 95% CI Lower Bounds > 0: {c4_ci_low_positive} (Seed 42 CI low: {s42_ci_low:+.4f}%p, Seed 43 CI low: {s43_ci_low:+.4f}%p)

## Final Gate Decision

**Decision**: {'PASS - ADOPT ZCR FOR AUDIORESNET FULL TRAINING' if passed_all else 'REJECT - DO NOT ADOPT ZCR, RETAIN BASELINE (92.4633%)'}
"""
    (BASE_DIR / "ZCR_CONFIRMATION_REPORT.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)

    decision_data = {
        "candidate": "H4_ZCR",
        "passed": bool(passed_all),
        "seed42_delta_accuracy_pp": s42_delta_last,
        "seed42_ci_95": [s42_ci_low, s42_ci_high],
        "seed42_delta_recall_0_pp": s42_r0_drop,
        "seed43_delta_accuracy_pp": s43_delta_last,
        "seed43_ci_95": [s43_ci_low, s43_ci_high],
        "seed43_delta_recall_0_pp": s43_r0_drop,
        "mean_delta_accuracy_pp": mean_delta_last,
        "action": "PROCEED_FULL_TRAINING" if passed_all else "RETAIN_OFFICIAL_BASELINE"
    }
    with open(BASE_DIR / "decision_summary.json", "w", encoding="utf-8") as f:
        json.dump(decision_data, f, indent=2)

if __name__ == "__main__":
    main()
