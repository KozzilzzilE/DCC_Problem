# -*- coding: utf-8 -*-
"""
DCC Mission 2: Path B (Fixed Additional Training) Runner.

Selection Rule: fixed_additional_last
- ReDimNet: Parent Epoch 9 -> +1 epoch fixed training -> Epoch 10
- ECAPA-TDNN: Parent Epoch 8 -> +2 epochs fixed training -> Epoch 10
- AudioResNet-50: Parent Epoch 10 -> 0 additional epochs (register with hash)
- Official Validation: ZERO evaluation during training, ZERO score-based best selection
- Smoke Isolation: Smoke strictly evaluates on training samples (evaluate_smoke_unified), never touching official validation data
- Preprocessing: Exact Librosa Slaney-scale Mel via m2.audio_features
- Decision Threshold: 0.50 fixed
- Ensemble: 3 models equal weighting (1/3 each)
- Output Isolation: runs/smoke/ vs runs/path_b_fixed_continue/ are strictly separated
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure UTF-8 stream
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from m2.path_b import (
    DCCAudioDatasetUnified,
    build_model,
    compute_sha256,
    evaluate_official_final_unified,
    evaluate_smoke_unified,
    export_submission_package,
    freeze_model_config,
    load_model_weights_adapted,
    train_path_b_model,
    validate_training_root,
)


def main():
    parser = argparse.ArgumentParser(description="DCC Mission 2 Path B Fixed Continue Runner")
    parser.add_argument(
        "--stage",
        default="define_only",
        choices=["define_only", "smoke", "full_train", "freeze_config", "eval_official", "export"],
        help="Execution stage (default: define_only)",
    )
    parser.add_argument(
        "--data_dir", default=str(BASE_DIR.parent / "data" / "train"), help="Training data root"
    )
    parser.add_argument(
        "--val_dir", default=str(BASE_DIR.parent / "data" / "val"), help="Validation data root (only for eval_official)"
    )
    parser.add_argument(
        "--output_root",
        default=str(BASE_DIR / "experiments" / "runs" / "path_b_fixed_continue"),
        help="Output run directory for full training",
    )
    parser.add_argument(
        "--smoke_output_root",
        default=str(BASE_DIR / "experiments" / "runs" / "smoke"),
        help="Output run directory for smoke testing",
    )
    parser.add_argument(
        "--export_dir",
        default=str(BASE_DIR / "checkpoints_path_b"),
        help="Submission export directory",
    )
    parser.add_argument(
        "--smoke_max_files", type=int, default=10, help="Max files for smoke run"
    )
    parser.add_argument(
        "--smoke_max_samples", type=int, default=200, help="Max samples for smoke run"
    )
    parser.add_argument(
        "--allow_smoke_freeze",
        action="store_true",
        help="Allow freezing config from smoke output (testing only)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_output_dir = Path(args.output_root)
    smoke_output_dir = Path(args.smoke_output_root)
    if full_output_dir.resolve() == smoke_output_dir.resolve():
        raise ValueError("Smoke and full-run directories must be different")
    if args.stage in {"smoke", "full_train"}:
        validate_training_root(args.data_dir, args.val_dir)

    parent_ckpts = {
        "redimnet": (BASE_DIR / "checkpoints" / "best_redimnet.pt", 9, 1),
        "ecapa_tdnn": (BASE_DIR / "checkpoints" / "best_ecapa_tdnn.pt", 8, 2),
        "resnet50": (BASE_DIR / "checkpoints" / "best_resnet50.pt", 10, 0),
    }

    if args.stage == "define_only":
        print("=" * 70)
        print(" [Path B Config Defined (define_only)]")
        print(f" - Device: {device}")
        print(f" - Training Data: {args.data_dir}")
        print(f" - Validation Data: {args.val_dir}")
        print(f" - Full Run Output Directory: {full_output_dir}")
        print(f" - Smoke Output Directory: {smoke_output_dir}")
        print(f" - Selection Rule: fixed_additional_last")
        print(" - Parent Checkpoints and Planned Additional Epochs:")
        for m, (p, ep, add_ep) in parent_ckpts.items():
            print(f"   * {m:10s}: Parent Ep {ep:2d} (+{add_ep} ep -> Ep {ep+add_ep:2d}) | Path: {p} (Exists: {p.exists()})")
        print(" [Notice] Default stage 'define_only' prevents automatic long training on Run All.")
        print(" To execute smoke test (training samples only): python run_path_b_fixed_continue.py --stage smoke")
        print(" To execute full training: python run_path_b_fixed_continue.py --stage full_train")
        print(" To execute official final eval: python run_path_b_fixed_continue.py --stage eval_official")
        print("=" * 70)
        return

    elif args.stage == "smoke":
        print("=" * 70)
        print(f" [Running Smoke Test] Max Files: {args.smoke_max_files}, Max Samples: {args.smoke_max_samples}")
        print(f" - Smoke Target Directory: {smoke_output_dir} (Separated from full run)")
        print(" - Validation Isolation: Official validation data is NEVER touched or evaluated.")
        print("=" * 70)
        smoke_output_dir.mkdir(parents=True, exist_ok=True)

        smoke_ds_80 = DCCAudioDatasetUnified(
            data_dir=args.data_dir,
            is_train=True,
            crop_mode="random",
            n_mels=80,
            n_fft=512,
            hop_length=160,
            max_files=args.smoke_max_files,
            max_samples=args.smoke_max_samples,
        )
        loader_80 = DataLoader(smoke_ds_80, batch_size=8, shuffle=True)
        print(f"Loaded smoke training samples: {len(smoke_ds_80)}")

        p_redim, ep_r, add_r = parent_ckpts["redimnet"]
        last_redim = train_path_b_model(
            "redimnet", p_redim, ep_r, add_r, smoke_output_dir, loader_80, device,
            is_smoke=True, run_stage="smoke"
        )

        p_ecapa, ep_e, add_e = parent_ckpts["ecapa_tdnn"]
        last_ecapa = train_path_b_model(
            "ecapa_tdnn", p_ecapa, ep_e, add_e, smoke_output_dir, loader_80, device,
            is_smoke=True, run_stage="smoke"
        )

        p_res, ep_res, add_res = parent_ckpts["resnet50"]
        last_res = train_path_b_model(
            "resnet50", p_res, ep_res, add_res, smoke_output_dir, loader_80, device,
            is_smoke=True, run_stage="smoke"
        )

        frozen_cfg = freeze_model_config(
            smoke_output_dir, last_redim, last_ecapa, last_res, allow_smoke=True
        )

        print("Running smoke verification on training samples (never touching official val)...")
        eval_metrics = evaluate_smoke_unified(
            frozen_config_path=frozen_cfg,
            train_data_root=Path(args.data_dir),
            device=device,
            batch_size=16,
            max_files=min(5, args.smoke_max_files),
            output_dir=smoke_output_dir / "eval_smoke",
        )
        print("Smoke verification completed successfully!")

    elif args.stage == "full_train":
        print("=" * 70)
        print(" [Running Path B Full Training]")
        print(f" - Target Output Directory: {full_output_dir}")
        print(" - Preprocessing: Librosa 80-Mel (ReDim, ECAPA) & 128-Mel (ResNet)")
        print(" - Official Validation: ZERO evaluation during training")
        print("=" * 70)
        full_output_dir.mkdir(parents=True, exist_ok=True)

        ds_80 = DCCAudioDatasetUnified(
            data_dir=args.data_dir,
            is_train=True,
            crop_mode="random",
            n_mels=80,
            n_fft=512,
            hop_length=160,
        )
        loader_80 = DataLoader(ds_80, batch_size=32, shuffle=True, num_workers=0)
        print(f"Loaded full training utterances: {len(ds_80):,} (audit: {ds_80.audit_info})")

        p_redim, ep_r, add_r = parent_ckpts["redimnet"]
        last_redim = train_path_b_model(
            "redimnet", p_redim, ep_r, add_r, full_output_dir, loader_80, device,
            is_smoke=False, run_stage="full_train"
        )

        p_ecapa, ep_e, add_e = parent_ckpts["ecapa_tdnn"]
        last_ecapa = train_path_b_model(
            "ecapa_tdnn", p_ecapa, ep_e, add_e, full_output_dir, loader_80, device,
            is_smoke=False, run_stage="full_train"
        )

        p_res, ep_res, add_res = parent_ckpts["resnet50"]
        last_res = train_path_b_model(
            "resnet50", p_res, ep_res, add_res, full_output_dir, loader_80, device,
            is_smoke=False, run_stage="full_train"
        )

        frozen_cfg = freeze_model_config(
            full_output_dir, last_redim, last_ecapa, last_res, allow_smoke=False
        )
        print(f"Full training and configuration freezing completed: {frozen_cfg}")

    elif args.stage == "freeze_config":
        redim_p = full_output_dir / "last_redimnet.pt"
        ecapa_p = full_output_dir / "last_ecapa_tdnn.pt"
        resnet_p = full_output_dir / "last_resnet50.pt"

        frozen_cfg = freeze_model_config(
            full_output_dir, redim_p, ecapa_p, resnet_p, allow_smoke=args.allow_smoke_freeze
        )
        print(f"Frozen config created: {frozen_cfg}")

    elif args.stage == "eval_official":
        frozen_cfg = full_output_dir / "final_model_config.json"
        if not frozen_cfg.exists():
            raise FileNotFoundError(
                f"Cannot run official evaluation without production frozen config: {frozen_cfg}. "
                "Smoke configurations are strictly prohibited for official evaluation."
            )

        eval_metrics = evaluate_official_final_unified(
            frozen_config_path=frozen_cfg,
            val_data_root=Path(args.val_dir),
            device=device,
            batch_size=32,
            output_dir=full_output_dir / "official_final",
        )

    elif args.stage == "export":
        frozen_cfg = full_output_dir / "final_model_config.json"
        if not frozen_cfg.exists():
            raise FileNotFoundError(f"Missing production frozen config for export: {frozen_cfg}")
        export_submission_package(frozen_cfg, Path(args.export_dir))


if __name__ == "__main__":
    main()
