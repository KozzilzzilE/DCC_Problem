"""Mission 3 KLUE-RoBERTa baseline 학습 진입점."""

from __future__ import annotations

import argparse
from pathlib import Path

from m3.training import BASELINE_MODEL_NAME, TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 3 KLUE-RoBERTa 다중 라벨 학습")
    parser.add_argument("--train-csv", required=True, help="학습 CSV 경로")
    parser.add_argument("--val-csv", required=True, help="Validation CSV 경로")
    parser.add_argument("--output-dir", required=True, help="실험 산출물 저장 경로")
    parser.add_argument("--model-name-or-path", default=BASELINE_MODEL_NAME)
    parser.add_argument("--model-revision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--loss-type", choices=("bce", "asl"), default="bce")
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=1.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--asl-eps", type=float, default=1e-8)
    parser.add_argument("--asl-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--checkpoint-metric",
        choices=("val_loss", "val_macro_f1"),
        default="val_loss",
        help="최적 모델(Best Checkpoint) 저장 기준 지표 (기본값: val_loss, 대회 지표 기준: val_macro_f1)",
    )
    parser.add_argument(
        "--encode-mode",
        choices=("truncate", "head_tail"),
        default="truncate",
        help="512 초과 통화 입력: truncate(앞만) 또는 head_tail(앞 128+꼬리)",
    )
    parser.add_argument(
        "--use-pure-nausea-sampling",
        action="store_true",
        help="Training의 pure-nausea(오심=1, 구토=0) row만 가중 재샘플링",
    )
    parser.add_argument(
        "--pure-nausea-weight",
        type=float,
        default=1.5,
        help="pure-nausea(C) Training row의 sampling weight (기본값: 1.5)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    output_dir = Path(args.output_dir)
    max_length = args.max_length
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    epochs = args.epochs
    max_train_samples = args.max_train_samples
    max_val_samples = args.max_val_samples
    max_steps = args.max_steps

    if args.smoke_test:
        output_dir = output_dir.parent / f"{output_dir.name}_smoke"
        max_length = min(max_length, 128)
        train_batch_size = min(train_batch_size, 2)
        val_batch_size = min(val_batch_size, 2)
        epochs = 1
        max_train_samples = min(max_train_samples or 64, 64)
        max_val_samples = min(max_val_samples or 32, 32)
        max_steps = min(max_steps or 2, 2)

    return TrainingConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=str(output_dir),
        model_name_or_path=args.model_name_or_path,
        model_revision=args.model_revision,
        seed=args.seed,
        max_length=max_length,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp,
        use_pos_weight=args.use_pos_weight,
        loss_type=args.loss_type,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_clip=args.asl_clip,
        asl_eps=args.asl_eps,
        asl_reduction=args.asl_reduction,
        local_files_only=args.local_files_only,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_steps=max_steps,
        smoke_test=args.smoke_test,
        checkpoint_metric=args.checkpoint_metric,
        encode_mode=args.encode_mode,
        use_pure_nausea_sampling=args.use_pure_nausea_sampling,
        pure_nausea_weight=args.pure_nausea_weight,
    )


def main() -> None:
    args = parse_args()
    metrics = run_training(build_config(args))
    print(
        f"학습 완료: best_epoch={metrics['best_epoch']}, "
        f"val_macro_f1@0.5={metrics['val_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
