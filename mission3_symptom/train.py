"""Mission 3 KLUE-RoBERTa 다중 라벨 분류 학습 진입점.

run_dev_selection.sh 가 dev 선택 학습과 최종 학습에 모두 이 스크립트를 쓴다. 예:

    python train.py --train-csv data_csv/mission3_train.csv --val-csv runs/devsel/data/dev_split.csv \\
        --output-dir runs/devsel/final_s42 --model-name-or-path runs/devsel/tapt --local-files-only \\
        --learning-rate 5e-5 --llrd-decay 0.8 --use-pos-weight --pos-weight-power 0.5 --epochs 3 \\
        --checkpoint-metric fixed_epoch --checkpoint-epoch 2 --max-length 512 --amp --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

from m3.config import DEFAULT_UTTERANCE_SEP_MODE, UTTERANCE_SEP_MODES
from m3.training import BASELINE_MODEL_NAME, CHECKPOINT_METRICS, TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    """학습 CLI 인자. 기본값은 원래 레시피(lr 2e-5, 3 epoch, 배치 8 x 누적 2, 512 토큰)이다."""
    parser = argparse.ArgumentParser(description="Mission 3 KLUE-RoBERTa 다중 라벨 학습")
    # 데이터·출력 경로
    parser.add_argument("--train-csv", required=True, help="학습 CSV 경로")
    parser.add_argument(
        "--val-csv", required=True,
        help="평가 CSV 경로. run_dev_selection.sh 는 Training 내부 dev 분할을 넘긴다 (epoch 별 기록·확률 저장용, 제공 Validation 으로 선택하지 않는다)",
    )
    parser.add_argument("--output-dir", required=True, help="실험 산출물 저장 경로")
    # 시작 모델 (공개 모델 이름 또는 TAPT 결과 폴더)
    parser.add_argument("--model-name-or-path", default=BASELINE_MODEL_NAME)
    parser.add_argument("--model-revision")
    # 일반 학습 설정
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
    # 불균형 보정 손실: BCE 양성 가중 pos_weight = (Training 음성/양성) ** power
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument(
        "--pos-weight-power",
        type=float,
        default=1.0,
        help="pos_weight = (negative/positive) ** power. 1.0 은 원래 공식. 제출 모델의 0.5 는 run_dev_selection.sh 3 단계(Training 내부 dev)가 정했다",
    )
    parser.add_argument("--local-files-only", action="store_true")
    # 저장할 체크포인트 선정 기준 (val_* 는 --val-csv 위의 값)
    parser.add_argument(
        "--checkpoint-metric",
        choices=CHECKPOINT_METRICS,
        default="fixed_epoch",
        help="저장 기준: fixed_epoch(기본, 평가 점수와 무관하게 --checkpoint-epoch 저장), val_loss, val_macro_f1",
    )
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        help="--checkpoint-metric fixed_epoch 일 때 저장할 epoch (생략하면 마지막 epoch). 학습률 스케줄은 --epochs 기준",
    )
    parser.add_argument(
        "--utterance-sep-mode",
        choices=tuple(UTTERANCE_SEP_MODES),
        default=DEFAULT_UTTERANCE_SEP_MODE,
        help="학습 CSV 의 발화 경계 표현. run_config.json 에 기록되어 추론이 같은 모드를 복원한다",
    )
    parser.add_argument(
        "--llrd-decay",
        type=float,
        default=1.0,
        help="layer-wise LR decay. 헤드는 learning-rate, 인코더 층마다 이 값을 곱한다 (1.0 은 끔)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    """CLI 인자를 TrainingConfig 로 그대로 옮긴다 (값 검증은 run_training 이 한다)."""
    return TrainingConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=str(Path(args.output_dir)),
        model_name_or_path=args.model_name_or_path,
        model_revision=args.model_revision,
        seed=args.seed,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp,
        use_pos_weight=args.use_pos_weight,
        pos_weight_power=args.pos_weight_power,
        local_files_only=args.local_files_only,
        checkpoint_metric=args.checkpoint_metric,
        checkpoint_epoch=args.checkpoint_epoch,
        utterance_sep_mode=args.utterance_sep_mode,
        llrd_decay=args.llrd_decay,
    )


def main() -> None:
    """학습 한 번을 실행하고 저장한 epoch 와 그 시점의 평가(dev) macro F1@0.5 를 출력한다."""
    args = parse_args()
    metrics = run_training(build_config(args))
    print(
        f"학습 완료: 저장 epoch={metrics['best_epoch']}, "
        f"평가 macro_f1@0.5={metrics['val_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
