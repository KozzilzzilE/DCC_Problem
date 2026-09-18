import os
import argparse
import pandas as pd
from torch.utils.data import DataLoader

from .config import MODEL_REGISTRY, COLAB_DATA_ROOT, LOCAL_DATA_ROOT, COLAB_DRIVE_BACKUP
from .dataset import UniversalSpeakerDataset
from .models import build_model
from .trainer import BenchmarkTrainer
from .ensemble import EnsembleEvaluator

def run_benchmark(models_to_run=None, epochs=15, max_train_files=None, max_val_files=None, 
                  data_root=None, output_dir="./results", skip_completed=True):
    """
    미션 2 다중 모델 벤치마크 일괄 실행 파이프라인
    """
    if data_root is None:
        data_root = COLAB_DATA_ROOT if os.path.exists(COLAB_DATA_ROOT) else LOCAL_DATA_ROOT

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    # 혹시 train/val 대신 상위 폴더 구조인 경우 자동 보정
    if not os.path.exists(train_dir):
        # AI-Hub 구조 확인 (Training / Validation)
        alt_train = os.path.join(data_root, "Training")
        alt_val = os.path.join(data_root, "Validation")
        if os.path.exists(alt_train):
            train_dir = alt_train
            val_dir = alt_val
        else:
            # data_root 내부 재귀 탐색
            train_dir = data_root
            val_dir = data_root

    trainer = BenchmarkTrainer(output_dir=output_dir, drive_backup_dir=COLAB_DRIVE_BACKUP)
    available_models = list(MODEL_REGISTRY.keys())
    target_models = models_to_run if models_to_run else available_models

    print(f"📊 [벤치마크 시작] 총 {len(target_models)}개 모델 테스트 예정: {target_models}")
    print(f"📁 데이터 경로: {data_root}")
    print(f"💾 결과 저장 경로: {output_dir}\n")

    for model_name in target_models:
        if model_name not in MODEL_REGISTRY:
            print(f"⚠️ {model_name}은(는) 등록되지 않은 모델입니다. 건너뜁니다.")
            continue

        cfg = MODEL_REGISTRY[model_name]

        # 1. 모델별 최적 데이터셋 로더 구축
        train_ds = UniversalSpeakerDataset(
            train_dir,
            model_name=model_name,
            config_dict=cfg,
            is_train=True,
            max_files=max_train_files
        )
        val_ds = UniversalSpeakerDataset(
            val_dir,
            model_name=model_name,
            config_dict=cfg,
            is_train=False,
            max_files=max_val_files
        )

        batch_size = cfg.get("batch_size", 32)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # 2. 모델 인스턴스 생성
        model = build_model(model_name, num_classes=2, pretrained=True)

        # 3. 학습 및 평가 루프 실행 (자동 백업 및 이어하기)
        trainer.fit(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=cfg.get("learning_rate", 1e-4),
            weight_decay=cfg.get("weight_decay", 1e-4),
            skip_if_done=skip_completed
        )

    # 최종 결과표 출력
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    if os.path.exists(csv_path):
        df_res = pd.read_csv(csv_path)
        print("\n=======================================================")
        print("🏆 [다중 모델 벤치마크 종합 결과 비교표]")
        print("=======================================================")
        print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCC Mission 2 Multi-Model Benchmark")
    parser.add_argument("--models", nargs="+", default=None, help="실행할 모델 목록 (예: redimnet ecapa_tdnn resnet50)")
    parser.add_argument("--epochs", type=int, default=15, help="에포크 수")
    parser.add_argument("--max_train", type=int, default=None, help="Train 파일 샘플링 수")
    parser.add_argument("--max_val", type=int, default=None, help="Val 파일 샘플링 수")
    parser.add_argument("--data_root", type=str, default=None, help="데이터 루트 경로")
    parser.add_argument("--output_dir", type=str, default="./results", help="결과 저장 폴더")
    parser.add_argument("--no_skip", action="store_true", help="기존 체크포인트 무시하고 재학습")

    args = parser.parse_args()
    run_benchmark(
        models_to_run=args.models,
        epochs=args.epochs,
        max_train_files=args.max_train,
        max_val_files=args.max_val,
        data_root=args.data_root,
        output_dir=args.output_dir,
        skip_completed=not args.no_skip
    )
