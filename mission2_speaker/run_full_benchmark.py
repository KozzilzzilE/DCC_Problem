import os
import sys

# 현재 디렉토리를 Python 모듈 검색 경로에 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from benchmark_suite.benchmark import run_benchmark

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 [DCC Mission 2] 전체 데이터 멀티 모델 벤치마크 통합 학습 시작")
    print("=" * 65)
    print("대상 모델: ResNet-50, ECAPA-TDNN, ReDimNet, CAM++, SSAST (총 5종)")
    print("학습 모드: 전체 데이터 100% (train 87만+, val 11만+)")
    print("안전 장치: AMP 혼합 정밀도 + 가중치 실시간 저장 + 완료 모델 Auto-Skip")
    print("=" * 65 + "\n")

    # 5개 모델 순차 실행 (RTX 3060 최적화)
    run_benchmark(
        models_to_run=["resnet50", "ecapa_tdnn", "redimnet", "campp", "ssast"],
        epochs=10,                  # 모델당 최적 에포크 수 (자동 최고 점수 가중치 저장)
        max_train_files=None,       # ⭐️ 전체 Train 데이터 100% 사용
        max_val_files=None,         # ⭐️ 전체 Val 데이터 100% 사용
        data_root=r"C:\Users\user\Desktop\DCC\data",
        output_dir="./results",
        skip_completed=True         # 이미 완료된 모델은 건너뛰고 다음 모델 학습
    )
