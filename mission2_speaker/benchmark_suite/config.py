import os

# ==============================================================================
# 1. 공통 기본 설정 (전역 파라미터)
# ==============================================================================
SEED = 42
SAMPLE_RATE = 16000
TARGET_DURATION = 3.0       # 기본 목표 음성 길이 (초)
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION) # 48,000 samples
NUM_CLASSES = 2             # 0: 119대원, 1: 신고자

# 구글 드라이브 및 로컬 경로 설정 (Colab 기준 자동 탐색)
COLAB_DATA_ROOT = "/content/data"
COLAB_DRIVE_BACKUP = "/content/drive/MyDrive/DCC/benchmark_results"
LOCAL_DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

# ==============================================================================
# 2. 모델별 최적 전처리 및 하이퍼파라미터 레지스트리
#    (speaker_diarization_analysis.md 및 각 모델 공식 권장 사양 반영)
# ==============================================================================
MODEL_REGISTRY = {
    # 1. ResNet-50 (2D CNN 베이스라인)
    "resnet50": {
        "category": "2D CNN",
        "input_type": "mel_spec",
        "n_mels": 128,
        "n_fft": 2048,
        "hop_length": 512,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "desc": "2D Mel-Spectrogram + 1ch ResNet-50 (ImageNet 평균 가중치)"
    },

    # 2. ECAPA-TDNN (화자 인식 표준 1D-CNN)
    "ecapa_tdnn": {
        "category": "1D CNN / Speaker Verification",
        "input_type": "fbank",
        "n_mels": 80,
        "n_fft": 512,
        "hop_length": 160,      # 10ms frame shift
        "batch_size": 64,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "desc": "Squeeze-and-Excitation + 통계적 풀링 특화 화자 임베딩 모델"
    },

    # 3. ReDimNet2-B2 (분석 문서 1순위 추천: 초경량 3.6M 혼합 구조)
    "redimnet": {
        "category": "Hybrid (2D+1D Conv + MHA)",
        "input_type": "fbank",
        "n_mels": 80,
        "n_fft": 512,
        "hop_length": 160,
        "batch_size": 64,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "desc": "2D/1D Conv와 Multi-Head Attention 결합 최신 화자 임베딩 모델"
    },

    # 4. CAM++ (분석 문서 추천: 국소 Conv + Context Masking)
    "campp": {
        "category": "Local Conv + Masking",
        "input_type": "fbank",
        "n_mels": 80,
        "n_fft": 512,
        "hop_length": 160,
        "batch_size": 64,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "desc": "3D-Speaker 기반 문맥 마스킹 고성능 화자 모델"
    },

    # 5. HuBERT-Base (대규모 자기지도학습 트랜스포머)
    "hubert": {
        "category": "Self-Supervised Transformer",
        "input_type": "waveform",
        "batch_size": 16,        # 트랜스포머 VRAM 절약
        "learning_rate": 2e-5,  # 파인튜닝용 낮은 LR
        "weight_decay": 1e-4,
        "desc": "Raw 파형 직접 입력, 음향 토큰 마스킹 사전학습 SOTA"
    },

    # 6. SSAST-Tiny (경량 오디오 ViT 가설 검증용)
    "ssast": {
        "category": "Audio Vision Transformer",
        "input_type": "mel_spec",
        "n_mels": 128,
        "n_fft": 2048,
        "hop_length": 512,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "weight_decay": 1e-4,
        "desc": "Patch 기반 스펙트로그램 트랜스포머 (ViT 가설 검증 대조군)"
    }
}
