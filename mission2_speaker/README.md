# 🎙️ Mission 2: 119 긴급 통화 음성 기반 화자 이진 분류 (Speaker Classification)

본 디렉토리는 **데이터+AI 크리에이터 캠프 본선 Mission 2 (상황실 접수요원 vs 신고자 음성 분류)**의 학습 파이프라인, 모델 아키텍처 및 추론 모듈을 포함합니다.

---

## 📌 1. 미션 개요 및 모델 아키텍처

- **목표**: 119 긴급 통화 음성에서 각 발화 세그먼트(`startAt` ~ `endAt`)를 분석하여 **상황실 접수요원(Class 0)**과 **신고자(Class 1)**를 실시간으로 정확하게 이진 분류.
- **핵심 아키텍처 (3대장 앙상블)**:
  1. **ECAPA-TDNN** (5.80M): 1D 시간 지연 합성곱 + Squeeze-and-Excitation + 통계적 풀링 기반 화자 고유 음색 시계열 모델
  2. **ReDimNet2-B2** (2.57M): 2D 합성곱 + Multi-Head Self-Attention 하이브리드 고효율 모델
  3. **AudioResNet-50** (23.50M): 2D 고해상도 Mel-Spectrogram 텍스처 특징 기반 앵커 모델
- **앙상블 및 분류 규칙**:
  - **대회 규정 준수**: 9/25 공통 FAQ에 따라 **결정 임계값 0.50 고정**
  - **사전 균등 가중치 (1/3씩)**: Validation 데이터 과적합(Leakage)을 원천 차단하기 위해 세 모델의 Sigmoid 확률을 단순 균등 평균(Simple Average)으로 결합
  - **검증 성능**: AI-Hub 정식 검증 데이터셋(111,919건 전수) 기준 **정확도 92.48%, RTF 0.0038(초당 350건 처리)** 달성

---

## 📂 2. 디렉토리 구조

```text
mission2_speaker/
├── m2/                         # Mission 2 전용 추론 패키지
│   ├── __init__.py
│   └── infer.py                # 전처리, 모델 로드, 앙상블 추론 엔진
├── benchmark_suite/            # 백본 모델 정의 (ReDimNet, ECAPA-TDNN 등)
│   ├── models/
│   │   ├── redimnet.py
│   │   └── ecapa_tdnn.py
│   └── ...
├── checkpoints/                # 3대 챔피언 모델 가중치 저장소
│   ├── best_redimnet.pt        # 9.8 MB
│   ├── best_ecapa_tdnn.pt      # 22.2 MB
│   └── best_resnet50.pt        # 89.9 MB
├── Local_Light_Train.ipynb     # RTX 3060 최적화 풀 훈련 및 앙상블 노트북
├── inference.py                # Mission 2 단독 실행용 CLI 스크립트
├── requirements.txt            # 필수 의존성 목록
└── README.md                   # 본 설명 문서
```

---

## 🚀 3. 추론 실행 방법 (How to Run)

### A. 미션 폴더 내 단독 실행
```bash
cd mission2_speaker
python inference.py --audio_dir ../data/val/audio                     --label_dir ../data/val/label                     --ckpt_path checkpoints/                     --output ../outputs/mission2.csv
```

### B. 프로젝트 루트에서 실행 (대회 표준 규격)
```bash
python inference.py --audio_dir data/val/audio                     --label_dir data/val/label                     --ckpt_path mission2_speaker/checkpoints/                     --output outputs/mission2.csv
```

* `--ckpt_path`에 `mission2_speaker/checkpoints/` 폴더를 지정하면 **3대장 앙상블(최고 성능)**로 자동 동작하며, 단일 `.pt` 파일 지정 시 해당 모델 단독으로 동작합니다.

---

## ⚙️ 4. 전처리 및 추론 기술 규격

1. **오디오 샘플링 레이트**: 16,000 Hz (Mono)
2. **윈도우 크기**: 3.0초 고정 (48,000 샘플, Zero Padding 및 Center Cropping)
3. **Mel-Spectrogram 규격**:
   - ReDimNet & ECAPA: `n_mels=80`, `n_fft=512`, `hop_length=160`
   - AudioResNet-50: `n_mels=128`, `n_fft=2048`, `hop_length=512`
4. **데시벨 변환 및 정규화**:
   - `mel_db = librosa.power_to_db(mel, ref=np.max)`
   - `mel_norm = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)`
5. **결정 임계값**: 0.50 고정 (`final_pred = 1 if prob >= 0.50 else 0`)
