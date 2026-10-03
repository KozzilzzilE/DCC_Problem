# Mission 2: 119 긴급 신고 전화 화자 분류 (Speaker Classification)

본 디렉토리는 2026 데이터+AI 혁신 챌린지 (대학부) Mission 2 (상황실 접수요원 vs 신고자 화자 이진 분류)의 최종 제출 코드 및 모델 가중치를 포함합니다.

---

## 1. 제출물 디렉토리 구조

```text
mission2_speaker/
├── checkpoints/                              # 학습 완료된 3대 챔피언 모델 가중치
│   ├── best_redimnet.pt                      # ReDimNet2-B2 가중치 (9.8 MB)
│   ├── best_ecapa_tdnn.pt                    # ECAPA-TDNN 가중치 (22.2 MB)
│   └── best_resnet50.pt                      # AudioResNet-50 가중치 (89.9 MB)
├── m2/                                       # Mission 2 전용 자립형 추론 모듈
│   ├── __init__.py
│   ├── models.py                             # 3대 모델 아키텍처 정의 (PyTorch)
│   └── infer.py                              # 전처리, 앙상블, 예측 엔진
├── Mission2_Speaker_Classification.ipynb     # [학습/실험] 전체 실험 흐름 및 학습 검증 단일 노트북
├── inference.py                              # [채점용] Mission 2 단독 실행 CLI 스크립트
├── requirements.txt                          # 실행 환경 재현을 위한 필수 패키지 목록
└── README.md                                 # 본 안내 문서
```

---

## 2. 모델 명세 및 계산 효율성 (주최 측 권장 명시 사항)

### (1) 파라미터 수 (Total & Active)
| 모델명 | 아키텍처 특성 | Total 파라미터 | Active 파라미터 | 모델 파일 크기 | 최고 정확도 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ReDimNet2-B2** | 2D Local Conv + 1D Dilated Conv + MHA Pooling | 2.57 M | 2.57 M | 9.8 MB | 91.89% |
| **ECAPA-TDNN** | 1D Multi-Scale Res2Net + Attentive Stats Pooling | 5.80 M | 5.80 M | 22.2 MB | 92.10% |
| **AudioResNet-50** | 1채널 2D 광대역 스펙트로그램 텍스처 앵커 | 23.50 M | 23.50 M | 89.9 MB | 91.20% |
| **3-모델 앙상블** | **Soft Voting (1/3 사전 균등 가중치)** | **31.88 M** | **31.88 M** | **121.9 MB** | **92.48%** |

* 모든 파라미터는 추론 시 활성화(Active)되어 앙상블 예측에 참여합니다.

### (2) 학습 및 추론 컴퓨팅 환경
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM, CUDA 11.8)
- **CPU**: 12th Gen Intel Core i7-12700H (14 Cores, 20 Threads, 2.30 GHz)
- **RAM**: 32 GB DDR5
- **OS**: Windows 11 Home 64-bit
- **SW 환경**: Python 3.11, PyTorch 2.7.1+cu118

### (3) Validation 추론 시 Batch Size
- **Batch Size**: 64 (발화 세그먼트 배치 단위 추론)
- 입력 오디오 규격: 16,000 Hz 모노, 3.0초 윈도우 (48,000 샘플, Zero-padding / Center-crop)

### (4) 실측 추론 속도 및 처리량
- **샘플당 평균 추론 지연시간 (RTX 3060 실측)**:
  * ReDimNet2-B2: 1.73 ms / sample (RTF: 0.0006, 초당 1,999.4건 처리)
  * ECAPA-TDNN: 5.12 ms / sample (RTF: 0.0017, 초당 1,304.2건 처리)
  * AudioResNet-50: 4.65 ms / sample (RTF: 0.0016, 초당 632.9건 처리)
  * **3-모델 앙상블 총 지연시간: ~11.50 ms / sample (RTF: 0.0038, 초당 260건 이상 처리)**
- **전체 Validation 데이터셋 (111,947개 발화) 총 추론 시간**:
  * 단일 모델 (ReDimNet2-B2): 약 1.8분 소요
  * 3-모델 앙상블: 약 7.2분 (432초) 소요

---

## 3. 추론 실행 방법 (How to Run)

### A. 미션 단독 실행 (권장)
`mission2_speaker` 디렉토리 내에서 아래 명령어로 단독 실행이 가능합니다:

```bash
cd mission2_speaker
python inference.py --audio_dir ../data/val/audio                     --label_dir ../data/val/label                     --ckpt_path checkpoints/                     --output ../outputs/mission2.csv
```

### B. 프로젝트 루트 통합 실행
프로젝트 최상위 루트 디렉토리에서도 동일하게 실행 가능합니다:

```bash
python inference.py --audio_dir data/val/audio                     --label_dir data/val/label                     --ckpt_path mission2_speaker/checkpoints/                     --output outputs/mission2.csv
```

* `--ckpt_path`에 `checkpoints/` 폴더를 전달하면 3대 모델 앙상블(최고 정확도 92.48%)로 자동 동작합니다.
* 단일 `.pt` 파일 경로를 전달할 경우 해당 모델 단독으로 고속 추론 모드가 동작합니다.

---

## 4. 대회 규정 준수 명세

1. **결정 임계값 0.50 고정 (9/25 공통 FAQ 준수)**:
   * Validation 튜닝 기반의 임계값(0.51 등)을 일절 사용하지 않으며, 전 클래스 0.50 고정 규칙을 엄격히 준수합니다.
2. **사전 균등 가중치 1/3 적용 (9/30 Q&A 준수)**:
   * Validation 데이터셋에 대한 그리드 서치(0.45 / 0.40 / 0.15)를 배제하고, 과적합(Data Leakage)을 원천 차단하기 위해 사전 정의된 균등 1/3 가중치를 적용합니다.
3. **오프라인 환경 완전 보장**:
   * 사전학습 모델의 외부 다운로드 의존성을 완전히 제거하고, `checkpoints/` 디렉토리에 저장된 로컬 가중치 파일로만 로드 및 추론을 수행합니다.
