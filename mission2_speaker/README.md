# Mission 2: 119 긴급 신고 전화 화자 분류 (Speaker Classification)

본 문서는 **2026 데이터+AI 혁신 챌린지 Mission 2 (상황실 접수요원 0 vs 신고자 1 이진 분류)** 제출용 단독 실행 가이드 및 모델 명세서입니다.

---

## 1. 빠른 실행 방법 (Quick Start)

추론은 `mission2_speaker` 폴더 안에서 `inference.py` 실행 **단 한 번**으로 완료됩니다.

### Step 1. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

### Step 2. 추론 실행 (단일 명령어)
```bash
python inference.py --audio_dir <wav_폴더경로> \
                    --label_dir <json_라벨폴더경로> \
                    --ckpt_path checkpoints/ \
                    --output ./outputs/mission2.csv
```

* **실제 데이터 실행 예시**:
  ```bash
  python inference.py --audio_dir ../data/val/audio \
                      --label_dir ../data/val/label \
                      --ckpt_path checkpoints/ \
                      --output ../outputs/mission2.csv
  ```

### Step 3. 결과 CSV 파일 확인
명령어 실행이 끝나면 `--output`으로 지정한 위치에 아래 규격의 CSV 파일이 자동 생성됩니다.

| audio file name | startAt | endAt | speaker |
| :--- | :---: | :---: | :---: |
| 651e464d69a4f266f0626837_20220101.wav | 299 | 949 | 0 |
| 651e464d69a4f266f0626837_20220101.wav | 2256 | 3074 | 1 |
| 651e464d69a4f266f0626837_20220101.wav | 3760 | 4577 | 0 |

* `speaker` 판정 기준: **0 = 상황실 접수요원**, **1 = 신고자**

---

## 2. CLI 실행 인자 (Arguments) 설명

| 인자명 | 필수 여부 | 기본값 | 설명 |
| :--- | :---: | :---: | :--- |
| `--audio_dir` | **필수** | - | 평가용 음성 파일(`.wav`)들이 위치한 디렉토리 경로 |
| `--label_dir` | **필수** | - | 전사 텍스트 및 발화 구간(`startAt`, `endAt`)이 적힌 `.json` 파일 디렉토리 경로 |
| `--ckpt_path` | 선택 | `checkpoints/` | 모델 가중치 경로.<br>• 폴더 지정 시: **3대 모델 앙상블(최고 성능 92.48%) 자동 가동**<br>• 단일 `.pt` 파일 지정 시: 해당 모델 단독 고속 추론 모드 |
| `--output` | **필수** | - | 추론 결과가 저장될 CSV 파일 전체 경로 (예: `outputs/mission2.csv`) |

---

## 3. 제출물 폴더 구성

```text
mission2_speaker/
├── checkpoints/                          # 사전학습 가중치 파일 (오프라인 로컬 로드)
│   ├── best_redimnet.pt                  # ReDimNet2-B2 가중치 (9.8 MB)
│   ├── best_ecapa_tdnn.pt                # ECAPA-TDNN 가중치 (22.2 MB)
│   └── best_resnet50.pt                  # AudioResNet-50 가중치 (89.9 MB)
├── m2/                                   # Mission 2 전용 자립형 핵심 엔진
│   ├── __init__.py
│   ├── models.py                         # 3대 모델 순수 PyTorch 신경망 아키텍처
│   └── infer.py                          # 3초 정규화 전처리, 앙상블, CSV 생성 로직
├── Mission2_Speaker_Classification.ipynb # [학습/실험] 전체 실험 흐름 및 검증 단일 노트북
├── inference.py                          # [채점용] 심사위원 단독 실행 CLI 스크립트
├── requirements.txt                      # 실행 환경 재현용 패키지 목록
└── README.md                             # 본 실행 가이드 및 모델 명세서
```

---

## 4. 모델 명세 및 계산 효율성 (주최 측 권장 명시 사항)

### (1) 모델 파라미터 수 (Total & Active)
* **총 파라미터 수**: **31.88 M**
* **Active 파라미터 수**: **31.88 M (100% 추론 활성화)**

| 모델명 | 주요 특징 | Total 파라미터 | Active 파라미터 | 모델 파일 크기 | 단독 검증 정확도 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ReDimNet2-B2** | 2D Local Conv + 1D Dilated Conv + MHA Pooling | 2.57 M | 2.57 M | 9.8 MB | 91.89% |
| **ECAPA-TDNN** | 1D Multi-Scale Res2Net + Attentive 통계적 풀링 | 5.80 M | 5.80 M | 22.2 MB | 92.10% |
| **AudioResNet-50** | 1채널 2D 광대역 스펙트로그램 텍스처 앵커 | 23.50 M | 23.50 M | 89.9 MB | 91.20% |
| **3대 챔피언 앙상블** | **Soft Voting (사전 1/3 균등 가중치)** | **31.88 M** | **31.88 M** | **121.9 MB** | **92.48%** |

### (2) 컴퓨팅 환경
* **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM, CUDA 11.8)
* **CPU**: 12th Gen Intel Core i7-12700H (14 Cores, 20 Threads, 2.30 GHz)
* **RAM**: 32 GB DDR5
* **OS**: Windows 11 Home 64-bit
* **소프트웨어**: Python 3.11, PyTorch 2.7.1+cu118

### (3) Validation 추론 시 Batch Size
* **Batch Size**: 64 (발화 세그먼트 단위 추론)
* 오디오 규격: 16,000 Hz 모노, 3.0초 고정 윈도우 (48,000 샘플, Zero-padding / Center-crop)

### (4) 실측 추론 시간 및 처리량 (RTX 3060 기준)
* **샘플당 평균 추론 지연시간 (Latency)**:
  * ReDimNet2-B2: 1.73 ms / sample (RTF: 0.0006, 초당 1,999.4건 처리)
  * ECAPA-TDNN: 5.12 ms / sample (RTF: 0.0017, 초당 1,304.2건 처리)
  * AudioResNet-50: 4.65 ms / sample (RTF: 0.0016, 초당 632.9건 처리)
  * **3대 앙상블 합산 지연시간: ~11.50 ms / sample (RTF: 0.0038, 초당 260건 이상 처리)**
* **전체 검증셋 (111,947개 발화) 총 추론 소요 시간**:
  * 단일 모델 (ReDimNet2-B2): 약 1.8분
  * 3대 앙상블 전체: **약 7.2분 (432초)**

---

## 5. 대회 공식 규정 준수 확인

1. **결정 임계값 0.50 고정 (9/25 공통 FAQ)**:
   * 검증셋에 맞춘 사후 임계값(0.51 등) 튜닝을 배제하고, 모든 클래스에 대해 0.50 고정 임계값을 적용했습니다.
2. **사전 1/3 균등 가중치 적용 (9/30 Q&A)**:
   * 검증셋에 대한 하이퍼파라미터 그리드 서치(0.45 / 0.40 / 0.15)를 배제하고, 사전 정의된 1/3 균등 가중치로 결합하여 과적합(Data Leakage)을 차단했습니다.
3. **오프라인 환경 보장**:
   * 실행 시 외부 다운로드 없이 `checkpoints/` 내의 로컬 가중치 파일로만 로드 및 추론을 수행합니다.
