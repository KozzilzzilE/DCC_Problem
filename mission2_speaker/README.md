# Mission 2 119 긴급 통화 화자 식별

본 모듈은 119 긴급 통화 음성 데이터에서 발화자 음성을 활용하여 상황실 요원(0)과 신고자(1)를 이진 분류하는 태스크입니다. 최종 공식 평가 지표는 Accuracy(정확도)이며, 결정 임계값은 0.50을 적용합니다.

## 1. 실행 방법

```bash
pip install -r requirements.txt

# 1) 기존 베이스라인 3모델 균등 앙상블 추론 (과거 checkpoints/)
python inference.py --audio_dir ../data/val/audio --label_dir ../data/val/label --ckpt_path checkpoints/ --output ../outputs/mission2.csv

# 2) 경로 B (고정 추가 학습) 체크포인트 추론 (checkpoints_path_b/)
python inference.py --audio_dir ../data/val/audio --label_dir ../data/val/label --ckpt_path checkpoints_path_b/ --output ../outputs/mission2.csv --ensemble_mode 3model

# 3) 초경량 2모델 (ReDimNet + ECAPA) 앙상블 추론 (옵션)
python inference.py --audio_dir ../data/val/audio --label_dir ../data/val/label --ckpt_path checkpoints/ --output ../outputs/mission2.csv --ensemble_mode 2model

# 4) 단일 모델 추론 (예: ECAPA-TDNN)
python inference.py --audio_dir ../data/val/audio --label_dir ../data/val/label --ckpt_path checkpoints/best_ecapa_tdnn.pt --output ../outputs/mission2.csv
```

- `--ckpt_path` 인자에 가중치가 있는 디렉터리를 지정하면 앙상블 모드가 활성화되며, 단일 `.pt` 파일을 지정하면 단일 모델 추론을 수행합니다.
- `--ensemble_mode {auto, 3model, 2model}` 플래그로 앙상블 방식을 명시할 수 있습니다.
- 출력 CSV 파일 규격: `audio file name`, `startAt`, `endAt`, `speaker` (UTF-8 BOM, 시간 오름차순 정렬).
- 음원 누락이나 로드 실패 시에도 JSON에 정의된 모든 발화 행을 누락 없이 처리하는 fail-safe 로직이 내장되어 있습니다.

## 2. 입력 및 전처리 규격 (비트 레벨 표준화)

- 샘플링 레이트: 16kHz 모노, 목표 길이 3.0초 (48,000 샘플).
- 시간 정규화: 3초 미만 발화는 우측 zero padding, 3초 초과 발화는 중앙 3초 crop.
- ReDimNet2-B2 및 ECAPA-TDNN: Mel 80 / FFT 512 / hop 160 (301 프레임).
- AudioResNet-50: Mel 128 / FFT 2048 / hop 512 (94 프레임).
- Mel 정규화: Librosa Slaney-scale Mel 스펙트로그램 생성 후 `librosa.power_to_db(mel, ref=max(1e-10, max(mel)), top_db=80.0)` 변환 및 `(mel_db + 80) / 80`을 거쳐 `[0.0, 1.0]` 범위로 클립.
- 학습(`m2.path_b.DCCAudioDatasetUnified`), 노트북, 추론 엔진(`m2.infer.Mission2InferenceEngine`)이 동일한 전처리 헬퍼를 공유하여 입력 불일치를 원천 차단했습니다.

## 3. 평가 데이터 규격

- 평가 대상 JSON: 3,640개 파일 (총 111,947개 발화).
- 결측 음원: 1건 (`651e5494386c2a48273e4ed2_20220305.json`, 28개 발화).
- 실제 평가 모집단 (분모): **111,919개 발화** (상황실 53,830개 / 신고자 58,089개).
- 학습 통화 29,142건과 검증 통화 3,640건 간 통화 식별 ID 중복률 0.0%로 엄격히 분리되었습니다.

## 4. 모델 성능 및 검증 상태

### 4.1 기존 보존 가중치 전수 실측치 (111,919건 전수, 임계값 0.50 고정)
*과거 공식 Validation 기반 최고 에폭 선택으로 보존된 가중치의 전수 실측 기록입니다.*

| 후보 ID | 모델 구성 | 검증 정확도 (Accuracy) | Macro F1 | 정답수 / 전체 | 파라미터 수 | 체크포인트 크기 |
|---|---|---|---|---|---|---|
| **E-REN** | **3모델 균등 앙상블 (1/3)** | **92.46%** | **0.9242** | **103,484 / 111,919** | 31.87M | 122.04 MiB |
| **E-RE** | **ReDimNet + ECAPA (2모델)** | **92.42%** | **0.9238** | **103,435 / 111,919** | **8.36M** | **32.08 MiB** |
| S-E | ECAPA-TDNN 단독 | 92.05% | 0.9201 | 103,018 / 111,919 | 5.80M | 22.23 MiB |
| E-EN | ECAPA + ResNet (2모델) | 91.78% | 0.9173 | 102,719 / 111,919 | 29.30M | 112.20 MiB |
| S-R | ReDimNet2-B2 단독 | 91.75% | 0.9171 | 102,685 / 111,919 | 2.57M | 9.84 MiB |
| E-RN | ReDim + ResNet (2모델) | 91.75% | 0.9171 | 102,686 / 111,919 | 26.07M | 99.81 MiB |
| S-N | AudioResNet-50 단독 | 91.17% | 0.9112 | 102,032 / 111,919 | 23.50M | 89.97 MiB |

### 4.2 경로 B (고정 추가 학습) 규격 및 성적 구분
- **배경**: 멘토 피드백에 따라 공식 Validation에 의한 점수 기반 best epoch 선택을 배제하고, 사전에 고정한 학습 횟수를 완료한 마지막 가중치(`fixed_additional_last`)를 선택하는 경로 B를 구현했습니다.
- **실행 규격**:
  - ReDimNet: 부모 에폭 9 가중치에서 1 에폭 고정 추가 학습 -> 최종 10 에폭 가중치
  - ECAPA-TDNN: 부모 에폭 8 가중치에서 2 에폭 고정 추가 학습 -> 최종 10 에폭 가중치
  - AudioResNet-50: 기존 10 에폭 완료 가중치를 Hash 등록 재사용 (추가 학습 0 에폭)
  - 학습 중 공식 Validation 사용 및 점수 기반 선택 완전 배제
- **성적 구분**:
  - 90발화 실측 결과(95.56%)는 격리 환경에서 수행한 CLI smoke 성적입니다.
  - 경로 B 새 가중치의 공식 전수 성적(111,919건)은 동결 설정(`final_model_config.json`)을 통해 1회성으로 평가됩니다.

## 5. 실측 추론 효율성 및 자원 소모 (NVIDIA RTX 3060 Laptop GPU)

- 단일 발화 추론 시간 (Batch 1): 21.84 ms/sample (RTF: 0.00728, 실시간 대비 137배 빠름).
- 배치 처리량: 657.0 샘플/초 (Batch 32), 719.6 샘플/초 (Batch 128).
- 체크포인트 메모리 점유: 122.67 MB, 최대 VRAM 점유: 1,129.12 MB.
