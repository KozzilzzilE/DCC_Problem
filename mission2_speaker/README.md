# Mission 2 119 긴급 통화 화자 분류

본 과제는 119 긴급 통화 음성의 단일 발화 구간만을 활용하여 상황실 접수요원(0)과 신고자(1)를 이진 분류하는 문제다. 공식 출제 규정에 따른 주 평가지표는 Accuracy(정확도)이며, 결정 임계값은 0.50으로 고정한다.

## 1. 실행 방법

```bash
pip install -r requirements.txt

# 기본 3모델 앙상블 실행
python inference.py --audio_dir ../data/val/audio --label_dir ../data/val/label --ckpt_path checkpoints/ --output ../outputs/mission2.csv

# 초경량 2모델 (ReDimNet + ECAPA) 앙상블 실행 (선택 옵션)
python inference.py --audio_dir ../data/val/audio --label_dir ../data/val/label --ckpt_path checkpoints/ --output ../outputs/mission2.csv --ensemble_mode 2model

# 단일 모델 실행 (예: ECAPA-TDNN)
python inference.py --audio_dir ../data/val/audio --label_dir ../data/val/label --ckpt_path checkpoints/best_ecapa_tdnn.pt --output ../outputs/mission2.csv
```

- `--ckpt_path` 인자에 세 가중치가 있는 폴더를 전달하면 앙상블 모드가 활성화되며, 개별 `.pt` 파일을 전달하면 단일 모델 모드로 동작한다.
- 출력 CSV 열 규격: `audio file name`, `startAt`, `endAt`, `speaker` (UTF-8 BOM, 시간은 정수 밀리초).
- 음원 파일 누락이나 로딩 실패 시에도 JSON에 정의된 모든 발화 행을 무음 처리 기반으로 보존하는 fail-safe 방어 로직이 적용되어 있다.

## 2. 입력 및 전처리 규격

- 샘플링 레이트: 16kHz 모노, 고정 길이 3.0초 (48,000 샘플).
- 길이 정규화: 3초 미만 발화는 뒤쪽 zero padding, 3초 초과 발화는 중앙 3초 crop.
- ReDimNet2-B2 및 ECAPA-TDNN: Mel 80 / FFT 512 / hop 160 (301 프레임).
- AudioResNet-50: Mel 128 / FFT 2048 / hop 512 (94 프레임).
- Mel 정규화: 상대 dB를 `(mel_db + 80) / 80`으로 선형 변환 후 `[0.0, 1.0]` 범위로 clip.

## 3. 평가 데이터 모집단

- 검증 라벨 JSON: 3,640개 파일 (총 111,947개 발화).
- 음원 누락 파일: 1개 (`651e5494386c2a48273e4ed2_20220305.json`, 28개 발화).
- 실제 평가 가능 모집단: **111,919개 발화** (상황실 53,830건 / 신고자 58,089건).
- 학습 통화 29,142건과 검증 통화 3,640건 간의 통화 세션 ID 중복률은 0.0%로 분리 독립성이 검증되었다.

## 4. 후보 모델별 실측 성능 비교 (111,919건 전수, 임계값 0.50)

| 후보 ID | 모델 구성 | 검증 정확도 (Accuracy) | Macro F1 | 정답수 / 전체 | 파라미터 수 | 체크포인트 크기 |
|---|---|---|---|---|---|---|
| **E-REN** | **3모델 균등 앙상블 (1/3)** | **92.46%** | **0.9242** | **103,484 / 111,919** | 31.87M | 122.04 MiB |
| **E-RE** | **ReDimNet + ECAPA (2모델)** | **92.42%** | **0.9238** | **103,435 / 111,919** | **8.37M** | **32.07 MiB** |
| S-E | ECAPA-TDNN 단독 | 92.05% | 0.9201 | 103,018 / 111,919 | 5.80M | 22.23 MiB |
| E-EN | ECAPA + ResNet (2모델) | 91.78% | 0.9173 | 102,719 / 111,919 | 29.30M | 112.20 MiB |
| S-R | ReDimNet2-B2 단독 | 91.75% | 0.9171 | 102,685 / 111,919 | 2.57M | 9.84 MiB |
| E-RN | ReDim + ResNet (2모델) | 91.75% | 0.9171 | 102,686 / 111,919 | 26.07M | 99.81 MiB |
| S-N | AudioResNet-50 단독 | 91.17% | 0.9112 | 102,032 / 111,919 | 23.50M | 89.97 MiB |
| *W-EER* | *(과거 참고) 가중 앙상블 .45/.40/.15* | 92.54% | 0.9250 | 103,565 / 111,919 | 31.87M | 122.04 MiB |

- 과거 Git 3638dab의 보존 기록(92.48%)은 이번 실측 3모델 앙상블(92.46%)과 0.02%p(17건) 차이로 정합성이 입증되었다.
- 2모델 앙상블(ReDim + ECAPA)은 92.42%로 3모델 대비 0.04%p(49건) 차이에 불과하면서, 파라미터와 용량을 74% 절감하고 전처리를 80-Mel로 단일화할 수 있는 경량 최적화 대안이다.

## 5. 실측 추론 효율성 및 자원 계측 (NVIDIA RTX 3060 Laptop GPU)

- 단일 발화 서빙 지연시간 (Batch 1): 21.84 ms/sample (RTF: 0.00728, 실시간 대비 137배 고속).
- 배치 처리량: 657.0 건/초 (Batch 32), 719.6 건/초 (Batch 128).
- 가중치 메모리 점유: 122.67 MB, 최대 VRAM 점유: 1,129.12 MB.

자세한 실험 배경, 도메인 가설, 전처리 대조 실험 결과는 [최종 실험 보고서](../docs/FINAL_EXPERIMENT_REPORT.md)에서 확인할 수 있다.
