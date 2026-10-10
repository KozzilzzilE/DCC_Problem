# Mission 2 119 긴급 통화 화자 식별

본 모듈은 119 긴급 통화 음성 데이터에서 발화자 음성을 활용하여 상황실 요원(0)과 신고자(1)를 이진 분류하는 태스크입니다. 최종 공식 평가 지표는 Accuracy(정확도)이며, 결정 임계값은 0.50을 적용합니다.

## 1. 실행 방법

### 1.1 추론 CLI (`inference.py`)

```bash
pip install -r requirements.txt

# 1) 기존 베이스라인 3모델 균등 앙상블 추론 (과거 checkpoints/)
python inference.py --audio_dir ../data/val/audio --label_dir ../data/val/label --ckpt_path checkpoints/ --output ../outputs/mission2.csv

# 2) 경로 B (고정 추가 학습) 최종 체크포인트 추론 (checkpoints_path_b_v2/)
python inference.py --audio_dir ../data/val/audio --label_dir ../data/val/label --ckpt_path checkpoints_path_b_v2/ --output ../outputs/mission2.csv --ensemble_mode 3model

# 3) 초경량 2모델 (ReDimNet + ECAPA) 앙상블 추론 (옵션)
python inference.py --audio_dir ../data/val/audio --label_dir ../data/val/label --ckpt_path checkpoints/ --output ../outputs/mission2.csv --ensemble_mode 2model

# 4) 단일 모델 추론 (예: ECAPA-TDNN)
python inference.py --audio_dir ../data/val/audio --label_dir ../data/val/label --ckpt_path checkpoints/best_ecapa_tdnn.pt --output ../outputs/mission2.csv
```

- `--ckpt_path` 인자에 가중치가 있는 디렉터리를 지정하면 앙상블 모드가 활성화되며, 단일 `.pt` 파일을 지정하면 단일 모델 추론을 수행합니다.
- `--ensemble_mode {auto, 3model, 2model}` 플래그로 앙상블 방식을 명시할 수 있습니다.
- 출력 CSV 파일 규격: `audio file name`, `startAt`, `endAt`, `speaker` (UTF-8 BOM, 시간 오름차순 정렬).
- 음원 누락이나 로드 실패 시에도 JSON에 정의된 모든 발화 행을 누락 없이 처리하는 fail-safe 로직이 내장되어 있습니다.

### 1.2 경로 B 파이프라인 러너 (`experiments/run_path_b_fixed_continue.py`)

```bash
# 기본 설정 확인 (데이터 학습/평가 없이 빠른 정의 확인)
python experiments/run_path_b_fixed_continue.py --stage define_only

# Training 표본 스모크 (실제 Training 경로를 지정; 공식 Validation은 평가하지 않음)
python experiments/run_path_b_fixed_continue.py --stage smoke

# 본 추가 학습 실행 (공식 Validation 평가 및 best 선택 0회)
python experiments/run_path_b_fixed_continue.py --stage full_train

# 완료 체크포인트 동결 설정 생성
python experiments/run_path_b_fixed_continue.py --stage freeze_config

# 동결 설정을 이용한 공식 Validation 전수 1회성 최종 평가
python experiments/run_path_b_fixed_continue.py --stage eval_official

# 해시 검증 및 제출용 패키지 export
python experiments/run_path_b_fixed_continue.py --stage export
```

실제 데이터가 다른 위치에 있으면 `--data_dir`에 Training 루트, `--val_dir`에 공식 Validation 루트를 지정한다. 같은 경로나 서로 포함하는 경로는 거부한다. 알려진 Validation 폴더명과 `VL_`/`VS_` 라벨 파일도 Training에서 거부한다. 폴더명이 정상이라는 사실만으로 내용의 출처가 증명되지는 않으므로, Windows 데이터 명세 및 기존 통화 단위 분리 감사 결과를 확인해야 한다.

중단 후 같은 명령어를 다시 실행하면 마지막 **완료 에포크**에서 재개한다. 에포크 중간의 진행은 마지막 저장 경계부터 다시 실행한다. 부모 해시, 표본 매핑/음원 파일 크기·mtime, 학습 계획, 전처리, 배치 설정, 학습률, seed, 코드 및 실행 환경이 다르면 기존 파일을 덮어쓰지 않고 실패한다. 데이터 fingerprint는 음원 내용 SHA-256을 대체하지 않는다. 완료된 학습·ResNet 등록·같은 설정 동결·같은 export는 재실행 시 기존 파일을 재사용한다.

이전 버전 파일에 필수 실행 계약/이력/스텝 수가 없으면 자동으로 완료 처리하거나 메타데이터를 만들어 넣지 않는다. 원래 `checkpoints/best_*.pt` 부모는 유지하고, 새 본 실행은 `--output_root experiments/runs/path_b_v2`처럼 새 폴더에서 시작한다. 이전 Windows 추가 학습의 사용 가능 여부는 실제 실행 근거를 따로 확인해야 한다. 다른 제출 가중치가 이미 export 폴더에 있으면 `--export_dir checkpoints_path_b_v2`처럼 새 폴더를 사용한다.

코드 검증: `python -m unittest discover -s tests -v` (이 디렉터리에서 실행). 작은 합성 모델/음원에 대한 실행 검증이며 프로젝트 성능 측정이 아니다. 실제 GPU 중단·재개와 신규 전수 성적은 Windows에서 확인한다.

## 2. 입력 및 전처리 규격 (비트 레벨 표준화)

- 공통 전처리 모듈: `m2/audio_features.py`에서 전체 통화 로딩·리샘플링, 발화 절단, 길이 정규화와 Mel 추출을 공유합니다. 같은 라이브러리 환경의 합성 8/16/44.1kHz WAV로 전체 입력을 대조하며 실제 신규 가중치의 CLI parity는 별도 확인합니다.
- 샘플링 레이트: 16kHz 모노, 목표 길이 3.0초 (48,000 샘플).
- 시간 정규화: 3초 미만은 우측 zero padding, 3초 초과는 Training에서 random crop, 평가/CLI에서 center crop.
- ReDimNet2-B2 및 ECAPA-TDNN: Mel 80 / FFT 512 / hop 160 (301 프레임).
- AudioResNet-50: Mel 128 / FFT 2048 / hop 512 (94 프레임).
- Mel 정규화: Librosa Slaney-scale Mel 스펙트로그램 생성 후 `librosa.power_to_db(mel, ref=max(1e-10, max(mel)), top_db=80.0)` 변환 및 `(mel_db + 80) / 80`을 거쳐 `[0.0, 1.0]` 범위로 클립.

## 3. 평가 데이터 규격

- 평가 대상 JSON: 3,640개 파일 (총 111,947개 발화).
- 결측 음원: 1건 (`651e5494386c2a48273e4ed2_20220305.json`, 28개 발화).
- 실제 평가 모집단 (분모): **111,919개 발화** (상황실 53,830개 / 신고자 58,089개).
- 학습 통화 29,142건과 검증 통화 3,640건 간 통화 식별 ID 중복률 0.0%로 엄격히 분리되었습니다.

## 4. 모델 성능 및 검증 상태

### 4.1 기존 보존 가중치 전수 실측치 (111,919건 전수, 임계값 0.50 고정)
*과거 공식 Validation 기반 최고 에폭 선택(`official_validation_best`)으로 보존된 가중치의 전수 실측 기록입니다.*

| 후보 ID | 모델 구성 | 검증 정확도 (Accuracy) | Macro F1 | 정답수 / 전체 | 파라미터 수 | 체크포인트 크기 |
|---|---|---|---|---|---|---|
| **E-REN** | **3모델 균등 앙상블 (1/3)** | **92.46%** | **0.9242** | **103,484 / 111,919** | 31.87M | 122.04 MiB |
| **E-RE** | **ReDimNet + ECAPA (2모델)** | **92.42%** | **0.9238** | **103,435 / 111,919** | **8.36M** | **32.08 MiB** |
| S-E | ECAPA-TDNN 단독 | 92.05% | 0.9201 | 103,018 / 111,919 | 5.80M | 22.23 MiB |
| E-EN | ECAPA + ResNet (2모델) | 91.78% | 0.9173 | 102,719 / 111,919 | 29.30M | 112.20 MiB |
| S-R | ReDimNet2-B2 단독 | 91.75% | 0.9171 | 102,685 / 111,919 | 2.57M | 9.84 MiB |
| E-RN | ReDim + ResNet (2모델) | 91.75% | 0.9171 | 102,686 / 111,919 | 26.07M | 99.81 MiB |
| S-N | AudioResNet-50 단독 | 91.17% | 0.9112 | 102,032 / 111,919 | 23.50M | 89.97 MiB |

### 4.2 경로 B (고정 추가 학습) 규격 및 확정 성적
- **배경**: 멘토 피드백에 따라 공식 Validation에 의한 점수 기반 best epoch 선택을 배제하고, 사전에 고정한 학습 횟수를 완료한 마지막 가중치(`fixed_additional_last`)를 선택하는 경로 B를 구현했습니다.
- **실행 규격**:
  - ReDimNet: 부모 에폭 9 가중치에서 1 에폭 고정 추가 학습 -> 최종 10 에폭 가중치 (`last_redimnet.pt`)
  - ECAPA-TDNN: 부모 에폭 8 가중치에서 2 에폭 고정 추가 학습 -> 최종 10 에폭 가중치 (`last_ecapa_tdnn.pt`)
  - AudioResNet-50: 기존 10 에폭 완료 가중치를 Hash 등록 재사용 (추가 학습 0 에폭, `last_resnet50.pt`)
  - 학습 중 공식 Validation 사용 및 점수 기반 선택 완전 배제
  - 에폭별 원자적 체크포인트 저장 및 전체 RNG 복원 기반 중단 후 재개(Resume) 지원
- **공식 전수 평가 실측 성적 (111,919건)**:
  - 동결 설정(`final_model_config.json`)을 기반으로 독립 1회성 전수 평가 완료
  - **정확도**: **92.4633%** (103,484 / 111,919건 정답)
  - **Macro F1**: **0.9242**
  - **클래스별 재현율**: 상황실(0) 88.32% (47,542 / 53,830), 신고자(1) 96.30% (55,942 / 58,089)
  - **혼동 행렬**: TN=47,542, FP=6,288, FN=2,147, TP=55,942
- **체크포인트 SHA-256 해시**:
  - ReDimNet (`best_redimnet.pt` / `last_redimnet.pt`): `74c6960c4a5e50fa2f7b332a762cb2e4113614ef296d1687e2807a42fd1cb242`
  - ECAPA-TDNN (`best_ecapa_tdnn.pt` / `last_ecapa_tdnn.pt`): `5701706f7053a461982e7046dfcd20a8d3a26e9a74d5a2f6debb71d9ea12ca41`
  - AudioResNet-50 (`best_resnet50.pt` / `last_resnet50.pt`): `55c5b3fbe7588d3159ef85ca2b0d0af2d3a65105017779dcb453d1638fff45aa`
- **CLI 정합성 및 제출 CSV 성적**:
  - 독립 CLI(`inference.py`)와 전수 검증 엔진 간 일치율: **99.9946%** (111,913 / 111,919건, 6건 미세 부동소수점 오차)
  - 실제 제출 CSV(`mission2.csv`) 유효 111,919건 실측치: **정확도 92.4651% (103,486건 정답)**, **Macro F1 0.9242**, 상황실 Recall 88.32%, 신고자 Recall 96.30%
  - 평가 대상: 라벨 JSON 3,640개, 유효 매핑 WAV 3,639개 (누락 1개 통화 28발화는 무음 fail-safe로 온전히 보존되어 총 111,947행 생성)
  - 최종 제출 패키지: `submission_mission2.zip` (용량: 169.72 MiB / 177,965,393 bytes = 177.97 MB, SHA-256: `402b684d9f4ce8c14509d414ad9ba8add0d87b70ad4a3f19f7f7717d7c94bdb4`)
  - 신규 실행에서 점수 기반 best 선택을 제거했으며, 부모의 과거 `official_validation_best` 이력은 보존합니다. 원래 optimizer 상태가 없는 부모에서 추가 학습한 결과를 원래 10에포크 학습 상태의 복원으로 표현하지 않습니다.

## 5. 실측 추론 효율성 및 자원 소모 (NVIDIA RTX 3060 Laptop GPU)

아래는 기존 가중치/당시 계측 구현의 Windows 보고 기록입니다. 신규 경로 B 또는 현재 CLI의 종단간 벤치마크를 대체하지 않습니다. RTF의 분모는 3초 정규화 입력입니다.

- 단일 발화 추론 시간 (Batch 1): 21.84 ms/sample (RTF: 0.00728, 실시간 대비 137배 빠름).
- 배치 처리량: 657.0 샘플/초 (Batch 32), 719.6 샘플/초 (Batch 128).
- 체크포인트 메모리 점유: 122.67 MB, 최대 VRAM 점유: 1,129.12 MB.
