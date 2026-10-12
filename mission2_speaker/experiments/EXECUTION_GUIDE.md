# Mission 2 — 119 긴급 통화 화자 분류 실행 가이드

음성만으로 상황실 접수요원(0)과 신고자(1)를 분류합니다. 최종 구성은 **ReDimNet2-B2 + ECAPA-TDNN + AudioResNet-50, 확률 균등 평균(각 1/3), 임계값 0.50**입니다. 명령은 `mission2_speaker` 폴더에서 실행합니다.

## 1. 노트북 확인

노트북은 **ResNet 기준선 → 윈도우·패딩 → 후보 아키텍처 → 음성 특화 대조 → 단일·앙상블 → 최종 성적** 순서로 이유·방법·결과·결정을 설명합니다.

HuBERT·WavLM은 OOM과 Colab 세션 만료, SSAST-Tiny는 OOM과 세션 제한으로 실행·검증을 중단했습니다. 추가 구조·학습 검증의 시간 비용을 고려해 해당 후보들을 보류하고 ResNet·ReDimNet·ECAPA를 앙상블 비교 후보로 좁혔습니다.

기본값 `RUN_STAGE = "recorded_results"`에서 Run All은 결과 표만 읽으며 학습을 시작하지 않습니다. 저장된 실험 표를 새 학습의 실행 로그로 사용하지 않습니다. 학습 실행은 해당 단계로 명시적으로 변경합니다.

```bash
python tools/show_recorded_results.py
python tools/save_recorded_outputs.py
python tools/show_recorded_results.py --format html --output recorded_results.html
```

첫 명령은 결과 확인, 두 번째는 결과 표시 셀의 출력 저장, 세 번째는 HTML 생성입니다. 결과 표시에는 Python 표준 라이브러리만 사용하고 GPU·음원 추론·학습을 실행하지 않습니다.

## 2. 환경 설치

검증 환경: Windows 11, Python 3.11.7, RTX 3060 Laptop GPU 6GB, PyTorch 2.7.1+cu118, Librosa 0.11.0.

```text
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\activate
```

Mission 2는 `torchaudio`를 사용하지 않으므로 설치하지 않습니다. 오디오 로딩·Mel 변환은 `soundfile`·`librosa`를 사용합니다.

활성화 후 `python`은 이 가상환경의 Python을 의미합니다. PowerShell에서는 `.\.venv\Scripts\Activate.ps1`을 사용하거나 `python` 대신 `.venv\Scripts\python`을 지정합니다. CPU 환경은 Torch index URL을 `https://download.pytorch.org/whl/cpu`로 바꿉니다. Mac은 일반 pip 설치를 사용하며, 현재 추론 코드는 CPU 또는 CUDA를 선택합니다.

노트북을 직접 열 때만 추가 설치:

```bash
python -m pip install -r requirements-notebook.txt
python -m jupyter lab Mission2_Speaker_Classification.ipynb
```

`requirements.txt`는 최소 버전 명세입니다. 최종 실행 환경은 `python -m pip freeze > requirements.windows.lock.txt`로 따로 보관합니다.

## 3. 최종 가중치와 제출 추론

별도 전달할 최종 가중치를 `checkpoints_final/`에 배치합니다. 폴더명이 다르면 `--ckpt_path`로 지정하면 됩니다.

```text
mission2_speaker/
├── inference.py
├── requirements.txt
├── m2/
└── checkpoints_final/
    ├── best_redimnet.pt
    ├── best_ecapa_tdnn.pt
    └── best_resnet50.pt
```

```bash
python inference.py --audio_dir "D:/data/val/audio" --label_dir "D:/data/val/label" --ckpt_path checkpoints_final --ensemble_mode 3model --output outputs/mission2.csv
```

- 추론은 로컬 모델 구조와 `.pt`를 로드하며 인터넷 다운로드에 의존하지 않습니다.
- WAV와 JSON을 동일한 stem으로 매칭하고 하위 폴더도 검색합니다.
- `utterances`(또는 `dialogs`, `dialogue`)의 밀리초 `startAt`, `endAt`을 읽습니다. 텍스트·정답 역할은 모델 입력으로 사용하지 않습니다.
- 출력은 UTF-8 BOM, `audio file name`, `startAt`, `endAt`, `speaker`의 4열입니다.
- 음원 누락·읽기 실패는 해당 JSON 발화에 무음 대체 예측을 생성합니다. NaN/Inf 확률은 오류로 처리합니다.
- `python inference.py --help`는 가중치를 로드하지 않습니다.

2모델 실행은 `--ensemble_mode 2model`, 단일 파일 실행은 `--ckpt_path checkpoints_final/best_ecapa_tdnn.pt --ensemble_mode auto`를 사용합니다. 후보 비교의 2모델 92.42%를 최종 가중치의 2모델 성적으로 자동 대체하지 않습니다.

## 4. 최종 결과와 입력 규격

| 항목 | 배치 평가 | 제출 CLI CSV |
|---|---:|---:|
| 평가 발화 | 111,919 | 111,919 |
| 정답 | 103,484 | 103,486 |
| Accuracy | 92.4633% | 92.4651% |
| Macro F1 | 0.9242 | 0.9242 |
| 상황실 Recall | 88.32% | 88.32% |
| 신고자 Recall | 96.30% | 96.30% |
| Batch size | 32 | 1 |

CSV는 **111,947행**이며 음원 누락 대체 28행을 포함합니다. 정상 음향 평가 분모는 111,919발화입니다. 배치·CLI 라벨 일치율은 99.9946%(6건 차이)입니다.

| 모델 | Total = Active 파라미터 | Mel / FFT / hop |
|---|---:|---|
| ReDimNet2-B2 | 2,567,489 | 80 / 512 / 160 |
| ECAPA-TDNN | 5,795,265 | 80 / 512 / 160 |
| AudioResNet-50 | 23,503,809 | 128 / 2048 / 512 |
| 합계 | 31,866,563 | - |

공통 규격은 16kHz 모노·3초, 짧으면 zero padding, 길면 학습 random crop / 추론 center crop입니다. `m2/audio_features.py`가 Mel 정규화를 공유합니다. CLI는 현재 실행의 시간을 출력하며 과거 효율성 계측과 구분합니다.

## 5. 10에포크 학습과 입력 조건 비교

```bash
python experiments/train_models.py --training-root "D:/data/train" --output-dir outputs/train10 --model all --epochs 10 --batch-size 32 --lr 0.0001 --seed 42 --device cuda
```

각 모델은 epoch 1~10의 단일 루프로 `last_<model>.pt`, `history.json`, `run_config.json`, `split_manifest.json`을 저장합니다. 마지막 epoch를 사용하고 내부 Dev 최고점으로 last를 교체하지 않습니다. 기본값은 FP32·Adam·cosine scheduler입니다. 이전 파일을 덮어쓰지 않도록 빈 출력 디렉터리를 사용합니다.

기본 초기화는 random입니다. ResNet의 로컬 ImageNet 초기화를 사용하려면 `--resnet-init "D:/weights/resnet50_imagenet.pth"`를 추가합니다. torchvision ResNet-50 state dict 형식이며 RGB 첫 합성곱을 채널 평균해 1채널로 바꿉니다. 초기화·학습 설정에 따라 결과가 달라지므로 새 실행 성적이 기존 실험 표와 같다고 가정하지 않습니다.

Training 내부 통화 단위 80:20 분할을 기본 사용합니다. 학습 함수에 공식 Validation을 전달하지 않습니다. `--dev-fraction 0`은 Training 전부를 사용합니다. BatchNorm을 위해 마지막 불완전 배치를 제외하며 제외 발화 수를 설정·history에 기록합니다.

네 입력 조건을 같은 분할·seed·예산으로 비교:

```bash
python experiments/train_models.py --training-root "D:/data/train" --output-dir outputs/windows --model resnet50 --window-suite --epochs 10 --seed 42 --device cuda
```

초기 탐색 표의 Validation 부분 집합과 새 내부 Dev 비교는 평가 범위가 다릅니다.

확정 가중치의 모델별 확률과 단일·균등 7조합 평가:

```bash
python experiments/evaluate_models.py --evaluation-root "D:/data/val" --checkpoints-dir checkpoints_final --output-dir outputs/new_evaluation --batch-size 32 --device cuda
```

평가 폴더 아래 `audio/`, `label/`을 배치합니다. 모델별 확률은 같은 발화 순서로 저장하고 SHA-256·batch size·시간을 함께 기록합니다. 출력은 `model_probabilities.json`, `ensemble_metrics.json`입니다. 음원 누락 발화는 유효 평가에서 제외합니다.

## 6. 음성 특화 실험

```bash
python experiments/quick_hypotheses.py prepare --training-root "D:/data/train" --cache-dir "D:/pilot500/cache" --max-calls 500 --max-per-call 0 --dev-fraction 0.2 --seed 42 --mel-profile resnet128
python experiments/quick_hypotheses.py probe --cache-dir "D:/pilot500/cache" --output-dir "D:/pilot500/probe_s42" --model resnet50 --conditions baseline short rms centroid zcr bandpass --epochs 5 --batch-size 16 --lr 0.0001 --seed 42 --head-dropout 0.3 --bootstrap 2000 --device cuda
```

seed 43은 같은 캐시에서 출력 경로와 seed를 바꿉니다. 조건은 해당 실행의 baseline과 비교합니다. 노트북에는 200통화 탐색, 500통화 비교, 1,000통화 ZCR 재확인 결과가 모두 있습니다.

1,000통화 캐시 준비에는 이전 500통화의 `selected_calls.json`을 `--exclude-calls`로 지정하고, `--dedup-mel`로 Mel을 공유합니다. 준비 후:

```bash
python experiments/run_zcr_confirmation.py --stage train --cache-dir "D:/pilot1000/cache" --output-dir "D:/pilot1000/run" --device cuda
```

채택 기준: 두 seed 모두 정확도 상승, 평균 +0.5%p 이상, 평균 Macro F1 비하락, 각 seed·양 클래스 Recall 하락 <1.0%p. 확대 확인은 두 seed CI 하한 >0도 요구합니다. RMS·bandpass의 미채택은 정확도 개선이 없다는 뜻이 아닙니다.

## 7. 코드와 제출 확인

| 파일 | 역할 |
|---|---|
| Mission2_Speaker_Classification.ipynb | 실험 이유·방법·결과·결정과 단계별 실행 |
| inference.py | 제출 CSV 생성 |
| m2/models.py, audio_features.py, infer.py | 모델·공유 전처리·추론 |
| m2/experiment_workflow.py | 입력 조건·통화 분할·10에포크 학습·7조합 계산 |
| experiments/train_models.py | 학습 / 윈도우 비교 명령 |
| experiments/evaluate_models.py | 모델별 확률·단일/균등 7조합 평가 |
| experiments/quick_hypotheses.py | 음향 통계·캐시·조건별 파일럿 |
| experiments/run_all_audio_hypotheses.py | 통합 비교 러너, 명시적 plan 사용 |
| experiments/run_zcr_confirmation.py | ZCR 재확인·판정 |
| m2/recorded_results.py, tools/, results/ | 집계 지표 검증·출력 저장·HTML 표 |

제출 전에는 최종 가중치의 SHA-256을 노트북 마지막 표와 대조하고 독립 폴더에서 실제 음성으로 `inference.py`를 실행합니다. 종료 코드·CSV 4열·행 수·이진 라벨을 확인하고 코드·가중치·PPT·노트북의 구성을 맞춥니다. 가중치를 포함해 제출 ZIP을 새로 만들면 해시도 다시 계산합니다.

음원·라벨·캐시·발화별 예측·대형 로그는 Git 업로드 대상에서 제외합니다. 결과를 확인하는 데 재학습은 필요하지 않으며, 이 정리본에는 가중치·원본 음원이 포함되어 있지 않습니다.

Windows 최종 테스트에는 `tools/verify_submission.py`를 사용합니다. `check-weights`는 최종 SHA-256, `prepare-smoke`는 실제 음성 표본·경계 사례 준비, `check-csv`는 발화 매핑·행 수·형식·성적, `run`은 UTF-8 실행 로그를 담당합니다. `python tools/verify_submission.py --help`에서 인자를 확인할 수 있습니다. 성적 비교에서 차이가 있으면 기능 검사 통과와 별도로 검토합니다.


## 재개 범위

`quick_hypotheses.py probe`는 동일 설정·출력 경로를 다시 실행하면 에포크 경계 체크포인트에서 재개합니다. 수정 전 구형 파일에 DataLoader 난수 상태가 없으면 새 출력 폴더를 사용해야 합니다. 실행 시간은 설정 동일성 판단에서 제외합니다.

`run_all_audio_hypotheses.py --resume`는 같은 계획에서 이미 완료된 조건을 검증한 뒤 건너뛰는 옵션입니다. 부분 학습을 이어가는 기능은 아닙니다. 계획이나 데이터를 변경할 때는 새 출력 디렉터리를 사용합니다.
