# Mission 2 — 119 긴급 통화 화자 분류

단일 발화 음성으로 상황실 접수요원(0)과 신고자(1)를 분류합니다. 최종 구성은 **ReDimNet2-B2 + ECAPA-TDNN + AudioResNet-50, 확률 균등 평균(각 1/3), 임계값 0.50**입니다. 아래 명령은 `mission2_speaker` 폴더에서 실행합니다.

## 설치

검증 환경: Windows 11 / Python 3.11.7 / RTX 3060 Laptop 6GB / PyTorch 2.7.1+cu118.

```text
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\activate
```

Mission 2는 `torchaudio`를 사용하지 않으므로 설치하지 않습니다. 오디오 로딩·Mel 변환은 `soundfile`·`librosa`를 사용합니다.

활성화 후 `python`은 위 가상환경의 Python입니다. PowerShell에서는 `.\.venv\Scripts\Activate.ps1`을 사용하거나 `python` 대신 `.venv\Scripts\python`을 지정합니다. CPU 환경은 Torch 설치 index URL을 `https://download.pytorch.org/whl/cpu`로 바꿉니다.

## 제출 추론

별도 전달하는 최종 가중치 세 파일을 `checkpoints_final/`에 넣습니다.

```text
checkpoints_final/best_redimnet.pt
checkpoints_final/best_ecapa_tdnn.pt
checkpoints_final/best_resnet50.pt
```

```bash
python inference.py --audio_dir "D:/data/val/audio" --label_dir "D:/data/val/label" --ckpt_path checkpoints_final --ensemble_mode 3model --output outputs/mission2.csv
```

- 실행 시 모델 다운로드 없이 로컬 가중치를 사용합니다. Git 저장소에는 최종 가중치가 포함되지 않습니다.
- WAV와 JSON을 파일 stem으로 연결하고, `startAt`, `endAt`의 밀리초 구간을 사용합니다. 대화 텍스트와 정답 역할은 모델 입력으로 사용하지 않습니다.
- 출력은 UTF-8 BOM CSV의 4열: `audio file name`, `startAt`, `endAt`, `speaker`. 음원 이름에는 `.wav` 확장자가 포함됩니다.
- 음원 누락·읽기 실패는 무음 대체 예측을 생성하고, NaN/Inf 확률은 오류로 처리합니다.
- `python inference.py --help`로 인자를 확인할 수 있습니다. 2모델 실행은 `--ensemble_mode 2model`입니다.

## 최종 기록

| 항목 | 배치 평가 | 제출 CLI CSV |
|---|---:|---:|
| 유효 평가 발화 | 111,919 | 111,919 |
| 정답 수 | 103,484 | 103,486 |
| Accuracy | 92.4633% | 92.4651% |
| Macro F1 | 0.9242 | 0.9242 |
| Batch size | 32 | 1 |

CSV 전체는 111,947행(음원 누락 대체 28행 포함)입니다. 배치·CLI 예측은 99.9946% 일치하며 6건 차이가 있습니다. Windows 최종 테스트 보고의 CLI 전체 추론시간은 3,133.80초, 평균 27.99ms/행입니다.

### 통화 단위 batch 추론과 FP32 고정 (10-12)

변경 내용:
- 제출 CLI 는 한 통화 안의 발화를 최대 `--batch_size`(기본 64)개씩 묶어 모델마다 한 번에 추론한다.
- Mel 추출 코드는 그대로이며, 발화별로 모델을 따로 호출하며 생기던 GPU 대기만 없앴다.

GPU 정밀도:
- 추론은 GPU 에서도 FP32 로 계산한다. `m2/infer.py` 에서 TF32 를 끈다.
- PyTorch 기본값인 TF32 합성곱에서는 cuDNN 이 batch 크기마다 다른 알고리즘을 골라 확률이 최대 약 0.009 달라졌다.
  - 그 결과 0.5 근처 발화의 판정이 실행 조건에 따라 바뀌었다 (TF32 batch 1 vs 64 에서 7발화).
  - 위 표의 배치·CLI 6건 차이도 같은 원인으로 보인다.
- FP32 고정 후에는 batch 1 과 64 의 확률 차이가 최대 5.6e-6, 판정 변동 0건이다 (2,946발화).

| 항목 (Validation 전체 111,947행, RTX 5060 8GB · Ryzen 5 9600 · torch 2.13.0+cu130) | 변경 전 (TF32, batch 1) | 변경 후 (FP32, batch 64) |
|---|---:|---:|
| 정답 수 (전 발화 111,947) | 103,509 | 103,511 |
| 정답 수 (위 표 분모 111,919, 누락 음원 파일 28발화 제외) | 103,486 | 103,488 |
| Accuracy / Macro F1 | 92.46% / 0.9242 | 92.46% / 0.9242 |
| 전체 추론 시간 | 1,182초 (10.6 ms/행) | 656초 (5.9 ms/행) |

- 변경 전후 예측이 다른 발화는 6건이고, 모두 확률이 0.5 에서 0.006 이내다.
- 측정 PC 에는 위 표에서 누락됐던 음원(`651e4f3cbd89bfb121b4678d_20220807`)이 있어 111,947발화 전체를 평가했다. 그 파일을 빼면 변경 전 결과가 위 표의 CLI 행렬과 정확히 같다.

Total / Active 파라미터 수는 모두 **31,866,563**입니다(ReDimNet 2,567,489 / ECAPA 5,795,265 / ResNet 23,503,809). 입력은 16kHz 모노·3초, zero padding과 추론 center crop입니다. Mel 규격은 ReDimNet·ECAPA 80/FFT512/hop160, ResNet 128/FFT2048/hop512입니다.

## 노트북과 실험 코드

[노트북](Mission2_Speaker_Classification.ipynb)은 기준선 → 입력 조건 → 후보 아키텍처 → 음성 특화 비교 → 앙상블 → 최종 결과를 정리합니다. 기본 `RUN_STAGE="recorded_results"`에서 Run All은 보존된 집계 결과를 표시하며 재학습하지 않습니다. 저장된 출력은 새 학습·추론 로그가 아닙니다.

```bash
python tools/show_recorded_results.py
```

노트북을 열려면 `python -m pip install -r requirements-notebook.txt`를 추가 실행합니다. 학습·평가 명령과 검증 도구는 [상세 실행 가이드](experiments/EXECUTION_GUIDE.md), [실험 대응표](experiments/EXPERIMENT_SOURCE_MAP.md)를 참고하세요. 과거 단계의 `m2/path_b.py`와 `run_path_b_fixed_continue.py`는 실행 이력 재현용이며 현재 노트북의 기본 진입점은 아닙니다.

원본 음원·라벨, 캐시, 발화별 예측, 가중치 및 인수인계 문서는 Git 업로드 대상에서 제외합니다.
