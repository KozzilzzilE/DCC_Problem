# Mission 2 — 119 긴급 통화 화자 분류

단일 발화 음성으로 상황실 접수요원(0)과 신고자(1)를 분류합니다. 최종 구성은 **ReDimNet2-B2 + ECAPA-TDNN + AudioResNet-50, 확률 균등 평균(각 1/3), 임계값 0.50**입니다. 아래 명령은 `mission2_speaker` 폴더에서 실행합니다.

## 설치

검증 환경: Windows 11 / Python 3.11.7 / RTX 3060 Laptop 6GB / PyTorch 2.7.1+cu118.

```text
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
.venv\Scripts\python -m pip install -r requirements.txt
```

이후 `python`은 위 가상환경의 Python입니다. CPU 환경은 Torch 설치 index URL을 `https://download.pytorch.org/whl/cpu`로 바꿉니다.

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

Total / Active 파라미터 수는 모두 **31,866,563**입니다(ReDimNet 2,567,489 / ECAPA 5,795,265 / ResNet 23,503,809). 입력은 16kHz 모노·3초, zero padding과 추론 center crop입니다. Mel 규격은 ReDimNet·ECAPA 80/FFT512/hop160, ResNet 128/FFT2048/hop512입니다.

## 노트북과 실험 코드

[노트북](Mission2_Speaker_Classification.ipynb)은 기준선 → 입력 조건 → 후보 아키텍처 → 음성 특화 비교 → 앙상블 → 최종 결과를 정리합니다. 기본 `RUN_STAGE="recorded_results"`에서 Run All은 보존된 집계 결과를 표시하며 재학습하지 않습니다. 저장된 출력은 새 학습·추론 로그가 아닙니다.

```bash
python tools/show_recorded_results.py
```

노트북을 열려면 `python -m pip install -r requirements-notebook.txt`를 추가 실행합니다. 학습·평가 명령과 검증 도구는 [상세 실행 가이드](experiments/EXECUTION_GUIDE.md), [실험 대응표](experiments/EXPERIMENT_SOURCE_MAP.md)를 참고하세요. 과거 단계의 `m2/path_b.py`와 `run_path_b_fixed_continue.py`는 실행 이력 재현용이며 현재 노트북의 기본 진입점은 아닙니다.

원본 음원·라벨, 캐시, 발화별 예측, 가중치 및 인수인계 문서는 Git 업로드 대상에서 제외합니다.
