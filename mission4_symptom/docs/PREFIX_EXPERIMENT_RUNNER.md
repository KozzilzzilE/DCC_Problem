# Prefix 공통 실행 연결

기존 팀 학습/판정/중단 알고리즘은 수정하지 않는다. Notebook은 팀 기록 조회용으로 그대로 유지한다.
`scripts/prefix_experiment.py`는 기존 `train_prefix.py`, `prefix_probs.py`, `m4.halting`,
`m4.simulate`, `StreamingSymptomModel`을 연결한다. T8 ASL 옵션과 CPU 검증은
[T8 구현 보고서](T8_ASYMMETRIC_LOSS.md)를 참고한다. 기본 실행은 기존 Weighted BCE다.

## PowerShell 실행

아래는 `mission4_symptom`에서 실행한다. Python은 프로젝트의 기존 가상환경을 사용한다.

```powershell
cd D:\assignment\programing\DCC\DCC_Problem\mission4_symptom
$py = (Resolve-Path ..\.venv\Scripts\python.exe).Path
```

1. 팀 확률표 + 고정 E026 검산. 분류기 학습/추론 없이 LightGBM 예측만 수행한다.

```powershell
& $py -B scripts/prefix_experiment.py verify --check-online
```

`--out ../outputs/mission4_prefix/teamcheck`를 추가하면 새 폴더에 세부 지표와 통화별 중단 배열을 저장한다.
이미 존재하는 출력 경로는 거부한다. 이번 작업의 실제 검산 결과는 `teamverified`,
온라인 특징 재생 검사는 `teamonline`, 최종 상세 산출물은 `teamfinal`에 저장한다. 같은 경로로 재실행하지 않는다.

2. 이후 명시적으로 로컬 E002 대조군 학습을 승인했을 때만 실행한다.

```powershell
& $py -B scripts/prefix_experiment.py run --out ../outputs/mission4_prefix/localcontrol
```

팀 검산 통과 → 기존 E002 설정의 학습 → Dev 확률 추출 → E001/E007 동일 비중 평균 →
저장된 E026 정책 적용 → 지표/중단 배열/팀 기준 차이 저장 순서다. 정책을 다시 학습하지 않는다.
Train 26,280 / Dev 2,920, TAPT 폴더, seed42 및 모든 기존 E002 인자를 유지한다.
CUDA 학습이며 완료 표식 재개/출력 덮어쓰기 기능은 제공하지 않는다.
실패한 실행도 보존하고, 재시도는 다른 새 출력 경로를 지정한다.

3. 이미 새 E002 확률표가 있으면 학습 없이 평가한다.

```powershell
& $py -B scripts/prefix_experiment.py evaluate `
  --e002-table ../outputs/mission4_prefix/localcontrol/dev_prefix.npz `
  --reference ../outputs/mission4_prefix/teamfinal/metrics.json `
  --out ../outputs/mission4_prefix/localcheck
```

후속 T8 결과는 `--reference`에 확정된 로컬 대조군의 metrics.json을 지정한다.
이 명령은 E026의 보정값/특징 순서/모델 문자열/β128/τ0.5/최소5발화를 유지한다.
정책과 E001/E007 확률표는 해시가 달라지면 중단한다. 여러 확률표의 통화 ID·offset·정답 순서를 검사한다.

4. 실제 순차 추론 시간은 별도 명령으로 측정한다. 이번 구성 작업에서는 실행하지 않는다.

```powershell
& $py -B scripts/prefix_experiment.py sequential `
  --e002-run ../outputs/mission4_prefix/localcontrol/training `
  --out ../outputs/mission4_prefix/localsequential --device cuda
```

E001은 `../mission3_symptom/runs/devsel/tl_s42/best_model`, E007은
`runs/E007_prefix_uniform_s43/best_model`만 사용하며 전체 Training 학습 제출 번들을 사용하지 않는다.
E002는 학습 통화 수와 fit ID 파일 해시를 검사한다. E001/E007은 팀 fit-only 경로와 확률표 메타 기록에
근거한다. 특히 E001의 원 학습 ID/가중치 해시를 대조할 별도 원 실행 영수증은 현재 미확보다.
모든 멤버의 파일 해시를 실측 결과에 남긴다. 경로명만으로 학습 출처를 독립 감사했다고 해석하지 않는다.

기존 번들 조립/Streaming 클래스를 재사용하여 fit-only 번들을 별도 출력에 만든다.
각 통화를 reset하고 발화를 한 개씩 전달한다. 4-1은 모든 발화를 처리하고, 4-2는 기존 정책이 멈출 때까지 처리한다.
CUDA 동기화 후 벽시계 시간을 측정한다. 모델 로딩은 별도 기록하며 워밍업은 하지 않는다.
캐시는 미리 메모리에 읽으므로 WAV/JSON 디스크 IO, 번들 조립 시간은 순차 추론 시간에서 제외된다.
이는 모델 파이프라인 실측이며 공식 CLI의 파일 IO 포함 전체 소요와 다르다.
FP16 번들/순차 연산과 배치 확률표 사이에는 수치 차이가 날 수 있다. 점수나 중단을 강제로 맞추지 않는다.

## 산출물과 지표

- `metrics.json`: 4-1/4-2, 클래스별 Precision/Recall/F1/TP/FP/FN/TN, 구간/발화별 4-1,
  평균/중앙값 중단 위치, 정책/입력/환경 해시와 설정. 분모가 0인 세부 Precision 등은 null.
- 4-1 Score는 기존 통화별 평균. 혼동행렬은 모든 발화 시점을 모아서 계산하므로 긴 통화 비중이 크다.
  앞/중간/뒤 및 주요 발화 위치는 기존 `score_4_1`의 집계 그대로다.
- 4-2는 각 통화 중단 시점 한 번의 예측으로 집계하며 기존 `score_halts`가 공식 Score를 계산한다.
- `halts.npz`: 통화 ID와 이번 계산의 중단 발화 위치. 과거 E026 Dev 통화별 중단 배열은 없으므로
  역사적 배열과 같다고 주장하지 않는다. `--check-online`은 기존 확률을 온라인 상태에 재생한 중단과 일괄 계산을 대조한다.
- `training/run_config.json`: 기존 학습기가 기록한 해당 PC의 E002 학습 시간/설정. 팀 과거 시간과 구분한다.
- `dev_prefix.meta.json`: 기존 추출기의 오프라인 배치 추출 시간. 순차 추론 시간으로 대체하지 않는다.
- `sequential_timing.json`: 실제 실행한 경우에만 4-1/4-2 처리 발화 수·초·발화당 시간·로딩 시간과 Score.
  순차 예측/중단도 NPZ로 보존한다. 실행하지 않은 시간은 null이며 추정하지 않는다.

## 보호와 한계

모든 출력은 프로젝트 `outputs/mission4_prefix/` 아래의 새 하위 폴더만 허용한다.
팀 `runs/`, Notebook, 정책, 체크포인트에 쓰지 않는다. 결과는 Git 무시 대상이며 원본 전사/개별 통화 결과를 커밋하지 않는다.
`run`은 기본 Baseline을 유지하며 `--loss asl`로 T8 단일 설정을 선택한다. 동일 evaluator를 사용한다.
학습 상태 보존/재개 방식은 기존 학습기 그대로이며 optimizer/RNG 재개 파일을 새로 제공하지 않는다.
Notebook의 셀16은 저장 결과 조회, 셀18은 단순 기대 점수 중단 평가이므로 새 실행 결과로 오해하지 않는다.
이번 연결에는 Notebook 변경이 필요하지 않다.
