# T4 Head-only → Full Fine-tuning 구현 및 CPU 검증

2026-11-18. 탐색적 통제 실험이다. 초기 backbone 업데이트가 기존 오류의 원인이라는 직접 증거는 없다. GPU 학습·실제 모델 초기화/forward/inference는 실행하지 않았다.

## 시작 상태 및 변경 범위

시작 브랜치 `mission4-prefix-t4-staged-fine-tuning`, 시작 HEAD와 `mission4-prefix` 모두 `1f69317e2cf280729451abdd8576576679b5c606`. 시작 작업 트리는 깨끗했다. 브랜치 생성·전환·커밋·Push·Merge는 하지 않았다.

- `m4/staged.py`: 학습 전용 입력 검증, freeze 경계, stage 제어, 경계 해시·state 확인.
- `scripts/train_prefix.py`: 선택 옵션, stage 진입, 노출·update/skip·메모리·시간 기록.
- `scripts/prefix_experiment.py`: 옵션 전달과 기존 배치·평가·순차 실행 연결. 평가 함수 자체는 불변.
- `tests/test_staged.py`: synthetic CPU 및 실제 main의 대역 실행 테스트.
- 이 문서. 검증 영수증은 `outputs/mission4_prefix/t4_cpu_verification/`.

## 경계 및 실제 구현

설치된 Transformers의 `RobertaForSequenceClassification` 소스에서 `roberta=RobertaModel(add_pooling_layer=False)`, `classifier=RobertaClassificationHead`를 확인했다. Head는 첫 토큰 선택 → Dropout → dense(768→768) → tanh → Dropout → out_proj(768→9)이다. Dropout은 파라미터가 없고 config의 classifier_dropout이 null이므로 hidden_dropout_prob=0.1을 사용한다. Backbone의 hidden/attention dropout도 기존0.1이다.

실제 localcontrol safetensors의 헤더만 읽어 다음을 확인했다. 가중치를 실제 모델로 로드하지 않았다.

| 구분 | 파라미터 수 | 이름 경계 |
|---|---:|---|
| Backbone | 110,027,520 | `roberta.*` / `model.base_model` |
| Head | 597,513 | `classifier.dense.{weight,bias}`, `classifier.out_proj.{weight,bias}` |
| 전체 | 110,625,033 | 두 집합이 중복 없이 전체를 덮음 |

학습 시에도 모듈 소속으로 경계의 중복·누락을 검사한다. Epoch1에서는 backbone requires_grad=False 및 grad=None, epoch2~3에서는 True다. `model.train()`을 매 epoch 그대로 호출하고 backbone을 eval로 전환하지 않는다. `no_grad` 문맥이나 새 모델 로드도 추가하지 않았다. Stage1의 backbone forward에서 Dropout은 계속 동작하며 head만 역전파한다.

## Optimizer·Scheduler

기존 `llrd_groups`는 requires_grad인 파라미터만 등록한다. 따라서 **모든 파라미터가 아직 활성인 시점에 기존 함수로 AdamW를 생성한 뒤** StageRecorder를 만들고 epoch 진입 시 freeze한다. LLRD 함수는 수정하지 않았다. 전체 파라미터가 Optimizer에 정확히1번 등록됐는지 검사한다.

Stage 전환에서 모델·Optimizer·Scheduler를 재생성하지 않는다. Head Adam state 객체와 모멘텀을 유지한다. Stage1 backbone에는 grad=None이므로 AdamW weight decay·state 생성이 발생하지 않는다. Stage2 첫 유효 gradient가 있는 성공 update에서 backbone Adam state가 생성된다. 첫 attempt가 AMP skip이면 그보다 늦어진다.

단일9,855-step linear scheduler, warmup985step(`int(.1*9855)`), lr5e-5/LLRD0.8/AdamW weight_decay0.01, clipping1.0, CUDA FP16 autocast/GradScaler를 유지한다. 기존처럼 AMP skip에도 scheduler는 진행한다. Stage2에서 restart·추가 warmup·별도 LR은 없다. Backbone은 첫 epoch의 warmup·초기 높은 LR에서 업데이트되지 않는다. 따라서 순서뿐 아니라 backbone 학습량과 실제 적용 LR 구간도 바뀐다.

## 계획 학습량과 로그

| Stage | epoch | trainable / frozen 파라미터 | 원본노출 | Head노출 | Backbone노출 | 계획 update 기회 |
|---|---|---|---:|---:|---:|---:|
| Head-only | 1 | 597,513 / 110,027,520 | 52,560 | 52,560 | 0 | 3,285 |
| Full | 2~3 | 110,625,033 / 0 | 105,120 | 105,120 | 105,120 | 6,570 |
| 전체 | 3epoch | — | 157,680 | 157,680 | 105,120 | 9,855 |

Baseline과 전체 원본 노출·계획step은 같지만 backbone 노출/업데이트 기회는1/3 감소한다. 추가 epoch는 없다. 위 노출은 중복 Prefix를 포함하며 고유 입력 수가 아니다. 성공update·실제 파라미터 변화 횟수는 AMP skip과 zero LR 등에 따라 계획과 다르다.

`run_config.json`에 branch/commit/TAPT hash, recipe, stage별 trainable/frozen 수, Optimizer 그룹별 크기/LR/initial_lr/decay, epoch별 Sampling hash, loss·원본/Head/Backbone 노출, 계획/시도/성공update·skip, epoch/누적/stage시간, epoch별 peak allocated/reserved 및 전체 최대값을 기록한다. Epoch 끝의 backbone Optimizer state 보유 파라미터 개수도 기록한다. Stage1 memory peak reset은 T4에서만 수행하며 전체 최대는 epoch peak의 max로 저장한다.

Stage 진입은 global_step0/3285/6570이다. Epoch2 직전에 backbone 전체 hash의 초기값 일치, head hash 변화, backbone Adam state 부재 및 Head state·Scheduler 보존을 기록한다. Backbone 변경/state 조기 생성이면 실패한다. 최종 backbone/head 변화와 state 생성 수도 기록한다. Head 변화 여부는 학습 진단이며 AMP skip·zero LR로 변화가 없을 가능성을 숨기지 않는다. Hash는 초기/epoch2경계/종료에만 계산하며 매 step 계산하지 않는다. CPU 전송 비용이 소량 추가된다.

## 통제 조건과 CLI

`--freeze-backbone-epochs`는0/1만 허용하며 기본0은 기존 single-forward Weighted BCE 전체학습 경로다. 옵션0에서는 stage controller·경계 hash를 실행하지 않는다. 로깅만 추가됐으며 기존 loss/forward/LLRD/sampling/AMP 업데이트 순서는 같다.

T4 옵션1에서는 확정 recipe와 모순되는 epochs/batch/accumulation/weight power/count aux/grad checkpointing 등은 학습 전에 거부한다. 원래 Fit 경로·ID hash·TAPT/Tokenizer hash를 localcontrol 영수증과 대조한다. Fit26,280/Dev2,920, Uniform2/epoch, seed42, max512/right/space, 기존 라벨·Dropout·전처리를 유지한다. T1/T3/T5/T8/T10/T9 목적함수·표집·정규화는 가져오지 않았다.

Runner는 기존 output reserve로 기존 폴더를 거부한다. 직접 train CLI도 T4에서는 기존 training 폴더를 거부한다. 모델·Tokenizer의 저장 구조와9출력은 같다. 기존 E001/E007·동일비중 Ensemble·raw0.5/빈예측 허용·Calibration/logit_shift·E026(beta128/tau0.5/min_step5)·공식 inference/score/bundle은 수정하지 않았다.

README의 외부 데이터·Validation 학습·미래정보·전체길이 입력·라벨별 입력전처리 금지 범위를 유지한다. 총길이는 Prefix 표집의 기존 범위와 사후 평가에만 사용하며 모델 특징이나 추론 분기 조건으로 추가하지 않는다. Dev는 평가에만 쓰고 학습에 넣지 않는다. 규정상 별도 명시 승인을 받은 기법이라는 주장은 하지 않는다.

## CPU 검증

최종 전체 Mission4 테스트: **46 passed**. 기존41개와 신규5개다. 기존 정책 테스트는 synthetic 데이터용 CPU 정책 계산이며 실제 RoBERTa나 실제 제출 정책을 새로 추론하지 않는다. 결과는 `t4_cpu_verification/pytest.txt`.

신규 테스트는 경량 synthetic network로 다음을 확인한다.

- 실제 LLRD 함수로 backbone 포함 전체 등록, 중복·누락 없음.
- Stage1 head-only gradient/update, backbone grad/state 없음 및 AdamW decay에도 가중치 불변.
- Stage2 전체 gradient/update와 첫 backbone Adam step=1.
- Head Adam state 객체·exp_avg·step 보존, 동일 Scheduler의 실제3285step 경계 LR 보존, 추가 warmup 없음.
- Dropout0.1과 train mode 유지.
- 실제 학습 main에 synthetic 모델/토크나이저·CPU AMP 대역을 연결해3epoch 전환·forward 횟수·노출/update/skip 및 run_config를 검증.
- Runner CLI의 T4·순차 실행 dispatch를 mock으로 검증. 실제 GPU subprocess는 실행하지 않음.

Synthetic 테스트를 실제 RoBERTa+CUDA 학습 검증으로 해석하지 않는다. 실제 모델 경계는 설치 소스·config·safetensors 헤더로 별도 확인했다.

Sampling은 **과거 Baseline 실제 영수증이 없어 Baseline 커밋의 실제 표집 구문을 CPU에서 재생**한 결과와 T4 현재 표집 구문을 비교했다. Fit 순서와 통화별 길이를 사용해 전체 배열·ID/위치·순서가 동일함을 확인했고 `epoch_1~3.npz`에 양쪽 배열과 통화 ID를 저장했다.

| epoch | 노출 | 계획step | ordered_items_sha256 |
|---|---:|---:|---|
| 1 | 52,560 | 3,285 | dc6bb9a2c78389425f00084ce4e78e4eb9d76934ac93862e952d1c7e5d6712b4 |
| 2 | 52,560 | 3,285 | 81a44e786d0b6275d678a4ea84d8510d2d49cd5fd87de596003a016f3dbb1313 |
| 3 | 52,560 | 3,285 | dca8c766aa61d1bd4c66fd147fcccbb84e5546f0779340ee6b46e73eac86d4df |

`cpu_verification.json`에 결과를 저장했다. 기존 평가 함수 evaluate/write_evaluation/sequential/prepare/aligned는 Baseline과 AST가 동일하다. 최종 source hash·Git status·보호 대상 일치는 `final_validation.json`에 기록한다.

## 미검증·예상 비용

실제 CUDA AMP/skip, 실제 backbone freeze 상태의 GPU 실행, Stage2 실제 update, 성능·일반화·지속누락 개선, GPU peak와 학습시간, 새 checkpoint/bundle 실물 호환성은 아직 GPU에서 검증되지 않았다. CPU 성공이 성능 개선을 뜻하지 않는다.

Baseline 학습37.84분/peak allocated4.66GiB를 참고해 T4 학습30~45분, 배치·순차 평가35~45분 정도를 계획값으로 둔다. 실측이 아니다. Stage1은 backward/Adam 상태가 줄지만 Stage2는 Baseline 전체학습 수준의 메모리가 필요하다. Checkpointing을 활성화하거나 Batch를 줄여 자동 회피하지 않는다. Reserved/다른 프로세스/실행 환경에 따라 OOM·시간 편차가 가능하다.

## 사용자 직접 실행 명령 (이번에는 실행하지 않음)

```powershell
Set-Location D:\assignment\programing\DCC\DCC_Problem\mission4_symptom
$py = (Resolve-Path ..\.venv\Scripts\python.exe).Path
& $py -B scripts/prefix_experiment.py run `
  --freeze-backbone-epochs 1 `
  --reference ../outputs/mission4_prefix/localcontrol/metrics.json `
  --out ../outputs/mission4_prefix/t4_staged_fine_tuning `
  --with-sequential
```

이 한 명령은 T4 학습 → E002 batch 확률 추출 → 기존 고정 Ensemble/E026 평가 → 기존 sequential4-1/4-2 평가를 연결한다. 중간 실패 시 후속 단계를 진행하지 않는다. 출력 폴더가 이미 있으면 덮어쓰지 않고 거부하므로 실패 원인을 확인해야 한다. 기본 Baseline 실행은 freeze 옵션을 생략한다.
