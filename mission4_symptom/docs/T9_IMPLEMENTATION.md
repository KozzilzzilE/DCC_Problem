# T9 Same-Prefix Consistency Learning 구현 및 CPU 검증

## 1. 브랜치 및 실험 가설

사용자가 생성한 `mission4-prefix-t9-same-prefix-consistency`에서 작업했다. 시작 작업 트리는 깨끗했으며
HEAD와 Baseline `mission4-prefix`는 모두 `1f69317e2cf280729451abdd8576576679b5c606`이었다.
브랜치 생성·전환·초기화 및 Commit/Push/Merge는 하지 않았다. 다른 실험 브랜치의 코드를 병합하지 않았다.

사전 보고서 `outputs/mission4_prefix/t9_preanalysis/T9_사전검토_보고서.md`의 단일 설정을 구현했다.
가설은 동일 Prefix의 독립 Dropout 출력 사이를 정규화하면 일반화와 분류 성능에 도움이 될 수 있다는 것이다.
서로 다른 발화 시점의 예측을 같게 유지하는 시간적 Consistency는 적용하지 않는다.

## 2. 수정·추가 파일

| 파일 | 변경 범위 |
|---|---|
| `m4/consistency.py` | FP32 Bernoulli 대칭 KL, 동일입력 목적함수, 확정 T9 설정 검사 |
| `scripts/train_prefix.py` | 기본 비활성 옵션, 두 view 목적함수 연결, 명시적 checkpoint 설정, 학습 통계 |
| `scripts/prefix_experiment.py` | 옵션 전달, Local 비교 기준, 기존 Sequential 평가 호출 연결 |
| `tests/test_consistency.py` | 실제 함수·실제 checkpoint 활성 구문·Dummy CPU 학습 및 RNG/Gradient 검사 |
| `scripts/verify_t9_cpu.py` | 전체 Sampling 재생, Fit w, Local 레시피, 불변 소스·모델·정책 해시 검증 |
| `docs/T9_IMPLEMENTATION.md` | 설정·검증·실행 명령·한계 문서 |

공식 score.py/inference.py 및 m4의 data/decision/halting/streaming/bundle/simulate,
일반 pack_bundle.py, prefix_probs.py는 수정하지 않았다. 평가 함수 evaluate/write_evaluation/sequential/prepare/aligned도 AST상 동일하다.

## 3. 실제 Loss와 학습 흐름

`--consistency-lambda` 기본값은0.0이다. 기본 또는0이면 기존과 같이 한 번만 forward하고 Weighted BCE를 반환한다.
1.0이면 같은 enc의 input_ids·attention_mask Tensor를 순차적으로 두 번 전달한다. 정답 target도 같다.
forward 사이 seed 재설정·RNG 복원은 없다. model.train() 아래에서 Dropout mask를 독립 표집한다.
우연히 같은 mask가 나올 가능성까지 배제하는 것은 아니며, 억지로 mask를 고르거나 재사용하지 않는다.

`L_supervised = (WeightedBCE(z1,y) + WeightedBCE(z2,y))/2`

`L_raw = mean(0.5*(sigmoid(z1)-sigmoid(z2))*(z1-z2))`

**`L_total = L_supervised + 1.0*L_raw`**

- BCE는 기존 Fit 기반 `(negative/positive)^0.5`, Batch×9 mean이다. 가중치·공식·마스킹·분모를 바꾸지 않았다.
- KL은 각 증상 Bernoulli 양방향 KL의 평균과 같은 Logit 동치식이다. Softmax나 확률 log를 사용하지 않는다.
- KL에 Pos Weight를 추가하지 않는다. 양쪽 출력 모두 Gradient를 받으며 detach는 없다.
- 모델 forward는 기존 CUDA FP16 autocast다. Loss에서는 autocast를 끄고 Logit을 FP32로 바꾼 뒤 sigmoid·차이·곱을 계산한다.
- `loss/grad_accum`은 기존 backward 위치에서 정확히 한 번만 적용한다. 두 view별 step 또는 별도 backward를 하지 않는다.
- 기존 unscale→clip_norm1.0→scaler.step→scaler.update→scheduler.step→zero_grad 순서를 유지했다.
- T9에서 Power0.6/보조출력/다른Batch·샘플량 등 확정 조건과 다른 설정은 학습 전 오류로 거부한다.

lambda1.0은 첫 탐색 실험 계수이며 최적값이 아니다. Loss 크기 비율을 Gradient 크기 비율로 해석하지 않는다.

## 4. Gradient Checkpointing

`--gradient-checkpointing`은 기존 `--grad-ckpt`의 별칭이다. 기본 비활성 상태는 그대로다.
T9 활성 시 이 옵션을 필수로 요구하고 아래 kwargs를 명시한다.

`gradient_checkpointing_kwargs={"use_reentrant":False,"preserve_rng_state":True}`

lambda0인 기존 checkpoint 경로는 원래의 `model.gradient_checkpointing_enable()` 호출을 유지했다.
각 원래 forward는 서로 다른 RNG 진행 상태를 사용한다. backward의 재계산만 각 view의 원래 RNG 상태를 재현한다.
이는 두 원래 forward 사이 RNG를 복원하는 것과 다르다.

두 그래프를 함께 보유하므로 checkpoint 활성화로 메모리 부담을 줄인다. OOM을 잡아 Batch를 내리는 처리나 재시도는 없다.
OOM은 오류로 전파되고 training.log를 확인해야 한다. 공통 실행기는 학습 subprocess 실패 시 평가를 진행하지 않는다.

## 5. 동일 조건과 달라진 조건

| 항목 | 설정·검증 |
|---|---|
| Fit/Dev | 기존26,280/2,920, ID 교집합 없음, 캐시/ID 해시 일치 |
| 초기화 | 기존 TAPT/Tokenizer 해시·seed42·Head 생성 방식·모델 이동 구문 동일 |
| Sampling | Uniform, epoch당 통화별 독립2개, include_full=False,3 epoch. 실제 소스 재생 배열 동일 |
| 원본 노출 | 157,680개. 중복 Prefix를 포함한 노출 수이며 고유 Prefix 개수라고 부르지 않음 |
| view 노출 | **315,360개**. checkpoint 내부 재계산은 view 노출에 포함하지 않음 |
| Batch | 물리16/forward, Accumulation1, 독립 원본 입력 기준 Effective16 |
| 계획 갱신 | 9,855회. view 수 때문에 두 배로 늘리지 않음 |
| 학습 | lr5e-5/LLRD0.8/AdamW/Decay0.01/Linear/Warmup0.1/AMP/Clipping 유지 |
| 입력 | space/max512/right truncation/배치 Padding 구문 및 data.py 동일 |
| Loss | 기존 BCE1회에서 두 stochastic BCE 평균+KL로 변경 |
| Checkpointing | Baseline 비활성에서 **T9 활성**으로 변경 |
| 평가 | E001/E007·동일 비중 앙상블·0.5 판정·Calibration/E026 고정, 기존 함수 유지 |

원본 NumPy Sampling RNG와 모델 Dropout Torch RNG는 분리돼 있다. 추가 forward가 원본 Prefix 선택을 바꾸지 않는다.
다만 Torch 난수 소비는 늘어나므로 이후 Dropout 경로까지 Baseline과 같다고 주장하지 않는다.

## 6. 기록과 산출물

run_config.json은 기존 필드를 유지하고 다음을 추가한다.

- 활성 여부/lambda/Loss 수식·dtype·mean·양방향 Gradient, Fit 분포와 실제 Pos Weight·증상 순서.
- checkpoint 활성/non-reentrant/RNG 보존, Physical/Effective Batch 및 기존 args.
- epoch별 supervised_bce/raw_consistency/weighted_consistency/train_loss,
  consistency_to_supervised_ratio, epoch_seconds 및 기존 누적 seconds.
- epoch/전체 원본 sample 및 stochastic view 노출, Sampling 해시·추출수.
- 예정 step·시도 step·실제 optimizer.step 수·AMP Skip.
- 기존 peak_cuda_memory_bytes(allocated), 추가 peak_cuda_reserved_memory_bytes.

일정 간격 로그에도 supervised/raw/weighted Consistency와 Total Loss가 함께 나온다.
실제 갱신은 AdamW step post-hook으로 센다. GradScaler가 step을 생략하면 증가하지 않는다.
Scheduler가 Skip에도 진행하는 기존 동작은 변경하지 않았다. 고비용 Gradient별 측정은 추가하지 않았다.

출력 기준: `outputs/mission4_prefix/t9_same_prefix_consistency/`.

| 경로 | 내용 |
|---|---|
| initial.json | recipe·lambda·checkpoint 옵션·초기 및 소스 해시 |
| training.log / training/train_log.txt | 적용 설정 및 학습 Loss 구성요소 로그 |
| training/run_config.json | 설정·실제 가중치·Loss/노출/시간/업데이트·Skip 기록 |
| training/best_model/ | 기존 모델·Tokenizer 형식 |
| dev_prefix.npz / dev_prefix.meta.json / extract.log | 기존 Batch 확률 추출 결과 |
| metrics.json / halts.npz | 기존 고정 정책 Batch 평가와 Local 기준 차이 |
| sequential/fit_only.pt | 기존 형식의 고정 평가 Bundle |
| sequential/sequential_timing.json | 기존 실제 Sequential 지표·세부 결과·시간 |
| sequential/sequential_4-1.npz / sequential_4-2.npz | 기존 순차 예측·중단 배열 |

추론은 기존 eval모드의1회/멤버다. Consistency나 추가 Dropout 추론을 넣지 않는다.
학습 helper는 파라미터를 추가하지 않으며 모델 출력과 저장 파라미터 구조도 바꾸지 않는다.
실제 RoBERTa 저장·Bundle 생성은 이번에 실행하지 않았으므로 구조적 호환성 검토와 Dummy state_dict 검증 범위다.

## 7. CPU 검증

관련 Mission4 전체 테스트: **57 passed**(기존41개, 신규16개).

- Log-sigmoid Bernoulli KL 참조식과 실제 FP32 동치식의 값·양방향 Gradient 일치.
- 실제 FP32 함수의 유한차분 검사, 동일 Logit에서0, 대칭성·대표 차이의 증가·mean 불변성.
- FP16/BF16/FP32 가상 입력, CPU autocast 아래 FP32 Loss, 극단±10,000 Logit의 유한 Loss/Gradient.
- 실제 목적함수에 연결한 Dummy에서 lambda0/1의 forward1회/2회, 동일 입력 Tensor·target 확인.
- Dummy train모드의 서로 다른 Dropout mask, 한 합성 backward와 CPU GradScaler·AdamW 단일 step.
- checkpoint의 역전파 재계산 mask가 각 원래 view와 같고, 비checkpoint 계산과 Loss/Gradient 일치.
- 실제 학습기의 checkpoint 활성 구문을 추출해 T9 명시 kwargs와 기존 비활성/기본 활성 경로를 검사.
- 누적 Loss Scaling1회, Dummy state_dict 키 동일, 실행 옵션 전달·기존 Sequential 연결 mock 검증.
- 전체3 epoch Sampling을 Baseline 및 T9 실제 소스 구문으로 재생해 배열·라벨 노출·기존 해시 일치.
- 실제 Fit 가중치 구문을 실행해 Local Baseline float32 Pos Weight와 정확히 일치.
- 초기·고정 모델·정책·데이터 해시와 불변 소스 검증. Local Baseline 레시피와 충돌 없음.

첫 테스트에서는 불완전 설정 입력에 AttributeError가 났다. 설정 검사에서 누락값도 명시적 ValueError로 다루도록 고친 뒤 통과했다.
Loss·checkpoint·RNG 테스트 실패는 없었다. GPU 테스트로 확대 해석하지 않는다.

CPU 산출물: `outputs/mission4_prefix/t9_cpu_verification/cpu_verification.json`,
`protected_files.json`, `pytest.txt`, `final_validation.json`.
전체 Tokenization은 재실행하지 않았으며 파일/설정/입력 구문 동일성을 확인했다.
과거 Baseline 실제 Sampling 영수증 대신 동일 데이터로 Baseline 소스를 재생한 비교다.

## 8. GPU 미검증 사항 및 해석의 한계

1. Dropout 불일치가 기존 오류의 주요 원인이라는 직접 근거는 없다. 실제 모델의 두 train출력 편차도 미측정이다.
2. 지속 FN·증상 근거 부족·라벨 오류를 직접 해결하지 않는다.
3. 두 stochastic BCE 평균 자체도 단일 BCE와 다른 Gradient 잡음 효과를 만들 수 있다.
4. Gradient Checkpointing 활성도 Baseline과 다른 실행 조건이다.
5. 따라서 결과는 두 view 평균+KL+checkpoint의 묶음 비교이며 KL 단독 효과로 해석할 수 없다.
6. GPU 메모리 및 **90~110분** 예상 학습 시간은 사전 추정이며 실측값이 아니다. 8GB에서 OOM 가능성은 남는다.
7. CUDA FP16 AMP·실제 RoBERTa checkpoint 재계산·실제 Skip·학습 완료·실제 초기 Tensor 동일성은 미검증이다.
8. 기존 Calibration/E026과 구현상 연결돼도 새 확률 분포의 적합성은 보장하지 않는다.

발견된 CPU 차단 문제는 해결됐다. GPU에서만 확인 가능한 위 항목은 남아 있다.
메모리 부족 시 자동 Batch/Accumulation 변경 없이 실패를 확인해야 한다.

## 9. 정확한 PowerShell 실행 명령

실제 가상환경과 CLI help를 확인했다. 아래 명령은 이번에 실행하지 않았다.
학습→Dev 확률 추출→기존 고정 Batch 평가→기존 Sequential4-1/4-2를 연결한다.
이미 존재하는 실험 출력 폴더는 거부하므로 기존 결과를 덮어쓰지 않는다.

```powershell
Set-Location D:\assignment\programing\DCC\DCC_Problem\mission4_symptom
$py = (Resolve-Path ..\.venv\Scripts\python.exe).Path
& $py -B scripts/prefix_experiment.py run `
  --consistency-lambda 1.0 `
  --gradient-checkpointing `
  --reference ../outputs/mission4_prefix/localcontrol/metrics.json `
  --out ../outputs/mission4_prefix/t9_same_prefix_consistency `
  --with-sequential
```

Baseline 실행은 두 실험 옵션을 생략하면 된다. lambda0에서도 불필요한 두 번째 forward는 없다.
이번 작업은 구현·CPU 검증까지만 완료했으며 실제 GPU 학습과 전체 모델 추론·평가는 하지 않았다.
