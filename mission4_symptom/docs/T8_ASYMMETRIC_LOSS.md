# T8 구현 및 학습 전 CPU 검증 보고서

작성일: 2026-11-16, 한국 시간. GPU 학습 및 실제 분류 모델 전체 추론은 실행하지 않았다.

## 1. 현재 브랜치 및 Baseline 분기 상태

- 현재 브랜치: `mission4-prefix-t8-asymmetric-loss`.
- 작업 시작 시 `HEAD`와 `mission4-prefix`는 모두 `1f69317`이며 양쪽 추가 커밋 수는 `0 / 0`이었다.
- 작업 시작 시 변경 파일은 없었으며 적용할 `AGENTS.md`도 발견되지 않았다.
- 현재 브랜치의 작업 파일만 수정했다. Baseline 브랜치 수정, Commit, Push, Merge는 하지 않았다.
- T1/T3/T5 학습 전략은 가져오지 않았다. 기존 공통 실행기 및 평가 경로를 재사용했다.

## 2. 변경·추가 파일과 목적

| 파일 | 목적 |
|---|---|
| `m4/losses.py` | FP32 ASL 및 기존 Weighted BCE를 선택하는 factory 추가 |
| `scripts/train_prefix.py` | 기본 BCE를 유지하는 `--loss` 옵션, 실제 Loss 설정·Sampling 해시·학습 통계·업데이트/Skip 기록 |
| `scripts/prefix_experiment.py` | ASL 전달, Local Baseline 비교, 학습→Batch→Sequential 연결, 추출 시간/발화당 시간 기록 |
| `scripts/verify_t8_cpu.py` | Baseline Git 소스와 전체 Fit Sampling/Tokenization을 CPU에서 재생·대조 |
| `tests/test_asymmetric_loss.py` | 수식·기울기·detach·mean·수치 안정성·Accumulation·BCE 유지 검증 |
| `tests/test_prefix_experiment.py` | ASL 인자 전달과 기존 레시피 유지, Sequential 연결의 mock 검증 |
| `docs/PREFIX_EXPERIMENT_RUNNER.md` | T8 구현 안내와 기본 BCE 유지 명시 |
| `docs/T8_ASYMMETRIC_LOSS.md` | 설정, 검증 근거, 제한, 실행 명령 및 저장 구조 설명 |

## 3. ASL 실제 구현 방식

[Alibaba-MIIL 공식 ASL](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py)의 확률 clipping 및 asymmetric focusing 수식을 참고했다.
공식 구현의 sum 대신 기존 Baseline과 동일한 mean을 적용했다.

- `gamma_pos=0`, `gamma_neg=4`, `clip=0.05`, `eps=1e-8`.
- `p = sigmoid(logit)`, `q = min(1, 1-p+0.05)`.
- 양성: `L+ = -log(max(p, eps))`.
- 음성: `L- = -detach((1-q)^4) * log(max(q, eps))`.
- `torch.no_grad()` 안에서 Focusing Weight를 계산한다. 외부의 Grad 활성 상태는 보존한다.
- 최종 반환값은 모든 원소의 평균이다. 증상 출력이 9개인 T8에서는 `sum(loss)/(Batch Size × 9)`다.
- 외부 AMP 문맥을 Loss 내부에서 비활성화하고 Logit/Target을 FP32로 변환한다. FP16 입력에서도 eps가 0으로 소실되지 않는다.
- 모델의 CUDA FP16 autocast, `loss/grad_accum`, GradScaler, Gradient Clipping은 기존 구문을 유지한다.

## 4. 기존 Weighted BCE와의 차이

기본값은 `--loss weighted-bce`다. 기존 `BCEWithLogitsLoss(pos_weight=...)`와 mean을 그대로 사용한다.
`--loss asl`일 때만 학습 Loss에서 Pos Weight를 무시한다. Pos Weight 계산 및 기록 자체는 유지한다.
E026의 `logit_shift`, 보정용 Pos Weight 관련 값, 평가 Threshold 및 정책은 그대로다.

이번 실험은 **Weighted BCE를 ASL로 교체하는 Loss 전략 비교**다.
Pos Weight 제거, 음성 clipping, asymmetric focusing의 개별 기여도를 분리하는 실험이 아니다.
ASL이 FP를 직접 줄이거나 성능을 반드시 개선한다는 전제를 두지 않는다.
두 Loss의 척도가 다르므로 학습 Loss 수치 자체를 직접 비교해 성능 개선을 판단하지 않는다.

## 5. 통제 변수 동일성 검증

실제 Local Baseline의 `outputs/mission4_prefix/localcontrol/initial.json`과
`training/run_config.json`, Baseline Git 소스, 현재 입력 파일을 대조했다.

| 통제 변수 | 확인 결과 및 범위 |
|---|---|
| Fit/Dev 분할·데이터 | 캐시 및 ID 파일 SHA256 일치. Fit 26,280 / Dev 2,920, 중복·교집합 검사 통과 |
| 데이터 순서 | 동일 `load_cache` 소스와 입력, 실제 전체 Sampling 배열 순서 일치 |
| TAPT/Tokenizer | 모델·Tokenizer·설정 파일의 기준 해시 및 Local 초기 기록 일치 |
| 모델 구조·초기화 | 모델 생성 구문·seed 설정 AST 동일. 실제 모델 초기 Tensor 비교는 미실행 |
| Prefix | 기존 Uniform Sampling 유지. epoch별 통화당 2개, 실제 선택 길이 및 shuffle 순서 일치 |
| 학습량 | epoch당 52,560개, 3 epoch 총 157,680개. Batch 16, Accumulation 1, 예정 업데이트 9,855회 |
| Optimizer·LR | AdamW, lr `5e-5`, LLRD `0.8`, Weight Decay `0.01` 구문 동일 |
| Scheduler | Linear, Warmup Ratio `0.1`(예정 985회), 기존 갱신 구문 동일 |
| 토큰 | max length 512, right truncation, space 형식. 실제 전체 토큰 ID·mask·padding·절단 통계 일치 |
| Mixed Precision | 기존 CUDA FP16 autocast/GradScaler 구문 동일. Loss만 내부 FP32 |
| Gradient Checkpointing | Local Baseline과 같이 기본 비활성. 옵션 및 활성화 구문 유지 |
| E001/E007 | 고정 모델 경로 유지, 고정 확률표 SHA256 일치. 모델 파일은 이번 작업에서 수정하지 않음 |
| 앙상블·판정 | 기존 3개 멤버 동일 비중 평균, Threshold 0.5, force_top1=False 유지 |
| Calibration·E026 | 정책 전체 해시 및 보정값 유지. beta 128 / tau 0.5 / min_step 5 유지 |
| 추론 정보 제한 | Streaming/Halting 소스 동일. 기존 온라인 특징·미래 정보 차단 테스트 통과 |

전체 Tokenization을 Baseline과 T8에서 각각 수행하여 다음 수치 및 토큰 ID/attention mask 해시를 대조했다.

| Epoch | 샘플 수 | Padding 제외 토큰 | Padding 토큰 | 절단 샘플 수 | 제거 토큰 수 |
|---|---:|---:|---:|---:|---:|
| 1 | 52,560 | 6,816,329 | 10,934,631 | 180 | 9,992 |
| 2 | 52,560 | 6,776,964 | 10,900,732 | 190 | 11,416 |
| 3 | 52,560 | 6,792,082 | 10,896,526 | 189 | 10,271 |
| 합계 | 157,680 | 20,385,375 | 32,731,889 | 559 | 31,679 |

절단은 원래 특수 토큰 포함 길이가 **512를 초과**한 경우다. 길이가 정확히 512인 입력은 절단으로 세지 않았다.
epoch별 양성 라벨 노출은 증상 순서대로
`[9372, 8078, 5208, 12124, 11434, 6162, 6054, 9606, 7084]`이며 두 경로가 일치했다.
음성 노출은 각 epoch의 52,560에서 해당 양성 노출을 뺀 값이므로 역시 일치한다.

이 결과는 **현재 동일 입력으로 Baseline 소스와 T8 소스를 재생한 검증**이다.
과거 Local 학습 당시의 Sampling/Tokenization 영수증이 없으므로 과거 실행 배열을 직접 대조한 것은 아니다.

## 6. CPU 검증 및 테스트 결과

`python -B -m pytest -p no:cacheprovider mission4_symptom/tests -q`: **66 passed**.
기존 Mission4 테스트 41개와 T8 추가 테스트 25개를 포함한다.

- 대표 확률별 양성·음성 Loss와 detach된 Logit Gradient가 독립적인 스칼라 수식과 일치했다.
- gamma_pos=0에서 양성 Loss/Gradient는 eps clamp가 작동하지 않는 범위의 비가중 BCE와 일치했다.
- `p <= clip`인 음성 Loss/Gradient가 0이었다.
- Focusing Weight 미분 포함 시의 Gradient와 다르고, 의도한 detach Gradient와 일치했다.
- Batch×9 mean, 혼합 Multi-label Batch, 동일 크기 microbatch Gradient Accumulation을 검증했다.
- FP16/BF16/FP32 입력 및 CPU BF16 autocast에서 극단 Logit ±10,000까지 Loss/Gradient가 유한했다.
- gamma_pos=gamma_neg=0, clip=0에서 혼합 Batch의 비가중 BCE Loss/Gradient와 일치했다(Logit -8~8).
- 기본 Weighted BCE의 Pos Weight 적용, ASL의 Pos Weight 비적용, CLI 기본값 유지가 통과했다.
- 기존 순차 평가 호출은 mock으로 검증했으며 실제 분류 모델은 실행하지 않았다.

아래는 1개 원소 기준이다. Batch mean에서 각 원소의 Gradient는 전체 원소 수로 나뉜다.

| p | 양성 Loss | 양성 dL/dz | 음성 Loss | 음성 dL/dz |
|---|---:|---:|---:|---:|
| 0.01 | 4.60517025 | -0.99000001 | 0 | 0 |
| 0.10 | 2.30258512 | -0.90000004 | 0.000000320583 | 0.000000592106 |
| 0.50 | 0.69314718 | -0.50000000 | 0.02451505 | 0.01863920 |
| 0.90 | 0.10536054 | -0.10000002 | 0.99030823 | 0.31320372 |
| 0.99 | 0.01005027 | -0.00999993 | 2.19656897 | 0.12882288 |

사전 분석의 원본 수치표는 현재 대화/저장소에서 확보하지 못했다. 따라서 그 표와의 직접 대조 완료로 보고하지 않는다.
위 수치는 지정된 수식에서 독립적으로 재계산하고 구현 결과와 비교한 것이다.

기존 확률표의 E026 CPU 검산 및 Dev 2,920통화 온라인 특징 재생도 통과했다.
기준 4-1 Score `0.5060006657883154`, 4-2 Score `0.5031464178054406`,
Macro-F1 `0.6063538200286547`, Earliness `0.8297901343833591`를 재현했다.
전체 통화에서 현재 Batch/온라인 재생의 중단 위치가 일치했다. 이 값들은 T8 모델 결과가 아니다.

검증 파일:

- `outputs/mission4_prefix/t8_cpu_verified/cpu_verification.json`
- 같은 폴더의 `epoch_1.json`, `epoch_2.json`, `epoch_3.json`, `asl_examples.json`
- `outputs/mission4_prefix/t8_policy_verified/metrics.json`, `halts.npz`
- `outputs/mission4_prefix/t8_cpu_verified/pytest.txt`, `source_receipt.json`

CPU 검증을 재실행하려면 저장소 루트에서 아래 명령을 사용한다. 이미 존재하는 출력 경로는 거부하므로 새 이름을 사용한다.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider mission4_symptom/tests -q
.\.venv\Scripts\python.exe -B mission4_symptom/scripts/verify_t8_cpu.py --out outputs/mission4_prefix/t8_cpu_recheck
```

## 7. 발견한 문제와 해결 과정

- Git이 Sandbox 사용자와 저장소 소유자 차이로 읽기를 거부했다. 명령별 `-c safe.directory=...`를 적용했으며 전역 Git 설정은 바꾸지 않았다.
- CPU 사전 검증기의 Gradient Clipping AST 검색이 함수의 전체 경로를 누락하여 첫 검증이 실패했다.
  검색을 `torch.nn.utils.clip_grad_norm_`로 수정한 뒤 전체 검증을 통과했다. 학습 코드의 Clipping은 변경하지 않았다.
  첫 실패 기록은 `outputs/mission4_prefix/t8_cpu_verification/failed_checks.json`에 보존했다.
- 공식 ASL의 전역 Grad 상태 변경 대신 지역 `torch.no_grad()`를 사용하여 외부 Grad 상태를 보존했다.
- FP16에서 eps/log 계산이 불안정해지지 않도록 Loss 내부 FP32 연산을 적용했다.
- 기존 `history.seconds`는 누적 시간이었다. 호환성을 위해 유지하고 `epoch_seconds`를 추가했다.
- 기존 `total_steps`는 microbatch 수다. 이를 유지하고 `planned_optimizer_updates`, `optimizer_updates`, `amp_skips`를 별도 기록했다.
- 기존 Scheduler는 AMP Skip 시에도 진행한다. 통제 변수 유지를 위해 이 동작을 변경하지 않았다.

## 8. 미검증 항목과 남은 위험

- GPU 학습, CUDA FP16 autocast/GradScaler의 실제 동작, 실제 업데이트/AMP Skip 수는 사용자 실행 전이라 미검증이다.
- 실제 모델의 초기 Tensor 동일성, CUDA 비결정성, 학습 완료/메모리/시간은 검증하지 않았다.
  초기 가중치 파일과 생성 조건의 동일성까지만 확인했다.
- 실제 T8 Batch/Sequential Score, Macro-F1, Earliness, FP/FN 및 추론 시간은 미측정이다.
- E001/E007의 모델 파일은 변경하지 않고 기존 경로를 유지한다. 별도 과거 모델 해시 영수증을 통한 출처의 독립 감사는 미완료다.
- 사전 분석 원본 수치표 및 과거 학습 Sampling/토큰 영수증은 확보하지 못했다.
- 이번 테스트 범위는 기존 Mission4 전체와 T8 추가 테스트다. 관련 없는 Mission1/3 등 저장소 전체 테스트는 실행하지 않았다.
- 공식 ASL처럼 eps clamp 및 FP32 sigmoid 포화 영역에서는 Gradient가 0이 될 수 있다.
  따라서 극단 Logit까지 BCEWithLogitsLoss와 완전히 같은 값/Gradient를 보장하지 않는다. 유한성은 별도 검증했다.
- 기록용 정수 집계 및 GradScaler scale 조회가 추가되므로 학습 시간에 일부 측정 오버헤드가 있다.
- Batch 확률 추출과 FP16 Sequential 번들은 수치 차이가 발생할 수 있다. 기존 평가 경로를 유지하고 결과를 각각 기록한다.
- ASL 학습에 Pos Weight를 쓰지 않더라도 E026 보정을 그대로 유지하는 것은 이번 통제 실험의 의도다. 그 보정의 T8 적합성은 보장하지 않는다.

## 9. GPU 학습 및 Sequential 평가 일괄 실행 명령

아래 명령은 **이번 작업에서 실행하지 않았다**. 사용자가 구현 결과를 검토한 뒤 직접 실행한다.
기존 팀 기준 검산 → T8 학습 → Dev 확률 추출 → 고정 E001/E007 앙상블 Batch 4-1/4-2 →
Sequential 4-1/4-2를 한 프로세스로 연결한다. 중간 단계 실패 시 후속 단계로 진행하지 않는다.

```powershell
Set-Location D:\assignment\programing\DCC\DCC_Problem\mission4_symptom
$py = (Resolve-Path ..\.venv\Scripts\python.exe).Path
& $py -B scripts/prefix_experiment.py run `
  --loss asl `
  --reference ../outputs/mission4_prefix/localcontrol/metrics.json `
  --out ../outputs/mission4_prefix/t8_asymmetric_loss `
  --with-sequential
```

3 epoch, seed 42, Uniform 통화당 2개, Batch 16, Accumulation 1 등 기존 레시피를 자동 전달한다.
추가 하이퍼파라미터 탐색을 제공하지 않는다. 출력 폴더가 이미 있으면 새 실험 폴더 이름을 사용한다.

## 10. 예상 결과 경로 및 파일

기준 경로: `outputs/mission4_prefix/t8_asymmetric_loss/`.

| 파일 | 내용 |
|---|---|
| `initial.json` | Loss 이름, 기존 레시피, 초기 가중치·Tokenizer·소스·분할 해시 |
| `training.log`, `training/train_log.txt` | 실제 설정 및 학습 진행 기록 |
| `training/run_config.json` | ASL 실제 설정, Loss Pos Weight 미적용, Pos Weight 원래 값, epoch Loss·시간·샘플·라벨/토큰/Padding 통계, Sampling 해시, 예정/실제 업데이트·AMP Skip |
| `training/best_model/` | 기존 형식의 모델 및 Tokenizer |
| `extract.log`, `dev_prefix.npz`, `dev_prefix.meta.json` | 교체 멤버의 Dev 확률표와 Batch 추출 시간 |
| `metrics.json`, `halts.npz` | Batch 4-1/4-2, 증상별 Precision/Recall/F1/TP/FP/FN/TN, 구간별 Score, Macro-F1/Earliness, 평균·중앙값 중단 위치, Local 기준 차이, 고정 정책·보정·해시 |
| `sequential/fit_only.pt` | 기존 조립 방식의 Fit-only 앙상블 번들 |
| `sequential/sequential_timing.json` | Sequential 4-1/4-2 지표, 증상별 세부 결과, 구간별 Score, 로딩·추론·발화당 시간, 중단 위치 |
| `sequential/sequential_4-1.npz`, `sequential/sequential_4-2.npz` | 실제 순차 예측 및 중단 배열 |

Batch 추출 시간은 **교체 멤버**의 Tokenization 및 추론 시간이다. 고정 E001/E007 재추론이나 모델 로딩은 포함하지 않는다.
`metrics.json`에 기존 추출 영수증 및 발화당 추출 시간을 기록하며, 실제 3개 모델 순차 추론 시간과 구분한다.
전체 학습 전 절단 통계는 CPU 검증 파일에 보존한다. 학습 시 Sampling 해시와 토큰/Padding 통계를 대조할 수 있다.

## 11. Git 변경 상태

기존 파일 4개 수정, 신규 파일 4개 추가이며 모두 미커밋이다. 목록은 2번과 같다.
검증 산출물은 기존 `.gitignore`에 따라 `outputs/` 아래 Git 무시 대상이다.
Baseline 및 현재 브랜치의 커밋 위치는 변경하지 않았다. Commit/Push/Merge는 실행하지 않았다.
