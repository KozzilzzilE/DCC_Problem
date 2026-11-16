# T10 구현 및 CPU 검증 보고서

## 1. 브랜치와 기준 커밋

사용자가 미리 생성한 `mission4-prefix-t10-positive-weight`에서 작업했다.
시작 시 작업 트리는 깨끗했고 HEAD와 `mission4-prefix`는 모두
`1f69317e2cf280729451abdd8576576679b5c606`이었다. 브랜치 생성·전환·초기화는 하지 않았다.
Baseline 브랜치 및 커밋을 변경하지 않았으며 Commit/Push/Merge를 하지 않았다.

## 2. 파일과 변경 이유

| 파일 | 변경 목적 |
|---|---|
| `scripts/prefix_experiment.py` | 기본0.5를 유지하며 공통 실행기에서0.6 전달. Local 기준 비교 및 기존 Sequential 평가 연결 |
| `scripts/train_prefix.py` | Loss·Sampling 변경 없이 Fit 분포·증상 순서·실제 노출·시간·업데이트/Skip 기록 추가 |
| `scripts/verify_t10_cpu.py` | 실제 소스 구문의 CPU 실행, 전체 Sampling 재생, 초기/고정 파일 해시와 통제 코드 검증 |
| `tests/test_positive_weight.py` | 기본값·Loss/Gradient·mean·옵션 전달·실제 CPU AMP Skip 계수·실행 보호 검사 |
| `docs/T10_IMPLEMENTATION.md` | 실제 설정, 검증 범위, 실행 명령과 산출물 설명 |

다른 실험 브랜치의 코드를 가져오지 않았다. T1/T3/T5/T8, 새 라벨·마스킹·증상별 지수는 적용하지 않는다.

## 3. 실제 계산과 증상 매핑

학습기에 기존부터 있던 `--pos-weight-power`를 그대로 사용한다. 기본값은0.5다.
실험 실행기에 같은 이름의 옵션을 연결했으며 선택값은0.5/0.6이다. 공통 RECIPE는 수정하지 않고 복사본에만 명시값을 적용한다.

`pos = labels.sum(axis=0)`

`pos_weight = ((len(labels)-pos)/np.maximum(pos,1)) ** args.pos_weight_power`

모든 Fit 클래스의 양성이0보다 크므로 `(negative_count/positive_count)**power`와 같다.
기존 `BCEWithLogitsLoss(pos_weight=...)`, `mean(Batch×9)`, FP32 Logit 입력, 음성 항을 그대로 유지한다.

| 증상 | Fit 양성 | Fit 음성 | Baseline0.5 | T10 실제0.6 |
|---|---:|---:|---:|---:|
| 고열 | 4686 | 21594 | 2.146671 | 2.501028 |
| 구토 | 4039 | 22241 | 2.346606 | 2.783097 |
| 두통 | 2604 | 23676 | 3.015322 | 3.760109 |
| 복통 | 6062 | 20218 | 1.826254 | 2.060029 |
| 어지러움 | 5717 | 20563 | 1.896528 | 2.155515 |
| 열상 | 3081 | 23199 | 2.744030 | 3.357891 |
| 오심 | 3027 | 23253 | 2.771617 | 3.398443 |
| 전신쇠약 | 4803 | 21477 | 2.114611 | 2.456273 |
| 호흡곤란 | 3542 | 22738 | 2.533680 | 3.051422 |

실제 학습 소스의 계산 구문을 실행해 기존 저장0.5 벡터와 float32 기준 정확히 일치함을 확인했다.
0.6은 사전 분석4자리 표와 절대오차0.00005 이내 일치했다. 위 증상 순서도 명시적으로 검사했다.
같은 출력에서 양성 Loss/Gradient만 w비율로 바뀌고 음성 Loss/Gradient는 동일하다.
학습 뒤 출력·성능 변화는 아직 알 수 없다. 계수가 달라진 학습 Loss 자체를 성능 개선 근거로 비교하지 않는다.

## 4. 변경 변수와 통제 변수

의도된 학습 변수는 **Power0.5→0.6 하나**다. 추가 변경은 실행 연결 및 결과 기록이다.

| 통제 항목 | 확인 결과 |
|---|---|
| Local 레시피 | 실제 localcontrol/training/run_config.json과 공통 레시피 및 세부 설정 일치 |
| 데이터 | Fit26,280, Dev2,920, 교집합 없음. 캐시/ID 기준 SHA256 일치 |
| Sampling | Baseline과 현재 소스의 실제 Sampling 구문을 실행해3 epoch 전체 배열 동일 |
| 노출 | 매 epoch 통화당2개,52,560개. 총157,680개, Batch16, Accumulation1, Effective16, 예정9,855회 |
| 초기 조건 | TAPT 가중치·Tokenizer 및 설정 해시 일치. seed·모델/Head 생성·이동 구문 AST 보존 |
| 입력 처리 | data.py, Tokenizer 생성·right truncation·max512·Padding 구문 유지 |
| 학습 | AdamW/lr5e-5/LLRD0.8/Decay0.01/Linear/Warmup0.1, AMP/Clipping/누적 조건 및 호출 구문 유지 |
| Gradient Checkpointing | 기존 Local과 같이 비활성, 기존 활성화 구문도 유지 |
| 평가 | E001/E007 모델·확률표 해시, Calibration/E026 정책 해시 일치 |
| 공용 코드 | decision/bundle/halting/streaming/simulate/data, 일반 pack_bundle, prefix_probs가 Baseline 소스와 동일 |

과거 Baseline 학습 시 Sampling 영수증은 없으므로 현재 Baseline 소스·동일 데이터의 재생을 비교한 것이다.
전체 Tokenization은 이번에 다시 실행하지 않았다. 입력 처리 동일성은 파일/Tokenizer 해시·설정·구문 비교로 확인했다.
원본 모델 및 보호 결과 파일은 CPU 검증 전후 해시가 같다.

## 5. CPU 검증 및 테스트

`python -B -m pytest -p no:cacheprovider mission4_symptom/tests -q`: 기존41개와 신규9개, **50 passed**.

- 실제 학습 구문에서 기본0.5 가중치 및 기존 Weighted BCE Loss 동일성 통과.
- 0.6 실계산·증상 순서·사전 표 비교 통과.
- 동일 Logit/양성 Label의 원소 Loss와 Gradient가 w비율로 변함을 확인.
- 음성 Label의 Loss/Gradient는 정확히 동일. 혼합 Batch×9 mean Gradient 수식 확인.
- 공통 실행기의0.5/0.6이 실제 train_prefix.py subprocess 인자로 전달되는지 mock 확인.
- Local 참조 해시가 다르면 학습 subprocess 이전에 중단하는지 확인.
- 단일 명령의 run→기존 sequential 연결을 mock 검증.
- 실제 학습 소스의 업데이트 hook을 CPU AdamW/CPU GradScaler에 연결해 정상 갱신과 Inf에 의한 Skip을 검사.
- 실제 소스의 전체 Sampling 배열 대조, 기존 CPU 사전 검증 해시 일치.
- 실제 CLI `run --help`, `train_prefix.py --help`와 가상환경 경로 확인.

검증 산출물은 `outputs/mission4_prefix/t10_cpu_verification/`의
`cpu_verification.json`, `protected_files.json`에 저장했다. 기존 결과를 덮어쓰지 않는다.
CPU 검증을 재실행할 때는 새 출력 폴더 이름을 사용한다.

## 6. 평가 경로 및 기록

학습 Power가 바뀌면 일반 `pack_bundle.py`의 멤버별w 일치 검사와 충돌할 수 있다.
이 검사를 제거하거나 새w의log로 Calibration을 바꾸지 않는다.
기존 `prefix_experiment.py sequential`은 저장 policy.logit_shift를 명시한 settings로 저수준 pack_bundle을 호출한다.
이 **기존 Local 평가 경로 자체를 변경하지 않고** 사용한다. 고정0.5 판정,3개 동일 비중 평균, E026 beta128/tau0.5/min_step5가 유지된다.

Batch evaluate 및 Sequential 구현 함수는 수정하지 않았다. 기존확률·통화별 중단·전체 Sequential 예측 배열을 보존하므로
학습 이후 동일 Baseline 중단 위치의 반사실적 분류 분석도 가능하다. 이번에는 실제 평가를 실행하지 않았다.

추가 학습 기록은 다음과 같다.

- Power·증상 순서·Fit 양성/음성 수·실제w·Loss/reduction.
- 매 epoch Sampling 순서 SHA256·추출수, 실제 소비 샘플 및 양성/음성 노출.
- 기존 누적 history.seconds 보존, 독립 epoch_seconds 추가.
- 예정 갱신수·시도수·실제 AdamW.step 횟수·AMP Skip 및 epoch별 계수.

hook은 실제 optimizer.step이 호출돼야 증가한다. Scheduler는 기존처럼 Skip에도 진행한다. 이 동작을 바꾸지 않았다.
실행 기록 추가의 소규모 CPU 오버헤드는 있을 수 있으나 난수를 소비하거나 Gradient를 바꾸지 않는다.

## 7. 미검증 사항과 위험

- GPU 학습·실제 CUDA AMP/Skip·메모리·완료 시간은 미검증이다. CPU Skip 검사로 CUDA 실동작까지 보장하지 않는다.
- 초기화된 Classification Head Tensor 자체는 생성·대조하지 않았다. 초기 파일과 seed/생성 조건까지만 확인했다.
- 실제 Batch/Sequential 점수·추론 시간·FP/FN 변화·확률 보정 적합성은 사용자 실행 이후 확인한다.
- 고정 E026과의 구현상 연결은 유지되지만 새 확률 분포와의 통계적 적합성은 보장하지 않는다.
- Baseline의 실제 AMP Skip 기록이 없어 미래 실제 업데이트 수까지 동일하다고 보장할 수 없다.
- 전체 저장소 테스트가 아니라 관련 Mission4 전체 테스트를 실행했다.
- 기존 cached policy verify는 최종 run의 선행 검사로 유지했으며, 이번에는 정책의 전체 데이터 평가를 재실행하지 않았다.

## 8. 정확한 PowerShell 실행 명령

아래 명령은 이번에 실행하지 않았다. 학습→교체 멤버 Dev 확률 추출→고정 앙상블 Batch 평가→기존 Sequential4-1/4-2 순서다.
선행 단계 실패 시 후속 단계로 진행하지 않는다. 이미 존재하는 결과 폴더는 거부한다.

```powershell
Set-Location D:\assignment\programing\DCC\DCC_Problem\mission4_symptom
$py = (Resolve-Path ..\.venv\Scripts\python.exe).Path
& $py -B scripts/prefix_experiment.py run `
  --pos-weight-power 0.6 `
  --reference ../outputs/mission4_prefix/localcontrol/metrics.json `
  --out ../outputs/mission4_prefix/t10_positive_weight `
  --with-sequential
```

## 9. 시간 추정과 예상 파일

기존 Local 학습2,270.6초, T1 2,311.7초, T8 2,311.5초를 참고하면
동일 장비·부하에서 T10 학습은 **약38~42분**으로 예상한다. 이번 실측값이 아니며 보장하지 않는다.
Batch 추출·Sequential 평가 시간은 별도다. 조기중단 위치가 변하면 Sequential4-2 총시간도 변할 수 있다.

기준 경로: `outputs/mission4_prefix/t10_positive_weight/`.

| 파일 | 내용 |
|---|---|
| initial.json | 실제Power가 반영된 recipe, 초기/소스/분할 해시 |
| training.log, training/train_log.txt | 기존 학습 로그 및 적용Power |
| training/run_config.json | 실제w·Fit분포·Sampling/노출·epochLoss/시간·성공/시도/Skip 갱신수 |
| training/best_model/ | 기존 형식의 모델·Tokenizer |
| extract.log, dev_prefix.npz, dev_prefix.meta.json | 교체 멤버 확률과 추출 시간/발화 수 |
| metrics.json, halts.npz | Batch4-1/4-2, Macro-F1/Earliness, 구간Score, 증상별Precision/Recall/F1/TP/FP/FN, 평균·중앙 중단, Local기준 차이 |
| sequential/fit_only.pt | 기존 방식의고정정책 번들 |
| sequential/sequential_timing.json | 실제Sequential 점수·세부지표·구간Score·추론/로딩/발화당 시간·모델 해시 |
| sequential/sequential_4-1.npz, sequential/sequential_4-2.npz | 실제 순차 예측·중단 배열 |

Batch 추출 시간은 교체 멤버의 Tokenization/추론이며 고정멤버 재추론·모델로딩을 포함하지 않는다.
발화당 Batch 추출 시간은 meta의 seconds/n_moments로 확인할 수 있다. 기존3멤버 Sequential 실측 시간과 구분한다.
결과는 기존 Git무시 대상이다. 현재 변경은 위5파일에 한정되며 모두 미커밋이다.
