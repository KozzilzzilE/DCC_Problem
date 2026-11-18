# Mission 4 실험 목록·판단·근거

요약 갱신: 2026-11-14. 시작 문서는 [에이전트 지침](../AGENTS.md)과 [기록·보존 기준](EXPERIMENT_RECORDING_STANDARD.md)이다.

이 목록은 현재 작업 폴더에 있는 Mission 4 자료의 연결표다. 예선 Mission 1~3 전체 산출물을 이 폴더에서 검증했다는 뜻은 아니다. 가중치 실물 확인과 보고서의 저장 경로 표기를 구분한다.

## 실험이 이어진 흐름

`데이터 특징·문맥 검토 → 전체/누적 학습 → 발화 경계·화자 표시 → 균등 시점 비교 → AI 문맥 정답 진단·보조 학습·미확인 강도 비교 → 10,000통화 확대 → 같은 자료 추가1epoch → 오답 분석 → 전체·국소 문맥 결합 → 출력 보정·앙상블·4-2 중단 → 22,509통화 확장 및 새 확인셋 평가`

쉽게 읽는 설명은 [실험 과정과 이유 요약](MISSION4_EXPERIMENT_STORY_20261114.md)에 정리한다. 아래 첫 표와 ‘당시 다음 담당자’는11월12일 시점의 기록을 보존한 것이다. 그 안의 ‘미실행·현재·대기’ 표현을 최신 상태로 해석하지 말고, 이어지는11월13~14일 완료·감사 항목을 함께 확인한다.

| ID/자료 | 왜 했는가 | 확인한 결과·판단 | 근거 자료 | 가중치/예측 보존 확인 |
|---|---|---|---|---|
| 데이터 분석·문맥 검토 | 짧은 응답·질문·다중 증상 등 입력 특성에서 실험 근거 찾기 | 자동 통계와 의미 검토 범위를 나눠 해석; 관찰을 곧바로 개선 효과로 단정하지 않음 | [자료 색인](MISSION4_DATA_ANALYSIS_INDEX_20261105.md), [데이터 중심 분석](mission4_data_led_discovery_20261105/HYPOTHESIS_ANSWERS_DATA_LED.md), [문맥 검토](mission4_context_review_20261106/CONTEXT_REVIEW_RESULT.md) | 분석 CSV/JSON/검토 기록 로컬 존재. 모델 학습 산출물과 별개 |
| B0/B1 파일럿 | 통화 전체만 학습하는 것보다 누적 시점도 학습하면 나은가 | 내부 Dev 512통화, 4-1 B0 0.3914 → B1 0.4240. B1을 후속 비교 기준으로 사용. 표본·단일 seed 한계 | [실험 보고서](mission4_baseline_strategy_20261108/MISSION4_B0_B1_PILOT_REPORT.md) | B0 예측·양쪽 가중치는 Drive 원격 재확인 대기. B1 확률은 후속 대조 폴더에 로컬 복원본 존재 |
| C2 파일럿 | 경계·화자 표시가 질문·응답 이해에 도움이 되는가 | 같은 512통화 파일럿에서 B1 0.424005 / C2 0.396671. 단순 채택하지 않고 원인 점검으로 연결 | [결과](mission4_c2_pilot_20261111/C2_PILOT_RESULT.md), [paired 대조](mission4_c2_pilot_20261111/paired_audit.json) | [B1 확률](mission4_c2_error_comparison_20261112/B1_dev_prefix.jsonl.gz), [C2 확률](mission4_c2_error_comparison_20261112/C2_dev_prefix.jsonl.gz) 로컬 존재. 원격 `.pt` 실물/해시 재확인 대기 |
| C2 원인 점검 | 낮은 점수가 화자 표시 때문인지 다른 통제 문제가 있는지 확인 | 입력 잘림과 작은 학습량 등의 혼재를 확인하여 공통 원문·확대 비교를 설계. 특정 원인을 단독 확정하지 않음 | [오답 대조](mission4_c2_error_comparison_20261112/B1_C2_ERROR_COMPARISON.md), [원인 점검](mission4_c2_cause_audit_20261112/CAUSE_AUDIT.md), [복원 이력](mission4_c2_error_comparison_20261112/analysis_provenance.json) | 원확률의 float32 값/순서를 복원한 로컬 자료 있음. 원격 gzip 바이트 그대로 복사한 파일은 아님 |
| `context_expanded_v2_20261112` | 공통 원문·확대 학습·동일 초기화로 B1/C1/C2 비교 | Train 2,000 / 내부 Dev 512, seed 42·43, 각 5epoch. 평균 B1 0.416200 / C1 0.415644 / C2 0.417691. C2 우위 방향이 seed별로 달라 채택 보류 | [프로토콜](mission4_context_expanded_20261112/EXPERIMENT_PROTOCOL.md), [완료 결과](mission4_context_expanded_20261112/RESULT_SUMMARY.md), [출력 노트북](mission4_context_expanded_20261112/Mission4_Colab_COMPLETED.ipynb) | **실행 완료 / 6개 예측 전송 해시 대조·재채점 완료 / 6가중치 실물·무결성 검증 대기**. [상세 보존 상태](mission4_context_expanded_20261112/EVIDENCE_STATUS.md) |
| 확대 비교 후속 오답 분석 | 전체 평균 외 단일/다중 증상에서 무엇이 달라졌는가 | 6개 각각 512통화·15,186시점 재채점과 고유 36통화 원문 검토 완료. C2 채택 보류, B1 기준선 유지. 다음은 B1 양성 손실 배수 1→2 진단 제안이며 새 학습은 하지 않음 | [최종 검토](mission4_context_error_audit_20261112/RESULT_REVIEW.md), [분석 코드](../mission4_colab/audit_context_v2.py), [증거·실행 기록](mission4_context_error_audit_20261112/EVIDENCE_AND_EXECUTION.md) | 원본 예측 6개 전송 해시 대조 완료. 가중치 실물 미검증 |
| 확대 비교 추가 진단 수치 검토 | B1/C2의 마지막 시점·초기 시점·클래스별 오류와 입력 잘림 수치 재집계 | Training 내부 Dev, seed 42/43별 기술 통계; 원인 및 채택은 미확정. 새 학습 없음 | [검증 기록과 재현 코드](mission4_diagnostic_claim_check_20261112/RECORD.md) | 예측 원본 재채점 확인; 가중치 실물·무결성 및 학습 재개 상태는 별도 검증 대기 |
| 시점 분포·빈 예측 추가 진단 | 마지막 시점 결과, 빈 예측 대체와 학습·평가 시점 비중 확인 | 기존 예측의 사후 재채점 및 10개 학습 계획 해시 확인; 새 학습·정책 선정 없음 | [추가 진단 기록](mission4_additional_diagnostic_20261112/RECORD.md) | 예측 재채점 및 계획 재생성 확인; 실제 학습 상태와 가중치 보존은 별도 확인 대기 |
| `loss_weight_b1_v1_20261112` | B1 양성 손실 공통 배수 1→2가 다중증상 FN을 줄이면서 전체 4-1 점수를 지키는가 | **사전 계획·실행기 준비 중 / 새 GPU 학습 미시작**. 원 2,000/512통화·시점 계획·seed42/43을 동결하고 배수1도 새로 학습하는 4회 비교. 두 seed 다중 FN 감소·전체 점수 비감소를 후속 확인 후보 기준으로 지정. 현재 Dev는 설계용이며 독립 검증 아님 | [사전 프로토콜](mission4_loss_weight_20261112/EXPERIMENT_PROTOCOL.md), [고정 입력·계획 해시](mission4_loss_weight_20261112/FROZEN_INPUTS.json) | 새 가중치·성능 결과 없음. 새 실행기는 final과 마지막 optimizer/AMP/RNG 재개 상태 및 로컬 자산을 모두 유지하도록 준비. 원격 계획 실물 대조는 미확인 |
| `sampling_b1_uniform_v1_20261112` | 학습의 마지막 시점 비중을 줄이고 통화 안 시점을 균등하게 뽑으면 B1의 누적 4-1 점수가 개선되는가 | **코드 준비·CPU 기능 시험 완료 / 본 학습 미실행**. 원 2,000/512통화·초기화·손실·epoch5를 고정한 두 seed의 기준/균등 4회 비교 준비. 두 seed 모두 전체 점수 증가 시 후속 확인 후보. 최신 사용자 지시에 따라 오늘 새 GPU 요청·본 학습 없음 | [사전 프로토콜](mission4_uniform_sampling_20261112/EXPERIMENT_PROTOCOL.md), [코드·증거 검토](mission4_uniform_sampling_20261112/RECORD.md), [결과별 다음 단계](mission4_uniform_sampling_20261112/NEXT_STEPS.md) | 새 실험 가중치·성능 결과 없음. 작은 CPU 중간/epoch 경계 재개와 slot 순서 시험 통과, 20 MiB CLI 왕복 기록. 실모델 offline·CUDA/AMP 재개·대형 상태 전송은 미검증. 기존 context 완료 원본은 수정하지 않음 |
| 후속 NLI·임베딩·4-2 개선 | 오류 유형이 확인됐을 때 추가 구조의 효용 검증 | **실험 후보**. 이 목록에서 신규 성능 검증이 완료됐다고 주장하지 않음 | 후속 질문·기준·프로토콜 확정 후 새 ID 부여 | 신규 가중치/결과 없음 |

위 점수는 공개 채점식의 4-1 점수이며 Accuracy가 아니다. 서로 다른 표본·문맥 정책·학습 예산의 파일럿과 확대 결과를 직접 빼서 특정 요소의 개선량이라고 해석하지 않는다.

## 원격 위치

- 초기 실행: `MyDrive/DCC/runs/pilot512_e3_B0_seed42/`, `pilot512_e3_B1_seed42/`, `pilot512_e3_C2_seed42/`.
- 확대 실행: `MyDrive/DCC/runs/context_expanded_v2_20261112/` 안 `B1/C1/C2_seed42/43`.
- 경로만 알고 있다는 것과 원격 실물·해시를 확인했다는 것은 다르다. 현재 이 문서에서는 가중치 실물을 재검증했다고 쓰지 않는다.

## 당시 다음 담당자가 할 일 (2026-11-12 기록)

1. [기록 양식](templates/EXPERIMENT_RECORD_TEMPLATE.md)과 [현재 보존 상태](mission4_context_expanded_20261112/EVIDENCE_STATUS.md)를 읽는다.
2. 확대 실행 원본을 접근 가능한 환경에서 별도 감사하고 필수 가중치·예측을 포함한 보관 사본을 만든다.
3. B1의 [균등 prefix 코드·검토 상태](mission4_uniform_sampling_20261112/RECORD.md)와 [결과별 다음 계획](mission4_uniform_sampling_20261112/NEXT_STEPS.md)을 이어받는다. 오늘 범위는 코드·계획이며 새 학습이나 NLI 채택은 하지 않았다. 양성 손실 배수 2는 별도 후보로 두고 동시에 변경하지 않는다.
4. 향후 실행 지시가 있을 때 실제 모델의 offline 복원·CUDA/AMP 재개·대형 상태 전송·저장 공간을 먼저 확인한다. 마지막 재개 상태를 끝까지 유지하며 기존 완료 소스는 그대로 둔다.
5. 새 결과가 나오면 seed별 성능·사전 기준 대조·결정·보존 검증 상태를 추가한다. 현재 Dev 반복 사용은 독립 검증이 아니므로 후속 확인 표본과 선정 이력을 별도로 정한다.

- **2026-11-13 `sampling_b1_uniform_v1_20261113` — 첫 256배치 후 재개 오류로 중단.** B1 전체·초기·후기 대 균등 prefix, seed 42/43 비교. 첫 모델을 실제 batch 256에서 계획 중단하여 재개 상태 백업을 확인한 뒤 같은 설정으로 이어갈 예정. T4 사용, Drive 새 런타임 연결은 미확인이라 VM→Mac 해시 검증 백업으로 진행. 결과·보존 상태는 [실행 기록](mission4_uniform_sampling_20261113/EXECUTION_LOG.md) 참조. 아직 성능 결과 없음.

- **2026-11-13 `sampling_b1_uniform_v2_20261113` — 실행 중.** v1의 재개 명령 기록 오류를 새 실행기에서 수정하여 처음부터 재실행할 계획. 기존 v1 첫 256배치와 재개 실패 증거는 보존한다. [실행 기록](mission4_uniform_sampling_v2_20261113/EXECUTION_LOG.md).

## 2026-11-13 결과 확정 및 원인 검토

이 항목은 위의 당시 진행 상태에 대한 최신 갱신이며 과거 기록은 보존한다.

- `sampling_b1_uniform_v2_20261113`: 실행 및 기존 Dev512 재채점 완료. full/early/late seed42/43 0.435203/0.418624, uniform 0.421117/0.415598. 두 seed 모두 균등 샘플링의 개선 근거를 얻지 못했다. 복구된 확률 및 동결 메타데이터는 비공개 gold_pilot source_metadata에 보존했다.
- `evidence_aux_v1_20261113`: 문맥 상태 보조 loss0 대0.25, 두 seed, 동일 replay256/추가2epoch. 네 실행192실제업데이트/skip0. 기존 Dev 평균차이+0.00498691로 사전+0.005 screen 미통과. 별도 등록한 한 차례 추가 확인 v2는 미사용 holdout512/15,428시점에서 평균+0.00130128, 95% CI[-0.00270958,+0.00537637], seed42 양수/43 음수로 채택·풀학습 보류.
- 원인 분석: 단일 증상 개선·다중 증상 악화가 두 seed 공통. 오탐 감소와 정답 누락 증가의 절충을 전수 확률에서 확인. 원문9통화를 읽어 실제 누락과 공식 라벨 근거 이슈를 분리했다. 학습 기전 가설은 원인 확정과 구분한다.
- 보존: 네 최종 가중치/마지막 재개 상태, tokenizer/config/frozen source, 전수 확률/로그/재채점/체크섬은 외장 `DCC/private_experiments/evidence_aux_v1_20261113/`에 보관. 결과 확인 시 Colab GPU 활성0.
- [실험 결과](evidence_pilot_20261113/RESULT_REVIEW.md), [원인 검토](evidence_pilot_20261113/CAUSE_REVIEW.md), [최종 상태](evidence_pilot_20261113/EXECUTION_STATUS.md). 이번 유보셋은 평가 완료했으므로 다음 실험에서 미사용 독립 확인 자료라고 부르지 않는다.

## 2026-11-13 미확인 상태 성분 제거 비교

`unknown_component_off_v1_20261113`: 미확인 상태 보조loss 분자만 제거, 기존 분모/긍정·부정 gradient 크기 유지. replay256/AI silver40/2epoch/seed42·43/임계값0.50 동일. 기존 완료 대조군·보조군은 재학습하지 않음. 로컬·T4 사전검사 및 실제2update 백업 검증 통과 후 고정 비교 실행 중. 성능은 아직 미확정. [프로토콜](unknown_ablation_20261113/PROTOCOL.md), [실행 상태](unknown_ablation_20261113/EXECUTION_STATUS.md).

### 미확인 성분 제거 비교 최종 결과

이번 변경만으로 채택 기준을 충족하지 못했다. 기존 기준선을 유지하고 자동 풀학습은 하지 않았다. 대조군 대비 평균차이+0.002721, 기존 보조 대비-0.002266. 기전 사전조건 충족True. [결과](unknown_ablation_20261113/RESULT_REVIEW.md). 모든 가중치/재개 상태/확률/코드 보존 검증 완료.

## 2026-11-14 미확인 절반 강도 비교

`unknown_component_half_v1_20261114`: 미확인 분자 기여만0.1→0.05, 기존 분모/긍정·부정 gradient 유지. 256replay/40AI통화/2epoch/seed42·43/임계값0.50 동일. 기존 세 조건은 해시 검증 예측 재사용. 로컬·T4 사전 검사와 실제2update 외장 재개 검증 후 실행 중. [사전 기준](unknown_half_20261114/PROTOCOL.md), [상태](unknown_half_20261114/EXECUTION_STATUS.md). 아직 성능/채택 미확정.

### 절반 강도 비교 최종 결과

사전 진입 기준 미통과. 절반 강도 조건은 채택하지 않고 이 강도 축의 추가 탐색을 종료한다. 기존 기준선을 유지하며 풀학습하지 않았다. 대조군 대비 평균차이+0.005476, 기존 보조 대비+0.000489. [결과](unknown_half_20261114/RESULT_REVIEW.md). 전체 확률·학습 로그·가중치·재개 상태를 비공개 보관했다.

## 2026-11-14 B1 학습 통화 확대 — 준비

run_id: b1_data_scale_10k_v1_20261114. 2,000×5epoch 대 10,000×1epoch, 30,000 입력/3,750 batch 맞춤. 기존 두 seed 기준선 재사용. 학습률·입력·양성 손실 가중치·임계값 고정. 공식 Validation/holdout 사용 없음. 원자료·코드·설정·재개 상태·전체 예측 보존. 상세: data_scale_20261114/PROTOCOL.md.

### B1 10,000통화 비교 완료

기존2k 평균0.426913 → 10k 평균0.442019; 차이+0.015106, CI[-0.0011439839851249795, 0.030593961931209692]. 사전 독립확인 진입 통과:False. 최종 채택/풀학습/공식Validation 평가 없음. raw예측 재채점 및 두seed 가중치/재개/오프라인 표본 검증 완료. 상세 data_scale_20261114/RESULT_REVIEW.md.

## 2026-11-14 同10,000통화 epoch1→epoch2 비교 — 준비

run_id:b1_10k_epoch2_v1_20261114. 기존 두seed epoch1의 전체 optimizer/scaler/RNG 상태로 새 경로에서1epoch 추가. 같은데이터/입력/학습률/손실/threshold. 원부모 보존 및lineage, 전체재개·최종가중치·원확률 기록. 공식Validation/holdout 사용없음. 상세 epoch2_20261114/PROTOCOL.md.

### 10,000통화 Epoch2 비교 완료

Epoch1평균0.442019→Epoch2평균0.446236, Δ+0.004217, CI[-0.006264057168190208, 0.01485756485631246]. 사전독립확인진입통과:False. 최종채택/새독립확인/공식Validation평가없음. 가중치/전체재개상태/원확률/환경/판정근거보존. 상세 epoch2_20261114/RESULT_REVIEW.md.


## 2026-11-14 Epoch1/2 오류 비교 완료

- 실행 ID: `epoch2_error_audit_20261114`. 신규 학습·GPU 할당 없음. 기존 내부 Dev512통화의 네 모델 예측60,744행을 비교하고 원본 JSON512개의 전사·순서·화자·9개 라벨 매핑을 전수 대조했다.
- 예측으로 선정한32통화·1,051발화를 처음부터 끝까지 AI가 검토했다. 무작위 표본이나 인간 인증 골든셋이 아니며, 전체512통화 의미 검토로 표현하지 않는다. 공식 라벨은 변경하지 않았다.
- 두 seed 평균 점수 변화+0.004217을 정확히 분해했다. 누락 회복(+0.043957)·오탐 제거(+0.027290)가 새 오탐(-0.042627)·기존 정답 탈락(-0.024404)에 대부분 상쇄됐다. 이는 산술 분해이며 인과효과 추정이 아니다.
- 선택한5통화·6시점의 입력 형태를 CPU에서40행 비교했다. 일부 누락은 관련 발화만 읽을 때 회복했지만 국소 입력으로 오탐이 커지는 반례도 있었다. 검토자가 사후 선택한 입력 검사로 배포 성능 개선을 주장하지 않는다.
- 다음 후보: 전체 문맥과 자동 국소 문맥을 학습형 결합으로 비교. 현재는 제안만 작성했으며 구현·학습·채택하지 않았다.
- [결과와 한계](epoch2_error_audit_20261114/RESULT_REVIEW.md), [재현 방법](epoch2_error_audit_20261114/REPRODUCTION.md), [다음 비교 제안](epoch2_error_audit_20261114/NEXT_EXPERIMENT.md). 원문·예측·라벨 이슈·검증표는 `artifacts_private/epoch2_error_audit_20261114/`에 비공개 보관한다.

## 2026-11-14 전체·국소 문맥 통제 비교 완료

- `local_fusion_v1_20261114`: Epoch 2(`b1_10k_epoch2_v1_20261114`) 두 인코더를 고정하고, 기존 10,000통화 밖의 Training 1,000통화에서 333파라미터 잔차 결합기를 학습해 내부 Dev 512(15,186시점)로 비교했다. 공식 Validation/holdout은 미사용.
- 결과(공개 4-1 점수): 참고 `unchanged_encoder` 평균 `0.446236`(`42: 0.451499 / 43: 0.440973`), 대조 `global_only` 평균 `0.434036`(`0.437786 / 0.430285`), 후보 `global_local` 평균 `0.433431`(`0.438494 / 0.428368`). `global_local − global_only` 평균 차이 **`-0.000605`**, paired call bootstrap 95% CI **`[-0.006786, +0.005714]`**.
- 결정: 사전 5개 기준 중 1개(`multi_FN_reduced`: `7,531.5→7,434.5`)만 충족하고 나머지 4개(두 seed 개선, `+0.005`, CI 하한>0, FP 증가 `<=10%` 대비 `+12.84%`) 미달로 **`do_not_adopt_under_registered_gates` (미채택)**. 무가중 BCE+`0.50` 고정 임계값으로 인해 양 결합기 모두 원 인코더 대비 빈 출력이 `15.48%→29.5~30.3%`로 급증했다.
- 감사 및 보존: 원격 `cache`/`fit` 완료 후 `compare` 단계에서 `numpy.bool_` JSON 직렬화 예외(`FAILED.json`)가 발생해, 동결된 `fusion.py`를 수정하지 않고 [audit_and_compare.py](../tools/local_fusion/audit_and_compare.py)로 동일 비교·재채점·캐시 3,024건 SHA-256·오프라인 CPU 추론 재현 및 GPU 반납(`SESSION_CLOSE.json`)을 검증했다. [결과 검토](local_fusion_20261114/RESULT_REVIEW.md), [실행·검증 상태](local_fusion_20261114/EXECUTION_STATUS.md), [계획 대비 차이](local_fusion_20261114/PROTOCOL_DEVIATIONS.md).

## 2026-11-14 4-1 후처리 보정·pos_weight 정렬 결합·앙상블·4-2 중단 비교 완료

- `decision_calibration_v1_20261114`: 추가 GPU 없이 `local_fusion_v1_20261114`의 `cal1k`(1,000통화) 및 `Dev 512`(15,186시점) 캐시를 사용해 누수 없이 CPU 비교 완료(`score.py` 전 조건 교차검증 일치, 단위테스트 4/4 통과).
- 4-1 단일 시드 결과(기준 `unchanged_050` `0.446236` 대비):
  - `top1_fallback_050`(튜닝 0개): **`0.474864`** (`+0.028628`, `CI [+0.0233, +0.0343]`, 다중 `0.3818→0.4054`, 빈 출력 `15.48%→0.00%`, **사전 4기준 전수 통과**).
  - `global_thresh_top1`(`0.500239`, `+0.0540`)·`class_thresh_top1`(`0.496734`, `+0.0505`)은 전체 점수를 크게 올렸으나 다중 점수가 하락(`0.3637 / 0.3726`)해 다중 비감소 기준 미달.
  - `pos_weight` 정렬 후 `pw_global_only_top1`(`0.481991`, 4기준 통과) vs `pw_global_local_top1`(`0.481665`, 4기준 통과) 직접 차이는 `-0.000326`(`CI [-0.0064, +0.0059]`)으로 국소 결합의 추가 이득 없음.
- 2시드 앙상블 참고(`0.5*(p42+p43)`): `ens_050` `0.457951` → `ens_top1_050` **`0.489892`**(다중 `0.414129`) → `ens_global_thresh_top1` **`0.505402`**(다중 `0.400260`) → `ens_class_thresh_top1` **`0.508124`**(다중 `0.385556`).
- 4-2 2-fold OOF 조기 중단: `S0_end` `0.3050~0.3091` → `top1_fallback_050 + S2` **`0.459201`(seed 42) / `0.459196`(앙상블)** (평균 중단 `10.3~10.4` 발화).
- [프로토콜](decision_calibration_20261114/PROTOCOL.md), [결과 검토](decision_calibration_20261114/RESULT_REVIEW.md), [실행·검증 상태](decision_calibration_20261114/EXECUTION_STATUS.md). 다음은 미사용 학습 데이터(`12,509~13,509`통화)를 반영한 GPU 백본 학습 확장 진행 예정.

## 2026-11-14 full22k L4 전환 상태 및 방법 검토

- `b1_full22k_cosine_v1_20261114`: 13:26 KST 원격 조회에서 seed42 confirm 추론 완료, seed43 학습8,438업데이트 완료 후 confirm 예측256/512 진행을 확인했다. A100 활성 세션 없고 L4만 활성. seed42 외장 final/resume/구제ZIP 실제 해시 일치. seed43 최종 모델·재개 상태는 원격 실물 존재하나 외장 보존·재채점은 미확인이다.
- 지정 Sol xhigh 방법 검토 및 Terra 실행 점검 결과, 현재 실행 중단·재학습의 직접 사유는 발견하지 못했다. 주 평가 규칙의 모호함, 모델-캐시 해시 연결, GPU 환경 차이, 확인셋 자체 정책 탐색과 독립 전이 구분, seed43 외장 백업이 잔여 항목이다. 완료 코드·원격 실행은 변경하지 않았다.

## 2026-11-14 B1 22,509통화 Cosine 확장 및 독립 확인 분할 검증 완료

- `b1_full22k_cosine_v1_20261114`: 기존 `10,000`통화 Epoch 2(`b1_10k_epoch2_v1_20261114`)에서 대칭 출발해 미사용 학습 통화 `12,509`건을 포함한 전체 `22,509`통화(`67,527` 프리픽스, `8,441` 배치) 1에폭(`lr=1.5e-5`, 5% warmup, cosine decay `min_ratio=0.1`)을 학습하고, 기존 `dev512` 및 미사용 독립 확인 분할 `confirm_dev512`(`512`통화, `15,497`시점)에서 평가했다.
- 하드웨어·보존 검증: `seed 42` 학습/`cal1k`/`dev512`는 A100에서 완료 후 사용자 지시에 따라 A100을 즉시 중단·종료했고, `seed 42 confirm_dev512` 및 `seed 43` 전체 학습·예측은 `NVIDIA L4`에서 완료 후 전 세션 종료(`Active assignments: 0`). 두 시드 `final.pt`·`training_state_last.pt`·`.npz` 전량 외장 디스크 SHA-256 검증, 오프라인 CPU 16-프리픽스 일치(`max_prob_err < 4.4e-7`), `score.py` 전수 재채점(`EVALUATE_FULL22K_OK`) 완료.
- 4-1 독립 확인 분할(`confirm_dev512`) 결과(공식 Mission 4-1 점수):
  - `unchanged_050`: 단일 평균 `0.481845 → 0.501201` (`+0.019356`, `CI [+0.0100, +0.0293]`, 4/4 통과), 2-시드 앙상블 `0.490452 → 0.505813`.
  - `top1_fallback_050`: 단일 평균 `0.509059 → 0.531479` (`+0.022419`, `CI [+0.0125, +0.0325]`, 4/4 통과), 2-시드 앙상블 **`0.520012 → 0.539550`** (`multi 0.384182`).
  - `global_thresh_top1`: 단일 평균 `0.521007 → 0.537173` (`+0.016166`, 4/4 통과), 2-시드 앙상블 **`0.529651 → 0.543369`** (`traj4_global_thresh_top1`: **`0.545611`**).
  - `class_thresh_top1`: 단일 평균 `0.521867 → 0.538275` (`+0.016408`, 4/4 통과), 2-시드 앙상블 **`0.532921 → 0.544811`**.
- 4-2 `confirm_dev512` 제로-리키지 전이(`dev512` 정책 그대로 적용): `full22k_ens + top1_fallback_050` **`0.469480`** (`traj4_best_cal1k`: **`0.470452`** vs 끝까지 듣기 `0.316281`).
- 결정: **`adopt_as_new_baseline` (신규 기준선 채택)**. [프로토콜](full22k_20261114/PROTOCOL.md), [실행·검증 상태](full22k_20261114/EXECUTION_STATUS.md), [결과 검토](full22k_20261114/RESULT_REVIEW.md). 다음은 L4/T4 기반 `klue/roberta-large` 확장(Track 2).
- [검토와 후속 체크](full22k_runtime_review_20261114/REVIEW.md). 원격 조회와 실제 파일 검사 자료는 비공개 `artifacts_private/full22k_runtime_review_20261114/`에 보관한다. 아직 최종 성능 채택 판정이 아니다.


## 2026-11-14 팀원 브랜치 비교 및 공개 요약 게시

- `mission4`와 실험용8개브랜치를 고정commit으로 읽고 보고서·집계·코드·노트북실행출력의 연결을 검토했다. 새 학습이나 가중치 교체는 하지 않았다.
- streaming은 원시0.5/Top-1/maxent/학습형중단 및 Dev/Validation 성적을 구분했다. 초기 Kc 노트북의 Validation 학습은 출력에서 확인되지만 높은보고성적과의동일실행연결은미확인이다. full Kc32,840통화캐시원천도미확인으로남겼다. 브랜치명만으로백본실행·우수성을확정하지않는다.
- 사용자확인: 증상 임계값0.5 고정, Top-1허용미확인. Training 전용동일조건Kc/RoBERTa비교를후보로제안했고아직실행하지않았다.
- [팀원검토](team_branch_review_20261114/TEAM_RESULT_REVIEW.md), [후속비교제안](team_branch_review_20261114/NEXT_COMPARISON.md), [공개게시영수증](team_branch_review_20261114/PUBLICATION_VERIFICATION.json). GitHub 새브랜치 `docs/mission4-experiment-review-20261114`, 원격확인commit `3611a7e`. 공개5파일은집계·요약·검토이며,원문·예측·가중치를추가업로드하지않았다.

## 2026-11-14 Training 전용 두 백본 공통 조건 재검증 준비

- 실행 ID `clean_backbone_v1_20261114`: fresh RoBERTa/KcELECTRA × seed42/43, Training10,000통화·동일3prefix·1epoch, raw0.5/Top-1 없음. 기존 완료 모델이나 공식 Validation을 학습에 섞지 않는다. Dev512는 반복 사용 개발용이다.
- 두 실제 모델의 encoder 로드·새head초기화와 CPU9출력, 공통 입력63,587개(각seed30,000학습입력 및 Dev15,186시점)의 전수 캐시, 합성 재개·캐시·monitor/predict 통합검사를 준비했다. 설계는Solxhigh, 코드 최종점검은Sol사용량한도후허용대체Astrahigh가 수행했다. 발견된 cachebuilder 속성과 체크포인트 조건식별 문제는 GPU 실행 전에 수정했다.
- 실제 GPU 학습·성능·외장 최종 보존은 이 준비 기록과 별개다. [설계](clean_backbone_20261114/DESIGN_REVIEW.md), [코드 점검](clean_backbone_20261114/CODE_REVIEW.md), [streaming 학습 설명](clean_backbone_20261114/STREAMING_TRAINING_EXPLANATION.md). 외장 비공개 실행 루트 `DCC/private_experiments/clean_backbone_v1_20261114/`.

## 2026-11-15 팀 기준선 변경 및 최신 업로드 논리 검토

- 사용자가 `mission4-streaming`을 앞으로의 기준선으로 지정했다. 최신 확인은 streaming `5d7539b`, 이를 병합한 mission4 `a16b61d`, 신규 실험 연결 브랜치 mission4-prefix `1f69317`이다. 코드와 커밋된 집계 기록을 검토했으며 새 학습·공식 Validation 평가·가중치 교체는 하지 않았다.
- 최신 운용 후보는 E026: raw 0.5/Top-1 없음, LightGBM 중단 점수의 기준값 0.5. 기록상 Validation 4-1 0.4867, 4-2 0.5070이다. E025 및 옛 maxent 결과와 구분한다. E026 Dev 개선은 약 +0.0042이나 95% CI가 0을 포함하며, 기존 기준 예외로 채택했다는 이력을 보존한다. 이번 검토에서 원본 예측을 재채점한 결과는 아니다.
- mission4-prefix는 기존 E002 학습과 고정 E026 평가를 연결하는 도구다. 문서상 T8 개선 전략은 미구현이며 재개 상태 저장도 미보완이다. 실행 도구 추가를 새 학습 방식의 성능 검증 완료로 해석하지 않는다.
- 다음 작업은 공통 평가·학습 출처 확인, prefix 효과를 분리하는 통제 비교, 중단 정책의 선택·검증 분리 보완이다. 기존 fresh 두 백본 비교는 준비 성적과 분리해 보존하며, 팀의 E021/E022를 확인한 뒤 우선순위를 재검토한다.
- [최신 검토 범위·소스 해시](mission4_streaming_baseline_20261115/LATEST_REVIEW_SCOPE.json). 이전 [E025 검토](mission4_streaming_baseline_20261115/REVIEW_AND_NEXT_EXPERIMENTS.md)는 당시 commit의 기록으로 보존한다.

### 최신 검토에서 확인한 보충 과제

- [업데이트 검토와 최소 실험 계획](mission4_streaming_baseline_20261115/UPDATES_REVIEW.md)에 E026 및 신규 prefix 연결기의 확인 범위와 역할을 정리했다. 이는 지정 Astra high의 코드·방법 검토이며 새 학습 결과가 아니다.
- E026 `wide/result.json`은 `E016_final/shift`를 사용하며 해당 설정은 Training 전체 29,200통화 라벨로 보정값을 계산했다고 명시한다. 따라서 내부 Dev 2,920통화의 라벨 분포도 중단용 보정에 간접 포함된다. raw 0.5인 4-1 판정과 구분하며, 공식 Validation이 학습에 유입됐다는 증거로 확대하지 않는다. 학습용 26,280통화만의 보정으로 중단 정책을 재비교하는 것이 우선 보충 과제다.
- Full 대 Prefix 효과를 같은 초기화·통화·노출량·학습 단계 수에서 분리하는 비교와, 중단 정책의 내부 선택·평가 분리 검증을 제안했다. 과거 모델 성적을 다른 판정 규칙으로 재사용하거나, 이미 본 Dev를 새 독립 확인셋이라고 부르지 않는다. 현재 실행 상태는 검토·계획 완료, 실제 보충 실험 미실행이다.

## 2026-11-16 E026 중단 정책 보충 실험 착수

- [E026 정적 감사](mission4_e026_audit_20261116/REPORT.md): 로컬 Git commit `5d7539b`의 `policy.json`·`result.json` 실물을 읽어 beta128, tau0.5, min_step5, fit26,280/Dev2,920, 기록된 Macro-F1×조기성 및 점수 차이 `+0.004193`의 산술 일치를 확인했다. 기록된 95% 구간은 0을 포함한다. **원확률 재채점·통화별 중단 재현·분할 누출 검증은 미실행**이다.
- [fit-only 보정 통제 비교 사전 계획](mission4_e026_fitonly_policy_20261116/PROTOCOL.md): 기존 E026을 재현한 후, 학습용 통화 라벨만으로 보정값과 정책을 다시 계산하고 beta0 대 고정 beta128을 비교한다. 현재 필수 원본 확률·OOF·fit/dev 목록을 이 Mac 작업 폴더에서 찾지 못해 **실제 정책 재학습·새 성능 결과는 아직 없다**. 다른 보관 위치에 있을 수 있으므로 실제 누락으로 단정하지 않는다.

## 2026-11-16 신규 4-2 안정성 중단 규칙 CPU 비교

- 실행 ID `m4_stop_stability3_cpu_20261116`. E026 재현이 아니다. 기존 full22k의 seed42/43 전 시점 logits를 고정하고, Training 내부에서 이미 사용된 dev512·confirm_dev512 각 512통화에서 B_end/B5/B10/S3를 비교했다. 증상 판정은 모든 클래스 `p>=0.5`, Top-1 없음. 새 학습이나 공식 Validation 접근은 없었다.
- 사전 주 비교 S3(5번째 이후 최근 3시점 증상 벡터 안정 시 중단) − B10(10번째 중단): 2-seed 확률 평균에서 dev `-0.039719`(95% 통화 bootstrap `[-0.063242,-0.017960]`), confirm `-0.046648`(`[-0.072879,-0.022038]`). seed42/43에서도 각각 두 분할 모두 음수였다. S3는 평균 약 6번째에 멈추며 빈 예측이 늘었다. **이 S3 규칙은 미채택**한다. B10이 최적이라는 결론이나 독립 확인 성적은 아니다.
- 원본 4개 NPZ sidecar SHA-256, seed 간 매핑·분할 비중복, 24개 조건의 공개 채점기 점수 일치, 미래 확률 변경 7,200건 인과성 점검을 통과했다. 통화별 확률·중단·예측·정답은 비공개 `artifacts_private/runs/m4_stop_stability3_cpu_20261116/`에 보존했다. [사전 프로토콜](m4_stop_stability3_cpu_20261116/PROTOCOL.md), [결과와 한계](m4_stop_stability3_cpu_20261116/EXPERIMENT_RECORD.md).
- E026과의 연결은 실험 후 [결과 기록의 별도 절](m4_stop_stability3_cpu_20261116/EXPERIMENT_RECORD.md)에 보충했다. S3는 E026 재현·동일 조건 우열 비교가 아니다. 직접 연결하려면 E026의 동일 통화·동일 시점 원확률과 정책을 확보해 E026 `β=0/128` 및 B10/S3를 함께 재생해야 한다. E026 fit 전용 보정의 엄밀한 재학습은 그 다음 별도 실험이다.

## 2026-11-16 E026 동일 입력 고정 정책 재생 완료

- 실행 ID `e026_same_input_replay_20261116`. 팀 공유 ZIP(`mission4_baseline_share.zip`, `mission4_share2_E001_E007_bundle.zip`, `mission4_symptom.zip`)에서 `E001`/`E002`/`E007`의 `dev_prefix.npz`, `fit_ids.txt`(`26,280`건)·`dev_ids.txt`(`2,920`건, 교집합·완전 전사 중복 0건), 누락 `m4/__init__.py`(`bec21f16...`)를 확보하고, 외장 `TL_서울_구급`에서 `train.jsonl`(`CRLF` 변환 시 원 노트북 해시 `9e00495204f96a69...` 일치)을 복원해 `lightgbm 4.7.0` CPU 환경에서 실행했다.
- **원점수 완전 재현**: 동일 `Dev 2,920`통화(`87,814`시점) `float32` 3모델 평균 + `raw p>=0.5`(Top-1 없음)에서 `beta0 = 0.4989532854214187`, `beta128 = 0.5031464178054406`이 기록 원수치와 오차 없이 일치했고 공개 채점기(`score.py`)도 경고 0건·일치 검증을 통과했다.
- **동일 입력 중단 규칙 비교**(`beta128` / `beta0` / `B10` / `S3`):
  - `beta128`: 4-2 **`0.503146`** (`Macro-F1 0.606354`, `earliness 0.829790`, 빈 예측 `43/2,920`)
  - `beta0`: 4-2 `0.498953` (`Macro-F1 0.613721`, `earliness 0.812998`, 빈 예측 `34/2,920`)
  - `B10`: 4-2 `0.450774` (`Macro-F1 0.556502`, `earliness 0.810013`, 빈 예측 `454/2,920`) → `beta128` 대비 `-0.052372` (`95% CI [-0.060279, -0.044314]`)
  - `S3`: 4-2 `0.412718` (`Macro-F1 0.463150`, `earliness 0.891111`, 빈 예측 `958/2,920`) → 초기 빈 예측 3연속 시 조기 종료(평균 `5.75`발화)로 `beta128` 대비 `-0.090428`, `B10` 대비 `-0.038056`
  - `beta128 - beta0`: `+0.004193` (`10,000`회 paired bootstrap `95% CI [-0.001003, +0.009525]`, `p_gt0=0.9405`)로 여전히 `0`을 포함. 또한 `E011_oof`는 미확보 상태이므로 fit-only 보정 재학습은 별도 과제로 남는다. [프로토콜](mission4_e026_same_input_replay_20261116/PROTOCOL.md), [실행 기록](mission4_e026_same_input_replay_20261116/EXPERIMENT_RECORD.md).

## 2026-11-16 E026 우선1 후속 CPU 근거 보완 점검 완료

- 실행 ID `e026_priority1_checks_20261116`. [업데이트 검토](mission4_streaming_baseline_20261115/UPDATES_REVIEW.md) 우선1 잔여 점검 전 항목을 CPU로 완료했다([보고서](mission4_e026_priority1_checks_20261116/REPORT.md), `artifacts_private/runs/e026_priority1_checks_20261116/`).
- **가중치 실물 ↔ 확률표 일치**: 공유 ZIP의 `E001(tl_s42)`·`E002`·`E007` `best_model`을 Dev 앞 8통화(`226`시점)에서 CPU FP32 추론한 결과, `dev_prefix.npz`와의 평균 확률 차이는 `~5.4e-05`이며 `0.5` 이진 판정 불일치는 3모델 모두 `0`건이었다.
- **온라인·일괄 완벽 일치**: `HaltFeatureState` 및 `should_halt`가 `Dev 2,920`통화 `87,814`시점에서 일괄 계산과 최대 특징 오차 `5.55e-16`, 중단 시점 `2,920/2,920`통화 일치했다.
- **`Dev 2,920` 4-1(`raw p>=0.5`, Top-1 없음, `score.py` 경고 `0`건)**: `E001` `0.501355`, `E002` `0.502733`, `E007` `0.504211`, `ENS3` **`0.506001`**.
- **`E025` vs `E026` 기준값 차이 해명 및 `fit shift` 민감도**: `fit 26,280` 기준 `shift`(`E002 run_config` 동일)에서 `beta0`는 **`0.498835`**로 `E025` 대조 점수와 정확히 일치하며, `29,200` 기준 `E026 beta0`(`0.498953`)과의 `+0.000118` 차이는 `shift` 대상(`8`통화 변경)만으로 설명됐다. 기존 정책에 `fit shift`만 대입한 민감도에서 `beta128_fit - beta0_fit = +0.003813`(`[-0.001434, +0.009202]`)였다.
- **신호 불일치 실측 빈도**: 합성 반례(`raw>=0.5` 빈 예측인데 `v(t)>=0.5`)는 `87,814`시점 중 단 `1`건(`0.0011%`), 실제 `beta0`/`beta128` 중단 시점 `2,920`건 중 **`0`건**이어서 `value` 신호 교체를 위한 `2×2` GPU 재학습 필요성이 낮음을 확인했다. 또한 `policy.json` fit OOF 격자에서 `β=32~256` 구간이 `0.4936~0.4941`의 완만한 평탄 영역임을 확인했다.

## 2026-11-16 Step 2 4-1 오류 원인 분해 완료 (`Dev 2,920` + 블라인드 `Dev16` + `G16 × full22k`)

- 실행 ID `m4_step2_error_analysis_20261116`. [보고서](mission4_step2_error_decomposition_20261116/REPORT.md), 코드 [run_step2_part1.py](../tools/step2_error_analysis/run_step2_part1.py)·[run_step2_part2_dev16.py](../tools/step2_error_analysis/run_step2_part2_dev16.py), 산출물 `artifacts_private/runs/m4_step2_error_analysis_20261116/`.
- **누수 차단**: 기존 `G01~G16` 16통화가 `mission4-streaming` `fit(26,280)`에 포함됨(`dev=0`)을 확인해, `G16`은 미학습인 `full22k` 대조에만 쓰고, `E001/E002/E007/ENS3` 분석은 미학습 `Dev 2,920` 전체(`87,814`시점) 및 내부 SHA-256 블라인드 16통화(`D01~D16`, `420`발화; AI 1차 초안·사람 검수 대기 `HUMAN_REVIEW_DEV16.csv`)로 분리 수행했다.
- **`Not So Fast` 부분 입력 원인 분리**:
  - `Dev 2,920` 어휘 단서 기준: `ENS3` FN(`58,374`건)의 **`97.35%`가 단서 등장 전**(`23.08%` 후반 등장 / `74.27%` 통화 내내 미등장)에서 발생했고, 단서 등장 후 Recall은 **`96.02%`**(`onset` 직전 `0.4539 → 0.8875`)였다.
  - 블라인드 `Dev16 × ENS3` 및 `G16 × full22k_ens`: 직접 근거 등장 후 Recall 모두 **`100.0%`**(`177/177`, `126/126`, retention drop `0`). 반면 전사에 없는 임상 라벨(`D01`, `D08`, `D10`, `D13` 등) 격차가 확인됐다.
  - 오탐(`FP`): `ENS3` 전체 FP(`36,299`건)의 **`77.7%`(`28,218`건)가 단서 전(`pre-cue`)** 발생. Prefix 모델(`E002` `31,989`, `E007` `32,210`)이 `E001`(`27,606`)보다 `pre-cue` 오탐이 더 많았으며, 클래스별로는 **`오심`**(`P=0.326, FP=5,408`)과 `전신쇠약`(`FP=6,350`), `어지러움`(`FP=5,730`) 비중이 컸다.
- **Q&A 전후 및 다중 증상(`Nature 119` + `TRIPOD+AI`)**:
  - 첫 증상 질문→짧은 부정(`158`건): 질문 때문에 새로 `>=0.5` 오탐이 된 경우는 **`1`건**뿐(`평균 0.1416→0.1421→0.1386`)이었으며, 주소 확인 뒤 탈락도 `1/173`건으로 극소였다. 단 `D11`처럼 앞 맥락으로 이미 `>=0.5`가 켜진 경우 부정 답변 뒤에도 확률이 내려가지 않고 잔존하는 오탐이 확인됐다.
  - 다중 증상(`962`통화): `ENS3` 마지막 발화 Recall이 단일 `82.64%` 대비 **`53.69%`**(`3개` `47.5%`, `4+개` `45.1%`)에 머물렀다.
- **후속 방향**: `E001`(TAPT 기반 전체 통화 학습, 후반 precision 강점) 대 `E002/E007`(TAPT+Prefix, 초반 recall 강점·`pre-cue` 오탐 증가)의 원인을 통제 분리하기 위해 **Step 3 (`Full vs Uniform Prefix`)**로 이어간다.

## 2026-11-16 `m4_precue_bce_v1_20261116` 사전 단서 양성 손실 완화 비교 완료 (`seed 42`, 미채택)

- 실행 ID `m4_precue_bce_v1_20261116`. [프로토콜](m4_precue_bce_v1_20261116/PROTOCOL.md), [결과 기록](m4_precue_bce_v1_20261116/EXPERIMENT_RECORD.md), 코드 [train_precue.py](../tools/m4_precue_bce/train_precue.py), 정량 결과 `artifacts_private/runs/m4_precue_bce_v1_20261116/s42_eval.json`, 외장 보관 `/Volumes/로컬 디스크/DCC/private_experiments/m4_precue_bce_v1_20261116/s42/`.
- **변경점 (`E002` `seed 42` 대비 단일 변수):** 동일 `TAPT`, `fit 26,280`, `3epoch`, `batch 16`, `lr 5e-5`, `LLRD 0.8`, `seed 42`, `raw p>=0.5`에서 `y=1` & `t < t_cue` (`30,938`개) 요소만 `pos_weight=1.0`으로 완화 (`post-cue`·`uncued`·`y=0` 동일).
- **결과 (`Dev 2,920`, `87,814`시점):** `pre-cue FP`가 `31,989 → 28,989`(`-3,000`건) 감소하고 `4-2` 단독 점수는 `fit_shift β=0` `0.492958 → 0.495434`(`+0.00248`), `E026 β=128` `0.500064 → 0.501891`로 올랐으나, 전역 확률 하향으로 `empty_rate`가 `15.15% → 17.14%`로 증가해 주 지표 **`4-1`은 `0.502733 → 0.497129`(`-0.005604`, `[-0.008890, -0.002368]`)**, `ENS3` 교체도 `0.506001 → 0.503910`으로 하락하여 **미채택**.
- **보존 확인:** 로컬 CPU `3`개 단위 테스트 + L4 CUDA `2`스텝 중단·재개 확인, `final.pt`/`training_state_last.pt`/`dev_prefix.npz` 외장 SHA-256 검증, 앞 4통화 `72`시점 오프라인 CPU 추론 일치(`0.5` 불일치 `0`건), `score.py` 재채점(경고 `0`) 모두 완료. 후속으로 `SoftJaccard + BCE` (`m4_softjaccard_v1_20261116`)로 전환.

## 2026-11-16 `m4_softjaccard_v1_20261116` `BCE + SoftJaccard` 비교 완료 (`seed 42`)

- 실행 ID `m4_softjaccard_v1_20261116`. [프로토콜](m4_softjaccard_v1_20261116/PROTOCOL.md), [결과 기록](m4_softjaccard_v1_20261116/EXPERIMENT_RECORD.md), 정량 결과 `artifacts_private/runs/m4_softjaccard_v1_20261116/s42_eval.json`, 외장 보관 `/Volumes/로컬 디스크/DCC/private_experiments/m4_softjaccard_v1_20261116/s42/`.
- **변경점 (`E002` `seed 42` 대비 단일 변수):** 동일 `TAPT`, `fit 26,280`, `3epoch`, `batch 16`, `lr 5e-5`, `LLRD 0.8`, `pos_weight^0.5`, `seed 42`, `raw p>=0.5`에서 손실에 `1.0·SoftJaccard` 추가.
- **결과 (`Dev 2,920`, `87,814`시점):**
  - **`4-1`:** `E002(0.502733) → softjaccard_s42` **`0.526393`**(**`+0.023660`**, `95% CI [+0.019270, +0.028127]`, `p_gt0=1.0`; 단일 `0.563053→0.596587`, 다중 `0.379961→0.383525`, 빈 예측률 `15.15%→8.16%`). `ENS3` 교체도 `0.506001 →` **`0.512504`**(`+0.006504`, `[+0.003906, +0.009139]`).
  - **`4-2`:** 저빈도(`오심`, `두통`) 재현율 저하로 단독 `Macro-F1`이 낮아져 `E026 β=128` 단독 `0.493339`(`E002 0.500064`), `ENS_rep` `0.503103`(`ENS3 0.503146`).
- **보존 확인:** `final.pt`/`training_state_last.pt`/`dev_prefix.npz` 외장 SHA-256 일치, `72`시점 오프라인 CPU 추론 일치(`0.5` 불일치 `0`건), `score.py` 재채점 완료. 후속으로 `0.5·SoftJaccard + 0.5·SoftMacroF1` (`m4_jacc_macrof1_v1_20261116`)로 연결.

## 2026-11-16 `m4_jacc_macrof1_v1_20261116` 1차 실행 (원격 유휴 회수 / 미회수)

- 실행 ID `m4_jacc_macrof1_v1_20261116`. [프로토콜](m4_jacc_macrof1_v1_20261116/PROTOCOL.md), [실행 상태](m4_jacc_macrof1_v1_20261116/EXECUTION_STATUS.md).
- `22:04` KST `L4`(`dcc-m4-precue-l4-20261116`, `PID 103432`)에서 정상 개시했으나, 대기 중 Mac 절전(`~22:07–23:18`)으로 확인이 지연되며 Google Drive 미마운트 임시 런타임이 자동 반납되어 산출물을 회수하지 못했다(`Active assignments: 0`, 잔여 `73.41` compute units). Drive 마운트 또는 절전 해제 상태에서 재실행 예정.

## 2026-11-17 `m4_jacc_macrof1_v1_20261116` 2차 실행 완료 및 `m4_softjaccard_v1_s43` 착수

- **오전 정지 원인 해소 및 `m4_jacc_macrof1_v1_s42` 2차 완료 (`L4`):** 오전 `10:09` 재시도(`task-2219`)가 `colab_cli` 커널 재시작 뒤 작업 디렉터리 이동 호출(`timeout=None`)에서 무한 대기하며 실제 학습이 시작되지 않았음을 진단·복구한 뒤, `16:47` L4 세션(`dcc-m4-stage2-l4-20261117`)에서 `m4_jacc_macrof1_v1_s42`(`BCE + 0.5·SoftJaccard + 0.5·SmoothMacroDice`)를 `1,771.9초` 만에 완주했다. [결과 기록](m4_jacc_macrof1_v1_20261116/EXPERIMENT_RECORD.md), 정량 JSON `/Volumes/로컬 디스크/DCC/private_experiments/m4_jacc_macrof1_v1_20261116/s42/eval_precue.json`.
  - **`4-1` (`Dev 2,920`, `raw p>=0.5`, Top-1 없음):** 단독 **`0.513424`**(`vs E002 +0.010691`, `95% CI [+0.006794, +0.014600]`, `score.py` 경고 `0`건), `ENS_rep_jm (E001+JM+E007)` **`0.509468`**(`vs ENS3 +0.003467`, `[+0.001495, +0.005548]`).
  - **`4-2` (`Dev 2,920`):** `E026` 고정 정책에서는 조기 중단 편향으로 단독 `0.495906`·`ENS_rep 0.503139`였으나, 자체 확률 궤적 기준 `nested 5-fold` 중단 적합 시 **`JM_s42` 단독 `0.501743`**(`Macro-F1 0.6072`)으로 `ENS3`(`0.501420`)을 상회하고 **`ENS_rep_jm` `0.503795`**(`Macro-F1 0.6073`, `+0.002375`)로 `4-1`·`4-2`를 동시 개선했다(단 `Dev` 탐색 결과이므로 `fit 26,280 OOF` 검증 필요).
  - **보존 확인:** `final.pt`·`training_state_last.pt`·`dev_prefix.npz` 외장 SHA-256 일치, `72`시점 오프라인 CPU 추론 일치(`max_abs_err=0.000592`, 불일치 `0`건), `score.py` 재채점 완료.
- **후속 실행 중 (`17:26 KST` 착수):** `SoftJaccard` multi-seed 재현성 확인을 위해 같은 L4 세션에서 `m4_softjaccard_v1_s43` (`seed 43`, `PID 120165`, GPU `85%`) 가동 중 (`30`분 뒤 확인 예정).

## 2026-11-17 `m4_softjaccard_v1_s43` (`seed 43`) 완료·검증 및 `m4_jacc_macrof1_v1_s43` 착수

- **`m4_softjaccard_v1_s43` 완료 및 2-seed 검증 통과 (`L4`, `1,764.3초`):** [보충 기록](m4_softjaccard_v1_20261116/SEED43_SUPPLEMENT.md), 외장 보관 `/Volumes/로컬 디스크/DCC/private_experiments/m4_softjaccard_v1_20261116/s43/`.
  - **`4-1` (`Dev 2,920`, `raw p>=0.5`):** `SJ_s43` 단독 **`0.524933`** (`vs E002 +0.022199`, `95% CI [+0.017440, +0.026946]`, `score.py` 경고 `0`건), `ENS_rep_sj_s43` **`0.513561`** (`vs ENS3 +0.007561`, `[+0.004938, +0.010186]`). `s42+s43` 2-seed 확률 평균 **`0.526039`** (`score.py` `0.5260387`, vs `ENS3 +0.020038`, `[+0.015980, +0.023926]`), 2-seed 단독 평균 `0.525663` (`std=0.000730`).
  - **`4-2`:** `E026 β=128` 단독 `0.491817`, `ENS_rep` `0.498757`; `Dev nested 5-fold` 탐색에서 `SJ_2seed_avg` `0.494349`, `E001+E007+SJ42+JM42` `0.502317`(`4-1 0.513647`).
  - **보존 확인:** `final.pt`·`training_state_last.pt`·`dev_prefix.npz` 외장 SHA-256 일치, `72`시점 오프라인 CPU 추론 일치(`max_abs_err=0.000639`, 불일치 `0`건), `score.py` 재채점 완료.
- **후속 실행 중 (`18:00 KST` 착수):** 균형형 `0.5·SoftJaccard + 0.5·SmoothMacroDice`의 multi-seed 확인을 위해 같은 L4 세션에서 `m4_jacc_macrof1_v1_s43` (`seed 43`, `PID 128457`, GPU `96%`) 가동 중 (`30`분 뒤 확인 예정). Gap ③ 해결용 [`remote_run_oof.py`](../tools/m4_precue_bce/remote_run_oof.py)도 원격 `/content/`에 사전 업로드 완료.

## 2026-11-17 `m4_jacc_macrof1_v1_s43` (`seed 43`) 완료·검증 및 Gap ③ 5-fold OOF (`e002_oof`) 착수

- **`m4_jacc_macrof1_v1_s43` 완료 및 2-seed 검증 통과 (`L4`, `1,766.4초`):** [보충 기록](m4_jacc_macrof1_v1_20261116/SEED43_SUPPLEMENT.md), 외장 보관 `/Volumes/로컬 디스크/DCC/private_experiments/m4_jacc_macrof1_v1_20261116/s43/`.
  - **`4-1` (`Dev 2,920`, `raw p>=0.5`):** `JM_s43` 단독 **`0.515594`** (`vs E002 +0.012860`, `95% CI [+0.008555, +0.017313]`, `score.py` 경고 `0`건), `ENS_rep_jm_s43` **`0.509867`** (`vs ENS3 +0.003866`, `[+0.001629, +0.006056]`). `JM42+JM43` 2-seed 확률 평균 **`0.515999`** (`score.py` `0.516000`, vs `E002 +0.013266`, vs `ENS3 +0.009999`), 2-seed 단독 평균 `0.514509` (`std=0.001085`).
  - **`4-2`:** `E026 β=128` 단독 `0.492091`, `ENS_rep` `0.498993`; `Dev nested 5-fold` 탐색에서 `JM_s43` `0.494486`, `JM_2seed_avg` `0.497788`, `E001+E007+JM42+JM43` `0.498166`으로 소표본 `Dev` 내 중단 선택 변동이 확인돼 `fit 26,280` 5-fold OOF 검증 필요성을 재확인했다.
  - **보존 확인:** `final.pt`·`training_state_last.pt`·`dev_prefix.npz` 외장 SHA-256 일치, `72`시점 오프라인 CPU 추론 일치(`max_abs_err=0.000470`, 불일치 `0`건), `score.py` 재채점 완료.
- **Gap ③ 1차 `e002_oof` 중단 / 미회수 (`18:35 → ~20:36 KST`):** `Dev` 라벨 누수 없는 `fit(26,280)` 기반 중단 정책 학습·비교(`v0_e026_fit_only` vs `v1_nonempty_fit_only`)를 위해 L4 세션 `dcc-m4-stage2-l4-20261117`에서 `e002_oof` 5-fold OOF (`PID 137196`)를 실행했다. `19:31 KST` 확인 시 `fold0`·`fold1` 완료 및 `fold2` `step 1600/7884` 정상 진행 중이었으나, `10:08 KST` 할당된 런타임의 총 유지 시간(`~10시간 28분`, `20:36 KST` 전후 소모량 기준 `55.32 → 53.22`, `2.10`유닛 소모) 한계로 전체 5겹 합산 파일 회수 전 세션이 원격 반납됐다(`Active assignments: 0`, 잔여 `53.22` compute units).
- **Gap ③ 2차 `e002_oof` 중단 / 원인 규명 (`21:00 → ~21:25 KST`):** 새 L4 세션 `dcc-m4-oof-l4-20261117`에서 분리 프로세스 방식으로 재시도했으나 `~25분` 만에 세션이 회수됐다(`53.10 → 52.29`, `0.81`유닛 소모). Mac `pmset` 로그상 절전은 없었으며, 분리 실행 후 Jupyter 커널이 유휴 상태로 전환되고 활성 WebSocket 연결이 닫힌 것이 야간 L4 유휴 회수 트리거로 작용한 것으로 파악됐다.
- **Gap ③ 3차 `e002_oof` 완료 및 fit-only 4-2 검증 통과 (`22:04 → 00:31 KST`, 보존 검증 완료):** 새 L4 세션 `dcc-m4-oof-l4-20261117-v2`에서 포그라운드 heartbeat + fold별 즉시 백업([`run_oof_resilient.py`](../tools/m4_precue_bce/run_oof_resilient.py))으로 5개 겹(`fold0..4`, 각 `fit 21,024 / held 5,256`, 총 `8,322.5s`)을 완주해 `fit_oof_merged.npz` (`26,280`통화·`801,544`시점, OOF 4-1 `0.485636`, SHA-256 `713433c83a34...`)를 확보하고 원격 세션을 자동 정지했다(`Active assignments: 0`, 잔여 `47.54` compute units). [Gap ③ 결과 기록](mission4_e026_fitonly_policy_20261116/EXPERIMENT_RECORD.md), 정량 결과 `artifacts_private/runs/e026_fitonly_policy_v1_20261117/summary.json`.
  - **`Dev 2,920` 누수 차단 4-2 검증 (`fit_only_shift`, `ref_tau=0.5`, `10,000`회 bootstrap `seed=20261116`, `score.py` 경고 `0`건):**
    - **`ENS3`** (`4-1=0.506001`): `C0(β=0) 0.498835`(`F1=0.6136`) → `C128_v0 0.507400`(`+0.008565`, `[+0.003310, +0.013915]`) → **`fitbest_v1_nonempty(β=256)` `0.507855`** (`F1=0.6142, E=0.8268`, **`+0.009019`**, `95% CI [+0.003671, +0.014361]`, `p=0.9995`, 빈 예측 `44→21`)로 사전 기준(`Δ4-2 > 0.002`, CI 하한 `> 0`) 및 `Macro-F1` 손실 해소를 모두 달성.
    - **`E002` 단독** (`0.502733`): `0.492958 → C128_v0 0.502187`(`+0.009229`, `[+0.003716, +0.014795]`) / `fitbest_v1 0.502279`(`+0.009322`).
    - **개선 분류기 조합도 전 조건 통과:** `ENS_rep_jm42` (`4-1=0.509468`) `fitbest_v1 0.504877`(`+0.007989`, `[+0.002460, +0.013628]`), `ENS4_jm` (`4-1=0.512240`) `0.504803`(`+0.009774`), `ENS_rep_sj42` (`4-1=0.512504`) `0.503642`(`+0.009162`), `ENS4_sj` (`4-1=0.516301`) `0.502472`(`+0.008276`).
  - **안전·보존 점검:** 미래 발화·확률 차단 불변성 및 온라인 `HaltFeatureState` 일치(`200/200`통화 오차 `<1e-12`), `SHA256SUMS.txt` 체크섬 기록, 로컬 단위 테스트(`test_train_precue.py` 3건 `OK`) 통과.

## 2026-11-18 상위 검토 모델(`GPT-6-astra high` / Argon High) 최종 감사 및 채택 판정 완료

- **코드·원본 대조 감사 및 초안 오류 정정:**
  1. **Gap ③ `E026`(`0.503146`) 대 신규 `C128_v0`(`0.507400`) 차이의 실제 원인 규명:** 기존 `E026`의 `policy.json` 실물을 감사해 이미 `ref_tau=0.5`, `n_rows=670,151`이었음을 확인하고 초안의 `ref_tau=0.52` 추측을 기각·정정했다. 동일한 신규 `e002_oof` 위에서 `29,200 shift`(`0.507455`) 대 `fit_only_shift`(`0.507400`, 차이 `-0.000055`)를 분리 실험하여 **기존 `E026`의 이득이 `Dev` 라벨 누수 덕분이 아니었음**을 입증했고, `+0.00425` 점수 차이는 두 OOF(`E011_oof` vs `e002_oof`) 간 회귀 예측 `δ̂`(`r=0.9886`, 평균 절대차 `0.0053`)가 `128`배 증폭되며 `561/2,920`통화(`19.2%`)의 중단 시점이 이동한 결과(OOF 생성 간 변동폭 `~0.004`)임을 규명했다.
  2. **Gap ② `SoftJaccard` 작동 메커니즘 명확화:** `SJ_2seed_avg`(`0.526039`)의 `+0.0233` 상승은 `pre-cue FP`(`31,989 → 33,765`건)나 다중 증상 마지막 발화 Recall(`54.37% → 51.76%`) 해결이 아니라, **초반 빈 예측률 축소(`15.15% → 8.16~9.74%`)와 단일 증상 통화(`67.1%`) 일치도 극대화(`0.5631 → 0.5968`), `오심` 정밀도 개선(`0.307 → 0.342`)** 에 의한 것임을 분리해 기록했다.
  3. **산출물 보존 상태 구분 (`AGENTS.md` 준수):** Gap ② 5개 실행(`precue_bce_s42`, `softjaccard_s42/s43`, `jacc_macrof1_s42/s43`)은 가중치·재개 상태·확률표 전량 외장 보존 완료; Gap ③ `e002_oof`는 5개 겹 원확률표·설정·로그·병합표 보존 완료이나 **5개 보조 fold의 모델 가중치(`final.pt`)·재개 상태는 원격 종료 시 미보존**으로 구분 표기했다.
- **최종 채택 판정 (개발용 재검증 기준):**
  - `m4_precue_bce_v1`: **미채택 확정** (전역 확률 하향으로 `empty_rate` 증가 및 4-1 `-0.0056` 하락).
  - **Mission 4-1 주 채택:** **`m4_softjaccard_v1` 2-seed (`s42 + s43`, `Dev 4-1 = 0.526039`)** 또는 기존 앙상블 결합 **`ENS4_sj` (`E001+E007+SJ42+SJ43`, `0.516301`)**.
  - **Mission 4-2 주 채택:** **`ENS3 (E001+E002+E007)` + `e026_fitonly_policy` `C128_v1_nonempty` (`β=128` 사전 고정 + 빈 예측 중단 금지 게이트, `Dev 4-2 = 0.507411`, `C0` 대비 `+0.008575`, CI `[+0.003325, +0.013903]`)**. (`β=256` `0.507855`는 격자 상한 평탄 구간이므로 탐색 참고치로 분류.)
  - **단일 확률표로 4-1·4-2 통합 운용 시 조건부 채택:** **`ENS4_jm` (`E001+E007+JM42+JM43`, `4-1 = 0.512240`, `4-2 C128_v1 = 0.503482`)** 또는 **`ENS_rep_jm42` (`4-1 = 0.509468`, `4-2 C128_v1 = 0.504324`)**.
