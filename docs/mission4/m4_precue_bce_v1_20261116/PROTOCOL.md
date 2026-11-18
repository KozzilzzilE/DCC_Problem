# `m4_precue_bce_v1_20261116` 사전 실험 프로토콜 (Step 2 병목 직접 개선)

- 작성일: 2026-11-16
- 작성·실행 모델: Claude Sonnet / Opus 계열 에이전트 (참고: `AGENTS.md` 권장 외부 `GPT-6.1-sol / GPT-6-astra` 사후 검토는 본 실행 후 별도 점검 대상으로 구분 기록)
- 연결 근거: [Step 2 4-1 오류 분해 보고서](../mission4_step2_error_decomposition_20261116/REPORT.md), [대회 공식 규정](../COMPETITION_RULES_20261114.md), [사전 진단 수치](../../artifacts_private/runs/m4_pre_gpu_diag_20261116/summary.json)

---

## 1. 왜 하는가 (관찰된 오류와 가설)

1. Step 2 분해에서 `ENS3` (`Dev 2,920`) 전체 오탐(FP `36,299`건)의 **`77.7%`(`28,218`건)** 가 해당 증상의 어휘 단서가 등장하기 전(`pre-cue`)에 발생했으며, Prefix 모델(`E002` `31,989`건, `E007` `32,210`건)이 통화 전체 모델(`E001` `27,606`건)보다 `pre-cue` 오탐을 약 `+4,400`건 더 냈다.
2. `train_prefix.py` 및 `fit(26,280)` 프로파일링을 대조한 결과, `E002`는 균등 Prefix `157,680`표본 중 아직 단서가 나오지 않은 **`30,938`개의 `pre-cue` 양성 쌍(`13.73%`)** 에도 통화 끝 정답과 똑같이 `pos_weight^0.5`(`1.83~3.02`)를 곱해 학습한다. 이로 인해 단서 등장 직전(`t_kw - 1`) 평균 확률이 `0.4648`(`E002`)까지 올라가 `0.5` 경계 바로 앞에서 대량 오탐을 유발한다.
3. 기존 `E024`(`pos_weight^0.3` 전역 축소)는 단서 등장 후·다중 증상 양성 손실까지 함께 낮춰 빈 예측이 `15.1% → 19.0%`로 늘고 점수가 떨어졌다(`0.502733 → 0.500408`).
4. **가설:** 추론 규칙(`raw p >= 0.5`, Top-1 없음)은 전혀 건드리지 않고, **학습 시 `fit` 내부의 단서 등장 전(`t < t_cue`, 단 통화 내내 단서가 없는 예외 양성은 유지) 구간에서만 `pos_weight` 배수를 `1.0`으로 낮추고 단서 등장 후(`t >= t_cue`)와 미등장 양성은 기존 `(neg/pos)^0.5`를 유지**하면, Bayes 최적 확률 과신을 억제해 `pre-cue` 오탐을 크게 줄이면서도 `post-cue`·다중 증상 재현율을 지킬 수 있다.

---

## 2. 무엇을 통제하고 무엇만 바꾸는가

| 항목 | 기준선 (`E002_prefix_uniform`, `seed 42`) | 후보 (`m4_precue_bce_v1_s42`) |
|---|---|---|
| 초기화 | `mission3_symptom/runs/devsel/tapt` | 동일 (`tapt`, SHA-256 `25424975...`) |
| 학습/개발 분할 | `fit_ids.txt` (`26,280`) / `dev_ids.txt` (`2,920`) | 동일 (무중복·원문 비중복 검증, `Validation` 미사용) |
| 입력·샘플링 | `fmt=space`, `max_length=512`, `right` 절단, `3epoch`, `samples_per_call=2`, `rng([seed, epoch])` | **완전 동일** |
| 최적화 | `batch_size=16`, `lr=5e-5`, `llrd=0.8`, `wd=0.01`, linear warmup `10%`, FP16 AMP | **동일** |
| **단일 변경 변수** | 전 구간 `pos_weight = (neg/pos)^0.5` | **`y=1` & `cue_step < 10^9` & `t < cue_step`에서만 `pos_weight = 1.0`** (`t >= cue_step`·uncued·`y=0`은 동일) |
| 판정·추론 규칙 | `raw p >= 0.5`, Top-1 없음 | **동일 (`0.5` 고정 규정 100% 준수)** |
| 보존 장치 | `best_model`만 저장 | `best_model` + `final.pt` + **`training_state_last.pt` (model/optimizer/scheduler/scaler/RNG)** |

---

## 3. 실행 단계 및 사전 판정 기준

- **하드웨어 제한:** Colab **`NVIDIA L4`** (또는 `T4`)만 사용 (`A100/H100` 미사용).
- **사전 점검(Stage 0):** 로컬 단위 테스트(`test_train_precue.py`: BCE 식 패리티 `<1e-6`, 누수 차단, CPU 2스텝 중단·재개 bit-identity) 통과 후, 원격 L4에서도 실제 2스텝 중단·재개 확인.
- **Stage 1 (`seed 42` 1회 학습 + `Dev 2,920` 예측):**
  - 주 비교: `m4_precue_bce_v1_s42` vs `E002_prefix_uniform (0.502733)` 및 `E001 + 후보 + E007` vs `ENS3 (0.506001)`.
  - 관찰 지표: 4-1 전체/단일/다중 점수, `pre-cue` FP 변화(`31,989` 대비), `post-cue` Recall(`95.50%` 대비), `오심`/`전신쇠약`/`어지러움` Precision/F1, `fit_shift` 기반 4-2(`beta=0` 및 `v1_nonempty_gate` 중첩 CV).
  - **Stage 2 진행 기준:** `seed 42` 단독 4-1이 `E002` 대비 **+0.002 이상** 이거나, `ENS3` 교체 시 `95% CI 하한 > 0` 방향의 뚜렷한 개선(예: `pre-cue` FP 대폭 감소 + 4-1/4-2 비악화 이상)을 보이면 **`seed 43` (vs `E007 0.504211`)** 및 **5-fold OOF (`Gap ③` 누수 없는 4-2 검증)** 로 즉시 확장한다. 미달 시 Step 2 오답 분해로 원인을 기록하고 다음 단일 레버(후반 다중 증상·명시 부정 하드네거티브/`left` 절단)로 전환한다.
