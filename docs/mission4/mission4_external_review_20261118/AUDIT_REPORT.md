# Mission 4 독립 감사 보고서 (코드·정량·제출 구조 종합 검증)

- **작성일:** 2026-11-18
- **검토 모델:** Claude Opus 5.5 High (1~72스텝 원본 대조 및 3개 CPU 검산 스크립트 작성·실행 후 쿼터 도달로 중단) → **GPT-6-astra high** (Orchestrator Argon, [`AGENTS.md`](../../AGENTS.md) 지정 상위 검토 모델)가 검산 산출물 인수 및 최종 감사·판정 완결 (2026-11-18)
- **제약 준수 확인:**
  - 공식 `Validation` 데이터(`/Volumes/로컬 디스크/DCC/DCC_dataset/Validation/`) 열람·목록 조회·평가 **0회**
  - GPU·Colab 실행, 새 모델 학습, 원격 작업 **0회**
  - 기존 실험 코드·기록·JSON·가중치 무수정 (신규 작성 파일은 본 감사 보고서 1건 및 `scratch/` 검산 스크립트 3종만 해당)

---

## 1. 감사 범위 및 대조한 실제 원본·커밋·검산 스크립트

### 1.1 직접 열람·대조한 기준 문서·코드·정량 파일
1. **기준 문서 및 브리프:**
   - [`AGENTS.md`](../../AGENTS.md), [`Docs/COMPETITION_RULES_20261114.md`](../COMPETITION_RULES_20261114.md), [`Docs/EXPERIMENT_RECORDING_STANDARD.md`](../EXPERIMENT_RECORDING_STANDARD.md)
   - [`Docs/MISSION4_EXTERNAL_AGENT_REVIEW_BRIEF_20261118.md`](../MISSION4_EXTERNAL_AGENT_REVIEW_BRIEF_20261118.md) (감사 대상 브리프)
   - [`Docs/mission4_streaming_baseline_20261115/UPDATES_REVIEW.md`](../mission4_streaming_baseline_20261115/UPDATES_REVIEW.md)
   - [`Docs/mission4_e026_same_input_replay_20261116/EXPERIMENT_RECORD.md`](../mission4_e026_same_input_replay_20261116/EXPERIMENT_RECORD.md)
   - [`Docs/mission4_e026_fitonly_policy_20261116/PROTOCOL.md`](../mission4_e026_fitonly_policy_20261116/PROTOCOL.md), [`EXPERIMENT_RECORD.md`](../mission4_e026_fitonly_policy_20261116/EXPERIMENT_RECORD.md)
   - [`Docs/m4_precue_bce_v1_20261116/EXPERIMENT_RECORD.md`](../m4_precue_bce_v1_20261116/EXPERIMENT_RECORD.md), [`Docs/m4_softjaccard_v1_20261116/EXPERIMENT_RECORD.md`](../m4_softjaccard_v1_20261116/EXPERIMENT_RECORD.md), [`SEED43_SUPPLEMENT.md`](../m4_softjaccard_v1_20261116/SEED43_SUPPLEMENT.md), [`Docs/m4_jacc_macrof1_v1_20261116/EXPERIMENT_RECORD.md`](../m4_jacc_macrof1_v1_20261116/EXPERIMENT_RECORD.md), [`SEED43_SUPPLEMENT.md`](../m4_jacc_macrof1_v1_20261116/SEED43_SUPPLEMENT.md)
2. **Git 저장소 원격 브랜치 (`github_review/DCC_Problem`, 읽기 전용 `git show`):**
   - `origin/mission4-streaming` (최신 commit **`5d7539b5444f94073d3a4cd370447c128c12743e`**, `2026-11-15 15:50:59 +0900`): `mission4_symptom/{inference.py, SUBMISSION_README.md, m4/bundle.py, m4/streaming.py, m4/halting.py, scripts/pack_bundle.py, runs/*/run_config.json}`
   - 로컬 스냅샷 `artifacts_private/team_branch_review_20261114/snapshots/mission4-streaming/mission4_symptom/` (이전 commit `355f5b2788e275dbed103220ce242e44423491f7` 시점의 `docs/`, `m4/`, `runs/`, `scripts/` 부분 복사본)
   - 동일 입력 재생 소스 `artifacts_private/e026_same_input_replay_20261116/source/`
3. **정량 요약·정책·실험 코드 및 외장 산출물:**
   - `artifacts_private/runs/e026_fitonly_policy_v1_20261117/{summary.json, gap2_summary_cmp.json, policy_fitonly_c128.json}`
   - [`tools/m4_precue_bce/run_gap3_complete.py`](../../tools/m4_precue_bce/run_gap3_complete.py)
   - `/Volumes/로컬 디스크/DCC/private_experiments/` 하위 `e026_same_input_replay_20261116`, `e026_fitonly_policy_v1_20261117`, `e002_oof_20261117`, `e026_priority1_checks_20261116/models/E001`, `m4_softjaccard_v1_20261116/s4{2,3}`, `m4_jacc_macrof1_v1_20261116/s4{2,3}`

### 1.2 CPU 검산 스크립트 3종 및 실행 출력

| 스크립트 | 실행 명령 | 핵심 출력 요약 |
|---|---|---|
| [`recheck_41.py`](scratch/recheck_41.py) | `python3 Docs/mission4_external_review_20261118/scratch/recheck_41.py` | 7개 npz(`87,814`시점) 원본으로 4-1(`raw p>=0.5`, Top-1 없음)·빈 예측률·`\|p-0.5\|<1e-3` 경계 비율 재계산: `E002 0.502733 (15.15%, 0.00057)`, `ENS3 0.506001 (16.43%, 0.00051)`, `SJ42 0.526393 (8.16%, 0.00089)`, `SJ43 0.524933 (9.74%, 0.00060)`, `SJ_2seed_avg 0.526039 (9.74%, 0.00073)`, `JM_2seed_avg 0.515999 (12.99%, 0.00047)`, `ENS4_sj 0.516301 (13.82%, 0.00055)`, `ENS4_jm 0.512240 (14.81%, 0.00051)` |
| [`check_empty_at_halt.py`](scratch/check_empty_at_halt.py) | `python3 Docs/mission4_external_review_20261118/scratch/check_empty_at_halt.py` | `ENS3 C128` `2,920`통화 중단 배열 대조: `v0` 빈 예측 **`43`건**(`last_utt=16, mid_call=27`, `mean_halt=9.685`), `v1_nonempty` 빈 예측 **`21`건**(`last_utt=21, mid_call=0`, `mean_halt=9.739`, `n_utt<5=0`) |
| [`check_shift.py`](scratch/check_shift.py) | `python3 Docs/mission4_external_review_20261118/scratch/check_shift.py` | `policy_fitonly_c128.json` `logit_shift` 대 `summary.fit_only_shift` 차이 `0.0`. `E002`/`E007`/`SJ42`/`SJ43`/`JM42`/`JM43`(`fit 26,280`)은 `max\|diff\|=3.762e-08`로 `tol 1e-5` 통과; 반면 `E016_final/shift` 및 `final/prefix_s42,s43`(`29,200`)은 `6.858e-03`, `e002_oof fold0`은 `9.677e-03`으로 불통과. `E001`은 `pos_weight` 미기재(`None`) |

---

## 2. 브리프(`MISSION4_EXTERNAL_AGENT_REVIEW_BRIEF_20261118.md`) 대조 결과

### 2.1 정량 수치 검증 결과 (핵심 점수·신뢰구간 100% 일치 확인)
- [`recheck_41.py`](scratch/recheck_41.py), `summary.json`, `gap2_summary_cmp.json`을 원본 대조한 결과, 브리프 3.1~3.3절의 4-1 점수(`E002 0.502733`, `ENS3 0.506001`, `SJ42 0.526393`, `SJ43 0.524933`, `SJ_2seed_avg 0.526039`, `JM42 0.513424`, `JM43 0.515594`, `JM_2seed_avg 0.515999`, `ENS4_sj 0.516301`, `ENS4_jm 0.512240`), 4-2 점수(`ENS3 C0 0.498835 → C128_v0 0.507400 → C128_v1_nonempty 0.507411`, CI `[+0.003325, +0.013903]` 등 전 6개 대상), Step 2 분해(`96.02%`, `97.35%`, `28,218/36,299 = 77.74%`), `29k_shift` 대 `fit_only_shift` 차이(`-0.000055`)가 **소수점 6자리까지 모두 실제 산출물과 일치**함을 확인했다.
- 브리프 3.2절 표에서 `-`로 비워져 있던 빈 예측률은 검산 결과 `JM_2seed_avg = 12.99%`, `ENS4_jm = 14.81%`로 확인되었다.

### 2.2 브리프에서 발견된 오류·누락 4건

1. **스냅샷 경로 불일치 및 `m4/bundle.py` 줄 번호 오기 (`오류 1`):**
   - 브리프 5절·6절이 링크한 `artifacts_private/team_branch_review_20261114/snapshots/mission4-streaming/mission4_symptom/`에는 `docs/`, `m4/`, `runs/`, `scripts/`만 존재하며, 브리프가 직접 참조한 `inference.py`, `SUBMISSION_README.md`, `m4/bundle.py`, `scripts/pack_bundle.py` 파일이 **로컬 스냅샷 폴더에 없다.** 실제 최신 제출 코드는 git 저장소 `github_review/DCC_Problem`의 `origin/mission4-streaming` (`5d7539b`) 브랜치에 있다.
   - 또한 브리프 147행이 인용한 `m4/bundle.py:55` (`shared_pos_weight()`)는 실제 `origin/mission4-streaming:mission4_symptom/m4/bundle.py` 기준 **`57`행**이다(`55`~`56`행은 공백, `learned_halt_setting()`은 `101`행 일치).

2. **공식 `Validation (3,640통화)` 팀 선행 확인 이력 누락 (`오류 2`):**
   - 브리프 4절 5번(`116`행)은 *"공식 Validation 3,640은 1회도 열어보지 않은 상태"* 라고 기술했으나, 이는 **사용자의 이번 Gap ②·③ 작업 세션 기준**일 뿐이다.
   - [`Docs/mission4_streaming_baseline_20261115/UPDATES_REVIEW.md:21-27`](../mission4_streaming_baseline_20261115/UPDATES_REVIEW.md) 및 `origin/mission4-streaming` 커밋 로그에 따르면, 팀원 브랜치에서는 이미 `E017`(2026-11-13), `E025`(2026-11-14 `d5d32f3`), `E026`(2026-11-15 `5d7539b`, `4-1 = 0.486713, 4-2 = 0.506968`) **3차례에 걸쳐 공식 `Validation`을 실행한 이력**이 존재한다. 따라서 이번 최종 검증 실행은 전체 프로젝트 기준 "최초 1회"가 아니라 "Gap ②·③ 최종 후보 동결 후 1회 확인"으로 정확히 구분해야 한다.

3. **기존 `E026` / `C128_v0` 빈 예측 중단 건수(`43`건 vs `44`건) 및 CI 출처 표기 혼재 (`오류 3`):**
   - 브리프 2절(`43`행)은 `E026`의 빈 예측 중단 통화를 `44`건으로 적었으나, `e026_same_input_replay` 기록(`65`행)과 [`check_empty_at_halt.py`](scratch/check_empty_at_halt.py) 검산 결과 `E026` 및 `ENS3 C128_v0`의 빈 예측 건수는 **`43`건**(`중간 중단 27건 + 마지막 발화 도달 16건`)이다. `44`건은 `fitbest_v0_b256` (`β=256`)의 수치다.
   - 브리프 2절은 `E026` 개선 CI를 `[-0.000893, +0.009200]`(`wide/result.json`, `n_boot=1,000, seed=2026`)으로 인용했는데, 브리프 3.3절(`10,000`회, `seed=20261116`)과 통일된 동일 입력 재생 CI는 `[-0.001003, +0.009525]`(`p=0.9405`)이므로 두 설정을 구분 병기하는 것이 정확하다.

4. **`test_causality` 실제 검사 통화 수 및 게이트 미포함 범위 (`오류 4`):**
   - 브리프 3.3절(`77`행)은 *"`200`통화 대상 미래 차단 불변성과 온라인 `HaltFeatureState` 일치 오차 `< 1e-12`"* 로 요약했으나, [`run_gap3_complete.py:65-89`](../../tools/m4_precue_bce/run_gap3_complete.py) 확인 결과 `n_utt < 4`인 5통화를 건너뛰어 **실제 검사는 `195`통화**였고, `pol.should_halt()`만 검사해 **`v1_nonempty` 게이트는 온라인 일치 테스트 대상에 포함되지 않았다.**

---

## 3. 5가지 핵심 판단 항목에 대한 감사 판정·근거·권고

### 판단 1. 최종 제출 번들 구조 — 방안 A (역할 분리형) vs 방안 B (공유형)
- **판정:** **`조건부 동의 (방안 A를 주 제출 구조로 권고 + 방안 B 백업 준비)`**
- **근거:**
  1. **템플릿 제약 부합:** `origin/mission4-streaming:mission4_symptom/inference.py:16-18, 340-376`에서 참가자 수정 허용 영역은 `Predictor` 클래스뿐이며, `Predictor.__init__(ckpt_path, mission, device)`가 이미 `self.mission` (`"4-1"` 또는 `"4-2"`)을 전달받는다. 또한 `m4/bundle.py:27, 32`도 이미 4-1(`rule`)과 4-2(`halt_rule`/`halt`) 설정을 분리해 보관한다.
  2. **정량적·연산적 우위:** 방안 A(`4-1 = SJ_2seed_avg 0.526039`, `4-2 = ENS3 + C128_v1_nonempty 0.507411`)는 방안 B 최고 후보인 `ENS4_sj` (`4-1 = 0.516301`, `4-2 = 0.502448`)나 `ENS4_jm` (`4-1 = 0.512240`, `4-2 = 0.503482`) 대비 **4-1(`+0.0097 ~ +0.0138`)과 4-2(`+0.0039 ~ +0.0050`) 모두 앞선다.** 또한 방안 A는 추론 시 4-1은 `2`개, 4-2는 `3`개 멤버만 실행하므로 `4`개 멤버를 매번 도는 방안 B보다 빠르다.
  3. **규정상 명시 공백:** [`COMPETITION_RULES_20261114.md`](../COMPETITION_RULES_20261114.md) 및 `inference.py`에는 `--mission 4-1`과 `--mission 4-2`가 단일 `.pt` 내부에서 서로 다른 서브모델을 호출하는 것에 대한 **명시적 허용 또는 금지 조항이 없다.** (명시적 금지 없음 / 필요 시 운영진 문의 대상이나, 두 미션의 별도 실행 및 별도 채점식 특성과 `Predictor` 설계상 자연스러운 확장이다.)
- **권고 조치:**
  - 단일 `ckpt/mission4.pt` 내부에 `4-1`용 멤버(`SJ42, SJ43`)와 `4-2`용 멤버(`E001, E002, E007` + `C128_v1_nonempty`)를 역할 태그(`members_by_mission` 또는 호환 settings 키)로 구분해 저장하고, `Predictor`가 `self.mission`에 맞는 멤버만 로드·추론하도록 구현한다.
  - 정성 평가(발표)에서 *"4-1은 시점 평균 Jaccard 정렬 손실(`SoftJaccard` 2-seed), 4-2는 희소 클래스 `Macro-F1` 보존이 핵심이므로 BCE 3-앙상블 + 무누수 학습형 중단 정책으로 역할을 분화했다"* 는 논리를 명확히 제시한다.
  - 동시에 만약의 단일 모델 공유 규정에 대비해 **`ENS4_sj` (또는 팀 합의 시 `ENS4_jm`) 백업 번들 및 스위치**를 함께 생성해 둔다.

### 판단 2. 4-2 `C128_v1_nonempty` 게이트의 규정 적합성 및 구현 상태
- **판정:** **`조건부 동의 (규정 100% 적합 및 무누수 확인 / 온라인 제출 코드 미구현 상태 — 즉시 반영 필수)`**
- **근거:**
  1. **규정 적합성:** 게이트 조건 `(v(t) + 128·δ̂(t) >= 0.5) and any(raw_p(t) >= 0.5)` (`t >= 5`)는 중단 점수 기준 `0.5`와 증상 판정 임계값 `0.5`를 동시에 만족할 때만 중단하며, 현재 시점 `t`까지의 누적 전사·시각·확률만 사용해 **미래 정보를 일절 참조하지 않는다.**
  2. **남은 `21`건 `empty_at_halt`의 정확한 원인 규명:** [`check_empty_at_halt.py`](scratch/check_empty_at_halt.py) 검산 결과, `v1_nonempty`에서 통화 중간에 빈 예측으로 조기 중단된 건수는 `v0` **`27`건 → `0`건**으로 완전히 사라졌다. 남은 `21`건은 **전부 `halt == n_utt` (`empty_at_last_utt = 21`)**, 즉 통화 끝(`finalize()`)까지 도달했는데도 `raw_p >= 0.5`를 넘는 증상이 끝내 없어 빈 리스트를 출력한 통화다(`v0`에서도 마지막까지 가서 빈 예측이었던 `16`건 + `v0`에서 중간에 성급히 끊겼다가 게이트 덕분에 끝까지 들었으나 끝내 `0.5` 미만으로 남은 `5`건). `n_utt < 5`인 짧은 통화도 `0`건이었다.
  3. **중요 코드 누락 발견:** 현재 이 게이트는 오프라인 평가 스크립트 [`run_gap3_complete.py:189`](../../tools/m4_precue_bce/run_gap3_complete.py#L189)에만 `cond & ev_hd.pred.any(axis=1)`로 구현되어 있으며, 제출용 온라인 코드 `origin/mission4-streaming:mission4_symptom/m4/halting.py:474-485` (`HaltPolicy.halts`, `HaltPolicy.should_halt`)와 `m4/streaming.py`, 그리고 [`policy_fitonly_c128.json`](../../artifacts_private/runs/e026_fitonly_policy_v1_20261117/policy_fitonly_c128.json)에는 **아직 들어 있지 않다.**
- **권고 조치:**
  - `HaltPolicy` (`require_nonempty: bool = True` 필드) 및 `should_halt()` / `halts()`와 `policy_fitonly_c128.json`에 이 게이트를 정식 통합한다 (`pred`는 `raw_probs >= 0.5` 여부로 판정).
  - 게이트 포함 상태에서 [`run_gap3_complete.py:61-91`](../../tools/m4_precue_bce/run_gap3_complete.py#L61-L91)과 같은 온라인/일괄 일치 검증 및 `Dev 2,920` 공식 `inference.py` → `score.py` E2E 검증을 다시 통과시킨다.

### 판단 3. `β=128` 사전 고정 채택 대 `β=256` OOF 최고점 선택
- **판정:** **`동의`**
- **근거:**
  1. [`Docs/mission4_e026_fitonly_policy_20261116/PROTOCOL.md:31`](../mission4_e026_fitonly_policy_20261116/PROTOCOL.md)는 *"β=128은 이번 실험 전에 고정한다. 새 β 탐색...을 섞지 않는다"* 고 명시했다. 따라서 사전 등록 기준인 `β=128`을 주 채택으로 유지한 것은 연구 규율(`AGENTS.md`)에 정확히 부합한다.
  2. `fit 26,280` 내부 5-fold OOF 격자에서 `β=128` (`0.494581`) 대비 `β=256` (`0.494725`, 격자 상한) 차이는 `+0.000144`에 불과한 평탄 구간이고, 높은 `β`일수록 OOF 재생성 오차 증폭(`128×` vs `256×`) 위험이 커진다.
- **권고 조치:** 사전 등록 원안 `C128_v0` (`0.507400`)와 비정상 조기중단 방어 게이트를 추가한 `C128_v1_nonempty` (`0.507411`, `empty 43→21`)의 관계를 보고서에 투명하게 병기하여 채택 근거로 삼는다.

### 판단 4. 잔여 Colab `47.54`유닛 활용 — `jm_oof` 및 `Training 29,200` 전체 재학습 여부
- **판정:** **`둘 다 불필요·반대 (검증된 fit 26,280 가중치 그대로 제출 권고)`**
- **근거:**
  1. **`jm_oof` (~4유닛):** 방안 A(`4-1: SJ 2-seed`, `4-2: ENS3 + C128_v1`)를 채택하면 `JM` 전용 OOF가 필요하지 않다. 또한 `JM_2seed` 단독 `final_step Macro-F1` (`0.6526`)이 이미 `ENS3` (`0.6562`)보다 낮아 `ENS4_jm`이 `ENS3 + C128_v1` (`0.507411`)을 역전한다는 보장이 없다.
  2. **`Training 29,200` 전체 재학습 (~2유닛) — 심각한 `logit_shift` 충돌 확인:**
     - [`check_shift.py`](scratch/check_shift.py) 실행 결과, `policy_fitonly_c128.json`의 `logit_shift`는 `summary.json`의 `fit_only_shift` (`26,280`통화 기준)와 정확히 일치한다 (`diff = 0.0`).
     - **`fit 26,280` 모델들(`E002`, `E007`, `SJ42`, `SJ43`, `JM42`, `JM43`)** 은 `policy_fitonly_c128.json`의 `logit_shift`와의 최대 차이가 **`3.762e-08`** 로 `m4/bundle.py:101` `learned_halt_setting(tol=1e-5)` 검사를 **모두 통과**한다.
     - 반면 **`Training 29,200` 전체 재학습 모델(`runs/final/prefix_s42`, `s43`)** 의 `log(pos_weight)`는 `fit_only_shift`와 **`6.858e-03`** 차이가 나서 `tol=1e-5` 검사를 통과하지 못하고 **`ValueError`로 중단**된다. (이전 `E026`은 `E016_final/shift`의 `29,200` 보정으로 학습된 정책이라 통과했으나, 이번 무누수 `policy_fitonly_c128`과 `29,200` 모델을 섞으면 보정 불일치 또는 재학습 비용이 발생한다.)
     - 게다가 `29,200`으로 재학습하면 `Dev 2,920`에서 회귀·점수 재확인이 불가능해지고, 팀원 브리프상 제출 기한(`2026-11-20`)이 임박해 회귀 검증·오프라인 일치를 새로 보장해야 하는 리스크가 크다.
- **권고 조치:** 추가 GPU 학습 없이 이미 `final.pt`·`training_state_last.pt`·SHA-256·72시점 CPU 오프라인 추론까지 전량 검증된 **`fit 26,280` 가중치 그대로** 제출 번들을 구성한다. (단, 4-1 전담 `SJ42+SJ43`만 4-2 정책과 분리해 `29,200` 2-seed를 추가로 고려하더라도 검증 불가능 리스크가 있어 권장하지 않는다.)

### 판단 5. `mission4_symptom` 코드 통합 시 기술적 함정 점검
감사 중 `origin/mission4-streaming` 코드와 [`check_shift.py`](scratch/check_shift.py), [`recheck_41.py`](scratch/recheck_41.py)에서 아래 **6가지 구체적 통합 함정**을 확인했다.

1. **[치명] `scripts/pack_bundle.py`와 `m4/bundle.py`의 기본값이 규정 위반 (`maxent` / `force_top1=True`):**
   - `m4/bundle.py:27, 32`의 `DEFAULT_SETTINGS`는 `"rule": {..., "force_top1": True}`, `"halt_rule": {..., "force_top1": True}`다.
   - `scripts/pack_bundle.py:45, 54, 104`는 `--rule` 기본값이 **`"maxent"`** 이고 `--no-force-top1`을 주지 않으면 `top1 = True`가 된다.
   - **해결:** 패키징 시 반드시 `--rule threshold --no-force-top1`을 강제하고(코드 기본값도 `force_top1=False`로 안전화), 생성된 번들의 `settings["rule"]`·`settings["halt_rule"]`이 모두 `{"type": "threshold", "threshold": 0.5, "force_top1": False}`인지 단위 테스트로 검증한다.
2. **[치명] `HaltPolicy` 온라인 `require_nonempty` 게이트 미구현:**
   - `m4/halting.py:479-484` `HaltPolicy.should_halt()`는 `raw_probs`의 `>= 0.5` 여부를 확인하지 않고 `signal >= self.tau`만 반환한다.
   - **해결:** `HaltPolicy`에 `require_nonempty` 속성을 추가해 `policy_fitonly_c128.json`에 기록하고, `StreamingSymptomModel.should_halt()` (또는 `HaltPolicy.should_halt()`)에서 `any(self._probs >= 0.5)`를 함께 확인한다.
3. **[주의] `E001` `pos_weight` 미기재 처리:**
   - [`check_shift.py`](scratch/check_shift.py) 확인 결과, `E001`의 `inference_config.json`에는 `pos_weight` 키가 없어 `member_pos_weight()`가 `None`을 반환한다 (`unknown` 목록 진입). `E002`와 `E007` (`fit 26,280`)의 `pos_weight`가 `fit_only_shift`와 `3.762e-08` 이내로 일치하므로 `E001+E002+E007` 묶음 시 `shared_pos_weight()`는 `E002/E007` 값을 채택해 정상 동작한다.
4. **[주의] `StreamingSymptomModel`의 입력 형식(`fmt="space"`, `max_length=512`, `truncation_side="right"`) 일치 확인:**
   - `E002`, `E007`, `SJ42`, `SJ43`, `JM42`, `JM43` 모두 동일하게 `fmt="space"`, `max_length=512`, `truncation_side="right"`임을 [`check_shift.py`](scratch/check_shift.py)로 확인했다 (`E001`은 `max_length=512`, 예선 `Mission 3` 모델로 `fmt="space"` 기반).
   - 방안 A 구현 시 번들 크기는 5멤버(`SJ42, SJ43, E001, E002, E007`) 기준 fp16 약 `1.1 GiB` 수준이며, `load_bundle()`이 `self.mission`에 해당하는 멤버만 메모리에 올려(예: `4-1`은 2개, `4-2`는 3개) 로드 시간·VRAM·추론 시간을 최적화하도록 구현하는 것이 안전하다.
5. **[경미] GPU `fp16` 변환 영향 확인 완료:**
   - `m4/bundle.py:163-164`는 CUDA 실행 시 `model.half()`로 추론한다. [`recheck_41.py`](scratch/recheck_41.py) 검산 결과 `Dev 87,814`시점 중 `|p - 0.5| < 1e-3` 경계 비율은 `0.047% ~ 0.089%` (`SJ_2seed_avg 0.073%`, `ENS3 0.051%`)로 매우 작으나, 최종 `ckpt/mission4.pt` 생성 후 실제 CUDA/CPU 환경에서 공식 `inference.py`로 `Dev 2,920` 재현성을 반드시 대조해야 한다.
6. **[경미] `inference.py` 구버전 docstring 및 fallback 설정 정리:**
   - `origin/mission4-streaming:mission4_symptom/inference.py:348-352`의 `Predictor` docstring이 구버전(maxent/항상 1개 이상 등)을 설명하고 있어 규정(`0.5` 고정·빈 리스트 허용)에 맞게 갱신해야 한다.
   - 또한 `pack_bundle.py:50, 101`의 `lightgbm` 미설치 시 fallback (`expected`, `tau=0.50`)에 대해 `requirements.txt` 내 `lightgbm` 필수 포함 여부를 점검한다.

---

## 4. 확인하지 못한 항목과 이유

1. **팀원 C (`Full vs Uniform Prefix`) 실행 결과:** 현재 합류 대기 중이어서 이번 감사 범위에 포함되지 않았다.
2. **공식 `Validation (3,640통화)` 성능:** 절대 금지 제약(`Validation` 미접근 원칙)에 따라 열람·실행하지 않았다. (팀 브랜치에서 선행 확인된 `E026 Validation` 수치와 이번 Gap ②·③ 후보의 실제 `Validation` 비교는 최종 번들 동결 후 1회 검증 시 확인 가능.)
3. **`E001` 원 학습 `fit` ID 및 `pos_weight` JSON 원문:** 외장 `models/E001/inference_config.json`에 `pos_weight`·학습 ID 목록이 기록되어 있지 않아 `E001`의 원 학습 `pos_weight` 일치 여부는 미확인 상태로 남긴다(`UPDATES_REVIEW.md:50`과 동일).
4. **`e002_oof` 보조 fold 모델 가중치(`final.pt`):** 원격 종료 시 삭제 정책에 따라 로컬에 확률표(`oof_prefix.npz`)·병합표(`fit_oof_merged.npz`)·설정·로그만 보존되고 보조 가중치는 미보존이다. (제출에 직접 쓰이는 모델은 아님.)

---

## 5. 최종 제출 전 우선순위 조치 목록

1. **[P0 — 필수] `mission4_symptom` 코드 통합 (로컬 작업):**
   - 최신 `origin/mission4-streaming` (`5d7539b`) 코드를 기반으로 작업 브랜치를 구성하고:
     - `m4/halting.py` · `m4/streaming.py` · `policy_fitonly_c128.json`에 `require_nonempty` (`any(raw_probs >= 0.5)`) 게이트 구현
     - `m4/bundle.py` · `streaming.py` · `inference.py` (`Predictor` 및 docstring)에 방안 A 역할 분리 번들 로딩(4-1: `SJ42+SJ43`, 4-2: `E001+E002+E007 + C128_v1_nonempty`) 및 기본값 `force_top1=False` 반영
     - 백업용 방안 B (`ENS4_sj`) 패키징 커맨드도 함께 문서화
2. **[P0 — 필수] 통합 회귀·온라인 일치·`Dev 2,920` E2E 검증 (로컬 CPU/GPU):**
   - `pytest` 전체 + 게이트(`v1_nonempty`) 포함 온라인 `StreamingSymptomModel` 대 오프라인 `run_gap3_complete.py` (`195+`통화/전체 `Dev 2,920`) 결과 일치(`< 1e-5`, 중단 step 100% 일치) 및 `score.py` 경고 `0`건 확인 (`4-1 = 0.526039`, `4-2 = 0.507411` 재현).
   - 인터넷 차단(`local_files_only`) 및 `lightgbm` 포함 `requirements.txt` 동작 확인.
3. **[P1 — 합류] 팀원 C (`Full vs Uniform Prefix`) 결과 합류 및 발표 논리 정리:**
   - Step 3 결과를 본선 스토리(`Full vs Prefix` → `SoftJaccard` 4-1 개선 + `fitonly_c128_v1` 4-2 무누수·빈예측 차단)와 연결한다. (이 시점 전까지 추가 GPU 학습 없음.)
4. **[P1 — 최종 1회] 모든 구성 동결 후 공식 `Validation` 1회 최종 검증 및 제출물 패키징:**
   - 모든 코드·정책·체크포인트를 동결한 뒤 공식 `inference.py --mission 4-1` / `--mission 4-2` 명령으로 `Validation` 1회 검증(결과 기반 재튜닝 금지) 후 SHA-256 영수증과 함께 제출물을 확정한다.
