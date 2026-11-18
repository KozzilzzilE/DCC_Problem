# m4_jacc_macrof1_v1_20261116 — SoftJaccard + SmoothMacroDice 결합 손실의 4-1·4-2 동시 개선 검증 (`seed 42`)

- 작성/갱신일: 2026-11-17
- 실행 담당: 보조 코딩 에이전트 (`AGENTS.md`상 상위 설계·최종 채택 검토 모델 `GPT-6.1-sol` / `GPT-6-astra` 미실행 상태)
- 현재 상태: **실행 완료 / 보존 검증 완료** (1차 `2026-11-16 22:04` 실행은 Mac 절전 중 Colab 유휴 회수로 유실, 2차 `2026-11-17 16:47` L4 재실행 완료 및 전체 산출물·해시·오프라인 추론 검증 완료)
- 이전 실험·관찰 자료: [SoftJaccard 실험 기록](../m4_softjaccard_v1_20261116/EXPERIMENT_RECORD.md), [Step 2 오류 분해 보고서](../mission4_step2_error_decomposition_20261116/REPORT.md), [사전 계획](PROTOCOL.md)
- 실제 실행 코드·보관 위치:
  - 학습/추론/평가 코드: [train_precue.py](../../tools/m4_precue_bce/train_precue.py), [predict_prefix.py](../../tools/m4_precue_bce/predict_prefix.py), [eval_precue.py](../../tools/m4_precue_bce/eval_precue.py), [remote_run_jacc_macro.py](../../tools/m4_precue_bce/remote_run_jacc_macro.py)
  - 정량 평가 원본: `/Volumes/로컬 디스크/DCC/private_experiments/m4_jacc_macrof1_v1_20261116/s42/eval_precue.json`
  - 외장 보존 사본 (`light/` + `weights/` + `weights.zip`): `/Volumes/로컬 디스크/DCC/private_experiments/m4_jacc_macrof1_v1_20261116/s42/`

## 1. 왜 이 실험을 하는가

선행 `m4_softjaccard_v1_s42`(`BCE + 1.0·SoftJaccard`)는 `Dev 2,920`에서 Mission 4-1 단독 점수를 `0.502733 → 0.526393`으로 크게 끌어올렸으나, 샘플별 Jaccard 가중이 고빈도 증상 위주로 작용하면서 저빈도 `오심`(`final F1 0.3613 → 0.3289`)·`두통`(`0.5621 → 0.5503`)이 약화되어 4-2 `Macro-F1` 및 단독 중단 점수(`E026 β=128`: `0.500064 → 0.493339`)가 하락하는 상충을 보였다.

검증할 질문: `BCE(pos_weight^0.5)`에 `0.5·SoftJaccard + 0.5·SmoothMacroDice`를 결합하면 4-1 개선을 일정 수준 유지하면서 4-2 `Macro-F1` 손실을 방어하거나 오히려 개선할 수 있는가.

## 2. 무엇을 어떻게 비교하는가

| 구분 | 기준선 (`E002`, `ENS3`) | 변경 조건 (`m4_jacc_macrof1_v1_s42`) | 공통으로 고정한 것 |
|---|---|---|---|
| 목적함수 | `BCE(pos_weight^0.5)` | `BCE(pos_weight^0.5) + 0.5·SoftJaccard + 0.5·(1 - SmoothMacroDice)` | `TAPT` backbone, `fit 26,280`, `dev 2,920` 미포함(`--forbid-ids`), `3 epoch`, `samples-per-call=2`, `batch=16`, `lr=5e-5`, `LLRD=0.8`, `seed=42`, `max_length=512`, `fmt=space`, `raw p>=0.5`, Top-1 미사용 |

- `SmoothMacroDice` 수식 (`train_precue.py:125-130`): 배치 내 각 증상 `c`별로 `macro_dice_c = (2 * sum(p * y) + 1) / (sum(p) + sum(y) + 1)`을 계산하고 `1 - mean_c(macro_dice_c)`를 `0.5` 가중으로 합산.
- 초기 모델(`TAPT` `model.safetensors`, classifier head `initial_state_sha256: 79fba79824755a486e608adc4c7901bbe43d7f9d988e172053cf1889911fd15f`), `fit_ids.txt`(`26,280`, `2f4bd4c3...`), `train.jsonl`(`b58498df...`) 모두 기준과 동일.
- 공식 `Validation` 미사용. `Dev 2,920`(`87,814`시점)에서 `4-1` 단독(`vs E002`) 및 `ENS_rep (E001 + JM_s42 + E007)`(`vs ENS3`) 비교, 더불어 기존 `ENS3` 기반 고정 `E026 β=128` 평가뿐 아니라 **각 후보 모델 자체 확률 궤적 위에서 `nested 5-fold`로 중단 정책(`beta`, `nonempty`)을 다시 맞춘 `4-2`**도 별도 측정한다.

## 3. 실행 전에 정한 판단 기준

- 주 지표 1 (`4-1`, `raw p>=0.5`): `ENS_rep > ENS3`(`0.506001`) 및 단독 > `E002`(`0.502733`), `10,000`회 paired bootstrap 95% 신뢰구간 하한 > 0.
- 주 지표 2 (`4-2`, `Macro-F1 × 조기성`): `SoftJaccard` 단독 대비 `Macro-F1` 회복 및 자체 확률 기반 `nested` 4-2에서 `ENS3` 동등 이상.

## 4. 실제 수행과 결과

- 1차 실행(`2026-11-16 22:04`, `dcc-m4-precue-l4-20261116`)은 Mac 절전 중 런타임 유휴 회수로 유실.
- 2차 실행(`2026-11-17 16:47–17:17 KST`, `dcc-m4-stage2-l4-20261117`, `NVIDIA L4 23,034 MiB`):
  - 전체 소요 `1771.9s` (`training_seconds: 1627.6s`, `peak_cuda_memory_bytes: 5,006,525,952`), `9,855` steps (`updates: 9,851`, `amp_skips: 4`, `train_loss`: `1.1387 → 1.0257 → 0.9717`).

### Mission 4-1 (`Dev 2,920`, `87,814`시점, `raw p>=0.5`, Top-1 없음)

| 조건 | 4-1 전체 | 단일(`1,958`) | 다중(`962`) | 빈 예측률 | 평균 예측 수 |
|---|---|---|---|---|---|
| `E002` (기준) | `0.502733` | `0.563053` | `0.379961` | `15.15%` | `1.2160` |
| `SoftJaccard_s42` (참고) | `0.526393` | `0.596587` | `0.383525` | `8.16%` | `1.2758` |
| **`JM_s42` (`0.5·SJ + 0.5·MacroDice`)** | **`0.513424`** | **`0.579092`** | `0.379768` | **`12.59%`** | `1.2267` |
| `ENS3` (`E001+E002+E007`) | `0.506001` | `0.569274` | `0.377218` | `16.43%` | `1.1682` |
| **`ENS_rep_jm` (`E001+JM_s42+E007`)** | **`0.509468`** | **`0.574824`** | `0.376446` | `15.44%` | `1.1706` |

- paired bootstrap(`10,000`회):
  - `JM_s42 - E002 = +0.010691` (`95% CI [+0.006794, +0.014600]`, `p_gt0=1.0`)
  - `ENS_rep_jm - ENS3 = +0.003467` (`95% CI [+0.001495, +0.005548]`, `p_gt0=0.9994`)
- 공식 채점기 `score.py` 재채점: `0.5134242181219014`, 경고 `0`건.
- `JM_s42`는 전 구간(`step_1_3` `0.1375→0.1492`부터 `final_step` `0.6333→0.6426`)에서 `E002`를 상회하며 `final_step Macro-F1`도 `0.6503`(`E002 0.6502`)으로 회복됐다(`오심 final F1`: `SJ_s42 0.3289 → JM_s42 0.3524`, `E002 0.3613`). 단, `pre-cue FP` 자체는 `E002 31,989 → JM_s42 33,047`로 줄지 않았으며, 짧은 부정 답변 뒤 FP 비율은 `0.0819 → 0.0760`으로 소폭 감소했다.

### Mission 4-2 (`Dev 2,920`, `fit_only_shift`)

- `ENS3` 위에 적합된 고정 `E026 β=128` 정책 대입 시: `JM_s42` 단독 `0.495906` (`Macro-F1=0.5842`, `earliness=0.8488`, `halt_median=6.0`, `E002 0.500064`), `ENS_rep_jm` `0.503139` (`Macro-F1=0.6020`, `earliness=0.8358`, `ENS3 0.503146`).
- **핵심 관찰 — 모델 자체 확률 궤적에 맞춘 `Dev nested 5-fold` (`v1_nonempty` 게이트, 탐색적):**
  - `ENS3`: `0.501420` (`Macro-F1=0.6008`, `earliness=0.8346`)
  - `SJ_s42`: `0.494407` (`Macro-F1=0.5972`, `earliness=0.8279`)
  - **`JM_s42` 단독**: **`0.501743`** (`Macro-F1=0.6072`, `earliness=0.8263`)
  - **`ENS_rep_jm` (`E001+JM_s42+E007`)**: **`0.503795`** (`Macro-F1=0.6073`, `earliness=0.8296`, `ENS3` 대비 `+0.002375`)
- 해석: `SJ_s42`와 `JM_s42`가 기존 고정 `E026` 정책에서 `4-2`가 낮아 보였던 주된 이유 중 하나는 BCE 기반 모델보다 이른 시점에 확률이 높아져 기존 정책이 조기 중단(`halt_median 5~6`)을 걸었기 때문이며, 자체 궤적에 맞춰 중단 정책을 재적합하면 `JM_s42`는 **단독 모델만으로도 4-1(`0.5134` vs `0.5060`)과 4-2(`0.5017` vs `0.5014`) 모두 `ENS3`를 넘고**, `ENS_rep_jm`은 4-1(`+0.0035`)과 4-2(`+0.0024`)를 동시에 개선한다. (단 `Dev` 내부 5-fold 탐색 결과이므로 `fit 26,280 OOF` 기반 독립 검증 전까지는 탐색적 증거로 구분한다.)

## 5. 결정과 다음 작업

- 판정: **단일 seed(`42`) 유망 후보 확보 (`4-1`·`4-2` 균형형 1순위; `4-1` 극대화는 `SoftJaccard` 우위)** — 단, `seed 43` 반복 검증과 `fit 26,280 OOF` 기반 `4-2` 누수 제거 비교가 남아 있어 최종 채택은 multi-seed 및 OOF 검증 후 결정한다.
- 다음 작업:
  1. `m4_softjaccard_v1_s43` (`seed 43`) 실행으로 `SoftJaccard` 재현성 확인 (진행 중).
  2. 필요 시 `m4_jacc_macrof1_v1_s43` (`seed 43`) 및 `Gap ③` 해결을 위한 `fit(26,280)` 5-fold OOF 중단 정책 재학습·검증 진행.

## 6. 산출물 보존 확인

| 항목 | 실제 경로/해시 | 상태·확인 일시 |
|---|---|---|
| 실행 설정·초기화·옵티마이저 감사·로그 | `/Volumes/로컬 디스크/DCC/private_experiments/m4_jacc_macrof1_v1_20261116/s42/light/{run_config,initialization,optimizer_audit,remote_quick_summary}.json`, `train_log.txt`, `runner_stdout.log` | 실물·해시 일치 확인 (`2026-11-17 17:19 KST`) |
| 가중치 ZIP 아카이브 (`132`청크 결합) | `.../s42/m4_jacc_macrof1_s42_weights.zip` (`2,213,595,825` B, `sha256: b7cc16c1d55d27b667b1e968f3750399ae7cad7d9bb8b4b5f69948de8a77d36a`) | 무결성 확인 (`2026-11-17 17:22 KST`) |
| 최종 가중치 + `best_model` + 토크나이저 | `.../s42/weights/content/runs/m4_jacc_macrof1_v1_s42/final.pt` (`442,552,911` B, `sha256: 4ee32f864e2fcbfc8a27460089a96e2a2645c42c9a46542909b6e710fbfde335`), `best_model/` | 원격 로그 SHA-256 일치 확인 (`2026-11-17 17:22 KST`) |
| 마지막 학습 재개 상태 | `.../s42/weights/.../training_state_last.pt` (`1,327,762,755` B, `sha256: 4cedc06b68e6ca741147ce28833c8b5ba95892cda3f0f0f91c728e588792e988`) | 실물·SHA-256 확인 (`2026-11-17 17:22 KST`) |
| `Dev 2,920` 전체 확률표 | `.../s42/light/dev_prefix.npz` (`sha256: 9e8b7eda897aadc7b874c6e103f46ac3491b40a80c16818c356f24cefbafc208`) | SHA-256·`score.py` 재채점(`warnings=[]`) 완료 (`2026-11-17 17:23 KST`) |
| 오프라인 CPU 추론 일치 | 앞 4통화(`72`시점) `best_model` CPU 추론 vs `dev_prefix.npz`: `max_abs_err=0.000592`, `mean_abs_err=0.00006718`, `0.5` 불일치 `0`건 | 검증 완료 (`2026-11-17 17:23 KST`) |
| 중복 청크 정리 | `s42/weight_chunks/` (`2.2GB` 임시 청크) 삭제 (`weights.zip`·압축해제본 확인 후) | 정리 완료 (`2026-11-17 17:24 KST`) |
