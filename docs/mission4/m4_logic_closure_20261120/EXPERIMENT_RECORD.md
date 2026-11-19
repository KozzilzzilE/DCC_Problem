# `m4_logic_closure_v1_20261120` — `Full vs Prefix` 통제 비교 및 `E008(KF-DeBERTa)` · `E019(RoBERTa-large)` `raw p >= 0.5` 재현 검증

- 작성/갱신일: 2026-11-20
- 담당자 / 설계·결론 검토: 이성규 / Planner (`exit_plan_mode` 사전 등록 프로토콜 및 결과 판정)
- 현재 상태: **보존 검증 완료** (`3`개 단계 가중치·재개 상태·예측 `.npz` SHA-256 일치, `score.py` 경고 `0`건, 오프라인 `CPU 8`통화 `226`시점 추론 일치, Colab L4 자동 정지 `No active sessions`)
- 사전 프로토콜: [`PROTOCOL.md`](PROTOCOL.md)
- 코드 및 검증: [`tools/m4_logic_closure/`](../../tools/m4_logic_closure/) (`common.py`, `train_stage.py`, `predict_stage.py`, `remote_run_stage.py`, `orchestrate_logic_closure.py`, `test_logic_closure.py`, `verify_logic_closure.py`)
- 로컬 · 공유 보관 위치:
  - 로컬 전체 산출물(재개 상태 포함): `artifacts_private/runs/m4_logic_closure_v1_20261120/`
  - Google Drive 공유 폴더(`06_logic_closure_full_e008_e019_20261120`): `https://drive.google.com/drive/folders/1EmmNH5_OS0pD51yyTi0HyZZbUEIZl8qJ` (`best_model.zip` · `light.zip` · 요약/기록 JSON/MD — 원본 전사·Validation 제외)

---

## 1. 왜 이 실험을 하는가

이전까지 Mission 4 스토리라인과 백본 비교표에는 두 가지 논리 연결 공백이 남아 있었다.
1. **`Full vs Prefix` 비교의 학습량·환경 비대칭:** 예선 모델 `E001` (`lr 2e-5`, `max_len 384`, `5 epoch`)과 본선 `Prefix` 모델 `E002` (`lr 5e-5`, `max_len 512`, `3 epoch × 2/통화 = 9,855`스텝)는 학습 조건이 달라, 동일한 `E002` `TAPT` 백본·하이퍼파라미터·학습 예산에서 입력 분포만 통화 전체(`Full-only`)로 바꿨을 때의 정량 거동이 확인되지 않았다.
2. **백본 비교표 `E008(KF-DeBERTa)` · `E019(RoBERTa-large)` 공식 `raw p >= 0.5` 수치 부재:** 팀원 결과 표에는 `E008`의 `구규칙(top1=True) 0.5354`와 `E019`의 `maxent 0.5451`만 남아 있었고 원본 `.npz`가 미확보 상태라, `Dev 2,920` 공식 `raw p >= 0.5`(Top-1 없음) 수치와 95% 신뢰구간이 비어 있었다.

**검증할 질문:**
1. 동일한 `TAPT KLUE-RoBERTa-base`·`fit 26,280`·`lr 5e-5`·`LLRD 0.8`·`seed 42` 조건에서 통화 전체(`Full-only`)로 `6 epoch`(`9,852`스텝, `epoch 3` 진단 스냅샷 병행)를 학습했을 때 `E002`(균등 Prefix) 대비 스트리밍 판별력(`AUROC`, `mAP`, `maxent`)과 `raw p >= 0.5` · `4-2 C128` 점수는 어떻게 달라지는가?
2. `E008(kakaobank/kf-deberta-base)` 및 `E019(klue/roberta-large)`를 팀원 `run_config.json` 레시피 그대로 Colab L4에서 독립 재현했을 때, 팀원 기존 진단치(`old_top1 0.5354`, `maxent 0.5451`)가 근사 재현되며 공식 `raw p >= 0.5`에서도 `E002(0.502733)` 및 `ENS3(0.506001)` 비채택 결론이 유지되는가?

---

## 2. 무엇을 어떻게 비교하는가

| 단계명 | 초기 모델 (`allow_network`) | 샘플링·노출 | 배치 / LR / LLRD / gc | 총 스텝 / 업데이트 | 비고 |
|---|---|---|---|---|---|
| `full_only_roberta_tapt_s42` | `E002` 동일 `TAPT klue/roberta-base` (로컬, SHA-256 `25424975...`) | `full_only`, `6 ep × 1/통화` (`ep3` 스냅샷 저장) | `16 × 1`, `5e-5`, `0.8`, `gc=False` | `9,852 / 9,850` (`ep3: 4,926`) | `skip=2` |
| `e008_kfdeberta_repro_s42` | `kakaobank/kf-deberta-base` (No-TAPT) | `uniform_prefix`, `3 ep × 2/통화` | `8 × 2` (실효 `16`), `2e-5`, `1.0`, `gc=False` | `19,710 / 9,853` | `use_fast=true`, `skip=2` |
| `e019_roberta_large_repro_s42` | `klue/roberta-large` (No-TAPT) | `uniform_prefix`, `3 ep × 2/통화` | `16 × 1`, `5e-5`, `0.8`, `gc=True` | `9,855 / 9,853` | `skip=2` |

- **데이터·분할·누수 방지:** `train.jsonl` 내 `fit_ids.txt` (`26,280`건) 학습, `dev_ids.txt` (`2,920`건, `87,814`시점) 평가. `assert_not_validation_path` 및 `verify_non_overlapping`(ID·정규화 전사 겹침 0건) 통과. 공식 `Validation(3,640)` 미접근.
- **공통 조건:** `fmt="space"`, `max_length=512`, `truncation_side="right"`, `pos_weight_power=0.5`, Weighted BCE (`E002` 동일), `seed=42`, AMP FP16.
- **사전 점검(`preflight_report.json`):** L4 `23.66 GB`, 디스크 `252.41 GB`. `TAPT` SHA-256 일치 확인; UNK율 `tapt=0.00023`, `kf-deberta(fast)=0.000047`, `roberta-large=0.00023`; LLRD 100% 커버리지(`adam_state_count == total_trainable_tensors`: `201/201`, `202/202`, `393/393`) 및 로딩 오류 `0`건 확인.
- **실행 환경:** Colab `NVIDIA L4`, 오케스트레이터 `orchestrate_logic_closure.py`, 총 소요 `15,669.6초`(`~4.35시간`, 단계별 청크 다운로드 및 SHA-256 검증 후 원격 정리·VM 자동 정지).

---

## 3. 사전 판단 기준과 주요 관찰

| 기준 | 사전 확인·채택 기준 | 근거 |
|---|---|---|
| 채점·정합성 | `score.py` 경고 `0`건, CPU 8통화 추론 일치 (`roberta < 1e-3`, `deberta-v2 < 5e-3`) | `AGENTS.md` 및 `PROTOCOL.md` |
| `Full vs Prefix` 해석 | `raw 0.5`뿐 아니라 구간별 정확도, 확률 분리도(`AUROC`/`mAP`), `maxent`, `4-2 C128` 함께 대조 | OOD 초반 빈칸 편향 분리 |
| `E008`/`E019` 재현 | 팀원 진단치(`old_top1 0.5354`, `maxent 0.5451`) 근사 재현(`±0.015`) 후 `raw 0.5` 및 `ENS3` 교체/추가 확인 | 전략 B(`m4_backbone_tapt`) 동일 게이트 |

---

## 4. 실제 수행과 결과 (`Dev 2,920`통화 · `87,814`시점, `verification_summary.json`)

### A. 1단계 `full_only_roberta_tapt_s42` vs `E002` (`Full vs Prefix` 통제 비교)

- 학습 손실(`full_only`): epoch 1 `0.5010`(359.1s) → epoch 2 `0.4148`(728.6s) → **epoch 3 `0.3865`(1,093.3s)** → epoch 4 `0.3553` → epoch 5 `0.3243` → **epoch 6 `0.2983`(2,210.3s, VRAM `4.67 GiB`)**.

| 모델 / 조건 | `4-1 (raw 0.5)` | `E002` 대비 `[95% CI]` | `AUROC` (전체/초반1/3) | `mAP` (전체/초반1/3) | `diag_maxent` | 1발화 비빈율 / `복통` 출력 수 | `4-2 C128` (`req_nonempty=True`) | 끝 `Macro-F1` |
|---|---|---|---|---|---:|---|---|---|
| **`E002` (`Prefix` 기준선)** | `0.502733` | 기준 (`0.000000`) | **`0.8438 / 0.7592`** | **`0.6170 / 0.4774`** | **`0.5485`** | `2.5%` / `2`건 | **`0.501900`** (`F1 0.6049`) | **`0.6502`** |
| `full_only_ep3` (`4,926`스텝) | `0.520431` | `+0.017697 [+0.012286, +0.023249]` | `0.8378 / 0.7445` | `0.6080 / 0.4598` | `0.5423` | **`89.8%` / `2,594`건 (`88.8%`)** | `0.487107` (`F1 0.5822`) | `0.6414` |
| `full_only_ep6` (`9,852`스텝) | `0.510817` | `+0.008083 [+0.002445, +0.013686]` | `0.8256 / 0.7280` | `0.5893 / 0.4419` | `0.5267` | **`97.8%` / `2,667`건 (`91.3%`)** | `0.462773` (`F1 0.5417`) | `0.6377` |
| *(대조)* **`SJ42+SJ43` (채택 `4-1`)** | **`0.526039`** | `full_ep3` 대비 `+0.005608` (`[+0.000559, +0.010610]`) | `0.8405 / 0.7560` | `0.6116 / 0.4764` | `0.5506` | `10.2%` / (`SJ42 283`건) | `0.497077` *(4-1 전용)* | `0.6484` |

- **핵심 메커니즘 규명 (`OOD` 1발화 다수 클래스 쏠림 역설):**
  - 순수 확률 판별력(`AUROC`, `mAP`), 분포 기반 `maxent`, 그리고 `4-2 C128`·`Macro-F1`에서는 **Prefix 학습(`E002`)이 `full_only_ep3`/`ep6`를 전 구간에서 일관되게 앞섰다** (`full_only_ep6`는 `3 epoch` 이후 급격한 과적합으로 `early AUROC -0.0312`, `4-2 -0.0391` 하락).
  - 그럼에도 `full_only`의 `raw p >= 0.5` 점수가 높게 나온 이유는 **OOD 다수 클래스 쏠림** 때문이다. 학습 때 1마디짜리 짧은 입력을 한 번도 못 본 `full_only`는 평가 시 1번째 발화가 들어오면 **`2,920`통화 중 `2,594`건(`88.8%`, `ep6 2,667`건 `91.3%`)에 다수 증상인 `복통`을 일괄 출력**했다(`1st step 4-1 = 0.1740 / 0.1835`). 반면 보정 균형이 잡힌 `E002`는 근거가 부족한 1발화에서 빈칸(`0`점, `1st step = 0.0076`)을 내므로, `Top-1` 금지 규칙 하에서 `full_only`의 다수 클래스 기본 찍기가 초반 빈칸 `0`점을 피하며 `raw 0.5` 점수만 끌어올렸다.
  - 앙상블에 넣어도 `ENS_swap_E1_full3` (`4-1 0.506921`, `4-2 C128_v1_nonempty 0.500132`), `ENS_swap_E2_full3` (`4-1 0.506773`, `4-2 C128_v1_nonempty 0.501128`), `ENS4_add_full3` (`4-1 0.506516`) 모두 `ENS3 4-2 (0.507411)`를 `-0.0063~-0.0073` 크게 훼손했다.
  - 탐색 참고: `SJ42+SJ43`(`0.526039`)은 `full_only_ep3`(`0.520431`)을 `+0.005608`(`p=0.9858`) 앞설 뿐 아니라 1발화 `복통` 쏠림 없이 초반 판별력(`early AUROC 0.7560 vs 0.7445`)을 유지한다. (`SJ42+SJ43+full3`는 참고 집계상 `0.528323`이나, `full_ep3`의 1발화 `88.8%` `복통` 편향 전이 및 사후 `Dev` 선택 규정에 따라 미채택.)

### B. 2·3단계 `E008(KF-DeBERTa)` · `E019(RoBERTa-large)` 독립 재현 (`No-TAPT`)

| 모델 / 조합 | 학습 시간 / peak VRAM | 기존 팀원 지표 vs 본 재현 진단치 | `4-1 (raw 0.5)` | `E002` / `ENS3` 대비 `[95% CI]` (`5,000`회) | `4-2 C128` (`v1_nonempty`) |
|---|---|---|---|---|---|
| **`E008` 재현 (`kf-deberta-base`)** | `3,327.7s` / `8.19 GiB` | `old_top1 0.5354` vs **`0.5230`** (`maxent 0.5372`) | **`0.494825`** | vs `E002`: **`-0.007908 [-0.013549, -0.002146]`** (`p=0.996` 열위) | `0.495119` (`F1 0.5943`) |
| `ENS_swap_E2_to_e008` / `ENS4_add_e008` | — | — | `0.505460` / `0.505109` | `-0.000541 [-0.003476, +0.002428]` / `-0.000891 [-0.003228, +0.001480]` | `0.506556` *(v0 `0.506619` / `0.508621`)* |
| **`E019` 재현 (`klue/roberta-large`)** | `6,875.6s` / `5.41 GiB` (`gc=True`) | `maxent 0.5451` vs **`0.543865`** *(diff -0.0012)* | **`0.498232`** | vs `E002`: `-0.004501 [-0.009464, +0.000549]` (`No-TAPT base 0.4987` 이하) | `0.502639` (`F1 0.6070`) |
| `ENS_swap_E2_to_e019` / `ENS4_add_e019` | — | — | `0.506018` / `0.505276` | `+0.000018 [-0.002412, +0.002568]` / `-0.000724 [-0.002690, +0.001274]` | `0.508641` / `0.508588` |

> 주: `verify_logic_closure.py` 내 `m4_2_c128_v1_nonempty` 열은 `e026_same_input_replay` 구버전 `HaltPolicy` 모듈 기준(`v0`, `ENS3=0.507400`)이며, 제출용 `mission4_symptom/m4/halting.py` (`require_nonempty=True`) 기준 실측 교차 대조는 위 표기값(`ENS3=0.507411`, `E002=0.501900`, `swap_E2_e019=0.508641`, `add_e019=0.508588`)이다.

---

## 5. 결정과 다음 작업

- **최종 판정:**
  1. **기존 Option A 제출 번들(`4-1: SJ42+SJ43`, `4-2: ENS3 + C128_v1_nonempty`) 현행 유지 — 신규 3종 모두 미채택.**
  2. **`Full vs Prefix` 논리 보강 완료:** `E002`(균등 Prefix)가 실제 확률 분리도(`AUROC`/`mAP`)·`maxent`·`4-2` 모두 `full_only`를 압도함을 입증했고, `full_only`의 높은 `raw 0.5`(`0.5204`)가 1발화 `복통` 88.8% OOD 일괄 출력에서 비롯됨을 분해해 Step 3(`0.5` 빈칸 병목) → Step 4-B(`SoftJaccard 0.5260`, OOD 편향 없이 초반 빈칸 해결)의 논리적 당위성을 완성했다.
  3. **`E008`/`E019` 백본 표 `raw 0.5` 공백 메움:** `E019`는 `maxent 0.5439`(원본 `0.5451`)로 정밀 재현됐고 `raw 0.5`는 `0.498232`(단독 열위, `ENS_swap_E2 0.506018` 무차이), `E008`은 `0.494825`(통계적 유의 열위)로 확인돼 `KLUE-RoBERTa-base + TAPT` 채택 및 비용-효율 배제 근거가 완전히 닫혔다. *(향후 팀원 원본 `E008`/`E019` `.npz`가 추가 회수되면 주 표 수치는 원본 재채점으로 대체하고 본 실행은 독립 재현 기록으로 병기한다.)*

---

## 6. 산출물 보존 확인

| 단계 | `final.pt` SHA-256 | `training_state_last.pt` SHA-256 | `dev_prefix.npz` SHA-256 | CPU 8통화(`226`시점) 일치 | 보관 위치 |
|---|---|---|---|---|---|
| `full_only_roberta_tapt_s42` (`ep6`) | `8fcaf2f817fff520...1040de` | `bc8e5a74485d7733...f845de0` | `6c177c42f998353d...16b2cceb` | `max=0.000972`, `bin=0/2034` | 로컬 + Drive `06_` (`best_model`/`light`) |
| `full_only_roberta_tapt_s42` (`ep3`) | `a909b73d978c766e...eaffaa6` | *(상동 ep6 연속)* | `8e90bcf80673d581...9b01754` | `max=0.000957`, `bin=0/2034` | 로컬 + Drive `06_` (`best_model`/`light`) |
| `e008_kfdeberta_repro_s42` | `471675ea05cdd420...89f88` | `4cbc294333f0431a...799f4beb` | `a6930c61f8f46c60...ee061c0a` | `max=0.002023`, `p99=0.000711`, `bin=2/2034` (`deberta-v2` FP16 허용 `< 5e-3`) | 로컬 + Drive `06_` (`best_model`/`light`) |
| `e019_roberta_large_repro_s42` | `6264d91e4e74f83e...fe4a57` | `21109ca3bd44b79f...49cc142` | `eb68093857c46dfe...cfd277` | `max=0.000899`, `bin=0/2034` | 로컬 + Drive `06_` (`best_model`/`light`) |

- 단위 테스트([test_logic_closure.py](../../tools/m4_logic_closure/test_logic_closure.py)): `7 passed in 7.09s`, Colab L4 정지 상태: `No active sessions found on server`.
- 미해결 항목: 공식 `Validation(3,640)` 1회 최종 검증은 팀 최종 freeze 합의 후 실행한다.
