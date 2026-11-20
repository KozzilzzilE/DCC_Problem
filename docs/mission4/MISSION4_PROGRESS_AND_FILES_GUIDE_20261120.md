# Mission 4 진행 현황, 파일 안내 및 `.npz` 확인 목록 (2026-11-20)

이 문서는 **DCC 2026 본선 Mission 4(119 구급 통화 증상 조기예측)** 에 대해 `2026-11-20` 기준 **현재까지 완료한 작업**, **요청 검토했던 `.npz` 목록**, **Google Drive·GitHub 파일 설명**, 그리고 **최종 동결 전후 남은 점검 사항**을 한눈에 파악할 수 있도록 정리한 안내서입니다.

> **핵심 규정 기준:** 모든 비교와 최종 채택은 대회 공식 규정(임계값 `0.5` 고정, Top-1 강제 금지, `4-2` 중단 기준값 `0.5` 기반, 미래 길이 미사용, `Validation 3,640` 학습·선택 미사용)을 엄격히 준수하여 `Training` 내부 학습 분할(`fit 26,280`)과 비학습 검증 분할(`dev 2,920`) 위에서 진행했습니다.

---

## 1. 현재까지 진행 현황 요약 (`2026-11-20`)

### 최종 채택 제출 구성: 방안 A (`version=2` 미션 분리형 번들)
단일 가중치 번들 `ckpt/mission4.pt` (`1,111,646,045 B` / `1,060 MiB`, SHA-256 `26c3ff2729e67916d2543f216a1c6c342d399f5b24e3ade9c7509e534d3c92b8`) 안에 `4-1` 전용 모델과 `4-2` 전용 모델·정책을 분리 탑재해 `--mission`에 따라 필요한 멤버만 적재합니다.

| 미션 | 채택 구성 | `Dev 2,920` (`87,814`시점) 성능 | 비고 |
|---|---|---|---|
| **Mission 4-1** | `KLUE-RoBERTa-base (TAPT 20ep)` + `Weighted BCE + 1.0 * Soft-Jaccard` 2시드 (`s42 + s43`) | **`0.526039`** (`raw p >= 0.5`, 빈 예측 강제 없음) | 기존 `ENS3` (`0.506001`) 대비 **`+0.020038`** (`95% CI [+0.015980, +0.023926]`), `score.py` 경고 `0`건 |
| **Mission 4-2** | `ENS3` (`E001` 통화전체 + `E002` 앞부분 s42 + `E007` 앞부분 s43) + 무누수 OOF 정책 `C128_v1_nonempty` | **`0.507411`** (`Macro-F1 0.613217 × 조기판단력 0.827457`) | 순수 `fit 26,280` 5겹 OOF 학습(`β=128, τ=0.5, require_nonempty=True`), 중간 빈 예측 중단 `0`건 |

### 단계별 완료 내역
1. **0단계 · 공식 규정 `0.5` 재정렬 (완료):** `maxent`·`top1`·`τ=0.64`를 제거하고 `raw p >= 0.5` 고정 기준으로 기준선을 재측정했습니다.
2. **1단계 · 4대 경량 백본 동일 TAPT 공정 비교 (`m4_backbone_tapt_v1_20261119`, 완료):**
   - `fit 26,280` 전사 20-epoch TAPT를 통일 적용한 공정 비교 결과(`Dev 2,920`, `raw p >= 0.5`, `seed 42`):
     - **`KLUE-RoBERTa-base + TAPT` (`E002`, 기준선): `0.502733`**
     - `KLUE-RoBERTa-base (No-TAPT)`: `0.498685` → **순수 TAPT 기여 `+0.004049` 입증**
     - `KcELECTRA-v2022 + TAPT`: `0.498099` (`-0.004634`)
     - `KoELECTRA-v3 + TAPT`: `0.495421` (`-0.007312`, `95% CI [-0.012621, -0.001935]`)
     - `KR-BERT-char16424 + TAPT`: `0.493317` (`-0.009416`, `95% CI [-0.014867, -0.003899]`)
   - **결론:** 동일 TAPT 기준 `KLUE-RoBERTa-base + TAPT` 유지를 확정했습니다. (`No-TAPT RoBERTa`조차 타 백본+TAPT 3종을 상회)
3. **2단계 · `4-1` 목적함수 정렬 (`SoftJaccard` 2-seed, 완료):**
   - 사전 단서 이전 양성 가중치 완화(`m4_precue_bce_v1`: `0.497129`)는 미채택하고, `4-1` 순간 Jaccard 지표에 직접 정렬한 `BCE + 1.0 * Soft-Jaccard` (`s42`: `0.526393`, `s43`: `0.524933`, 2시드 앙상블 **`0.526039`**)를 채택했습니다.
   - `SoftJaccard` 단독은 저빈도 클래스(`오심`, `두통`) 재현율 하락으로 `4-2 Macro-F1`이 낮아져(`0.5732` vs `ENS3 0.6136`), 미션별 최적 모델을 분리하는 **방안 A**를 채택했습니다.
4. **3단계 · `4-2` 무누수 OOF 중단 정책 (`e002_oof` + `C128_v1_nonempty`, 완료):**
   - `dev 2,920`이 전혀 섞이지 않은 순수 `fit 26,280` 5겹 교차학습 확률표(`801,544`시점)와 `fit_only_shift`만으로 `LightGBM` 조기중단 정책을 재학습하고 `require_nonempty` 게이트(`any(raw_p >= 0.5)`)를 적용해 `ENS3` 기준 `0.498835 → 0.507411` (`+0.008575`, `95% CI [+0.003325, +0.013903]`, 중간 빈 예측 중단 `27→0`건)을 달성했습니다.
5. **4단계 · 팀원 결과 통합 및 논리 공백 보강 (`m4_logic_closure_v1_20261120`, 완료):**
   - 팀원 백본 벤치마크(`ohyun0628`: `309`건 비공식 검증 제거·`32,840` 누수 격리·`6절` 규칙 분리 반영) 및 Prefix 소거법(`okjunges`: `T0~T10` 순차 통일표, `T4/T8` 차이·`localcontrol T0` vs `E026` 차이 검증)을 통합했습니다.
   - Colab L4 통제 실험 3건을 완주·해시 검증했습니다:
     - **`Full-only (6ep, ep3 스냅샷 포함)` vs `E002`:** `ep3 raw 0.5 = 0.520431` / `ep6 0.510817`은 1발화 OOD 입력에서 `2,594/2,920`건(`88.8%`)에 다수 클래스 `복통`을 일괄 출력해 생긴 편향 역설임을 규명했고, `AUROC`(`0.8438` vs `0.8378`)·`mAP`(`0.6170` vs `0.6080`)·`4-2 C128`(`0.5019` vs `0.4871`)은 모두 `E002`가 앞섰으며 채택 `SJ42+SJ43`(`0.526039`)이 `full_ep3`보다 `+0.005608` 유의 우위임을 확인했습니다.
     - **`E008 KF-DeBERTa` L4 재현 (`No-TAPT`):** `diag_old_top1 0.5230`, 공식 **`raw 0.5 = 0.494825`** (`E002` 대비 `-0.007908`, `95% CI [-0.013549, -0.002146]`) → 비채택 확정.
     - **`E019 KLUE-RoBERTa-large` L4 재현 (`No-TAPT`):** `diag_maxent 0.543865`(원본 `0.545105` 대비 `-0.0012` 재현), 공식 **`raw 0.5 = 0.498232`** (`-0.004501`, `3~4배` 비용-지연) → 비채택 확정.
6. **5단계 · 제출 패키지 완결 검증 (완료, `mission4-submission-v2`):**
   - `mission4.pt` (`version=2`) 재패킹, `Dev 2,920` 2계층 E2E 검증 (`Layer 1` 전수 일치 + `Layer 2` 16통화 노모킹 오프라인 CLI `inference.py` + `score.py` 경고 `0`건, fp16 오차 `<= 0.000624`).
   - `scripts/train_prefix.py` `softjaccard` 기본 가중치(`0.5→1.0`) 수정, `scripts/run_final_training.sh` 방안 A 재현 기준 전면 개편, `model_train.ipynb` 11/11 코드 셀 실행 출력 수록(원본 `29,200`건 캐시 줄 단위 일치 `same=True`, 개인 경로·통화 ID·전사 누출 `0`건).
   - `scripts/build_submission.py` 실제 조립 드라이런 후 `PYTHONPATH` 없는 독립 조립 폴더·오프라인 환경에서 `pytest 48/48` 통과 및 `Dev` 4통화(`83`시점) `inference.py` + `score.py` 경고 `0`건 검증 완료.

---

## 2. 팀원에게 요청하려 했던 `.npz` 파일 목록 및 현재 상태

과거 팀원 실험 결과 중 확률 원본(`.npz`)이 저장소/전달본에 빠져 있어 확인·요청 대상으로 잡았던 목록과, **현재(`11-20`) 기준 실제 필요 여부**입니다.

### ① 최우선 요청 대상이었던 파일 — *L4 독립 재현으로 공백 해소 완료 (원본 보유 시 선택 교체)*
| 원래 요청 대상 파일 (팀원 PC 기준) | 왜 필요했는가 | 현재 상태 (`2026-11-20`) |
|---|---|---|
| `runs/E008_kfdeberta_prefix/dev_prefix.npz` | 기존 로그에 구규칙(`top1=True`, `0.5354`)만 남아 있고 규정 준수 **`raw p >= 0.5`**·`95% CI`가 없었음 | **`m4_logic_closure` L4 독립 재현으로 해소 완료** (`raw 0.5 = 0.494825`, `-0.007908`, `95% CI [-0.013549, -0.002146]`). 팀원 PC에 원본이 남아 있으면 발표 주 수치 교체용으로만 선택 전달 |
| `runs/E019_large_gc/dev_prefix.npz` | 기존 로그에 **`maxent` (`0.5451`)** 만 남아 있고 규정 준수 **`raw p >= 0.5`**·`ENS_swap`·`4-2 C128` 수치가 없었음 | **`m4_logic_closure` L4 독립 재현으로 해소 완료** (`diag_maxent 0.5439`, **`raw 0.5 = 0.498232`**). 팀원 PC에 원본이 남아 있으면 발표 주 수치 교체용으로만 선택 전달 |

### ② 선택 보완용 (핵심 채택·논리 결론에 영향 없음)
| 파일 (팀원 PC 기준) | 남아 있으면 얻을 수 있는 보완 정보 | 필수 여부 |
|---|---|---|
| `runs/E021_kcelectra/dev_prefix.npz`<br>`runs/E022_kcelectra_tapt/dev_prefix.npz` | `analysis.txt`에 이미 `raw 0.5`(`0.4957`, `0.4981`)가 기록되어 있으나, 전략 B 동일 스크립트로 `5,000`회 부트스트랩·`4-2 C128` 표를 통일하려면 `.npz` 필요 | 선택(없어도 논리 검증에 지장 없음) |
| `runs/E014_count_aux/dev_prefix.npz`<br>`runs/E018_speaker/dev_prefix.npz` | 과거 `maxent` 기준 미채택 보조 실험(`증상 개수 보조출력`, `[대원]/[신고자] 태그`)을 `raw 0.5` 기준 열까지 완전히 채우고 싶을 때 필요 | 선택(없어도 지장 없음) |
| `okjunges` T-series 중 **`T1`, `T3`, `T5`** 순차 예측 결과 (`dev_prefix.npz` 또는 `sequential` JSON) | `Mission4_Prefix_Training_Experiment_Results.zip`에 나머지 실험은 CI가 있었으나 `T1`·`T3`·`T5`만 `T0` 대비 `95% CI`가 미수록됨 | 선택(3건 모두 `T0` 대비 소폭 변동·미채택 확정) |

### ③ 더 이상 요청이 불필요한 파일 (확보·대체 완료)
- **`runs/E011_oof/fold0..4/oof_prefix.npz`:** `02_gap3_fitonly_oof_and_policy/e002_oof_20261117.zip` (`fit 26,280`, `801,544`시점)으로 5겹 OOF 전체를 새로 생성해 완전히 대체했습니다.
- **`E001`, `E002`, `E007` 가중치(`model.safetensors`) 및 `dev_prefix.npz`:** 실물 확보 및 `okjunges` SHA-256 해시 대조까지 100% 완료했습니다.

---

## 3. Google Drive 공유 폴더 및 파일 상세 설명

### A. 팀 공유 루트 폴더 (`1RMInYmZ-eFP6q3FoyjX2goGd3ZzcyMPN`)
| 이름 | 설명 | 비고 |
|---|---|---|
| **`미션4_진행현황_및_파일안내_20261120.md`** | **본 최신 안내 문서** (진행 현황, `.npz` 요청 목록, Drive·GitHub 전체 파일 가이드, 남은 작업) | `2026-11-20` 신규 업로드 |
| **`mission4_sunggyu_share_20261119/`** | `11/16~11/20` 보강 실험·제출 번들·검증 자료 전체 보관 폴더 (아래 B항 상세 참조) | 최신 검증 산출물(`00~06`) |
| `Mission4_Prefix_Training_Experiment_Results.zip` | 팀원 `okjunges` Prefix 소거 실험(`T0~T10`) 원본·비교표·부트스트랩 묶음 (`309.9 KB`) | 감사 통합 완료 |
| `11/14 본선 멘토링`, `미션4 데이터 분석 및 문제 ...` | 대회 멘토링·초기 데이터 분석 문서 | 기존 공유 문서 |
| `KcELECTRA/`, `mission4_baseline_share.zip`, `mission4_share2_E001_E007_bundle.zip` | 백본 비교 및 기준선(`E001~E007`) 모델 번들 | 초기 공유 자산 |
| `mission4_symptom.zip`, `미션4_진행정리.md` | **구버전(`11/15`)** 단일 `E026` 기준 공유본 (`4-1 0.5060 / 4-2 0.5031`, Validation `0.4867 / 0.5070`) | **참고:** 최종 제출본은 `05_` 폴더의 방안 A(`version=2`) 사용 권장 |

### B. `mission4_sunggyu_share_20261119/` 내부 상세 (`1Tdspzl8XsoNhNzk25YyPoPamYPv-RGFZ`)
> 개인정보·저작권·규정 보호를 위해 원본 전사(`train.jsonl`)와 `Validation 3,640`은 업로드하지 않았습니다.

| 폴더 / 파일 | 포함 파일 | 역할 및 핵심 내용 |
|---|---|---|
| **`00_README_INDEX.md`** | 안내 인덱스 (1개) | 전체 6개 하위 폴더의 구성, 수치 요약, GitHub 커밋 기준 안내 |
| **`SHA256SUMS.txt`** | 루트 해시 파일 (1개) | `01~04` 폴더 내 모든 ZIP·JSON의 무결성 검증용 SHA-256 목록 |
| **`01_gap2_loss_models/`** | `15`개 (`m4_precue_s42`, `m4_softjaccard_s42/s43`, `m4_jacc_macrof1_s42/s43` 각 `best_model.zip` · `light.zip` · `eval.json`) | **4-1 손실함수 개선 실험 5건.** `m4_softjaccard_s42` (`0.526393`) + `s43` (`0.524933`) 2시드 평균 **`0.526039`** 가 4-1 채택 모델 |
| **`02_gap3_fitonly_oof_and_policy/`** | `e002_oof_20261117.zip`, `e026_fitonly_policy_v1_20261117.zip` (2개) | **4-2 무누수 중단 정책.** 순수 `fit 26,280` 5겹 OOF(`801,544`시점)로 학습한 `C128_v1_nonempty` (`4-2 = 0.507411`) 원본·부트스트랩 결과 |
| **`03_audit_and_error_analysis/`** | `mission4_audit_summaries_20261119.zip` (1개) | `E026` 동일 입력 재생(`e026_same_input_replay`), `87,814`시점 단서 전/후 오류 분해(`m4_step2_error_analysis`), 사전 진단 JSON 모음 |
| **`04_strategy_b_backbone_tapt_20261119/`** | `11`개 (`roberta_notapt_s42`, `krbert_tapt20_s42`, `koelectra_tapt20_s42` 가중치/TAPT/light + 요약 JSON 3종) | **4대 경량 백본 동일 TAPT 공정 비교.** `KLUE-RoBERTa-base + TAPT (0.502733)` 유지 확정 및 `No-TAPT` 대조(`+0.004049`) 증거 |
| **`05_submission_bundle_option_a_20261119/`** | `mission4.pt`, `policy_fitonly_c128_v1_nonempty.json`, `e2e_verification_summary.json`, `submission_dryrun_verification.json`, `SHA256SUMS.txt` (5개) | **최종 제출 가중치 번들(`version=2`, `1,060 MiB`, `26c3ff27...`) 및 E2E·독립 조립 드라이런 검증 결과** |
| **`06_logic_closure_full_e008_e019_20261120/`** | `12`개 (`full_only_roberta_tapt_s42`, `e008_kfdeberta_repro_s42`, `e019_roberta_large_repro_s42` 각 `best_model`/`light.zip` + `PROTOCOL.md`, `EXPERIMENT_RECORD.md`, `verification_summary.json`, `preflight_report.json`, `ORCHESTRATION_COMPLETE.json`, `SHA256SUMS.txt`) | **논리 보강 통제 실험.** `Full-only` OOD 1발화 편향 역설(`복통` `88.8%`) 규명 및 `E008`/`E019` `raw 0.5` 독립 재현·비채택 확정 |

> *(대용량 학습 재개 상태 `training_state_last.pt` · `final.pt`는 파일당 ~1.25 GB 이상이라 로컬·외장 디스크 `artifacts_private`에 SHA-256과 함께 별도 보관되어 있습니다.)*

---

## 4. GitHub 저장소 파일 설명 (`KozzilzzilE/DCC_Problem`, 브랜치 `mission4-submission-v2`)

### A. 실제 제출 폴더 (`mission4_symptom/`)
`python scripts/build_submission.py --out <제출폴더>/mission4_symptom`을 실행하면 아래 **추론·학습·테스트 코드 `26`개 + `ckpt/mission4.pt`** 만 깔끔하게 내보내집니다(`runs/`·`docs/` 제외).

| 경로 | 역할 및 설명 |
|---|---|
| `inference.py` | 주최측 공식 추론 템플릿. `class Predictor`만 작성했고 나머지 데이터 로딩·CSV 저장 코드는 수정 없음 |
| `score.py` | 주최측 공식 채점 스크립트 (수정 없음) |
| `SUBMISSION_README.md` | 제출 폴더 내 `README.md`로 복사되는 실행·모델·규정·재현 안내서 |
| `model_train.ipynb` | 방안 A(`fit 26,280`, `version=2`) 전체 학습·검증 기록 노트북 (**`11/11` 코드 셀 실행 출력 수록 완료**, 개인정보·원문 누출 `0`건) |
| `requirements.txt` | 파이썬 의존 패키지 목록 (`torch==2.13.0+cu130`, `transformers==5.15.0`, `lightgbm==4.7.0` 등 팀원 Windows 검증 사양 기준) |
| `m4/__init__.py`, `data.py` | 9종 증상 정의, 라벨 JSON → 통화 단위 정렬·필터링, `prefix_texts()` 누적 문자열 생성 |
| `m4/bundle.py` | `version=2` 미션 분리 번들 패킹(`pack_mission_split_bundle`) 및 미션별 멤버 선택 적재(`load_bundle`), `pos_weight`/`logit_shift` 정합성 검사 |
| `m4/decision.py` | `threshold 0.5` 고정(`force_top1=False`), 보정(`calibrate`), 기대 점수 계산 등 결정 규칙 |
| `m4/halting.py` | `4-2` 학습형 중단 정책(`57`개 인과적 특징 추출 `HaltFeatureState`, `LightGBM` 부스터 복원, `require_nonempty` 게이트, 부트스트랩/점수 계산) |
| `m4/streaming.py`, `m4/simulate.py` | 발화 단위 스트리밍 추론 상태기(`StreamingSymptomModel`) 및 `score.py` 일치 모의 채점기 |
| `scripts/train_prefix.py` | 앞부분·전체(`--full-only`) 통화 학습기 (`--loss {bce,softjaccard,jacc_macrof1}`, `softjaccard` 기본 가중치 `1.0`) |
| `scripts/run_final_training.sh` | 방안 A 전체 파이프라인(데이터 검증 → TAPT → `4-1 SoftJaccard` → `4-2 ENS3` → `C128_v1_nonempty` → 번들 패킹) 통합 재현 셸 스크립트 |
| `scripts/train_halt_policy.py` | `5`겹 OOF 기반 `LightGBM` 중단 정책 학습·`--require-nonempty` 지원 |
| `scripts/pack_bundle.py`, `build_submission.py`, `build_notebook.py` | 번들 패커, 커밋(`HEAD`) + 번들 해시 검증 제출 조립기, 노트북 생성·실행기 |
| `scripts/build_cache.py`, `tapt_mlm.py`, `prefix_probs.py`, `run_oof_tables.sh` | 캐시 구축, TAPT 20ep MLM 학습, 확률표(`.npz`) 추론, 5겹 OOF 테이블 생성 스크립트 |
| `tests/test_*.py` (`5`개 파일, `48`테스트) | 번들, 결정 규칙, 중단 정책 인과성/`require_nonempty`, 채점식 일치, 스트리밍 일관성 단위 테스트 |

### B. 저장소 문서 (`docs/mission4/`)
- **`MISSION4_PROGRESS_AND_FILES_GUIDE_20261120.md`**: 지금 읽고 계신 종합 진행·파일 안내 문서
- **`MISSION4_FULL_PIPELINE_AND_STORYLINE_20261118.md`**: 발표·심사 대응용 전체 실험 논리 흐름(0단계~방안 A 채택, 6대 논리 연결 완결본)
- **`MISSION4_EXPERIMENT_LEDGER.md`**: 날짜별 전체 실험·검증 원장 (`11-11 ~ 11-20`)
- **`m4_backbone_tapt_20261119/`**, **`m4_logic_closure_20261120/`**, **`team_results_audit_20261119/`**: 최근 보강 실험·팀원 감사 프로토콜 및 검증 기록
- *(주의: `mission4_symptom/docs/` 안의 `MISSION4_정리.md`, `PLAN.md`, `PROGRESS_REPORT.md`는 `11/18` 이전 `mission4-streaming` 시절 기록이므로, 최신 수치는 반드시 `docs/mission4/`와 `SUBMISSION_README.md`를 참조하세요.)*

---

## 5. 완결성 최종 점검: 끝난 것 vs 부족한 것

### 완료 및 검증 끝난 항목
- [x] `KLUE-RoBERTa-base + TAPT` 백본 확정 (4대 `base` 백본 동일 TAPT 공정 비교 + `No-TAPT` 대조)
- [x] `4-1` (`SoftJaccard 2-seed = 0.526039`) · `4-2` (`ENS3 + C128_v1_nonempty = 0.507411`) 미션 분리 방안 A 확정 및 규정(`0.5` 고정, 미래 길이 미사용) 준수 검증
- [x] `Full-only` vs `Prefix` 통제 실험 및 `E008`/`E019` `raw 0.5` 독립 재현 완료, Colab 유휴 런타임 종료(`0` 활성)
- [x] `mission4.pt` (`version=2`) SHA-256 해시 대조 + `Dev 2,920` 전수/`16`통화 노모킹 오프라인 CLI E2E 검증
- [x] `model_train.ipynb` 전 셀(`11/11`) 실행 출력 수록 (`29,200`건 캐시 줄 일치 검증, 누출 `0`건)
- [x] `build_submission.py` 독립 조립 드라이런 + 오프라인 `pytest 48/48` + `4`통화 `inference.py`/`score.py` 경고 `0`건 통과

### 아직 남아 있는 항목 (팀 합의 · 동결 후 진행)
1. **[필수 — 동결 직후 1회] 공식 `Validation 3,640` 최종 검증 및 실측 반영**
   - 팀 최종 모델 동결 합의 직후 **딱 1회** `inference.py` + `score.py`를 실행해 실측 점수·경고 `0`건·실제 소요 시간을 확정하고 `SUBMISSION_README.md`의 예정 문구를 교체해야 합니다.
2. **[팀 합의 필요] `fit 26,280` 동결 확정 · `main` 병합 · Drive 구버전 정리**
   - `fit 26,280` 기반 방안 A로 동결할지 팀 합의 후 `mission4-submission-v2 → main` PR·병합이 필요합니다. (Google Drive 루트의 `11/15` 구형 `mission4_symptom.zip`을 최종 제출본으로 교체·정리할지도 결정 필요)
3. **[환경 점검 필수] `requirements.txt` 채점 서버 호환성 확인**
   - 현재 `requirements.txt`가 팀원 Windows 환경(`Python 3.14.6`, `torch==2.13.0+cu130`, `transformers==5.15.0`) 기준 `==`로 고정돼 있어, 심사 서버가 Linux/Python 3.11~3.12/CUDA 12.x라면 `pip install` 충돌 가능성을 팀과 확인·조정해야 합니다.
4. **[정리·발표 권장] 구형 `mission4_symptom/docs/` 정리 여부 & 최종 발표 PPT 작성**
   - 제출 폴더 조립(`build_submission.py`) 시에는 제외되지만, 저장소 열람 혼선을 막으려면 `mission4_symptom/docs/MISSION4_정리.md` 등 구형 문서에 최신 문서 링크 안내를 달거나 정리하는 편이 안전합니다.
5. **[기록상 보존 상태 주석] 초기 파일럿 `context_expanded_v2_20261112` `final.pt` 원격 실물 대조**
   - 최종 제출 모델과는 무관한 `11/12` 초기 파일럿 6건의 가중치(`final.pt`)는 Mac 사본·해시 대조가 완료되지 않은 원격 미확인 상태로 보존 원장에 구분 기록해 두었습니다.
