# Mission 4 Gap 2 & Gap 3 연구·감사·스토리라인 브랜치 (`mission4-softjaccard-fitonly`)

기준 브랜치: `origin/mission4` (`a16b61d`)  
업로드일: 2026-11-18  

이 브랜치는 `origin/mission4` (`mission4-streaming` 병합본) 이후 진행된 **Gap 2 (4-1 손실함수 정렬 개선)**, **Gap 3 (`fit 26,280` 5-fold OOF 재생성 및 빈 예측 조기 중단 금지 정책)**, **독립 코드·정량 감사**, 그리고 **전체 파이프라인·발표 스토리라인 정리** 산출물을 공유하기 위해 분리 생성한 연구·기록 브랜치다.

*(제출용 `mission4_symptom/m4/` 추론 코드 통합은 다음 단계에서 진행하며, 이 브랜치는 기존 `m4/` · `inference.py` 파일을 수정하지 않았다.)*

## 포함된 주요 경로

| 경로 | 내용 |
|---|---|
| `mission4_symptom/scripts/m4_precue_bce/` | `train_precue.py` (`precue_bce`, `SoftJaccard`, `SmoothMacroDice`), `predict_prefix.py`, `eval_precue.py`, `eval_halting_gap3.py`, `run_gap3_complete.py`, `run_oof_resilient.py`, Colab 원격 실행기, 단위 테스트(`test_train_precue.py`), CPU 검산 3종(`recheck_41.py`, `check_empty_at_halt.py`, `check_shift.py`) |
| `mission4_symptom/runs/m4_precue_bce_v1_s42/s42_eval.json` | `m4_precue_bce_v1` `seed 42` `Dev 2,920` 집계 결과 (기각 근거) |
| `mission4_symptom/runs/m4_softjaccard_v1_s42/s42_eval.json` | `m4_softjaccard_v1` `seed 42` `Dev 2,920` 집계 결과 |
| `mission4_symptom/runs/e026_fitonly_policy_v1/` | `summary.json` (Gap 3 6개 모델 `4-2` 및 10,000회 bootstrap 전체 결과), `gap2_summary_cmp.json` (Step 2 기전 분해), `policy_fitonly_c128.json` (`fit 26,280` 전용 LightGBM 중단 정책) |
| `docs/mission4/MISSION4_FULL_PIPELINE_AND_STORYLINE_20261118.md` | 예선 3 재사용부터 본선 4-1·4-2 최종 해법까지의 전체 파이프라인 다이어그램·해설·Q&A·보완 체크리스트 |
| `docs/mission4/MISSION4_QNA_AND_IMPROVEMENT_NOTES_20261118.md` | 백본/TAPT 비교 비대칭성 및 다중 시드 효율성 분석 메모 |
| `docs/mission4/mission4_external_review_20261118/AUDIT_REPORT.md` | 코드·정량·제출 구조(방안 A vs B, `logit_shift` 충돌, `pack_bundle.py` 함정) 독립 감사 보고서 |
| `docs/mission4/MISSION4_EXTERNAL_AGENT_REVIEW_BRIEF_20261118.md` | 외부 검토용 브리프 (감사 정정 반영본) |
| `docs/mission4/{COMPETITION_RULES_20261114.md, MISSION4_EXPERIMENT_LEDGER.md}` | 본선 규정 정리 및 Mission 4 전체 실험 원장 |
| `docs/mission4/{m4_precue_bce_*, m4_softjaccard_*, m4_jacc_macrof1_*, mission4_e026_*}/` | 각 실험별 사전 프로토콜(`PROTOCOL.md`) 및 결과 기록(`EXPERIMENT_RECORD.md`, `SEED43_SUPPLEMENT.md`) |

## 공개 저장소 제외 항목 (비공개 보관 안내)

대회 및 `AGENTS.md` 보안·보존 기준에 따라 아래 산출물은 이 공개 Git 브랜치에 포함하지 않았으며 로컬/외장 비공개 디스크(`/Volumes/로컬 디스크/DCC/private_experiments/` 및 작업 디렉터리 `artifacts_private/`)에 SHA-256 체크섬과 함께 별도 보관한다.

- 모델 가중치(`final.pt`, `best_model/`), 학습 재개 상태(`training_state_last.pt`)
- 시점별 원확률 배열(`.npz`), OOF 병합 확률표(`fit_oof_merged.npz`), 통화별 중단 배열(`*_calls.json`)
- 원본 전사 텍스트 발췌가 포함된 Step 2 원문 대조 보고서 및 블라인드 사례 주석 파일
