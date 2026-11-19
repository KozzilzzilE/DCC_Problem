# `m4_logic_closure_v1_20261120` 사전 프로토콜 (Colab L4)

작성일: 2026-11-20  
실행 ID: `m4_logic_closure_v1_20261120`  
목적:
1. **`Full vs Prefix` 동일 조건 통제 비교 완결:** `E001`(예선 전체 통화)과 `E002`(본선 `Prefix` `9,855`스텝) 간 학습량·레시피 비대칭을 제거하고, `E002`와 동일한 `TAPT KLUE-RoBERTa-base`(초기 SHA-256 `254249750ed0dae0b2e50a93874fad537f5329dabfd83a3cc4de2a4f58f42b4e`)·`fit 26,280`·`lr 5e-5`·`LLRD 0.8`·`seed 42`·`9,855 steps`(`157,680` 표본 노출) 조건에서 **입력만 통화 전체(`Full-only`)** 로 학습해 `E002`와 정면 비교한다. (`epoch 3` 진단 스냅샷도 함께 저장해 `동일 epoch` 대비 `동일 step` 효과를 함께 분리한다.)
2. **백본 표 `E008(KF-DeBERTa)` · `E019(RoBERTa-large)` `raw p >= 0.5` 재현 확보:** 팀원 원본 `.npz` 미보유 공백을 채우기 위해 팀원 원본 `run_config.json` 레시피 그대로 Colab L4에서 독립 재현(`fit 26,280 / Dev 2,920`)하고, 팀원 원본 지표(`E008` `0.5354`, `E019` `maxent 0.5451`) 근사 재현 여부를 교차 확인한 뒤 공식 `raw p >= 0.5`(Top-1 없음) 점수와 `95%` 쌍 부트스트랩 구간을 산출한다. (만약 추후 팀원 원본 `.npz`가 확보되면 백본 표 주 수치는 팀원 원본 재채점으로 대체하고 본 실행은 독립 재현 기록으로 분리 보존한다.)

---

## 1. 통제 조건 및 단계 정의

- **데이터·분할:** `train.jsonl` 내 `fit_ids.txt` (`26,280`통화) 학습, `dev_ids.txt` (`2,920`통화, `87,814`시점) 평가. 공식 `Validation` (`3,640`통화) 열람·접근 금지.
- **공통 조건:** `fmt="space"`, `max_length=512`, `truncation_side="right"`, `pos_weight_power=0.5`, `seed=42`, Weighted BCE(`E002` 동일).

| 순서 | 단계명 | 초기 모델 | 샘플링·노출 | 배치 / LR / LLRD / gc | 총 스텝 / 업데이트 |
|---:|---|---|---|---|---|
| **1** | `full_only_roberta_tapt_s42` | `E002` 동일 `TAPT klue/roberta-base` | `full_only`, `6 epoch × 1/통화` (`epoch 3` 스냅샷 병행) | `batch 16 × 1`, `lr 5e-5`, `LLRD 0.8`, `gc=False` | `9,855 / 9,855` (`epoch3: 4,928`) |
| **2** | `e008_kfdeberta_repro_s42` | `kakaobank/kf-deberta-base` (No-TAPT) | `uniform_prefix`, `3 epoch × 2/통화` | `batch 8 × 2` (실효 `16`), `lr 2e-5`, `LLRD 1.0`, `gc=False` | `19,710 / 9,855` |
| **3** | `e019_roberta_large_repro_s42` | `klue/roberta-large` (No-TAPT) | `uniform_prefix`, `3 epoch × 2/통화` | `batch 16 × 1`, `lr 5e-5`, `LLRD 0.8`, `gc=True` | `9,855 / 9,855` |

---

## 2. 보존 및 검증 기준

- 각 단계는 완료 즉시 `best_model/`, `final.pt`, `training_state_last.pt`, `initialization.json`, `run_config.json`, `optimizer_audit.json`, `train_log.txt`, `dev_prefix.npz`, `dev_prefix.meta.json` (1단계는 `snapshot_ep3` 포함)을 로컬 Mac `artifacts_private/runs/m4_logic_closure_v1_20261120/`으로 내려받아 SHA-256 검증 후 원격 임시 파일을 정리한다.
- 모든 모델은 오프라인 CPU `8`프리픽스 일치, `score.py` 경고 `0`건, `Dev 2,920` `raw p >= 0.5` (`5,000`회 paired bootstrap vs `E002` · `ENS3`), `ENS_swap`/`ENS4_add`, `4-2 C128_v1_nonempty` 검증을 거친다.
- 전체 종료 시 `finally`에서 Colab VM을 정지해 `Active assignments: 0`을 확인한다.
