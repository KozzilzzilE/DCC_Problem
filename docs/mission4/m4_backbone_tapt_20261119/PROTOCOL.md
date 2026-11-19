# `m4_backbone_tapt_20261119` 사전 실험 프로토콜 (전략 B: 4대 백본 동일 TAPT 공정 비교)

- 작성일: 2026-11-19
- 실행 환경 예정: Colab L4 (잔여 `47.54` compute units, 현재 활성 세션 `0`)
- 자체 종료 기한: 2026-11-21
- 직전 근거: `mission4-streaming` `E002` (`RoBERTa+TAPT 0.502733`), `E021` (`KcELECTRA No-TAPT 0.4957`), `E022` (`KcELECTRA+TAPT 0.498099`), 및 팀원 `ohyun0628` 백본 벤치마크(`KR-BERT-char`, `KoELECTRA-v3`, `RoBERTa-large`, `KF-DeBERTa`, `KoGPT2`)

---

## 1. 실험 목적 및 검증 질문

기존 백본 비교는 기준선 `klue/roberta-base`(`E002`)와 `KcELECTRA`(`E022`)에만 `20-epoch TAPT`가 적용되었고, `KR-BERT-char` 및 `KoELECTRA-v3`는 `fit 26,280` 공정 재판정 및 `TAPT` 검증이 부재했으며, `No-TAPT klue/roberta-base` 대조군도 부재했다. 본 실험(전략 B)은 배포 가능한 경량 `base`급 4대 한국어 백본 전원에 동일한 20-epoch TAPT를 적용하고 `No-TAPT RoBERTa` 대조군을 추가해 아래 세 질문에 답한다.

1. **순수 Mission 4 TAPT 기여도:** `klue/roberta-base`에서 TAPT 20 epoch 유무(`roberta_notapt_s42` vs `E002 0.502733`)가 공식 `raw p >= 0.5` 4-1 점수에 주는 순수 이득은 얼마인가?
2. **4대 경량 백본 동일 TAPT 정면 비교:** 동일 `fit 26,280` 본문 TAPT 20 epoch + 동일 `E002` Prefix 3 epoch 레시피에서 `KLUE-RoBERTa-base (E002)`, `KcELECTRA-base-v2022 (E022)`, `KR-BERT-char16424 (krbert_tapt20_s42)`, `KoELECTRA-base-v3-discriminator (koelectra_tapt20_s42)`의 공식 `Dev 2,920` `raw p >= 0.5` 성능 서열은 어떠한가?
3. **동일 `No-TAPT` 기준점 보완:** `roberta_notapt_s42`는 기존 `No-TAPT` 실행들(`E021 0.4957`, `E019`, `E008`)과의 동일 조건 비교 기준선 역할을 함께 수행한다.

---

## 2. 실행 대상 및 고정 하이퍼파라미터

### 실행 3건 (모두 `fit 26,280 / dev 2,920` 분할, `seed=42`, 공식 `raw p >= 0.5` · Top-1 강제 없음)
1. **`roberta_notapt_s42`**: `klue/roberta-base` (No-TAPT) → 3-epoch Prefix + `dev_prefix.npz`
2. **`krbert_tapt20_s42`**: `snunlp/KR-BERT-char16424` → 20-epoch TAPT MLM → 3-epoch Prefix + `dev_prefix.npz`
3. **`koelectra_tapt20_s42`**: `monologg/koelectra-base-v3-discriminator` → 20-epoch TAPT MLM → 3-epoch Prefix + `dev_prefix.npz`

### TAPT MLM 고정 설정 (팀 `runs/devsel/tapt` 및 `E022` 동일)
- **입력 데이터:** `fit_ids.txt` (`26,280`건)의 전사 본문만 사용 (`" ".join(t.strip() for t in c.texts if t and t.strip())`). 라벨·`dev_ids.txt`·공식 `Validation` 일체 미사용(코드 내 `forbid-ids` 겹침 및 `"val"` 경로 차단 검사).
- **하이퍼파라미터:** `epochs=20`, `learning_rate=5e-5`, `batch_size=8`, `gradient_accumulation_steps=2` (실효 배치 `16`), `warmup_ratio=0.06`, `weight_decay=0.01`, `max_length=512`, `mlm_probability=0.15`, `seed=42`, AMP FP16.
- **재개 보존:** 매 epoch 종료 시 `training_state_last.pt` (모델·옵티마이저·스케줄러·scaler·DataLoader generator·RNG·history) 저장 및 최종 완료 시에도 보존.

### Prefix 분류 학습 고정 설정 (팀 `E002` 레시피 완전 동일)
- **입력 데이터:** `fit_ids.txt` (`26,280`건), `dev_ids.txt` (`2,920`건, 학습 중 `forbid-ids`로 ID·정규화 전사 겹침 0건 검증).
- **하이퍼파라미터:** `epochs=3`, `samples_per_call=2`, `batch_size=16`, `grad_accum=1`, `lr=5e-5`, `llrd=0.8`, `pos_weight_power=0.5`, plain weighted BCE (`precue_pos_weight=-1.0`, `jaccard_weight=0.0`, `macro_f1_weight=0.0`), `max_length=512`, `fmt=space`, `truncation_side=right`, `seed=42`, AMP FP16.
- **아키텍처·토크나이저 안전 검사:**
  - Pre-flight 단계에서 3종 토크나이저 UNK 비율(`< 0.05` 필수) 및 L4 2스텝 스모크(OOM/NaN 없음) 사전 검증.
  - 분류 헤드 및 풀러 키 외 `mismatched_keys` / `error_msgs` 불허, `llrd_groups` 전체 `requires_grad` 파라미터 100% 포함 단언.
  - 최종 `best_model/`, `final.pt`, `training_state_last.pt`, `dev_prefix.npz` 전량 로컬 회수 및 SHA-256 · CPU 오프라인 추론 일치 검증.

---

## 3. 사전 등록 판단 기준 (결과 확인 전 고정)

| 기준 | 판정 규칙 | 근거 |
|---|---|---|
| 주 지표 (`Dev 2,920` 4-1 `raw p >= 0.5`) | 신규 TAPT 백본이 `E002` (`0.502733`)을 **`> +0.002` 초과**하고(쌍 부트스트랩 가능 시 95% CI 하한 `> 0`), 2026-11-21 기한 내 후속 파이프라인(`SoftJaccard` + 5겹 OOF) 재학습 실익이 입증되는 경우에 한해 백본 교체 검토. 그렇지 않으면 `klue/roberta-base + TAPT` 유지. | `E021`/`E022` 사전 게이트 및 11/21 마감·하류 재학습 비용 고려 |
| 사전 중단(Re-plan) 조건 | Pre-flight UNK율 `>= 0.05`, 단일 TAPT 예상시간 `> 5h` / L4 OOM, 비유한 loss/NaN 발산, 잔여 유닛 `< 20` 급감, 또는 신규 백본의 `E002 +0.002` 초과 달성 시 즉시 계획 모델에 보고·판단 이관. | 자원 보호 및 과학적 결론의 상위 모델 검토 원칙 (`AGENTS.md`) |
