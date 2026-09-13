# Mission 3 — 환자 증상 다중 라벨 분류: 512 토큰 Truncation 복구 검증 결과

- **분석 대상 백본**: `KLUE-RoBERTa-base` (seed 42, ASL)
- **검증 데이터**: Validation 3,640건 (`mission3_val.csv`)
- **분석 목적**: 오분류 분석의 Head-Tail Truncation 가설이 Validation Macro F1을 개선하는지, **추론 입력 변경**과 **학습 입력 변경** 두 단계로 정량 검증함.

---

## 1. 개요 및 분석 배경

KoBERT / BERT tokenizer의 기본 `truncation=True`는 앞쪽 `max_length`(512)만 보존함. 119 신고 통화는 초반 신원·주소 문진이 길어, 후반부 동반 증상 호소가 입력에서 절단되어 FN이 발생할 수 있음.

본 검증은 아래 두 실험을 수행함.

| 실험 | 내용 |
|---|---|
| **실험 1** | 기학습 체크포인트에 추론 입력만 `truncate` / `head_tail` / `sliding`으로 변경 |
| **실험 2** | `dataset.py`의 `encode_mode=head_tail`로 학습·검증 입력을 동일하게 맞춘 뒤 재학습 |

구현: `mission3_symptom/m3/truncation.py`, `eval_truncation.py`, `encode_mode` 옵션 (`dataset.py`, `train.py`).  
`inference.py` Mission 3 제출 경로는 연결하지 않음. 기본 `encode_mode`는 `truncate`로 유지함.

---

## 2. 실험 1 — 추론 입력만 변경

- **체크포인트**: `klue_roberta_base_asl_seed42_val_loss` (앞 512 truncation으로 학습)
- **스크립트**: `mission3_symptom/eval_truncation.py`

| 입력 방식 | 기준(0.5) Macro F1 | Optimized Macro F1 | 기준선 대비 F1@0.5 | 기준선 대비 Optimized |
|---|---:|---:|---:|---:|
| **truncate** | 0.5309 | 0.6501 | — | — |
| **head_tail** | 0.5305 | 0.6500 | **-0.0003** | **-0.0001** |
| **sliding** | 0.5295 | 0.6500 | **-0.0013** | **0.0000** |

- **512토큰 초과 표본**: 98건 / 3,640건 (약 `2.69%`)
- **첫 창 밖에만 정답 증상 문자열이 존재하는 표본** (`cut_calls`): **0건**

원본: `mission3_symptom/reports/truncation_val.json` (코랩 산출물, 저장소 미업로드)

---

## 3. 실험 2 — 학습 입력 Head-Tail 재학습

학습과 검증 모두 512 초과 통화에 대해 **앞 128토큰 + 꼬리**를 사용함. 그 외 설정은 실험 1 기준선과 동일함 (ASL, seed 42, 3 epoch, `val_loss` checkpoint).

| Run | `encode_mode` | 기준(0.5) Macro F1 | 기준선 대비 |
|---|---|---:|---:|
| `klue_roberta_base_asl_seed42_val_loss` | truncate | 0.5309 | — |
| `klue_roberta_base_asl_seed42_head_tail_v2` | head_tail | 0.5303 | **-0.0006** |

노트북 설정 및 해당 run의 `run_config.json`에 `"encode_mode": "head_tail"`이 확인된 재학습 결과임.

---

## 4. 관찰 및 해석

- **관찰 현상**: 추론만 바꿔도, 학습까지 Head-Tail로 맞춰도 Macro F1@0.5는 기준선과 오차 수준으로 동일함.
- **원인 분석**:
  1. **Truncation 영향 범위가 제한적**: Validation에서 512 초과는 98건(약 2.7%)에 불과함.
  2. **후반부 타겟 문자열 절단이 드묾**: 실험 1에서 정답 증상 문자열이 첫 창 밖에만 있는 표본은 0건이었음.
  3. **학습-추론 불일치만의 문제는 아님**: 실험 2에서 입력을 학습·검증 모두 Head-Tail로 통일해도 F1이 상승하지 않음.

---

## 5. 결론 및 시사점

1. **추론 단계 Head-Tail / Sliding은 현 체크포인트에서 Macro F1을 견인하지 못함.**
2. **학습 단계 Head-Tail 재학습도 동일 조건에서 Macro F1@0.5를 개선하지 못함** (0.5309 → 0.5303).
3. 오분류 보고서의 Truncation FN 가설은 정성 분석으로는 유효할 수 있으나, **본 정량 실험에서는 전체 지표 개선으로 이어지지 않음.**
4. 코드의 `encode_mode=head_tail` 옵션은 유지하되, **기본 학습·제출 경로는 `truncate`를 유지하는 것이 타당함.**
