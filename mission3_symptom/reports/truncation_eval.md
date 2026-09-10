# Mission 3 — 환자 증상 다중 라벨 분류: 512 토큰 Truncation 복구 검증 결과

- **분석 대상 모델**: `KLUE-RoBERTa-base` (seed 42, ASL, run `klue_roberta_base_asl_seed42_val_loss`)
- **검증 데이터**: Validation 3,640건 (`mission3_val.csv`)
- **담당**: 권오현
- **분석 목적**: 오분류 분석에서 도출한 Head-Tail Truncation 가설이, 기존 학습 체크포인트의 Validation Macro F1을 실제로 개선하는지 정량 검증함.

---

## 1. 개요 및 분석 배경

KoBERT / BERT tokenizer의 기본 `truncation=True`는 앞쪽 `max_length`(512)만 보존함. 119 신고 통화는 초반 신원·주소 문진이 길어, 후반부 동반 증상 호소가 입력에서 절단되어 FN이 발생할 수 있음.

이에 따라 학습 파이프라인(`dataset.py`)은 변경하지 않고, 기학습 `best_model`에 대해 **추론 입력만** 세 가지로 바꿔 Validation 성능을 비교하였음.

| 입력 방식 | 설명 |
|---|---|
| **truncate** | 기존과 동일하게 앞 512토큰만 사용 (기준선) |
| **head_tail** | 앞 128토큰(주호소) + 뒤 나머지(세부 문진)를 결합하여 1회 forward |
| **sliding** | 본문을 중첩 창으로 분할한 뒤 창별 확률을 클래스별 `max`로 병합 |

구현 위치: `mission3_symptom/m3/truncation.py`, 평가 스크립트: `mission3_symptom/eval_truncation.py`.  
`inference.py`의 Mission 3 제출 경로는 본 실험에 연결하지 않음.

---

## 2. Validation 성능 비교

| 입력 방식 | 기준(0.5) Macro F1 | Optimized Macro F1 | 기준선 대비 F1@0.5 | 기준선 대비 Optimized |
|---|---:|---:|---:|---:|
| **truncate** | 0.5309 | 0.6501 | — | — |
| **head_tail** | 0.5305 | 0.6500 | **-0.0003** | **-0.0001** |
| **sliding** | 0.5295 | 0.6500 | **-0.0013** | **0.0000** |

- **512토큰 초과 표본**: 98건 / 3,640건 (약 `2.69%`, 기존 KLUE truncation 비율과 일치)
- **첫 창 밖에만 정답 증상 문자열이 존재하는 표본** (`cut_calls`): **0건**

원본 산출물: `mission3_symptom/reports/truncation_val.json` (로컬/코랩 run 파일, 저장소에는 업로드하지 않음)

---

## 3. 관찰 및 해석

- **관찰 현상**: Head-Tail 및 Sliding 모두 기준선(앞 512 truncation) 대비 Macro F1이 개선되지 않음. 변화량은 오차 수준이며, Sliding은 F1@0.5에서 소폭 하락함.
- **원인 분석**:
  1. **Truncation 영향 범위가 제한적**: Validation에서 512를 넘는 통화는 98건에 불과하여, 입력 방식 변경이 전체 Macro F1을 움직이기 어려움.
  2. **후반부 타겟 문자열 절단이 드묾**: 9개 공식 증상 문자열이 첫 창 밖에만 위치한 표본은 본 탐지 기준으로 0건이었음. 후반 FN이 존재하더라도 표현이 타겟 문자열과 일치하지 않거나, 초반에도 동일 증상이 언급된 경우가 많음을 시사함.
  3. **학습-추론 입력 불일치**: 본 실험은 앞 512 truncation으로 학습된 체크포인트에 추론만 꼬리를 재투입한 설정임. 모델이 후반 문맥을 학습하지 않은 상태에서 입력만 바꾸면 이득이 제한됨.

---

## 4. 결론 및 시사점

1. **추론 단계만의 Head-Tail / Sliding은 현 체크포인트에서 Macro F1을 견인하지 못함.**
2. 오분류 보고서의 Truncation FN 가설은 정성 분석으로는 유효하나, **정량 Validation에서는 전체 지표 개선으로 이어지지 않음.**
3. `dataset.py`에 Head-Tail을 삽입하여 재학습하는 방안은 학습 입력이 바뀌므로 모델링 담당(김완수)의 재학습이 필요함. 본 수치만으로는 재학습 우선순위가 높지 않으며, 적용 여부는 팀 합의 후 결정하는 것이 타당함.
