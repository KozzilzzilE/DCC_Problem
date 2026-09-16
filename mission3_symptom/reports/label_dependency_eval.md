# Mission 3 — 환자 증상 다중 라벨 분류: Label Dependency Loss 검증 결과

- **분석 대상 백본**: `KLUE-RoBERTa-base` (seed 42, plain BCE 대비)
- **검증 데이터**: Validation 3,640건 (`mission3_val.csv`)
- **분석 목적**: 증상 간 동시 발생(Co-occurrence)을 BCE에 페널티로 더한 `LabelDependencyLoss`가 Validation Macro F1 및 오심/구토 F1을 개선하는지 정량 검증함.

---

## 1. 개요 및 분석 배경

9개 증상은 독립이 아니라 통화 안에서 함께 등장하는 쌍이 있음. 특히 오심과 구토는 텍스트상 동반 표현이 잦음.

`LabelDependencyLoss`는 기본 BCE에, 학습셋에서 구한 동시 발생 행렬을 곱한 확률 차이 페널티를 더함. 자주 같이 나오는 증상끼리는 예측 확률이 비슷해지도록 유도함.

오분류·라벨 확인에서는 Validation의 `오심=0, 구토=1`이 다수였고, 구토가 있으면 오심을 안 찍는 쪽에 가까웠음. 본 실험은 그 제약과 loss 가정이 충돌하지 않는지 F1으로 확인함.

구현: `m3/losses.py`의 `LabelDependencyLoss`, `--loss-type dependency`, `--dependency-alpha`(기본 0.1).  
기본 학습 경로의 `loss_type`은 `bce`로 유지함.

---

## 2. 실험 설정

기준선과 **loss만** 다름. backbone, seed, epoch, LR, batch, max_length는 동일함.

| 항목 | 기준선 | 본 실험 |
|---|---|---|
| Run | KLUE-RoBERTa + plain BCE seed42 | `klue_roberta_base_bce_dependency_a01_seed42_v4` |
| `loss_type` | bce | dependency |
| `dependency_alpha` | — | 0.1 |
| 동시 발생 행렬 | — | Training 라벨 `P(i,j)` |
| epoch | 3 | 3 |
| Best checkpoint | epoch 2 | epoch 2 |

---

## 3. Validation 성능 (임계값 0.5)

| 지표 | BCE (기준선) | dependency | 차이 |
|---|---:|---:|---:|
| Macro F1@0.5 | 0.6003 | 0.5981 | **-0.0022** |
| 오심 F1@0.5 | 0.0529 | 0.0359 | **-0.0170** |
| 구토 F1@0.5 | 0.5860 | 0.5817 | **-0.0043** |

클래스별 F1@0.5 (dependency):

| 증상 | F1@0.5 |
|---|---:|
| 고열 | 0.6811 |
| 구토 | 0.5817 |
| 두통 | 0.5177 |
| 복통 | 0.8153 |
| 어지러움 | 0.6474 |
| 열상 | 0.8884 |
| **오심** | **0.0359** |
| 전신쇠약 | 0.5529 |
| 호흡곤란 | 0.6629 |

원본: `mission3_symptom/runs/klue_roberta_base_bce_dependency_a01_seed42_v4/` (코랩 산출물, 저장소 미업로드)

---

## 4. 관찰 및 해석

- **관찰 현상**: Macro F1@0.5는 기준선과 오차 수준이며 소폭 하락함. 오심 F1@0.5는 더 낮아짐.
- **원인 분석**:
  1. 동시 발생 페널티는 오심·구토 확률을 붙이려 함.
  2. 공식 라벨은 구토가 있으면 오심을 0으로 두는 경우가 많음.
  3. 그 결과 오심 양성을 더 올리려다 0.5 기준 오심 F1이 하락한 것으로 해석함.

---

## 5. 결론 및 시사점

1. **Label Dependency Loss(alpha=0.1)는 현 KLUE + BCE 기준선을 Macro F1에서 개선하지 못함.**
2. 병목인 오심 F1도 개선되지 않았고, 0.5 기준에서는 하락함.
3. `--loss-type dependency` 옵션은 코드에 유지하되, **기본 학습·제출 경로는 plain BCE를 유지하는 것이 타당함.**
4. 다음 실험 후보는 Pairwise Ranking이며, 개선이 없으면 KLUE + plain BCE로 고정하고 제출 경로를 정리하는 것이 맞음.
