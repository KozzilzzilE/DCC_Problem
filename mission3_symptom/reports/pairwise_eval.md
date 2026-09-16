# Mission 3 — 환자 증상 다중 라벨 분류: Pairwise Ranking Loss 검증 결과

- **분석 대상 백본**: `KLUE-RoBERTa-base` (seed 42, plain BCE 대비)
- **검증 데이터**: Validation 3,640건 (`mission3_val.csv`)
- **분석 목적**: 양성/음성 증상 쌍에 ranking 제약을 더한 `PairwiseRankingLoss`가 Validation Macro F1 및 오심/구토 F1을 개선하는지 정량 검증함.

---

## 1. 개요 및 분석 배경

기본 BCE는 9개 증상을 독립 이진 분류로 학습함. Pairwise Ranking은 같은 통화에서 정답이 1인 증상의 logit이 0인 증상보다 커지도록 `softplus(s_neg - s_pos)` 평균을 BCE에 더함.

Label Dependency Loss는 동시 발생 확률을 붙이는 제약이었고 Macro F1을 개선하지 못했음. 본 실험은 같은 KLUE 기준선에서 **loss만** ranking으로 바꿔 한 번 더 확인함.

구현: `m3/losses.py`의 `PairwiseRankingLoss`, `--loss-type pairwise`, `--pairwise-alpha`(기본 0.1).  
기본 학습 경로의 `loss_type`은 `bce`로 유지함.

---

## 2. 실험 설정

기준선과 **loss만** 다름. backbone, seed, epoch, LR, batch, max_length, pooling(`cls`), encode_mode(`truncate`)는 동일함.

| 항목 | 기준선 | 본 실험 |
|---|---|---|
| Run | KLUE-RoBERTa + plain BCE seed42 | `klue_roberta_base_bce_pairwise_a01_seed42` |
| `loss_type` | bce | pairwise |
| `pairwise_alpha` | — | 0.1 |
| epoch | 3 | 3 |
| Best checkpoint | epoch 2 | epoch 2 |
| Train / Val | 29,200 / 3,640 | 29,200 / 3,640 |

---

## 3. Validation 성능

### 3.1 임계값 0.5

| 지표 | BCE (기준선) | pairwise | 차이 |
|---|---:|---:|---:|
| Macro F1@0.5 | 0.6003 | 0.5979 | **-0.0024** |
| 오심 F1@0.5 | 0.0529 | 0.0359 | **-0.0170** |
| 구토 F1@0.5 | 0.5860 | 0.5852 | **-0.0008** |

클래스별 F1@0.5 (pairwise):

| 증상 | F1@0.5 |
|---|---:|
| 고열 | 0.6870 |
| 구토 | 0.5852 |
| 두통 | 0.5169 |
| 복통 | 0.8160 |
| 어지러움 | 0.6424 |
| 열상 | 0.8946 |
| **오심** | **0.0359** |
| 전신쇠약 | 0.5480 |
| 호흡곤란 | 0.6553 |

### 3.2 클래스별 임계값 최적화

같은 Validation에서 threshold를 고른 결과이며, 제출 점수보다 낙관적임.

| 지표 | BCE (기준선) | pairwise |
|---|---:|---:|
| Optimized Macro F1 | 0.6554 | 0.6551 |
| 오심 optimized F1 | 0.4030 | 0.4012 |

원본: `mission3_symptom/runs/klue_roberta_base_bce_pairwise_a01_seed42/` (코랩 산출물, 저장소 미업로드)

---

## 4. 관찰 및 해석

- **관찰 현상**: Macro F1@0.5는 기준선과 오차 수준이며 소폭 하락함. Optimized Macro F1도 BCE와 거의 동일함. 오심 F1@0.5는 Label Dependency와 같이 하락함.
- **Label Dependency와의 비교**: dependency Macro F1@0.5 0.5981, 오심 0.0359. pairwise와 차이가 없음.
- **원인 분석**: ranking 제약은 “있는 증상 점수가 없는 증상보다 커야 한다”는 가정임. 공식 라벨에서 구토만 1이고 오심은 0인 경우가 많아, 오심 점수를 올리는 방향이 0.5 기준 오심 F1에 불리할 수 있음.
- pooling(Label-wise Attention)은 이미 별도 ablation에서 기각되었으므로, pairwise와 조합하는 추가 실험은 하지 않음.

---

## 5. 결론 및 시사점

1. **Pairwise Ranking Loss(alpha=0.1)는 현 KLUE + BCE 기준선을 Macro F1에서 개선하지 못함.**
2. 병목인 오심 F1도 개선되지 않았고, 0.5 기준에서는 하락함.
3. `--loss-type pairwise` 옵션은 코드에 유지하되, **기본 학습·제출 경로는 plain BCE를 유지하는 것이 타당함.**
4. loss·truncation·pooling·backbone 교체는 모두 0.655 부근을 넘지 못했으므로, 추가 loss 조합보다 **KLUE + CLS + plain BCE로 고정하고 제출 경로를 정리**하는 것이 맞음.
