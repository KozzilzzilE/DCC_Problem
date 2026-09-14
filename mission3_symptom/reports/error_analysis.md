# Mission 3 — 환자 증상 다중 라벨 분류 Global Error Analysis

- **기준 run**: `klue_roberta_base_plain_bce_seed42_val_loss`
- **기준 설정**: `klue/roberta-base`, plain BCE, seed 42, `encode_mode=truncate`, sampling OFF
- **체크포인트 기준**: minimum validation loss
- **기준 성능**: optimized Macro F1 `0.6554208986`
- **분석 목적**: 오심/구토 중심의 초기 오분류 분석과 truncation·sampling ablation을 9개 클래스 전체 분석으로 확장하여 다음 구조적 개선 후보를 결정함.

이 문서는 공개 가능한 집계 통계만 포함한다. 원문 transcript, 개별 Validation 표본, `call_id` 및 그 밖의 표본 식별 정보는 포함하지 않는다.

---

## 1. 분석 경과와 결론의 변화

초기에는 F1이 가장 낮은 오심과 의미적으로 인접한 구토를 중심으로 대표 FP·FN·TP 95건을 살폈다. 정성 분석에서는 오심 표현의 라벨 누락 가능성, 긴 신고 통화의 후반부 증상 유실, 오심 단독 표본 부족이 주요 가설로 제기되었다. 이에 따라 다음 두 실험을 순서대로 확인했다.

1. `head_tail` 및 `sliding`으로 512 토큰 이후의 정보를 복구하는 truncation 실험
2. 오심=1·구토=0인 pure-nausea 그룹만 약하게 더 노출하는 sampling 실험

두 실험 모두 전체 Macro F1을 개선하지 못했다. 이후 분석 범위를 9개 클래스 전체와 label co-occurrence로 넓힌 결과, 더 큰 병목은 특정 두 클래스의 혼동이나 입력 길이보다 **한 표본에 여러 증상이 함께 있을 때 실제 positive label을 누락하는 FN 증가**로 나타났다. 따라서 다음 후보는 데이터 재가중보다 클래스별로 서로 다른 token evidence를 모으는 **label-wise evidence aggregation**으로 이동한다.

---

## 2. 기준 모델의 클래스별 성능

| 클래스 | Optimized F1 |
|---|---:|
| 오심 | 0.4030 |
| 두통 | 0.5525 |
| 전신쇠약 | 0.5894 |
| 구토 | 0.6066 |
| 어지러움 | 0.6584 |
| 호흡곤란 | 0.6757 |
| 고열 | 0.7023 |
| 복통 | 0.8239 |
| 열상 | 0.8869 |

오심이 가장 취약하지만 두통·전신쇠약·구토도 상대적으로 낮다. 특히 뒤의 global 분석에서 확인되듯 이들은 단일한 오심/구토 축으로만 설명되지 않는다.

---

## 3. 오심/구토 집중 분석에서 확인한 사항

### 3.1 오심: 낮은 threshold와 score overlap

- F1 `0.4030`, precision `0.3165`, recall `0.5548`
- FP `514`, FN `191`
- optimized threshold `0.19`
- AUROC `0.7752`, AP `0.3230`

오심은 threshold를 0.5에서 0.19로 내리면 F1이 크게 회복되므로 calibration 영향이 있다. 그러나 optimized threshold에서도 positive와 negative score가 상당히 겹치고 FP가 514건 남아 있어 threshold 문제만으로 볼 수 없다.

기존 대표 오류 정성 분석에서는 구토와 함께 언급된 오심 표현이 정답에 반영되지 않았을 가능성, 즉 annotation ambiguity가 일부 관찰되었다. 반대로 명시적인 오심 표현이 충분히 드러난 표본은 상대적으로 안정적으로 검출되었다. 다만 이 관찰은 제한된 표본에서 얻은 정성 가설이며, 전체 오류를 라벨 노이즈 하나로 단정할 근거는 아니다.

### 3.2 오심과 구토의 직접 혼동은 주된 global 병목이 아님

`A`가 실제 positive인데 누락되고 `B`가 실제 negative인데 잘못 활성화된 경우를 방향성 pair로 집계하면 다음과 같다.

- 구토 → 오심: `28 / 369 = 7.59%` (전체 pair 중 약 16위)
- 오심 → 구토: `4 / 261 = 1.53%` (약 58위)

의미적 연관성과 일부 혼동은 존재하지만, 두 방향 모두 전체 Macro F1을 지배하는 최상위 오류는 아니다. 따라서 오심/구토만을 더 강하게 분리하거나 재샘플링하는 접근의 기대효과는 제한적이다.

### 3.3 truncation 가설의 정량 검증

신고 통화는 초반의 위치·상황 확인 뒤에 세부 증상이 등장할 수 있어, 앞 512 토큰만 남기는 방식이 FN을 만들 수 있다는 가설 자체는 타당했다. KLUE-RoBERTa의 Validation 512 토큰 초과 비율은 KoBERT의 `5.38%`보다 낮은 `98 / 3,640 = 2.69%`였고, 전체적으로 TP와 FN의 입력 길이 차이도 작았다.

별도 truncation 평가에서도 다음 결과가 확인되었다.

- 추론 입력만 `head_tail` 또는 `sliding`으로 바꿔도 optimized Macro F1이 개선되지 않음
- 학습·검증을 모두 `head_tail`로 맞춘 재학습에서도 Macro F1@0.5가 `0.5309 → 0.5303`으로 개선되지 않음

따라서 truncation은 일부 개별 오류를 설명할 수 있으나 현 Validation 전체의 주된 병목은 아니다. `head_tail`은 재현 가능한 옵션으로 남기되 기본 경로는 `truncate`가 적절하다.

### 3.4 pure-nausea sampling 결과

Training의 pure-nausea 그룹(오심=1, 구토=0)에만 `WeightedRandomSampler` weight `1.5`를 적용한 결과는 다음과 같다.

| 지표 | Baseline | Sampling | 변화 |
|---|---:|---:|---:|
| Optimized Macro F1 | 0.6554 | 0.6485 | -0.0069 |
| 오심 F1 | 0.4030 | 0.4081 | +0.0051 |
| 오심 precision | 0.3165 | 0.3767 | +0.0602 |
| 오심 recall | 0.5548 | 0.4452 | -0.1096 |
| 구토 F1 | 0.6066 | 0.5886 | -0.0180 |
| 두통 F1 | 0.5525 | 0.5231 | -0.0294 |

오심의 TP는 47건 줄고 FP는 198건 줄었으며, pure-nausea Validation 그룹 recall도 `47.89% → 35.25%`로 하락했다. 즉 오심 F1의 소폭 상승은 pure-nausea 복구보다 precision/recall trade-off에 가깝고, 다른 클래스의 동반 하락으로 전체 성능은 낮아졌다. 더 강한 pure-nausea sampling은 현재 우선순위가 낮으며 sampling 기본값은 OFF가 타당하다.

---

## 4. Global Error Analysis

### 4.1 positive label 수가 늘수록 FN이 급증

| 표본의 positive label 수 | FN rate |
|---:|---:|
| 1개 | 23.15% |
| 2개 | 39.79% |
| 3개 | 44.29% |
| 4개 이상 | 57.84% |

단일 증상 표본보다 다증상 표본에서 실제 label을 빠뜨리는 비율이 일관되게 높다. 4개 이상에서는 절반이 넘는 positive instance가 누락된다. 이는 클래스 하나의 희소성보다 **공유된 단일 문장 표현이 여러 label의 서로 다른 evidence를 충분히 보존하지 못하는 문제**를 우선 검토해야 한다는 근거다.

### 4.2 co-occurrence 상황에서의 recall 저하

| 클래스 | 단독 등장 recall | 다른 label과 동반 시 recall |
|---|---:|---:|
| 호흡곤란 | 77.85% | 44.32% |
| 고열 | 77.88% | 55.75% |
| 두통 | 53.69% | 40.44% |

두통은 전체 `374`개 positive 중 `203`개를 놓쳐 FN rate가 `54.28%`이고, 동반 label이 있을 때 recall이 더 낮다. 반면 오심은 단독 recall `34.48%`, 동반 recall `57.00%`로 반대 양상을 보여 단순히 “동반 label이 많으면 모든 클래스가 동일하게 어려워진다”거나 “positive 표본이 적어서 그렇다”고 일반화할 수 없다. 클래스별 evidence aggregation이 필요한 이유도 이 이질성에 있다.

### 4.3 confusion-like pair는 전신쇠약·오심으로의 과활성화가 큼

| 누락된 실제 label → 잘못 활성화된 label | 건수 |
|---|---:|
| 어지러움 → 전신쇠약 | 89 |
| 복통 → 전신쇠약 | 55 |
| 호흡곤란 → 전신쇠약 | 47 |
| 고열 → 전신쇠약 | 47 |
| 전신쇠약 → 오심 | 47 |
| 복통 → 오심 | 47 |
| 구토 → 전신쇠약 | 40 |
| 두통 → 오심 | 38 |
| 두통 → 전신쇠약 | 38 |
| 오심 → 전신쇠약 | 38 |

전신쇠약은 precision `0.5072`, FP `445`로 여러 구체적 증상 문맥에서 generic한 양성으로 활성화되는 경향이 있다. 오심도 일부 클래스의 evidence를 흡수하는 FP sink로 나타난다. 하나의 global representation을 모든 label이 공유하면 넓은 의미의 label이 구체적인 label의 신호를 흡수할 수 있다는 가설과 맞닿아 있다.

### 4.4 backbone을 바꿔도 남는 공통 오류

KoBERT, KoELECTRA, KLUE-RoBERTa의 저장된 prediction artifact를 같은 기준으로 비교하면 다음과 같다.

- 세 backbone 공통 FN: `1,479 / 5,236 = 28.25%`
- 세 backbone 공통 FP: `1,207 / 27,524 = 4.39%`
- 클래스별 공통 FN: 두통 `44.39%`, 구토 `38.92%`, 오심 `38.00%`

백본 교체만으로 해소되지 않는 오류가 상당하다. 현재 KLUE-RoBERTa가 세 모델 중 가장 높은 기준 성능을 보인 점까지 고려하면, 다음 실험은 새 backbone보다 같은 encoder에서 label별 증거를 분리해 모으는 head 구조가 더 직접적인 가설이다.

---

## 5. Follow-up: Label-wise Attention

### 5.1 가설과 설계

Global Error Analysis에서 positive label 수가 많을수록 FN rate가 증가한 원인을, 하나의 CLS representation이 여러 증상의 서로 다른 token evidence를 충분히 보존하지 못하기 때문이라고 가정했다. 이를 검증하기 위해 다른 학습 조건은 유지하고 pooling만 바꾼 단일 ablation을 수행했다.

- **기존 CLS**: first token representation 하나를 shared classifier에 전달해 9개 logit 생성
- **Label-wise Attention**: 9개 label별 learnable query가 token hidden state에 attention하고, label별 weighted representation을 기존 classifier의 해당 output weight와 결합해 logit 생성
- **파라미터 변화**: `110,625,033 → 110,631,945`(+6,912, 약 0.006%)
- **공통 조건**: KLUE-RoBERTa-base, plain BCE, seed 42, `encode_mode=truncate`, sampling OFF, max length 512, learning rate `2e-5`, 3 epochs, minimum `val_loss` checkpoint

### 5.2 Full Training 결과

| 지표 | CLS baseline | Label Attention | 변화 |
|---|---:|---:|---:|
| Best epoch | 2 | 2 | 0 |
| Validation loss | 0.254921 | 0.255887 | +0.000966 |
| Macro F1 @ 0.5 | 0.600329 | 0.594089 | -0.006240 |
| Optimized Macro F1 | 0.655421 | 0.655291 | -0.000130 |
| Macro AUROC | 0.883142 | 0.882844 | -0.000298 |
| Macro AP | 0.683620 | 0.682569 | -0.001051 |

Optimized Macro F1은 사실상 동률이지만 개선은 아니며, 고정 threshold 0.5와 ranking 지표도 낮아졌다.

### 5.3 Multi-label FN 비교

동일한 3,640개 Validation label 배열과 각 run에 저장된 optimized threshold를 사용해 비교했다.

| positive label 수 | CLS FN rate | Label Attention FN rate | 변화 |
|---:|---:|---:|---:|
| 1개 | 23.15% | 24.56% | +1.41%p |
| 2개 | 39.79% | 41.57% | +1.79%p |
| 3개 | 44.29% | 45.24% | +0.95%p |
| 4개 이상 | 57.84% | 57.84% | 0 |

핵심 목표였던 2개 이상 label 영역에서 FN rate가 개선되지 않았고 2개·3개 그룹은 오히려 악화됐다. 따라서 CLS pooling 자체가 multi-label FN 증가의 주된 원인이라는 가설은 이번 실험에서 지지되지 않는다.

### 5.4 Co-occurrence와 FP/FN 패턴

일부 label에서는 다른 증상과 함께 등장할 때 recall이 올랐지만 일관된 개선은 아니었다.

| 클래스 | CLS co-occurrence recall | Label Attention | 변화 |
|---|---:|---:|---:|
| 두통 | 40.44% | 42.22% | +1.78%p |
| 호흡곤란 | 44.32% | 48.86% | +4.55%p |
| 고열 | 55.75% | 50.44% | -5.31%p |
| 전신쇠약 | 63.82% | 57.94% | -5.88%p |

전신쇠약 FP는 `445 → 368`(-77), 오심 FP는 `514 → 450`(-64)으로 감소했다. 그러나 recall도 각각 `70.35% → 65.59%`, `55.48% → 51.52%`로 낮아져, broad FP 완화는 FN 증가와 맞바꾼 결과에 가깝다.

Positive-label instance에서는 `FN → TP` 86건보다 `TP → FN` 160건이 많아 TP가 순 74건 감소했다. Negative-label instance에서는 `FP → TN` 352건, `TN → FP` 185건으로 FP가 순 167건 감소했다. Exact match는 `1,543 / 3,640 (42.39%) → 1,572 / 3,640 (43.19%)`로 0.80%p 올랐지만, 주 지표와 multi-label FN이 개선되지 않아 채택 근거로 보지 않는다.

### 5.5 해석과 결정

Global Error Analysis에서 발견한 문제를 바탕으로 가설을 세우고 Label-wise Attention으로 검증했으나, 핵심 지표와 multi-label FN이 개선되지 않았다. 두통·호흡곤란에서 label-specific recall의 약한 신호는 있었지만 고열·전신쇠약 등에서 반대 결과가 나타났고, 전체적으로는 증상을 덜 놓치기보다 양성 예측을 더 보수적으로 만드는 변화였다.

따라서 pooling 구조는 현재 주요 병목이 아니라고 판단하며 Label-wise Attention을 최종 모델로 채택하지 않는다. 이 방향은 여기서 종료하고 query 수 조정이나 attention 구조 변형 같은 세부 튜닝의 우선순위도 낮게 둔다.

---

## 6. Learning rate `1e-5` ablation

### 6.1 목적과 설정

Baseline과 Label Attention 모두 epoch 2에서 minimum validation loss를 기록하고 epoch 3에서는 train loss만 계속 감소했다. 구조를 더 추가하기 전에 fine-tuning update를 완만하게 하면 Validation ranking과 Macro F1이 개선되는지 확인했다.

Run `klue_roberta_base_plain_bce_seed42_lr1e5_val_loss`는 최고 CLS baseline과 비교해 learning rate만 `2e-5 → 1e-5`로 변경했다. KLUE-RoBERTa-base, CLS pooling, plain BCE, seed 42, `truncate`, sampling OFF, max length 512, train/Validation batch 8/16, gradient accumulation 2, 3 epochs, weight decay 0.01, warmup ratio 0.1, AMP와 minimum `val_loss` checkpoint 기준은 모두 동일하다.

### 6.2 전체 결과와 학습 흐름

| 지표 | LR `2e-5` baseline | LR `1e-5` | 변화 |
|---|---:|---:|---:|
| Best epoch | 2 | 2 | 0 |
| Best validation loss | 0.254921 | 0.256425 | +0.001504 |
| Macro F1 @ 0.5 | 0.600329 | 0.596253 | -0.004076 |
| Optimized Macro F1 | 0.655421 | 0.656180 | +0.000759 |
| Threshold optimization gain | +0.055092 | +0.059927 | +0.004835 |

| epoch | baseline train/val loss | LR `1e-5` train/val loss | baseline/LR `1e-5` F1@0.5 |
|---:|---:|---:|---:|
| 1 | 0.3164 / 0.2626 | 0.3334 / 0.2665 | 0.5832 / 0.5787 |
| 2 | 0.2486 / 0.2549 | 0.2539 / 0.2564 | 0.6003 / 0.5963 |
| 3 | 0.2279 / 0.2567 | 0.2387 / 0.2568 | 0.6003 / 0.5926 |

낮은 LR에서는 train loss 감소가 더 느렸고 best validation loss와 F1@0.5도 baseline보다 낮았다. 다만 저장된 class-wise optimized threshold를 적용한 Macro F1은 약 `+0.0008` 높았다. 같은 Validation에서 threshold를 선택한 point estimate이고 기존 seed 변화에서도 약 0.001 이상의 변동이 관찰됐으므로, LR `1e-5`가 명확히 우수하다고 판단하지 않는다.

### 6.3 클래스별 변화

| 클래스 | threshold | F1@0.5 | baseline optimized F1 | LR `1e-5` optimized F1 | 변화 |
|---|---:|---:|---:|---:|---:|
| 고열 | 0.33 | 0.6797 | 0.7023 | 0.7051 | +0.0028 |
| 구토 | 0.31 | 0.5863 | 0.6066 | 0.6100 | +0.0034 |
| 두통 | 0.24 | 0.5222 | 0.5525 | 0.5435 | -0.0090 |
| 복통 | 0.46 | 0.8190 | 0.8239 | 0.8202 | -0.0037 |
| 어지러움 | 0.41 | 0.6405 | 0.6584 | 0.6608 | +0.0024 |
| 열상 | 0.50 | 0.8855 | 0.8869 | 0.8855 | -0.0014 |
| 오심 | 0.19 | 0.0046 | 0.4030 | 0.4133 | +0.0102 |
| 전신쇠약 | 0.35 | 0.5626 | 0.5894 | 0.5947 | +0.0052 |
| 호흡곤란 | 0.27 | 0.6659 | 0.6757 | 0.6727 | -0.0030 |

오심·전신쇠약·구토·고열·어지러움은 optimized F1이 상승했지만 두통·복통·호흡곤란은 하락했다. 특히 오심의 F1@0.5는 0.0046으로 매우 낮고 threshold 0.19에서 회복돼, 전체 개선은 기본 score calibration의 개선보다 threshold 사후 조정에 의존한다. LR `1e-5`는 **weak positive signal**로 기록하되 명확한 채택 근거로 보지 않으며, `1.5e-5`, `7e-6`, `5e-6` 또는 epoch 증가와 같은 추가 LR tuning은 종료한다.

---

## 7. 주요 ablation 종합

| 실험 | Optimized Macro F1 | 해석 |
|---|---:|---|
| KoBERT | 0.642988 | 정상 tokenizer 기준 초기 backbone |
| KoELECTRA | 0.646446 | KoBERT 대비 상승 |
| KLUE CLS baseline | 0.655421 | 현재 기준 standalone |
| `pos_weight` | 약 0.6413 | 개선 없음 |
| ASL | 0.652710 | ranking은 일부 상승했으나 F1 하락 |
| BCE+ASL probability ensemble | 0.655897 | baseline 대비 작은 상승 |
| 2-seed probability ensemble | 0.656804 | 현재 저장 결과 중 최고 point estimate이나 상승 폭 작음 |
| truncation variants | 약 0.650 | 전체 개선 없음 |
| pure-nausea sampling | 0.6485 | 오심 일부 변화와 다른 클래스 하락 |
| Label-wise Attention | 0.655291 | multi-label FN 개선 없음 |
| LR `1e-5` | 0.656180 | 약 +0.0008, weak positive signal |

Loss, sampling, truncation, pooling과 learning rate를 하나씩 바꾼 실험은 대체로 `0.655~0.657` 부근의 좁은 범위에 수렴하거나 baseline보다 낮았다. 작은 hyperparameter 조정을 반복하기보다, 기존 실험과 가설이 구분되는 구조적 변화 또는 더 강한 공개 pretrained encoder를 우선 검토하는 편이 남은 시간 대비 합리적이다.

---

## 8. 다음 실험 후보 분석

### 8.1 A. Label dependency modeling

9개 baseline logit에 `9 × 9` dependency correction을 더하되 diagonal을 제외하고 zero initialization하는 방식이다. Label Attention이 “각 label이 어떤 token을 볼 것인가”를 검증했다면 이 후보는 “한 label의 예측이 다른 label의 예측에 어떤 보정을 줄 것인가”를 직접 모델링한다.

- **장점**: multi-label co-occurrence FN과 confusion-like pair에 직접 연결된다. off-diagonal weight만 사용하면 추가 파라미터는 72개이며 학습·추론 비용도 9차원 행렬 연산 수준이다. correction을 zero initialization하고 별도 loss weight 없이 BCE로 end-to-end 학습하면 단일 Full Training의 해석도 비교적 명확하다.
- **위험**: Training label correlation을 의미적 인과처럼 학습해 broad FP를 늘릴 수 있다. Validation에 우연히 강한 dependency에 맞으면 일반화되지 않을 수 있고, 전신쇠약·오심 같은 FP sink가 다시 강화될 가능성이 있다.
- **판단**: error analysis와 가장 직접적으로 연결되는 저비용 구조 실험이다. 다만 전체 표현력이 늘어나는 것은 아니므로 기대 상승 폭은 제한적일 수 있다.

### 8.2 B. Pairwise ranking auxiliary loss

Plain BCE를 유지하면서 같은 sample의 positive label score가 negative label score보다 높아지도록 pairwise ranking term을 추가하는 방식이다.

- **장점**: positive/negative score separation을 직접 최적화하며, 한 sample 안에서 누락되는 positive label을 끌어올릴 가능성이 있다. label 수가 9개라 pair 계산량은 작고 inference 경로는 baseline과 동일하다. 오심·두통처럼 score overlap이 큰 클래스에 도움이 될 가능성이 있다.
- **위험**: 클래스별 prevalence와 calibration이 다른데 모든 cross-label score를 직접 비교하면 불필요한 순서 제약이 생길 수 있다. auxiliary weight라는 새 hyperparameter가 필요해 한 번의 실패가 아이디어의 실패인지 weight 선택 실패인지 구분하기 어렵다. 강한 weight는 negative label을 과도하게 낮추거나 positive label을 일괄 상승시켜 FP/FN 균형을 흔들 수 있다.
- **판단**: 계산·제출 비용은 낮지만 단일 실험의 Yes/No 해석력이 세 후보 중 가장 낮다.

### 8.3 C. Stronger public pretrained backbone

Backbone 변경은 실제 비교에서 `KoBERT 0.642988 < KoELECTRA 0.646446 < KLUE-RoBERTa 0.655421`로 가장 명확한 차이를 만들었다. 제공 Training 데이터만 fine-tuning하고 공개 pretrained weight를 사용하는 전제에서 다음 후보를 검토할 수 있다.

- [`kakaobank/kf-deberta-base`](https://huggingface.co/kakaobank/kf-deberta-base): DeBERTa-v2 기반 12-layer/hidden 768 모델이다. 모델 카드의 KLUE benchmark는 KLUE-RoBERTa-large보다 높은 평균을 보고해 가장 강한 1차 후보지만, 범용+금융 corpus와 응급 통화 사이의 domain mismatch 및 큰 vocabulary가 위험이다. 실제 실험 전 tokenizer·모델 호환성과 save/load 동작을 짧게 검증해야 한다.
- [`beomi/KcELECTRA-base`](https://huggingface.co/beomi/KcELECTRA-base): 12-layer/hidden 768 ELECTRA로 댓글 기반 noisy Korean pretraining이 구어체·비정형 transcript에 유리할 가능성이 있다. Base 계열이라 기존 pipeline에 적용하기 비교적 단순하지만, 제작자도 일반 corpus task에서는 KoELECTRA가 더 나을 수 있다고 설명하므로 KLUE baseline을 넘을지는 불확실하다. 재현 시 revision을 고정해야 한다.
- [`klue/roberta-large`](https://huggingface.co/klue/roberta-large): 동일 KLUE 계열의 24-layer/hidden 1024 모델로 표현력 증가는 가장 명확하지만 current baseline보다 규모가 크게 증가한다. 기존 학습 조건을 그대로 유지하기 어려울 수 있어 공정한 단일 변수 비교와 제출 운용성이 떨어진다.

공개 pretrained model 자체는 규칙 전제에 부합하지만, 실제 사용 전 license와 대회 허용 범위를 다시 확인하고 모델·tokenizer를 제출 환경에 함께 저장해 offline inference가 되는지 검증해야 한다.

### 8.4 후보 비교와 우선순위

| 기준 | A. Label dependency | B. Pairwise ranking | C. Stronger backbone |
|---|---|---|---|
| 예상 성능 상승 가능성 | 중간 | 중간 | 중간~높음 |
| 구현 난이도 | 낮음 | 낮음~중간 | 낮음~중간 |
| 1회 Yes/No 판단 | 비교적 명확 | 낮음: loss weight 영향 | 비교적 명확 |
| error analysis 직접성 | 높음 | 높음 | 중간 |
| 남은 시간 대비 효율 | 높음 | 중간 | 높음 |
| competition rule 안전성 | 높음 | 높음 | 공개 weight/license 재확인 필요 |
| inference/submission 복잡도 | 거의 증가 없음 | 증가 없음 | 모델 규모에 따라 증가 |

최종 우선순위는 **1순위 C → 2순위 A → 3순위 B**다. 다음 Full Training은 `kakaobank/kf-deberta-base`를 추천한다. 기존 backbone 비교에서 확인된 가장 강한 실증 신호를 활용하면서 Large 모델보다 baseline과 가까운 규모의 후보이기 때문이다. 단, 성능 향상을 보장하지 않으며 향후 실행 시 전체 학습 전에 tokenizer·모델 호환성과 offline save/load를 먼저 확인해야 한다.

---

## 9. 결론

- LR `1e-5`는 optimized Macro F1을 `0.655421 → 0.656180`으로 `+0.000759` 높였지만 F1@0.5와 validation loss는 악화됐다.
- 클래스별 상승과 하락이 섞여 있고 기존 seed 변동보다 작은 차이이므로 weak positive signal로만 기록하며, LR 추가 tuning은 종료한다.
- 지금까지 단순 loss·sampling·truncation·pooling·LR 변경은 기준 성능을 명확히 넘지 못했다.
- 다음 우선순위는 stronger public pretrained backbone, zero-initialized label dependency residual, pairwise ranking auxiliary loss 순이다.
- 다음 Full Training 후보는 `kakaobank/kf-deberta-base`이며, 공개 pretrained weight만 초기화에 사용하고 제공 Training 데이터만 fine-tuning한다.
