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

## 5. 통합 해석과 다음 실험 우선순위

현재 증거를 종합하면 우선순위는 다음과 같다.

1. **Label-wise evidence aggregation 검토**: label마다 별도의 query로 token evidence를 모아 다증상 표본의 FN과 generic label 과활성화를 줄일 수 있는지 확인한다.
2. **오심 threshold 및 score 분리 모니터링**: threshold 최적화 효과는 유지하되, 다음 모델에서도 오심 AP·AUROC·precision/recall을 함께 비교한다.
3. **truncation과 pure-nausea sampling은 낮은 우선순위의 ablation 옵션으로 유지**: 기본값은 각각 `truncate`, sampling OFF를 유지한다.

Label-wise Attention Pooling은 global 분석에서 드러난 병목과 직접 연결되는 후보지만 성능 향상을 보장하지 않는다. encoder hidden state에 필요한 정보가 이미 없으면 pooling만으로 복구할 수 없고, 희소 label query가 과적합하거나 CLS baseline보다 최적화가 불안정해질 수 있다. 따라서 다음 실험에서는 다른 모든 조건을 고정하고 pooling 방식 하나만 바꾸는 ablation으로 검증해야 한다.

---

## 6. 결론

- 초기 오심/구토 분석은 취약 클래스와 annotation ambiguity 가능성을 찾는 데 유효했지만, 두 클래스의 직접 혼동은 global 최상위 오류가 아니다.
- Validation에서 512 토큰 초과 비율은 2.69%이고 `head_tail`/`sliding` 실험도 개선되지 않아 truncation은 현 시점의 주된 병목이 아니다.
- pure-nausea weight 1.5 sampling은 오심 F1을 소폭 높였으나 recall과 다른 클래스 성능을 낮춰 Macro F1이 `0.6554 → 0.6485`로 하락했다.
- 가장 강한 신호는 positive label 수가 많아질수록 FN rate가 `23.15% → 57.84%`로 증가하고, 여러 클래스에서 co-occurrence recall이 낮아지는 현상이다.
- 다음 구조 후보는 KLUE-RoBERTa와 학습 조건을 유지한 채 CLS pooling만 label-wise token attention pooling으로 교체하는 최소 ablation이다.
