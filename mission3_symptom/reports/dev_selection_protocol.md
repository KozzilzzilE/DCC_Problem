# Mission 3 하이퍼파라미터 재선정 절차 (Training 내부 dev, 사전 등록)

작성 2026-10-01. 이 문서는 dev 결과를 보기 전에 커밋한다. 아래 규칙은 결과를 보고 바꾸지 않는다.

## 왜 다시 정하나

2026-09-30경 받은 주최 측 답변으로 기준이 바뀌었다. 우리가 보낸 질문 Q1~Q4 에 대해 답변은 다음과 같다.

| 질문 | 답변 |
|---|---|
| Q1. Training 라벨로 계산한 pos_weight = (음성/양성)^p 를 쓰는 것 | 가능 |
| Q2. p 를 고르는 방법 | 제공된 Validation 성능이 아니라 Training 내부 dev 나 OOF 로 정한다 |
| Q3. 앙상블 가중치 w, 로지스틱 회귀 C | 같은 기준으로 Training 내부에서 정한다. Validation 은 결정된 모델의 성능 확인에만 쓴다 |
| Q4. 클래스별 p·w | Validation 으로 클래스별 값을 최적화하지 않는다. 임계값은 모든 클래스 0.5 고정 |

지금까지 제출 후보들은 p(0.5), w(0.3), C(0.15), TAPT 길이·LLRD, 체크포인트 epoch 을 Validation 을 보며 골랐다. 그래서 이 값들을 모두 Training 내부 dev 로 다시 정한다.

## 데이터

- **Training CSV:** `mission3_train.csv` (29,200건). `make_csv.py` 로 원본 JSON 에서 다시 만들 수 있다.
- **dev 분할:** `make_dev_split.py --dev-fraction 0.1 --seed 1234` 로 만든다.
  - 통화 단위 무작위 분할이다. 학습용 26,280건, dev 2,920건.
  - 원본 sha256 과 라벨 분포는 `split.json` 에 남긴다.
- **Validation 은 1~9 단계에서 읽지 않는다.**
  - `devsel.py` 는 모든 run 의 `run_config.json` 이 학습용 분할로 학습하고 dev 분할로 평가했는지 확인한다. 아니면 멈춘다.
- **임계값은 모든 단계에서 0.5 고정이다.**

## 고정값 (탐색하지 않음)

다음 값은 원래 레시피의 기본값을 그대로 쓴다.

- **모델·입력:** klue/roberta-base, 최대 512 토큰 앞부분, 발화 공백 결합
- **학습:**
  - 배치 8 × 누적 2, weight decay 0.01, warmup 0.1, AMP
  - 3 epoch 학습률 스케줄
- **TF-IDF:**
  - 특징은 char_wb 2-4 + 단어 1-2 gram, min_df 3
  - 분류기는 balanced LogisticRegression

TAPT 설정(20 epoch, lr 5e-5)과 LLRD 설정(0.8, 분류 lr 5e-5)은 "후보 레시피"로만 쓴다. 이 후보는 이전 탐색에서 나온 것이고, 채택할지는 아래 dev 규칙으로 정한다.

## 단계와 규칙

| 단계 | 내용 | 규칙 |
|---|---|---|
| 1 | dev 분할 | 위 설정 그대로 |
| 2 | p 후보 {0, 0.5, 1.0} | 원래 레시피, seed 42, 학습용 분할로 3 epoch 학습한다. epoch 마다 dev macro F1@0.5 를 잰다 (p=0 은 가중치 없는 BCE) |
| 3 | p 결정 | 후보마다 epoch 별 dev F1 중 최고값을 비교해 가장 큰 p. 같으면 먼저 적은 후보 |
| 4 | TAPT | 학습용 분할 본문만으로 MLM 20 epoch, 평가 CSV 없음 |
| 5 | 레시피 후보 | 원래 레시피와 TAPT+LLRD 0.8 을 각각 seed 42·43 으로, 정한 p 로 학습한다. 원래 레시피 seed 42 는 2 단계 run 을 재사용 |
| 6a | 레시피·epoch | 레시피마다 seed 평균 dev F1 이 가장 높은 epoch 를 찾는다(같으면 뒤 epoch). 그 값이 큰 레시피를 고른다(같으면 원래 레시피) |
| 6b | C, w | 고른 레시피의 그 epoch dev 확률을 seed 평균한다. 학습용 분할로 적합한 TF-IDF 를 C ∈ {0.05, 0.15, 0.5, 1.5}, w ∈ {0, 0.1, …, 0.6} 격자에서 섞는다. dev F1 최고인 (C, w) 를 고른다. 같으면 w 가 작은 쪽, 그다음 C 가 0.15 에 가까운 쪽. w=0 이면 TF-IDF 를 쓰지 않는다 |
| 6c | 모델 수 | seed 평균 블렌드의 dev F1 이 단일 seed 블렌드 평균보다 높으면 4개, 아니면 1개 |
| 7 | 최종 학습 | Training 전체(29,200건)로 학습한다. 정한 레시피·p 를 쓰고, 3 epoch 스케줄 중 정한 epoch 에서 저장한다(`--checkpoint-metric fixed_epoch`). seed 는 42부터. TAPT 레시피면 4 단계 TAPT 가중치에서 시작 |
| 8 | 최종 TF-IDF | Training 전체로 정한 C (w>0 일 때만) |
| 9 | 번들 | `devsel.py assemble`, fp16 |
| 10 | Validation 확인 | 번들로 `inference.py` 를 **한 번** 돌려 macro F1@0.5 를 보고한다. 이 결과로 아무것도 바꾸지 않는다 |

### 7 단계 보충

- 학습 스크립트가 평가 CSV 를 요구한다. 그래서 dev 분할을 넘기지만, 이 dev 는 학습 데이터 안에 있으므로 진행 기록일 뿐이고 아무 결정에도 쓰지 않는다.

### 10 단계 보충

- 한 번 실행하면 완료 표식이 남아서 다시 돌지 않는다.
- 결과 점수가 이전 제출 후보보다 낮게 나와도 이 번들을 쓴다. 이전 후보들은 Validation 으로 고른 값이라 이번 기준에서 쓸 수 없다.

## 한계 (미리 적어 둠)

- **후보 집합:** p·C·w 격자와 두 레시피 후보는 이전 Validation 탐색을 알고 만든 것이다. 다만 값의 선택은 전부 dev 규칙이 한다.
- **dev 크기:** 2,920건이라 macro F1 차이 0.005 안팎은 잡음 범위다. 규칙은 동점 처리까지 미리 정해 두었다.
- **6c 모델 수 판단:** seed 2개로 판단한다. 4개를 실제로 dev 에서 재지는 않는다.

## 실행

```bash
bash run_dev_selection.sh <mission3_train.csv> <validation label 폴더>
```

결정 기록은 `runs/devsel/decisions/`(power.json, final.json, validation_once.json)에 남는다. 결과는 `reports/dev_selection_eval.md` 로 정리한다.
