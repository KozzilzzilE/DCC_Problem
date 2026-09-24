# Mission 3 — 환자 증상 다중 라벨 분류 (평가 지표 및 임계값 최적화)

담당: 권오현 (데이터 전처리, 평가 지표 구현 및 임계값 최적화)  
담당: 김완수 (텍스트 데이터 정제 및 KoBERT 모델링)

119 신고 전화 대화 전사(Transcript) 텍스트로부터 환자의 주요 증상(9개 타겟 증상)을 다중 라벨(Multi-label)로 분류하고, **Macro F1-score**를 극대화하기 위한 클래스별 최적 임계값(Threshold)을 탐색.

---

## 📋 대회 규정 대상 9개 증상

| 번호 | 타겟 증상 | 비고 |
|:---:|:---:|:---|
| 0 | **고열** | 발현 빈도 높음 (주요 클래스) |
| 1 | **구토** | 주요 소화기 응급 증상 |
| 2 | **두통** | 다빈도 호소 증상 |
| 3 | **복통** | 다빈도 호소 증상 |
| 4 | **어지러움** | 중등도 빈도 증상 |
| 5 | **열상** | 발현 빈도 매우 낮음 (외상류) |
| 6 | **오심** | 구토와 동반 빈도 높음 |
| 7 | **전신쇠약** | 노인/만성 질환 다빈도 호소 |
| 8 | **호흡곤란** | 초응급 증상 |

> **대회 규정**: 9개 평가 대상 외 증상(예: `골절`, `찰과상`, `화상` 등)은 학습/평가 라벨 벡터에서 제거하며, 모델 입력으로는 대화 본문 텍스트(`utterances[].text`)만 허용. (화자, 시간, 인적사항 메타데이터 원천 배제)

---

## 🎯 접근 전략 — 클래스별 임계값(Class-wise Threshold) 최적화

1. **지표 특성 (Macro F1)**:
   $$\text{Macro F1} = \frac{1}{9} \sum_{c=1}^{9} \text{F1}_c$$
   - 다빈도 클래스(두통, 복통)보다 **희귀 클래스(열상, 고열 등)의 F1 개선이 전체 점수에 동일한 가중치(1/9)**를 가짐.
2. **기본 0.5 임계값의 한계**:
   - 불균형 데이터에서는 모델이 양성 예측을 소극적으로 하여 Recall이 급락하고 F1이 0에 수렴하는 문제 발생.
3. **해결책**:
   - 검증셋(Validation)에서 각 증상별로 0.05~0.95 구간을 $0.01$ 단위로 그리드 탐색하여 **증상별 최적 임계값 벡터**를 산출하고, 이를 `best_thresholds.json`으로 저장하여 최종 `inference.py` 추론에 적용.

---

## 📁 구조

```text
mission3_symptom/
├── m3/                          # 핵심 기능 모듈 패키지
│   ├── config.py                # 9개 타겟 증상 상수, 경로 설정 및 발화 경계 모드
│   ├── labels.py                # 대회 규정 강제 라벨 파서 및 데이터 로더
│   ├── metrics.py               # 이진 F1 및 규정 준수 Macro F1 계산 함수
│   ├── threshold.py             # 9개 증상별 최적 임계값 그리드 탐색기
│   ├── report.py                # 성과 리포트(MD) 및 최적 임계값(JSON) 생성기
│   ├── dataset.py               # CSV 검증, text-only Dataset 및 DataLoader
│   ├── kobert_tokenizer.py       # KoBERT SentencePiece tokenizer 및 BERT 입력 형식
│   ├── model.py                 # Hugging Face backbone 기반 9-label 모델 생성 및 저장
│   ├── training.py              # 학습, 검증 및 실험 산출물 저장
│   ├── tfidf_member.py          # Training 전용 TF-IDF+LR 보조 멤버 (제출 블렌드용)
│   ├── infer.py                 # 제출 추론 (단일 run 또는 ensemble.json 번들, 임계값 0.5 고정)
│   └── __init__.py              # m3 통합 인터페이스 export
├── reports/
│   ├── comparison.md            # 기본 0.5 vs 최적 임계값 전/후 F1 성과 리포트
├── extract_labels.ipynb         # ★ 1단계: 원본 zip(001~013)에서 JSON 라벨 32,840건 고속 추출 노트북
├── data_preprocessing.ipynb     # ★ 2단계: 규정 준수 텍스트 정제 & 9개 타겟 증상 CSV 생성 전처리 노트북
├── model_train.ipynb            # ★ 3단계: 실제 baseline 결과 검증 및 시각화 노트북
├── train.py                     # 공통 backbone 학습 CLI
├── train_tfidf_member.py        # Training 전용 TF-IDF+LR 멤버 학습 CLI
├── inference.py                 # 미션 폴더 단독 제출 진입점
├── tests/                       # 데이터 및 label shape 단위 테스트
└── README.md                    # 현재 문서
```

---

## 🧩 `m3` 모듈별 상세 스펙 및 역할

| 모듈 파일 | 핵심 역할 | 주요 함수 / 클래스 | 세부 설명 |
|:---|:---|:---|:---|
| **`config.py`** | 글로벌 설정 및 표준 상수 | `TARGET_SYMPTOMS`<br>`NUM_CLASSES = 9`<br>`SYMPTOM_TO_IDX`<br>`UTTERANCE_SEP_MODES`<br>`resolve_utterance_sep()` | 9개 공식 증상 목록, 증상별 정수 인덱스 매핑, 리포트 저장 경로 및 발화 경계 표현 모드 정의. |
| **`labels.py`** | 대회 규칙 강제 전처리 파이프라인 | `load_transcripts_dataframe()`<br>`load_transcripts_dir()`<br>`read_transcript()` | 수만 건의 JSON에서 화자/시간 메타데이터를 원천 배제하고 순수 대화 본문만 추출. `sep_mode` 로 발화 경계 표시 방식을 선택(기본 `space`). 비타겟 증상 필터링 후 9차원 원-핫(이진) 벡터로 변환. Windows/macOS(NFC 정규화)/Colab 다중 인코딩 Fallback 지원. |
| **`metrics.py`** | 공식 평가지표 계산기 | `eval_macro_f1()`<br>`calculate_binary_f1()` | 증상별 정밀도(Precision)와 재현율(Recall) 기반 이진 F1 계산(ZeroDivision 안전 처리) 및 산술 평균 기반 대회 공식 Macro F1 산출. |
| **`threshold.py`** | 임계값 최적화 엔진 | `find_best_thresholds()`<br>`apply_thresholds()`<br>`get_threshold_curves()` | 0.05~0.95 구간(0.01 간격) 그리드 탐색을 통해 9개 증상별 F1을 극대화하는 황금 임계값 벡터 산출 및 시각화용 반응 곡선 데이터 생성. |
| **`report.py`** | 성과 문서 & 추론 설정 자동화 | `generate_comparison_markdown()`<br>`save_thresholds_json()` | 기준선(0.5) 대비 F1 상승폭을 마크다운 리포트(`comparison.md`)로 자동 작성하고, 최종 추론용 `best_thresholds.json` 파일 저장. |
| **`kobert_tokenizer.py`** | KoBERT tokenizer | `KoBertTokenizer` | `spiece.model`을 사용해 한국어를 SentencePiece subword로 변환하고 `[CLS] text [SEP]` 입력 형식과 local 저장/재로드를 보장. |
| **`__init__.py`** | 패키지 인터페이스 허브 | `m3.*` 공개 API 노출 | 모델링 담당자가 복잡한 내부 구조를 몰라도 `from m3 import ...` 한 줄로 전처리 및 평가 함수를 즉시 호출 가능하도록 구성. |

---

## 🚀 빠른 실행 가이드

### KLUE-RoBERTa plain BCE baseline 학습

학습 코드는 실행 환경의 경로를 가정하지 않으며, CSV와 출력 경로를 명령행 인자로 받는다. Google Drive 마운트나 파일 다운로드는 학습 모듈의 책임이 아니다.

정상 동작만 확인하는 smoke test:

```bash
python mission3_symptom/train.py \
  --train-csv <mission3_train.csv> \
  --val-csv <mission3_val.csv> \
  --output-dir mission3_symptom/runs/klue_roberta_base_bce_seed42 \
  --smoke-test \
  --amp
```

smoke test는 최대 Train 64건, Validation 32건, optimizer step 2회로 제한되며 결과는 별도 `<output-dir-name>_smoke` 디렉터리에 저장된다. 이 점수는 성능 비교에 사용하지 않는다.

전체 학습:

```bash
python mission3_symptom/train.py \
  --train-csv <mission3_train.csv> \
  --val-csv <mission3_val.csv> \
  --output-dir mission3_symptom/runs/klue_roberta_base_bce_seed42 \
  --model-name-or-path klue/roberta-base \
  --seed 42 \
  --max-length 512 \
  --train-batch-size 8 \
  --val-batch-size 16 \
  --gradient-accumulation-steps 2 \
  --learning-rate 2e-5 \
  --epochs 3 \
  --loss-type bce \
  --encode-mode truncate \
  --amp
```

`--device auto`가 기본값이며 사용 가능한 실행 환경을 자동으로 선택한다. `--amp`는 지원되는 환경에서만 활성화된다.

정식 run은 `best_model/`, `run_config.json`, `history.json`, `baseline_metrics.json`, `val_logits.npy`, `val_probs.npy`, `val_labels.npy`를 생성한다. 평가는 기존 `m3.metrics` 및 `m3.threshold.apply_thresholds`를 사용하여 threshold 0.5를 기준으로 수행한다.

> `reports/best_thresholds.json`과 `reports/comparison.md`는 실제 모델 학습 이전에 synthetic validation 데이터로 생성된 기존 결과이므로 실제 Full Training 성능으로 해석하지 않는다. 실제 threshold 결과는 각 정식 run의 `threshold_metrics.json`과 `optimized_thresholds.json`으로 별도 관리하며 기존 reports 파일을 수정하지 않는다.

### 주피터 노트북 실행 (`model_train.ipynb`)
VS Code 또는 Jupyter 환경에서 `mission3_symptom/model_train.ipynb`를 열고 순서대로 셀을 실행하면:
1. 정식 run의 설정과 저장 파일 검증
2. 실제 Validation logits, probabilities, labels 로드
3. threshold 0.5 Macro F1 및 클래스별 F1 재계산
4. epoch별 loss와 Validation Macro F1 시각화
5. 기존 threshold 탐색 함수와의 연결 상태 확인

필수 산출물이 없으면 synthetic 데이터로 대체하지 않고 오류를 발생시킨다. Threshold 탐색은 기본적으로 비활성화되어 있으며 기존 reports 파일을 수정하지 않는다.

### `model_train.ipynb` 실험 방법

노트북을 새로 열거나 kernel을 다시 시작했다면 0번 환경 준비부터 실행한다. 1번 학습 설정 셀에서 사용하는 변수는 다음과 같다.

| 변수 | 설명 |
|---|---|
| `TRAIN_CSV` | Training CSV 경로 |
| `VAL_CSV` | Validation CSV 경로 |
| `RUN_NAME` | 정식 실험과 산출물을 구분하는 이름 |
| `OUTPUT_DIR` | `RUN_NAME`에 따라 결정되는 결과 저장 위치 |
| `MODEL_NAME` | pretrained model 이름 또는 local model 경로 |
| `SEED` | 데이터 순서와 모델 초기화 재현을 위한 random seed |
| `MAX_LENGTH` | tokenizer가 모델에 전달하는 최대 token 길이 |
| `TRAIN_BATCH_SIZE` | 한 번의 training forward/backward에 사용하는 sample 수 |
| `VAL_BATCH_SIZE` | Validation inference batch의 sample 수 |
| `GRAD_ACCUM_STEPS` | optimizer update 전에 gradient를 누적하는 step 수 |
| `LEARNING_RATE` | AdamW learning rate |
| `WEIGHT_DECAY` | 과적합 완화를 위한 AdamW weight decay |
| `WARMUP_RATIO` | 전체 optimizer step 중 learning-rate warmup 비율 |
| `EPOCHS` | 전체 Training 데이터를 반복 학습하는 횟수 |
| `USE_AMP` | 지원되는 환경에서 mixed precision을 사용할지 여부 |
| `USE_POS_WEIGHT` | Training label에서 계산한 클래스별 `negative / positive` 가중치를 BCE에 적용할지 여부. 기본값은 `False`이며 class imbalance ablation에서만 활성화 |
| `ENCODE_MODE` | 기본값 `truncate`. Head-tail truncation ablation에서만 `head_tail` 사용 |
| `USE_PURE_NAUSEA_SAMPLING` | pure-nausea group-aware sampling 활성화 여부. CLI/TrainingConfig 기본값은 `False` |
| `PURE_NAUSEA_WEIGHT` | C 그룹(오심=1, 구토=0)에만 적용하는 sampling weight. 첫 실험값은 `1.5` |

Pure-nausea sampling은 `--use-pure-nausea-sampling --pure-nausea-weight 1.5`로 활성화한다. 이 옵션은 Training loader에만 `WeightedRandomSampler`를 적용하며 Validation loader에는 영향을 주지 않는다.

최초 실행에서는 `TRAIN_CSV`와 `VAL_CSV`가 실제 CSV를 가리키는지 확인한다. 기본 경로와 다른 위치에 데이터가 있다면 이 두 값만 실행 환경에 맞게 변경한다.

새로운 정식 실험에서는 모델, learning rate, epoch 등 비교할 조건을 수정하고 기존 결과와 섞이지 않도록 `RUN_NAME`도 새 값으로 변경한다. `OUTPUT_DIR`은 `RUN_NAME`으로부터 자동 구성되며, 기존 정식 run이 있는 디렉터리는 덮어쓰지 않는다.

권장 실행 흐름:

- 같은 환경에서 단순 hyperparameter만 변경: 0번을 이미 실행한 상태에서 `1 → 3 → 4 → 5 → 6(선택) → 7 → 8`
- 새로운 환경, PC, Colab에서 실행하거나 모델/tokenizer를 변경한 경우: `0 → 1 → 2 Smoke Test → 3 Full Training → 4 → 5 → 6(선택) → 7 → 8`
- dataset/model/training pipeline 코드를 변경한 경우에도 2번 Smoke Test를 먼저 실행

2번 Smoke Test는 데이터 로딩부터 결과 저장까지 pipeline correctness를 확인하는 용도다. 일부 sample과 매우 적은 step만 사용하므로 Smoke loss, Macro F1, 클래스별 F1은 정식 성능으로 해석하거나 다른 실험과 비교하지 않는다.

3번 Full Training이 전체 Train/Validation 데이터를 사용하는 정식 학습이다. 4번은 해당 run의 저장 결과를 불러오고, 5번은 공통 기준인 threshold 0.5의 실제 Validation 성능을 재검증한다. 6번은 epoch별 loss와 Macro F1을 시각화한다.

7번은 해당 checkpoint가 생성한 `val_probs`에서 class-wise optimized threshold를 탐색한다. 기본값은 `RUN_THRESHOLD_SEARCH = False`이므로 팀 확인 후 `True`로 변경해 실행한다. 8번은 탐색한 threshold를 적용하여 threshold 0.5 대비 Macro F1, 클래스별 F1과 개선량을 비교한다. 두 단계 모두 기존 reports 파일을 자동으로 저장하거나 수정하지 않는다.

Threshold는 학습 hyperparameter가 아니라 학습 완료 후 probability에 적용하는 post-processing parameter이므로 threshold 탐색을 위해 모델을 다시 학습할 필요는 없다. 다만 모델이나 학습 조건이 바뀌면 probability 분포도 달라질 수 있으므로, 다른 checkpoint에서 얻은 threshold를 그대로 재사용하지 않고 각 정식 run의 `val_probs`를 기준으로 다시 계산하는 것을 원칙으로 한다.

### 발화 경계(턴 구분) 전처리

대회 Q&A 답변(2026-09-20)으로 `speaker` 값을 사용하지 않고 발화 경계만 입력에 남기는 전처리가 허용됐다. 기존 전처리는 `utterances[].text`를 공백으로 이어 붙여 턴 경계를 지웠으므로, 입력 표현은 지금까지 한 번도 변경되지 않은 축이다.

`m3.config.UTTERANCE_SEP_MODES`가 결합 방식을 정의하고 `m3.labels`의 파싱 함수들이 `sep_mode` 인자로 이를 받는다. 기본값은 `space`이며 기존 CSV와 기존 실험 결과의 재현성은 그대로 유지된다.

| 모드 | 결합 결과 | 비고 |
|---|---|---|
| `space` | `발화1 발화2` | 기존 baseline. 경계 정보 없음 |
| `sep` | `발화1 [SEP] 발화2` | tokenizer 기본 어휘의 경계 토큰 재사용. vocab 변경 불필요 |
| `turn` | `발화1 [TURN] 발화2` | 전용 special token. tokenizer 등록과 embedding resize 가 선행돼야 함 |

개행(`\n`)은 모드로 제공하지 않는다. KLUE-RoBERTa 등 BERT 계열 tokenizer 는 basic tokenization 단계에서 개행을 공백과 동일하게 처리하므로 token 열이 바뀌지 않아 `space` 와 결과가 같다.

구분자는 발화 사이에만 들어가며 전체 텍스트의 앞뒤를 감싸지 않는다. 빈 발화는 결합 전에 제거하므로 구분자가 연속으로 붙어 가짜 턴 경계가 생기지 않는다. 메타데이터 차단은 모드와 무관하게 유지되며 `tests/test_utterance_sep.py`가 세 모드 모두에서 `speaker`, `startAt`, `endAt`, 인적사항이 본문에 섞이지 않는지 검증한다.

CSV 재생성은 `data_preprocessing.ipynb`의 `UTTERANCE_SEP_MODE`만 바꿔 실행한다. 이 노트북은 전처리 로직을 복제하지 않고 `m3.labels`를 그대로 호출하므로 학습 CSV와 추론 경로가 어긋나지 않는다. 출력은 `mission3_train_<mode>.csv`, `mission3_val_<mode>.csv`와 전처리 이력 `mission3_preprocess_<mode>.json`이다.

재학습 전에 노트북 4번 셀로 512 token 초과 비율 변화를 먼저 측정한다. 구분자만큼 입력이 길어지므로 이 값을 재지 않으면 경계 정보의 효과와 절단 증가의 부작용이 섞여 결과를 해석할 수 없다. baseline(`space`)의 Validation 초과 비율은 KLUE-RoBERTa 기준 `98 / 3,640 = 2.69%`다.

### 임계값 0.5 고정 기준 권장 레시피 (2026-09-24)

제출 지표는 **macro F1@0.5** 다. 아래 "현재 정상 Full Training 실험 결과" 절의 `Optimized Macro F1` 은 클래스별 임계값을 Validation 에서 고른 연구 기록이라 제출 성능이 아니다. 전체 수치와 근거는 `reports/calibration_eval.md` 에 있다.

| 설정 (KLUE-RoBERTa-base, 로컬 Validation 3,640건) | macro F1@0.5 | 오심 F1@0.5 |
|---|---:|---:|
| plain BCE (기존 기준선, val_loss 기준 체크포인트) | 0.5967 | 0.032 |
| `--use-pos-weight` (negative/positive, val_macro_f1 기준) | 0.6189 | 0.363 |
| **`--use-pos-weight --pos-weight-power 0.5`** (val_macro_f1 기준, val_loss 기준과 같은 epoch) | **0.6496** | 0.387 |
| 위 + Training 전용 TF-IDF 블렌드 (w=0.3) | 0.6536 | 0.401 |
| power 0.5 시드 42~45 평균 + TF-IDF 블렌드 | 0.6546 | 0.395 |

체크포인트 선택 기준이 행마다 다르다. plain BCE 를 val_macro_f1 기준으로 맞추면 epoch 3 의 0.6021 이고, 그래도 power 0.5 가 +0.0475 높다. TF-IDF 의 C 와 w 는 Validation 을 보며 고른 하이퍼파라미터라 이 두 행은 약간 낙관적이다 (자세한 경위는 보고서 6절).

- 기존 `--use-pos-weight` 는 Training 라벨의 negative/positive 를 그대로 써서 모든 클래스를 과보정했다. `--pos-weight-power 0.5` 로 제곱근을 쓰면 9개 클래스가 모두 오르고, 클래스별 임계값을 따로 골라도 더 얻을 것이 없다. 시드 42~45 에서 0.6453~0.6496 으로 재현된다.
- 발화 경계(`sep`) 입력은 같은 레시피에서 −0.0033 (95% CI −0.008~+0.002) 로 이득이 없어 공백 결합을 유지한다.
- 가중치는 Training 라벨 개수로만 계산하고 power 는 9개 클래스 공통 스칼라 하나다. 판정 임계값은 0.5 그대로다.

1. 트랜스포머 학습:

```bash
python mission3_symptom/train.py \
  --train-csv <mission3_train.csv> --val-csv <mission3_val.csv> \
  --output-dir mission3_symptom/runs/klue_posw0.5_seed42 \
  --model-name-or-path klue/roberta-base --seed 42 --max-length 512 \
  --train-batch-size 8 --val-batch-size 16 --gradient-accumulation-steps 2 \
  --learning-rate 2e-5 --epochs 3 --loss-type bce --encode-mode truncate --amp \
  --use-pos-weight --pos-weight-power 0.5 --checkpoint-metric val_macro_f1
```

2. (선택) TF-IDF 보조 멤버 학습 — **Training CSV 만** 넣는다. Validation 을 넣으면 Validation 을 학습에 쓴 것이 된다.

```bash
python mission3_symptom/train_tfidf_member.py \
  --train-csv <mission3_train.csv> \
  --output mission3_symptom/runs/tfidf_lr_c0.15/tfidf_lr.joblib
```

3. (선택) 번들 — 폴더 하나에 `ensemble.json` 과 그것이 쓰는 파일을 **모두** 넣고 `--ckpt_path` 로 그 폴더를 준다. 제출되는 것은 이 폴더 하나이므로 멤버 경로는 `./` 로 시작하는 폴더 안 경로로 적는다 (폴더 밖을 가리키면 추론 시 경고가 나온다). 멤버는 각 run 의 `best_model` 폴더만 복사해도 된다 (안의 `inference_config.json` 이 경계 모드·인코딩·max_length 를 복원한다). 트랜스포머 멤버는 균등 평균, TF-IDF 는 9개 클래스 공통 가중치 하나로 섞고 0.5 로 판정한다. 클래스별 가중치나 `threshold` 키는 거부된다.

```text
runs/submit_bundle/
├── ensemble.json
├── seed42/          # runs/klue_posw0.5_seed42/best_model 복사본
├── seed43/
└── tfidf/tfidf_lr.joblib
```

```json
{
  "members": ["./seed42", "./seed43"],
  "tfidf_member": "./tfidf/tfidf_lr.joblib",
  "tfidf_weight": 0.3
}
```

4. 제출 추론 — run 디렉터리를 주면 단일 모델, 번들 디렉터리를 주면 앙상블·블렌드다.

```bash
# 저장소 루트에서 (주최 측 실행 형태)
python inference.py --audio_dir <wav 폴더> --label_dir <json 폴더> --ckpt_path mission3_symptom/runs/<run 또는 번들> --output ./outputs/mission3.csv
```

```bash
# 또는 mission3_symptom/ 폴더 안에서 단독 실행
python inference.py --label_dir <json 폴더> --ckpt_path runs/<run 또는 번들> --output ./outputs/mission3.csv
```

로컬 Validation 추론 시간은 단일 모델 27.5초, 단일+TF-IDF 42초, 4-seed+TF-IDF 101초였다. TF-IDF 멤버는 scikit-learn 버전이 다르면 경고를 낸다 (학습 1.9.0).

### 현재 정상 Full Training 실험 결과

초기 AutoTokenizer 기반 KoBERT run은 encoder와 맞지 않는 tokenizer가 선택된 상태였으므로 정상 성능 비교에서 제외한다. 이름이 `_smoke`로 끝나는 run과 `reports/`의 synthetic 결과도 아래 Full Training benchmark에 포함하지 않는다.

#### KoBERT loss ablation

정상 SentencePiece `KoBertTokenizer`를 적용한 두 KoBERT run의 결과는 다음과 같다.

| 실험 | Macro F1 @ 0.5 | Optimized Macro F1 | Cross-fitted optimized Macro F1 |
|---|---:|---:|---:|
| Correct tokenizer + plain BCE | 0.5737 | **0.6430** | **약 0.6360** |
| Correct tokenizer + Training-derived pos_weight | **0.6035** | 0.6413 | 약 0.6319 |

Pos_weight는 threshold 0.5에서 recall과 Macro F1을 높였지만 ranking/AP와 threshold 최적화 후 Macro F1은 개선하지 못했다. 따라서 현재 대표 configuration은 correct `KoBertTokenizer`와 plain BCE이며, pos_weight 옵션은 실제 ablation 재현을 위해 유지한다.

기존 cross-fitted 수치는 참고값이다. 현재 repository에는 fold 정의, split 방식, seed, threshold protocol이 완전히 고정된 재현 코드와 산출물이 없으므로 다른 backbone의 수치를 임의로 추가하지 않는다. Protocol을 고정한 뒤 모든 backbone의 저장된 Validation prediction에 동일 방식으로 재계산할 예정이다.

#### Backbone benchmark

네 backbone은 모델과 그에 맞는 tokenizer만 변경했다. 동일 Training 29,200건/Validation 3,640건, plain BCE, seed 42, 3 epochs, learning rate `2e-5`, max length 512, physical batch 8, gradient accumulation 2(effective batch 16), weight decay 0.01, warmup ratio 0.1, AMP, minimum `val_loss` checkpoint 선택과 동일한 class-wise threshold 탐색을 사용했다. 공개 pretrained checkpoint는 초기화에만 사용했으며, 제공 Training 데이터만 supervised fine-tuning에 사용하고 Validation은 평가와 threshold 선택에만 사용했다.

| Model | Best Epoch | F1 @ 0.5 | Optimized Macro F1 |
|---|---:|---:|---:|
| KoBERT (`skt/kobert-base-v1`) | 3 | 0.573737 | 0.642988 |
| KoELECTRA (`monologg/koelectra-base-v3-discriminator`) | 3 | 0.581073 | 0.646446 |
| KLUE-RoBERTa (`klue/roberta-base`) | 2 | 0.600329 | 0.655421 |
| KF-DeBERTa (`kakaobank/kf-deberta-base`) | 2 | **0.606132** | **0.655760** |

Backbone 교체는 초기 세 모델에서 `KoBERT → KoELECTRA → KLUE-RoBERTa` 순으로 비교적 분명한 상승을 만들었다. KF-DeBERTa는 단일 Validation run에서 가장 높은 point estimate를 기록했지만, optimized Macro F1은 KLUE-RoBERTa보다 `0.000339` 높은 수준에 그쳤다. 같은 Validation에서 threshold를 선택하고 평가한 결과이므로 이 작은 차이를 명확한 성능 우위로 해석하지 않는다.

#### KF-DeBERTa Full Training 및 threshold 최적화

Run `kf_deberta_base_plain_bce_seed42_val_loss`는 `kakaobank/kf-deberta-base` revision `363b171d71443b0874b0bf9cea053eb5b1650633`을 사용했다. Standard CLS classification path에서 Label Attention, `pos_weight`, ASL과 weighted sampling을 모두 끄고 plain BCE로 학습했다. 나머지 조건은 backbone benchmark와 동일하며 입력은 `truncate` 방식으로 최대 512 token까지만 사용했다.

| Epoch | Train loss | Validation loss | Validation Macro F1 @ 0.5 | 선택 |
|---:|---:|---:|---:|---|
| 1 | 0.302746 | 0.258396 | 0.576790 | - |
| 2 | 0.243850 | **0.254455** | **0.606132** | best checkpoint |
| 3 | **0.221921** | 0.258950 | 0.601708 | - |

Epoch 3에서 train loss는 계속 감소했지만 Validation loss가 다시 증가했고 Macro F1@0.5도 epoch 2를 넘지 못했다. 따라서 minimum Validation loss 기준으로 저장된 epoch 2 checkpoint가 이 run의 적절한 선택이며, epoch 2 이후에는 가벼운 과적합 조짐이 관찰된 것으로 해석한다.

Best checkpoint의 class-wise threshold 최적화 결과는 다음과 같다. 이는 모델 표현을 다시 학습한 결과가 아니라, 같은 Validation probability에 적용하는 **decision boundary 후처리**다.

| 증상 | F1 @ 0.5 | Optimized threshold | Optimized F1 | Delta |
|---|---:|---:|---:|---:|
| 고열 | 0.6691 | 0.31 | 0.6931 | +0.0240 |
| 구토 | 0.5831 | 0.33 | 0.5992 | +0.0161 |
| 두통 | 0.5009 | 0.23 | 0.5663 | +0.0654 |
| 복통 | 0.8205 | 0.35 | 0.8239 | +0.0034 |
| 어지러움 | 0.6480 | 0.34 | 0.6671 | +0.0191 |
| 열상 | 0.8927 | 0.47 | 0.8934 | +0.0007 |
| 오심 | 0.1452 | 0.19 | 0.3964 | +0.2513 |
| 전신쇠약 | 0.5498 | 0.22 | 0.5923 | +0.0425 |
| 호흡곤란 | 0.6460 | 0.27 | 0.6702 | +0.0241 |
| **Macro F1** | **0.606132** | class-wise | **0.655760** | **+0.049628** |

오심은 threshold `0.19`에서 F1이 `0.1452 → 0.3964`로 가장 크게 변했다. 이는 오심의 낮은 score scale에 맞춘 후처리가 필요하다는 근거이며, backbone 자체가 오심 표현 문제를 해결했다는 의미는 아니다.

KF-DeBERTa의 `BertTokenizer`는 vocabulary 130,000개를 사용했고 Validation의 512 token 초과 비율은 `55 / 3,640 = 1.51%`였다. KLUE-RoBERTa의 `98 / 3,640 = 2.69%`보다 truncation 노출은 낮았지만, 이 tokenizer상의 이점이 Macro F1의 명확한 상승으로 이어지지는 않았다.

저장된 단일 로컬 run의 Validation wall-clock은 KF-DeBERTa가 epoch 1 `1132.206초`, epoch 2 `770.013초`였고, KLUE-RoBERTa best-checkpoint 평가는 `19.260초`였다. 실행별 초기화와 환경 영향을 받는 측정이므로 일반적인 속도 배수로 단정하지 않지만, 이 프로젝트의 동일 로컬 실험 조건에서는 KF-DeBERTa의 계산 비용이 크게 증가했다.

#### 현재 기준 backbone 선정

KF-DeBERTa의 optimized Macro F1 `0.6557598681`은 KLUE-RoBERTa seed 42의 `0.6554208986`보다 약 `0.000339` 높다. 그러나 KLUE 자체도 seed 42 `0.6554209`, seed 43 `0.6539106`으로 약 `0.0015` 변동했기 때문에, KF와 KLUE의 차이는 단일 run의 변동 범위보다 작다.

따라서 KF-DeBERTa는 유효한 비교 후보이자 약간 높은 단일 Validation 점수를 기록한 모델로 남기되, **성능 차이의 불확실성, 계산 비용, 기존 실험의 재현성과 운용 효율을 함께 고려해 KLUE-RoBERTa를 현재 기준 backbone(selected baseline backbone)으로 유지한다.** 더 강한 backbone으로 교체하는 것만으로 현재 `0.655~0.657` 부근의 Validation Macro F1 병목이 크게 해소되지는 않았으며, 이는 향후 label dependency와 같은 구조적 가설을 검토할 근거가 된다. 아직 실행하지 않은 후속 실험의 성능은 가정하지 않는다.

#### KLUE-RoBERTa baseline threshold 결과

KLUE의 best checkpoint는 epoch 2(`val_loss=0.2549`)였다. Train loss는 epoch 3까지 감소했지만 val loss는 epoch 2에서 최소인 뒤 0.2567로 소폭 상승했고 F1@0.5는 `0.6003 → 0.6003`으로 거의 동일해, epoch 2 이후 Validation 개선이 제한적이었다.

| KLUE class | F1 @ 0.5 | Optimized threshold | Optimized F1 |
|---|---:|---:|---:|
| 고열 | 0.6813 | 0.27 | 0.7023 |
| 구토 | 0.5860 | 0.38 | 0.6066 |
| 두통 | 0.5147 | 0.36 | 0.5525 |
| 복통 | 0.8201 | 0.44 | 0.8239 |
| 어지러움 | 0.6429 | 0.37 | 0.6584 |
| 열상 | 0.8859 | 0.51 | 0.8869 |
| 오심 | 0.0529 | 0.19 | 0.4030 |
| 전신쇠약 | 0.5570 | 0.23 | 0.5894 |
| 호흡곤란 | 0.6622 | 0.36 | 0.6757 |

#### KLUE seed 재현성 및 2-seed probability ensemble

KLUE-RoBERTa plain BCE baseline의 1차 seed 재현성을 확인하기 위해 seed 43을 추가로 학습했다. seed 42와 seed 43은 `RUN_NAME`과 seed만 다르고 데이터 split, 모델, loss, optimizer 및 학습 설정, `val_loss` checkpoint 선택 기준과 threshold 탐색 protocol은 동일하다. 두 run 모두 epoch 2가 best checkpoint로 선택됐다.

| 평가 대상 | F1 @ 0.5 | Optimized Macro F1 | Macro AUROC | Macro AP | Best epoch |
|---|---:|---:|---:|---:|---:|
| KLUE seed 42 | 0.600329 | **0.655421** | **0.883142** | **0.683620** | 2 |
| KLUE seed 43 | **0.607258** | 0.653911 | 0.882340 | 0.679988 | 2 |
| seed 42·43 probability ensemble | 0.603682 | **0.656804** | **0.885324** | **0.685990** | - |

Ensemble은 동일한 Validation 3,640건에서 두 모델의 class-wise probability를 단순 평균한 뒤 standalone과 동일한 방식으로 threshold 0.5 지표와 클래스별 optimized threshold를 계산했다. 평균 전에 sample 순서, call ID와 label alignment가 동일한지 확인했다.

| Ensemble delta (vs seed 42) | 변화량 |
|---|---:|
| Optimized Macro F1 | +0.001383 |
| Macro AUROC | +0.002182 |
| Macro AP | +0.002370 |

seed 42와 seed 43의 standalone optimized Macro F1 차이는 약 0.0015로 작았다. 두 seed만 비교한 제한은 있지만, KLUE plain BCE baseline이 두 run에서 대체로 비슷한 수준으로 재현된 1차 결과로 해석한다. 두 모델 probability의 Pearson correlation은 0.975552로 상당히 높아 prediction diversity는 제한적이었다.

Ensemble은 현재 Validation point estimate에서 가장 높은 optimized Macro F1, Macro AUROC와 Macro AP를 기록했지만 seed 42 대비 개선 폭은 작다. 또한 모든 취약 클래스가 함께 개선된 것은 아니다. optimized F1 기준 구토, 어지러움과 전신쇠약은 소폭 개선됐지만 두통은 `0.5525 → 0.5423`, 오심은 `0.4030 → 0.3988`로 seed 42보다 낮았다. 같은 Validation에서 threshold를 선택하고 평가한 결과이므로 ensemble의 우위를 통계적이거나 확정적인 결론으로 해석하지 않는다.

추가 seed가 제공할 정보 대비 seed 44 Full Training의 우선순위는 낮게 두고, 다음 ablation으로 backbone과 나머지 조건을 유지한 KLUE + Asymmetric Loss(ASL)를 확인했다.

#### KLUE loss ablation: plain BCE vs ASL

seed 42 KLUE-RoBERTa에서 loss만 plain BCE에서 ASL(`gamma_neg=4`, `gamma_pos=1`, `clip=0.05`)로 변경하고 나머지 학습 및 평가 조건은 동일하게 유지했다.

| Loss | F1 @ 0.5 | Optimized Macro F1 | Macro AUROC | Macro AP | Best epoch |
|---|---:|---:|---:|---:|---:|
| Plain BCE | **0.600329** | **0.655421** | 0.883142 | 0.683620 | 2 |
| ASL | 0.529330 | 0.652710 | **0.885651** | **0.686941** | 3 |

ASL은 BCE보다 Macro AUROC `+0.002509`, Macro AP `+0.003321`로 ranking 지표가 소폭 상승했지만 optimized Macro F1은 `-0.002711` 하락해 전체 성능 개선으로 이어지지 않았다. 오심 optimized F1은 `0.4030 → 0.4092`로 소폭 상승했으나 문제를 해결한 수준은 아니며, 복통과 열상 외 다수 클래스의 optimized F1도 하락했다. ASL의 optimized threshold는 전체 클래스에서 0.59~0.72로 BCE의 0.19~0.51보다 높아져 확률 스케일이 전반적으로 위쪽으로 이동했다. 따라서 현재 standalone 대표 설정은 plain BCE로 유지하며, ASL의 효과를 과도하게 해석하지 않는다.

### Pure-nausea sampling ablation

- **목적**: Training의 pure-nausea 그룹(`오심=1, 구토=0`) 노출을 완만하게 늘려 취약한 오심 분류를 개선할 수 있는지 확인했다.
- **설정**: KLUE-RoBERTa, plain BCE, seed 42, `encode_mode=truncate`, max length 512 baseline에서 `WeightedRandomSampler`를 사용해 C 그룹에만 weight `1.5`를 적용했다. Training 그룹은 A 22,687건, B 3,172건, C 2,050건, D 1,291건이다.
- **결과**: best epoch 1, F1@0.5 `0.5855`, optimized Macro F1 `0.6485`였다. optimized F1은 오심 `0.4030 → 0.4081`로 소폭 상승했지만 구토 `0.6066 → 0.5886`, 두통 `0.5525 → 0.5231`로 하락했고, 전체도 `0.655421 → 0.6485`로 감소했다.
- **결론**: C weight 1.5 sampling은 최종 baseline으로 채택하지 않는다. 구현은 재현 가능한 ablation 옵션으로 유지하되 기본값은 OFF로 유지한다.

### Label-wise Attention ablation

- **가설**: CLS representation 하나가 여러 증상의 token evidence를 충분히 보존하지 못해 multi-label 표본의 FN이 증가하는지 검증했다.
- **설정**: KLUE-RoBERTa, plain BCE, seed 42, `encode_mode=truncate`, sampling OFF baseline에서 pooling만 label별 learnable query 기반 token attention으로 변경했다. 추가 파라미터는 6,912개(약 0.006%)다.
- **결과**: best epoch는 모두 2였고, optimized Macro F1은 CLS `0.655421`에서 Label Attention `0.655291`로 `-0.000130` 변했다. positive label 2개 표본의 FN rate는 `39.79% → 41.57%`, 3개 표본은 `44.29% → 45.24%`로 개선되지 않았다.
- **결론**: 일부 클래스의 co-occurrence recall과 broad FP는 개선됐지만 FN 증가와 trade-off가 있었으므로 최종 모델로 채택하지 않는다. pooling 구조는 현재 주요 병목이 아니라고 판단해 이 방향을 종료한다.

### Learning-rate ablation

- **설정**: 최고 성능 KLUE CLS baseline의 learning rate만 `2e-5 → 1e-5`로 낮추고 model, plain BCE, seed 42, `truncate`, sampling OFF 및 나머지 학습 조건을 유지했다.
- **결과**: best epoch 2, F1@0.5 `0.596253`, optimized Macro F1 `0.656180`으로 baseline `0.655421` 대비 `+0.000759`였다.
- **결론**: optimized 지표의 weak positive signal은 있지만 F1@0.5는 `0.600329 → 0.596253`으로 하락했고 클래스별 변화도 혼재했다. LR `1e-5`를 명확히 우수한 설정으로 채택하지 않으며 추가 LR tuning은 종료한다.
- **다음 방향**: stronger public pretrained backbone으로 KF-DeBERTa까지 비교했지만 KLUE 대비 상승 폭이 매우 작고 계산 비용이 컸다. 작은 hyperparameter나 backbone 교체를 반복하기보다 label dependency modeling과 pairwise ranking loss 순으로 구조적 가설을 검토한다.

#### 오심 관찰

- 세 backbone의 optimized 오심 F1은 KoBERT 0.3930, KoELECTRA 0.4006, KLUE-RoBERTa 0.4030으로 거의 개선되지 않아 현재 가장 큰 class-level bottleneck으로 남았다.
- KLUE에서도 threshold 0.5 F1은 0.0529였고 threshold를 0.19로 낮춘 뒤 0.4030이 됐다.
- 원인을 전처리, label noise 또는 구토와의 의미 중첩으로 단정하지 않고 세 모델의 오심/구토 FP/FN 원문을 우선 분석한다.

오심/구토 Validation 오류에 대한 수동 검토와 loss ablation 결과를 함께 고려해 후속 실험의 우선순위를 정한다.

---

## 🤝 협업 및 모델링 연동 가이드

### 1. 데이터 로드 및 전처리 파이프라인
* 학습에는 전처리가 완료된 CSV의 `text`와 9개 binary label column만 사용한다.
* `symptoms`와 문자열 형태의 `label_vector`는 학습 입력으로 사용하지 않는다.
* 원본 JSON 추론 시에는 `m3.labels`의 text-only 추출 규칙을 재사용한다.
* 각 backbone tokenizer는 512 토큰을 상한으로 적용하며 실제 truncation 비율을 run config에 기록한다.

### 2. 다중 라벨 분류(Multi-label) 모델 설계
* **모델 구조**: Hugging Face backbone 위에 9개 타겟 증상 출력을 위한 sequence classification head 구성 (`NUM_CLASSES = 9`). 현재 기준 backbone은 KLUE-RoBERTa이며 backbone benchmark에는 KoBERT, KoELECTRA와 KF-DeBERTa를 포함한다.
* **손실 함수(Loss)**: 각 증상의 발생 여부가 독립적인 다중 라벨 분류이므로, `CrossEntropyLoss` 대신 반드시 `BCEWithLogitsLoss`를 사용.
* **출력 확률 추출**: 검증 및 추론 시 모델 로짓(Logits)에 Sigmoid 함수를 적용하여 [0.0, 1.0] 범위의 확률 행렬(`val_probs`)을 산출.

### 3. 검증 평가 및 최적 임계값 모듈 연동
* **기준 성능**: 실제 검증셋 예측 확률에 threshold 0.5를 적용하여 Macro F1과 클래스별 F1을 기록.
* **임계값 연결**: 저장된 `val_probs.npy`와 `val_labels.npy`를 기존 `m3.threshold.find_best_thresholds()`에 전달할 수 있도록 구성.
* **실행 보류**: 실제 임계값 최적화와 reports 갱신은 별도 확인 후 수행.

---

### ⚠️ 모델링 및 대회 규칙 핵심 주의사항
1. **입력 제약 엄수**: 대회 규정상 대화 본문 텍스트(`utterances[].text`)만 모델 입력으로 사용 가능. 화자, 발화 시간, 인적사항 등 메타데이터 사용 시 규정 위반이므로 `m3.labels` 모듈 사용 필수.
2. **비타겟 증상 노이즈**: 골절, 찰과상, 화상 등 9개 외 증상은 타겟에서 제외되어 0으로 처리되므로, 본문에 통증 호소가 강하더라도 정답 라벨이 0이 되는 데이터 노이즈 특성에 유의.
3. **평가 지표 특성(Macro F1)**: 대회 공식 지표는 클래스별 단순 평균이므로 Accuracy가 아니라 Macro F1과 클래스별 F1을 함께 확인해야 함.
