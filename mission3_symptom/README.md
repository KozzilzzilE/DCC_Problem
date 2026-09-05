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
│   ├── config.py                # 9개 타겟 증상 상수 및 경로 설정
│   ├── labels.py                # 대회 규정 강제 라벨 파서 및 데이터 로더
│   ├── metrics.py               # 이진 F1 및 규정 준수 Macro F1 계산 함수
│   ├── threshold.py             # 9개 증상별 최적 임계값 그리드 탐색기
│   ├── report.py                # 성과 리포트(MD) 및 최적 임계값(JSON) 생성기
│   └── __init__.py              # m3 통합 인터페이스 export
├── reports/
│   ├── comparison.md            # 기본 0.5 vs 최적 임계값 전/후 F1 성과 리포트
│   └── best_thresholds.json     # 최종 추론 시 로드되는 9개 최적 임계값 파일
├── model_train.ipynb            # ★ 메인 시각화 노트북 (F1 반응 곡선 차트, 비교표)
└── README.md                    # 현재 문서
```

---

## 🧩 `m3` 모듈별 상세 스펙 및 역할

| 모듈 파일 | 핵심 역할 | 주요 함수 / 클래스 | 세부 설명 |
|:---|:---|:---|:---|
| **`config.py`** | 글로벌 설정 및 표준 상수 | `TARGET_SYMPTOMS`<br>`NUM_CLASSES = 9`<br>`SYMPTOM_TO_IDX` | 9개 공식 증상 목록, 증상별 정수 인덱스 매핑 및 리포트 저장 경로 정의. |
| **`labels.py`** | 대회 규칙 강제 전처리 파이프라인 | `load_transcripts_dataframe()`<br>`load_transcripts_dir()`<br>`read_transcript()` | 수만 건의 JSON에서 화자/시간 메타데이터를 원천 배제하고 순수 대화 본문만 추출. 비타겟 증상 필터링 후 9차원 원-핫(이진) 벡터로 변환. Windows/macOS(NFC 정규화)/Colab 다중 인코딩 Fallback 지원. |
| **`metrics.py`** | 공식 평가지표 계산기 | `eval_macro_f1()`<br>`calculate_binary_f1()` | 증상별 정밀도(Precision)와 재현율(Recall) 기반 이진 F1 계산(ZeroDivision 안전 처리) 및 산술 평균 기반 대회 공식 Macro F1 산출. |
| **`threshold.py`** | 임계값 최적화 엔진 | `find_best_thresholds()`<br>`apply_thresholds()`<br>`get_threshold_curves()` | 0.05~0.95 구간(0.01 간격) 그리드 탐색을 통해 9개 증상별 F1을 극대화하는 황금 임계값 벡터 산출 및 시각화용 반응 곡선 데이터 생성. |
| **`report.py`** | 성과 문서 & 추론 설정 자동화 | `generate_comparison_markdown()`<br>`save_thresholds_json()` | 기준선(0.5) 대비 F1 상승폭을 마크다운 리포트(`comparison.md`)로 자동 작성하고, 최종 추론용 `best_thresholds.json` 파일 저장. |
| **`__init__.py`** | 패키지 인터페이스 허브 | `m3.*` 공개 API 노출 | 모델링 담당자가 복잡한 내부 구조를 몰라도 `from m3 import ...` 한 줄로 전처리 및 평가 함수를 즉시 호출 가능하도록 구성. |

---

## 🚀 빠른 실행 가이드

### 주피터 노트북 실행 (`model_train.ipynb`)
VS Code 또는 Jupyter 환경에서 `mission3_symptom/model_train.ipynb`를 열고 순서대로 셀을 실행하면:
1. 검증셋 데이터 로드 (가중치 미완성 시 시뮬레이션 데이터 자동 가동)
2. 기준선 0.5 F1 측정
3. 9개 증상별 F1 반응 곡선(3x3 서브플롯) 시각화
4. Before / After F1 상승폭 막대그래프 출력
5. `reports/comparison.md` 및 `reports/best_thresholds.json` 자동 생성.

---

## 🤝 협업 및 모델링 연동 가이드

### 1. 데이터 로드 및 전처리 파이프라인 (완수 님 연동)
* **DataFrame으로 바로 가져오기 (권장)**:
  ```python
  from m3.labels import load_transcripts_dataframe
  
  # 원본 라벨 폴더만 지정하면 규정 준수 정제 데이터프레임 즉시 반환
  train_df = load_transcripts_dataframe("data/train/label")
  val_df = load_transcripts_dataframe("data/val/label")
  
  # train_df['text'] -> 순수 대화 전사 텍스트
  # train_df['label_vector'] -> 9차원 이진 리스트 ([0, 1, 0, 0, 0, 0, 0, 1, 0])
  ```
* **토큰화**: KoBERT 사전학습 토크나이저를 적용하여 텍스트 인코딩 진행 (권장 Max Length: 512 토큰).

### 2. 다중 라벨 분류(Multi-label) 모델 설계
* **모델 구조**: KoBERT 백본 위에 9개 타겟 증상 출력을 위한 선형 분류 헤드(Linear Head) 구성 (`NUM_CLASSES = 9`).
* **손실 함수(Loss)**: 각 증상의 발생 여부가 독립적인 다중 라벨 분류이므로, `CrossEntropyLoss` 대신 반드시 `BCEWithLogitsLoss`를 사용.
* **출력 확률 추출**: 검증 및 추론 시 모델 로짓(Logits)에 Sigmoid 함수를 적용하여 [0.0, 1.0] 범위의 확률 행렬(`val_probs`)을 산출.

### 3. 검증 평가 및 최적 임계값 모듈 연동
* **임계값 최적화**: 학습 완료 후 검증셋 예측 확률(`val_probs`)과 실제 정답(`val_labels`)을 `m3.threshold.find_best_thresholds()`에 전달하여 9개 클래스별 최적 임계값 도출.
* **성과표 및 임계값 갱신**: `m3.report` 모듈을 통해 `reports/best_thresholds.json` 및 `reports/comparison.md`를 자동 갱신하여 최종 `inference.py` 제출에 연동.

---

### ⚠️ 모델링 및 대회 규칙 핵심 주의사항
1. **입력 제약 엄수**: 대회 규정상 대화 본문 텍스트(`utterances[].text`)만 모델 입력으로 사용 가능. 화자, 발화 시간, 인적사항 등 메타데이터 사용 시 규정 위반이므로 `m3.labels` 모듈 사용 필수.
2. **비타겟 증상 노이즈**: 골절, 찰과상, 화상 등 9개 외 증상은 타겟에서 제외되어 0으로 처리되므로, 본문에 통증 호소가 강하더라도 정답 라벨이 0이 되는 데이터 노이즈 특성에 유의.
3. **평가 지표 특성(Macro F1)**: 대회 공식 지표는 클래스별 단순 평균인 Macro F1이므로, 다빈도 클래스에 편향된 Accuracy 대신 희귀 클래스를 살릴 수 있는 오현 님의 클래스별 임계값 튜닝을 필수로 거쳐야 함.