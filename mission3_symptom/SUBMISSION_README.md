# Mission 3 — 환자 증상 인식 (예선 제출본)

119 신고 통화 전사(`utterances[].text`)만 보고 9개 증상(고열·구토·두통·복통·어지러움·열상·오심·전신쇠약·호흡곤란)을 다중 라벨로 분류한다.

- **Validation macro F1@0.5 = 0.6593** (3,640건, 이 폴더의 `inference.py` 로 측정)
- 결정 임계값은 **모든 클래스 0.5 고정**이다 (대회 규정). 클래스별·튜닝된 임계값은 쓰지 않는다.

## 폴더 구성

```text
mission3_symptom/
├── inference.py            # 추론 진입점 — 한 번 실행으로 결과 CSV 생성
├── m3/                     # 전처리·모델·추론·학습 모듈
├── ckpt/                   # 제출 번들 (tokenizer/config/가중치 포함, 인터넷 불필요)
│   ├── ensemble.json       # 번들 진입점: 멤버 4개, TF-IDF 가중치 0.3, precision fp16
│   ├── seed42 ~ seed45/    # KLUE-RoBERTa-base 분류 모델 (시드별)
│   └── tfidf/              # Training 전용 TF-IDF + LogisticRegression
├── make_csv.py             # 학습 1단계: 원본 JSON -> 학습 CSV
├── tapt_mlm.py             # 학습 2단계: TAPT (MLM 추가 사전학습)
├── train.py                # 학습 3단계: 분류 학습
├── train_tfidf_member.py   # 학습 4단계: TF-IDF 멤버
├── model_train.ipynb       # 학습 과정·로그·Validation 평가 기록
├── reports/                # 실험 보고서: calibration_eval.md(보정 손실), improvement_eval.md(TAPT·LLRD·앙상블)
├── requirements.txt
└── README.md
```

## 설치

Python 3.12 이상 (검증 3.14.6). 이 폴더에서:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 는 PyTorch CUDA 13.0 빌드(torch 2.13.0+cu130)를 받는다. NVIDIA 드라이버 580 이상이 필요하며, 드라이버가 더 오래됐으면 파일 안의 `cu130` 을 `cu126` 으로, GPU 가 없으면 `cpu` 로 바꾼다.

## 추론 (주최 측 명령 형식)

이 폴더에서:

```bash
python inference.py --audio_dir <wav 폴더> --label_dir <json 폴더> --ckpt_path ckpt/ensemble.json --output ./outputs/mission3.csv
```

- `--audio_dir` 는 받기만 하고 읽지 않는다 (입력은 대화 본문뿐).
- `--ckpt_path` 는 `ckpt/ensemble.json` 파일 또는 `ckpt` 폴더를 받는다.
- 출력 CSV: `label file name`, `symptom`. symptom 은 `"['두통', '복통']"` 형식이고 증상이 없으면 `"[]"` 다.
- `ensemble.json` 의 `"precision": "fp16"` 은 CUDA 에서만 적용된다. GPU 가 없거나 GPU 에서 실패하면 자동으로 CPU fp32 로 끝까지 추론한다. 이 키는 이 폴더의 `m3/infer.py` 가 읽는다.
- 가중치는 Hugging Face 표준 `model.safetensors`(멤버별)와 scikit-learn `joblib`(TF-IDF 멤버)이고, `ensemble.json` 이 번들 진입점이다. 출제문제 12쪽의 `.pt/.pth/.ckpt` 예시와 확장자가 다르지만 모두 이 폴더의 로더(`m3/model.py`, `m3/tfidf_member.py`)가 로컬 경로에서 불러온다.

## 계산 효율

| 항목 | 값 |
|---|---|
| Total 파라미터 | **445,681,425** = KLUE-RoBERTa-base 110,625,033 × 4 + TF-IDF LogisticRegression 3,181,293 (9 × 353,477) |
| Active 파라미터 | **445,681,425** (앙상블 멤버가 모두 매 샘플 추론에 쓰이는 dense 구조) |
| 학습·추론 환경 | NVIDIA GeForce RTX 5060 8GB (드라이버 610.62, CUDA 13.0), AMD Ryzen 5 9600 (6코어 12스레드), RAM 31GB, Windows 10, Python 3.14.6, torch 2.13.0+cu130, transformers 5.15.0 |
| Validation 추론 batch size | 16 |
| Validation 전체 추론 시간 | 3,640건 **약 42초 (샘플당 11.6 ms)**, fp16, 모델 로딩·TF-IDF·CSV 저장 포함 (fp32 로 돌리면 약 104초, 샘플당 28.7 ms) |
| GPU 없는 환경 (참고) | CPU fp32 샘플당 약 1초 (12건 표본, 로딩 포함). Validation 전체면 약 1시간으로 추정 |
| 학습 시간 | TAPT 20 epoch 약 2시간 35분 + 분류 학습 시드당 약 19분 × 4 + TF-IDF 약 30초 |

## 방법

1. **보정 손실 (임계값 0.5 고정 대응)**: 불균형 때문에 0.5 에서 양성을 놓치는 문제를 임계값이 아니라 학습 손실로 해결했다. BCE 의 양성 가중을 `pos_weight = (Training 음성/양성)^0.5` 로 두면 9개 클래스 모두 F1@0.5 가 오른다 (0.5967 → 0.6496).
2. **TAPT**: KLUE-RoBERTa 는 통화 전사문에서 MLM 손실이 약 5.1 로 도메인 차이가 크다. Training 본문만으로 MLM 을 20 epoch 이어 학습했다.
3. **LLRD 0.8**: 분류 학습에서 분류 헤드 lr 5e-5, 인코더 층을 내려갈수록 0.8 배씩 줄여 사전학습 표현을 덜 흔든다.
4. **앙상블**: 시드 42~45 네 모델의 확률을 균등 평균하고, Training 전용 TF-IDF LogisticRegression 을 전역 가중치 0.3 으로 섞은 뒤 0.5 로 판정한다.

| 설정 (Validation macro F1@0.5) | 값 |
|---|---:|
| KLUE-RoBERTa-base, plain BCE (기준선) | 0.5967 |
| + pos_weight^0.5 | 0.6496 |
| + TAPT 20ep + LLRD 0.8 (시드 42~45 단일 모델 평균) | 0.6543 |
| **+ 4시드 앙상블 + TF-IDF (제출)** | **0.6593** |

실험 비교와 근거는 `reports/improvement_eval.md`, `reports/calibration_eval.md`, 학습 로그는 `model_train.ipynb` 에 있다.

## 학습 재현

이 폴더에서 순서대로 실행한다 (RTX 5060 기준 약 4시간). 시작점 `klue/roberta-base`(공개 모델)는 TAPT 첫 실행 때 한 번 내려받는다. 추론에는 인터넷이 필요 없다.

```bash
python make_csv.py --label-dir <data>/train/label --output data_csv/mission3_train.csv
python make_csv.py --label-dir <data>/val/label --output data_csv/mission3_val.csv
python tapt_mlm.py --train-csv data_csv/mission3_train.csv --output-dir runs/tapt_klue_base_e20 --amp --epochs 20
python train.py --train-csv data_csv/mission3_train.csv --val-csv data_csv/mission3_val.csv --model-name-or-path runs/tapt_klue_base_e20 --local-files-only --learning-rate 5e-5 --llrd-decay 0.8 --use-pos-weight --pos-weight-power 0.5 --checkpoint-metric val_macro_f1 --amp --seed 42 --output-dir runs/seed42
python train_tfidf_member.py --train-csv data_csv/mission3_train.csv --output runs/tfidf/tfidf_lr.joblib
```

`train.py` 는 시드 43·44·45 도 같은 명령으로 돌린다. 끝나면 각 `runs/seed*/best_model` 을 `ckpt/seed*` 로, `runs/tfidf/` 를 `ckpt/tfidf/` 로 복사하고 `ckpt/ensemble.json` 을 위 구성대로 둔다.

## 규정 준수

- 학습(역전파)에는 Training(서울) 데이터만 썼다. Validation 은 체크포인트 선택·하이퍼파라미터 선택·평가에만 썼다.
- 입력은 대화 본문 텍스트뿐이다. 화자·시간·인적사항은 파싱 단계(`m3/labels.py`)에서 버린다.
- 결정 임계값은 모든 클래스 0.5 고정이다 (`m3/infer.py` 의 `DECISION_THRESHOLD`). `m3/threshold.py`·`m3/report.py` 의 임계값 탐색 함수는 초기 분석용이며 추론·학습 경로에서 쓰지 않는다.
- 공개 사전학습 모델(klue/roberta-base)만 썼고, 모델 학습·추론에 상용 API 는 쓰지 않았다.
