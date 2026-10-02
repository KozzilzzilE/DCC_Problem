# Mission 3 — 환자 증상 인식 (예선 제출본)

119 신고 통화 전사(`utterances[].text`)만 보고 9개 증상(고열·구토·두통·복통·어지러움·열상·오심·전신쇠약·호흡곤란)을 다중 라벨로 분류한다.

- **Validation macro F1@0.5 = 0.6599** (3,640건). 하이퍼파라미터를 Training 내부 dev 로 모두 정한 뒤, 이 폴더의 `inference.py` 로 확인한 값이다 (Validation 은 선택에 쓰지 않았다).
- 결정 임계값은 **모든 클래스 0.5 고정**이다 (대회 규정). 클래스별·튜닝된 임계값은 쓰지 않는다.

## 폴더 구성

```text
mission3_symptom/
├── inference.py            # 추론 진입점 — 한 번 실행으로 결과 CSV 생성
├── ckpt/
│   └── mission3.pt         # 제출 번들 .pt 하나: KLUE-RoBERTa-base 분류 모델 4개(TAPT + LLRD, seed 42~45)의
│                           #   가중치(fp32 저장)·tokenizer·config, 추론 정밀도 fp16. 인터넷 불필요
├── m3/                     # 전처리·모델·학습·추론 모듈
│   ├── config.py           #   9개 증상, 발화 경계 모드 등 상수
│   ├── labels.py           #   라벨 JSON -> 대화 본문·정답 (본문 외 메타데이터는 파싱 단계에서 버림)
│   ├── dataset.py          #   학습 CSV -> 토큰화(512 토큰 앞부분) -> DataLoader
│   ├── model.py            #   KLUE-RoBERTa 분류 모델·토크나이저 로드와 호환성 검사
│   ├── training.py         #   학습 루프 (pos_weight BCE, LLRD, 체크포인트 저장)
│   ├── metrics.py          #   임계값 0.5 macro F1
│   ├── tfidf_member.py     #   TF-IDF 보조 멤버 (dev 에서 블렌드 가중치가 0 보다 크게 정해질 때만 사용)
│   ├── bundle.py           #   .pt 제출 번들 묶기·불러오기
│   └── infer.py            #   제출 추론 (멤버 평균, 0.5 판정, CSV 규격)
├── make_csv.py             # 원본 JSON -> 학습 CSV
├── make_dev_split.py       # Training 내부 dev 분할 (하이퍼파라미터 선택 전용)
├── tapt_mlm.py             # TAPT (MLM 추가 사전학습)
├── train.py                # 분류 학습
├── devsel.py               # dev 기준 선택 규칙·번들(.pt) 조립·최종 확인 채점
├── train_tfidf_member.py   # TF-IDF 멤버 학습
├── run_dev_selection.sh    # 위 단계를 순서대로 실행하는 학습 스크립트
├── model_train.ipynb       # 학습 과정·로그·dev 선택 결과·Validation 평가 기록
├── build_notebook.py       # model_train.ipynb 생성 스크립트 (git 저장소에만 있음, 제출 폴더에서는 뺌)
├── reports/                # dev_selection_protocol.md(사전 등록 규칙), dev_selection_eval.md(결과)
├── tests/                  # 단위·통합 테스트 (git 저장소에만 있음, 제출 폴더에서는 뺌)
├── requirements.txt
└── README.md
```

`*.pt` 는 저장소 `.gitignore` 대상이라 git 저장소에는 빈 `ckpt/` 만 있다. 제출 폴더에는 `run_dev_selection.sh` 가 만든 번들을 `ckpt/mission3.pt` 로 넣는다.

## 설치

Python 3.12 이상 (검증 3.14.6). 이 폴더에서:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 는 PyTorch CUDA 13.0 빌드(torch 2.13.0+cu130)를 받는다. NVIDIA 드라이버 580 이상이 필요하며, 드라이버가 더 오래됐으면 파일 안의 `cu130` 을 `cu126` 으로 바꾼다.

## 추론 (주최 측 명령 형식)

이 폴더에서:

```bash
python inference.py --audio_dir <wav 폴더> --label_dir <json 폴더> --ckpt_path ckpt/mission3.pt --output ./outputs/mission3.csv
```

- **`--audio_dir`:** 받기만 하고 읽지 않는다 (입력은 대화 본문뿐).
- **`--ckpt_path`:** 제출 번들 `ckpt/mission3.pt` 파일을 받는다. 학습 run 폴더(`best_model` 포함) 하나를 주면 그 모델 하나로 추론한다 (단일 모델 확인용).
- **출력 CSV:** 열은 `label file name`, `symptom` 이다. symptom 은 `"['두통', '복통']"` 형식이고, 증상이 없으면 `"[]"` 다.
- **추론 정밀도 fp16:** 번들에 저장된 `"precision": "fp16"` 은 CUDA 에서만 적용되고, 이 폴더의 `m3/infer.py` 가 읽는다.
  - GPU 에서 메모리 부족이 나면 배치를 나눠 다시 돌린다.
  - 그래도 안 되거나 CUDA 실행 오류가 날 때만 CPU fp32 로 내려 결과 파일을 끝까지 만든다. 1회 실행이 중간에 죽지 않게 하는 안전장치다.
- **가중치 형식:** 출제문제 12쪽의 `.pt` 파일 하나다. `torch.save` 한 dict 에 멤버 4개의 가중치와 tokenizer·config 파일 원문이 모두 들어 있고, 이 폴더의 로더(`m3/bundle.py`)가 Hugging Face 표준 로더로 다시 조립한다.

## 계산 효율

| 항목 | 값 |
|---|---|
| Total 파라미터 | **442,500,132** = KLUE-RoBERTa-base 110,625,033 × 4 |
| Active 파라미터 | **442,500,132** (앙상블 멤버가 모두 매 샘플 추론에 쓰이는 dense 구조) |
| 학습·추론 환경 | NVIDIA GeForce RTX 5060 8GB (드라이버 610.62, CUDA 13.0), AMD Ryzen 5 9600 (6코어 12스레드), RAM 31GB, Windows 10, Python 3.14.6, torch 2.13.0+cu130, transformers 5.15.0 |
| Validation 추론 batch size | 16 |
| Validation 전체 추론 시간 | 3,640건 **약 43초 (샘플당 약 12 ms)**, `--ckpt_path ckpt/mission3.pt`, fp16, 모델 로딩·CSV 저장 포함 (10-02 여러 번 실측 42.7~45.1초). 단 .pt 를 막 복사했거나 학습 직후처럼 파일 캐시가 비어 있는 첫 실행은 더 걸린다 (제출 폴더로 복사한 직후 .pt 55.4초, 10-01 학습 직후 같은 가중치의 폴더 번들 50.8초) |
| 학습 시간 | 약 5시간 10분 (`run_dev_selection.sh` 전체) |

학습 시간 내역:
- TAPT 20 epoch: 141분
- dev 선택용 분류 학습 6회: 약 1시간 50분
- 최종 분류 학습 4개: 각 13분

## 방법

1. **보정 손실 (임계값 0.5 고정 대응)**
   - 불균형 때문에 0.5 에서 양성을 놓치는 문제를 임계값이 아니라 학습 손실로 다룬다.
   - BCE 의 양성 가중을 `pos_weight = (Training 음성/양성)^p` 로 둔다.
   - 음성·양성 수는 Training 라벨로만 계산한다.
2. **TAPT:** KLUE-RoBERTa 는 통화 전사문에서 MLM 손실이 약 5.1 로 도메인 차이가 크다. 그래서 Training 본문만으로 MLM 을 20 epoch 이어 학습했다.
3. **LLRD 0.8:** 분류 헤드 lr 5e-5 에서 시작해, 인코더 층을 내려갈수록 0.8 배씩 줄인다.
4. **앙상블:** seed 42~45 네 모델의 확률을 균등 평균하고 0.5 로 판정한다.

### 하이퍼파라미터 선택 — Training 내부 dev

Training 을 통화 단위로 90/10 분할했다 (`make_dev_split.py`, seed 1234). 학습용 분할로 후보를 학습하고 dev 분할(2,920건)로 아래 값을 정했다. 선택 규칙은 결과를 보기 전에 `reports/dev_selection_protocol.md` 로 커밋했다.

| 결정 | 결과 | dev macro F1@0.5 |
|---|---|---|
| pos_weight 지수 p | **0.5** | 0.6503 (p=0: 0.6189, p=1.0: 0.6141) |
| 레시피 | **TAPT + LLRD 0.8** | 0.6542 (원래 레시피 0.6505), seed 2개 평균 |
| 저장 epoch | **2** (3 epoch 스케줄) | 0.6542 (3 epoch: 0.6533) |
| TF-IDF 블렌드 | **쓰지 않음** (w=0) | 0.6580 (w=0.3: 0.6550) |
| 모델 수 | **4개** | 2-seed 앙상블 0.6580 > 단일 평균 0.6542 |

정한 설정으로 Training 전체(29,200건)를 다시 학습했다. 3 epoch 스케줄의 2 epoch 시점에서 평가 점수를 보지 않고 저장했다. 그 번들로 Validation 을 확인한 결과는 **0.6599** 다. 단계별 표와 비용은 `reports/dev_selection_eval.md`, 학습 로그는 `model_train.ipynb` 에 있다.

## 학습 재현

이 폴더에서 실행한다 (RTX 5060 기준 약 5시간 10분).
- 시작점 `klue/roberta-base`(공개 모델)는 첫 줄 명령으로 한 번 내려받는다. 학습 스크립트는 오프라인 모드로 돌므로 이 캐시가 필요하다.
- 추론에는 인터넷이 필요 없다.

```bash
python -c "from transformers import AutoModelForMaskedLM, AutoTokenizer; AutoTokenizer.from_pretrained('klue/roberta-base'); AutoModelForMaskedLM.from_pretrained('klue/roberta-base')"
python make_csv.py --label-dir <data>/train/label --output data_csv/mission3_train.csv
bash run_dev_selection.sh data_csv/mission3_train.csv <data>/val/label
```

스크립트가 하는 일:
1. dev 분할과 후보 학습, 규칙에 따른 결정을 한다.
2. Training 전체로 최종 학습하고 번들(`runs/devsel/mission3.pt`)을 조립한다 (`devsel.py assemble`).
3. 마지막에 그 번들로 Validation 을 한 번 추론해 성능을 확인한다. Validation 은 이 마지막 확인에만 쓴다.

끝나면 `runs/devsel/mission3.pt` 를 `ckpt/mission3.pt` 로 복사한다.

테스트는 git 저장소의 이 폴더에서 `python -m pytest tests` 로 돌린다 (pytest 는 따로 설치. 제출 폴더에는 `tests/` 가 없다).

## 규정 준수

- **학습 데이터:** 역전파에는 Training(서울) 데이터만 썼다.
- **하이퍼파라미터 선택:** p, 레시피, 저장 epoch, TF-IDF 블렌드 C·w, 모델 수는 모두 Training 내부 dev 로 정했다.
  - 이는 주최 측 답변을 따른 것이다. p·w·C 는 Validation 이 아니라 Training 내부 dev/OOF 로 정하고, Validation 은 결정된 모델의 성능 확인에만 쓴다.
  - `devsel.py` 는 선택에 쓰는 모든 run 이 dev 분할로 평가됐는지 확인한다.
  - 다만 dev 가 비교한 레시피 후보 두 개(원래 레시피, TAPT 20 epoch + LLRD 0.8·lr 5e-5)는 9/30 답변 이전의 Validation 탐색에서 나온 것이다. dev 는 두 후보 중 무엇을 쓸지와 위 값들을 정했고, 레시피 안의 값(20 epoch, LLRD 0.8, lr 5e-5)은 다시 고르지 않았다 (`reports/dev_selection_protocol.md` '한계' 절에 실행 전부터 적어 둠).
- **Validation 사용:** 결정이 모두 끝난 번들의 성능 확인에만 썼다.
- **입력:** 대화 본문 텍스트뿐이다. 화자·시간·인적사항은 파싱 단계(`m3/labels.py`)에서 버린다.
- **결정 임계값:** 모든 클래스 0.5 고정이다 (`m3/infer.py` 의 `DECISION_THRESHOLD`). 판정은 이 값 하나로만 한다. 번들 안 `inference_config.json` 에는 학습 때 기록한 `threshold: 0.5` 가 있지만 추론은 이 값을 읽지 않는다.
- **사전학습 모델:** 공개 모델(klue/roberta-base)만 썼고, 모델 학습·추론에 상용 API 는 쓰지 않았다.
