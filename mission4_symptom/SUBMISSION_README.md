# Mission 4 환자 증상 조기예측 — 제출물

119 신고 통화를 발화 하나씩 받으면서 증상 9종을 예측한다. 4-1 은 매 순간 예측하고, 4-2 는 스스로 판단을 멈출 시점을 정한다.

## 실행

이 폴더에서 공식 명령 그대로 실행한다. 가중치는 `ckpt/mission4.pt` 하나다.

```bash
python inference.py --mission 4-1 --audio_dir {wav folder} --label_dir {json folder} \
    --ckpt_path ckpt/mission4.pt --output ./outputs/mission4-1.csv
python inference.py --mission 4-2 --audio_dir {wav folder} --label_dir {json folder} \
    --ckpt_path ckpt/mission4.pt --output ./outputs/mission4-2.csv
```

- 설치: `pip install -r requirements.txt` (설치 방법과 검증 환경은 파일 머리말에 적었다).
- 번들 안에 tokenizer·config·가중치·결정 규칙·멈춤 기준이 모두 들어 있어 추론 때 인터넷 접속이 없다.
- 실측 시간 (RTX 5060 8GB, cuda, Validation 3,640통화): 4-1 2,178초, 4-2 949초.

## 파일

| 경로 | 내용 |
|---|---|
| `inference.py` | 공식 템플릿. `Predictor` 만 작성했다 (`m4/streaming.py` 를 부른다) |
| `ckpt/mission4.pt` | 제출 가중치 번들 (666,798,993 B, sha256 `eb7e0584213e9ee889284f58164a5a076d939e531bab2c38f827e197e5270352`) |
| `m4/` | 모델 코드. `streaming.py` 추론 상태, `bundle.py` 번들 형식, `decision.py` 4-1 결정 규칙, `halting.py` 학습형 중단 정책(제출 번들에서는 쓰지 않음), `data.py` 입력 형식, `simulate.py` 채점식 |
| `model_train.ipynb` | 학습 기록 노트북 (실행 출력 포함: 데이터 캐시, 레시피, TAPT 기록, 최종 학습 로그, 번들 구성, 학습형 중단 정책 실험 기록) |
| `scripts/` | 학습 코드. 실행 순서는 아래 "다시 학습하기" |
| `tests/` | 채점식 일치, 번들, 결정 규칙, 중단 정책, 스트리밍 경로 테스트 (`pytest`) |
| `score.py` | 주최측 채점 스크립트 (자체 채점용, 수정 없음) |

## 모델

1. **증상 분류기 3개 평균.** 모두 KLUE-RoBERTa-base 를 Training 전사로 추가 사전학습(TAPT, MLM 20 epoch)한 백본에서 출발한다.
   - 앞부분 학습 2개 (seed 42, 43): 통화마다 앞부분 길이를 1..n 에서 균등하게 뽑아 학습한다. 4-1 이 모든 순간을 같은 무게로 평균하므로 학습 분포도 순간에 대해 균등하게 맞췄다.
   - 통화 전체 학습 1개: 예선 Mission 3 최종 모델(같은 백본·입력 형식).
   - 공통 레시피: LLRD 0.8, lr 5e-5, pos_weight=(neg/pos)^0.5, 3 epoch, 512 토큰, 발화를 공백으로 이어 붙인 누적 전사.
2. **4-1 판정.** 대회 규정대로 판정 임계값은 0.5 로 고정한다. 3개 모델의 평균 확률(보정 없음)이 0.5 이상인 증상을 모두 낸다. 0.5 를 넘는 증상이 없으면 빈 리스트를 낸다.
3. **4-2 멈춤.** 멈춤 기준도 0.5 로 고정한다. 다섯 번째 발화부터 매 순간 "지금 답하면 받을 것으로 기대되는 순간 점수"를 계산해 0.5 이상이면 멈춘다. 기대 점수는 평균 확률에서 학습 손실의 pos_weight 만큼 로짓을 되돌린 보정 확률로 계산한다. 멈출 때의 예측은 4-1 과 같은 0.5 판정이다(보정 없는 평균 확률 0.5 이상인 증상).
   - 개발 중에는 LightGBM 으로 멈춤 시점을 학습하는 정책(기준 0.64)도 만들었으나, 멈춤 기준도 0.5 고정이라는 규정에 따라 제출 번들에서 뺐다. 코드(`m4/halting.py`, `scripts/train_halt_policy.py`)와 노트북 기록은 남겨 두었다.

## 규정 준수

- **입력:** 매 순간 그때까지 받은 발화의 `text`·`speaker`·`start_ms`·`end_ms` 만 쓴다 (제출 번들은 `text` 만 쓴다. `speaker`·시각은 쓰지 않은 학습형 중단 정책의 특징이었다). 총 발화 수·총 통화 시간·이후 발화·라벨 필드·원본 wav 는 읽지 않는다.
- **학습 데이터:** AI-Hub Training 의 서울 구급 통화 29,200건(9종 필터 뒤)만 썼다. Validation 은 학습·TAPT·값 선택 어디에도 쓰지 않았다.
- **값 선택:** 레시피의 모든 값과 멈춤을 시작하는 발화 수(5)는 Training 안에서 나눈 dev(2,920통화)와 교차 검증으로 정했다. 최종 모델은 같은 값으로 Training 전체를 다시 학습했다.
- **사전학습 모델:** `klue/roberta-base` (공개 오픈소스). 상용 API 는 쓰지 않는다.

## 성능

dev (Training 내부 2,920통화, 같은 구성을 학습용 26,280통화로만 학습한 모델):

| | 4-1 | 4-2 |
|---|---|---|
| 제출 번들 구성 (판정·멈춤 모두 0.5 고정) | 0.5060 | 0.4988 |

최종 번들은 dev 통화까지 학습했으므로 dev 로 잴 수 없어, Validation 으로 한 번 확인했다 (이 결과로 바꾼 값은 없다):

| | 4-1 | 4-2 |
|---|---|---|
| 제출 번들, Validation 3,640통화 | 0.4867 | 0.4883 (Macro-F1 0.6069 × 조기판단력 0.8046) |

## 다시 학습하기

```bash
python scripts/build_cache.py ...            # Training 라벨 json → runs/cache/train.jsonl
python scripts/tapt_mlm.py ...               # TAPT (또는 run_final_training.sh 의 TAPT_MODE=full)
bash scripts/run_final_training.sh           # Training 전체로 앞부분 학습 모델 2개
python scripts/pack_bundle.py ...            # 번들 → ckpt/mission4.pt
```

각 명령의 실제 인자와 실행 로그는 `model_train.ipynb` 에 있다.
