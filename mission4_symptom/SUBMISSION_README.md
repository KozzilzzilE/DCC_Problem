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
- 번들 안에 tokenizer·config·가중치·결정 규칙·중단 정책이 모두 들어 있어 추론 때 인터넷 접속이 없다.
- 4-2 중단 정책은 LightGBM 모델이다. lightgbm 이 설치돼 있지 않으면 번들에 든 대체 기준(기대 점수 0.50)으로 내려가며 경고를 출력한다.
- 실측 시간 (RTX 5060 8GB, cuda, Validation 3,640통화): 4-1 1,411초, 4-2 410초.

## 파일

| 경로 | 내용 |
|---|---|
| `inference.py` | 공식 템플릿. `Predictor` 만 작성했다 (`m4/streaming.py` 를 부른다) |
| `ckpt/mission4.pt` | 제출 가중치 번들 (667,128,618 B, sha256 `ce35f219ab347989350cb25ec7c935b026144126e6356d45a5f15b6eb95af140`) |
| `m4/` | 모델 코드. `streaming.py` 추론 상태, `bundle.py` 번들 형식, `decision.py` 4-1 결정 규칙, `halting.py` 4-2 중단 정책, `data.py` 입력 형식, `simulate.py` 채점식 |
| `model_train.ipynb` | 학습 기록 노트북 (실행 출력 포함: 데이터 캐시, 레시피, TAPT 기록, 최종 학습 로그, 번들 구성, 중단 정책 학습) |
| `scripts/` | 학습 코드. 실행 순서는 아래 "다시 학습하기" |
| `tests/` | 채점식 일치, 번들, 결정 규칙, 중단 정책, 스트리밍 경로 테스트 (`pytest`) |
| `score.py` | 주최측 채점 스크립트 (자체 채점용, 수정 없음) |

## 모델

1. **증상 분류기 3개 평균.** 모두 KLUE-RoBERTa-base 를 Training 전사로 추가 사전학습(TAPT, MLM 20 epoch)한 백본에서 출발한다.
   - 앞부분 학습 2개 (seed 42, 43): 통화마다 앞부분 길이를 1..n 에서 균등하게 뽑아 학습한다. 4-1 이 모든 순간을 같은 무게로 평균하므로 학습 분포도 순간에 대해 균등하게 맞췄다.
   - 통화 전체 학습 1개: 예선 Mission 3 최종 모델(같은 백본·입력 형식).
   - 공통 레시피: LLRD 0.8, lr 5e-5, pos_weight=(neg/pos)^0.5, 3 epoch, 512 토큰, 발화를 공백으로 이어 붙인 누적 전사.
2. **4-1 결정 규칙.** 평균 확률을 pos_weight 만큼 로짓 보정한 뒤, Training 라벨의 증상 조합 분포를 사전분포로 써서 순간 점수의 기댓값이 가장 큰 증상 집합을 고른다. 빈 예측은 내지 않는다.
3. **4-2 중단 정책.** 매 순간 "지금 멈출 때의 기대 점수 + 16 × 더 들으면 얻을 것으로 예측한 이득" 을 LightGBM 이 추정하고, 0.64 를 넘으면 멈춘다. 멈출 때의 예측은 임계값 0.5 (비면 확률 최대 1개)다.

## 규정 준수

- **입력:** 매 순간 그때까지 받은 발화의 `text`·`speaker`·`start_ms`·`end_ms` 만 쓴다 (`speaker`·시각은 4-2 중단 정책의 특징에만 쓴다). 총 발화 수·총 통화 시간·이후 발화·라벨 필드·원본 wav 는 읽지 않는다. 통화 길이는 학습 때 목표(더 들으면 얻을 이득)를 만드는 데만 썼다.
- **학습 데이터:** AI-Hub Training 의 서울 구급 통화 29,200건(9종 필터 뒤)만 썼다. Validation 은 학습·TAPT·값 선택 어디에도 쓰지 않았다.
- **값 선택:** 레시피·결정 규칙·중단 정책의 모든 값은 Training 안에서 나눈 dev(2,920통화)와 교차 검증으로 정했다. 최종 모델은 같은 값으로 Training 전체를 다시 학습했다.
- **사전학습 모델:** `klue/roberta-base` (공개 오픈소스). 상용 API 는 쓰지 않는다.

## 성능

| | Validation (최종 번들, 확인 1회) | 참고: dev (Training 내부, 같은 구성의 fit-only 모델) |
|---|---|---|
| 4-1 | 0.5302 | 0.5517 |
| 4-2 | 0.5042 (Macro-F1 0.6166 × 조기판단력 0.8177) | 0.5064 |

Validation 은 모든 값을 정한 뒤 한 번만 돌렸고, 그 결과로 바꾼 값은 없다.

## 다시 학습하기

```bash
python scripts/build_cache.py ...            # Training 라벨 json → runs/cache/train.jsonl
python scripts/tapt_mlm.py ...               # TAPT (또는 run_final_training.sh 의 TAPT_MODE=full)
bash scripts/run_oof_tables.sh               # 5겹 교차 학습 확률표 (중단 정책 학습 자료)
python scripts/train_halt_policy.py ...      # 4-2 중단 정책
bash scripts/run_final_training.sh           # Training 전체로 앞부분 학습 모델 2개
python scripts/pack_bundle.py ...            # 번들 → ckpt/mission4.pt
```

각 명령의 실제 인자와 실행 로그는 `model_train.ipynb` 에 있다.
