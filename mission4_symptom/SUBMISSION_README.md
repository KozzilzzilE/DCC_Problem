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
- 번들 안에 tokenizer·config·가중치·결정 규칙·멈춤 정책이 모두 들어 있어 추론 때 인터넷 접속이 없다.
- 실측 시간 (RTX 5060 8GB, cuda, Validation 3,640통화): 4-1 1,333초, 4-2 461초.
- 4-2 멈춤 정책은 LightGBM 모델이다. lightgbm 이 설치돼 있지 않으면 번들에 든 대체 기준(기대 점수 ≥ 0.5)으로 내려가며 경고를 출력한다.

## 파일

| 경로 | 내용 |
|---|---|
| `inference.py` | 공식 템플릿. `Predictor` 만 작성했다 (`m4/streaming.py` 를 부른다) |
| `ckpt/mission4.pt` | 제출 가중치 번들 (`version 2` 미션 분리형, 1,111,644,586 B / 1,060 MiB, sha256 `f681d55a80daec5e3db4a5ad1ccafc8790c06fcc56880a6efdfc5586e5ab18e2`) |
| `m4/` | 모델 코드. `streaming.py` 추론 상태, `bundle.py` 번들 형식, `decision.py` 4-1 결정 규칙, `halting.py` 4-2 멈춤 정책, `data.py` 입력 형식, `simulate.py` 채점식 |
| `model_train.ipynb` | 학습 기록 노트북 (실행 출력 포함: 데이터 캐시, 레시피, TAPT 기록, 최종 학습 로그, 번들 구성, 멈춤 정책 학습) |
| `scripts/` | 학습 코드. 실행 순서는 아래 "다시 학습하기" |
| `tests/` | 채점식 일치, 번들, 결정 규칙, 중단 정책, 스트리밍 경로 테스트 (`pytest`) |
| `score.py` | 주최측 채점 스크립트 (자체 채점용, 수정 없음) |

## 모델

단일 제출 파일 `ckpt/mission4.pt`(`version 2` 미션 분리 번들) 안에 `4-1` 전용 모델과 `4-2` 전용 모델·정책이 분리 수록돼 있으며, 실행 시 `--mission` 에 해당하는 멤버만 메모리에 적재한다. 모두 `klue/roberta-base` 를 AI-Hub Training 서울 구급 전사로 추가 사전학습(TAPT, MLM 20 epoch)한 백본에서 출발했다(`fit 26,280`통화 학습).

1. **4-1 증상 분류 (`SoftJaccard` 2-seed 앙상블: seed 42 + 43).**
   - 학습: 앞부분(1..n 균등 샘플링) 학습에 `BCE(pos_weight^0.5) + SoftJaccard` 손실을 직접 결합해(시드 42·43) 초반 빈 예측을 줄이고 순간 Jaccard 채점식에 정렬했다.
   - 판정: 대회 규정대로 판정 임계값은 `0.5` 로 고정한다(`force_top1=False`). 2개 모델의 평균 확률(보정 없음)이 `0.5` 이상인 증상을 모두 내며, 없으면 빈 리스트를 낸다.
2. **4-2 증상 분류 및 멈춤 (`ENS3`: E001 + E002 + E007 3모델 평균 + 무누수 학습형 멈춤 정책 `C128_v1_nonempty`).**
   - 분류기: `Macro-F1` 을 유지하는 예선 전체 통화 학습 모델 1개(`E001`) + 앞부분 균등 학습 모델 2개(`E002`, `E007`)의 보정 없는 평균 확률이 `0.5` 이상인 증상을 낸다(1위 강제 없음).
   - 멈춤 조건: 5번째 발화부터 ① **현재 보정 전 평균 확률 `>= 0.5` 인 증상이 최소 1개 이상(`require_nonempty=True`)** 이고, ② **"지금 답하면 받을 기대 점수 + 128 × 더 들으면 얻을 예상 이득 ≥ 0.5"** 이면 멈춘다.
   - 예측 이득(`LightGBM` 회귀)과 기대 점수용 `logit_shift` 는 `dev 2,920`·`Validation 3,640` 을 제외한 순수 `fit 26,280`통화 5겹 OOF(`e002_oof`)만으로 학습했으며, 57개 특징에는 그 순간까지 받은 `text`·`speaker`·`start_ms`·`end_ms` 와 확률만 들어간다.

## 규정 준수

- **입력:** 매 순간 그때까지 받은 발화의 `text`·`speaker`·`start_ms`·`end_ms` 만 쓴다 (`speaker`·시각은 4-2 멈춤 정책의 특징에만 쓴다). 총 발화 수·총 통화 시간·이후 발화·라벨 필드·원본 wav 는 읽지 않는다. 통화 길이는 학습 때 목표(더 들으면 얻을 이득)를 만드는 데만 썼다.
- **판정 임계값:** 0/1 을 가르는 기준은 모두 0.5 다 (4-1 판정, 4-2 멈춤, 멈출 때 판정). 클래스별 임계값이나 Top-1 강제는 없다.
- **학습 데이터:** AI-Hub Training 의 서울 구급 통화 중 `fit 26,280`건(9종 필터 뒤)만 썼다. `Dev 2,920`·`Validation 3,640` 은 가중치 학습·TAPT·`logit_shift`·멈춤 정책 학습 어디에도 쓰지 않았다.
- **사전학습 모델:** `klue/roberta-base` (공개 오픈소스). 상용 API 는 쓰지 않는다.

## 성능

Training 내부 비학습 `Dev 2,920`통화 (`87,814`시점, 공식 `score.py` 경고 `0`건, 16통화 노모킹 오프라인 CLI `inference.py` E2E 검증 통과):

| 구성 | 4-1 (`Dev 2,920`) | 4-2 (`Dev 2,920`) |
|---|---|---|
| 기존 `E026` 기준 (`fit` 재학습 전 / `ENS3`) | 0.5060 | 0.5031 |
| **방안 A 최종 번들 (`SJ42+43` / `ENS3+C128_v1_nonempty`)** | **0.526039** | **0.507411** (`Macro-F1 0.6132 × 조기판단력 0.8275`) |

- 공식 `Validation 3,640`통화 최종 1회 검증은 팀 최종 freeze 합의 후 별도 실행 예정(참고: 이전 단일 `E026` 번들의 Validation 1회 기록은 `4-1 0.4867`, `4-2 0.5070`).

## 다시 학습하기

```bash
python scripts/build_cache.py ...            # Training 라벨 json → runs/cache/train.jsonl
python scripts/tapt_mlm.py ...               # TAPT (또는 run_final_training.sh 의 TAPT_MODE=full)
bash scripts/run_oof_tables.sh               # 5겹 교차 학습 확률표 (멈춤 정책 학습 자료)
python scripts/train_halt_policy.py ... --taus 0.5   # 4-2 멈춤 정책 (기준 0.5 고정, β 만 선택)
bash scripts/run_final_training.sh           # Training 전체로 앞부분 학습 모델 2개
python scripts/pack_bundle.py ...            # 번들 → ckpt/mission4.pt
```

각 명령의 실제 인자와 실행 로그는 `model_train.ipynb` 에 있다.
