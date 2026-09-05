# Mission 1 — 신고자 성별 분류 (CNN / Vision 접근)

담당: 김승윤

음성으로부터 신고자의 성별(남/여)을 분류한다. 같은 전처리 캐시 위에 **2D CNN
(ResNet50)** 갈래와 **음성 특화 파인튜닝(Wav2Vec2)** 갈래를 올려 정확도와 속도를
비교한다.

## 접근 — 2단 구조

정답 `gender`는 통화 단위 라벨인데, 통화 음성에는 신고자와 119대원이 섞여 있다
(평균 72.5초). 규칙상 추론 시에도 `startAt`/`endAt`/`speaker`를 쓸 수 있으므로:

1. **조각 단위 학습** — `speaker == 1`(신고자) 발화 구간만 잘라, 각 조각에 그 통화의
   성별을 라벨로 붙여 이진 분류기를 학습
2. **통화 단위 집계** — 한 통화의 조각별 확률을 평균(soft voting)해 남/여 결정

통화당 신고자 조각이 평균 15.8개라 집계 효과가 크다. 실측으로 조각 정확도 0.877 →
통화 정확도 0.979 (Validation 3,640통화, ResNet50 기준).

## 데이터 실측

| 항목 | 값 |
|---|---|
| WAV 포맷 | **8 kHz mono 16-bit** (16 kHz 아님) |
| 통화 길이 | 평균 72.5초 |
| 신고자 발화 | 통화당 평균 15.8개 / 33.5초 (최소 1개) |
| 조각 길이 | 평균 2.12초, p50 1.49초, p90 4.66초, max 24.2초 |
| 전체 규모 | Training 29,200통화 → 462,190조각 (15.67 GB 캐시) |
| Validation | 3,640통화 → 58,101조각 |
| 다수결 기준선 | Validation 통화 단위 **0.538** |

## 대회 규칙 대응

| 규칙 | 대응 |
|---|---|
| `startAt`/`endAt`/`speaker`만 사용 | `labels.py`가 그 외 필드를 **반환하지 않는다**. `text`·`symptom`·`address`는 파싱 단계에서 버려져 하위 코드가 접근할 수 없고, 테스트로 고정 |
| Validation 학습 금지 | Training 내부를 **통화 ID 기준** 90/10 분할해 dev 구성. Validation은 최종 리포트에만 사용 |
| label별 상이한 전처리 금지 | 전처리 함수 시그니처가 label을 받지 않는다 (`test_frontend_signature_takes_no_label`) |
| 서울 데이터만 | 제공 데이터 전량이 `Seoul`임을 확인 |
| 상용 API 금지 | ImageNet ResNet50 / Wav2Vec2 공개 체크포인트만 사용 |

## 구조

```
m1/
  labels.py      라벨 JSON -> 허용 필드만 담은 frozen dataclass
  cache.py       통화당 wav 1회 읽기 -> 신고자 조각 8kHz int16 캐시
  features.py    torch.stft 기반 log-Mel / MFCC (librosa와 수치 대조 검증)
  datasets.py    캐시 위의 Dataset (두 갈래 공용) + sliding window
  models/        resnet.py / w2v2.py / factory.py (체크포인트 입출력)
  aggregate.py   조각 확률 -> 통화 라벨 (soft voting)
  train.py       학습 CLI (--branch resnet|w2v2)
  evaluate.py    통화 단위 accuracy / 혼동행렬
  benchmark.py   갈래 비교표 생성
  infer.py       캐시 없이 wav/json 폴더에서 바로 추론
```

### 왜 캐시를 두는가

조각 하나를 꺼낼 때마다 통화 전체 wav를 다시 디코딩하면, 46만 조각 × 약 1.1 MB의
읽기가 매 epoch 반복된다. 통화당 wav를 한 번만 읽어 신고자 조각을 미리 잘라두면
그 비용이 사라진다. 전체 29,200통화 캐시 빌드에 **92초**, 결과물 15.67 GB.

원본 8 kHz를 그대로 보관하므로 CNN 갈래(스펙트로그램)와 Wav2Vec2 갈래(16 kHz
업샘플)가 같은 캐시를 공유한다. 두 갈래의 속도 차이가 I/O가 아니라 모델에서
나오게 하려는 의도다.

### 왜 torchaudio를 안 쓰는가

설치된 `torch 2.13.0+cu130`에 맞는 torchaudio 빌드가 없다 (cu130 채널 최대 2.11).
그래서 `torch.stft` + 자체 mel 필터뱅크로 직접 구현했고, librosa와 수치가 일치함을
테스트에서 대조 검증한다 (`test_mel_filterbank_matches_librosa`,
`test_logmel_matches_librosa_melspectrogram`).

## 실행

### 1. 데이터 배치

`data/train/{audio,label}`, `data/val/{audio,label}`에 wav/json을 둔다
(파일명 stem이 통화 ID로 일치해야 한다).

### 2. 캐시 빌드

```bash
python -c "import sys;sys.path.insert(0,'mission1_gender');from m1.cache import build_cache,default_workers;build_cache('data/train/label','data/train/audio','cache/train',workers=default_workers(),progress=True)"
```

### 3. 학습

```bash
PYTHONPATH=mission1_gender python -m m1.train --branch resnet --cache cache/train --out mission1_gender/ckpt/resnet_full.pt --epochs 6
```

```bash
PYTHONPATH=mission1_gender python -m m1.train --branch w2v2 --cache cache/train --out mission1_gender/ckpt/w2v2_full.pt --epochs 3 --batch-size 16
```

### 4. 비교표

```bash
PYTHONPATH=mission1_gender python -m m1.benchmark --ckpt mission1_gender/ckpt/resnet_full.pt --ckpt mission1_gender/ckpt/w2v2_full.pt
```

### 5. 제출 규격 추론

```bash
python inference.py --audio_dir ./data/val/audio --label_dir ./data/val/label --ckpt_path ./mission1_gender/ckpt/resnet_full.pt --output ./outputs/mission1.csv
```

체크포인트에 `FeatureConfig`와 갈래 이름이 함께 저장되므로, `--ckpt_path`만 바꾸면
전처리가 자동으로 그에 맞게 복원된다.

### 테스트

```bash
python -m pytest -q
```

## 환경 주의사항

**GPU 스택은 `requirements.txt`에서 버전을 고정하지 않는다.** 원래 `torch==2.7.1+cu118`로
고정돼 있었는데, RTX 5060은 Blackwell(sm_120)이라 그 조합은 설치조차 되지 않는다
(`No matching distribution found`). 머신마다 GPU가 다르므로 torch/torchvision은 각자
자기 GPU에 맞는 인덱스에서 먼저 설치한 뒤 `requirements.txt`를 적용한다 — 설치 명령은
`requirements.txt` 안에 적어 두었다. 이 작업은 `torch 2.13.0+cu130` +
`torchvision 0.28.0+cu130`으로 진행했다.

**한국어 base 크기 Wav2Vec2는 공개된 것이 없다.** 한국어는 large(24층)만 있어
(`kresnik/wav2vec2-large-xlsr-korean`), 8 GB VRAM과 통제된 비교를 고려해 기본
비교 대상은 `facebook/wav2vec2-base`로 두었다. 성별 판별의 단서는 F0·포먼트 같은
음향 특성이라 언어 의존도가 낮다. 한국어 large는 `--w2v2-model`로 지정해 추가
실험할 수 있다.
