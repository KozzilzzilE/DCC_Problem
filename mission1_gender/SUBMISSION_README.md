# Mission 1 — 신고자 성별 분류 (예선 제출본)

119 신고 통화 음성에서 신고자의 성별(남/여)을 통화 단위로 분류한다.

- **제출 모델 `ckpt/w2v2_full.pt` (Wav2Vec2-base 파인튜닝): Validation 통화 정확도 0.9835** (3,640통화, 이 폴더의 `inference.py` 로 측정)
- 결정 임계값은 **0.5 고정**이다 (대회 규정). 통화의 신고자 조각 확률 평균이 0.5 이상이면 `F`, 미만이면 `M`.

## 폴더 구성

```text
mission1_gender/
├── inference.py        # 추론 진입점 — 한 번 실행으로 결과 CSV 생성
├── m1/                 # 라벨 파싱·오디오 전처리·모델·학습·추론 모듈
│   └── models/         # w2v2.py(제출 모델), resnet.py(비교 실험), factory.py(모델 생성·체크포인트 입출력)
├── ckpt/               # 가중치 (.pt, HF config 동봉 — 인터넷 불필요)
│   └── w2v2_full.pt    (+ w2v2_full.history.json: 학습 로그)
├── model_train.ipynb   # 학습 과정·로그·Validation 평가 기록
├── reports/            # 모델 비교 결과 (모두 임계값 0.5 기준)
├── requirements.txt
└── README.md
```

## 설치

Python 3.12 이상 (검증 3.14.6). 이 폴더에서:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 는 PyTorch CUDA 13.0 빌드(torch 2.13.0+cu130, torchvision 0.28.0+cu130)를 받는다. NVIDIA 드라이버 580 이상이 필요하며, 드라이버가 더 오래됐으면 파일 안의 `cu130` 을 `cu126` 으로 바꾼다.

## 추론 (주최 측 명령 형식)

이 폴더에서:

```bash
python inference.py --audio_dir <wav 폴더> --label_dir <json 폴더> --ckpt_path ckpt/w2v2_full.pt --output ./outputs/mission1.csv
```

- 출력 CSV: `audio file name`, `gender`. 값은 라벨 원값 `M`/`F` 를 그대로 쓴다 (출제 PDF 7쪽 '라벨링데이터의 gender'). 입력 통화마다 한 행이다.
- 실행 첫 줄에 `threshold=0.500 (규정 고정)` 이 찍힌다.
- 사전학습 모델의 config 와 가중치가 `.pt` 안에 들어 있어 추론 시 인터넷에 접속하지 않는다.

## 계산 효율

| 항목 | 제출 `w2v2_full.pt` |
|---|---|
| Total 파라미터 | 94,372,481 (94.4M) |
| Active 파라미터 | 94.4M (dense, 전 파라미터 사용) |
| 학습 시 trainable | 90.2M (feature encoder 4.2M 동결) |
| 가중치 파일 | 378 MB |
| 학습·추론 환경 | NVIDIA GeForce RTX 5060 8GB (드라이버 610.62, CUDA 13.0), AMD Ryzen 5 9600 (6코어 12스레드), RAM 31GB, Windows 10, Python 3.14.6, torch 2.13.0+cu130, AMP |
| Validation 추론 batch size | 32 |
| Validation 전체 추론 시간 | 3,640통화 **약 220~252초 (통화당 61~69 ms, 09-29·10-02 실측 4회)**, 모델 로딩·wav 디코딩·리샘플·CSV 저장 포함 |
| 학습 시간 | 3 epoch 약 2.3시간 (epoch 당 약 45분) |

추론 시간은 이 폴더의 `inference.py` 를 인터넷 차단(`HF_HUB_OFFLINE=1`) 상태로 실행해 잰 벽시계 시간이다.
w2v2 배치를 128 로 올리면 8 GB GPU 에서 VRAM 을 넘겨 Windows 가 시스템 RAM 으로 페이징하므로 약 10배 느려진다.
그래서 기본 배치를 32 로 뒀다 (예약 GPU 메모리 약 3.2 GB). VRAM 이 8 GB 보다 작으면 `--batch_size 16` 을 준다 (속도 차이 없음).

## 방법

1. **조각 단위 학습**: 라벨 JSON 의 `speaker == 1`(신고자) 발화 구간(`startAt`/`endAt`)만 잘라 각 조각에 통화의 성별을 붙여 이진 분류기를 학습한다. 원본 8 kHz 음성을 16 kHz 로 올려 Wav2Vec2-base(`facebook/wav2vec2-base`)를 파인튜닝했다.
2. **통화 단위 집계**: 한 통화의 조각별 확률을 평균(soft voting)해 0.5 로 판정한다. 통화당 신고자 조각이 평균 15.8개라 조각 오류가 상쇄된다 (조각 정확도 약 0.88~0.90 → 통화 정확도 0.98대).
3. **모델 선택**: Training 폴더만 통화 ID 기준 90/10 으로 나눈 dev 로 골랐고, Validation 은 결과 보고에만 썼다.

## 제출 모델 선정 (dev 기준, 임계값 0.5)

dev(center) 는 학습 중 매 epoch 찍는 값(조각당 창 1개)이고, dev(sliding) 은 제출 경로와 같은 방식(창 여러 개 평균)으로 두 후보만 따로 잰 값이다.

| 모델 (임계값 0.5) | dev (center) | dev (sliding) | Validation |
|---|---:|---:|---:|
| **Wav2Vec2-base (`w2v2_full.pt`, 제출)** | **0.9897** | **0.9897** | **0.9835** |
| ResNet50 + SpecAug + log-Mel 80 (비교 실험) | 0.9866 | 0.9863 | 0.9821 |
| ResNet50 기준, log-Mel 64 (비교 실험) | 0.9860 | — | 0.9791 |

다수결 기준선 0.538. Wav2Vec2-base 가 ResNet50 + SpecAug + log-Mel 80 대비 dev(center) +0.31%p, Validation +0.14%p(5통화)로 두 집합에서 같은 방향으로 앞서고, 제출 경로와 같은 sliding dev 에서도 0.9897 vs 0.9863 이다.
학습 로그는 `model_train.ipynb` 와 `ckpt/w2v2_full.history.json`, 모델 비교는 `reports/comparison.md` 에 있다.
`reports/` 는 출력 표기를 `M`/`F` 로 바꾸기 전에 만든 기록이라 혼동행렬 키가 남/여(`'남->여'` 는 정답 M, 예측 F)다.
그 표의 `audeering_full` 행은 제출 정리 때 코드에서 뺀 비교 실험(전화 음성 사전학습 `audeering/wav2vec2-large-robust-6-ft-age-gender`, 라이선스 CC-BY-NC-SA-4.0)이다.

**단일 시드의 한계.** 모든 모델을 시드 하나(1234)로만 학습했고 반복 실험이 없다. Validation 오답 겹침으로 McNemar 정확검정을 하면 Wav2Vec2-base 는 기준 ResNet50 보다 유의하게 낫지만(한쪽만 오답 22 vs 6, p=0.004), ResNet50 + SpecAug + log-Mel 80 과는 분리되지 않는다(17 vs 12, p=0.46). 따라서 ResNet50 + SpecAug + log-Mel 80 보다 낫다는 것은 dev·Validation 두 집합에서 일관된 방향이라는 근거이지 통계적 확정이 아니다.

## 오류 분석

- 고정 평균 앙상블(임계값 0.5)은 비교 실험 6개 모델(ResNet50 변형 5개 + Wav2Vec2-base)의 어떤 조합도 최고 0.9841 로, Wav2Vec2-base 단독(0.9835)보다 2통화 앞서지만 노이즈 범위이고 dev 근거가 없어 채택하지 않았다. 6개 모델이 모두 틀리는 통화 44건(1.21%)이 바닥이다 (`reports/overlap_resnet_full_resnet_aug_resnet_aug_m80_resnet_aug_mfcc_resnet_kd_w2v2_full.json`).
- 남은 오답은 결정 경계 근처(Wav2Vec2-base 오답 60통화의 확신도 중앙 0.20, 절반이 0.2 미만)의 애매한 목소리이고, 아동·구간오류·통화유형·라벨오류 가설은 모두 기각됐다.

## 학습 재현

이 폴더에서 실행한다. 시작점 `facebook/wav2vec2-base`(공개 모델)는 첫 학습 때 한 번 내려받는다.

```bash
python -c "from m1.cache import build_cache, default_workers; build_cache('<data>/train/label', '<data>/train/audio', 'cache/train', workers=default_workers(), progress=True)"
python -m m1.train --branch w2v2 --cache cache/train --out ckpt/w2v2_full.pt --epochs 3 --lr 3e-5 --batch-size 32 --num-workers 6
```

캐시는 Training 신고자 조각(462,190개)을 미리 잘라 둔 것이다 (약 15.7 GB).

## 규정 준수

- 추론·학습 입력은 음성과 라벨의 `startAt`/`endAt`/`speaker` 뿐이다. 그 외 필드는 `m1/labels.py` 가 파싱 단계에서 버린다.
- 학습·모델 선택에는 Training(서울) 데이터만 썼다. Validation 은 결과 보고에만 썼다.
- 결정 임계값은 0.5 고정이다 (`m1/models/factory.py` 의 `DEFAULT_THRESHOLD`, 제출 체크포인트에는 다른 임계값이 저장돼 있지 않다).
- 공개 사전학습 모델(facebook/wav2vec2-base)만 썼고, 모델 학습·추론에 상용 API 는 쓰지 않았다.
