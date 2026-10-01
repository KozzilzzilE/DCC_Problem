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

통화당 신고자 조각이 평균 15.8개라 집계 효과가 크다. 실측으로 조각 정확도 0.877~0.897 →
통화 정확도 0.9791~0.9835 (Validation 3,640통화, 결정 임계값 0.5 고정, 갈래별 표 참고).

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

## 제출 모델 선정 — dev 기준, 결정 임계값 0.5 고정

**결정 임계값은 대회 규정으로 0.5 고정이다.** 조각 확률의 통화 평균이 0.5 이상이면 여, 아니면 남.
`m1.models.decision_threshold` 가 항상 0.5 를 돌려주고 제출 경로(`m1.infer`)와 비교표
(`benchmark`/`analysis`/`overlap`)가 이 값을 쓴다. 한때 dev 에서 보정한 임계값(0.515 등)을
체크포인트에 저장해 썼으나 규정에 맞춰 폐기했고, 제출 체크포인트 두 개에서는 저장값 자체를
제거했다 (아래 5절은 연구 기록).

**모델·설정은 Training 내부 dev(2,920통화)로 고르고, Validation 은 보고만 한다.** 규정상
Validation 으로 모델을 골라도 되지만, 그렇게 하면 보고 수치가 낙관적으로 치우치므로 dev 로
고른다. 출제문제 11쪽에 평가는 "주최 측이 제공하는 Validation 또는 Test 입력 폴더" 로 한다고
돼 있어 채점 집합이 미정이므로, 어느 쪽이든 안전한 dev 기준을 택했다.
dev 는 학습 중 매 epoch 찍는 `center`(조각당 창 1개) 값이고, 제출 경로와 같은
`sliding`(창 여러 개 평균) dev 는 두 후보만 따로 쟀다 (아래 문단).

| 모델 | dev (center) | Validation | 오답 |
|---|---|---|---|
| **Wav2Vec2-base** (`w2v2_full.pt`) | **0.9897** | **0.9835** | 60 |
| ResNet50 + SpecAug + logmel/80 (`resnet_aug_m80.pt`) | **0.9866** | 0.9821 | 65 |
| ResNet50 + SpecAug + logmel/80 + 증류 (`resnet_kd.pt`) | 0.9856 | 0.9824 | 64 |
| ResNet50 + SpecAug + MFCC | 0.9866 | 0.9805 | 71 |
| ResNet50 + SpecAug | 0.9856 | 0.9813 | 68 |
| ResNet50 기준 (logmel/64) | 0.9860 | 0.9791 | 76 |
| audeering (전화 음성 사전학습) | 0.9873 | 0.9821 | 65 |

다수결 기준선 0.5382. 전부 임계값 0.5, 같은 시드(1234)·같은 dev 분할.

**제출 1안: `w2v2_full.pt`** — dev +0.31%p, Validation +0.14%p(5통화)로 두 집합에서 같은 방향으로
앞선다. 제출 경로와 같은 sliding dev 에서도 w2v2 0.9897 vs ResNet 0.9863 다.
체크포인트에 HF config 를 동봉해 오프라인 환경에서도 로딩된다 (`HF_HUB_OFFLINE=1` 실증).

**폴백: `resnet_aug_m80.pt`** — dev 기준 ResNet 최고. 94 MB, 추론 8배 빠름. 채점 머신에 GPU 가
없거나 시간 제한이 있으면 이쪽. `--ckpt_path` 만 바꾸면 된다. `resnet_kd.pt` 는 Validation 만
보면 ResNet 최고(0.9824)지만 dev 가 가장 낮아(0.9856) Validation 으로 고른 셈이 되므로 쓰지 않는다.

**단일 시드의 한계.** 모든 갈래를 시드 하나로만 학습했고 반복 실험이 없다. Validation 오답
겹침으로 McNemar 정확검정을 하면 w2v2 는 기준 ResNet 보다 유의하게 낫지만(한쪽만 오답 22 vs 6,
p=0.004), 폴백 `resnet_aug_m80` 과는 분리되지 않는다(17 vs 12, p=0.46). ResNet 변형끼리는 전부
노이즈 범위다. "w2v2 가 폴백보다 낫다" 는 두 집합에서 일관된 방향이라는 근거이지 통계적
확정이 아니다.

### ResNet 끌어올리기 실험에서 배운 것

되찾을 수 있는 건 w2v2 와 겹치지 않는 ResNet 오답 22 건뿐이라 상한은 w2v2 수준이었다.
SpecAugment 는 과적합을 잡았지만(train 0.95 → 0.89) 통화 정확도로 거의 안 옮겨갔고,
MFCC 는 dev 동률·Validation 하락 — dev 2,920 통화가 0.001 차이를 구분하지 못한다.
고정 평균 앙상블(임계값 0.5)은 6 개 모델 어떤 조합도 최고 0.9841 로, w2v2 단독(0.9835)보다
2 통화 앞서지만 노이즈 범위이고 dev 근거가 없어 채택하지 않는다. 6 개 전부 틀리는 통화
44 건(1.21%)이 바닥이다.

## 최종 결과 — 세 갈래 비교 (Validation 3,640통화, 결정 임계값 0.5 고정)

| 갈래 | 통화 Acc | 오답 | 파라미터 | 학습 s/epoch | 추론 ms/통화 | VRAM |
|---|---|---|---|---|---|---|
| ResNet50 (기준) | 0.9791 | 76 | **23.5M** | **326** | **12.6** | **363 MB** |
| **Wav2Vec2-base** | **0.9835** | **60** | 94.4M | 2,708 | 90.5 | 4,224 MB |
| audeering (전화 음성 사전학습) | 0.9821 | 65 | 88.7M | 3,269 | 94.8 | 5,541 MB |

다수결 기준선 0.5382. **세 모델이 모두 틀리는 통화 49개(1.35%)** 가 이 데이터의 오답
바닥이다 — 매 통화마다 맞는 모델을 고를 수 있어도 0.9865 가 상한이고, 고정 평균 앙상블은
어느 조합도 Wav2Vec2 단독을 넘지 못한다 (최고 0.9827; `m1.overlap`, `reports/overlap_*.json`).

**닫힌 개선 경로** (전부 측정으로 확인): 더 긴 학습(과적합), 가중 집계(dev 가 구분 못 함),
앙상블(오류 겹침 74~90%), 전처리(raw waveform 을 보는 갈래가 같은 통화에서 실패),
전화 음성 사전학습(파인튜닝하면 같은 오류로 수렴 — zero-shot oracle 0.9890 → 파인튜닝 후 0.9854).
남은 오답은 결정 경계 근처(w2v2 오답 60 통화의 확신도 중앙 0.20, 절반이 0.2 미만)의 애매한
목소리이고, 아동·구간오류·통화유형·라벨오류 가설은 모두 기각됐다.

위 개선 경로 실험들은 규정 확인 전 dev 보정 임계값으로 돌린 것이라 커밋 메시지·`reports/` 의
당시 수치는 여기 표와 소수점 셋째 자리에서 다를 수 있다. 결론은 0.5 에서도 같다 (이 절의 표와
`reports/comparison.md`, `reports/overlap_*.json` 은 0.5 로 재생성).

## 대회 규칙 대응

| 규칙 | 대응 |
|---|---|
| `startAt`/`endAt`/`speaker`만 사용 | `labels.py`가 그 외 필드를 **반환하지 않는다**. `text`·`symptom`·`address`는 파싱 단계에서 버려져 하위 코드가 접근할 수 없고, 테스트로 고정 |
| Validation 학습 금지 | Training 내부를 **통화 ID 기준** 90/10 분할해 dev 구성. Validation은 최종 리포트에만 사용 |
| label별 상이한 전처리 금지 | 전처리 함수 시그니처가 label을 받지 않는다 (`test_frontend_signature_takes_no_label`) |
| 서울 데이터만 | 제공 데이터 전량이 `Seoul`임을 확인 |
| 상용 API 금지 | ImageNet ResNet50 / Wav2Vec2 공개 체크포인트만 사용 |
| 결정 임계값 0.5 고정 | `m1.models.decision_threshold` 가 항상 0.5 를 반환하고 제출 경로가 이것만 쓴다. 제출 체크포인트에서 보정값 제거. 테스트로 고정 (`test_decision_threshold_is_fixed_at_half_for_submission`, `test_inference_ignores_stored_threshold`) |

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
PYTHONPATH=mission1_gender python -m m1.train --branch w2v2 --cache cache/train --out mission1_gender/ckpt/w2v2_full.pt --epochs 3 --lr 3e-5 --batch-size 32
```

폴백 ResNet (`resnet_aug_m80.pt`):

```bash
PYTHONPATH=mission1_gender python -m m1.train --branch resnet --cache cache/train --out mission1_gender/ckpt/resnet_aug_m80.pt --epochs 6 --spec-augment --n-mels 80
```

위 명령이 이 저장소의 체크포인트를 만든 설정 그대로다 (`ckpt/*.history.json` 의 `train_config`).
lr·weight decay·배치는 탐색하지 않고 고정했다 (ResNet lr 1e-4·배치 64, 16 kHz 갈래 lr 3e-5·배치 32,
AdamW wd 1e-4, cosine, AMP). dev 로 고른 것은 epoch 수(w2v2 3 vs 8)와 갈래별 변형뿐이다.

```bash
PYTHONPATH=mission1_gender python -m m1.train --branch audeering --cache cache/train --out mission1_gender/ckpt/audeering_full.pt --epochs 3 --batch-size 32
```

세 번째 갈래 `audeering` 은 `audeering/wav2vec2-large-robust-6-ft-age-gender` 를 백본으로
쓴다. Fisher/Switchboard **전화 음성**으로 사전학습된 유일한 후보라, 16 kHz 고음질로만
사전학습된 앞의 두 갈래와 오류 프로파일이 다른지 보려고 넣었다 (zero-shot 진단에서
우리 오답 70 통화 중 30 을 맞혔다 — `reports/audeering_diag.json`). 층별 학습 가중치와
latent 시간 마스킹이 들어 있다.

- **라이선스 CC-BY-NC-SA-4.0** (비상업). 이 갈래를 제출하면 문서에 명시해야 한다.
- feature encoder 가 layer-norm 이라 **평가 배치 256 에서 OOM** 난다. `calibrate` /
  `benchmark` / `analysis` 에 `--batch-size 64` 를 줄 것 (학습은 32 로 정상).

### 4. 비교표

```bash
PYTHONPATH=mission1_gender python -m m1.benchmark --ckpt mission1_gender/ckpt/resnet_full.pt --ckpt mission1_gender/ckpt/w2v2_full.pt
```

규정대로 0.5 로 채점한다. 체크포인트에 남아 있는 보정값으로 채점하려면(연구용)
`--use-ckpt-threshold` (`analysis`/`overlap` 도 같은 플래그).

### 5. (연구 기록) 결정 임계값 보정 — 제출에는 쓰지 않는다

```bash
PYTHONPATH=mission1_gender python -m m1.calibrate --ckpt mission1_gender/ckpt/resnet_full.pt --dry-run
```

조각 확률을 통화 단위로 평균하면 0.5 가 최적 경계가 아니다. dev 에서 고른 0.515 를 쓰면
ResNet 기준 Validation 0.9791 -> 0.9808, w2v2 는 0.9835 -> 0.9843 이 된다 (계산 비용 0).
**그러나 대회 규정이 임계값 0.5 고정이라 제출에는 쓰지 않는다.** 제출 경로와 비교표 기본값은
저장값을 무시하고, 제출 체크포인트에서는 값을 제거했다. `--dry-run` 없이 실행하면 체크포인트에
값이 다시 기록되지만 판정에는 여전히 쓰이지 않는다.

같은 이유로 Validation 에서 임계값을 고르는 것도 하지 않았다. Validation 최적값(0.540)은
+0.28%p 로 보이지만 평가 데이터에 맞춘 값이고, dev 에서 고른 값의 실제 이득은 +0.16%p 였다.

### 6. 제출 규격 추론 — 이 폴더만으로 실행

이 폴더(`mission1_gender/`) 안의 `inference.py` 가 미션 단독 진입점이다. 같은 폴더의
`m1` 패키지만 import 하고 다른 미션의 의존성은 쓰지 않는다.

```bash
cd mission1_gender
python inference.py --audio_dir <wav 폴더> --label_dir <json 폴더> --ckpt_path ckpt/w2v2_full.pt --output ./outputs/mission1.csv
```

체크포인트에 `FeatureConfig`·갈래·HF config 가 함께 저장되므로 `--ckpt_path` 만 바꾸면
전처리가 자동으로 복원되고, **인터넷 접속 없이** 로딩된다 (`HF_HUB_OFFLINE=1` 로 실증).
결정 임계값은 체크포인트와 무관하게 0.5 다. 폴백은 `ckpt/resnet_aug_m80.pt`.

리포지터리 루트의 `inference.py` 는 세 미션 공용 진입점이며 같은 함수를 호출한다.

### 7. 제출 직전 체크리스트

`.pt` 는 git 에 없으므로(`.gitignore`) 제출 패키지에 파일을 직접 넣어야 한다. 넣은 그 파일로
아래를 **한 번** 실행하고 첫 줄을 확인한다:

```bash
cd mission1_gender
python inference.py --audio_dir <wav 폴더> --label_dir <json 폴더> --ckpt_path ckpt/<제출할 .pt> --output ./outputs/mission1.csv
```

- 첫 줄 `[Mission 1] branch=... threshold=0.500 (규정 고정) ...` — **0.500 이 아니면 옛 코드**다.
  `ckpt 저장값 ... 은 무시` 가 덧붙으면 보정값이 남은 체크포인트이니 제출본에서는 제거할 것
- 종료 코드 0, `outputs/mission1.csv` 행 수 = 입력 통화 수, 값은 `남`/`여` 만
- w2v2 갈래면 `HF_HUB_OFFLINE=1` 을 붙여 한 번 더 실행해 허브 없이 로딩되는지 확인

### 테스트

```bash
python -m pytest -q
```

## 계산 효율성 (채점 안내 2-8 항목)

| 항목 | 제출 1안 `w2v2_full.pt` | 폴백 `resnet_aug_m80.pt` |
|---|---|---|
| Validation 통화 Acc (임계값 0.5) | **0.9835** | 0.9821 |
| Total 파라미터 | 94.4M | 23.5M |
| Active 파라미터 (추론) | 94.4M — dense 모델, 전 파라미터 사용 | 23.5M |
| 학습 시 trainable | 90.2M (feature encoder 4.2M 동결) | 23.5M |
| 학습·추론 환경 | NVIDIA GeForce RTX 5060 (8 GB), Windows 10, Python 3.14, torch 2.13.0+cu130, AMP | 동일 |
| Validation 추론 batch size | **32** (16 kHz 갈래 기본값, 아래 참고; `inference.py --batch_size` 로 변경 가능) | 128 |
| Validation 전체 추론 시간 (3,640통화) | **221초 (3.7분)** | **27.2초** |
| 샘플(통화)당 평균 | **60.7 ms** | **7.5 ms** |
| 가중치 파일 | 378 MB | 94 MB |

전체 추론 시간은 이 폴더의 `inference.py` 를 `HF_HUB_OFFLINE=1` 로 1회 실행해 잰 벽시계
시간(모델 로딩·wav 디코딩·리샘플·추론·CSV 저장 포함)이다. Validation 3,640 통화, 2026-09-19 실측.

**w2v2 는 배치 크기를 128 로 올리면 8 GB GPU 에서 10배 느려진다 (통화당 616 ms, 31 분).**
16 kHz 창(48,896 샘플) 128 개를 한 번에 넣으면 예약 GPU 메모리가 9.2 GB 로 VRAM 8 GB 를
넘고, Windows WDDM 이 OOM 을 내는 대신 시스템 RAM 으로 조용히 페이징한다. 배치 64 이하
(예약 4.4 GB 이하)면 통화당 58 ms 다. 그래서 `m1/infer.py` 는 16 kHz 갈래 기본 배치를
32 로 둔다 (예약 3.2 GB). 150 통화 표본 프로파일 (CPU 로드·리샘플은 통화당 11 ms 로 병목 아님):

| batch | 통화당 | 예약 GPU 메모리 |
|---|---|---|
| 128 | 616 ms | 9.21 GB (VRAM 초과) |
| 64 | 58 ms | 4.37 GB |
| 32 | 57 ms | 3.17 GB |
| 16 | 59 ms | 1.98 GB |

VRAM 이 8 GB 보다 작은 채점 머신이면 `--batch_size 16` 으로 내린다 (속도 손해 없음).
ResNet 은 8 kHz 멜 입력이라 128 도 안전하다. 시간 제한이 매우 빡빡하면 폴백
`resnet_aug_m80.pt` (정확도 0.9821, 27.2초)를 쓴다.

## 제출 패키징 주의

- `ckpt/*.pt` 는 `.gitignore` 라 **리포지터리에 없다.** 제출 폴더에는 `ckpt/w2v2_full.pt`
  (와 폴백 `ckpt/resnet_aug_m80.pt`)를 직접 넣을 것
- 폴더 구성: `inference.py`, `m1/`, `ckpt/`, `requirements.txt`, `model_train.ipynb`, `README.md`
- 넣은 그 `.pt` 로 위 체크리스트를 한 번 실행해 첫 줄이 `threshold=0.500 (규정 고정)` 이고
  `저장값 ... 무시` 문구가 없는지 확인 (제출 체크포인트 두 개는 보정값을 제거해 둔 상태)

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
