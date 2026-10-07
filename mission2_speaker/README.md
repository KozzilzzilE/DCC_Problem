# Mission 2 119 긴급 통화 화자 분류

음성 발화 구간으로 상황실 접수요원 0과 신고자 1을 분류한다. 최종 설정은 세 모델 확률의 균등 평균과 고정 임계값 0.50이다.

보존된 학습 노트북 로그는 Validation **111,919건에서 정확도 92.48%, Macro F1 0.9244**를 보고한다. 2026년 10월 8일에 학습 및 제출 전처리 차이를 수정했으며, 이 수치는 수정된 추론 엔진을 새로 실행한 결과가 아니다. 원본 데이터에서 성능과 표본 수를 다시 확인해야 한다.

## 실행

미션 폴더에서 다음 명령을 실행한다.

```bash
pip install -r requirements.txt
python inference.py --audio_dir ../data/val/audio --label_dir ../data/val/label --ckpt_path checkpoints/ --output ../outputs/mission2.csv
```

`--ckpt_path`에 세 가중치가 있는 폴더를 주면 앙상블을, 특정 `.pt` 파일을 주면 그 모델 하나를 사용한다. 세 checkpoint 파일은 `best_redimnet.pt`, `best_ecapa_tdnn.pt`, `best_resnet50.pt`이다. 출력 CSV의 열은 `audio file name`, `startAt`, `endAt`, `speaker`다. `startAt`과 `endAt`은 밀리초이며 화자 판정에 텍스트나 발화 순서를 사용하지 않는다.

## 전처리

입력은 16kHz 모노 3초로 맞춘다. 짧으면 뒤쪽 zero padding, 길면 중앙 crop을 사용한다. ReDimNet 계열 로컬 구현과 ECAPA는 Mel 80 / FFT 512 / hop 160, ResNet은 Mel 128 / FFT 2048 / hop 512다. 상대 dB를 `(mel_db + 80) / 80`으로 변환하고 0~1로 제한한다.

## 과거 실험과 최종 설정의 차이

Git bf38dd1의 가중 앙상블은 ECAPA 0.45 / ReDim 0.40 / ResNet 0.15로, 임계값 0.50에서 92.57%, 0.51에서 92.58% 및 Macro F1 0.9254였다. 그 실행의 임계값 0.50 Macro F1은 stdout에 출력되지 않았다. 최종 선택은 이후 Git 3638dab의 균등 평균과 임계값 0.50이다.

Git 2c6cb4a의 3에폭 FT는 ReDim 91.57%, ECAPA 91.82%, ResNet 90.70%로 완료됐다. 이후 ReDim 추가 실행은 8에폭 완료 후 9에폭 학습 중단, 최고 91.58%였다. 초기 Wav2Vec2는 Git d6cdee7의 부분 검증 9,138건에서 89.72% / Macro F1 0.8965를 기록했다.

## 기존 평가와 체크포인트

| 모델 | 기록 정확도 | Macro F1 | 파라미터 | 파일 크기 MiB |
|---|---|---|---|---|
| ReDimNet 계열 로컬 구현 | 91.89% | 0.9185 | 2,567,489 | 9.84 |
| ECAPA-TDNN 로컬 구현 | 92.10% | 0.9206 | 5,795,265 | 22.23 |
| AudioResNet-50 | 91.20% | 0.9115 | 23,503,809 | 89.97 |
| 균등 평균 앙상블 임계값 0.50 | 92.48% | 0.9244 | 31,866,563 | 122.04 |

`ReDimNet2_B2`는 로컬 클래스 이름이며 공식 ReDimNet2-B2의 구조·사전학습 가중치를 재현했다는 의미가 아니다. 파일 크기는 1MiB = 1,048,576 bytes다. 파라미터 수는 체크포인트 shape에서 BatchNorm running 통계와 counter를 제외한 계산이다.

## 효율성과 재현성

기존 합성 입력 벤치마크의 forward 지연시간은 1.73 / 5.12 / 4.65ms다. 시간축 300프레임의 무작위 tensor를 썼으며 오디오 읽기, 리샘플링과 Mel 계산이 제외되었다. 실제 ResNet 3초 입력은 94프레임이다. 앙상블 11.50ms, 260~350건/초, Validation 7.2분을 종단간 실측치로 주장하지 않는다. 제출 엔진은 발화별로 실행하며 batch 64 추론을 구현하지 않는다.

노트북에는 Git 3638dab의 기존 실행 출력을 복원했다. 최초 분류표의 클래스 이름이 반대로 기재되어 있었으므로 첫 행은 상황실 0, 둘째 행은 신고자 1로 읽는다. 수정된 코드는 다음 실행에서 이름을 올바르게 출력하고 확률 배열과 발화 목록을 `reports/`에 저장한다. 저장된 과거 출력은 수정된 코드의 신규 실행 결과가 아니다.

자세한 근거와 제한은 [최종 보고서](../docs/FINAL_EXPERIMENT_REPORT.md)에 정리되어 있다. 이번 검토 환경에는 PyTorch/librosa와 원본 데이터가 없어 학습·전수 추론 및 실제 가중치 로딩을 새로 검증하지 않았다.
