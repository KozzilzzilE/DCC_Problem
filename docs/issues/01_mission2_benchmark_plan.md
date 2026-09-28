# [Issue] Mission 2 지능형 화자 분류 다중 모델 벤치마크 총괄 계획

## 1. 배경 및 목표
- 미션 2는 119 신고 통화 음성 조각(`startAt` ~ `endAt`)만으로 **신고자(1) vs 119대원(0)**을 판별하는 이진 분류 과제입니다.
- 단일 모델(ResNet-50)에 안주하지 않고, 화자 인식 분석 보고서(`speaker_diarization_analysis.md`) 및 글로벌 SOTA 아키텍처를 기반으로 **6개 모델의 정확도, 지연시간, 파라미터 수**를 종합 비교하는 벤치마크를 추진합니다.

## 2. 세부 태스크
- [x] 모델별 최적 전처리 및 하이퍼파라미터 정의 (`config.py`)
- [x] Mel-Spec, Filterbank, Waveform 3종 입력 지원 통합 데이터셋 구축 (`dataset.py`)
- [x] 6대 모델 래퍼 모듈 구현 (`models/`)
- [x] 무중단 안전장치(구글 드라이브 실시간 동기화 + Auto-Resume) 탑재 트레이너 구축 (`trainer.py`)
- [x] 다중 모델 Soft Voting 앙상블 모듈 구현 (`ensemble.py`)
- [x] Colab Pro 고속 학습 노트북 생성 (`Multi_Model_Benchmark.ipynb`)

## 3. 평가 지표 및 비교 기준
- 평가 지표: Accuracy (공식 평가 기준), Macro F1, Precision, Recall
- 효율성 지표: 단일 샘플 추론 지연시간 (ms/sample), 파라미터 수 (Million), 학습 소요 시간
- 산출물: `benchmark_results.csv`, 모델별 성능 비교 시각화 차트, 최종 `best_model.pt`
