# [Issue / Model] ReDimNet2-B2 초경량 하이브리드 화자 모델 벤치마크

## 1. 모델 개요
- 모델명: ReDimNet2-B2
- 카테고리: Hybrid (2D Conv + 1D Dilated Conv + Multi-Head Attention)
- 파라미터 수: 약 3.6M (전체 비교군 중 최소 크기)
- 선정 근거: 분석 보고서(`speaker_diarization_analysis.md`) 1순위 추천 모델. VoxCeleb에서 입증된 경량화 및 SOTA급 화자 분별력.

## 2. 입력 및 전처리 프로토콜
- 입력 특징: Log Mel-Filterbank (80 Mels, n_fft=512, hop_length=160)
- 정규화: Cepstral Mean and Variance Normalization (CMVN)
- 윈도우 길이: 3.0초 (부족분은 zero-padding 방어)

## 3. 실험 목표 및 가설
- 가설: 2D Conv로 Pitch/Formant 등 국소 성도 단서를 보존하고, MHA Time Pooling으로 화자 표현을 안정적으로 집계하여 ResNet-50(23.5M) 대비 1/6 이하의 파라미터로 동등 이상의 성능을 달성할 것이다.
- 목표 정확도: 88.0% ~ 91.0%+
- 목표 지연시간: 15ms 이하
