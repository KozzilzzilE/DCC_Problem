---
name: Mission 2 단일 모델 분석 및 튜닝
about: 특정 모델의 아키텍처 튜닝, 오분류 분석 및 하이퍼파라미터 최적화
title: "[Mission 2 / Model] "
labels: "mission2, model-tuning"
assignees: ""
---

## 대상 모델
- 모델명: (예: ReDimNet2-B2 / ECAPA-TDNN / HuBERT-Base 등)
- 입력 형태: (Mel-Spectrogram / Log Mel-Filterbank / Raw Waveform)
- 파라미터 수: 

## 아키텍처 및 전처리 설정
- 전처리 파라미터 (Window, Hop, FFT, Mels 등):
- 학습 하이퍼파라미터 (LR, Batch Size, Optimizer, Weight Decay):

## 검증 성능
- Validation Accuracy:
- Macro F1:
- Precision / Recall:
- 평균 추론 지연시간 (ms):

## 혼동 행렬 및 오분류 패턴 분석
- 신고자(1)를 119대원(0)으로 오분류한 케이스 특징:
- 119대원(0)을 신고자(1)로 오분류한 케이스 특징:
- 개선 방안 및 앙상블 기여 가능성:
