# [Issue / Model] HuBERT-Base 대규모 음향 자기지도학습 파인튜닝

## 1. 모델 개요
- 모델명: HuBERT-Base (facebook/hubert-base-ls960)
- 카테고리: Self-Supervised Waveform Transformer
- 파라미터 수: 약 95.0M (트랜스포머 기반 대규모 모델)
- 선정 근거: 스펙트로그램 변환 없이 1D 원본 파형에서 직접 학습된 음향 토큰 마스킹 사전학습 SOTA 모델.

## 2. 입력 및 전처리 프로토콜
- 입력 특징: 정규화된 1D Raw Waveform (48,000 samples = 3.0초)
- 전략: CNN Feature Extractor는 Freeze하고, 트랜스포머 상위 레이어 및 분류 헤드만 파인튜닝하여 VRAM 절약.
- 배치 크기: 16 (GPU 메모리 최적화)

## 3. 실험 목표 및 가설
- 가설: 음성 표현력이 가장 뛰어나 화자 간 미세한 음향적 차이(다급함, 떨림 vs 차분함)를 가장 정밀하게 포착하여 단일 모델 최고 정확도를 기록할 가능성이 높으나, 추론 지연시간이 길 수 있다.
- 목표 정확도: 91.0% ~ 93.0%+
