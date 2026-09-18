# [Issue / Model] WavLM-Base+ 화자 인식 및 노이즈 강건성 특화 파인튜닝

## 1. 모델 개요
- 모델명: WavLM-Base+ (`microsoft/wavlm-base-plus`)
- 카테고리: Self-Supervised Speech Representation (화자 구별 및 노이즈 제거 특화)
- 파라미터 수: 약 95.0M (트랜스포머 기반 대규모 모델)
- 제작사: Microsoft
- 선정 근거:
  - HuBERT의 후속 모델로서 음향 마스킹 토큰 복원뿐만 아니라 **화자 구별(Speaker Verification) 목적함수**와 **노이즈 시뮬레이션(Gated Relative Position Bias)**이 사전학습에 직접 내장됨.
  - 119 신고 통화 특유의 열악한 음향 환경(사이렌, 배경 소음, 전화 회선 노이즈) 및 신고자의 다급한 음성과 접수원의 정돈된 음성을 구분하는 데 현존 SSL 모델 중 가장 적합.

## 2. 입력 및 전처리 프로토콜
- 입력 특징: 정규화된 1D Raw Waveform (48,000 samples = 3.0초 윈도우)
- 데이터 증강: `VerifiedSpeechDataset` 내 가우시안 백색 잡음 및 진폭(Gain) 변조 적용
- 최적화 전략:
  - 1D CNN Feature Extractor는 고정(`freeze_cnn=True`)하여 VRAM 절약 및 수렴 속도 극대화
  - 트랜스포머 백본(lr=2e-5)과 분류 헤드(lr=3e-4)에 차등 학습률(Discriminative Learning Rate) 적용
  - CosineAnnealingLR 스케줄러 적용 (5 에포크)
- 가중치 저장 경로: `/content/drive/MyDrive/DCC/benchmark_results/checkpoints/best_WavLM.pt`

## 3. 실험 목표 및 가설
- 가설: 일반 음성 인식 중심의 HuBERT나 Wav2Vec 2.0보다 화자 구별 태스크에서 더 높은 단일 판별력(90~92%)을 발휘할 것이며, 시각적 2D ResNet-50과의 앙상블 결합 시 92% 이상의 최종 정확도를 달성할 것이다.
- 목표 정확도: 단일 모델 90.0% ~ 92.0%+, 앙상블 결합 시 92.5%+

## 4. 진행 현황
- **현황**: `Multi_Model_Benchmark.ipynb` 내 [Step 7-2] 파인튜닝 셀 및 [Step 10] 다중 앙상블 비교 파이프라인 구현 완료. 사용자의 Colab 학습 실행 대기 중.
