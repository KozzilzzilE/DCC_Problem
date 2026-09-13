# [Issue / Model] ECAPA-TDNN 화자 인식 표준 모델 벤치마크

## 1. 모델 개요
- 모델명: ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)
- 카테고리: 1D CNN / Speaker Verification Standard
- 파라미터 수: 약 6.1M
- 선정 근거: 화자 인식 챌린지(VoxSRC)의 사실상 표준 베이스라인. Res2Net 기반 멀티스케일 필터와 Squeeze-and-Excitation 채널 어텐션, 통계적 어텐션 풀링을 채택함.

## 2. 입력 및 전처리 프로토콜
- 입력 특징: Log Mel-Filterbank (80 Mels, n_fft=512, hop_length=160)
- 풀링 기법: Attentive Statistics Pooling (평균 + 표준편차 어텐션)
- 윈도우 길이: 3.0초 (Zero-padding)

## 3. 실험 목표 및 가설
- 가설: 화자 통계적 특성(Mean, Std)을 어텐션 기반으로 요약하므로 통화 음질 편차나 마이크 특성 차이에 가장 강건하게 대응할 것이다.
- 목표 정확도: 89.0% ~ 92.0%
