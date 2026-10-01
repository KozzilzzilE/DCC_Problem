# [Issue / Model] SSAST-Tiny 경량 오디오 Vision Transformer 벤치마크

## 1. 모델 개요
- 모델명: SSAST-Tiny (Spectrogram Audio Vision Transformer)
- 카테고리: Patch-based Spectrogram ViT
- 파라미터 수: 약 6.0M
- 선정 근거: 보고서(`speaker_diarization_analysis.md`)에서 "CNN vs ViT" 가설을 객관적으로 검증하기 위해 제시된 경량 ViT 대조군.

## 2. 입력 및 전처리 프로토콜
- 입력 특징: Mel-Spectrogram (128 Mels, 비대칭 패치 임베딩 16x16)
- 윈도우 길이: 3.0초

## 3. 실험 목표 및 가설
- 가설: 음성 도메인에서 사전학습 없이 단순히 ViT 구조만 적용할 경우, 국소적 음향 단서(Formant)를 학습하는 데 CNN 대비 더 많은 데이터가 필요하여 정확도나 수렴 속도가 뒤처질 가능성을 검증한다.
