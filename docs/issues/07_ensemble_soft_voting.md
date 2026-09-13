# [Issue / Ensemble] 이종 아키텍처 결합 Soft Voting 앙상블

## 1. 개요
- 목표: 서로 다른 도메인 사전학습 지식(ImageNet CNN + 화자 특화 1D CNN + 자기지도 트랜스포머)을 가진 상위 모델들의 예측 확률을 가중 결합하여 단일 모델 대비 +1.5~2.5%p 성능 향상 도모.
- 결합 대상 후보: ReDimNet2-B2, ECAPA-TDNN, AudioResNet-50, HuBERT-Base

## 2. 앙상블 알고리즘
- Soft Voting (Weighted Average):
  `P_final = w1 * P_redimnet + w2 * P_ecapa + w3 * P_resnet + w4 * P_hubert`
- 최적 가중치(w)는 검증 데이터셋에서 Grid Search를 통해 탐색.

## 3. 성공 기준
- 단일 최고 모델 대비 Macro F1 및 Validation Accuracy 유의미한 상승 확인.
- 최종 제출용 `best_model.pt` 및 추론 가중치 확정.
