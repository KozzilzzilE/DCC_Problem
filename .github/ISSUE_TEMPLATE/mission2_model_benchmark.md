---
name: Mission 2 모델 벤치마크 및 성능 비교
about: 미션 2 화자 분류를 위한 다중 모델 벤치마크 결과 및 비교 분석 기록
title: "[Mission 2] 다중 모델 벤치마크 성능 비교: "
labels: "mission2, benchmark, enhancement"
assignees: ""
---

## 실험 목적
- 미션 2(신고자 vs 119대원 음성 화자 분류)에 대해 여러 아키텍처 접근법을 동일 조건에서 비교 평가합니다.
- 단순 정확도뿐만 아니라 파라미터 수, 추론 지연시간(Latency, ms), 메모리 사용량을 다차원 분석합니다.

## 비교 대상 모델 체크리스트
- [ ] ReDimNet2-B2 (3.6M, Hybrid 2D+1D Conv + Multi-Head Attention)
- [ ] ECAPA-TDNN (6.1M, 1D CNN + Attentive Statistics Pooling)
- [ ] CAM++ (7.2M, Dense TDNN + Context-Aware Masking)
- [ ] AudioResNet-50 (23.5M, 2D CNN ImageNet 가중치 평균 변환)
- [ ] HuBERT-Base (95.0M, Self-Supervised Waveform Transformer)
- [ ] SSAST-Tiny (6.0M, Patch-based Spectrogram Vision Transformer)
- [ ] Soft Voting 앙상블 (상위 모델 가중 결합)

## 벤치마크 결과 비교표
| 모델명 | 카테고리 | Val Accuracy (%) | Macro F1 | 파라미터 (M) | 지연시간 (ms/sample) | 학습 시간 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| ReDimNet2-B2 | Hybrid | | | 3.6 | | |
| ECAPA-TDNN | 1D CNN | | | 6.1 | | |
| CAM++ | Dense TDNN | | | 7.2 | | |
| AudioResNet-50 | 2D CNN | | | 23.5 | | |
| HuBERT-Base | Transformer | | | 95.0 | | |
| SSAST-Tiny | Audio ViT | | | 6.0 | | |
| Ensemble | Soft Voting | | | - | | |

## 결과 분석 및 시사점
- 최고 성능 모델 및 선정 사유:
- 경량화 및 효율성(속도/메모리) 관점 분석:
- 오분류 주요 원인(짧은 발화, 배경 소음 등):
- 최종 제출 가중치(best_model.pt) 채택 여부:
