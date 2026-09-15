# [Issue / Integration] inference.py에 Mission 2 추론 로직 연동

## 1. 개요
- 목표: 대회 제출 규격에 맞춰 `inference.py`의 `mission2_inference()` 함수를 최종 가중치와 연동
- 현황: 현재는 단일 ResNet-50 하드코딩 상태이며, 최종 앙상블 조합이나 남은 모델(CAM++, HuBERT 등)의 학습이 끝난 후 최고 성능의 모델로 대체해야 함.

## 2. 작업 내용
- [ ] 나머지 모델(CAM++, HuBERT, SSAST) 학습 완료 대기
- [ ] 최종 앙상블 조합(Soft Voting) 가중치 확정
- [ ] 확정된 가중치를 로드하고 앙상블 또는 단일 모델 추론을 수행하는 로직을 `mission2_inference()`에 구현
- [ ] 로컬에서 `python inference.py --output ./outputs/mission2.csv` 실행하여 정상 작동 및 포맷(CSV) 확인

## 3. 진행 현황
- 대기 중 (다른 모델 학습 결과 도출 후 진행 예정)
