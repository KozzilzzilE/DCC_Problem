# `m4_jacc_macrof1_v1_20261116` 실행 상태 기록 (1차 시도 원격 유휴 회수)

- 작성일: 2026-11-16
- 실행·정리 모델: Claude 계열 에이전트
- 현재 상태: **`실행 중단 / 원격 유휴 회수 (산출물 미회수)`**
- 잔여 Colab 연산량: `73.41` compute units (`Active assignments: 0`, `Usage rate: 0.00/hr`)

## 경위

1. 2026-11-16 `22:04` KST: 로컬 단위 테스트 `3`건 통과(`test_train_precue.py`, `18.378s`) 후 `dcc-m4-precue-l4-20261116` (`NVIDIA L4`)에서 `m4_jacc_macrof1_v1_s42` (`PID 103432`) 학습을 정상 개시하고 `32`분 타이머(`22:37` 예정)를 걸었다.
2. Mac이 대기 중 절전 상태(`~22:07–23:18`)로 들어가면서 타이머 알림이 `23:18` KST에 지연 실행됐고, 그 사이 Colab 임시 런타임(`/content`, Google Drive 미마운트)이 유휴 시간 초과로 자동 반납되어 `dev_prefix.npz`와 가중치 추출 전 소멸됐다.
3. 로컬·외장 기존 실험(`m4_precue_bce_v1_s42`, `m4_softjaccard_v1_s42`) 산출물에는 영향 없음.

## 재실행 전 보완 사항

- Google Drive 마운트(`colab drivemount`는 터미널 사용자 상호작용 필요) 또는 Mac 절전 해제 상태에서 재실행해 회수 전 자동 반납이 일어나지 않도록 한다.
- 다음 GPU 재실행 대상:
  1. `m4_jacc_macrof1_v1_s42` (`4-1` 개선 유지 + `4-2 Macro-F1` 회복 확인)
  2. `m4_softjaccard_v1_s43` (`seed 43` 반복 일관성 vs `E007 0.504211`)
  3. `fit(26,280)` 5-fold OOF 재생성 (`Gap ③` 누수 없는 4-2 `v1_nonempty` 검증)
