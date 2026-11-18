# E026 동일 입력 고정 정책 재생 준비

상태: **합성 smoke만 완료 / 원본 재생 미실행**. 새 학습은 하지 않는다. 이전 E026 결과와 S3 결과를 섞지 않고 E001/E002/E007의 동일 Dev 2,920 통화 확률 평균에서 β0, β128, B10, S3를 재생한다. β0/β128은 E026 원 shift·고정 policy를 사용한다. 정답 raw p ≥ 0.5, Top-1 없음. B10은 10번째 또는 그보다 짧으면 마지막 발화. S3는 5번째부터 3개 연속 동일한 0.5 이진 벡터에서 중단, 없으면 마지막 발화다.

비공개 준비 위치: `artifacts_private/e026_same_input_replay_20261116/`. `source/`는 commit `5d7539b5444f94073d3a4cd370447c128c12743e`의 고정 source 및 policy, shift 사본이다. SHA256SUMS 참조. **`m4/__init__.py`는 Git blob 자체 미확보** (기대 Git blob `bec21f16da26b098cfa0e67caa1213bae5cd8c2b`); 원본 실행 전 복구해야 한다. LightGBM도 현재 환경에 없다. 환경 버전과 실행 명령은 실제 실행 시 별도 기록한다.

필수 입력은 각 원본의 로컬 경로 4개와 독립 확인한 SHA256 JSON: `--e001 <E001/dev_prefix.npz> --e002 <E002/dev_prefix.npz> --e007 <E007/dev_prefix.npz> --calls <original/train.jsonl> --hashes <inputs.json> --out <새 비공개 출력 경로>`. JSON은 `{"e001":"<sha256>","e002":"<sha256>","e007":"<sha256>","calls":"<sha256>"}`. 입력 자료와 각 해시 출처·생성 체크포인트 및 fit/dev ID 이력은 확보되지 않았다. 해시 일치만으로 출처를 입증하지 않는다. OOF 자료는 고정 정책 재생에는 필수 아님.

실행기는 해시, 2,920 고유 ID, 각 통화 offsets/라벨/확률 일치 및 범위, 원문 캐시 길이·라벨 일치, float32 평균, 보정값과 정책 일치, 공개 채점기 점수 일치, 통화 짝 bootstrap을 확인한다. 개별 중단·정답·예측은 비공개 새 출력 폴더에 저장한다. 기록된 β0 0.4989532854214187, β128 0.5031464178054406과 재생 점수의 일치 여부를 기록하며 불일치 시 원인 분석 전 후속 해석을 중단한다. 기존의 전체 Training 기반 shift에는 Dev 라벨 분포가 반영되어 있다. 원래 정책 및 자료 출처·실행 환경 검증은 별도 필수다.

검증: `python3 artifacts_private/e026_same_input_replay_20261116/smoke.py`. 합성 통화에서 정렬 거부, 평균, 중단 및 공개 채점기 일치, bootstrap을 확인했다. 이 테스트의 β128은 LightGBM 대신 가짜 정책을 쓰므로 실제 모델 로딩·예측 검증이 아니다.
