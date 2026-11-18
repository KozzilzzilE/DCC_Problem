# `e026_same_input_replay_20261116` — E026 동일 Dev 2,920 입력 고정 정책 재생 및 중단 규칙 비교

- 작성/갱신일: 2026-11-16
- 담당자, 실행·기록 모델: 사용자 + Gemini 4 Hydrogen (XHigh) *(사전 [PROTOCOL.md](PROTOCOL.md) 기반으로 실행·기록; 지정 GPT-6.1-sol xhigh / GPT-6-astra high 최종 검토 미실시 상태)*
- 현재 상태: **실행 완료 / 원점수 재현 및 동일 입력 비교 완료** *(한계: 기존 `E016_final/shift`의 Training 29,200 분포 간접 포함 및 `E011_oof` 미확보로 fit-only 정책 재학습은 별도 과제로 남음)*
- 이전 실험·관찰 자료: [PROTOCOL.md](PROTOCOL.md), [E026 정적 감사](../mission4_e026_audit_20261116/REPORT.md), [업데이트 검토](../mission4_streaming_baseline_20261115/UPDATES_REVIEW.md), [S3 CPU 비교 기록](../m4_stop_stability3_cpu_20261116/EXPERIMENT_RECORD.md)
- 코드/비공개 보관 위치:
  - 실행기/고정 소스: `artifacts_private/e026_same_input_replay_20261116/` (`commit 5d7539b` 사본 + `mission4_symptom.zip`에서 복구한 `m4/__init__.py`)
  - 복원 입력(`dev_prefix.npz` 3종, `train.jsonl`, `fit_ids.txt`, `dev_ids.txt`, `inputs.json`): `/Volumes/로컬 디스크/DCC/private_experiments/e026_same_input_replay_20261116/inputs/`
  - 실행 결과(`results.json`, `per_call.json`, `breakdown.json`, `environment.json`, `input_manifest.json`): `artifacts_private/runs/e026_same_input_replay_20261116/`

---

## 1. 왜 이 실험을 하는가

팀의 `mission4-streaming` E026 보고서(`commit 5d7539b`)는 LightGBM 학습형 중단 정책(`β=128, τ=0.5`)으로 Dev 2,920에서 4-2 점수 `0.498953 → 0.503146`(`+0.004193`)을 기록했으나, 로컬 감사 시점에는 Dev 시점별 원확률(`.npz`)과 원문 캐시가 없어 통화별 재채점이나 단순 중단 규칙(`B10`, `S3`)과의 **동일 입력 비교**를 수행하지 못했었다.

검증할 질문:
1. 팀 공유 ZIP(`mission4_baseline_share.zip`, `mission4_share2_E001_E007_bundle.zip`, `mission4_symptom.zip`) 및 원본 Training 라벨 JSON으로 복원한 동일 `Dev 2,920` 입력에서 기존 `E026`의 `β=0`(`0.4989532854214187`) 및 `β=128`(`0.5031464178054406`) 점수가 정확히 재현되는가?
2. **동일한 `E001+E002+E007` float32 평균 확률과 `p >= 0.5` Top-1 없음 기준** 위에서 `beta128`, `beta0`, `B10`, `S3` 네 중단 규칙은 어떤 차이를 보이는가?

---

## 2. 무엇을 어떻게 비교하는가

- **공통 입력(고정)**: `E001_m3_baseline`, `E002_prefix_uniform`, `E007_prefix_uniform_s43`의 `Dev 2,920`통화(`87,814`시점) `float32` 원확률 3개 단순 평균, `raw p >= 0.5` (Top-1 강제 출력 없음).
- **비교 정책 4종**:
  - **`beta0`**: `E016_final/shift` 기반 기대 점수 `v(t) >= 0.5`, 최소 `5`발화 (`β=0`)
  - **`beta128`**: `E026` 학습형 LightGBM 회귀 정책 `v(t) + 128·δ̂(t) >= 0.5`, 최소 `5`발화
  - **`B10`**: 10번째 발화(10발화 미만 통화는 마지막 발화) 고정 중단
  - **`S3`**: 5번째 발화부터 최근 3시점 `p >= 0.5` 이진 예측 벡터가 연속 동일하면 즉시 중단, 없으면 마지막 발화
- **사전 입력 무결성·출처 복원 검증**:
  - `m4/__init__.py`: `mission4_symptom.zip`(`/Volumes/로컬 디스크/다운로드/mission4_symptom.zip`, SHA-256 `a8a8c0ad...`)에서 추출 후 Git blob `bec21f16da26b098cfa0e67caa1213bae5cd8c2b` 완전 일치 확인(SHA-256 `ff4d8fb4...`).
  - `e001_dev_prefix.npz` (`mission4_share2_E001_E007_bundle.zip`에서 추출, SHA-256 `a1a6e195...`)
  - `e002_dev_prefix.npz` (`mission4_baseline_share.zip`에서 추출, SHA-256 `ec52ff81...`)
  - `e007_dev_prefix.npz` (`mission4_share2_E001_E007_bundle.zip`에서 추출, SHA-256 `f7eef245...`)
  - `train.jsonl`: `/Volumes/로컬 디스크/DCC/DCC_dataset/Training/2.라벨링데이터/TL_서울_구급`에서 `m4.data.build_cache`로 `29,200`통화 재생성 (`LF SHA-256: b58498df...`; Windows `CRLF` 변환 시 팀 `model_train.ipynb` 기록 해시 `9e00495204f96a697f9c7a914e3ce5d39d5ed7f3948dfe64db0f481f6e08097a`와 바이트 단위 완전 일치 확인).
  - 분할 정합성: `dev_ids.txt`(`2,920`건)와 `fit_ids.txt`(`26,280`건) 교집합 `0`, 정규화 전사 완전 중복 통화 `0`건, `per_call.json` ID 순서가 `dev_ids.txt`와 100% 일치 확인. `E011_oof`는 미보유.
  - 공식 Validation: 미접근.
- **실행 환경**: macOS-26.5.2-arm64 (CPU), Python `3.12.7`, `numpy 1.26.4`, `scipy 1.13.1`, `torch 2.9.0`, `lightgbm 4.7.0`.
- **실행 명령**:
```bash
python3 -u artifacts_private/e026_same_input_replay_20261116/run.py \
  --e001 "/Volumes/로컬 디스크/DCC/private_experiments/e026_same_input_replay_20261116/inputs/e001_dev_prefix.npz" \
  --e002 "/Volumes/로컬 디스크/DCC/private_experiments/e026_same_input_replay_20261116/inputs/e002_dev_prefix.npz" \
  --e007 "/Volumes/로컬 디스크/DCC/private_experiments/e026_same_input_replay_20261116/inputs/e007_dev_prefix.npz" \
  --calls "/Volumes/로컬 디스크/DCC/private_experiments/e026_same_input_replay_20261116/inputs/train.jsonl" \
  --hashes "/Volumes/로컬 디스크/DCC/private_experiments/e026_same_input_replay_20261116/inputs/inputs.json" \
  --out "/Users/sunggyu/Desktop/이성규/DCC/artifacts_private/runs/e026_same_input_replay_20261116"
```

---

## 3. 실제 수행과 결과

### 3.1 원점수 일치 검증 (`score.py` 직접 교차 대조 통과)
- `original_reference.beta0 = 0.4989532854214187` → 재생 결과 `0.4989532854214187` (**`reference_match.beta0 = true`**)
- `original_reference.beta128 = 0.5031464178054406` → 재생 결과 `0.5031464178054406` (**`reference_match.beta128 = true`**)
- 4개 조건 모두 공개 채점기(`score.score_mission_4_2`) 경고 0건 및 `1e-12` 이내 일치 검증 통과.

### 3.2 동일 입력(`Dev 2,920`통화) 네 정책 성능 비교

| 정책 | 4-2 점수 (`Macro-F1 × Earliness`) | Macro-F1 | 평균 Earliness | 중단 중앙 / 평균 발화 | 끝까지 간 비율 | 빈 예측 수 / 2,920 (`비율`) | 단일(1,958) F1 | 다중(962) F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`beta128`** (E026 학습형) | **`0.503146`** | `0.606354` | `0.829790` | `7.0 / 9.63` | `2.02%` | `43 (1.47%)` | `0.595835` | `0.567673` |
| **`beta0`** (E026 기대점수 대조) | `0.498953` | `0.613721` | `0.812998` | `6.0 / 10.64` | `8.08%` | `34 (1.16%)` | `0.608999` | `0.574714` |
| **`B10`** (10발화 고정) | `0.450774` | `0.556502` | `0.810013` | `10.0 / 10.00` | `0.38%` | `454 (15.55%)` | `0.555747` | `0.515173` |
| **`S3`** (3연속 안정성) | `0.412718` | `0.463150` | `0.891111` | `5.0 / 5.75` | `0.00%` | `958 (32.81%)` | `0.466279` | `0.423953` |

*(Dev `2,920`통화 평균 발화 길이 `30.07`발화; 단일 증상 `1,958`통화, 다중 증상 `962`통화)*

### 3.3 통화 단위 paired bootstrap (10,000회, seed `20261116`)

| 비교 쌍 | 4-2 점수 차이 | 95% CI | `P(차이 > 0)` |
|---|---:|---|---:|
| `beta128 - beta0` | `+0.004193` | `[-0.001003, +0.009525]` | `0.9405` |
| `S3 - B10` | `-0.038056` | `[-0.048177, -0.028448]` | `0.0000` |
| `S3 - beta128` | `-0.090428` | `[-0.101545, -0.079614]` | `0.0000` |
| `B10 - beta128` | `-0.052372` | `[-0.060279, -0.044314]` | `0.0000` |

---

## 4. 해석 및 결정

1. **원본 재채점 완전 재현 확인**: `E001/E002/E007` 원확률과 `train.jsonl`을 복원해 실행한 결과, `E026` 보고 수치(`beta0=0.4989532854214187`, `beta128=0.5031464178054406`)와 공개 채점기 결과가 오차 없이 완전히 재현되었다.
2. **단순 규칙(`B10`, `S3`) 대비 `beta0/beta128`의 우위 확인**: 동일한 `E001+E002+E007` 확률 위에서도 `S3`는 초기 빈 예측(`[0,...,0]`) 상태가 3연속 이어지면 평균 `5.75`발화 만에 곧바로 끊어 **전체의 `32.81%`(958통화)가 빈 예측으로 종료**되면서 `Macro-F1`이 `0.463150`까지 떨어졌고, `B10` 역시 `15.55%`(454통화)가 빈 예측이어서 `beta128`보다 `-0.052372` 낮았다.
3. **`beta128` 대 `beta0`의 실제 작동 방식과 한계 유지**:
   - `beta128`은 `Macro-F1`이 `beta0`보다 소폭 하락(`0.613721 → 0.606354`, 단일/다중 모두 감소)했지만 끝까지 듣는 비율을 `8.08% → 2.02%`로 줄여 `earliness`를 `0.812998 → 0.829790`으로 끌어올렸다.
   - 다만 10,000회 통화 paired bootstrap 95% CI가 `[-0.001003, +0.009525]`로 **여전히 0을 포함**한다.
4. **남은 미해결 과제**:
   - 현재 재생한 `beta0`/`beta128`은 여전히 Training 전체 `29,200`통화 라벨로 계산한 `E016_final/shift`를 사용한다. 엄밀한 fit-only 검증([fit-only 보정 통제 비교 계획](../mission4_e026_fitonly_policy_20261116/PROTOCOL.md))을 하려면 아직 미확보 상태인 `runs/E011_oof/fold0..4/oof_prefix.npz`(`fit 26,280` 5-fold 교차확률)를 확보하거나 이번에 받은 `TAPT` 백본·`fit_ids.txt`로 5-fold OOF를 다시 생성해야 한다.
   - `Dev 2,920`은 이미 정책·모델 탐색에 반복 사용된 내부 분할이므로, 이 재생 결과를 새로운 독립 확인 성적으로 쓰지 않는다.

---

## 5. 산출물 보존 확인

- `artifacts_private/e026_same_input_replay_20261116/source/m4/__init__.py` (`bec21f16da26b098cfa0e67caa1213bae5cd8c2b`) 복구 완료 및 `SHA256SUMS` 검증 통과.
- `/Volumes/로컬 디스크/DCC/private_experiments/e026_same_input_replay_20261116/inputs/`: `e001_dev_prefix.npz`, `e002_dev_prefix.npz`, `e007_dev_prefix.npz`, `train.jsonl`, `dev_ids.txt`, `fit_ids.txt`, `dev_split.csv`, `inputs.json` 보존.
- `artifacts_private/runs/e026_same_input_replay_20261116/`: `results.json` (`b04af542...`), `per_call.json` (`1294f3bf...`), `breakdown.json`, `environment.json`, `input_manifest.json` 보존.
