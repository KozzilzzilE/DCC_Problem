# m4_jacc_macrof1_v1_20261116 — `seed 43` 반복 검증 및 2-seed 결과 보충

- 작성일: 2026-11-17 (상위 검토 갱신: 2026-11-18)
- 실행 담당 및 최종 검토: 보조 코딩 에이전트 실행 / **`GPT-6-astra high` (Argon High) 설계·원인 분석·채택 최종 검토 완료**
- 기준 문서: [seed 42 실험 기록](EXPERIMENT_RECORD.md), [사전 계획](PROTOCOL.md)
- 현재 상태: **실행 완료 / 보존 검증 완료 (4-1·4-2 단일 모델 통합 운용 시 조건부 채택 후보; 분리 운용 시 `SJ` + `ENS3` 조합 대비 열위)**

## 1. `seed 43` 실행 요약

- 실행 ID / 조건: `m4_jacc_macrof1_v1_s43` (`seed 43`, `BCE(pos_weight^0.5) + 0.5·SoftJaccard + 0.5·(1 - SmoothMacroDice)`, 나머지 조건 `s42`와 동일)
- 실행 환경·시간: `dcc-m4-stage2-l4-20261117` (`NVIDIA L4`), `1,766.4초` (`training_seconds: 1,634.5s`, `peak_cuda_memory_bytes: 5,023,180,288`, `9,855` steps, `updates: 9,850`, `amp_skips: 5`, `train_loss: 1.1403 → 1.0240 → 0.9692`)
- config digest: `186435908ed95454d01934f21fecc9bfac7ca0f1626d6da50f989eb1c401c24a`

## 2. `Dev 2,920` (`87,814`시점) 결과

### Mission 4-1 (`raw p>=0.5`, Top-1 없음)

| 조건 | 4-1 전체 | 단일(`1,958`) | 다중(`962`) | 빈 예측률 | 평균 예측 수 |
|---|---|---|---|---|---|
| `E002` (기준) | `0.502733` | `0.563053` | `0.379961` | `15.15%` | `1.2160` |
| `JM_s42` | `0.513424` | `0.579092` | `0.379768` | `12.59%` | `1.2267` |
| **`JM_s43`** | **`0.515594`** | **`0.582494`** | `0.379429` | **`12.86%`** | `1.2187` |
| **2-seed 평균 (`JM42+JM43` 확률 평균)** | **`0.515999`** | - | - | - | - |
| `ENS3` (`E001+E002+E007`) | `0.506001` | `0.569274` | `0.377218` | `16.43%` | `1.1682` |
| `ENS_rep_jm_s43` (`E001+JM43+E007`) | **`0.509867`** | **`0.575153`** | `0.376988` | `15.41%` | `1.1725` |
| `E001+E007+JM42+JM43` | `0.512240` | - | - | - | - |
| `SJ42+SJ43+JM42+JM43` | `0.521154` | - | - | - | - |

- 2-seed 단독 평균 / 표준편차: **`0.514509`** (`std=0.001085`, 통화별 `s42`–`s43` 상관계수 `r=0.9368`).
- `10,000`회 paired bootstrap:
  - `JM_s43 - E002 = +0.012860` (`95% CI [+0.008555, +0.017313]`, `p_gt0=1.0`)
  - `ENS_rep_jm_s43 - ENS3 = +0.003866` (`95% CI [+0.001629, +0.006056]`, `p_gt0=1.0`)
  - `JM_2seed_avg - E002 = +0.013266` (`95% CI [+0.009439, +0.017109]`)
  - `JM_2seed_avg - ENS3 = +0.009999` (`95% CI [+0.006741, +0.013345]`)
  - `E001+E007+JM42+JM43 - ENS3 = +0.006239` (`95% CI [+0.003927, +0.008488]`)
- 공식 `score.py` 4-1 재채점:
  - `JM_s43`: `0.5155937235468306` (`warnings: []`)
  - `JM_2seed_avg`: `0.5159993879708982` (`warnings: []`)
- `JM_s43` `final_step Macro-F1` 9종: `고열 0.6743`, `구토 0.5743`, `두통 0.5585`, `복통 0.8251`, `어지러움 0.6720`, `열상 0.8934`, `오심 0.3599`, `전신쇠약 0.6176`, `호흡곤란 0.6833`.

### Mission 4-2 (`Dev 2,920`, `fit_only_shift`)

- `frozen E026 β=128`: `JM_s43` 단독 `0.492091` (`Macro-F1 0.5806`, `earliness 0.8476`), `ENS_rep_jm_s43` `0.498993` (`Macro-F1 0.5968`, `earliness 0.8361`).
- `Dev nested 5-fold` (자체 확률 + `nonempty` 게이트, 탐색):
  - `JM_s43`: `0.494486` (`Macro-F1 0.5992`, `earliness 0.8252`) vs `JM_s42` `0.501743`
  - `JM_2seed_avg`: `0.497788` (`Macro-F1 0.6022`, `earliness 0.8266`)
  - `E001+E007+JM42+JM43`: `0.498166` (`Macro-F1 0.5999`, `earliness 0.8304`)
  - `E001+E007+SJ42+SJ43+JM42+JM43`: `0.499395` (`Macro-F1 0.6049`, `earliness 0.8256`, `4-1 0.516903`)
- 해석: 4-1 개선(`~+0.011~0.013` vs `E002`, 2-seed avg `0.5160`)은 두 seed 모두에서 일관되게 재현됐다. 반면 `Dev 2,920` 내 소규모 5-fold(`2,336`통화) 기반 `4-2` `beta` 선택은 seed·fold 변동이 확인되므로, **Gap ③ (`fit 26,280` 5-fold OOF)** 을 통한 안정적·무누수 중단 정책 학습이 필수적이다.

## 3. 산출물 보존 확인 (`s43`)

| 항목 | 실제 경로/해시 | 상태·확인 일시 |
|---|---|---|
| 평가·설정·로그 | `/Volumes/로컬 디스크/DCC/private_experiments/m4_jacc_macrof1_v1_20261116/s43/eval_precue.json`, `s43/light/{run_config,optimizer_audit,initialization,remote_quick_summary}.json`, `train_log.txt` | 실물·일치 확인 (`2026-11-17 18:33 KST`) |
| `dev_prefix.npz` | `s43/light/dev_prefix.npz` (`sha256: 9cb0d93a308f8461cc3ef2ac930d59e9a6384289ec9e128ebea9aea6a59361fc`) | SHA-256·`score.py` 경고 `0` 확인 (`2026-11-17 18:33 KST`) |
| 가중치 ZIP (`132`청크 결합) | `s43/m4_jacc_macrof1_s43_weights.zip` (`2,213,595,697` B, `sha256: a67d276e16e94dca8d3e842ff4fd280745a316b29b4ebffac8d090d0b218ac3e`) | 무결성 확인 (`2026-11-17 18:35 KST`) |
| `final.pt` + `best_model/` + 토크나이저 | `s43/weights/content/runs/m4_jacc_macrof1_v1_s43/final.pt` (`442,552,911` B, `sha256: 286da530a233e976a8c72e1edf60da2feb051b827b43854b3ecdf99123cd398d`) | 원격 로그·`run_config` SHA 일치 (`2026-11-17 18:35 KST`) |
| 마지막 학습 재개 상태 | `s43/weights/.../training_state_last.pt` (`1,327,762,627` B, `sha256: 6e6bdf9ab241bc61451dfcdde7bf7f0b4a79628aa3f9d84c39830a0836ac18d2`) | 실물·SHA 일치 확인 (`2026-11-17 18:35 KST`) |
| 오프라인 CPU 추론 일치 | 앞 4통화(`72`시점) `best_model` CPU vs `dev_prefix.npz`: `max_abs_err=0.000470`, `mean_abs_err=0.00007847`, `0.5` 불일치 `0`건 | 검증 완료 (`2026-11-17 18:35 KST`) |
