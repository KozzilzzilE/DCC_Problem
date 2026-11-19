# `m4_backbone_tapt_20261119` — 전략 B: 4대 경량 백본 동일 TAPT 공정 비교 및 `No-TAPT RoBERTa` 대조 실험

- 작성일: 2026-11-19
- 담당자: 이성규 (계획 설계·실행·오프라인 재검증)
- 현재 상태: **실행 완료 / 보존 검증 완료** (3종 가중치·TAPT 인코더·재개 상태·확률표 SHA-256 일치, `score.py` 재채점 경고 `0`건, 오프라인 CPU `16`프리픽스 추론 일치, Colab L4 VM 정지 `Active assignments: 0`)
- 사전 프로토콜: [`PROTOCOL.md`](PROTOCOL.md)
- 코드 위치: [`tools/m4_backbone_tapt/`](../../tools/m4_backbone_tapt/) (`tapt_mlm_resumable.py`, `train_prefix_backbone.py`, `predict_prefix_backbone.py`, `remote_run_stage.py`, `orchestrate_strategy_b.py`, `verify_offline_and_score.py`, `test_backbone_tapt.py`)
- 로컬·공유 보관 위치:
  - 로컬 전체 산출물(재개 상태 포함): `artifacts_private/runs/m4_backbone_tapt_v1_20261119/`
  - Google Drive 공유 폴더(`본선/mission4_sunggyu_share_20261119/04_strategy_b_backbone_tapt_20261119/`): `https://drive.google.com/drive/folders/10s1pRZO6ksX0GC3WRMG2_ylbddZFLFqn` (`best_model/`, `tapt_model/`, `*_light.zip`, 요약 JSON — 원본 전사·Validation 제외)

---

## 1. 왜 이 실험을 하는가

기존 Mission 4 백본 비교(`E002`, `E008`, `E019`, `E021`, `E022` 및 팀원 `ohyun0628` 벤치마크)에는 두 가지 공정성 공백이 있었다.
1. 기준선 `klue/roberta-base` (`E002`, `0.502733`)에는 20-epoch TAPT가 적용된 반면, **`No-TAPT klue/roberta-base` 대조군이 없어** Mission 4에서 TAPT 자체의 순수 기여도와 No-TAPT 동일 조건 기준점이 비어 있었다.
2. 1차 3통 스모크에서 2·3위였던 **`KR-BERT-char` (`snunlp/KR-BERT-char16424`)** 와 **`KoELECTRA-v3` (`monologg/koelectra-base-v3-discriminator`)** 는 `fit 26,280 / dev 2,920` 공정 재판정과 `20-epoch TAPT` 검증이 모두 부재했다(`KcELECTRA`만 `E022`에서 20ep TAPT 완료).

**검증할 질문:**
1. `klue/roberta-base`에서 TAPT 20 epoch 유무(`roberta_notapt_s42` vs `E002`)의 순수 `raw p >= 0.5` 4-1 차이는 얼마인가?
2. 배포 가능한 경량 `base`급 4대 백본(`KLUE-RoBERTa-base`, `KcELECTRA-base-v2022`, `KoELECTRA-base-v3`, `KR-BERT-char16424`) 전원에 **동일한 `fit 26,280` 전사 20-epoch TAPT + 동일 `E002` Prefix 3-epoch 레시피**를 적용했을 때 공식 `raw p >= 0.5` 서열이 뒤집히는가?

---

## 2. 무엇을 어떻게 비교하는가

| 구분 | `roberta_notapt_s42` | `krbert_tapt20_s42` | `koelectra_tapt20_s42` | 기존 기준선 `E002` / `E022` |
|---|---|---|---|---|
| 백본 | `klue/roberta-base` | `snunlp/KR-BERT-char16424` | `monologg/koelectra-base-v3-discriminator` | `klue/roberta-base` / `KcELECTRA-base-v2022` |
| 사전 도메인 적응(TAPT) | **미적용 (대조군)** | **`fit 26,280` 20ep MLM** | **`fit 26,280` 20ep MLM** | `fit 26,280` 20ep MLM |
| 어휘 크기 / UNK율 | `32,000` / `0.00023` | `16,424` / `0.000008~0.000015` | `35,000` / `0.000019~0.000027` | `32,000` / `54,343` |
| Prefix 레시피 (고정) | `fit 26,280`, `3ep`, `samples_per_call=2`, `bs=16`, `lr=5e-5`, `llrd=0.8`, `pos_weight^0.5`, `precue=-1.0`, `seed=42`, AMP FP16 | 동일 | 동일 | 동일 (`E022`는 8GB VRAM 제약으로 Prefix `8x2`) |

- **데이터·누수 차단:** `fit_ids.txt` (`26,280`건) vs `dev_ids.txt` (`2,920`건, `87,814`시점). 코드 내 `verify_non_overlapping` 및 `assert_not_validation_path`로 ID·정규화 전사 겹침 0건 및 `Validation` 접근 전면 차단 확인.
- **아키텍처·LLRD 정합 검사:** RoBERTa/BERT/ELECTRA별 `AutoModelForMaskedLM` 및 `AutoModelForSequenceClassification` 키 검증 통과, 3종 모두 `adam_state_count == total_trainable_tensors == 201` (100% LLRD 커버리지 확인).
- **사전 등록 판단 기준 (`PROTOCOL.md`):** `E002` (`0.502733`) 대비 `> +0.002` (`> 0.504733`, 95% CI > 0) 초과 백본이 나오지 않으면 `KLUE-RoBERTa-base + TAPT`를 최종 유지.

---

## 3. 실제 수행과 결과 (`Dev 2,920`, 공식 `raw p >= 0.5`, Top-1 없음)

- **실행 일시·환경:** `2026-11-19 16:33 ~ 23:11 KST` (`23,926.2초`), Colab `NVIDIA L4` (`22.03 GB`, 세션 `dcc-m4-tapt-b-l4-20261119`), 잔여 유닛 `47.54 → 34.78` (**총 `12.76`유닛 소모**, 종료 후 `Active assignments: 0` 확인).
- **Pre-flight 검증 (`preflight_report.json`):** 3종 토크나이저 UNK율 전부 `0.00023` 이하(`< 0.05` 기준 통과), `KR-BERT` fast/slow 토크나이저 결과 동일(`use_fast="true"`), 20-update L4 TAPT 스모크로 `KR-BERT 2.13h`·`KoELECTRA 2.38h` 추정(`5h` 상한 통과).

### 공식 채점 비교표 (`score.py` 경고 `0`건, `Dev 2,920`통화 · `87,814`시점)

| 순위 | 모델 / 조건 | TAPT 여부 | `4-1` (`raw 0.5`) | `E002` 대비 차이 `[95% CI]` (`5,000`회) | 단일 / 다중 증상 `4-1` | 앞 / 중 / 뒤 1/3 | 빈 예측률 | 끝발화 Macro-F1 | `ENS_swap` (`4-1`) |
|---:|---|---|---:|---|---|---|---:|---:|---:|
| **1** | **`E002` (`KLUE-RoBERTa-base`)** | **TAPT 20ep** | **`0.502733`** | 기준선 (`0.000000`) | `0.5646 / 0.3772` | `0.3271 / 0.5517 / 0.6250` | `15.15%` | `0.6392` | `ENS3: 0.506001` |
| **2** | **`roberta_notapt_s42` (`KLUE-RoBERTa-base`)** | No-TAPT | **`0.498685`** | `-0.004049 [-0.008864, +0.000983]` | `0.5598 / 0.3743` | `0.3215 / 0.5462 / 0.6245` | `16.18%` | `0.6488` | `0.503727` (`-0.002273`) |
| **3** | `E022` (`KcELECTRA-base-v2022`) *(팀원)* | TAPT 20ep | `0.498099` | `-0.004634 [-0.009569, +0.000469]` | — | — | `15.4%` | `0.6428` | `0.505466` (`-0.000535`) |
| **4** | `E021` (`KcELECTRA-base-v2022`) *(팀원)* | No-TAPT | `0.4957` | `-0.0070 [-0.0127, -0.0010]` | — | — | `16.5%` | `0.6333` | `0.5050` (`-0.0010`) |
| **5** | **`koelectra_tapt20_s42` (`KoELECTRA-v3`)** | **TAPT 20ep** | **`0.495421`** | **`-0.007312 [-0.012621, -0.001935]`** (`p=0.0032`) | `0.5540 / 0.3762` | `0.3219 / 0.5444 / 0.6183` | **`15.10%`** | `0.6446` | `0.503507` (`-0.002493*`) |
| **6** | **`krbert_tapt20_s42` (`KR-BERT-char16424`)** | **TAPT 20ep** | **`0.493317`** | **`-0.009416 [-0.014867, -0.003899]`** (`p=0.0002`) | `0.5517 / 0.3745` | `0.3193 / 0.5404 / 0.6146` | `15.60%` | `0.6424` | `0.504329` (`-0.001672`) |

> 주: `E022`/`E021` 수치는 `mission4-streaming` `runs/E021_kcelectra/analysis.txt`, `runs/E022_kcelectra_tapt/analysis.txt` 기록 기준(부트스트랩 2,000회). 전략 B 3종의 `E002`·`ENS3` 대비 5,000회 쌍 부트스트랩은 외장 디스크 `e001`/`e002`/`e007_dev_prefix.npz` 위에서 직접 재계산했다([`offline_verification_summary.json`](../../artifacts_private/runs/m4_backbone_tapt_v1_20261119/offline_verification_summary.json)).

### 학습 시간 및 손실 곡선 요약
- **`roberta_notapt_s42`:** Prefix `1,649.7s` (Loss `0.5425 → 0.4842 → 0.4553`), 추론 `104.95s`, VRAM `4.66 GiB`.
- **`krbert_tapt20_s42`:** TAPT 20ep `7,927.5s` (`2h 12m`, MLM loss `1.4407 → 0.8317`, `amp_skips=10`), Prefix `1,703.9s` (`0.5377 → 0.4767 → 0.4457`), 추론 `113.88s`, VRAM `4.53 GiB`.
- **`koelectra_tapt20_s42`:** TAPT 20ep `8,790.2s` (`2h 27m`, MLM loss `3.9728 → 1.1145`, `amp_skips=13`), Prefix `1,693.0s` (`0.5506 → 0.4831 → 0.4586`), 추론 `106.59s`, VRAM `4.68 GiB`.

---

## 4. 결정과 해석

- **판정:** **`KLUE-RoBERTa-base + TAPT` 백본 유지 확정 (신규 백본 3종 모두 `E002` 미달)**.
- **확인된 핵심 사실 3가지:**
  1. **순수 Mission 4 TAPT 효과 확인 (`+0.004049`):** 동일 `KLUE-RoBERTa-base`에서 20-epoch TAPT 유무만으로 `0.498685 → 0.502733` (`+0.004049`, `p=0.9424` 양방향 단측)이 상승해 예선 도메인 적응 투자의 정당성이 본선에서도 재확인됐다. (`No-TAPT` 기준에서도 `RoBERTa 0.498685`가 `KcELECTRA E021 0.4957`을 앞선다.)
  2. **4대 경량 백본 동일 TAPT 정면 비교 완결:** `KR-BERT-char + TAPT (0.493317)`와 `KoELECTRA-v3 + TAPT (0.495421)` 모두 `E002 (0.502733)` 대비 각각 **`-0.009416` (`95% CI [-0.0149, -0.0039]`)**, **`-0.007312` (`95% CI [-0.0126, -0.0019]`)** 로 95% 신뢰구간이 `0`을 완전히 벗어난 통계적으로 유의한 열위로 확인됐다. 기존 `E022 (KcELECTRA + TAPT 0.498099)`까지 포함해 배포 가능한 `base`급 4대 백본 전원이 `KLUE-RoBERTa-base + TAPT`를 넘지 못했으며, 오히려 **`No-TAPT RoBERTa (0.498685)`가 다른 3종 + TAPT보다도 높았다**.
  3. **앙상블 다양성 효과도 없음:** `ENS3`의 `E002`를 교체(`ENS_swap 0.5035~0.5043`)하거나 4번째 멤버로 추가(`ENS4_add 0.5040~0.5048`)해도 모두 기존 `ENS3 (0.506001)`보다 하락했다(특히 `KoELECTRA+TAPT`는 교체 `-0.002493`·추가 `-0.001993` 모두 CI `< 0` 유의 하락).
- **한계:** `KF-DeBERTa (E008)`와 `RoBERTa-large (E019)`는 20-epoch TAPT 미적용 상태이나, `base` 대비 2.5~4배의 스트리밍 추론 비용·메모리 제약 및 `No-TAPT` 단계에서의 낮은 비용-효율로 배포 후보에서 제외했다(`E008/E019`의 공식 `raw 0.5` 수치는 팀원 PC `.npz` 회수 또는 별도 CPU 재채점 시 표에 보충 가능).

---

## 5. 산출물 보존 확인

| 항목 | 상태·해시 | 보관 위치 |
|---|---|---|
| 실행 코드·단위 테스트 | `test_backbone_tapt.py` `2 passed`, `py_compile` 통과 | `tools/m4_backbone_tapt/` |
| Pre-flight & 오케스트레이터 요약 | `preflight_report.json`, `ORCHESTRATOR_SUMMARY.json` | `artifacts_private/runs/m4_backbone_tapt_v1_20261119/` |
| `roberta_notapt_s42` 가중치·재개·확률표 | `final.pt` (`fb8d0a2b...`), `resume` (`e54d392d...`), `dev_prefix.npz` (`9ba83c74...`), CPU 16-prefix `max_err=0.000568` | 로컬 + Drive `04_strategy_b...` |
| `krbert_tapt20_s42` TAPT·Prefix·재개·확률표 | `tapt` (`e136e21c...`), `tapt_resume` (`c7606bd4...`), `final.pt` (`44113142...`), `resume` (`142d3ff8...`), `npz` (`5e4a941b...`), CPU `max_err=0.000275` | 로컬 + Drive `04_strategy_b...` |
| `koelectra_tapt20_s42` TAPT·Prefix·재개·확률표 | `tapt` (`0d686dc9...`), `tapt_resume` (`c1d4ed72...`), `final.pt` (`83d9c15a...`), `resume` (`1d91d5cc...`), `npz` (`16fb6606...`), CPU `max_err=0.000459` | 로컬 + Drive `04_strategy_b...` |
| 공식 `score.py` 재채점 + `5,000`회 부트스트랩 | 경고 `0`건, 전체 SHA-256 일치 (`offline_verification_summary.json`) | `artifacts_private/runs/m4_backbone_tapt_v1_20261119/` + Drive |
