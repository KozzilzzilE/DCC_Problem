# Mission 4 팀원 결과 검토와 다음 비교 제안

검토일: 2026-11-14. 보고서·집계 지표·설정·코드를 읽었다. 원본 전사·개별 통화 ID·예측 배열·가중치는 공개하지 않았고 새 학습도 시작하지 않았다. 아래 4-1은 공개 채점 점수이며 Accuracy가 아니다.

## 판단

**KcELECTRA는 Training 전용 동일 조건 비교를 준비할 우선 후보다. 현재 높은 성적만으로 모델을 교체하기에는 데이터 출처와 평가 연결 증거가 부족하다.** 초기 노트북에는 공식 Validation을 학습에 읽은 실행 기록이 있다. 전체 학습 기록의 32,840통화도 출처 확인이 필요하다.

`mission4-streaming`의 TAPT·층별 학습률 감쇠·누적 입력 학습, 앙상블, 학습용 OOF 확률 기반 중단 정책은 발전시킬 가치가 있다. 다만 maxent는 고정 0.5 판정과 다르고, threshold 규칙에도 기본으로 Top-1 강제 출력이 포함된다.

현재 사용자 확인은 **증상 판정 임계값 0.5 고정**, **Top-1 허용 여부 미확인**이다. 다음 보수적 비교는 `raw probability >= 0.5`, `force_top1=False`로 둔다. 과거 임계값 탐색·Top-1 성적은 탐색 결과로 보존한다.

## 브랜치별 비교

| 브랜치 | 모델·데이터/평가 분모 | 4-1 / 0.5·Top-1 구분 | 4-2 | 확인 수준 |
|---|---|---|---:|---|
| `mission4` | 본선 규격 기준 | 실험 성적 아님 | — | 비교 기준 브랜치 |
| `mission4-streaming` | TAPT RoBERTa; fit26,280 / Dev2,920·87,814시점, 최종 Training29,200 / Validation3,640 | E002 raw0.5·Top-1없음 **0.502733**; 3개평균maxent Dev0.5517; 최종Validation **0.530223** | 최종Validation **0.504228** | 코드·설정·집계JSON 확인. 모델실물/분할ID 감사 미실시 |
| `mission4-prefix-speaker` | RoBERTa prefix+화자+pos_weight^.35; 보고서는001zip, 노트북은Validation경로 사용 | 보고0.6388; 0.5표기, 해당점수 실행 연결 미확인 | 보고0.4385~0.4852, 조건·표본 다름 | 노트북Validation경로 확인. 화자와손실 동시변경 |
| `mission4-kcelectra` | KcELECTRA; 노트북Validation3,640→학습3,094/내부평가546 | 보고0.7484; 정확한채점분모·가중치 연결 미확인 | 모델대응점수 미확인 | Validation학습출력 확인; 높은성적 독립검증근거로 사용보류 |
| `mission4-kcelectra-full` | KcELECTRA, 캐시32,840×2epoch / Validation3,640 | 보고0.5 **0.6870**(Top-1여부미확인), maxent0.6933 | 보고 **0.6438** | 출처32,840 및 가중치/예측계보 미검증 |
| `mission4-koelectra` | KoELECTRA 후보; 고유학습자료 미확인 | 개별성적 미확인 | 미확인 | 공통RoBERTa노트북이며 모델고유실행 확인못함 |
| `mission4-krbert` | KR-BERT; 실제학습/평가분모 미확인 | 보고0.6579; 판정·가중치연결 미확인 | 미확인 | 공통RoBERTa/Kc노트북; 고유실행 미확인 |
| `mission4-kogpt2` | KoGPT2; 실제학습/평가분모 미확인 | 보고0.0000 | 미확인 | 단방향구조가실패원인이라는주장 미검증 |
| `mission4-roberta-large` | 초기large; 실제분모 미확인. 별개streaming E019는Dev2,920 | 초기보고0.5456; streaming E019 maxent0.5451 | E019보고0.4899 | 두실험구분필요. 초기원인·가중치연결 미확인 |

초기 수치는 [누적 보고서][B1], streaming은 [실험 집계][S1]·[E001 지표][S2]·[E002 지표][S3], Validation은 [4-1 JSON][S4]·[4-2 JSON][S5], 전체 Kc는 [EX 기록][K1]에 근거한다. 다른 표본·학습량·규칙의 성적을 빼서 개선량으로 계산하지 않는다.

우리의 1만통화 Epoch2 + Top-1 **0.474864**, 두 seed + Top-1 **0.489892**는 반복 사용한 Dev512의 점수다. 팀원의 Dev2,920·Validation3,640과 직접 우열을 정할 수 없다. 우리 raw 0.5 평균 **0.446236**과 Top-1 성적도 구분한다. full22k의 최종 비교·채택은 이번 검토에서 확정하지 않는다.

## 먼저 확인할 문제

1. **초기 노트북의 Validation 사용은 출력으로 확인된다.** Kc 노트북의 자동 탐색은 Training 이외 라벨 폴더도 먼저 고를 수 있다. 실제 출력이 `Validation/VL_서울_구급`, 3,640통화, 학습 3,094통화다. 이후 통화 단위 분할을 해도 공식 Validation이라는 원천은 달라지지 않는다. 단, 이 노트북과 모든 보고 성적이 동일 실행이라고 단정하지 않는다. [출력·탐색][K2], [분할·설정][K3]
2. **전체 Kc 32,840통화는 혼합 가능성 감사가 필요하다.** streaming의 Training 29,200 + Validation 3,640과 정확히 같다. 강한 확인 신호지만 숫자 일치만으로 실제 혼합을 확정하지 않는다. 캐시 원천별 수·학습 ID·분할 해시·실행 로그 대조 전까지 독립 Validation 성능 근거로 사용을 보류한다. [기록][K1]
3. **브랜치명이 실행 증거는 아니다.** 여섯 초기 백본 브랜치의 `model_train.ipynb` 코드는 동일한 RoBERTa dense-prefix 코드다. Kc·KR-BERT·KoGPT2·large 브랜치의 `KcELECTRA.ipynb` 코드도 동일했다. 모델별 실제 실행·seed·가중치 해시·전체 예측 연결이 더 필요하다.
4. **streaming 분할은 설계와 실물 확인을 구분한다.** 코드상 fit/dev는 통화 단위이며 정책 학습·평가 ID가 겹치면 멈춘다. E011 다섯 설정은 각 21,024학습 통화와 fold 경로를 기록한다. 그러나 확인한 Git 트리에는 실제 fit/dev 및 fold ID 목록이 없어 이번에 ID·정규화 전사 중복을 재검사하지 못했다. 비공개 분할과 해시 영수증이 필요하다. [분할][S6], [겹침 검사][S7], [OOF 코드][S8]
5. **읽은 온라인 코드에서는 미래 정보 사용을 발견하지 못했다.** 받은 전사·화자·시각만 누적하고 reset으로 지운다. 전체 길이·마지막 확률은 정책 학습 표적과 채점용이며 확인한 온라인 특징에는 없다. 원격 실행 전체를 독립 재현했다는 뜻은 아니다. [온라인 처리][S9], [특징·표적][S10]
6. **학습 실행기를 그대로 가져오면 보존 기준에 맞지 않는다.** `train_prefix.py`는 최종 추론 모델을 저장하지만 optimizer/scaler/RNG 재개 상태 저장은 없다. OOF 실행기는 확률 추출 뒤 fold 가중치를 지운다. 통합 전에 모든 조건×seed의 최종 가중치와 마지막 재개 상태를 보존해야 한다. [저장][S11], [OOF 정리][S8]

## 규칙과 해석 주의

- 초기 보고서는 주최 측 답변을 근거로 Top-1 금지까지 주장하지만 공식 회신 원문은 확인하지 못했다. streaming은 규칙 적용을 미확인으로 적었다. **0.5 고정은 사용자 확인, Top-1 금지는 아직 미확인**으로 구분한다. [초기 주장][R1], [미확인 기록][R2]
- streaming `threshold` 기본값은 `force_top1=True`다. 이름에 thr0.5가 있어도 순수 0.5라는 보장은 없다. maxent/expected는 0.5를 넘은 집합 대신 기대 점수로 출력 개수를 고르는 별도 방법이다. [결정 코드][S12]
- 초기 halting 도구는 추출된 3·5·8·12·마지막 시점 행을 돌며 연속성을 센다. 모든 발화의 연속 턴을 보는 실제 추론과 일치하는지 확인해야 한다. 같은 표에서 고른 격자 최고점은 독립 평가가 아니다. [초기 중단 도구][R3]
- streaming maxent는 한 증상 통화에서 얻고 다중 증상에서 잃는 절충이 있다. 학습형 중단은 초기 중첩 검증에서 단순 중단 기준 재선택 대비 추가 이득이 작았고, 후속 학습용 OOF 비교에서는 Dev **+0.00694**가 기록됐다. 서로 다른 단계의 결과이며 같은 Dev의 반복 선택 이력도 남는다. [절충·한계][S13], [후속 결과][S14]
- 공식 라벨이 전사에서 언급되지 않았다는 통계와 키워드 부재를 검증된 오라벨·인간 임상 판정으로 바꾸어 쓰지 않는다. 모델·입력 방향 전체가 포화됐다는 일반화도 현재 제한된 조건만으로 확정하지 않는다.

## 유망한 발전 방향과 다음 순서

**첫 단계는 GPU 없이 캐시 출처 감사와 비교 준비다.** Kc 32,840 캐시의 원천별 수·Training/Validation 교집합·정규화 전사 중복·가중치 계보를 읽기 전용으로 확인한다. 실물 접근 전에는 누수 여부를 확정하지 않고 기존 기록을 덮어쓰지 않는다.

후속 학습에서는 RoBERTa-base 대 KcELECTRA-base, seed42·43의 작은 고정 비교를 권한다. 두 백본 모두 각자의 **고정 revision 사전학습 모델에서 새로 시작**하고 같은 Training 전용 ID·prefix·입력 정책·배치 계획·학습 예산·손실·마지막 epoch 선택을 쓴다. 1만통화 Epoch2 이어학습 RoBERTa와 fresh Kc를 비교해 백본 단독 효과라고 부르지 않는다. 첫 비교에서 TAPT·화자·pos_weight까지 동시에 바꾸지 않으며 토큰화와 실제 연산량 차이를 기록한다.

평가는 **raw 0.5, Top-1 없음**으로 고정한다. 전체·단일·다중 점수, 빈 출력, FN/FP, seed별 차이와 통화 paired CI를 남긴다. 두 seed 개선·평균 +0.005 이상·CI 하한 양수·다중 점수 비악화 같은 기준은 새 확인셋을 보기 전에 동결한다. 이미 본 Dev는 개발용이며, 과거 보조 학습 후속에서 원 holdout512도 평가했으므로 holdout이라는 이름만으로 미사용이라 가정하지 않고 전체 평가 이력을 대조해 확인 표본을 예약한다.

기존 팀원 가중치부터 비교하려면 **fit-only 체크포인트의 학습 ID와 우리 학습 ID를 대조해 양쪽 모두 보지 않은 통화**를 찾아야 한다. 우리 Dev512가 팀원에게도 미사용이었다고 가정하면 안 된다. Training 전체로 학습한 최종 번들은 Training 내부 Dev 독립 비교 후보가 아니다.

clean 백본 비교 뒤에는 TAPT와 층별 학습률 감쇠를 한 요소씩 시험하거나, 두 계열 확률의 고정 평균 뒤 0.5를 적용하는 앙상블을 본다. 정책 개선은 별도 Training OOF 확률에서 학습하고, 정책 학습·선택에 쓰지 않은 통화에서 평가한다. 중단 신호 자체에 적용되는 규칙을 확인하기 전에는 maxent나 학습형 가변 기준을 보수적 제출 구성에 넣지 않는다.

현재 완료 범위는 **증거 검토와 다음 비교 설계**다. 새 학습·GPU 요청·가중치 교체는 하지 않았다. 다음 순서는 캐시 감사 → 비공개 분할·해시 확보 → 규칙 확인 → 사전 기준 동결 → 최종·재개 상태 보존 시험이다.

## 확인한 commit

| 브랜치 | commit |
|---|---|
| mission4 | `b1aaea0a2f89602d17c41baccf0edee5bd29dff5` |
| mission4-streaming | `355f5b2788e275dbed103220ce242e44423491f7` |
| mission4-prefix-speaker | `745b73ae932a310c1883fec21bb6de37e36ba597` |
| mission4-kcelectra | `a5b406ea523b6f7d87d2914b3931f42c2daaea51` |
| mission4-kcelectra-full | `e0a31fb8032396033b6b9927f5ed2076e70d06d9` |
| mission4-koelectra | `d10da59ce9be561673f6fc7d9fc6eca7c67d800c` |
| mission4-kogpt2 | `0088b5270548a386c4fa15be106ae322d87a0185` |
| mission4-krbert | `f382352d124a07306483b424dfa2ccf0cf6e53fa` |
| mission4-roberta-large | `8204005fcb5a7732aaea8cc64b3e780ab7173e27` |

[B1]: https://github.com/KozzilzzilE/DCC_Problem/blob/8204005fcb5a7732aaea8cc64b3e780ab7173e27/mission4_symptom/reports/mission4_progress_report.md#L60-L82
[S1]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/docs/results.csv
[S2]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/runs/E001_m3_baseline/metrics_dev.json
[S3]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/runs/E002_prefix_uniform/metrics_dev.json
[S4]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/runs/E016_final/val_check/score4-1.json
[S5]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/runs/E016_final/val_check/score4-2.json
[S6]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/scripts/build_cache.py#L23-L47
[S7]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/scripts/train_halt_policy.py#L75-L98
[S8]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/scripts/run_oof_tables.sh#L20-L55
[S9]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/m4/streaming.py#L61-L119
[S10]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/m4/halting.py#L66-L95
[S11]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/scripts/train_prefix.py#L187-L193
[S12]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/m4/decision.py#L132-L177
[S13]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/docs/PROGRESS_REPORT.md#L89-L108
[S14]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/runs/E012_halt_policy/ens3/result.json
[K1]: https://github.com/KozzilzzilE/DCC_Problem/blob/e0a31fb8032396033b6b9927f5ed2076e70d06d9/mission4_symptom/docs/EXPERIMENT_LOG.md#L598-L609
[K2]: https://github.com/KozzilzzilE/DCC_Problem/blob/a5b406ea523b6f7d87d2914b3931f42c2daaea51/mission4_symptom/KcELECTRA.ipynb#L155-L193
[K3]: https://github.com/KozzilzzilE/DCC_Problem/blob/a5b406ea523b6f7d87d2914b3931f42c2daaea51/mission4_symptom/KcELECTRA.ipynb#L598-L618
[R1]: https://github.com/KozzilzzilE/DCC_Problem/blob/745b73ae932a310c1883fec21bb6de37e36ba597/mission4_symptom/reports/mission4_progress_report.md#L18-L26
[R2]: https://github.com/KozzilzzilE/DCC_Problem/blob/355f5b2788e275dbed103220ce242e44423491f7/mission4_symptom/docs/PROGRESS_REPORT.md#L159-L165
[R3]: https://github.com/KozzilzzilE/DCC_Problem/blob/745b73ae932a310c1883fec21bb6de37e36ba597/mission4_symptom/tune_halting.py#L111-L139
