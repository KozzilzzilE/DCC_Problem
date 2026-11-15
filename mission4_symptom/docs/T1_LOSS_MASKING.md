# T1: 오심 음성 Loss 마스킹 (이전 명칭 T8-2)

## 목적 및 범위

라벨 오류를 확정하거나 원본 라벨을 수정하지 않는다. E002 학습의 오심 음성 BCE 항만
수동 검토된 근거가 있는 Prefix에서 제외한다. 다른 8종과 오심 양성 항은 항상 유지한다.
추론/앙상블/고정 E026/평가 함수는 바꾸지 않는다. Notebook 변경은 없다.

## Fit 후보 검토

기존 255건 후보를 동일 탐색식/동일 Fit 캐시 순서로 재구성했다. Dev/Validation 성능이나 대화는
조건 설계에 사용하지 않았다. 255건 모두의 오심 관련 발화와 인접 응답을 선별 검토했지만,
255건 전체 대화를 정밀 검토했다고 주장하지 않는다.

- 현재 시점의 명확한 오심성 진술을 가진 30건의 환자 귀속/질문·응답/시제 문맥을 추가 검토했다.
- 25건을 허용 목록으로 고정했다. 임의 통화에 정규식이 맞는다고 후보를 자동 확장하지 않는다.
- 5건은 호흡·가슴 불편과 울렁거림 의미 불명확 1, 당시 환자 확인 불충분 1,
  발화 주체 전환 불명확 1, 신고자/다른 환자 혼재 1, 환자 소개/일인칭 호소 혼재 1로 제외했다.
- 나머지 225건은 추가 문맥 검토 미완료로 제외했다. 오류가 없는 통화로 판정한 것이 아니다.
- 구역질/실제 구토 단독, 상담사의 재진술·질문, 부정·해소된 과거·다른 사람의 호소는 자동 포함하지 않는다.

검토 범위는 소스의 `DEEP_REVIEW`/`EXCLUDED`, 실행용 기록은 별도 manifest에 남는다.
manifest는 통화 ID·근거 발화 번호·발화 해시만 보존하며 개인정보 포함 원문을 복사하지 않는다.
이 작은 파일럿을 전체 누락 의심 사례에 대한 일반 규칙으로 해석하지 않는다.

## 실제 마스크 조건

오심 원 라벨0 + 동결된 Fit 허용 목록 + 검토된 현재 환자 발화 + 신고자 평서 진술을 모두 요구한다.
메스꺼움/울렁거림/토할 것 같은 느낌을 직접 호소한 경우를 대상으로 한다. 질문·부정·다른 사람·
해소된 과거가 있는 근거는 제외한다. 단순 구토/구역질만으로는 선택하지 않는다.
현재 Prefix까지 나온 근거만 사용하며, 뒤에만 있는 근거는 앞 Prefix를 마스킹하지 않는다.
관측된 후속 부정/해소/주체 전환은 이후 Prefix에서만 근거를 취소한다.
기존 512토큰 오른쪽 잘림으로 근거가 모델 입력에서 빠지면 마스킹하지 않는다.

Tokenizer의 offset만 조회하며 별도 분류 모델은 없다. Prefix 원문/토큰/샘플/라벨은 바꾸지 않는다.
마스크 계획 계산은 RNG를 호출하지 않는다. 실제 학습 루프의 items/order 생성은 기존 코드 그대로다.

```python
element_loss = BCEWithLogitsLoss(pos_weight=existing_pos_weight, reduction="none")(logits, labels)
loss = (element_loss * loss_mask).mean()
```

분모는 원래 B*9다. 공유 Encoder 파라미터 업데이트는 달라지므로 다른 증상의 최종 예측까지
동일하다고 주장하지 않는다. 기본 Baseline은 원래 reduction="mean" 경로다.
원소별 BCE + 외부 mean은 내장 mean과 수학적으로 같지만 역전파 연산 순서의 반올림 차이는 가능하다.

## 준비 결과 및 통계

이번 작성 시 `rulesvone`을 생성했다. 학습이나 모델 추론은 실행하지 않았다.
seed42, epoch별 2개 균등 샘플과 shuffle을 CPU에서 재생한 예상치:

| 항목 | 예상 |
|---|---:|
| 검토 허용 Fit 통화 | 25 |
| 실제 샘플링에서 마스킹되는 고유 통화 | 24 |
| Epoch1/2/3 마스킹 노출 | 32 / 37 / 30 |
| 전체 Prefix 노출 | 157,680 |
| 마스킹 노출 | 99 (0.062785%) |
| 오심 음성 Loss 중 비율 | 0.070959% |
| 전체 9종 Loss 원소 중 비율 | 0.006976% |
| 계획된 Optimizer Steps | 9,855 |

Prefix 길이 분포·epoch별 분모/비율·샘플링 해시는 `expected_masking_stats.json`에 있다.
실제 학습이 끝나면 별도의 `training/masking_stats.json`에 실제 카운터를 저장한다.
예상 규모가 작으므로 유의미한 성능 개선을 기대하거나 개선량을 미리 제시하지 않는다.

## 규정 확인

`README.md` 70~78행은 미래/전체 통화 길이 입력, Validation 학습, 라벨에 따른 전처리를 금지한다.
`EXPERIMENT_LOG.md` E024는 Pos Weight 등 손실 설계 허용에 대한 팀 전달 기록이다.
이번 구현은 입력 전처리/원본 라벨/평가 후처리를 바꾸지 않는 학습 Loss 설계지만,
선택적 Loss 마스킹의 명시적 허용 회신은 없다. 라벨별 전처리 금지의 적용 범위에 관해
주최측 확인이 필요하며, 이 문서는 허용 판정을 대신하지 않는다.

## 실행 명령 (PowerShell)

```powershell
cd D:\assignment\programing\DCC\DCC_Problem\mission4_symptom
$py = (Resolve-Path ..\.venv\Scripts\python.exe).Path
$manifest = '../outputs/mission4_prefix/t1_loss_masking/rulesvone/manifest.json'
```

명시적으로 실행을 결정한 뒤에만 전체 학습/배치/순차를 연속 실행한다:

```powershell
& $py -B scripts/prefix_experiment.py run --strategy t1 `
  --mask-manifest $manifest --all-stages `
  --reference ../outputs/mission4_prefix/localcontrol/metrics.json `
  --out ../outputs/mission4_prefix/t1_loss_masking/runone
```

학습 → 배치 확률 추출/평가 → 순차 추론/평가 순서이며 예외가 나면 다음 단계로 진행하지 않는다.
기존 Baseline `run --out ...`의 기본 동작은 그대로며 `--all-stages`는 명시적 옵션이다.
T1 활성화는 Runner `--strategy t1`과 학습기 `--t1-mask-manifest`로만 가능하다.

개별 단계(각각 출력 폴더가 아직 없을 때):

```powershell
# 학습만: 이미 성공한 모델이 있다면 다시 실행하지 않는다.
& $py -B scripts/prefix_experiment.py train --strategy t1 --mask-manifest $manifest `
  --out ../outputs/mission4_prefix/t1_loss_masking/trainonly

# 성공한 학습에서 배치 확률 추출/평가 재시도
& $py -B scripts/prefix_experiment.py extract `
  --e002-run ../outputs/mission4_prefix/t1_loss_masking/runone/training `
  --reference ../outputs/mission4_prefix/localcontrol/metrics.json `
  --out ../outputs/mission4_prefix/t1_loss_masking/batchretry

# 확률표까지 성공했고 평가만 실패한 경우
& $py -B scripts/prefix_experiment.py evaluate `
  --e002-table ../outputs/mission4_prefix/t1_loss_masking/runone/dev_prefix.npz `
  --reference ../outputs/mission4_prefix/localcontrol/metrics.json `
  --out ../outputs/mission4_prefix/t1_loss_masking/evalretry

# 순차만 재시도: 성공한 학습 경로를 지정한다.
& $py -B scripts/prefix_experiment.py sequential `
  --e002-run ../outputs/mission4_prefix/t1_loss_masking/runone/training `
  --out ../outputs/mission4_prefix/t1_loss_masking/seqretry --device cuda
```

학습 자체가 실패하면 기존 학습기의 optimizer/RNG 재개 Checkpoint가 없어 새 출력 폴더에서
처음부터 재실행해야 한다. 실패 파일을 지우거나 덮어쓰지 않는다.
`trainonly`를 사용했다면 후속 `--e002-run`도 `trainonly/training`으로 지정한다.

Fit 준비 기록을 새 이름으로 다시 만들 필요가 있을 때만 다음을 실행한다(기존 폴더 재사용 금지):

```powershell
& $py -B scripts/prepare_t1_mask.py --out ../outputs/mission4_prefix/t1_loss_masking/rulesvtwo
```

## 산출물

- `rulesvone/manifest.json`, `expected_masking_stats.json`: 동결된 Fit 검토·선정·제외·예상 통계.
- `runone/initial.json`: 설정·초기 가중치/Tokenizer/분할/소스/마스크 manifest 해시.
- `runone/training/`: 기존 모델/Tokenizer/학습 설정/로그와 실제 `masking_stats.json`.
- `runone/dev_prefix.npz`, `.meta.json`: 기존 형식의 확률표와 배치 추출 시간.
- `runone/metrics.json`, `halts.npz`: 기존 고정 정책의 배치 평가 및 세부 지표/중단 배열.
- `runone/sequential/sequential_timing.json`: 실제 순차 성능·세부 지표·시간.
- `runone/sequential/sequential_4-1.npz`, `sequential_4-2.npz`: 기존과 같은 배열 형식.

순차 평가를 성능 판단의 주 기준으로 사용한다. 순차 시간은 캐시 로딩 후 토큰화/분류기/정책 실행이며
WAV/JSON IO·번들 조립 제외, 로딩 시간은 별도다. 배치 추출 시간과 합치거나 대체하지 않는다.
기존 evaluator/Streaming 클래스/고정 모델/NPZ/정책/Notebook은 T1 로직으로 변경하지 않았다.
