# reports/ 안내 — 무엇이 실제 결과이고 무엇이 아닌가

## 1. 대회 규정: 결정 임계값 0.5 고정

주최 측 규정으로 제출 시 결정 임계값은 **0.5 고정**이다. 따라서 이 폴더의 문서에 나오는
`optimized Macro F1`, `class-wise threshold` 수치는 **제출 성능이 아니라 연구 기록**이다.
제출 성능을 인용할 때는 각 run 의 `baseline_metrics.json` 에 있는 `val_macro_f1`
(threshold 0.5 기준)을 쓴다.

제출 경로(`m3/infer.py`, `inference.py`)는 이 폴더의 어떤 파일도 읽지 않는다.

## 2. 실제 학습 이전에 만들어진 synthetic 산출물

아래 두 파일은 **실제 모델 학습 전에 synthetic validation 데이터로 생성**된 것이다.
실제 성능으로 해석하면 안 되고, 추론에 사용해서도 안 된다.

| 파일 | 내용 | 주의 |
|---|---|---|
| `comparison.md` | Macro F1 `0.8583 -> 0.9231` | synthetic 데이터 결과. 실제 baseline 은 F1@0.5 `0.600329` |
| `best_thresholds.json` | 임계값 `0.51~0.68` | synthetic 값. 실제 탐색값은 `0.19~0.51` 로 완전히 다르며, 애초에 규정상 사용 불가 |

특히 `best_thresholds.json` 은 이름과 형식이 그럴듯해서 추론 코드가 무심코 읽기 쉽다.
읽는 코드를 추가하지 마라.

## 3. 실제 결과 문서

| 파일 | 내용 |
|---|---|
| `error_analysis.md` | 9개 클래스 전체 오류 분석, 다중 라벨 FN 증가, FP sink |
| `truncation_eval.md` | 512 토큰 절단 복구 실험 (개선 없음) |
| `label_dependency_eval.md` | Label Dependency Loss 검증 (개선 없음) |
| `pairwise_eval.md` | Pairwise Ranking Loss 검증 (개선 없음) |
| `calibration_eval.md` | 임계값 0.5 고정 기준 pos_weight 거듭제곱·발화 경계·오심·TF-IDF 블렌드 검증 (F1@0.5 0.5967 → 0.6496, 블렌드 0.6536~0.6546) |
