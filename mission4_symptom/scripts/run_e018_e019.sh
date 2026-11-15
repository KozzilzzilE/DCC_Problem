#!/usr/bin/env bash
# E018 화자 표시 입력, E019 큰 백본(klue/roberta-large) 을 차례로 학습하고 dev 로 판정한다.
# 둘 다 E002 와 같은 자료(fit 26,280통화)·같은 레시피이고 바뀌는 것은 표시한 한 가지뿐이다.
#   bash scripts/run_e018_e019.sh
PY=../../../venv/Scripts/python
TAPT=../../mission3-calibration/mission3_symptom/runs/devsel/tapt
PRIOR="--maxent-prior-ids runs/cache/fit_ids.txt"

run_eval () {            # $1 = 실험 폴더, $2 = fmt
  $PY scripts/prefix_probs.py --model "$1/best_model" --calls runs/cache/train.jsonl \
      --ids runs/cache/dev_ids.txt --out "$1/dev_prefix.npz" --fmt "$2" || return 1
  # 단독 성능
  $PY scripts/eval_combo.py --shift-from "$1" --tables "$1/dev_prefix.npz" $PRIOR \
      --out "$1/eval_single.json" || return 1
  # 현재 제출 후보(3개 평균)에서 E002 자리를 이 모델로 바꾼 경우
  $PY scripts/eval_combo.py --shift-from "$1" \
      --tables runs/E001_m3_baseline/dev_prefix.npz "$1/dev_prefix.npz" runs/E007_prefix_uniform_s43/dev_prefix.npz \
      $PRIOR --out "$1/eval_swap.json" || return 1
  # 현재 3개 평균에 추가한 경우 (4개)
  $PY scripts/eval_combo.py --shift-from "$1" \
      --tables runs/E001_m3_baseline/dev_prefix.npz runs/E002_prefix_uniform/dev_prefix.npz \
      runs/E007_prefix_uniform_s43/dev_prefix.npz "$1/dev_prefix.npz" \
      $PRIOR --out "$1/eval_add.json" || return 1
}

echo "=== E018 화자 표시 ($(date +%H:%M)) ==="
if $PY scripts/train_prefix.py --calls runs/cache/train.jsonl --ids runs/cache/fit_ids.txt \
      --backbone "$TAPT" --out runs/E018_speaker --fmt speaker --seed 42; then
  run_eval runs/E018_speaker speaker && echo "E018 완료" || echo "E018 판정 실패"
else
  echo "E018 학습 실패"
fi

echo "=== E019 큰 백본 klue/roberta-large ($(date +%H:%M)) ==="
# 파라미터가 3배라 한 묶음을 4로 줄이고 기울기를 4번 모아 같은 유효 묶음 16 을 쓴다.
if $PY scripts/train_prefix.py --calls runs/cache/train.jsonl --ids runs/cache/fit_ids.txt \
      --backbone klue/roberta-large --out runs/E019_large --seed 42 \
      --batch-size 4 --grad-accum 4; then
  run_eval runs/E019_large space && echo "E019 완료" || echo "E019 판정 실패"
else
  echo "E019 학습 실패"
fi
echo "=== 끝 ($(date +%H:%M)) ==="
