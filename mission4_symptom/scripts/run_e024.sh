#!/usr/bin/env bash
# E024 pos_weight 지수 실험 (판정 임계값 0.5 고정 대응). E002 와 같은 레시피에서 지수만 0.3 / 0.75 로 바꿔 학습하고 dev 로 판정한다.
# GPU 를 같이 쓰면 메모리가 넘치므로 E022 가 끝날 때까지 기다렸다가 시작한다.
#   bash scripts/run_e024.sh
PY=../../../venv/Scripts/python
OUT=runs/E024_posweight
TAPT=../../mission3-calibration/mission3_symptom/runs/devsel/tapt     # E002 와 같은 TAPT 백본
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8

echo "=== E022 끝나기 기다림 ($(date +%H:%M)) ==="
until grep -q "^=== 끝" runs/E022_kcelectra_tapt/run.log 2>/dev/null; do sleep 60; done

for SPEC in "0.3 p030" "0.75 p075"; do
  set -- $SPEC; P=$1; TAG=$2
  echo "=== 학습 pos_weight 지수 $P → $OUT/$TAG ($(date +%H:%M)) ==="
  if [ ! -f "$OUT/$TAG/best_model/model.safetensors" ]; then
    $PY scripts/train_prefix.py --calls runs/cache/train.jsonl --ids runs/cache/fit_ids.txt \
        --backbone "$TAPT" --out "$OUT/$TAG" --seed 42 --batch-size 16 --grad-accum 1 --pos-weight-power "$P" \
        || { echo "학습 실패 $P"; echo "=== 끝 ($(date +%H:%M)) ==="; exit 1; }
  fi
  $PY scripts/prefix_probs.py --model "$OUT/$TAG/best_model" --calls runs/cache/train.jsonl \
      --ids runs/cache/dev_ids.txt --out "$OUT/$TAG/dev_prefix.npz" --fmt space \
      || { echo "확률표 실패 $P"; echo "=== 끝 ($(date +%H:%M)) ==="; exit 1; }
done

echo "=== 판정 ($(date +%H:%M)) ==="
$PY "$OUT/analyze.py" | tee "$OUT/analysis.txt"
echo "=== 끝 ($(date +%H:%M)) ==="
