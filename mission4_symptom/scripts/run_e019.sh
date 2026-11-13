#!/usr/bin/env bash
# E019 큰 백본(klue/roberta-large, TAPT 없음)을 E002 와 같은 자료·레시피·묶음 16 으로 학습하고 dev 로 판정한다.
# 8GB 에 맞추려고 gradient checkpointing 을 켠다 (계산 결과는 같고 메모리만 줄어든다).
#   bash scripts/run_e019.sh
PY=../../../venv/Scripts/python
OUT=runs/E019_large_gc
export HF_HUB_OFFLINE=1

echo "=== E019 학습 ($(date +%H:%M)) ==="
$PY scripts/train_prefix.py --calls runs/cache/train.jsonl --ids runs/cache/fit_ids.txt \
    --backbone klue/roberta-large --out "$OUT" --seed 42 --batch-size 16 --grad-accum 1 --grad-ckpt \
    || { echo "E019 학습 실패"; echo "=== 끝 ($(date +%H:%M)) ==="; exit 1; }

echo "=== 확률표 ($(date +%H:%M)) ==="
$PY scripts/prefix_probs.py --model "$OUT/best_model" --calls runs/cache/train.jsonl \
    --ids runs/cache/dev_ids.txt --out "$OUT/dev_prefix.npz" --fmt space \
    || { echo "E019 확률표 실패"; echo "=== 끝 ($(date +%H:%M)) ==="; exit 1; }

echo "=== 판정 ($(date +%H:%M)) ==="
$PY "$OUT/analyze.py" | tee "$OUT/analysis.txt"
$PY scripts/eval_combo.py --shift-from "$OUT" --tables "$OUT/dev_prefix.npz" \
    --maxent-prior-ids runs/cache/fit_ids.txt --out "$OUT/eval_single.json"
$PY scripts/eval_combo.py --shift-from "$OUT" \
    --tables runs/E001_m3_baseline/dev_prefix.npz "$OUT/dev_prefix.npz" runs/E007_prefix_uniform_s43/dev_prefix.npz \
    --maxent-prior-ids runs/cache/fit_ids.txt --out "$OUT/eval_swap.json"
echo "E019 완료"
echo "=== 끝 ($(date +%H:%M)) ==="
