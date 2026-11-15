#!/usr/bin/env bash
# E021 KcELECTRA(beomi/KcELECTRA-base-v2022, revision 8728557, TAPT 없음)를 E002 와 같은 자료·레시피로 학습하고 dev 로 판정한다.
# 팀원 KcELECTRA 점수는 Validation 을 학습에 넣은 결과라 비교 근거가 될 수 없어, Training 만으로 다시 잰다.
#   bash scripts/run_e021.sh
PY=../../../venv/Scripts/python
OUT=runs/E021_kcelectra
SNAP="C:/Users/Administrator/.cache/huggingface/hub/models--beomi--KcELECTRA-base-v2022/snapshots/87285578b4686b6d190abd40e947a2a416add6de"
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8

echo "=== E021 학습 ($(date +%H:%M)) ==="
$PY scripts/train_prefix.py --calls runs/cache/train.jsonl --ids runs/cache/fit_ids.txt \
    --backbone "$SNAP" --out "$OUT" --seed 42 --batch-size 16 --grad-accum 1 \
    || { echo "E021 학습 실패"; echo "=== 끝 ($(date +%H:%M)) ==="; exit 1; }

echo "=== 확률표 ($(date +%H:%M)) ==="
$PY scripts/prefix_probs.py --model "$OUT/best_model" --calls runs/cache/train.jsonl \
    --ids runs/cache/dev_ids.txt --out "$OUT/dev_prefix.npz" --fmt space \
    || { echo "E021 확률표 실패"; echo "=== 끝 ($(date +%H:%M)) ==="; exit 1; }

echo "=== 판정 ($(date +%H:%M)) ==="
$PY "$OUT/analyze.py" | tee "$OUT/analysis.txt"
$PY scripts/eval_combo.py --shift-from "$OUT" --tables "$OUT/dev_prefix.npz" \
    --maxent-prior-ids runs/cache/fit_ids.txt --out "$OUT/eval_single.json"
$PY scripts/eval_combo.py --shift-from "$OUT" \
    --tables runs/E001_m3_baseline/dev_prefix.npz "$OUT/dev_prefix.npz" runs/E007_prefix_uniform_s43/dev_prefix.npz \
    --maxent-prior-ids runs/cache/fit_ids.txt --out "$OUT/eval_swap.json"
$PY scripts/eval_combo.py --shift-from "$OUT" \
    --tables runs/E001_m3_baseline/dev_prefix.npz runs/E002_prefix_uniform/dev_prefix.npz runs/E007_prefix_uniform_s43/dev_prefix.npz "$OUT/dev_prefix.npz" \
    --maxent-prior-ids runs/cache/fit_ids.txt --out "$OUT/eval_add.json"
echo "E021 완료"
echo "=== 끝 ($(date +%H:%M)) ==="
