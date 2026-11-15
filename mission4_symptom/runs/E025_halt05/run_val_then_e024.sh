#!/usr/bin/env bash
# E025 제출 번들 Validation 확인 (사용자 승인 11-14, 1회, 값 변경 없음) → 이어서 E024 pos_weight 실험.
# GPU 를 같이 쓰지 않도록 E022 가 끝날 때까지 기다린다. 공식 명령을 제출 폴더에서 그대로 실행한다.
W=$(pwd)
PY="C:/Users/Administrator/Desktop/DCC_Problem/venv/Scripts/python"
SUB="C:/Users/Administrator/Desktop/DCC_submission/mission4_symptom"
DATA="C:/Users/Administrator/Desktop/DCC_Problem/data/val"
OUT="$W/runs/E025_halt05/val_check"
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8

echo "=== E022 끝나기 기다림 ($(date +%H:%M)) ==="
until grep -q "^=== 끝" runs/E022_kcelectra_tapt/run.log 2>/dev/null; do sleep 60; done

echo "=== Validation 확인 시작 ($(date +%H:%M)) ==="
sha256sum "$SUB/ckpt/mission4.pt" > "$OUT/bundle.sha256"
for M in 4-1 4-2; do
  S=$(date +%s)
  (cd "$SUB" && "$PY" inference.py --mission $M --audio_dir "$DATA/audio" --label_dir "$DATA/label" \
      --ckpt_path ckpt/mission4.pt --device cuda --output "$OUT/mission$M.csv") > "$OUT/run$M.log" 2>&1 \
      || echo "추론 실패 $M"
  echo "$M 추론 $(( $(date +%s) - S ))초" >> "$OUT/elapsed.txt"
  (cd "$SUB" && "$PY" score.py --mission $M --label_dir "$DATA/label" --audio_dir "$DATA/audio" \
      --pred "$OUT/mission$M.csv" --out_json "$OUT/score$M.json") > "$OUT/score$M.log" 2>&1 \
      || echo "채점 실패 $M"
  tail -5 "$OUT/score$M.log"
done
echo "=== Validation 확인 끝 ($(date +%H:%M)) ==="

bash scripts/run_e024.sh > runs/E024_posweight/run.log 2>&1
echo "=== E024 끝 ($(date +%H:%M)) ==="
