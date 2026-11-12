#!/usr/bin/env bash
# 학습용 통화(fit 26,280건)의 교차 학습(out-of-fold) 누적 확률표를 만든다.
#
# fit 통화를 5겹으로 나누고, 겹마다 나머지 4겹으로 E002 레시피 모델을 학습한 뒤 그 모델이 보지 않은
# 겹의 모든 누적 입력에 대해 확률표를 뽑는다. 이렇게 하면 "모델이 보지 않은 통화에서의 확률"이
# 학습용 통화 전체에 대해 생기므로, 중단 정책처럼 확률 위에 얹는 2단계 모델을 dev 를 건드리지 않고
# 학습할 수 있다. dev 2,920통화는 판정에만 쓴다. Validation 은 쓰지 않는다.
#
#   bash scripts/run_oof_tables.sh          # 끝난 겹은 건너뛴다
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PY="${PYTHON:-../../../venv/Scripts/python}"
OUT="${OOF_ROOT:-runs/E011_oof}"
BACKBONE="${BACKBONE:-../../mission3-calibration/mission3_symptom/runs/devsel/tapt}"
FOLDS=5
mkdir -p "$OUT/logs"

# 겹 나누기: fit 목록을 seed 2026 으로 섞어 5등분한다 (통화 단위, 한 번 만들면 다시 만들지 않는다)
if [ ! -f "$OUT/fold0_held.txt" ]; then
  "$PY" - "$OUT" "$FOLDS" <<'PYEOF'
import sys
from pathlib import Path
import numpy as np
out, folds = Path(sys.argv[1]), int(sys.argv[2])
ids = [l.strip() for l in open("runs/cache/fit_ids.txt", encoding="utf-8") if l.strip()]
fold = np.random.default_rng(2026).permutation(len(ids)) % folds
for k in range(folds):
    held = [i for i, f in zip(ids, fold) if f == k]
    fit = [i for i, f in zip(ids, fold) if f != k]
    (out / f"fold{k}_held.txt").write_text("\n".join(held) + "\n", encoding="utf-8")
    (out / f"fold{k}_fit.txt").write_text("\n".join(fit) + "\n", encoding="utf-8")
    print(f"fold {k}: 학습 {len(fit)}통화, 보류 {len(held)}통화")
PYEOF
fi

for k in $(seq 0 $((FOLDS - 1))); do
  RUN="$OUT/fold$k"
  if [ -f "$RUN/oof_prefix.npz" ]; then
    echo "[$(date +%H:%M:%S)] fold $k: 이미 끝남, 건너뜀" | tee -a "$OUT/logs/queue.log"
    continue
  fi
  echo "[$(date +%H:%M:%S)] fold $k: 학습 시작" | tee -a "$OUT/logs/queue.log"
  rm -rf "$RUN"
  # 레시피는 E002 와 같다 (앞부분 균등 표본 2개 x 3 epoch, LLRD 0.8, lr 5e-5, pos_weight 지수 0.5, seed 42)
  "$PY" scripts/train_prefix.py --calls runs/cache/train.jsonl --ids "$OUT/fold${k}_fit.txt" \
      --backbone "$BACKBONE" --out "$RUN" > "$OUT/logs/fold$k.train.log" 2>&1
  echo "[$(date +%H:%M:%S)] fold $k: 보류 통화 확률표 계산" | tee -a "$OUT/logs/queue.log"
  "$PY" scripts/prefix_probs.py --model "$RUN/best_model" --calls runs/cache/train.jsonl \
      --ids "$OUT/fold${k}_held.txt" --out "$RUN/oof_prefix.tmp.npz" > "$OUT/logs/fold$k.probs.log" 2>&1
  mv "$RUN/oof_prefix.tmp.npz" "$RUN/oof_prefix.npz"
  mv "$RUN/oof_prefix.tmp.meta.json" "$RUN/oof_prefix.meta.json"
  rm -rf "$RUN/best_model"          # 가중치(약 440MB)는 확률표를 뽑은 뒤 지운다. 설정과 로그는 남긴다
  echo "[$(date +%H:%M:%S)] fold $k: 끝" | tee -a "$OUT/logs/queue.log"
done
echo "[$(date +%H:%M:%S)] 전체 끝" | tee -a "$OUT/logs/queue.log"
