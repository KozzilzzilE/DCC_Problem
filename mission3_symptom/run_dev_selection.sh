#!/usr/bin/env bash
# Mission 3: Training 내부 dev 로 하이퍼파라미터를 정하고, Training 전체로 최종 번들을 만든 뒤
# Validation 을 마지막에 한 번만 추론해 성능을 확인한다. 규칙은 reports/dev_selection_protocol.md.
#
#   bash run_dev_selection.sh <mission3_train.csv> <validation label 폴더>
#
# - 1~9 단계는 Validation 을 읽지 않는다. 10 단계만 Validation 을 읽고, 한 번 끝나면 다시 돌지 않는다.
# - 단계마다 완료 표식(<ROOT>/.done/<단계>)을 남겨, 끊긴 뒤 다시 실행하면 이어서 한다.
# - 환경 변수: PYTHON(기본 python), DEVSEL_ROOT(기본 runs/devsel), TAPT_EPOCHS(기본 20),
#   MAX_LENGTH(기본 512), BASE_MODEL(기본 klue/roberta-base).
set -euo pipefail
cd "$(dirname "$0")"

TRAIN_CSV=${1:?Training CSV 경로가 필요합니다}
VAL_LABEL_DIR=${2:?Validation label 폴더가 필요합니다 (10 단계 확인용)}
PY=${PYTHON:-python}
ROOT=${DEVSEL_ROOT:-runs/devsel}
TAPT_EPOCHS=${TAPT_EPOCHS:-20}
MAX_LENGTH=${MAX_LENGTH:-512}
BASE_MODEL=${BASE_MODEL:-klue/roberta-base}
POWERS=(0 0.5 1.0)

export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$ROOT/.done" "$ROOT/decisions" "$ROOT/logs"
DATA="$ROOT/data"
TRAIN_SPLIT="$DATA/train_split.csv"
DEV_SPLIT="$DATA/dev_split.csv"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$ROOT/logs/queue.log"; }
done_mark() { touch "$ROOT/.done/$1"; }
is_done() { [ -f "$ROOT/.done/$1" ]; }

# 분류 학습 공통 인자 (원래 레시피 기본값: lr 2e-5, 3 epoch, 배치 8 x 누적 2, 512 토큰, AMP)
common_train=(--max-length "$MAX_LENGTH" --amp --local-files-only --use-pos-weight --epochs 3)

train_dev() {  # 학습용 분할로 학습, dev 분할로 epoch 별 평가 (선택 단계 전용)
  local name=$1; shift
  is_done "$name" && { log "건너뜀 $name"; return; }
  rm -rf "$ROOT/$name"
  log "시작 $name"
  "$PY" -B train.py --train-csv "$TRAIN_SPLIT" --val-csv "$DEV_SPLIT" --output-dir "$ROOT/$name" \
    --checkpoint-metric val_macro_f1 "${common_train[@]}" "$@" > "$ROOT/logs/$name.log" 2>&1
  done_mark "$name"; log "끝 $name"
}

# 1. Training 내부 dev 분할 (Validation 미사용)
if ! is_done split; then
  log "1. dev 분할"
  "$PY" -B make_dev_split.py --train-csv "$TRAIN_CSV" --out-dir "$DATA" | tee -a "$ROOT/logs/queue.log"
  done_mark split
fi

# 2. pos_weight power p 후보: 원래 레시피, seed 42
for p in "${POWERS[@]}"; do
  train_dev "p${p}_s42" --model-name-or-path "$BASE_MODEL" --pos-weight-power "$p" --seed 42
done

# 3. p 결정
run_args=()
for p in "${POWERS[@]}"; do run_args+=(--run "$p=$ROOT/p${p}_s42"); done
if ! is_done decide_power; then
  "$PY" -B devsel.py power --train-csv "$TRAIN_SPLIT" --dev-csv "$DEV_SPLIT" "${run_args[@]}" \
    --out "$ROOT/decisions/power.json" > "$ROOT/logs/decide_power.log"
  done_mark decide_power
fi
P=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['choice'])" "$ROOT/decisions/power.json")
log "3. 결정 p=$P"

# 4. TAPT: 학습용 분할 본문만으로 MLM (평가 CSV 없음)
if ! is_done tapt; then
  rm -rf "$ROOT/tapt"
  log "4. TAPT ${TAPT_EPOCHS} epoch 시작"
  "$PY" -B tapt_mlm.py --train-csv "$TRAIN_SPLIT" --output-dir "$ROOT/tapt" --model-name-or-path "$BASE_MODEL" \
    --epochs "$TAPT_EPOCHS" --max-length "$MAX_LENGTH" --amp --local-files-only > "$ROOT/logs/tapt.log" 2>&1
  done_mark tapt; log "4. TAPT 끝"
fi

# 5. 레시피 후보 두 개 x seed 2개 (원래 레시피 seed 42 는 2 단계 run 재사용)
train_dev "orig_s43" --model-name-or-path "$BASE_MODEL" --pos-weight-power "$P" --seed 43
for s in 42 43; do
  train_dev "tl_s$s" --model-name-or-path "$ROOT/tapt" --learning-rate 5e-5 --llrd-decay 0.8 --pos-weight-power "$P" --seed "$s"
done

# 6. 레시피·epoch·TF-IDF(C, w)·앙상블 크기 결정
if ! is_done decide_final; then
  "$PY" -B devsel.py final --train-csv "$TRAIN_SPLIT" --dev-csv "$DEV_SPLIT" --power-decision "$ROOT/decisions/power.json" \
    --arm "original=$ROOT/p${P}_s42,$ROOT/orig_s43" --arm "tapt_llrd=$ROOT/tl_s42,$ROOT/tl_s43" \
    --out "$ROOT/decisions/final.json" > "$ROOT/logs/decide_final.log"
  done_mark decide_final
fi
read -r RECIPE EPOCH C W NMEM < <("$PY" -c "
import json,sys
d=json.load(open(sys.argv[1], encoding='utf-8'))['decision']
print(d['recipe'], d['epoch'], d['tfidf_C'], d['tfidf_weight'], d['n_members'])" "$ROOT/decisions/final.json")
log "6. 결정 recipe=$RECIPE epoch=$EPOCH C=$C w=$W members=$NMEM"

# 7. 최종 학습: Training 전체, 정한 epoch 에서 저장 (dev 분할은 진행 기록용, 선택에 쓰지 않음)
if [ "$RECIPE" = "tapt_llrd" ]; then
  recipe_args=(--model-name-or-path "$ROOT/tapt" --learning-rate 5e-5 --llrd-decay 0.8)
else
  recipe_args=(--model-name-or-path "$BASE_MODEL")
fi
members=()
for ((i = 0; i < NMEM; i++)); do
  s=$((42 + i)); name="final_s$s"; members+=(--member "$ROOT/$name")
  if ! is_done "$name"; then
    rm -rf "$ROOT/$name"
    log "7. 시작 $name"
    "$PY" -B train.py --train-csv "$TRAIN_CSV" --val-csv "$DEV_SPLIT" --output-dir "$ROOT/$name" \
      --checkpoint-metric fixed_epoch --checkpoint-epoch "$EPOCH" "${common_train[@]}" "${recipe_args[@]}" \
      --pos-weight-power "$P" --seed "$s" > "$ROOT/logs/$name.log" 2>&1
    done_mark "$name"; log "7. 끝 $name"
  fi
done

# 8. 최종 TF-IDF: Training 전체, 정한 C
tfidf_args=()
if "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "$W"; then
  if ! is_done final_tfidf; then
    rm -rf "$ROOT/final_tfidf"
    "$PY" -B train_tfidf_member.py --train-csv "$TRAIN_CSV" --output "$ROOT/final_tfidf/tfidf_lr.joblib" --C "$C" \
      > "$ROOT/logs/final_tfidf.log" 2>&1
    done_mark final_tfidf
  fi
  tfidf_args=(--tfidf "$ROOT/final_tfidf/tfidf_lr.joblib" --tfidf-weight "$W")
fi

# 9. 번들 조립 (fp16)
if ! is_done bundle; then
  rm -rf "$ROOT/bundle"
  "$PY" -B devsel.py assemble "${members[@]}" "${tfidf_args[@]}" --out "$ROOT/bundle" > "$ROOT/logs/bundle.log"
  done_mark bundle; log "9. 번들 $ROOT/bundle"
fi

# 10. Validation 확인 1회 (결정이 모두 끝난 뒤. 다시 돌지 않는다)
if is_done validation_once; then
  log "10. Validation 확인은 이미 한 번 했습니다: $ROOT/decisions/validation_once.json"
else
  touch "$ROOT/.done/validation_once"
  log "10. Validation 확인 시작"
  "$PY" -B inference.py --label_dir "$VAL_LABEL_DIR" --ckpt_path "$ROOT/bundle/ensemble.json" \
    --output "$ROOT/validation_once.csv" > "$ROOT/logs/validation_once.log" 2>&1
  "$PY" -B devsel.py score --pred "$ROOT/validation_once.csv" --label-dir "$VAL_LABEL_DIR" \
    --out "$ROOT/decisions/validation_once.json" > /dev/null
  log "10. 끝: $("$PY" -c "import json,sys; print('macro F1@0.5', round(json.load(open(sys.argv[1], encoding='utf-8'))['macro_f1'], 4))" "$ROOT/decisions/validation_once.json")"
fi
