#!/usr/bin/env bash
# Mission 3: Training 내부 dev 로 하이퍼파라미터를 정하고, Training 전체로 최종 번들을 만든 뒤
# Validation 을 마지막에 한 번만 추론해 성능을 확인한다. 규칙은 reports/dev_selection_protocol.md.
#
#   bash run_dev_selection.sh <mission3_train.csv> <validation label 폴더>
#
# - 1~9 단계는 Validation 을 읽지 않는다. 10 단계만 Validation 을 읽고, 한 번 끝나면 다시 돌지 않는다.
# - 단계마다 완료 표식(<ROOT>/.done/<단계>)을 남겨, 끊긴 뒤 다시 실행하면 이어서 한다.
#   어떤 단계를 실제로 다시 돌리면 그 뒤 단계의 표식은 모두 지워 함께 다시 돌린다 (옛 결과가 섞이지 않게).
#   다만 10 단계(Validation 확인)를 한 번이라도 시작한 ROOT 에서는 앞 단계를 다시 돌리지 않고 멈춘다
#   (Validation 을 본 뒤 번들이 바뀌지 않게). 설정을 바꿔 다시 하려면 새 DEVSEL_ROOT 를 쓴다.
# - 환경 변수: PYTHON(기본 python), DEVSEL_ROOT(기본 runs/devsel), TAPT_EPOCHS(기본 20),
#   MAX_LENGTH(기본 512), BASE_MODEL(기본 klue/roberta-base).
set -euo pipefail

# 인자는 cd 전에 절대 경로로 바꾸고 바로 확인한다 (틀린 경로가 5시간 뒤 10 단계에서야 드러나지 않게)
TRAIN_CSV=${1:?Training CSV 경로가 필요합니다}
VAL_LABEL_DIR=${2:?Validation label 폴더가 필요합니다 (10 단계 확인용)}
[ -f "$TRAIN_CSV" ] || { echo "Training CSV 가 없습니다: $TRAIN_CSV" >&2; exit 1; }
[ -d "$VAL_LABEL_DIR" ] || { echo "label 폴더가 없습니다: $VAL_LABEL_DIR" >&2; exit 1; }
abs_dir() { (cd "$1" && (pwd -W 2>/dev/null || pwd)); }   # Git Bash 는 Windows 경로(C:/...), 그 밖은 pwd
TRAIN_CSV="$(abs_dir "$(dirname "$TRAIN_CSV")")/$(basename "$TRAIN_CSV")"
VAL_LABEL_DIR="$(abs_dir "$VAL_LABEL_DIR")"
compgen -G "$VAL_LABEL_DIR/*.json" > /dev/null || { echo "label 폴더에 JSON 이 없습니다: $VAL_LABEL_DIR" >&2; exit 1; }
cd "$(dirname "$0")"
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
RESULT="$ROOT/decisions/validation_once.json"   # 10 단계 결과

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$ROOT/logs/queue.log"; }
done_mark() { touch "$ROOT/.done/$1"; }
is_done() { [ -f "$ROOT/.done/$1" ]; }

# 1~9 단계 표식을 실행 순서대로 둔다. begin_stage 는 단계를 실제로 돌리기 직전에 부른다.
STAGES=(split)
for p in "${POWERS[@]}"; do STAGES+=("p${p}_s42"); done
STAGES+=(decide_power tapt orig_s43 tl_s42 tl_s43 decide_final final_s42 final_s43 final_s44 final_s45 final_tfidf bundle_pt)
begin_stage() {
  local name=$1 seen=0 s
  # 결과가 나온 확인뿐 아니라 결과 없이 끝난 시도(validation_started)도 막는다. Validation 예측 파일이 이미 남았을 수 있다.
  if [ -f "$RESULT" ] || is_done validation_started; then
    log "$name 을 다시 돌려야 하지만 이 ROOT 는 Validation 확인을 이미 시작했습니다 (logs/validation_attempts.log). 확인한 번들이 바뀌지 않게 멈춥니다. 새 DEVSEL_ROOT 를 쓰세요"
    exit 1
  fi
  for s in "${STAGES[@]}"; do
    if [ "$seen" = 1 ] && is_done "$s"; then
      rm -f "$ROOT/.done/$s"; log "  뒤 단계 표식 지움: $s ($name 을 다시 돌리므로)"
    fi
    if [ "$s" = "$name" ]; then seen=1; fi
  done
}

# 분류 학습 공통 인자 (원래 레시피 기본값: lr 2e-5, 3 epoch, 배치 8 x 누적 2, 512 토큰, AMP)
common_train=(--max-length "$MAX_LENGTH" --amp --local-files-only --use-pos-weight --epochs 3)

train_dev() {  # 학습용 분할로 학습, dev 분할로 epoch 별 평가 (선택 단계 전용)
  local name=$1; shift
  is_done "$name" && { log "건너뜀 $name"; return; }
  begin_stage "$name"
  rm -rf "$ROOT/$name"
  log "시작 $name"
  "$PY" -B train.py --train-csv "$TRAIN_SPLIT" --val-csv "$DEV_SPLIT" --output-dir "$ROOT/$name" \
    --checkpoint-metric val_macro_f1 "${common_train[@]}" "$@" > "$ROOT/logs/$name.log" 2>&1
  done_mark "$name"; log "끝 $name"
}

# 1. Training 내부 dev 분할 (Validation 미사용)
if ! is_done split; then
  begin_stage split
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
  begin_stage decide_power
  "$PY" -B devsel.py power --train-csv "$TRAIN_SPLIT" --dev-csv "$DEV_SPLIT" "${run_args[@]}" \
    --out "$ROOT/decisions/power.json" > "$ROOT/logs/decide_power.log"
  done_mark decide_power
fi
# Windows 의 python 출력은 줄 끝이 CRLF 라 CR 을 지운다 (남으면 경로·산술에서 깨진다)
P=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['choice'])" "$ROOT/decisions/power.json" | tr -d '\r')
log "3. 결정 p=$P"

# 4. TAPT: 학습용 분할 본문만으로 MLM (평가 CSV 없음)
if ! is_done tapt; then
  begin_stage tapt
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
  begin_stage decide_final
  "$PY" -B devsel.py final --train-csv "$TRAIN_SPLIT" --dev-csv "$DEV_SPLIT" --power-decision "$ROOT/decisions/power.json" \
    --arm "original=$ROOT/p${P}_s42,$ROOT/orig_s43" --arm "tapt_llrd=$ROOT/tl_s42,$ROOT/tl_s43" \
    --out "$ROOT/decisions/final.json" > "$ROOT/logs/decide_final.log"
  done_mark decide_final
fi
read -r RECIPE EPOCH C W NMEM < <("$PY" -c "
import json,sys
d=json.load(open(sys.argv[1], encoding='utf-8'))['decision']
print(d['recipe'], d['epoch'], d['tfidf_C'], d['tfidf_weight'], d['n_members'])" "$ROOT/decisions/final.json" | tr -d '\r')
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
    begin_stage "$name"
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
    begin_stage final_tfidf
    rm -rf "$ROOT/final_tfidf"
    "$PY" -B train_tfidf_member.py --train-csv "$TRAIN_CSV" --output "$ROOT/final_tfidf/tfidf_lr.joblib" --C "$C" \
      > "$ROOT/logs/final_tfidf.log" 2>&1
    done_mark final_tfidf
  fi
  tfidf_args=(--tfidf "$ROOT/final_tfidf/tfidf_lr.joblib" --tfidf-weight "$W")
fi

# 9. 번들 조립: 최종 멤버 가중치(fp32)·tokenizer·config (+TF-IDF) 를 .pt 파일 하나로, 추론 정밀도 fp16
#  - 완료 표식은 bundle_pt 다. 10 단계를 시작하기 전이라면, 표식이 없거나(예전 폴더 번들 시절의 표식 bundle 만
#    남은 ROOT 포함) .pt 파일이 없을 때 다시 만든다. 10 단계를 시작한 뒤에는 다시 만들지 않고 멈춘다.
if ! is_done bundle_pt || [ ! -f "$ROOT/mission3.pt" ]; then
  if [ ! -f "$ROOT/mission3.pt" ] && { [ -f "$RESULT" ] || is_done validation_started; }; then
    log "9. $ROOT/mission3.pt 가 없습니다. Validation 확인을 시작한 ROOT 라 다시 만들지 않습니다. 복사해 둔 ckpt/mission3.pt 를 이 경로로 되돌리거나 새 DEVSEL_ROOT 를 쓰세요"
    exit 1
  fi
  begin_stage bundle_pt
  rm -f "$ROOT/mission3.pt"
  "$PY" -B devsel.py assemble "${members[@]}" "${tfidf_args[@]}" --out "$ROOT/mission3.pt" > "$ROOT/logs/bundle.log"
  done_mark bundle_pt; log "9. 번들 $ROOT/mission3.pt"
fi

# 10. Validation 확인 1회 (결정이 모두 끝난 뒤. 결과가 나온 뒤에는 다시 돌지 않는다)
#  - 시도마다 시각·번들 해시·label 폴더를 logs/validation_attempts.log 에 덧붙여, 몇 번 열었는지 남긴다.
#  - 결과 없이 끝난 시도가 있으면 멈춘다. 원인을 확인하고 .done/validation_started 를 지운 뒤 다시 실행한다.
if [ -f "$RESULT" ]; then
  log "10. Validation 확인은 이미 했습니다: $RESULT"
elif is_done validation_started; then
  log "10. 이전 Validation 시도가 결과 없이 끝났습니다. logs/validation_attempts.log 와 로그를 확인한 뒤 .done/validation_started 를 지우고 다시 실행하세요"
  exit 1
else
  stamp=$(date '+%Y%m%d_%H%M%S')
  {
    echo "== attempt $stamp label_dir=$VAL_LABEL_DIR"
    sha256sum "$ROOT/mission3.pt" 2>/dev/null || true
  } >> "$ROOT/logs/validation_attempts.log"
  done_mark validation_started
  log "10. Validation 확인 시작 ($stamp)"
  "$PY" -B inference.py --label_dir "$VAL_LABEL_DIR" --ckpt_path "$ROOT/mission3.pt" \
    --output "$ROOT/validation_once_$stamp.csv" > "$ROOT/logs/validation_once_$stamp.log" 2>&1
  "$PY" -B devsel.py score --pred "$ROOT/validation_once_$stamp.csv" --label-dir "$VAL_LABEL_DIR" \
    --out "$RESULT" > /dev/null
  echo "result $stamp -> $RESULT" >> "$ROOT/logs/validation_attempts.log"
  log "10. 끝: $("$PY" -c "import json,sys; print('macro F1@0.5', round(json.load(open(sys.argv[1], encoding='utf-8'))['macro_f1'], 4))" "$RESULT" | tr -d '\r')"
fi
