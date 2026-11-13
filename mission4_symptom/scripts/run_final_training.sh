#!/usr/bin/env bash
# Mission 4 최종 학습: Training 전체(29,200통화)로 제출용 모델을 다시 학습한다.
#
#   bash scripts/run_final_training.sh
#
# dev 로 정한 레시피(E002: 통화 앞부분 길이 균등 표본, 통화마다 epoch 당 2개, 3 epoch, lr 5e-5, LLRD 0.8,
# pos_weight=(neg/pos)^0.5, 512토큰, 발화 공백 결합)를 값 하나 바꾸지 않고, 학습 통화만 26,280 → 29,200 으로 늘린다.
#
# - Validation 은 어디에서도 읽지 않는다. 이 스크립트에는 Validation 경로가 없다.
#   입력은 Training 라벨 캐시(runs/cache/train.jsonl) 하나뿐이고, --ids 를 주지 않아 캐시 전체로 학습한다.
# - 단계마다 완료 표식(<FINAL_ROOT>/.done/<단계>)을 남긴다. 끊긴 뒤 다시 실행하면 끝난 단계는 건너뛴다.
#   끝나지 않은 단계는 그 단계의 출력 폴더를 지우고 처음부터 다시 돈다.
# - 모든 출력은 <FINAL_ROOT>/logs/ 에도 남는다 (queue.log = 단계 진행, <단계>.log = 단계별 전체 출력).
# - 어느 폴더에서 실행해도 된다. 경로는 이 스크립트 위치를 기준으로 푼다.
#
# 환경 변수
#   PYTHON      파이썬 실행 파일 (기본: 저장소의 venv, 없으면 python)
#   FINAL_ROOT  출력 폴더 (기본 runs/final, mission4_symptom 기준 상대 경로)
#   SEEDS       앞부분 학습 모델의 seed 목록 (기본 "42 43")
#   M3_DIR      예선 Mission 3 폴더 (기본 ../../mission3-calibration/mission3_symptom)
#   TAPT_MODE   reuse(기본) = dev 선택에 쓴 TAPT 백본(<M3_DIR>/runs/devsel/tapt, 학습용 26,280통화 전사)을 그대로 쓴다.
#                             예선 최종 학습(final_s42~45)도 같은 방식이었다 (예선 reports/dev_selection_protocol.md 7단계).
#               full        = Training 전체 29,200통화 전사로 TAPT(MLM 20 epoch)를 다시 돌려 그 백본을 쓴다 (약 2시간 40분 추가).
#   MAX_STEPS   점검용. 0 이 아니면 그 스텝에서 멈춘다. 이때는 FINAL_ROOT 를 runs/final 이 아닌 곳으로 줘야 한다.
set -euo pipefail

abs_dir() { (cd "$1" && (pwd -W 2>/dev/null || pwd)); }   # Git Bash 는 Windows 경로(C:/...), 그 밖은 pwd
HERE="$(abs_dir "$(dirname "${BASH_SOURCE[0]}")/..")"      # mission4_symptom/
cd "$HERE"

if [ -z "${PYTHON:-}" ]; then
  if [ -x "$HERE/../../../venv/Scripts/python.exe" ]; then PYTHON="$HERE/../../../venv/Scripts/python.exe"; else PYTHON=python; fi
fi
PY=$PYTHON
FINAL=${FINAL_ROOT:-runs/final}
SEEDS=${SEEDS:-"42 43"}
M3_DIR=${M3_DIR:-../../mission3-calibration/mission3_symptom}
TAPT_MODE=${TAPT_MODE:-reuse}
MAX_STEPS=${MAX_STEPS:-0}
CALLS=runs/cache/train.jsonl          # Training 라벨 캐시 (scripts/build_cache.py 가 만든다)
EXPECT_CALLS=29200                    # Training 전체 통화 수 (9종 필터 뒤)
TAPT_EPOCHS=20                        # 예선 dev 선택 때와 같은 TAPT 설정

# 레시피 (dev 로 정한 E002 값. 바꾸지 않는다)
EPOCHS=3; SAMPLES_PER_CALL=2; BATCH=16; LR=5e-5; LLRD=0.8; POW=0.5; MAXLEN=512

export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOGS="$FINAL/logs"

# 인자 확인은 출력 폴더를 만들기 전에 한다
case "$TAPT_MODE" in reuse|full) ;; *) echo "TAPT_MODE 는 reuse 또는 full 이어야 합니다: $TAPT_MODE" >&2; exit 1 ;; esac
if [ "$MAX_STEPS" != 0 ] && [ "$FINAL" = "runs/final" ]; then
  echo "MAX_STEPS 점검은 FINAL_ROOT 를 다른 폴더로 주고 돌리십시오 (최종 학습 폴더에 점검 산출물이 섞이지 않게)" >&2; exit 1
fi
mkdir -p "$FINAL/.done" "$LOGS"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOGS/queue.log"; }
done_mark() { touch "$FINAL/.done/$1"; }
is_done() { [ -f "$FINAL/.done/$1" ]; }
die() { log "중단: $*"; exit 1; }
# 같은 출력 폴더에서 백본 방식을 바꾸면 옛 표식과 새 백본이 섞인다. 처음 정한 방식을 기록해 두고 다르면 멈춘다.
if [ -f "$FINAL/.done/tapt_mode" ]; then
  prev=$(tr -d '\r\n' < "$FINAL/.done/tapt_mode")
  [ "$prev" = "$TAPT_MODE" ] || die "이 폴더($FINAL)는 TAPT_MODE=$prev 로 시작했습니다. 방식을 바꾸려면 새 FINAL_ROOT 를 쓰십시오"
else
  echo "$TAPT_MODE" > "$FINAL/.done/tapt_mode"
fi

log "최종 학습 시작: FINAL_ROOT=$FINAL SEEDS=[$SEEDS] TAPT_MODE=$TAPT_MODE MAX_STEPS=$MAX_STEPS"
log "python: $("$PY" -c "import sys, torch, transformers; print(sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" | tr -d '\r')"
log "GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | tr -d '\r' || echo '확인 못 함')"

# ---------------------------------------------------------------------------------------------
# 0. 입력 확인 (매번 돈다. 몇 초)
#    캐시가 Training 전체인지 줄 수로 확인한다. 학습 스크립트는 --ids 가 없으면 캐시의 모든 줄을 쓴다.
# ---------------------------------------------------------------------------------------------
[ -f "$CALLS" ] || die "Training 캐시가 없습니다: $CALLS (scripts/build_cache.py 로 먼저 만든다)"
n_calls=$(wc -l < "$CALLS" | tr -d ' \r')
[ "$n_calls" = "$EXPECT_CALLS" ] || die "캐시 통화 수가 $n_calls 입니다 (Training 전체 $EXPECT_CALLS 이어야 함): $CALLS"
log "0. 캐시 확인: $CALLS $n_calls 통화 (Training 전체)"

# ---------------------------------------------------------------------------------------------
# 1. TAPT 백본
#    reuse: 예선 dev 선택 단계의 백본을 그대로 쓴다 (학습 없음).
#    full : Training 전체 전사(통화 전체, 발화 공백 결합, Mission 4 와 같은 발화 순서)로 MLM 을 다시 돌린다.
#           라벨은 쓰지 않는다. 캐시의 본문만 CSV 로 옮겨 scripts/tapt_mlm.py 에 넘긴다.
# ---------------------------------------------------------------------------------------------
if [ "$TAPT_MODE" = reuse ]; then
  BACKBONE="$M3_DIR/runs/devsel/tapt"
  [ -f "$BACKBONE/model.safetensors" ] && [ -f "$BACKBONE/tapt_config.json" ] || die "TAPT 백본이 없습니다: $BACKBONE"
else
  BACKBONE="$FINAL/tapt"
  if is_done tapt && [ -f "$BACKBONE/model.safetensors" ]; then
    log "1. 건너뜀 tapt"
  else
    rm -f "$FINAL/.done/tapt"
    for s in $SEEDS; do rm -f "$FINAL/.done/prefix_s$s"; done     # 백본이 바뀌므로 뒤 단계도 다시 돈다
    rm -rf "$FINAL/tapt"
    log "1. TAPT 본문 CSV 만들기 (Training 캐시 전체, 라벨은 형식만 채움)"
    "$PY" -B - "$CALLS" "$FINAL/tapt_train_text.csv" <<'PYEOF' 2>&1 | tee "$LOGS/tapt_csv.log"
import csv, json, sys
classes = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
src, dst = sys.argv[1], sys.argv[2]
if "val" in src.replace("\\", "/").split("/")[-1].lower():
    raise SystemExit("Validation 캐시는 TAPT 에 쓸 수 없습니다")
n = 0
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8-sig", newline="") as g:
    w = csv.writer(g)
    w.writerow(["call_id", "text"] + classes)
    for line in f:
        c = json.loads(line)
        # 통화 전체 본문: 빈 발화를 빼고 공백으로 잇는다 (m4.data.prefix_texts 의 마지막 원소와 같다)
        text = " ".join(t.strip() for t in c["texts"] if t and t.strip())
        w.writerow([c["call_id"], text] + [int(v) for v in c["labels"]])
        n += 1
print(f"TAPT 본문 {n}건 -> {dst}", flush=True)
PYEOF
    log "1. TAPT $TAPT_EPOCHS epoch 시작 (진행 막대는 $LOGS/tapt.log)"
    "$PY" -B scripts/tapt_mlm.py --train-csv "$FINAL/tapt_train_text.csv" --output-dir "$FINAL/tapt" \
        --model-name-or-path klue/roberta-base --epochs "$TAPT_EPOCHS" --max-length "$MAXLEN" --amp --local-files-only \
        > "$LOGS/tapt.log" 2>&1 || die "TAPT 실패. $LOGS/tapt.log 를 확인하십시오"
    grep -a "^Epoch " "$LOGS/tapt.log" | tee -a "$LOGS/queue.log" || true
    [ -f "$BACKBONE/model.safetensors" ] || die "TAPT 결과가 없습니다: $BACKBONE"
    done_mark tapt; log "1. TAPT 끝"
  fi
fi
log "1. 백본: $BACKBONE — $("$PY" -c "
import json, sys
d = json.load(open(sys.argv[1] + '/tapt_config.json', encoding='utf-8'))
print('본문', d['num_texts'], '건,', d['epochs'], 'epoch, 마지막 MLM loss', round(d['history'][-1]['mlm_loss'], 4), ',', d['source'])" "$BACKBONE" | tr -d '\r')"

# ---------------------------------------------------------------------------------------------
# 2. 앞부분 학습 모델 (seed 마다 하나). --ids 없음 = 캐시 전체 29,200통화.
#    스텝 수: 29,200 x 2 / 16 = 3,650 스텝/epoch, 3 epoch = 10,950 스텝 (warmup 10% = 1,095, 그 뒤 선형 감쇠).
# ---------------------------------------------------------------------------------------------
for s in $SEEDS; do
  name="prefix_s$s"; out="$FINAL/$name"
  if is_done "$name" && [ -f "$out/best_model/model.safetensors" ]; then
    log "2. 건너뜀 $name"; continue
  fi
  rm -f "$FINAL/.done/$name"
  rm -rf "$out"
  log "2. 시작 $name"
  "$PY" -B scripts/train_prefix.py --calls "$CALLS" --backbone "$BACKBONE" --out "$out" \
      --fmt space --truncation-side right --max-length "$MAXLEN" --epochs "$EPOCHS" \
      --samples-per-call "$SAMPLES_PER_CALL" --batch-size "$BATCH" --lr "$LR" --llrd "$LLRD" \
      --pos-weight-power "$POW" --seed "$s" --max-steps "$MAX_STEPS" 2>&1 | tee "$LOGS/$name.log"
  # 학습 설정 기록으로 Training 전체·정한 스텝 수로 끝났는지 확인한 뒤에만 완료 표식을 남긴다
  "$PY" -c "
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
n, max_steps = int(sys.argv[2]), int(sys.argv[3])
want = max_steps or n * d['samples_per_call'] // d['batch_size'] * d['epochs']
assert d['ids'] is None, 'ids 가 지정됨: %r' % d['ids']
assert d['n_calls'] == n, '학습 통화 수 %d' % d['n_calls']
assert d['total_steps'] == want, '스텝 수 %d (기대 %d)' % (d['total_steps'], want)
print('확인: 통화 %d, 스텝 %d, %.0f초, GPU 최대 %.2f GiB' % (d['n_calls'], d['total_steps'], d['training_seconds'], d['peak_cuda_memory_bytes'] / 2**30))
" "$out/run_config.json" "$EXPECT_CALLS" "$MAX_STEPS" 2>&1 | tr -d '\r' | tee -a "$LOGS/queue.log" || die "$name 학습 기록 확인 실패"
  done_mark "$name"; log "2. 끝 $name"
done

# ---------------------------------------------------------------------------------------------
# 3. 번들 조립 — 여기에 넣지 않는다.
#    scripts/pack_bundle.py 가 정리되면 이 자리에서 호출한다. 멤버 후보:
#      $FINAL/prefix_s42  $FINAL/prefix_s43  (이 스크립트가 만든 앞부분 학습 모델)
#      $M3_DIR/runs/devsel/final_s42         (예선 최종 모델: Training 전체로 학습한 통화 전체 모델, 재학습 불필요)
#    maxent 사전분포는 Training 전체 라벨로 만든다 (--calls runs/cache/train.jsonl, 통화 목록 없음).
# ---------------------------------------------------------------------------------------------
log "모든 학습 단계 끝. 번들 조립은 scripts/pack_bundle.py 로 따로 한다"
