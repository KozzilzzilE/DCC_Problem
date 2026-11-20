#!/usr/bin/env bash
# Mission 4 최종 학습 (방안 A, version=2 미션 분리형, fit 26,280통화 기준).
#
#   bash scripts/run_final_training.sh
#
# 구성 (방안 A):
#   1) TAPT 백본 준비 (klue/roberta-base + fit 26,280 전사 20 epoch MLM; dev/Validation 미사용)
#   2) 4-1 전용 분류기 2종: SoftJaccard (Weighted BCE + 1.0*Soft-Jaccard, seed 42·43)
#   3) 4-2 전용 분류기 3종 (ENS3): E001(통화 전체 6ep) + E002(앞부분 s42) + E007(앞부분 s43)
#   4) 4-2 무누수 학습형 중단 정책: fit 26,280 5겹 OOF 기반 C128_v1_nonempty (tau=0.5, beta=128, require_nonempty)
#   5) 미션 분리 번들 패킹: scripts/pack_bundle.py -> $FINAL/ckpt/mission4.pt (version=2)
#
# - Validation 폴더는 어디에서도 읽지 않는다 (학습·정책 모두 fit 26,280만 사용, dev 2,920은 판정 전용).
# - 단계마다 완료 표식($FINAL/.done/<단계>)을 남겨 중단 후 이어서 실행할 수 있다.
# - 모든 출력은 $FINAL/logs/ 에 저장된다.
#
# 주요 환경 변수:
#   PYTHON      파이썬 실행 파일 (기본: 저장소 venv 또는 python3)
#   FINAL_ROOT  출력 폴더 (기본: runs/final_option_a)
#   SEEDS       4-1 SoftJaccard seed 목록 (기본: "42 43")
#   TAPT_MODE   reuse(기본, TAPT_DIR 사용) 또는 train(fit 26,280 전사로 MLM 20ep 실행)
#   TAPT_DIR    TAPT_MODE=reuse 일 때 사용할 백본 경로
#   E001_DIR    기존 E001(통화 전체 학습) 모델 경로 (비어 있으면 --full-only 6ep 로 fit 26,280 학습)
#   POLICY_JSON 기존 C128_v1_nonempty 정책 JSON 경로 (비어 있으면 OOF 기반 policy_fitonly_c128_v1_nonempty.json 학습)
#   OOF_ROOT    5겹 OOF 확률표 폴더 (기본: runs/E011_oof)
#   SKIP_BUNDLE 1 이면 5단계 pack_bundle.py 를 건너뛴다
#   MAX_STEPS   점검용 스텝 제한 (0 이면 전체 학습)
set -euo pipefail

abs_dir() { (cd "$1" && (pwd -W 2>/dev/null || pwd)); }
HERE="$(abs_dir "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$HERE"

if [ -z "${PYTHON:-}" ]; then
  if [ -x "$HERE/../../../venv/Scripts/python.exe" ]; then
    PYTHON="$HERE/../../../venv/Scripts/python.exe"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    PYTHON=python
  fi
fi
PY="$PYTHON"

FINAL="${FINAL_ROOT:-runs/final_option_a}"
SEEDS="${SEEDS:-42 43}"
TAPT_MODE="${TAPT_MODE:-reuse}"
TAPT_DIR="${TAPT_DIR:-../../mission3-calibration/mission3_symptom/runs/devsel/tapt}"
E001_DIR="${E001_DIR:-}"
POLICY_JSON="${POLICY_JSON:-}"
OOF_ROOT="${OOF_ROOT:-runs/E011_oof}"
SKIP_BUNDLE="${SKIP_BUNDLE:-0}"
MAX_STEPS="${MAX_STEPS:-0}"

CALLS="runs/cache/train.jsonl"
FIT_IDS="runs/cache/fit_ids.txt"
DEV_IDS="runs/cache/dev_ids.txt"
EXPECT_TRAIN=29200
EXPECT_FIT=26280
EXPECT_DEV=2920
TAPT_EPOCHS=20

# 공통 분류기 하이퍼파라미터 (dev 선택 레시피)
EPOCHS=3
SAMPLES_PER_CALL=2
BATCH=16
LR=5e-5
LLRD=0.8
POW=0.5
MAXLEN=512

export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
if [ "${ALLOW_NETWORK:-0}" != "1" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
LOGS="$FINAL/logs"

case "$TAPT_MODE" in
  reuse|train) ;;
  *) echo "TAPT_MODE 는 reuse 또는 train 이어야 합니다: $TAPT_MODE" >&2; exit 1 ;;
esac
if [ "$MAX_STEPS" != 0 ] && [ "$FINAL" = "runs/final_option_a" ]; then
  echo "MAX_STEPS 점검은 FINAL_ROOT 를 별도 폴더로 지정하십시오." >&2
  exit 1
fi
mkdir -p "$FINAL/.done" "$LOGS"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOGS/queue.log"; }
done_mark() { touch "$FINAL/.done/$1"; }
is_done() { [ -f "$FINAL/.done/$1" ]; }
die() { log "중단: $*"; exit 1; }

if [ -f "$FINAL/.done/tapt_mode" ]; then
  prev=$(tr -d '\r\n' < "$FINAL/.done/tapt_mode")
  [ "$prev" = "$TAPT_MODE" ] || die "이 폴더($FINAL)는 TAPT_MODE=$prev 로 시작했습니다. 방식을 바꾸려면 새 FINAL_ROOT 를 쓰십시오."
else
  echo "$TAPT_MODE" > "$FINAL/.done/tapt_mode"
fi

log "방안 A 학습 파이프라인 시작: FINAL_ROOT=$FINAL SEEDS=[$SEEDS] TAPT_MODE=$TAPT_MODE MAX_STEPS=$MAX_STEPS"
log "python: $("$PY" -c "import sys, torch, transformers; print(sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" | tr -d '\r')"

# ---------------------------------------------------------------------------------------------
# 0. 입력 캐시 및 fit/dev 분할 검증 (Validation 미사용 보장)
# ---------------------------------------------------------------------------------------------
[ -f "$CALLS" ] || die "Training 캐시가 없습니다: $CALLS (scripts/build_cache.py 로 먼저 생성)"
[ -f "$FIT_IDS" ] || die "fit ID 목록이 없습니다: $FIT_IDS"
[ -f "$DEV_IDS" ] || die "dev ID 목록이 없습니다: $DEV_IDS"

"$PY" -B - "$CALLS" "$FIT_IDS" "$DEV_IDS" "$EXPECT_TRAIN" "$EXPECT_FIT" "$EXPECT_DEV" <<'PYEOF' 2>&1 | tee -a "$LOGS/queue.log"
import json, sys
from pathlib import Path

calls_p, fit_p, dev_p = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ex_train, ex_fit, ex_dev = map(int, sys.argv[4:7])
if "val" in calls_p.name.lower():
    raise SystemExit("Validation 캐시는 학습에 사용할 수 없습니다.")
call_ids = {json.loads(line)["call_id"] for line in calls_p.open(encoding="utf-8")}
fit_ids = [l.strip() for l in fit_p.read_text(encoding="utf-8").splitlines() if l.strip()]
dev_ids = [l.strip() for l in dev_p.read_text(encoding="utf-8").splitlines() if l.strip()]
assert len(call_ids) == ex_train, f"train.jsonl 통화 수 불일치: {len(call_ids)} != {ex_train}"
assert len(fit_ids) == ex_fit and len(set(fit_ids)) == ex_fit, f"fit_ids 불일치: {len(fit_ids)}"
assert len(dev_ids) == ex_dev and len(set(dev_ids)) == ex_dev, f"dev_ids 불일치: {len(dev_ids)}"
overlap = set(fit_ids) & set(dev_ids)
assert not overlap, f"fit/dev 중복 발견: {len(overlap)}건"
print(f"0. 입력 확인 완료: train={len(call_ids)}, fit={len(fit_ids)}, dev={len(dev_ids)}, overlap=0")
PYEOF

# ---------------------------------------------------------------------------------------------
# 1. TAPT 백본 준비 (fit 26,280 전사만 사용)
# ---------------------------------------------------------------------------------------------
if [ "$TAPT_MODE" = reuse ]; then
  BACKBONE="$TAPT_DIR"
  [ -f "$BACKBONE/model.safetensors" ] && [ -f "$BACKBONE/tapt_config.json" ] || die "TAPT 백본이 없습니다: $BACKBONE"
else
  BACKBONE="$FINAL/tapt"
  if is_done tapt && [ -f "$BACKBONE/model.safetensors" ]; then
    log "1. 건너뜀 tapt"
  else
    rm -f "$FINAL/.done/tapt"
    rm -rf "$BACKBONE"
    log "1. TAPT 본문 CSV 생성 (fit 26,280 전사만 사용, dev/Validation 제외)"
    "$PY" -B - "$CALLS" "$FIT_IDS" "$FINAL/tapt_fit_text.csv" <<'PYEOF' 2>&1 | tee "$LOGS/tapt_csv.log"
import csv, json, sys
from pathlib import Path

classes = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
calls_p, fit_p, dst_p = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
fit_set = {l.strip() for l in fit_p.read_text(encoding="utf-8").splitlines() if l.strip()}
n = 0
with calls_p.open(encoding="utf-8") as f, dst_p.open("w", encoding="utf-8-sig", newline="") as g:
    w = csv.writer(g)
    w.writerow(["call_id", "text"] + classes)
    for line in f:
        c = json.loads(line)
        if c["call_id"] not in fit_set:
            continue
        text = " ".join(t.strip() for t in c["texts"] if t and t.strip())
        w.writerow([c["call_id"], text] + [int(v) for v in c["labels"]])
        n += 1
assert n == len(fit_set), f"fit TAPT 건수 불일치: {n} != {len(fit_set)}"
print(f"TAPT 본문 {n}건 저장 -> {dst_p}", flush=True)
PYEOF
    log "1. TAPT ${TAPT_EPOCHS} epoch 시작 ($LOGS/tapt.log)"
    TAPT_ARGS=()
    if [ "${ALLOW_NETWORK:-0}" != "1" ]; then
      TAPT_ARGS+=(--local-files-only)
    fi
    "$PY" -B scripts/tapt_mlm.py --train-csv "$FINAL/tapt_fit_text.csv" --output-dir "$BACKBONE" \
        --model-name-or-path klue/roberta-base --epochs "$TAPT_EPOCHS" --max-length "$MAXLEN" --amp "${TAPT_ARGS[@]}" \
        > "$LOGS/tapt.log" 2>&1 || die "TAPT 실패 ($LOGS/tapt.log 확인)"
    [ -f "$BACKBONE/model.safetensors" ] || die "TAPT 결과 없음: $BACKBONE"
    done_mark tapt
    log "1. TAPT 완료"
  fi
fi
log "1. 사용 백본: $BACKBONE"

verify_prefix_run() {
  local name="$1" out="$2" expect_jw="$3" expect_steps="$4"
  "$PY" -B - "$out/run_config.json" "$EXPECT_FIT" "$MAX_STEPS" "$expect_jw" "$expect_steps" <<'PYEOF' 2>&1 | tr -d '\r' | tee -a "$LOGS/queue.log"
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
n_fit, max_steps = int(sys.argv[2]), int(sys.argv[3])
ex_jw, ex_steps = float(sys.argv[4]), int(sys.argv[5])
assert d.get("ids") == "runs/cache/fit_ids.txt" or Path(d.get("ids") or "").name == "fit_ids.txt", f"ids 불일치: {d.get('ids')}"
assert d["n_calls"] == n_fit, f"학습 통화 수 불일치: {d['n_calls']} != {n_fit}"
want = max_steps or ex_steps
assert d["total_steps"] == want, f"스텝 수 불일치: {d['total_steps']} != {want}"
assert abs(float(d.get("jaccard_weight", 0.0)) - ex_jw) < 1e-6, f"jaccard_weight 불일치: {d.get('jaccard_weight')} != {ex_jw}"
print(f"확인: 통화 {d['n_calls']}, 스텝 {d['total_steps']}, jaccard_weight={d.get('jaccard_weight', 0.0)}, {d['training_seconds']:.1f}s")
PYEOF
}

# ---------------------------------------------------------------------------------------------
# 2. 4-1 전용 SoftJaccard 2-seed 학습 (fit 26,280, jaccard_weight=1.0)
# ---------------------------------------------------------------------------------------------
SJ_MEMBERS=()
for s in $SEEDS; do
  name="m4_softjaccard_v1_s$s"
  out="$FINAL/$name"
  SJ_MEMBERS+=("$out/best_model")
  if is_done "$name" && [ -f "$out/best_model/model.safetensors" ]; then
    log "2. 건너뜀 $name"
    continue
  fi
  rm -f "$FINAL/.done/$name"
  rm -rf "$out"
  log "2. 시작 $name (--loss softjaccard, jaccard_weight=1.0)"
  "$PY" -B scripts/train_prefix.py --calls "$CALLS" --ids "$FIT_IDS" --backbone "$BACKBONE" --out "$out" \
      --fmt space --truncation-side right --max-length "$MAXLEN" --epochs "$EPOCHS" \
      --samples-per-call "$SAMPLES_PER_CALL" --batch-size "$BATCH" --lr "$LR" --llrd "$LLRD" \
      --pos-weight-power "$POW" --loss softjaccard --seed "$s" --max-steps "$MAX_STEPS" 2>&1 | tee "$LOGS/$name.log"
  verify_prefix_run "$name" "$out" "1.0" "9855" || die "$name 검증 실패"
  done_mark "$name"
  log "2. 완료 $name"
done

# ---------------------------------------------------------------------------------------------
# 3. 4-2 전용 ENS3 (E001 통화 전체 + E002 앞부분 s42 + E007 앞부분 s43, fit 26,280)
# ---------------------------------------------------------------------------------------------
if [ -n "$E001_DIR" ]; then
  [ -d "$E001_DIR" ] || die "지정한 E001_DIR 이 없습니다: $E001_DIR"
  E001_BEST="$E001_DIR"
  log "3. E001 기존 경로 재사용: $E001_BEST"
else
  name="E001"
  out="$FINAL/$name"
  E001_BEST="$out/best_model"
  if is_done "$name" && [ -f "$E001_BEST/model.safetensors" ]; then
    log "3. 건너뜀 $name"
  else
    rm -f "$FINAL/.done/$name"
    rm -rf "$out"
    log "3. 시작 $name (--full-only --samples-per-call 1 --epochs 6 --loss bce)"
    "$PY" -B scripts/train_prefix.py --calls "$CALLS" --ids "$FIT_IDS" --backbone "$BACKBONE" --out "$out" \
        --fmt space --truncation-side right --max-length "$MAXLEN" --epochs 6 --samples-per-call 1 --full-only \
        --batch-size "$BATCH" --lr "$LR" --llrd "$LLRD" --pos-weight-power "$POW" --loss bce \
        --seed 42 --max-steps "$MAX_STEPS" 2>&1 | tee "$LOGS/$name.log"
    verify_prefix_run "$name" "$out" "0.0" "9852" || die "$name 검증 실패"
    done_mark "$name"
    log "3. 완료 $name"
  fi
fi

for pair in "E002:42" "E007:43"; do
  name="${pair%%:*}"
  seed="${pair##*:}"
  out="$FINAL/$name"
  if is_done "$name" && [ -f "$out/best_model/model.safetensors" ]; then
    log "3. 건너뜀 $name"
    continue
  fi
  rm -f "$FINAL/.done/$name"
  rm -rf "$out"
  log "3. 시작 $name (--loss bce --seed $seed)"
  "$PY" -B scripts/train_prefix.py --calls "$CALLS" --ids "$FIT_IDS" --backbone "$BACKBONE" --out "$out" \
      --fmt space --truncation-side right --max-length "$MAXLEN" --epochs "$EPOCHS" \
      --samples-per-call "$SAMPLES_PER_CALL" --batch-size "$BATCH" --lr "$LR" --llrd "$LLRD" \
      --pos-weight-power "$POW" --loss bce --seed "$seed" --max-steps "$MAX_STEPS" 2>&1 | tee "$LOGS/$name.log"
  verify_prefix_run "$name" "$out" "0.0" "9855" || die "$name 검증 실패"
  done_mark "$name"
  log "3. 완료 $name"
done
ENS3_MEMBERS=("$E001_BEST" "$FINAL/E002/best_model" "$FINAL/E007/best_model")

# ---------------------------------------------------------------------------------------------
# 4. 4-2 무누수 학습형 중단 정책 C128_v1_nonempty (fit 26,280 5겹 OOF)
# ---------------------------------------------------------------------------------------------
if [ -n "$POLICY_JSON" ]; then
  [ -f "$POLICY_JSON" ] || die "지정한 POLICY_JSON 이 없습니다: $POLICY_JSON"
  USE_POLICY="$POLICY_JSON"
  log "4. 기존 중단 정책 재사용: $USE_POLICY"
else
  POL_OUT="$FINAL/halt_policy"
  USE_POLICY="$POL_OUT/policy.json"
  if is_done halt_policy && [ -f "$USE_POLICY" ]; then
    log "4. 건너뜀 halt_policy"
  else
    log "4. 5겹 OOF 확률표 확인 및 생성 ($OOF_ROOT)"
    PYTHON="$PY" BACKBONE="$BACKBONE" OOF_ROOT="$OOF_ROOT" bash scripts/run_oof_tables.sh
    for mdir in "E001:$E001_BEST" "E002:$FINAL/E002/best_model" "E007:$FINAL/E007/best_model"; do
      mname="${mdir%%:*}"
      mpath="${mdir#*:}"
      if [ ! -f "$FINAL/${mname}_dev_prefix.npz" ]; then
        log "4. Dev 확률표 생성: $mname"
        "$PY" -B scripts/prefix_probs.py --model "$mpath" --calls "$CALLS" --ids "$DEV_IDS" --out "$FINAL/${mname}_dev_prefix.npz"
      fi
    done
    log "4. LightGBM 중단 정책 학습 (tau=0.5 고정, beta=128, require_nonempty)"
    "$PY" -B scripts/train_halt_policy.py \
        --fit-tables "$OOF_ROOT/fold0/oof_prefix.npz" "$OOF_ROOT/fold1/oof_prefix.npz" "$OOF_ROOT/fold2/oof_prefix.npz" "$OOF_ROOT/fold3/oof_prefix.npz" "$OOF_ROOT/fold4/oof_prefix.npz" \
        --eval-tables "$FINAL/E001_dev_prefix.npz" "$FINAL/E002_dev_prefix.npz" "$FINAL/E007_dev_prefix.npz" \
        --shift-from "$FINAL/E002" --calls "$CALLS" \
        --taus 0.5 --betas 128 --base-tau 0.5 --ref-tau 0.5 --require-nonempty \
        --out "$POL_OUT" 2>&1 | tee "$LOGS/halt_policy.log"
    done_mark halt_policy
    log "4. 중단 정책 학습 완료: $USE_POLICY"
  fi
fi

# ---------------------------------------------------------------------------------------------
# 5. 방안 A (version=2) 미션 분리 번들 패킹
# ---------------------------------------------------------------------------------------------
if [ "$SKIP_BUNDLE" = "1" ]; then
  log "5. SKIP_BUNDLE=1 이므로 번들 패킹을 건너뜁니다."
else
  mkdir -p "$FINAL/ckpt"
  "$PY" -B scripts/pack_bundle.py \
      --members-4-1 "${SJ_MEMBERS[@]}" \
      --members-4-2 "${ENS3_MEMBERS[@]}" \
      --halt-policy "$USE_POLICY" \
      --require-nonempty --rule threshold --no-force-top1 \
      --note "Mission 4 Option A (version=2, fit 26,280): 4-1 SoftJaccard s42+s43, 4-2 ENS3 + C128_v1_nonempty" \
      --out "$FINAL/ckpt/mission4.pt" 2>&1 | tee "$LOGS/pack_bundle.log"
  done_mark bundle
  log "5. 번들 패킹 완료: $FINAL/ckpt/mission4.pt"
fi

log "모든 단계 완료"
