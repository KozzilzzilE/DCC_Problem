#!/usr/bin/env bash
# E022 KcELECTRA(beomi/KcELECTRA-base-v2022, revision 8728557)에 RoBERTa 와 같은 TAPT 를 한 뒤 E002 와 같은 자료·레시피로 학습하고 dev 로 판정한다.
# 팀원 KcELECTRA 점수는 Validation 을 학습에 넣은 결과라 비교 근거가 될 수 없어, Training 만으로 다시 잰다.
#   bash scripts/run_e022.sh
PY=../../../venv/Scripts/python
OUT=runs/E022_kcelectra_tapt
SNAP="C:/Users/Administrator/.cache/huggingface/hub/models--beomi--KcELECTRA-base-v2022/snapshots/87285578b4686b6d190abd40e947a2a416add6de"
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8

TAPT_CSV=../../mission3-calibration/mission3_symptom/runs/devsel/data/train_split.csv   # RoBERTa TAPT 와 같은 본문 (학습용 26,280통화, Validation 없음)
echo "=== TAPT ($(date +%H:%M)) ==="
# 설정은 예선 TAPT(runs/devsel/tapt)와 같다: MLM 20 epoch, lr 5e-5, 실효 배치 16. 어휘가 커서 8GB 에 맞추려고 4x4 로 나눴다.
# KcELECTRA 는 판별기만 공개돼 MLM 머리는 새로 초기화된다.
if [ ! -f "$OUT/tapt/model.safetensors" ]; then
  $PY scripts/tapt_mlm.py --train-csv "$TAPT_CSV" --output-dir "$OUT/tapt" --model-name-or-path "$SNAP" \
      --epochs 20 --amp --local-files-only --batch-size 4 --gradient-accumulation-steps 4 \
      || { echo "E022 TAPT 실패"; echo "=== 끝 ($(date +%H:%M)) ==="; exit 1; }
fi

# 배치 16 은 8GB 를 넘쳐 공유 메모리로 새서 17배 느려졌다(11-14 실측 3초/스텝). 실효 배치 16 은 같게 8x2 로 나눴다.
echo "=== E022 학습 ($(date +%H:%M)) ==="
$PY scripts/train_prefix.py --calls runs/cache/train.jsonl --ids runs/cache/fit_ids.txt \
    --backbone "$OUT/tapt" --out "$OUT" --seed 42 --batch-size 8 --grad-accum 2 \
    || { echo "E022 학습 실패"; echo "=== 끝 ($(date +%H:%M)) ==="; exit 1; }

echo "=== 확률표 ($(date +%H:%M)) ==="
$PY scripts/prefix_probs.py --model "$OUT/best_model" --calls runs/cache/train.jsonl \
    --ids runs/cache/dev_ids.txt --out "$OUT/dev_prefix.npz" --fmt space \
    || { echo "E022 확률표 실패"; echo "=== 끝 ($(date +%H:%M)) ==="; exit 1; }

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
echo "E022 완료"
echo "=== 끝 ($(date +%H:%M)) ==="
