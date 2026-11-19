"""Mission 4 제출용 학습 기록 노트북(model_train.ipynb)을 만들고 실행한다.

학습은 명령줄에서 `bash scripts/run_final_training.sh` 로 돌린다 (단계별 이어하기, 로그는 runs/final/logs/).
노트북은 그 기록을 순서대로 읽어 셀 출력으로 남긴다. 예선 Mission 3 의 model_train.ipynb 와 같은 방식이다.
  1) 데이터 준비: 원본 Training 라벨 JSON 으로 캐시를 다시 만들어, 학습에 쓴 캐시와 같은지 확인한다.
  2) dev 로 정한 레시피와 그 근거 수치를 보여 준다.
  3) 최종 학습(Training 전체)의 로그 전문과 설정 기록을 출력한다.
  4) 제출 번들(.pt)의 구성과 추론 설정을 출력한다.
  5) 4-2 중단 정책의 학습·판정 기록을 출력한다.
  6) dev 평가 결과를 요약한다.
노트북은 Validation 폴더를 읽지 않는다.

    python scripts/build_notebook.py --no-exec   # 생성만 (학습 전: 출력이 빈 노트북)
    python scripts/build_notebook.py             # 생성 + 실행 (학습이 끝난 뒤: 로그가 셀 출력으로 들어간다)

환경 변수 M4_RUN_TRAINING=1 을 주고 실행하면 4절 셀이 학습 스크립트를 직접 돌린다 (끝난 단계는 건너뛴다).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent.parent          # mission4_symptom/
NOTEBOOK = HERE / "model_train.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


CELLS = [
    md(r"""
# Mission 4 — 환자 증상 조기예측: 최종 제출 모델 학습 기록

**과제**: 119 신고 통화의 발화가 하나씩 들어올 때마다 증상 9종을 예측한다 (4-1: 매 순간 예측, 4-2: 스스로 멈추고 예측).

**최종 모델**
1. KLUE-RoBERTa-base 를 Training 통화 전사로 TAPT(MLM) 한 백본에서 시작한다 (예선 Mission 3 에서 만든 백본).
2. **통화 앞부분으로 학습한다.** 통화마다 epoch 당 앞부분 2개를 뽑고, 길이는 1..n 발화에서 균등하게 뽑는다. 라벨은 통화 전체의 증상이다.
3. 그 밖의 설정은 예선 최종 레시피와 같다: 3 epoch, 배치 16, lr 5e-5, 층별 학습률 감쇠(LLRD) 0.8, pos_weight=(음성/양성)^0.5, 최대 512토큰, 발화 공백 결합.
4. seed 가 다른 앞부분 학습 모델과, 통화 전체로 학습한 예선 최종 모델의 확률을 평균한다. 구성은 5절에서 번들 파일을 읽어 출력한다.

**규정 준수**
- **데이터:** 학습에는 Training 폴더만 썼다. 이 노트북과 학습 스크립트는 Validation 폴더를 읽지 않는다.
- **값 고르기:** 레시피·결정 규칙·중단 기준은 모두 Training 내부 dev(10%, 2,920통화, 예선과 같은 분할)로 정했다.
- **입력:** 그 순간까지 받은 발화의 전사만 쓴다. 통화 전체 길이, 뒤 발화, 라벨 JSON 의 다른 필드는 쓰지 않는다.
- **모델:** 공개 사전학습 모델(klue/roberta-base)만 썼고, 외부 데이터와 상용 API 는 쓰지 않았다.

**이 노트북의 구조**
- 학습은 명령줄에서 `bash scripts/run_final_training.sh` 로 돌렸다. 한 번에 1시간 이상 걸리는 학습이라, 끊겨도 단계별로 이어서 돌 수 있고 로그가 파일로 남는 방식을 택했다.
- 각 셀은 그 스크립트가 남긴 로그와 설정 기록(`runs/final/`)을 읽어 출력한다. 4절에 학습 로그 전문이 들어 있다.
- 제출 폴더에는 `runs/` 가 없으므로 저장된 셀 출력으로 확인한다.
"""),
    code(r"""
import hashlib, json, os, platform, subprocess, sys, tempfile, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*IProgress.*")     # 진행 막대 위젯이 없다는 안내는 학습과 무관하다
import numpy as np
import pandas as pd
import torch, transformers

HERE = Path.cwd()                         # mission4_symptom/
sys.path.insert(0, str(HERE))
from m4 import SYMPTOM_CLASSES
from m4.data import build_cache, label_matrix, load_cache

FINAL = HERE / os.environ.get("FINAL_ROOT", "runs/final")      # 최종 학습 출력 (scripts/run_final_training.sh)
CACHE = HERE / "runs" / "cache" / "train.jsonl"                # Training 라벨 캐시
BUNDLE = HERE / os.environ.get("M4_BUNDLE", "ckpt/mission4.pt")  # 제출 번들
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42 43").split()]

def find_data_root(start: Path) -> Path:
    # 위 폴더로 올라가며 data/train/label 을 찾는다 (빈 자리표시 폴더는 건너뛴다)
    for parent in [start, *start.parents]:
        label_dir = parent / "data" / "train" / "label"
        if label_dir.is_dir() and next(label_dir.glob("*.json"), None) is not None:
            return parent / "data"
    raise FileNotFoundError("data/train/label 폴더를 찾지 못했습니다. DCC_DATA_ROOT 를 지정하세요.")

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def show_text(path, missing="(아직 없음 — 학습 전)"):
    # 로그 파일 전문을 그대로 출력한다
    path = Path(path)
    print(f"----- {path.relative_to(HERE).as_posix()} -----")
    print(path.read_text(encoding="utf-8", errors="replace").rstrip() if path.is_file() else missing)

DATA_ROOT = Path(os.environ["DCC_DATA_ROOT"]) if os.environ.get("DCC_DATA_ROOT") else find_data_root(HERE)
print("Python", platform.python_version(), "| torch", torch.__version__, "| transformers", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음", "| OS:", platform.platform())
print("Training 라벨 폴더:", DATA_ROOT / "train" / "label")
"""),
    md(r"""
## 1. 데이터 준비 — Training 라벨 JSON 을 통화 캐시로

`m4.data.read_call` 이 라벨 JSON 하나를 통화 하나로 읽는다.
- **발화 순서:** `startAt`, `endAt` 으로 정렬한다. 대회 추론 템플릿(`inference.py` 의 `load_samples`)과 같은 순서다.
- **입력으로 남기는 것:** 발화의 전사·화자·시각뿐이다. 학습 입력에는 전사만 쓴다 (`m4.data.prefix_texts`, 발화를 공백으로 잇는다).
- **라벨:** `symptom` 에서 9종만 0/1 로 남긴다. 9종이 하나도 없는 통화는 뺀다 (템플릿과 같은 제외 규칙).

3만 개 JSON 을 매번 여는 시간을 줄이려고 JSONL 한 파일로 모아 둔다. 명령줄로는 `python scripts/build_cache.py ...` 다.
아래 셀은 원본 Training JSON 으로 캐시를 다시 만들어, 학습에 실제로 쓴 캐시와 줄 단위로 같은지 확인한다.
"""),
    code(r"""
t0 = time.perf_counter()
with tempfile.TemporaryDirectory() as tmp:
    rebuilt = Path(tmp) / "train.jsonl"
    n_rebuilt = build_cache(DATA_ROOT / "train" / "label", rebuilt)       # Training 폴더만 읽는다
    if not CACHE.is_file():                                               # 캐시가 없으면 지금 만든 것을 쓴다
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(rebuilt.read_bytes())
    same = rebuilt.read_text(encoding="utf-8").splitlines() == CACHE.read_text(encoding="utf-8").splitlines()
print(f"원본 JSON 에서 다시 만든 통화 {n_rebuilt:,}건 ({time.perf_counter() - t0:.1f}초) / 학습에 쓴 캐시와 같은가: {same}")
print("캐시 sha256:", hashlib.sha256(CACHE.read_bytes()).hexdigest()[:16], "…")

calls = load_cache(CACHE)
labels = label_matrix(calls)
n_utt = np.array([c.n_utt for c in calls])
print(f"Training 통화 {len(calls):,}건, 발화 수 중앙값 {int(np.median(n_utt))} (10% {int(np.percentile(n_utt, 10))}, 90% {int(np.percentile(n_utt, 90))}, 최대 {n_utt.max()})")
count = labels.sum(axis=1)
print("통화당 증상 수:", {int(k): f"{(count == k).mean():.1%}" for k in sorted(set(count.tolist()))})
pd.DataFrame({"양성 통화": labels.sum(axis=0), "비율": labels.mean(axis=0).round(3),
              "pos_weight=(음성/양성)^0.5": (((len(labels) - labels.sum(axis=0)) / labels.sum(axis=0)) ** 0.5).round(3)},
             index=SYMPTOM_CLASSES)
"""),
    md(r"""
## 2. 레시피 정하기 — Training 내부 dev

Training 을 통화 단위로 90/10 으로 나눴다 (예선 Mission 3 과 같은 분할, seed 1234).
- **학습용 26,280통화:** 후보 모델을 학습한다.
- **dev 2,920통화:** 학습 입력 방식, 결정 규칙, 중단 기준, 평균할 모델을 정한다.

dev 채점은 주최측 `score.py` 와 같은 식을 쓴다 (`m4/simulate.py`, 값 일치 테스트 `tests/`).
아래 표는 실험별 dev 점수다 (`docs/results.csv`). 실험의 명령·설정·해석은 `docs/EXPERIMENT_LOG.md` 에 있다.

정한 것:
- **학습 입력:** 통화 전체가 아니라 앞부분(길이 균등 표본)으로 학습한다 (`E002`). 4-1 채점이 모든 순간을 같은 무게로 평균하므로 학습 분포도 순간에 대해 균등하게 맞췄다.
- **4-1 모델·결정 규칙 (Option A 분리):** `KLUE-RoBERTa-base (TAPT)` 에 `Weighted BCE(pos_weight^0.5) + Soft-Jaccard (weight 0.5)` 를 결합한 `2`개 시드(`seed 42·43`, `m4_softjaccard_v1`)를 평균하고, 대회 규정대로 판정 임계값 `0.5` 고정(`force_top1=False`)으로 예측한다.
- **4-2 모델·중단 기준 (Option A 분리):** 4-2 는 통화 끝(중단 시점) F1 이 핵심이므로 `final_step Macro-F1` 이 더 높은 `ENS3`(`E001` 통화 전체 1종 + `E002/E007` 앞부분 2종)을 쓰고, `fit 26,280`통화 OOF 로 학습한 `LightGBM` 조기중단 정책(`C128_v1_nonempty`, `β = 128`, `τ = 0.5` 고정, `min_step = 5`, `require_nonempty = True`)으로 멈춘다.
- **해 봤지만 채택하지 않은 것:** 다른 계열 인코더(`E008`/`DeBERTa` 등), 4멤버 확장(`ENS4`), 증상 개수 보조 출력(`E014`), `Soft-Jaccard`/`Soft-Macro-F1` 의 4-2 직접 적용(`final_step Macro-F1` 하락으로 4-1 에만 적용).
"""),
    code(r"""
ids = {name: (HERE / "runs" / "cache" / f"{name}_ids.txt") for name in ("fit", "dev")}
if all(p.is_file() for p in ids.values()):
    print({name: sum(1 for line in open(p, encoding="utf-8") if line.strip()) for name, p in ids.items()}, "(학습용 / dev 통화 수)")

results = pd.read_csv(HERE / "docs" / "results.csv")
display(results[["exp_id", "what", "m4_1_score", "m4_1_rule", "m4_2_score", "m4_2_policy", "m4_2_macro_f1", "m4_2_earliness"]].fillna(""))

# dev 선택 단계의 앞부분 학습 모델(E002, 학습용 26,280통화)의 설정. 최종 학습은 이 값을 그대로 쓴다.
RECIPE_KEYS = ["fmt", "truncation_side", "max_length", "epochs", "samples_per_call", "include_full", "batch_size",
               "lr", "llrd", "pos_weight_power"]
e002 = HERE / "runs" / "E002_prefix_uniform" / "run_config.json"
if e002.is_file():
    cfg = load_json(e002)
    print("E002 레시피:", {k: cfg[k] for k in RECIPE_KEYS})
    print(f"E002 학습: 통화 {cfg['n_calls']:,}건, {cfg['total_steps']:,} 스텝, {cfg['training_seconds'] / 60:.1f}분")
"""),
    md(r"""
## 3. TAPT 백본

KLUE-RoBERTa-base 를 119 통화 전사로 MLM 추가 사전학습한 가중치다. 예선 Mission 3 의 `tapt_mlm.py` 로 만들었다.
- **데이터:** Training 통화 전사만 쓴다. 라벨은 쓰지 않고 Validation 은 읽지 않는다.
- **설정:** 20 epoch, 15% 마스킹, lr 5e-5, warmup 6%, 선형 감쇠, 최대 512토큰.

아래 셀은 최종 학습이 실제로 쓴 백본의 기록(`tapt_config.json`)을 출력한다.
"""),
    code(r"""
def backbone_dir():
    # 최종 학습 기록에 적힌 백본 경로. 학습 전이면 기본 경로(예선 dev 선택 단계의 TAPT)를 본다.
    cfg_path = FINAL / f"prefix_s{SEEDS[0]}" / "run_config.json"
    raw = load_json(cfg_path)["backbone"] if cfg_path.is_file() else "../../mission3-calibration/mission3_symptom/runs/devsel/tapt"
    return (HERE / raw).resolve()

tapt_path = backbone_dir() / "tapt_config.json"
if tapt_path.is_file():
    tapt = load_json(tapt_path)
    print("백본:", backbone_dir())
    print(f"본문 {tapt['num_texts']:,}건, {tapt['epochs']} epoch, 학습 시간 {tapt['training_seconds'] / 60:.1f}분")
    print("데이터 출처:", tapt["source"])
    display(pd.DataFrame(tapt["history"]).set_index("epoch").round(4).T)
else:
    print("TAPT 기록을 찾지 못했습니다:", tapt_path)
"""),
    md(r"""
## 4. 최종 학습 — Training 전체 29,200통화

2절에서 정한 레시피 그대로, 학습 통화만 Training 전체로 늘려 다시 학습한다 (`scripts/run_final_training.sh` → `scripts/train_prefix.py`).
- **통화 목록 없음:** `--ids` 를 주지 않으면 캐시 전체로 학습한다. dev 통화도 학습에 들어가므로, 이 모델들은 dev 로 다시 채점하지 않는다.
- **스텝 수:** 29,200통화 × 2개 ÷ 배치 16 = 3,650 스텝/epoch, 3 epoch = 10,950 스텝. warmup 은 처음 10%, 그 뒤 선형 감쇠다.
- **저장:** 3 epoch 를 끝낸 가중치를 저장한다. 평가 점수를 보고 고르지 않는다.
- **pos_weight:** Training 전체 라벨로 계산한다 (1절 표와 같은 값).

아래 첫 셀은 학습 명령이다. 환경 변수 `M4_RUN_TRAINING=1` 이면 이 셀이 학습을 직접 돌리고(끝난 단계는 건너뜀), 아니면 명령만 보여 준다.
그다음 셀이 학습 로그 전문을 출력한다.
"""),
    code(r"""
TRAIN_CMD = ["bash", "scripts/run_final_training.sh"]
print("학습 명령 (mission4_symptom 폴더에서):", " ".join(TRAIN_CMD))
if os.environ.get("M4_RUN_TRAINING") == "1":
    # 스크립트 출력(단계 진행과 학습 로그)을 줄 단위로 그대로 셀에 남긴다
    proc = subprocess.Popen(TRAIN_CMD, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        if "Loading weights" not in line and "Writing model shards" not in line:     # 진행 막대는 뺀다
            print(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"학습 스크립트가 실패했습니다 (종료 코드 {proc.returncode})")
else:
    print("이 실행에서는 학습을 돌리지 않았습니다. 아래 셀은 이미 끝난 학습의 로그를 읽습니다.")
"""),
    code(r"""
# 단계 진행 기록, 그리고 seed 별 학습 로그 전문 (200 스텝마다 평균 loss·학습률·경과 시간)
show_text(FINAL / "logs" / "queue.log")
for seed in SEEDS:
    print()
    show_text(FINAL / f"prefix_s{seed}" / "train_log.txt")
"""),
    code(r"""
# 학습 설정 기록으로 확인: Training 전체로, 정한 레시피로, 정한 스텝 수만큼 학습했는가
rows = []
for seed in SEEDS:
    path = FINAL / f"prefix_s{seed}" / "run_config.json"
    if not path.is_file():
        print(f"prefix_s{seed}: 아직 없음 — 학습 전")
        continue
    cfg = load_json(path)
    assert cfg["ids"] is None and cfg["n_calls"] == len(calls), "Training 전체로 학습한 모델이 아닙니다"
    assert "val" not in Path(cfg["calls"]).name.lower(), "Validation 캐시로 학습했습니다"
    assert cfg["total_steps"] == len(calls) * cfg["samples_per_call"] // cfg["batch_size"] * cfg["epochs"], "스텝 수가 다릅니다"
    if e002.is_file():
        assert {k: cfg[k] for k in RECIPE_KEYS} == {k: load_json(e002)[k] for k in RECIPE_KEYS}, "dev 로 정한 레시피와 다릅니다"
    for h in cfg["history"]:
        rows.append({"seed": seed, "epoch": h["epoch"], "train_loss": round(h["train_loss"], 4),
                     "누적 시간(분)": round(h["seconds"] / 60, 1), "학습 통화": cfg["n_calls"], "전체 스텝": cfg["total_steps"],
                     "GPU 최대(GiB)": round(cfg["peak_cuda_memory_bytes"] / 2**30, 2)})
    print(f"prefix_s{seed}: 백본 {cfg['backbone']}")
    print("  pos_weight:", {c: round(w, 3) for c, w in zip(SYMPTOM_CLASSES, cfg["pos_weight"])})
pd.DataFrame(rows).set_index(["seed", "epoch"]) if rows else None
"""),
    md(r"""
## 5. 제출 번들 (Option A 미션 분리, version=2)

학습이 남긴 `best_model` 폴더들을 `.pt` 파일 하나로 묶는다 (`scripts/pack_bundle.py`, `m4/bundle.py`).
- **멤버마다:** tokenizer·config 파일 원문과 가중치(fp16 저장)가 들어 있어 인터넷 없이 로드된다.
- **미션 분리 구성 (`version=2`):**
  - **4-1:** `m4_softjaccard_v1_s42` · `m4_softjaccard_v1_s43` (2멤버, `threshold 0.5`, `force_top1=False`, `halt never`).
  - **4-2:** `E001` (통화 전체 `fit` 모델) · `E002` · `E007` (앞부분 `fit` 2종) + `policy_fitonly_c128_v1_nonempty.json` (`β = 128`, `τ = 0.5`, `require_nonempty=True`).
- Option B (공유 `ENS3` 단일 번들) fallback 시에도 같은 로더가 `version=1` 을 지원한다.

아래 셀은 번들 파일을 직접 열어 구성과 설정을 출력한다.
"""),
    code(r"""
def _print_bundle_section(tag, members, settings):
    params = []
    print(f"[{tag}]")
    for member in members:
        n = sum(int(t.numel()) for t in member["state_dict"].values())
        params.append(n)
        print(f"  멤버 {member['name']}: 파라미터 {n:,}, 파일 {sorted(member['files'])}")
    for key in ("fmt", "max_length", "truncation_side", "rule", "halt_rule"):
        print(f"  {key}: {settings.get(key)}")
    halt_raw = settings.get("halt") or {}
    halt = {k: (f"(LightGBM 모델 {len(v):,}자)" if k == "model_str" else f"({len(v)}개)" if k == "feature_names" else
                [round(x, 3) for x in v] if k == "logit_shift" and v else v)
            for k, v in halt_raw.items() if k != "info"}
    print(f"  halt: {halt}")
    print("  logit_shift:", None if settings.get("logit_shift") is None else [round(v, 3) for v in settings["logit_shift"]])
    return params


if BUNDLE.is_file():
    payload = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    print(f"{BUNDLE.relative_to(HERE).as_posix()}: {BUNDLE.stat().st_size / 2**20:.0f} MiB, "
          f"version={payload.get('version', 1)}, sha256 {hashlib.sha256(BUNDLE.read_bytes()).hexdigest()[:16]}…")
    print("note:", payload.get("note", ""))
    member_params = []
    if "missions" in payload:
        for mkey, block in payload["missions"].items():
            member_params.extend(_print_bundle_section(mkey, block["members"], block["settings"]))
    else:
        member_params.extend(_print_bundle_section("shared", payload["members"], payload["settings"]))
else:
    member_params = []
    print("번들이 아직 없습니다:", BUNDLE)
"""),
    md(r"""
## 6. 4-2 중단 정책 학습

`scripts/train_halt_policy.py` 가 정책을 학습하고 판정한다 (수식과 특징은 `m4/halting.py`).
- **학습 자료:** 학습용 26,280통화의 교차 학습 확률표 (`scripts/run_oof_tables.sh`). 5겹으로 나눠, 겹마다 나머지 4겹으로 학습한 모델이 보지 않은 겹의 확률을 낸다.
- **기준 τ 는 0.5 로 고정** 하고 `--require-nonempty` 게이트(보정 전 확률 `≥ 0.5` 인 증상이 1개 이상일 때만 조기 중단)를 켜며, 예상 이득에 곱하는 `β` 는 fit 26,280통화 안에서 5겹 교차 학습 예측으로 고른다 (`C128_v1_nonempty`: `β = 128`). `dev` 는 판정에만 사용한다.
- **멈출 때 예측:** 학습 목표와 채점 모두 제출과 같은 0.5 판정(1위 강제 없음)으로 계산한다.
- **판정:** dev 2,920통화에 그대로 적용해 `ENS3` 기준 `0.5074` (`Macro-F1 0.6132 × 조기판단력 0.8275`), 중간 빈 예측 조기 중단 `0`건을 확인했다.

아래 셀은 제출 번들에 넣은 정책의 학습 기록(`result.json`)을 출력한다.
"""),
    code(r"""
POLICY = HERE / os.environ.get("M4_POLICY", "runs/E026_learned_halt05/wide")
if (POLICY / "result.json").is_file():
    res = load_json(POLICY / "result.json")
    run = res["runs"][0]
    print("학습용 확률표:", [Path(p).as_posix() for p in res["args"]["fit_tables"]])
    print("판정용 확률표:", [Path(p).as_posix() for p in res["args"]["eval_tables"]])
    print(f"특징 {res['n_features']}개, 고른 값 beta {run['beta']:g} / tau {run['tau']:.2f}, 판정 통화 {run['n_eval_calls']:,}건")
    names = {"base": f"기존 기준 (기대 점수 ≥ {run['base_tau']:.2f})", "tau_only": "τ 만 다시 고름", "learned": "학습형 정책 (제출)"}
    display(pd.DataFrame({names[k]: {"4-2": round(v["score"], 4), "Macro-F1": round(v["macro_f1"], 4), "조기판단력": round(v["earliness"], 4),
                                     "중단 발화(중앙값)": v["halt_median"], "끝까지 들은 비율": round(v["never_halted"], 3)}
                          for k, v in run["scores"].items()}).T)
    d = run["learned_vs_base"]
    print(f"학습형 − 기존 기준: {d['diff']:+.4f}, 95% 구간 [{d['lo']:+.4f}, {d['hi']:+.4f}]")
else:
    print("정책 학습 기록이 없습니다:", POLICY)
"""),
    md(r"""
## 7. dev 평가 결과 요약

수치는 모두 Training 내부 dev 2,920통화 기준이고, **학습용 26,280통화로 학습한 같은 레시피의 모델**로 잰 것이다.
Option A (`version=2`) 제출은 미션별 최선 모델을 분리 탑재한다:
- **4-1 제출 (`m4_softjaccard_v1_s42` + `s43`, 2멤버):** `Dev 2,920 = 0.526039` (`raw p >= 0.5`, `force_top1=False`), `score.py` 경고 `0`건.
- **4-2 제출 (`ENS3: E001+E002+E007` + `C128_v1_nonempty`, 3멤버):** `Dev 2,920 = 0.507411` (`Macro-F1 0.613217 × E 0.827457`), 중간 빈 예측 조기 중단 `0`건.
"""),
    code(r"""
from m4 import simulate as sim
from m4.decision import calibrate, expected_rule

DEV = {
    "E001": "runs/E001_m3_baseline",
    "E002": "runs/E002_prefix_uniform",
    "E007": "runs/E007_prefix_uniform_s43",
    "SJ42": "runs/m4_softjaccard_v1_s42",
    "SJ43": "runs/m4_softjaccard_v1_s43",
}
tables = {k: sim.PrefixTable.load(HERE / v / "dev_prefix.npz") for k, v in DEV.items() if (HERE / v / "dev_prefix.npz").is_file()}
rows = []
if {"E001", "E002", "E007"}.issubset(tables):
    shift = np.log(load_json(HERE / DEV["E002"] / "run_config.json")["pos_weight"])
    combos = [
        ("앞부분 학습 seed 42 (E002)", ["E002"]),
        ("앞부분 학습 seed 43 (E007)", ["E007"]),
        ("4-2 채택 앙상블 (ENS3: E001+E002+E007)", ["E001", "E002", "E007"]),
    ]
    if {"SJ42", "SJ43"}.issubset(tables):
        combos.append(("4-1 채택 앙상블 (SJ42+SJ43)", ["SJ42", "SJ43"]))
    for label, names in combos:
        t = tables[names[0]]
        probs = np.mean([tables[k].probs for k in names], axis=0)
        pred = probs >= 0.5                                      # 제출 규칙: 0.5 고정, 넘는 것이 없으면 빈 예측
        gold = np.repeat(t.labels, t.n_utt, axis=0)
        per_call = np.add.reduceat(sim.moment_scores(pred, gold), t.offsets[:-1]) / t.n_utt
        value = expected_rule(calibrate(probs, shift))[1]          # 기대 점수 (보정 확률)
        halts = []
        for a, b in zip(t.offsets[:-1], t.offsets[1:]):
            hit = np.flatnonzero((value[a:b] >= 0.5) & pred[a:b].any(axis=1))
            hit = hit[hit >= 4]                                  # 5번째 발화부터 (require_nonempty)
            halts.append(int(hit[0]) + 1 if len(hit) else int(b - a))
        halts = np.array(halts)
        f1 = sim.macro_f1(pred[t.offsets[:-1] + halts - 1], t.labels)["macro_f1"]
        e = float(sim.earliness(halts, t.n_utt).mean())
        rows.append({"모델": label, "4-1": round(float(per_call.mean()), 4), "빈 예측 순간": f"{(~pred.any(1)).mean():.1%}",
                     "4-2 (기대 점수 ≥ 0.5, nonempty)": round(f1 * e, 4), "Macro-F1": round(f1, 4), "조기판단력": round(e, 4),
                     "중단 발화(중앙값)": float(np.median(halts))})
pd.DataFrame(rows).set_index("모델") if rows else print("dev 확률표(runs/E00*/dev_prefix.npz)가 없습니다")
"""),
    md(r"""
## 8. 계산 효율
"""),
    code(r"""
train_minutes = [load_json(p)["training_seconds"] / 60 for p in (FINAL / f"prefix_s{s}" / "run_config.json" for s in SEEDS) if p.is_file()]
pd.DataFrame({"값": [
    f"{sum(member_params):,} (KLUE-RoBERTa-base × {len(member_params)} 멤버 — 4-1 2개, 4-2 3개)" if member_params else "(번들 없음)",
    ", ".join(f"{m:.1f}분" for m in train_minutes) if train_minutes else "(최종 번들은 26,280통화 fit-only Option A 사용)",
    f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}, Python {platform.python_version()}, torch {torch.__version__}",
]}, index=["번들 파라미터 수", "학습 시간 / 레시피 비고", "학습 환경"])
"""),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-exec", action="store_true", help="실행하지 않고 생성만 한다")
    parser.add_argument("--out", default=str(NOTEBOOK), help="저장 경로 (기본 model_train.ipynb)")
    args = parser.parse_args()
    nb = build()
    if not args.no_exec:
        from nbclient import NotebookClient

        # 셀 안에서 학습을 돌릴 때(M4_RUN_TRAINING=1)는 시간 제한을 두지 않는다
        timeout = None if os.environ.get("M4_RUN_TRAINING") == "1" else 1800
        NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    out = Path(args.out)
    nbf.write(nb, out)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
