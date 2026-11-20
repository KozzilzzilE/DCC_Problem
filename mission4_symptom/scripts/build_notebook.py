"""Mission 4 제출용 학습 기록 노트북(model_train.ipynb)을 만들고 실행한다.

학습은 명령줄에서 `bash scripts/run_final_training.sh` (방안 A, fit 26,280통화 기준) 로 돌린다.
노트북은 데이터 일치 검증, 레시피 근거, TAPT·분류기 학습 기록, 제출 번들(version=2) 점검,
4-2 무누수 중단 정책(C128_v1_nonempty), Dev 2,920 2계층 E2E 검증 결과를 셀 출력으로 남긴다.
노트북은 Validation 폴더를 절대 읽지 않는다.

    python scripts/build_notebook.py --no-exec   # 생성만 (출력 없음)
    python scripts/build_notebook.py             # 생성 + 실행 (검증·로그 출력 수록)

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
# Mission 4 — 환자 증상 조기예측: 최종 제출 모델 학습·검증 기록 (방안 A)

**과제**: 119 신고 통화의 발화가 하나씩 들어올 때마다 증상 9종을 예측한다 (`4-1`: 매 순간 예측, `4-2`: 스스로 멈추고 예측).

**최종 채택 모델 (방안 A, `version=2` 미션 분리 번들)**
1. **공통 백본·분할:** `klue/roberta-base` 를 Training 내부 학습 분할 `fit 26,280`통화 전사로 추가 사전학습(TAPT, MLM `20 epoch`)한 백본을 출발점으로 쓴다 (`dev 2,920`통화는 판정·비교 전용, `Validation 3,640`은 미사용).
2. **4-1 증상 분류 (`SoftJaccard` 2-seed 앙상블: `s42 + s43`):** 통화 앞부분 균등 표본(`1..n`, epoch당 `2`개, `3 epoch`, `lr=5e-5`, `LLRD=0.8`, `pos_weight=(음성/양성)^0.5`, `max_length=512`)에 `Weighted BCE + 1.0 * Soft-Jaccard` 손실을 직접 결합해 초반 빈 예측을 줄이고 `4-1` 순간 Jaccard 식에 정렬했다.
3. **4-2 증상 분류 및 중단 (`ENS3: E001+E002+E007` + `C128_v1_nonempty`):** 통화 끝 `Macro-F1` 이 가장 높은 통화 전체 학습 모델 `E001` 과 앞부분 학습 모델 `E002`·`E007` 의 평균 확률(`>= 0.5` 고정)을 쓰고, `fit 26,280` 5겹 교차 학습(OOF)으로만 학습한 `LightGBM` 조기중단 정책(`β=128`, `τ=0.5`, `min_step=5`, `require_nonempty=True`)을 결합했다.
4. **대회 규정 준수:** 판정·중단 임계값은 모두 `0.5` 고정(`force_top1=False`)이며, 추론 입력에는 그 순간까지 받은 발화의 `text`·`speaker`·`start_ms`·`end_ms` 만 쓴다.

**이 노트북의 구조**
- 통합 재현 스크립트 `bash scripts/run_final_training.sh` 와 개별 실험 로그·설정·번들·Dev E2E 요약을 읽어 실행 출력을 남긴다.
- 출력에는 개인 로컬 절대경로·통화 ID·전사 원문을 남기지 않는다.
"""),
    code(r"""
import hashlib, json, os, platform, subprocess, sys, tempfile, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*IProgress.*")
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch, transformers

HERE = Path.cwd() if (Path.cwd() / "m4").is_dir() else Path.cwd().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from m4 import SYMPTOM_CLASSES
from m4.data import build_cache, label_matrix, load_cache, read_ids

def resolve_input(name: str) -> Path:
    env_dir = os.environ.get("M4_INPUT_DIR")
    if env_dir and (Path(env_dir) / name).is_file():
        return Path(env_dir) / name
    return HERE / "runs" / "cache" / name

def find_train_label_dir() -> Path | None:
    if os.environ.get("DCC_TRAIN_LABEL_DIR"):
        p = Path(os.environ["DCC_TRAIN_LABEL_DIR"])
        if p.is_dir():
            return p
    for parent in [HERE, *HERE.parents]:
        cand = parent / "data" / "train" / "label"
        if cand.is_dir() and next(cand.glob("*.json"), None) is not None:
            return cand
    return None

def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def safe_label(path: Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(HERE.resolve()).as_posix()
    except ValueError:
        return f"<외부 보관>/{p.parent.name}/{p.name}" if p.parent.name else f"<외부 보관>/{p.name}"

TRAIN_LABEL_DIR = find_train_label_dir()
CACHE = resolve_input("train.jsonl")
FIT_IDS_PATH = resolve_input("fit_ids.txt")
DEV_IDS_PATH = resolve_input("dev_ids.txt")
BUNDLE = Path(os.environ.get("M4_BUNDLE") or (HERE / "ckpt" / "mission4.pt"))
TAPT_CONFIG = Path(os.environ.get("M4_TAPT_CONFIG") or (HERE / "../../mission3-calibration/mission3_symptom/runs/devsel/tapt/tapt_config.json"))
SJ42_DIR = Path(os.environ.get("M4_SJ42_DIR") or (HERE / "runs" / "m4_softjaccard_v1_s42"))
SJ43_DIR = Path(os.environ.get("M4_SJ43_DIR") or (HERE / "runs" / "m4_softjaccard_v1_s43"))
E002_DIR = HERE / "runs" / "E002_prefix_uniform"
E007_DIR = HERE / "runs" / "E007_prefix_uniform_s43"
FINAL = HERE / os.environ.get("FINAL_ROOT", "runs/final_option_a")

print("Python", platform.python_version(), "| torch", torch.__version__, "| transformers", transformers.__version__, "| lightgbm", lgb.__version__)
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "| OS:", platform.system(), platform.machine())
print("작업 폴더:", HERE.name, "| 번들 존재:", BUNDLE.is_file(), "| Training 라벨 준비:", TRAIN_LABEL_DIR is not None)
"""),
    md(r"""
## 1. 데이터 준비 — Training 라벨 JSON 과 `fit 26,280` / `dev 2,920` 분할 검증

`m4.data.read_call` 이 라벨 JSON 하나를 통화 하나로 읽는다.
- **발화 순서:** `startAt`, `endAt` 기준 오름차순 정렬(추론 템플릿 `inference.py` 와 동일).
- **입력으로 남기는 것:** 발화의 `text`·`speaker`·`startAt`·`endAt` 만 사용한다 (`Validation` 폴더는 전혀 읽지 않음).
- **라벨 필터:** 9종 증상 필터 후 빈 통화는 제외한다 (`29,200`건 유지).

아래 셀은 원본 Training 라벨 JSON 에서 캐시를 다시 만들어 학습에 쓴 `train.jsonl` 과 줄 단위로 일치하는지 검증하고, `fit 26,280` 기준 양성 비율·`pos_weight` 를 확인한다.
"""),
    code(r"""
if TRAIN_LABEL_DIR is not None and CACHE.is_file():
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = Path(tmp) / "train.jsonl"
        n_rebuilt = build_cache(TRAIN_LABEL_DIR, rebuilt)
        same = rebuilt.read_text(encoding="utf-8").splitlines() == CACHE.read_text(encoding="utf-8").splitlines()
    assert same and n_rebuilt == 29200, f"Training 캐시 불일치: n={n_rebuilt}, same={same}"
    print(f"원본 Training JSON 재구축 통화 {n_rebuilt:,}건 ({time.perf_counter() - t0:.1f}초) / 학습 캐시 줄 단위 일치: {same}")
    print("train.jsonl sha256:", hashlib.sha256(CACHE.read_bytes()).hexdigest()[:16], "…")
else:
    print("원본 라벨 JSON 또는 train.jsonl 이 현재 환경에 없어 캐시 검증을 건너뜁니다.")

if CACHE.is_file() and FIT_IDS_PATH.is_file() and DEV_IDS_PATH.is_file():
    calls = load_cache(CACHE)
    fit_ids = read_ids(FIT_IDS_PATH)
    dev_ids = read_ids(DEV_IDS_PATH)
    assert len(calls) == 29200 and len(fit_ids) == 26280 and len(dev_ids) == 2920
    assert len(set(fit_ids) & set(dev_ids)) == 0
    fit_calls = [c for c in calls if c.call_id in set(fit_ids)]
    labels_all = label_matrix(calls)
    labels_fit = label_matrix(fit_calls)
    n_utt = np.array([c.n_utt for c in calls])
    print(f"분할 확인: Training 전체 {len(calls):,}건 = fit {len(fit_calls):,}건 + dev {len(dev_ids):,}건 (중복 0건)")
    print(f"발화 수 분포: 중앙값 {int(np.median(n_utt))}, 10% {int(np.percentile(n_utt, 10))}, 90% {int(np.percentile(n_utt, 90))}, 최대 {int(n_utt.max())}")
    count = labels_all.sum(axis=1)
    print("통화당 증상 수 비율:", {int(k): f"{(count == k).mean():.1%}" for k in sorted(set(count.tolist()))})
    pw_fit = ((len(labels_fit) - labels_fit.sum(axis=0)) / labels_fit.sum(axis=0)) ** 0.5
    display(pd.DataFrame({
        "fit 양성 통화": labels_fit.sum(axis=0),
        "fit 양성 비율": labels_fit.mean(axis=0).round(4),
        "pos_weight=(음성/양성)^0.5": pw_fit.round(4),
    }, index=SYMPTOM_CLASSES))
"""),
    md(r"""
## 2. 레시피 정하기 — Training 내부 dev

Training 을 통화 단위로 90/10 으로 나눴다 (예선 Mission 3 과 같은 분할, seed 1234).
- **학습용 26,280통화 (`fit`):** 후보 모델·TAPT·5겹 OOF 중단 정책을 학습한다.
- **내부 검증 2,920통화 (`dev`):** 학습 입력 방식, 손실 함수, 중단 기준, 앙상블 조합을 공정 비교한다.

dev 채점은 주최측 `score.py` 와 동일한 식을 쓰며(`m4/simulate.py`, 단위 테스트로 값 일치 검증), 아래 표는 `docs/results.csv` 와 `E002` 기본 레시피 기록이다.

정한 것:
- **학습 입력:** 통화 전체가 아니라 앞부분(길이 균등 표본)으로 학습한다 (`E002`). `4-1` 채점이 모든 순간을 같은 무게로 평균하므로 학습 분포도 순간에 대해 균등하게 맞췄다.
- **4-1 모델·결정 규칙 (Option A 분리):** `KLUE-RoBERTa-base (TAPT)` 에 `Weighted BCE(pos_weight^0.5) + Soft-Jaccard (jaccard_weight = 1.0)` 를 결합한 `2`개 시드(`seed 42·43`, `m4_softjaccard_v1`)를 평균하고, 대회 규정대로 판정 임계값 `0.5` 고정(`force_top1=False`)으로 예측한다.
- **4-2 모델·중단 기준 (Option A 분리):** `4-2` 는 통화 끝(중단 시점) F1 이 핵심이므로 `final_step Macro-F1` 이 더 높은 `ENS3`(`E001` 통화 전체 1종 + `E002/E007` 앞부분 2종)을 쓰고, `fit 26,280`통화 OOF 로 학습한 `LightGBM` 조기중단 정책(`C128_v1_nonempty`, `β = 128`, `τ = 0.5` 고정, `min_step = 5`, `require_nonempty = True`)으로 멈춘다.
- **해 봤지만 채택하지 않은 것:** 다른 계열 인코더(`E008 KF-DeBERTa`, `E019 RoBERTa-large`, `E021/E022 KcELECTRA`), `Full-only` 단독 학습(`raw 0.5` 1발화 폭주 취약), 4멤버 확장(`ENS4`), 증상 개수 보조 출력(`E014`), `Soft-Jaccard`/`Soft-Macro-F1` 의 `4-2` 직접 적용(`final_step Macro-F1` 하락).
"""),
    code(r"""
results = pd.read_csv(HERE / "docs" / "results.csv")
print(f"실험 기록 수: {len(results)}행 (docs/results.csv)")
display(results[["exp_id", "what", "m4_1_score", "m4_1_rule", "m4_2_score", "m4_2_policy", "m4_2_macro_f1", "m4_2_earliness"]].fillna(""))

RECIPE_KEYS = ["fmt", "truncation_side", "max_length", "epochs", "samples_per_call", "include_full", "batch_size",
               "lr", "llrd", "pos_weight_power"]
e002_cfg_path = E002_DIR / "run_config.json"
if e002_cfg_path.is_file():
    cfg = load_json(e002_cfg_path)
    print("E002 기본 레시피:", {k: cfg[k] for k in RECIPE_KEYS})
    print(f"E002 학습 요약: 통화 {cfg['n_calls']:,}건, {cfg['total_steps']:,} 스텝, {cfg['training_seconds'] / 60:.1f}분")
"""),
    md(r"""
## 3. TAPT 백본 (`fit 26,280` 전사 20 epoch MLM)

`klue/roberta-base` 를 119 구급 통화 전사(`fit 26,280`건, 라벨·`dev`·`Validation` 미사용)로 Masked LM 추가 사전학습한 백본이다.
- **설정:** `20 epoch`, `mlm_probability = 0.15`, `lr = 5e-5`, `warmup_ratio = 0.06`, 선형 감쇠, `max_length = 512`.
- 아래 셀은 실제 사용한 TAPT 백본의 `tapt_config.json` 과 epoch별 MLM 손실 추이를 출력한다.
"""),
    code(r"""
if TAPT_CONFIG.is_file():
    tapt = load_json(TAPT_CONFIG)
    print("TAPT 기록:", safe_label(TAPT_CONFIG))
    print(f"기반 모델: {tapt['model_name_or_path']} | 본문: {tapt['num_texts']:,}건 | {tapt['epochs']} epoch | 소요: {tapt['training_seconds'] / 60:.1f}분")
    print("데이터 출처:", tapt["source"])
    hist_df = pd.DataFrame(tapt["history"]).set_index("epoch").round(4).T
    display(hist_df)
    print(f"MLM loss 변화: epoch 1 ({tapt['history'][0]['mlm_loss']:.4f}) -> epoch 20 ({tapt['history'][-1]['mlm_loss']:.4f})")
else:
    print("TAPT 기록을 찾지 못했습니다.")
"""),
    md(r"""
## 4. 방안 A 모델 학습 기록 (`fit 26,280`통화)

제출 번들(`ckpt/mission4.pt`, `version=2`)에 탑재된 `fit 26,280` 기반 모델들의 학습 로그와 설정을 검증한다 (`bash scripts/run_final_training.sh` 로 전체 재현 가능).
- **4-1 멤버 (2종):** `m4_softjaccard_v1_s42`, `m4_softjaccard_v1_s43` (`Weighted BCE + 1.0 * Soft-Jaccard`, `fit 26,280`, `9,855` 스텝).
- **4-2 멤버 (3종):** `E001` (통화 전체 `fit` 모델) + `E002` (앞부분 `seed 42`) + `E007` (앞부분 `seed 43`).
"""),
    code(r"""
TRAIN_CMD = ["bash", "scripts/run_final_training.sh"]
print("방안 A 통합 재현 명령 (mission4_symptom 폴더):", " ".join(TRAIN_CMD))
if os.environ.get("M4_RUN_TRAINING") == "1":
    proc = subprocess.Popen(TRAIN_CMD, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"학습 스크립트 실패 (exit {proc.returncode})")
else:
    print("이번 실행에서는 GPU 재학습을 수행하지 않고, 완료된 학습 설정·로그를 검증해 출력합니다.")
"""),
    code(r"""
def print_compact_log(tag: str, log_path: Path) -> None:
    print(f"----- {tag} ({safe_label(log_path)}) -----")
    lines = [l.rstrip() for l in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    if len(lines) <= 18:
        chosen = lines
    else:
        chosen = lines[:6] + ["... (중간 스텝 생략) ..."] + lines[-8:]
    for line in chosen:
        print(line)

for tag, run_dir in [("SJ42 (4-1)", SJ42_DIR), ("SJ43 (4-1)", SJ43_DIR), ("E002 (4-2)", E002_DIR), ("E007 (4-2)", E007_DIR)]:
    lp = run_dir / "train_log.txt"
    if lp.is_file():
        print_compact_log(tag, lp)
        print()
    else:
        print(f"{tag}: train_log.txt 없음")
"""),
    code(r"""
TRAINED_MEMBERS = [
    ("SJ42", "4-1 (BCE+SoftJaccard 1.0)", SJ42_DIR),
    ("SJ43", "4-1 (BCE+SoftJaccard 1.0)", SJ43_DIR),
    ("E002", "4-2 (BCE prefix s42)", E002_DIR),
    ("E007", "4-2 (BCE prefix s43)", E007_DIR),
]
rows = []
for name, role, rdir in TRAINED_MEMBERS:
    cfg_p = rdir / "run_config.json"
    if not cfg_p.is_file():
        print(f"{name}: run_config.json 없음")
        continue
    cfg = load_json(cfg_p)
    ids_name = Path(str(cfg.get("ids") or "")).name
    assert ids_name == "fit_ids.txt" and cfg["n_calls"] == 26280, f"{name} 분할 오류"
    assert "val" not in str(cfg.get("calls", "")).lower(), f"{name} Validation 참조 발견"
    assert int(cfg["total_steps"]) == 9855, f"{name} 스텝 수 불일치: {cfg['total_steps']}"
    jw = float(cfg.get("jaccard_weight", 0.0))
    expect_jw = 1.0 if name.startswith("SJ") else 0.0
    assert abs(jw - expect_jw) < 1e-6, f"{name} jaccard_weight={jw}"
    for h in cfg["history"]:
        rows.append({
            "모델": name,
            "역할": role,
            "seed": cfg["seed"],
            "epoch": h["epoch"],
            "train_loss": round(float(h["train_loss"]), 4),
            "누적 시간(분)": round(float(h["seconds"]) / 60.0, 1),
            "학습 통화": cfg["n_calls"],
            "전체 스텝": cfg["total_steps"],
            "GPU 최대(GiB)": round(float(cfg["peak_cuda_memory_bytes"]) / 2**30, 2),
        })
    print(f"{name} 검증 통과: calls={cfg['n_calls']:,}, steps={cfg['total_steps']:,}, jaccard_weight={jw}, elapsed={cfg['training_seconds']/60:.1f}분")
    print("  pos_weight:", {c: round(w, 3) for c, w in zip(SYMPTOM_CLASSES, cfg["pos_weight"])})

display(pd.DataFrame(rows).set_index(["모델", "epoch"])) if rows else None
"""),
    md(r"""
## 5. 제출 번들 (`ckpt/mission4.pt`, Option A `version=2`) 점검

`scripts/pack_bundle.py` 와 `m4/bundle.py` 로 묶인 단일 제출 가중치 번들을 열어 구성·무결성·규정 준수 메타데이터를 확인한다.
- **미션 분리 탑재:** `4-1` 실행 시 `SoftJaccard 2멤버` 만 적재하고, `4-2` 실행 시 `ENS3 3멤버 + C128_v1_nonempty` 정책만 적재한다.
- 번들 안에 `tokenizer`, `config`, `fp16` 가중치, 결정 규칙, 중단 정책이 모두 포함돼 오프라인 환경에서 외부 다운로드 없이 추론한다.
"""),
    code(r"""
def _print_bundle_section(tag, members, settings):
    params = []
    print(f"[{tag}]")
    for member in members:
        n = sum(int(t.numel()) for t in member["state_dict"].values())
        params.append(n)
        print(f"  멤버 {member['name']}: 파라미터 {n:,}, 내장 파일 {sorted(member['files'])}")
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
    bundle_sha = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    assert bundle_sha == "26c3ff2729e67916d2543f216a1c6c342d399f5b24e3ade9c7509e534d3c92b8", f"번들 해시 불일치: {bundle_sha}"
    payload = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    print(f"번들: {safe_label(BUNDLE)} | {BUNDLE.stat().st_size:,} B ({BUNDLE.stat().st_size / 2**20:.1f} MiB) | version={payload.get('version', 1)}")
    print(f"sha256: {bundle_sha}")
    print("note:", payload.get("note", ""))
    member_params = []
    for mkey, block in payload["missions"].items():
        member_params.extend(_print_bundle_section(mkey, block["members"], block["settings"]))
    print(f"전체 번들 파라미터 합계: {sum(member_params):,} ({len(member_params)}개 멤버)")
else:
    member_params = []
    print("번들이 없습니다:", safe_label(BUNDLE))
"""),
    md(r"""
## 6. 4-2 무누수 학습형 중단 정책 (`C128_v1_nonempty`)

`scripts/train_halt_policy.py` 와 `m4/halting.py` 를 사용해, **`dev 2,920` 과 `Validation 3,640` 을 제외한 순수 `fit 26,280` 5겹 OOF 확률표(`e002_oof`)만** 으로 중단 정책을 학습했다.
- **중단 기준:** `5`번째 발화부터 ① 현재 보정 전 평균 확률 `>= 0.5` 인 증상이 최소 `1`개 이상(`require_nonempty=True`) 이고 ② `기대 점수 + 128 × δ̂ ≥ 0.5` (`τ = 0.5` 고정)일 때 멈춘다.
- **특징 (`57`개):** 그 순간까지 누적된 `text`, `speaker`, `start_ms`, `end_ms` 와 보정 확률만 사용(미래 길이 미사용, `200`통화 인과성/온라인 일치 검증 완료).
"""),
    code(r"""
fitonly_summary_p = HERE / "runs" / "e026_fitonly_policy_v1" / "summary.json"
if fitonly_summary_p.is_file():
    pol_res = load_json(fitonly_summary_p)
    caus = pol_res["causality_and_online_parity"]
    print(f"정책 학습 범위: fit {pol_res['n_fit']:,}통화 OOF ({Path(pol_res['oof_source']).name}) | 판정 전용 dev {pol_res['n_eval']:,}통화")
    print(f"OOF δ 회귀 R²: {pol_res['crossfit_r2']:.4f} | 인과성·온라인 일치 검증({caus['checked_calls']}통화): future_invariance={caus['future_invariance_ok']}, parity={caus['online_batch_parity_ok']}")
    ens3 = pol_res["targets"]["ENS3"]
    policy_names = {
        "C0_beta0": "기준 정책 (β=0, τ=0.5, nonempty)",
        "C128_v0": "학습형 v0 (β=128, τ=0.5, empty 허용)",
        "C128_v1_nonempty": "최종 채택 C128_v1_nonempty (β=128, τ=0.5, require_nonempty)",
    }
    pol_rows = {}
    for k, label in policy_names.items():
        v = ens3[k]
        pol_rows[label] = {
            "4-2 점수": round(v["score"], 6),
            "Macro-F1": round(v["macro_f1"], 4),
            "조기판단력": round(v["earliness"], 4),
            "중단 발화(중앙값)": v["halt_median"],
            "끝까지 들은 비율": round(v["never_halted"], 4),
            "중단 빈 예측 건수": v["empty_at_halt"],
        }
    display(pd.DataFrame(pol_rows).T)
    d_c0 = ens3["C128_v1_nonempty"]["diff_vs_C0"]
    print(f"C128_v1_nonempty - 기준(C0): {d_c0['diff']:+.6f} (95% CI [{d_c0['lo']:+.6f}, {d_c0['hi']:+.6f}], P(>0)={d_c0['p_gt0']:.4f})")
else:
    print("정책 요약 파일이 없습니다.")
"""),
    md(r"""
## 7. Dev 2,920 평가 및 2계층 E2E 오프라인 검증 결과

Training 내부 비학습 `dev 2,920`통화(`87,814`시점)에서 개별·앙상블 `.npz` 확률표를 직접 채점하고, 실제 제출 번들 `ckpt/mission4.pt` 를 `inference.py` + 공식 `score.py` 로 전수·E2E 검증한 결과를 요약한다.
"""),
    code(r"""
from m4 import simulate as sim
from m4.decision import calibrate, expected_rule

NPZ_SOURCES = {
    "E001": resolve_input("e001_dev_prefix.npz"),
    "E002": resolve_input("e002_dev_prefix.npz"),
    "E007": resolve_input("e007_dev_prefix.npz"),
    "SJ42": SJ42_DIR / "dev_prefix.npz",
    "SJ43": SJ43_DIR / "dev_prefix.npz",
}
tables = {k: sim.PrefixTable.load(p) for k, p in NPZ_SOURCES.items() if p.is_file()}
if {"E001", "E002", "E007", "SJ42", "SJ43"}.issubset(tables):
    base_t = tables["E002"]
    for k, tbl in tables.items():
        assert tbl.call_ids == base_t.call_ids and np.array_equal(tbl.offsets, base_t.offsets), f"{k} 정렬 불일치"
    shift = np.log(load_json(E002_DIR / "run_config.json")["pos_weight"])
    combos = [
        ("E001 (통화 전체 BCE)", ["E001"]),
        ("E002 (앞부분 BCE s42)", ["E002"]),
        ("E007 (앞부분 BCE s43)", ["E007"]),
        ("ENS3 (4-2 채택: E001+E002+E007)", ["E001", "E002", "E007"]),
        ("SJ42 (앞부분 SoftJaccard s42)", ["SJ42"]),
        ("SJ43 (앞부분 SoftJaccard s43)", ["SJ43"]),
        ("SJ42+SJ43 (4-1 채택 2앙상블)", ["SJ42", "SJ43"]),
    ]
    eval_rows = []
    for label, names in combos:
        probs = np.mean([tables[k].probs for k in names], axis=0)
        pred = probs >= 0.5
        gold = np.repeat(base_t.labels, base_t.n_utt, axis=0)
        per_call = np.add.reduceat(sim.moment_scores(pred, gold), base_t.offsets[:-1]) / base_t.n_utt
        value = expected_rule(calibrate(probs, shift))[1]
        halts = []
        for a, b in zip(base_t.offsets[:-1], base_t.offsets[1:]):
            hit = np.flatnonzero((value[a:b] >= 0.5) & pred[a:b].any(axis=1))
            hit = hit[hit >= 4]
            halts.append(int(hit[0]) + 1 if len(hit) else int(b - a))
        halts = np.array(halts)
        f1 = sim.macro_f1(pred[base_t.offsets[:-1] + halts - 1], base_t.labels)["macro_f1"]
        e = float(sim.earliness(halts, base_t.n_utt).mean())
        eval_rows.append({
            "구성": label,
            "4-1 (raw>=0.5)": round(float(per_call.mean()), 6),
            "빈 예측 순간 비율": f"{(~pred.any(1)).mean():.2%}",
            "4-2 (기준 expected>=0.5, nonempty)": round(f1 * e, 6),
            "Macro-F1": round(f1, 4),
            "조기판단력": round(e, 4),
        })
    print("1) .npz 확률표 기반 Dev 2,920통화(87,814시점) 직접 채점 결과:")
    display(pd.DataFrame(eval_rows).set_index("구성"))

e2e_path = HERE / "runs" / "mission4_submission_v2_20261119" / "e2e_verification_summary.json"
if e2e_path.is_file():
    e2e = load_json(e2e_path)
    l1 = e2e["layer1_full_dev2920"]
    l2 = e2e["layer2_real_cli_subset16"]
    print("\n2) 실제 번들(ckpt/mission4.pt, version=2) 2계층 E2E 오프라인 검증 요약:")
    print(f"  - Layer 1 (Dev 2,920 전수, {l1['n_moments']:,}시점): 4-1={l1['mission_4_1']['score']:.6f} (적재 {l1['mission_4_1']['loaded_members']}멤버, 경고 {len(l1['mission_4_1']['warnings'])}건)")
    print(f"  - Layer 1 (Dev 2,920 전수, C128_v1_nonempty):   4-2={l1['mission_4_2']['score']:.6f} = Macro-F1 {l1['mission_4_2']['macro_f1']:.4f} x E {l1['mission_4_2']['mean_earliness']:.4f} (적재 {l1['mission_4_2']['loaded_members']}멤버, 중간 빈 중단 {l1['mission_4_2']['mid_call_empty_halts']}건, 불일치 {l1['mission_4_2']['mismatches_vs_ENS3_C128_v1_nonempty']}건)")
    print(f"  - Layer 2 (16통화 노모킹 오프라인 CLI CPU):       4-1 {l2['mission_4_1']['cli_seconds']:.2f}s (이진 차이 {l2['mission_4_1']['binary_row_diffs_vs_layer1']}건, max_prob_diff={l2['mission_4_1']['max_prob_diff_fp16_bundle_vs_npz']:.6f})")
    print(f"  - Layer 2 (16통화 노모킹 오프라인 CLI CPU):       4-2 {l2['mission_4_2']['cli_seconds']:.2f}s (중단 차이 {l2['mission_4_2']['call_diffs_vs_layer1']}건, max_prob_diff={l2['mission_4_2']['max_prob_diff_fp16_bundle_vs_npz']:.6f})")
    f1_df = pd.DataFrame.from_dict(l1["mission_4_2"]["per_class_f1"], orient="index", columns=["4-2 중단 시점 F1"]).round(4)
    display(f1_df.T)
"""),
    md(r"""
## 8. 계산 효율
"""),
    code(r"""
sj42_m = load_json(SJ42_DIR / "run_config.json")["training_seconds"] / 60 if (SJ42_DIR / "run_config.json").is_file() else None
sj43_m = load_json(SJ43_DIR / "run_config.json")["training_seconds"] / 60 if (SJ43_DIR / "run_config.json").is_file() else None
e002_m = load_json(E002_DIR / "run_config.json")["training_seconds"] / 60 if (E002_DIR / "run_config.json").is_file() else None
e007_m = load_json(E007_DIR / "run_config.json")["training_seconds"] / 60 if (E007_DIR / "run_config.json").is_file() else None
e2e_cli = load_json(HERE / "runs" / "mission4_submission_v2_20261119" / "e2e_verification_summary.json")["layer2_real_cli_subset16"]

summary_rows = {
    "총 번들 파라미터 / 런타임 적재": f"{sum(member_params):,} (KLUE-RoBERTa-base x 5, 실행 시 4-1 2멤버 / 4-2 3멤버 분리 적재)" if member_params else "확인 필요",
    "번들 파일 크기 (fp16)": f"{BUNDLE.stat().st_size / 2**20:.1f} MiB ({BUNDLE.stat().st_size:,} B)" if BUNDLE.is_file() else "없음",
    "4-1 학습 시간 (L4 AMP, 9,855스텝)": f"SJ42 {sj42_m:.1f}분, SJ43 {sj43_m:.1f}분 (각 fit 26,280)" if sj42_m and sj43_m else "기록 확인",
    "4-2 앞부분 학습 시간 (9,855스텝)": f"E002 {e002_m:.1f}분, E007 {e007_m:.1f}분 (fit 26,280)" if e002_m and e007_m else "기록 확인",
    "오프라인 CLI 실측 (CPU 16통화)": f"4-1 {e2e_cli['mission_4_1']['cli_seconds']:.2f}초 / 4-2 {e2e_cli['mission_4_2']['cli_seconds']:.2f}초 (HF_HUB_OFFLINE=1)",
}
display(pd.DataFrame({"값": list(summary_rows.values())}, index=list(summary_rows.keys())))
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

        timeout = None if os.environ.get("M4_RUN_TRAINING") == "1" else 1800
        NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    out = Path(args.out)
    nbf.write(nb, out)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
