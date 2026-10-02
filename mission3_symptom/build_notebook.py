"""Mission 3 제출용 학습 기록 노트북(model_train.ipynb)을 만들고 실행한다.

학습은 `run_dev_selection.sh` 로 이미 끝냈다 (약 5시간). 노트북이 하는 일:
  1) 원본 JSON 에서 학습 CSV 를 다시 만들어, 실제 학습에 쓴 CSV 와 같은지 확인한다.
  2) Training 내부 dev 로 하이퍼파라미터를 정한 각 단계의 로그와 결정 기록을 보여 준다.
     (runs/devsel/*/history.json, decisions/*.json)
  3) 결정이 끝난 제출 번들로 Validation 전체를 추론해 점수와 시간을 기록한다.
RUN_TRAINING=True 로 바꾸면 같은 스크립트로 처음부터 다시 학습한다.

    python build_notebook.py            # 노트북 생성 + 실행 (학습 작업공간의 runs/devsel 필요)
    python build_notebook.py --no-exec  # 생성만
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "model_train.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


CELLS = [
    md("""
# Mission 3 — 환자 증상 인식: 최종 제출 모델 학습 기록

**최종 모델**:
1. KLUE-RoBERTa-base 를 Training 대화 본문으로 TAPT(MLM) 20 epoch 추가 학습한다.
2. 층별 학습률 감쇠(LLRD 0.8, 최상위 lr 5e-5)와 pos_weight=(neg/pos)^0.5 로 9개 증상 다중 라벨 분류를 학습한다 (seed 42~45).
3. 3 epoch 스케줄의 2 epoch 시점 가중치 4개의 확률을 균등 평균하고, **모든 클래스 임계값 0.5** 로 판정한다.

**하이퍼파라미터는 모두 Training 내부 dev(10%, 2,920건)로 정했다.** 대상은 p, 레시피, 저장 epoch, TF-IDF 블렌드(C·w), 모델 수다. 주최 측 답변(p·w·C 는 Validation 이 아니라 Training 내부 dev/OOF 로 정하고, Validation 은 결정된 모델의 성능 확인에만 쓴다)을 따른 것이다. 선택 규칙은 결과를 보기 전에 `reports/dev_selection_protocol.md` 로 커밋했다.

**Validation macro F1@0.5 = 0.6599** (3,640건). 결정이 모두 끝난 뒤 확인한 값이며, Validation 은 선택에 쓰지 않았다.

규정 준수:
- **데이터:** 학습(역전파)에는 Training 데이터만 썼다. Validation 은 결정이 끝난 모델의 성능 확인에만 썼다.
- **입력:** 대화 본문(`utterances[].text`)만 쓴다. 화자·시간·인적사항은 파싱 단계에서 버린다.
- **판정:** 결정 임계값은 모든 클래스 0.5 고정이고, 클래스별 값은 없다.
- **모델:** 공개 사전학습 모델(klue/roberta-base)만 썼고, 상용 API 는 쓰지 않았다.

이 노트북은 학습 작업공간(`runs/devsel` 이 있는 폴더)에서 실행한 기록이다.
- **학습 방식:** `run_dev_selection.sh` 로 CLI 에서 학습했다. 각 셀은 그 스크립트가 남긴 로그를 읽어 출력한다.
- **제출 폴더에서:** `runs/` 가 없으므로 저장된 셀 출력으로 확인한다.
- **처음부터 재현:** `RUN_TRAINING = True` 로 바꾸면 같은 스크립트로 다시 학습한다 (약 5시간 10분, RTX 5060 기준).
"""),
    code("""
import json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch, transformers, sklearn

HERE = Path.cwd()                       # mission3_symptom/
sys.path.insert(0, str(HERE))
from m3.config import TARGET_SYMPTOMS
from m3.labels import load_transcripts_dir

RUN_TRAINING = False                    # True 면 run_dev_selection.sh 로 다시 학습 (약 5시간 10분)
DEVSEL = HERE / "runs" / "devsel"
DECISIONS = DEVSEL / "decisions"
BUNDLE = DEVSEL / "bundle"
SEEDS = [42, 43, 44, 45]

def find_data_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        label_dir = parent / "data" / "train" / "label"
        if label_dir.is_dir() and next(label_dir.glob("*.json"), None) is not None:   # 빈 자리표시 폴더는 건너뛴다
            return parent / "data"
    raise FileNotFoundError("data/train/label 폴더를 찾지 못했습니다. DCC_DATA_ROOT 를 지정하세요.")

def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            return subprocess.check_output(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                           text=True).strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()

def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def history_table(run: Path, label: str) -> pd.DataFrame:
    rows = [{"run": label, "epoch": e["epoch"], "train_loss": e["train_loss"], "dev_loss": e["val_loss"],
             "dev_F1@0.5": e["val_macro_f1_at_0_5"]} for e in load_json(run / "history.json")["epochs"]]
    return pd.DataFrame(rows)

DATA_ROOT = Path(os.environ.get("DCC_DATA_ROOT", "")) if os.environ.get("DCC_DATA_ROOT") else find_data_root(HERE)
print("Python", platform.python_version(), "| torch", torch.__version__, "| transformers", transformers.__version__, "| scikit-learn", sklearn.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음", "| CPU:", cpu_name(), "| OS:", platform.platform())
print("데이터:", DATA_ROOT)
"""),
    md("""
## 1. 데이터 준비 — 원본 JSON 에서 학습 CSV 만들기

`m3.labels.load_transcripts_dir` 가 대화 본문만 공백으로 이어 붙이고, 9개 대상 증상만 라벨 벡터로 남긴다. 추론 경로도 같은 함수를 쓰므로 학습 입력과 추론 입력이 어긋나지 않는다. 명령줄로는 `python make_csv.py --label-dir <data>/train/label --output data_csv/mission3_train.csv` 다.

실제 학습에는 팀이 먼저 만든 Training CSV 를 썼다. 아래 셀이 다시 만든 것과 행 단위로 대조한다.
"""),
    code("""
def build_dataframe(label_dir: Path) -> pd.DataFrame:
    rows = []
    for record in load_transcripts_dir(label_dir, sep_mode="space"):
        row = {"call_id": record.call_id, "text": record.text}
        row.update({s: int(record.label_vector[i]) for i, s in enumerate(TARGET_SYMPTOMS)})
        rows.append(row)
    return pd.DataFrame(rows)

t0 = time.perf_counter()
train_df = build_dataframe(DATA_ROOT / "train" / "label")
print(f"Training {len(train_df):,}건 ({time.perf_counter() - t0:.1f}초)")

used_train_csv = load_json(DEVSEL / "final_s42" / "run_config.json")["train_csv"]
original = pd.read_csv(used_train_csv, encoding="utf-8-sig").set_index("call_id")
rebuilt = train_df.set_index("call_id").loc[original.index]
text_diff = rebuilt["text"] != original["text"]
squash = lambda s: s.str.split().str.join(" ")
ws_only = int((text_diff & (squash(rebuilt["text"]) == squash(original["text"]))).sum())
label_same = int((rebuilt[TARGET_SYMPTOMS].values == original[TARGET_SYMPTOMS].values).all(axis=1).sum())
print(f"학습 CSV {len(original):,}행 중 본문 동일 {len(original) - int(text_diff.sum()):,}행"
      f" (다른 {int(text_diff.sum())}행 중 공백만 다른 행 {ws_only}), 라벨 동일 {label_same:,}행  [{Path(used_train_csv).name}]")

pd.DataFrame({
    "Training 양성": train_df[TARGET_SYMPTOMS].sum(),
    "비율": train_df[TARGET_SYMPTOMS].mean().round(3),
    "neg/pos": ((len(train_df) - train_df[TARGET_SYMPTOMS].sum()) / train_df[TARGET_SYMPTOMS].sum()).round(2),
})
"""),
    md("""
## 2. 하이퍼파라미터 선택 절차 — Training 내부 dev

Training 을 통화 단위로 90/10 분할했다 (`make_dev_split.py`, seed 1234).
- **학습용 분할:** 후보 모델을 학습한다.
- **dev 분할:** 다음 값을 정한다.
  - pos_weight 지수 p
  - 레시피 (원래 레시피 vs TAPT+LLRD)
  - 저장 epoch
  - TF-IDF 블렌드의 C·w
  - 모델 수

규칙(동점 처리 포함)은 `reports/dev_selection_protocol.md` 에 있고, dev 결과를 보기 전에 커밋했다. `devsel.py` 는 선택에 쓰는 run 이 모두 학습용 분할로 학습하고 dev 분할로 평가됐는지 `run_config.json` 으로 확인하며, 아니면 멈춘다. 그래서 선택 단계에서 Validation 이 쓰일 수 없다.
"""),
    code("""
split = load_json(DEVSEL / "data" / "split.json")
print(f"원본 {split['source_csv']} {split['source_rows']:,}건 (sha256 {split['source_sha256'][:16]}…)")
print(f"학습용 {split['train_rows']:,}건 / dev {split['dev_rows']:,}건, seed {split['seed']}")
print("dev 양성:", split["dev_positives"])

TRAIN_CMD = ["bash", "run_dev_selection.sh", used_train_csv, str(DATA_ROOT / "val" / "label")]
print("\\n재현 명령:", " ".join(TRAIN_CMD))
if RUN_TRAINING:
    subprocess.run(TRAIN_CMD, check=True)
"""),
    md("""
## 3. pos_weight 지수 p — 원래 레시피, seed 42

손실은 BCE 이고, pos_weight = (Training 음성/양성)^p 다 (음성·양성 수는 학습용 분할 라벨로 계산).
- **p=0:** 가중치 없는 BCE 다.
- **선택 규칙:** epoch 별 dev macro F1@0.5 중 최고값이 가장 큰 p 를 고른다.
"""),
    code("""
power = load_json(DECISIONS / "power.json")
table = pd.DataFrame({f"p={p}": d["dev_f1_by_epoch"] for p, d in power["details"].items()}).T
table.columns = [f"epoch {c}" for c in table.columns]
table["최고"] = table.max(axis=1)
print("결정: p =", power["choice"])
table.round(4)
"""),
    md("""
## 4. TAPT — 학습용 분할 본문으로 MLM 추가 사전학습 (20 epoch)

원본 KLUE-RoBERTa 는 이 통화 전사문에서 MLM 손실이 약 5.1 로, 문어체 사전학습 코퍼스와 도메인 차이가 크다.
- **데이터:** 학습용 분할 26,280건의 text 만 쓴다. 라벨은 쓰지 않고, dev 본문과 Validation 은 넣지 않았다.
- **설정:** 15% 마스킹, lr 5e-5, warmup 6%, 선형 감쇠.
- **평가 CSV 없음.**
"""),
    code("""
tapt = load_json(DEVSEL / "tapt" / "tapt_config.json")
print(f"학습 시간 {tapt['training_seconds'] / 60:.1f}분, 본문 {tapt['num_texts']:,}건, eval_csv={tapt['eval_csv']}")
pd.DataFrame(tapt["history"]).set_index("epoch").round(4).T
"""),
    md("""
## 5. 레시피와 저장 epoch — 정한 p 로 seed 42·43

후보 레시피는 두 가지다.
- **원래 레시피:** klue/roberta-base 에서 시작, lr 2e-5.
- **TAPT + LLRD 0.8:** 위 TAPT 모델에서 시작. 분류 헤드는 lr 5e-5 이고, 인코더 층을 내려갈수록 0.8 배씩 줄인다.

둘 다 3 epoch, 유효 배치 16(8×2), max_length 512, AMP 로 학습한다.

선택 규칙: 레시피마다 seed 평균 dev F1 이 가장 높은 epoch 를 찾고, 그 값이 큰 레시피를 고른다.
"""),
    code("""
final = load_json(DECISIONS / "final.json")
runs = {"original s42": DEVSEL / f"p{power['choice']}_s42", "original s43": DEVSEL / "orig_s43",
        "tapt_llrd s42": DEVSEL / "tl_s42", "tapt_llrd s43": DEVSEL / "tl_s43"}
display(pd.concat([history_table(path, name) for name, path in runs.items()]).set_index(["run", "epoch"]).round(4))
arms = pd.DataFrame({name: arm["mean_dev_f1_by_epoch"] for name, arm in final["arms"].items()}).T
arms.columns = [f"epoch {c}" for c in arms.columns]
print("결정:", final["decision"]["recipe"], "/ epoch", final["decision"]["epoch"])
arms.round(4)
"""),
    md("""
## 6. TF-IDF 블렌드 (C, w) 와 모델 수

선택된 레시피의 2 epoch dev 확률(seed 평균)에 TF-IDF LogisticRegression 확률을 섞는다. TF-IDF 는 학습용 분할로 적합했다 (char_wb 2-4 + 단어 1-2 gram, min_df 3, balanced).
- **(C, w):** 격자에서 dev F1 최고를 고른다.
- **w = 0:** TF-IDF 를 쓰지 않는다는 뜻이다.
- **모델 수:** 2-seed 평균이 단일 seed 평균보다 높으면 4개 앙상블을 쓴다.
"""),
    code("""
grid = pd.DataFrame(final["tfidf_table"]).pivot(index="w", columns="C", values="f1")
print("결정:", {k: final["decision"][k] for k in ["tfidf_C", "tfidf_weight", "n_members"]})
print("모델 수 판단:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in final["members_check"].items()})
grid.round(4)
"""),
    md("""
## 7. 최종 학습 — Training 전체, 정한 설정, seed 42~45

정한 설정 그대로 Training 전체(29,200건)로 학습했다.
- **설정:** TAPT 가중치에서 시작, lr 5e-5, LLRD 0.8, p 0.5.
- **저장:** 3 epoch 스케줄의 **2 epoch 시점에서 저장하고 멈춘다** (`--checkpoint-metric fixed_epoch --checkpoint-epoch 2`). 평가 점수를 보지 않는다.
- **진행 기록의 dev 점수:** 학습 스크립트가 평가 CSV 를 요구해 dev 분할을 넘겼다. 이 dev 는 학습 데이터 안에 있어 점수가 높게 나오며, 아무 결정에도 쓰지 않았다.
"""),
    code("""
cfg = load_json(DEVSEL / "final_s42" / "run_config.json")
print({k: cfg[k] for k in ["model_name_or_path", "learning_rate", "llrd_decay", "pos_weight_power", "epochs",
                           "train_batch_size", "gradient_accumulation_steps", "max_length", "amp"]})
print("체크포인트:", cfg["checkpoint_selection_criterion"])
print("pos_weight:", {s: round(v["pos_weight"], 3) for s, v in cfg["train_pos_weight_statistics"].items()})
rows = []
for seed in SEEDS:
    run = DEVSEL / f"final_s{seed}"
    metrics = load_json(run / "baseline_metrics.json")
    for e in load_json(run / "history.json")["epochs"]:
        rows.append({"seed": seed, "epoch": e["epoch"], "train_loss": round(e["train_loss"], 4),
                     "저장": "✓" if e["epoch"] == metrics["best_epoch"] else "",
                     "학습 시간(분)": round(metrics["training_seconds"] / 60, 1),
                     "GPU 최대(GiB)": round(metrics["peak_cuda_memory_bytes"] / 2**30, 2)})
pd.DataFrame(rows).set_index(["seed", "epoch"])
"""),
    md("""
## 8. 제출 번들

최종 4개의 `best_model` 을 한 폴더에 모으고 `ensemble.json` 에 구성과 정밀도를 적는다. 제출 폴더에서는 이 번들이 `ckpt/` 이고, `--ckpt_path ckpt/ensemble.json` 으로 추론한다. 멤버 폴더에는 tokenizer·config·가중치가 모두 있어 인터넷 없이 로드된다.
"""),
    code("""
print(json.dumps(load_json(BUNDLE / "ensemble.json"), ensure_ascii=False, indent=2))
for member in sorted(p for p in BUNDLE.iterdir() if p.is_dir()):
    print(f"{member.name}/: " + ", ".join(sorted(f.name for f in member.iterdir())))
"""),
    md("""
## 9. Validation 확인 — 결정이 끝난 번들로 전체 추론

결정은 모두 끝난 상태다. 이 단계는 성능 확인일 뿐이다.
- **첫 확인 기록:** 2026-10-01 22:42 에 `run_dev_selection.sh` 10 단계에서 한 번 확인했다. 그때의 점수와 번들 해시가 `decisions/validation_once.json`, `logs/validation_attempts.log` 에 있다.
- **이 셀:** 같은 번들로 제출과 같은 명령을 다시 실행해 시간을 재고, 첫 확인과 예측이 같은지 본다.
- **판정 임계값:** 0.5 고정이다.
"""),
    code("""
once = load_json(DECISIONS / "validation_once.json")
print(f"첫 확인 (결정 후 1회): macro F1@0.5 = {once['macro_f1']:.4f}, {once['rows']:,}건")
print((DEVSEL / "logs" / "validation_attempts.log").read_text(encoding="utf-8").splitlines()[0])

val_df = build_dataframe(DATA_ROOT / "val" / "label")
out_csv = HERE / "outputs" / "mission3_val.csv"
INFER_CMD = [sys.executable, "inference.py", "--audio_dir", str(DATA_ROOT / "val"), "--label_dir", str(DATA_ROOT / "val" / "label"),
             "--ckpt_path", str(BUNDLE / "ensemble.json"), "--output", str(out_csv)]
t0 = time.perf_counter()
result = subprocess.run(INFER_CMD, capture_output=True, text=True, encoding="utf-8", errors="replace")
elapsed = time.perf_counter() - t0
print("\\n".join(line for line in result.stdout.splitlines() if any(k in line for k in ["정밀도", "완료", "증상 0개", "구성"])))
print(f"전체 실행 시간(프로세스 시작~종료) {elapsed:.1f}초, 샘플당 {elapsed / len(val_df) * 1000:.1f} ms")
"""),
    code("""
import ast
pred = pd.read_csv(out_csv, encoding="utf-8-sig")
pred["call_id"] = pred["label file name"].str.replace(r"\\.json$", "", regex=True)
gold = val_df.set_index("call_id").loc[pred["call_id"]]
P = np.array([[s in ast.literal_eval(v) for s in TARGET_SYMPTOMS] for v in pred["symptom"]], dtype=int)
Y = gold[TARGET_SYMPTOMS].to_numpy()
tp, fp, fn = (P & Y).sum(0), (P & (1 - Y)).sum(0), ((1 - P) & Y).sum(0)
f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
first = sorted((DEVSEL).glob("validation_once_*.csv"))[0]
same = (pd.read_csv(first, encoding="utf-8-sig").sort_values("label file name").reset_index(drop=True)
        .equals(pred.drop(columns="call_id").sort_values("label file name").reset_index(drop=True)))
print(f"Validation macro F1@0.5 = {f1.mean():.4f}  (예측 {len(pred):,}건, 증상 0개 예측 {(P.sum(1) == 0).sum()}건)")
print("첫 확인과 예측이 같은가:", same)
pd.DataFrame({"precision": tp / np.maximum(tp + fp, 1), "recall": tp / np.maximum(tp + fn, 1), "F1@0.5": f1}, index=TARGET_SYMPTOMS).round(4)
"""),
    md("""
## 10. 주요 실험 결과

**dev 기준 선택 (Training 내부 dev 2,920건, macro F1@0.5)**

| 단계 | 비교 | dev F1 |
|---|---|---:|
| 손실 가중 | 가중치 없음 / **(neg/pos)^0.5** / neg/pos | 0.6189 / **0.6503** / 0.6141 |
| 레시피 (seed 2개 평균) | 원래 레시피 / **TAPT 20ep + LLRD 0.8** | 0.6505 / **0.6542** |
| TF-IDF 블렌드 | **w=0** / w=0.3 (C=0.15) | **0.6580** / 0.6550 |
| 모델 수 | 단일 평균 / **2-seed 앙상블** | 0.6542 / **0.6580** |

**최종 번들 Validation 확인 (1회): 0.6599**

참고 (선택에는 쓰지 않음): 이 절차 전에 Validation 을 보며 했던 탐색 기록은 저장소의 `reports/calibration_eval.md`, `reports/improvement_eval.md` 에 있다.
- 기준선 plain BCE 0.5967
- pos_weight^0.5 0.6496
- 다른 백본(KoELECTRA·KF-DeBERTa·RoBERTa-large)은 이득 없음

주최 측 답변 이후 그 값들은 제출 모델 선택에 쓰지 않았다.
"""),
    md("""
## 11. 계산 효율
"""),
    code("""
from safetensors import safe_open
member_params = []
for member in sorted(p for p in BUNDLE.iterdir() if p.is_dir()):
    with safe_open(str(member / "model.safetensors"), framework="pt") as f:
        member_params.append(sum(int(np.prod(f.get_slice(k).get_shape())) for k in f.keys()))
total = sum(member_params)
pd.DataFrame({
    "값": [f"{total:,} (KLUE-RoBERTa-base {member_params[0]:,} × {len(member_params)})",
          f"{total:,} (모든 멤버가 매 샘플에 쓰이는 dense 앙상블)",
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}, {cpu_name()}, Python {platform.python_version()}, torch {torch.__version__}",
          "16",
          f"{elapsed:.1f}초 / 샘플당 {elapsed / len(val_df) * 1000:.1f} ms (3,640건, fp16, 모델 로딩·CSV 저장 포함)"],
}, index=["Total 파라미터", "Active 파라미터", "학습·추론 환경", "Validation batch size", "Validation 전체 추론 시간"])
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
    parser.add_argument("--no-exec", action="store_true")
    args = parser.parse_args()
    nb = build()
    if not args.no_exec:
        from nbclient import NotebookClient

        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, NOTEBOOK)
    print(f"저장: {NOTEBOOK}")


if __name__ == "__main__":
    main()
