"""Mission 3 제출용 학습 기록 노트북(model_train.ipynb)을 만들고 실행한다.

학습(TAPT 약 2.5시간 + 시드 4개 × 약 20분)은 CLI 로 이미 끝냈다. 노트북은
  1) 원본 JSON 에서 학습 CSV 를 다시 만들어 실제 학습에 쓴 CSV 와 같은지 확인하고,
  2) 각 단계의 실행 명령과 그 단계가 남긴 로그(tapt_config.json, history.json, baseline_metrics.json)를 보여 주고,
  3) 제출 번들로 Validation 전체를 실제로 추론해 점수와 시간을 기록한다.
RUN_TRAINING=True 로 바꾸면 같은 명령으로 처음부터 다시 학습한다.

    python build_notebook.py            # 노트북 생성 + 실행 (학습 작업공간의 runs/ 필요)
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

**최종 모델**: KLUE-RoBERTa-base 에 Training 대화 본문으로 TAPT(MLM) 20 epoch → 층별 학습률 감쇠(LLRD 0.8, 최상위 lr 5e-5) + pos_weight=(neg/pos)^0.5 로 9개 증상 다중 라벨 분류 학습(시드 42~45) → 4개 확률 균등 평균 + Training 전용 TF-IDF LogisticRegression 을 전역 가중치 0.3 으로 섞음 → **모든 클래스 임계값 0.5** 로 판정.

**Validation macro F1@0.5 = 0.6593** (3,640건, 제출 경로 `inference.py` 로 재현).

규정 준수:
- 학습(역전파)에는 Training(서울) 데이터만 썼다. Validation 은 체크포인트·하이퍼파라미터 선택과 평가에만 썼다.
- 입력은 대화 본문(`utterances[].text`)뿐이다. 화자·시간·인적사항은 파싱 단계에서 버린다.
- 결정 임계값은 0.5 고정이다. 공개 사전학습 모델(klue/roberta-base)만 썼고 상용 API 는 쓰지 않았다.

이 노트북은 학습 작업공간(`runs/` 가 있는 폴더)에서 실행한 기록이다. 학습은 아래 명령들로 CLI 에서 수행했고, 각 셀은 그 명령이 남긴 로그를 읽어 출력한다. `RUN_TRAINING = True` 로 바꾸면 같은 명령으로 다시 학습한다.
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

RUN_TRAINING = False                    # True 면 아래 명령으로 다시 학습 (약 4시간)
RUNS = HERE / "runs"
SEEDS = [42, 43, 44, 45]
TAPT_DIR = RUNS / "tapt_klue_base_e20"
RUN_DIRS = {s: RUNS / f"klue_tapt_e20_llrd0.8_lr5e-5_posw0.5_seed{s}" for s in SEEDS}
TFIDF_DIR = RUNS / "tfidf_lr_c0.15"
BUNDLE = RUNS / "submit_tapt20_llrd08_4seed_tfidf"

def find_data_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        label_dir = parent / "data" / "train" / "label"
        if label_dir.is_dir() and next(label_dir.glob("*.json"), None) is not None:   # 빈 자리표시 폴더는 건너뛴다
            return parent / "data"
    raise FileNotFoundError("data/train/label 폴더를 찾지 못했습니다. DATA_ROOT 를 직접 지정하세요.")

def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            return subprocess.check_output(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                           text=True).strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()

DATA_ROOT = Path(os.environ.get("DCC_DATA_ROOT", "")) if os.environ.get("DCC_DATA_ROOT") else find_data_root(HERE)
print("Python", platform.python_version(), "| torch", torch.__version__, "| transformers", transformers.__version__, "| scikit-learn", sklearn.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음", "| CPU:", cpu_name(), "| OS:", platform.platform())
print("데이터:", DATA_ROOT)
"""),
    md("""
## 1. 데이터 준비 — 원본 JSON 에서 학습 CSV 만들기

`m3.labels.load_transcripts_dir` 가 대화 본문만 공백으로 이어 붙이고(발화 경계 모드 `space`), 9개 대상 증상만 라벨 벡터로 남긴다. 추론 경로(`m3.infer.read_texts`)도 같은 함수를 쓰므로 학습 입력과 추론 입력이 어긋나지 않는다.
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
val_df = build_dataframe(DATA_ROOT / "val" / "label")
print(f"Training {len(train_df):,}건, Validation {len(val_df):,}건 ({time.perf_counter() - t0:.1f}초)")

counts = pd.DataFrame({
    "Training 양성": train_df[TARGET_SYMPTOMS].sum(),
    "Training 비율": train_df[TARGET_SYMPTOMS].mean().round(3),
    "Validation 양성": val_df[TARGET_SYMPTOMS].sum(),
    "neg/pos": ((len(train_df) - train_df[TARGET_SYMPTOMS].sum()) / train_df[TARGET_SYMPTOMS].sum()).round(2),
})
print("통화당 평균 증상 수 (Training):", round(train_df[TARGET_SYMPTOMS].sum(axis=1).mean(), 2))
counts
"""),
    code("""
# 실제 학습에 쓴 CSV (run_config.json 에 기록된 경로) 와 본문·라벨이 같은지 확인한다.
used = json.loads((RUN_DIRS[42] / "run_config.json").read_text(encoding="utf-8"))
for name, rebuilt, csv_path in [("Training", train_df, used["train_csv"]), ("Validation", val_df, used["val_csv"])]:
    original = pd.read_csv(csv_path, encoding="utf-8-sig").set_index("call_id")
    rebuilt = rebuilt.set_index("call_id").loc[original.index]
    same_text = (rebuilt["text"] == original["text"]).mean()
    same_label = (rebuilt[TARGET_SYMPTOMS].values == original[TARGET_SYMPTOMS].values).all(axis=1).mean()
    print(f"{name}: 학습 CSV {len(original):,}행과 본문 일치 {same_text:.2%}, 라벨 일치 {same_label:.2%}  ({csv_path})")
"""),
    md("""
## 2. TAPT — Training 본문으로 MLM 추가 사전학습 (20 epoch)

원본 KLUE-RoBERTa 는 이 통화 전사문에서 MLM 손실이 약 5.1 로, 문어체 사전학습 코퍼스와 도메인 차이가 크다. Training CSV 의 text 만으로 15% 마스킹 MLM 을 20 epoch 이어 학습했다(lr 5e-5, warmup 6%, 선형 감쇠, 항상 마지막 epoch 저장). `--eval-csv` 는 Validation 256건의 MLM 손실을 no_grad 로 기록하는 진단용이며 가중치에 영향이 없다.
"""),
    code("""
TAPT_CMD = [sys.executable, "tapt_mlm.py", "--train-csv", used["train_csv"], "--eval-csv", used["val_csv"],
            "--output-dir", str(TAPT_DIR), "--local-files-only", "--amp", "--epochs", "20"]
print(" ".join(TAPT_CMD[1:]))
if RUN_TRAINING:
    subprocess.run(TAPT_CMD, check=True)

tapt = json.loads((TAPT_DIR / "tapt_config.json").read_text(encoding="utf-8"))
print(f"\\n학습 시간 {tapt['training_seconds'] / 60:.1f}분, 역전파 본문: Training {tapt['num_texts']:,}건 (라벨 미사용)")
print(f"eval_csv (no_grad 손실 기록 전용, 선택에 미사용): {Path(tapt['eval_csv']).name}, 표본 {tapt['eval_samples']}건")
pd.DataFrame(tapt["history"]).set_index("epoch").round(4)
"""),
    md("""
## 3. 분류 학습 — LLRD 0.8 + pos_weight^0.5, 시드 42~45

- 시작점: 위 TAPT 모델. 분류 헤드는 lr 5e-5, 인코더 층을 내려갈수록 0.8 배씩 감쇠(임베딩 약 2.7e-6).
- 손실: BCE, pos_weight = (Training 음성/양성)^0.5. 임계값 0.5 에서 확률이 제대로 판정되도록 하는 calibration 고려 손실 설계다(임계값은 옮기지 않는다).
- 3 epoch, 유효 배치 16(8×2), max_length 512, AMP. 체크포인트는 Validation macro F1@0.5 가 가장 높은 epoch(선택용으로만 사용).
"""),
    code("""
def train_cmd(seed: int):
    return [sys.executable, "train.py", "--train-csv", used["train_csv"], "--val-csv", used["val_csv"],
            "--model-name-or-path", str(TAPT_DIR), "--local-files-only", "--learning-rate", "5e-5", "--llrd-decay", "0.8",
            "--use-pos-weight", "--pos-weight-power", "0.5", "--checkpoint-metric", "val_macro_f1", "--amp",
            "--seed", str(seed), "--output-dir", str(RUN_DIRS[seed])]

print(" ".join(train_cmd(42)[1:]))
if RUN_TRAINING:
    for seed in SEEDS:
        subprocess.run(train_cmd(seed), check=True)

cfg = json.loads((RUN_DIRS[42] / "run_config.json").read_text(encoding="utf-8"))
print({k: cfg[k] for k in ["learning_rate", "llrd_decay", "pos_weight_power", "epochs", "train_batch_size",
                           "gradient_accumulation_steps", "max_length", "amp", "checkpoint_metric"]})
print("pos_weight:", {s: round(v["pos_weight"], 3) for s, v in cfg["train_pos_weight_statistics"].items()})
"""),
    code("""
rows = []
for seed in SEEDS:
    history = json.loads((RUN_DIRS[seed] / "history.json").read_text(encoding="utf-8"))["epochs"]
    metrics = json.loads((RUN_DIRS[seed] / "baseline_metrics.json").read_text(encoding="utf-8"))
    for e in history:
        rows.append({"seed": seed, "epoch": e["epoch"], "train_loss": e["train_loss"], "val_loss": e["val_loss"],
                     "val_macro_f1@0.5": e["val_macro_f1_at_0_5"],
                     "선택": "✓" if e["epoch"] == metrics["best_epoch"] else ""})
    print(f"seed {seed}: best epoch {metrics['best_epoch']}, Validation F1@0.5 {metrics['val_macro_f1']:.4f}, "
          f"학습 {metrics['training_seconds'] / 60:.1f}분, GPU 최대 {metrics['peak_cuda_memory_bytes'] / 2**30:.2f}GiB")
pd.DataFrame(rows).set_index(["seed", "epoch"]).round(4)
"""),
    md("""
## 4. TF-IDF 보조 멤버 — Training 전용

문자 2-4gram(char_wb) + 공백 토큰 1-2gram TF-IDF 위에 증상별 LogisticRegression(C=0.15, class_weight=balanced). Training CSV 로만 적합하고 Validation 은 transform·평가에만 쓴다. 트랜스포머와 다르게 틀리는 멤버라 블렌드 효과가 있다(C·가중치 0.3 은 Validation 으로 고른 전역 하이퍼파라미터이고 클래스별 값은 없다).
"""),
    code("""
TFIDF_CMD = [sys.executable, "train_tfidf_member.py", "--train-csv", used["train_csv"], "--output", str(TFIDF_DIR / "tfidf_lr.joblib")]
print(" ".join(TFIDF_CMD[1:]))
if RUN_TRAINING:
    subprocess.run(TFIDF_CMD, check=True)
meta = json.loads((TFIDF_DIR / "tfidf_lr.json").read_text(encoding="utf-8"))
{k: meta[k] for k in ["train_rows", "C", "min_df", "class_weight", "input", "sklearn_version", "note"]}
"""),
    md("""
## 5. 제출 번들

시드별 `best_model` 4개와 TF-IDF 멤버를 한 폴더에 모으고, `ensemble.json` 에 구성·가중치·정밀도를 적는다. 제출 폴더에서는 이 번들이 `ckpt/` 이고, `--ckpt_path ckpt/ensemble.json` 으로 추론한다. 멤버 폴더에는 tokenizer·config·가중치가 모두 있어 인터넷 없이 로드된다.
"""),
    code("""
manifest = json.loads((BUNDLE / "ensemble.json").read_text(encoding="utf-8"))
manifest["precision"] = "fp16"   # 제출본: CUDA 에서 fp16 autocast (GPU 가 없으면 자동으로 fp32)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
for member in sorted(p for p in BUNDLE.iterdir() if p.is_dir()):
    print(f"{member.name}/: " + ", ".join(sorted(f.name for f in member.iterdir())))
"""),
    md("""
## 6. Validation 평가 — 제출 경로(`inference.py`)로 전체 추론

원본 Validation JSON 3,640건을 제출과 같은 명령으로 추론하고, 결과 CSV 를 정답과 대조한다. 판정 임계값은 0.5 고정이다.
"""),
    code("""
out_csv = HERE / "outputs" / "mission3_val.csv"
INFER_CMD = [sys.executable, "inference.py", "--audio_dir", str(DATA_ROOT / "val"), "--label_dir", str(DATA_ROOT / "val" / "label"),
             "--ckpt_path", str(BUNDLE / "ensemble.json"), "--output", str(out_csv), "--precision", "fp16", "--batch_size", "16"]
t0 = time.perf_counter()
result = subprocess.run(INFER_CMD, capture_output=True, text=True, encoding="utf-8", errors="replace")
elapsed = time.perf_counter() - t0
print("\\n".join(line for line in result.stdout.splitlines() if any(k in line for k in ["정밀도", "완료", "증상 0개", "앙상블", "구성"])))
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
print(f"Validation macro F1@0.5 = {f1.mean():.4f}  (예측 {len(pred):,}건, 증상 0개 예측 {(P.sum(1) == 0).sum()}건)")
pd.DataFrame({"precision": tp / np.maximum(tp + fp, 1), "recall": tp / np.maximum(tp + fn, 1), "F1@0.5": f1}, index=TARGET_SYMPTOMS).round(4)
"""),
    md("""
## 7. 주요 실험 결과

모든 수치는 Validation macro F1@0.5(임계값 0.5 고정)다. 전체 근거는 `reports/calibration_eval.md`, `reports/improvement_eval.md`.

| 단계 | 설정 | F1@0.5 |
|---|---|---:|
| 기준선 | KLUE-RoBERTa-base, plain BCE | 0.5967 |
| 보정 손실 | pos_weight = neg/pos | 0.6189 |
| 보정 손실 | **pos_weight = (neg/pos)^0.5** (9개 클래스 모두 상승) | 0.6496 |
| 블렌드 | + Training 전용 TF-IDF (가중치 0.3) | 0.6536 |
| 4시드 앙상블 | 위 레시피 시드 42~45 + TF-IDF | 0.6546 |
| 도메인 적응 | TAPT 20ep (시드 42~45 평균, 단일 모델) | 0.6516 |
| 미세조정 | LLRD 0.8 (시드 42~45 평균, 단일 모델) | 0.6519 |
| **최종** | **TAPT 20ep + LLRD 0.8 단일 모델 (시드 평균)** | **0.6543** (원본 대비 +0.0068 [+0.0025, +0.0111]) |
| **최종 제출** | **위 4시드 + TF-IDF 번들** | **0.6593** |

효과가 없던 시도: 다른 백본(KoELECTRA 0.6398, KF-DeBERTa 0.6504, RoBERTa-large 0.6495, 모두 단일 시드), 발화 경계 `[SEP]` 입력(−0.0033), pos_weight power 미세 탐색(0.45~0.60 평탄), TF-IDF n-gram 변형·LightGBM, 학습률만 올리기. 남은 오류의 약 절반은 모든 모델이 똑같이 틀리는 행으로, 라벨이 주호소(처음 언급 증상) 위주로 달린 관례와 통화 밖 정보가 원인으로 보인다.
"""),
    md("""
## 8. 계산 효율
"""),
    code("""
from safetensors import safe_open
member_params = []
for seed_dir in sorted(p for p in BUNDLE.iterdir() if p.name.startswith("seed")):
    with safe_open(str(seed_dir / "model.safetensors"), framework="pt") as f:
        member_params.append(sum(int(np.prod(f.get_slice(k).get_shape())) for k in f.keys()))
from m3.tfidf_member import load_tfidf_member
tfidf_params = sum(m.coef_.size + m.intercept_.size for m in load_tfidf_member(BUNDLE / "tfidf" / "tfidf_lr.joblib").models)
total = sum(member_params) + tfidf_params
pd.DataFrame({
    "값": [f"{total:,} (멤버 {member_params[0]:,} × {len(member_params)} + TF-IDF LR {tfidf_params:,})",
          f"{total:,} (모든 멤버가 매 샘플에 쓰이는 dense 앙상블)",
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}, {cpu_name()}, Python {platform.python_version()}, torch {torch.__version__}",
          "16",
          f"{elapsed:.1f}초 / 샘플당 {elapsed / len(val_df) * 1000:.1f} ms (3,640건, fp16, 로딩·TF-IDF·CSV 저장 포함)"],
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
