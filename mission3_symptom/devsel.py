"""Training 내부 dev 로 Mission 3 하이퍼파라미터를 정하는 규칙과 번들 조립.

사전 등록한 절차와 규칙은 reports/dev_selection_protocol.md 에 있고, 순서대로 돌리는 스크립트는
run_dev_selection.sh 다. 이 파일의 power / final 단계는 Validation 을 읽지 않는다. 모든 run 의
run_config.json 이 dev 분할로 평가됐는지 확인하고, 아니면 멈춘다.

    python devsel.py power --train-csv D/train_split.csv --dev-csv D/dev_split.csv \\
        --run 0=runs/devsel/p0_s42 --run 0.5=runs/devsel/p0.5_s42 --run 1.0=runs/devsel/p1.0_s42 \\
        --out runs/devsel/decisions/power.json
    python devsel.py final --train-csv D/train_split.csv --dev-csv D/dev_split.csv \\
        --power-decision runs/devsel/decisions/power.json \\
        --arm original=runs/devsel/p0.5_s42,runs/devsel/orig_s43 \\
        --arm tapt_llrd=runs/devsel/tl_s42,runs/devsel/tl_s43 --out runs/devsel/decisions/final.json
    python devsel.py assemble --member runs/devsel/final_s42 ... [--tfidf X.joblib --tfidf-weight W] \\
        --out runs/devsel/mission3.pt                                      # 제출 번들 .pt 파일 하나
    python devsel.py score --pred pred.csv --label-dir <val label 폴더>      # 마지막 확인 1회에만
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_C_GRID = (0.05, 0.15, 0.5, 1.5)
DEFAULT_W_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
DEFAULT_C = 0.15
FULL_ENSEMBLE_SIZE = 4
EPS = 1e-12


# ---------------------------------------------------------------- 점수·선택 규칙 (순수 함수)

def macro_f1_at_half(labels, probs) -> float:
    """임계값 0.5 고정 macro F1."""
    import numpy as np

    from m3.metrics import eval_macro_f1

    return float(eval_macro_f1(np.asarray(labels, dtype=int), (np.asarray(probs) >= 0.5).astype(int)))


def blend(transformer_probs, tfidf_probs, weight: float):
    """제출 추론과 같은 식: (1 - w) x 트랜스포머 평균 + w x TF-IDF."""
    import numpy as np

    if tfidf_probs is None or weight == 0.0:
        return np.asarray(transformer_probs, dtype=np.float64)
    return (1.0 - weight) * np.asarray(transformer_probs, dtype=np.float64) + weight * np.asarray(tfidf_probs)


def argmax_first(scores: Mapping[str, float]) -> str:
    """점수가 가장 높은 키. 같으면 먼저 넣은 키 (규칙에 적은 우선순위)."""
    best_key, best_value = None, None
    for key, value in scores.items():
        if best_value is None or value > best_value + EPS:
            best_key, best_value = key, value
    if best_key is None:
        raise ValueError("후보가 없습니다")
    return best_key


def choose_epoch(mean_by_epoch: Mapping[int, float]) -> int:
    """시드 평균 dev F1 이 가장 높은 epoch. 같으면 뒤 epoch."""
    if not mean_by_epoch:
        raise ValueError("epoch 후보가 없습니다")
    best = max(mean_by_epoch.values())
    return int(max(epoch for epoch, value in mean_by_epoch.items() if value >= best - EPS))


def choose_tfidf(table: Sequence[Mapping[str, float]], default_c: float = DEFAULT_C) -> Mapping[str, float]:
    """(C, w) 표에서 dev F1 최고. 같으면 w 가 작은 쪽, 그다음 C 가 기본값(0.15)에 가까운 쪽."""
    if not table:
        raise ValueError("TF-IDF 후보 표가 비었습니다")
    best = max(row["f1"] for row in table)
    tied = [row for row in table if row["f1"] >= best - EPS]
    tied.sort(key=lambda row: (row["w"], abs(row["C"] - default_c), row["C"]))
    return tied[0]


def decide_members(seed_probs: Sequence, tfidf_probs, weight: float, labels) -> Dict[str, float]:
    """시드 평균 블렌드가 단일 시드 블렌드 평균보다 높으면 4개 앙상블, 아니면 1개."""
    import numpy as np

    singles = [macro_f1_at_half(labels, blend(p, tfidf_probs, weight)) for p in seed_probs]
    ensemble = macro_f1_at_half(labels, blend(np.mean(np.stack(seed_probs), axis=0), tfidf_probs, weight))
    single_mean = float(np.mean(singles))
    return {
        "single_scores": singles,
        "single_mean": single_mean,
        "ensemble_score": ensemble,
        "n_members": FULL_ENSEMBLE_SIZE if ensemble > single_mean + EPS else 1,
    }


# ---------------------------------------------------------------- run 읽기 + Validation 차단

def _same_file(a, b) -> bool:
    """두 경로가 같은 파일을 가리키는가 (상대·절대 경로 차이는 무시)."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def check_run_split(run_dir: Path, train_csv: Path, dev_csv: Path) -> Dict[str, object]:
    """run 이 Training 학습용 분할로 학습하고 dev 분할로 평가됐는지 확인한다. 아니면 멈춘다."""
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if not _same_file(config.get("val_csv", ""), dev_csv):
        raise ValueError(f"{run_dir}: 평가 CSV 가 dev 분할이 아닙니다 ({config.get('val_csv')}). Validation 으로 고를 수 없습니다")
    if not _same_file(config.get("train_csv", ""), train_csv):
        raise ValueError(f"{run_dir}: 학습 CSV 가 Training 학습용 분할이 아닙니다 ({config.get('train_csv')})")
    return config


def load_epoch_probs(run_dir: Path, dev_labels) -> Dict[int, object]:
    """epoch 별 dev 확률. 행 순서가 dev CSV 와 맞는지 history 의 F1 으로 다시 확인한다."""
    import numpy as np

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))["epochs"]
    probs = {}
    for row in history:
        epoch = int(row["epoch"])
        path = run_dir / f"val_probs_epoch{epoch}.npy"
        values = np.load(path)
        if values.shape != np.asarray(dev_labels).shape:
            raise ValueError(f"{path}: 형상 {values.shape} 이 dev 라벨 {np.asarray(dev_labels).shape} 와 다릅니다")
        recomputed = macro_f1_at_half(dev_labels, values)
        if abs(recomputed - float(row["val_macro_f1_at_0_5"])) > 1e-6:
            raise ValueError(f"{path}: 행 순서가 dev CSV 와 다릅니다 (F1 {recomputed:.6f} vs 기록 {row['val_macro_f1_at_0_5']:.6f})")
        probs[epoch] = values
    return probs


def load_split_labels(csv_path: Path):
    """분할 CSV 를 읽어 `(DataFrame, (n, 9) 정수 라벨)` 을 돌려준다. 행 순서는 CSV 그대로다."""
    from m3.dataset import labels_from_dataframe, load_symptom_csv

    frame = load_symptom_csv(csv_path)
    return frame, labels_from_dataframe(frame).astype(int)


# ---------------------------------------------------------------- 단계

def stage_power(args) -> Dict[str, object]:
    """pos_weight 지수 p 를 정한다 (run_dev_selection.sh 3 단계).

    `--run p=run폴더` 마다 run_config 가 학습용/dev 분할과 그 p 로 학습됐는지 확인하고, epoch 별 dev
    macro F1@0.5 중 최고값을 그 p 의 점수로 삼아 가장 높은 p 를 고른다 (같으면 먼저 적은 후보).
    """
    _, dev_labels = load_split_labels(Path(args.dev_csv))
    scores: Dict[str, float] = {}
    details = {}
    for spec in args.run:
        power, run = spec.split("=", 1)
        run_dir = Path(run)
        config = check_run_split(run_dir, Path(args.train_csv), Path(args.dev_csv))
        if abs(float(config.get("pos_weight_power", -1)) - float(power)) > 1e-9 or not config.get("use_pos_weight"):
            raise ValueError(f"{run_dir}: run_config 의 pos_weight 설정이 p={power} 와 다릅니다")
        epoch_probs = load_epoch_probs(run_dir, dev_labels)
        per_epoch = {epoch: macro_f1_at_half(dev_labels, p) for epoch, p in epoch_probs.items()}
        scores[power] = max(per_epoch.values())
        details[power] = {"run": str(run_dir), "dev_f1_by_epoch": per_epoch, "dev_f1_best": scores[power]}
    choice = argmax_first(scores)
    return {
        "stage": "power",
        "rule": "후보 p 마다 원래 레시피 seed 42 를 학습용 분할로 3 epoch 학습, epoch 별 dev macro F1@0.5 중 최고값이 가장 큰 p (같으면 먼저 적은 후보)",
        "candidates": list(scores),
        "details": details,
        "choice": choice,
    }


def stage_final(args) -> Dict[str, object]:
    """레시피·저장 epoch·TF-IDF (C, w)·모델 수를 차례로 정한다 (run_dev_selection.sh 6 단계).

    1. 레시피(`--arm 이름=run1,run2`)마다 seed 평균 dev F1 이 최고인 epoch 를 찾고, 그 값이 큰 레시피를 고른다.
    2. 고른 레시피의 그 epoch seed 평균 확률에 학습용 분할로 적합한 TF-IDF 를 섞어 (C, w) 격자에서 dev F1 최고를 고른다.
    3. seed 평균 블렌드가 단일 seed 블렌드 평균보다 높으면 4개 앙상블, 아니면 1개로 정한다.
    모든 run 은 학습용/dev 분할과 3 단계에서 정한 p 로 학습됐는지 먼저 확인한다. Validation 은 읽지 않는다.
    """
    import numpy as np

    from m3.tfidf_member import fit_tfidf_member

    train_csv, dev_csv = Path(args.train_csv), Path(args.dev_csv)
    train_frame, train_labels = load_split_labels(train_csv)
    dev_frame, dev_labels = load_split_labels(dev_csv)
    power = json.loads(Path(args.power_decision).read_text(encoding="utf-8"))["choice"]

    arms: Dict[str, Dict[str, object]] = {}
    for spec in args.arm:
        name, runs = spec.split("=", 1)
        seed_probs: List[Dict[int, object]] = []
        run_dirs = [Path(run) for run in runs.split(",")]
        for run_dir in run_dirs:
            config = check_run_split(run_dir, train_csv, dev_csv)
            if abs(float(config.get("pos_weight_power", -1)) - float(power)) > 1e-9:
                raise ValueError(f"{run_dir}: p={config.get('pos_weight_power')} 가 결정된 p={power} 와 다릅니다")
            seed_probs.append(load_epoch_probs(run_dir, dev_labels))
        epochs = sorted(set.intersection(*[set(p) for p in seed_probs]))
        mean_by_epoch = {
            epoch: float(np.mean([macro_f1_at_half(dev_labels, p[epoch]) for p in seed_probs])) for epoch in epochs
        }
        epoch = choose_epoch(mean_by_epoch)
        arms[name] = {
            "runs": [str(r) for r in run_dirs],
            "mean_dev_f1_by_epoch": mean_by_epoch,
            "epoch": epoch,
            "score": mean_by_epoch[epoch],
            "_probs": [p[epoch] for p in seed_probs],
        }

    recipe = argmax_first({name: arm["score"] for name, arm in arms.items()})
    chosen = arms[recipe]
    ensemble_probs = np.mean(np.stack(chosen["_probs"]), axis=0)

    table = []
    tfidf_probs_by_c = {}
    for c in args.c_grid:
        member = fit_tfidf_member(train_frame["text"].astype(str).tolist(), train_labels, C=c)
        tfidf_probs_by_c[c] = member.predict_proba(dev_frame["text"].astype(str).tolist())
        for w in args.w_grid:
            table.append({"C": c, "w": w, "f1": macro_f1_at_half(dev_labels, blend(ensemble_probs, tfidf_probs_by_c[c], w))})
    tfidf_choice = choose_tfidf(table)
    members = decide_members(chosen["_probs"], tfidf_probs_by_c[tfidf_choice["C"]], tfidf_choice["w"], dev_labels)

    for arm in arms.values():
        arm.pop("_probs")
    return {
        "stage": "final",
        "rules": {
            "recipe": "레시피마다 seed 2개 epoch 별 dev F1 평균 -> 최고 epoch 의 값이 큰 레시피 (같으면 먼저 적은 레시피)",
            "epoch": "선택된 레시피의 seed 평균 dev F1 이 최고인 epoch (같으면 뒤 epoch)",
            "tfidf": "선택된 레시피 seed 평균 확률에 학습용 분할로 적합한 TF-IDF 를 섞어 (C, w) 격자에서 dev F1 최고 (같으면 작은 w, C 는 0.15 에 가까운 쪽). w=0 이면 TF-IDF 를 쓰지 않음",
            "members": "seed 평균 블렌드 dev F1 이 단일 seed 블렌드 평균보다 높으면 4개 앙상블, 아니면 1개",
        },
        "power": power,
        "arms": arms,
        "tfidf_table": table,
        "members_check": members,
        "decision": {
            "recipe": recipe,
            "pos_weight_power": float(power),
            "epoch": int(chosen["epoch"]),
            "tfidf_C": float(tfidf_choice["C"]),
            "tfidf_weight": float(tfidf_choice["w"]),
            "n_members": int(members["n_members"]),
            "dev_f1_at_decision": float(tfidf_choice["f1"]),
        },
    }


def stage_assemble(args) -> Dict[str, object]:
    """최종 run 들의 best_model 과 (w > 0 이면) TF-IDF 멤버를 제출용 `.pt` 파일 하나로 묶는다 (m3/bundle.py)."""
    from m3.bundle import describe, load_bundle, pack_bundle

    out = Path(args.out)
    if out.suffix != ".pt":
        raise ValueError(f"--out 은 .pt 파일 경로여야 합니다: {out}")
    if out.exists():
        raise FileExistsError(f"이미 있는 번들 파일입니다: {out}")
    # 멤버는 run 폴더(안의 best_model)를 그대로 받는다. 없거나 가중치가 빠졌으면 pack_bundle 이 멈춘다.
    use_tfidf = bool(args.tfidf) and args.tfidf_weight > 0
    pack_bundle(
        args.member,
        out,
        precision=args.precision,
        tfidf_path=args.tfidf if use_tfidf else None,
        tfidf_weight=args.tfidf_weight if use_tfidf else 0.0,
        note=args.note,
    )
    # 다시 읽어 형식을 확인하고, 구성(멤버·파라미터 수·정밀도·TF-IDF 가중치)을 기록으로 남긴다.
    bundle = load_bundle(out)
    result = {
        "stage": "assemble",
        "bundle": str(out),
        "members": [member.name for member in bundle.members],
        "precision": bundle.precision,
        "tfidf_weight": bundle.tfidf_weight,
        "note": bundle.note,
        "summary": describe(bundle),
    }
    del bundle  # mmap 으로 연 가중치를 바로 놓는다
    return result


def stage_score(args) -> Dict[str, object]:
    """제출 CSV 를 라벨 JSON 정답과 대조한다. 결정이 끝난 모델의 최종 확인에만 쓴다."""
    import ast

    import numpy as np
    import pandas as pd

    from m3.config import TARGET_SYMPTOMS
    from make_csv import build_dataframe

    gold = build_dataframe(Path(args.label_dir)).set_index("call_id")
    pred = pd.read_csv(args.pred, encoding="utf-8-sig")
    ids = pred["label file name"].astype(str).str.replace(r"\.json$", "", regex=True)
    if not ids.is_unique or set(ids) != set(gold.index):
        raise ValueError("예측 CSV 의 파일 집합이 라벨 폴더와 다릅니다")
    predicted = np.array([[symptom in ast.literal_eval(value) for symptom in TARGET_SYMPTOMS] for value in pred["symptom"]], dtype=int)
    labels = gold.loc[ids, TARGET_SYMPTOMS].to_numpy().astype(int)
    from m3.metrics import eval_macro_f1

    macro, per_class = eval_macro_f1(labels, predicted, return_per_class=True)
    return {"stage": "score", "rows": int(len(ids)), "macro_f1": float(macro), "per_class_f1": per_class}


# ---------------------------------------------------------------- CLI

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mission 3 Training 내부 dev 하이퍼파라미터 선택")
    sub = p.add_subparsers(dest="stage", required=True)

    power = sub.add_parser("power")
    power.add_argument("--train-csv", required=True)
    power.add_argument("--dev-csv", required=True)
    power.add_argument("--run", action="append", required=True, help="p=run 디렉터리 (적은 순서가 동점 우선순위)")
    power.add_argument("--out", required=True)

    final = sub.add_parser("final")
    final.add_argument("--train-csv", required=True)
    final.add_argument("--dev-csv", required=True)
    final.add_argument("--power-decision", required=True)
    final.add_argument("--arm", action="append", required=True, help="이름=run1,run2 (적은 순서가 동점 우선순위)")
    final.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    final.add_argument("--w-grid", type=float, nargs="+", default=list(DEFAULT_W_GRID))
    final.add_argument("--out", required=True)

    assemble = sub.add_parser("assemble")
    assemble.add_argument("--member", action="append", required=True, help="최종 학습 run 디렉터리 (적은 순서대로 담긴다)")
    assemble.add_argument("--tfidf", help="TF-IDF 멤버 .joblib (--tfidf-weight 가 0 보다 클 때만 담는다)")
    assemble.add_argument("--tfidf-weight", type=float, default=0.0, help="9개 클래스 공통 블렌드 가중치 w")
    assemble.add_argument("--precision", choices=("fp32", "fp16"), default="fp16", help="추론 정밀도 (가중치는 fp32 로 저장)")
    assemble.add_argument("--note", default="Training 내부 dev 로 하이퍼파라미터를 정한 제출 번들 (reports/dev_selection_protocol.md)")
    assemble.add_argument("--out", required=True, help="저장할 번들 파일 경로 (.pt)")

    score = sub.add_parser("score")
    score.add_argument("--pred", required=True)
    score.add_argument("--label-dir", required=True)
    score.add_argument("--out")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = {"power": stage_power, "final": stage_final, "assemble": stage_assemble, "score": stage_score}[args.stage](args)
    text = json.dumps(result, ensure_ascii=False, indent=2, default=float)
    out = getattr(args, "out", None)
    if out and args.stage != "assemble":
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
