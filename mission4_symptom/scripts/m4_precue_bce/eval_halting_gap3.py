"""Gap 3 4-2 evaluation helper: nested 5-fold fit-only OOF & Dev 2,920 comparison (E026 v0 vs nonempty gate v1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/sunggyu/Desktop/이성규/DCC")
SRC = ROOT / "artifacts_private/e026_same_input_replay_20261116/source"
INP = Path("/Volumes/로컬 디스크/DCC/private_experiments/e026_same_input_replay_20261116/inputs")

sys.path.insert(0, str(SRC))
import score as sc  # noqa: E402
from m4 import SYMPTOM_CLASSES, halting as hl, simulate as sim  # noqa: E402
from m4.data import label_matrix, load_cache, read_ids  # noqa: E402


def merge_tables_average_or_concat(paths: list[Path]) -> sim.PrefixTable:
    tables = [sim.PrefixTable.load(p) for p in paths]
    first = tables[0]
    if all(t.call_ids == first.call_ids and np.array_equal(t.offsets, first.offsets) for t in tables[1:]):
        probs = np.mean([t.probs for t in tables], axis=0).astype(np.float32) if len(tables) > 1 else first.probs
        return sim.PrefixTable(first.call_ids, first.offsets, probs, first.labels)
    ids = [cid for t in tables for cid in t.call_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("Tables neither share identical call_ids nor are disjoint.")
    offsets = np.concatenate([[0], np.cumsum(np.concatenate([t.n_utt for t in tables]))]).astype(np.int64)
    return sim.PrefixTable(ids, offsets, np.concatenate([t.probs for t in tables]), np.concatenate([t.labels for t in tables]))


def train_and_eval_policy(fit_data: hl.HaltData, eval_data: hl.HaltData, nonempty_gate: bool, seed: int = 0) -> dict:
    betas = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0)
    delta, rows, consts = hl.delta_target(fit_data, ref_tau=0.5)
    oof = hl.crossfit_delta(fit_data, delta, rows, folds=5, seed=seed, threads=4)
    resid = delta[rows] - oof[rows]
    r2 = 1.0 - float((resid**2).sum()) / max(float(((delta[rows] - delta[rows].mean())**2).sum()), 1e-12)

    ones = np.ones((1, len(fit_data)))
    best_b, best_s = 0.0, -1.0
    for b in betas:
        cond = (fit_data.value + b * oof) >= 0.5
        if nonempty_gate:
            cond = cond & fit_data.pred.any(axis=1)
        s = float(hl._weighted_scores(hl._score_parts(fit_data, hl.first_halt(cond, fit_data, hl.MIN_STEP)), ones)[0])
        if s > best_s + 1e-12:
            best_s, best_b = s, b

    booster = hl.fit_gbm(fit_data.X[rows], delta[rows], threads=4)
    dh_ev = booster.predict(eval_data.X, num_threads=1)
    cond_ev = (eval_data.value + best_b * dh_ev) >= 0.5
    if nonempty_gate:
        cond_ev = cond_ev & eval_data.pred.any(axis=1)
    h_learned = hl.first_halt(cond_ev, eval_data, hl.MIN_STEP)
    h_base = hl.first_halt(eval_data.value >= 0.5, eval_data, hl.MIN_STEP)

    gold_map = {
        cid: sc.GoldSample(
            label_file=cid,
            gold=[SYMPTOM_CLASSES[k] for k in np.flatnonzero(eval_data.labels[i])],
            n_utt=int(eval_data.n_utt[i]),
            texts=[],
            speakers=[],
        )
        for i, cid in enumerate(eval_data.call_ids)
    }
    pred_42 = {}
    for i, cid in enumerate(eval_data.call_ids):
        row = int(eval_data.offsets[i] + h_learned[i] - 1)
        pred_42[cid] = (int(h_learned[i]), [SYMPTOM_CLASSES[k] for k in np.flatnonzero(eval_data.pred[row])])
    sc.set_quiet(True)
    off42 = sc.score_mission_4_2(pred_42, gold_map)
    if off42["warnings"]:
        raise RuntimeError(f"score.py 4-2 warnings: {off42['warnings']}")
    learned_metrics = hl.score_halts(eval_data, h_learned)
    assert abs(float(off42["score"]) - float(learned_metrics["score"])) < 1e-12

    return {
        "beta": best_b,
        "crossfit_r2": round(r2, 4),
        "fit_selected_score": round(best_s, 6),
        "eval_base": hl.score_halts(eval_data, h_base),
        "eval_learned": learned_metrics,
        "official_score_py_4_2": float(off42["score"]),
        "eval_diff_vs_base": hl.paired_bootstrap(eval_data, h_learned, h_base, n_boot=10000, seed=2026),
        "empty_at_halt_base": int((~eval_data.pred[eval_data.offsets[:-1] + h_base - 1].any(axis=1)).sum()),
        "empty_at_halt_learned": int((~eval_data.pred[eval_data.offsets[:-1] + h_learned - 1].any(axis=1)).sum()),
        "_halts": h_learned,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-oof", nargs="+", required=True)
    ap.add_argument("--eval-dev", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    fit_tbl = merge_tables_average_or_concat([Path(p) for p in args.fit_oof])
    ev_tbl = merge_tables_average_or_concat([Path(p) for p in args.eval_dev])
    if set(fit_tbl.call_ids) & set(ev_tbl.call_ids):
        raise ValueError("Overlap between fit OOF and eval dev calls!")

    calls = {c.call_id: c for c in load_cache(INP / "train.jsonl", set(fit_tbl.call_ids) | set(ev_tbl.call_ids))}
    fit_calls = [calls[cid] for cid in fit_tbl.call_ids]
    ev_calls = [calls[cid] for cid in ev_tbl.call_ids]

    # Strictly compute fit-only shift from fit_calls labels
    fit_lab = label_matrix(fit_calls)
    pos = fit_lab.sum(axis=0)
    fit_shift = np.log(np.sqrt((len(fit_calls) - pos) / np.maximum(pos, 1)))

    fit_hd = hl.HaltData.build(fit_tbl, fit_calls, fit_shift)
    ev_hd = hl.HaltData.build(ev_tbl, ev_calls, fit_shift)

    v0 = train_and_eval_policy(fit_hd, ev_hd, nonempty_gate=False)
    v1 = train_and_eval_policy(fit_hd, ev_hd, nonempty_gate=True)
    h0 = v0.pop("_halts")
    h1 = v1.pop("_halts")
    res = {
        "n_fit": len(fit_hd),
        "n_eval": len(ev_hd),
        "fit_only_shift": [float(x) for x in fit_shift],
        "v0_e026_fit_only": v0,
        "v1_nonempty_fit_only": v1,
        "v1_vs_v0_diff": hl.paired_bootstrap(ev_hd, h1, h0, n_boot=10000, seed=2026),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
