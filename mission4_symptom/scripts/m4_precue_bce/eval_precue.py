"""Evaluate trained pre-cue prefix run(s) against E002 / E007 / ENS3 on Dev 2,920 (raw p >= 0.5 + 4-2)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/sunggyu/Desktop/이성규/DCC")
SRC = ROOT / "artifacts_private/e026_same_input_replay_20261116/source"
INP = Path("/Volumes/로컬 디스크/DCC/private_experiments/e026_same_input_replay_20261116/inputs")
STEP2_TOOLS = ROOT / "tools/step2_error_analysis"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(STEP2_TOOLS))

import score as sc  # noqa: E402
from m4 import SYMPTOM_CLASSES, decision as dec, halting as hl, simulate as sim  # noqa: E402
from m4.data import label_matrix, load_cache, read_ids  # noqa: E402
import run_step2_part1 as st2  # noqa: E402


def paired_41(cand: np.ndarray, ref: np.ndarray, n_boot: int = 5000, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    N = len(cand)
    diff = cand - ref
    idx = rng.integers(0, N, size=(n_boot, N))
    boots = diff[idx].mean(axis=1)
    return {
        "diff": round(float(diff.mean()), 6),
        "ci95": [round(float(np.percentile(boots, 2.5)), 6), round(float(np.percentile(boots, 97.5)), 6)],
        "p_gt0": round(float((boots > 0).mean()), 4),
    }


def score_table_41(table: sim.PrefixTable, calls, gold_map) -> tuple[dict, np.ndarray]:
    pred = dec.threshold_rule(table.probs.copy(), force_top1=False)
    gold = np.repeat(table.labels.astype(bool), table.n_utt, axis=0)
    ms = sim.moment_scores(pred, gold)
    per_call = np.add.reduceat(ms, table.offsets[:-1]) / table.n_utt

    pred_dict = {}
    for i, c in enumerate(calls):
        a, b = int(table.offsets[i]), int(table.offsets[i + 1])
        pred_dict[c.call_id] = {
            s: [SYMPTOM_CLASSES[k] for k in np.flatnonzero(pred[a + s - 1])]
            for s in range(1, b - a + 1)
        }
    sc.set_quiet(True)
    off = sc.score_mission_4_1(pred_dict, gold_map)
    if off["warnings"]:
        raise RuntimeError(f"Official scorer warnings: {off['warnings']}")
    assert abs(float(off["score"]) - float(per_call.mean())) < 1e-9

    lcnt = table.labels.sum(axis=1)
    return {
        "m4_1": round(float(per_call.mean()), 6),
        "single_4_1": round(float(per_call[lcnt == 1].mean()), 6),
        "multi_4_1": round(float(per_call[lcnt >= 2].mean()), 6),
        "empty_rate": round(float((~pred.any(axis=1)).mean()), 6),
        "mean_pred_labels": round(float(pred.sum(axis=1).mean()), 4),
    }, per_call


def eval_42_with_shift(probs: np.ndarray, t1: sim.PrefixTable, calls, fit_shift: np.ndarray, policy_29k_path: Path):
    tbl = sim.PrefixTable(t1.call_ids, t1.offsets, probs.astype(np.float32), t1.labels)
    hdata_fit = hl.HaltData.build(tbl, calls, fit_shift)
    h_beta0 = hl.first_halt(hdata_fit.value >= 0.5, hdata_fit, hl.MIN_STEP)
    sc_beta0 = hl.score_halts(hdata_fit, h_beta0)

    h_nonempty = hl.first_halt((hdata_fit.value >= 0.5) & hdata_fit.pred.any(axis=1), hdata_fit, hl.MIN_STEP)
    sc_nonempty = hl.score_halts(hdata_fit, h_nonempty)

    # Also apply frozen E026 policy using its exact stored logit_shift
    pol = hl.HaltPolicy.load(policy_29k_path)
    hdata_pol = hl.HaltData.build(tbl, calls, pol.logit_shift)
    sc_pol_b0 = hl.score_halts(hdata_pol, hl.first_halt(hdata_pol.value >= 0.5, hdata_pol, hl.MIN_STEP))
    sc_pol_128 = hl.score_halts(hdata_pol, pol.halts(hdata_pol))
    return {
        "fit_shift_beta0": sc_beta0,
        "fit_shift_beta0_nonempty": sc_nonempty,
        "frozen_e026_policy_shift_beta0": sc_pol_b0,
        "frozen_e026_policy_beta128": sc_pol_128,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-npz", required=True)
    ap.add_argument("--cand-name", default="cand_s42")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    dev_ids = read_ids(INP / "dev_ids.txt")
    calls = load_cache(INP / "train.jsonl", dev_ids)
    gold_map = {
        c.call_id: sc.GoldSample(c.call_id, [SYMPTOM_CLASSES[k] for k, v in enumerate(c.labels) if v], len(c.texts), c.texts, c.speakers)
        for c in calls
    }

    t1 = sim.PrefixTable.load(INP / "e001_dev_prefix.npz")
    t2 = sim.PrefixTable.load(INP / "e002_dev_prefix.npz")
    t7 = sim.PrefixTable.load(INP / "e007_dev_prefix.npz")
    tc = sim.PrefixTable.load(args.cand_npz)
    assert t1.call_ids == t2.call_ids == t7.call_ids == tc.call_ids
    assert np.array_equal(t1.offsets, tc.offsets)
    assert np.array_equal(t1.labels, tc.labels)

    ens3_p = ((t1.probs + t2.probs + t7.probs) / 3.0).astype(np.float32)
    ens3 = sim.PrefixTable(t1.call_ids, t1.offsets, ens3_p, t1.labels)

    ens_rep_p = ((t1.probs + tc.probs + t7.probs) / 3.0).astype(np.float32)
    ens_rep = sim.PrefixTable(t1.call_ids, t1.offsets, ens_rep_p, t1.labels)

    fit_ids = read_ids(INP / "fit_ids.txt")
    fit_calls = load_cache(INP / "train.jsonl", fit_ids)
    fit_lab = label_matrix(fit_calls)
    pos_fit = fit_lab.sum(axis=0)
    fit_shift = np.log(np.sqrt((len(fit_calls) - pos_fit) / np.maximum(pos_fit, 1)))
    pol_path = ROOT / "artifacts_private/e026_same_input_replay_20261116/source/runs/E026_learned_halt05/wide/policy.json"

    s_e2, pc_e2 = score_table_41(t2, calls, gold_map)
    s_cand, pc_cand = score_table_41(tc, calls, gold_map)
    s_ens3, pc_ens3 = score_table_41(ens3, calls, gold_map)
    s_rep, pc_rep = score_table_41(ens_rep, calls, gold_map)

    step_arr = np.concatenate([np.arange(1, int(n) + 1) for n in t1.n_utt])
    rel_arr = np.concatenate([np.arange(1, int(n) + 1, dtype=np.float64) / float(n) for n in t1.n_utt])
    is_last = np.zeros(len(step_arr), dtype=bool)
    is_last[t1.offsets[1:] - 1] = True

    decomp_e2 = st2.analyze_model_dev2920(t2, step_arr, rel_arr, is_last)
    decomp_cand = st2.analyze_model_dev2920(tc, step_arr, rel_arr, is_last)
    decomp_ens3 = st2.analyze_model_dev2920(ens3, step_arr, rel_arr, is_last)
    decomp_rep = st2.analyze_model_dev2920(ens_rep, step_arr, rel_arr, is_last)

    lex_qa = st2.analyze_lexicon_and_qa(calls, {
        "E002": t2,
        args.cand_name: tc,
        "ENS3": ens3,
        f"ENS_rep_{args.cand_name}": ens_rep,
    })

    out = {
        "scores_4_1": {
            "E002": s_e2,
            args.cand_name: s_cand,
            "ENS3": s_ens3,
            f"ENS_rep_{args.cand_name}": s_rep,
        },
        "paired_bootstrap": {
            f"{args.cand_name}_vs_E002": paired_41(pc_cand, pc_e2),
            f"ENS_rep_{args.cand_name}_vs_ENS3": paired_41(pc_rep, pc_ens3),
        },
        "step2_decomp": {
            "E002": decomp_e2,
            args.cand_name: decomp_cand,
            "ENS3": decomp_ens3,
            f"ENS_rep_{args.cand_name}": decomp_rep,
        },
        "lexicon_and_qa": lex_qa,
        "scores_4_2": {
            "E002": eval_42_with_shift(t2.probs, t1, calls, fit_shift, pol_path),
            args.cand_name: eval_42_with_shift(tc.probs, t1, calls, fit_shift, pol_path),
            "ENS3": eval_42_with_shift(ens3.probs, t1, calls, fit_shift, pol_path),
            f"ENS_rep_{args.cand_name}": eval_42_with_shift(ens_rep.probs, t1, calls, fit_shift, pol_path),
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scores_4_1": out["scores_4_1"], "paired_bootstrap": out["paired_bootstrap"], "scores_4_2": out["scores_4_2"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
