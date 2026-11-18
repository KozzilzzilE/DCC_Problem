"""Complete Gap 3 fit-only OOF policy training & Dev 2,920 evaluation (PROTOCOL R0 / C0 / C128 / fit-selected beta + v1_nonempty)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/sunggyu/Desktop/이성규/DCC")
SRC = ROOT / "artifacts_private/e026_same_input_replay_20261116/source"
INP = Path("/Volumes/로컬 디스크/DCC/private_experiments/e026_same_input_replay_20261116/inputs")
OOF_NPZ = Path("/Volumes/로컬 디스크/DCC/private_experiments/e002_oof_20261117/fit_oof_merged.npz")
SJ42 = Path("/Volumes/로컬 디스크/DCC/private_experiments/m4_softjaccard_v1_20261116/s42/light/dev_prefix.npz")
SJ43 = Path("/Volumes/로컬 디스크/DCC/private_experiments/m4_softjaccard_v1_20261116/s43/light/dev_prefix.npz")
JM42 = Path("/Volumes/로컬 디스크/DCC/private_experiments/m4_jacc_macrof1_v1_20261116/s42/light/dev_prefix.npz")
JM43 = Path("/Volumes/로컬 디스크/DCC/private_experiments/m4_jacc_macrof1_v1_20261116/s43/light/dev_prefix.npz")
OUT_PRIVATE = ROOT / "artifacts_private/runs/e026_fitonly_policy_v1_20261117"
OUT_EXT = Path("/Volumes/로컬 디스크/DCC/private_experiments/e026_fitonly_policy_v1_20261117")

sys.path.insert(0, str(SRC))
import score as sc  # noqa: E402
from m4 import SYMPTOM_CLASSES, halting as hl, simulate as sim  # noqa: E402
from m4.data import Call, label_matrix, load_cache  # noqa: E402


def merge_eval(paths: list[Path]) -> sim.PrefixTable:
    tables = [sim.PrefixTable.load(p) for p in paths]
    first = tables[0]
    for t in tables[1:]:
        assert t.call_ids == first.call_ids and np.array_equal(t.offsets, first.offsets)
    probs = np.mean([t.probs for t in tables], axis=0).astype(np.float32) if len(tables) > 1 else first.probs
    return sim.PrefixTable(first.call_ids, first.offsets, probs, first.labels)


def score_official_42(eval_hd: hl.HaltData, halts: np.ndarray) -> tuple[dict, list]:
    gold_map = {
        cid: sc.GoldSample(
            label_file=cid,
            gold=[SYMPTOM_CLASSES[k] for k in np.flatnonzero(eval_hd.labels[i])],
            n_utt=int(eval_hd.n_utt[i]), texts=[], speakers=[],
        )
        for i, cid in enumerate(eval_hd.call_ids)
    }
    pred_42 = {}
    rows = []
    for i, cid in enumerate(eval_hd.call_ids):
        h = int(halts[i])
        row = int(eval_hd.offsets[i] + h - 1)
        syms = [SYMPTOM_CLASSES[k] for k in np.flatnonzero(eval_hd.pred[row])]
        pred_42[cid] = (h, syms)
        rows.append((cid, h, syms, gold_map[cid].gold))
    sc.set_quiet(True)
    off = sc.score_mission_4_2(pred_42, gold_map)
    if off["warnings"]:
        raise RuntimeError(f"score.py warnings: {off['warnings']}")
    return off, rows


def test_causality(pol: hl.HaltPolicy, calls: list[Call], tbl: sim.PrefixTable) -> dict:
    # Verify prefix features and halting decisions are strictly invariant to future utterances/probabilities
    rng = np.random.default_rng(20261117)
    checked = 0
    for idx in range(min(200, len(calls))):
        c = calls[idx]
        if c.n_utt < 4:
            continue
        k = int(c.n_utt // 2)
        p_full = tbl.call(idx)
        c_trunc = Call(call_id=c.call_id, texts=c.texts[:k], speakers=c.speakers[:k],
                       starts=c.starts[:k], ends=c.ends[:k], labels=c.labels)
        t_full = sim.PrefixTable([c.call_id], np.array([0, c.n_utt], dtype=np.int64), p_full, tbl.labels[idx:idx+1])
        t_trunc = sim.PrefixTable([c.call_id], np.array([0, k], dtype=np.int64), p_full[:k], tbl.labels[idx:idx+1])
        hd_full = hl.HaltData.build(t_full, [c], pol.logit_shift)
        hd_trunc = hl.HaltData.build(t_trunc, [c_trunc], pol.logit_shift)
        assert np.allclose(hd_full.X[:k], hd_trunc.X, atol=1e-12)
        assert np.allclose(hd_full.value[:k], hd_trunc.value, atol=1e-12)
        h_trunc_sig = ((hd_trunc.value + pol.beta * pol.delta_hat(hd_trunc.X)) >= pol.tau)
        h_full_sig = ((hd_full.value[:k] + pol.beta * pol.delta_hat(hd_full.X[:k])) >= pol.tau)
        assert np.array_equal(h_trunc_sig, h_full_sig)
        # Also test HaltFeatureState online vs batch parity
        state = hl.HaltFeatureState(pol.logit_shift)
        online_halts = []
        for s in range(k):
            feats = state.update(c.texts[s], c.speakers[s], c.starts[s], c.ends[s], p_full[s])
            assert np.max(np.abs(feats - hd_trunc.X[s])) < 1e-12
            online_halts.append(pol.should_halt(state, feats))
        assert np.array_equal(np.array([bool(x) for x in online_halts])[hl.MIN_STEP-1:], h_trunc_sig[hl.MIN_STEP-1:])
        checked += 1
    return {"checked_calls": checked, "future_invariance_ok": True, "online_batch_parity_ok": True}


def main() -> int:
    t0 = time.time()
    OUT_PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT_EXT.mkdir(parents=True, exist_ok=True)

    fit_tbl = sim.PrefixTable.load(OOF_NPZ)
    ev_e002 = sim.PrefixTable.load(INP / "e002_dev_prefix.npz")
    assert not (set(fit_tbl.call_ids) & set(ev_e002.call_ids))

    all_calls = {c.call_id: c for c in load_cache(INP / "train.jsonl", set(fit_tbl.call_ids) | set(ev_e002.call_ids))}
    fit_calls = [all_calls[cid] for cid in fit_tbl.call_ids]
    ev_calls = [all_calls[cid] for cid in ev_e002.call_ids]

    # Strictly fit-only shift (26,280 calls only)
    fit_lab = label_matrix(fit_calls)
    pos = fit_lab.sum(axis=0)
    fit_shift = np.log(np.sqrt((len(fit_calls) - pos) / np.maximum(pos, 1)))

    fit_hd = hl.HaltData.build(fit_tbl, fit_calls, fit_shift)
    delta, rows, consts = hl.delta_target(fit_hd, ref_tau=0.5)
    oof_delta_hat = hl.crossfit_delta(fit_hd, delta, rows, folds=5, seed=0, threads=4)
    resid = delta[rows] - oof_delta_hat[rows]
    r2 = 1.0 - float((resid**2).sum()) / max(float(((delta[rows] - delta[rows].mean())**2).sum()), 1e-12)

    betas = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0)
    ones = np.ones((1, len(fit_hd)))
    grid_v0, grid_v1 = [], []
    best_v0, best_v1 = (-1.0, 0.0), (-1.0, 0.0)
    for b in betas:
        raw_cond = (fit_hd.value + b * oof_delta_hat) >= 0.5
        s0 = float(hl._weighted_scores(hl._score_parts(fit_hd, hl.first_halt(raw_cond, fit_hd, hl.MIN_STEP)), ones)[0])
        s1 = float(hl._weighted_scores(hl._score_parts(fit_hd, hl.first_halt(raw_cond & fit_hd.pred.any(axis=1), fit_hd, hl.MIN_STEP)), ones)[0])
        grid_v0.append({"beta": b, "score": round(s0, 6)})
        grid_v1.append({"beta": b, "score": round(s1, 6)})
        if s0 > best_v0[0] + 1e-12: best_v0 = (s0, b)
        if s1 > best_v1[0] + 1e-12: best_v1 = (s1, b)

    booster = hl.fit_gbm(fit_hd.X[rows], delta[rows], threads=4)
    model_str = booster.model_to_string()
    pol_c128 = hl.HaltPolicy(
        model_str=model_str, beta=128.0, tau=0.5, logit_shift=[float(x) for x in fit_shift],
        n_fit_calls=len(fit_hd), info={"target": consts, "crossfit_r2": r2, "grid_v0": grid_v0, "grid_v1": grid_v1},
        _booster=booster,
    )
    pol_c128.save(OUT_PRIVATE / "policy_fitonly_c128.json")
    pol_c128.save(OUT_EXT / "policy_fitonly_c128.json")

    causality = test_causality(pol_c128, ev_calls, ev_e002)

    targets = {
        "E002": [INP / "e002_dev_prefix.npz"],
        "ENS3": [INP / "e001_dev_prefix.npz", INP / "e002_dev_prefix.npz", INP / "e007_dev_prefix.npz"],
        "ENS_rep_jm42": [INP / "e001_dev_prefix.npz", JM42, INP / "e007_dev_prefix.npz"],
        "ENS4_e1_e7_jm42_jm43": [INP / "e001_dev_prefix.npz", INP / "e007_dev_prefix.npz", JM42, JM43],
        "ENS_rep_sj42": [INP / "e001_dev_prefix.npz", SJ42, INP / "e007_dev_prefix.npz"],
        "ENS4_e1_e7_sj42_sj43": [INP / "e001_dev_prefix.npz", INP / "e007_dev_prefix.npz", SJ42, SJ43],
    }

    summary = {
        "n_fit": len(fit_hd),
        "n_eval": len(ev_e002),
        "oof_source": str(OOF_NPZ),
        "fit_only_shift": [float(x) for x in fit_shift],
        "crossfit_r2": round(r2, 4),
        "fit_grid_v0": grid_v0,
        "fit_grid_v1_nonempty": grid_v1,
        "fit_best_beta_v0": best_v0[1],
        "fit_best_beta_v1": best_v1[1],
        "causality_and_online_parity": causality,
        "bootstrap_seed": 20261116,
        "n_boot": 10000,
        "targets": {},
    }

    for name, paths in targets.items():
        ev_tbl = merge_eval(paths)
        ev_hd = hl.HaltData.build(ev_tbl, ev_calls, fit_shift)
        dh = booster.predict(ev_hd.X, num_threads=1)
        h_c0 = hl.first_halt(ev_hd.value >= 0.5, ev_hd, hl.MIN_STEP)
        h_c0_ne = hl.first_halt((ev_hd.value >= 0.5) & ev_hd.pred.any(axis=1), ev_hd, hl.MIN_STEP)

        res_t = {
            "C0_beta0": {**hl.score_halts(ev_hd, h_c0), "empty_at_halt": int((~ev_hd.pred[ev_hd.offsets[:-1] + h_c0 - 1].any(axis=1)).sum())},
            "C0_nonempty": {**hl.score_halts(ev_hd, h_c0_ne), "empty_at_halt": int((~ev_hd.pred[ev_hd.offsets[:-1] + h_c0_ne - 1].any(axis=1)).sum())},
        }
        policies = [
            ("C128_v0", 128.0, False),
            ("C128_v1_nonempty", 128.0, True),
            (f"fitbest_v0_b{int(best_v0[1])}", best_v0[1], False),
            (f"fitbest_v1_nonempty_b{int(best_v1[1])}", best_v1[1], True),
        ]
        halts_map = {}
        for pname, beta, ne in policies:
            cond = (ev_hd.value + beta * dh) >= 0.5
            if ne:
                cond = cond & ev_hd.pred.any(axis=1)
            h = hl.first_halt(cond, ev_hd, hl.MIN_STEP)
            halts_map[pname] = h
            sc_m = hl.score_halts(ev_hd, h)
            off42, call_rows = score_official_42(ev_hd, h)
            assert abs(float(off42["score"]) - float(sc_m["score"])) < 1e-12
            res_t[pname] = {
                **sc_m,
                "official_score_py_4_2": float(off42["score"]),
                "empty_at_halt": int((~ev_hd.pred[ev_hd.offsets[:-1] + h - 1].any(axis=1)).sum()),
                "diff_vs_C0": hl.paired_bootstrap(ev_hd, h, h_c0, n_boot=10000, seed=20261116),
                "per_class_f1": off42["per_class_f1"],
            }
            if name == "ENS3" and pname in ("C128_v0", "C128_v1_nonempty"):
                # Save per-call halting decisions cleanly for private audit
                (OUT_EXT / f"{name}_{pname}_calls.json").write_text(
                    json.dumps([{"call_id": cid, "halt": h_i, "pred": p_i, "gold": g_i} for cid, h_i, p_i, g_i in call_rows], ensure_ascii=False),
                    encoding="utf-8",
                )
        res_t["C128_v1_vs_v0_diff"] = hl.paired_bootstrap(ev_hd, halts_map["C128_v1_nonempty"], halts_map["C128_v0"], n_boot=10000, seed=20261116)
        summary["targets"][name] = res_t

        print(f"\n=== {name} ===")
        print(f"  C0 (beta=0):          score={res_t['C0_beta0']['score']:.6f} f1={res_t['C0_beta0']['macro_f1']:.4f} e={res_t['C0_beta0']['earliness']:.4f} empty={res_t['C0_beta0']['empty_at_halt']}")
        for pname in [p[0] for p in policies]:
            r = res_t[pname]
            d = r["diff_vs_C0"]
            print(f"  {pname:23s}: score={r['score']:.6f} f1={r['macro_f1']:.4f} e={r['earliness']:.4f} empty={r['empty_at_halt']:2d} | diff={d['diff']:+.6f} CI=[{d['lo']:+.6f}, {d['hi']:+.6f}] p={d['p_gt0']:.4f}")
        dv = res_t["C128_v1_vs_v0_diff"]
        print(f"  C128 v1 vs v0:        diff={dv['diff']:+.6f} CI=[{dv['lo']:+.6f}, {dv['hi']:+.6f}] p={dv['p_gt0']:.4f}")

    summary["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT_PRIVATE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_EXT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDONE in {summary['elapsed_sec']}s -> {OUT_PRIVATE / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
