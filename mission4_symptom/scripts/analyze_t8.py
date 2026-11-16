"""저장된 T8/Baseline 산출물만 읽는 사후 진단. 모델 로드·추론·정책 최적화 없음."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from m4 import SYMPTOM_CLASSES as CLASSES
from m4 import decision as dec, halting as hl, simulate as sim
from m4.data import load_cache, read_ids

RUNS = ROOT / "outputs/mission4_prefix"
BASE = RUNS / "localcontrol"
T8 = RUNS / "t8_asymmetric_loss"
SEQ = [RUNS / "localsequential", T8 / "sequential"]


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dump(p, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding="utf-8")


def confusion(p, y):
    p, y = np.asarray(p, bool), np.asarray(y, bool)
    counts = np.stack([(p & y).sum(0), (p & ~y).sum(0), (~p & y).sum(0), (~p & ~y).sum(0)], axis=-1)
    result = {}
    for c, (tp, fp, fn, tn) in zip(CLASSES, counts):
        result[c] = dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                         precision=float(tp / (tp + fp)) if tp + fp else 0.,
                         recall=float(tp / (tp + fn)) if tp + fn else 0.,
                         f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.)
    return {"per_class": result, "totals": dict(zip(["tp", "fp", "fn", "tn"], counts.sum(0).tolist())),
            "macro_f1": sim.macro_f1(p, y)["macro_f1"],
            "mean_labels": float(p.sum(1).mean()), "empty_rate": float((~p.any(1)).mean()),
            "label_count_distribution": {str(k): int((p.sum(1) == k).sum()) for k in range(10)}}


def stop_metrics(p, y, h, n):
    result = confusion(p, y)
    e = float(sim.earliness(h, n).mean())
    return {**result, "earliness": e, "score": result["macro_f1"] * e,
            "halt_mean": float(h.mean()), "halt_median": float(np.median(h)),
            "at_final_rate": float((h == n).mean())}


def distribution(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "quantiles": dict(zip(["0", "1", "5", "25", "50", "75", "95", "99", "100"],
                                  np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100]).tolist()))}


def probability_metrics(p, y):
    bins = []
    for low in np.arange(0, 1, .1):
        mask = (p >= low) & (p < low + .1 if low < .9 else p <= 1)
        bins.append({"low": round(float(low), 1), "count": int(mask.sum()),
                     "confidence": float(p[mask].mean()) if mask.any() else None,
                     "frequency": float(y[mask].mean()) if mask.any() else None})
    ece = sum(b["count"] * abs(b["confidence"] - b["frequency"]) for b in bins if b["count"]) / p.size
    return {"all": distribution(p), "positive": distribution(p[y]), "negative": distribution(p[~y]),
            "brier": float(np.mean((p.astype(float) - y) ** 2)), "ece_10": ece,
            "auroc_macro": float(roc_auc_score(y, p, average="macro")),
            "auprc_macro": float(average_precision_score(y, p, average="macro")),
            "per_class": {c: {"positive_mean": float(p[y[:, k], k].mean()),
                               "negative_mean": float(p[~y[:, k], k].mean()),
                               "brier": float(np.mean((p[:, k].astype(float) - y[:, k]) ** 2)),
                               "quantiles": distribution(p[:, k])["quantiles"],
                               "auroc": float(roc_auc_score(y[:, k], p[:, k])),
                               "auprc": float(average_precision_score(y[:, k], p[:, k]))}
                          for k, c in enumerate(CLASSES)}, "reliability_bins": bins}


def transitions(p0, p1, y):
    add, remove = ~p0 & p1, p0 & ~p1
    return {"new_positive": int(add.sum()), "new_positive_true": int((add & y).sum()),
            "new_positive_false": int((add & ~y).sum()), "removed_positive": int(remove.sum()),
            "removed_positive_true": int((remove & y).sum()), "removed_positive_false": int((remove & ~y).sum())}


def bootstrap(y, pred0, pred1, hb, ht, n, off, moment, rel, seed=20261116, repeats=2000):
    rng = np.random.default_rng(seed)
    N = len(n)
    # 동일한 통화 재표집을 모든 지표에 공통으로 사용한다.
    W = rng.multinomial(N, np.full(N, 1 / N), size=repeats).astype(float)
    def f1(p):
        tp, fp, fn = [W @ v.astype(float) for v in (p & y, p & ~y, ~p & y)]
        den = 2 * tp + fp + fn
        valid = den > 0
        return ((2 * tp / np.maximum(den, 1)) * valid).sum(1) / valid.sum(1)
    ib, it = off[:-1] + hb - 1, off[:-1] + ht - 1
    fA, fD, fB = f1(pred0[ib]), f1(pred1[it]), f1(pred1[ib])
    e0, e1 = sim.earliness(hb, n), sim.earliness(ht, n)
    E0, E1 = W @ e0 / N, W @ e1 / N
    d = {"4-1 Score": W @ (moment[1] - moment[0]) / N,
         "4-2 Score": fD * E1 - fA * E0, "Macro-F1": fD - fA,
         "Earliness": E1 - E0, "동일 Baseline 중단 Macro-F1": fB - fA}
    for name, low, high in [("초반 Score", 0, 1/3), ("중반 Score", 1/3, 2/3), ("후반 Score", 2/3, 1)]:
        mask = (rel > low) & (rel <= high)
        den = np.add.reduceat(mask.astype(float), off[:-1])
        num = np.add.reduceat((moment[3] - moment[2]) * mask, off[:-1])
        d[name] = (W @ num) / (W @ den)
    return {name: {"delta": float(values.mean()), "lo": float(np.percentile(values, 2.5)),
                   "hi": float(np.percentile(values, 97.5)),
                   "includes_zero": bool(np.percentile(values, 2.5) <= 0 <= np.percentile(values, 97.5)),
                   "bootstrap_positive_fraction": float((values > 0).mean())} for name, values in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=RUNS / "t8_analysis")
    args = ap.parse_args()
    out = args.out.resolve()
    if out.parent != RUNS.resolve() or out in [BASE.resolve(), T8.resolve(), *[x.resolve() for x in SEQ]]:
        raise ValueError("분석 전용 새 출력 폴더를 지정하십시오")
    out.mkdir(exist_ok=False)
    files = [p for d in [BASE, T8, SEQ[0]] for p in d.rglob("*") if p.is_file() and p.suffix in [".json", ".npz", ".log", ".txt"]]
    manifest = {str(p.relative_to(ROOT)): digest(p) for p in files}
    tables = [sim.PrefixTable.load(d / "dev_prefix.npz") for d in (BASE, T8)]
    a, b = tables
    assert a.call_ids == b.call_ids and np.array_equal(a.offsets, b.offsets) and np.array_equal(a.labels, b.labels)
    off, n, y = a.offsets, a.n_utt, a.labels.astype(bool)
    N = len(n)
    assert N == 2920 and off[-1] == 87814
    models = [read(d / "metrics.json") for d in (BASE, T8)]
    cfgs = [read(d / "training/run_config.json") for d in (BASE, T8)]
    initials = [read(d / "initial.json") for d in (BASE, T8)]
    seqrecords = [read(d / "sequential_timing.json") for d in SEQ]
    hs = []
    for d in (BASE, T8):
        z = np.load(d / "halts.npz")
        assert z["call_ids"].tolist() == a.call_ids
        hs.append(z["halts"])
    hb, ht = hs
    fixed = [sim.PrefixTable.load(HERE / "runs" / rel / "dev_prefix.npz") for rel in ["E001_m3_baseline", "E007_prefix_uniform_s43"]]
    for f in fixed:
        assert f.call_ids == a.call_ids and np.array_equal(f.offsets, off) and np.array_equal(f.labels, y)
    probs = [np.mean([fixed[0].probs, t.probs, fixed[1].probs], axis=0).astype(np.float32) for t in tables]
    assert all(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() for p in probs)
    pred = [p >= .5 for p in probs]
    yy = np.repeat(y, n, axis=0)
    step = np.concatenate([np.arange(1, k + 1) for k in n])
    rel = step / np.repeat(n, n)
    callidx = np.repeat(np.arange(N), n)
    ms = [sim.moment_scores(p, yy) for p in pred]
    callscore = [np.add.reduceat(s, off[:-1]) / n for s in ms]
    batch = [{"4-1": {"score": float(s.mean()), **confusion(p, yy)},
              "4-2": stop_metrics(p[off[:-1] + h - 1], y, h, n)} for s, p, h in zip(callscore, pred, hs)]
    for i in range(2):
        assert abs(batch[i]["4-1"]["score"] - models[i]["mission_4_1"]["score"]) < 1e-12
        assert abs(batch[i]["4-2"]["score"] - models[i]["mission_4_2"]["score"]) < 1e-12
    print("저장된 Batch 지표 재계산 일치", flush=True)
    segments = {}
    masks = {"초반": rel <= 1/3, "중반": (rel > 1/3) & (rel <= 2/3), "후반": rel > 2/3,
             "1~4": step <= 4, "5~10": (step >= 5) & (step <= 10),
             "11~20": (step >= 11) & (step <= 20), "21 이상": step >= 21}
    for k in [1, 2, 3, 4, 5, 8, 10, 15, 20, 30]:
        masks[f"발화 {k}"] = step == k
    finalmask = np.zeros(len(step), bool)
    finalmask[off[1:] - 1] = True
    masks["마지막"] = finalmask
    for name, mask in masks.items():
        segments[name] = {"moments": int(mask.sum()), "calls": int(np.unique(callidx[mask]).size),
            "baseline": {"score": float(ms[0][mask].mean()), **confusion(pred[0][mask], yy[mask])},
            "t8": {"score": float(ms[1][mask].mean()), **confusion(pred[1][mask], yy[mask])},
            "transitions": transitions(pred[0][mask], pred[1][mask], yy[mask])}
    lengths = {}
    for label, lo, hi in [("1~10", 1, 10), ("11~20", 11, 20), ("21~40", 21, 40), ("41 이상", 41, 100000)]:
        selected = (n >= lo) & (n <= hi)
        lengths[label] = {"calls": int(selected.sum()), "baseline": float(callscore[0][selected].mean()),
                          "t8": float(callscore[1][selected].mean()),
                          "halt_delta_mean": float((ht - hb)[selected].mean())}
    cf = {}
    for key, p, h in [("A", pred[0], hb), ("B", pred[1], hb), ("C", pred[0], ht), ("D", pred[1], ht)]:
        cf[key] = stop_metrics(p[off[:-1] + h - 1], y, h, n)
    sequential, seqcf, seq_checks = [], {}, []
    spreds, shalts = [], []
    for i, folder in enumerate(SEQ):
        z1, z2 = np.load(folder / "sequential_4-1.npz"), np.load(folder / "sequential_4-2.npz")
        assert z1["call_ids"].tolist() == z2["call_ids"].tolist() == a.call_ids
        p, h, actual = z1["pred"], z2["halts"], z2["pred"]
        score = float((np.add.reduceat(sim.moment_scores(p, yy), off[:-1]) / n).mean())
        rec = {"4-1": {"score": score, **confusion(p, yy)}, "4-2": stop_metrics(actual, y, h, n)}
        assert abs(score - seqrecords[i]["4-1"]["score"]) < 1e-12
        assert abs(rec["4-2"]["score"] - seqrecords[i]["4-2"]["score"]) < 1e-12
        sequential.append(rec)
        seq_checks.append({"batch_vs_sequential_41_label_disagreements": int((p != pred[i]).sum()),
                           "batch_vs_sequential_halt_disagreements": int((h != hs[i]).sum()),
                           "seq41_at_seq42_halt_vs_actual42_label_disagreements": int((p[off[:-1] + h - 1] != actual).sum())})
        spreds.append(p); shalts.append(h)
    for key, p, h in [("A", spreds[0], shalts[0]), ("B", spreds[1], shalts[0]),
                       ("C", spreds[0], shalts[1]), ("D", spreds[1], shalts[1])]:
        seqcf[key] = stop_metrics(p[off[:-1] + h - 1], y, h, n)
    print("Sequential 저장 배열 검증 및 반사실적 조합 완료", flush=True)
    policy = models[0]["provenance"]["policy"]
    shift = policy["logit_shift"]
    probability = {}
    for name, pp in [("E002 단독", [t.probs for t in tables]), ("최종 앙상블", probs)]:
        probability[name] = {"raw": [probability_metrics(p, yy) for p in pp],
                             "calibrated": [probability_metrics(dec.calibrate(p, shift), yy) for p in pp],
                             "transitions": transitions(pp[0] >= .5, pp[1] >= .5, yy),
                             "classification": [{"score": float((np.add.reduceat(sim.moment_scores(p >= .5, yy), off[:-1]) / n).mean()),
                                                  **confusion(p >= .5, yy)} for p in pp]}
    print("단독/앙상블 확률·보정·순위 지표 계산 완료", flush=True)
    boot = bootstrap(y, *pred, hb, ht, n, off, (*callscore, *ms), rel)
    point = {"4-1 Score": batch[1]["4-1"]["score"] - batch[0]["4-1"]["score"],
             "4-2 Score": cf["D"]["score"] - cf["A"]["score"],
             "Macro-F1": cf["D"]["macro_f1"] - cf["A"]["macro_f1"],
             "Earliness": cf["D"]["earliness"] - cf["A"]["earliness"],
             "동일 Baseline 중단 Macro-F1": cf["B"]["macro_f1"] - cf["A"]["macro_f1"],
             **{name + " Score": segments[name]["t8"]["score"] - segments[name]["baseline"]["score"] for name in ["초반", "중반", "후반"]}}
    for key in boot:
        boot[key]["bootstrap_mean_delta"] = boot[key].pop("delta")
        boot[key]["delta"] = point[key]
    print("통화 단위 Paired Bootstrap 2,000회 완료", flush=True)
    # E026의 회귀 모델을 로드/호출하지 않고, 저장 확률에 대한 수식과 텍스트 특징만 계산한다.
    calls_by_id = {c.call_id: c for c in load_cache(HERE / "runs/cache/train.jsonl", read_ids(HERE / "runs/cache/dev_ids.txt"))}
    calls = [calls_by_id[i] for i in a.call_ids]
    assert np.array_equal([c.labels for c in calls], y)
    signals = [hl.moment_signals(p, shift) for p in probs]
    features = [hl.batch_features(p, off, calls, shift, sig) for p, sig in zip(probs, signals)]
    dh = ht - hb
    fsummary, correlations = {}, []
    common = off[:-1] + np.minimum(5, n) - 1
    common = common[n >= 5]
    for k, name in enumerate(hl.FEATURE_NAMES):
        v0, v1 = features[0][:, k], features[1][:, k]
        fsummary[name] = {"all_moments": [distribution(v0), distribution(v1)],
                          "same_step_5": [distribution(v0[common]), distribution(v1[common])],
                          "at_baseline_halt": [distribution(v0[off[:-1] + hb - 1]), distribution(v1[off[:-1] + hb - 1])],
                          "at_own_halt": [distribution(v0[off[:-1] + hb - 1]), distribution(v1[off[:-1] + ht - 1])]}
        change = (v1 - v0)[off[:-1] + hb - 1]
        if np.std(change) > 0:
            rho = float(spearmanr(change, dh).statistic)
            correlations.append({"feature": name, "rho_delta_feature_vs_delay": rho,
                                 "change_mean": float(change.mean())})
    conv_same = np.array_equal(features[0][:, len(hl.PROB_NAMES):], features[1][:, len(hl.PROB_NAMES):])
    group = {}
    for label, select in [("빨라짐", dh < 0), ("동일", dh == 0), ("늦어짐", dh > 0), ("10발화 이상 지연", dh >= 10)]:
        ix = off[:-1][select] + hb[select] - 1
        group[label] = {"calls": int(select.sum()), "length_mean": float(n[select].mean()),
                        "halt_delta_mean": float(dh[select].mean()),
                        "feature_change_at_baseline_halt": {name: float((features[1][ix, k] - features[0][ix, k]).mean()) for k, name in enumerate(hl.PROB_NAMES)}}
    halt = {"earlier": int((dh < 0).sum()), "same": int((dh == 0).sum()), "later": int((dh > 0).sum()),
            "delta": distribution(dh), "baseline": distribution(hb), "t8": distribution(ht),
            "at_final": [int((h == n).sum()) for h in hs],
            "at_final_rate": [float((h == n).mean()) for h in hs],
            "delay_bins": {name: int(mask.sum()) for name, mask in [("음수", dh < 0), ("0", dh == 0),
                 ("1~4", (dh > 0) & (dh <= 4)), ("5~9", (dh >= 5) & (dh <= 9)),
                 ("10~19", (dh >= 10) & (dh <= 19)), ("20 이상", dh >= 20)]},
            "groups": group, "conv_features_identical_at_matched_moments": conv_same,
            "feature_correlations": sorted(correlations, key=lambda v: abs(v["rho_delta_feature_vs_delay"]), reverse=True),
            "policy_signal_status": "미확보. E026 회귀 출력과 최종 signal은 저장되지 않아 새 모델 추론 없이 복원 불가",
            "never_condition_status": "미확보. halt==통화길이는 마지막 시점 조건 충족과 조건 미충족의 fallback을 구분하지 못함"}
    print("새 정책 추론 없이 E026 입력 Feature 분석 완료", flush=True)
    # 원문 사례는 증상 관련 발화만 최소 발췌하고 개인정보 표기를 추가 제거한다.
    change = callscore[1] - callscore[0]
    earlydiff = np.zeros(N)
    for i in range(N):
        mask = rel[off[i]:off[i+1]] <= 1/3
        earlydiff[i] = (ms[1][off[i]:off[i+1]][mask] - ms[0][off[i]:off[i+1]][mask]).mean() if mask.any() else 0
    categories = {"초반 개선": np.argsort(-earlydiff), "후반 오탐 실패": np.argsort(change), "중단 지연": np.argsort(-dh)}
    cases = []
    for category, order in categories.items():
        chosen = 0
        for i in order:
            if n[i] < 10:
                continue
            utterances = []
            for j, text in enumerate(calls[i].texts, 1):
                if re.search(r"메스|울렁|구토|토했|토하|머리|두통|아프|아파|어지|호흡|숨이|열이|기운|힘이", text):
                    clean = re.sub(r"\d{2,}", "[숫자 생략]", text)
                    if re.search(r"아파트|번지|도로|연락처|전화번호", clean):
                        continue
                    utterances.append({"step": j, "text": clean[:160]})
            if not utterances:
                continue
            ixb, ixt = int(off[i] + hb[i] - 1), int(off[i] + ht[i] - 1)
            cases.append({"category": category, "case": hashlib.sha256(a.call_ids[i].encode()).hexdigest()[:10],
                          "n": int(n[i]), "gold": [c for c, flag in zip(CLASSES, y[i]) if flag],
                          "score41": [float(s[i]) for s in callscore], "early_delta": float(earlydiff[i]),
                          "halts": [int(hb[i]), int(ht[i])], "snippets": utterances[:6],
                          "at_baseline_halt": [[c for c, flag in zip(CLASSES, p[ixb]) if flag] for p in pred],
                          "at_t8_halt": [[c for c, flag in zip(CLASSES, p[ixt]) if flag] for p in pred],
                          "at_final": [[c for c, flag in zip(CLASSES, p[off[i+1]-1]) if flag] for p in pred],
                          "value_at_baseline_halt": [float(sig[1][ixb]) for sig in signals]})
            chosen += 1
            if chosen == 2:
                break
    # 직접 해시 대조와 저장 설정·CPU 재생 근거를 별도로 보존한다.
    controls = {"recipe_equal": initials[0]["recipe"] == initials[1]["recipe"],
                "training_config_equal_except_loss_output_records": {k: cfgs[0].get(k) == cfgs[1].get(k) for k in
                ["calls", "ids", "backbone", "fmt", "truncation_side", "max_length", "epochs", "samples_per_call", "include_full",
                 "batch_size", "grad_accum", "lr", "llrd", "pos_weight_power", "seed", "count_classes", "count_weight", "max_steps", "grad_ckpt", "n_calls", "total_steps"]},
                "initial_receipts_equal": {k: initials[0][k] == initials[1][k] for k in
                    ["initial_weights_sha256", "initial_tokenizer_sha256", "fit_ids_sha256", "dev_ids_sha256"]},
                "policy_equal": models[0]["provenance"]["policy"] == models[1]["provenance"]["policy"],
                "fixed_input_receipts_equal": {k: models[0]["provenance"][k] == models[1]["provenance"][k] for k in
                    ["policy_sha256", "cache_sha256", "fit_ids_sha256", "dev_ids_sha256"]},
                "loss_config": cfgs[1]["loss_config"], "optimizer_updates": cfgs[1]["optimizer_updates"], "amp_skips": cfgs[1]["amp_skips"],
                "baseline_optimizer_status": "별도 실측 기록 없음",
                "training_seconds": [c["training_seconds"] for c in cfgs], "history": [c["history"] for c in cfgs]}
    controls["baseline_optimizer_status"] = "실제 업데이트/AMP Skip의 별도 기록 없음. total_steps=9855는 예정/시도 횟수 근거"
    cpu = read(RUNS / "t8_cpu_verified/cpu_verification.json")
    controls["sampling_cpu_replay_match"] = all(s["ordered_items_sha256"] == e["ordered_items_sha256"] for s, e in zip(cfgs[1]["sampling_history"], cpu["epochs"]))
    controls["runtime_tokens_labels_vs_cpu_replay"] = all(all(h[k] == e["t8"][k] for k in ["nonpadding_tokens", "padding_tokens", "label_positive_exposures"])
                                                          and h["consumed_samples"] == e["t8"]["samples"] for h, e in zip(cfgs[1]["history"], cpu["epochs"]))
    controls["cpu_epochs"] = cpu["epochs"]
    controls["current_source_vs_t8_initial"] = {str(Path(p).relative_to(ROOT)): digest(p) == expected for p, expected in initials[1]["source_sha256"].items()}
    controls["current_initial_files"] = {"weights": digest(Path(cfgs[1]["backbone"]) / "model.safetensors") == initials[1]["initial_weights_sha256"],
                                           "tokenizer": digest(Path(cfgs[1]["backbone"]) / "tokenizer.json") == initials[1]["initial_tokenizer_sha256"]}
    checkpoints = []
    for i, run in enumerate((BASE, T8)):
        actual = digest(run / "training/best_model/model.safetensors")
        hashes = seqrecords[i]["model_hashes"]
        checkpoints.append({"current_sha256": actual, "batch_sha256": models[i]["provenance"]["e002_checkpoint_sha256"],
                            "sequential_record_contains_same_sha256": actual in hashes.values(),
                            "batch_matches_current": actual == models[i]["provenance"]["e002_checkpoint_sha256"]})
    controls["checkpoints"] = checkpoints
    controls["fixed_models_recorded_hashes_equal"] = list(seqrecords[0]["model_hashes"].values())[:2] == list(seqrecords[1]["model_hashes"].values())[:2]
    controls["fixed_tables_recorded_hashes_equal"] = list(models[0]["provenance"]["tables"].values())[::2] == list(models[1]["provenance"]["tables"].values())[::2]
    prior = {}
    for name, directory in [("T3", "t3_mixed_training"), ("T5", "t5_early_sampling")]:
        p = RUNS / directory / "runone/metrics.json"
        if p.is_file():
            rec = read(p)
            prior[name] = {"4-1": rec["mission_4_1"], "4-2": rec["mission_4_2"]}
    result = {"scope": "저장된 Dev 결과의 사후 진단. 새로운 분류기/정책 추론, 학습, 최적화 없음",
              "n_calls": N, "n_moments": int(off[-1]), "classes": CLASSES, "controls": controls,
              "batch": batch, "sequential": sequential, "sequential_checks": seq_checks,
              "counterfactual": cf, "sequential_counterfactual": seqcf, "segments": segments,
              "length_groups": lengths, "probabilities": probability, "bootstrap": boot,
              "bootstrap_settings": {"repeats": 2000, "seed": 20261116, "unit": "통화", "ci": "percentile 95%"},
              "halting": halt, "features": fsummary, "cases": cases, "prior": prior,
              "timing": {"batch_extraction": [read(d / "dev_prefix.meta.json") for d in (BASE, T8)], "sequential": seqrecords}}
    assert all(digest(ROOT / p) == expected for p, expected in manifest.items())
    dump(out / "analysis.json", result)
    dump(out / "input_manifest.json", {"before": manifest, "after_equal": True, "analysis_source_sha256": digest(__file__)})
    np.savez_compressed(out / "diagnostic_arrays.npz", baseline_halts=hb, t8_halts=ht,
                        baseline_call_scores=callscore[0], t8_call_scores=callscore[1], delay=dh)
    print(f"원본 산출물 해시 유지 확인. 분석 결과: {out}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with threadpool_limits(limits=1):
        main()
