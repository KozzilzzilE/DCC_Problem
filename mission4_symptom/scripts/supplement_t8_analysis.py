"""T8 사후 분석 보충: 초기 개선의 예측 확대 분해와 확률 순위 비교. 모델 추론 없음."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from analyze_t8 import BASE, T8, HERE, ROOT, RUNS, CLASSES, dump, read, digest
from m4 import simulate as sim
from m4.data import load_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=RUNS / "t8_analysis")
    out = ap.parse_args().out.resolve()
    if out.parent != RUNS.resolve() or out in [BASE.resolve(), T8.resolve(), (RUNS / "localsequential").resolve()]:
        raise ValueError("분석 결과 폴더만 지정하십시오")
    r = read(out / "analysis.json")
    direct = {}
    metrics = [read(d / "metrics.json") for d in (BASE, T8)]
    for rel, key in [("runs/cache/train.jsonl", "cache_sha256"), ("runs/cache/fit_ids.txt", "fit_ids_sha256"),
                     ("runs/cache/dev_ids.txt", "dev_ids_sha256"), ("runs/E026_learned_halt05/wide/policy.json", "policy_sha256")]:
        direct[rel] = digest(HERE / rel) == metrics[0]["provenance"][key] == metrics[1]["provenance"][key]
    for model, folder in [("Baseline", RUNS / "localsequential"), ("T8", T8 / "sequential")]:
        hashes = read(folder / "sequential_timing.json")["model_hashes"]
        for path, expected in list(hashes.items())[:2]:
            direct[model + "_" + str(Path(path).relative_to(ROOT))] = digest(Path(path) / "model.safetensors") == expected
    for rel in ["E001_m3_baseline", "E007_prefix_uniform_s43"]:
        path = HERE / "runs" / rel / "dev_prefix.npz"
        direct[rel + "_probability"] = all(digest(path) == record["provenance"]["tables"][str(path)] for record in metrics)
    configs = [read(d / "training/best_model/config.json") for d in (BASE, T8)]
    direct["model_config_equal"] = configs[0] == configs[1]
    tables = [sim.PrefixTable.load(d / "dev_prefix.npz") for d in (BASE, T8)]
    a = tables[0]
    n, off, y = a.n_utt, a.offsets, a.labels.astype(bool)
    yy = np.repeat(y, n, axis=0)
    fixed = [sim.PrefixTable.load(HERE / "runs" / name / "dev_prefix.npz").probs for name in ["E001_m3_baseline", "E007_prefix_uniform_s43"]]
    ensemble = [np.mean([fixed[0], t.probs, fixed[1]], axis=0).astype(np.float32) for t in tables]
    pred = [p >= .5 for p in ensemble]
    scores = [sim.moment_scores(p, yy) for p in pred]
    step = np.concatenate([np.arange(1, k+1) for k in n])
    rel = step / np.repeat(n, n)
    stages = {"초반": rel <= 1/3, "중반": (rel > 1/3) & (rel <= 2/3), "후반": rel > 2/3,
              "1~4": step <= 4, "마지막": np.isin(np.arange(len(step)), off[1:]-1)}
    data = {}
    for name, mask in stages.items():
        data[name] = {"ranking": {model: [
            {"auroc_macro": float(roc_auc_score(yy[mask], p[mask], average="macro")),
             "auprc_macro": float(average_precision_score(yy[mask], p[mask], average="macro")),
             "top1_in_gold": float(yy[mask][np.arange(mask.sum()), p[mask].argmax(1)].mean())}
            for p in pp] for model, pp in [("E002 단독", [t.probs for t in tables]), ("앙상블", ensemble)]}}
        den = int(mask.sum())
        empty0, empty1 = [~p.any(1) for p in pred]
        groups = {"빈 예측에서 비어 있지 않은 예측": empty0 & ~empty1,
                  "기존 비어 있지 않은 예측": ~empty0,
                  "양쪽 모두 빈 예측": empty0 & empty1}
        data[name]["score_change_groups"] = {label: {"moments": int((select & mask).sum()),
            "contribution_to_segment_delta": float(((scores[1]-scores[0]) * (select & mask)).sum() / den),
            "baseline_score": float(scores[0][select & mask].mean()) if (select & mask).any() else None,
            "t8_score": float(scores[1][select & mask].mean()) if (select & mask).any() else None}
            for label, select in groups.items()}
        if name != "마지막":
            count = np.add.reduceat(mask.astype(int), off[:-1])
            valid = count > 0
            data[name]["call_balanced_score"] = [float((np.add.reduceat(s * mask, off[:-1])[valid] / count[valid]).mean()) for s in scores]
    cf = r["counterfactual"]
    FA, FD, EA, ED = [cf[c][k] for c, k in [("A", "macro_f1"), ("D", "macro_f1"), ("A", "earliness"), ("D", "earliness")]]
    decomposition = {"classification_at_baseline_stop": cf["B"]["score"]-cf["A"]["score"],
                     "later_context_and_earliness_with_t8": cf["D"]["score"]-cf["B"]["score"],
                     "baseline_later_stop": cf["C"]["score"]-cf["A"]["score"],
                     "classification_at_t8_stop": cf["D"]["score"]-cf["C"]["score"],
                     "shapley_model": ((cf["B"]["score"]-cf["A"]["score"])+(cf["D"]["score"]-cf["C"]["score"]))/2,
                     "shapley_stop": ((cf["C"]["score"]-cf["A"]["score"])+(cf["D"]["score"]-cf["B"]["score"]))/2,
                     "deltaF_term": EA*(FD-FA), "deltaE_term": FA*(ED-EA), "interaction": (FD-FA)*(ED-EA)}
    hb, ht = [np.load(d / "halts.npz")["halts"] for d in [BASE, T8]]
    dh = ht-hb
    length_corr = float(spearmanr(n, dh).statistic)
    matched_count = [p[off[:-1]+hb-1].sum(1) for p in pred]
    groups = {"예측 증상 수 증가": matched_count[1] > matched_count[0], "예측 증상 수 동일": matched_count[1] == matched_count[0]}
    delays = {name: {"calls": int(mask.sum()), "delay_mean": float(dh[mask].mean()), "length_mean": float(n[mask].mean())} for name, mask in groups.items()}
    by_id = {hashlib.sha256(c.call_id.encode()).hexdigest()[:10]: c for c in load_cache(HERE / "runs/cache/train.jsonl")}
    ids = {hashlib.sha256(cid.encode()).hexdigest()[:10]: i for i, cid in enumerate(a.call_ids)}
    cases = []
    for case in r["cases"]:
        call, i = by_id[case["case"]], ids[case["case"]]
        early = []
        for j, text in enumerate(call.texts[:12], 1):
            if re.search(r"아파트|번지|도로|연락처|전화번호|주소|[동로길]\s*\d", text):
                continue
            if not re.search(r"구급차|배가|구토|메스|울렁|머리|두통|아프|아파|어지|호흡|숨|열이|기운|힘이|입니다|말씀", text):
                continue
            clean = re.sub(r"\d{2,}", "[숫자 생략]", text)
            early.append({"step": j, "text": clean[:120],
                          "predictions": [[c for c, yes in zip(CLASSES, p[off[i]+j-1]) if yes] for p in pred]})
        cases.append({"case": case["case"], "early_utterances": early})
    patterns = [r"열이|열나", r"구토|토했|토하", r"머리|두통", r"배가|복통", r"어지럽|어지러", r"찢|피가", r"메스|울렁", r"기운|힘이", r"숨|호흡"]
    extras = []
    for identifier, call in by_id.items():
        if identifier not in ids:
            continue
        i = ids[identifier]
        for j, text in enumerate(call.texts[:min(n[i]//3, 10)]):
            for k, pattern in enumerate(patterns):
                ix = off[i]+j
                if y[i,k] and re.search(pattern,text) and not pred[0][ix,k] and pred[1][ix,k] and pred[1][ix].sum() <= 2:
                    extras.append({"case": identifier, "n": int(n[i]), "step": j+1, "text": re.sub(r"\d{2,}", "[숫자 생략]",text)[:160],
                                   "gold": [c for c,yes in zip(CLASSES,y[i]) if yes],
                                   "predictions": [[c for c,yes in zip(CLASSES,p[ix]) if yes] for p in pred],
                                   "moment_scores": [float(s[ix]) for s in scores], "halts": [int(hb[i]),int(ht[i])]})
            if len(extras) >= 3:
                break
        if len(extras) >= 3:
            break
    dump(out / "supplement.json", {"stages": data, "score42_decomposition": decomposition,
                                    "delay_vs_length_spearman": length_corr, "delay_by_prediction_count": delays,
                                    "case_contexts": cases, "explicit_early_symptom_examples": extras,
                                    "direct_file_checks": direct})


if __name__ == "__main__":
    main()
