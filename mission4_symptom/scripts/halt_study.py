"""4-2 중단 정책 비교 (학습 없이 dev 확률표만 사용).

    python scripts/halt_study.py --tables runs/E002_prefix_uniform/dev_prefix.npz --shift-from runs/E002_prefix_uniform --out runs/E004_halt_study/e002.json

A. 규칙형: (중단 신호) x (기준값) x (중단 시점 결정 규칙) 격자.
B. 학습형: 그 순간까지의 정보로 만든 특징에서 "지금 끊는 것이 이후 어느 시점보다 낫다"를 맞히는 분류기.

기준값을 dev 에서 고른 점수는 낙관적이다. 그래서 통화 단위 5겹 교차검증(4겹에서 고르고 남은 1겹에서 채점)
점수를 함께 낸다. 교차검증 점수가 "고른 뒤에도 남는" 성능의 추정치다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4 import decision as dec  # noqa: E402
from m4 import simulate as sim  # noqa: E402
from m4.data import load_cache  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

GRACE = 5


def first_stop(stop: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """통화별로 stop 이 처음 참인 step(1-based). GRACE 이전은 무시하고, 없으면 통화 길이."""
    n = np.diff(offsets)
    step = np.concatenate([np.arange(1, k + 1) for k in n])
    cand = np.where(stop & (step >= GRACE), step, 10 ** 9)
    first = np.minimum.reduceat(cand, offsets[:-1])
    return np.minimum(first, n)


def score(halts: np.ndarray, pred_all: np.ndarray, table: sim.PrefixTable, calls=None) -> dict:
    """중단 시점과 순간별 예측으로 4-2 점수를 낸다. calls 를 주면 그 통화들만 채점한다."""
    idx = np.arange(len(table)) if calls is None else calls
    n = table.n_utt[idx]
    pred = pred_all[table.offsets[:-1][idx] + halts[idx] - 1]
    f1 = sim.macro_f1(pred, table.labels[idx])["macro_f1"]
    e = float(sim.earliness(halts[idx], n).mean())
    return {"score": f1 * e, "macro_f1": f1, "earliness": e, "halt_median": float(np.median(halts[idx]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", required=True, help="확률표 (여러 개면 확률 평균)")
    ap.add_argument("--shift-from", required=True, help="pos_weight 를 읽을 run 폴더")
    ap.add_argument("--calls", default="runs/cache/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    tables = [sim.PrefixTable.load(p) for p in args.tables]
    table = tables[0]
    if len(tables) > 1:
        table = sim.PrefixTable(table.call_ids, table.offsets, np.mean([t.probs for t in tables], axis=0), table.labels)
    shift = np.log(json.loads((Path(args.shift_from) / "run_config.json").read_text(encoding="utf-8"))["pos_weight"])
    n = table.n_utt
    off = table.offsets
    M = int(off[-1])
    step = np.concatenate([np.arange(1, k + 1) for k in n])
    gold = np.repeat(table.labels, n, axis=0)
    rng = np.random.default_rng(0)
    fold = rng.permutation(len(table)) % args.folds

    cal = dec.calibrate(table.probs, shift)
    pred_exp, value = dec.expected_rule(cal)
    preds = {"임계 0.5(비면 1개)": dec.threshold_rule(table.probs.copy()), "기대 점수 최대화": pred_exp}
    signals = {"최대 확률": table.probs.max(axis=1), "기대 점수": value}
    out = {"tables": args.tables, "n_calls": len(table), "rule_grid": [], "learned": {}}

    # ------------------------------------------------------------------ A. 규칙형
    print("== A. 규칙형 (dev 전체 / 5겹 교차검증)")
    taus = np.round(np.arange(0.40, 0.99, 0.02), 2)
    for sig_name, sig in signals.items():
        for pred_name, pred in preds.items():
            halts = {t: first_stop(sig >= t, off) for t in taus}
            full = {t: score(h, pred, table) for t, h in halts.items()}
            best = max(taus, key=lambda t: full[t]["score"])
            cv = []
            for k in range(args.folds):
                tr, te = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
                pick = max(taus, key=lambda t: score(halts[t], pred, table, tr)["score"])
                cv.append((score(halts[pick], pred, table, te)["score"], pick))
            row = {"signal": sig_name, "halt_rule": pred_name, "best_tau": float(best), **full[best],
                   "cv_score": float(np.mean([c[0] for c in cv])), "cv_taus": [float(c[1]) for c in cv]}
            out["rule_grid"].append(row)
            print(f"  신호 {sig_name:<6} 규칙 {pred_name:<14} τ={best:.2f}  {row['score']:.4f} = F1 {row['macro_f1']:.4f} x "
                  f"{row['earliness']:.4f}  중단 중앙 {row['halt_median']:.0f}  | 교차검증 {row['cv_score']:.4f} (τ {row['cv_taus']})")

    # ------------------------------------------------------------------ B. 학습형
    from sklearn.ensemble import HistGradientBoostingClassifier

    calls = {c.call_id: c for c in load_cache(Path(args.calls), set(table.call_ids))}
    ends = np.concatenate([np.asarray(calls[i].ends, dtype=np.float64) for i in table.call_ids]) / 1000.0
    chars = np.concatenate([np.cumsum([len(t) for t in calls[i].texts]) for i in table.call_ids]).astype(np.float64)
    caller = np.concatenate([np.cumsum([s == 1 for s in calls[i].speakers]) for i in table.call_ids]) / step
    top = -np.sort(-cal, axis=1)[:, :3]
    first = np.zeros(M, dtype=bool)
    first[off[:-1]] = True
    prev_value = np.where(first, value, np.roll(value, 1))
    arg = cal.argmax(axis=1)
    changed = first | (arg != np.roll(arg, 1))
    # 최고 확률 증상이 마지막으로 바뀐 뒤 지난 발화 수
    last_change = np.maximum.accumulate(np.where(changed, np.arange(M), 0))
    since = np.arange(M) - last_change
    feats = np.column_stack([top, value, value - prev_value, table.probs.max(axis=1), cal.sum(axis=1),
                             step, ends, chars, caller, since])
    names = ["보정확률 1위", "2위", "3위", "기대 점수", "기대 점수 변화", "최대 확률", "보정확률 합",
             "발화 순번", "경과 초", "누적 글자 수", "신고자 발화 비율", "1위 유지 발화 수"]

    for pred_name, pred in preds.items():
        # 사후 정답: 지금 끊었을 때의 (순간 점수 x 조기판단력)이 이후 어느 시점보다 작지 않으면 1
        reward = sim.moment_scores(pred, gold) * sim.earliness(step, np.repeat(n, n))
        future = np.empty(M)
        for i in range(len(table)):
            r = reward[off[i]:off[i + 1]]
            f = np.maximum.accumulate(r[::-1])[::-1]
            future[off[i]:off[i + 1]] = np.append(f[1:], -1.0)
        target = (reward >= future - 1e-9).astype(np.int8)
        usable = step >= GRACE
        call_of = np.repeat(np.arange(len(table)), n)
        prob = np.zeros(M)
        for k in range(args.folds):
            tr = usable & (fold[call_of] != k)
            te = fold[call_of] == k
            clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, random_state=0)
            clf.fit(feats[tr], target[tr])
            prob[te] = clf.predict_proba(feats[te])[:, 1]
        thetas = np.round(np.arange(0.20, 0.91, 0.02), 2)
        halts = {t: first_stop(prob >= t, off) for t in thetas}
        full = {t: score(h, pred, table) for t, h in halts.items()}
        best = max(thetas, key=lambda t: full[t]["score"])
        cv = []
        for k in range(args.folds):
            tr, te = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
            pick = max(thetas, key=lambda t: score(halts[t], pred, table, tr)["score"])
            cv.append((score(halts[pick], pred, table, te)["score"], pick))
        row = {"halt_rule": pred_name, "best_theta": float(best), **full[best],
               "cv_score": float(np.mean([c[0] for c in cv])), "cv_thetas": [float(c[1]) for c in cv],
               "features": names}
        out["learned"][pred_name] = row
        print(f"== B. 학습형 (규칙 {pred_name}) θ={best:.2f}  {row['score']:.4f} = F1 {row['macro_f1']:.4f} x {row['earliness']:.4f}  "
              f"중단 중앙 {row['halt_median']:.0f}  | 교차검증 {row['cv_score']:.4f} (θ {row['cv_thetas']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
