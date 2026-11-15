"""E014 판정: 증상 개수 보조 출력으로 학습한 모델이 4-1 을 올리는지 dev 2,920통화에서 본다.

    python runs/E014_count_aux/analyze.py

보는 것
  1. 보조 출력을 학습에만 쓴 경우 (추론은 지금과 같음): E014 단독 vs E002 단독, 3개 평균에서 E002 를 E014 로 바꾼 경우.
  2. 개수 출력의 품질: 모델이 낸 개수 분포 vs maxent 분포가 암시하는 개수 분포 (정답 개수 칸의 로그 손실).
  3. 개수 출력을 결정에도 쓴 경우: maxent 정답 집합 분포에 (개수 출력 / maxent 가 암시한 개수 분포)^α 를 곱해 다시 정규화.
     α=1 을 미리 정한 값으로 보고, 0.5 는 민감도 확인용이다.
차이는 통화 단위 짝 부트스트랩 95% 구간과 함께 낸다. Validation 은 쓰지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from m4 import decision as dec  # noqa: E402
from m4 import simulate as sim  # noqa: E402
from m4.data import label_matrix, load_cache, read_ids  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

M3 = HERE / "../../mission3-calibration/mission3_symptom/runs/devsel"
TABLES = {
    "E001": HERE / "runs/E001_m3_baseline/dev_prefix.npz",
    "E002": HERE / "runs/E002_prefix_uniform/dev_prefix.npz",
    "E007": HERE / "runs/E007_prefix_uniform_s43/dev_prefix.npz",
    "E014": HERE / "runs/E014_count_aux/dev_prefix.npz",
}
N_BUCKET = 3
BUCKET = np.minimum(dec._SET_SIZE, N_BUCKET).astype(int) - 1          # 정답 집합 511가지의 개수 칸


def per_call_41(pred: np.ndarray, t: sim.PrefixTable) -> np.ndarray:
    gold = np.repeat(t.labels, t.n_utt, axis=0)
    return np.add.reduceat(sim.moment_scores(pred, gold), t.offsets[:-1]) / t.n_utt


def boot(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 2026) -> dict:
    d = a - b
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}


def fmt(r: dict) -> str:
    return f"{r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"


def count_rule(cal: np.ndarray, count_probs: np.ndarray, prior, marg, alpha: float, max_k: int = 4, chunk: int = 15000):
    """maxent 분포에 개수 출력을 곱해 다시 정규화한 뒤 기대 순간 점수가 가장 큰 상위 k개를 낸다."""
    pred = np.zeros(cal.shape, dtype=bool)
    implied = np.zeros((len(cal), N_BUCKET))
    onehot = np.eye(N_BUCKET)[BUCKET]                                   # (511, 3)
    for s in range(0, len(cal), chunk):
        q = cal[s:s + chunk]
        w = dec.maxent_weights(q, prior, marg)
        m = w @ onehot                                                  # maxent 가 암시하는 개수 분포
        implied[s:s + chunk] = m
        if alpha > 0:
            ratio = (np.clip(count_probs[s:s + chunk], 1e-4, 1) / np.clip(m, 1e-4, 1)) ** alpha
            w = w * ratio[:, BUCKET]
            w /= w.sum(axis=1, keepdims=True)
        pred[s:s + chunk] = dec._best_top_k(w, q, max_k)[0]
    return pred, implied


def main() -> int:
    fit_labels = label_matrix(load_cache(HERE / "runs/cache/train.jsonl", read_ids(HERE / "runs/cache/fit_ids.txt")))
    prior, marg = dec.label_set_prior(fit_labels)
    shift = np.log(json.loads((HERE / "runs/E002_prefix_uniform/run_config.json").read_text(encoding="utf-8"))["pos_weight"])
    shift14 = np.log(json.loads((HERE / "runs/E014_count_aux/run_config.json").read_text(encoding="utf-8"))["pos_weight"])
    assert np.allclose(shift, shift14), "pos_weight 가 E002 와 다릅니다"
    t = {k: sim.PrefixTable.load(p) for k, p in TABLES.items()}
    ref = t["E002"]
    for x in t.values():
        assert x.call_ids == ref.call_ids and np.array_equal(x.offsets, ref.offsets)
    count_probs = np.load(HERE / "runs/E014_count_aux/dev_prefix.count.npy")
    out = {}

    def avg(names):
        return dec.calibrate(np.mean([t[n].probs for n in names], axis=0), shift)

    print("== 1. 보조 출력을 학습에만 쓴 경우 (4-1, maxent 규칙)")
    combos = {"E002": ["E002"], "E014": ["E014"], "3개 평균(현재)": ["E001", "E002", "E007"],
              "E002→E014 교체": ["E001", "E014", "E007"], "4개 평균(+E014)": ["E001", "E002", "E007", "E014"]}
    pc = {}
    for name, members in combos.items():
        pred = dec.maxent_rule(avg(members), prior, marg)[0]
        pc[name] = per_call_41(pred, ref)
        print(f"  {name:<18} {pc[name].mean():.4f}  평균 개수 {pred.sum(axis=1).mean():.2f}")
        out[f"m41/{name}"] = float(pc[name].mean())
    for a, b in (("E014", "E002"), ("E002→E014 교체", "3개 평균(현재)"), ("4개 평균(+E014)", "3개 평균(현재)")):
        r = boot(pc[a], pc[b])
        out[f"m41_diff/{a} - {b}"] = r
        print(f"  {a} − {b}: {fmt(r)}")

    print("== 2. 개수 출력의 품질 (E014 단독, 순간 단위)")
    cal14 = avg(["E014"])
    n = ref.n_utt
    true_b = np.repeat(np.minimum(ref.labels.sum(axis=1), N_BUCKET).astype(int) - 1, n)
    pred0, implied = count_rule(cal14, count_probs, prior, marg, 0.0)
    step = np.concatenate([np.arange(1, k + 1) for k in n])
    rel = step / np.repeat(n, n)
    w_moment = 1.0 / np.repeat(n, n)                                    # 통화마다 같은 무게
    base = np.bincount(np.minimum(fit_labels.sum(axis=1), N_BUCKET).astype(int) - 1, minlength=N_BUCKET) / len(fit_labels)
    rows = np.arange(len(true_b))
    for name, lo, hi in (("전체", 0, 1), ("앞 1/3", 0, 1 / 3), ("가운데", 1 / 3, 2 / 3), ("뒤 1/3", 2 / 3, 1)):
        sel = (rel > lo) & (rel <= hi)
        w = w_moment[sel] / w_moment[sel].sum()
        nll = {k: float(-(w * np.log(np.clip(p[rows[sel], true_b[sel]], 1e-6, 1))).sum())
               for k, p in (("개수 출력", count_probs), ("maxent 암시", implied), ("사전 비율", np.tile(base, (len(true_b), 1))))}
        acc = {k: float((w * (p[sel].argmax(axis=1) == true_b[sel])).sum()) for k, p in (("개수 출력", count_probs), ("maxent 암시", implied))}
        out[f"count_quality/{name}"] = {"nll": nll, "acc": acc}
        print(f"  {name:<7} 로그 손실: 개수 출력 {nll['개수 출력']:.4f} / maxent 암시 {nll['maxent 암시']:.4f} / 사전 비율 {nll['사전 비율']:.4f}"
              f"   정확도: {acc['개수 출력']:.3f} / {acc['maxent 암시']:.3f}")

    print("== 3. 개수 출력을 결정에도 쓴 경우 (E014 단독, 4-1)")
    pc0 = per_call_41(pred0, ref)
    for alpha in (1.0, 0.5):
        pred, _ = count_rule(cal14, count_probs, prior, marg, alpha)
        p = per_call_41(pred, ref)
        r = boot(p, pc0)
        r2 = boot(p, pc["E002"])
        out[f"count_rule/alpha{alpha}"] = {"score": float(p.mean()), "mean_labels": float(pred.sum(axis=1).mean()),
                                           "vs_E014_maxent": r, "vs_E002_maxent": r2}
        print(f"  α={alpha}: {p.mean():.4f} (평균 개수 {pred.sum(axis=1).mean():.2f})  E014 maxent 대비 {fmt(r)}  E002 maxent 대비 {fmt(r2)}")
    # 3개 평균의 확률 + E014 의 개수 출력
    cal3 = avg(["E001", "E014", "E007"])
    for alpha in (1.0, 0.5):
        pred, _ = count_rule(cal3, count_probs, prior, marg, alpha)
        p = per_call_41(pred, ref)
        r = boot(p, pc["3개 평균(현재)"])
        out[f"count_rule_ens/alpha{alpha}"] = {"score": float(p.mean()), "vs_current": r}
        print(f"  교체한 3개 평균 + 개수 출력 α={alpha}: {p.mean():.4f}  현재 제출 후보 대비 {fmt(r)}")

    (HERE / "runs/E014_count_aux/analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
