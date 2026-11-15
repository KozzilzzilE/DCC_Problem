"""E022 판정: KcELECTRA-base-v2022 + TAPT이 4-1 을 올리는지 dev 2,920통화에서 본다.

    python runs/E022_kcelectra_tapt/analyze.py

E018 판정(runs/E018_speaker/analyze.py)과 같은 절차다. 바꾼 것은 비교 대상 확률표뿐이다.
차이는 통화 단위 짝 부트스트랩 95% 구간과 함께 낸다. Validation 은 쓰지 않는다.
사전 등록 기준: 구간이 0 을 벗어나고 차이가 0.002 를 넘을 때만 채택한다.
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

NEW = "E022"
NEW_DIR = HERE / "runs/E022_kcelectra_tapt"
TABLES = {
    "E001": HERE / "runs/E001_m3_baseline/dev_prefix.npz",
    "E002": HERE / "runs/E002_prefix_uniform/dev_prefix.npz",
    "E007": HERE / "runs/E007_prefix_uniform_s43/dev_prefix.npz",
    NEW: NEW_DIR / "dev_prefix.npz",
}


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


def main() -> int:
    fit_labels = label_matrix(load_cache(HERE / "runs/cache/train.jsonl", read_ids(HERE / "runs/cache/fit_ids.txt")))
    prior, marg = dec.label_set_prior(fit_labels)
    shift = np.log(json.loads((HERE / "runs/E002_prefix_uniform/run_config.json").read_text(encoding="utf-8"))["pos_weight"])
    shift_new = np.log(json.loads((NEW_DIR / "run_config.json").read_text(encoding="utf-8"))["pos_weight"])
    assert np.allclose(shift, shift_new), "pos_weight 가 E002 와 다릅니다"
    t = {k: sim.PrefixTable.load(p) for k, p in TABLES.items()}
    ref = t["E002"]
    for x in t.values():
        assert x.call_ids == ref.call_ids and np.array_equal(x.offsets, ref.offsets)

    def avg(names):
        return dec.calibrate(np.mean([t[n].probs for n in names], axis=0), shift)

    combos = {"E002 단독": ["E002"], f"{NEW} 단독": [NEW], "3개 평균(현재)": ["E001", "E002", "E007"],
              f"E002→{NEW} 교체": ["E001", NEW, "E007"], f"4개 평균(+{NEW})": ["E001", "E002", "E007", NEW]}
    pc, out = {}, {}
    print("== 4-1 (maxent 규칙), dev 2,920통화")
    for name, members in combos.items():
        pred = dec.maxent_rule(avg(members), prior, marg)[0]
        pc[name] = per_call_41(pred, ref)
        out[f"m41/{name}"] = float(pc[name].mean())
        print(f"  {name:<16} {pc[name].mean():.4f}  평균 개수 {pred.sum(axis=1).mean():.2f}")
    for a, b in ((f"{NEW} 단독", "E002 단독"), (f"E002→{NEW} 교체", "3개 평균(현재)"), (f"4개 평균(+{NEW})", "3개 평균(현재)")):
        r = boot(pc[a], pc[b])
        out[f"m41_diff/{a} - {b}"] = r
        print(f"  {a} − {b}: {fmt(r)}")

    # 참고: 임계값 0.5 고정 규칙(보정 없는 원 확률 >= 0.5, 빈 예측 허용)일 때. 판정에는 쓰지 않는다.
    print("== 참고: 4-1, 원 확률 >= 0.5 (보정·top1 없음)")
    pc05 = {}
    for name, members in combos.items():
        raw = np.mean([t[n].probs for n in members], axis=0)
        pred = dec.threshold_rule(raw, 0.5, force_top1=False)
        pc05[name] = per_call_41(pred, ref)
        out[f"m41_raw05/{name}"] = float(pc05[name].mean())
        print(f"  {name:<16} {pc05[name].mean():.4f}  빈 예측 순간 {(pred.sum(axis=1) == 0).mean():.3f}")
    for a, b in ((f"{NEW} 단독", "E002 단독"), (f"E002→{NEW} 교체", "3개 평균(현재)")):
        r = boot(pc05[a], pc05[b])
        out[f"m41_raw05_diff/{a} - {b}"] = r
        print(f"  {a} − {b}: {fmt(r)}")

    print("== 통화 전체를 들은 시점의 F1 (참고)")
    for name, members in combos.items():
        cal = avg(members)
        last = ref.offsets[1:] - 1
        pred = dec.maxent_rule(cal[last], prior, marg)[0]
        gold = ref.labels
        tp = (pred & (gold > 0)).sum(axis=0)
        f1 = 2 * tp / np.maximum(pred.sum(axis=0) + (gold > 0).sum(axis=0), 1)
        out[f"full_f1/{name}"] = float(f1.mean())
        print(f"  {name:<16} {f1.mean():.4f}")

    (NEW_DIR / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
