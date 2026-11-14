"""E024 판정: 판정 임계값 0.5 고정(보정 없는 확률 >= 0.5, 빈 예측 허용)에서 pos_weight 지수가 4-1 을 바꾸는지 dev 2,920통화로 본다.

    python runs/E024_posweight/analyze.py

비교 대상은 E002(지수 0.5, seed 42)다. 새 모델은 지수만 바꾸고 나머지(TAPT 백본, 앞부분 균등 표본, lr, LLRD, seed 42)는 같다.
차이는 통화 단위 짝 부트스트랩 95% 구간과 함께 낸다. Validation 은 쓰지 않는다.
사전 등록 기준: "E002→새 모델 교체" 가 3개 평균 대비 0.002 를 넘고 구간이 0 을 벗어날 때만 채택 후보.
4-2 는 참고로 낸다 (멈춤: 기대 점수 >= 0.50, 보정 확률. 멈출 때 예측: 원 확률 >= 0.5).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from m4 import decision as dec  # noqa: E402
from m4 import simulate as sim  # noqa: E402
from halt_study import first_stop, score  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

OUT = HERE / "runs/E024_posweight"
BASE = {"E001": "runs/E001_m3_baseline", "E002": "runs/E002_prefix_uniform", "E007": "runs/E007_prefix_uniform_s43"}
NEW = {"p030": "runs/E024_posweight/p030", "p075": "runs/E024_posweight/p075"}


def per_call_41(pred, t):
    gold = np.repeat(t.labels, t.n_utt, axis=0)
    return np.add.reduceat(sim.moment_scores(pred, gold), t.offsets[:-1]) / t.n_utt


def boot(a, b, n=2000, seed=2026):
    d = a - b
    m = d[np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}


def main() -> int:
    t = {k: sim.PrefixTable.load(HERE / v / "dev_prefix.npz") for k, v in {**BASE, **NEW}.items()
         if (HERE / v / "dev_prefix.npz").exists()}
    ref = t["E002"]
    for x in t.values():
        assert x.call_ids == ref.call_ids and np.array_equal(x.offsets, ref.offsets)
    # 4-2 멈춤 신호는 모든 조합에서 같게 둔다 (현재 3개 평균의 보정 확률로 만든 기대 점수 >= 0.50)
    shift = np.log(json.loads((HERE / BASE["E002"] / "run_config.json").read_text(encoding="utf-8"))["pos_weight"])
    _, val = dec.expected_rule(dec.calibrate(np.mean([t[k].probs for k in BASE], axis=0), shift))
    halts = first_stop(val >= 0.50, ref.offsets)

    combos = {"E002 단독": ["E002"], "3개 평균(현재)": ["E001", "E002", "E007"]}
    for k in NEW:
        if k in t:
            combos[f"{k} 단독"] = [k]
            combos[f"E002→{k} 교체"] = ["E001", k, "E007"]
    out, pc = {}, {}
    print("== 4-1 원 확률 >= 0.5 (빈 예측 허용), dev 2,920통화  |  4-2 참고")
    for name, mem in combos.items():
        pred = np.mean([t[m].probs for m in mem], axis=0) >= 0.5
        pc[name] = per_call_41(pred, ref)
        r42 = score(halts, pred, ref)
        out[name] = {"m41": float(pc[name].mean()), "empty": float((~pred.any(1)).mean()),
                     "n_labels": float(pred.sum(1).mean()), "m42": r42["score"], "m42_f1": r42["macro_f1"]}
        print(f"  {name:<16} 4-1 {pc[name].mean():.4f}  빈 예측 {out[name]['empty']:.3f}  개수 {out[name]['n_labels']:.2f}"
              f"  | 4-2 {r42['score']:.4f} (F1 {r42['macro_f1']:.4f})")
    for k in NEW:
        if k not in t:
            continue
        for a, b in ((f"{k} 단독", "E002 단독"), (f"E002→{k} 교체", "3개 평균(현재)")):
            r = boot(pc[a], pc[b])
            out[f"diff/{a} - {b}"] = r
            print(f"  {a} − {b}: {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    (OUT / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
