"""감사용 검산: Dev 2,920 시점별 원확률(npz)로 4-1(시점 평균 Jaccard, raw p>=0.5, Top-1 없음)과 빈 예측률을 재계산한다. 읽기 전용."""
import sys, numpy as np
SRC = "/Users/sunggyu/Desktop/이성규/DCC/artifacts_private/e026_same_input_replay_20261116/source"
sys.path.insert(0, SRC)
from m4 import simulate as sim
EXT = "/Volumes/로컬 디스크/DCC/private_experiments"
INP = f"{EXT}/e026_same_input_replay_20261116/inputs"
P = {
 "E001": f"{INP}/e001_dev_prefix.npz", "E002": f"{INP}/e002_dev_prefix.npz", "E007": f"{INP}/e007_dev_prefix.npz",
 "SJ42": f"{EXT}/m4_softjaccard_v1_20261116/s42/light/dev_prefix.npz", "SJ43": f"{EXT}/m4_softjaccard_v1_20261116/s43/light/dev_prefix.npz",
 "JM42": f"{EXT}/m4_jacc_macrof1_v1_20261116/s42/light/dev_prefix.npz", "JM43": f"{EXT}/m4_jacc_macrof1_v1_20261116/s43/light/dev_prefix.npz",
}
T = {k: sim.PrefixTable.load(v) for k, v in P.items()}
base = T["E002"]
for k, t in T.items():
    assert t.call_ids == base.call_ids and np.array_equal(t.offsets, base.offsets) and np.array_equal(t.labels, base.labels), k
combos = {"E002": ["E002"], "ENS3": ["E001","E002","E007"], "SJ_2seed_avg": ["SJ42","SJ43"], "JM_2seed_avg": ["JM42","JM43"],
          "ENS4_sj": ["E001","E007","SJ42","SJ43"], "ENS4_jm": ["E001","E007","JM42","JM43"], "SJ42": ["SJ42"], "SJ43": ["SJ43"]}
rule = lambda p: sim.decide(p, 0.5, force_top1=False)
for name, mem in combos.items():
    probs = np.mean([T[m].probs for m in mem], axis=0).astype(np.float32)
    tbl = sim.PrefixTable(base.call_ids, base.offsets, probs, base.labels)
    r = sim.score_4_1(tbl, rule)
    pred = probs >= 0.5
    empty = float((~pred.any(axis=1)).mean())
    near = float((np.abs(probs - 0.5) < 1e-3).mean())
    print(f"{name:14s} 4-1={r['score']:.6f} empty_rate={empty*100:.2f}% n_rows={len(probs)} |p-0.5|<1e-3 frac={near:.5f}")
