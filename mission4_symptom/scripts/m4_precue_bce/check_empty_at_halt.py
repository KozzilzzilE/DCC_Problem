"""감사용 검산: ENS3 C128 v0/v1 통화별 중단 기록에서 빈 예측 중단이 어느 시점(중간 vs 마지막 발화)에서 생겼는지 센다. 읽기 전용."""
import json, numpy as np
EXT = "/Volumes/로컬 디스크/DCC/private_experiments"
z = np.load(f"{EXT}/e026_same_input_replay_20261116/inputs/e002_dev_prefix.npz", allow_pickle=True)
print("npz keys:", list(z.keys()))
ids = [str(x) for x in z["call_ids"]]
off = z["offsets"]
nutt = {c: int(off[i+1]-off[i]) for i, c in enumerate(ids)}
for tag in ("v0", "v1_nonempty"):
    rows = json.load(open(f"{EXT}/e026_fitonly_policy_v1_20261117/ENS3_C128_{tag}_calls.json"))
    empty = [r for r in rows if not r["pred"]]
    at_end = [r for r in empty if r["halt"] == nutt[r["call_id"]]]
    short = [r for r in empty if nutt[r["call_id"]] < 5]
    mid = [r for r in empty if r["halt"] < nutt[r["call_id"]]]
    print(tag, "calls", len(rows), "empty", len(empty), "empty_at_last_utt", len(at_end), "empty_mid_call", len(mid), "n_utt<5", len(short))
    halts_end = sum(1 for r in rows if r["halt"] == nutt[r["call_id"]])
    print("   total halts at last utterance:", halts_end, " mean halt step:", np.mean([r["halt"] for r in rows]))
