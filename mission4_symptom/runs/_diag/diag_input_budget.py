"""지금 입력이 무엇을 담고 무엇을 버리는지 센다 (학습 없음, dev 2,920통화).

    python runs/_diag/diag_input_budget.py

보는 것
  1. 512토큰 한도를 넘는 순간의 비율과, 넘은 뒤 버려지는 것이 어느 쪽인지 (지금은 최신 발화가 잘린다).
  2. 입력 토큰 중 119대원 발화가 차지하는 몫 (화자 표시·선별의 여지).
  3. 전사가 비었거나 아주 짧은 발화의 비율 (음성에만 있는 정보의 여지).
Validation 은 쓰지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from m4.data import load_cache, read_ids  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

MAXLEN = 512


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HERE / "../../mission3-calibration/mission3_symptom/runs/devsel/tapt")
    calls = load_cache(HERE / "runs/cache/train.jsonl", read_ids(HERE / "runs/cache/dev_ids.txt"))
    print(f"dev 통화 {len(calls)}건")

    n_tok, is_op, n_char = [], [], []
    for c in calls:                                   # 발화마다 토큰 수 (공백 이어붙임과 같은 셈)
        ids = tok([t for t in c.texts], add_special_tokens=False)["input_ids"]
        n_tok.append(np.array([len(x) for x in ids]))
        is_op.append(np.array([s == 0 for s in c.speakers]))
        n_char.append(np.array([len(t.strip()) for t in c.texts]))

    tot_moments = sum(len(x) for x in n_tok)
    over, over_lost, over_call = 0, [], 0
    op_share_at_end, op_share_all = [], []
    for nt, op in zip(n_tok, is_op):
        cum = np.cumsum(nt) + 2                       # [CLS]/[SEP]
        o = cum > MAXLEN
        over += int(o.sum())
        if o.any():
            over_call += 1
            over_lost.append(float((cum[-1] - MAXLEN) / cum[-1]))   # 마지막 순간에 버려지는 몫
        op_share_all.append(float(nt[op].sum() / max(nt.sum(), 1)))
        k = int(np.searchsorted(cum, MAXLEN, side="right"))          # 한도 안에 들어가는 발화 수
        if k > 0:
            op_share_at_end.append(float(nt[:k][op[:k]].sum() / max(nt[:k].sum(), 1)))

    print(f"\n1. 512토큰 한도")
    print(f"  순간 {over}/{tot_moments} = {over/tot_moments:.3%} 가 한도를 넘는다")
    print(f"  통화 {over_call}/{len(calls)} = {over_call/len(calls):.1%} 가 한 번이라도 넘는다")
    if over_lost:
        q = np.percentile(over_lost, [50, 90, 99])
        print(f"  넘는 통화의 마지막 순간에 버려지는 토큰 몫: 중앙값 {q[0]:.1%}, 90% {q[1]:.1%}, 99% {q[2]:.1%}")
    print(f"  지금 설정(truncation_side=right)은 앞을 남기고 **최신 발화를 버린다**")

    print(f"\n2. 119대원 발화가 차지하는 입력 토큰 몫")
    print(f"  통화 전체 기준 평균 {np.mean(op_share_all):.1%} (중앙값 {np.median(op_share_all):.1%})")
    print(f"  512 한도 안 기준 평균 {np.mean(op_share_at_end):.1%}")

    allc = np.concatenate(n_char)
    allop = np.concatenate(is_op)
    print(f"\n3. 전사 길이")
    print(f"  빈 전사 {float((allc == 0).mean()):.2%}, 3자 이하 {float((allc <= 3).mean()):.1%}, 중앙값 {int(np.median(allc))}자")
    print(f"  신고자 발화만: 빈 전사 {float((allc[~allop] == 0).mean()):.2%}, 3자 이하 {float((allc[~allop] <= 3).mean()):.1%}")
    print(f"  통화당 발화 수 중앙값 {int(np.median([len(x) for x in n_tok]))}, 90% {int(np.percentile([len(x) for x in n_tok], 90))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
