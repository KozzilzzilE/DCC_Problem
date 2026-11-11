"""4-1: '주변확률 일치(CM)' 수정이 확률표가 달라도 같은 방향인지, 이득이 어디서 나오는지 확인한다 (고르는 값 없음).

기존 expected_rule 은 증상 독립 분포에서 빈 집합을 빼고 다시 정규화한다. 그러면 증상별 주변확률이 보정 확률보다
커져서(ENS 기준 1.129배) 증상을 너무 많이 낸다. CM 은 '빈 집합을 뺀 뒤의 주변확률 = 보정 확률' 이 되도록
독립 모수를 다시 푼 것이다 (setmodel.consistent_marginals). 값은 고르지 않는다.

출력
  1) 확률표 6종에서 기존 vs CM (통화 단위 짝 부트스트랩 1,000회)
  2) 발화 순번 구간별, 통화의 정답 증상 수별 차이
  3) 개수 상한(max_k=1..4)별 점수 (기존 / CM). 상한은 dev 전체에서 본 값이라 참고용.
"""
from __future__ import annotations

import numpy as np

import cw
import setmodel as sm
from cw import Ctx
from s09_direct41 import fast_rule, to_pred

SETS = {"E001": ("E001",), "E002": ("E002",), "E007": ("E007",), "E003": ("E003",),
        "E002+E007": ("E002", "E007"), "E001+E002+E007": ("E001", "E002", "E007")}
EDGES = np.array([5, 10, 20])
BNAME = ["1-5", "6-10", "11-20", "21+"]


def main():
    W = cw.boot_weights(2920)
    out = {}
    for key, names in SETS.items():
        c = Ctx(names)
        gold = c.gold.astype(np.float64)
        q0 = np.clip(c.cal, 1e-4, 1 - 1e-4)
        q1 = sm.consistent_marginals(c.cal)
        res = {}
        preds = {}
        for tag, q in (("기존", q0), ("CM", q1)):
            for mk in (1, 2, 3, 4):
                bk, order, _ = fast_rule(q, max_k=mk)
                p = to_pred(bk, order)
                preds[(tag, mk)] = p
                res[f"{tag}|max_k={mk}"] = {"m4_1": float(c.per_call_41(p).mean()), "mean_k": float(bk.mean())}
        base_pc = c.per_call_41(preds[("기존", 4)])
        cm_pc = c.per_call_41(preds[("CM", 4)])
        d, lo, hi = cw.boot_41(cm_pc, base_pc, W)
        res["CM-기존"] = {"delta": d, "lo": lo, "hi": hi}
        print(f"\n===== {key}: 기존 {base_pc.mean():.4f} (개수 {res['기존|max_k=4']['mean_k']:.2f}) → CM {cm_pc.mean():.4f} "
              f"(개수 {res['CM|max_k=4']['mean_k']:.2f})  차이 {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
        # 구간별 (순간 평균)
        ms0 = cw.sim.moment_scores(preds[("기존", 4)], c.gold)
        ms1 = cw.sim.moment_scores(preds[("CM", 4)], c.gold)
        bucket = np.searchsorted(EDGES, c.step, side="left")
        line = []
        res["by_step"] = {}
        for b in range(4):
            m = bucket == b
            k0 = preds[("기존", 4)][m].sum(axis=1).mean()
            k1 = preds[("CM", 4)][m].sum(axis=1).mean()
            res["by_step"][BNAME[b]] = {"base": float(ms0[m].mean()), "cm": float(ms1[m].mean()),
                                        "k_base": float(k0), "k_cm": float(k1), "true_k": float(gold[m].sum(axis=1).mean())}
            line.append(f"{BNAME[b]}: {ms0[m].mean():.3f}→{ms1[m].mean():.3f} (개수 {k0:.2f}→{k1:.2f})")
        print("   순번 구간별 순간 점수: " + "  ".join(line))
        nlab = c.labels.sum(axis=1)
        line = []
        res["by_nlabel"] = {}
        for name, m in (("1개", nlab == 1), ("2개", nlab == 2), ("3개+", nlab >= 3)):
            res["by_nlabel"][name] = {"n": int(m.sum()), "base": float(base_pc[m].mean()), "cm": float(cm_pc[m].mean())}
            line.append(f"{name}({int(m.sum())}통화): {base_pc[m].mean():.3f}→{cm_pc[m].mean():.3f}")
        print("   정답 증상 수별 통화 점수: " + "  ".join(line))
        print("   개수 상한별 4-1 (기존 / CM): " + "  ".join(
            f"k≤{mk}: {res[f'기존|max_k={mk}']['m4_1']:.4f} / {res[f'CM|max_k={mk}']['m4_1']:.4f}" for mk in (1, 2, 3, 4)))
        # 증상별: 4-1 예측에 그 증상을 넣은 순간의 정밀도, 넣은 횟수
        line = []
        res["per_class"] = {}
        for j, name in enumerate(cw.CLS):
            a, b2 = preds[("기존", 4)][:, j], preds[("CM", 4)][:, j]
            pa = c.gold[a, j].mean() if a.any() else float("nan")
            pb = c.gold[b2, j].mean() if b2.any() else float("nan")
            ra = a[c.gold[:, j]].mean()
            rb = b2[c.gold[:, j]].mean()
            res["per_class"][name] = {"n_base": int(a.sum()), "n_cm": int(b2.sum()), "prec_base": float(pa), "prec_cm": float(pb),
                                      "rec_base": float(ra), "rec_cm": float(rb)}
            line.append(f"{name} {int(a.sum())}→{int(b2.sum())} P{pa:.2f}→{pb:.2f} R{ra:.2f}→{rb:.2f}")
        print("   증상별 예측 횟수·정밀도·재현율(순간): " + " | ".join(line))
        out[key] = res
    cw.save_json("s14_cm_check.json", out)


if __name__ == "__main__":
    main()
