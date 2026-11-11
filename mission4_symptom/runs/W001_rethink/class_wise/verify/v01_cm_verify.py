"""반박 검증: '주변확률 일치(CM)' 4-1 이득 주장을 독립 코드로 다시 계산한다.

class_wise 의 cw.py / setmodel.py / s09 / s14 는 가져오지 않는다. m4/ 도 채점·규칙 계산에는 쓰지 않고
(확률표는 np.load 로 직접 읽는다), 끝에서 내 기준 규칙이 m4.decision.expected_rule 과 같은지 대조할 때만 쓴다.

- 기준 규칙: 증상 독립 분포(보정 확률 q)에서 511가지 정답 집합을 모두 열거해 상위 k(1..4)의 기대 순간 점수를 구한다.
- CM(정확 해): p'_c = q_c * u, u 는 prod(1 - q_c u) = 1 - u 의 (0,1) 해. 이분법으로 푼다.
  sum(q) <= 1 이면 해가 없다(빈 집합이 아닌 분포의 주변확률 합은 1 이상). 이때의 극한 분포는 '정답 1개, 확률 q_c/sum q'.
- CM(반복): 주장 쪽 구현 방식(p <- q(1 - prod(1 - p)), 바닥 q*1e-3)을 내 코드로 다시 써서 반복 횟수 민감도를 본다.
- 예측에 쓰는 것은 그 순간의 9종 확률뿐이다. n_utt 는 채점에만 쓴다. Validation 은 읽지 않는다.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

RUNS = {"E001": "E001_m3_baseline", "E002": "E002_prefix_uniform", "E007": "E007_prefix_uniform_s43",
        "E003": "E003_prefix_plus_full_s43"}
COMBOS = {"E001": ("E001",), "E002": ("E002",), "E007": ("E007",), "E003": ("E003",),
          "E002+E007": ("E002", "E007"), "ENS(E001+E002+E007)": ("E001", "E002", "E007")}
SETS = np.array(list(itertools.product([0, 1], repeat=9)), dtype=np.float64)[1:]     # (511, 9), 빈 집합 제외
SIZE = SETS.sum(axis=1)


# ------------------------------------------------------------------ 자료
def load(name):
    z = np.load(ROOT / "runs" / RUNS[name] / "dev_prefix.npz", allow_pickle=False)
    return [str(x) for x in z["call_ids"]], z["offsets"].astype(np.int64), z["probs"].astype(np.float64), z["labels"].astype(bool)


def calibrate(p, shift):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) - shift)))


# ------------------------------------------------------------------ 규칙 (독립 구현)
def topk_expected(p, max_k=4, chunk=15000):
    """독립 모수 p 의 '빈 집합 제외' 분포에서 상위 k 의 기대 순간 점수. (k (m,), 기대 점수 (m,), 정렬 (m,9))"""
    m = len(p)
    best_k = np.ones(m, dtype=np.int64)
    best_v = np.full(m, -1.0)
    order_all = np.argsort(-p, axis=1, kind="stable")
    for s in range(0, m, chunk):
        q = p[s:s + chunk]
        lw = np.log(q) @ SETS.T + np.log1p(-q) @ (1 - SETS).T
        lw -= lw.max(axis=1, keepdims=True)
        w = np.exp(lw)
        w /= w.sum(axis=1, keepdims=True)
        order = order_all[s:s + chunk]
        chosen = np.zeros(q.shape)
        bv = np.full(len(q), -1.0)
        bk = np.ones(len(q), dtype=np.int64)
        for k in range(1, max_k + 1):
            chosen[np.arange(len(q)), order[:, k - 1]] = 1.0
            inter = chosen @ SETS.T
            v = (w * inter / (k + SIZE[None, :] - inter)).sum(axis=1)
            up = v > bv + 1e-12
            bv[up] = v[up]
            bk[up] = k
        best_k[s:s + chunk] = bk
        best_v[s:s + chunk] = bv
    return best_k, best_v, order_all


def to_pred(k, order):
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(9)[None, :].repeat(len(order), axis=0), axis=1)
    return rank < k[:, None]


def cm_exact(q):
    """prod(1 - q_c u) = 1 - u 의 (0,1) 해 u. sum(q) <= 1 이면 해가 없어 u = 0 (degenerate=True)."""
    q = np.clip(q, 1e-4, 1 - 1e-4)
    deg = q.sum(axis=1) <= 1.0
    lo = np.full(len(q), 1e-12)
    hi = np.full(len(q), 1 - 1e-13)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        d = np.log1p(-q * mid[:, None]).sum(axis=1) - np.log1p(-mid)     # <0 이면 해가 더 오른쪽
        neg = d < 0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    u = 0.5 * (lo + hi)
    u[deg] = 0.0
    return q, u, deg


def rule_cm_exact(cal, max_k=4):
    q, u, deg = cm_exact(cal)
    # 수치상 u 가 너무 작으면(합이 1 을 겨우 넘는 경우) 극한 분포와 같으므로 같이 처리
    deg = deg | (u < 1e-9)
    p = q * np.where(deg, 1.0, u)[:, None]
    k, v, order = topk_expected(p, max_k)
    # 해가 없는 순간: 극한 분포 = 정답 1개 (확률 q_c / sum q) → 1개만 내는 것이 최적, 기대 점수 = max q / sum q
    k[deg] = 1
    v[deg] = q[deg].max(axis=1) / q[deg].sum(axis=1)
    order_q = np.argsort(-q, axis=1, kind="stable")
    return to_pred(k, order_q), v, k, deg, u


def rule_cm_iter(cal, iters=200, floor=1e-3, max_k=4):
    q = np.clip(cal, 1e-4, 1 - 1e-4)
    p = q.copy()
    for _ in range(iters):
        p = np.maximum(q * (1.0 - np.prod(1.0 - p, axis=1, keepdims=True)), q * floor)
    k, v, order = topk_expected(p, max_k)
    return to_pred(k, order), v, k


# ------------------------------------------------------------------ 채점 (독립 구현)
def per_call_41(pred, labels, off):
    n = np.diff(off)
    gold = np.repeat(labels, n, axis=0)
    inter = (pred & gold).sum(axis=1)
    union = (pred | gold).sum(axis=1)
    ms = inter / union                       # 정답이 항상 1개 이상이라 union >= 1
    cs = np.concatenate([[0.0], np.cumsum(ms)])
    return (cs[off[1:]] - cs[off[:-1]]) / n, ms


def boot_ci(d, B=1000, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    bs = d[idx].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs > 0).mean()), float((bs > 0.002).mean())


def first_stop(stop, off, grace=5):
    n = np.diff(off)
    halts = n.copy()
    for i in range(len(n)):
        s = stop[off[i]:off[i + 1]]
        hit = np.nonzero(s[grace - 1:])[0]
        if len(hit):
            halts[i] = hit[0] + grace
    return halts


def macro_f1(pred, gold):
    tp = (pred & gold).sum(axis=0).astype(float)
    fp = (pred & ~gold).sum(axis=0).astype(float)
    fn = (~pred & gold).sum(axis=0).astype(float)
    seen = (tp + fp + fn) > 0
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1.0)
    return float(f1[seen].mean())


def earliness(h, n):
    return 0.5 + 0.5 * np.maximum(0.0, 1.0 - np.maximum(h, 5) / n)


def score42(halts, pred_all, labels, off, idx=None):
    n = np.diff(off)
    idx = np.arange(len(n)) if idx is None else idx
    ph = pred_all[off[:-1][idx] + halts[idx] - 1]
    return macro_f1(ph, labels[idx]) * float(earliness(halts[idx], n[idx]).mean())


def raw_rule(p):
    pred = p >= 0.5
    empty = ~pred.any(axis=1)
    pred[empty, p[empty].argmax(axis=1)] = True
    return pred


def boot42(h_a, h_b, pred_all, labels, off, B=1000, seed=7):
    n = np.diff(off)
    pa = pred_all[off[:-1] + h_a - 1]
    pb = pred_all[off[:-1] + h_b - 1]
    ea, eb = earliness(h_a, n), earliness(h_b, n)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(n), np.full(len(n), 1 / len(n)), size=B).astype(float)

    def sc(p, e):
        tp = W @ (p & labels).astype(float); fp = W @ (p & ~labels).astype(float); fn = W @ (~p & labels).astype(float)
        return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)).mean(axis=1) * (W @ e / len(n))
    d = sc(pa, ea) - sc(pb, eb)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ------------------------------------------------------------------ 본문
def main():
    shift = np.log(np.asarray(json.loads((ROOT / "runs" / RUNS["E002"] / "run_config.json").read_text(encoding="utf-8"))["pos_weight"]))
    tabs = {k: load(k) for k in RUNS}
    ids0, off, _, labels = tabs["E002"]
    for k, (ids, o, p, lab) in tabs.items():
        assert ids == ids0 and (o == off).all() and (lab == labels).all(), k
    N = len(ids0)
    n = np.diff(off)
    step = np.concatenate([np.arange(1, x + 1) for x in n])
    call_of = np.repeat(np.arange(N), n)

    # 자료 점검: dev 와 학습 목록이 겹치지 않는지, 확률표 통화가 dev 목록인지
    dev_ids = set(x.strip() for x in (ROOT / "runs/cache/dev_ids.txt").read_text(encoding="utf-8").split() if x.strip())
    fit_ids = set(x.strip() for x in (ROOT / "runs/cache/fit_ids.txt").read_text(encoding="utf-8").split() if x.strip())
    print(f"[자료] 통화 {N}, 순간 {off[-1]}, 확률표 통화 ⊆ dev 목록: {set(ids0) <= dev_ids}, dev ∩ 학습 목록: {len(dev_ids & fit_ids)}")
    nlab = labels.sum(axis=1)
    print(f"[자료] dev 정답 증상 수 분포: 1개 {np.mean(nlab == 1):.3f}, 2개 {np.mean(nlab == 2):.3f}, 3개 {np.mean(nlab == 3):.3f}, 4개+ {np.mean(nlab >= 4):.3f}")

    my_fold = np.random.default_rng(987).permutation(N) % 5          # 내 분할
    their_fold = np.random.default_rng(0).permutation(N) % 5        # 기준 문서와 같은 분할 (4-2 τ 고르기 비교용)
    out = {}
    keep = {}
    for key, names in COMBOS.items():
        raw = np.mean([tabs[x][2] for x in names], axis=0)
        cal = calibrate(raw, shift)
        qb = np.clip(cal, 1e-4, 1 - 1e-4)
        kb, vb, ob = topk_expected(qb)
        pred_b = to_pred(kb, ob)
        pred_c, vc, kc, deg, u = rule_cm_exact(cal)
        pc_b, ms_b = per_call_41(pred_b, labels, off)
        pc_c, ms_c = per_call_41(pred_c, labels, off)
        d = pc_c - pc_b
        lo, hi, ppos, p002 = boot_ci(d)
        r = {"base": float(pc_b.mean()), "cm_exact": float(pc_c.mean()), "delta": float(d.mean()), "lo": lo, "hi": hi,
             "P(d>0)": ppos, "P(d>0.002)": p002, "k_base": float(kb.mean()), "k_cm": float(kc.mean()),
             "degenerate_frac": float(deg.mean())}
        print(f"\n===== {key}")
        print(f"  4-1 기준 {pc_b.mean():.4f} (개수 {kb.mean():.2f}) → CM(정확 해) {pc_c.mean():.4f} (개수 {kc.mean():.2f})  "
              f"차이 {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  P(차이>0)={ppos:.3f}  P(차이>0.002)={p002:.3f}")
        # 반복 횟수 민감도 (주장 쪽 구현 방식)
        r["iter"] = {}
        line = []
        for it in (20, 50, 200, 1000, 5000):
            pi, vi, ki = rule_cm_iter(cal, iters=it)
            pci, _ = per_call_41(pi, labels, off)
            l2, h2, _, _ = boot_ci(pci - pc_b)
            r["iter"][it] = {"m4_1": float(pci.mean()), "delta": float((pci - pc_b).mean()), "lo": l2, "hi": h2,
                             "k": float(ki.mean()), "diff_moments_vs_exact": float((pi != pred_c).any(axis=1).mean())}
            line.append(f"{it}회 {pci.mean():.4f}({(pci - pc_b).mean():+.4f}, 개수 {ki.mean():.2f}, 정확 해와 다른 순간 {100 * (pi != pred_c).any(axis=1).mean():.2f}%)")
            if it == 200:
                keep[(key, "iter200")] = (pi, vi)
        print("  반복 횟수별(바닥 q*1e-3): " + " | ".join(line))
        # 해가 없는 순간(sum q <= 1)과 있는 순간으로 이득 분해 (순간 점수를 1/n 가중으로 더하면 통화 평균과 같다)
        wmom = 1.0 / n[call_of] / N
        dm = (ms_c - ms_b) * wmom
        chg = (pred_c != pred_b).any(axis=1)
        r["decomp"] = {"deg_part": float(dm[deg].sum()), "nondeg_part": float(dm[~deg].sum()), "changed_frac": float(chg.mean()),
                       "changed_mean_gain": float((ms_c - ms_b)[chg].mean())}
        print(f"  sum q<=1 이라 해가 없는 순간 {100 * deg.mean():.1f}% (CM 은 무조건 1개) → 이득 기여 {dm[deg].sum():+.4f}, "
              f"해가 있는 순간 기여 {dm[~deg].sum():+.4f}; 예측이 바뀐 순간 {100 * chg.mean():.1f}%, 그 순간의 평균 변화 {(ms_c - ms_b)[chg].mean():+.4f}")
        # 순번 구간별
        line = []
        for name, m in (("1-5", step <= 5), ("6-10", (step > 5) & (step <= 10)), ("11-20", (step > 10) & (step <= 20)), ("21+", step > 20)):
            line.append(f"{name}: {ms_b[m].mean():.3f}→{ms_c[m].mean():.3f} (개수 {pred_b[m].sum(1).mean():.2f}→{pred_c[m].sum(1).mean():.2f})")
        print("  순번 구간별 순간 점수: " + "  ".join(line))
        # 정답 증상 수별
        r["by_nlabel"] = {}
        line = []
        dk = {}
        for name, m in (("1개", nlab == 1), ("2개", nlab == 2), ("3개+", nlab >= 3)):
            l3, h3, _, _ = boot_ci(d[m])
            dk[name] = float(d[m].mean())
            r["by_nlabel"][name] = {"n": int(m.sum()), "base": float(pc_b[m].mean()), "cm": float(pc_c[m].mean()), "delta": dk[name], "lo": l3, "hi": h3}
            line.append(f"{name}({int(m.sum())}): {pc_b[m].mean():.3f}→{pc_c[m].mean():.3f} ({dk[name]:+.4f} [{l3:+.4f},{h3:+.4f}])")
        print("  정답 증상 수별 통화 점수: " + "  ".join(line))
        # 증상 1개 통화 비율이 달라질 때의 손익분기 (2개 : 3개+ 비율은 dev 그대로)
        rest = (dk["2개"] * (nlab == 2).sum() + dk["3개+"] * (nlab >= 3).sum()) / (nlab >= 2).sum()
        be = -rest / (dk["1개"] - rest)
        r["break_even_single_frac"] = float(be)
        tr_mix = 0.670 * dk["1개"] + 0.244 * dk["2개"] + 0.086 * dk["3개+"]
        print(f"  증상 1개 통화 비율이 {be:.3f} 아래면 손해 (dev {np.mean(nlab == 1):.3f}, Training 0.670). Training 비율로 재가중한 차이 {tr_mix:+.4f}")
        # 겹별 부호
        fd = [float(d[my_fold == k].mean()) for k in range(5)]
        fd2 = [float(d[their_fold == k].mean()) for k in range(5)]
        r["fold_delta_mine"], r["fold_delta_theirs"] = fd, fd2
        print("  겹별 차이(내 분할): " + " ".join(f"{x:+.4f}" for x in fd) + "  | (기준 분할): " + " ".join(f"{x:+.4f}" for x in fd2))
        out[key] = r
        keep[key] = dict(raw=raw, cal=cal, pred_b=pred_b, vb=vb, pred_c=pred_c, vc=vc, pc_b=pc_b, pc_c=pc_c)

    # seed 흔들림 (4-1): 같은 설정 E002 vs E007
    a, b = keep["E002"], keep["E007"]
    sd_b = b["pc_b"] - a["pc_b"]; sd_c = b["pc_c"] - a["pc_c"]
    l, h, _, _ = boot_ci(sd_b)
    print(f"\n[seed] E007 - E002 (4-1): 기준 규칙 {sd_b.mean():+.4f} [{l:+.4f},{h:+.4f}], CM 규칙 {sd_c.mean():+.4f}")
    out["seed_gap_41"] = {"base": float(sd_b.mean()), "cm": float(sd_c.mean()), "lo": l, "hi": h}
    # CM 이득의 seed 간 차이 (E002 의 이득 - E007 의 이득)
    dd = (a["pc_c"] - a["pc_b"]) - (b["pc_c"] - b["pc_b"])
    l, h, _, _ = boot_ci(dd)
    print(f"[seed] CM 이득의 seed 간 차이 (E002 - E007): {dd.mean():+.4f} [{l:+.4f},{h:+.4f}]")

    # 내 기준 규칙이 채택 규칙(m4.decision.expected_rule)과 같은 예측을 내는지 대조
    sys.path.insert(0, str(ROOT))
    from m4 import decision as dec
    ens = keep["ENS(E001+E002+E007)"]
    p_ref, v_ref = dec.expected_rule(dec.calibrate(ens["raw"], shift))
    print(f"\n[대조] ENS 기준 규칙: 내 구현과 m4.decision.expected_rule 예측이 다른 순간 {int((p_ref != ens['pred_b']).any(axis=1).sum())}개, "
          f"기대 점수 최대 차이 {np.abs(v_ref - ens['vb']).max():.2e}")

    # ---------------------------------------------------------------- 4-2: CM 기대 점수를 중단 신호로
    print("\n===== 4-2: 중단 신호를 CM 기대 점수로 바꾸면 (중단 시점 예측은 보정 전 ≥0.5 그대로, τ 는 5겹)")
    taus = np.round(np.arange(0.36, 0.801, 0.02), 2)
    out["halt"] = {}
    for key in ("ENS(E001+E002+E007)", "E002", "E007", "E001"):
        kk = keep[key]
        pred42 = raw_rule(kk["raw"].copy())
        h_base = first_stop(kk["vb"] >= 0.52, off)
        s_base = score42(h_base, pred42, labels, off)
        row = {"base_tau0.52": s_base}
        txt = [f"{key}: 기준(τ=0.52 고정) {s_base:.4f}"]
        for sname, val in (("기존 신호", kk["vb"]), ("CM 신호(정확 해)", kk["vc"]), ("CM 신호(200회 반복)", keep[(key, "iter200")][1])):
            H = {t: first_stop(val >= t, off) for t in taus}
            for fname, fold in (("기준 분할", their_fold), ("내 분할", my_fold)):
                oh = np.zeros(N, dtype=np.int64); fs = []; picks = []
                for k in range(5):
                    tr, te = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
                    g = max(taus, key=lambda t: score42(H[t], pred42, labels, off, tr))
                    oh[te] = H[g][te]; fs.append(score42(H[g], pred42, labels, off, te)); picks.append(float(g))
                pooled = score42(oh, pred42, labels, off)
                lo, hi = boot42(oh, h_base, pred42, labels, off)
                row[f"{sname}|{fname}"] = {"fold_mean": float(np.mean(fs)), "oof_pooled": pooled, "delta_vs_base": pooled - s_base, "lo": lo, "hi": hi, "picks": picks}
                txt.append(f"   {sname:<18} {fname}: 겹평균 {np.mean(fs):.4f}, 겹밖 풀링 {pooled:.4f}, 기준 대비 {pooled - s_base:+.4f} [{lo:+.4f},{hi:+.4f}], τ {picks}")
        out["halt"][key] = row
        print("\n".join(txt))

    (OUT / "v01_cm_verify.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
