"""W001 4-2 중단 정책 재설계: 공용 도구.

- 확률표와 통화 원문(dev)만 읽는다. Validation 은 읽지 않는다.
- 채점은 m4.simulate / scripts.halt_study 와 같은 식이다 (exp01 에서 값이 같은지 확인).
- 교차검증: 통화 단위 5겹, 기존 스크립트와 같은 분할(rng 0 순열 % 5).
- 부트스트랩: 통화 단위 짝 재표집 1,000회.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # mission4_symptom
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from m4 import decision as dec  # noqa: E402
from m4 import simulate as sim  # noqa: E402
from m4.data import load_cache  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

GRACE = 5
FOLDS = 5
CLASSES = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
TABLES = {
    "E001": "runs/E001_m3_baseline/dev_prefix.npz",
    "E002": "runs/E002_prefix_uniform/dev_prefix.npz",
    "E007": "runs/E007_prefix_uniform_s43/dev_prefix.npz",
}
MODELS = {"E002": ["E002"], "E007": ["E007"], "ENS3": ["E001", "E002", "E007"]}
SHIFT = np.log(json.loads((ROOT / "runs/E002_prefix_uniform/run_config.json").read_text(encoding="utf-8"))["pos_weight"])


class Ctx:
    """모델(확률표 하나 또는 평균) 하나의 순간별 배열 묶음."""

    def __init__(self, name: str, probs: np.ndarray | None = None):
        ts = [sim.PrefixTable.load(ROOT / TABLES[k]) for k in MODELS[name]]
        t = ts[0]
        p = np.mean([x.probs for x in ts], axis=0) if probs is None else probs
        self.name = name
        self.table = sim.PrefixTable(t.call_ids, t.offsets, p.astype(np.float32), t.labels)
        self.members = [x.probs for x in ts]
        self.off = t.offsets
        self.n = t.n_utt
        self.N = len(t)
        self.M = int(t.offsets[-1])
        self.step = np.concatenate([np.arange(1, k + 1) for k in self.n])
        self.call_of = np.repeat(np.arange(self.N), self.n)
        self.labels = t.labels.astype(bool)
        self.gold = np.repeat(self.labels, self.n, axis=0)
        self.raw = self.table.probs.astype(np.float64)
        self.cal = dec.calibrate(self.table.probs, SHIFT)
        self.pred41, self.value = dec.expected_rule(self.cal)
        self.pred_thr = dec.threshold_rule(self.table.probs.copy())      # 채택 규칙: 보정 전 0.5, 비면 1개
        self.fold = np.random.default_rng(0).permutation(self.N) % FOLDS
        self.first = np.zeros(self.M, dtype=bool)
        self.first[self.off[:-1]] = True

    def prev(self, x: np.ndarray, fill=None) -> np.ndarray:
        """같은 통화의 직전 순간 값 (첫 순간은 자기 값 또는 fill)."""
        r = np.roll(x, 1, axis=0)
        r[self.first] = x[self.first] if fill is None else fill
        return r


def first_stop(stop: np.ndarray, off: np.ndarray, min_step: int = GRACE) -> np.ndarray:
    """통화별로 stop 이 처음 참인 step(1-based). min_step 이전은 무시, 없으면 통화 길이(finalize)."""
    n = np.diff(off)
    step = np.concatenate([np.arange(1, k + 1) for k in n])
    cand = np.where(stop & (step >= min_step), step, 10 ** 9)
    return np.minimum(np.minimum.reduceat(cand, off[:-1]), n)


def earliness(h: np.ndarray, n: np.ndarray) -> np.ndarray:
    return 0.5 + 0.5 * np.maximum(0.0, 1.0 - np.maximum(h, GRACE) / n)


def f1_counts(pred: np.ndarray, gold: np.ndarray, w: np.ndarray | None = None):
    pred = pred.astype(bool)
    gold = gold.astype(bool)
    if w is None:
        return (pred & gold).sum(0).astype(float), (pred & ~gold).sum(0).astype(float), (~pred & gold).sum(0).astype(float)
    return w @ (pred & gold), w @ (pred & ~gold), w @ (~pred & gold)


def macro_f1_from(tp, fp, fn):
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)
    seen = den > 0
    return (f1 * seen).sum(-1) / np.maximum(seen.sum(-1), 1), f1


def score_hp(h: np.ndarray, pred: np.ndarray, labels: np.ndarray, n: np.ndarray, idx=None) -> dict:
    """통화별 중단 시점 h 와 그때의 예측 pred(통화 x 9)로 4-2 점수."""
    if idx is not None:
        h, pred, labels, n = h[idx], pred[idx], labels[idx], n[idx]
    mf, f1 = macro_f1_from(*f1_counts(pred, labels))
    e = float(earliness(h, n).mean())
    return {"score": float(mf) * e, "macro_f1": float(mf), "earliness": e, "per_class": f1.tolist(),
            "halt_median": float(np.median(h)), "halt_mean": float(h.mean())}


def take(ctx: Ctx, h: np.ndarray, pred_all: np.ndarray) -> np.ndarray:
    """순간별 예측표에서 중단 시점의 행을 뽑는다."""
    return pred_all[ctx.off[:-1] + h - 1]


def score(ctx: Ctx, h: np.ndarray, pred_all: np.ndarray, idx=None) -> dict:
    return score_hp(h, take(ctx, h, pred_all), ctx.labels, ctx.n, idx)


def baseline(ctx: Ctx, tau: float = 0.52):
    h = first_stop(ctx.value >= tau, ctx.off)
    return h, take(ctx, h, ctx.pred_thr)


def cv_select(ctx: Ctx, candidates: dict, label: str = ""):
    """candidates: {키: (h, pred(통화 x 9))}. 4겹에서 점수가 가장 높은 키를 골라 남은 1겹에 적용한다.

    돌려주는 것: dict(dev 전체 최고 키·점수, 겹 평균 교차검증 점수, 겹별 고른 키, OOF h·pred)
    """
    keys = list(candidates)
    H = np.stack([candidates[k][0] for k in keys])                 # (K, N)
    P = np.stack([candidates[k][1] for k in keys]).astype(bool)    # (K, N, 9)
    lab = ctx.labels
    E = earliness(H, ctx.n[None, :])                               # (K, N)
    tpi, fpi, fni = P & lab[None], P & ~lab[None], ~P & lab[None]

    def scores(mask):
        mf, _ = macro_f1_from(tpi[:, mask].sum(1), fpi[:, mask].sum(1), fni[:, mask].sum(1))
        return mf * E[:, mask].mean(1)

    full = scores(np.ones(ctx.N, dtype=bool))
    best = int(np.argmax(full))
    oof_h = np.empty(ctx.N, dtype=np.int64)
    oof_p = np.zeros((ctx.N, 9), dtype=bool)
    cvs, picks = [], []
    for k in range(FOLDS):
        tr, te = ctx.fold != k, ctx.fold == k
        j = int(np.argmax(scores(tr)))
        cvs.append(float(scores(te)[j]))
        picks.append(keys[j])
        oof_h[te], oof_p[te] = H[j, te], P[j, te]
    return {"label": label, "best_key": keys[best], "dev_all": float(full[best]), "cv": float(np.mean(cvs)),
            "cv_folds": cvs, "picks": picks, "oof_h": oof_h, "oof_p": oof_p,
            "best_h": H[best], "best_p": P[best]}


def fold_mean(ctx: Ctx, h: np.ndarray, pred: np.ndarray) -> float:
    """고정 정책의 겹 평균 점수 (교차검증 점수와 같은 잣대로 비교하기 위함)."""
    return float(np.mean([score_hp(h, pred, ctx.labels, ctx.n, np.nonzero(ctx.fold == k)[0])["score"] for k in range(FOLDS)]))


_BOOT_W = {}


def boot_weights(N: int, B: int = 1000, seed: int = 0) -> np.ndarray:
    if (N, B, seed) not in _BOOT_W:
        rng = np.random.default_rng(seed)
        _BOOT_W[(N, B, seed)] = rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(np.float64)
    return _BOOT_W[(N, B, seed)]


def boot_scores(ctx: Ctx, h: np.ndarray, pred: np.ndarray, B: int = 1000) -> np.ndarray:
    W = boot_weights(ctx.N, B)
    lab = ctx.labels
    pred = pred.astype(bool)
    mf, _ = macro_f1_from(W @ (pred & lab), W @ (pred & ~lab), W @ (~pred & lab))
    e = (W @ earliness(h, ctx.n)) / ctx.N
    return mf * e


def paired_boot(ctx: Ctx, cand, base, B: int = 1000) -> dict:
    """(h, pred) 두 쌍의 점수 차이(cand - base)와 통화 단위 짝 부트스트랩 95% 구간."""
    d = boot_scores(ctx, *cand, B=B) - boot_scores(ctx, *base, B=B)
    point = score_hp(cand[0], cand[1], ctx.labels, ctx.n)["score"] - score_hp(base[0], base[1], ctx.labels, ctx.n)["score"]
    return {"diff": float(point), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_gt0": float((d > 0).mean())}


def report(ctx: Ctx, res: dict, base, base_fm: float) -> dict:
    """교차검증 결과를 기준(같은 모델, τ=0.52 고정)과 비교한 한 줄 요약."""
    oof = (res["oof_h"], res["oof_p"])
    s_oof = score_hp(*oof, ctx.labels, ctx.n)
    pb = paired_boot(ctx, oof, base)
    row = {"model": ctx.name, "method": res["label"], "best_key": str(res["best_key"]), "dev_all": res["dev_all"],
           "cv": res["cv"], "cv_minus_base": res["cv"] - base_fm, "oof_pooled": s_oof["score"],
           "oof_f1": s_oof["macro_f1"], "oof_earl": s_oof["earliness"], "oof_halt_median": s_oof["halt_median"],
           "boot_diff": pb["diff"], "boot_lo": pb["lo"], "boot_hi": pb["hi"], "boot_p_gt0": pb["p_gt0"],
           "picks": [str(x) for x in res["picks"]]}
    print(f"  [{ctx.name}] {res['label']:<44} dev {row['dev_all']:.4f}  교차검증 {row['cv']:.4f} ({row['cv_minus_base']:+.4f})  "
          f"OOF {row['oof_pooled']:.4f} = {row['oof_f1']:.4f} x {row['oof_earl']:.4f}  "
          f"차이 {pb['diff']:+.4f} [{pb['lo']:+.4f}, {pb['hi']:+.4f}]  고른 값 {row['picks']}")
    return row


def load_dev_calls(ctx: Ctx):
    calls = {c.call_id: c for c in load_cache(ROOT / "runs/cache/train.jsonl", set(ctx.table.call_ids))}
    return [calls[i] for i in ctx.table.call_ids]


def save_json(name: str, obj) -> None:
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv), encoding="utf-8")


def cv_fit(ctx: Ctx, label: str, fit, apply, base, base_fm: float) -> dict:
    """fit(통화 mask) -> 파라미터, apply(파라미터) -> (h, pred(통화 x 9)). 5겹 교차검증 + 부트스트랩 요약."""
    allm = np.ones(ctx.N, dtype=bool)
    p_all = fit(allm)
    h, pr = apply(p_all)
    dev_all = score_hp(h, pr, ctx.labels, ctx.n)["score"]
    oof_h = np.empty(ctx.N, dtype=np.int64)
    oof_p = np.zeros((ctx.N, 9), dtype=bool)
    cvs, picks = [], []
    for k in range(FOLDS):
        prm = fit(ctx.fold != k)
        hh, pp = apply(prm)
        te = np.nonzero(ctx.fold == k)[0]
        oof_h[te], oof_p[te] = hh[te], pp[te]
        cvs.append(score_hp(hh, pp, ctx.labels, ctx.n, te)["score"])
        picks.append(prm)
    res = {"label": label, "best_key": p_all, "dev_all": dev_all, "cv": float(np.mean(cvs)), "cv_folds": cvs,
           "picks": picks, "oof_h": oof_h, "oof_p": oof_p}
    return report(ctx, res, base, base_fm)
