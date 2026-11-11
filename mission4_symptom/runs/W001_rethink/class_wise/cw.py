"""class_wise 관점 공통 도구. 기존 코드는 읽기만 한다 (m4/, scripts/ 수정 없음).

규칙
- Validation 은 읽지 않는다. dev 확률표(runs/E00x/dev_prefix.npz)와 Training 캐시만 쓴다.
- 예측·중단 판단에 쓰는 값은 그 순간까지의 확률과 발화 순번(step)뿐이다. n_utt 는 채점에만 쓴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from m4 import SYMPTOM_CLASSES  # noqa: E402
from m4 import decision as dec  # noqa: E402
from m4 import simulate as sim  # noqa: E402
from halt_study import first_stop  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

CLS = list(SYMPTOM_CLASSES)
RUNS = {"E001": "E001_m3_baseline", "E002": "E002_prefix_uniform", "E007": "E007_prefix_uniform_s43",
        "E003": "E003_prefix_plus_full_s43"}
FOLDS = 5
GRACE = 5
TAU0 = 0.52


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Ctx:
    """확률표 하나(또는 평균)와 채점에 필요한 색인."""

    def __init__(self, names=("E001", "E002", "E007")):
        tabs = [sim.PrefixTable.load(ROOT / "runs" / RUNS[n] / "dev_prefix.npz") for n in names]
        t = tabs[0]
        self.names = tuple(names)
        self.members = [x.probs.astype(np.float64) for x in tabs]
        self.probs = np.mean(self.members, axis=0)            # 보정 전 확률 (기존 방식: 확률 평균)
        self.table = sim.PrefixTable(t.call_ids, t.offsets, self.probs.astype(np.float32), t.labels)
        self.off = t.offsets
        self.n = t.n_utt
        self.N = len(t.call_ids)
        self.M = int(self.off[-1])
        self.step = np.concatenate([np.arange(1, k + 1) for k in self.n])
        self.call_of = np.repeat(np.arange(self.N), self.n)
        self.labels = t.labels.astype(bool)
        self.gold = np.repeat(self.labels, self.n, axis=0)
        self.fold = np.random.default_rng(0).permutation(self.N) % FOLDS     # halt_study / eval_combo 와 같은 분할
        self.mfold = self.fold[self.call_of]
        cfg = json.loads((ROOT / "runs" / RUNS["E002"] / "run_config.json").read_text(encoding="utf-8"))
        self.pos_weight = np.asarray(cfg["pos_weight"], dtype=np.float64)
        self.shift = np.log(self.pos_weight)
        self.cal = dec.calibrate(self.probs, self.shift)        # 기존 보정 (로짓 - log pos_weight)
        self._base = None

    # ------------------------------------------------------------------ 기준 (채택 규칙)
    def base(self):
        if self._base is None:
            pred41, value = dec.expected_rule(self.cal)
            pred42 = dec.threshold_rule(self.probs.copy())
            halts = first_stop(value >= TAU0, self.off)
            self._base = {"pred41": pred41, "value": value, "pred42": pred42, "halts": halts}
        return self._base

    # ------------------------------------------------------------------ 4-1
    def per_call_41(self, pred):
        ms = sim.moment_scores(pred, self.gold)
        return np.add.reduceat(ms, self.off[:-1]) / self.n

    # ------------------------------------------------------------------ 4-2
    def halt_pred(self, halts, pred_all):
        return pred_all[self.off[:-1] + halts - 1]

    def score42(self, halts, pred_all, calls=None):
        idx = np.arange(self.N) if calls is None else calls
        pred = self.halt_pred(halts, pred_all)[idx]
        f1 = sim.macro_f1(pred, self.labels[idx])
        e = float(sim.earliness(halts[idx], self.n[idx]).mean())
        return {"score": f1["macro_f1"] * e, "macro_f1": f1["macro_f1"], "earliness": e,
                "halt_median": float(np.median(halts[idx])), "per_class": f1["per_class"]}

    def score42_from(self, pred_h, halts, calls=None):
        """중단 시점 예측(통화 x 9)을 직접 받아 채점."""
        idx = np.arange(self.N) if calls is None else calls
        f1 = sim.macro_f1(pred_h[idx], self.labels[idx])
        e = float(sim.earliness(halts[idx], self.n[idx]).mean())
        return {"score": f1["macro_f1"] * e, "macro_f1": f1["macro_f1"], "earliness": e,
                "per_class": f1["per_class"]}

    def fold_mean_42(self, pred_h, halts):
        return float(np.mean([self.score42_from(pred_h, halts, np.nonzero(self.fold == k)[0])["score"]
                              for k in range(FOLDS)]))


# ---------------------------------------------------------------------- 부트스트랩 (통화 단위, 짝)
def boot_weights(N, B=1000, seed=20261111):
    rng = np.random.default_rng(seed)
    return rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(np.float64)       # (B, N)


def boot_41(per_call_a, per_call_b, W=None):
    """a - b 의 평균 차이와 95% 구간."""
    d = per_call_a - per_call_b
    if W is None:
        W = boot_weights(len(d))
    bs = W @ d / len(d)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _boot_score42(pred_h, labels, earl, W):
    tp = W @ (pred_h & labels).astype(np.float64)
    fp = W @ (pred_h & ~labels).astype(np.float64)
    fn = W @ (~pred_h & labels).astype(np.float64)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    return f1.mean(axis=1) * (W @ earl / W.shape[1]), f1.mean(axis=1), W @ earl / W.shape[1]


def boot_42(ctx: Ctx, pred_a, halts_a, pred_b, halts_b, W=None):
    """4-2 점수(풀링 Macro-F1 x 평균 조기판단력) 차이 a - b 의 점 추정과 95% 구간."""
    if W is None:
        W = boot_weights(ctx.N)
    ea, eb = sim.earliness(halts_a, ctx.n), sim.earliness(halts_b, ctx.n)
    sa, fa, _ = _boot_score42(pred_a.astype(bool), ctx.labels, ea, W)
    sb, fb, _ = _boot_score42(pred_b.astype(bool), ctx.labels, eb, W)
    point = ctx.score42_from(pred_a, halts_a)["score"] - ctx.score42_from(pred_b, halts_b)["score"]
    d = sa - sb
    df = fa - fb
    return {"delta": float(point), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "dF1_lo": float(np.percentile(df, 2.5)), "dF1_hi": float(np.percentile(df, 97.5))}


def ece(p, y, bins=15):
    """동일 폭 구간 ECE."""
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    cnt = np.bincount(b, minlength=bins).astype(float)
    sp = np.bincount(b, weights=p, minlength=bins)
    sy = np.bincount(b, weights=y.astype(float), minlength=bins)
    ok = cnt > 0
    return float((np.abs(sp[ok] - sy[ok])).sum() / len(p))


def save_json(name, obj):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv), encoding="utf-8")
