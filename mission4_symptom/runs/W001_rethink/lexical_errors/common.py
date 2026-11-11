"""W001 lexical_errors 공용 도구: 자료 읽기, 후보 채점, 5겹 교차검증 선택, 통화 단위 짝 부트스트랩.

기존 코드(m4/, scripts/)는 읽기만 한다. Validation 은 읽지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # mission4_symptom
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from halt_study import first_stop  # noqa: E402
from m4 import SYMPTOM_CLASSES  # noqa: E402
from m4 import decision as dec  # noqa: E402
from m4 import simulate as sim  # noqa: E402
from m4.data import load_cache, read_ids  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

C9 = SYMPTOM_CLASSES
TABLES = {"E001": "runs/E001_m3_baseline/dev_prefix.npz", "E002": "runs/E002_prefix_uniform/dev_prefix.npz",
          "E007": "runs/E007_prefix_uniform_s43/dev_prefix.npz", "E003": "runs/E003_prefix_plus_full_s43/dev_prefix.npz"}
TAUS = np.round(np.arange(0.40, 0.71, 0.02), 2)
FOLDS = 5


def load_table(key: str) -> sim.PrefixTable:
    return sim.PrefixTable.load(ROOT / TABLES[key])


def load_shift() -> np.ndarray:
    cfg = json.loads((ROOT / "runs/E002_prefix_uniform/run_config.json").read_text(encoding="utf-8"))
    return np.log(np.asarray(cfg["pos_weight"], dtype=np.float64))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dev_calls(table: sim.PrefixTable):
    by = {c.call_id: c for c in load_cache(ROOT / "runs/cache/train.jsonl", set(table.call_ids))}
    return [by[i] for i in table.call_ids]


def fit_calls():
    ids = read_ids(ROOT / "runs/cache/fit_ids.txt")
    by = {c.call_id: c for c in load_cache(ROOT / "runs/cache/train.jsonl", set(ids))}
    return [by[i] for i in ids]


def fold_of(n_calls: int) -> np.ndarray:
    """scripts/halt_study.py, eval_combo.py 와 같은 5겹 배정."""
    return np.random.default_rng(0).permutation(n_calls) % FOLDS


class Geometry:
    def __init__(self, table: sim.PrefixTable):
        self.table = table
        self.n = table.n_utt
        self.off = table.offsets
        self.M = int(self.off[-1])
        self.step = np.concatenate([np.arange(1, k + 1) for k in self.n])
        self.call_of = np.repeat(np.arange(len(table)), self.n)
        self.gold = np.repeat(table.labels, self.n, axis=0).astype(bool)
        self.labels = table.labels.astype(bool)
        self.fold = fold_of(len(table))


class Candidate:
    """보정 확률(cal: 실제 빈도에 맞춘 것)과 4-2 용 확률(raw: pos_weight 만큼 올라간 공간)로 채택 규칙을 적용한다."""

    def __init__(self, g: Geometry, cal: np.ndarray, raw: np.ndarray, name: str = ""):
        self.name = name
        self.g = g
        pred41, value = dec.expected_rule(cal)
        self.pred41 = pred41
        self.value = value
        ms = sim.moment_scores(pred41, g.gold)
        self.ms = ms
        self.sc41 = np.add.reduceat(ms, g.off[:-1]) / g.n                 # 통화별 4-1 점수
        self.pred42 = dec.threshold_rule(raw.copy())
        self.halts = {}
        self.predh = {}
        self.earl = {}
        for t in TAUS:
            h = first_stop(value >= t, g.off)
            self.halts[t] = h
            self.predh[t] = self.pred42[g.off[:-1] + h - 1]
            self.earl[t] = sim.earliness(h, g.n)

    def s41(self, idx=None) -> float:
        return float(self.sc41.mean() if idx is None else self.sc41[idx].mean())

    def s42(self, tau: float, idx=None) -> float:
        if idx is None:
            idx = np.arange(len(self.g.table))
        f1 = sim.macro_f1(self.predh[tau][idx], self.g.labels[idx])["macro_f1"]
        return f1 * float(self.earl[tau][idx].mean())

    def parts42(self, tau: float):
        f1 = sim.macro_f1(self.predh[tau], self.g.labels)
        return f1["macro_f1"], float(self.earl[tau].mean()), f1["per_class"]


def cand_from_tables(g: Geometry, keys, shift, name="") -> Candidate:
    raw = np.mean([load_table(k).probs for k in keys], axis=0)
    return Candidate(g, dec.calibrate(raw, shift), raw, name or "+".join(keys))


def f1x_e(pred: np.ndarray, gold: np.ndarray, earl: np.ndarray, weight=None) -> float:
    """macro-F1 x 평균 조기판단력. weight 는 통화별 중복 횟수(부트스트랩)."""
    w = np.ones(len(pred)) if weight is None else weight
    tp = (w[:, None] * (pred & gold)).sum(0)
    fp = (w[:, None] * (pred & ~gold)).sum(0)
    fn = (w[:, None] * (~pred & gold)).sum(0)
    seen = (tp + fp + fn) > 0
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
    return float(f1[seen].mean()) * float((w * earl).sum() / w.sum())


def cv_select(g: Geometry, cands: list):
    """후보 목록에서 4겹으로 고르고 남은 1겹에서 채점한다.

    4-1: 4겹 평균 점수가 가장 큰 후보. 4-2: (후보, τ) 쌍을 함께 고른다.
    돌려주는 값: 겹 평균 점수(기존 기록의 '교차검증'과 같은 정의), 겹 밖 예측을 이어 붙인 통화별 결과(부트스트랩용).
    """
    nc = len(g.table)
    oof41 = np.zeros(nc)
    oofp = np.zeros((nc, 9), dtype=bool)
    oofe = np.zeros(nc)
    f41, f42, pick41, pick42 = [], [], [], []
    for k in range(FOLDS):
        tr, te = np.nonzero(g.fold != k)[0], np.nonzero(g.fold == k)[0]
        b = max(range(len(cands)), key=lambda i: cands[i].s41(tr))
        oof41[te] = cands[b].sc41[te]
        f41.append(cands[b].s41(te))
        pick41.append(cands[b].name)
        best = max(((i, t) for i in range(len(cands)) for t in TAUS), key=lambda it: cands[it[0]].s42(it[1], tr))
        c, t = cands[best[0]], best[1]
        oofp[te] = c.predh[t][te]
        oofe[te] = c.earl[t][te]
        f42.append(c.s42(t, te))
        pick42.append((c.name, float(t)))
    return {"cv41": float(np.mean(f41)), "cv42": float(np.mean(f42)), "pick41": pick41, "pick42": pick42,
            "oof41": oof41, "oof_pred": oofp, "oof_earl": oofe,
            "pooled42": f1x_e(oofp, g.labels, oofe)}


def boot_diff(g: Geometry, a: dict, b: dict, n_boot: int = 1000, seed: int = 0):
    """통화 단위 짝 부트스트랩. a − b 의 95% 구간 (4-1, 4-2). a, b 는 cv_select 결과 또는 fixed_result 결과."""
    rng = np.random.default_rng(seed)
    nc = len(g.table)
    d41, d42 = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, nc, nc), minlength=nc).astype(np.float64)
        d41.append(float((w * (a["oof41"] - b["oof41"])).sum() / nc))
        d42.append(f1x_e(a["oof_pred"], g.labels, a["oof_earl"], w) - f1x_e(b["oof_pred"], g.labels, b["oof_earl"], w))
    d41, d42 = np.asarray(d41), np.asarray(d42)
    return {"d41": float((a["oof41"] - b["oof41"]).mean()), "ci41": [float(np.percentile(d41, 2.5)), float(np.percentile(d41, 97.5))],
            "d42": float(f1x_e(a["oof_pred"], g.labels, a["oof_earl"]) - f1x_e(b["oof_pred"], g.labels, b["oof_earl"])),
            "ci42": [float(np.percentile(d42, 2.5)), float(np.percentile(d42, 97.5))]}


def fixed_result(c: Candidate, tau: float = 0.52) -> dict:
    """값을 고르지 않는 후보(τ 고정)의 결과를 cv_select 와 같은 모양으로."""
    return {"cv41": c.s41(), "cv42": c.s42(tau), "oof41": c.sc41, "oof_pred": c.predh[tau], "oof_earl": c.earl[tau],
            "pooled42": c.s42(tau)}


def fmt_ci(ci) -> str:
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
