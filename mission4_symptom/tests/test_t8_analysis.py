"""저장 배열 사후 분석의 집계·Bootstrap을 작은 CPU 배열로 독립 검증한다."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_t8 import bootstrap, confusion, stop_metrics, sim


def test_confusion_and_pooled_macro_f1():
    y = np.array([[1, 0] * 4 + [1], [0, 1] * 4 + [0]], bool)
    p = np.ones_like(y)
    result = confusion(p, y)
    assert result["totals"] == {"tp": 9, "fp": 9, "fn": 0, "tn": 0}
    assert result["macro_f1"] == pytest.approx(2/3)
    assert result["mean_labels"] == 9


def test_stop_score_uses_pooled_f1_times_mean_earliness():
    y = np.array([[1] * 9, [0] * 9], bool)
    p = np.array([[1] * 9, [1] * 9], bool)
    result = stop_metrics(p, y, np.array([5, 10]), np.array([10, 10]))
    assert result["macro_f1"] == pytest.approx(2/3)
    assert result["earliness"] == pytest.approx(.625)
    assert result["score"] == pytest.approx((2/3)*.625)


def test_paired_bootstrap_matches_explicit_resampled_call_calculation():
    rng = np.random.default_rng(12)
    n = np.array([6, 9, 12, 15])
    off = np.r_[0, n.cumsum()]
    y = rng.random((4, 9)) > .5
    p0, p1 = rng.random((n.sum(), 9)) > .5, rng.random((n.sum(), 9)) > .5
    yy = np.repeat(y, n, axis=0)
    ms = [sim.moment_scores(p, yy) for p in [p0, p1]]
    cs = [np.add.reduceat(s, off[:-1]) / n for s in ms]
    hb, ht = np.array([5, 5, 6, 7]), np.array([6, 7, 9, 10])
    rel = np.concatenate([np.arange(1, k+1)/k for k in n])
    actual = bootstrap(y, p0, p1, hb, ht, n, off, (*cs, *ms), rel, seed=21, repeats=30)
    W = np.random.default_rng(21).multinomial(4, np.full(4, .25), size=30)
    naive, thirds = [], []
    for w in W:
        ix = np.repeat(np.arange(4), w)
        v = []
        for p, h in [(p0, hb), (p1, ht)]:
            pred = p[off[ix] + h[ix] - 1]
            v.append(sim.macro_f1(pred, y[ix])["macro_f1"] * sim.earliness(h[ix], n[ix]).mean())
        naive.append(v[1]-v[0])
        rows = np.concatenate([np.arange(off[i], off[i+1]) for i in ix])
        mask = rel[rows] <= 1/3
        thirds.append((ms[1][rows][mask]-ms[0][rows][mask]).mean())
    for key, values in [("4-2 Score", naive), ("초반 Score", thirds)]:
        assert actual[key]["delta"] == pytest.approx(np.mean(values))
        assert actual[key]["lo"] == pytest.approx(np.percentile(values, 2.5))
        assert actual[key]["hi"] == pytest.approx(np.percentile(values, 97.5))
