"""모의 채점(m4.simulate)이 주최측 score.py 와 같은 값을 내는지, 중단 정책이 미래를 보지 않는지 확인한다."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4 import SYMPTOM_CLASSES  # noqa: E402
from m4 import simulate as sim  # noqa: E402

_spec = importlib.util.spec_from_file_location("official_score", HERE / "score.py")
official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(official)


def _names(row) -> list:
    return [c for c, v in zip(SYMPTOM_CLASSES, row) if v]


def _random_table(seed: int = 0, n_calls: int = 60) -> sim.PrefixTable:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 40, size=n_calls)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    probs = rng.random((offsets[-1], 9)).astype(np.float32) ** 2
    labels = (rng.random((n_calls, 9)) < 0.2).astype(np.int8)
    labels[labels.sum(axis=1) == 0, 0] = 1
    return sim.PrefixTable([f"c{i}" for i in range(n_calls)], offsets, probs, labels)


def test_4_1_matches_official():
    table = _random_table()
    for kwargs in ({}, {"force_top1": True}, {"force_top1": True, "max_labels": 2}):
        rule = lambda p, kw=kwargs: sim.decide(p, **kw)
        ours = sim.score_4_1(table, rule)["score"]
        pred, gold = {}, {}
        for i, cid in enumerate(table.call_ids):
            rows = rule(table.call(i).copy())
            pred[cid] = {t + 1: _names(r) for t, r in enumerate(rows)}
            gold[cid] = official.GoldSample(cid, _names(table.labels[i]), len(rows), [], [])
        theirs = official.score_mission_4_1(pred, gold)["score"]
        assert abs(ours - theirs) < 1e-12, (kwargs, ours, theirs)


def test_4_2_matches_official():
    table = _random_table(1)
    rule = lambda p: sim.decide(p, force_top1=True)
    for halt in (sim.halt_full, sim.halt_fixed(7), sim.halt_confidence(0.8), sim.halt_stable()):
        ours = sim.score_4_2(table, halt, rule)
        pred, gold = {}, {}
        for i, cid in enumerate(table.call_ids):
            p = table.call(i)
            h = halt(p) or len(p)
            pred[cid] = (h, _names(rule(p[h - 1:h])[0]))
            gold[cid] = official.GoldSample(cid, _names(table.labels[i]), len(p), [], [])
        theirs = official.score_mission_4_2(pred, gold)
        assert abs(ours["score"] - theirs["score"]) < 1e-12
        assert abs(ours["macro_f1"] - theirs["macro_f1"]) < 1e-12
        assert abs(ours["mean_earliness"] - theirs["mean_earliness"]) < 1e-12


def test_halting_does_not_look_ahead():
    """step h 에서 중단했다면, h 뒤의 확률을 바꿔도 같은 h 가 나와야 한다."""
    table = _random_table(2)
    rng = np.random.default_rng(3)
    for halt in (sim.halt_fixed(7), sim.halt_confidence(0.8), sim.halt_stable()):
        for i in range(len(table)):
            p = table.call(i).copy()
            h = halt(p)
            if h is None or h == len(p):
                continue
            q = p.copy()
            q[h:] = rng.random(q[h:].shape)
            assert halt(q) == h
