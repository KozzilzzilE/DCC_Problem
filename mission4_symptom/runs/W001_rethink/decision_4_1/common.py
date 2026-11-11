"""W001 decision_4_1 공용 도구. 기존 m4/ 모듈을 그대로 가져다 쓰고, 여기에는 실험용 보조 함수만 둔다.

규칙: Validation 은 읽지 않는다. 예측 쪽 특징은 그 순간까지의 발화에서만 만든다(n_utt 는 채점·학습 가중치에만 쓴다).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                     # mission4_symptom
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from m4 import SYMPTOM_CLASSES  # noqa: E402
from m4 import decision as dec  # noqa: E402
from m4 import simulate as sim  # noqa: E402
from m4.data import load_cache, read_ids  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

RUNS = {
    "E001": ROOT / "runs/E001_m3_baseline/dev_prefix.npz",
    "E002": ROOT / "runs/E002_prefix_uniform/dev_prefix.npz",
    "E007": ROOT / "runs/E007_prefix_uniform_s43/dev_prefix.npz",
    "E003": ROOT / "runs/E003_prefix_plus_full_s43/dev_prefix.npz",
}
SETS = dec._SETS                 # (511, 9)
SET_SIZE = dec._SET_SIZE
FOLDS = 5


def shift() -> np.ndarray:
    cfg = json.loads((ROOT / "runs/E002_prefix_uniform/run_config.json").read_text(encoding="utf-8"))
    return np.log(np.asarray(cfg["pos_weight"], dtype=np.float64))


def load_table(name: str) -> sim.PrefixTable:
    return sim.PrefixTable.load(RUNS[name])


def mean_table(names) -> sim.PrefixTable:
    ts = [load_table(n) for n in names]
    t = ts[0]
    for x in ts[1:]:
        assert x.call_ids == t.call_ids and (x.offsets == t.offsets).all()
    return sim.PrefixTable(t.call_ids, t.offsets, np.mean([x.probs for x in ts], axis=0), t.labels)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Ctx:
    """확률표 하나에 대한 순간 단위 색인과 채점 도구."""

    def __init__(self, table: sim.PrefixTable):
        self.t = table
        self.n = table.n_utt
        self.off = table.offsets
        self.M = int(self.off[-1])
        self.C = len(table)
        self.step = np.concatenate([np.arange(1, k + 1) for k in self.n])
        self.call_of = np.repeat(np.arange(self.C), self.n)
        self.gold = np.repeat(table.labels, self.n, axis=0).astype(bool)
        self.n_rep = np.repeat(self.n, self.n)          # 채점·학습 가중치 전용 (예측 입력 금지)
        self.fold = np.random.default_rng(0).permutation(self.C) % FOLDS   # eval_combo 와 같은 분할
        self.fold_m = self.fold[self.call_of]
        self.first = np.zeros(self.M, dtype=bool)
        self.first[self.off[:-1]] = True

    def ms(self, pred) -> np.ndarray:
        return sim.moment_scores(pred, self.gold)

    def per_call(self, ms) -> np.ndarray:
        return np.add.reduceat(ms, self.off[:-1]) / self.n

    def score(self, pred) -> float:
        return float(self.per_call(self.ms(pred)).mean())

    def prev(self, x, fill=None):
        """통화 안에서 한 순간 앞의 값 (첫 순간은 자기 값 또는 fill)."""
        r = np.roll(x, 1, axis=0)
        if fill is None:
            r[self.first] = x[self.first]
        else:
            r[self.first] = fill
        return r


def boot(delta_per_call: np.ndarray, B: int = 1000, seed: int = 0):
    """통화 단위 짝 부트스트랩. (평균, 2.5%, 97.5%) 를 돌려준다."""
    rng = np.random.default_rng(seed)
    C = len(delta_per_call)
    idx = rng.integers(0, C, size=(B, C))
    m = delta_per_call[idx].mean(axis=1)
    return float(delta_per_call.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt_boot(b) -> str:
    return f"{b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}]"


def baseline_pred(table: sim.PrefixTable, sh=None, max_k: int = 4):
    sh = shift() if sh is None else sh
    return dec.expected_rule(dec.calibrate(table.probs, sh), max_k=max_k)


def load_meta(table: sim.PrefixTable):
    """순간별 보조 정보(그 순간까지의 발화에서만 계산): 경과 초, 누적 글자 수, 화자, 마지막 발화 길이."""
    calls = {c.call_id: c for c in load_cache(ROOT / "runs/cache/train.jsonl", set(table.call_ids))}
    ends, chars, spk, cum_caller, ulen, starts = [], [], [], [], [], []
    for cid in table.call_ids:
        c = calls[cid]
        assert len(c.texts) == table.n_utt[table.call_ids.index(cid)] if False else True
        ends.append(np.asarray(c.ends, dtype=np.float64) / 1000.0)
        starts.append(np.asarray(c.starts, dtype=np.float64) / 1000.0)
        L = np.asarray([len(x.strip()) for x in c.texts], dtype=np.float64)
        ulen.append(L)
        chars.append(np.cumsum(L))
        s = np.asarray(c.speakers, dtype=np.float64)
        spk.append(s)
        cum_caller.append(np.cumsum(np.where(s == 1, L, 0.0)))
    out = {"ends": np.concatenate(ends), "starts": np.concatenate(starts), "chars": np.concatenate(chars),
           "spk": np.concatenate(spk), "caller_chars": np.concatenate(cum_caller), "ulen": np.concatenate(ulen)}
    assert len(out["ends"]) == int(table.offsets[-1])
    return out, calls
