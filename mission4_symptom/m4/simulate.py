"""누적 입력 확률표(통화 x 순간 x 9종)로 4-1 결정 규칙과 4-2 중단 정책을 모의 채점한다.

채점식은 score.py(주최측 공개본)를 그대로 따른다. 벡터 연산으로 다시 쓴 부분은
tests/test_simulate.py 에서 score.py 의 함수와 값이 같은지 확인한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import SYMPTOM_CLASSES

EARLINESS_GRACE = 5
EARLINESS_FLOOR = 0.5


@dataclass
class PrefixTable:
    """통화별 누적 확률. probs[offsets[i]:offsets[i+1]] 가 i번째 통화의 step 1..n 확률이다."""

    call_ids: List[str]
    offsets: np.ndarray          # (통화 수 + 1,)
    probs: np.ndarray            # (전체 순간 수, 9) float32
    labels: np.ndarray           # (통화 수, 9) 0/1

    def __len__(self) -> int:
        return len(self.call_ids)

    def call(self, i: int) -> np.ndarray:
        return self.probs[self.offsets[i]:self.offsets[i + 1]]

    @property
    def n_utt(self) -> np.ndarray:
        return np.diff(self.offsets)

    def save(self, path) -> None:
        np.savez_compressed(path, call_ids=np.asarray(self.call_ids), offsets=self.offsets,
                            probs=self.probs.astype(np.float32), labels=self.labels)

    @classmethod
    def load(cls, path) -> "PrefixTable":
        z = np.load(path, allow_pickle=False)
        return cls([str(x) for x in z["call_ids"]], z["offsets"], z["probs"], z["labels"])


# ----------------------------------------------------------------------------- 결정 규칙
def decide(probs: np.ndarray, threshold: float = 0.5, force_top1: bool = False,
           max_labels: Optional[int] = None) -> np.ndarray:
    """확률 (m, 9) -> 예측 0/1 (m, 9).

    force_top1 : 임계값을 넘는 증상이 없으면 확률이 가장 높은 1개를 낸다 (빈 예측은 4-1 에서 0점).
    max_labels : 임계값을 넘는 증상이 많아도 확률 상위 k개만 남긴다.
    """
    pred = probs >= threshold
    if max_labels is not None:
        rank = np.argsort(-probs, axis=1).argsort(axis=1)
        pred &= rank < max_labels
    if force_top1:
        empty = ~pred.any(axis=1)
        pred[empty, probs[empty].argmax(axis=1)] = True
    return pred


# ----------------------------------------------------------------------------- 4-1
def moment_scores(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """순간 점수 = 맞힌 클래스 수 / (정답 또는 예측에 있는 클래스 수). score.moment_score 와 같다."""
    pred = pred.astype(bool)
    gold = gold.astype(bool)
    inter = (pred & gold).sum(axis=1)
    union = (pred | gold).sum(axis=1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def score_4_1(table: PrefixTable, rule: Callable[[np.ndarray], np.ndarray]) -> Dict[str, object]:
    """4-1 점수: 순간 점수 -> 통화 평균 -> 전체 평균. 통화 진행 구간별 점수도 함께 낸다."""
    pred = rule(table.probs)
    n = table.n_utt
    gold = np.repeat(table.labels, n, axis=0)
    ms = moment_scores(pred, gold)
    per_call = np.add.reduceat(ms, table.offsets[:-1]) / n
    # 통화 안 상대 위치(앞 1/3, 가운데, 뒤 1/3)별 평균: 어디서 점수를 잃는지 본다
    step = np.concatenate([np.arange(1, k + 1) for k in n])
    rel = step / np.repeat(n, n)
    thirds = {name: float(ms[(rel > lo) & (rel <= hi)].mean())
              for name, lo, hi in (("앞", 0, 1 / 3), ("가운데", 1 / 3, 2 / 3), ("뒤", 2 / 3, 1))}
    by_step = {int(t): float(ms[step == t].mean()) for t in (1, 2, 3, 5, 8, 10, 15, 20, 30)}
    return {"score": float(per_call.mean()), "empty_rate": float((~pred.any(axis=1)).mean()),
            "mean_labels": float(pred.sum(axis=1).mean()), "by_third": thirds, "by_step": by_step}


# ----------------------------------------------------------------------------- 4-2
def earliness(halt: np.ndarray, n_utt: np.ndarray) -> np.ndarray:
    r = 1.0 - np.maximum(halt, EARLINESS_GRACE) / n_utt
    return EARLINESS_FLOOR + (1.0 - EARLINESS_FLOOR) * np.maximum(0.0, r)


def macro_f1(pred: np.ndarray, gold: np.ndarray) -> Dict[str, object]:
    """전체를 모아 클래스별 F1 을 구하고 평균한다. 정답·예측 모두 없는 클래스는 뺀다 (score.macro_f1)."""
    pred = pred.astype(bool)
    gold = gold.astype(bool)
    tp = (pred & gold).sum(axis=0).astype(float)
    fp = (pred & ~gold).sum(axis=0).astype(float)
    fn = (~pred & gold).sum(axis=0).astype(float)
    seen = (tp + fp + fn) > 0
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return {"macro_f1": float(f1[seen].mean()) if seen.any() else 0.0,
            "per_class": {c: (float(f1[k]) if seen[k] else None) for k, c in enumerate(SYMPTOM_CLASSES)}}


def score_4_2(table: PrefixTable, halt_fn: Callable[[np.ndarray], Optional[int]],
              rule: Callable[[np.ndarray], np.ndarray]) -> Dict[str, object]:
    """4-2 점수: 중단 시점 예측의 Macro-F1 x 평균 조기판단력.

    halt_fn 은 한 통화의 확률 (n, 9) 를 받아 중단할 step(1-based)을 돌려준다. None 이면 끝까지 듣는다.
    정책은 step t 를 정할 때 t 이전 행만 보도록 작성해야 한다 (tests 에서 확인).
    """
    halts = np.empty(len(table), dtype=np.int64)
    preds = np.zeros((len(table), len(SYMPTOM_CLASSES)), dtype=bool)
    for i in range(len(table)):
        p = table.call(i)
        h = halt_fn(p)
        h = len(p) if h is None else int(h)
        halts[i] = h
        preds[i] = rule(p[h - 1:h])[0]
    n = table.n_utt
    f1 = macro_f1(preds, table.labels)
    e = earliness(halts, n)
    return {"score": f1["macro_f1"] * float(e.mean()), "macro_f1": f1["macro_f1"],
            "mean_earliness": float(e.mean()), "per_class": f1["per_class"],
            "halt_median": float(np.median(halts)), "halt_mean": float(halts.mean()),
            "halt_frac_median": float(np.median(halts / n)),
            "never_halted": float((halts == n).mean()), "empty_rate": float((~preds.any(axis=1)).mean())}


# ----------------------------------------------------------------------------- 중단 정책
def halt_full(_p: np.ndarray) -> Optional[int]:
    return None


def halt_fixed(k: int) -> Callable[[np.ndarray], Optional[int]]:
    return lambda p: k if len(p) >= k else None


def halt_confidence(tau: float, min_step: int = EARLINESS_GRACE) -> Callable[[np.ndarray], Optional[int]]:
    """min_step 이후 최대 확률이 tau 이상인 첫 순간에 중단한다."""
    def fn(p: np.ndarray) -> Optional[int]:
        hit = np.nonzero(p[min_step - 1:].max(axis=1) >= tau)[0]
        return int(hit[0]) + min_step if len(hit) else None
    return fn


def halt_stable(threshold: float = 0.5, patience: int = 2,
                min_step: int = EARLINESS_GRACE) -> Callable[[np.ndarray], Optional[int]]:
    """비어 있지 않은 같은 예측이 patience 번 연속 유지되면 중단한다 (팀원 브랜치의 규칙)."""
    def fn(p: np.ndarray) -> Optional[int]:
        pred = p >= threshold
        run = 0
        for t in range(len(p)):
            if pred[t].any() and t > 0 and (pred[t] == pred[t - 1]).all():
                run += 1
            elif pred[t].any():
                run = 1
            else:
                run = 0
            if t + 1 >= min_step and run >= patience:
                return t + 1
        return None
    return fn
