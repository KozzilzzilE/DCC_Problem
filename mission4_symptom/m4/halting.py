"""4-2 학습형 중단 정책: "기대 점수 + β·δ̂ ≥ τ" (5번째 발화 이후 첫 순간에 중단).

기존 정책은 "지금 답했을 때의 기대 순간 점수 v(t) 가 τ 이상이면 중단"이다. v(t) 는 지금 확률이 얼마나
확실한지만 보므로, "더 들어도 달라지지 않을 통화"(끝까지 가도 기준을 못 넘는 통화)를 끝까지 듣게 된다.
학습형 정책은 여기에 δ̂(t) 를 더한다.

    δ(t)  = (지금 t 에서 끊을 때의 효용) − (끊지 않고 기준 정책을 따라 다음에 끊게 될 시점의 효용)
    δ̂(t) = δ(t) 를 그 순간까지의 정보(특징 57개)로 맞히는 LightGBM 회귀의 예측
    중단  : step ≥ 5 이고  v(t) + β·δ̂(t) ≥ τ  인 첫 순간.  끝까지 없으면 마지막 발화에서 답한다.

δ̂ 가 양수면 "기다려 봐야 손해"라는 뜻이라 중단 쪽으로, 음수면 "더 들으면 나아진다"는 뜻이라 계속 듣는
쪽으로 민다. β = 0 이면 기존 정책과 같다.

효용(4-2 점수의 1차 근사)
    4-2 점수 S = F × E,  F = 전체 통화를 모아 계산한 Macro-F1,  E = 통화별 조기판단력의 평균.
    통화 하나의 중단 시점을 바꾸면 S 는 대략  E·ΔF + F·ΔE  만큼 변한다.
      ΔE : 그 통화의 조기판단력 변화 / N  (N = 통화 수)
      ΔF : 클래스 c 에서 놓친 정답 하나를 맞히면(FN→TP) F1_c 가 약 (2 − F1_c)/D_c 오르고,
           틀린 예측 하나를 더 내면(FP) 약 F1_c/D_c 내린다.  D_c = 2TP_c + FP_c + FN_c.
           Macro-F1 은 9개 클래스 평균이므로 9 로 나눈다.
    양변에 N 을 곱해 통화 하나의 효용으로 쓴다.
      u(t) = E · (N/9) · Σ_c pred_c(t) · [ q_c·(2 − F1_c) − (1 − q_c)·F1_c ] / D_c  +  F · 조기판단력(t)
    pred_c(t) 는 t 에서 끊을 때 낼 예측(보정 전 확률 ≥ 0.5, 없으면 1개)이다.
    q_c 는 "클래스 c 가 정답일 확률"인데, 라벨(0/1) 대신 통화 끝의 보정 확률을 쓴다. 라벨을 그대로 쓰면
    목표의 잡음이 커서 회귀가 거의 학습되지 않는다 (W001: R² 0.03~0.05 대 0.21~0.25).
    F, E, F1_c, D_c 는 학습용 통화에서 기준 정책을 돌렸을 때의 값(운영점 상수)이다.

미래 정보
    통화 전체 길이, 통화 끝 확률, 뒤 발화, 라벨은 δ(학습 목표)와 채점에만 쓴다. 특징에는 그 순간까지의
    발화(전사·화자·시각)와 그 순간까지의 모델 확률만 들어간다. 추정 통화 길이 특징은 넣지 않는다.
    특징은 두 방식으로 계산한다: `batch_features`(확률표 + 통화 원문, 학습·모의 채점용)와
    `HaltFeatureState`(발화를 하나씩 받는 추론용). 둘이 같은 값을 내는지는 tests/test_halting.py 에서 확인한다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import decision as dec
from . import simulate as sim

MIN_STEP = sim.EARLINESS_GRACE      # 5번째 발화 전에는 중단하지 않는다 (그 전에 끊어도 조기판단력이 같다)
N_CLASSES = dec.N_CLASSES

# ----------------------------------------------------------------------------- 특징 이름
# 대화가 어디까지 진행됐는지 알려 주는 낱말 묶음. 발화 전사에 하나라도 있으면 1.
LEXICON = {
    "addr": r"주소|어디|위치|몇 동|몇 호|아파트|빌라|번지|건물",
    "pii": r"\[개인정보\]",
    "dispatch": r"구급차|출동|보내|가볼게|갈게|갑니다|가고 있|도착|기다리",
    "question": r"\?",
    "conscious": r"의식|호흡|숨",
    "age": r"나이|연세|몇 살|살이|년생|남자|여자|할머니|할아버지",
    "symptom": r"아프|아파|통증|열이|열나|토하|토를|구토|어지|피가|피를|찢|숨이|숨을|힘이|기운|쓰러|메스|울렁|머리|배가",
    "phone": r"전화|번호|연락",
    "bye": r"수고|감사|알겠습니다|끊",
}
_LEX_RE = {k: re.compile(v) for k, v in LEXICON.items()}
_SINCE = ("addr", "pii", "dispatch")

# 확률 계열 22개
PROB_NAMES = ([f"cal_{k}" for k in range(N_CLASSES)]      # 보정 확률 9개 (증상 순서는 m4.SYMPTOM_CLASSES)
              + ["top1", "top2", "top3",                  # 보정 확률 상위 3개
                 "value",                                 # 지금 답했을 때의 기대 순간 점수 v(t)
                 "value_delta",                           # v(t) − v(t−1)
                 "value_max",                             # 지금까지 v 의 최대값
                 "same_pred_run",                         # 중단 시점 예측이 직전과 같았던 연속 횟수
                 "cal_sum",                               # 보정 확률 합 (예상 증상 수)
                 "raw_max",                               # 보정 전 확률의 최대값
                 "n_pred",                                # 지금 끊으면 낼 증상 수
                 "step",                                  # 지금까지 받은 발화 수
                 "cal_rise_max",                          # 직전 대비 가장 많이 오른 증상의 확률 변화
                 "cal_change_abs"])                       # 직전 대비 확률 변화의 절대값 합
# 대화 진행 35개 (전사·화자·시각만 사용, 모델 확률은 쓰지 않는다)
CONV_NAMES = (["sec",                                     # 지금 발화가 끝난 시각(초)
               "chars",                                   # 누적 글자 수
               "caller_ratio",                            # 신고자 발화 비율
               "last_dur", "mean_dur",                    # 이번 발화 길이(초), 평균 발화 길이
               "last_chars", "mean_chars",                # 이번 발화 글자 수, 평균 글자 수
               "char_rate",                               # 초당 글자 수
               "last_spk",                                # 이번 발화의 화자 (0 대원, 1 신고자)
               "turn_rate",                               # 화자가 바뀐 비율
               "short_cum",                               # 3글자 이하 발화 수 (네, 예 등)
               "gap_mean"]                                # 발화 사이 침묵의 평균(초)
              + [f"{k}_cum" for k in LEXICON]             # 낱말 묶음별 누적 등장 발화 수
              + [f"{k}_last" for k in LEXICON]            # 이번 발화에 등장했는지
              + [f"since_{k}" for k in _SINCE]            # 마지막 등장 뒤 지난 발화 수 (없었으면 −1)
              + ["caller_sym_cum",                        # 신고자가 증상 낱말을 말한 발화 수
                 "disp_q_cum"])                           # 대원이 질문한 발화 수
FEATURE_NAMES = PROB_NAMES + CONV_NAMES

# LightGBM 설정 (W001 검증에서 쓴 값 그대로. dev 로 조정하지 않는다)
GBM_PARAMS = {"objective": "regression", "learning_rate": 0.06, "num_leaves": 15, "min_data_in_leaf": 100,
              "bagging_fraction": 0.7, "bagging_freq": 1, "feature_fraction": 0.8, "lambda_l2": 5.0,
              "seed": 0, "deterministic": True, "force_row_wise": True, "verbose": -1}
GBM_ROUNDS = 200
# 2026-11-12: 한 겹 시험에서 beta 가 격자 끝(8)에서 골라져, 5겹 본실험 전에 16·32 와 tau 0.68~0.80 을 미리 추가했다.
# 값은 학습용 통화 안에서만 고른다 (판정용 통화는 선택에 쓰이지 않는다).
BETAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
TAUS = tuple(round(0.40 + 0.02 * i, 2) for i in range(21))          # 0.40 ~ 0.80


# ----------------------------------------------------------------------------- 순간별 신호
def moment_signals(raw: np.ndarray, logit_shift: Optional[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """보정 전 확률 (m, 9) -> (보정 확률, 기대 순간 점수 v, 중단 시점 예측 0/1).

    보정 확률은 학습 때 pos_weight 로 올라간 로짓을 되돌린 것이고, v 는 그 확률로 계산한다.
    중단 시점 예측은 채택 규칙 그대로다: 보정 전 확률 ≥ 0.5, 하나도 없으면 가장 높은 1개.
    """
    raw = np.asarray(raw, dtype=np.float32)
    cal = dec.calibrate(raw, logit_shift).astype(np.float64)
    value = dec.expected_rule(cal)[1]
    pred = dec.threshold_rule(raw.copy())
    return cal, value, pred


# ----------------------------------------------------------------------------- 특징: 일괄 계산
def _prob_features_call(raw: np.ndarray, cal: np.ndarray, value: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """한 통화의 확률 계열 특징 (n, 22). t 행은 0..t 행만으로 만든다."""
    n = len(raw)
    idx = np.arange(n)
    prev_cal = np.vstack([cal[:1], cal[:-1]])                  # 첫 순간의 "직전"은 자기 자신 (변화 0)
    prev_val = np.concatenate([value[:1], value[:-1]])
    changed = np.ones(n, dtype=bool)                           # 예측이 직전과 달라진 순간 (첫 순간 포함)
    changed[1:] = (pred[1:] != pred[:-1]).any(axis=1)
    run = idx - np.maximum.accumulate(np.where(changed, idx, 0))
    top3 = -np.sort(-cal, axis=1)[:, :3]
    diff = cal - prev_cal
    return np.column_stack([cal, top3, value, value - prev_val, np.maximum.accumulate(value), run, cal.sum(axis=1),
                            raw.max(axis=1).astype(np.float64), pred.sum(axis=1), idx + 1, diff.max(axis=1),
                            np.abs(diff).sum(axis=1)])


def conv_features(texts: Sequence[str], speakers: Sequence[int], starts: Sequence[int], ends: Sequence[int]) -> np.ndarray:
    """한 통화의 대화 진행 특징 (n, 35). 모두 누적값이거나 이번 발화의 값이라 뒤 발화에 의존하지 않는다."""
    n = len(texts)
    t = np.arange(1, n + 1, dtype=np.float64)
    st = np.asarray(starts, dtype=np.float64) / 1000.0
    en = np.asarray(ends, dtype=np.float64) / 1000.0
    spk = np.asarray(speakers)
    sec = np.maximum.accumulate(en)                            # 지금까지 가장 늦은 끝 시각
    length = np.array([len(x) for x in texts], dtype=np.float64)
    dur = np.maximum(en - st, 0.0)
    chars = np.cumsum(length)
    turn = np.concatenate([[0.0], (spk[1:] != spk[:-1]).astype(np.float64)])
    gap = np.concatenate([[0.0], np.maximum(st[1:] - sec[:-1], 0.0)])
    cols = [sec, chars, np.cumsum(spk == 1) / t, dur, np.cumsum(dur) / t, length, chars / t, chars / np.maximum(sec, 0.5),
            spk.astype(np.float64), np.cumsum(turn) / t, np.cumsum(length <= 3), np.cumsum(gap) / t]
    hits = {k: np.array([1.0 if r.search(x) else 0.0 for x in texts]) for k, r in _LEX_RE.items()}
    cols += [np.cumsum(hits[k]) for k in LEXICON]
    cols += [hits[k] for k in LEXICON]
    pos = np.arange(n)
    for k in _SINCE:
        last = np.maximum.accumulate(np.where(hits[k] > 0, pos, -1))
        cols.append(np.where(last >= 0, pos - last, -1).astype(np.float64))
    cols.append(np.cumsum(hits["symptom"] * (spk == 1)))
    cols.append(np.cumsum(hits["question"] * (spk == 0)))
    return np.column_stack(cols)


def batch_features(raw: np.ndarray, offsets: np.ndarray, calls: Sequence, logit_shift: Optional[Sequence[float]],
                   signals: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """확률표(raw, offsets)와 통화 원문(calls: m4.data.Call, 확률표와 같은 순서)으로 전 순간의 특징 (M, 57).

    통화 전체 길이는 "몇 행을 만들지"에만 쓰이고 값에는 들어가지 않는다 (통화를 잘라도 앞 행이 같다).
    """
    cal, value, pred = signals if signals is not None else moment_signals(raw, logit_shift)
    out = np.empty((len(raw), len(FEATURE_NAMES)), dtype=np.float64)
    for i, c in enumerate(calls):
        a, b = int(offsets[i]), int(offsets[i + 1])
        if b - a != len(c.texts):
            raise ValueError(f"확률표와 통화 원문의 발화 수가 다릅니다: {c.call_id} ({b - a} 대 {len(c.texts)})")
        out[a:b, :len(PROB_NAMES)] = _prob_features_call(raw[a:b], cal[a:b], value[a:b], pred[a:b])
        out[a:b, len(PROB_NAMES):] = conv_features(c.texts, c.speakers, c.starts, c.ends)
    return out


# ----------------------------------------------------------------------------- 특징: 온라인 계산
class HaltFeatureState:
    """발화를 하나씩 받으며 특징을 만든다 (추론용). 받은 것만 누적하므로 뒤 발화를 볼 방법이 없다.

        state = HaltFeatureState(logit_shift)
        state.reset()                                   # 새 통화
        x = state.update(text, speaker, start_ms, end_ms, raw_probs)     # (57,)
        state.value                                     # 이번 순간의 기대 점수 v(t)
    """

    def __init__(self, logit_shift: Optional[Sequence[float]] = None) -> None:
        self.logit_shift = logit_shift
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.value = 0.0
        self._prev_cal: Optional[np.ndarray] = None
        self._prev_value = 0.0
        self._prev_pred: Optional[np.ndarray] = None
        self._value_max = -np.inf
        self._run = 0
        self._sec = 0.0
        self._chars = self._dur_sum = self._gap_sum = 0.0
        self._n_caller = self._n_turn = self._n_short = 0
        self._cum = {k: 0 for k in LEXICON}
        self._last_seen: Dict[str, Optional[int]] = {k: None for k in _SINCE}
        self._caller_sym = self._disp_q = 0
        self._prev_spk: Optional[int] = None

    def update(self, text: str, speaker: int, start_ms: int, end_ms: int, raw_probs: np.ndarray) -> np.ndarray:
        self.step += 1
        t = self.step
        # ---- 확률 계열
        raw = np.asarray(raw_probs, dtype=np.float32)[None, :]
        cal2, value2, pred2 = moment_signals(raw, self.logit_shift)
        cal, value, pred = cal2[0], float(value2[0]), pred2[0]
        first = self._prev_cal is None
        prev_cal = cal if first else self._prev_cal
        prev_value = value if first else self._prev_value
        self._run = 0 if first or (pred != self._prev_pred).any() else self._run + 1
        self._value_max = max(self._value_max, value)
        diff = cal - prev_cal
        top3 = -np.sort(-cal)[:3]
        prob_part = [*cal, *top3, value, value - prev_value, self._value_max, self._run, cal.sum(), float(raw.max()),
                     int(pred.sum()), t, diff.max(), np.abs(diff).sum()]
        self._prev_cal, self._prev_value, self._prev_pred, self.value = cal, value, pred, value
        # ---- 대화 진행
        s, e = start_ms / 1000.0, end_ms / 1000.0
        if t > 1:
            self._gap_sum += max(s - self._sec, 0.0)
            self._n_turn += int(speaker != self._prev_spk)
        self._sec = max(self._sec, e) if t > 1 else e
        length = float(len(text))
        dur = max(e - s, 0.0)
        self._chars += length
        self._dur_sum += dur
        self._n_caller += int(speaker == 1)
        self._n_short += int(length <= 3)
        hit = {k: (1 if r.search(text) else 0) for k, r in _LEX_RE.items()}
        for k in LEXICON:
            self._cum[k] += hit[k]
        for k in _SINCE:
            if hit[k]:
                self._last_seen[k] = t
        self._caller_sym += hit["symptom"] * int(speaker == 1)
        self._disp_q += hit["question"] * int(speaker == 0)
        self._prev_spk = speaker
        conv_part = ([self._sec, self._chars, self._n_caller / t, dur, self._dur_sum / t, length, self._chars / t,
                      self._chars / max(self._sec, 0.5), float(speaker), self._n_turn / t, self._n_short, self._gap_sum / t]
                     + [self._cum[k] for k in LEXICON] + [hit[k] for k in LEXICON]
                     + [(t - self._last_seen[k]) if self._last_seen[k] is not None else -1 for k in _SINCE]
                     + [self._caller_sym, self._disp_q])
        return np.asarray(prob_part + conv_part, dtype=np.float64)


# ----------------------------------------------------------------------------- 자료 묶음
@dataclass
class HaltData:
    """확률표 하나(또는 평균)와 통화 원문에서 만든 순간별 배열. 순간은 통화 순서대로 이어 붙어 있다."""

    call_ids: List[str]
    offsets: np.ndarray        # (N + 1,)
    labels: np.ndarray         # (N, 9) bool   — 채점과 운영점 상수에만 쓴다
    cal: np.ndarray            # (M, 9) 보정 확률
    value: np.ndarray          # (M,)   기대 순간 점수 v(t)
    pred: np.ndarray           # (M, 9) bool, t 에서 끊을 때 낼 예측
    X: np.ndarray              # (M, 57) 특징

    @classmethod
    def build(cls, table: sim.PrefixTable, calls: Sequence, logit_shift: Optional[Sequence[float]]) -> "HaltData":
        if [c.call_id for c in calls] != list(table.call_ids):
            raise ValueError("통화 원문의 순서가 확률표와 다릅니다")
        raw = np.asarray(table.probs, dtype=np.float32)
        signals = moment_signals(raw, logit_shift)
        X = batch_features(raw, table.offsets, calls, logit_shift, signals)
        return cls(list(table.call_ids), np.asarray(table.offsets, dtype=np.int64), np.asarray(table.labels).astype(bool),
                   signals[0], signals[1], signals[2], X)

    def __len__(self) -> int:
        return len(self.call_ids)

    @property
    def n_utt(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def call_of(self) -> np.ndarray:
        """순간 -> 통화 번호"""
        return np.repeat(np.arange(len(self)), self.n_utt)

    @property
    def step(self) -> np.ndarray:
        """순간 -> 통화 안 발화 순번(1부터)"""
        return np.arange(int(self.offsets[-1])) - np.repeat(self.offsets[:-1], self.n_utt) + 1

    def subset(self, idx: np.ndarray) -> "HaltData":
        """통화 번호 idx 만 남긴 새 묶음 (통화 단위 분할용)."""
        idx = np.asarray(idx, dtype=np.int64)
        n = self.n_utt[idx]
        rows = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in idx]) if len(idx) else np.zeros(0, int)
        return HaltData([self.call_ids[i] for i in idx], np.concatenate([[0], np.cumsum(n)]).astype(np.int64),
                        self.labels[idx], self.cal[rows], self.value[rows], self.pred[rows], self.X[rows])


# ----------------------------------------------------------------------------- 중단 시점과 채점
def first_halt(stop: np.ndarray, data: HaltData, min_step: int = MIN_STEP) -> np.ndarray:
    """통화별 중단 step(1부터). min_step 이후 stop 이 처음 참인 순간이고, 없으면 통화 길이(끝까지 듣고 답함)."""
    step = data.step
    cand = np.where(stop & (step >= min_step), step, np.iinfo(np.int64).max)
    return np.minimum(np.minimum.reduceat(cand, data.offsets[:-1]), data.n_utt)


def score_halts(data: HaltData, halts: np.ndarray) -> Dict[str, float]:
    """4-2 점수 = (중단 시점 예측의 전체 풀링 Macro-F1) × (평균 조기판단력). m4.simulate 의 함수를 그대로 쓴다."""
    pred = data.pred[data.offsets[:-1] + halts - 1]
    n = data.n_utt
    f1 = sim.macro_f1(pred, data.labels)["macro_f1"]
    e = float(sim.earliness(halts, n).mean())
    return {"score": f1 * e, "macro_f1": f1, "earliness": e, "halt_median": float(np.median(halts)),
            "never_halted": float((halts == n).mean())}


def _score_parts(data: HaltData, halts: np.ndarray):
    """통화별 (TP, FP, FN, 조기판단력). 통화 가중치만 바꿔 가며 점수를 여러 번 낼 때 쓴다."""
    pred = data.pred[data.offsets[:-1] + halts - 1]
    lab = data.labels
    return ((pred & lab).astype(np.float64), (pred & ~lab).astype(np.float64), (~pred & lab).astype(np.float64),
            sim.earliness(halts, data.n_utt))


def _weighted_scores(parts, weights: np.ndarray) -> np.ndarray:
    """weights (B, N): 통화별 가중치(재표집 횟수 또는 0/1). 행마다 4-2 점수를 낸다. score_halts 와 같은 식이다."""
    tp, fp, fn, e = parts
    w = np.atleast_2d(weights).astype(np.float64)
    TP, FP, FN = w @ tp, w @ fp, w @ fn
    den = 2 * TP + FP + FN
    seen = den > 0                                              # 정답·예측 모두 없는 클래스는 평균에서 뺀다
    f1 = np.where(seen, 2 * TP / np.maximum(den, 1e-12), 0.0)
    macro = f1.sum(axis=1) / np.maximum(seen.sum(axis=1), 1)
    return macro * (w @ e) / w.sum(axis=1)


def paired_bootstrap(data: HaltData, halts_a: np.ndarray, halts_b: np.ndarray, n_boot: int = 1000, seed: int = 2026) -> Dict[str, float]:
    """점수 차이(a − b)와 통화 단위 짝 부트스트랩 95% 구간. 같은 재표집을 두 정책에 함께 적용한다."""
    N = len(data)
    W = np.random.default_rng(seed).multinomial(N, np.full(N, 1.0 / N), size=n_boot).astype(np.float64)
    d = _weighted_scores(_score_parts(data, halts_a), W) - _weighted_scores(_score_parts(data, halts_b), W)
    point = score_halts(data, halts_a)["score"] - score_halts(data, halts_b)["score"]
    return {"diff": float(point), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_gt0": float((d > 0).mean())}


# ----------------------------------------------------------------------------- 학습 목표 δ
def delta_target(data: HaltData, ref_tau: float, min_step: int = MIN_STEP) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """학습 목표 δ(t) = u(t) − u(기준 정책이 t 뒤에 끊는 순간) 와, 학습에 쓸 행 표시를 돌려준다.

    기준 정책은 "v ≥ ref_tau 인 첫 순간(5번째 이후)에 중단"이다. t 에서 끊지 않으면 그 뒤 기준 정책이
    처음 끊는 순간에 끊게 되고, 끝까지 없으면 마지막 발화에서 답한다.
    학습 행은 5번째 이후이면서 마지막 발화가 아닌 순간이다 (마지막 발화에서는 "더 듣기"가 없다).
    여기서 쓰는 통화 길이·통화 끝 확률·라벨은 목표에만 들어가고 특징과는 무관하다.
    """
    N, n, off, step = len(data), data.n_utt, data.offsets, data.step
    M = int(off[-1])
    stop0 = (data.value >= ref_tau) & (step >= min_step)
    # 운영점 상수: 기준 정책을 이 통화들에 돌렸을 때의 F, E, 클래스별 F1 과 분모
    h0 = first_halt(stop0, data, min_step)
    p0 = data.pred[off[:-1] + h0 - 1]
    tp = (p0 & data.labels).sum(axis=0).astype(np.float64)
    fp = (p0 & ~data.labels).sum(axis=0).astype(np.float64)
    fn = (~p0 & data.labels).sum(axis=0).astype(np.float64)
    D = np.maximum(2 * tp + fp + fn, 1.0)
    f1 = 2 * tp / D
    F = float(f1.mean())
    E = float(sim.earliness(h0, n).mean())
    # 효용 u(t): 모듈 설명의 식. q 는 통화 끝의 보정 확률
    q_end = np.repeat(data.cal[off[1:] - 1], n, axis=0)
    gain = (q_end * (2 - f1) - (1 - q_end) * f1) / D
    u = E * N / 9.0 * (data.pred * gain).sum(axis=1) + F * sim.earliness(step, np.repeat(n, n))
    # nxt[j]: 순간 j 뒤(> j)에 기준 정책이 끊는 순간. 없으면 그 통화의 마지막 발화
    nxt = np.full(M, -1, dtype=np.int64)
    for i in range(N):
        a, b = int(off[i]), int(off[i + 1])
        stops = np.nonzero(stop0[a:b])[0]
        local = np.full(b - a, b - a - 1)
        if len(stops):
            k = np.searchsorted(stops, np.arange(b - a), side="right")   # j 보다 뒤에 있는 첫 중단 후보의 위치
            local = np.where(k < len(stops), stops[np.minimum(k, len(stops) - 1)], b - a - 1)
        nxt[a:b] = a + local
    rows = (step >= min_step) & (step < np.repeat(n, n))
    delta = np.zeros(M)
    delta[rows] = u[rows] - u[nxt[rows]]
    consts = {"ref_tau": float(ref_tau), "macro_f1": F, "earliness": E, "n_calls": N, "n_rows": int(rows.sum()),
              "delta_mean": float(delta[rows].mean()) if rows.any() else 0.0,
              "delta_pos_rate": float((delta[rows] > 0).mean()) if rows.any() else 0.0}
    return delta, rows, consts


# ----------------------------------------------------------------------------- LightGBM
def fit_gbm(X: np.ndarray, y: np.ndarray, params: Optional[Dict[str, object]] = None, rounds: int = GBM_ROUNDS, threads: int = 4):
    """δ 회귀 모델(LightGBM Booster)을 학습한다."""
    import lightgbm as lgb

    p = {**GBM_PARAMS, **(params or {}), "num_threads": int(threads)}
    return lgb.train(p, lgb.Dataset(X, label=y, feature_name=list(FEATURE_NAMES), free_raw_data=False), num_boost_round=rounds)


def crossfit_delta(data: HaltData, delta: np.ndarray, rows: np.ndarray, folds: int = 5, seed: int = 0,
                   params: Optional[Dict[str, object]] = None, rounds: int = GBM_ROUNDS, threads: int = 4) -> np.ndarray:
    """통화 단위 교차 학습 예측 δ̂ (M,). 각 통화의 δ̂ 는 그 통화를 빼고 학습한 모델이 낸다.

    (β, τ) 를 고를 때 이 값을 쓴다. 자기 통화로 학습한 δ̂ 로 고르면 δ̂ 가 실제보다 잘 맞아 보여 β 가 크게 뽑힌다.
    """
    fold = np.random.default_rng(seed).permutation(len(data)) % folds
    row_fold = fold[data.call_of]
    out = np.zeros(len(delta))
    for k in range(folds):
        tr = rows & (row_fold != k)
        te = row_fold == k
        out[te] = fit_gbm(data.X[tr], delta[tr], params, rounds, threads).predict(data.X[te])
    return out


def select_beta_tau(data: HaltData, delta_hat: np.ndarray, betas: Sequence[float] = BETAS, taus: Sequence[float] = TAUS,
                    min_step: int = MIN_STEP) -> Dict[str, object]:
    """(β, τ) 격자에서 data 의 4-2 점수가 가장 높은 조합을 고른다. 동점이면 먼저 나온 것(β 작은 쪽).

    β = 0 행은 "τ 만 고르는 기존 정책"이므로, 그 행의 최고값을 따로 돌려준다 (학습형이 τ 재선택보다 나은지 보는 기준).
    """
    ones = np.ones((1, len(data)))
    best, tau_only = (-1.0, None), (-1.0, None)
    grid = []
    for b in betas:
        signal = data.value + b * delta_hat
        for t in taus:
            s = float(_weighted_scores(_score_parts(data, first_halt(signal >= t, data, min_step)), ones)[0])
            grid.append({"beta": float(b), "tau": float(t), "score": s})
            if s > best[0] + 1e-12:
                best = (s, (float(b), float(t)))
            if b == 0 and s > tau_only[0] + 1e-12:
                tau_only = (s, float(t))
    return {"beta": best[1][0], "tau": best[1][1], "score": best[0], "tau_only": tau_only[1], "tau_only_score": tau_only[0], "grid": grid}


# ----------------------------------------------------------------------------- 정책
@dataclass
class HaltPolicy:
    """학습된 중단 정책. v(t) + beta·δ̂(t) ≥ tau 인 첫 순간(min_step 이후)에 중단한다."""

    model_str: str                      # LightGBM 모델 (Booster.model_to_string)
    beta: float
    tau: float
    min_step: int = MIN_STEP
    logit_shift: Optional[List[float]] = None
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    n_fit_calls: int = 0
    info: Dict[str, object] = field(default_factory=dict)       # 학습 기록 (운영점 상수, 선택 과정 등)
    _booster: object = field(default=None, repr=False, compare=False)

    def booster(self):
        if self._booster is None:
            import lightgbm as lgb

            if list(self.feature_names) != list(FEATURE_NAMES):
                raise ValueError("저장된 정책의 특징 목록이 지금 코드(m4.halting.FEATURE_NAMES)와 다릅니다")
            self._booster = lgb.Booster(model_str=self.model_str)
        return self._booster

    def delta_hat(self, X: np.ndarray) -> np.ndarray:
        return self.booster().predict(np.atleast_2d(X), num_threads=1)

    def halts(self, data: HaltData) -> np.ndarray:
        """모의 채점용: 통화별 중단 step."""
        signal = data.value + (self.beta * self.delta_hat(data.X) if self.beta else 0.0)
        return first_halt(signal >= self.tau, data, self.min_step)

    def should_halt(self, state: HaltFeatureState, features: np.ndarray) -> bool:
        """추론용: 방금 state.update 가 돌려준 features 로 지금 중단할지 정한다."""
        if state.step < self.min_step:
            return False
        signal = state.value + (self.beta * float(self.delta_hat(features)[0]) if self.beta else 0.0)
        return bool(signal >= self.tau)

    def to_dict(self) -> Dict[str, object]:
        return {"type": "learned", "beta": float(self.beta), "tau": float(self.tau), "min_step": int(self.min_step),
                "logit_shift": None if self.logit_shift is None else [float(x) for x in self.logit_shift],
                "feature_names": list(self.feature_names), "n_fit_calls": int(self.n_fit_calls), "info": self.info,
                "model_str": self.model_str}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "HaltPolicy":
        return cls(model_str=d["model_str"], beta=float(d["beta"]), tau=float(d["tau"]), min_step=int(d.get("min_step", MIN_STEP)),
                   logit_shift=d.get("logit_shift"), feature_names=list(d["feature_names"]), n_fit_calls=int(d.get("n_fit_calls", 0)),
                   info=dict(d.get("info") or {}))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "HaltPolicy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def train_policy(data: HaltData, ref_tau: float = 0.52, logit_shift: Optional[Sequence[float]] = None, folds: int = 5,
                 seed: int = 0, betas: Sequence[float] = BETAS, taus: Sequence[float] = TAUS,
                 params: Optional[Dict[str, object]] = None, rounds: int = GBM_ROUNDS, threads: int = 4) -> HaltPolicy:
    """학습용 통화(data)만으로 정책 전체를 만든다. 다른 통화(판정용)는 어디에도 쓰이지 않는다.

    1. 기준 정책(v ≥ ref_tau)을 data 에 돌려 운영점 상수와 목표 δ 를 만든다.
    2. 통화 단위 교차 학습으로 δ̂ 를 만들고, 그 δ̂ 로 (β, τ) 를 고른다.
    3. data 전체로 δ 회귀 모델을 다시 학습한다. 이 모델과 2 의 (β, τ) 가 정책이다.
    """
    delta, rows, consts = delta_target(data, ref_tau)
    oof = crossfit_delta(data, delta, rows, folds, seed, params, rounds, threads)
    resid = delta[rows] - oof[rows]
    r2 = 1.0 - float((resid ** 2).sum()) / max(float(((delta[rows] - delta[rows].mean()) ** 2).sum()), 1e-12)
    pick = select_beta_tau(data, oof, betas, taus)
    booster = fit_gbm(data.X[rows], delta[rows], params, rounds, threads)
    ref = score_halts(data, first_halt(data.value >= ref_tau, data))
    info = {"target": consts, "crossfit_r2": r2, "folds": folds, "seed": seed, "gbm_params": {**GBM_PARAMS, **(params or {})},
            "gbm_rounds": rounds, "fit_score_ref_tau": ref["score"], "fit_score_tau_only": pick["tau_only_score"],
            "fit_tau_only": pick["tau_only"], "fit_score_selected": pick["score"], "grid": pick["grid"]}
    return HaltPolicy(model_str=booster.model_to_string(), beta=pick["beta"], tau=pick["tau"],
                      logit_shift=None if logit_shift is None else [float(x) for x in logit_shift],
                      n_fit_calls=len(data), info=info, _booster=booster)
