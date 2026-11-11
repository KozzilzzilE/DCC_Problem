"""그 순간까지의 발화만으로 만드는 대화 진행 특징 (모델 확률을 쓰지 않는다).

모든 값은 누적/직전 값이라 뒤 발화와 통화 전체 길이에 의존하지 않는다.
"""
from __future__ import annotations

import re

import numpy as np

PAT = {
    "addr": re.compile(r"주소|어디|위치|몇 동|몇 호|아파트|빌라|번지|건물"),
    "pii": re.compile(r"\[개인정보\]"),
    "dispatch": re.compile(r"구급차|출동|보내|가볼게|갈게|갑니다|가고 있|도착|기다리"),
    "question": re.compile(r"\?"),
    "conscious": re.compile(r"의식|호흡|숨"),
    "age": re.compile(r"나이|연세|몇 살|살이|년생|남자|여자|할머니|할아버지"),
    "symptom": re.compile(r"아프|아파|통증|열이|열나|토하|토를|구토|어지|피가|피를|찢|숨이|숨을|힘이|기운|쓰러|메스|울렁|머리|배가"),
    "phone": re.compile(r"전화|번호|연락"),
    "bye": re.compile(r"수고|감사|알겠습니다|끊"),
}
NAMES = ["step", "sec", "chars", "caller_ratio", "last_dur", "mean_dur", "last_chars", "mean_chars", "char_rate", "last_spk",
         "turn_rate", "short_cum", "gap_mean"] + [f"{k}_cum" for k in PAT] + [f"{k}_last" for k in PAT] + \
        ["since_addr", "since_pii", "since_dispatch", "caller_sym_cum", "disp_q_cum"]


def call_feats(c) -> np.ndarray:
    n = len(c.texts)
    t = np.arange(1, n + 1, dtype=np.float64)
    st = np.asarray(c.starts, dtype=np.float64) / 1000.0
    en = np.asarray(c.ends, dtype=np.float64) / 1000.0
    spk = np.asarray(c.speakers)
    sec = np.maximum.accumulate(en)
    ln = np.array([len(x) for x in c.texts], dtype=np.float64)
    dur = np.maximum(en - st, 0.0)
    chars = np.cumsum(ln)
    turn = np.concatenate([[0], (spk[1:] != spk[:-1]).astype(float)])
    gap = np.concatenate([[0.0], np.maximum(st[1:] - sec[:-1], 0.0)])
    cols = [t, sec, chars, np.cumsum(spk == 1) / t, dur, np.cumsum(dur) / t, ln, chars / t, chars / np.maximum(sec, 0.5), spk.astype(float),
            np.cumsum(turn) / t, np.cumsum(ln <= 3), np.cumsum(gap) / t]
    hits = {k: np.array([1.0 if p.search(x) else 0.0 for x in c.texts]) for k, p in PAT.items()}
    cols += [np.cumsum(hits[k]) for k in PAT]
    cols += [hits[k] for k in PAT]
    for k in ("addr", "pii", "dispatch"):
        seen = np.maximum.accumulate(np.where(hits[k] > 0, np.arange(n), -1))
        cols.append(np.where(seen >= 0, np.arange(n) - seen, -1).astype(float))
    cols.append(np.cumsum(hits["symptom"] * (spk == 1)))
    cols.append(np.cumsum(hits["question"] * (spk == 0)))
    return np.column_stack(cols)


def build(calls) -> np.ndarray:
    return np.concatenate([call_feats(c) for c in calls])
