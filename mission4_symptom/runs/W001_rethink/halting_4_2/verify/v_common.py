"""반박 검증용 공용 코드 (검증자가 따로 쓴 구현).

대상 에이전트의 common.py / feats.py / exp*.py 를 불러 쓰지 않는다. 쓰는 것은 대회 기준 모듈(m4.simulate, m4.decision, m4.data)뿐이다.
- Validation 은 읽지 않는다 (runs/cache/train.jsonl 과 dev 확률표만).
- 모든 특징은 그 순간까지의 발화로만 만든다. 통화 길이 n 은 채점과 학습 목표(라벨)에만 쓴다.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]           # mission4_symptom
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from m4 import decision as dec  # noqa: E402
from m4 import simulate as sim  # noqa: E402
from m4.data import load_cache  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

TABLE_PATH = {
    "E001": "runs/E001_m3_baseline/dev_prefix.npz",
    "E002": "runs/E002_prefix_uniform/dev_prefix.npz",
    "E007": "runs/E007_prefix_uniform_s43/dev_prefix.npz",
    "E003": "runs/E003_prefix_plus_full_s43/dev_prefix.npz",
}
MEMBERS = {"E001": ["E001"], "E002": ["E002"], "E007": ["E007"], "E003": ["E003"],
           "ENS3": ["E001", "E002", "E007"], "ENS2": ["E002", "E007"]}
SHIFT = np.log(json.loads((ROOT / "runs/E002_prefix_uniform/run_config.json").read_text(encoding="utf-8"))["pos_weight"])
GRACE = 5


class Model:
    """확률표(하나 또는 평균)에서 순간별 배열을 만든다. 통화별 패딩 행렬도 함께 둔다."""

    def __init__(self, name: str):
        tabs = [sim.PrefixTable.load(ROOT / TABLE_PATH[k]) for k in MEMBERS[name]]
        t0 = tabs[0]
        for t in tabs[1:]:
            assert t.call_ids == t0.call_ids and (t.offsets == t0.offsets).all()
        self.name = name
        self.call_ids = t0.call_ids
        self.off = t0.offsets.astype(np.int64)
        self.n = np.diff(self.off)
        self.N = len(self.n)
        self.M = int(self.off[-1])
        self.labels = t0.labels.astype(bool)
        self.raw = np.mean([t.probs.astype(np.float64) for t in tabs], axis=0).astype(np.float32)
        self.cal = dec.calibrate(self.raw, SHIFT)
        _, self.v = dec.expected_rule(self.cal)
        # 중단 시점 예측: 보정 전 확률 >= 0.5, 비면 최고 1개 (직접 구현)
        p = self.raw >= 0.5
        empty = ~p.any(axis=1)
        p[empty, self.raw[empty].argmax(axis=1)] = True
        self.pred = p
        self.call_of = np.repeat(np.arange(self.N), self.n)
        self.step = np.arange(self.M) - self.off[:-1][self.call_of] + 1
        self.L = int(self.n.max())
        self.valid = self.pad(np.ones(self.M, dtype=bool), False)
        self.stepM = np.broadcast_to(np.arange(1, self.L + 1), (self.N, self.L))

    def pad(self, x: np.ndarray, fill) -> np.ndarray:
        m = np.full((self.N, self.L), fill, dtype=x.dtype if x.dtype != np.float32 else np.float64)
        m[self.call_of, self.step - 1] = x
        return m

    def halts(self, scoreM: np.ndarray, tau: float) -> np.ndarray:
        """5번째 이후 score >= tau 인 첫 step. 없으면 통화 길이(끝까지 = finalize)."""
        stop = (scoreM >= tau) & self.valid & (self.stepM >= GRACE)
        return np.where(stop.any(axis=1), stop.argmax(axis=1) + 1, self.n)

    def pred_at(self, h: np.ndarray) -> np.ndarray:
        return self.pred[self.off[:-1] + h - 1]


def earl(h, n):
    return sim.earliness(np.asarray(h), np.asarray(n))


def score42(md: Model, h: np.ndarray, idx=None) -> dict:
    """4-2 점수 = 풀링 Macro-F1 x 평균 조기판단력 (m4.simulate 의 함수 사용)."""
    p = md.pred_at(h)
    lab, n = md.labels, md.n
    if idx is not None:
        p, lab, n, h = p[idx], lab[idx], n[idx], h[idx]
    f1 = sim.macro_f1(p, lab)["macro_f1"]
    e = float(earl(h, n).mean())
    return {"score": f1 * e, "f1": f1, "earl": e}


class FastScorer:
    """후보 정책을 많이 채점할 때 쓰는 빠른 채점기 (통화 부분집합 가중치로 TP/FP/FN 합산). score42 와 값이 같은지 v01 에서 확인."""

    def __init__(self, md: Model):
        self.md = md

    def parts(self, h: np.ndarray):
        p = self.md.pred_at(h)
        lab = self.md.labels
        return (p & lab).astype(np.float64), (p & ~lab).astype(np.float64), (~p & lab).astype(np.float64), earl(h, self.md.n)

    @staticmethod
    def from_parts(tp, fp, fn, e, w):
        """w: (B, N) 또는 (N,) 가중치(통화 재표집 횟수 또는 0/1 마스크)."""
        w = np.atleast_2d(w).astype(np.float64)
        TP, FP, FN = w @ tp, w @ fp, w @ fn
        den = 2 * TP + FP + FN
        seen = den > 0
        f1 = np.where(seen, 2 * TP / np.maximum(den, 1e-12), 0.0)
        mf = f1.sum(axis=1) / np.maximum(seen.sum(axis=1), 1)
        return mf * (w @ e) / w.sum(axis=1)

    def score(self, h, mask=None) -> float:
        w = np.ones(self.md.N) if mask is None else mask.astype(np.float64)
        return float(self.from_parts(*self.parts(h), w)[0])


def boot_weights(N: int, B: int = 1000, seed: int = 2026) -> np.ndarray:
    return np.random.default_rng(seed).multinomial(N, np.full(N, 1.0 / N), size=B).astype(np.float64)


def paired_boot(md: Model, h_a: np.ndarray, h_b: np.ndarray, W: np.ndarray) -> dict:
    fs = FastScorer(md)
    a = fs.from_parts(*fs.parts(h_a), W)
    b = fs.from_parts(*fs.parts(h_b), W)
    d = a - b
    return {"diff": fs.score(h_a) - fs.score(h_b), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_le0": float((d <= 0).mean())}


# ----------------------------------------------------------------------------- 특징 (검증자 구현)
LEX = {
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
LEX_RE = {k: re.compile(v) for k, v in LEX.items()}
CONV_NAMES = (["sec", "chars", "caller_ratio", "last_dur", "mean_dur", "last_chars", "mean_chars", "char_rate", "last_spk",
               "turn_rate", "short_cum", "gap_mean"] + [f"{k}_cum" for k in LEX] + [f"{k}_last" for k in LEX]
              + ["since_addr", "since_pii", "since_dispatch", "caller_sym_cum", "disp_q_cum"])


def conv_feats_prefix(texts, speakers, starts, ends) -> np.ndarray:
    """발화를 하나씩 받아 가며(스트리밍) 대화 진행 특징 35개를 만든다. 뒤 발화는 볼 수 없는 구조로 썼다."""
    rows = []
    sec = 0.0
    chars = dur_sum = gap_sum = 0.0
    n_caller = n_turn = n_short = 0
    cum = {k: 0 for k in LEX}
    last_seen = {"addr": None, "pii": None, "dispatch": None}
    caller_sym = disp_q = 0
    prev_spk = None
    for t, (tx, sp, s_ms, e_ms) in enumerate(zip(texts, speakers, starts, ends), start=1):
        s, e = s_ms / 1000.0, e_ms / 1000.0
        if t > 1:
            gap_sum += max(s - sec, 0.0)
            n_turn += int(sp != prev_spk)
        sec = max(sec, e)
        ln = float(len(tx))
        d = max(e - s, 0.0)
        chars += ln
        dur_sum += d
        n_caller += int(sp == 1)
        n_short += int(ln <= 3)
        hit = {k: (1 if r.search(tx) else 0) for k, r in LEX_RE.items()}
        for k in LEX:
            cum[k] += hit[k]
        for k in last_seen:
            if hit[k]:
                last_seen[k] = t
        caller_sym += hit["symptom"] * int(sp == 1)
        disp_q += hit["question"] * int(sp == 0)
        rows.append([sec, chars, n_caller / t, d, dur_sum / t, ln, chars / t, chars / max(sec, 0.5), float(sp),
                     n_turn / t, n_short, gap_sum / t]
                    + [cum[k] for k in LEX] + [hit[k] for k in LEX]
                    + [(t - last_seen[k]) if last_seen[k] is not None else -1 for k in ("addr", "pii", "dispatch")]
                    + [caller_sym, disp_q])
        prev_spk = sp
    return np.asarray(rows, dtype=np.float64)


def conv_feats(calls) -> np.ndarray:
    return np.concatenate([conv_feats_prefix(c.texts, c.speakers, c.starts, c.ends) for c in calls])


def prob_feats(md: Model) -> np.ndarray:
    """확률 계열 22개. 통화마다 앞에서부터 차례로 계산한다."""
    out = np.zeros((md.M, 22))
    for i in range(md.N):
        a, b = md.off[i], md.off[i + 1]
        cal, v, pr, raw = md.cal[a:b], md.v[a:b], md.pred[a:b], md.raw[a:b]
        m = b - a
        prev_cal = np.vstack([cal[:1], cal[:-1]])
        prev_v = np.concatenate([v[:1], v[:-1]])
        run = np.zeros(m)
        for t in range(1, m):
            run[t] = run[t - 1] + 1 if (pr[t] == pr[t - 1]).all() else 0
        top3 = -np.sort(-cal, axis=1)[:, :3]
        d = cal - prev_cal
        out[a:b] = np.column_stack([cal, top3, v, v - prev_v, np.maximum.accumulate(v), run, cal.sum(axis=1), raw.max(axis=1),
                                    pr.sum(axis=1), np.arange(1, m + 1), d.max(axis=1), np.abs(d).sum(axis=1)])
    return out


def dev_calls(md: Model):
    by = {c.call_id: c for c in load_cache(ROOT / "runs/cache/train.jsonl", set(md.call_ids))}
    return [by[i] for i in md.call_ids]


def save_json(path: Path, obj) -> None:
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv), encoding="utf-8")
