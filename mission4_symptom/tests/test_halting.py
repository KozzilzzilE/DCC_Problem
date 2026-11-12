"""학습형 중단 정책(m4.halting): 특징이 미래를 보지 않는지, 일괄 계산과 온라인 계산이 같은지, 정책이 규칙을 지키는지 확인한다.

모델(인코더)을 불러오지 않는다. 확률표와 통화 원문은 난수로 만든다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4 import halting as hl  # noqa: E402
from m4 import simulate as sim  # noqa: E402
from m4.data import Call  # noqa: E402

WORDS = ["네", "예", "주소가 어디세요?", "[개인정보] 아파트요", "구급차 보내드릴게요", "배가 아파요", "숨을 못 쉬어요",
         "연세가 어떻게 되세요?", "전화번호 알려주세요", "의식 있으세요?", "머리가 어지러워요", "알겠습니다 감사합니다", "", "토했어요"]
SHIFT = np.log(np.linspace(1.8, 3.0, 9))


def _make(seed: int = 0, n_calls: int = 40, min_len: int = 1, max_len: int = 30):
    """난수 통화와 확률표. 시각은 겹치거나 뒤로 가는 경우도 섞는다."""
    rng = np.random.default_rng(seed)
    calls, probs = [], []
    for i in range(n_calls):
        n = int(rng.integers(min_len, max_len + 1))
        starts = np.cumsum(rng.integers(0, 4000, size=n))
        starts[rng.random(n) < 0.1] -= 1500                      # 일부는 앞 발화와 겹치게
        starts = np.maximum(starts, 0)
        ends = starts + rng.integers(0, 6000, size=n)
        labels = (rng.random(9) < 0.2).astype(int)
        labels[0] = max(labels[0], int(labels.sum() == 0))
        calls.append(Call(f"c{i}", [WORDS[k] for k in rng.integers(0, len(WORDS), size=n)],
                          [int(x) for x in rng.integers(0, 2, size=n)], [int(x) for x in starts], [int(x) for x in ends],
                          [int(x) for x in labels]))
        probs.append((rng.random((n, 9)) ** 2).astype(np.float32))
    offsets = np.concatenate([[0], np.cumsum([c.n_utt for c in calls])]).astype(np.int64)
    table = sim.PrefixTable([c.call_id for c in calls], offsets, np.concatenate(probs), np.asarray([c.labels for c in calls], dtype=np.int8))
    return table, calls


def _online(call: Call, probs: np.ndarray) -> np.ndarray:
    state = hl.HaltFeatureState(SHIFT)
    return np.stack([state.update(call.texts[t], call.speakers[t], call.starts[t], call.ends[t], probs[t])
                     for t in range(call.n_utt)])


def test_feature_names():
    assert len(hl.PROB_NAMES) == 22 and len(hl.CONV_NAMES) == 35
    assert len(set(hl.FEATURE_NAMES)) == 57
    assert not any("n_utt" in name or "nhat" in name or "len" in name for name in hl.FEATURE_NAMES)     # 통화 길이 계열 없음


def test_batch_equals_online():
    table, calls = _make(1)
    batch = hl.batch_features(table.probs, table.offsets, calls, SHIFT)
    assert batch.shape == (int(table.offsets[-1]), 57) and np.isfinite(batch).all()
    for i, call in enumerate(calls):
        online = _online(call, table.call(i))
        np.testing.assert_allclose(online, batch[table.offsets[i]:table.offsets[i + 1]], rtol=0, atol=1e-9)


def test_online_state_resets():
    """reset 뒤에는 앞 통화의 흔적이 남지 않는다 (통화 순서를 바꿔도 결과가 같다)."""
    table, calls = _make(2, n_calls=3, min_len=8)
    state = hl.HaltFeatureState(SHIFT)
    for t in range(calls[0].n_utt):
        state.update(calls[0].texts[t], calls[0].speakers[t], calls[0].starts[t], calls[0].ends[t], table.call(0)[t])
    state.reset()
    again = np.stack([state.update(calls[1].texts[t], calls[1].speakers[t], calls[1].starts[t], calls[1].ends[t], table.call(1)[t])
                      for t in range(calls[1].n_utt)])
    np.testing.assert_array_equal(again, _online(calls[1], table.call(1)))


def test_features_do_not_see_future():
    """k 번째 발화 뒤의 전사·화자·시각·확률을 바꾸거나 통화를 k 에서 잘라도, 앞 k 개 순간의 특징은 그대로다."""
    table, calls = _make(3, min_len=6)
    base = hl.batch_features(table.probs, table.offsets, calls, SHIFT)
    rng = np.random.default_rng(99)
    for i, call in enumerate(calls):
        n = call.n_utt
        k = int(rng.integers(1, n))                          # 1..n-1
        p = table.call(i).copy()
        head = base[table.offsets[i]:table.offsets[i] + k]
        # (1) 뒤쪽을 다른 내용으로 바꾼다
        p2 = p.copy()
        p2[k:] = rng.random((n - k, 9)).astype(np.float32)
        changed = Call(call.call_id, call.texts[:k] + ["구급차 주소 [개인정보] 감사합니다?"] * (n - k), call.speakers[:k] + [1] * (n - k),
                       call.starts[:k] + [s + 777 for s in call.starts[k:]], call.ends[:k] + [e + 99999 for e in call.ends[k:]], call.labels)
        got = hl.batch_features(p2, np.array([0, n]), [changed], SHIFT)
        np.testing.assert_allclose(got[:k], head, rtol=0, atol=1e-12)
        # (2) 통화를 k 에서 자른다 (전체 길이가 달라져도 같아야 한다)
        cut = Call(call.call_id, call.texts[:k], call.speakers[:k], call.starts[:k], call.ends[:k], call.labels)
        got = hl.batch_features(p[:k], np.array([0, k]), [cut], SHIFT)
        np.testing.assert_allclose(got, head, rtol=0, atol=1e-12)
        # (3) 뒤에 발화를 덧붙여 통화를 늘린다
        longer = Call(call.call_id, call.texts + ["네"] * 7, call.speakers + [0] * 7, call.starts + [call.starts[-1] + 1000] * 7,
                      call.ends + [call.ends[-1] + 2000] * 7, call.labels)
        got = hl.batch_features(np.vstack([p, np.full((7, 9), 0.9, dtype=np.float32)]), np.array([0, n + 7]), [longer], SHIFT)
        np.testing.assert_allclose(got[:n], base[table.offsets[i]:table.offsets[i + 1]], rtol=0, atol=1e-12)


def test_features_do_not_depend_on_labels():
    table, calls = _make(4)
    a = hl.HaltData.build(table, calls, SHIFT)
    flipped = sim.PrefixTable(table.call_ids, table.offsets, table.probs, 1 - table.labels)
    b = hl.HaltData.build(flipped, calls, SHIFT)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.value, b.value)


def test_score_matches_simulate():
    """빠른 채점(가중치 방식)과 score_halts 가 m4.simulate.score_4_2 와 같은 값을 낸다."""
    table, calls = _make(5, n_calls=80)
    data = hl.HaltData.build(table, calls, SHIFT)
    halts = hl.first_halt(data.value >= 0.5, data)
    by_call = iter(halts)
    ref = sim.score_4_2(table, lambda p: int(next(by_call)), lambda p: sim.decide(p, force_top1=True))
    ours = hl.score_halts(data, halts)
    fast = float(hl._weighted_scores(hl._score_parts(data, halts), np.ones((1, len(data))))[0])
    assert abs(ours["score"] - ref["score"]) < 1e-12 and abs(fast - ref["score"]) < 1e-12


def test_first_halt_respects_min_step():
    table, calls = _make(6, n_calls=60, min_len=1, max_len=12)
    data = hl.HaltData.build(table, calls, SHIFT)
    always = hl.first_halt(np.ones(len(data.X), dtype=bool), data)
    np.testing.assert_array_equal(always, np.where(data.n_utt >= 5, 5, data.n_utt))      # 5번째, 짧은 통화는 끝
    never = hl.first_halt(np.zeros(len(data.X), dtype=bool), data)
    np.testing.assert_array_equal(never, data.n_utt)                     # 못 끊으면 끝까지 (finalize)


def test_delta_target_definition():
    """δ 를 통화 하나씩 풀어서 다시 계산한 값과 같다. 학습 행은 5번째 이후·마지막 발화 전이다."""
    table, calls = _make(7, n_calls=50, min_len=3)
    data = hl.HaltData.build(table, calls, SHIFT)
    ref_tau = float(np.quantile(data.value, 0.7))
    delta, rows, consts = hl.delta_target(data, ref_tau)
    step, n_rep = data.step, np.repeat(data.n_utt, data.n_utt)
    np.testing.assert_array_equal(rows, (step >= 5) & (step < n_rep))
    h0 = hl.first_halt(data.value >= ref_tau, data)
    p0 = data.pred[data.offsets[:-1] + h0 - 1]
    tp, fp, fn = (p0 & data.labels).sum(0), (p0 & ~data.labels).sum(0), (~p0 & data.labels).sum(0)
    D = np.maximum(2 * tp + fp + fn, 1)
    f1 = 2 * tp / D
    E = sim.earliness(h0, data.n_utt).mean()
    for i in range(len(data)):
        a, b = data.offsets[i], data.offsets[i + 1]
        n = b - a
        q = data.cal[b - 1]
        u = [E * len(data) / 9 * (data.pred[a + t] * (q * (2 - f1) - (1 - q) * f1) / D).sum()
             + f1.mean() * (0.5 + 0.5 * max(0.0, 1 - max(t + 1, 5) / n)) for t in range(n)]
        for t in range(4, n - 1):                             # step = t + 1
            later = [s for s in range(t + 1, n) if data.value[a + s] >= ref_tau]
            nxt = later[0] if later else n - 1
            assert abs(delta[a + t] - (u[t] - u[nxt])) < 1e-9
    assert consts["n_rows"] == int(rows.sum())


def test_policy_end_to_end():
    """작은 정책을 학습해: 5번째 발화 전에는 중단하지 않고, 저장·불러오기 뒤에도 같고, 온라인 판단이 모의 채점과 같다."""
    pytest.importorskip("lightgbm")
    table, calls = _make(8, n_calls=120, min_len=1, max_len=25)
    data = hl.HaltData.build(table, calls, SHIFT)
    policy = hl.train_policy(data, ref_tau=0.5, logit_shift=SHIFT, folds=3, params={"min_data_in_leaf": 5}, rounds=20, threads=1)
    assert policy.beta in hl.BETAS and policy.tau in hl.TAUS and policy.n_fit_calls == len(data)

    # 어떤 τ 로도 5번째 발화 전에는 끊지 않는다 (통화가 5발화보다 짧으면 끝에서 답한다)
    eager = hl.HaltPolicy(policy.model_str, beta=8.0, tau=-1e9, logit_shift=list(SHIFT))
    halts = eager.halts(data)
    np.testing.assert_array_equal(halts, np.where(data.n_utt >= 5, 5, data.n_utt))

    # 저장 → 불러오기
    loaded = hl.HaltPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
    forced = hl.HaltPolicy(loaded.model_str, beta=4.0, tau=policy.tau, logit_shift=loaded.logit_shift)
    np.testing.assert_allclose(forced.delta_hat(data.X), policy.booster().predict(data.X), rtol=0, atol=1e-12)

    # 온라인 판단(발화를 하나씩 넣음)과 모의 채점의 중단 시점이 같다
    sim_halts = forced.halts(data)
    state = hl.HaltFeatureState(forced.logit_shift)
    for i, call in enumerate(calls[:40]):
        state.reset()
        got = call.n_utt
        for t in range(call.n_utt):
            x = state.update(call.texts[t], call.speakers[t], call.starts[t], call.ends[t], table.call(i)[t])
            if forced.should_halt(state, x):
                got = t + 1
                break
        assert got == sim_halts[i], (i, got, sim_halts[i])


def test_policy_rejects_changed_feature_list():
    pytest.importorskip("lightgbm")
    bad = hl.HaltPolicy(model_str="", beta=1.0, tau=0.5, feature_names=hl.FEATURE_NAMES[:-1])
    with pytest.raises(ValueError):
        bad.booster()
