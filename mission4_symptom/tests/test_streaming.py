"""스트리밍 추론기(m4.streaming)의 학습형 중단 분기: 발화를 하나씩 넣어 만든 특징·중단 시점이 오프라인 계산과 같은지 확인한다.

인코더는 불러오지 않는다. `_infer`(누적 전사 -> 확률)만 미리 만든 확률표로 바꿔 끼우고, 나머지(push, reset,
should_halt, symptoms)는 실제 코드를 그대로 탄다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4 import SYMPTOM_CLASSES  # noqa: E402
from m4 import halting as hl  # noqa: E402
from m4.bundle import DEFAULT_SETTINGS, learned_halt_setting  # noqa: E402
from m4.streaming import StreamingSymptomModel  # noqa: E402

from test_halting import SHIFT, _make  # noqa: E402


def _model(halt: dict) -> StreamingSymptomModel:
    """번들 없이 추론 설정만으로 만든 추론기. 확률은 `feed` 에 넣어 둔 행을 순서대로 낸다."""
    model = StreamingSymptomModel.__new__(StreamingSymptomModel)
    model.members = []
    model.settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    model.settings.update({"logit_shift": [float(x) for x in SHIFT], "halt": halt,
                           "rule": {"type": "threshold", "threshold": 0.5, "force_top1": True}})
    model._configure()
    model.feed = iter(())
    model._infer = lambda joined: np.asarray(next(model.feed), dtype=np.float32)
    return model


def _policy(seed: int = 8):
    """작은 학습형 정책과 그 학습에 쓴 자료. tau 는 중단 시점이 통화마다 갈리도록 신호의 중앙값으로 둔다."""
    table, calls = _make(seed, n_calls=120, min_len=1, max_len=25)
    data = hl.HaltData.build(table, calls, SHIFT)
    fitted = hl.train_policy(data, ref_tau=0.5, logit_shift=SHIFT, folds=3, params={"min_data_in_leaf": 5}, rounds=20, threads=1)
    signal = data.value + 4.0 * fitted.delta_hat(data.X)
    policy = hl.HaltPolicy(fitted.model_str, beta=4.0, tau=float(np.quantile(signal[data.step >= 5], 0.8)),
                           logit_shift=[float(x) for x in SHIFT])
    return policy, table, calls, data


def test_learned_features_and_halts_match_offline():
    """push 로 하나씩 넣은 특징이 batch_features 와 같고, should_halt 의 첫 중단 시점이 모의 채점(policy.halts)과 같다."""
    pytest.importorskip("lightgbm")
    policy, table, calls, data = _policy()
    model = _model(learned_halt_setting(policy.to_dict(), [float(x) for x in SHIFT]))
    sim_halts = policy.halts(data)
    assert len(set(sim_halts.tolist())) > 3 and (sim_halts < data.n_utt).any() and (sim_halts == data.n_utt).any()
    for i, call in enumerate(calls):
        a = int(table.offsets[i])
        model.reset()
        model.feed = iter(table.call(i))
        first = None
        for t in range(call.n_utt):
            model.push(call.texts[t], call.speakers[t], call.starts[t], call.ends[t])
            np.testing.assert_allclose(model._halt_x, data.X[a + t], rtol=0, atol=1e-9)
            halt_now = model.should_halt()
            assert not (halt_now and t + 1 < 5)                          # 5번째 발화 전에는 중단하지 않는다
            if halt_now and first is None:
                first = t + 1
                # 중단 시점 예측은 채택 규칙 그대로다: 보정 전 확률 >= 0.5, 없으면 1개
                want = [c for c, on in zip(SYMPTOM_CLASSES, data.pred[a + t]) if on]
                assert model.symptoms(at_halt=True) == want and want
        assert (first or call.n_utt) == sim_halts[i], (i, first, sim_halts[i])


def test_learned_reset_clears_state():
    """reset 뒤에는 앞 통화의 흔적이 없다: 통화 순서를 바꿔 넣어도 특징이 같다."""
    pytest.importorskip("lightgbm")
    policy, table, calls, data = _policy()
    model = _model(learned_halt_setting(policy.to_dict(), [float(x) for x in SHIFT]))
    for i in (7, 3, 7, 0):
        model.reset()
        assert model._halt_x is None and model._halt_state.step == 0
        model.feed = iter(table.call(i))
        for t in range(calls[i].n_utt):
            model.push(calls[i].texts[t], calls[i].speakers[t], calls[i].starts[t], calls[i].ends[t])
            np.testing.assert_allclose(model._halt_x, data.X[int(table.offsets[i]) + t], rtol=0, atol=1e-9)


def test_expected_halt_unchanged_without_times():
    """예전 방식(expected)은 시각 없이 push(text, speaker) 로 불러도 그대로 동작하고, lightgbm 정책을 만들지 않는다."""
    table, calls = _make(11, n_calls=30, min_len=1, max_len=20)
    data = hl.HaltData.build(table, calls, SHIFT)
    model = _model({"type": "expected", "tau": 0.5, "min_step": 5})
    assert model._policy is None and model._halt_state is None
    want = hl.first_halt(data.value >= 0.5, data)
    for i, call in enumerate(calls):
        model.reset()
        model.feed = iter(table.call(i))
        first = None
        for t in range(call.n_utt):
            model.push(call.texts[t], call.speakers[t])
            if first is None and model.should_halt():
                first = t + 1
        assert (first or call.n_utt) == want[i]


def test_learned_falls_back_without_lightgbm(monkeypatch, capsys):
    """lightgbm 을 읽지 못하면 번들의 대체 기준(기대 점수)으로 내려가고, 대체 기준이 없으면 그대로 실패한다."""
    pytest.importorskip("lightgbm")
    policy, table, calls, data = _policy()
    setting = learned_halt_setting(policy.to_dict(), [float(x) for x in SHIFT])
    monkeypatch.setitem(sys.modules, "lightgbm", None)          # import lightgbm -> ImportError
    with pytest.raises(ImportError):
        _model(dict(setting))
    fallback = {"type": "expected", "tau": 0.5, "min_step": 5}
    model = _model(dict(setting, fallback=fallback))
    assert model.settings["halt"] == fallback and model._policy is None and model._halt_state is None
    assert "대체 기준" in capsys.readouterr().err
    reference = _model(dict(fallback))                          # 처음부터 기대 점수 기준인 추론기와 같은 판단을 낸다
    for i in range(10):
        rows = table.call(i)
        for m in (model, reference):
            m.reset()
            m.feed = iter(rows)
        for j in range(len(rows)):
            model.push("말", 0, 1000 * j, 1000 * j + 500)
            reference.push("말", 0)
            assert model.should_halt() == reference.should_halt()


def test_predictor_step_passes_utterance_times():
    """inference.py 의 Predictor.step 이 발화 시각(start_ms, end_ms)을 추론기에 넘긴다 (학습형 특징의 입력)."""
    import ast

    source = (HERE / "inference.py").read_text(encoding="utf-8")
    cls = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef) and n.name == "Predictor")
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    pushes = [n for n in ast.walk(step) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "push"]
    assert len(pushes) == 1
    passed = {ast.unparse(a) for a in pushes[0].args} | {ast.unparse(k.value) for k in pushes[0].keywords}
    assert {"utt.text", "utt.speaker", "utt.start_ms", "utt.end_ms"} <= passed
