"""번들을 묶기 전에 멤버들의 학습 설정을 대조하는 로직을 확인한다 (모델을 읽지 않는다)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4 import SYMPTOM_CLASSES  # noqa: E402
from m4.bundle import (  # noqa: E402
    BUNDLE_SPLIT_VERSION,
    DEFAULT_SETTINGS,
    learned_halt_setting,
    load_bundle,
    member_pos_weight,
    pack_mission_split_bundle,
    shared_pos_weight,
    shared_prior_source,
)

WEIGHTS = [2.1, 2.3, 3.0, 1.8, 1.9, 2.7, 2.8, 2.1, 2.5]


def _final(weights=WEIGHTS, calls="runs/cache/train.jsonl", ids="runs/cache/fit_ids.txt") -> dict:
    """본선 학습 스크립트가 남기는 형식."""
    return {"pos_weight": list(weights), "calls": calls, "ids": ids}


def _prelim(weights=WEIGHTS) -> dict:
    """예선 학습 스크립트가 남기는 형식 (증상 이름별 통계, calls·ids 없음). 키 순서를 일부러 뒤집는다."""
    stats = {c: {"pos_weight": w} for c, w in zip(SYMPTOM_CLASSES, weights)}
    return {"train_pos_weight_statistics": dict(reversed(list(stats.items())))}


def test_member_pos_weight_reads_both_formats():
    assert member_pos_weight(_final()) == WEIGHTS
    assert member_pos_weight(_prelim()) == WEIGHTS          # 키 순서가 아니라 증상 이름으로 맞춘다
    assert member_pos_weight({}) is None
    assert member_pos_weight({"train_pos_weight_statistics": {"고열": {"pos_weight": 2.0}}}) is None


def test_member_pos_weight_rejects_broken_values():
    with pytest.raises(ValueError):
        member_pos_weight({"pos_weight": WEIGHTS[:8]})
    with pytest.raises(ValueError):
        member_pos_weight({"pos_weight": [0.0] + WEIGHTS[1:]})


def test_shared_pos_weight_mixed_formats():
    # float32 로 저장된 값과 float64 값의 차이(1e-7 수준)는 같은 값으로 본다
    near = [w + 1e-7 for w in WEIGHTS]
    weights, missing = shared_pos_weight([_prelim(), _final(near), {}], ["a", "b", "c"])
    assert weights == WEIGHTS and missing == ["c"]


def test_shared_pos_weight_missing_first_member():
    weights, missing = shared_pos_weight([{}, _final()], ["a", "b"])
    assert weights == WEIGHTS and missing == ["a"]
    assert shared_pos_weight([{}, {}], ["a", "b"]) == (None, ["a", "b"])


def test_shared_pos_weight_stops_on_mismatch():
    other = list(WEIGHTS)
    other[4] += 0.01
    with pytest.raises(ValueError, match="pos_weight"):
        shared_pos_weight([_final(), _prelim(other)], ["a", "b"])


def test_shared_prior_source():
    src = ("runs/cache/train.jsonl", "runs/cache/fit_ids.txt")
    assert shared_prior_source([_prelim(), _final(), _final()], ["a", "b", "c"]) == src
    assert shared_prior_source([_prelim(), {}], ["a", "b"]) == (None, None)
    assert shared_prior_source([_final(ids=None)], ["a"]) == ("runs/cache/train.jsonl", None)
    with pytest.raises(ValueError, match="학습 통화"):
        shared_prior_source([_final(), _final(ids="other.txt")], ["a", "b"])


def _policy_dict(shift) -> dict:
    return {"type": "learned", "beta": 16.0, "tau": 0.64, "min_step": 5, "logit_shift": shift,
            "feature_names": ["a", "b"], "model_str": "tree"}


def test_learned_halt_setting_checks_logit_shift():
    """학습형 중단 정책을 번들에 넣기 전에 보정값을 대조한다. 다르면 멈추고, 저장 정밀도 차이(1e-7 수준)는 넘긴다."""
    shift = [0.76, 0.85, 1.10, 0.60, 0.64, 1.01, 1.02, 0.75, 0.93]
    near = [x + 4e-8 for x in shift]
    got = learned_halt_setting(_policy_dict(shift), near)
    assert got["type"] == "learned" and got["logit_shift"] == shift and got["beta"] == 16.0      # 정책의 값을 그대로 둔다
    other = list(shift)
    other[3] += 0.01
    with pytest.raises(ValueError, match="logit_shift"):
        learned_halt_setting(_policy_dict(shift), other)
    with pytest.raises(ValueError, match="logit_shift"):
        learned_halt_setting(_policy_dict(shift), None)
    with pytest.raises(ValueError, match="logit_shift"):
        learned_halt_setting(_policy_dict(None), shift)
    with pytest.raises(ValueError, match="logit_shift"):
        learned_halt_setting(_policy_dict(shift), shift[:8])


def test_learned_halt_setting_rejects_other_policies():
    with pytest.raises(ValueError, match="학습형"):
        learned_halt_setting({"type": "expected", "tau": 0.5, "min_step": 5}, None)
    broken = _policy_dict(None)
    del broken["model_str"]
    with pytest.raises(ValueError, match="model_str"):
        learned_halt_setting(broken, None)


def test_default_settings_comply_with_competition_rules():
    assert DEFAULT_SETTINGS["rule"] == {"type": "threshold", "threshold": 0.5, "force_top1": False, "max_labels": None}
    assert DEFAULT_SETTINGS["halt_rule"] == {"type": "threshold", "threshold": 0.5, "force_top1": False}
    assert DEFAULT_SETTINGS["halt"]["require_nonempty"] is True


def test_learned_halt_setting_preserves_require_nonempty():
    shift = [0.76] * 9
    assert learned_halt_setting(_policy_dict(shift), shift)["require_nonempty"] is False
    flagged = dict(_policy_dict(shift), require_nonempty=True)
    assert learned_halt_setting(flagged, shift)["require_nonempty"] is True


def _write_tiny_member(dirpath: Path, seed: int) -> None:
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    dirpath.mkdir(parents=True, exist_ok=True)
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "고열": 4, "구토": 5}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]"
    )
    fast_tok.save_pretrained(str(dirpath))

    torch.manual_seed(seed)
    cfg = BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, num_labels=len(SYMPTOM_CLASSES),
    )
    model = BertForSequenceClassification(cfg)
    cfg.save_pretrained(str(dirpath))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(dirpath / "model.safetensors"))


def test_pack_and_load_mission_split_bundle_offline(tmp_path, monkeypatch):
    import torch

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    m41 = tmp_path / "sj42"
    m42_a = tmp_path / "e001"
    m42_b = tmp_path / "e002"
    _write_tiny_member(m41, 1)
    _write_tiny_member(m42_a, 2)
    _write_tiny_member(m42_b, 3)

    bundle_path = tmp_path / "split.pt"
    pack_mission_split_bundle(
        {
            "4-1": ([m41], {"rule": {"type": "threshold", "threshold": 0.5, "force_top1": False}, "halt": {"type": "never"}}),
            "4-2": ([m42_a, m42_b], {"rule": {"type": "threshold", "threshold": 0.5, "force_top1": False},
                                     "halt": {"type": "expected", "tau": 0.5, "min_step": 5, "require_nonempty": True}}),
        },
        bundle_path,
        note="unit-test",
    )
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    assert payload["version"] == BUNDLE_SPLIT_VERSION

    mem41, set41 = load_bundle(bundle_path, torch.device("cpu"), mission="4-1")
    mem42, set42 = load_bundle(bundle_path, torch.device("cpu"), mission="4-2")
    assert len(mem41) == 1 and set41["halt"]["type"] == "never"
    assert len(mem42) == 2 and set42["halt"]["require_nonempty"] is True
    with pytest.raises(ValueError, match="mission"):
        load_bundle(bundle_path, torch.device("cpu"), mission=None)
    with pytest.raises(ValueError, match="mission"):
        load_bundle(bundle_path, torch.device("cpu"), mission="4-3")


def test_pack_members_uses_parent_name_for_best_model(tmp_path):
    from m4.bundle import _pack_members

    bm = tmp_path / "m4_softjaccard_v1_s42" / "best_model"
    _write_tiny_member(bm, 7)
    plain = tmp_path / "E002_prefix_uniform"
    _write_tiny_member(plain, 8)
    members = _pack_members([bm, plain], half=True)
    assert [m["name"] for m in members] == ["m4_softjaccard_v1_s42", "E002_prefix_uniform"]


def test_pack_bundle_records_halt_source_basename_only(tmp_path, monkeypatch):
    import json
    import types

    scripts_dir = str(HERE / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import pack_bundle  # noqa: E402

    pol_path = tmp_path / "private_dir" / "policy_fitonly_c128.json"
    pol_path.parent.mkdir(parents=True)
    shift = [0.76] * 9
    pol_path.write_text(json.dumps(_policy_dict(shift)), encoding="utf-8")

    fake_args = types.SimpleNamespace(
        halt_policy=str(pol_path),
        allow_empty_halt=False,
        require_nonempty=True,
        halt_fallback_tau=0.5,
    )
    halt = pack_bundle._build_halt_setting(fake_args, shift)
    assert halt["source"] == "policy_fitonly_c128.json"
    assert halt["require_nonempty"] is True


def test_train_prefix_losses_and_full_only_sampling():
    import numpy as np
    import torch

    scripts_dir = str(HERE / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import train_prefix  # noqa: E402

    assert train_prefix.resolve_loss_weights("bce") == (0.0, 0.0)
    assert train_prefix.resolve_loss_weights("softjaccard") == (0.5, 0.0)
    assert train_prefix.resolve_loss_weights("jacc_macrof1") == (0.5, 0.2)
    assert train_prefix.resolve_loss_weights("softjaccard", jaccard_weight=0.25) == (0.25, 0.0)

    torch.manual_seed(0)
    logits = torch.randn(4, len(SYMPTOM_CLASSES), requires_grad=True)
    target = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=torch.float32)
    pw = torch.tensor(WEIGHTS, dtype=torch.float32)

    bce_ref = torch.nn.BCEWithLogitsLoss(pos_weight=pw)(logits, target)
    l_bce = train_prefix.compute_loss(logits, target, pw, 0.0, 0.0)
    l_sj = train_prefix.compute_loss(logits, target, pw, 0.5, 0.0)
    l_jm = train_prefix.compute_loss(logits, target, pw, 0.5, 0.2)
    assert torch.allclose(l_bce, bce_ref, atol=1e-6)
    assert torch.isfinite(l_sj) and torch.isfinite(l_jm)
    assert l_sj.item() > l_bce.item() and l_jm.item() > l_sj.item()
    l_jm.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    prefixes = [["a", "b", "c"], ["d", "e"]]
    rng = np.random.default_rng(42)
    full_items = train_prefix.sample_epoch_items(prefixes, 2, rng, include_full=False, full_only=True)
    assert full_items == [(0, 3), (0, 3), (1, 2), (1, 2)]
    with pytest.raises(ValueError):
        train_prefix.sample_epoch_items(prefixes, 1, rng, include_full=True, full_only=True)


def test_train_halt_policy_forwards_require_nonempty(monkeypatch):
    import types
    import numpy as np

    scripts_dir = str(HERE / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import train_halt_policy as thp  # noqa: E402
    from m4 import halting as hl  # noqa: E402

    captured = {}

    class _FakeBooster:
        def feature_importance(self, importance_type="gain"):
            return np.ones(len(hl.FEATURE_NAMES), dtype=np.float64)

    class _FakePolicy:
        beta = 16.0
        tau = 0.5
        feature_names = hl.FEATURE_NAMES
        info = {
            "target": {"n_rows": 4, "delta_pos_rate": 0.5},
            "crossfit_r2": 0.1,
            "fit_score_ref_tau": 0.4,
            "fit_tau_only": 0.5,
            "fit_score_tau_only": 0.41,
            "fit_score_selected": 0.45,
        }

        def halts(self, ev):
            return np.zeros(len(ev), dtype=np.int64)

        def booster(self):
            return _FakeBooster()

    def _fake_train_policy(fit, **kwargs):
        captured.update(kwargs)
        return _FakePolicy()

    class _FakeHaltData:
        def __init__(self, call_ids):
            self.call_ids = call_ids
        value = np.array([0.6, 0.7])
        pred = np.array([[True] + [False] * 8, [False] * 9])
        step = np.array([5, 5])
        n_utt = np.array([5, 5])
        X = np.zeros((2, len(hl.FEATURE_NAMES)))

        def __len__(self):
            return 2

    monkeypatch.setattr(hl, "train_policy", _fake_train_policy)
    monkeypatch.setattr(hl, "first_halt", lambda stop, ev: np.zeros(len(ev), dtype=np.int64))
    monkeypatch.setattr(hl, "score_halts", lambda ev, h: {"score": 0.5, "macro_f1": 0.6, "earliness": 0.8, "halt_median": 5.0, "never_halted": 0.0})
    monkeypatch.setattr(hl, "paired_bootstrap", lambda ev, a, b, boot: {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p_gt0": 0.5})

    args = types.SimpleNamespace(
        ref_tau=0.5, folds=5, seed=0, threads=1, taus=[0.5], betas=[16.0],
        base_tau=0.5, boot=10, require_nonempty=True,
    )
    row, _, _ = thp.run_once(_FakeHaltData(["f1", "f2"]), _FakeHaltData(["e1", "e2"]), args, [0.5] * 9, "test")
    assert captured["require_nonempty"] is True
    assert row["require_nonempty"] is True

