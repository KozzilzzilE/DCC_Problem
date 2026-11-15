"""CPU 텐서/정적검사만 사용. 학습기 및 분류 모델을 실행하지 않는다."""
import ast
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from m4 import t1_loss_masking as t1
from m4.data import Call
import prefix_experiment as runner


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(min(510, len(text)))]}


def call(texts, label=0):
    labels = [0] * 9
    labels[6] = label
    return Call("test", texts, [1] * len(texts), [0] * len(texts), [1] * len(texts), labels)


def record(c, step, confirmed=True):
    return {"decision": "accepted", "evidence": [{"step": step,
        "text_sha256": t1.text_sha(c.texts[step - 1]), "target_confirmed": confirmed}]}


def actual_reduction(elements, mask):
    # 실제 학습 코드에 연결된 Loss 축약식을 읽어 CPU 텐서로 실행한다.
    tree = ast.parse((HERE / "scripts/train_prefix.py").read_text(encoding="utf-8"))
    expr = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "loss" for t in n.targets)
                and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == "mean")
    return eval(compile(ast.Expression(expr), "train_prefix_loss", "eval"), {"loss": elements, "mask": mask})


class T1Tests(unittest.TestCase):
    def gradients(self, masked):
        z = torch.arange(18, dtype=torch.float64).reshape(2, 9).div(10).requires_grad_()
        y = torch.tensor([[0] * 9, [1] * 9], dtype=torch.float64)
        weights = torch.arange(1, 10, dtype=torch.float64)
        if masked is None:
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=weights)(z, y)
        else:
            elements = torch.nn.BCEWithLogitsLoss(pos_weight=weights, reduction="none")(z, y)
            loss = actual_reduction(elements, masked)
        loss.backward()
        return loss.detach(), z.grad

    def test_all_ones_equal_loss_and_gradient(self):
        a, ga = self.gradients(None)
        b, gb = self.gradients(torch.ones((2, 9), dtype=torch.float64))
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        # 내부 mean과 외부 mean의 역전파 연산 순서에 따른 float64 반올림만 허용.
        torch.testing.assert_close(ga, gb, atol=1e-15, rtol=1e-14)

    def test_only_negative_nausea_direct_gradient_zero(self):
        _, baseline = self.gradients(None)
        mask = torch.ones((2, 9), dtype=torch.float64)
        mask[0, 6] = 0
        _, grad = self.gradients(mask)
        self.assertEqual(grad[0, 6].item(), 0)
        baseline[0, 6] = 0
        torch.testing.assert_close(grad, baseline, atol=1e-15, rtol=1e-14)

    def test_positive_and_other_classes_never_masked(self):
        y = torch.zeros((2, 9))
        y[1, 6] = 1
        mask = t1.loss_mask(y, ["x", "y"], [3, 3], {"x": {3}, "y": {3}})
        self.assertEqual(mask[0, 6].item(), 0)
        self.assertEqual(mask[1, 6].item(), 1)
        self.assertEqual(int((mask == 0).sum()), 1)
        c = call(["제가 지금 메스꺼워요"], label=1)
        self.assertEqual(t1.maskable_prefixes(c, record(c, 1), FakeTokenizer()), set())

    def test_future_evidence_never_masks_earlier_prefix(self):
        c = call(["여보세요", "제가 아파요", "지금 속이 울렁거려요"])
        plan = t1.maskable_prefixes(c, record(c, 3), FakeTokenizer())
        self.assertEqual(plan, {3})

    def test_unreviewed_call_does_not_mask(self):
        c = call(["제가 메스꺼워요"])
        self.assertEqual(t1.maskable_prefixes(c, {"decision": "not_deep_reviewed"}, FakeTokenizer()), set())

    def test_question_denial_other_person_and_resolved_past_excluded(self):
        cases = [("속이 울렁거리세요?", 0, True), ("토할 거 같애?", 1, True),
                 ("지금 속이 메스껍지 않아요", 1, True),
                 ("다른 사람이 토할 거 같다고 해요", 1, True),
                 ("어제는 메스껍고 지금은 괜찮아요", 1, True),
                 ("친구가 토할 것 같다고 해요", 1, False),
                 ("계속 구토하고 있어요", 1, True), ("헛구역질만 해요", 1, True)]
        for text, speaker, confirmed in cases:
            with self.subTest(text=text):
                self.assertFalse(t1.valid_evidence(text, speaker, confirmed))
        self.assertTrue(t1.valid_evidence("제가 지금 속이 메스껍거든요. 구급차 보내주세요?", 1, True))

    def test_later_resolution_cancels_without_changing_past(self):
        c = call(["제가 메스꺼워요", "지금은 괜찮아요"])
        self.assertEqual(t1.maskable_prefixes(c, record(c, 1), FakeTokenizer()), {1})

    def test_truncated_evidence_is_not_used(self):
        c = call(["가" * 520, "지금 토할 것 같아요"])
        self.assertEqual(t1.maskable_prefixes(c, record(c, 2), FakeTokenizer()), set())

    def test_sampling_source_and_rng_order_unchanged(self):
        original = subprocess.check_output(["git", "-c", "safe.directory=" + str(HERE.parent).replace("\\", "/"),
            "show", "HEAD:mission4_symptom/scripts/train_prefix.py"], cwd=HERE).decode("utf-8")
        modified = (HERE / "scripts/train_prefix.py").read_text(encoding="utf-8")
        def assignments(source):
            return [ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(source))
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"rng", "items", "order"} for t in n.targets)]
        self.assertEqual(assignments(original), assignments(modified))
        a, b = np.random.default_rng([42, 1]), np.random.default_rng([42, 1])
        expected = [int(a.integers(1, 31)) for _ in range(20)]
        actual = []
        for _ in range(20):
            actual.append(int(b.integers(1, 31)))
            t1.loss_mask(torch.zeros((1, 9)), ["x"], [actual[-1]], {"x": {5}})
        self.assertEqual(expected, actual)
        self.assertEqual(a.random(), b.random())

    def test_default_baseline_loss_and_no_mask_flag(self):
        source = (HERE / "scripts/train_prefix.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        reduction = next(kw.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "BCEWithLogitsLoss"
                         for kw in n.keywords if kw.arg == "reduction")
        from types import SimpleNamespace
        self.assertEqual(eval(compile(ast.Expression(reduction), "reduction", "eval"),
                         {"args": SimpleNamespace(t1_mask_manifest=None)}), "mean")

    def test_t1_output_isolation_before_training(self):
        with patch.object(runner, "verify") as verify:
            with self.assertRaises(ValueError):
                runner.run(runner.TEAM / "bad", strategy="t1", mask_manifest="unused")
            verify.assert_not_called()

    def test_extract_failure_blocks_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "check_fit_run"), \
             patch.object(runner, "invoke", side_effect=RuntimeError("extract failed")), \
             patch.object(runner, "write_evaluation") as evaluate:
            (Path(tmp) / "best_model").mkdir()
            (Path(tmp) / "best_model/model.safetensors").touch()
            with self.assertRaises(RuntimeError):
                runner.extract_stage(Path(tmp), Path(tmp), existing=True)
            evaluate.assert_not_called()

    def test_plan_does_not_consume_torch_or_numpy_rng(self):
        numpy_state = copy.deepcopy(np.random.get_state())
        torch_state = torch.random.get_rng_state().clone()
        c = call(["제가 아파요", "제가 지금 속이 메스꺼워요"])
        t1.maskable_prefixes(c, record(c, 2), FakeTokenizer())
        self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))

    def test_train_failure_blocks_extract_and_sequential(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "verify", return_value=({}, None, None)), \
             patch.object(runner, "read", return_value=runner.RECIPE), \
             patch.object(runner, "BACKBONE_SHA256", {}), patch.object(runner, "reserve", return_value=Path(tmp)), \
             patch.object(runner, "save"), patch.object(runner, "digest", return_value="test"), \
             patch.object(runner, "invoke", side_effect=RuntimeError("train failed")), \
             patch.object(runner, "extract_stage") as extract, patch.object(runner, "sequential") as seq:
            with self.assertRaises(RuntimeError):
                runner.run(Path(tmp), all_stages=True)
            extract.assert_not_called()
            seq.assert_not_called()

    def test_evaluation_failure_blocks_sequential(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "verify", return_value=({}, None, None)), \
             patch.object(runner, "read", return_value=runner.RECIPE), \
             patch.object(runner, "BACKBONE_SHA256", {}), patch.object(runner, "reserve", return_value=Path(tmp)), \
             patch.object(runner, "save"), patch.object(runner, "digest", return_value="test"), \
             patch.object(runner, "invoke"), \
             patch.object(runner, "extract_stage", side_effect=RuntimeError("eval failed")), \
             patch.object(runner, "sequential") as seq:
            with self.assertRaises(RuntimeError):
                runner.run(Path(tmp), all_stages=True)
            seq.assert_not_called()

    def test_t1_argument_is_explicit_and_sequence_order(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "OUTPUTS", Path(tmp)), \
             patch.object(runner, "verify", return_value=({}, None, None)), \
             patch.object(runner, "read", side_effect=lambda p: runner.RECIPE if p == runner.E002 / "run_config.json" else {}), \
             patch.object(runner, "BACKBONE_SHA256", {}), patch.object(runner, "save"), \
             patch.object(runner, "digest", return_value="test"), patch.object(runner, "invoke") as invoke, \
             patch.object(runner, "extract_stage") as extract, patch.object(runner, "sequential") as seq:
            runner.run(Path(tmp) / "t1_loss_masking/run", strategy="t1", mask_manifest="manifest.json", all_stages=True)
            args = invoke.call_args.args[1]
            self.assertIn("--t1-mask-manifest", args)
            self.assertEqual(args[args.index("--seed") + 1], "42")
            extract.assert_called_once()
            seq.assert_called_once()

    def test_frozen_team_inputs_unchanged(self):
        self.assertEqual(runner.digest(runner.POLICY), runner.POLICY_SHA256)
        for path, expected in runner.INPUT_SHA256.items():
            self.assertEqual(runner.digest(path), expected)
        for path, expected in zip(runner.FIXED, runner.FIXED_SHA256):
            self.assertEqual(runner.digest(path), expected)

    def test_existing_model_files_match_baseline_sequential_receipt(self):
        receipt = runner.read(runner.OUTPUTS / "localsequential/sequential_timing.json")
        for source, expected in receipt["model_hashes"].items():
            path = Path(source)
            if not path.is_absolute():
                path = HERE / path
            self.assertEqual(runner.digest(path / "model.safetensors"), expected)


if __name__ == "__main__":
    unittest.main()
