"""Synthetic checks of the experiment runner; no pretrained weights or audio."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/run_all_audio_hypotheses.py"
spec = importlib.util.spec_from_file_location("hypothesis_runner_under_test", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class HypothesisRunnerTests(unittest.TestCase):
    def test_changed_training_plan_cannot_reuse_completed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            plan = self._plan(root)
            plan['optimizer'] = {**plan['optimizer'], 'lr': 0.002}
            with self.assertRaisesRegex(ValueError, 'configuration or cache changed'):
                runner.stage_compare(plan, resume=True)

    def test_changed_cache_identity_cannot_reuse_completed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            (root / 'cache_500_resnet128/config.json').write_text(json.dumps({
                'ready': True, 'cache_id': 'changed-cache'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'configuration or cache changed'):
                runner.stage_compare(self._plan(root), resume=True)

    def test_legacy_metadata_cannot_reuse_completed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            (root / 'compare_s42/baseline/run_config.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'configuration or cache changed'):
                runner.stage_compare(self._plan(root), resume=True)

    def test_nonfinite_saved_probabilities_cannot_enter_bootstrap(self):
        with self.assertRaises(ValueError):
            runner.compute_paired_call_bootstrap(np.array(["a", "a"]), np.array([0, 1]),
                                                np.array([0.1, np.nan]), np.array([0.1, 0.9]))

    def test_nonfinite_evaluation_logits_are_rejected(self):
        class InvalidModel(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, mel, extra):
                return torch.full((len(mel),), self.value)

        loader = [(torch.zeros(2, 1, 8, 8), torch.zeros(2, 6), torch.tensor([0, 1]))]
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                runner.evaluate_model(InvalidModel(value), loader, torch.device("cpu"))

    def test_cached_comparison_reports_actual_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            plan = self._plan(root)
            runner.stage_compare(plan, resume=True)
            table = pd.read_csv(root / "ALL_HYPOTHESES_COMPARISON.csv")
            for column, expected in (("train_calls", 2), ("dev_calls", 1),
                                     ("train_utterances", 3), ("dev_utterances", 2)):
                self.assertTrue((table[column] == expected).all(), column)

    def test_passing_candidate_report_remains_pending_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            plan = self._plan(root)
            runner.stage_compare(plan, resume=True)
            (root / "confirmation_candidate.json").write_text(json.dumps({
                "candidate": "zcr", "action": "proceed_to_r5_confirmation"}), encoding="utf-8")
            runner.stage_report(plan)
            text = (root / "ADOPTION_DECISION.md").read_text(encoding="utf-8")
            self.assertIn("확대 재확인 대상", text)
            self.assertNotIn("후보가 없으므로", text)
            report = (root / "REMAINING_HYPOTHESES_RESULT.md").read_text(encoding="utf-8")
            self.assertIn("Training 3통화 (5발화, Train 3 / Dev 2", report)

    @staticmethod
    def _plan(root):
        return {"output_root": str(root), "epochs": 1, "seeds": [42, 43],
                "initial_conditions": ["baseline", "short", "rms", "zcr", "bandpass"],
                "bootstrap_iterations": 2, "threshold": 0.5, "batch_size": 2,
                "head_dropout": 0.3, "optimizer": {"lr": 0.0001, "weight_decay": 0.01}}

    @staticmethod
    def _fixture(root):
        cache = root / "cache_500_resnet128"
        cache.mkdir()
        (cache / 'config.json').write_text(json.dumps({
            'ready': True, 'cache_id': 'synthetic_completed_conditions'}), encoding='utf-8')
        np.savez(cache / "features.npz", x=np.zeros((5, 7)),
                 dev=np.array([False, False, False, True, True]),
                 calls=np.array(["a", "a", "b", "c", "c"]))
        for seed in (42, 43):
            for cond in ("baseline", "short", "rms", "zcr", "bandpass"):
                folder = root / f"compare_s{seed}" / cond
                folder.mkdir(parents=True)
                (folder / "best_dev.pt").touch()
                (folder / "last.pt").touch()
                np.savez(folder / "best_dev_predictions.npz", y=np.array([0, 1]),
                         probability=np.array([0.1, 0.9]), calls=np.array(["c", "c"]))
                np.savez(folder / "last_predictions.npz", y=np.array([0, 1]),
                         probability=np.array([0.1, 0.9]), calls=np.array(["c", "c"]))
                (folder / 'run_config.json').write_text(json.dumps({
                    'resume_identity': runner.condition_resume_identity(cond, seed, HypothesisRunnerTests._plan(root), cache)
                }), encoding='utf-8')
                pd.DataFrame([{"epoch": 1, "accuracy": 1.0, "macro_f1": 1.0,
                               "recall_0": 1.0, "recall_1": 1.0,
                               "epoch_seconds": 0.001}]).to_csv(folder / "epoch_metrics.csv", index=False)


if __name__ == "__main__":
    unittest.main()
