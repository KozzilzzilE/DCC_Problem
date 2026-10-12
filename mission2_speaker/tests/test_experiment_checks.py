"""Regression checks for experiment resume identity and cache population."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2.experiment_checks import stable_spec_id, call_split_counts


class ExperimentChecksTests(unittest.TestCase):
    def test_elapsed_initialization_time_does_not_change_resume_identity(self):
        first = {"seed": 42, "epochs": 5, "model_init_seconds": 0.101}
        second = {"seed": 42, "epochs": 5, "model_init_seconds": 0.987}
        self.assertEqual(stable_spec_id(first), stable_spec_id(second))

    def test_training_config_change_changes_resume_identity(self):
        first = {"seed": 42, "epochs": 5, "lr": 0.0001}
        self.assertNotEqual(stable_spec_id(first), stable_spec_id({**first, "lr": 0.001}))

    def test_real_call_and_utterance_counts(self):
        self.assertEqual(call_split_counts(["a", "a", "b", "c", "c"], [False, False, False, True, True]),
                         {"train_calls": 2, "dev_calls": 1, "train_utterances": 3, "dev_utterances": 2})

    def test_split_call_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            call_split_counts(["a", "a"], [False, True])

    def test_misaligned_split_is_rejected(self):
        with self.assertRaises(ValueError):
            call_split_counts(["a"], [])


if __name__ == "__main__":
    unittest.main()
