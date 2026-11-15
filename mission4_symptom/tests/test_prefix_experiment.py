"""실행 연결의 보호 장치 검증. 모델 학습/추론은 호출하지 않는다."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("prefix_experiment", Path(__file__).resolve().parents[1] / "scripts/prefix_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ExperimentSafetyTests(unittest.TestCase):
    def test_reject_misaligned_calls_even_when_shapes_match(self):
        a = runner.sim.PrefixTable(["a", "b"], np.array([0, 1, 2]), np.full((2, 9), .2), np.zeros((2, 9)))
        b = runner.sim.PrefixTable(["b", "a"], a.offsets, a.probs, a.labels)
        with self.assertRaises(ValueError):
            runner.aligned([a, b])

    def test_refuse_overwrite_and_team_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner, "OUTPUTS", Path(tmp)):
                target = runner.reserve(Path(tmp) / "new")
                with self.assertRaises(FileExistsError):
                    runner.reserve(target)
                with self.assertRaises(ValueError):
                    runner.reserve(Path(tmp).parent / "outside")
                with self.assertRaises(ValueError):
                    runner.reserve(Path(tmp))

    def test_metrics_undefined_precision_and_confusion(self):
        pred = np.zeros((3, 9), bool)
        gold = np.zeros((3, 9), bool)
        gold[0, 0] = True
        result = runner.confusion(pred, gold)[runner.SYMPTOM_CLASSES[0]]
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (0, 0, 1, 2))
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0)

    def test_changed_reference_blocks_training_before_subprocess(self):
        with patch.object(runner, "verify", side_effect=ValueError("기준 불일치")):
            with patch.object(runner, "invoke") as invoke:
                with self.assertRaises(ValueError):
                    runner.run(Path("unused"))
                invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
