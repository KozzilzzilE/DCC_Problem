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

    def test_asl_run_forwards_loss_and_keeps_baseline_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            hashes = {**{str(p): h for p, h in runner.INPUT_SHA256.items()},
                      **{str(runner.BACKBONE / p): h for p, h in runner.BACKBONE_SHA256.items()}}
            with patch.object(runner, "verify", return_value=({}, None, None)), \
                 patch.object(runner, "read", return_value=runner.RECIPE), \
                 patch.object(runner, "digest", side_effect=lambda p: hashes.get(str(p), "test")), \
                 patch.object(runner, "reserve", return_value=out), \
                 patch.object(runner, "save"), patch.object(runner, "invoke") as invoke, \
                 patch.object(runner, "write_evaluation") as evaluate:
                runner.run(out, loss="asl")
                script, args, _ = invoke.call_args_list[0].args
                self.assertEqual(script, "train_prefix.py")
                self.assertEqual(args[args.index("--loss") + 1], "asl")
                for key, value in runner.RECIPE.items():
                    flag = "--" + key.replace("_", "-")
                    if key == "include_full":
                        self.assertNotIn(flag, args)
                    else:
                        self.assertEqual(args[args.index(flag) + 1], str(value))
                self.assertEqual(args[args.index("--grad-accum") + 1], "1")
                self.assertEqual(args[args.index("--count-classes") + 1], "0")
                evaluate.assert_called_once()

    def test_with_sequential_calls_existing_evaluator_after_training(self):
        with patch.object(runner.os, "chdir"), patch.object(runner, "run") as run, \
             patch.object(runner, "sequential") as sequential, \
             patch("sys.argv", ["prefix_experiment", "run", "--out", "new", "--loss", "asl", "--with-sequential"]):
            runner.main()
            run.assert_called_once_with(Path("new"), "asl", None)
            sequential.assert_called_once_with(Path("new/sequential"), Path("new/training"), "cuda")


if __name__ == "__main__":
    unittest.main()
