"""Training 내부 dev 선택 절차(make_dev_split.py, devsel.py) 단위 테스트."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

import devsel  # noqa: E402
import make_dev_split  # noqa: E402
from m3.config import TARGET_SYMPTOMS  # noqa: E402


def _frame(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {"call_id": [f"c{i:03d}" for i in range(n)], "text": [f"본문 {i}" for i in range(n)]}
    for index, symptom in enumerate(TARGET_SYMPTOMS):
        column = rng.integers(0, 2, n)
        column[index % n] = 1
        column[(index + 1) % n] = 0
        data[symptom] = column
    return pd.DataFrame(data)


class SplitTest(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_complete(self) -> None:
        frame = _frame(50)
        train_a, dev_a = make_dev_split.split_frame(frame, 0.1, 1234)
        train_b, dev_b = make_dev_split.split_frame(frame, 0.1, 1234)
        self.assertEqual(list(dev_a["call_id"]), list(dev_b["call_id"]))
        self.assertEqual(len(dev_a), 5)
        self.assertEqual(len(train_a), 45)
        self.assertFalse(set(train_a["call_id"]) & set(dev_a["call_id"]))
        self.assertEqual(set(train_a["call_id"]) | set(dev_a["call_id"]), set(frame["call_id"]))
        # 각 부분은 원본 순서를 유지한다
        self.assertEqual(list(dev_a["call_id"]), sorted(dev_a["call_id"]))
        _, dev_other = make_dev_split.split_frame(frame, 0.1, 99)
        self.assertNotEqual(list(dev_a["call_id"]), list(dev_other["call_id"]))

    def test_main_writes_csvs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "train.csv"
            _frame(40).to_csv(source, index=False, encoding="utf-8-sig")
            out = Path(tmp) / "split"
            self.assertEqual(make_dev_split.main(["--train-csv", str(source), "--out-dir", str(out)]), 0)
            manifest = json.loads((out / "split.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["train_rows"] + manifest["dev_rows"], 40)
            self.assertEqual(manifest["dev_rows"], 4)
            dev = pd.read_csv(out / "dev_split.csv", encoding="utf-8-sig")
            self.assertEqual(list(dev.columns)[:2], ["call_id", "text"])

    def test_rejects_duplicate_call_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame = _frame(20)
            frame.loc[1, "call_id"] = frame.loc[0, "call_id"]
            source = Path(tmp) / "train.csv"
            frame.to_csv(source, index=False, encoding="utf-8-sig")
            with self.assertRaises(ValueError):
                make_dev_split.main(["--train-csv", str(source), "--out-dir", str(Path(tmp) / "o")])


class RuleTest(unittest.TestCase):
    def test_argmax_first_breaks_ties_by_order(self) -> None:
        self.assertEqual(devsel.argmax_first({"0": 0.60, "0.5": 0.65, "1.0": 0.62}), "0.5")
        self.assertEqual(devsel.argmax_first({"original": 0.65, "tapt_llrd": 0.65}), "original")

    def test_choose_epoch_prefers_later_on_tie(self) -> None:
        self.assertEqual(devsel.choose_epoch({1: 0.60, 2: 0.66, 3: 0.65}), 2)
        self.assertEqual(devsel.choose_epoch({1: 0.60, 2: 0.66, 3: 0.66}), 3)

    def test_choose_tfidf_tie_rules(self) -> None:
        table = [
            {"C": 0.5, "w": 0.3, "f1": 0.70},
            {"C": 0.15, "w": 0.3, "f1": 0.70},
            {"C": 0.05, "w": 0.2, "f1": 0.70},
            {"C": 1.5, "w": 0.1, "f1": 0.69},
        ]
        self.assertEqual(devsel.choose_tfidf(table), {"C": 0.05, "w": 0.2, "f1": 0.70})
        self.assertEqual(devsel.choose_tfidf(table[:2]), {"C": 0.15, "w": 0.3, "f1": 0.70})

    def test_blend_matches_submission_formula(self) -> None:
        t = np.array([[0.2, 0.8]])
        f = np.array([[1.0, 0.0]])
        np.testing.assert_allclose(devsel.blend(t, f, 0.3), [[0.44, 0.56]])
        np.testing.assert_allclose(devsel.blend(t, f, 0.0), t)
        np.testing.assert_allclose(devsel.blend(t, None, 0.3), t)

    def test_decide_members(self) -> None:
        labels = np.array([[1] * 9, [0] * 9])
        # 각 seed 는 한 행씩 틀리고 평균은 둘 다 맞힌다 -> 앙상블이 낫다
        seed_a = np.array([[0.9] * 9, [0.6] * 9])
        seed_b = np.array([[0.4] * 9, [0.1] * 9])
        result = devsel.decide_members([seed_a, seed_b], None, 0.0, labels)
        self.assertEqual(result["n_members"], 4)
        same = np.array([[0.9] * 9, [0.1] * 9])
        self.assertEqual(devsel.decide_members([same, same], None, 0.0, labels)["n_members"], 1)


class RunGuardTest(unittest.TestCase):
    def _run(self, root: Path, train_csv: str, val_csv: str) -> Path:
        run = root / "run"
        run.mkdir()
        (run / "run_config.json").write_text(json.dumps({"train_csv": train_csv, "val_csv": val_csv}), encoding="utf-8")
        return run

    def test_rejects_run_evaluated_on_other_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("train_split.csv", "dev_split.csv", "mission3_val.csv"):
                (root / name).write_text("x", encoding="utf-8")
            run = self._run(root, str(root / "train_split.csv"), str(root / "mission3_val.csv"))
            with self.assertRaises(ValueError):
                devsel.check_run_split(run, root / "train_split.csv", root / "dev_split.csv")
            (run / "run_config.json").write_text(
                json.dumps({"train_csv": str(root / "train_split.csv"), "val_csv": str(root / "dev_split.csv")}),
                encoding="utf-8")
            devsel.check_run_split(run, root / "train_split.csv", root / "dev_split.csv")

    def test_epoch_probs_must_match_recorded_f1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            labels = np.array([[1] * 9, [0] * 9])
            probs = np.array([[0.9] * 9, [0.1] * 9])
            np.save(run / "val_probs_epoch1.npy", probs)
            history = {"epochs": [{"epoch": 1, "val_macro_f1_at_0_5": devsel.macro_f1_at_half(labels, probs)}]}
            (run / "history.json").write_text(json.dumps(history), encoding="utf-8")
            self.assertIn(1, devsel.load_epoch_probs(run, labels))
            np.save(run / "val_probs_epoch1.npy", probs[::-1])  # 행 순서가 어긋난 경우
            with self.assertRaises(ValueError):
                devsel.load_epoch_probs(run, labels)


def _fake_run(root: Path, name: str, value: float = 0.0) -> Path:
    """config·토크나이저 파일과 작은 가중치 하나만 있는 최종 run (번들 조립만 확인한다)."""
    import torch
    from safetensors.torch import save_file

    model = root / name / "best_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "inference_config.json").write_text(json.dumps({"utterance_sep_mode": "space"}), encoding="utf-8")
    save_file({"w": torch.full((2, 3), value)}, str(model / "model.safetensors"))
    return root / name


class AssembleTest(unittest.TestCase):
    def test_assemble_writes_loadable_pt_bundle(self) -> None:
        import gc

        import torch

        from m3.bundle import load_bundle

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            members = []
            for index, seed in enumerate((42, 43)):
                members += ["--member", str(_fake_run(root, f"final_s{seed}", value=float(index)))]
            tfidf = root / "tfidf" / "tfidf_lr.joblib"
            tfidf.parent.mkdir()
            tfidf.write_bytes(b"joblib-bytes")
            out = root / "mission3.pt"
            devsel.main(["assemble", *members, "--tfidf", str(tfidf), "--tfidf-weight", "0.2", "--out", str(out)])

            bundle = load_bundle(out)
            self.assertEqual([m.name for m in bundle.members], ["final_s42", "final_s43"])
            self.assertEqual(bundle.precision, "fp16")          # 기본값: CUDA 추론 fp16
            self.assertEqual(bundle.tfidf_weight, 0.2)
            self.assertEqual(bundle.tfidf_joblib, b"joblib-bytes")
            self.assertIn("dev_selection_protocol", bundle.note)
            self.assertEqual(sorted(bundle.members[0].files), ["config.json", "inference_config.json", "tokenizer.json"])
            self.assertTrue(torch.equal(bundle.members[1].state_dict["w"], torch.ones(2, 3)))
            self.assertEqual(bundle.members[0].inference_config(), {"utterance_sep_mode": "space"})
            del bundle
            gc.collect()

    def test_assemble_without_tfidf_when_weight_zero(self) -> None:
        import gc

        from m3.bundle import load_bundle

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            run = _fake_run(root, "final_s42")
            out = root / "mission3.pt"
            devsel.main(["assemble", "--member", str(run), "--tfidf", str(root / "none.joblib"),
                         "--tfidf-weight", "0", "--precision", "fp32", "--out", str(out)])
            bundle = load_bundle(out)
            self.assertIsNone(bundle.tfidf_joblib)
            self.assertEqual(bundle.tfidf_weight, 0.0)
            self.assertEqual(bundle.precision, "fp32")
            del bundle
            gc.collect()

    def test_assemble_refuses_non_pt_output_and_existing_file(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            run = _fake_run(root, "final_s42")
            with self.assertRaisesRegex(ValueError, ".pt"):
                devsel.main(["assemble", "--member", str(run), "--out", str(root / "bundle")])
            existing = root / "mission3.pt"
            existing.write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                devsel.main(["assemble", "--member", str(run), "--out", str(existing)])
            self.assertEqual(existing.read_bytes(), b"old")

    def test_assemble_refuses_member_without_weights(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            model = root / "final_s42" / "best_model"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                devsel.main(["assemble", "--member", str(root / "final_s42"), "--out", str(root / "m.pt")])


if __name__ == "__main__":
    unittest.main()
