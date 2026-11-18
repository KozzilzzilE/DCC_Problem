"""Unit tests for pre-cue prefix weighting, leak prevention, and exact checkpoint/resume parity."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_precue as tp  # noqa: E402
from m4.data import Call  # noqa: E402

TAPT_DIR = Path("/Volumes/로컬 디스크/DCC/private_experiments/m4_precue_bce_v1_20261116/assets/mission3_symptom/runs/devsel/tapt")


class TestTrainPrecue(unittest.TestCase):
    def test_bce_parity_and_precue_damping(self):
        rng = np.random.default_rng(7)
        logits = torch.tensor(rng.normal(size=(8, 9)), dtype=torch.float32)
        target = torch.tensor(rng.integers(0, 2, size=(8, 9)), dtype=torch.float32)
        pw = torch.tensor([2.1, 2.3, 3.0, 1.8, 1.9, 2.7, 2.8, 2.1, 2.5], dtype=torch.float32)

        ref = torch.nn.BCEWithLogitsLoss(pos_weight=pw)(logits, target)
        got = tp.compute_prefix_bce_loss(logits, target, pw, None, -1.0)
        self.assertLess(abs(float(ref) - float(got)), 1e-6)

        # Compare pre-cue mask behavior directly
        mask = torch.zeros((8, 9), dtype=torch.bool)
        mask[:, 0] = True
        loss_precue = tp.compute_prefix_bce_loss(logits, target, pw, mask, 1.0)
        elem = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pw_exp = pw.unsqueeze(0).expand_as(target).clone()
        pw_exp[:, 0] = 1.0
        manual = (elem * torch.where(target > 0.5, pw_exp, 1.0)).mean()
        self.assertLess(abs(float(loss_precue) - float(manual)), 1e-6)

        # Compare SoftJaccard addition against explicit BCE + 1.0 * (1 - SoftIoU)
        target_nonempty = target.clone()
        target_nonempty[:, 0] = 1.0
        loss_jac = tp.compute_prefix_bce_loss(logits, target_nonempty, pw, None, -1.0, jaccard_weight=1.0)
        bce_ref = torch.nn.BCEWithLogitsLoss(pos_weight=pw)(logits, target_nonempty)
        pr = torch.sigmoid(logits)
        jac_ref = (1.0 - (pr * target_nonempty).sum(1) / (pr + target_nonempty - pr * target_nonempty).sum(1)).mean()
        self.assertLess(abs(float(loss_jac) - float(bce_ref + jac_ref)), 1e-6)

        loss_jm = tp.compute_prefix_bce_loss(logits, target_nonempty, pw, None, -1.0, jaccard_weight=0.5, macro_f1_weight=0.5)
        tp_c = (pr * target_nonempty).sum(0)
        macro_ref = 1.0 - ((2.0 * tp_c + 1.0) / (pr.sum(0) + target_nonempty.sum(0) + 1.0)).mean()
        self.assertLess(abs(float(loss_jm) - float(bce_ref + 0.5 * jac_ref + 0.5 * macro_ref)), 1e-6)

    def test_cue_steps_and_leak_check(self):
        c1 = Call("c1", ["네 119입니다", "배가 아파요", "토도 했어요"], [0, 1, 1], [0, 10, 20], [5, 15, 25], [0, 1, 0, 1, 0, 0, 0, 0, 0])
        c2 = Call("c2", ["안녕하세요", "기운이 없어요"], [0, 1], [0, 10], [5, 15], [0, 0, 0, 0, 0, 0, 0, 1, 0])
        cs = tp.cue_steps_matrix([c1, c2])
        # c1: 구토(idx 1) at step 3, 복통(idx 3) at step 2
        self.assertEqual(int(cs[0, 1]), 3)
        self.assertEqual(int(cs[0, 3]), 2)
        self.assertEqual(int(cs[1, 7]), 2)
        self.assertGreater(int(cs[0, 0]), 10**8)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            calls_path = p / "calls.jsonl"
            calls_path.write_text(
                "\n".join(json.dumps(c.__dict__, ensure_ascii=False) for c in (c1, c2)) + "\n",
                encoding="utf-8",
            )
            forbid = p / "forbid.txt"
            forbid.write_text("c2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                tp.verify_non_overlapping([c1, c2], calls_path, forbid)

    @unittest.skipUnless(TAPT_DIR.exists(), "TAPT backbone directory not staged")
    def test_stop_and_resume_bit_identical_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # Make 32 synthetic calls (32 * 2 = 64 samples => 4 batches of 16 per epoch)
            calls = []
            for idx in range(32):
                labs = [0] * 9
                labs[idx % 9] = 1
                calls.append(
                    Call(
                        f"syn_{idx}",
                        ["여보세요 119입니다", "배가 아파요 열이 나요", "감사합니다"],
                        [0, 1, 0],
                        [0, 1000, 2000],
                        [900, 1900, 2900],
                        labs,
                    )
                )
            calls_path = p / "calls.jsonl"
            calls_path.write_text("\n".join(json.dumps(c.__dict__, ensure_ascii=False) for c in calls) + "\n", encoding="utf-8")
            ids_path = p / "fit_ids.txt"
            ids_path.write_text("\n".join(c.call_id for c in calls) + "\n", encoding="utf-8")

            # Uninterrupted 1-epoch (4 steps)
            out_full = p / "full"
            args_full = tp.build_parser().parse_args([
                "--calls", str(calls_path), "--ids", str(ids_path), "--backbone", str(TAPT_DIR),
                "--out", str(out_full), "--epochs", "1", "--batch-size", "16", "--precue-pos-weight", "1.0",
                "--seed", "42", "--cpu", "--log-every", "2",
            ])
            self.assertEqual(tp.run_training(args_full), 0)

            # Interrupted at step 2, then resumed to completion
            out_res = p / "resumed"
            args_stop = tp.build_parser().parse_args([
                "--calls", str(calls_path), "--ids", str(ids_path), "--backbone", str(TAPT_DIR),
                "--out", str(out_res), "--epochs", "1", "--batch-size", "16", "--precue-pos-weight", "1.0",
                "--seed", "42", "--cpu", "--log-every", "2", "--stop-after-steps", "2",
            ])
            self.assertEqual(tp.run_training(args_stop), 0)
            self.assertTrue((out_res / "training_state_last.pt").exists())
            self.assertFalse((out_res / "final.pt").exists())

            args_cont = tp.build_parser().parse_args([
                "--calls", str(calls_path), "--ids", str(ids_path), "--backbone", str(TAPT_DIR),
                "--out", str(out_res), "--epochs", "1", "--batch-size", "16", "--precue-pos-weight", "1.0",
                "--seed", "42", "--cpu", "--log-every", "2", "--resume",
            ])
            self.assertEqual(tp.run_training(args_cont), 0)

            cfg_f = json.loads((out_full / "run_config.json").read_text())
            cfg_r = json.loads((out_res / "run_config.json").read_text())
            self.assertEqual(cfg_f["history"][0]["train_loss"], cfg_r["history"][0]["train_loss"])
            st_f = torch.load(out_full / "final.pt", map_location="cpu", weights_only=True)
            st_r = torch.load(out_res / "final.pt", map_location="cpu", weights_only=True)
            self.assertEqual(st_f["state_hash"], st_r["state_hash"])


if __name__ == "__main__":
    unittest.main()
