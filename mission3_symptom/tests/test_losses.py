"""BCE/ASL loss 선택과 multi-label tensor 처리 테스트."""

from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

MISSION3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MISSION3_DIR))

from m3.config import NUM_CLASSES
from m3.losses import AsymmetricLoss, LabelDependencyLoss, build_loss
from m3.training import TrainingConfig, _validate_config


class LossTest(unittest.TestCase):
    def test_asl_finite_and_backward(self) -> None:
        logits = torch.randn(4, NUM_CLASSES, requires_grad=True)
        targets = torch.randint(0, 2, (4, NUM_CLASSES), dtype=torch.float32)

        loss = AsymmetricLoss()(logits, targets)
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_asl_none_reduction_preserves_batch_and_label_shape(self) -> None:
        logits = torch.zeros(3, NUM_CLASSES)
        targets = torch.zeros_like(logits)
        loss = AsymmetricLoss(reduction="none")(logits, targets)
        self.assertEqual(loss.shape, (3, NUM_CLASSES))

    def test_asl_without_focusing_or_clipping_matches_plain_bce(self) -> None:
        logits = torch.randn(4, NUM_CLASSES)
        targets = torch.randint(0, 2, logits.shape, dtype=torch.float32)
        asl = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0)(logits, targets)
        bce = torch.nn.BCEWithLogitsLoss()(logits, targets)
        self.assertTrue(torch.allclose(asl, bce, atol=1e-6, rtol=1e-6))

    def test_asl_clipping_discards_very_easy_negative(self) -> None:
        logits = torch.tensor([[-6.0]])
        targets = torch.zeros_like(logits)
        loss = AsymmetricLoss(clip=0.05)(logits, targets)
        self.assertEqual(loss.item(), 0.0)

    def test_asl_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            AsymmetricLoss()(
                torch.zeros(2, NUM_CLASSES),
                torch.zeros(2, NUM_CLASSES - 1),
            )

    def test_bce_selection_preserves_plain_and_pos_weight_behavior(self) -> None:
        logits = torch.tensor([[0.5, -0.5]])
        targets = torch.tensor([[1.0, 0.0]])
        pos_weight = torch.tensor([2.0, 3.0])

        plain = build_loss("bce")(logits, targets)
        weighted = build_loss("bce", pos_weight=pos_weight)(logits, targets)

        self.assertTrue(torch.equal(
            plain,
            torch.nn.BCEWithLogitsLoss()(logits, targets),
        ))
        self.assertTrue(torch.equal(
            weighted,
            torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets),
        ))

    def test_asl_config_is_serializable_and_bce_remains_default(self) -> None:
        bce_config = TrainingConfig("train.csv", "val.csv", "output")
        asl_config = TrainingConfig(
            "train.csv",
            "val.csv",
            "output",
            loss_type="asl",
        )

        self.assertEqual(bce_config.loss_type, "bce")
        serialized = asdict(asl_config)
        self.assertEqual(serialized["loss_type"], "asl")
        self.assertEqual(serialized["asl_gamma_neg"], 4.0)
        self.assertEqual(serialized["asl_gamma_pos"], 1.0)
        self.assertEqual(serialized["asl_clip"], 0.05)
        self.assertEqual(serialized["asl_eps"], 1e-8)
        self.assertEqual(serialized["asl_reduction"], "mean")
        self.assertTrue(serialized["asl_disable_focal_loss_grad"])

    def test_asl_and_pos_weight_cannot_be_combined(self) -> None:
        config = TrainingConfig(
            "train.csv",
            "val.csv",
            "output",
            loss_type="asl",
            use_pos_weight=True,
        )
        with self.assertRaisesRegex(ValueError, "use_pos_weight"):
            _validate_config(config)

    def test_dependency_loss_finite_and_backward(self) -> None:
        logits = torch.randn(4, NUM_CLASSES, requires_grad=True)
        targets = torch.randint(0, 2, (4, NUM_CLASSES), dtype=torch.float32)
        co_occ = torch.rand(NUM_CLASSES, NUM_CLASSES)
        
        loss = LabelDependencyLoss(co_occurrence_matrix=co_occ)(logits, targets)
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_dependency_loss_zero_alpha_matches_bce(self) -> None:
        logits = torch.randn(4, NUM_CLASSES)
        targets = torch.randint(0, 2, logits.shape, dtype=torch.float32)
        co_occ = torch.rand(NUM_CLASSES, NUM_CLASSES)
        
        dep_loss = LabelDependencyLoss(co_occurrence_matrix=co_occ, alpha=0.0)(logits, targets)
        bce = torch.nn.BCEWithLogitsLoss()(logits, targets)
        self.assertTrue(torch.allclose(dep_loss, bce, atol=1e-6, rtol=1e-6))



if __name__ == "__main__":
    unittest.main()
