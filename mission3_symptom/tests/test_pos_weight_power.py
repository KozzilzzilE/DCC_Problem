"""pos_weight 거듭제곱(`--pos-weight-power`) 단위 테스트.

결정 임계값이 0.5 로 고정이라, 학습 손실의 양성 가중으로 모델 출력 자체를 옮겨야 한다.
Training 라벨의 `negative / positive` 를 그대로 쓰면(power=1) 모든 클래스가 과보정되고,
power=0.5 가 로컬 실측 최적이었다. 가중치는 Training 라벨 개수로만 계산한다.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

from m3.config import NUM_CLASSES, TARGET_SYMPTOMS
from m3.training import TrainingConfig, _calculate_pos_weights, _validate_config

import train as train_entry


def one_positive_in_five() -> pd.DataFrame:
    """모든 클래스가 양성 1건, 음성 4건 -> negative/positive = 4."""
    return pd.DataFrame({symptom: [1, 0, 0, 0, 0] for symptom in TARGET_SYMPTOMS})


def make_config(**overrides) -> TrainingConfig:
    return TrainingConfig(train_csv="train.csv", val_csv="val.csv", output_dir="output", **overrides)


class PosWeightPowerTest(unittest.TestCase):
    def test_default_power_keeps_negative_over_positive(self) -> None:
        weights, _ = _calculate_pos_weights(one_positive_in_five(), torch.device("cpu"))

        self.assertTrue(torch.allclose(weights, torch.full((NUM_CLASSES,), 4.0)))

    def test_power_raises_ratio_to_the_given_exponent(self) -> None:
        weights, statistics = _calculate_pos_weights(
            one_positive_in_five(), torch.device("cpu"), power=0.5
        )

        self.assertTrue(torch.allclose(weights, torch.full((NUM_CLASSES,), 2.0)))
        for symptom in TARGET_SYMPTOMS:
            self.assertAlmostEqual(statistics[symptom]["negative_over_positive"], 4.0)
            self.assertAlmostEqual(statistics[symptom]["pos_weight"], 2.0)
            self.assertAlmostEqual(statistics[symptom]["pos_weight_power"], 0.5)

    def test_power_zero_is_plain_bce(self) -> None:
        weights, _ = _calculate_pos_weights(one_positive_in_five(), torch.device("cpu"), power=0.0)

        self.assertTrue(torch.allclose(weights, torch.ones(NUM_CLASSES)))

    def test_config_default_power_is_one(self) -> None:
        self.assertEqual(make_config().pos_weight_power, 1.0)

    def test_config_rejects_negative_or_non_finite_power(self) -> None:
        for bad in (-0.5, math.inf, math.nan):
            with self.subTest(power=bad):
                with self.assertRaisesRegex(ValueError, "pos_weight_power"):
                    _validate_config(make_config(use_pos_weight=True, pos_weight_power=bad))

    def test_config_rejects_power_without_pos_weight(self) -> None:
        # --use-pos-weight 없이 power 만 바꾸면 조용히 무시되어 실험 기록이 거짓이 된다.
        with self.assertRaisesRegex(ValueError, "use_pos_weight"):
            _validate_config(make_config(use_pos_weight=False, pos_weight_power=0.5))

    def test_cli_passes_power_to_config(self) -> None:
        required = ["train.py", "--train-csv", "train.csv", "--val-csv", "val.csv", "--output-dir", "output"]
        with patch.object(sys, "argv", required):
            default_config = train_entry.build_config(train_entry.parse_args())
        with patch.object(sys, "argv", [*required, "--use-pos-weight", "--pos-weight-power", "0.5"]):
            power_config = train_entry.build_config(train_entry.parse_args())

        self.assertEqual(default_config.pos_weight_power, 1.0)
        self.assertTrue(power_config.use_pos_weight)
        self.assertEqual(power_config.pos_weight_power, 0.5)


if __name__ == "__main__":
    unittest.main()
