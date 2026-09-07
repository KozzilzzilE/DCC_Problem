"""Best Checkpoint 선정 기준(val_loss vs val_macro_f1) 단위 테스트."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

# 로컬 테스트 환경에 torch/transformers가 없을 경우를 대비한 안전한 Mocking
for mod in [
    "torch",
    "torch.nn",
    "torch.nn.utils",
    "torch.utils",
    "torch.utils.data",
    "transformers",
    "sentencepiece",
    "pandas",
    "tqdm",
    "tqdm.auto",
]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from m3.training import TrainingConfig, _validate_config


class CheckpointMetricTest(unittest.TestCase):
    def test_default_checkpoint_metric_is_val_loss(self) -> None:
        """기존 코드와 100% 호환되도록 기본값은 val_loss여야 함."""
        config = TrainingConfig(
            train_csv="train.csv",
            val_csv="val.csv",
            output_dir="output",
        )
        self.assertEqual(config.checkpoint_metric, "val_loss")

    def test_custom_checkpoint_metric_val_macro_f1(self) -> None:
        """대회 공식 평가지표인 val_macro_f1 옵션 정상 설정 확인."""
        config = TrainingConfig(
            train_csv="train.csv",
            val_csv="val.csv",
            output_dir="output",
            checkpoint_metric="val_macro_f1",
        )
        self.assertEqual(config.checkpoint_metric, "val_macro_f1")

    def test_invalid_checkpoint_metric_raises_error(self) -> None:
        """지원하지 않는 이상한 메트릭 이름 입력 시 ValueError 발생 확인."""
        config = TrainingConfig(
            train_csv="train.csv",
            val_csv="val.csv",
            output_dir="output",
            checkpoint_metric="accuracy",
        )
        with self.assertRaises(ValueError) as ctx:
            _validate_config(config)
        self.assertIn("지원하지 않는 checkpoint_metric입니다", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
