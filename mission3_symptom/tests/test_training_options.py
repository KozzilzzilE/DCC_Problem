"""학습 옵션 단위 테스트: layer-wise LR decay(`--llrd-decay`)와 `--gradient-checkpointing`.

- LLRD 는 분류 헤드에 기본 LR 을 주고, 인코더 층을 내려갈수록 decay 를 한 번씩 곱한다.
  임베딩은 맨 아래 층보다 한 단계 더 낮다. 1.0(기본값)이면 기존 두 그룹 동작 그대로다.
- gradient checkpointing 은 8GB GPU 에서 large 백본을 돌리기 위한 메모리 옵션이다.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from transformers import RobertaConfig, RobertaForSequenceClassification

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

from m3 import training
from m3.config import TARGET_SYMPTOMS
from m3.training import TrainingConfig, _build_optimizer, _validate_config

import train as train_entry

REQUIRED_ARGS = ["train.py", "--train-csv", "train.csv", "--val-csv", "val.csv", "--output-dir", "output"]


def tiny_roberta(num_layers: int = 2) -> RobertaForSequenceClassification:
    config = RobertaConfig(
        vocab_size=50, hidden_size=8, num_hidden_layers=num_layers, num_attention_heads=2,
        intermediate_size=16, max_position_embeddings=40, num_labels=len(TARGET_SYMPTOMS),
    )
    return RobertaForSequenceClassification(config)


def lr_by_name(model, optimizer) -> dict:
    ids = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            ids[id(parameter)] = (group["lr"], group["weight_decay"])
    return {name: ids[id(parameter)] for name, parameter in model.named_parameters()}


def make_config(**overrides) -> TrainingConfig:
    return TrainingConfig(train_csv="train.csv", val_csv="val.csv", output_dir="output", **overrides)


class LayerwiseLrDecayTest(unittest.TestCase):
    def test_default_keeps_single_learning_rate(self) -> None:
        model = tiny_roberta()
        optimizer = _build_optimizer(model, 1e-4, 0.01)

        rates = {lr for lr, _ in lr_by_name(model, optimizer).values()}
        self.assertEqual(rates, {1e-4})
        self.assertEqual(len(optimizer.param_groups), 2)

    def test_decay_lowers_rate_per_layer_from_head_to_embeddings(self) -> None:
        model = tiny_roberta(num_layers=2)
        optimizer = _build_optimizer(model, 1e-4, 0.01, llrd_decay=0.5)
        rates = lr_by_name(model, optimizer)

        self.assertAlmostEqual(rates["classifier.dense.weight"][0], 1e-4)
        self.assertAlmostEqual(rates["classifier.out_proj.bias"][0], 1e-4)
        self.assertAlmostEqual(rates["roberta.encoder.layer.1.attention.self.query.weight"][0], 0.5e-4)
        self.assertAlmostEqual(rates["roberta.encoder.layer.0.output.dense.weight"][0], 0.25e-4)
        self.assertAlmostEqual(rates["roberta.embeddings.word_embeddings.weight"][0], 0.125e-4)

    def test_decay_keeps_weight_decay_exclusions(self) -> None:
        model = tiny_roberta()
        optimizer = _build_optimizer(model, 1e-4, 0.01, llrd_decay=0.5)
        rates = lr_by_name(model, optimizer)

        self.assertEqual(rates["roberta.encoder.layer.0.output.dense.weight"][1], 0.01)
        self.assertEqual(rates["roberta.encoder.layer.0.output.dense.bias"][1], 0.0)
        self.assertEqual(rates["roberta.encoder.layer.0.output.LayerNorm.weight"][1], 0.0)

    def test_every_trainable_parameter_is_optimized_once(self) -> None:
        model = tiny_roberta()
        optimizer = _build_optimizer(model, 1e-4, 0.01, llrd_decay=0.8)

        grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.parameters() if p.requires_grad})

    def test_backbone_params_outside_layers_get_embedding_rate(self) -> None:
        # DeBERTa-v2 의 공유 상대위치 임베딩·인코더 LayerNorm 은 층 번호가 없지만 백본이다.
        # 헤드 학습률(가장 큰 값)이 아니라 임베딩과 같은 가장 낮은 학습률을 받아야 한다.
        from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

        config = DebertaV2Config(
            vocab_size=50, hidden_size=8, num_hidden_layers=2, num_attention_heads=2, intermediate_size=16,
            max_position_embeddings=40, relative_attention=True, position_biased_input=False,
            norm_rel_ebd="layer_norm", num_labels=len(TARGET_SYMPTOMS),
        )
        model = DebertaV2ForSequenceClassification(config)
        rates = lr_by_name(model, _build_optimizer(model, 1e-4, 0.01, llrd_decay=0.5))

        self.assertAlmostEqual(rates["deberta.encoder.rel_embeddings.weight"][0], 0.125e-4)
        self.assertAlmostEqual(rates["deberta.embeddings.word_embeddings.weight"][0], 0.125e-4)
        self.assertAlmostEqual(rates["deberta.encoder.layer.1.attention.self.query_proj.weight"][0], 0.5e-4)
        self.assertAlmostEqual(rates["classifier.weight"][0], 1e-4)
        self.assertAlmostEqual(rates["pooler.dense.weight"][0], 1e-4)

    def test_config_rejects_decay_outside_zero_one(self) -> None:
        for bad in (0.0, -0.1, 1.5, math.nan, math.inf):
            with self.subTest(decay=bad):
                with self.assertRaisesRegex(ValueError, "llrd_decay"):
                    _validate_config(make_config(llrd_decay=bad))


class TrainingOptionCliTest(unittest.TestCase):
    def test_cli_defaults_are_off(self) -> None:
        with patch.object(sys, "argv", REQUIRED_ARGS):
            config = train_entry.build_config(train_entry.parse_args())

        self.assertEqual(config.llrd_decay, 1.0)
        self.assertFalse(config.gradient_checkpointing)

    def test_cli_passes_options_to_config(self) -> None:
        argv = [*REQUIRED_ARGS, "--llrd-decay", "0.9", "--gradient-checkpointing"]
        with patch.object(sys, "argv", argv):
            config = train_entry.build_config(train_entry.parse_args())

        self.assertEqual(config.llrd_decay, 0.9)
        self.assertTrue(config.gradient_checkpointing)


class RunTrainingOptionWiringTest(unittest.TestCase):
    def _run_until_loss(self, **overrides):
        class _Stop(Exception):
            pass

        frame = pd.DataFrame({"text": ["a"] * 5, **{s: [1, 0, 0, 0, 0] for s in TARGET_SYMPTOMS}})
        model = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(training, "load_symptom_csv", return_value=frame), \
                patch.object(training, "verify_utterance_sep_mode"), \
                patch.object(training, "build_tokenizer_and_model", return_value=(MagicMock(), model)), \
                patch.object(training, "calculate_token_length_stats", return_value={}), \
                patch.object(training, "_create_train_val_loaders", return_value=(None, None, None)), \
                patch.object(training, "build_loss", side_effect=_Stop):
            config = TrainingConfig(
                train_csv="train.csv", val_csv="val.csv", output_dir=str(Path(tmp) / "out"),
                device="cpu", **overrides,
            )
            with self.assertRaises(_Stop):
                training.run_training(config)
        return model

    def test_gradient_checkpointing_is_enabled_on_request(self) -> None:
        model = self._run_until_loss(gradient_checkpointing=True)

        model.gradient_checkpointing_enable.assert_called_once()

    def test_gradient_checkpointing_is_off_by_default(self) -> None:
        model = self._run_until_loss()

        model.gradient_checkpointing_enable.assert_not_called()


if __name__ == "__main__":
    unittest.main()
