"""학습 옵션 단위 테스트: layer-wise LR decay(`--llrd-decay`)와 train.py CLI.

- LLRD 는 분류 헤드에 기본 LR 을 주고, 인코더 층을 내려갈수록 decay 를 한 번씩 곱한다.
  임베딩은 맨 아래 층보다 한 단계 더 낮다. 1.0(기본값)이면 기존 두 그룹 동작 그대로다.
- 학습 loader 는 seed 로 섞고, 평가 loader 는 CSV 행 순서를 지킨다 (devsel.py 가 이 순서에 기댄다).
- run_dev_selection.sh 가 쓰는 옵션이 모두 TrainingConfig 로 그대로 옮겨지는지, 정리해 뺀 실험 옵션은
  거부되는지 고정한다.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import RobertaConfig, RobertaForSequenceClassification

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

from m3.config import TARGET_SYMPTOMS
from m3.training import TrainingConfig, _build_optimizer, _create_train_val_loaders, _validate_config

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

    def test_only_classifier_gets_head_rate(self) -> None:
        # RoBERTa 백본 파라미터는 모두 인코더 층이나 임베딩이라 헤드 학습률(가장 큰 값)을 받지 않는다.
        model = tiny_roberta(num_layers=2)
        rates = lr_by_name(model, _build_optimizer(model, 1e-4, 0.01, llrd_decay=0.5))

        for name, (rate, _) in rates.items():
            with self.subTest(name=name):
                if name.startswith("classifier."):
                    self.assertAlmostEqual(rate, 1e-4)
                else:
                    self.assertTrue(name.startswith("roberta."), name)
                    self.assertLess(rate, 1e-4)

    def test_config_rejects_decay_outside_zero_one(self) -> None:
        for bad in (0.0, -0.1, 1.5, math.nan, math.inf):
            with self.subTest(decay=bad):
                with self.assertRaisesRegex(ValueError, "llrd_decay"):
                    _validate_config(make_config(llrd_decay=bad))


class RowIndexTokenizer:
    """본문 '행 N' 을 [1, N, 2] 로 바꾸는 더미 tokenizer. 배치에서 원래 행 번호를 읽기 위한 것이다."""

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, int(text.split()[-1]), 2], "attention_mask": [1, 1, 1]}

    def pad(self, features, padding=True, pad_to_multiple_of=None, return_tensors="pt"):
        return {key: torch.tensor([f[key] for f in features], dtype=torch.long) for key in features[0]}


def row_frame(n: int) -> pd.DataFrame:
    rows = []
    for index in range(n):
        row = {"call_id": f"call-{index}", "text": f"행 {index}"}
        row.update({symptom: index % 2 for symptom in TARGET_SYMPTOMS})
        rows.append(row)
    return pd.DataFrame(rows)


def row_order(loader) -> list:
    return [int(value) for batch in loader for value in batch["input_ids"][:, 1]]


class TrainValLoaderTest(unittest.TestCase):
    """devsel.py 는 epoch 별 평가 확률(val_probs_epoch*.npy)을 dev CSV 행 순서로 읽는다.

    그래서 평가 loader 는 섞지 않고(SequentialSampler) CSV 순서대로 돌아야 하고, 학습 loader 만 seed 로 섞는다.
    """

    def _loaders(self, n: int = 12, seed: int = 42):
        config = make_config(seed=seed, train_batch_size=4, val_batch_size=5)
        return _create_train_val_loaders(row_frame(n), row_frame(n), RowIndexTokenizer(), config, pin_memory=False)

    def test_returns_train_then_validation_loader(self) -> None:
        train_loader, val_loader = self._loaders()

        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(val_loader.sampler, SequentialSampler)
        self.assertEqual((train_loader.batch_size, val_loader.batch_size), (4, 5))

    def test_validation_loader_keeps_csv_row_order(self) -> None:
        _, val_loader = self._loaders()

        self.assertEqual(row_order(val_loader), list(range(12)))
        self.assertEqual(row_order(val_loader), list(range(12)))  # 다시 돌려도 같은 순서

    def test_train_loader_shuffles_reproducibly_by_seed(self) -> None:
        first = row_order(self._loaders(seed=42)[0])
        second = row_order(self._loaders(seed=42)[0])

        self.assertEqual(sorted(first), list(range(12)))
        self.assertEqual(first, second)


class TrainingOptionCliTest(unittest.TestCase):
    def _config(self, *extra: str) -> TrainingConfig:
        with patch.object(sys, "argv", [*REQUIRED_ARGS, *extra]):
            return train_entry.build_config(train_entry.parse_args())

    def test_cli_defaults(self) -> None:
        config = self._config()

        self.assertEqual(config.llrd_decay, 1.0)
        self.assertEqual(config.encode_mode, "truncate")
        self.assertEqual(config.utterance_sep_mode, "space")
        self.assertEqual(config.checkpoint_metric, "fixed_epoch")
        self.assertIsNone(config.checkpoint_epoch)

    def test_final_training_command_reaches_config(self) -> None:
        """run_dev_selection.sh 7 단계(최종 학습)와 같은 인자가 그대로 설정이 된다."""
        config = self._config(
            "--model-name-or-path", "runs/devsel/tapt", "--local-files-only", "--learning-rate", "5e-5",
            "--llrd-decay", "0.8", "--use-pos-weight", "--pos-weight-power", "0.5", "--epochs", "3",
            "--checkpoint-metric", "fixed_epoch", "--checkpoint-epoch", "2", "--max-length", "512",
            "--amp", "--seed", "43",
        )

        self.assertEqual(config.model_name_or_path, "runs/devsel/tapt")
        self.assertTrue(config.local_files_only)
        self.assertEqual(config.learning_rate, 5e-5)
        self.assertEqual(config.llrd_decay, 0.8)
        self.assertTrue(config.use_pos_weight)
        self.assertEqual(config.pos_weight_power, 0.5)
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.checkpoint_metric, "fixed_epoch")
        self.assertEqual(config.checkpoint_epoch, 2)
        self.assertEqual(config.max_length, 512)
        self.assertTrue(config.amp)
        self.assertEqual(config.seed, 43)
        self.assertEqual((config.train_batch_size, config.gradient_accumulation_steps), (8, 2))
        _validate_config(config)

    def test_removed_experiment_options_are_rejected(self) -> None:
        removed = (
            ["--gradient-checkpointing"], ["--loss-type", "asl"], ["--pooling-type", "label_attention"],
            ["--encode-mode", "head_tail"], ["--use-pure-nausea-sampling"],
            ["--smoke-test"], ["--max-steps", "2"], ["--max-train-samples", "8"], ["--max-val-samples", "8"],
        )
        for argv in removed:
            with self.subTest(argv=argv), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    self._config(*argv)

    def test_config_accepts_only_truncate_encoding(self) -> None:
        _validate_config(make_config(encode_mode="truncate"))
        with self.assertRaisesRegex(ValueError, "encode_mode"):
            _validate_config(make_config(encode_mode="head_tail"))


if __name__ == "__main__":
    unittest.main()
