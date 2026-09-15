from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler

MISSION3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MISSION3_DIR))

from m3.config import NUM_CLASSES, TARGET_SYMPTOMS
from m3.losses import build_loss
from m3.sampling import (
    build_pure_nausea_sampler,
    calculate_nausea_group_counts,
    classify_nausea_vomit_groups,
)
from m3.training import BASELINE_MODEL_NAME, TrainingConfig, _create_train_val_loaders
import train as train_entry


class DummyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [1, len(text), 2], "attention_mask": [1, 1, 1]}

    def pad(self, features, **kwargs):
        return {
            key: torch.tensor([feature[key] for feature in features], dtype=torch.long)
            for key in features[0]
        }


class TinyMultiLabelModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 8)
        self.classifier = torch.nn.Linear(8, NUM_CLASSES)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids).mean(dim=1)
        return SimpleNamespace(logits=self.classifier(hidden))


def make_group_dataframe() -> pd.DataFrame:
    rows = []
    for index, (nausea, vomit) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        row = {"call_id": f"synthetic-{index}", "text": f"synthetic text {index}"}
        row.update({symptom: 0 for symptom in TARGET_SYMPTOMS})
        row["오심"] = nausea
        row["구토"] = vomit
        rows.append(row)
    rows[2]["두통"] = 1
    return pd.DataFrame(rows)


class PureNauseaSamplingTest(unittest.TestCase):
    def test_baseline_defaults_are_explicit(self) -> None:
        config = TrainingConfig("train.csv", "val.csv", "output")
        self.assertEqual(BASELINE_MODEL_NAME, "klue/roberta-base")
        self.assertEqual(config.model_name_or_path, "klue/roberta-base")
        self.assertEqual(config.loss_type, "bce")
        self.assertFalse(config.use_pos_weight)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.encode_mode, "truncate")
        self.assertEqual(config.max_length, 512)
        self.assertFalse(config.use_pure_nausea_sampling)

    def test_group_classification_covers_a_b_c_d(self) -> None:
        dataframe = make_group_dataframe()
        self.assertEqual(
            classify_nausea_vomit_groups(dataframe).tolist(),
            ["A", "B", "C", "D"],
        )
        self.assertEqual(
            calculate_nausea_group_counts(dataframe),
            {"A": 1, "B": 1, "C": 1, "D": 1},
        )

    def test_cli_sampling_is_off_by_default_and_configurable(self) -> None:
        required = [
            "train.py",
            "--train-csv",
            "train.csv",
            "--val-csv",
            "val.csv",
            "--output-dir",
            "output",
        ]
        with patch.object(sys, "argv", required):
            default_config = train_entry.build_config(train_entry.parse_args())
        with patch.object(
            sys,
            "argv",
            [*required, "--use-pure-nausea-sampling", "--pure-nausea-weight", "1.75"],
        ):
            sampling_config = train_entry.build_config(train_entry.parse_args())

        self.assertFalse(default_config.use_pure_nausea_sampling)
        self.assertEqual(default_config.pure_nausea_weight, 1.5)
        self.assertTrue(sampling_config.use_pure_nausea_sampling)
        self.assertEqual(sampling_config.pure_nausea_weight, 1.75)

    def test_pure_nausea_weight_changes_sampling_probability(self) -> None:
        sampler, summary = build_pure_nausea_sampler(
            make_group_dataframe(), pure_nausea_weight=1.5, seed=42
        )
        probabilities = {
            group: stats["expected_sampling_probability"]
            for group, stats in summary["group_statistics"].items()
        }
        self.assertIsInstance(sampler, WeightedRandomSampler)
        self.assertAlmostEqual(probabilities["C"], 1.5 / 4.5)
        self.assertGreater(probabilities["C"], probabilities["A"])
        self.assertAlmostEqual(summary["label_exposure"]["두통"]["relative_exposure"], 4 / 3)

    def test_seed_42_sampler_is_reproducible(self) -> None:
        first, _ = build_pure_nausea_sampler(
            make_group_dataframe(), pure_nausea_weight=1.5, seed=42
        )
        second, _ = build_pure_nausea_sampler(
            make_group_dataframe(), pure_nausea_weight=1.5, seed=42
        )
        self.assertEqual(list(iter(first)), list(iter(second)))

    def test_sampling_off_keeps_existing_train_and_val_loader_path(self) -> None:
        dataframe = make_group_dataframe()
        config = TrainingConfig(
            "train.csv",
            "val.csv",
            "output",
            train_batch_size=2,
            val_batch_size=2,
        )
        train_loader, val_loader, summary = _create_train_val_loaders(
            dataframe, dataframe, DummyTokenizer(), config, pin_memory=False
        )
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(val_loader.sampler, SequentialSampler)
        self.assertFalse(summary["enabled"])
        self.assertEqual(train_loader.dataset.encode_mode, "truncate")
        self.assertEqual(val_loader.dataset.encode_mode, "truncate")

    def test_sampling_on_applies_only_to_training_loader(self) -> None:
        dataframe = make_group_dataframe()
        config = TrainingConfig(
            "train.csv",
            "val.csv",
            "output",
            train_batch_size=2,
            val_batch_size=2,
            use_pure_nausea_sampling=True,
            pure_nausea_weight=1.5,
        )
        train_loader, val_loader, summary = _create_train_val_loaders(
            dataframe, dataframe, DummyTokenizer(), config, pin_memory=False
        )
        self.assertIsInstance(train_loader.sampler, WeightedRandomSampler)
        self.assertIsInstance(val_loader.sampler, SequentialSampler)
        self.assertEqual(len(train_loader.sampler), len(dataframe))
        self.assertTrue(train_loader.sampler.replacement)
        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["source"], "training_rows_only")
        self.assertEqual(train_loader.dataset.encode_mode, "truncate")
        self.assertEqual(val_loader.dataset.encode_mode, "truncate")

    def test_one_batch_sampling_smoke_forward_and_plain_bce(self) -> None:
        dataframe = make_group_dataframe()
        config = TrainingConfig(
            "train.csv",
            "val.csv",
            "output",
            train_batch_size=2,
            val_batch_size=2,
            use_pure_nausea_sampling=True,
            pure_nausea_weight=1.5,
        )
        train_loader, _, _ = _create_train_val_loaders(
            dataframe, dataframe, DummyTokenizer(), config, pin_memory=False
        )
        batch = next(iter(train_loader))
        model = TinyMultiLabelModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        loss = build_loss("bce")(logits, batch["labels"])
        loss.backward()
        optimizer.step()

        self.assertEqual(logits.shape, (2, NUM_CLASSES))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
