from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

MISSION3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MISSION3_DIR))

from m3.config import NUM_CLASSES, TARGET_SYMPTOMS
from m3.dataset import MultiLabelCollator, SymptomDataset, load_symptom_csv
from m3.training import TrainingConfig, _calculate_pos_weights, evaluate


class DummyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [1, len(text), 2], "attention_mask": [1, 1, 1]}

    def pad(
        self,
        features,
        padding=True,
        pad_to_multiple_of=None,
        return_tensors="pt",
    ):
        return {
            key: torch.tensor([feature[key] for feature in features], dtype=torch.long)
            for key in features[0]
        }


class DummyModel:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        return SimpleNamespace(logits=torch.full((batch_size, NUM_CLASSES), 10.0))


def make_dataframe() -> pd.DataFrame:
    rows = []
    for index, text in enumerate(("환자가 숨이 차요", "머리가 아프고 토했어요")):
        row = {"call_id": f"sample-{index}", "text": text}
        row.update({symptom: 0 for symptom in TARGET_SYMPTOMS})
        rows.append(row)
    rows[0]["호흡곤란"] = 1
    rows[1]["구토"] = 1
    rows[1]["두통"] = 1
    return pd.DataFrame(rows)


class BaselinePipelineTest(unittest.TestCase):
    def test_pos_weight_is_disabled_by_default(self) -> None:
        config = TrainingConfig(
            train_csv="train.csv",
            val_csv="val.csv",
            output_dir="output",
        )

        self.assertFalse(config.use_pos_weight)

    def test_pos_weight_uses_training_negative_over_positive(self) -> None:
        dataframe = pd.DataFrame({
            symptom: [1, 0, 0, 0]
            for symptom in TARGET_SYMPTOMS
        })

        weights, statistics = _calculate_pos_weights(
            dataframe,
            torch.device("cpu"),
        )

        self.assertTrue(torch.equal(weights, torch.full((NUM_CLASSES,), 3.0)))
        for symptom in TARGET_SYMPTOMS:
            self.assertEqual(statistics[symptom]["positive_count"], 1)
            self.assertEqual(statistics[symptom]["negative_count"], 3)
            self.assertEqual(statistics[symptom]["pos_weight"], 3.0)

    def test_pos_weight_rejects_class_without_positive_sample(self) -> None:
        dataframe = pd.DataFrame({
            symptom: [1, 0]
            for symptom in TARGET_SYMPTOMS
        })
        dataframe[TARGET_SYMPTOMS[0]] = 0

        with self.assertRaisesRegex(ValueError, "positive sample"):
            _calculate_pos_weights(dataframe, torch.device("cpu"))

    def test_dataset_uses_text_and_fixed_label_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "samples.csv"
            make_dataframe().to_csv(csv_path, index=False, encoding="utf-8-sig")

            dataframe = load_symptom_csv(csv_path)
            dataset = SymptomDataset(dataframe, DummyTokenizer(), max_length=16)
            first = dataset[0]

        self.assertEqual(set(first), {"input_ids", "attention_mask", "labels"})
        self.assertEqual(first["labels"].shape, (NUM_CLASSES,))
        self.assertEqual(
            first["labels"][TARGET_SYMPTOMS.index("호흡곤란")].item(),
            1.0,
        )

    def test_collator_preserves_multilabel_shape(self) -> None:
        dataset = SymptomDataset(make_dataframe(), DummyTokenizer(), max_length=16)
        batch = MultiLabelCollator(DummyTokenizer())([dataset[0], dataset[1]])

        self.assertEqual(batch["labels"].dtype, torch.float32)
        self.assertEqual(batch["labels"].shape, (2, NUM_CLASSES))
        self.assertEqual(batch["input_ids"].shape, (2, 3))

    def test_csv_rejects_non_binary_label(self) -> None:
        dataframe = make_dataframe()
        dataframe.loc[0, "고열"] = 2
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "invalid.csv"
            dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")

            with self.assertRaisesRegex(ValueError, "0 또는 1"):
                load_symptom_csv(csv_path)

    def test_evaluate_reuses_macro_f1_at_half(self) -> None:
        batch = {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
            "labels": torch.ones((2, NUM_CLASSES), dtype=torch.float32),
        }
        result = evaluate(
            DummyModel(),
            [batch],
            torch.nn.BCEWithLogitsLoss(),
            torch.device("cpu"),
            amp_enabled=False,
        )

        self.assertAlmostEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["logits"].shape, (2, NUM_CLASSES))
        self.assertEqual(result["probabilities"].shape, (2, NUM_CLASSES))
        self.assertEqual(result["labels"].shape, (2, NUM_CLASSES))


if __name__ == "__main__":
    unittest.main()
