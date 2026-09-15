"""Cache-gated integration checks for the pinned KF-DeBERTa artifact."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import torch

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

from m3.config import NUM_CLASSES
from m3.model import CLS_POOLING, build_tokenizer_and_model


class KfDebertaIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source_value = os.environ.get("KF_DEBERTA_LOCAL_PATH")
        if not source_value:
            raise unittest.SkipTest("KF_DEBERTA_LOCAL_PATH is not set.")
        source = Path(source_value)
        if not source.is_dir():
            raise unittest.SkipTest(f"KF-DeBERTa local artifact is absent: {source}")
        cls.tokenizer, cls.model = build_tokenizer_and_model(
            source,
            local_files_only=True,
            pooling_type=CLS_POOLING,
        )
        cls.model.eval()

    def test_model_and_tokenizer_contract(self) -> None:
        self.assertEqual(type(self.tokenizer).__name__, "BertTokenizer")
        self.assertTrue(self.tokenizer.is_fast)
        self.assertEqual(self.tokenizer.vocab_size, 130000)
        self.assertEqual(len(self.tokenizer), 130000)
        self.assertEqual(
            self.model.get_input_embeddings().num_embeddings,
            130000,
        )
        self.assertEqual(self.tokenizer.model_max_length, 512)
        self.assertEqual(self.model.config.max_position_embeddings, 512)
        self.assertEqual(self.model.config.model_type, "deberta-v2")
        self.assertEqual(
            type(self.model).__name__,
            "DebertaV2ForSequenceClassification",
        )
        self.assertEqual(self.model.config.num_labels, NUM_CLASSES)
        self.assertEqual(
            self.model.config.problem_type,
            "multi_label_classification",
        )

    def test_special_tokens_single_sequence_and_truncation(self) -> None:
        self.assertEqual(
            {
                "pad": self.tokenizer.pad_token_id,
                "unk": self.tokenizer.unk_token_id,
                "cls": self.tokenizer.cls_token_id,
                "sep": self.tokenizer.sep_token_id,
                "mask": self.tokenizer.mask_token_id,
            },
            {"pad": 0, "unk": 1, "cls": 2, "sep": 3, "mask": 4},
        )
        text = "환자가 고열과 구토가 있고 숨쉬기 힘들다고 합니다. " * 300
        encoded = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            add_special_tokens=True,
        )
        self.assertEqual(len(encoded["input_ids"]), 512)
        self.assertEqual(encoded["input_ids"][0], self.tokenizer.cls_token_id)
        self.assertEqual(encoded["input_ids"][-1], self.tokenizer.sep_token_id)
        self.assertEqual(len(encoded["attention_mask"]), 512)
        self.assertIn("token_type_ids", encoded)
        self.assertEqual(set(encoded["token_type_ids"]), {0})

    def test_standard_auto_classifier_logits_shape(self) -> None:
        inputs = self.tokenizer(
            ["환자가 숨쉬기 어렵습니다.", "머리가 아프고 토했어요."],
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        self.assertEqual(tuple(logits.shape), (2, NUM_CLASSES))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
