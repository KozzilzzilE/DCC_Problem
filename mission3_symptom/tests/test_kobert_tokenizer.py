from __future__ import annotations

import gc
import re
import sys
import tempfile
import unittest
from pathlib import Path

MISSION3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MISSION3_DIR))

from m3.kobert_tokenizer import KoBertTokenizer
from m3.model import (
    DEFAULT_MODEL_NAME,
    build_tokenizer_and_model,
    load_saved_model,
    save_model_bundle,
)


JAMO_PATTERN = re.compile(r"[\u1100-\u11ff\u3130-\u318f]")
SAMPLE_TEXT = "한국어 모델을 공유합니다."
EXPECTED_TOKENS = ["▁한국", "어", "▁모델", "을", "▁공유", "합니다", "."]
EXPECTED_CONTENT_IDS = [4958, 6855, 2046, 7088, 1050, 7843, 54]


class KoBertTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.tokenizer, cls.model = build_tokenizer_and_model(
                DEFAULT_MODEL_NAME,
                local_files_only=True,
            )
        except OSError as error:
            raise unittest.SkipTest(f"KoBERT local cache가 없습니다: {error}") from error

    def test_uses_sentencepiece_without_jamo_decomposition(self) -> None:
        tokens = self.tokenizer.tokenize(SAMPLE_TEXT)

        self.assertIsInstance(self.tokenizer, KoBertTokenizer)
        self.assertEqual(tokens, EXPECTED_TOKENS)
        self.assertFalse(any(JAMO_PATTERN.search(token) for token in tokens))

    def test_unknown_ratio_and_special_token_order(self) -> None:
        encoded = self.tokenizer(SAMPLE_TEXT, add_special_tokens=True)
        content_ids = encoded["input_ids"][1:-1]
        unknown_ratio = sum(
            token_id == self.tokenizer.unk_token_id for token_id in content_ids
        ) / len(content_ids)

        self.assertEqual(content_ids, EXPECTED_CONTENT_IDS)
        self.assertLessEqual(unknown_ratio, 0.1)
        self.assertEqual(encoded["input_ids"][0], 2)
        self.assertEqual(encoded["input_ids"][-1], 3)

    def test_padding_attention_mask_and_token_type_ids(self) -> None:
        batch = self.tokenizer(
            [SAMPLE_TEXT, "환자가 숨을 쉬기 어렵습니다."],
            padding=True,
            return_tensors="pt",
        )

        self.assertEqual(batch["input_ids"].shape, batch["attention_mask"].shape)
        self.assertEqual(batch["input_ids"].shape, batch["token_type_ids"].shape)
        self.assertTrue((batch["token_type_ids"] == 0).all().item())
        self.assertTrue(
            ((batch["input_ids"] == 1) == (batch["attention_mask"] == 0)).all().item()
        )

    def test_truncation_preserves_sep_token(self) -> None:
        encoded = self.tokenizer(
            "환자가 숨을 쉬기 어렵습니다. " * 100,
            max_length=16,
            truncation=True,
        )

        self.assertEqual(len(encoded["input_ids"]), 16)
        self.assertEqual(encoded["input_ids"][0], 2)
        self.assertEqual(encoded["input_ids"][-1], 3)

    def test_vocabulary_matches_model_embeddings(self) -> None:
        embedding_size = self.model.get_input_embeddings().num_embeddings

        self.assertEqual(self.tokenizer.vocab_size, 8002)
        self.assertEqual(len(self.tokenizer), embedding_size)
        self.assertEqual(self.tokenizer.vocab_size, embedding_size)

    def test_save_and_local_reload_preserve_token_ids(self) -> None:
        expected = self.tokenizer(SAMPLE_TEXT)["input_ids"]
        with tempfile.TemporaryDirectory() as directory:
            save_model_bundle(self.model, self.tokenizer, directory)
            reloaded_tokenizer, reloaded_model = load_saved_model(directory)
            actual = reloaded_tokenizer(SAMPLE_TEXT)["input_ids"]
            del reloaded_model
            gc.collect()

        self.assertIsInstance(reloaded_tokenizer, KoBertTokenizer)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
