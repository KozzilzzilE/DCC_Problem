"""다중 백본 모델 및 토크나이저 호환성 단위 테스트."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

# 로컬 테스트 환경에 torch/transformers가 없을 경우를 대비한 안전한 Mocking
if "transformers" not in sys.modules:
    try:
        import transformers  # noqa: F401
    except ImportError:
        sys.modules["transformers"] = MagicMock()
if "sentencepiece" not in sys.modules:
    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        sys.modules["sentencepiece"] = MagicMock()

from m3.model import (
    is_kobert_model,
    validate_generic_tokenizer_model_compatibility,
    validate_tokenizer_model_compatibility,
)


class DummyEmbedding:
    def __init__(self, num_embeddings: int):
        self.num_embeddings = num_embeddings


class DummyModel:
    def __init__(self, num_embeddings: int = 1000):
        self._embedding = DummyEmbedding(num_embeddings)

    def get_input_embeddings(self):
        return self._embedding

    def resize_token_embeddings(self, new_num_tokens: int):
        self._embedding.num_embeddings = new_num_tokens


class DummyGenericTokenizer:
    def __init__(self, cls_id: int = 2, sep_id: int = 3, unk_id: int = 1, vocab_size: int = 1000):
        self.cls_token_id = cls_id
        self.sep_token_id = sep_id
        self.unk_token_id = unk_id
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size

    def tokenize(self, text: str):
        return ["한국어", "모델을", "공유", "##합니다"]

    def __call__(self, text: str, add_special_tokens: bool = True):
        return {
            "input_ids": [self.cls_token_id, 10, 20, 30, self.sep_token_id],
            "attention_mask": [1, 1, 1, 1, 1],
        }


class DummyBrokenJamoTokenizer(DummyGenericTokenizer):
    def tokenize(self, text: str):
        return ["ㅎ", "ㅏ", "ㄴ", "ㄱ", "ㅜ", "ㄱ"]


class MultiModelTokenizerTest(unittest.TestCase):
    def test_is_kobert_model_detection(self) -> None:
        self.assertTrue(is_kobert_model("skt/kobert-base-v1"))
        self.assertTrue(is_kobert_model("my-kobert-finetuned"))
        self.assertFalse(is_kobert_model("monologg/koelectra-base-v3-discriminator"))
        self.assertFalse(is_kobert_model("klue/roberta-base"))
        self.assertFalse(is_kobert_model("klue/bert-base"))

    def test_generic_tokenizer_sanity_pass(self) -> None:
        tokenizer = DummyGenericTokenizer()
        model = DummyModel(1000)
        result = validate_generic_tokenizer_model_compatibility(tokenizer, model)
        self.assertEqual(result["tokenizer_class"], "DummyGenericTokenizer")
        self.assertEqual(result["first_token_id"], 2)
        self.assertEqual(result["last_token_id"], 3)
        self.assertEqual(result["unk_ratio"], 0.0)

    def test_generic_tokenizer_jamo_rejection(self) -> None:
        tokenizer = DummyBrokenJamoTokenizer()
        model = DummyModel(1000)
        with self.assertRaises(ValueError) as ctx:
            validate_generic_tokenizer_model_compatibility(tokenizer, model)
        self.assertIn("한국어가 자모 단위로 분해되었습니다", str(ctx.exception))

    def test_generic_tokenizer_resizes_embedding_if_needed(self) -> None:
        tokenizer = DummyGenericTokenizer(vocab_size=1500)
        model = DummyModel(1000)
        result = validate_generic_tokenizer_model_compatibility(tokenizer, model)
        self.assertEqual(model.get_input_embeddings().num_embeddings, 1500)
        self.assertEqual(result["embedding_size"], 1500)


if __name__ == "__main__":
    unittest.main()
