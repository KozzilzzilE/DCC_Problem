"""모델·토크나이저 로드와 호환성 검사 단위 테스트."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

# 로컬 테스트 환경에 torch/transformers가 없을 경우를 대비한 안전한 Mocking
if "transformers" not in sys.modules:
    try:
        import transformers  # noqa: F401
    except ImportError:
        sys.modules["transformers"] = MagicMock()

from m3.config import NUM_CLASSES, TARGET_SYMPTOMS
from m3.model import build_tokenizer_and_model, validate_tokenizer_model_compatibility


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


class TokenizerSanityTest(unittest.TestCase):
    def test_tokenizer_sanity_pass(self) -> None:
        tokenizer = DummyGenericTokenizer()
        model = DummyModel(1000)
        result = validate_tokenizer_model_compatibility(tokenizer, model)
        self.assertEqual(result["tokenizer_class"], "DummyGenericTokenizer")
        self.assertEqual(result["first_token_id"], 2)
        self.assertEqual(result["last_token_id"], 3)
        self.assertEqual(result["unk_ratio"], 0.0)

    def test_tokenizer_jamo_rejection(self) -> None:
        tokenizer = DummyBrokenJamoTokenizer()
        model = DummyModel(1000)
        with self.assertRaises(ValueError) as ctx:
            validate_tokenizer_model_compatibility(tokenizer, model)
        self.assertIn("한국어가 자모 단위로 분해되었습니다", str(ctx.exception))

    def test_tokenizer_resizes_embedding_if_needed(self) -> None:
        tokenizer = DummyGenericTokenizer(vocab_size=1500)
        model = DummyModel(1000)
        result = validate_tokenizer_model_compatibility(tokenizer, model)
        self.assertEqual(model.get_input_embeddings().num_embeddings, 1500)
        self.assertEqual(result["embedding_size"], 1500)


class BuildModelTest(unittest.TestCase):
    def test_builds_nine_label_multilabel_head_with_load_options(self) -> None:
        """분류 헤드 설정(9개 라벨·순서·multi-label)과 로드 옵션이 표준 로더에 그대로 넘어가는지 고정한다."""
        tokenizer, model = MagicMock(), MagicMock()
        with patch("m3.model.AutoTokenizer.from_pretrained", return_value=tokenizer) as tokenizer_loader, \
                patch("m3.model.AutoModelForSequenceClassification.from_pretrained",
                      return_value=model) as model_loader, \
                patch("m3.model.validate_tokenizer_model_compatibility") as validate:
            built = build_tokenizer_and_model("runs/devsel/tapt", local_files_only=True, revision="abc")

        self.assertEqual(built, (tokenizer, model))
        tokenizer_loader.assert_called_once_with("runs/devsel/tapt", local_files_only=True, revision="abc")
        model_loader.assert_called_once_with(
            "runs/devsel/tapt",
            num_labels=NUM_CLASSES,
            label2id={symptom: index for index, symptom in enumerate(TARGET_SYMPTOMS)},
            id2label={index: symptom for index, symptom in enumerate(TARGET_SYMPTOMS)},
            problem_type="multi_label_classification",
            local_files_only=True,
            revision="abc",
        )
        validate.assert_called_once_with(tokenizer, model)


if __name__ == "__main__":
    unittest.main()
