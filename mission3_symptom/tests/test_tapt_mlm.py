"""TAPT 스크립트의 진단용 held-out 평가가 학습 난수에 영향을 주지 않는지 고정한다.

`--eval-csv` 는 평가 전용이어야 한다. 마스크를 만들 때 전역 난수(특히 CUDA)를 다시 시드하면
옵션을 켜느냐에 따라 TAPT 가중치가 달라진다.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BertTokenizer, DataCollatorForLanguageModeling

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

import tapt_mlm

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "네", "119", "입니다", "머리", "아파요", "배", "열", "나요"]


def tiny_tokenizer(tmp: str) -> BertTokenizer:
    path = Path(tmp) / "vocab.txt"
    path.write_text("\n".join(VOCAB), encoding="utf-8")
    return BertTokenizer(str(path))


TEXTS = {"seen_train": ["네 119 입니다 머리 아파요"] * 6, "unseen_eval": ["배 아파요 열 나요"] * 5}


class EvalBatchesTest(unittest.TestCase):
    def test_global_rng_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tok = tiny_tokenizer(tmp)
            collator = DataCollatorForLanguageModeling(tok, mlm_probability=0.5)
            torch.manual_seed(7)
            cpu_before = torch.random.get_rng_state()
            cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            tapt_mlm.build_eval_batches(tok, collator, TEXTS, max_length=16, seed=2024)

            self.assertTrue(torch.equal(cpu_before, torch.random.get_rng_state()))
            if cuda_before is not None:
                for before, after in zip(cuda_before, torch.cuda.get_rng_state_all()):
                    self.assertTrue(torch.equal(before, after))

    def test_masks_are_deterministic_for_a_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tok = tiny_tokenizer(tmp)
            collator = DataCollatorForLanguageModeling(tok, mlm_probability=0.5)
            first = tapt_mlm.build_eval_batches(tok, collator, TEXTS, max_length=16, seed=2024)
            second = tapt_mlm.build_eval_batches(tok, collator, TEXTS, max_length=16, seed=2024)

        self.assertEqual(set(first), set(TEXTS))
        for name in TEXTS:
            for a, b in zip(first[name], second[name]):
                self.assertTrue(torch.equal(a["labels"], b["labels"]))


class EvalHelpersTest(unittest.TestCase):
    def test_sample_size_is_capped_by_available_texts(self) -> None:
        self.assertEqual(len(tapt_mlm.sample_texts(["a", "b", "c"], 256, seed=2024)), 3)
        self.assertEqual(len(tapt_mlm.sample_texts([str(i) for i in range(10)], 4, seed=2024)), 4)

    def test_provenance_mentions_eval_only_use(self) -> None:
        without = tapt_mlm.provenance_note(None)
        with_eval = tapt_mlm.provenance_note("mission3_val.csv")

        self.assertIn("Validation unused", without)
        self.assertNotIn("Validation unused", with_eval)
        self.assertIn("no_grad", with_eval)
        self.assertIn("mission3_val.csv", with_eval)


if __name__ == "__main__":
    unittest.main()
