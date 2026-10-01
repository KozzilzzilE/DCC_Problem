"""TAPT 스크립트가 Training CSV 본문만으로 MLM 을 학습하고 결과 폴더를 남기는지 고정한다.

- 다른 CSV 를 받는 옵션(예전 진단용 `--eval-csv`)은 없다. 주면 인자 단계에서 멈춘다.
- 인터넷 없이 그 자리에서 만든 소형 BERT MLM 으로 1 epoch 를 실제로 돌려, 결과 폴더가
  `train.py --model-name-or-path` 로 읽을 수 있는 형태인지와 기록(tapt_config.json)을 확인한다.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from transformers import AutoTokenizer, BertConfig, BertForMaskedLM, BertTokenizer

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

import tapt_mlm
from m3.config import TARGET_SYMPTOMS

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "네", "119", "입니다", "머리", "아파요", "배", "열", "나요"]
TEXTS = ["네 119 입니다 머리 아파요", "배 아파요 열 나요", "머리 열 나요", "네 배 아파요"]


def build_tiny_mlm(model_dir: Path) -> None:
    """인터넷 없이 로드되는 소형 BERT MLM 과 tokenizer 를 만든다."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "vocab.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "BertTokenizer", "do_lower_case": False}), encoding="utf-8")
    BertTokenizer.from_pretrained(str(model_dir)).save_pretrained(model_dir)
    config = BertConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=64)
    BertForMaskedLM(config).save_pretrained(model_dir)


def write_train_csv(path: Path) -> None:
    rows = []
    for index, text in enumerate(TEXTS * 3):
        row = {"call_id": f"call-{index}", "text": text}
        row.update({symptom: 0 for symptom in TARGET_SYMPTOMS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


class TaptArgumentsTest(unittest.TestCase):
    def test_extra_csv_options_are_rejected(self) -> None:
        required = ["--train-csv", "train.csv", "--output-dir", "out"]
        for extra in (["--eval-csv", "dev.csv"], ["--eval-samples", "256"]):
            with self.subTest(extra=extra), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    tapt_mlm.parse_args([*required, *extra])

    def test_defaults(self) -> None:
        args = tapt_mlm.parse_args(["--train-csv", "train.csv", "--output-dir", "out"])

        self.assertEqual(args.model_name_or_path, "klue/roberta-base")
        self.assertEqual((args.learning_rate, args.mlm_probability, args.seed), (5e-5, 0.15, 42))
        self.assertEqual((args.batch_size, args.gradient_accumulation_steps), (8, 2))


class TaptRunTest(unittest.TestCase):
    def test_one_epoch_saves_model_and_training_only_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_tiny_mlm(root / "base")
            write_train_csv(root / "train.csv")
            out = root / "tapt"
            argv = ["tapt_mlm.py", "--train-csv", str(root / "train.csv"), "--output-dir", str(out),
                    "--model-name-or-path", str(root / "base"), "--local-files-only",
                    "--epochs", "1", "--max-length", "16", "--batch-size", "4",
                    # 짧은 본문이라 0.15 면 마스크가 하나도 없는 배치(손실 NaN)가 나올 수 있다.
                    "--mlm-probability", "0.5"]
            with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=False):
                tapt_mlm.main()

            meta = json.loads((out / "tapt_config.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["source"], tapt_mlm.SOURCE_NOTE)
            self.assertIn("Validation unused", meta["source"])
            self.assertEqual(meta["num_texts"], len(TEXTS) * 3)
            self.assertEqual([row["epoch"] for row in meta["history"]], [1])
            self.assertEqual(set(meta["history"][0]), {"epoch", "mlm_loss"})
            self.assertNotIn("eval_csv", meta)
            # 분류 학습이 --local-files-only 로 바로 읽을 수 있어야 한다.
            tokenizer = AutoTokenizer.from_pretrained(str(out), local_files_only=True)
            self.assertEqual(tokenizer.mask_token, "[MASK]")
            self.assertTrue((out / "config.json").is_file())


if __name__ == "__main__":
    unittest.main()
