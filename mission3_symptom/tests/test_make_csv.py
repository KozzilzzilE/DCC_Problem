"""make_csv.py: 원본 JSON -> 학습 CSV 가 학습 로더(load_symptom_csv)와 추론 입력 규칙을 지키는지."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

import make_csv
from m3.config import TARGET_SYMPTOMS
from m3.dataset import load_symptom_csv
from m3.infer import read_texts


def write_call(label_dir: Path, call_id: str, symptoms, utterances) -> None:
    payload = {"symptom": symptoms, "utterances": [
        {"speaker": str(i % 2), "startAt": i, "endAt": i + 1, "text": text} for i, text in enumerate(utterances)]}
    (label_dir / f"{call_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class MakeCsvTest(unittest.TestCase):
    def test_csv_matches_training_loader_and_inference_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            label_dir = Path(tmp) / "label"
            label_dir.mkdir()
            write_call(label_dir, "a", ["두통", "골절"], ["119입니다", "머리가 아파요"])
            write_call(label_dir, "b", [], ["네", "주소 알려주세요"])
            out = Path(tmp) / "csv" / "train.csv"

            make_csv.main(["--label-dir", str(label_dir), "--output", str(out)])
            frame = load_symptom_csv(out)
            names, texts = read_texts(label_dir, "space")

        self.assertEqual(list(frame["call_id"]), ["a", "b"])
        self.assertEqual(frame.loc[0, "두통"], 1)
        self.assertEqual(int(frame.loc[0, TARGET_SYMPTOMS].sum()), 1)   # 골절은 대상 외라 버린다
        self.assertEqual(int(frame.loc[1, TARGET_SYMPTOMS].sum()), 0)
        self.assertEqual(list(frame["text"]), texts)                   # 추론 입력과 같은 본문

    def test_empty_folder_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                make_csv.build_dataframe(Path(tmp))


if __name__ == "__main__":
    unittest.main()
