"""Training 전용 TF-IDF + LogisticRegression 보조 멤버 테스트.

이 멤버는 대화 본문(공백 결합 utterances[].text)만 입력으로 받고, Training CSV 로만 적합한다.
제출 추론에서 트랜스포머 확률과 전역 가중치 하나로 섞인다 (임계값은 0.5 그대로).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

from m3.config import NUM_CLASSES, TARGET_SYMPTOMS
from m3.tfidf_member import (
    DEFAULT_C,
    fit_tfidf_member,
    load_tfidf_member,
    save_tfidf_member,
)

CUES = {
    "고열": "열이 펄펄 나요", "구토": "계속 토했어요", "두통": "머리가 아파요",
    "복통": "배가 아파요", "어지러움": "어지러워요", "열상": "손이 찢어졌어요",
    "오심": "속이 메스꺼워요", "전신쇠약": "기운이 없어요", "호흡곤란": "숨을 못 쉬어요",
}


def toy_corpus():
    """클래스마다 양성·음성이 모두 있는 작은 말뭉치."""
    texts, labels = [], []
    for index, symptom in enumerate(TARGET_SYMPTOMS):
        other = TARGET_SYMPTOMS[(index + 1) % NUM_CLASSES]
        for repeat in range(2):
            texts.append(f"여보세요 119죠 {CUES[symptom]} 네 알겠습니다 {repeat}")
            row = [0] * NUM_CLASSES
            row[index] = 1
            labels.append(row)
        texts.append(f"여보세요 {CUES[symptom]} 그리고 {CUES[other]}")
        row = [0] * NUM_CLASSES
        row[index] = 1
        row[(index + 1) % NUM_CLASSES] = 1
        labels.append(row)
    return texts, np.array(labels)


class TfidfMemberTest(unittest.TestCase):
    def test_default_regularisation_matches_train_oof_choice(self) -> None:
        self.assertEqual(DEFAULT_C, 0.15)

    def test_predict_proba_shape_and_range(self) -> None:
        texts, labels = toy_corpus()
        member = fit_tfidf_member(texts, labels, min_df=1)

        probs = member.predict_proba(texts[:5])

        self.assertEqual(probs.shape, (5, NUM_CLASSES))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_learns_the_obvious_cue(self) -> None:
        texts, labels = toy_corpus()
        member = fit_tfidf_member(texts, labels, C=10.0, min_df=1)

        probs = member.predict_proba(["속이 메스꺼워요"])[0]

        self.assertEqual(int(np.argmax(probs)), TARGET_SYMPTOMS.index("오심"))

    def test_save_load_roundtrip_gives_identical_probabilities(self) -> None:
        texts, labels = toy_corpus()
        member = fit_tfidf_member(texts, labels, min_df=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_tfidf_member(member, Path(tmp) / "tfidf_lr.joblib")
            loaded = load_tfidf_member(path)

        np.testing.assert_array_equal(member.predict_proba(texts), loaded.predict_proba(texts))
        self.assertEqual(loaded.C, member.C)
        self.assertEqual(loaded.train_rows, len(texts))

    def test_rejects_label_matrix_with_wrong_shape(self) -> None:
        texts, labels = toy_corpus()
        with self.assertRaisesRegex(ValueError, "라벨"):
            fit_tfidf_member(texts, labels[:, :8], min_df=1)
        with self.assertRaisesRegex(ValueError, "라벨"):
            fit_tfidf_member(texts[:-1], labels, min_df=1)

    def test_rejects_class_without_both_outcomes(self) -> None:
        texts, labels = toy_corpus()
        labels = labels.copy()
        labels[:, TARGET_SYMPTOMS.index("열상")] = 0
        with self.assertRaisesRegex(ValueError, "열상"):
            fit_tfidf_member(texts, labels, min_df=1)

    def test_load_rejects_foreign_file(self) -> None:
        import joblib

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "other.joblib"
            joblib.dump({"format": "something-else"}, path)
            with self.assertRaisesRegex(ValueError, "형식"):
                load_tfidf_member(path)

    def test_load_rejects_different_symptom_order(self) -> None:
        import joblib

        texts, labels = toy_corpus()
        member = fit_tfidf_member(texts, labels, min_df=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_tfidf_member(member, Path(tmp) / "tfidf_lr.joblib")
            payload = joblib.load(path)
            payload["symptoms"] = list(reversed(payload["symptoms"]))
            joblib.dump(payload, path)
            with self.assertRaisesRegex(ValueError, "증상 순서"):
                load_tfidf_member(path)


class TrainTfidfCliTest(unittest.TestCase):
    def _write_csv(self, directory: Path, texts, labels) -> Path:
        frame = pd.DataFrame({"call_id": [f"c{i}" for i in range(len(texts))], "text": texts})
        for index, symptom in enumerate(TARGET_SYMPTOMS):
            frame[symptom] = labels[:, index]
        path = directory / "train.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def test_cli_fits_saves_and_records_provenance(self) -> None:
        import train_tfidf_member as cli

        texts, labels = toy_corpus()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = self._write_csv(root, texts, labels)
            output = root / "member" / "tfidf_lr.joblib"
            argv = ["train_tfidf_member.py", "--train-csv", str(csv_path), "--output", str(output), "--min-df", "1"]
            with patch.object(sys, "argv", argv):
                self.assertEqual(cli.main(), 0)

            self.assertTrue(output.is_file())
            record = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(record["train_rows"], len(texts))
            self.assertEqual(record["C"], DEFAULT_C)
            self.assertEqual(record["input"], "space-joined utterances[].text")
            self.assertEqual(len(load_tfidf_member(output).predict_proba(["머리가 아파요"])[0]), NUM_CLASSES)

    def test_cli_refuses_boundary_marked_text(self) -> None:
        import train_tfidf_member as cli

        texts, labels = toy_corpus()
        texts = [text.replace(" ", " [SEP] ", 1) for text in texts]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = self._write_csv(root, texts, labels)
            argv = ["train_tfidf_member.py", "--train-csv", str(csv_path), "--output", str(root / "m.joblib"), "--min-df", "1"]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "space"):
                    cli.main()

    def test_cli_refuses_to_overwrite(self) -> None:
        import train_tfidf_member as cli

        texts, labels = toy_corpus()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = self._write_csv(root, texts, labels)
            output = root / "tfidf_lr.joblib"
            output.write_bytes(b"existing")
            argv = ["train_tfidf_member.py", "--train-csv", str(csv_path), "--output", str(output), "--min-df", "1"]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    cli.main()

    def test_cli_rejects_json_output_path(self) -> None:
        # 출처 기록(.json)이 모델 파일 자체를 덮어쓰면 안 된다.
        import train_tfidf_member as cli

        texts, labels = toy_corpus()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = self._write_csv(root, texts, labels)
            argv = ["train_tfidf_member.py", "--train-csv", str(csv_path), "--output", str(root / "member.json"), "--min-df", "1"]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "json"):
                    cli.main()

    def test_cli_refuses_to_overwrite_existing_sidecar(self) -> None:
        import train_tfidf_member as cli

        texts, labels = toy_corpus()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = self._write_csv(root, texts, labels)
            (root / "tfidf_lr.json").write_text("{}", encoding="utf-8")
            argv = ["train_tfidf_member.py", "--train-csv", str(csv_path), "--output", str(root / "tfidf_lr.joblib"), "--min-df", "1"]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    cli.main()


if __name__ == "__main__":
    unittest.main()
