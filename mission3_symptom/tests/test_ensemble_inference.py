"""앙상블·블렌드 제출 추론 테스트 (.pt 번들).

`--ckpt_path` 가 `.pt` 제출 번들이면 트랜스포머 멤버 확률을 균등 평균하고, (번들에 있으면)
Training 전용 TF-IDF 멤버와 전역 가중치 하나로 섞은 뒤 **임계값 0.5** 로 판정한다.
클래스별 가중치·임계값은 없다 (대회 규정: 결정 임계값 0.5 고정).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Sequence
from unittest.mock import patch

import numpy as np

MISSION3_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for path in (MISSION3_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m3 import bundle as bundle_module
from m3 import infer
from m3.config import NUM_CLASSES, TARGET_SYMPTOMS
from m3.infer import bundle_probabilities, predict_directory

OSIM = TARGET_SYMPTOMS.index("오심")
GUTO = TARGET_SYMPTOMS.index("구토")


def write_fake_bundle(
    path: Path,
    member_names: Sequence[str],
    precision: Optional[object] = "fp16",
    tfidf_weight: Optional[float] = None,
) -> Path:
    """가중치 없이 형식만 맞춘 .pt 번들. 모델 로드를 흉내 내는 테스트에서 load_bundle 을 통과시킨다.

    precision 에 None 을 주면 키 자체를 뺀다 (구버전 번들처럼).
    """
    import torch

    payload = {
        "format": bundle_module.BUNDLE_FORMAT,
        "version": bundle_module.BUNDLE_VERSION,
        "members": [{"name": name, "files": {}, "state_dict": {}} for name in member_names],
        "tfidf": None if tfidf_weight is None else {"weight": tfidf_weight, "joblib": b"x"},
        "note": "test",
    }
    if precision is not None:
        payload["precision"] = precision
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def write_labels(label_dir: Path, n: int = 3) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    for index in range(n):
        payload = {"utterances": [
            {"speaker": "0", "startAt": 0, "endAt": 1, "text": "119입니다"},
            {"speaker": "1", "startAt": 1, "endAt": 2, "text": f"속이 안 좋아요 {index}"},
        ]}
        (label_dir / f"call-{index}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class FakeTfidf:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value
        self.seen_texts = None

    def predict_proba(self, texts):
        self.seen_texts = list(texts)
        return np.tile(self.value, (len(texts), 1))


class BlendMathTest(unittest.TestCase):
    """멤버 평균 -> 전역 가중 블렌드 -> 0.5 판정 순서를 숫자로 고정한다."""

    def _run(self, member_probs: dict, tfidf_value, weight):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            pt = write_fake_bundle(root / "b.pt", list(member_probs), tfidf_weight=weight)
            fake = None if tfidf_value is None else FakeTfidf(np.asarray(tfidf_value, dtype=float))
            label_dir = root / "labels"
            write_labels(label_dir)
            names = sorted(p.name for p in label_dir.glob("*.json"))
            members = iter(member_probs.values())   # 번들에 담긴 순서대로 한 번씩 불린다

            def fake_transformer(label_dir_arg, ckpt_path, batch_size=16, device=None, precision="fp32", loader=None):
                vector = np.asarray(next(members), dtype=float)
                return names, np.tile(vector, (len(names), 1))

            with patch.object(infer, "transformer_probabilities", side_effect=fake_transformer), \
                    patch.object(bundle_module, "load_bundle_tfidf", return_value=fake):
                frame = predict_directory(label_dir, pt)
            return frame, fake

    def test_members_are_averaged_equally_before_the_fixed_threshold(self) -> None:
        a = [0.0] * NUM_CLASSES
        b = [0.0] * NUM_CLASSES
        a[OSIM], b[OSIM] = 0.8, 0.3   # 평균 0.55 -> 양성
        a[GUTO], b[GUTO] = 0.9, 0.0   # 평균 0.45 -> 음성
        frame, _ = self._run({"a": a, "b": b}, None, None)

        self.assertTrue((frame["symptom"] == "['오심']").all())

    def test_global_tfidf_weight_blends_every_class_then_uses_half(self) -> None:
        transformer = [0.0] * NUM_CLASSES
        tfidf = [0.0] * NUM_CLASSES
        transformer[OSIM], tfidf[OSIM] = 0.45, 0.9   # 0.7*0.45 + 0.3*0.9 = 0.585 -> 양성
        transformer[GUTO], tfidf[GUTO] = 0.55, 0.1   # 0.7*0.55 + 0.3*0.1 = 0.415 -> 음성
        frame, _ = self._run({"a": transformer}, tfidf, 0.3)

        self.assertTrue((frame["symptom"] == "['오심']").all())

    def test_tfidf_member_reads_space_joined_text(self) -> None:
        _, fake = self._run({"a": [0.0] * NUM_CLASSES}, [0.0] * NUM_CLASSES, 0.3)

        self.assertEqual(len(fake.seen_texts), 3)
        self.assertTrue(all("[SEP]" not in text for text in fake.seen_texts))
        self.assertTrue(all(text.startswith("119입니다 속이 안 좋아요") for text in fake.seen_texts))

    def test_member_file_order_mismatch_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            pt = write_fake_bundle(root / "b.pt", ["a", "b"])
            label_dir = root / "labels"
            write_labels(label_dir)
            orders = iter([["x.json", "y.json"], ["y.json", "x.json"]])

            def fake_transformer(label_dir_arg, ckpt_path, batch_size=16, device=None, precision="fp32", loader=None):
                return next(orders), np.zeros((2, NUM_CLASSES))

            with patch.object(infer, "transformer_probabilities", side_effect=fake_transformer):
                with self.assertRaisesRegex(RuntimeError, "순서"):
                    predict_directory(label_dir, pt)

    def test_blend_probabilities_are_exactly_weighted(self) -> None:
        transformer = np.linspace(0.05, 0.85, NUM_CLASSES)
        tfidf = np.linspace(0.9, 0.1, NUM_CLASSES)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            pt = write_fake_bundle(root / "b.pt", ["a"], tfidf_weight=0.3)
            label_dir = root / "labels"
            write_labels(label_dir)
            names = sorted(p.name for p in label_dir.glob("*.json"))
            with patch.object(infer, "transformer_probabilities",
                              return_value=(names, np.tile(transformer, (len(names), 1)))), \
                    patch.object(bundle_module, "load_bundle_tfidf", return_value=FakeTfidf(tfidf)):
                _, blended = bundle_probabilities(pt, label_dir)

        np.testing.assert_allclose(blended, np.tile(0.7 * transformer + 0.3 * tfidf, (len(names), 1)))

    def test_old_ensemble_manifest_is_not_a_checkpoint_any_more(self) -> None:
        """ensemble.json 폴더 번들은 .pt 로 대체됐다. 조용히 다른 경로로 새지 않고 멈춘다."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bundle").mkdir()
            (root / "bundle" / "ensemble.json").write_text(json.dumps({"members": ["./a"]}), encoding="utf-8")
            write_labels(root / "labels")
            with self.assertRaises(FileNotFoundError):
                predict_directory(root / "labels", root / "bundle")


try:
    import torch  # noqa: F401
    from test_inference_end_to_end import HAS_TORCH, write_labels as write_e2e_labels, write_run
except ImportError:  # pragma: no cover
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch/transformers 가 설치된 환경에서만 실행")
class BundleEndToEndTest(unittest.TestCase):
    def test_real_members_and_tfidf_member_produce_submission_rows(self) -> None:
        import gc

        from m3.bundle import pack_bundle
        from m3.tfidf_member import fit_tfidf_member, save_tfidf_member
        from test_tfidf_member import toy_corpus

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            low = write_run(root / "low", sep_mode="sep", bias=-6.0)     # 모든 확률 약 0.0025
            high = write_run(root / "high", sep_mode="space", bias=6.0)  # 모든 확률 약 0.9975
            for run, sep_mode in ((low, "sep"), (high, "space")):
                # 학습은 best_model/inference_config.json 을 남긴다. 번들에는 best_model 폴더 파일만 담긴다.
                (run / "best_model" / "inference_config.json").write_text(json.dumps(
                    {"utterance_sep_mode": sep_mode, "encode_mode": "truncate", "max_length": 128}), encoding="utf-8")
            texts, labels = toy_corpus()
            tfidf_path = save_tfidf_member(fit_tfidf_member(texts, labels, min_df=1), root / "m.joblib")
            pt = pack_bundle([low, high], root / "mission3.pt", precision="fp32",
                             tfidf_path=tfidf_path, tfidf_weight=0.3)
            label_dir = root / "labels"
            write_e2e_labels(label_dir)

            frame = predict_directory(label_dir, pt, batch_size=2, device="cpu")
            gc.collect()

        # write_e2e_labels 는 통화 3건 + 본문이 빈 통화 1건을 만든다.
        self.assertEqual(len(frame), 4)
        self.assertEqual(list(frame.columns), ["label file name", "symptom"])


class SingleRunTest(unittest.TestCase):
    def test_nan_logit_class_is_dropped_like_before_not_a_crash(self) -> None:
        """단일 모델 경로는 NaN 확률 클래스를 0.5 미만으로 보고 버린다. 죽으면 안 된다."""
        import contextlib
        import io
        from types import SimpleNamespace

        import torch

        class StubTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

            def pad(self, features, padding=True, return_tensors="pt"):
                return {k: torch.tensor([f[k] for f in features]) for k in features[0]}

        class StubModel:
            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, **batch):
                logits = torch.full((batch["input_ids"].shape[0], NUM_CLASSES), 5.0)
                logits[:, 0] = float("nan")
                return SimpleNamespace(logits=logits)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            (run_dir / "best_model").mkdir(parents=True)
            (run_dir / "best_model" / "config.json").write_text("{}", encoding="utf-8")
            (run_dir / "run_config.json").write_text(json.dumps({"utterance_sep_mode": "space"}), encoding="utf-8")
            label_dir = root / "labels"
            write_labels(label_dir)
            buffer = io.StringIO()
            with patch("m3.model.load_saved_model", return_value=(StubTokenizer(), StubModel())), \
                    contextlib.redirect_stdout(buffer):
                frame = predict_directory(label_dir, run_dir, device="cpu")

        expected = str([s for s in TARGET_SYMPTOMS if s != TARGET_SYMPTOMS[0]])
        self.assertTrue((frame["symptom"] == expected).all())
        self.assertIn("NaN", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
