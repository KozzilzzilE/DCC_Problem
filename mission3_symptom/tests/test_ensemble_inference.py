"""앙상블·블렌드 제출 추론 테스트.

`--ckpt_path` 가 `ensemble.json` 을 가리키면 트랜스포머 멤버 확률을 균등 평균하고,
Training 전용 TF-IDF 멤버와 전역 가중치 하나로 섞은 뒤 **임계값 0.5** 로 판정한다.
클래스별 가중치·임계값은 없다 (대회 규정: 결정 임계값 0.5 고정).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

MISSION3_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for path in (MISSION3_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m3 import infer
from m3.config import NUM_CLASSES, TARGET_SYMPTOMS
from m3.infer import (
    ENSEMBLE_MANIFEST_NAME,
    find_ensemble_manifest,
    load_ensemble_spec,
    predict_directory,
)

OSIM = TARGET_SYMPTOMS.index("오심")
GUTO = TARGET_SYMPTOMS.index("구토")


def make_run(root: Path, name: str) -> Path:
    run_dir = root / name
    (run_dir / "best_model").mkdir(parents=True)
    (run_dir / "best_model" / "config.json").write_text("{}", encoding="utf-8")
    return run_dir


def write_manifest(directory: Path, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ENSEMBLE_MANIFEST_NAME
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def write_labels(label_dir: Path, n: int = 3) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    for index in range(n):
        payload = {"utterances": [
            {"speaker": "0", "startAt": 0, "endAt": 1, "text": "119입니다"},
            {"speaker": "1", "startAt": 1, "endAt": 2, "text": f"속이 안 좋아요 {index}"},
        ]}
        (label_dir / f"call-{index}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class ManifestTest(unittest.TestCase):
    def test_finds_manifest_in_directory_or_as_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "a")
            manifest = write_manifest(root / "bundle", {"members": ["../a"]})
            self.assertEqual(find_ensemble_manifest(root / "bundle"), manifest)
            self.assertEqual(find_ensemble_manifest(manifest), manifest)
            self.assertIsNone(find_ensemble_manifest(root / "a"))

    def test_member_paths_are_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = make_run(root, "a"), make_run(root, "b")
            (root / "tfidf").mkdir()
            (root / "tfidf" / "m.joblib").write_bytes(b"x")
            manifest = write_manifest(root / "bundle", {
                "members": ["../a", "../b"], "tfidf_member": "../tfidf/m.joblib", "tfidf_weight": 0.3})

            spec = load_ensemble_spec(manifest)

            self.assertEqual([p.resolve() for p in spec.members], [a.resolve(), b.resolve()])
            self.assertEqual(spec.tfidf_path.resolve(), (root / "tfidf" / "m.joblib").resolve())
            self.assertEqual(spec.tfidf_weight, 0.3)

    def test_rejects_invalid_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "a")
            (root / "m.joblib").write_bytes(b"x")
            cases = {
                "no members": {"members": []},
                "missing member": {"members": ["../missing"]},
                "weight without member": {"members": ["../a"], "tfidf_weight": 0.3},
                "member without weight": {"members": ["../a"], "tfidf_member": "../m.joblib"},
                "weight too large": {"members": ["../a"], "tfidf_member": "../m.joblib", "tfidf_weight": 1.0},
                "negative weight": {"members": ["../a"], "tfidf_member": "../m.joblib", "tfidf_weight": -0.1},
                "per-class weight": {"members": ["../a"], "tfidf_member": "../m.joblib", "tfidf_weight": [0.3] * 9},
                "missing tfidf file": {"members": ["../a"], "tfidf_member": "../nope.joblib", "tfidf_weight": 0.3},
                "threshold override": {"members": ["../a"], "threshold": 0.4},
            }
            for name, payload in cases.items():
                with self.subTest(case=name):
                    manifest = write_manifest(root / "bundle", payload)
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        load_ensemble_spec(manifest)


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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in member_probs:
                make_run(root, name)
            payload = {"members": [f"../{name}" for name in member_probs]}
            fake = None
            if tfidf_value is not None:
                (root / "m.joblib").write_bytes(b"x")
                payload.update({"tfidf_member": "../m.joblib", "tfidf_weight": weight})
                fake = FakeTfidf(np.asarray(tfidf_value, dtype=float))
            manifest = write_manifest(root / "bundle", payload)
            label_dir = root / "labels"
            write_labels(label_dir)
            names = sorted(p.name for p in label_dir.glob("*.json"))

            def fake_transformer(label_dir_arg, ckpt_path, batch_size=16, device=None):
                vector = np.asarray(member_probs[Path(ckpt_path).name], dtype=float)
                return names, np.tile(vector, (len(names), 1))

            with patch.object(infer, "transformer_probabilities", side_effect=fake_transformer), \
                    patch.object(infer, "load_tfidf_member", return_value=fake):
                frame = predict_directory(label_dir, manifest.parent)
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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "a"), make_run(root, "b")
            manifest = write_manifest(root / "bundle", {"members": ["../a", "../b"]})
            label_dir = root / "labels"
            write_labels(label_dir)

            def fake_transformer(label_dir_arg, ckpt_path, batch_size=16, device=None):
                names = ["x.json", "y.json"] if Path(ckpt_path).name == "a" else ["y.json", "x.json"]
                return names, np.zeros((2, NUM_CLASSES))

            with patch.object(infer, "transformer_probabilities", side_effect=fake_transformer):
                with self.assertRaisesRegex(RuntimeError, "순서"):
                    predict_directory(label_dir, manifest.parent)


try:
    import torch  # noqa: F401
    from test_inference_end_to_end import HAS_TORCH, write_labels as write_e2e_labels, write_run
except ImportError:  # pragma: no cover
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch/transformers 가 설치된 환경에서만 실행")
class EnsembleEndToEndTest(unittest.TestCase):
    def test_real_members_and_tfidf_member_produce_submission_rows(self) -> None:
        from m3.tfidf_member import fit_tfidf_member, save_tfidf_member
        from test_tfidf_member import toy_corpus

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            low = write_run(root / "low", sep_mode="sep", bias=-6.0)     # 모든 확률 약 0.0025
            high = write_run(root / "high", sep_mode="space", bias=6.0)  # 모든 확률 약 0.9975
            texts, labels = toy_corpus()
            tfidf_path = save_tfidf_member(fit_tfidf_member(texts, labels, min_df=1), root / "m.joblib")
            manifest = write_manifest(root / "bundle", {
                "members": [str(low), str(high)], "tfidf_member": str(tfidf_path), "tfidf_weight": 0.3})
            label_dir = root / "labels"
            write_e2e_labels(label_dir)

            frame = predict_directory(label_dir, manifest.parent, batch_size=2)

        # write_e2e_labels 는 통화 3건 + 본문이 빈 통화 1건을 만든다.
        self.assertEqual(len(frame), 4)
        self.assertEqual(list(frame.columns), ["label file name", "symptom"])


class ReviewFollowUpTest(unittest.TestCase):
    """PR 전 리뷰에서 확인된 문제들의 회귀 테스트."""

    def test_manifest_with_utf8_bom_loads(self) -> None:
        # Windows PowerShell 5.1 의 Out-File -Encoding utf8 은 BOM 을 붙인다.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "a")
            bundle = root / "bundle"
            bundle.mkdir()
            manifest = bundle / ENSEMBLE_MANIFEST_NAME
            manifest.write_text(json.dumps({"members": ["../a"]}), encoding="utf-8-sig")

            spec = load_ensemble_spec(manifest)

        self.assertEqual(len(spec.members), 1)

    def test_self_contained_bundle_can_be_moved_alone(self) -> None:
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            make_run(bundle, "seed42")
            (bundle / "tfidf").mkdir()
            (bundle / "tfidf" / "m.joblib").write_bytes(b"x")
            write_manifest(bundle, {"members": ["./seed42"], "tfidf_member": "./tfidf/m.joblib", "tfidf_weight": 0.3})
            (root / "elsewhere").mkdir()
            moved = Path(shutil.move(str(bundle), str(root / "elsewhere" / "submission")))

            spec = load_ensemble_spec(moved / ENSEMBLE_MANIFEST_NAME)

        self.assertEqual(spec.members[0].name, "seed42")
        self.assertEqual(spec.tfidf_path.name, "m.joblib")

    def test_warns_when_bundle_points_outside_itself(self) -> None:
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "a")
            manifest = write_manifest(root / "bundle", {"members": ["../a"]})
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                load_ensemble_spec(manifest)

        self.assertIn("번들 밖", buffer.getvalue())

    def test_blend_probabilities_are_exactly_weighted(self) -> None:
        from m3.infer import ensemble_probabilities

        transformer = np.linspace(0.05, 0.85, NUM_CLASSES)
        tfidf = np.linspace(0.9, 0.1, NUM_CLASSES)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "a")
            (root / "m.joblib").write_bytes(b"x")
            manifest = write_manifest(root / "bundle", {
                "members": ["../a"], "tfidf_member": "../m.joblib", "tfidf_weight": 0.3})
            label_dir = root / "labels"
            write_labels(label_dir)
            names = sorted(p.name for p in label_dir.glob("*.json"))
            with patch.object(infer, "transformer_probabilities",
                              return_value=(names, np.tile(transformer, (len(names), 1)))), \
                    patch.object(infer, "load_tfidf_member", return_value=FakeTfidf(tfidf)):
                _, blended = ensemble_probabilities(load_ensemble_spec(manifest), label_dir)

        np.testing.assert_allclose(blended, np.tile(0.7 * transformer + 0.3 * tfidf, (len(names), 1)))

    def test_nan_logit_class_is_dropped_like_before_not_a_crash(self) -> None:
        """기존 단일 모델 경로는 NaN 확률 클래스를 0.5 미만으로 보고 버렸다. 그대로여야 한다."""
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
            run_dir = make_run(root, "run")
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
