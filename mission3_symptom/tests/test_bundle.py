"""제출 번들(.pt) 묶기·불러오기 테스트 (m3/bundle.py).

학습이 남긴 best_model 폴더를 `pack_bundle` 로 묶고 `load_bundle` / `load_member_model` 로 다시 만든 모델이,
같은 폴더를 Hugging Face 표준 로더로 읽은 모델과 가중치·확률까지 같은지 확인한다.
작은 무작위 RoBERTa 와 로컬 Hugging Face 캐시의 klue/roberta-base 토크나이저를 쓴다 (없으면 건너뛴다).
"""

from __future__ import annotations

import contextlib
import gc
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

MISSION3_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for _path in (MISSION3_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from m3.bundle import (
    BUNDLE_FORMAT,
    describe,
    load_bundle,
    load_bundle_tfidf,
    load_member_model,
    pack_bundle,
)
from m3.config import NUM_CLASSES, TARGET_SYMPTOMS
from m3.infer import build_inference_config, bundle_probabilities, transformer_probabilities
from m3.model import load_saved_model, save_model_bundle


def klue_tokenizer():
    """로컬 캐시의 klue/roberta-base 토크나이저. 없으면 None (인터넷에 접속하지 않는다)."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("klue/roberta-base", local_files_only=True)
    except Exception:  # noqa: BLE001 - 캐시가 없는 환경에서는 테스트를 건너뛴다
        return None


TOKENIZER = klue_tokenizer()


def write_tiny_run(root: Path, name: str, seed: int, max_length: int = 64) -> Path:
    """학습(run_training)이 남기는 것과 같은 모양의 run 폴더: best_model 가중치·토크나이저·inference_config."""
    from transformers import RobertaConfig, RobertaForSequenceClassification

    torch.manual_seed(seed)
    config = RobertaConfig(
        vocab_size=len(TOKENIZER), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=514, type_vocab_size=1, pad_token_id=1,
        bos_token_id=0, eos_token_id=2, num_labels=NUM_CLASSES, problem_type="multi_label_classification",
        label2id={s: i for i, s in enumerate(TARGET_SYMPTOMS)}, id2label={i: s for i, s in enumerate(TARGET_SYMPTOMS)},
    )
    model = RobertaForSequenceClassification(config)
    best_model = root / name / "best_model"
    save_model_bundle(model, TOKENIZER, best_model)
    training_config = SimpleNamespace(utterance_sep_mode="space", encode_mode="truncate",
                                      max_length=max_length, model_name_or_path="klue/roberta-base")
    (best_model / "inference_config.json").write_text(
        json.dumps(build_inference_config(training_config), ensure_ascii=False), encoding="utf-8")
    return root / name


def write_labels(label_dir: Path) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    texts = [["여보세요 119죠", "머리가 아프고 토했어요"], ["숨을 못 쉬겠어요"], ["열이 나요"] * 30, []]
    for index, utterances in enumerate(texts):
        payload = {"utterances": [{"speaker": "0", "startAt": 0, "endAt": 1, "text": t} for t in utterances],
                   "gender": "여"}
        (label_dir / f"call-{index}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@unittest.skipIf(TOKENIZER is None, "로컬 Hugging Face 캐시에 klue/roberta-base 토크나이저가 없다")
class PackLoadRoundTripTest(unittest.TestCase):
    def test_round_trip_restores_files_weights_and_settings(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            runs = [write_tiny_run(root, "final_s42", 42), write_tiny_run(root, "final_s43", 43)]
            pt = pack_bundle(runs, root / "mission3.pt", note="round trip")
            bundle = load_bundle(pt)

            self.assertEqual(bundle.precision, "fp16")            # 기본 추론 정밀도
            self.assertEqual([m.name for m in bundle.members], ["final_s42", "final_s43"])
            self.assertIsNone(bundle.tfidf_joblib)
            self.assertEqual(bundle.note, "round trip")
            for run, member in zip(runs, bundle.members):
                saved = {f.name: f.read_bytes() for f in (run / "best_model").iterdir()
                         if f.name != "model.safetensors"}
                self.assertEqual(member.files, saved)             # 설정·토크나이저 파일 원문 그대로
                self.assertEqual(member.inference_config()["encode_mode"], "truncate")
                self.assertEqual(member.inference_config()["max_length"], 64)

                _, reference = load_saved_model(run / "best_model")
                _, rebuilt = load_member_model(member)
                expected = reference.state_dict()
                actual = rebuilt.state_dict()
                self.assertEqual(sorted(expected), sorted(actual))
                for key in expected:
                    self.assertTrue(torch.equal(expected[key], actual[key]), key)
                self.assertEqual(rebuilt.config.num_labels, NUM_CLASSES)

            lines = describe(bundle)
            params = sum(int(t.numel()) for t in bundle.members[0].state_dict.values())
            self.assertIn(f"{params:,}", lines[1])
            del bundle
            gc.collect()

    def test_bundle_member_predicts_exactly_like_the_run_directory(self) -> None:
        """같은 가중치를 .pt 로 묶어 읽어도 run 폴더에서 읽은 것과 확률이 비트 단위로 같다 (CPU fp32)."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            run = write_tiny_run(root, "final_s42", 42)
            label_dir = root / "labels"
            write_labels(label_dir)
            pt = pack_bundle([run], root / "mission3.pt", precision="fp32")

            names_run, probs_run = transformer_probabilities(label_dir, run, batch_size=2, device="cpu")
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                names_pt, probs_pt = bundle_probabilities(pt, label_dir, batch_size=2, device="cpu")
            gc.collect()

        self.assertEqual(names_run, names_pt)
        np.testing.assert_array_equal(probs_run, probs_pt)
        # 설정은 번들 안의 inference_config.json 에서 복원했다고 찍혀야 한다 (기본값으로 돈 것처럼 보이면 안 된다).
        self.assertIn("설정 출처: mission3.pt:final_s42", log.getvalue())
        self.assertNotIn("없음(기본값 사용)", log.getvalue())

    def test_tfidf_member_round_trip(self) -> None:
        from m3.tfidf_member import fit_tfidf_member, save_tfidf_member
        from test_tfidf_member import toy_corpus

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            run = write_tiny_run(root, "final_s42", 42)
            texts, labels = toy_corpus()
            member = fit_tfidf_member(texts, labels, min_df=1)
            joblib_path = save_tfidf_member(member, root / "tfidf_lr.joblib")
            pt = pack_bundle([run], root / "mission3.pt", tfidf_path=joblib_path, tfidf_weight=0.3)
            bundle = load_bundle(pt)
            restored = load_bundle_tfidf(bundle)

            self.assertEqual(bundle.tfidf_weight, 0.3)
            self.assertEqual(bundle.tfidf_joblib, joblib_path.read_bytes())
            np.testing.assert_array_equal(restored.predict_proba(texts), member.predict_proba(texts))
            self.assertIn("TF-IDF", describe(bundle)[-1])
            del bundle
            gc.collect()


class BundleValidationTest(unittest.TestCase):
    """가중치가 필요 없는 형식 검사. 토크나이저 캐시가 없어도 돈다."""

    def _fake_run(self, root: Path) -> Path:
        from safetensors.torch import save_file

        model = root / "run" / "best_model"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        save_file({"w": torch.zeros(3)}, str(model / "model.safetensors"))
        return root / "run"

    def test_pack_rejects_bad_arguments(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            run = self._fake_run(root)
            (root / "m.joblib").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "precision"):
                pack_bundle([run], root / "a.pt", precision="int8")
            with self.assertRaises(ValueError):
                pack_bundle([], root / "b.pt")
            with self.assertRaises(FileNotFoundError):
                pack_bundle([root / "missing"], root / "c.pt")
            with self.assertRaisesRegex(ValueError, "tfidf_weight"):
                pack_bundle([run], root / "d.pt", tfidf_path=root / "m.joblib", tfidf_weight=1.0)

    def test_zero_weight_drops_the_tfidf_member(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            run = self._fake_run(root)
            (root / "m.joblib").write_bytes(b"x")
            bundle = load_bundle(pack_bundle([run], root / "a.pt", tfidf_path=root / "m.joblib", tfidf_weight=0.0))
            self.assertIsNone(bundle.tfidf_joblib)
            self.assertIsNone(load_bundle_tfidf(bundle))
            del bundle
            gc.collect()

    def test_load_rejects_foreign_or_newer_files(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            torch.save({"state_dict": {}}, root / "foreign.pt")
            torch.save({"format": BUNDLE_FORMAT, "version": 99, "members": []}, root / "newer.pt")
            torch.save({"format": BUNDLE_FORMAT, "version": 1, "members": []}, root / "empty.pt")
            for name in ("foreign.pt", "newer.pt", "empty.pt"):
                with self.subTest(file=name), self.assertRaises(ValueError):
                    load_bundle(root / name)
            gc.collect()


if __name__ == "__main__":
    unittest.main()
