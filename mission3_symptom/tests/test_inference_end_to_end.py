"""제출 경로 end-to-end 검증.

tokenizer.pad -> model -> sigmoid -> 임계값 0.5 -> CSV 까지 실제로 통과시킨다.
주최 측은 `inference.py` 를 한 번만 실행하므로, 이 경로가 죽으면 곧바로 0점이다.
외부 다운로드 없이 임의 초기화한 소형 모델을 그 자리에서 만들어 쓴다.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

MISSION3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MISSION3_DIR))

from m3.config import NUM_CLASSES, TARGET_SYMPTOMS

try:
    import torch
    from transformers import AutoModelForSequenceClassification, BertConfig, BertTokenizer
    HAS_TORCH = True
except ImportError:  # pragma: no cover - 실행 환경에 따라 건너뛴다
    HAS_TORCH = False

SANITY_TEXT = "한국어 모델을 공유합니다."
SAMPLE_TEXTS = [
    ["여보세요 119죠", "어머니가 머리가 아프고 토해요"],
    ["환자가 숨을 못 쉬어요"],
    ["열이 많이 나요", "어지럽다고 합니다"],
]


def build_offline_bundle(model_dir: Path) -> None:
    """인터넷 없이 로드 가능한 최소 번들을 만든다."""
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    corpus = SANITY_TEXT + "".join("".join(t) for t in SAMPLE_TEXTS) + "[SEP][TURN]"
    chars = sorted({c for c in corpus if not c.isspace()})
    # 한글은 BERT 의 CJK 문자 분리 대상이 아니라 wordpiece 가 "##" 접두 조각을 찾는다.
    vocab = specials + [c for c in chars if c not in specials] + [f"##{c}" for c in chars]

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    # transformers 5.x 의 BertTokenizer 는 vocab_file 인자가 아니라 디렉터리 로드로 vocab 을 읽는다.
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "BertTokenizer", "do_lower_case": False}),
        encoding="utf-8")
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    tokenizer.save_pretrained(model_dir)

    config = BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=512,
        num_labels=NUM_CLASSES, problem_type="multi_label_classification",
        label2id={s: i for i, s in enumerate(TARGET_SYMPTOMS)},
        id2label={i: s for i, s in enumerate(TARGET_SYMPTOMS)},
    )
    model = AutoModelForSequenceClassification.from_config(config)
    model.save_pretrained(model_dir, safe_serialization=True)


def force_logits(model_dir: Path, bias: float) -> None:
    """분류기 bias 를 강제해 모든 확률을 임계값 위/아래로 몬다."""
    from safetensors.torch import load_file, save_file

    weights_path = model_dir / "model.safetensors"
    state = load_file(str(weights_path))
    for key in state:
        if key.endswith("classifier.weight"):
            state[key] = torch.zeros_like(state[key])
        if key.endswith("classifier.bias"):
            state[key] = torch.full_like(state[key], bias)
    save_file(state, str(weights_path), metadata={"format": "pt"})


def write_run(root: Path, sep_mode: str = "sep", bias: float | None = None) -> Path:
    run_dir = root / "runs" / "e2e"
    build_offline_bundle(run_dir / "best_model")
    if bias is not None:
        force_logits(run_dir / "best_model", bias)
    (run_dir / "run_config.json").write_text(json.dumps({
        "utterance_sep_mode": sep_mode, "encode_mode": "truncate", "max_length": 128,
        "train_csv": "mission3_train_sep.csv",
    }, ensure_ascii=False), encoding="utf-8")
    return run_dir


def write_labels(label_dir: Path) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    for index, utterances in enumerate(SAMPLE_TEXTS):
        payload = {
            "utterances": [
                {"speaker": str(i % 2), "startAt": i * 1000, "endAt": (i + 1) * 1000, "text": t}
                for i, t in enumerate(utterances)
            ],
            "gender": "여",
            "address": "서울시 어딘가",
        }
        (label_dir / f"call-{index:03d}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    # 본문이 비어 있는 통화도 평가 데이터에 섞일 수 있다.
    (label_dir / "call-empty.json").write_text(
        json.dumps({"utterances": []}, ensure_ascii=False), encoding="utf-8")


@unittest.skipUnless(HAS_TORCH, "torch/transformers 가 설치된 환경에서만 실행")
class SubmissionPathEndToEndTest(unittest.TestCase):
    def test_predicts_every_file_in_submission_format(self) -> None:
        from m3.infer import OUTPUT_COLUMNS, predict_directory

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = write_run(root)
            label_dir = root / "labels"
            write_labels(label_dir)

            frame = predict_directory(label_dir, run_dir, batch_size=2)

            self.assertEqual(list(frame.columns), OUTPUT_COLUMNS)
            self.assertEqual(len(frame), len(list(label_dir.glob("*.json"))))
            self.assertEqual(
                frame["label file name"].tolist(),
                sorted(p.name for p in label_dir.glob("*.json")),
            )
            for value in frame["symptom"]:
                parsed = ast.literal_eval(value)          # 채점 측이 복원할 수 있어야 한다
                self.assertIsInstance(parsed, list)
                for symptom in parsed:
                    self.assertIn(symptom, TARGET_SYMPTOMS)

    def test_all_probabilities_below_threshold_give_empty_lists(self) -> None:
        from m3.infer import predict_directory

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = write_run(root, bias=-20.0)       # sigmoid(-20) ~ 0
            label_dir = root / "labels"
            write_labels(label_dir)
            frame = predict_directory(label_dir, run_dir, batch_size=2)
            self.assertEqual(set(frame["symptom"]), {"[]"})

    def test_all_probabilities_above_threshold_give_all_nine(self) -> None:
        from m3.infer import predict_directory

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = write_run(root, bias=20.0)        # sigmoid(20) ~ 1
            label_dir = root / "labels"
            write_labels(label_dir)
            frame = predict_directory(label_dir, run_dir, batch_size=2)
            expected = str(list(TARGET_SYMPTOMS))
            self.assertEqual(set(frame["symptom"]), {expected})

    def test_csv_roundtrip_preserves_the_list_string(self) -> None:
        """콤마가 들어간 문자열이 CSV 를 왕복해도 원래 리스트로 복원돼야 한다."""
        import pandas as pd
        from m3.infer import predict_directory

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = write_run(root, bias=20.0)
            label_dir = root / "labels"
            write_labels(label_dir)
            frame = predict_directory(label_dir, run_dir, batch_size=2)

            out = root / "mission3.csv"
            frame.to_csv(out, index=False, encoding="utf-8-sig")   # main() 과 동일하게 저장
            restored = pd.read_csv(out, encoding="utf-8-sig")

            self.assertEqual(list(restored.columns), list(frame.columns))
            self.assertEqual(restored["symptom"].tolist(), frame["symptom"].tolist())
            self.assertEqual(ast.literal_eval(restored["symptom"].iloc[0]), list(TARGET_SYMPTOMS))

    def test_separator_mode_reaches_the_tokenizer(self) -> None:
        """run_config 의 sep_mode 가 실제 모델 입력까지 전달되는지 확인한다."""
        from m3.infer import read_texts, resolve_settings

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = write_run(root, sep_mode="sep")
            label_dir = root / "labels"
            write_labels(label_dir)

            settings = resolve_settings(run_dir)
            self.assertEqual(settings.sep_mode, "sep")
            _, texts = read_texts(label_dir, settings.sep_mode)
            self.assertIn("[SEP]", texts[0])

            tokenizer = BertTokenizer.from_pretrained(str(run_dir / "best_model"))
            ids = tokenizer(texts[0], add_special_tokens=True)["input_ids"]
            self.assertEqual(
                ids.count(tokenizer.sep_token_id), 2,
                "본문의 [SEP] 가 특수 토큰으로 처리되지 않았다면 경계 실험의 전제가 깨진다",
            )


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_TORCH, "torch/transformers 가 설치된 환경에서만 실행")
class StandaloneEntryPointTest(unittest.TestCase):
    """mission3_symptom/ 만 제출해도 동작해야 한다 (mission1 과 같은 구조)."""

    def test_writes_submission_csv(self) -> None:
        import importlib

        entry = importlib.import_module("inference") if str(MISSION3_DIR) in sys.path else None
        self.assertIsNotNone(entry, "mission3_symptom 이 sys.path 에 있어야 한다")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = write_run(root, bias=20.0)
            label_dir = root / "labels"
            write_labels(label_dir)
            out = root / "outputs" / "mission3.csv"

            exit_code = entry.main([
                "--label_dir", str(label_dir),
                "--ckpt_path", str(run_dir),
                "--output", str(out),
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue(out.is_file())

            import pandas as pd
            frame = pd.read_csv(out, encoding="utf-8-sig")
            self.assertEqual(list(frame.columns), ["label file name", "symptom"])
            self.assertEqual(len(frame), len(list(label_dir.glob("*.json"))))
            self.assertEqual(ast.literal_eval(frame["symptom"].iloc[0]), list(TARGET_SYMPTOMS))

    def test_entry_point_does_not_need_other_mission_dependencies(self) -> None:
        """librosa/torchvision 이 없어도 Mission 3 제출 경로는 살아 있어야 한다."""
        import builtins
        import importlib

        blocked_names = {"librosa", "torchvision", "torchaudio"}
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in blocked_names:
                raise ModuleNotFoundError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded
        try:
            for name in ("m3.infer", "m3.labels", "m3.model"):
                importlib.reload(importlib.import_module(name))
        finally:
            builtins.__import__ = real_import
