from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

MISSION3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MISSION3_DIR))

from m3.config import NUM_CLASSES, TARGET_SYMPTOMS
from m3.infer import (
    DECISION_THRESHOLD,
    OUTPUT_COLUMNS,
    decision_threshold,
    format_symptoms,
    load_run_config,
    read_texts,
    resolve_model_dir,
    resolve_sep_mode,
    resolve_settings,
    symptoms_from_probabilities,
)


def write_transcript(directory: Path, call_id: str, texts, symptoms=()) -> Path:
    path = directory / f"{call_id}.json"
    payload = {
        "utterances": [
            {"speaker": str(i % 2), "startAt": i * 1000, "endAt": (i + 1) * 1000, "text": t}
            for i, t in enumerate(texts)
        ],
        "symptom": list(symptoms),
        "gender": "여",
        "address": "서울시 어딘가",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def make_bundle(root: Path, run_config: dict | None = None) -> Path:
    """run 디렉터리 형태의 최소 번들을 만든다 (가중치는 이 테스트에 필요 없다)."""
    run_dir = root / "runs" / "some_run"
    (run_dir / "best_model").mkdir(parents=True)
    (run_dir / "best_model" / "config.json").write_text("{}", encoding="utf-8")
    if run_config is not None:
        (run_dir / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False), encoding="utf-8")
    return run_dir


class FixedThresholdTest(unittest.TestCase):
    def test_threshold_is_competition_fixed_value(self) -> None:
        """대회 규정: 결정 임계값 0.5 고정. 저장된 보정값이 끼어들 자리가 없어야 한다."""
        self.assertEqual(DECISION_THRESHOLD, 0.5)
        self.assertEqual(decision_threshold(), 0.5)

    def test_probability_exactly_at_threshold_is_positive(self) -> None:
        probs = [0.0] * NUM_CLASSES
        probs[TARGET_SYMPTOMS.index("오심")] = 0.5
        self.assertEqual(symptoms_from_probabilities(probs), ["오심"])

    def test_probability_just_below_threshold_is_negative(self) -> None:
        probs = [0.499999] * NUM_CLASSES
        self.assertEqual(symptoms_from_probabilities(probs), [])

    def test_all_positive(self) -> None:
        self.assertEqual(symptoms_from_probabilities([0.9] * NUM_CLASSES), list(TARGET_SYMPTOMS))

    def test_wrong_vector_length_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            symptoms_from_probabilities([0.9] * 8)


class OutputFormatTest(unittest.TestCase):
    def test_zero_symptoms_is_empty_list_string(self) -> None:
        self.assertEqual(format_symptoms([]), "[]")

    def test_string_matches_submission_spec(self) -> None:
        self.assertEqual(format_symptoms(["두통", "복통"]), "['두통', '복통']")

    def test_order_follows_target_symptom_order(self) -> None:
        self.assertEqual(format_symptoms(["복통", "두통"]), "['두통', '복통']")

    def test_non_target_symptom_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            format_symptoms(["골절"])

    def test_output_columns_match_submission_spec(self) -> None:
        self.assertEqual(OUTPUT_COLUMNS, ["label file name", "symptom"])


class CheckpointResolutionTest(unittest.TestCase):
    def test_run_directory_resolves_to_best_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = make_bundle(Path(tmp), {"utterance_sep_mode": "sep"})
            model_dir, resolved_run = resolve_model_dir(run_dir)
            self.assertEqual(model_dir, run_dir / "best_model")
            self.assertEqual(resolved_run, run_dir)

    def test_best_model_directory_is_accepted_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = make_bundle(Path(tmp), {"utterance_sep_mode": "sep"})
            model_dir, resolved_run = resolve_model_dir(run_dir / "best_model")
            self.assertEqual(model_dir, run_dir / "best_model")
            self.assertEqual(resolved_run, run_dir)

    def test_file_inside_bundle_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = make_bundle(Path(tmp), {"utterance_sep_mode": "sep"})
            weights = run_dir / "best_model" / "model.safetensors"
            weights.write_bytes(b"")
            model_dir, _ = resolve_model_dir(weights)
            self.assertEqual(model_dir, run_dir / "best_model")

    def test_missing_bundle_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                resolve_model_dir(Path(tmp))


class SettingsRestoreTest(unittest.TestCase):
    def test_sep_mode_comes_from_run_config(self) -> None:
        self.assertEqual(resolve_sep_mode({"utterance_sep_mode": "sep"}), "sep")
        self.assertEqual(resolve_sep_mode({"utterance_sep_mode": "turn"}), "turn")

    def test_sep_mode_falls_back_to_train_csv_name(self) -> None:
        """구버전 run 은 모드를 기록하지 않았으므로 CSV 이름에서 유추한다."""
        self.assertEqual(resolve_sep_mode({"train_csv": "/x/mission3_train_sep.csv"}), "sep")

    def test_sep_mode_defaults_to_space_when_unknown(self) -> None:
        self.assertEqual(resolve_sep_mode({"train_csv": "/x/mission3_train.csv"}), "space")

    def test_settings_restore_encode_mode_and_max_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = make_bundle(Path(tmp), {
                "utterance_sep_mode": "sep", "encode_mode": "head_tail", "max_length": 256,
            })
            settings = resolve_settings(run_dir)
            self.assertEqual(settings.sep_mode, "sep")
            self.assertEqual(settings.encode_mode, "head_tail")
            self.assertEqual(settings.max_length, 256)

    def test_inference_config_overrides_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = make_bundle(Path(tmp), {"utterance_sep_mode": "space"})
            (run_dir / "best_model" / "inference_config.json").write_text(
                json.dumps({"utterance_sep_mode": "turn"}), encoding="utf-8")
            self.assertEqual(resolve_settings(run_dir).sep_mode, "turn")


class ReadTextsTest(unittest.TestCase):
    def test_reads_in_filename_order_and_applies_separator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_transcript(directory, "call-b", ["둘째 통화"])
            write_transcript(directory, "call-a", ["여보세요", "머리가 아파요"])
            names, texts = read_texts(directory, "sep")
            self.assertEqual(names, ["call-a.json", "call-b.json"])
            self.assertEqual(texts[0], "여보세요 [SEP] 머리가 아파요")

    def test_metadata_never_reaches_the_model_input(self) -> None:
        """대회 규정 회귀 방지: speaker/시간/인적사항이 본문에 섞이면 안 된다."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_transcript(directory, "call-a", ["여보세요"], symptoms=["두통"])
            for mode in ("space", "sep", "turn"):
                _, texts = read_texts(directory, mode)
                for forbidden in ("speaker", "startAt", "endAt", "1000", "서울시"):
                    self.assertNotIn(forbidden, texts[0], f"{mode} 모드에서 {forbidden} 누출")

    def test_empty_directory_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                read_texts(tmp, "space")


if __name__ == "__main__":
    unittest.main()


class SubmissionRobustnessTest(unittest.TestCase):
    """채점은 1회 실행이다. 여기서 죽거나 조용히 어긋나면 그대로 점수로 직결된다."""

    def test_uppercase_json_extension_is_found(self) -> None:
        """평가 데이터가 .JSON 으로 오더라도 찾지 못해 죽으면 안 된다."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = write_transcript(directory, "call-a", ["여보세요"])
            upper = directory / "CALL-B.JSON"
            upper.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

            names, texts = read_texts(directory, "space")
            self.assertEqual(len(names), 2, f"대소문자 확장자를 놓쳤다: {names}")
            self.assertIn("CALL-B.JSON", names)

    def test_missing_label_directory_raises_clear_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_texts("/존재하지/않는/폴더", "space")

    def test_best_model_alone_still_restores_settings(self) -> None:
        """best_model/ 만 제출해도 학습 설정이 복원돼야 한다 (부모 run_config 없이)."""
        with tempfile.TemporaryDirectory() as tmp:
            standalone = Path(tmp) / "best_model"
            standalone.mkdir(parents=True)
            (standalone / "config.json").write_text("{}", encoding="utf-8")
            (standalone / "inference_config.json").write_text(
                json.dumps({
                    "utterance_sep_mode": "sep",
                    "encode_mode": "head_tail",
                    "max_length": 256,
                }, ensure_ascii=False), encoding="utf-8")

            settings = resolve_settings(standalone)
            self.assertEqual(settings.sep_mode, "sep")
            self.assertEqual(settings.encode_mode, "head_tail")
            self.assertEqual(settings.max_length, 256)

    def test_submission_path_never_reads_optimized_thresholds(self) -> None:
        """규정상 임계값은 0.5 고정이다.

        reports/best_thresholds.json 은 이름이 그럴듯한 데다 synthetic 값이라,
        제출 경로가 이걸 읽기 시작하면 규정 위반인 동시에 성능 사고다.
        """
        import m3.infer as infer_module

        source = Path(infer_module.__file__).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines()
            if not line.strip().startswith("#") and not line.strip().startswith("(")
        )
        for forbidden in ("best_thresholds", "BEST_THRESHOLDS", "find_best_thresholds",
                          "apply_thresholds", "optimized_thresholds"):
            self.assertNotIn(
                forbidden, code.split('"""')[-1],
                f"제출 경로가 {forbidden} 를 참조하면 안 된다",
            )


class InferenceConfigBuilderTest(unittest.TestCase):
    """학습이 번들에 남기는 설정이 추론이 읽는 것과 실제로 맞물리는지 확인한다."""

    def test_round_trips_through_resolve_settings(self) -> None:
        from types import SimpleNamespace
        from m3.infer import build_inference_config

        training_config = SimpleNamespace(
            utterance_sep_mode="turn", encode_mode="head_tail",
            max_length=384, model_name_or_path="klue/roberta-base",
        )
        payload = build_inference_config(training_config)
        self.assertEqual(payload["threshold"], DECISION_THRESHOLD)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "best_model"
            bundle.mkdir(parents=True)
            (bundle / "config.json").write_text("{}", encoding="utf-8")
            (bundle / "inference_config.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            settings = resolve_settings(bundle)
            self.assertEqual(settings.sep_mode, "turn")
            self.assertEqual(settings.encode_mode, "head_tail")
            self.assertEqual(settings.max_length, 384)

    def test_defaults_when_fields_are_absent(self) -> None:
        from types import SimpleNamespace
        from m3.infer import build_inference_config

        payload = build_inference_config(SimpleNamespace())
        self.assertEqual(payload["utterance_sep_mode"], "space")
        self.assertEqual(payload["encode_mode"], "truncate")
        self.assertEqual(payload["max_length"], 512)
