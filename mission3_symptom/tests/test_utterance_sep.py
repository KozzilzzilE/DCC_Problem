from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

MISSION3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MISSION3_DIR))

from m3.config import DEFAULT_UTTERANCE_SEP_MODE, UTTERANCE_SEP_MODES, resolve_utterance_sep
from m3.labels import _parse_dialogue_text, load_transcripts_dir, read_transcript

# 화자·시간 메타데이터가 섞인 원본 형태. 규정상 text 외에는 절대 새어나가면 안 된다.
SAMPLE_UTTERANCES = [
    {"speaker": "1", "startAt": 0, "endAt": 1500, "text": "119 상황실입니다"},
    {"speaker": "0", "startAt": 1500, "endAt": 4200, "text": "어머니가 토하고 어지럽다고 해요"},
]


def write_sample_json(directory: Path, call_id: str = "sample-001") -> Path:
    path = directory / f"{call_id}.json"
    payload = {
        "utterances": SAMPLE_UTTERANCES,
        "symptom": ["구토", "어지러움", "골절"],
        "gender": "여",
        "address": "서울시 어딘가",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class UtteranceSeparatorTest(unittest.TestCase):
    def test_default_mode_keeps_existing_space_join(self) -> None:
        """기본값은 기존 동작 그대로여야 기존 CSV·실험 결과가 재현된다."""
        self.assertEqual(DEFAULT_UTTERANCE_SEP_MODE, "space")
        self.assertEqual(resolve_utterance_sep(DEFAULT_UTTERANCE_SEP_MODE), " ")
        self.assertEqual(
            _parse_dialogue_text(SAMPLE_UTTERANCES),
            "119 상황실입니다 어머니가 토하고 어지럽다고 해요",
        )

    def test_sep_mode_inserts_boundary_between_utterances(self) -> None:
        text = _parse_dialogue_text(SAMPLE_UTTERANCES, resolve_utterance_sep("sep"))
        self.assertEqual(
            text,
            "119 상황실입니다 [SEP] 어머니가 토하고 어지럽다고 해요",
        )

    def test_boundary_marker_never_wraps_the_whole_text(self) -> None:
        """구분자는 발화 사이에만 들어간다. 앞뒤에 붙으면 tokenizer 특수토큰과 충돌한다."""
        text = _parse_dialogue_text(SAMPLE_UTTERANCES, resolve_utterance_sep("sep"))
        self.assertFalse(text.startswith("[SEP]"))
        self.assertFalse(text.endswith("[SEP]"))
        self.assertEqual(text.count("[SEP]"), len(SAMPLE_UTTERANCES) - 1)

    def test_single_utterance_has_no_separator(self) -> None:
        text = _parse_dialogue_text(SAMPLE_UTTERANCES[:1], resolve_utterance_sep("sep"))
        self.assertNotIn("[SEP]", text)

    def test_blank_utterances_do_not_create_empty_turns(self) -> None:
        """빈 발화가 연속 구분자를 만들면 모델이 가짜 턴 경계를 학습한다."""
        utterances = [
            {"text": "첫 발화"},
            {"text": "   "},
            {"text": None},
            {"not_text": "무시"},
            {"text": "둘째 발화"},
        ]
        text = _parse_dialogue_text(utterances, resolve_utterance_sep("sep"))
        self.assertEqual(text, "첫 발화 [SEP] 둘째 발화")

    def test_turn_mode_uses_dedicated_marker(self) -> None:
        text = _parse_dialogue_text(SAMPLE_UTTERANCES, resolve_utterance_sep("turn"))
        self.assertEqual(text.count("[TURN]"), len(SAMPLE_UTTERANCES) - 1)
        self.assertNotIn("[SEP]", text)

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_utterance_sep("newline")
        self.assertEqual(set(UTTERANCE_SEP_MODES), {"space", "sep", "turn"})

    def test_metadata_never_leaks_in_any_mode(self) -> None:
        """대회 규정 회귀 방지: 구분자를 넣어도 speaker/시간/인적사항은 절대 포함되지 않는다."""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_sample_json(Path(tmp))
            for mode in UTTERANCE_SEP_MODES:
                record = read_transcript(path, sep_mode=mode)
                for forbidden in ("speaker", "startAt", "endAt", "1500", "4200", "서울시", "여"):
                    self.assertNotIn(forbidden, record.text, f"{mode} 모드에서 {forbidden} 누출")

    def test_read_transcript_still_filters_non_target_symptoms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_sample_json(Path(tmp))
            record = read_transcript(path, sep_mode="sep")
            self.assertEqual(record.symptoms, ("구토", "어지러움"))  # 골절 제거
            self.assertEqual(int(record.label_vector.sum()), 2)
            self.assertIn("[SEP]", record.text)

    def test_load_transcripts_dir_propagates_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_sample_json(Path(tmp), "call-a")
            write_sample_json(Path(tmp), "call-b")
            records = load_transcripts_dir(tmp, sep_mode="sep")
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertIn("[SEP]", record.text)


if __name__ == "__main__":
    unittest.main()


class SepModeVerificationTest(unittest.TestCase):
    """학습 CSV 와 선언한 모드가 어긋나면 조용히 점수만 떨어지므로 즉시 실패시킨다."""

    def test_declared_sep_but_text_has_no_marker(self) -> None:
        from m3.labels import verify_utterance_sep_mode
        with self.assertRaises(ValueError):
            verify_utterance_sep_mode(['경계 없는 본문입니다'], 'sep', source='train.csv')

    def test_declared_space_but_text_has_marker(self) -> None:
        from m3.labels import verify_utterance_sep_mode
        with self.assertRaises(ValueError):
            verify_utterance_sep_mode(['앞 [SEP] 뒤'], 'space')

    def test_matching_modes_pass(self) -> None:
        from m3.labels import verify_utterance_sep_mode
        verify_utterance_sep_mode(['앞 [SEP] 뒤', '다른 본문'], 'sep')
        verify_utterance_sep_mode(['경계 없는 본문'], 'space')
        verify_utterance_sep_mode(['앞 [TURN] 뒤'], 'turn')

    def test_empty_sample_is_rejected(self) -> None:
        from m3.labels import verify_utterance_sep_mode
        with self.assertRaises(ValueError):
            verify_utterance_sep_mode([], 'space')
