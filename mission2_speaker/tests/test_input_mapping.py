"""Submission input ambiguity checks; no Torch or real audio required."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2.input_mapping import discover_input_files


class InputMappingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio, self.labels = self.root / "audio", self.root / "label"
        self.audio.mkdir()
        self.labels.mkdir()

    def touch(self, root, relative):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    def test_nested_paths_and_uppercase_wav_extension(self):
        label = self.touch(self.labels, "nested/call.json")
        wav = self.touch(self.audio, "nested/call.WAV")
        labels, wavs = discover_input_files(self.audio, self.labels)
        self.assertEqual(labels, [label])
        self.assertEqual(wavs, {"call": wav})

    def test_missing_audio_preserves_label_for_fallback(self):
        label = self.touch(self.labels, "missing.json")
        labels, wavs = discover_input_files(self.audio, self.labels)
        self.assertEqual(labels, [label])
        self.assertNotIn("missing", wavs)

    def test_duplicate_json_stems_fail_instead_of_dropping_rows(self):
        first = self.touch(self.labels, "a/call.json")
        second = self.touch(self.labels, "b/call.json")
        with self.assertRaisesRegex(ValueError, "Ambiguous duplicate JSON") as error:
            discover_input_files(self.audio, self.labels)
        self.assertIn(str(first), str(error.exception))
        self.assertIn(str(second), str(error.exception))

    def test_duplicate_wav_stems_fail_instead_of_selecting_first(self):
        self.touch(self.labels, "call.json")
        first = self.touch(self.audio, "a/call.wav")
        second = self.touch(self.audio, "b/call.WAV")
        with self.assertRaisesRegex(ValueError, "Ambiguous duplicate WAV") as error:
            discover_input_files(self.audio, self.labels)
        self.assertIn(str(first), str(error.exception))
        self.assertIn(str(second), str(error.exception))

    def test_empty_label_directory_fails(self):
        with self.assertRaisesRegex(FileNotFoundError, "No JSON"):
            discover_input_files(self.audio, self.labels)

    def test_non_directory_input_fails(self):
        file = self.touch(self.root, "audio_file")
        with self.assertRaisesRegex(FileNotFoundError, "audio_dir is not a directory"):
            discover_input_files(file, self.labels)


if __name__ == "__main__":
    unittest.main()
