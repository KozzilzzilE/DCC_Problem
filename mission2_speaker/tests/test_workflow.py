"""CPU-free checks of split isolation, label mapping and ensemble decisions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2.experiment_workflow import index_utterances, split_training_calls, score_probabilities, compare_ensembles


class WorkflowTests(unittest.TestCase):
    def test_call_split_keeps_related_utterances_together(self):
        samples = [{"call_id": f"c{i}", "sample_id": f"c{i}:{j}"} for i in range(10) for j in range(3)]
        train, dev = split_training_calls(samples)
        self.assertFalse({x['call_id'] for x in train} & {x['call_id'] for x in dev})
        self.assertEqual(len(train) + len(dev), 30)
        self.assertEqual((train, dev), split_training_calls(samples))

    def test_prefixed_json_matches_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'audio').mkdir()
            (root / 'label').mkdir()
            (root / 'audio/TS_call01.wav').touch()
            obj = {'utterances': [{'startAt': 0, 'endAt': 1000, 'speaker': '신고자'}]}
            (root / 'label/TL_call01.json').write_text(json.dumps(obj), encoding='utf-8')
            rows, audit = index_utterances(root)
            self.assertEqual(rows[0]['label'], 1)
            self.assertEqual(audit['valid_samples'], 1)

    def test_validation_label_rejected_in_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'VS_call.wav').touch()
            (root / 'VL_call.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                index_utterances(root)

    def test_threshold_boundary_and_confusion(self):
        metrics = score_probabilities([0, 0, 1, 1], [0.1, 0.5, 0.4, 0.9])
        self.assertEqual(metrics['confusion_matrix'], [[1, 1], [1, 1]])
        self.assertEqual(metrics['accuracy'], 0.5)
        self.assertEqual(metrics['macro_f1'], 0.5)

    def test_nonfinite_probability_cannot_become_class_zero(self):
        for p in (float('nan'), float('inf'), -0.1, 1.1):
            with self.assertRaises(ValueError):
                score_probabilities([0], [p])

    def test_ensemble_selects_seven_equal_combinations(self):
        vectors = {'resnet50': [0.8, 0.8], 'redimnet': [0.1, 0.7], 'ecapa_tdnn': [0.1, 0.8]}
        rows = compare_ensembles([0, 1], vectors)
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[-1]['accuracy'], 1.0)
        self.assertEqual(rows[-1]['weights'], [1 / 3] * 3)

    def test_mismatched_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            compare_ensembles([0, 1], {'resnet50': [0], 'redimnet': [0, 1], 'ecapa_tdnn': [0, 1]})


if __name__ == '__main__':
    unittest.main()
