"""CPU regressions for Path B; synthetic inputs are not project performance evidence."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from m2 import path_b as pb
from m2 import infer
from m2.audio_features import extract_utterance_clip, load_and_resample_call


def tiny_model():
    return nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Dropout(.2), nn.Linear(4, 1))


class ToyData(Dataset):
    data_identity = {"sha256": "a" * 64, "method": "synthetic"}
    sr, window_sec, n_mels, n_fft, hop_length, crop_mode = 16000, 3., 80, 512, 160, "random"
    def __len__(self): return 8
    def __getitem__(self, i):
        noise = random.random() + float(np.random.rand()) + float(torch.rand(()))
        return torch.tensor([i / 8 + noise / 10]), torch.tensor(float(i % 2)), str(i)


class InterruptedLoader(DataLoader):
    def __iter__(self):
        self.iterations = getattr(self, 'iterations', 0) + 1
        if self.iterations == 2:
            raise RuntimeError('synthetic interruption')
        return super().__iter__()


class PathBTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.parent = self.root / 'parent.pt'
        torch.save(tiny_model().state_dict(), self.parent)
        self.model_patch = patch.object(pb, 'build_model', lambda _: tiny_model())
        self.model_patch.start()
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stderr = contextlib.redirect_stderr(io.StringIO())
        self.stdout.__enter__()
        self.stderr.__enter__()

    def tearDown(self):
        self.stderr.__exit__(None, None, None)
        self.stdout.__exit__(None, None, None)
        self.model_patch.stop()
        self.tmp.cleanup()

    def loader(self, cls=DataLoader, data=None):
        return cls(data or ToyData(), batch_size=4, shuffle=True,
                   generator=torch.Generator().manual_seed(101))

    def train(self, name='ecapa_tdnn', directory='run', loader=None, **kwargs):
        parent, extra = pb.MODEL_PLAN[name]
        return pb.train_path_b_model(name, self.parent, parent, extra,
                                     self.root / directory, loader or self.loader(),
                                     torch.device('cpu'), **kwargs)

    def config(self, smoke=False):
        paths = [self.train(name, is_smoke=smoke) for name in pb.MODEL_PLAN]
        return pb.freeze_model_config(self.root / 'run', *paths, allow_smoke=smoke), paths

    def test_interrupted_resume_equals_continuous_with_shuffle_dropout_and_three_rngs(self):
        uninterrupted = self.train(directory='continuous')
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.train(directory='resumed', loader=self.loader(InterruptedLoader))
        partial = torch.load(self.root / 'resumed/last_ecapa_tdnn.pt', weights_only=False)
        self.assertEqual(partial['completed_epoch'], 9)
        resumed = self.train(directory='resumed')
        a, b = [torch.load(p, weights_only=False) for p in (uninterrupted, resumed)]
        for k in a['model_state_dict']:
            self.assertTrue(torch.equal(a['model_state_dict'][k], b['model_state_dict'][k]), k)
        self.assertEqual(a['history'], b['history'])
        self.assertEqual(b['completed_epoch'], 10)
        self.assertEqual(b['resumed_from_epoch'], 9)

    def test_completed_training_and_resnet_registration_are_not_overwritten(self):
        for name in ['ecapa_tdnn', 'resnet50']:
            p = self.train(name)
            before = p.read_bytes()
            self.train(name)
            self.assertEqual(before, p.read_bytes())

    def test_changed_plan_data_lr_or_corrupt_metadata_fails_without_overwriting(self):
        p = self.train()
        before = p.read_bytes()
        with self.assertRaisesRegex(ValueError, 'different'):
            self.train(lr=1e-4)
        data = ToyData()
        data.data_identity = {'sha256': 'b' * 64}
        with self.assertRaisesRegex(ValueError, 'different'):
            self.train(loader=self.loader(data=data))
        with self.assertRaises(ValueError):
            pb.train_path_b_model('ecapa_tdnn', self.parent, 8, 1, self.root/'run', self.loader(), torch.device('cpu'))
        self.assertEqual(before, p.read_bytes())
        state = torch.load(p, weights_only=False)
        del state['rng_states']
        torch.save(state, p)
        corrupt = p.read_bytes()
        with self.assertRaises(ValueError): self.train()
        self.assertEqual(corrupt, p.read_bytes())

    def test_rng_roundtrip(self):
        state = pb.serialize_rng_state()
        expected = [random.random(), np.random.rand(), torch.rand(())]
        pb.restore_rng_state(state)
        actual = [random.random(), np.random.rand(), torch.rand(())]
        for a, b in zip(expected, actual): self.assertEqual(a, b)
        with self.assertRaises(ValueError): pb.restore_rng_state({'numpy': []})

    def test_freeze_rejects_incomplete_missing_history_smoke_and_changed_threshold(self):
        cfg, paths = self.config()
        original = paths[0].read_bytes()
        state = torch.load(paths[0], weights_only=False)
        for changed in [dict(state, completed_epoch=9), dict(state, history=[]), dict(state, is_smoke=True)]:
            torch.save(changed, paths[0])
            with self.assertRaises(ValueError): pb.freeze_model_config(self.root/'run', *paths)
        paths[0].write_bytes(original)
        with self.assertRaises(ValueError): pb.freeze_model_config(self.root/'run', *paths, threshold=.51)

    def test_export_checks_frozen_hash_and_preserves_existing_package(self):
        cfg, paths = self.config()
        destination = self.root/'submission'
        pb.export_submission_package(cfg, destination)
        original = (destination/'best_redimnet.pt').read_bytes()
        pb.export_submission_package(cfg, destination)
        state = torch.load(paths[0], weights_only=False)
        state['changed_after_freeze'] = True
        torch.save(state, paths[0])
        with self.assertRaisesRegex(ValueError, 'changed'):
            pb.export_submission_package(cfg, destination)
        self.assertEqual(original, (destination/'best_redimnet.pt').read_bytes())
        self.assertFalse((self.root/'submission2').exists())
        with self.assertRaises(ValueError): pb.export_submission_package(cfg, self.root/'submission2')
        self.assertFalse((self.root/'submission2').exists())

    def test_failed_export_leaves_no_partial_package(self):
        cfg, _ = self.config()
        destination = self.root/'failed_export'
        with patch.object(pb.shutil,'copy2',side_effect=OSError('synthetic copy failure')):
            with self.assertRaises(OSError): pb.export_submission_package(cfg,destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob('.failed_export-*')),[])

    def test_official_evaluation_rejects_smoke_before_loading_data(self):
        cfg, _ = self.config(smoke=True)
        with patch.object(pb, 'DCCAudioDatasetUnified', side_effect=AssertionError('data should not be loaded')):
            with self.assertRaises(ValueError):
                pb.evaluate_official_final_unified(cfg, self.root/'validation', torch.device('cpu'))

    def test_official_evaluation_rejects_misaligned_ids_and_labels(self):
        cfg, _ = self.config()
        for mismatch in ['ids', 'labels']:
            class FakeData(Dataset):
                def __init__(self, *a, n_mels=80, **k):
                    self.n_mels = n_mels
                    self.audit_info = {}
                def __len__(self): return 2
                def __getitem__(self, i):
                    sid = str(1-i) if self.n_mels == 128 and mismatch == 'ids' else str(i)
                    label = 1-i if self.n_mels == 128 and mismatch == 'labels' else i
                    return torch.tensor([float(i)]), torch.tensor(float(label)), sid
            with patch.object(pb, 'DCCAudioDatasetUnified', FakeData):
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    pb.evaluate_official_final_unified(cfg, self.root/'validation', torch.device('cpu'))

    def test_shared_preprocessing_matches_cli_for_8k_16k_44k_and_long_short_silent_clips(self):
        rng = np.random.default_rng(2)
        for sr in [8000, 16000, 44100]:
            root = self.root / f'audio{sr}'
            (root/'audio').mkdir(parents=True)
            (root/'label').mkdir()
            wave = rng.normal(0, .02, sr*5).astype(np.float32)
            sf.write(root/'audio/call.wav', wave, sr)
            utterances = [{'startAt':123, 'endAt':987, 'speaker':0},
                          {'startAt':100, 'endAt':4800, 'speaker':1}]
            (root/'label/call.json').write_text(json.dumps({'utterances':utterances}))
            full = load_and_resample_call(str(root/'audio/call.wav'))
            for mel, fft, hop in [(80,512,160),(128,2048,512)]:
                ds = pb.DCCAudioDatasetUnified(root, is_train=False, n_mels=mel,n_fft=fft,hop_length=hop)
                for i, utt in enumerate(utterances):
                    clip = extract_utterance_clip(full,utt['startAt'],utt['endAt'])
                    expected = infer.Mission2InferenceEngine._extract_normalized_mel(clip,16000,mel,fft,hop)
                    self.assertTrue(np.array_equal(ds[i][0].squeeze(0).numpy(),expected))
                silent = np.zeros(48000,np.float32)
                a = pb.extract_normalized_mel(silent,16000,mel,fft,hop)
                b = infer.Mission2InferenceEngine._extract_normalized_mel(silent,16000,mel,fft,hop)
                self.assertTrue(np.array_equal(a,b))
                self.assertTrue(np.isfinite(a).all())

    def test_training_validation_root_and_invalid_json_are_rejected(self):
        with self.assertRaises(ValueError): pb.validate_training_root(self.root/'validation')
        with self.assertRaises(ValueError): pb.validate_training_root(self.root/'x',self.root/'x')
        (self.root/'label').mkdir()
        (self.root/'label/bad.json').write_text('{invalid')
        with self.assertRaises(ValueError): pb.DCCAudioDatasetUnified(self.root)

    def test_fresh_notebook_define_only_finishes_without_model_or_dataset_execution(self):
        nb = json.loads((PKG/'Mission2_Speaker_Classification.ipynb').read_text())
        namespace = {'__name__':'__main__'}
        with patch.object(pb,'build_model',side_effect=AssertionError('model execution')), \
             patch.object(pb,'DCCAudioDatasetUnified',side_effect=AssertionError('dataset execution')):
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    exec(compile(''.join(cell['source']),f'cell{i}','exec'),namespace)
        self.assertEqual(namespace['RUN_STAGE'],'define_only')
        self.assertEqual(namespace['FROZEN_CONFIG_PATH'],namespace['OUTPUT_ROOT']/'final_model_config.json')

    def test_smoke_runner_never_calls_official_evaluation(self):
        spec = importlib.util.spec_from_file_location('runner',PKG/'experiments/run_path_b_fixed_continue.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        seen = []
        with patch.object(sys,'argv',['runner','--stage','smoke','--data_dir',str(self.root/'train'),
                                     '--val_dir',str(self.root/'validation'),'--output_root',str(self.root/'full'),
                                     '--smoke_output_root',str(self.root/'smoke')]), \
             patch.object(runner,'DCCAudioDatasetUnified',return_value=ToyData()), \
             patch.object(runner,'train_path_b_model',return_value=self.parent), \
             patch.object(runner,'freeze_model_config',return_value=self.root/'config.json'), \
             patch.object(runner,'evaluate_smoke_unified',side_effect=lambda **k:seen.append(k['train_data_root'])), \
             patch.object(runner,'evaluate_official_final_unified',side_effect=AssertionError('official eval')):
            runner.main()
        self.assertEqual(seen,[self.root/'train'])


if __name__ == '__main__': unittest.main()
