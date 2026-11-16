import importlib.util
from pathlib import Path
import subprocess
import pytest

P = Path(__file__).resolve().parents[1] / 'scripts/prefix_experiment.py'
spec = importlib.util.spec_from_file_location('t5_runner', P)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def test_failed_train_blocks_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUTPUTS', tmp_path)
    monkeypatch.setattr(r, 'verify', lambda: ({}, None, None))
    monkeypatch.setattr(r, 'read', lambda p: r.RECIPE)
    monkeypatch.setattr(r, 'digest', lambda p: r.BACKBONE_SHA256.get(Path(p).name, 'test'))
    calls = []
    def fail(script, args, log):
        calls.append(script)
        raise subprocess.CalledProcessError(1, script)
    monkeypatch.setattr(r, 'invoke', fail)
    with pytest.raises(subprocess.CalledProcessError):
        r.run(tmp_path / 't5_early_sampling/run', 'weighted')
    assert calls == ['train_prefix.py']


def test_existing_output_protected(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUTPUTS', tmp_path)
    existing = tmp_path / 'existing'
    existing.mkdir()
    with pytest.raises(FileExistsError):
        r.reserve(existing)


def test_invoke_failure_propagates_and_retry_log_preserved(tmp_path, monkeypatch):
    log = tmp_path/'train.log'
    log.write_text('original')
    def fail(*a, **kw):
        assert kw['check'] is True
        raise subprocess.CalledProcessError(1, 'test')
    monkeypatch.setattr(r.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        r.invoke('train_prefix.py', [], log)
    assert log.read_text() == 'original'


def test_sequential_difference_uses_existing_by_third():
    def report(thirds):
        return {'4-1': {'score': .5, 'seconds': 10, 'seconds_per_utterance': .1,
                        'metrics': {'by_third': thirds}},
                '4-2': {'score': .4, 'seconds': 5, 'seconds_per_utterance': .2}}
    baseline = report({'앞': .2, '가운데': .5, '뒤': .7})
    current = report({'앞': .3, '가운데': .4, '뒤': .7})
    diff = r.sequential_difference(current, baseline)
    assert diff['4-1']['metrics']['by_third'] == pytest.approx(
        {'앞': .1, '가운데': -.1, '뒤': 0})
    assert r.sequential_difference(baseline, baseline)['4-1']['metrics']['by_third'] == {
        '앞': 0, '가운데': 0, '뒤': 0}
    assert baseline['4-1']['metrics']['by_third']['앞'] == .2
    source = P.read_text(encoding='utf-8')
    assert 'prefix_segments' not in source
    assert 'segment_scores' not in source
