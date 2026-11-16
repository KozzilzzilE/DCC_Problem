import ast
import importlib.util
import sys
from pathlib import Path
import subprocess
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from m4.mixed_sampling import sample_epoch, validate, summary
spec = importlib.util.spec_from_file_location('mixed_runner', HERE/'scripts/prefix_experiment.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


@pytest.mark.parametrize('epoch', [1,2,3])
def test_original_uniform_and_shuffle(epoch):
    ns = [1,2,4,5,10,21,98]*30
    old = np.random.default_rng([42,epoch])
    expected = [(i,int(old.integers(1,n+1))) for i,n in enumerate(ns) for _ in range(2)]
    order = old.permutation(len(expected))
    for mode,p in [('uniform',.2),('mixed',0),('mixed',.2),('mixed',1)]:
        pairs,rng,flags = sample_epoch(ns,42,epoch,mode,p)
        assert np.array_equal(rng.permutation(len(pairs)), order)
        assert len(pairs)==2*len(ns)
        assert all(1<=t<=ns[i] for i,t in pairs)
        if mode=='uniform' or p==0:
            assert pairs==expected and not any(flags)
        if p==1:
            assert all(t==ns[i] for i,t in pairs) and all(flags)
        assert pairs==sample_epoch(ns,42,epoch,mode,p)[0]


@pytest.mark.parametrize('p', [-.1,1.1,float('nan'),float('inf')])
def test_bad_probability(p):
    with pytest.raises(ValueError):
        sample_epoch([1,20],42,1,'mixed',p)


def test_conflicting_include_full():
    with pytest.raises(ValueError):
        validate('mixed',.2,2,True)
    with pytest.raises(ValueError):
        validate('mixed',.2,3,False)


def test_fit_counts_and_label_independence():
    from m4.data import load_cache,read_ids
    calls = load_cache(r.CALLS, read_ids(r.FIT))
    ns = [c.n_utt for c in calls]
    items,flags = [],[]
    for e in (1,2,3):
        a,_,f=sample_epoch(ns,42,e,'mixed',.2)
        # Independent transcription of the pre-analysis algorithm.
        g=np.random.default_rng([42,e]);mix=np.random.default_rng([42,e,3])
        expected=[(i,int(g.integers(1,n+1))) for i,n in enumerate(ns) for _ in range(2)]
        expected=[(i,int(ns[i])) if mix.random()<.2 else (i,t) for i,t in expected]
        assert a==expected
        items+=a;flags+=f
    before=sample_epoch(ns,42,1,'mixed')[0]
    for c in calls:
        c.labels=[1-x for x in c.labels]  # in-memory only
    assert before==sample_epoch([c.n_utt for c in calls],42,1,'mixed')[0]
    result=summary(items,ns,flags)
    assert result['exposures']==157680 and result['full_exposures']==36085
    assert result['bins_1_4_5_10_11_20_21plus']==[18954,28644,47711,62371]
    assert result['full_never_calls']==5672 and result['early_never_calls']==8695


def prepare(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'OUTPUTS',tmp_path)
    monkeypatch.setattr(r,'verify',lambda: ({},None,None))
    monkeypatch.setattr(r,'read',lambda p: dict(r.RECIPE,sampling_strategy='mixed',full_probability=.2))
    monkeypatch.setattr(r,'digest',lambda p: r.BACKBONE_SHA256.get(Path(p).name,'test'))


def test_runner_passes_options_and_failure_stops(monkeypatch,tmp_path):
    prepare(monkeypatch,tmp_path)
    invoked=[]
    def fail(script,args,log):
        invoked.append(script)
        assert args[-4:]==['--sampling-strategy','mixed','--full-probability','0.2']
        raise subprocess.CalledProcessError(1,script)
    monkeypatch.setattr(r,'invoke',fail)
    with pytest.raises(subprocess.CalledProcessError):
        r.run(tmp_path/'t3_mixed_training/run','mixed',sequential_after=True)
    assert invoked==['train_prefix.py']


def test_output_protection(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'OUTPUTS',tmp_path)
    with pytest.raises(ValueError):
        r.run(tmp_path/'localcontrol','mixed')
    existing=tmp_path/'existing';existing.mkdir()
    with pytest.raises(FileExistsError):
        r.reserve(existing)


def test_batch_resume_does_not_train(monkeypatch,tmp_path):
    prepare(monkeypatch,tmp_path)
    out=tmp_path/'t3_mixed_training/run';out.mkdir(parents=True)
    (out/'initial.json').write_text('{}')
    invoked=[]
    monkeypatch.setattr(r,'invoke',lambda script,args,log: invoked.append(script))
    monkeypatch.setattr(r,'write_evaluation',lambda *a,**kw: invoked.append('evaluate'))
    r.run(out,'mixed','batch')
    assert invoked==['prefix_probs.py','evaluate']


def test_sources_parse_and_evaluation_unchanged():
    for name in ['train_prefix.py','prefix_experiment.py','audit_t3_sampling.py']:
        ast.parse((HERE/'scripts'/name).read_text(encoding='utf-8'))
    source=(HERE/'scripts/prefix_experiment.py').read_text(encoding='utf-8')
    assert 'prefix_segments' not in source
    assert 'check=True' in source
    assert 'mixed_sampling' not in (HERE/'m4/streaming.py').read_text(encoding='utf-8')


def test_batch_failure_blocks_sequential(monkeypatch,tmp_path):
    prepare(monkeypatch,tmp_path)
    out=tmp_path/'t3_mixed_training/run';out.mkdir(parents=True)
    (out/'initial.json').write_text('{}')
    def fail(*a,**kw):
        raise subprocess.CalledProcessError(1,'extract')
    monkeypatch.setattr(r,'invoke',fail)
    monkeypatch.setattr(r,'sequential',lambda *a: pytest.fail('Sequential must not start'))
    with pytest.raises(subprocess.CalledProcessError):
        r.run(out,'mixed','batch',True)


def test_by_third_delta_uses_existing_metric():
    def report(x):
        return {m: dict(score=x,seconds=1,seconds_per_utterance=.1,
            metrics=({'by_third':{'앞':x,'가운데':x,'뒤':x}} if m=='4-1' else
                dict(macro_f1=x,earliness=x,halt_mean=x,halt_median=x))) for m in ('4-1','4-2')}
    delta=r.sequential_difference(report(.3),report(.2))
    assert delta['4-1']['metrics']['by_third']==pytest.approx({'앞':.1,'가운데':.1,'뒤':.1})
