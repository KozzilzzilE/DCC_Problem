import ast
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m4.prefix_sampling import sample_epoch, summary


@pytest.mark.parametrize('epoch', [1, 2, 3])
def test_original_and_shuffle(epoch):
    ns = [2, 4, 5, 10, 11, 98] * 100
    r = np.random.default_rng([42, epoch])
    expected = [(i, int(r.integers(1, n+1))) for i,n in enumerate(ns) for _ in range(2)]
    items, rng = sample_epoch(ns, 42, epoch)
    assert items == expected
    _, other = sample_epoch(ns, 42, epoch, 'weighted')
    expected_order = r.permutation(len(items))
    assert np.array_equal(rng.permutation(len(items)), expected_order)
    assert np.array_equal(other.permutation(len(items)), expected_order)


def test_weighted_bounds_reproducibility_and_coverage():
    ns = [1, 2, 4, 5, 10, 11, 98] * 100
    a = sample_epoch(ns, 42, 1, 'weighted')[0]
    assert a == sample_epoch(ns, 42, 1, 'weighted')[0]
    assert len(a) == len(ns)*2
    assert all(1 <= t <= ns[i] for i,t in a)
    assert all(sum(j == i for j,t in a) == 2 for i in range(len(ns)))
    assert any(t > 20 for i,t in a)
    assert summary(a, ns)['bin_fractions'][1] > summary(sample_epoch(ns,42,1)[0],ns)['bin_fractions'][1]
    # API accepts only lengths, never labels or dialogue contents.
    assert list(__import__('inspect').signature(sample_epoch).parameters) == ['lengths','seed','epoch','strategy','samples']
    with pytest.raises(ValueError):
        sample_epoch(ns,42,1,'weighted',1)


def test_sources_compile_and_inference_unchanged_interface():
    for name in ['train_prefix.py','prefix_experiment.py','analyze_t5.py']:
        ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'))
    text = (ROOT/'scripts/prefix_experiment.py').read_text(encoding='utf-8')
    assert 'check=True' in text
    assert 'exist_ok=False' in text
    assert 'sampling-strategy' not in (ROOT/'m4/streaming.py').read_text(encoding='utf-8')
