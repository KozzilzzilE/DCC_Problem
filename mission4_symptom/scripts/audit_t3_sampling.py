"""Fit-only CPU audit of the actual training sampler; never loads a classifier."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from m4.data import load_cache, read_ids, prefix_texts
from m4.mixed_sampling import sample_epoch, summary


def main():
    from transformers import AutoTokenizer
    calls = load_cache(HERE / 'runs/cache/train.jsonl', read_ids(HERE / 'runs/cache/fit_ids.txt'))
    lengths = [c.n_utt for c in calls]
    items, flags, orders = [], [], []
    for epoch in (1, 2, 3):
        pairs, rng, replaced = sample_epoch(lengths, 42, epoch, 'mixed', .2)
        orders.append(rng.permutation(len(pairs)))
        items.extend(pairs)
        flags.extend(replaced)
    needed = [set() for c in calls]
    for i,t in items:
        needed[i].add(t)
    tok = AutoTokenizer.from_pretrained(HERE.parent / 'mission3_symptom/runs/devsel/tapt', local_files_only=True)
    sizes, signatures = {}, {}
    for i,c in enumerate(calls):
        ps = prefix_texts(c)
        ts = sorted(needed[i])
        raw = tok([ps[t-1] for t in ts], truncation=False)['input_ids']
        clipped = tok([ps[t-1] for t in ts], truncation=True, max_length=512)['input_ids']
        for t,ids,visible in zip(ts,raw,clipped):
            sizes[i,t] = len(ids)
            signatures[i,t] = hashlib.sha256(np.array(visible, dtype=np.int32).tobytes()).hexdigest()
    result = summary(items, lengths, flags)
    result.update(scope='pretraining CPU reproduction, not actual training measurements',
        nominal_full_replacement_probability=.2, planned_optimizer_steps=len(items)//16,
        input_tokens=sum(min(sizes[x],512) for x in items),
        truncated_inputs=sum(sizes[x]>512 for x in items),
        unique_token_inputs=len({(i,signatures[i,t]) for i,t in items}))
    result['token_duplicate_fraction'] = 1-result['unique_token_inputs']/len(items)
    result['padded_tokens'] = sum(int(np.array([min(sizes[x],512) for x in items[e*52560:(e+1)*52560]])
        [order].reshape(-1,16).max(axis=1).sum())*16 for e,order in enumerate(orders))
    assert len(items)==157680 and result['full_exposures']==36085
    assert result['bins_1_4_5_10_11_20_21plus']==[18954,28644,47711,62371]
    assert result['full_never_calls']==5672 and result['early_never_calls']==8695
    assert result['unique_pairs']==133568 and result['unique_token_inputs']==133269
    assert result['input_tokens']==24174197 and result['padded_tokens']==60163392
    assert result['truncated_inputs']==1291 and result['planned_optimizer_steps']==9855
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
