"""Fit-only CPU sampling/tokenization audit. No classifier is loaded."""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from m4.data import load_cache, read_ids, prefix_texts
from m4.prefix_sampling import sample_epoch, summary


def main():
    from transformers import AutoTokenizer
    calls = load_cache(HERE / 'runs/cache/train.jsonl', read_ids(HERE / 'runs/cache/fit_ids.txt'))
    lengths = [c.n_utt for c in calls]
    plans = {s: [x for ep in (1, 2, 3) for x in sample_epoch(lengths, 42, ep, s)[0]]
             for s in ('uniform', 'early1', 'early5', 'weighted')}
    needed = [set() for _ in calls]
    for items in plans.values():
        for i, t in items:
            needed[i].add(t)
    tok = AutoTokenizer.from_pretrained(HERE.parent / 'mission3_symptom/runs/devsel/tapt', local_files_only=True)
    sizes = {}
    for i, c in enumerate(calls):
        ps = prefix_texts(c)
        ts = sorted(needed[i])
        ns = tok([ps[t-1] for t in ts], truncation=False, return_length=True)['length']
        sizes.update({(i, t): n for t, n in zip(ts, ns)})
    result = {'calls': len(calls), 'length_quantiles': dict(zip(['0', '25', '50', '75', '90', '95', '99', '100'],
        np.percentile(lengths, [0,25,50,75,90,95,99,100]).tolist())),
        'call_length_bins': summary(list(enumerate(lengths)), lengths)['bins_1_4_5_10_11_20_21plus'], 'candidates': {}}
    for s, items in plans.items():
        r = summary(items, lengths)
        r.update(input_tokens=sum(min(sizes[x],512) for x in items),
                 truncated_inputs=sum(sizes[x]>512 for x in items))
        result['candidates'][s] = r
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
