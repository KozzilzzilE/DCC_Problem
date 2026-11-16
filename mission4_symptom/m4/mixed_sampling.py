"""Training-only A20 sampling. No labels or inference state are accepted."""
import math
import numpy as np


def validate(strategy, probability, samples=2, include_full=False):
    if strategy not in ("uniform", "mixed"):
        raise ValueError("Unknown sampling strategy")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Full replacement probability must be finite and in [0, 1]")
    if strategy == "mixed" and (samples != 2 or include_full):
        raise ValueError("T3 requires two samples per call and include_full=False")


def sample_epoch(lengths, seed, epoch, strategy="uniform", probability=0.2, samples=2):
    validate(strategy, probability, samples)
    if any(n < 1 for n in lengths):
        raise ValueError("Calls must contain at least one utterance")
    rng = np.random.default_rng([seed, epoch])
    items = [(i, int(rng.integers(1, n + 1))) for i, n in enumerate(lengths)
             for _ in range(samples)]
    replaced = [False] * len(items)
    if strategy == "mixed":
        mix = np.random.default_rng([seed, epoch, 3])
        for j, (i, t) in enumerate(items):
            if mix.random() < probability:
                items[j] = (i, int(lengths[i]))
                replaced[j] = True
    return items, rng, replaced


def summary(items, lengths, replaced):
    ts = np.array([t for i, t in items])
    full = np.array([t == lengths[i] for i, t in items])
    def bins(mask):
        return [int(((ts >= a) & (ts <= b) & mask).sum())
                for a, b in ((1, 4), (5, 10), (11, 20), (21, max(lengths)))]
    full_calls = {i for (i, t), f in zip(items, full) if f}
    early_calls = {i for i, t in items if 5 <= t <= 10}
    unique = len(set(items))
    return dict(exposures=len(items), full_exposures=int(full.sum()),
        full_fraction=float(full.mean()), replacement_branch_full=int(sum(replaced)),
        uniform_branch_full=int(sum(f and not r for f, r in zip(full, replaced))),
        full_never_calls=len(lengths)-len(full_calls),
        full_never_fraction=1-len(full_calls)/len(lengths),
        bins_1_4_5_10_11_20_21plus=bins(np.ones(len(items), bool)),
        nonfull_bins=bins(~full), early_never_calls=len(lengths)-len(early_calls),
        early_never_fraction=1-len(early_calls)/len(lengths), unique_pairs=unique,
        duplicate_fraction=1-unique/len(items), unique_prefixes_per_call=unique/len(lengths))
