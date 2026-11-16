"""Label-independent training-only sampling; baseline RNG/shuffle is preserved."""
import numpy as np


def sample_epoch(lengths, seed, epoch, strategy="uniform", samples=2):
    if strategy not in ("uniform", "early1", "early5", "weighted"):
        raise ValueError(strategy)
    if strategy != "uniform" and samples != 2:
        raise ValueError("Early sampling requires two samples per call")
    rng = np.random.default_rng([seed, epoch])
    items = [(i, int(rng.integers(1, n + 1))) for i, n in enumerate(lengths)
             for _ in range(samples)]
    # Independent stream: uniform first slot and baseline shuffle remain identical.
    early = np.random.default_rng([seed, epoch, 5])
    if strategy != "uniform":
        for j, (i, t) in enumerate(items):
            n = lengths[i]
            if strategy == "weighted":
                weights = np.ones(n)
                weights[4:min(10, n)] = 2
                items[j] = (i, int(early.choice(n, p=weights / weights.sum())) + 1)
            elif j % samples == 1:
                low = 5 if strategy == "early5" and n >= 5 else 1
                items[j] = (i, int(early.integers(low, min(10, n) + 1)))
    return items, rng


def summary(items, lengths):
    ts = np.array([t for _, t in items])
    pairs = set(items)
    early_calls = {i for i, t in items if 5 <= t <= 10}
    counts = [int(((ts >= lo) & (ts <= hi)).sum())
              for lo, hi in ((1, 4), (5, 10), (11, 20), (21, max(lengths)))]
    return {"exposures": len(items), "bins_1_4_5_10_11_20_21plus": counts,
            "bin_fractions": [x / len(items) for x in counts],
            "unique_pairs": len(pairs), "duplicate_exposures": len(items) - len(pairs),
            "duplicate_fraction": 1 - len(pairs) / len(items),
            "unique_prefixes_per_call_mean": len(pairs) / len(lengths),
            "calls_without_5_10": len(lengths) - len(early_calls),
            "calls_without_5_10_fraction": 1 - len(early_calls) / len(lengths)}
