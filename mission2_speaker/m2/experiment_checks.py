"""Dependency-free experiment identity and call-split audit helpers."""
import hashlib
import json


def stable_spec_id(spec):
    """Runtime timing is metadata, not part of a resumable run's identity."""
    identity = {key: value for key, value in spec.items() if key != "model_init_seconds"}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def call_split_counts(calls, dev):
    """Read population counts from the actual cache rather than a prior run."""
    calls, dev = list(calls), [bool(value) for value in dev]
    if len(calls) != len(dev) or not calls:
        raise ValueError("Calls and split flags must be nonempty and aligned")
    train_calls = {str(call) for call, is_dev in zip(calls, dev) if not is_dev}
    dev_calls = {str(call) for call, is_dev in zip(calls, dev) if is_dev}
    if train_calls & dev_calls:
        raise ValueError("A call occurs in both Training and internal Dev")
    return {"train_calls": len(train_calls), "dev_calls": len(dev_calls),
            "train_utterances": len(dev) - sum(dev), "dev_utterances": sum(dev)}
