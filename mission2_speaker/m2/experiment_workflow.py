"""Reviewable experiment workflow: input settings, 10-epoch training, ensemble scoring.

Heavy dependencies are loaded only when an experiment is explicitly started.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time

MODEL_PROFILES = {
    "resnet50": {"n_mels": 128, "n_fft": 2048, "hop_length": 512},
    "redimnet": {"n_mels": 80, "n_fft": 512, "hop_length": 160},
    "ecapa_tdnn": {"n_mels": 80, "n_fft": 512, "hop_length": 160},
}
WINDOW_CONDITIONS = [
    {"name": "Pre-1", "window_sec": 1.5, "padding": "zero", "n_fft": 2048},
    {"name": "Pre-2", "window_sec": 3.0, "padding": "repeat", "n_fft": 2048},
    {"name": "Pre-3", "window_sec": 3.0, "padding": "zero", "n_fft": 2048},
    {"name": "Pre-4", "window_sec": 3.0, "padding": "zero", "n_fft": 1024},
]


def canonical(stem):
    for prefix in ("TS_", "TL_", "VS_", "VL_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def index_utterances(root, training=True):
    """Match WAV/JSON by call ID; keep labels only as loss/evaluation targets."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if training and any("validation" in part.lower() for part in root.parts):
        raise ValueError("Training root must not be the official Validation directory")
    audio = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() == ".wav":
            key = canonical(path.stem)
            if key in audio:
                raise ValueError(f"Ambiguous WAV call ID: {key}")
            audio[key] = path
    samples, missing = [], []
    seen = set()
    for path in sorted(root.rglob("*.json")):
        if training and path.stem.startswith(("VS_", "VL_")):
            raise ValueError(f"Validation label in Training root: {path.name}")
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if "utterances" not in obj:
            continue
        call_id = canonical(path.stem)
        if call_id not in audio:
            missing.append(call_id)
            continue
        for i, utt in enumerate(obj["utterances"]):
            speaker = utt.get("speaker", utt.get("role"))
            if speaker in (0, "0", "상황실"):
                label = 0
            elif speaker in (1, "1", "신고자"):
                label = 1
            else:
                raise ValueError(f"Invalid speaker in {path.name}, utterance {i}")
            start, end = float(utt["startAt"]), float(utt["endAt"])
            if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start:
                raise ValueError(f"Invalid boundary in {path.name}, utterance {i}")
            sample_id = f"{call_id}:{start:g}:{end:g}"
            if sample_id in seen:
                raise ValueError(f"Duplicate utterance: {sample_id}")
            seen.add(sample_id)
            samples.append({"sample_id": sample_id, "call_id": call_id,
                            "wav_path": str(audio[call_id]), "start_ms": start,
                            "end_ms": end, "label": label})
    if not samples:
        raise ValueError("No matched labelled utterances")
    return samples, {"missing_audio_calls": sorted(set(missing)), "valid_samples": len(samples)}


def split_training_calls(samples, dev_fraction=0.2, seed=42):
    """A fixed call-level split reused across all compared conditions."""
    if not 0 <= dev_fraction < 1:
        raise ValueError("dev_fraction must be in [0,1)")
    calls = sorted({s["call_id"] for s in samples})
    random.Random(seed).shuffle(calls)
    n_dev = max(1, round(len(calls) * dev_fraction)) if dev_fraction else 0
    if n_dev >= len(calls):
        raise ValueError("At least one Training call must remain")
    dev_calls = set(calls[:n_dev])
    train = [s for s in samples if s["call_id"] not in dev_calls]
    dev = [s for s in samples if s["call_id"] in dev_calls]
    return train, dev


def make_dataset(samples, model_name, training=False, window_sec=3.0, padding="zero", n_fft=None):
    import numpy as np
    import torch
    from .audio_features import load_and_resample_call, normalize_clip, extract_normalized_mel

    profile = dict(MODEL_PROFILES[model_name])
    if n_fft is not None:
        profile["n_fft"] = n_fft
    if window_sec <= 0 or padding not in ("zero", "repeat"):
        raise ValueError("Invalid window or padding")

    class AudioDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            row = samples[idx]
            audio = load_and_resample_call(row["wav_path"], target_sr=16000)
            left = max(0, int(row["start_ms"] * 16))
            right = min(len(audio), int(row["end_ms"] * 16))
            clip = audio[left:right]
            target = round(16000 * window_sec)
            if padding == "repeat" and 0 < len(clip) < target:
                clip = np.tile(clip, math.ceil(target / len(clip)))[:target]
            clip = normalize_clip(clip, target_length=target,
                                  crop_mode="random" if training else "center")
            mel = extract_normalized_mel(clip, **profile)
            return torch.from_numpy(mel).unsqueeze(0), torch.tensor(row["label"], dtype=torch.float32), row["sample_id"]

    return AudioDataset()


def build_model(model_name, resnet_init=None):
    """Same classifier structures as the submission inference engine."""
    import torch
    from .models import AudioResNet, ReDimNet2_B2, ECAPA_TDNN
    constructors = {"resnet50": AudioResNet, "redimnet": ReDimNet2_B2, "ecapa_tdnn": ECAPA_TDNN}
    model = constructors[model_name](num_classes=1)
    if model_name == "resnet50" and resnet_init:
        # Optional local torchvision ImageNet backbone; never download implicitly.
        state = torch.load(resnet_init, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        if state["conv1.weight"].shape[1] != 3:
            raise ValueError("Expected a 3-channel torchvision ResNet-50 initialization")
        state["conv1.weight"] = state["conv1.weight"].mean(dim=1, keepdim=True)
        missing, unexpected = model.resnet.load_state_dict(state, strict=False)
        if set(missing) != {"fc.1.weight", "fc.1.bias"} or unexpected:
            raise ValueError("Local ImageNet backbone keys do not match ResNet-50")
    return model


def score_probabilities(targets, probabilities, threshold=0.5):
    """Binary metrics from aligned probabilities; reject NaN/Inf explicitly."""
    targets, probabilities = list(targets), list(probabilities)
    if not targets or len(targets) != len(probabilities):
        raise ValueError("Targets/probabilities must be nonempty and aligned")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    cm = [[0, 0], [0, 0]]
    for y, p in zip(targets, probabilities):
        if y not in (0, 1) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid binary target/probability")
        cm[int(y)][int(p >= threshold)] += 1
    tn, fp = cm[0]
    fn, tp = cm[1]
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    return {"n": len(targets), "correct": tn + tp, "accuracy": (tn + tp) / len(targets),
            "macro_f1": (f0 + f1) / 2, "confusion_matrix": cm,
            "recall_0": tn / (tn + fp) if tn + fp else 0,
            "recall_1": tp / (tp + fn) if tp + fn else 0}


def compare_ensembles(targets, probabilities):
    """Three singles, three equal pairs and an equal triple, threshold fixed 0.50."""
    from itertools import combinations
    names = list(MODEL_PROFILES)
    if set(probabilities) != set(names):
        raise ValueError("Provide probabilities for all three models")
    if any(len(probabilities[name]) != len(targets) for name in names):
        raise ValueError("Each probability vector must match the target length")
    # Validate each constituent before averaging, including out-of-range outputs.
    for name in names:
        score_probabilities(targets, probabilities[name])
    rows = []
    for size in (1, 2, 3):
        for members in combinations(names, size):
            mean_p = [sum(probabilities[m][i] for m in members) / size for i in range(len(targets))]
            rows.append({"models": list(members), "weights": [1 / size] * size,
                         **score_probabilities(targets, mean_p)})
    return rows


def evaluate_models(evaluation_root, checkpoints_dir, output_dir, batch_size=32, device="auto"):
    """Aligned batch probabilities and seven ensemble scores from local weights."""
    import hashlib
    import torch
    from torch.utils.data import DataLoader
    from .infer import load_model_weights

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_dir, checkpoints_dir = Path(output_dir), Path(checkpoints_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use an empty evaluation output directory")
    samples, audit = index_utterances(evaluation_root, training=False)
    selected = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    filenames = {"resnet50": "best_resnet50.pt", "redimnet": "best_redimnet.pt", "ecapa_tdnn": "best_ecapa_tdnn.pt"}
    paths = {name: checkpoints_dir / filename for name, filename in filenames.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    targets = [row["label"] for row in samples]
    sample_ids = [row["sample_id"] for row in samples]
    probabilities, hashes, times = {}, {}, {}
    for name in MODEL_PROFILES:
        model = load_model_weights(build_model(name).to(selected), paths[name], selected)
        loader = DataLoader(make_dataset(samples, name), batch_size=batch_size, shuffle=False, num_workers=0)
        values, ids = [], []
        started = time.perf_counter()
        with torch.inference_mode():
            for inputs, _, current_ids in loader:
                values.extend(model(inputs.to(selected)).reshape(-1).sigmoid().cpu().tolist())
                ids.extend(current_ids)
        if ids != sample_ids:
            raise ValueError(f"Prediction order mismatch: {name}")
        score_probabilities(targets, values)
        probabilities[name] = values
        times[name] = time.perf_counter() - started
        hashes[name] = hashlib.sha256(paths[name].read_bytes()).hexdigest()
        del model
    result = {"targets": targets, "sample_ids": sample_ids, "probabilities": probabilities,
              "config": {"batch_size": batch_size, "device": str(selected), "threshold": 0.5,
                         "checkpoint_sha256": hashes, "model_seconds": times, "audit": audit}}
    comparisons = compare_ensembles(targets, probabilities)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, obj in (("model_probabilities.json", result), ("ensemble_metrics.json", comparisons)):
        (output_dir / filename).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    return comparisons


def train_model(model_name, training_root, output_dir, epochs=10, batch_size=32,
                lr=1e-4, seed=42, dev_fraction=0.2, device="auto",
                window_sec=3.0, padding="zero", n_fft=None, resnet_init=None):
    """Train epochs 1..10 in one loop; save the fixed final epoch, not Dev best."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    if epochs < 1 or batch_size < 2 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("epochs>=1, batch_size>=2 and finite lr>0 required")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory to preserve previous runs")
    samples, audit = index_utterances(training_root)
    train, dev = split_training_calls(samples, dev_fraction, seed)
    if len(train) < batch_size:
        raise ValueError("Training set is smaller than the selected batch size")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = build_model(model_name, resnet_init).to(device)
    dataset = make_dataset(train, model_name, True, window_sec, padding, n_fft)
    generator = torch.Generator().manual_seed(seed)
    # BatchNorm heads require at least two samples. The dropped tail is recorded.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        drop_last=True, generator=generator)
    dev_loader = DataLoader(make_dataset(dev, model_name, False, window_sec, padding, n_fft),
                            batch_size=batch_size, shuffle=False, num_workers=0) if dev else None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    config = {"model": model_name, "epochs": epochs, "batch_size": batch_size,
              "lr": lr, "seed": seed, "device": str(device), "selection_rule": "fixed_last",
              "window_sec": window_sec, "padding": padding, "mel_profile": MODEL_PROFILES[model_name],
              "n_fft_override": n_fft, "train_samples": len(train), "dev_samples": len(dev),
              "drop_last": True, "dropped_samples_per_epoch": len(train) % batch_size,
              "resnet_init": str(resnet_init) if resnet_init else "random", "precision": "float32", "audit": audit}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "split_manifest.json").write_text(json.dumps({
        "train_calls": sorted({s['call_id'] for s in train}),
        "dev_calls": sorted({s['call_id'] for s in dev})}, indent=2), encoding="utf-8")
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum = seen = 0
        for inputs, labels, sample_ids in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs).reshape(-1)
            loss = criterion(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training output: {list(sample_ids)}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Nonfinite training gradient")
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_samples_seen": seen}
        if dev_loader is not None:
            model.eval()
            targets, probs = [], []
            with torch.inference_mode():
                for inputs, labels, _ in dev_loader:
                    p = model(inputs.to(device)).reshape(-1).sigmoid().cpu().tolist()
                    targets.extend(labels.tolist())
                    probs.extend(p)
            row["internal_dev"] = score_probabilities(targets, probs)
        row["epoch_seconds"] = time.perf_counter() - started
        history.append(row)
        payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "scheduler_state_dict": scheduler.state_dict(), "completed_epoch": epoch,
                   "planned_epochs": epochs, "config": config, "history": history,
                   "rng_states": {"python": random.getstate(), "numpy": np.random.get_state(),
                                  "torch": torch.get_rng_state(), "loader": generator.get_state(),
                                  "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}}
        temp = output_dir / "last.tmp"
        torch.save(payload, temp)
        temp.replace(output_dir / f"last_{model_name}.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"{model_name} epoch {epoch}/{epochs}: loss={row['train_loss']:.4f}, {row['epoch_seconds']:.1f}s", flush=True)
    return output_dir / f"last_{model_name}.pt"
