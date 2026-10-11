"""Training-only acoustic audit and small CNN probes; never loads DCC best weights.

prepare requires librosa, soundfile, scipy, numpy. audit uses numpy only.
probe additionally requires torch (and torchvision for resnet50). Cached features and checkpoints stay local.
These are exploratory results, not full-model or official Validation scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

FEATURES = ['duration_s', 'rms_mean', 'rms_std', 'centroid_mean_hz',
            'centroid_std_hz', 'zcr_mean', 'zcr_std']
GROUPS = {'length': [0], 'rms': [1, 2], 'centroid': [3, 4], 'zcr': [5, 6]}
CONDITIONS = ['baseline', 'short', 'rms', 'centroid', 'zcr', 'bandpass']
SR = 16000
SCHEMA = 2

MEL_PROFILES = {
    'default': {
        'n_mels': 80,
        'n_fft': 512,
        'hop_length': 160,
        'target_duration': 3.0,
        'frames': 301,
        'dtype': 'float16',
    },
    'resnet128': {
        'n_mels': 128,
        'n_fft': 2048,
        'hop_length': 512,
        'target_duration': 3.0,
        'frames': 94,
        'dtype': 'float32',
    }
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(values, name):
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite values: {name}')


def acoustic_features(y):
    """Unpadded utterance; magnitude centroid, frame RMS and crossing fraction.

    Left-aligned 2048-sample frames, hop 512; short utterances use their actual
    length. No padding, VAD, text or role-dependent transformations.
    """
    y = np.asarray(y, dtype=np.float64)
    if not len(y):
        raise ValueError('Empty audio')
    finite(y, 'audio')
    frame_size = min(2048, len(y))
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::512]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    magnitude = np.abs(np.fft.rfft(frames * np.hanning(frame_size), axis=1))
    hz = np.fft.rfftfreq(frame_size, 1 / SR)
    centroid = (magnitude @ hz) / np.maximum(magnitude.sum(axis=1), 1e-12)
    zcr = np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1) if frame_size > 1 else np.zeros(1)
    result = np.array([len(y) / SR, rms.mean(), rms.std(), centroid.mean(),
                       centroid.std(), zcr.mean(), zcr.std()])
    finite(result, 'features')
    return result


def canonical(stem):
    for prefix in ('VS_', 'VL_', 'TS_', 'TL_'):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def split_calls(calls, dev_fraction, seed):
    calls = sorted(set(calls))
    if len(calls) < 5:
        raise ValueError('At least 5 usable calls are required')
    order = np.random.default_rng(seed).permutation(len(calls))
    n_dev = max(1, min(len(calls) - 1, round(len(calls) * dev_fraction)))
    dev = {calls[i] for i in order[:n_dev]}
    return {call: ('dev' if call in dev else 'train') for call in calls}


def prepare(args):
    import librosa
    import soundfile as sf
    from scipy.signal import butter, filtfilt
    from m2.audio_features import extract_normalized_mel, normalize_clip

    root = args.training_root.resolve()
    if any(p.lower() in {'val', 'validation'} for p in root.parts):
        raise ValueError('Use Seoul Training, not provided Validation')
    if not root.is_dir():
        raise ValueError(f'Training directory not found: {root}')
    if args.cache_dir.exists() and any(args.cache_dir.iterdir()):
        raise ValueError('Cache directory must be empty; use a new run directory')
    labels = sorted(root.rglob('*.json'))
    wav_files = sorted(root.rglob('*.wav'))
    if any(part.lower() in {'val', 'validation'} for path in labels + wav_files for part in path.relative_to(root).parts):
        raise ValueError('The supplied root includes Validation; point to Training only')
    wav_map = {}
    for wav in wav_files:
        key = canonical(wav.stem)
        if key in wav_map and wav_map[key] != wav:
            raise ValueError(f'Ambiguous WAV stem: {key}; provide a unique Training layout')
        wav_map[key] = wav
    call_map = json.loads(args.call_map.read_text(encoding='utf-8')) if args.call_map else {}
    calls = {}
    missing = []
    seen = set()
    for label in labels:
        key = canonical(label.stem)
        if key not in wav_map:
            missing.append(str(label))
            continue
        metadata = json.loads(label.read_text(encoding='utf-8-sig'))
        if 'utterances' not in metadata:
            raise ValueError(f'Missing utterances schema: {label}')
        call = str(call_map.get(key, key))
        for utterance in metadata['utterances']:
            speaker = str(utterance.get('speaker')).strip()
            if speaker not in {'0', '1'}:
                raise ValueError(f'Unknown role in {label}: {speaker}')
            start, end = float(utterance['startAt']), float(utterance['endAt'])
            if not np.isfinite([start, end]).all() or start < 0 or end <= start:
                raise ValueError(f'Invalid millisecond boundaries: {label} {start}-{end}')
            sample_id = f'{key}:{start:g}:{end:g}'
            if sample_id in seen:
                continue
            seen.add(sample_id)
            calls.setdefault(call, []).append(dict(sample_id=sample_id, call_id=call,
                wav=str(wav_map[key]), start_ms=start, end_ms=end, speaker=int(speaker)))

    keys = sorted(calls)
    if getattr(args, 'exclude_calls', None) and args.exclude_calls.exists():
        exclude_set = set(json.loads(args.exclude_calls.read_text(encoding='utf-8')))
        keys = [k for k in keys if k not in exclude_set]
    if not keys:
        raise ValueError('No valid calls found')

    # 1. Train/Dev assignment on full Training calls using split seed
    assignments = split_calls(keys, args.dev_fraction, args.seed)

    # 2. Distinct sample-seed permutation for nested pilot selection
    sample_seed = getattr(args, 'sample_seed', None)
    if sample_seed is None:
        sample_seed = args.seed
    sample_rng = np.random.default_rng(sample_seed)
    permuted_calls = sample_rng.permutation(keys).tolist()
    selected_calls = sorted(permuted_calls[:min(args.max_calls, len(keys))])

    samples = []
    rng = np.random.default_rng(args.seed)
    for call in selected_calls:
        items = calls[call]
        if args.max_per_call and len(items) > args.max_per_call:
            items = [items[i] for i in sorted(rng.choice(len(items), args.max_per_call, replace=False))]
        samples.extend(dict(item, split=assignments[call]) for item in items)
    if not samples:
        raise ValueError('No matching Training WAV/JSON utterances')

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    profile_name = getattr(args, 'mel_profile', 'default')
    if profile_name not in MEL_PROFILES:
        raise ValueError(f'Unknown mel-profile: {profile_name}; choose from {list(MEL_PROFILES)}')
    prof = MEL_PROFILES[profile_name]
    n_mels = prof['n_mels']
    n_fft = prof['n_fft']
    hop_length = prof['hop_length']
    mel_dtype = np.float32 if prof['dtype'] == 'float32' else np.float16

    frames_3s = int(3.0 * SR / hop_length) + 1 if profile_name != 'default' else 301
    frames_1_5s = int(1.5 * SR / hop_length) + 1 if profile_name != 'default' else 151

    settings = dict(
        schema=SCHEMA, training_root=str(root), seed=args.seed, sample_seed=sample_seed,
        max_calls=args.max_calls, max_per_call=args.max_per_call, dev_fraction=args.dev_fraction,
        mel_profile=profile_name, sample_rate=SR, frame_length=2048, feature_hop=512,
        mel_bins=n_mels, mel_fft=n_fft, mel_hop=hop_length, mel_dtype=prof['dtype'],
        crop='center, no random augmentation', padding='zeros',
        bandpass='butter(4, [200,4000], fs=16000), filtfilt; padlen=min(27,n-1)',
        statistics='unpadded utterance, population std; centroid magnitude-weighted',
        grouping='explicit call map' if args.call_map else 'canonical WAV stem',
        model_selection='Training internal dev only', ready=False
    )
    settings['source_metadata'] = {str(wav_map[k]): [wav_map[k].stat().st_size, wav_map[k].stat().st_mtime_ns]
                                   for k in sorted({canonical(Path(x['wav']).stem) for x in samples})}
    settings['split_sha256'] = digest(assignments)
    settings['selected_calls_sha256'] = digest(selected_calls)
    settings['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    settings['cache_id'] = digest(dict(settings, samples=samples))
    write_json(args.cache_dir / 'config.json', settings)
    write_json(args.cache_dir / 'selected_calls.json', selected_calls)

    dedup_mel = getattr(args, 'dedup_mel', False)
    matrix_specs = [('baseline', frames_3s)] if dedup_mel else [('baseline', frames_3s), ('short', frames_1_5s), ('bandpass', frames_3s)]
    matrices = {name: np.lib.format.open_memmap(args.cache_dir / f'{name}.npy',
        mode='w+', dtype=mel_dtype, shape=(len(samples), n_mels, frames))
        for name, frames in matrix_specs}
    if not dedup_mel:
        b, a = butter(4, [200, 4000], btype='band', fs=SR)

    def compute_mel(y, seconds):
        target = int(seconds * SR)
        clip = normalize_clip(y, target_length=target, crop_mode='center')
        result = extract_normalized_mel(clip, sr=SR, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        finite(result, f'Mel_{profile_name}')
        return result.astype(mel_dtype)

    valid, features, excluded = [], [], []
    begin = time.perf_counter()
    current_wav, audio = None, None
    for row, item in enumerate(samples):
        if item['wav'] != current_wav:
            try:
                audio, sample_rate = sf.read(item['wav'], dtype='float32', always_2d=True)
                audio = audio.mean(axis=1)
                if sample_rate != SR:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SR)
                finite(audio, item['wav'])
            except Exception as exc:
                raise RuntimeError(f'Audio decoding failed: {item["wav"]}: {exc}') from exc
            current_wav = item['wav']
        st, et = int(item['start_ms'] * SR / 1000), int(item['end_ms'] * SR / 1000)
        if et > len(audio) + int(.01 * SR):
            raise ValueError(f'Utterance end exceeds WAV: {item["sample_id"]}')
        y = audio[st:min(et, len(audio))]
        if len(y) < 256:
            excluded.append(dict(item, reason='pilot minimum 256 real samples; not a submission rule'))
            continue
        features.append(acoustic_features(y))
        matrices['baseline'][row] = compute_mel(y, 3.)
        if not dedup_mel:
            matrices['short'][row] = compute_mel(y, 1.5)
            filtered = filtfilt(b, a, y, padlen=min(27, len(y) - 1)).astype(np.float32)
            matrices['bandpass'][row] = compute_mel(filtered, 3.)
        valid.append(dict(item, cache_row=row))
        if len(valid) % 500 == 0 or len(valid) == len(samples):
            print(f'Prepared {len(valid)}/{len(samples)} utterances', flush=True)

    for matrix in matrices.values():
        matrix.flush()
    if not valid:
        raise ValueError('No usable audio; cache is incomplete')
    y = np.array([r['speaker'] for r in valid], dtype=np.int64)
    dev = np.array([r['split'] == 'dev' for r in valid])
    for mask in [dev, ~dev]:
        if set(y[mask]) != {0, 1}:
            raise ValueError('Each split needs both roles; enlarge the sample before running')
    np.savez_compressed(args.cache_dir / 'features.npz',
        x=np.vstack(features), y=y, dev=dev,
        calls=np.array([r['call_id'] for r in valid]),
        rows=np.array([r['cache_row'] for r in valid]),
        sample_ids=np.array([r['sample_id'] for r in valid]))
    with (args.cache_dir / 'manifest.jsonl').open('w', encoding='utf-8') as f:
        for item in valid:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    write_json(args.cache_dir / 'split_manifest.json', assignments)
    if args.call_map:
        write_json(args.cache_dir / 'call_map.json', call_map)
    write_json(args.cache_dir / 'exclusions.json', dict(missing_audio_json=missing, excluded_utterances=excluded))
    settings.update(ready=True, valid_utterances=len(valid), cache_rows=len(samples),
        prepare_seconds=time.perf_counter() - begin, train_utterances=int((~dev).sum()),
        dev_utterances=int(dev.sum()), cache_bytes=sum(p.stat().st_size for p in args.cache_dir.glob('*.npy')))
    write_json(args.cache_dir / 'config.json', settings)
    print(json.dumps(settings, ensure_ascii=False, indent=2))


def load_cache(folder):
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if config['schema'] not in {1, 2} or not config['ready']:
        raise ValueError('Incomplete or unsupported cache; do not use its partial results')
    data = dict(np.load(folder / 'features.npz', allow_pickle=False))
    finite(data['x'], 'cached features')
    if set(data['calls'][data['dev']]) & set(data['calls'][~data['dev']]):
        raise ValueError('Call overlap between train and internal dev')
    return config, data


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def logistic_fit_predict(train, labels, dev):
    """Fixed L2=1 logistic regression, standardization fitted on train only."""
    mean, std = train.mean(axis=0), np.maximum(train.std(axis=0), 1e-6)
    x = np.column_stack([(train - mean) / std, np.ones(len(train))])
    xd = np.column_stack([(dev - mean) / std, np.ones(len(dev))])
    weights = np.zeros(x.shape[1])
    penalty = np.eye(len(weights))
    penalty[-1, -1] = 0
    for _ in range(100):
        p = sigmoid(x @ weights)
        hessian = (x.T * (p * (1 - p))) @ x + penalty + np.eye(len(weights)) * 1e-8
        step = np.linalg.solve(hessian, x.T @ (p - labels) + penalty @ weights)
        weights -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    probability = sigmoid(xd @ weights)
    finite(probability, 'logistic probabilities')
    return probability


def metrics(y, probability):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    finite(probability, 'predicted probabilities')
    pred = (probability >= .5).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    r0 = tn / max(1, tn + fp)
    r1 = tp / max(1, tp + fn)
    p0 = tn / max(1, tn + fn)
    p1 = tp / max(1, tp + fp)
    f0 = 2 * p0 * r0 / max(1e-12, p0 + r0)
    f1 = 2 * p1 * r1 / max(1e-12, p1 + r1)
    return dict(accuracy=float((y == pred).mean()), macro_f1=float(.5 * (f0 + f1)),
        recall_0=float(r0), recall_1=float(r1), confusion_matrix=[[tn, fp], [fn, tp]])


def paired_ci(y, p_cand, p_base, calls, iterations, seed):
    delta = ((p_cand >= .5) == y).astype(float) - ((p_base >= .5) == y).astype(float)
    keys, inverse = np.unique(calls, return_inverse=True)
    correct_delta = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        choice = rng.integers(len(keys), size=len(keys))
        values.append(correct_delta[choice].sum() / counts[choice].sum())
    return np.quantile(values, [.025, .975]).tolist()


def audit(args):
    config, data = load_cache(args.cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x, y, dev, calls = (data[k] for k in ['x', 'y', 'dev', 'calls'])
    full = logistic_fit_predict(x[~dev], y[~dev], x[dev])
    all_metrics = dict(full=metrics(y[dev], full))
    majority = float(y[~dev].mean() >= .5)
    all_metrics['majority'] = metrics(y[dev], np.full(dev.sum(), majority))
    rows, predictions = [], dict(y=y[dev], calls=calls[dev], full=full)
    for name, omitted in GROUPS.items():
        columns = [i for i in range(len(FEATURES)) if i not in omitted]
        reduced = logistic_fit_predict(x[~dev][:, columns], y[~dev], x[dev][:, columns])
        value = metrics(y[dev], reduced)
        ci = paired_ci(y[dev], full, reduced, calls[dev], args.bootstrap, args.seed)
        rows.append(dict(hypothesis=name, full_accuracy=all_metrics['full']['accuracy'],
            without_group_accuracy=value['accuracy'], full_minus_without_pp=100*(all_metrics['full']['accuracy']-value['accuracy']),
            paired_call_ci_low_pp=100*ci[0], paired_call_ci_high_pp=100*ci[1],
            without_group_macro_f1=value['macro_f1'], dev_samples=int(dev.sum())))
        all_metrics[f'without_{name}'] = value
        predictions[f'without_{name}'] = reduced
    distribution = []
    for index, name in enumerate(FEATURES):
        for split, mask in [('train', ~dev), ('dev', dev)]:
            for role in [0, 1]:
                values = x[mask & (y == role), index]
                distribution.append(dict(feature=name, split=split, role=role, n=len(values),
                    mean=float(values.mean()), std=float(values.std()), median=float(np.median(values))))
    write_csv(args.output_dir / 'hypothesis_ablation.csv', rows)
    write_csv(args.output_dir / 'feature_statistics.csv', distribution)
    write_json(args.output_dir / 'audit_metrics.json', dict(cache_id=config['cache_id'], metrics=all_metrics,
        interpretation='Exploratory association and logistic ablation; not causation or final CNN accuracy',
        bootstrap='paired by call, exploratory CI; no multiple-testing correction', seed=args.seed))
    np.savez_compressed(args.output_dir / 'audit_predictions.npz', **predictions)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def probe(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    config, data = load_cache(args.cache_dir)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Standardization fitted strictly on internal Train
    train_mask = ~data['dev']
    feat_mean = data['x'][train_mask].mean(axis=0)
    feat_std = np.maximum(data['x'][train_mask].std(axis=0), 1e-6)
    standardized = ((data['x'] - feat_mean) / feat_std).astype(np.float32)

    feature_stats = {
        'features': FEATURES,
        'train_mean': feat_mean.tolist(),
        'train_std': feat_std.tolist(),
        'train_samples': int(train_mask.sum()),
        'dev_samples': int(data['dev'].sum()),
    }
    write_json(args.output_dir / 'feature_stats.json', feature_stats)

    head_dropout = getattr(args, 'head_dropout', 0.0)
    is_smoke = bool(getattr(args, 'smoke', False))

    class PilotCNN(nn.Module):
        def __init__(self):
            super().__init__()
            if args.model == 'resnet50':
                from torchvision.models import resnet50, ResNet50_Weights
                self.encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
                old_conv = self.encoder.conv1
                self.encoder.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
                self.encoder.conv1.weight.data.copy_(old_conv.weight.data.mean(dim=1, keepdim=True))
                self.encoder.fc = nn.Identity()
                embedding_size = 2048
                if args.freeze_backbone:
                    self.encoder.requires_grad_(False)
            else:
                self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
                embedding_size = 64
            self.drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
            self.head = nn.Linear(embedding_size + 6, 1)

        def forward(self, mel, extra):
            emb = self.encoder(mel).flatten(1)
            feat = torch.cat([emb, extra], dim=1)
            return self.head(self.drop(feat)).flatten()

    class Clips(Dataset):
        def __init__(self, indices, condition):
            self.indices = indices
            self.condition = condition
            mel_name = condition if condition in {'short', 'bandpass'} else 'baseline'
            self.mels = np.load(args.cache_dir / f'{mel_name}.npy', mmap_mode='r')

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            index = self.indices[i]
            mel = np.array(self.mels[data['rows'][index]], dtype=np.float32)[None]
            extra = np.zeros(6, dtype=np.float32)
            if self.condition == 'rms':
                extra[0] = standardized[index, 1]
                extra[1] = standardized[index, 2]
            elif self.condition == 'centroid':
                extra[2] = standardized[index, 3]
                extra[3] = standardized[index, 4]
            elif self.condition == 'zcr':
                extra[4] = standardized[index, 5]
                extra[5] = standardized[index, 6]
            return torch.from_numpy(mel), torch.from_numpy(extra), torch.tensor(float(data['y'][index]))

    train_indices = np.flatnonzero(train_mask)
    dev_indices = np.flatnonzero(data['dev'])
    dev_sample_ids = data['sample_ids'][dev_indices] if 'sample_ids' in data else np.array([f'sample_{i}' for i in dev_indices])
    dev_calls = data['calls'][dev_indices]
    dev_y = data['y'][dev_indices]

    summary = []
    for condition in args.conditions:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        init_start = time.perf_counter()
        model = PilotCNN().to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        init_seconds = time.perf_counter() - init_start

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
        initial_hash = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            initial_hash.update(name.encode())
            initial_hash.update(tensor.detach().cpu().numpy().tobytes())

        spec = dict(
            initial_state_sha256=initial_hash.hexdigest(), cache_id=config['cache_id'],
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            model=args.model, initialization='ImageNet' if args.model == 'resnet50' else 'random',
            freeze_backbone=args.freeze_backbone, head_dropout=head_dropout, is_smoke=is_smoke,
            condition=condition, epochs=args.epochs, seed=args.seed, batch_size=args.batch_size,
            lr=args.lr, precision='FP32', device=str(device),
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            mean=feat_mean.tolist(), std=feat_std.tolist(), threshold=.5,
            model_init_seconds=init_seconds
        )
        run = args.output_dir / f'{condition}_seed{args.seed}'
        run.mkdir(parents=True, exist_ok=True)
        spec_id = digest(spec)
        write_json(run / 'run_config.json', spec)

        last = run / 'last.pt'
        start_epoch, history, best = 0, [], -1.
        if last.exists():
            state = torch.load(last, map_location='cpu', weights_only=False)
            if state['spec_id'] != spec_id:
                raise ValueError('Resume config changed; use a new output directory')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = state['epoch']
            history = state['history']
            best = state['best_accuracy']
            if 'torch_rng' in state and state['torch_rng'] is not None:
                torch.set_rng_state(state['torch_rng'])
            if device.type == 'cuda' and 'cuda_rng' in state and state['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            if 'numpy_rng' in state:
                np.random.set_state(state['numpy_rng'])
            if 'python_rng' in state:
                random.setstate(state['python_rng'])
            print(f'Resumed {condition} from completed epoch {start_epoch}/{args.epochs}')

        criterion = nn.BCEWithLogitsLoss()
        g_loader = torch.Generator()
        g_loader.manual_seed(args.seed)
        train_loader = DataLoader(Clips(train_indices, condition), batch_size=args.batch_size,
                                  shuffle=True, generator=g_loader)
        dev_loader = DataLoader(Clips(dev_indices, condition), batch_size=args.batch_size, shuffle=False)

        for epoch in range(start_epoch + 1, args.epochs + 1):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            model.train()
            total_loss = 0.0
            for mel_b, extra_b, y_b in train_loader:
                mel_b, extra_b, y_b = mel_b.to(device), extra_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits = model(mel_b, extra_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(y_b)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_train = time.perf_counter() - t_start

            if device.type == 'cuda':
                torch.cuda.synchronize()
            eval_start = time.perf_counter()
            model.eval()
            probs = []
            with torch.no_grad():
                for mel_b, extra_b, _ in dev_loader:
                    logits = model(mel_b.to(device), extra_b.to(device))
                    probs.append(torch.sigmoid(logits).cpu().numpy())
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_eval = time.perf_counter() - eval_start

            probability = np.concatenate(probs) if probs else np.array([])
            value = metrics(dev_y, probability)
            improved = value['accuracy'] > best
            best = max(best, value['accuracy'])
            history.append(dict(
                epoch=epoch, accuracy=value['accuracy'], macro_f1=value['macro_f1'],
                recall_0=value['recall_0'], recall_1=value['recall_1'],
                train_loss=total_loss / max(1, len(train_indices)),
                train_seconds=t_train, eval_seconds=t_eval, epoch_seconds=t_train + t_eval,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == 'cuda' else None
            ))
            state = dict(
                spec_id=spec_id, model=model.state_dict(), optimizer=optimizer.state_dict(),
                epoch=epoch, history=history, best_accuracy=best, is_smoke=is_smoke,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                numpy_rng=np.random.get_state(), python_rng=random.getstate()
            )
            if improved:
                torch.save(state, run / 'best_dev.tmp')
                os.replace(run / 'best_dev.tmp', run / 'best_dev.pt')
                np.savez_compressed(run / 'best_dev_predictions.npz', probability=probability,
                    y=dev_y, calls=dev_calls, sample_id=dev_sample_ids)
            np.savez_compressed(run / 'last_predictions.npz', probability=probability,
                y=dev_y, calls=dev_calls, sample_id=dev_sample_ids)
            write_csv(run / 'epoch_metrics.csv', history)
            torch.save(state, run / 'last.tmp')
            os.replace(run / 'last.tmp', last)
            print(f'{condition} epoch {epoch}/{args.epochs}: accuracy={value["accuracy"]:.4f}, macro_f1={value["macro_f1"]:.4f}, train={t_train:.2f}s, eval={t_eval:.2f}s', flush=True)

        best_prediction = dict(np.load(run / 'best_dev_predictions.npz', allow_pickle=False))
        value_best = metrics(best_prediction['y'], best_prediction['probability'])
        last_prediction = dict(np.load(run / 'last_predictions.npz', allow_pickle=False))
        value_last = metrics(last_prediction['y'], last_prediction['probability'])
        summary.append(dict(
            condition=condition,
            best_internal_dev_accuracy=value_best['accuracy'],
            best_macro_f1=value_best['macro_f1'],
            best_recall_0=value_best['recall_0'],
            best_recall_1=value_best['recall_1'],
            last_epoch_accuracy=value_last['accuracy'],
            last_epoch_macro_f1=value_last['macro_f1'],
            last_epoch_recall_0=value_last['recall_0'],
            last_epoch_recall_1=value_last['recall_1'],
            parameters=spec['parameters'], epochs=args.epochs, seed=args.seed,
            is_smoke=is_smoke, head_dropout=head_dropout,
            train_samples=len(train_indices), dev_samples=len(dev_indices)
        ))

    write_csv(args.output_dir / f'pilot_comparison_seed{args.seed}.csv', summary)

    baseline_best_path = args.output_dir / f'baseline_seed{args.seed}' / 'best_dev_predictions.npz'
    baseline_last_path = args.output_dir / f'baseline_seed{args.seed}' / 'last_predictions.npz'
    if baseline_best_path.exists() and baseline_last_path.exists():
        baseline_spec = json.loads((baseline_best_path.parent / 'run_config.json').read_text(encoding='utf-8'))
        for key in ['cache_id', 'model', 'epochs', 'seed', 'batch_size', 'lr', 'precision', 'freeze_backbone', 'head_dropout']:
            if baseline_spec.get(key) != spec.get(key):
                raise ValueError(f'Baseline comparison config mismatch: {key}')
        baseline_best = dict(np.load(baseline_best_path, allow_pickle=False))
        baseline_last = dict(np.load(baseline_last_path, allow_pickle=False))
        comparisons = []
        n_boot = getattr(args, 'bootstrap', 2000)
        for row in summary:
            if row['condition'] == 'baseline':
                continue
            cand_best_path = args.output_dir / f'{row["condition"]}_seed{args.seed}' / 'best_dev_predictions.npz'
            cand_last_path = args.output_dir / f'{row["condition"]}_seed{args.seed}' / 'last_predictions.npz'
            cand_best = dict(np.load(cand_best_path, allow_pickle=False))
            cand_last = dict(np.load(cand_last_path, allow_pickle=False))
            if not np.array_equal(cand_best['y'], baseline_best['y']) or not np.array_equal(cand_best['calls'], baseline_best['calls']):
                raise ValueError('Unaligned dev predictions')
            ci_best = paired_ci(cand_best['y'], cand_best['probability'], baseline_best['probability'], cand_best['calls'], n_boot, args.seed)
            ci_last = paired_ci(cand_last['y'], cand_last['probability'], baseline_last['probability'], cand_last['calls'], n_boot, args.seed)
            comparisons.append(dict(
                condition=row['condition'],
                delta_accuracy_last_pp=100 * (row['last_epoch_accuracy'] - metrics(baseline_last['y'], baseline_last['probability'])['accuracy']),
                paired_ci_last_low_pp=100 * ci_last[0],
                paired_ci_last_high_pp=100 * ci_last[1],
                delta_macro_f1_last=row['last_epoch_macro_f1'] - metrics(baseline_last['y'], baseline_last['probability'])['macro_f1'],
                delta_recall_0_last_pp=100 * (row['last_epoch_recall_0'] - metrics(baseline_last['y'], baseline_last['probability'])['recall_0']),
                delta_recall_1_last_pp=100 * (row['last_epoch_recall_1'] - metrics(baseline_last['y'], baseline_last['probability'])['recall_1']),
                delta_accuracy_best_pp=100 * (row['best_internal_dev_accuracy'] - metrics(baseline_best['y'], baseline_best['probability'])['accuracy']),
                paired_ci_best_low_pp=100 * ci_best[0],
                paired_ci_best_high_pp=100 * ci_best[1],
                seed=args.seed,
                is_smoke=is_smoke
            ))
        write_csv(args.output_dir / f'pilot_differences_seed{args.seed}.csv', comparisons)


def self_test():
    from m2.audio_features import extract_normalized_mel, normalize_clip
    import torch
    import torch.nn as nn

    # 1. Split calls symmetry
    calls = [f'call{i}' for i in range(20)]
    split = split_calls(calls, .2, 42)
    assert split == split_calls(list(reversed(calls)), .2, 42)
    assert sum(v == 'dev' for v in split.values()) == 4

    # 2. Acoustic features
    sine = .2 * np.sin(2 * np.pi * 1000 * np.arange(SR) / SR)
    feat = acoustic_features(sine)
    assert abs(feat[0] - 1.) < 1e-12 and abs(feat[1] - .2 / np.sqrt(2)) < .001
    assert abs(feat[3] - 1000) < 2 and .11 < feat[5] < .14
    assert np.isfinite(acoustic_features(np.zeros(100))).all()
    try:
        acoustic_features(np.array([np.nan]))
        raise AssertionError('NaN was accepted')
    except ValueError:
        pass

    # 3. Logistic fit & metrics
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 2))
    y = (x[:, 0] > 0).astype(int)
    p = logistic_fit_predict(x[:150], y[:150], x[150:])
    assert metrics(y[150:], p)['accuracy'] >= .94
    perfect = np.array([0, 0, 1, 1])
    prob = np.array([.1, .2, .8, .9])
    assert metrics(perfect, prob)['confusion_matrix'] == [[2, 0], [0, 2]]
    assert metrics(perfect, prob)['macro_f1'] == 1.
    assert paired_ci(perfect, prob, prob, np.array(['a', 'a', 'b', 'b']), 50, 42) == [0., 0.]

    # 4. Mel extraction numerical & shape check with m2.audio_features
    dummy_audio = np.random.randn(48000).astype(np.float32)
    mel80 = extract_normalized_mel(dummy_audio, sr=16000, n_mels=80, n_fft=512, hop_length=160)
    mel128 = extract_normalized_mel(dummy_audio, sr=16000, n_mels=128, n_fft=2048, hop_length=512)
    assert mel80.shape == (80, 301)
    assert mel128.shape == (128, 94)
    assert np.all(np.isfinite(mel80)) and np.all(np.isfinite(mel128))
    assert 0.0 <= mel128.min() and mel128.max() <= 1.0

    # 5. Silence finite test
    silence = np.zeros(48000, dtype=np.float32)
    silence_mel = extract_normalized_mel(silence, sr=16000, n_mels=128, n_fft=2048, hop_length=512)
    assert silence_mel.shape == (128, 94)
    assert np.all(np.isfinite(silence_mel))

    # 6. Sample seed permutation nesting check
    test_keys = [f'c_{i:04d}' for i in range(1000)]
    s_rng = np.random.default_rng(4242)
    p_keys = s_rng.permutation(test_keys).tolist()
    sub500 = set(p_keys[:500])
    sub1000 = set(p_keys[:1000])
    assert sub500.issubset(sub1000)

    # 7. Model parameter count & initial state hash repeatability check
    def make_cnn(seed, dropout):
        torch.manual_seed(seed)
        enc = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        head = nn.Linear(16 + 6, 1)
        h = hashlib.sha256()
        for n, t in nn.Sequential(enc, head).state_dict().items():
            h.update(n.encode())
            h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest(), sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in head.parameters())

    h1, p1 = make_cnn(42, 0.3)
    h2, p2 = make_cnn(42, 0.3)
    h3, p3 = make_cnn(43, 0.3)
    assert h1 == h2 and h1 != h3
    assert p1 == p2 == p3

    print('Self-test passed: call split, audio features, non-finite rejection, logistic fit, metrics, paired bootstrap, mel128 shape/range, silence robustness, nested sample permutation, model param & init hash reproducibility')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('self-test')
    prep = commands.add_parser('prepare')
    prep.add_argument('--training-root', type=Path, required=True)
    prep.add_argument('--cache-dir', type=Path, required=True)
    prep.add_argument('--call-map', type=Path, help='Optional canonical WAV stem -> session ID JSON')
    prep.add_argument('--max-calls', type=int, default=200)
    prep.add_argument('--max-per-call', type=int, default=80)
    prep.add_argument('--dev-fraction', type=float, default=.2)
    prep.add_argument('--seed', type=int, default=42)
    prep.add_argument('--sample-seed', type=int, default=4242, help='Separate seed for sampling calls to avoid selection bias')
    prep.add_argument('--mel-profile', choices=list(MEL_PROFILES), default='default', help='Mel profile: default (80) or resnet128 (128)')
    prep.add_argument('--exclude-calls', type=Path, help='Optional JSON list of call IDs to exclude')
    prep.add_argument('--dedup-mel', action='store_true', help='Extract only baseline.npy for shared mel input and features.npz')
    for name in ['audit', 'probe']:
        sub = commands.add_parser(name)
        sub.add_argument('--cache-dir', type=Path, required=True)
        sub.add_argument('--output-dir', type=Path, required=True)
        sub.add_argument('--seed', type=int, default=42)
        if name == 'audit':
            sub.add_argument('--bootstrap', type=int, default=500)
        else:
            sub.add_argument('--bootstrap', type=int, default=2000, help='Bootstrap iterations for paired CIs')
            sub.add_argument('--model', choices=['tiny', 'resnet50'], default='tiny')
            sub.add_argument('--freeze-backbone', action='store_true', help='Head-only ImageNet ResNet probe')
            sub.add_argument('--head-dropout', type=float, default=0.0, help='Dropout probability before head Linear layer')
            sub.add_argument('--smoke', action='store_true', help='Mark as exploratory timing/smoke run')
            sub.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
            sub.add_argument('--epochs', type=int, default=3)
            sub.add_argument('--batch-size', type=int, default=32)
            sub.add_argument('--lr', type=float, default=.001)
            sub.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    if args.command == 'prepare' and (args.max_calls < 5 or args.max_per_call < 0 or not 0 < args.dev_fraction < 1):
        parser.error('max-calls >= 5, max-per-call >= 0, and 0 < dev-fraction < 1 required')
    if args.command == 'audit' and args.bootstrap < 1:
        parser.error('bootstrap must be positive')
    if args.command == 'probe' and args.freeze_backbone and args.model != 'resnet50':
        parser.error('freeze-backbone is for the ImageNet ResNet probe only')
    if args.command == 'probe' and (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0):
        parser.error('Positive epochs, batch size and learning rate required')
    {'prepare': prepare, 'audit': audit, 'probe': probe, 'self-test': lambda _: self_test()}[args.command](args)


if __name__ == '__main__':
    main()
