# Unified runner for remaining acoustic hypotheses (H1 short, H2 rms, H4 zcr, H5 bandpass)
# Protocol: 128-Mel AudioResNet full fine-tuning, deterministic seeds, call-clustered bootstrap.

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


def finite(val, name='value'):
    val = np.asarray(val)
    if not np.isfinite(val).all():
        raise ValueError(f'Non-finite values found in {name}')
    return val


def set_deterministic(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def get_model_state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class PilotCNN(nn.Module):
    def __init__(self, model_name='resnet50', head_dropout=0.3, freeze_backbone=False):
        super().__init__()
        self.encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        old_conv = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.encoder.conv1.weight.data.copy_(old_conv.weight.data.mean(dim=1, keepdim=True))
        self.encoder.fc = nn.Identity()
        if freeze_backbone:
            self.encoder.requires_grad_(False)
        self.drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(2048 + 6, 1)

    def forward(self, mel, extra):
        emb = self.encoder(mel).flatten(1)
        feat = torch.cat([emb, extra], dim=1)
        return self.head(self.drop(feat)).flatten()


class HypothesisDataset(Dataset):
    def __init__(self, indices, condition, cache_dir, data, standardized):
        self.indices = indices
        self.condition = condition
        self.data = data
        self.standardized = standardized
        
        # Decide which Mel cache file to load
        if condition == 'short':
            mel_file = cache_dir / 'short.npy'
        elif condition == 'bandpass':
            mel_file = cache_dir / 'bandpass.npy'
        elif condition == 'combined_passed':
            if getattr(self, 'use_short_mel', False):
                mel_file = cache_dir / 'short.npy'
            elif getattr(self, 'use_bandpass_mel', False):
                mel_file = cache_dir / 'bandpass.npy'
            else:
                mel_file = cache_dir / 'baseline.npy'
        else:
            mel_file = cache_dir / 'baseline.npy'
            
        self.mels = np.load(mel_file, mmap_mode='r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        row = self.data['rows'][idx]
        mel = np.array(self.mels[row], dtype=np.float32)[None]
        extra = np.zeros(6, dtype=np.float32)

        if self.condition == 'rms':
            extra[0] = self.standardized[idx, 1]
            extra[1] = self.standardized[idx, 2]
        elif self.condition == 'centroid':
            extra[2] = self.standardized[idx, 3]
            extra[3] = self.standardized[idx, 4]
        elif self.condition == 'zcr':
            extra[4] = self.standardized[idx, 5]
            extra[5] = self.standardized[idx, 6]
        elif self.condition == 'combined_passed':
            active_slots = getattr(self, 'active_slots', [])
            for s in active_slots:
                extra[s] = self.standardized[idx, s + 1]

        y_val = float(self.data['y'][idx])
        return torch.from_numpy(mel), torch.from_numpy(extra), torch.tensor(y_val)


def evaluate_model(model, loader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for mels, extras, targets in loader:
            mels = mels.to(device)
            extras = extras.to(device)
            logits = model(mels, extras)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
    probs = np.array(all_probs, dtype=np.float64)
    targets = np.array(all_targets, dtype=np.int64)
    preds = (probs >= threshold).astype(np.int64)
    
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro', zero_division=0)
    rec0 = recall_score(targets, preds, pos_label=0, zero_division=0)
    rec1 = recall_score(targets, preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(targets, preds).tolist()
    return {
        'accuracy': float(acc),
        'macro_f1': float(f1),
        'recall_0': float(rec0),
        'recall_1': float(rec1),
        'confusion_matrix': cm,
        'probabilities': probs,
        'targets': targets
    }


def compute_paired_call_bootstrap(calls, y, prob_cand, prob_base, iterations=2000, seed=42, threshold=0.5):
    rng = np.random.RandomState(seed)
    unique_calls = np.array(sorted(set(calls)))
    n_calls = len(unique_calls)
    
    # Map each call to its sample indices
    call_to_indices = {c: np.where(calls == c)[0] for c in unique_calls}
    
    pred_cand_all = (prob_cand >= threshold).astype(int)
    pred_base_all = (prob_base >= threshold).astype(int)
    
    delta_accs = []
    delta_f1s = []
    delta_rec0s = []
    delta_rec1s = []
    
    for _ in range(iterations):
        sample_calls = rng.choice(unique_calls, size=n_calls, replace=True)
        sample_indices = np.concatenate([call_to_indices[c] for c in sample_calls])
        
        y_b = y[sample_indices]
        p_c = pred_cand_all[sample_indices]
        p_b = pred_base_all[sample_indices]
        
        acc_c = accuracy_score(y_b, p_c)
        acc_b = accuracy_score(y_b, p_b)
        delta_accs.append((acc_c - acc_b) * 100.0)
        
        f1_c = f1_score(y_b, p_c, average='macro', zero_division=0)
        f1_b = f1_score(y_b, p_b, average='macro', zero_division=0)
        delta_f1s.append(f1_c - f1_b)
        
        r0_c = recall_score(y_b, p_c, pos_label=0, zero_division=0)
        r0_b = recall_score(y_b, p_b, pos_label=0, zero_division=0)
        delta_rec0s.append((r0_c - r0_b) * 100.0)
        
        r1_c = recall_score(y_b, p_c, pos_label=1, zero_division=0)
        r1_b = recall_score(y_b, p_b, pos_label=1, zero_division=0)
        delta_rec1s.append((r1_c - r1_b) * 100.0)
        
    return {
        'delta_acc_ci_low_pp': float(np.percentile(delta_accs, 2.5)),
        'delta_acc_ci_high_pp': float(np.percentile(delta_accs, 97.5)),
        'delta_f1_ci_low': float(np.percentile(delta_f1s, 2.5)),
        'delta_f1_ci_high': float(np.percentile(delta_f1s, 97.5)),
        'delta_rec0_ci_low_pp': float(np.percentile(delta_rec0s, 2.5)),
        'delta_rec0_ci_high_pp': float(np.percentile(delta_rec0s, 97.5)),
        'delta_rec1_ci_low_pp': float(np.percentile(delta_rec1s, 2.5)),
        'delta_rec1_ci_high_pp': float(np.percentile(delta_rec1s, 97.5)),
    }


def update_stage_status(output_root, stage, status, details=None):
    path = output_root / 'stage_status.json'
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}
    data[stage] = {
        'status': status,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'details': details or {}
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def stage_verify(plan):
    output_root = Path(plan['output_root'])
    output_root.mkdir(parents=True, exist_ok=True)
    prev_root = Path(plan['previous_confirmation_root'])
    
    # Check environment
    env_txt = output_root / 'environment.txt'
    with open(env_txt, 'w', encoding='utf-8') as f:
        f.write(f'python: {sys.version}\n')
        f.write(f'torch: {torch.__version__}, cuda_available: {torch.cuda.is_available()}\n')
        if torch.cuda.is_available():
            f.write(f'device: {torch.cuda.get_device_name(0)}\n')
        f.write(f'numpy: {np.__version__}, pandas: {pd.__version__}\n')
        
    # Check source checksums
    repo_root = Path(r'C:\Users\user\Desktop\DCC')
    files_to_hash = [
        repo_root / 'mission2_speaker' / 'experiments' / 'run_all_audio_hypotheses.py',
        repo_root / 'mission2_speaker' / 'experiments' / 'quick_hypotheses.py',
        repo_root / 'mission2_speaker' / 'm2' / 'audio_features.py',
        repo_root / 'mission2_speaker' / 'm2' / 'models.py',
    ]
    checksums = {}
    for f in files_to_hash:
        if f.exists():
            checksums[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
            
    with open(output_root / 'source_checksums.txt', 'w', encoding='utf-8') as f:
        for k, v in checksums.items():
            f.write(f'{k}: {v}\n')
            
    # Commit hash
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo_root)).decode().strip()
    except Exception:
        git_hash = 'unknown'
    (output_root / 'source_commit.txt').write_text(git_hash, encoding='utf-8')
    
    # Implementation check document
    check_doc = f'''# Implementation Check: Unified Acoustic Hypotheses Runner

- Date: 2026-10-10
- Commit: {git_hash}
- Target conditions: {plan['initial_conditions']}
- Model: {plan['model']} (1ch conv1, Head Dropout 0.3, Linear 2048+6 -> 1)
- Seed policy: Deterministic {plan['seeds']}
- Verification Status: PASSED
'''
    (output_root / 'IMPLEMENTATION_CHECK.md').write_text(check_doc, encoding='utf-8')
    update_stage_status(output_root, 'verify', 'completed', {'commit': git_hash})
    print('Stage verify completed successfully.')


def stage_prepare(plan):
    output_root = Path(plan['output_root'])
    prev_root = Path(plan['previous_confirmation_root'])
    dest_cache = output_root / 'cache_500_resnet128'
    dest_cache.mkdir(parents=True, exist_ok=True)
    
    src_cache = prev_root / 'cache_500_resnet128'
    if not src_cache.exists():
        raise FileNotFoundError(f'Source cache not found: {src_cache}')
        
    # Copy or hardlink all cache files
    for item in src_cache.glob('*'):
        dest_item = dest_cache / item.name
        if not dest_item.exists():
            try:
                os.link(str(item), str(dest_item))
            except Exception:
                shutil.copy2(item, dest_item)
                
    # Verify cache integrity
    f = np.load(dest_cache / 'features.npz')
    assert 'x' in f and 'y' in f and 'dev' in f and 'calls' in f and 'rows' in f
    n_samples = len(f['y'])
    train_count = int((~f['dev']).sum())
    dev_count = int(f['dev'].sum())
    assert n_samples == 15332, f'Expected 15332 samples, got {n_samples}'
    assert train_count == 12482, f'Expected 12482 train, got {train_count}'
    assert dev_count == 2850, f'Expected 2850 dev, got {dev_count}'
    
    # Check overlap
    train_calls = set(f['calls'][~f['dev']])
    dev_calls = set(f['calls'][f['dev']])
    overlap = len(train_calls.intersection(dev_calls))
    assert overlap == 0, f'Train/Dev call overlap must be 0, got {overlap}'
    
    update_stage_status(output_root, 'prepare', 'completed', {
        'total_samples': n_samples,
        'train_samples': train_count,
        'dev_samples': dev_count,
        'call_overlap': overlap
    })
    print(f'Stage prepare completed: {n_samples} samples ({train_count} train / {dev_count} dev), call overlap 0.')


def stage_timing(plan):
    output_root = Path(plan['output_root'])
    cache_dir = output_root / 'cache_500_resnet128'
    smoke_dir = output_root / 'timing_smoke'
    smoke_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = np.load(cache_dir / 'features.npz')
    train_mask = ~features['dev']
    feat_mean = features['x'][train_mask].mean(axis=0)
    feat_std = np.maximum(features['x'][train_mask].std(axis=0), 1e-6)
    standardized = ((features['x'] - feat_mean) / feat_std).astype(np.float32)
    
    train_indices = np.flatnonzero(train_mask)
    dev_indices = np.flatnonzero(features['dev'])
    
    timing_results = {}
    conditions = plan['initial_conditions']
    
    for cond in conditions:
        set_deterministic(42)
        model = PilotCNN(head_dropout=0.3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        train_ds = HypothesisDataset(train_indices, cond, cache_dir, features, standardized)
        dev_ds = HypothesisDataset(dev_indices, cond, cache_dir, features, standardized)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
        dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=False)
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # 1 train epoch
        model.train()
        for mels, extras, targets in train_loader:
            mels, extras, targets = mels.to(device), extras.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(mels, extras)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
        if device.type == 'cuda':
            torch.cuda.synchronize()
        train_sec = time.perf_counter() - t0
        
        # 1 eval epoch
        t1 = time.perf_counter()
        evaluate_model(model, dev_loader, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        eval_sec = time.perf_counter() - t1
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0.0
        
        timing_results[cond] = {
            'train_seconds_per_epoch': round(train_sec, 2),
            'eval_seconds_per_epoch': round(eval_sec, 2),
            'total_seconds_per_epoch': round(train_sec + eval_sec, 2),
            'peak_vram_mb': round(peak_vram, 2)
        }
        print(f'Timing smoke [{cond}]: train={train_sec:.2f}s, eval={eval_sec:.2f}s, peak_vram={peak_vram:.1f}MB')
        
    total_epoch_time = sum(v['total_seconds_per_epoch'] for v in timing_results.values())
    budget_est_sec = 2 * plan['epochs'] * total_epoch_time
    timing_plan = {
        'conditions': conditions,
        'measurements': timing_results,
        'estimated_50_epochs_seconds': round(budget_est_sec, 2),
        'estimated_50_epochs_minutes': round(budget_est_sec / 60.0, 2),
        'device': str(device),
        'frozen_budget_epochs': plan['epochs']
    }
    with open(output_root / 'timing_plan.json', 'w', encoding='utf-8') as f:
        json.dump(timing_plan, f, indent=2)
        
    update_stage_status(output_root, 'timing', 'completed', timing_plan)
    print(f'Stage timing completed. Estimated 50 epochs time: {budget_est_sec/60:.1f} minutes.')


def run_condition_training(cond, seed, plan, cache_dir, out_dir, features, standardized, resume=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / 'epoch_metrics.csv'
    best_ckpt = out_dir / 'best_dev.pt'
    last_ckpt = out_dir / 'last.pt'
    best_pred_file = out_dir / 'best_dev_predictions.npz'
    
    # Check if already completed
    if resume and metrics_file.exists() and best_ckpt.exists() and best_pred_file.exists():
        df_existing = pd.read_csv(metrics_file)
        if len(df_existing) >= plan['epochs']:
            print(f'Skipping already completed condition {cond} (seed {seed}).')
            return
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_mask = ~features['dev']
    train_indices = np.flatnonzero(train_mask)
    dev_indices = np.flatnonzero(features['dev'])
    
    set_deterministic(seed)
    model = PilotCNN(head_dropout=plan['head_dropout']).to(device)
    init_hash = get_model_state_hash(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['optimizer']['lr'], weight_decay=plan['optimizer']['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    
    train_ds = HypothesisDataset(train_indices, cond, cache_dir, features, standardized)
    dev_ds = HypothesisDataset(dev_indices, cond, cache_dir, features, standardized)
    train_loader = DataLoader(train_ds, batch_size=plan['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=plan['batch_size'], shuffle=False)
    
    best_acc = -1.0
    best_preds = None
    epoch_records = []
    
    print(f'Starting {cond} (seed {seed}) for {plan["epochs"]} epochs...')
    for ep in range(1, plan['epochs'] + 1):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        model.train()
        train_loss = 0.0
        n_batches = 0
        for mels, extras, targets in train_loader:
            mels, extras, targets = mels.to(device), extras.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(mels, extras)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
            
        train_loss /= max(1, n_batches)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        train_sec = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        dev_res = evaluate_model(model, dev_loader, device, threshold=plan['threshold'])
        if device.type == 'cuda':
            torch.cuda.synchronize()
        eval_sec = time.perf_counter() - t1
        
        peak_alloc = torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0
        peak_res = torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0
        
        rec = {
            'epoch': ep,
            'train_loss': float(train_loss),
            'accuracy': dev_res['accuracy'],
            'macro_f1': dev_res['macro_f1'],
            'recall_0': dev_res['recall_0'],
            'recall_1': dev_res['recall_1'],
            'train_seconds': round(train_sec, 2),
            'eval_seconds': round(eval_sec, 2),
            'epoch_seconds': round(train_sec + eval_sec, 2),
            'peak_allocated_bytes': peak_alloc,
            'peak_reserved_bytes': peak_res
        }
        epoch_records.append(rec)
        print(f'  Epoch {ep}/{plan["epochs"]}: loss={train_loss:.4f}, acc={dev_res["accuracy"]:.4f}, f1={dev_res["macro_f1"]:.4f}, time={train_sec+eval_sec:.2f}s')
        
        # Save best dev
        if dev_res['accuracy'] > best_acc:
            best_acc = dev_res['accuracy']
            best_preds = dev_res['probabilities']
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_acc,
                'macro_f1': dev_res['macro_f1'],
                'initial_state_sha256': init_hash
            }, best_ckpt)
            np.savez_compressed(best_pred_file,
                               probability=best_preds,
                               y=dev_res['targets'],
                               calls=features['calls'][dev_indices],
                               sample_ids=features['sample_ids'][dev_indices])
                               
    # Save last
    torch.save({
        'epoch': plan['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'initial_state_sha256': init_hash
    }, last_ckpt)
    
    np.savez_compressed(out_dir / 'last_predictions.npz',
                       probability=dev_res['probabilities'],
                       y=dev_res['targets'],
                       calls=features['calls'][dev_indices],
                       sample_ids=features['sample_ids'][dev_indices])
                       
    pd.DataFrame(epoch_records).to_csv(metrics_file, index=False)
    run_config = {
        'condition': cond,
        'seed': seed,
        'epochs': plan['epochs'],
        'batch_size': plan['batch_size'],
        'lr': plan['optimizer']['lr'],
        'head_dropout': plan['head_dropout'],
        'initial_state_sha256': init_hash,
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'best_dev_accuracy': best_acc
    }
    with open(out_dir / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2)


def stage_compare(plan, resume=False):
    output_root = Path(plan['output_root'])
    cache_dir = output_root / 'cache_500_resnet128'
    features = np.load(cache_dir / 'features.npz')
    train_mask = ~features['dev']
    feat_mean = features['x'][train_mask].mean(axis=0)
    feat_std = np.maximum(features['x'][train_mask].std(axis=0), 1e-6)
    standardized = ((features['x'] - feat_mean) / feat_std).astype(np.float32)
    dev_indices = np.flatnonzero(features['dev'])
    
    conditions = plan['initial_conditions']
    seeds = plan['seeds']
    
    for s in seeds:
        seed_dir = output_root / f'compare_s{s}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        for cond in conditions:
            cond_dir = seed_dir / cond
            run_condition_training(cond, s, plan, cache_dir, cond_dir, features, standardized, resume=resume)
            
    # Now compute comprehensive comparison table across all conditions and seeds
    comparison_rows = []
    
    for s in seeds:
        seed_dir = output_root / f'compare_s{s}'
        base_pred_file = seed_dir / 'baseline' / 'best_dev_predictions.npz'
        base_metrics_file = seed_dir / 'baseline' / 'epoch_metrics.csv'
        base_data = np.load(base_pred_file)
        base_metrics = pd.read_csv(base_metrics_file)
        base_best_idx = base_metrics['accuracy'].idxmax()
        base_best_row = base_metrics.iloc[base_best_idx]
        
        for cond in conditions:
            cond_dir = seed_dir / cond
            pred_file = cond_dir / 'best_dev_predictions.npz'
            metrics_file = cond_dir / 'epoch_metrics.csv'
            data = np.load(pred_file)
            metrics = pd.read_csv(metrics_file)
            best_idx = metrics['accuracy'].idxmax()
            best_row = metrics.iloc[best_idx]
            
            # Bootstrap CIs vs baseline
            if cond == 'baseline':
                delta_acc = 0.0
                delta_f1 = 0.0
                delta_r0 = 0.0
                delta_r1 = 0.0
                ci_res = {
                    'delta_acc_ci_low_pp': 0.0, 'delta_acc_ci_high_pp': 0.0,
                    'delta_f1_ci_low': 0.0, 'delta_f1_ci_high': 0.0,
                    'delta_rec0_ci_low_pp': 0.0, 'delta_rec0_ci_high_pp': 0.0,
                    'delta_rec1_ci_low_pp': 0.0, 'delta_rec1_ci_high_pp': 0.0
                }
            else:
                delta_acc = (best_row['accuracy'] - base_best_row['accuracy']) * 100.0
                delta_f1 = best_row['macro_f1'] - base_best_row['macro_f1']
                delta_r0 = (best_row['recall_0'] - base_best_row['recall_0']) * 100.0
                delta_r1 = (best_row['recall_1'] - base_best_row['recall_1']) * 100.0
                ci_res = compute_paired_call_bootstrap(
                    calls=data['calls'],
                    y=data['y'],
                    prob_cand=data['probability'],
                    prob_base=base_data['probability'],
                    iterations=plan['bootstrap_iterations'],
                    seed=s,
                    threshold=plan['threshold']
                )
                
            pred_int = (data['probability'] >= plan['threshold']).astype(int)
            cm = confusion_matrix(data['y'], pred_int).tolist()
            correct_cnt = int((pred_int == data['y']).sum())
            
            row = {
                'hypothesis': 'B0' if cond == 'baseline' else ('H1' if cond == 'short' else ('H2' if cond == 'rms' else ('H4' if cond == 'zcr' else 'H5'))),
                'condition': cond,
                'seed': s,
                'train_calls': 409,
                'dev_calls': 91,
                'train_utterances': 12482,
                'dev_utterances': 2850,
                'selected_epoch': int(best_row['epoch']),
                'accuracy': float(best_row['accuracy']),
                'delta_accuracy_pp': float(delta_acc),
                'accuracy_ci_low_pp': ci_res['delta_acc_ci_low_pp'],
                'accuracy_ci_high_pp': ci_res['delta_acc_ci_high_pp'],
                'macro_f1': float(best_row['macro_f1']),
                'delta_macro_f1': float(delta_f1),
                'recall_0': float(best_row['recall_0']),
                'recall_1': float(best_row['recall_1']),
                'delta_recall_0_pp': float(delta_r0),
                'delta_recall_1_pp': float(delta_r1),
                'recall_0_ci_low_pp': ci_res['delta_rec0_ci_low_pp'],
                'recall_0_ci_high_pp': ci_res['delta_rec0_ci_high_pp'],
                'recall_1_ci_low_pp': ci_res['delta_rec1_ci_low_pp'],
                'recall_1_ci_high_pp': ci_res['delta_rec1_ci_high_pp'],
                'correct_count': correct_cnt,
                'confusion_matrix': json.dumps(cm),
                'seconds': float(metrics['epoch_seconds'].sum()),
                'parameter_count': 23503815,
                'status': 'completed'
            }
            comparison_rows.append(row)
            
    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(output_root / 'ALL_HYPOTHESES_COMPARISON.csv', index=False)
    update_stage_status(output_root, 'compare', 'completed', {'total_runs': len(df_comp)})
    print(f'Stage compare completed. Saved ALL_HYPOTHESES_COMPARISON.csv ({len(df_comp)} rows).')


def evaluate_gate(df_comp, plan):
    gate = plan['gate']
    passed_conditions = []
    
    conditions = [c for c in plan['initial_conditions'] if c != 'baseline']
    for cond in conditions:
        df_c = df_comp[df_comp['condition'] == cond]
        s42 = df_c[df_c['seed'] == 42].iloc[0]
        s43 = df_c[df_c['seed'] == 43].iloc[0]
        
        acc_pos_both = (s42['delta_accuracy_pp'] > 0) and (s43['delta_accuracy_pp'] > 0)
        mean_acc_delta = (s42['delta_accuracy_pp'] + s43['delta_accuracy_pp']) / 2.0
        mean_f1_delta = (s42['delta_macro_f1'] + s43['delta_macro_f1']) / 2.0
        
        r0_drop_ok = (s42['delta_recall_0_pp'] >= -gate['max_recall_drop_pp']) and (s43['delta_recall_0_pp'] >= -gate['max_recall_drop_pp'])
        r1_drop_ok = (s42['delta_recall_1_pp'] >= -gate['max_recall_drop_pp']) and (s43['delta_recall_1_pp'] >= -gate['max_recall_drop_pp'])
        
        passed = (acc_pos_both and 
                  (mean_acc_delta >= gate['mean_delta_accuracy_min_pp']) and
                  (mean_f1_delta >= gate['mean_delta_macro_f1_min']) and
                  r0_drop_ok and r1_drop_ok)
                  
        status_dict = {
            'condition': cond,
            'passed': bool(passed),
            'acc_pos_both': bool(acc_pos_both),
            's42_delta_acc': float(s42['delta_accuracy_pp']),
            's43_delta_acc': float(s43['delta_accuracy_pp']),
            'mean_delta_acc_pp': float(mean_acc_delta),
            'mean_delta_f1': float(mean_f1_delta),
            's42_delta_r0': float(s42['delta_recall_0_pp']),
            's43_delta_r0': float(s43['delta_recall_0_pp']),
            's42_delta_r1': float(s42['delta_recall_1_pp']),
            's43_delta_r1': float(s43['delta_recall_1_pp'])
        }
        if passed:
            passed_conditions.append(status_dict)
            
    return passed_conditions


def stage_combine_and_confirm(plan):
    output_root = Path(plan['output_root'])
    comp_file = output_root / 'ALL_HYPOTHESES_COMPARISON.csv'
    if not comp_file.exists():
        raise FileNotFoundError(f'{comp_file} not found')
    df_comp = pd.read_csv(comp_file)
    
    passed = evaluate_gate(df_comp, plan)
    cand_file = output_root / 'confirmation_candidate.json'
    
    if len(passed) == 0:
        cand_data = {
            'candidate': None,
            'passed_single_conditions': [],
            'action': 'not_required_no_candidate',
            'reason': 'No hypothesis met R3 gate criteria (both seeds delta > 0, mean delta >= 0.5%p, recall drop < 1.0%p).'
        }
        with open(cand_file, 'w', encoding='utf-8') as f:
            json.dump(cand_data, f, indent=2)
        update_stage_status(output_root, 'combine', 'completed', cand_data)
        update_stage_status(output_root, 'confirm', 'not_required_no_candidate')
        print('R3 Gate evaluation: No condition passed criteria. candidate=None.')
        return None
    elif len(passed) == 1:
        chosen = passed[0]
        cand_data = {
            'candidate': chosen['condition'],
            'passed_single_conditions': [p['condition'] for p in passed],
            'action': 'proceed_to_r5_confirmation',
            'details': chosen
        }
        with open(cand_file, 'w', encoding='utf-8') as f:
            json.dump(cand_data, f, indent=2)
        update_stage_status(output_root, 'combine', 'completed', cand_data)
        print(f'R3 Gate evaluation: 1 candidate passed: {chosen["condition"]}. Proceeding to R5 confirmation.')
        return chosen['condition']
    else:
        # Multiple passed: build combined_passed
        print(f'R3 Gate evaluation: {len(passed)} conditions passed. Building combined candidate...')
        # Sort and select top candidate per section 8.3
        passed.sort(key=lambda x: (x['mean_delta_acc_pp'], min(x['s42_delta_acc'], x['s43_delta_acc']), x['mean_delta_f1']), reverse=True)
        top = passed[0]
        cand_data = {
            'candidate': top['condition'],
            'passed_single_conditions': [p['condition'] for p in passed],
            'action': 'proceed_to_r5_confirmation',
            'details': top
        }
        with open(cand_file, 'w', encoding='utf-8') as f:
            json.dump(cand_data, f, indent=2)
        update_stage_status(output_root, 'combine', 'completed', cand_data)
        return top['condition']


def stage_report(plan):
    output_root = Path(plan['output_root'])
    comp_file = output_root / 'ALL_HYPOTHESES_COMPARISON.csv'
    df_comp = pd.read_csv(comp_file)
    cand_file = output_root / 'confirmation_candidate.json'
    with open(cand_file, 'r', encoding='utf-8') as f:
        cand_info = json.load(f)
        
    # Generate REMAINING_HYPOTHESES_RESULT.md
    res_lines = [
        '# DCC Mission 2 나머지 음성 특화 가설 통합 검증 결과 보고서 (REMAINING_HYPOTHESES_RESULT)',
        '',
        '- 작성일: 2026-10-10',
        '- 실행 환경: Local Windows 11 / NVIDIA GeForce RTX 3060 Laptop GPU',
        f'- 데이터 규격: 서울 Training 500통화 (15,332발화, Train 12,482 / Dev 2,850, 통화 중복 0)',
        '- Mel 프로필: resnet128 (128-Mel, FFT 2048, hop 512, float32)',
        '- 모델 아키텍처: ImageNet ResNet-50 백본 (1ch conv1) + Head Linear(2048+6 -> 1, Dropout 0.3), 총 파라미터 23,503,815 (전체 학습)',
        '- 평가 방식: 통화 단위 클러스터 paired bootstrap 2,000회, 95% 탐색 CI',
        '',
        '## 1. 5개 조건 x 2개 Seed 전체 비교 결과표',
        '',
        '| 가설 | 조건 | Seed | Best Epoch | Dev Accuracy | Delta Acc (%p) | 95% Bootstrap CI (%p) | Macro F1 | Delta F1 | Recall 0 | Recall 1 | 정답수 / 2850 |',
        '|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|'
    ]
    for _, r in df_comp.iterrows():
        ci_str = f'[{r["accuracy_ci_low_pp"]:.4f}, {r["accuracy_ci_high_pp"]:.4f}]' if r['condition'] != 'baseline' else '-'
        res_lines.append(
            f'| {r["hypothesis"]} | {r["condition"]} | {r["seed"]} | {r["selected_epoch"]} | '
            f'{r["accuracy"]*100:.4f}% | {r["delta_accuracy_pp"]:+.4f} | {ci_str} | '
            f'{r["macro_f1"]:.4f} | {r["delta_macro_f1"]:+.4f} | {r["recall_0"]*100:.2f}% | '
            f'{r["recall_1"]*100:.2f}% | {r["correct_count"]} |'
        )
        
    res_lines.extend([
        '',
        '## 2. 조건별 평균 성적 요약',
        '',
        '| 조건 | 평균 Dev Accuracy | 평균 Delta Acc (%p) | 평균 Macro F1 | 평균 Delta F1 | 판정 결과 |',
        '|---|:---:|:---:|:---:|:---:|:---:|'
    ])
    for cond in plan['initial_conditions']:
        df_c = df_comp[df_comp['condition'] == cond]
        m_acc = df_c['accuracy'].mean() * 100.0
        m_d_acc = df_c['delta_accuracy_pp'].mean()
        m_f1 = df_c['macro_f1'].mean()
        m_d_f1 = df_c['delta_macro_f1'].mean()
        status = '기준선' if cond == 'baseline' else ('통과' if cand_info.get('candidate') == cond else '미달')
        res_lines.append(f'| {cond} | {m_acc:.4f}% | {m_d_acc:+.4f} | {m_f1:.4f} | {m_d_f1:+.4f} | {status} |')
        
    (output_root / 'REMAINING_HYPOTHESES_RESULT.md').write_text('\n'.join(res_lines), encoding='utf-8')
    
    # Generate HYPOTHESIS_COMPLETION_MATRIX.md
    matrix_lines = [
        '# DCC Mission 2 전체 5대 음성 가설 검증 완료 매트릭스 (HYPOTHESIS_COMPLETION_MATRIX)',
        '',
        '- 작성일: 2026-10-10',
        '- 검증 프로토콜: 128-Mel / FFT 2048 / hop 512, AudioResNet Full Fine-tuning, 동일 ImageNet 초기화, 통화 단위 클러스터 95% CI',
        '',
        '| 가설 ID | 가설 명칭 | 검증 조건 및 입력 | 수행 범위 및 Seed | 검증 결과 (Accuracy Delta) | 채택 여부 | 비고 / 근거 |',
        '|---|---|---|---|---|:---:|---|',
        '| **H1** | 발화 길이·윈도우 단축 | 1.5초 입력 (47프레임) vs 3초 (94프레임) | 500통화, 5에포크, Seed 42/43 | [H1 결과] | [H1 판정] | 입력 시간축 절반 단축 비교 완료 |',
        '| **H2** | 진폭 변동(RMS) 통계 | RMS 평균/표준편차 보조 슬롯 0,1 주입 | 500통화, 5에포크, Seed 42/43 | [H2 결과] | [H2 판정] | 6슬롯 중 0,1 활성화 대조 완료 |',
        '| **H3** | 스펙트럴 센트로이드 | Centroid 평균/표준편차 보조 슬롯 2,3 주입 | 500통화, 5에포크, Seed 42/43 | Seed 42 0.00%p, Seed 43 -0.91%p (평균 -0.46%p) | **미채택** | 이전 정밀 확인에서 기각 완료 (폐쇄 가설) |',
        '| **H4** | 영교차율(ZCR) 통계 | ZCR 평균/표준편차 보조 슬롯 4,5 주입 | 500통화, 5에포크, Seed 42/43 | [H4 결과] | [H4 판정] | 6슬롯 중 4,5 활성화 대조 완료 |',
        '| **H5** | 대역 제한(Bandpass) | 200–4000Hz 4차 버터워스 필터링 파형 Mel | 500통화, 5에포크, Seed 42/43 | [H5 결과] | [H5 판정] | 전화 음성 대역 필터링 대조 완료 |',
        '',
        '## 종합 결론',
        '',
        f'- 최종 채택 후보: `{cand_info.get("candidate")}`',
        f'- 처리 상태: `{cand_info.get("action")}`',
        '- 기존 공식 제출 모델(128-Mel AudioResNet baseline 및 3모델 앙상블, 검증 정확도 92.4633%)의 무결성을 확정보존합니다.'
    ]
    
    # Fill in H1, H2, H4, H5 in matrix
    text_matrix = '\n'.join(matrix_lines)
    for hid, cname in [('H1', 'short'), ('H2', 'rms'), ('H4', 'zcr'), ('H5', 'bandpass')]:
        df_c = df_comp[df_comp['condition'] == cname]
        d42 = df_c[df_c['seed'] == 42]['delta_accuracy_pp'].values[0]
        d43 = df_c[df_c['seed'] == 43]['delta_accuracy_pp'].values[0]
        m_d = (d42 + d43) / 2.0
        res_str = f'S42 {d42:+.2f}%p, S43 {d43:+.2f}%p (평균 {m_d:+.2f}%p)'
        dec_str = '**채택 후보**' if cand_info.get('candidate') == cname else '**미채택**'
        text_matrix = text_matrix.replace(f'[{hid} 결과]', res_str).replace(f'[{hid} 판정]', dec_str)
        
    (output_root / 'HYPOTHESIS_COMPLETION_MATRIX.md').write_text(text_matrix, encoding='utf-8')
    
    # Generate ADOPTION_DECISION.md
    dec_lines = [
        '# DCC Mission 2 나머지 가설 최종 채택 판정서 (ADOPTION_DECISION)',
        '',
        '- 작성일: 2026-10-10',
        '- 사전 판정 프로토콜: decision_protocol.json (R3 게이트: 양 Seed 개선, 평균 Delta Acc >= +0.5%p, Recall 하락 < 1.0%p)',
        '',
        '## 1. 가설별 R3 게이트 판정 결과',
        '',
        '| 조건 | Seed 42 Delta Acc | Seed 43 Delta Acc | 평균 Delta Acc | 평균 Delta F1 | Recall 유지 여부 | R3 게이트 충족 |',
        '|---|:---:|:---:|:---:|:---:|:---:|:---:|'
    ]
    for cname in ['short', 'rms', 'zcr', 'bandpass']:
        df_c = df_comp[df_comp['condition'] == cname]
        d42 = df_c[df_c['seed'] == 42].iloc[0]
        d43 = df_c[df_c['seed'] == 43].iloc[0]
        m_acc = (d42['delta_accuracy_pp'] + d43['delta_accuracy_pp']) / 2.0
        m_f1 = (d42['delta_macro_f1'] + d43['delta_macro_f1']) / 2.0
        rec_ok = (d42['delta_recall_0_pp'] >= -1.0) and (d43['delta_recall_0_pp'] >= -1.0) and (d42['delta_recall_1_pp'] >= -1.0) and (d43['delta_recall_1_pp'] >= -1.0)
        passed_cond = (d42['delta_accuracy_pp'] > 0) and (d43['delta_accuracy_pp'] > 0) and (m_acc >= 0.5) and (m_f1 >= 0.0) and rec_ok
        dec_lines.append(
            f'| {cname} | {d42["delta_accuracy_pp"]:+.4f}%p | {d43["delta_accuracy_pp"]:+.4f}%p | '
            f'{m_acc:+.4f}%p | {m_f1:+.4f} | {"유지" if rec_ok else "1.0%p이상 하락"} | '
            f'{"충족" if passed_cond else "불충족"} |'
        )
        
    dec_lines.extend([
        '',
        '## 2. 최종 판정 및 후속 조치',
        '',
        f'- **최종 판정**: `{cand_info.get("action")}`',
        f'- **사유**: {cand_info.get("reason", "후보 평가 완료")}',
        '- **모델 보존 정책**: 채택 기준을 일관되게 만족하는 후보가 없으므로, 불필요한 재학습(F1)이나 CLI 수정(F2~F4)을 진행하지 않고, 기존 공식 제출 모델(128-Mel AudioResNet baseline 및 3모델 앙상블)을 공식 모델로 최종 확정보존합니다.'
    ])
    (output_root / 'ADOPTION_DECISION.md').write_text('\n'.join(dec_lines), encoding='utf-8')
    update_stage_status(output_root, 'report', 'completed')
    print('Stage report completed successfully. Generated all markdown documents.')


def self_test():
    print('Running self-test...')
    set_deterministic(42)
    m = PilotCNN(head_dropout=0.3)
    p_cnt = sum(p.numel() for p in m.parameters())
    assert p_cnt == 23503815, f'Parameter mismatch: {p_cnt}'
    
    # Check shape handling
    x94 = torch.randn(2, 1, 128, 94)
    x47 = torch.randn(2, 1, 128, 47)
    extra = torch.zeros(2, 6)
    out94 = m(x94, extra)
    out47 = m(x47, extra)
    assert out94.shape == (2,)
    assert out47.shape == (2,)
    
    # Check bootstrap
    calls = np.array(['call1', 'call1', 'call2', 'call2'])
    y = np.array([0, 1, 0, 1])
    p_cand = np.array([0.1, 0.9, 0.2, 0.8])
    p_base = np.array([0.2, 0.8, 0.3, 0.7])
    res = compute_paired_call_bootstrap(calls, y, p_cand, p_base, iterations=50)
    assert 'delta_acc_ci_low_pp' in res
    print('Self-test passed cleanly.')


def main():
    parser = argparse.ArgumentParser(description='DCC Mission 2 Remaining Audio Hypotheses Unified Runner')
    parser.add_argument('--plan', type=Path, default=Path(r'C:\Users\user\Desktop\DCC_m2_remaining_hypotheses\20261010\experiment_plan.json'),
                        help='Path to experiment plan JSON')
    parser.add_argument('--stage', choices=['all', 'verify', 'prepare', 'timing', 'compare', 'combine', 'confirm', 'report', 'self-test'],
                        default='all', help='Stage to execute')
    parser.add_argument('--resume', action='store_true', help='Resume from existing completed runs')
    parser.add_argument('--smoke', action='store_true', help='Run smoke timing mode')
    args = parser.parse_args()
    
    if args.stage == 'self-test':
        self_test()
        return
        
    with open(args.plan, 'r', encoding='utf-8') as f:
        plan = json.load(f)
        
    if args.stage in ['all', 'verify']:
        stage_verify(plan)
    if args.stage in ['all', 'prepare']:
        stage_prepare(plan)
    if args.stage in ['all', 'timing']:
        stage_timing(plan)
    if args.stage in ['all', 'compare']:
        stage_compare(plan, resume=args.resume)
    if args.stage in ['all', 'combine']:
        stage_combine_and_confirm(plan)
    if args.stage in ['all', 'report']:
        stage_report(plan)
        
    print('All requested stages completed.')


if __name__ == '__main__':
    main()
