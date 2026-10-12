"""Portable ZCR runner. Default report stage reads existing CSV results only."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

QUICK_SCRIPT = Path(__file__).with_name('quick_hypotheses.py')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def run_probe(cache_dir, output_dir, seed, device):
    command = [sys.executable, str(QUICK_SCRIPT), 'probe', '--cache-dir', str(cache_dir),
               '--output-dir', str(output_dir), '--model', 'resnet50', '--conditions',
               'baseline', 'zcr', '--epochs', '5', '--batch-size', '16', '--lr', '0.0001',
               '--seed', str(seed), '--device', device, '--head-dropout', '0.3', '--bootstrap', '2000']
    log_path = output_dir.parent / f'probe_s{seed}.log'
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            print(line, end='')
            log.write(line)
        if process.wait():
            raise RuntimeError(f'Probe failed. See {log_path}')


def report(output_dir, cache_dir=None, report_dir=None):
    report_dir = report_dir or output_dir / 'report_recheck'
    report_dir.mkdir(parents=True, exist_ok=True)
    comparisons, differences = [], []
    seed_metrics = []
    for seed in (42, 43):
        folder = output_dir / f'probe_s{seed}'
        comp = read_csv(folder / f'pilot_comparison_seed{seed}.csv')
        diff = read_csv(folder / f'pilot_differences_seed{seed}.csv')
        if any(row.get('is_smoke', '').lower() == 'true' for row in comp):
            raise ValueError('Smoke results cannot determine final adoption')
        row = next(row for row in diff if row['condition'] == 'zcr')
        seed_metrics.append({
            'seed': seed, 'delta_accuracy_pp': float(row['delta_accuracy_last_pp']),
            'delta_macro_f1': float(row['delta_macro_f1_last']),
            'delta_recall_0_pp': float(row['delta_recall_0_last_pp']),
            'delta_recall_1_pp': float(row['delta_recall_1_last_pp']),
            'ci_95_pp': [float(row['paired_ci_last_low_pp']), float(row['paired_ci_last_high_pp'])],
        })
        comparisons.extend(comp)
        differences.extend(diff)
    for filename, rows in [('ZCR_PILOT_COMPARISON_ALL.csv', comparisons),
                           ('ZCR_PILOT_DIFFERENCES_ALL.csv', differences)]:
        with (report_dir / filename).open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    mean = sum(row['delta_accuracy_pp'] for row in seed_metrics) / 2
    criteria = {
        'both_seeds_positive': all(row['delta_accuracy_pp'] > 0 for row in seed_metrics),
        'mean_delta_at_least_0_5_pp': mean >= 0.5,
        'mean_macro_f1_nonnegative': sum(row['delta_macro_f1'] for row in seed_metrics) / 2 >= 0,
        'both_classes_recall_drop_less_than_1_pp': all(row[key] > -1.0 for row in seed_metrics
            for key in ('delta_recall_0_pp', 'delta_recall_1_pp')),
        'both_ci_lower_bounds_positive': all(row['ci_95_pp'][0] > 0 for row in seed_metrics),
    }
    split = None
    if cache_dir is not None:
        import numpy as np
        with np.load(cache_dir / 'features.npz', allow_pickle=False) as features:
            dev = features['dev'].astype(bool)
            calls = features['calls']
            split = {'train_calls': len(np.unique(calls[~dev])), 'dev_calls': len(np.unique(calls[dev])),
                     'train_utterances': int((~dev).sum()), 'dev_utterances': int(dev.sum())}
    passed = all(criteria.values())
    decision = {'candidate': 'H4_ZCR', 'primary_epoch': 5, 'passed': passed,
                'mean_delta_accuracy_pp': mean, 'criteria': criteria, 'seeds': seed_metrics,
                'actual_split': split,
                'action': 'PROCEED_FULL_TRAINING' if passed else 'RETAIN_OFFICIAL_BASELINE'}
    (report_dir / 'decision_summary.json').write_text(json.dumps(decision, indent=2) + '\n', encoding='utf-8')
    print(f'Report output: {report_dir}')
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['report', 'train'], default='report')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--report-dir', type=Path, help='Separate report output; default output-dir/report_recheck')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    args = parser.parse_args()
    if args.stage == 'train':
        if args.cache_dir is None:
            parser.error('--cache-dir is required for training')
        for file in ('baseline.npy', 'features.npz'):
            if not (args.cache_dir / file).is_file():
                parser.error(f'Missing prepared cache: {file}')
        # Protect earlier runs. Reports may be regenerated, weights may not be overwritten.
        for seed in (42, 43):
            if (args.output_dir / f'probe_s{seed}').exists():
                raise FileExistsError('Use a new output directory for a new training execution')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for seed in (42, 43):
            run_probe(args.cache_dir, args.output_dir / f'probe_s{seed}', seed, args.device)
    report(args.output_dir, args.cache_dir, args.report_dir)


if __name__ == '__main__':
    main()
