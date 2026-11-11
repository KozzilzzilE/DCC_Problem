#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission 4-2 적응형 중단 규칙 격자 탐색 (Grid Search) 스크립트

Colab에서 추출한 검증셋 확률 테이블(val_probabilities_v2.csv)을 기반으로,
다양한 중단 조건(최소 턴, 연속 일치 턴, 최대 대기 턴 등)을 시뮬레이션하여
Macro-F1 x 평균 조기판단력이 최대가 되는 최적 규칙을 도출합니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

TARGET_SYMPTOMS = [
    "고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"
]
SYMPTOM_SET = set(TARGET_SYMPTOMS)
DECISION_THRESHOLD = 0.5
EARLINESS_GRACE = 5
EARLINESS_FLOOR = 0.5


def earliness_weight(halt: int, n_utt: int) -> float:
    if n_utt <= 0:
        return EARLINESS_FLOOR
    r = 1.0 - max(halt, EARLINESS_GRACE) / n_utt
    return EARLINESS_FLOOR + (1.0 - EARLINESS_FLOOR) * max(0.0, r)


def macro_f1(pairs: List[Tuple[List[str], List[str]]]) -> float:
    vals: List[float] = []
    for c in TARGET_SYMPTOMS:
        tp = fp = fn = 0
        for g, p in pairs:
            gi, pi = c in set(g), c in set(p)
            if gi and pi:
                tp += 1
            elif pi and not gi:
                fp += 1
            elif gi and not pi:
                fn += 1
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        vals.append(f1)
    return sum(vals) / len(vals) if vals else 0.0


def load_golds_and_lengths(label_dir: str) -> Dict[str, Tuple[List[str], int]]:
    result = {}
    p = Path(label_dir)
    for fn in os.listdir(p):
        if not fn.endswith('.json'):
            continue
        with open(p / fn, 'r', encoding='utf-8') as f:
            d = json.load(f)
        raw = set(d.get('symptom') or [])
        valid = [s for s in TARGET_SYMPTOMS if s in raw]
        if not valid:
            continue
        utts = d.get('utterances') or []
        if not utts:
            continue
        result[fn] = (valid, len(utts))
    return result


def load_prob_table(prob_csv_path: str) -> Dict[str, List[Tuple[int, List[float]]]]:
    calls = defaultdict(list)
    with open(prob_csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['call_id']
            step = int(row['step'])
            probs = [float(row[s]) for s in TARGET_SYMPTOMS]
            calls[cid].append((step, probs))
    for cid in calls:
        calls[cid].sort(key=lambda x: x[0])
    return calls


def simulate(
    calls: Dict[str, List[Tuple[int, List[float]]]],
    golds: Dict[str, Tuple[List[str], int]],
    min_halt: int,
    consecutive_k: int,
    fallback_step: int = 999,
) -> Tuple[float, float, float]:
    pairs = []
    earliness = []

    for cid, (gold_symptoms, total_utts) in golds.items():
        steps = calls.get(cid, [])
        halted = False
        halt_step = total_utts
        pred_at_halt = []

        consecutive_count = 0
        last_pred = []

        for step, probs in steps:
            curr_pred = [s for s, p in zip(TARGET_SYMPTOMS, probs) if p >= DECISION_THRESHOLD]

            if curr_pred and curr_pred == last_pred:
                consecutive_count += 1
            elif curr_pred:
                last_pred = list(curr_pred)
                consecutive_count = 1
            else:
                last_pred = []
                consecutive_count = 0

            # 1. Fallback 도달 시 강제 중단
            if step >= fallback_step:
                halt_step = step
                pred_at_halt = list(last_pred)
                halted = True
                break

            # 2. 최소 턴 이상 + 연속 k턴 일치 시 중단
            if step >= min_halt and curr_pred and consecutive_count >= consecutive_k:
                halt_step = step
                pred_at_halt = list(curr_pred)
                halted = True
                break

        if not halted:
            halt_step = total_utts
            pred_at_halt = list(last_pred)

        e = earliness_weight(halt_step, total_utts)
        pairs.append((gold_symptoms, pred_at_halt))
        earliness.append(e)

    mf1 = macro_f1(pairs)
    mean_e = sum(earliness) / len(earliness) if earliness else 0.0
    return mf1 * mean_e, mf1, mean_e


def main():
    parser = argparse.ArgumentParser(description="4-2 중단 규칙 격자 탐색")
    parser.add_argument("--prob_csv", default="val_probabilities_v2.csv", help="확률 CSV 경로")
    parser.add_argument("--label_dir", default="../data/Training/2.라벨링데이터/TL_서울_구급", help="라벨 폴더")
    args = parser.parse_args()

    if not os.path.exists(args.prob_csv):
        print(f"[Error] 확률 파일이 없습니다: {args.prob_csv}")
        return 1

    print(f"라벨 로딩 중: {args.label_dir}")
    golds = load_golds_and_lengths(args.label_dir)
    print(f"유효 통화: {len(golds)}건")

    print(f"확률표 로딩 중: {args.prob_csv}")
    calls = load_prob_table(args.prob_csv)
    print(f"확률표 통화: {len(calls)}건")

    eval_calls = {k: golds[k] for k in calls if k in golds}
    print(f"공통 평가 대상: {len(eval_calls)}건\n")

    print(f"{'min_halt':<10}{'consecutive_k':<15}{'fallback':<12}{'Score':<10}{'Macro-F1':<12}{'Earliness':<10}")
    print("-" * 70)

    best_score = -1.0
    best_params = None

    for min_halt in [5, 6, 7]:
        for k in [1, 2, 3]:
            for fb in [999, 12, 10, 8]:
                score, mf1, me = simulate(calls, eval_calls, min_halt, k, fb)
                if score > best_score:
                    best_score = score
                    best_params = (min_halt, k, fb, mf1, me)
                fb_str = str(fb) if fb != 999 else "None"
                print(f"{min_halt:<10}{k:<15}{fb_str:<12}{score:.4f}    {mf1:.4f}      {me:.4f}")

    print("\n" + "=" * 70)
    print(f"⭐️ [최적 규칙 조합 발견]")
    print(f"  - 최소 대기 턴 (min_halt) : {best_params[0]}턴")
    print(f"  - 연속 일치 턴 (consec_k) : {best_params[1]}턴")
    print(f"  - 강제 중단 턴 (fallback) : {best_params[2]}")
    print(f"  - 최고 4-2 Score          : {best_score:.4f} (F1: {best_params[3]:.4f}, 조기판단력: {best_params[4]:.4f})")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
