"""Mission 3 토큰 절단 오분류 탐지와 창 복구 로직."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MISSION3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MISSION3_DIR))

from m3.truncation import (
    analyze_truncated_symptoms,
    content_budget,
    content_token_windows,
    find_symptom_mentions,
    head_tail_concat,
    merge_chunk_probs,
    split_ids_for_windows,
)


def _char_spans(text: str):
    """테스트용: 한글 한 글자 = 토큰 하나."""
    return [(i, i + 1) for i in range(len(text))]


def test_mentions_find_all_occurrences():
    text = "배가 아프고 복통이 있고 또 복통"
    hits = find_symptom_mentions(text)["복통"]
    assert len(hits) == 2
    assert text[hits[0][0] : hits[0][1]] == "복통"


def test_short_text_is_not_truncated():
    text = "두통이 심해요"
    report = analyze_truncated_symptoms(
        text, ["두통"], _char_spans(text), max_length=32, num_special=2
    )
    assert report["is_truncated"] is False
    assert report["cut_symptoms"] == []
    assert report["kept_symptoms"] == ["두통"]


def test_tail_only_symptom_is_marked_cut():
    prefix = "대원입니다. 주소 확인했습니다. " * 20
    text = prefix + "호흡곤란 있어요"
    budget = content_budget(max_length=40, num_special=2)
    assert len(text) > budget

    report = analyze_truncated_symptoms(
        text, ["호흡곤란"], _char_spans(text), max_length=40, num_special=2
    )
    assert report["is_truncated"] is True
    assert report["cut_symptoms"] == ["호흡곤란"]
    assert report["kept_symptoms"] == []


def test_symptom_in_head_is_kept_even_if_call_is_long():
    text = "고열 있어요. " + ("문진 이어집니다. " * 30)
    report = analyze_truncated_symptoms(
        text, ["고열"], _char_spans(text), max_length=40, num_special=2
    )
    assert report["is_truncated"] is True
    assert report["kept_symptoms"] == ["고열"]
    assert report["cut_symptoms"] == []


def test_sliding_windows_cover_the_tail():
    n = 100
    wins = content_token_windows(n, max_content=20, stride=10)
    assert wins[0] == (0, 20)
    assert wins[-1] == (80, 100)
    assert all(e - s == 20 for s, e in wins)


def test_short_sequence_is_single_window():
    assert content_token_windows(10, max_content=50) == [(0, 10)]
    assert content_token_windows(0, max_content=50) == [(0, 0)]


def test_split_ids_last_chunk_contains_tail_symptom_ids():
    text = ("가" * 80) + "열상"
    ids = list(range(len(text)))
    chunks = split_ids_for_windows(ids, max_length=22, num_special=2)
    last = chunks[-1]
    tail_ids = ids[-len("열상") :]
    assert all(t in last for t in tail_ids)


def test_head_tail_keeps_both_ends():
    idx = head_tail_concat(100, max_content=20, head_ratio=0.5)
    assert idx[:10] == list(range(10))
    assert idx[-10:] == list(range(90, 100))
    assert len(idx) == 20


def test_head_tail_default_uses_128_head():
    idx = head_tail_concat(1000, max_content=510)
    assert idx[:128] == list(range(128))
    assert idx[-382:] == list(range(1000 - 382, 1000))
    assert len(idx) == 510


def test_merge_chunk_probs_max_recovers_tail_positive():
    chunks = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.92],
        ]
    )
    merged = merge_chunk_probs(chunks, method="max")
    assert merged[-1] == pytest.approx(0.92)
    meaned = merge_chunk_probs(chunks, method="mean")
    assert meaned[-1] < 0.5
