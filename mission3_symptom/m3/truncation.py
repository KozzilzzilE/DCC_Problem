"""Mission 3: max_length 토큰 절단으로 생기는 오분류를 찾고 복구하는 모듈.

담당: 권오현 (오분류 분석 + 잘린 토큰 구간 처리)

오분류 패턴
    KoBERT / BERT tokenizer 기본값 truncation=True 는 앞쪽 max_length 만 남긴다.
    119 통화는 초반 주호소 뒤에 대원 문진이 길어져, 정답 증상 문자열이
    첫 512토큰(실제 본문은 CLS/SEP 제외 510) 밖에만 남는 경우가 있다.
    그 샘플은 모델이 증상을 한 번도 못 보고 FN 이 된다.

처리
    1. analyze_truncated_symptoms 로 "첫 창 밖에만 있는 정답 증상"을 표시
    2. content_token_windows / head_tail_concat 으로 잘린 꼬리를 다시 넣음
    3. merge_chunk_probs 는 다중라벨이므로 청크별 확률을 클래스별 max 로 합침
       (한 청크에서만 나온 증상을 살림. mean 은 긴 통화에서 희석된다)

기본 head-tail 은 오분류 보고서 전략 1과 같이 앞 128 + 뒤 (예산-128) 이다.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .config import NUM_CLASSES, TARGET_SYMPTOMS

# KoBERT 기본 한도. [CLS] + 본문 + [SEP] 이므로 본문 예산은 max_length - 2.
DEFAULT_MAX_LENGTH = 512
DEFAULT_NUM_SPECIAL = 2
# error_analysis.md 전략 1: 앞 128 + 뒤 384 (512 한도 기준 본문 창)
DEFAULT_HEAD_TOKENS = 128


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def content_budget(max_length: int = DEFAULT_MAX_LENGTH, num_special: int = DEFAULT_NUM_SPECIAL) -> int:
    """특수 토큰을 뺀 실제 본문 토큰 한도."""
    if max_length <= num_special:
        raise ValueError(f"max_length({max_length}) 가 특수 토큰 수({num_special})보다 커야 한다.")
    return max_length - num_special


def find_symptom_mentions(text: str) -> Dict[str, List[Tuple[int, int]]]:
    """정규화한 텍스트에서 9개 타겟 증상의 문자 구간을 모두 찾는다."""
    text = normalize_text(text)
    mentions: Dict[str, List[Tuple[int, int]]] = {sym: [] for sym in TARGET_SYMPTOMS}
    for sym in TARGET_SYMPTOMS:
        start = 0
        while True:
            idx = text.find(sym, start)
            if idx < 0:
                break
            mentions[sym].append((idx, idx + len(sym)))
            start = idx + 1
    return mentions


def content_token_windows(
    n_tokens: int,
    max_content: int,
    stride: int | None = None,
) -> List[Tuple[int, int]]:
    """본문 토큰 [0, n_tokens) 를 덮는 고정 길이 창.

    짧으면 창 하나. 길면 Mission 1 오디오와 같이 마지막 창을 끝에 붙여
    꼬리가 다시 잘리지 않게 한다.
    """
    if n_tokens < 0 or max_content <= 0:
        raise ValueError("n_tokens 는 0 이상, max_content 는 1 이상이어야 한다.")
    if n_tokens == 0:
        return [(0, 0)]
    if n_tokens <= max_content:
        return [(0, n_tokens)]

    if stride is None:
        stride = max(1, max_content // 2)
    if stride <= 0:
        raise ValueError("stride 는 1 이상이어야 한다.")

    starts = list(range(0, n_tokens - max_content + 1, stride))
    windows = [(s, s + max_content) for s in starts]
    if windows[-1][1] != n_tokens:
        windows.append((n_tokens - max_content, n_tokens))
    return windows


def head_tail_concat(
    n_tokens: int,
    max_content: int,
    head_ratio: float | None = None,
    head_tokens: int | None = None,
) -> List[int]:
    """한 번의 forward 로 초반+후반을 넣기 위한 본문 인덱스.

    기본은 앞 128토큰 + 나머지 꼬리. head_ratio 를 주면 비율로 나눈다.
    """
    if n_tokens <= max_content:
        return list(range(n_tokens))

    if head_tokens is None:
        if head_ratio is None:
            head_tokens = DEFAULT_HEAD_TOKENS
        else:
            head_tokens = max(1, int(round(max_content * head_ratio)))

    head = max(1, min(int(head_tokens), max_content - 1))
    tail = max_content - head
    return list(range(head)) + list(range(n_tokens - tail, n_tokens))


def _span_covered(mention: Tuple[int, int], visible_char_ranges: Sequence[Tuple[int, int]]) -> bool:
    m0, m1 = mention
    for v0, v1 in visible_char_ranges:
        overlap = min(m1, v1) - max(m0, v0)
        if overlap > 0:
            return True
    return False


def visible_char_ranges(
    token_char_spans: Sequence[Tuple[int, int]],
    start_token: int,
    end_token: int,
) -> List[Tuple[int, int]]:
    """창 [start_token, end_token) 이 덮는 (빈 토큰 제외) 문자 구간."""
    ranges: List[Tuple[int, int]] = []
    for s, e in token_char_spans[start_token:end_token]:
        if e > s:
            ranges.append((s, e))
    return ranges


def analyze_truncated_symptoms(
    text: str,
    gold_symptoms: Sequence[str],
    token_char_spans: Sequence[Tuple[int, int]],
    max_length: int = DEFAULT_MAX_LENGTH,
    num_special: int = DEFAULT_NUM_SPECIAL,
) -> Dict[str, object]:
    """첫 창 truncation 때문에 모델이 못 보는 정답 증상을 집계한다.

    token_char_spans: 본문 토큰 i 가 덮는 원문 문자 구간 (start, end).
    특수 토큰은 넣지 않는다. 빈 구간 (0, 0) 은 무시한다.

    Returns:
        n_content_tokens, first_window, cut_symptoms, kept_symptoms, is_truncated
    """
    text = normalize_text(text)
    gold = [normalize_text(s) for s in gold_symptoms if normalize_text(s) in TARGET_SYMPTOMS]
    budget = content_budget(max_length, num_special)
    n_tokens = len(token_char_spans)
    first_end = min(n_tokens, budget)
    visible = visible_char_ranges(token_char_spans, 0, first_end)
    mentions = find_symptom_mentions(text)

    cut: List[str] = []
    kept: List[str] = []
    for sym in gold:
        hits = mentions.get(sym, [])
        if not hits:
            # 정답 라벨은 있으나 전사가 다른 표현인 경우는 절단 문제가 아님
            continue
        if any(_span_covered(m, visible) for m in hits):
            kept.append(sym)
        else:
            cut.append(sym)

    return {
        "n_content_tokens": n_tokens,
        "first_window": (0, first_end),
        "is_truncated": n_tokens > budget,
        "cut_symptoms": cut,
        "kept_symptoms": kept,
    }


def split_ids_for_windows(
    content_ids: Sequence[int],
    max_length: int = DEFAULT_MAX_LENGTH,
    num_special: int = DEFAULT_NUM_SPECIAL,
    stride: int | None = None,
) -> List[List[int]]:
    """본문 id 를 슬라이딩 창으로 나눈다. 학습/추론 공통 입력 준비용."""
    budget = content_budget(max_length, num_special)
    windows = content_token_windows(len(content_ids), budget, stride)
    return [list(content_ids[s:e]) for s, e in windows]


def merge_chunk_probs(
    chunk_probs: Iterable,
    method: str = "max",
) -> np.ndarray:
    """청크 확률 (C, 9) 를 샘플 하나 확률 (9,) 로 합친다."""
    arr = np.asarray(chunk_probs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != NUM_CLASSES:
        raise ValueError(f"chunk_probs 형상은 (C, {NUM_CLASSES}) 여야 한다: {arr.shape}")
    if method == "max":
        return np.max(arr, axis=0)
    if method == "mean":
        return np.mean(arr, axis=0)
    raise ValueError(f"지원하지 않는 병합: {method}")
