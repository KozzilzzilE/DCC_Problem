"""조각 오디오 캐시가 신고자 구간을 정확히 잘라 저장하는지 검증."""
import json

import numpy as np
import pytest
import soundfile as sf

from m1.cache import CacheIndex, build_cache

SR = 8000


def make_call(tmp_path, call_id, utterances, gender, n_samples=8000 * 20):
    """utterances: list of (start_ms, end_ms, speaker)."""
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "label"
    audio_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)

    # 샘플 인덱스를 그대로 값으로 넣어 어느 구간이 잘렸는지 정확히 추적한다.
    wave = np.arange(n_samples, dtype=np.int16)
    sf.write(audio_dir / f"{call_id}.wav", wave, SR, subtype="PCM_16")

    payload = {
        "gender": gender,
        "symptom": ["복통"],
        "address": "서울특별시",
        "utterances": [
            {"startAt": s, "endAt": e, "speaker": sp, "text": "금지된 텍스트"}
            for s, e, sp in utterances
        ],
    }
    (label_dir / f"{call_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return audio_dir, label_dir


def test_cache_extracts_only_caller_segments(tmp_path):
    audio_dir, label_dir = make_call(
        tmp_path,
        "c1",
        [(0, 1000, 0), (1000, 2000, 1), (2000, 3000, 0), (3000, 5000, 1)],
        gender="F",
    )
    index = build_cache(label_dir, audio_dir, tmp_path / "cache")

    rows = index.rows
    assert len(rows) == 2
    assert [r.length for r in rows] == [SR * 1, SR * 2]
    assert all(r.gender == "F" for r in rows)
    assert all(r.call_id == "c1" for r in rows)


def test_cached_samples_match_source_slice(tmp_path):
    audio_dir, label_dir = make_call(tmp_path, "c1", [(1000, 2500, 1)], gender="M")
    index = build_cache(label_dir, audio_dir, tmp_path / "cache")

    seg = index.load_segment(index.rows[0])
    # wav 샘플값 == 샘플 인덱스이므로 정확한 구간 검증이 가능하다
    np.testing.assert_array_equal(seg, np.arange(8000, 20000, dtype=np.int16))


def test_segments_clamped_to_audio_length(tmp_path):
    audio_dir, label_dir = make_call(
        tmp_path, "c1", [(19000, 25000, 1)], gender="M", n_samples=SR * 20
    )
    index = build_cache(label_dir, audio_dir, tmp_path / "cache")

    assert len(index.rows) == 1
    assert index.rows[0].length == SR * 1  # 19s~20s 만 남는다


def test_segments_shorter_than_minimum_dropped(tmp_path):
    audio_dir, label_dir = make_call(
        tmp_path, "c1", [(0, 50, 1), (1000, 3000, 1)], gender="F"
    )
    index = build_cache(label_dir, audio_dir, tmp_path / "cache", min_segment_ms=100)
    assert len(index.rows) == 1
    assert index.rows[0].length == SR * 2


def test_call_with_no_caller_segments_is_skipped(tmp_path):
    audio_dir, label_dir = make_call(tmp_path, "c1", [(0, 2000, 0)], gender="F")
    index = build_cache(label_dir, audio_dir, tmp_path / "cache")
    assert index.rows == []


def test_missing_wav_is_skipped(tmp_path):
    audio_dir, label_dir = make_call(tmp_path, "c1", [(0, 2000, 1)], gender="F")
    (audio_dir / "c1.wav").unlink()
    index = build_cache(label_dir, audio_dir, tmp_path / "cache")
    assert index.rows == []


def test_build_is_idempotent(tmp_path):
    audio_dir, label_dir = make_call(tmp_path, "c1", [(0, 2000, 1), (3000, 4000, 1)], gender="M")
    cache_dir = tmp_path / "cache"

    first = build_cache(label_dir, audio_dir, cache_dir)
    second = build_cache(label_dir, audio_dir, cache_dir)

    assert [r.as_tuple() for r in first.rows] == [r.as_tuple() for r in second.rows]


def test_index_round_trips_through_csv(tmp_path):
    audio_dir, label_dir = make_call(tmp_path, "c1", [(0, 2000, 1), (3000, 4000, 1)], gender="M")
    cache_dir = tmp_path / "cache"
    built = build_cache(label_dir, audio_dir, cache_dir)

    reloaded = CacheIndex.load(cache_dir)
    assert [r.as_tuple() for r in reloaded.rows] == [r.as_tuple() for r in built.rows]
    np.testing.assert_array_equal(
        reloaded.load_segment(reloaded.rows[1]), built.load_segment(built.rows[1])
    )


def test_index_groups_rows_by_call(tmp_path):
    audio_dir, label_dir = make_call(tmp_path, "c1", [(0, 2000, 1), (3000, 4000, 1)], gender="M")
    make_call(tmp_path, "c2", [(0, 2000, 1)], gender="F")
    index = build_cache(label_dir, audio_dir, tmp_path / "cache")

    groups = index.by_call()
    assert set(groups) == {"c1", "c2"}
    assert len(groups["c1"]) == 2
    assert len(groups["c2"]) == 1


def test_no_label_text_written_into_cache(tmp_path):
    audio_dir, label_dir = make_call(tmp_path, "c1", [(0, 2000, 1)], gender="F")
    cache_dir = tmp_path / "cache"
    build_cache(label_dir, audio_dir, cache_dir)

    blob = (cache_dir / "index.csv").read_bytes()
    assert "금지된 텍스트".encode("utf-8") not in blob
    assert "복통".encode("utf-8") not in blob


def test_gender_absent_is_written_as_empty(tmp_path):
    audio_dir, label_dir = make_call(tmp_path, "c1", [(0, 2000, 1)], gender=None)
    index = build_cache(label_dir, audio_dir, tmp_path / "cache")
    assert index.rows[0].gender is None
