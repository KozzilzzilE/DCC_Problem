"""라벨 리더가 대회 규칙(startAt/endAt/speaker만 사용)을 구조적으로 지키는지 검증."""
import dataclasses
import json

import pytest

from m1.labels import CALLER, CallRecord, Utterance, caller_utterances, iter_calls, read_call

FULL_JSON = {
    "_id": "abc123",
    "audioPath": "20231112/Seoul/2022/20220119/converted_[x].wav",
    "recordId": "r1",
    "status": 12,
    "startAt": 0,
    "endAt": 73920,
    "mediaType": "Mobile",
    "gender": "M",
    "address": "서울특별시 송파구 가락동",
    "disasterLarge": "구급",
    "disasterMedium": "사고",
    "urgencyLevel": "중",
    "sentiment": "기타부정",
    "symptom": ["복통", "열상"],
    "triage": "응급증상",
    "utterances": [
        {"id": "u0", "startAt": 1020, "endAt": 2203, "text": "119상황실입니다.", "speaker": 0},
        {"id": "u1", "startAt": 2311, "endAt": 4985, "text": "여기 사람이 쓰러졌어요", "speaker": 1},
        {"id": "u2", "startAt": 5000, "endAt": 6000, "text": "어디신가요", "speaker": 0},
        {"id": "u3", "startAt": 6100, "endAt": 9000, "text": "가락동이요", "speaker": 1},
    ],
}


def write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_read_call_exposes_only_whitelisted_fields(tmp_path):
    rec = read_call(write_json(tmp_path, "abc123_20220119.json", FULL_JSON))

    assert {f.name for f in dataclasses.fields(rec)} == {"call_id", "utterances", "gender"}
    assert {f.name for f in dataclasses.fields(rec.utterances[0])} == {
        "start_ms",
        "end_ms",
        "speaker",
    }


def test_no_forbidden_content_reachable_from_record(tmp_path):
    """text/symptom/address 등 금지 필드가 레코드 어디에도 남아 있지 않아야 한다."""
    rec = read_call(write_json(tmp_path, "abc123_20220119.json", FULL_JSON))

    blob = repr(rec)
    for forbidden in ("119상황실입니다", "쓰러졌", "복통", "열상", "송파구", "응급증상", "기타부정"):
        assert forbidden not in blob


def test_records_are_frozen(tmp_path):
    rec = read_call(write_json(tmp_path, "abc123_20220119.json", FULL_JSON))

    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.gender = "F"
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.utterances[0].speaker = 1


def test_call_id_comes_from_filename_stem(tmp_path):
    rec = read_call(write_json(tmp_path, "abc123_20220119.json", FULL_JSON))
    assert rec.call_id == "abc123_20220119"


def test_gender_and_utterances_parsed(tmp_path):
    rec = read_call(write_json(tmp_path, "abc123_20220119.json", FULL_JSON))

    assert rec.gender == "M"
    assert len(rec.utterances) == 4
    assert rec.utterances[1] == Utterance(start_ms=2311, end_ms=4985, speaker=1)


def test_caller_utterances_filters_to_speaker_1(tmp_path):
    rec = read_call(write_json(tmp_path, "abc123_20220119.json", FULL_JSON))

    callers = caller_utterances(rec)
    assert len(callers) == 2
    assert all(u.speaker == CALLER for u in callers)
    assert [u.start_ms for u in callers] == [2311, 6100]


def test_non_positive_duration_utterances_dropped(tmp_path):
    payload = dict(FULL_JSON)
    payload["utterances"] = [
        {"startAt": 100, "endAt": 100, "speaker": 1},
        {"startAt": 500, "endAt": 200, "speaker": 1},
        {"startAt": 800, "endAt": 1800, "speaker": 1},
    ]
    rec = read_call(write_json(tmp_path, "c_1.json", payload))

    assert len(rec.utterances) == 1
    assert rec.utterances[0].start_ms == 800


def test_missing_gender_becomes_none(tmp_path):
    """평가용 입력에는 정답 gender가 없을 수 있다."""
    payload = {k: v for k, v in FULL_JSON.items() if k != "gender"}
    rec = read_call(write_json(tmp_path, "c_1.json", payload))
    assert rec.gender is None


def test_unexpected_gender_value_becomes_none(tmp_path):
    payload = dict(FULL_JSON, gender="unknown")
    assert read_call(write_json(tmp_path, "c_1.json", payload)).gender is None


def test_iter_calls_reads_directory_sorted(tmp_path):
    write_json(tmp_path, "b_2.json", FULL_JSON)
    write_json(tmp_path, "a_1.json", FULL_JSON)
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    assert [r.call_id for r in iter_calls(tmp_path)] == ["a_1", "b_2"]


def test_missing_utterances_key_yields_empty(tmp_path):
    payload = {k: v for k, v in FULL_JSON.items() if k != "utterances"}
    rec = read_call(write_json(tmp_path, "c_1.json", payload))
    assert rec.utterances == ()
    assert caller_utterances(rec) == ()


def test_record_is_hashable_for_use_as_dict_key(tmp_path):
    rec = read_call(write_json(tmp_path, "c_1.json", FULL_JSON))
    assert isinstance(hash(rec), int)
    assert isinstance(rec, CallRecord)
