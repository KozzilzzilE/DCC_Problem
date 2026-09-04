"""제출 규격 검증 — inference.py 1회 실행으로 올바른 CSV 가 나오는가.

대회 실행 규칙:
    python inference.py --audio_dir {wav} --label_dir {json} --ckpt_path {ckpt}
                        --output ./outputs/mission1.csv
    Mission 1 csv: [audio file name], [gender]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from m1.config import FeatureConfig
from m1.infer import OUTPUT_COLUMNS, predict_directory
from m1.models import build_model, save_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
SR = 8000


@pytest.fixture
def dataset(tmp_path):
    """신고자/대원 발화가 섞인 통화 3건."""
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "label"
    audio_dir.mkdir()
    label_dir.mkdir()

    rng = np.random.RandomState(0)
    for i in range(3):
        wave = (rng.randn(SR * 20) * 3000).astype(np.int16)
        sf.write(audio_dir / f"call{i}.wav", wave, SR, subtype="PCM_16")
        payload = {
            "gender": "F" if i % 2 else "M",
            "symptom": ["복통"],
            "utterances": [
                {"startAt": 0, "endAt": 2000, "speaker": 0, "text": "119상황실입니다"},
                {"startAt": 2000, "endAt": 6000, "speaker": 1, "text": "도와주세요"},
                {"startAt": 7000, "endAt": 9000, "speaker": 1, "text": "가락동이요"},
            ],
        }
        (label_dir / f"call{i}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    return audio_dir, label_dir


@pytest.fixture
def ckpt(tmp_path):
    cfg = FeatureConfig()
    model = build_model("resnet", cfg, pretrained=False)
    return save_checkpoint(tmp_path / "m1.pt", model, "resnet", cfg, metrics={"dev_call_accuracy": 0.5})


def test_predict_directory_schema(dataset, ckpt):
    audio_dir, label_dir = dataset
    df = predict_directory(audio_dir, label_dir, ckpt, device="cpu", verbose=False)

    assert list(df.columns) == OUTPUT_COLUMNS == ["audio file name", "gender"]
    assert len(df) == 3
    assert sorted(df["audio file name"]) == ["call0.wav", "call1.wav", "call2.wav"]
    assert set(df["gender"]) <= {"남", "여"}


def test_every_call_gets_exactly_one_row(dataset, ckpt):
    audio_dir, label_dir = dataset
    df = predict_directory(audio_dir, label_dir, ckpt, device="cpu", verbose=False)
    assert df["audio file name"].is_unique


def test_call_without_caller_segments_still_emits_row(tmp_path, ckpt):
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "label"
    audio_dir.mkdir()
    label_dir.mkdir()
    sf.write(audio_dir / "c.wav", np.zeros(SR * 5, dtype=np.int16), SR, subtype="PCM_16")
    (label_dir / "c.json").write_text(
        json.dumps({"utterances": [{"startAt": 0, "endAt": 2000, "speaker": 0}]}),
        encoding="utf-8",
    )

    df = predict_directory(audio_dir, label_dir, ckpt, device="cpu", verbose=False)
    assert len(df) == 1
    assert df.iloc[0]["gender"] in {"남", "여"}


def test_wav_without_label_json_still_emits_row(dataset, ckpt):
    audio_dir, label_dir = dataset
    sf.write(audio_dir / "orphan.wav", np.zeros(SR * 3, dtype=np.int16), SR, subtype="PCM_16")

    df = predict_directory(audio_dir, label_dir, ckpt, device="cpu", verbose=False)
    assert len(df) == 4
    assert "orphan.wav" in set(df["audio file name"])


def test_resamples_non_8k_input(tmp_path, ckpt):
    """평가 데이터가 16 kHz 로 와도 동작해야 한다."""
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "label"
    audio_dir.mkdir()
    label_dir.mkdir()
    rng = np.random.RandomState(1)
    sf.write(audio_dir / "c.wav", (rng.randn(16000 * 10) * 2000).astype(np.int16), 16000, subtype="PCM_16")
    (label_dir / "c.json").write_text(
        json.dumps({"gender": "F", "utterances": [{"startAt": 1000, "endAt": 5000, "speaker": 1}]}),
        encoding="utf-8",
    )

    df = predict_directory(audio_dir, label_dir, ckpt, device="cpu", verbose=False)
    assert len(df) == 1
    assert df.iloc[0]["gender"] in {"남", "여"}


def test_checkpoint_round_trip_preserves_feature_config(tmp_path):
    from m1.models import load_checkpoint

    cfg = FeatureConfig(kind="mfcc", n_mfcc=20, n_mels=48, window_frames=128)
    model = build_model("resnet", cfg, pretrained=False)
    path = save_checkpoint(tmp_path / "m.pt", model, "resnet", cfg)

    _, branch, loaded_cfg, _ = load_checkpoint(path, device="cpu")
    assert branch == "resnet"
    assert loaded_cfg == cfg


def test_rejects_foreign_checkpoint(tmp_path):
    from m1.models import load_checkpoint

    path = tmp_path / "foreign.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="m1 체크포인트가 아닙니다"):
        load_checkpoint(path, device="cpu")


def test_cli_end_to_end(dataset, ckpt, tmp_path):
    """대회가 실제로 실행하는 명령 그대로 한 번 돌려본다."""
    audio_dir, label_dir = dataset
    output = tmp_path / "outputs" / "mission1.csv"

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "inference.py"),
            "--audio_dir", str(audio_dir),
            "--label_dir", str(label_dir),
            "--ckpt_path", str(ckpt),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()

    df = pd.read_csv(output)
    assert list(df.columns) == ["audio file name", "gender"]
    assert len(df) == 3
    assert set(df["gender"]) <= {"남", "여"}
