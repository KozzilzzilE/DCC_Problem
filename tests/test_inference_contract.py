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


class _FirstSampleModel(torch.nn.Module):
    """창의 첫 샘플만 보고 logit 을 내는 결정적 스텁.

    패딩(뒤쪽 0)에 영향받지 않으므로, 조각 길이가 달라도 창마다 의도한 확률을
    정확히 재현할 수 있다.
    """

    def forward(self, waveform):
        return waveform[:, 0] * 10.0


def test_windows_are_averaged_per_segment_before_per_call(tmp_path, monkeypatch):
    """긴 조각이 창 개수만큼 가중되면 안 된다.

    창 -> 통화로 한 번에 평균하면 창이 6개 나오는 긴 조각이 창 1개짜리 짧은
    조각을 압도한다. m1.evaluate 는 창 -> 조각 -> 통화 순으로 두 번 평균하므로,
    제출 경로도 같아야 보고한 수치와 실제 결과가 일치한다.
    """
    from m1 import infer

    cfg = FeatureConfig()
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "label"
    audio_dir.mkdir()
    label_dir.mkdir()

    # 0~10초: 값 +0.46 (여성 쪽 logit +4.6), 11~12초: 값 -0.69 (남성 쪽 logit -6.9)
    wave = np.zeros(SR * 13, dtype=np.int16)
    wave[: SR * 10] = int(0.46 * 32768)
    wave[SR * 11 : SR * 12] = int(-0.69 * 32768)
    sf.write(audio_dir / "c.wav", wave, SR, subtype="PCM_16")
    (label_dir / "c.json").write_text(
        json.dumps({"utterances": [
            {"startAt": 0, "endAt": 10000, "speaker": 1},
            {"startAt": 11000, "endAt": 12000, "speaker": 1},
        ]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        infer, "load_checkpoint",
        lambda path, device="cpu": (_FirstSampleModel(), "resnet", cfg, {}),
    )

    df = infer.predict_directory(audio_dir, label_dir, "unused.pt", device="cpu", verbose=False)

    # 조각별 평균: (0.99 + 0.001) / 2 = 0.4955 -> 남
    # 창을 한 번에 평균했다면: (6*0.99 + 0.001) / 7 = 0.849 -> 여
    assert df.iloc[0]["gender"] == "남"


def test_call_windows_tags_segments_not_windows(tmp_path):
    """긴 조각에서 나온 창들은 같은 조각 번호를 공유해야 한다."""
    from m1.infer import _call_windows
    from m1.labels import read_call

    cfg = FeatureConfig()
    label_dir = tmp_path / "label"
    label_dir.mkdir()
    (label_dir / "c.json").write_text(
        json.dumps({"utterances": [
            {"startAt": 0, "endAt": 10000, "speaker": 1},
            {"startAt": 11000, "endAt": 12000, "speaker": 1},
        ]}),
        encoding="utf-8",
    )

    audio = np.zeros(SR * 13, dtype=np.int16)
    tags = [seg_idx for seg_idx, _ in _call_windows(read_call(label_dir / "c.json"), audio, cfg, "resnet")]

    assert set(tags) == {0, 1}
    assert tags.count(0) > 1, "10초 조각은 창이 여러 개 나와야 한다"
    assert tags.count(1) == 1, "1초 조각은 패딩되어 창 하나"
