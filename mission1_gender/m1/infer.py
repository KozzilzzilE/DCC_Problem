"""제출용 추론 — 캐시 없이 원본 wav/json 폴더에서 바로 예측한다.

평가 시 주최 측은 임의의 audio/label 폴더를 주므로 캐시를 전제할 수 없다.
통화당 wav 를 한 번만 읽고 신고자 조각을 잘라 창 단위로 배치 추론한 뒤,
soft voting 으로 통화 라벨을 정한다.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

from .aggregate import call_label, call_probability
from .config import FeatureConfig
from .datasets import RESAMPLE_BRANCHES, crop_or_pad, to_waveform
from .features import sliding_windows
from .labels import CallRecord, caller_utterances, iter_calls
from .models import checkpoint_threshold, decision_threshold, load_checkpoint

OUTPUT_COLUMNS = ["audio file name", "gender"]
MIN_SEGMENT_MS = 100
STRIDE_RATIO = 0.5
EMPTY_CACHE_EVERY = 200   # 이 배치 수마다 torch.cuda.empty_cache() (m1.evaluate 와 동일)


def suggested_batch_size(branch: str) -> int:
    """갈래별 기본 추론 배치. 8 GB GPU 에서 VRAM 을 넘기지 않는 값.

    w2v2 는 16 kHz 창(48,896 샘플)이라 배치 128 이면 예약 메모리가 9.2 GB 로 8 GB 를
    넘어 Windows WDDM 이 시스템 RAM 으로 페이징한다 — OOM 없이 조용히 10배 느려진다
    (통화당 616 ms). 배치 32 는 3.2 GB, 통화당 58 ms. ResNet 은 8 kHz 멜이라 128 도 안전.
    """
    return 32 if branch in RESAMPLE_BRANCHES else 128


def _load_call_audio(wav_path: Path, target_sr: int) -> np.ndarray | None:
    """통화 오디오를 int16 mono, target_sr 로 읽는다."""
    try:
        audio, sr = sf.read(wav_path, dtype="int16", always_2d=False)
    except Exception:
        return None

    if audio.ndim > 1:
        audio = audio[:, 0]

    if sr != target_sr:
        # 제공 데이터는 8 kHz 지만, 평가 데이터가 다른 레이트여도 동작하게 한다.
        divisor = gcd(int(target_sr), int(sr))
        audio = resample_poly(audio.astype(np.float32), target_sr // divisor, sr // divisor)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)

    return audio


def _call_windows(
    record: CallRecord, audio: np.ndarray, cfg: FeatureConfig, branch: str
) -> Iterator[tuple[int, np.ndarray]]:
    """한 통화의 신고자 조각들을 (조각 번호, 고정 길이 창) 으로 펼친다.

    조각 번호를 함께 내보내는 이유: 창 확률을 먼저 조각 단위로 평균한 뒤 통화
    단위로 평균해야 한다. 창을 통화 전체에서 한 번에 평균하면 창이 많이 나오는
    긴 조각에 가중치가 쏠려, m1.evaluate 가 보고하는 수치와 제출 결과가 달라진다.
    """
    window = cfg.window_samples
    stride = max(1, int(window * STRIDE_RATIO))
    min_samples = max(1, int(MIN_SEGMENT_MS * cfg.sample_rate / 1000))

    seg_idx = 0
    for utt in caller_utterances(record):
        start = max(0, int(utt.start_ms * cfg.sample_rate / 1000))
        end = min(len(audio), int(utt.end_ms * cfg.sample_rate / 1000))
        segment = audio[start:end]
        if len(segment) < min_samples:
            continue

        if len(segment) < window:
            yield seg_idx, to_waveform(crop_or_pad(segment, window, None), branch)
        else:
            for win_start, _ in sliding_windows(len(segment), window, stride):
                yield seg_idx, to_waveform(segment[win_start : win_start + window], branch)
        seg_idx += 1


@torch.no_grad()
def predict_directory(
    audio_dir: str | Path,
    label_dir: str | Path,
    ckpt_path: str | Path,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """label_dir 의 통화마다 한 행씩, [audio file name, gender] DataFrame 을 만든다.

    batch_size 가 None 이면 갈래별 기본값(suggested_batch_size)을 쓴다.
    """
    audio_dir = Path(audio_dir)
    label_dir = Path(label_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    model, branch, cfg, payload = load_checkpoint(ckpt_path, device=device)
    threshold = decision_threshold(payload)          # 대회 규정: 0.5 고정
    if batch_size is None:
        batch_size = suggested_batch_size(branch)
    if verbose:
        trained = payload.get("metrics", {}).get("dev_call_accuracy")
        stored = (payload.get("extra") or {}).get("decision_threshold")
        print(f"[Mission 1] branch={branch} device={device} feature={cfg.kind}"
              f" threshold={threshold:.3f} (규정 고정) batch_size={batch_size}"
              + (f" dev_call_acc={trained:.4f}" if trained else "")
              + (f" | ckpt 저장값 {checkpoint_threshold(payload):.3f} 은 무시" if stored is not None else ""))

    records = list(iter_calls(label_dir))

    batch: list[np.ndarray] = []
    batch_keys: list[tuple[str, int]] = []
    # (call_id, seg_idx) -> 창 확률의 합/개수. 조각 단위로 먼저 평균한다.
    sums: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], int] = {}
    flushes = 0

    def flush() -> None:
        nonlocal flushes
        if not batch:
            return
        flushes += 1
        if device.type == "cuda" and flushes % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()
        waveform = torch.from_numpy(np.stack(batch)).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(waveform)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        for key, prob in zip(batch_keys, probs):
            sums[key] = sums.get(key, 0.0) + float(prob)
            counts[key] = counts.get(key, 0) + 1
        batch.clear()
        batch_keys.clear()

    for n, record in enumerate(records, start=1):
        audio = _load_call_audio(audio_dir / f"{record.call_id}.wav", cfg.sample_rate)
        if audio is not None:
            for seg_idx, chunk in _call_windows(record, audio, cfg, branch):
                batch.append(chunk)
                batch_keys.append((record.call_id, seg_idx))
                if len(batch) >= batch_size:
                    flush()
        if verbose and n % 500 == 0:
            print(f"  {n}/{len(records)} calls", flush=True)
    flush()

    # 창 -> 조각 평균, 그 다음 조각 -> 통화 평균 (m1.evaluate 와 동일한 순서)
    segment_probs: dict[str, list[float]] = {}
    for (call_id, _seg_idx), total in sums.items():
        segment_probs.setdefault(call_id, []).append(total / counts[(call_id, _seg_idx)])

    call_probs: dict[str, float | None] = {
        record.call_id: call_probability(np.array(segment_probs.get(record.call_id, [])))
        for record in records
    }

    rows = [
        {"audio file name": f"{record.call_id}.wav",
         "gender": call_label(call_probs[record.call_id], threshold)}
        for record in records
    ]

    # 라벨 JSON 이 없는 wav 도 빠뜨리지 않는다 (행 수 보존).
    seen = {record.call_id for record in records}
    for wav in sorted(audio_dir.glob("*.wav")):
        if wav.stem not in seen:
            rows.append({"audio file name": wav.name, "gender": call_label(None, threshold)})

    missing = sum(1 for r in records if not segment_probs.get(r.call_id))
    if verbose and missing:
        print(f"  경고: {missing}개 통화에서 신고자 조각을 얻지 못해 다수 클래스로 폴백")

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
