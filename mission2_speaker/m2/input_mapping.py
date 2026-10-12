"""Unambiguous WAV/JSON discovery for the submission CLI (stdlib only)."""

from pathlib import Path


def discover_input_files(audio_dir, label_dir):
    """Index each call once; reject duplicate IDs rather than drop/select files.

    A label without a corresponding WAV remains in the returned label list so
    the inference caller can preserve its utterances through silence fallback.
    """
    audio_dir, label_dir = Path(audio_dir), Path(label_dir)
    for name, directory in (("audio_dir", audio_dir), ("label_dir", label_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"[Error] {name} is not a directory: {directory}")

    labels = sorted(label_dir.rglob("*.json"))
    if not labels:
        raise FileNotFoundError(f"[Error] No JSON files found in {label_dir}")
    label_ids = {}
    for path in labels:
        if path.stem in label_ids:
            raise ValueError(
                f"[Error] Ambiguous duplicate JSON call ID '{path.stem}': "
                f"{label_ids[path.stem]} and {path}"
            )
        label_ids[path.stem] = path

    wavs = {}
    for path in sorted(audio_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        if path.stem in wavs:
            raise ValueError(
                f"[Error] Ambiguous duplicate WAV call ID '{path.stem}': "
                f"{wavs[path.stem]} and {path}"
            )
        wavs[path.stem] = path
    return labels, wavs
