"""Windows submission checks using only the Python standard library.

These commands verify files/CSV, not model quality through new training.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["audio file name", "startAt", "endAt", "speaker"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def dialogues(obj):
    for key in ("utterances", "dialogs", "dialogue"):
        if obj.get(key):
            return key, obj[key]
    raise ValueError("Label JSON has no nonempty utterance list")


def wav_index(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() == ".wav":
            if path.name in result:
                raise ValueError(f"Ambiguous duplicate WAV filename: {path.name}")
            result[path.name] = path
    return result


def label_files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(root.rglob("*.json"))
    if not paths:
        raise ValueError("No label JSON files")
    names = [p.stem for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate JSON stems: resolve ambiguity before testing")
    return paths


def speaker_target(utterance):
    speaker = utterance.get("speaker", utterance.get("role"))
    if speaker is None:
        return None
    if speaker in (0, "0", "상황실"):
        return 0
    if speaker in (1, "1", "신고자"):
        return 1
    raise ValueError(f"Unrecognized ground-truth speaker: {speaker!r}")


def boundary(utt):
    start = float(utt.get("startAt", utt.get("start_time", 0)))
    end = float(utt.get("endAt", utt.get("end_time", 0)))
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError(f"Invalid utterance boundary: {start}, {end}")
    # Match the standalone CLI's output conversion to integer milliseconds.
    return int(start), int(end)


def check_weights(args):
    models = read_json(ROOT / "results/recorded_summary.json")["models"]
    rows = []
    for model in models:
        path = args.checkpoints_dir / model["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as file:
            digest = hashlib.sha256()
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual.lower() != model["sha256"].lower():
            raise ValueError(f"Wrong checkpoint SHA-256: {path.name}; actual={actual}")
        rows.append({"file": path.name, "sha256": actual, "bytes": path.stat().st_size})
    if args.output:
        save_json(args.output, {"passed": True, "checkpoints": rows})
    print(json.dumps({"passed": True, "checkpoints": rows}, indent=2))


def prepare_smoke(args):
    wavs = wav_index(args.audio_dir)
    paths = label_files(args.label_dir)
    if args.output_dir.exists():
        raise FileExistsError("Use a new smoke directory")
    selected = []
    for path in paths:
        wav_name = path.stem + ".wav"
        if wav_name not in wavs:
            continue
        obj = read_json(path)
        key, utterances = dialogues(obj)
        subset = utterances[:args.max_utterances_per_call]
        for utt in subset:
            boundary(utt)
        selected.append((path, wavs[wav_name], obj, key, subset))
        if len(selected) == args.calls:
            break
    if len(selected) != args.calls:
        raise ValueError(f"Need {args.calls} matched calls; found {len(selected)}")
    audio, labels = args.output_dir / "audio", args.output_dir / "label"
    audio.mkdir(parents=True)
    labels.mkdir()
    rows = 0
    for label, wav, obj, key, subset in selected:
        shutil.copy2(wav, audio / wav.name)
        obj[key] = subset
        save_json(labels / label.name, obj)
        rows += len(subset)
    if args.include_boundary_cases:
        # Separate, unlabelled probes: zero-energy short audio and missing WAV.
        name = "__probe_silence__"
        if any(label.stem == name for label, *_ in selected):
            raise ValueError("Probe name collision")
        with wave.open(str(audio / (name + ".wav")), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(16000)
            file.writeframes(b"\0\0" * 16000)
        save_json(labels / (name + ".json"), {"utterances": [{"startAt": 0, "endAt": 500}]})
        save_json(labels / "__probe_missing__.json", {"utterances": [{"startAt": 0, "endAt": 1000}]})
        rows += 2
    manifest = {"source_calls": [p.stem for p, *_ in selected], "expected_rows": rows,
                "boundary_probes": 2 if args.include_boundary_cases else 0,
                "allowed_missing_rows": 1 if args.include_boundary_cases else 0,
                "note": "Smoke subset only; synthetic boundary probes have no ground-truth label"}
    save_json(args.output_dir / "smoke_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def expected_samples(label_dir, audio_dir):
    wavs = wav_index(audio_dir)
    expected = {}
    for path in label_files(label_dir):
        _, utterances = dialogues(read_json(path))
        name = path.stem + ".wav"
        for utt in utterances:
            start, end = boundary(utt)
            key = (name, start, end)
            if key in expected:
                raise ValueError(f"Duplicate utterance key in labels: {key}")
            expected[key] = {"target": speaker_target(utt), "audio_exists": name in wavs}
    return expected


def load_csv(path):
    with Path(path).open("rb") as file:
        if file.read(3) != b"\xef\xbb\xbf":
            raise ValueError("CSV must use UTF-8 BOM")
    rows = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Wrong CSV columns: {reader.fieldnames}")
        for row in reader:
            if set(row) != set(COLUMNS) or any(row[name] is None or row[name] == "" for name in COLUMNS):
                raise ValueError(f"Missing/extra CSV fields: {row}")
            if row["speaker"] not in ("0", "1"):
                raise ValueError(f"Not a binary prediction: {row}")
            key = (row[COLUMNS[0]], int(row["startAt"]), int(row["endAt"]))
            if key in rows:
                raise ValueError(f"Duplicate prediction key: {key}")
            rows[key] = int(row["speaker"])
    if not rows:
        raise ValueError("Empty CSV")
    return rows


def check_csv(args):
    expected = expected_samples(args.label_dir, args.audio_dir)
    actual = load_csv(args.csv)
    if expected.keys() != actual.keys():
        missing = list(expected.keys() - actual.keys())[:5]
        extra = list(actual.keys() - expected.keys())[:5]
        raise ValueError(f"Utterance mapping mismatch: missing={missing}, extra={extra}")
    missing_rows = sum(not info["audio_exists"] for info in expected.values())
    if missing_rows != args.allowed_missing_rows:
        raise ValueError(f"Unexpected missing-audio rows: {missing_rows}; allowed={args.allowed_missing_rows}")
    if args.expected_rows is not None and len(actual) != args.expected_rows:
        raise ValueError(f"Unexpected CSV row count: {len(actual)}")
    if args.log:
        data = args.log.read_bytes()
        encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        log = data.decode(encoding, errors="replace")
        if "Failed to load audio" in log or "Traceback (most recent call last)" in log:
            raise ValueError("Inference log contains an audio-load failure or traceback")
    cm = [[0, 0], [0, 0]]
    unscored = 0
    for key, info in expected.items():
        if not info["audio_exists"] or info["target"] is None:
            unscored += 1
        else:
            cm[info["target"]][actual[key]] += 1
    tn, fp = cm[0]
    fn, tp = cm[1]
    n = tn + fp + fn + tp
    if args.expected_valid_rows is not None and n != args.expected_valid_rows:
        raise ValueError(f"Unexpected scored denominator: {n}")
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    report = {"functional_checks_passed": True, "csv_rows": len(actual), "missing_audio_rows": missing_rows,
              "unscored_rows": unscored, "evaluation_denominator": n, "correct": tn + tp,
              "accuracy": (tn + tp) / n if n else None, "macro_f1": (f0 + f1) / 2 if n else None,
              "recall_0": tn / (tn + fp) if tn + fp else None,
              "recall_1": tp / (tp + fn) if tp + fn else None, "confusion_matrix": cm}
    if args.reference_metrics:
        reference = read_json(args.reference_metrics)
        if n != reference["evaluation_denominator"]:
            raise ValueError("Reference metric denominator differs")
        report["reference_accuracy"] = reference["accuracy"]
        report["accuracy_delta_pp"] = (report["accuracy"] - reference["accuracy"]) * 100
        report["correct_delta"] = report["correct"] - reference["total_correct"]
        report["reference_exact_correct_count_match"] = report["correct_delta"] == 0
    if args.compare_csv:
        reference_rows = load_csv(args.compare_csv)
        if actual.keys() != reference_rows.keys():
            raise ValueError("Comparison CSV has different utterance keys")
        diff = [dict(zip(COLUMNS[:3], key), reference=reference_rows[key], actual=actual[key])
                for key in actual if actual[key] != reference_rows[key]]
        report["different_predictions"] = len(diff)
        report["prediction_agreement_percent"] = 100 * (len(actual) - len(diff)) / len(actual)
        path = args.output.with_name(args.output.stem + "_differences.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=[*COLUMNS[:3], "reference", "actual"])
            writer.writeheader()
            writer.writerows(diff)
    save_json(args.output, report)
    print(json.dumps(report, indent=2))
    if report.get("reference_exact_correct_count_match") is False:
        print("REVIEW_REQUIRED: metric differs; retain new CSV and logs, do not overwrite reference results.")


def run_logged(args):
    command = args.arguments[1:] if args.arguments and args.arguments[0] == "--" else args.arguments
    if not command:
        raise ValueError("Provide a Python script/module command after --")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create keeps earlier execution evidence intact.
    with args.log.open("x", encoding="utf-8") as log:
        log.write("Python: " + sys.executable + "\nCommand: " + json.dumps(command, ensure_ascii=False) + "\n")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        process = subprocess.Popen([sys.executable, *command], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
        log.write(f"\nExit code: {code}\n")
    return code


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    weights = sub.add_parser("check-weights")
    weights.add_argument("--checkpoints-dir", type=Path, required=True)
    weights.add_argument("--output", type=Path)
    smoke = sub.add_parser("prepare-smoke")
    smoke.add_argument("--audio-dir", type=Path, required=True)
    smoke.add_argument("--label-dir", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--calls", type=int, default=2)
    smoke.add_argument("--max-utterances-per-call", type=int, default=20)
    smoke.add_argument("--include-boundary-cases", action="store_true")
    check = sub.add_parser("check-csv")
    check.add_argument("--csv", type=Path, required=True)
    check.add_argument("--audio-dir", type=Path, required=True)
    check.add_argument("--label-dir", type=Path, required=True)
    check.add_argument("--allowed-missing-rows", type=int, default=0)
    check.add_argument("--expected-rows", type=int)
    check.add_argument("--expected-valid-rows", type=int)
    check.add_argument("--log", type=Path)
    check.add_argument("--reference-metrics", type=Path)
    check.add_argument("--compare-csv", type=Path)
    check.add_argument("--output", type=Path, required=True)
    logged = sub.add_parser("run", help="Run a Python command and save combined UTF-8 output")
    logged.add_argument("--log", type=Path, required=True)
    logged.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "prepare-smoke" and (args.calls < 1 or args.max_utterances_per_call < 1):
        parser.error("Positive call/utterance counts required")
    if args.command == "run":
        raise SystemExit(run_logged(args))
    {"check-weights": check_weights, "prepare-smoke": prepare_smoke, "check-csv": check_csv}[args.command](args)


if __name__ == "__main__":
    main()
