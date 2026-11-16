"""팀 Baseline 연결 실행기. 정책/학습 알고리즘을 바꾸지 않는다.

verify는 기존 확률표와 고정 E026만 평가한다(분류 모델 실행 없음).
run은 명시적으로 요청할 때만 기존 E002 학습/확률 추출 스크립트를 호출한다.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from m4 import SYMPTOM_CLASSES
from m4 import halting as hl, simulate as sim
from m4.data import load_cache, read_ids

TEAM = HERE / "runs"
POLICY = TEAM / "E026_learned_halt05/wide/policy.json"
CALLS = TEAM / "cache/train.jsonl"
FIT = TEAM / "cache/fit_ids.txt"
DEV = TEAM / "cache/dev_ids.txt"
E002 = TEAM / "E002_prefix_uniform"
FIXED = [TEAM / "E001_m3_baseline/dev_prefix.npz", TEAM / "E007_prefix_uniform_s43/dev_prefix.npz"]
MODEL1 = HERE.parent / "mission3_symptom/runs/devsel/tl_s42/best_model"
MODEL7 = TEAM / "E007_prefix_uniform_s43/best_model"
BACKBONE = HERE.parent / "mission3_symptom/runs/devsel/tapt"
OUTPUTS = HERE.parent / "outputs/mission4_prefix"
POLICY_SHA256 = "80e69bcd99f6a1321586bb39411e512147a56dc5a3aa68623fe06595554689f1"
FIXED_SHA256 = ["a1a6e1959f0767c18e56efdad1fc3e97ff05c075b01bd1b835101b8e5611a4f5",
                "f7eef2456999b8bca0130d0a6f5cffd49c5fd488be9b82ac52a0eb1cd3b5d136"]
INPUT_SHA256 = {
    CALLS: "9e00495204f96a697f9c7a914e3ce5d39d5ed7f3948dfe64db0f481f6e08097a",
    FIT: "2f4bd4c3139784949aaaa234bd569a3bcecfd9348fc4c4efc48d8077831db022",
    DEV: "b3e0446e8be6217aaf8bba9e1c54b5bfc6d44ae09140b11e218efb0e32aa2b2e",
}
BACKBONE_SHA256 = {
    "config.json": "d3c69af98d4de8e11599af9a7a1a02ed0e75a4a2d02385eed9e9003cb69db14d",
    "tokenizer_config.json": "fe5faa5bd4f6d95566555775b242be3e211287e9989337c0225214707d428363",
    "model.safetensors": "254249750ed0dae0b2e50a93874fad537f5329dabfd83a3cc4de2a4f58f42b4e",
    "tokenizer.json": "200c2fd5a65daf3b67aa532740d3a26736deceaa8b28e82826038aede9377cf0",
}
RECIPE = {"fmt": "space", "truncation_side": "right", "max_length": 512,
          "epochs": 3, "samples_per_call": 2, "include_full": False,
          "batch_size": 16, "lr": 5e-5, "llrd": 0.8, "pos_weight_power": 0.5,
          "seed": 42}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def reserve(path):
    """기존 결과와 팀 runs/를 출력 대상으로 허용하지 않는다."""
    path = Path(path).resolve()
    if path == OUTPUTS.resolve() or not path.is_relative_to(OUTPUTS.resolve()):
        raise ValueError(f"출력은 {OUTPUTS} 아래의 새 실험 폴더여야 합니다")
    path.mkdir(parents=True, exist_ok=False)
    return path


def aligned(tables):
    first = tables[0]
    for table in tables:
        if (table.call_ids != first.call_ids or
                not np.array_equal(table.offsets, first.offsets) or
                not np.array_equal(table.labels, first.labels)):
            raise ValueError("확률표의 통화 순서/발화 구간/정답이 다릅니다")
        if not np.isfinite(table.probs).all() or np.any((table.probs < 0) | (table.probs > 1)):
            raise ValueError("확률 범위 오류")
    return sim.PrefixTable(first.call_ids, first.offsets,
                          np.mean([t.probs for t in tables], axis=0).astype(np.float32), first.labels)


def confusion(pred, gold):
    pred, gold = np.asarray(pred, bool), np.asarray(gold, bool)
    result = {}
    for k, name in enumerate(SYMPTOM_CLASSES):
        p, y = pred[:, k], gold[:, k]
        tp, fp, fn, tn = (int(x.sum()) for x in (p & y, p & ~y, ~p & y, ~p & ~y))
        result[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "precision": tp / (tp + fp) if tp + fp else None,
                        "recall": tp / (tp + fn) if tp + fn else None,
                        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    return result


def prepare(e002_table):
    if digest(POLICY) != POLICY_SHA256 or any(digest(p) != h for p, h in zip(FIXED, FIXED_SHA256)):
        raise ValueError("고정 E001/E007 확률표 또는 E026 정책 파일이 변경됨")
    if any(digest(p) != h for p, h in INPUT_SHA256.items()):
        raise ValueError("팀 캐시 또는 Train/Dev 분할 파일이 변경됨")
    paths = [FIXED[0], Path(e002_table), FIXED[1]]
    table = aligned([sim.PrefixTable.load(p) for p in paths])
    dev, fit = read_ids(DEV), read_ids(FIT)
    if (len(dev) != 2920 or len(fit) != 26280 or len(set(dev)) != 2920 or
            len(set(fit)) != 26280 or set(dev) & set(fit) or set(table.call_ids) != set(dev)):
        raise ValueError("팀 Train/Dev 분할 불일치")
    calls_by_id = {c.call_id: c for c in load_cache(CALLS, dev)}
    calls = [calls_by_id[i] for i in table.call_ids]
    if any(c.n_utt != n for c, n in zip(calls, table.n_utt)):
        raise ValueError("캐시와 확률표 발화 수 불일치")
    if not np.array_equal([c.labels for c in calls], table.labels):
        raise ValueError("캐시와 확률표 정답 불일치")
    policy = hl.HaltPolicy.load(POLICY)
    if (policy.beta, policy.tau, policy.min_step) != (128.0, 0.5, 5):
        raise ValueError("E026 고정 정책 설정 불일치")
    policy.booster()  # 특징 순서 검증 포함, 정책 재학습은 하지 않음
    return paths, table, calls, policy


def evaluate(e002_table):
    paths, table, calls, policy = prepare(e002_table)
    started = time.perf_counter()
    data = hl.HaltData.build(table, calls, policy.logit_shift)
    halts = policy.halts(data)
    score42 = hl.score_halts(data, halts)
    idx = table.offsets[:-1] + halts - 1
    score42.update({"halt_mean": float(halts.mean()),
                    "halt_fraction_mean": float((halts / table.n_utt).mean()),
                    "per_class": confusion(data.pred[idx], table.labels)})
    result = {
        "mission_4_1": {**sim.score_4_1(table, sim.decide),
            "per_class_all_moments": confusion(sim.decide(table.probs), np.repeat(table.labels, table.n_utt, axis=0)),
            "aggregation": "세부 혼동행렬은 발화 시점 단위, 공식 Score는 통화별 평균"},
        "mission_4_2": score42,
        "timing": {"cached_evaluation_seconds": time.perf_counter() - started,
                   "sequential_inference_seconds": None, "seconds_per_utterance": None,
                   "note": "확률표 평가 시간은 분류 모델 추론 시간이 아님"},
        "provenance": {"tables": {str(p): digest(p) for p in paths},
            "policy_sha256": digest(POLICY), "cache_sha256": digest(CALLS),
            "fit_ids_sha256": digest(FIT), "dev_ids_sha256": digest(DEV),
            "policy": {"beta": policy.beta, "tau": policy.tau, "min_step": policy.min_step,
                       "logit_shift": policy.logit_shift, "feature_names": policy.feature_names},
            "versions": {n: importlib.metadata.version(n) for n in ("numpy", "torch", "transformers", "lightgbm")}},
    }
    cfg_path = Path(e002_table).parent / "training/run_config.json"
    if not cfg_path.is_file():
        cfg_path = Path(e002_table).parent / "run_config.json"
    meta_path = Path(e002_table).with_suffix(".meta.json")
    result["timing"]["e002_training_record"] = read(cfg_path) if cfg_path.is_file() else None
    result["timing"]["e002_offline_extraction_record"] = read(meta_path) if meta_path.is_file() else None
    result["timing"]["fixed_members_offline_extraction_records"] = {
        str(p): read(p.with_suffix(".meta.json")) for p in FIXED}
    return result, table, calls, halts


def verify(check_online=False):
    result, table, _, halts = evaluate(E002 / "dev_prefix.npz")
    recorded = read(TEAM / "E026_learned_halt05/wide/result.json")["runs"][0]["scores"]["learned"]
    checks = {k: abs(result["mission_4_2"][k] - recorded[k]) <= 1e-10
              for k in ("score", "macro_f1", "earliness", "halt_median", "never_halted")}
    checks["4-1_rounded"] = round(result["mission_4_1"]["score"], 4) == 0.5060
    if check_online:
        _, _, calls, policy = prepare(E002 / "dev_prefix.npz")
        online = []
        for i, call in enumerate(calls):
            state = hl.HaltFeatureState(policy.logit_shift)
            for j, (text, spk, start, end) in enumerate(zip(call.texts, call.speakers, call.starts, call.ends)):
                features = state.update(text, spk, start, end, table.probs[table.offsets[i] + j])
                if policy.should_halt(state, features):
                    break
            online.append(j + 1)
        checks["cached_batch_vs_online_halts_all_calls"] = bool(np.array_equal(online, halts))
    # 과거 Dev의 통화별 E026 halt 배열은 저장되어 있지 않다. 평균/중앙값만으로 동일함을 주장하지 않는다.
    result["verification"] = {"checks": checks, "passed": all(checks.values()),
        "historical_per_call_halts": "미확보: 이번 계산값을 새 기준 배열로 보존 가능"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit("팀 기준점과 불일치. 학습 전에 원인을 확인하십시오")
    return result, table, halts


def invoke(script, args, log):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if log.exists():
        log = log.with_name(log.stem + ".retry-" + str(time.time_ns()) + log.suffix)
    with log.open("x", encoding="utf-8") as out:
        subprocess.run([sys.executable, "-B", str(HERE / "scripts" / script), *args], cwd=HERE,
                       env=env, stdout=out, stderr=subprocess.STDOUT, check=True)


def run(out, strategy="uniform", stage="all"):
    out = Path(out).resolve()
    if strategy != "uniform" and not out.is_relative_to((OUTPUTS / "t5_early_sampling").resolve()):
        raise ValueError("T5 output must be below outputs/mission4_prefix/t5_early_sampling")
    # 먼저 팀 기록과 동일한 평가 경로인지 확인한다. 불일치 시 새 학습을 시작하지 않는다.
    reference, _, _ = verify()
    cfg = read(E002 / "run_config.json")
    if any(cfg[k] != v for k, v in RECIPE.items()):
        raise ValueError("E002 팀 레시피가 예상값과 다름")
    if any(digest(BACKBONE / p) != h for p, h in BACKBONE_SHA256.items()):
        raise ValueError("팀 TAPT 초기 가중치 또는 Tokenizer가 변경됨")
    if stage in ("all", "train"):
        out = reserve(out)
    elif not (out / "initial.json").is_file():
        raise FileNotFoundError(out / "initial.json")
    if stage in ("all", "train"):
        save(out / "initial.json", {"recipe": RECIPE, "sampling_strategy": strategy, "backbone": str(BACKBONE),
        "initial_weights_sha256": digest(BACKBONE / "model.safetensors"),
        "initial_tokenizer_sha256": digest(BACKBONE / "tokenizer.json"),
        "fit_ids_sha256": digest(FIT), "dev_ids_sha256": digest(DEV),
        "source_sha256": {str(p): digest(p) for p in [Path(__file__), HERE / "scripts/train_prefix.py",
            HERE / "scripts/prefix_probs.py", HERE / "m4/halting.py", HERE / "m4/simulate.py"]}})
    args = ["--calls", str(CALLS), "--ids", str(FIT), "--backbone", str(BACKBONE), "--out", str(out / "training")]
    for key, value in RECIPE.items():
        if key != "include_full":
            args += ["--" + key.replace("_", "-"), str(value)]
    args += ["--grad-accum", "1", "--count-classes", "0", "--max-steps", "0"]
    if strategy != "uniform":
        args += ["--sampling-strategy", strategy]
    if stage in ("all", "train"):
        invoke("train_prefix.py", args, out / "training.log")
    if stage == "train":
        return
    cfg_local = read(out / "training/run_config.json")
    if cfg_local.get("sampling_strategy", "uniform") != strategy:
        raise ValueError("Training strategy mismatch")
    if (out / "metrics.json").exists():
        raise FileExistsError(out / "metrics.json")
    if not (out / "dev_prefix.npz").exists():
        invoke("prefix_probs.py", ["--model", str(out / "training/best_model"), "--calls", str(CALLS),
        "--ids", str(DEV), "--out", str(out / "dev_prefix.npz"), "--fmt", "space",
        "--max-length", "512", "--truncation-side", "right", "--token-budget", "24000"], out / "extract.log")
    write_evaluation(out / "dev_prefix.npz", out, existing=True, reference=reference)


def write_evaluation(table_path, out, existing=False, reference=None):
    out = Path(out) if existing else reserve(out)
    result, table, _, halts = evaluate(table_path)
    cfg_path = Path(table_path).parent / "training/run_config.json"
    checkpoint = cfg_path.parent / "best_model/model.safetensors"
    if checkpoint.is_file():
        result["provenance"]["e002_checkpoint_sha256"] = digest(checkpoint)
    if reference is not None:
        for key in ("policy_sha256", "cache_sha256", "fit_ids_sha256", "dev_ids_sha256"):
            if result["provenance"][key] != reference["provenance"][key]:
                raise ValueError(f"비교 기준의 고정 입력이 다릅니다: {key}")
        result["difference_from_reference"] = {
            "mission_4_1": result["mission_4_1"]["score"] - reference["mission_4_1"]["score"],
            **{k: result["mission_4_2"][k] - reference["mission_4_2"][k]
               for k in ("score", "macro_f1", "earliness", "halt_mean", "halt_median")}}
    save(out / "metrics.json", result)
    np.savez_compressed(out / "halts.npz", call_ids=np.array(table.call_ids), halts=halts)
    print(json.dumps({"out": str(out), "4-1": result["mission_4_1"]["score"],
                      "4-2": result["mission_4_2"]["score"]}, ensure_ascii=False))


def sequential_difference(report, baseline):
    difference = {m: {k: report[m][k] - baseline[m][k]
        for k in ("score", "seconds", "seconds_per_utterance") if k in baseline[m]}
        for m in ("4-1", "4-2")}
    current = report["4-1"]["metrics"]["by_third"]
    reference = baseline["4-1"]["metrics"]["by_third"]
    difference["4-1"]["metrics"] = {"by_third": {
        name: current[name] - reference[name] for name in current}}
    return difference


def sequential(out, e002_run, device):
    """실측 전용: fit-only 멤버로 기존 StreamingSymptomModel을 실행한다."""
    import torch
    from m4.bundle import pack_bundle
    from m4.streaming import StreamingSymptomModel
    cfg = read(Path(e002_run) / "run_config.json")
    if cfg["n_calls"] != 26280 or cfg["ids"] is None or digest(FIT) != digest(cfg["ids"] if Path(cfg["ids"]).is_absolute() else HERE / cfg["ids"]):
        raise ValueError("E002는 기존 fit ID 26,280통화로만 학습한 모델이어야 합니다")
    # E001/E007은 고정된 팀 fit-only 모델 경로만 사용. 최종 제출 번들 사용 금지.
    for p in (MODEL1, MODEL7, Path(e002_run) / "best_model"):
        if not (p / "model.safetensors").is_file():
            raise FileNotFoundError(p)
    _, table, calls, policy = prepare(E002 / "dev_prefix.npz")
    out = reserve(out)
    settings = {"fmt": "space", "max_length": 512, "truncation_side": "right",
        "logit_shift": policy.logit_shift, "rule": {"type": "threshold", "threshold": 0.5, "force_top1": False},
        "halt_rule": {"type": "threshold", "threshold": 0.5, "force_top1": False}, "halt": policy.to_dict()}
    pack_bundle([MODEL1, Path(e002_run) / "best_model", MODEL7], out / "fit_only.pt", settings)
    started = time.perf_counter()
    model = StreamingSymptomModel(str(out / "fit_only.pt"), device)
    if device == "cuda" and model.device.type != "cuda":
        raise RuntimeError("CUDA가 없어 CPU로 대체됨")
    sync = lambda: torch.cuda.synchronize() if model.device.type == "cuda" else None
    sync()
    load_seconds = time.perf_counter() - started
    report = {"model_load_seconds": load_seconds, "device": str(model.device),
        "scope": "캐시를 메모리에 읽은 뒤 기존 StreamingSymptomModel 순차 실행. WAV/JSON IO·번들 조립 제외. 워밍업 없음",
        "model_hashes": {str(p): digest(p / "model.safetensors") for p in (MODEL1, MODEL7, Path(e002_run) / "best_model")}}
    for mission in ("4-1", "4-2"):
        predicted, hs, count = [], [], 0
        sync(); started = time.perf_counter()
        for call in calls:
            model.reset()
            for t, (text, spk, start, end) in enumerate(zip(call.texts, call.speakers, call.starts, call.ends), 1):
                model.push(text, spk, start, end)
                count += 1
                if mission == "4-1":
                    symptoms = set(model.symptoms())
                    predicted.append([c in symptoms for c in SYMPTOM_CLASSES])
                elif model.should_halt():
                    break
            if mission == "4-2":
                hs.append(t)
                symptoms = set(model.symptoms(at_halt=True))
                predicted.append([c in symptoms for c in SYMPTOM_CLASSES])
        sync(); seconds = time.perf_counter() - started
        p = np.asarray(predicted, bool)
        if mission == "4-1":
            ms = sim.moment_scores(p, np.repeat(table.labels, table.n_utt, axis=0))
            score = float((np.add.reduceat(ms, table.offsets[:-1]) / table.n_utt).mean())
        else:
            score = sim.macro_f1(p, table.labels)["macro_f1"] * float(sim.earliness(np.array(hs), table.n_utt).mean())
        report[mission] = {"score": score, "seconds": seconds, "processed_utterances": count,
                           "seconds_per_utterance": seconds / count,
                           "seconds_with_model_load": seconds + load_seconds}
        if mission == "4-1":
            binary_table = sim.PrefixTable(table.call_ids, table.offsets, p.astype(np.float32), table.labels)
            report[mission]["metrics"] = {**sim.score_4_1(binary_table, sim.decide),
                "per_class_all_moments": confusion(p, np.repeat(table.labels, table.n_utt, axis=0))}
        else:
            report[mission]["metrics"] = {**sim.macro_f1(p, table.labels),
                "earliness": float(sim.earliness(np.array(hs), table.n_utt).mean()),
                "halt_mean": float(np.mean(hs)), "halt_median": float(np.median(hs)),
                "confusion": confusion(p, table.labels)}
        np.savez_compressed(out / f"sequential_{mission}.npz", call_ids=np.array(table.call_ids), pred=p, halts=np.array(hs))
    reference_path = OUTPUTS / "localsequential/sequential_timing.json"
    if reference_path.is_file():
        baseline = read(reference_path)
        report["baseline_reference_sha256"] = digest(reference_path)
        report["delta_from_local_sequential"] = sequential_difference(report, baseline)
    save(out / "sequential_timing.json", report)


def main():
    os.chdir(HERE)
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify", help="분류 모델 없이 팀 기준점 검산")
    v.add_argument("--out", type=Path)
    v.add_argument("--check-online", action="store_true", help="기존 확률로 온라인 특징/중단을 재생해 통화별 일치 검사")
    r = sub.add_parser("run", help="기존 E002 재학습→확률 추출→고정 정책 평가")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--sampling-strategy", choices=["uniform", "weighted"], default="uniform")
    r.add_argument("--stage", choices=["all", "train", "batch"], default="all")
    r.add_argument("--sequential-after", action="store_true")
    e = sub.add_parser("evaluate", help="기존 확률표에 고정 정책만 적용")
    e.add_argument("--e002-table", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--reference", type=Path, help="비교할 기존 metrics.json (팀 기준 또는 로컬 기준)")
    s = sub.add_parser("sequential", help="명시적 실제 분류 모델 순차 추론/시간 측정")
    s.add_argument("--e002-run", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = ap.parse_args()
    if args.command == "verify":
        result, table, halts = verify(args.check_online)
        if args.out:
            out = reserve(args.out)
            save(out / "metrics.json", result)
            np.savez_compressed(out / "halts.npz", call_ids=np.array(table.call_ids), halts=halts)
    elif args.command == "run":
        run(args.out, args.sampling_strategy, args.stage)
        if args.sequential_after:
            if args.stage == "train":
                raise ValueError("Use all/batch for sequential-after")
            sequential(Path(args.out) / "sequential", Path(args.out) / "training", "cuda")
    elif args.command == "evaluate":
        write_evaluation(args.e002_table, args.out, reference=read(args.reference) if args.reference else None)
    else:
        sequential(args.out, args.e002_run, args.device)


if __name__ == "__main__":
    main()
