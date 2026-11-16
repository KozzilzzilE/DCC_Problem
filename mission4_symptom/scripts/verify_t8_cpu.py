"""모델 없이 Baseline 소스와 T8의 전체 Sampling/Tokenization을 CPU에서 대조한다."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from transformers import AutoTokenizer

import prefix_experiment as runner
from m4.data import label_matrix, load_cache, prefix_texts, read_ids
from m4.losses import ASL_CONFIG

ROOT = runner.HERE.parent
GIT = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]


def git(*args):
    return subprocess.check_output([*GIT, *args], cwd=ROOT).decode("utf-8")


def sampler(source):
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in main.body if isinstance(n, ast.For) and ast.unparse(n.target) == "epoch")
    # Baseline에서 rng 생성부터 permutation까지 그대로 실행한다. 학습 루프는 실행하지 않는다.
    code = []
    for node in loop.body:
        code.append(node)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "order":
            break
    return compile(ast.Module(body=code, type_ignores=[]), "sampling_cpu", "exec")


def token_statistics(tokenizer, prefixes, labels, ordered, batch_size):
    ids_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
    stats = {"samples": len(ordered), "nonpadding_tokens": 0, "padding_tokens": 0,
             "raw_tokens": 0, "truncated_samples": 0, "removed_tokens": 0}
    exposure = np.zeros(9, dtype=np.int64)
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start:start + batch_size]
        texts = [prefixes[i][t - 1] for i, t in batch]
        raw = tokenizer(texts, truncation=False, padding=False, verbose=False)["input_ids"]
        enc = tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="np")
        ids, mask = enc["input_ids"], enc["attention_mask"]
        for value, digest in ((ids, ids_hash), (mask, mask_hash)):
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(np.asarray(value, dtype="<i8").tobytes())
        lengths = np.array([len(x) for x in raw])
        stats["raw_tokens"] += int(lengths.sum())
        stats["truncated_samples"] += int((lengths > 512).sum())
        stats["removed_tokens"] += int(np.maximum(lengths - 512, 0).sum())
        stats["nonpadding_tokens"] += int(mask.sum())
        stats["padding_tokens"] += int(mask.size - mask.sum())
        exposure += labels[batch[:, 0]].sum(axis=0)
    return {**stats, "label_positive_exposures": exposure.tolist(),
            "input_ids_sha256": ids_hash.hexdigest(), "attention_mask_sha256": mask_hash.hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if git("branch", "--show-current").strip() != "mission4-prefix-t8-asymmetric-loss":
        raise ValueError("T8 브랜치에서만 검증합니다")
    out = runner.reserve(args.out)
    baseline = git("show", "mission4-prefix:mission4_symptom/scripts/train_prefix.py")
    current = (runner.HERE / "scripts/train_prefix.py").read_text(encoding="utf-8")
    cfg = runner.read(ROOT / "outputs/mission4_prefix/localcontrol/training/run_config.json")
    initial = runner.read(ROOT / "outputs/mission4_prefix/localcontrol/initial.json")
    recipe = dict(runner.RECIPE, grad_accum=1, count_classes=0, count_weight=.5,
                  max_steps=0, grad_ckpt=False, log_every=200)
    checks = {"local_recipe": all(cfg[k] == v for k, v in recipe.items()),
              "local_initial_weights": runner.digest(runner.BACKBONE / "model.safetensors") == initial["initial_weights_sha256"],
              "local_initial_tokenizer": runner.digest(runner.BACKBONE / "tokenizer.json") == initial["initial_tokenizer_sha256"]}
    for path, expected in {**runner.INPUT_SHA256,
                           **{runner.BACKBONE / p: h for p, h in runner.BACKBONE_SHA256.items()},
                           runner.POLICY: runner.POLICY_SHA256,
                           **dict(zip(runner.FIXED, runner.FIXED_SHA256))}.items():
        checks[str(path.relative_to(ROOT))] = runner.digest(path) == expected
    # 학습·추론 통제 코드와 E026은 Baseline blob과 직접 대조한다.
    for rel in ["m4/data.py", "m4/halting.py", "m4/decision.py", "m4/simulate.py",
                "m4/streaming.py", "m4/bundle.py", "scripts/prefix_probs.py"]:
        checks[rel + "_source"] = (runner.HERE / rel).read_text(encoding="utf-8") == git("show", "mission4-prefix:mission4_symptom/" + rel)
    # 초기화/Tokenization/갱신의 개별 구문이 보존되었는지 AST로 확인한다.
    def statements(source):
        return [ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(source)) if isinstance(n, ast.stmt)]
    new_stmts = set(statements(current))
    for fragment in ["random.seed", "np.random.seed", "torch.manual_seed", "torch.cuda.manual_seed_all",
                     "model =", "tokenizer =", "tokenizer.truncation_side", "device =", "model.to",
                     "model.gradient_checkpointing_enable", "optimizer =", "scheduler =", "scaler =",
                     "enc =", "target =", "scaler.scale", "scaler.unscale_", "torch.nn.utils.clip_grad_norm_",
                     "scaler.step", "scaler.update", "scheduler.step", "optimizer.zero_grad"]:
        nodes = [n for n in ast.walk(ast.parse(baseline)) if isinstance(n, (ast.Assign, ast.Expr))
                 and ast.unparse(n).startswith(fragment)]
        checks[fragment + "_ast"] = bool(nodes) and all(ast.dump(n, include_attributes=False) in new_stmts for n in nodes)
    # 실행 조건(autocast/accumulation)도 비교한다.
    for cls, text in [(ast.With, "torch.autocast"), (ast.If, "step % args.grad_accum == 0")]:
        old = [n for n in ast.walk(ast.parse(baseline)) if isinstance(n, cls) and text in ast.unparse(n).split(":")[0]]
        new = [n for n in ast.walk(ast.parse(current)) if isinstance(n, cls) and text in ast.unparse(n).split(":")[0]]
        checks[text + "_condition"] = len(old) == len(new) == 1 and ast.dump(ast.parse(ast.unparse(old[0]).split(":")[0] + ":\n pass")) == ast.dump(ast.parse(ast.unparse(new[0]).split(":")[0] + ":\n pass"))
    if not all(checks.values()):
        runner.save(out / "failed_checks.json", checks)
        raise ValueError(f"통제 변수 불일치: {[k for k, v in checks.items() if not v]}")
    calls = load_cache(runner.CALLS, read_ids(runner.FIT))
    labels = label_matrix(calls).astype(np.int64)
    prefixes = [prefix_texts(c, recipe["fmt"]) for c in calls]
    tokenizer = AutoTokenizer.from_pretrained(runner.BACKBONE, local_files_only=True)
    tokenizer.truncation_side = recipe["truncation_side"]
    records = []
    for epoch in range(1, 4):
        ordered = []
        for source in (baseline, current):
            env = dict(np=np, args=SimpleNamespace(**recipe), epoch=epoch, prefixes=prefixes)
            exec(sampler(source), env)
            ordered.append(np.asarray(env["items"], dtype="<i8")[env["order"]])
        assert np.array_equal(*ordered)
        assert np.array_equal(np.bincount(ordered[0][:, 0]), np.full(len(calls), 2))
        summaries = [token_statistics(tokenizer, prefixes, labels, x, recipe["batch_size"]) for x in ordered]
        assert summaries[0] == summaries[1]
        record = {"epoch": epoch, "ordered_items_sha256": hashlib.sha256(ordered[0].tobytes()).hexdigest(),
                  "baseline": summaries[0], "t8": summaries[1], "exact_match": True}
        records.append(record)
        runner.save(out / f"epoch_{epoch}.json", record)
        print(f"epoch {epoch}: 전체 Sampling/Tokenization 일치", flush=True)
    total = sum(x["t8"]["samples"] for x in records)
    assert total == 157680 and total // 16 == 9855
    report = {"checks": checks, "passed": True, "loss": ASL_CONFIG, "epochs": records,
              "samples": total, "planned_optimizer_updates": total // 16,
              "baseline_commit": git("rev-parse", "mission4-prefix").strip(),
              "initialization_scope": "가중치/Tokenizer 해시, seed, 모델 생성 구문 동일. 실제 모델 로드 및 GPU 연산은 미실행",
              "historical_scope": "저장된 Local Baseline 설정과 현재 입력의 재생 비교. 과거 학습 당시 Token/Sampling 영수증은 없음",
              "not_run": ["GPU 학습", "CUDA AMP", "실제 Optimizer 업데이트/AMP Skip", "전체 분류 모델 Batch/Sequential 추론"]}
    runner.save(out / "cpu_verification.json", report)


if __name__ == "__main__":
    main()

