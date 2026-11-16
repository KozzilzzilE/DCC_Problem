"""T10 학습 전 CPU 검증. 모델/정책 로드·추론 없이 소스·가중치·Sampling을 확인한다."""
from __future__ import annotations
import argparse
import ast
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import prefix_experiment as runner
from m4.data import label_matrix, load_cache, read_ids

ROOT = runner.HERE.parent
GIT = ["git", "-c", "safe.directory=" + ROOT.as_posix()]


def git(*args):
    return subprocess.check_output([*GIT, *args], cwd=ROOT).decode("utf-8")


def sampling_code(source):
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in main.body if isinstance(n, ast.For) and ast.unparse(n.target) == "epoch")
    nodes = []
    for node in loop.body:
        nodes.append(node)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "order":
            break
    return compile(ast.Module(body=nodes, type_ignores=[]), "actual_sampling", "exec")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert git("branch", "--show-current").strip() == "mission4-prefix-t10-positive-weight"
    assert git("rev-parse", "HEAD") == git("rev-parse", "mission4-prefix")
    out = runner.reserve(args.out)
    protected = [p for folder in [runner.E002, ROOT/"outputs/mission4_prefix/localcontrol", ROOT/"outputs/mission4_prefix/localsequential"]
                 for p in folder.rglob("*") if p.is_file() and p.suffix in {".json", ".npz", ".txt", ".log"}]
    protected += [runner.POLICY, *runner.FIXED, *(runner.BACKBONE/p for p in runner.BACKBONE_SHA256),
                  runner.MODEL1/"model.safetensors", runner.MODEL7/"model.safetensors", *runner.INPUT_SHA256]
    before = {str(p): runner.digest(p) for p in protected}
    old = git("show", "mission4-prefix:mission4_symptom/scripts/train_prefix.py")
    new = (runner.HERE/"scripts/train_prefix.py").read_text(encoding="utf-8")
    cfg = runner.read(ROOT/"outputs/mission4_prefix/localcontrol/training/run_config.json")
    initial = runner.read(ROOT/"outputs/mission4_prefix/localcontrol/initial.json")
    expected = dict(runner.RECIPE, grad_accum=1, count_classes=0, max_steps=0, grad_ckpt=False)
    checks = {"local_baseline_recipe": all(cfg[k] == v for k,v in expected.items())}
    checks.update({str(p): before[str(p)] == h for p,h in runner.INPUT_SHA256.items()})
    checks.update({"tapt_"+p: before[str(runner.BACKBONE/p)] == h for p,h in runner.BACKBONE_SHA256.items()})
    checks["policy_hash"] = before[str(runner.POLICY)] == runner.POLICY_SHA256
    checks["fixed_tables"] = all(before[str(p)] == h for p,h in zip(runner.FIXED, runner.FIXED_SHA256))
    checks["local_initial_weights"] = before[str(runner.BACKBONE/"model.safetensors")] == initial["initial_weights_sha256"]
    checks["local_initial_tokenizer"] = before[str(runner.BACKBONE/"tokenizer.json")] == initial["initial_tokenizer_sha256"]
    seq = runner.read(ROOT/"outputs/mission4_prefix/localsequential/sequential_timing.json")
    checks["fixed_models"] = all(before[str(p/"model.safetensors")] == seq["model_hashes"][str(p)] for p in [runner.MODEL1,runner.MODEL7])
    for rel in ["m4/data.py", "m4/decision.py", "m4/bundle.py", "m4/halting.py", "m4/streaming.py", "m4/simulate.py", "scripts/pack_bundle.py", "scripts/prefix_probs.py"]:
        checks[rel] = (runner.HERE/rel).read_text(encoding="utf-8") == git("show", "mission4-prefix:mission4_symptom/"+rel)
    current_nodes = {ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(new))}
    prefixes = ["random.seed", "np.random.seed", "torch.manual_seed", "torch.cuda.manual_seed_all", "tokenizer =", "tokenizer.truncation_side", "model =",
                "model.to", "model.gradient_checkpointing_enable", "optimizer =", "scheduler =", "scaler =", "loss_fn =", "pos =", "pos_weight =", "enc =", "target =",
                "scaler.scale", "scaler.unscale_", "torch.nn.utils.clip_grad_norm_", "scaler.step", "scaler.update", "scheduler.step", "optimizer.zero_grad"]
    for prefix in prefixes:
        nodes = [n for n in ast.walk(ast.parse(old)) if isinstance(n, (ast.Assign, ast.Expr)) and ast.unparse(n).startswith(prefix)]
        checks[prefix] = bool(nodes) and all(ast.dump(n, include_attributes=False) in current_nodes for n in nodes)
    # autocast 및 누적 갱신 조건을 포함하는 헤더도 동일하다.
    for marker in ["with torch.autocast", "if step % args.grad_accum"]:
        checks[marker] = [line.strip() for line in old.splitlines() if line.strip().startswith(marker)] == [line.strip() for line in new.splitlines() if line.strip().startswith(marker)]
    assert all(checks.values()), [k for k,v in checks.items() if not v]
    calls = load_cache(runner.CALLS, read_ids(runner.FIT))
    labels = label_matrix(calls).astype(np.float32)
    assert len(calls) == 26280
    assert len(set(read_ids(runner.DEV))) == 2920 and not set(read_ids(runner.FIT)) & set(read_ids(runner.DEV))
    rows = []
    values = {}
    for power in [.5,.6]:
        env = dict(np=np, labels=labels, args=SimpleNamespace(pos_weight_power=power))
        nodes = [n for n in ast.walk(ast.parse(new)) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {"pos", "pos_weight"}]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),"actual_pos_weight","exec"),env)
        values[str(power)] = env["pos_weight"]
    np.testing.assert_array_equal(values["0.5"], np.asarray(cfg["pos_weight"],dtype=np.float32))
    expected_t10 = np.array([2.5010,2.7831,3.7601,2.0600,2.1555,3.3579,3.3984,2.4563,3.0514])
    np.testing.assert_allclose(values["0.6"],expected_t10,rtol=0,atol=.00005)
    assert list(runner.SYMPTOM_CLASSES) == ["고열","구토","두통","복통","어지러움","열상","오심","전신쇠약","호흡곤란"]
    for i,name in enumerate(runner.SYMPTOM_CLASSES):
        rows.append({"class":name,"positive":int(labels[:,i].sum()),"negative":int(len(labels)-labels[:,i].sum()),
                     "baseline":float(values['0.5'][i]),"t10":float(values['0.6'][i])})
    cpu = runner.read(ROOT/"outputs/mission4_prefix/t8_cpu_verified/cpu_verification.json")
    for epoch in [1,2,3]:
        result = []
        for source,power in [(old,.5),(new,.6)]:
            env = dict(np=np,args=SimpleNamespace(seed=42,samples_per_call=2,include_full=False,pos_weight_power=power),
                       epoch=epoch,prefixes=[range(c.n_utt) for c in calls])
            exec(sampling_code(source),env)
            ordered = np.asarray(env['items'],dtype='<i8')[env['order']]
            result.append(ordered)
        np.testing.assert_array_equal(*result)
        assert len(result[0]) == 52560
        np.testing.assert_array_equal(np.bincount(result[0][:,0]),np.full(len(calls),2))
        digest = hashlib.sha256(result[0].tobytes()).hexdigest()
        assert digest == cpu['epochs'][epoch-1]['ordered_items_sha256']
        checks['sampling_epoch_'+str(epoch)] = True
    runner.save(out/'cpu_verification.json',{'checks':checks,'passed':True,'weights':rows,'samples':157680,'planned_updates':9855,
        'baseline_commit':git('rev-parse','mission4-prefix').strip(),
        'scope':'실제 소스의 Loss 계산/초기화 구문·Sampling 실행 및 파일 해시 대조. 모델 Tensor 초기화·CUDA AMP·전체 추론 미실행',
        'tokenization_scope':'Tokenizer/입력 구문·설정 동일. 전체 Tokenization 재실행은 하지 않음'})
    assert all(runner.digest(Path(p)) == h for p,h in before.items())
    runner.save(out/'protected_files.json',{'before_sha256':before,'unchanged':True})
    print('T10 CPU 검증 통과: 실제 계산 가중치, 전체 Sampling, 통제 소스 및 보호 파일 해시 일치')


if __name__ == '__main__':
    main()
