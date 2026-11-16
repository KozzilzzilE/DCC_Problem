"""T10의 실제 학습 구문과 실행 옵션 전달을 CPU에서 검증한다."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
SOURCE = (HERE / "scripts/train_prefix.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
spec = importlib.util.spec_from_file_location("t10_runner", HERE / "scripts/prefix_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def actual_loss(labels, power):
    """학습기의 실제 Pos Weight 계산 및 Loss 생성 구문만 실행한다."""
    nodes = [n for n in ast.walk(TREE) if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {"pos", "pos_weight", "loss_fn"}]
    env = dict(labels=labels, np=np, torch=torch, args=SimpleNamespace(pos_weight_power=power), device=torch.device("cpu"))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "actual_training_loss", "exec"), env)
    return env["pos_weight"], env["loss_fn"]


def test_default_is_unchanged():
    node = next(n for n in ast.walk(TREE) if isinstance(n, ast.Call) and n.args
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--pos-weight-power")
    assert next(k.value.value for k in node.keywords if k.arg == "default") == .5
    labels = (np.arange(90).reshape(10, 9) % 4 == 0).astype(np.float32)
    w, fn = actual_loss(labels, .5)
    expected = ((len(labels)-labels.sum(0))/labels.sum(0))**.5
    np.testing.assert_array_equal(w, expected)
    x, y = torch.randn(2, 9), torch.ones(2, 9)
    torch.testing.assert_close(fn(x, y), torch.nn.functional.binary_cross_entropy_with_logits(x, y, pos_weight=torch.tensor(expected)))
    assert fn.reduction == "mean"


@pytest.mark.parametrize("positive", [False, True])
def test_actual_loss_and_gradient_change_only_for_positive_labels(positive):
    labels = (np.arange(90).reshape(10, 9) % 4 == 0).astype(np.float32)
    x = torch.linspace(-5, 5, 27).reshape(3, 9).requires_grad_()
    y = torch.full_like(x, float(positive))
    w0, f0 = actual_loss(labels, .5)
    w1, f1 = actual_loss(labels, .6)
    g0 = torch.autograd.grad(f0(x, y), x)[0]
    g1 = torch.autograd.grad(f1(x, y), x)[0]
    if positive:
        torch.testing.assert_close(g1, g0 * torch.tensor(w1/w0))
        element0 = torch.nn.functional.binary_cross_entropy_with_logits(x, y, pos_weight=torch.tensor(w0), reduction="none")
        element1 = torch.nn.functional.binary_cross_entropy_with_logits(x, y, pos_weight=torch.tensor(w1), reduction="none")
        torch.testing.assert_close(element1, element0 * torch.tensor(w1/w0))
    else:
        torch.testing.assert_close(f0(x, y), f1(x, y), rtol=0, atol=0)
        torch.testing.assert_close(g0, g1, rtol=0, atol=0)


def test_mixed_multilabel_mean_gradient():
    labels = (np.arange(90).reshape(10, 9) % 4 == 0).astype(np.float32)
    w, fn = actual_loss(labels, .6)
    x = torch.linspace(-5, 5, 27).reshape(3, 9).requires_grad_()
    y = (torch.arange(27).reshape(3, 9) % 2).float()
    g = torch.autograd.grad(fn(x, y), x)[0]
    torch.testing.assert_close(g, (y*torch.tensor(w)*(x.sigmoid()-1)+(1-y)*x.sigmoid())/27)


@pytest.mark.parametrize("power", [.5, .6])
def test_power_reaches_training_subprocess_without_recipe_mutation(tmp_path, power):
    hashes = {str(runner.BACKBONE/p): h for p,h in runner.BACKBONE_SHA256.items()}
    original = dict(runner.RECIPE)
    with patch.object(runner, "verify", return_value=({}, None, None)), \
         patch.object(runner, "read", return_value=original), \
         patch.object(runner, "digest", side_effect=lambda p: hashes.get(str(p), "test")), \
         patch.object(runner, "reserve", return_value=tmp_path), patch.object(runner, "save") as save, \
         patch.object(runner, "invoke") as invoke, patch.object(runner, "write_evaluation"):
        runner.run(tmp_path, power)
        script, args, _ = invoke.call_args_list[0].args
        assert script == "train_prefix.py"
        assert args[args.index("--pos-weight-power")+1] == str(power)
        for key,value in original.items():
            if key not in {"pos_weight_power", "include_full"}:
                assert args[args.index("--"+key.replace("_","-"))+1] == str(value)
        assert save.call_args.args[1]["recipe"]["pos_weight_power"] == power
    assert runner.RECIPE == original


def test_single_command_sequential_connection():
    with patch.object(runner.os, "chdir"), patch.object(runner, "run") as run, \
         patch.object(runner, "sequential") as seq, \
         patch("sys.argv", ["runner", "run", "--out", "new", "--pos-weight-power", "0.6", "--with-sequential"]):
        runner.main()
        run.assert_called_once_with(Path("new"), .6, None)
        seq.assert_called_once_with(Path("new/sequential"), Path("new/training"), "cuda")


def test_optimizer_hook_records_only_successful_steps():
    node = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == "record_update")
    env = {"update_counter": {"updates": 0}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "actual_hook", "exec"), env)
    param = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([param])
    optimizer.register_step_post_hook(env["record_update"])
    param.grad = torch.ones_like(param)
    optimizer.step()
    assert env["update_counter"]["updates"] == 1
    scaler = torch.amp.GradScaler("cpu")
    optimizer.zero_grad()
    scaler.scale(param.square()).backward()
    scaler.step(optimizer)
    scaler.update()
    assert env["update_counter"]["updates"] == 2
    optimizer.zero_grad()
    scaler.scale(param.square()).backward()
    param.grad.fill_(float("inf"))
    scaler.step(optimizer)
    scaler.update()
    assert env["update_counter"]["updates"] == 2


def test_bad_reference_stops_before_training(tmp_path):
    with patch.object(runner, "verify", return_value=({"provenance": {"policy_sha256":"a"}}, None, None)), \
         patch.object(runner, "read", return_value={"provenance": {"policy_sha256":"b"}}), \
         patch.object(runner, "invoke") as invoke:
        with pytest.raises(ValueError): runner.run(tmp_path, .6, Path("reference"))
        invoke.assert_not_called()
