"""T8 CPU 수식·기울기·수치 안정성·기존 경로 검증."""
import ast
import math
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m4.losses import AsymmetricLoss, make_loss


@pytest.mark.parametrize("p", [.01, .10, .50, .90, .99])
@pytest.mark.parametrize("y", [0., 1.])
def test_representative_loss_and_detached_logit_gradient(p, y):
    x = torch.tensor([[math.log(p / (1 - p))]], requires_grad=True)
    loss = AsymmetricLoss()(x, torch.full_like(x, y))
    loss.backward()
    q = min(1., 1 - p + .05)
    expected_loss = -math.log(p) if y else -(1 - q) ** 4 * math.log(q)
    expected_gradient = p - 1 if y else ((1 - q) ** 4 * p * (1 - p) / q if p > .05 else 0.)
    assert loss.item() == pytest.approx(expected_loss, rel=2e-5, abs=1e-8)
    assert x.grad.item() == pytest.approx(expected_gradient, rel=2e-5, abs=1e-8)


def test_positive_equals_unweighted_bce_and_mean_over_batch_times_nine():
    x = torch.linspace(-8, 8, 27).reshape(3, 9).requires_grad_()
    y = torch.ones_like(x)
    actual = AsymmetricLoss()(x, y)
    expected = F.binary_cross_entropy_with_logits(x, y)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0])


@pytest.mark.parametrize("p", [.001, .01, .049, .05])
def test_clipped_easy_negatives_have_zero_loss_and_gradient(p):
    x = torch.tensor(math.log(p / (1 - p)), requires_grad=True)
    loss = AsymmetricLoss()(x, torch.zeros_like(x))
    loss.backward()
    assert loss.item() == 0
    assert x.grad.item() == 0


def test_focusing_is_detached_instead_of_differentiated():
    x = torch.tensor(.7, requires_grad=True)
    actual = AsymmetricLoss()(x, torch.zeros_like(x))
    actual_gradient = torch.autograd.grad(actual, x)[0]
    p = x.sigmoid()
    q = 1 - p + .05
    differentiated = -(1 - q).pow(4) * q.log()
    full_gradient = torch.autograd.grad(differentiated, x)[0]
    assert not torch.isclose(actual_gradient, full_gradient)
    assert torch.is_grad_enabled()
    with torch.no_grad():
        assert not AsymmetricLoss()(x, torch.zeros_like(x)).requires_grad
        assert not torch.is_grad_enabled()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_extreme_logits_and_cpu_amp_mixed_multilabel_batch(dtype):
    x = torch.tensor([[-10000, -100, -20, -1, 0, 1, 20, 100, 10000]] * 2,
                     dtype=dtype, requires_grad=True)
    y = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = AsymmetricLoss()(x, y)
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_zero_gamma_no_clip_equals_bce_for_mixed_labels():
    x = torch.linspace(-8, 8, 36).reshape(4, 9).requires_grad_()
    y = (torch.arange(36).reshape(4, 9) % 3 == 0).float()
    actual = AsymmetricLoss(gamma_pos=0, gamma_neg=0, clip=0)(x, y)
    expected = F.binary_cross_entropy_with_logits(x, y)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0],
                               rtol=2e-5, atol=2e-6)


def test_mixed_mean_and_gradient_accumulation():
    x = torch.linspace(-3, 3, 36).reshape(4, 9).requires_grad_()
    y = (torch.arange(36).reshape(4, 9) % 2).float()
    fn = AsymmetricLoss()
    full = fn(x, y)
    individual = torch.stack([fn(x[i, j], y[i, j]) for i in range(4) for j in range(9)]).mean()
    torch.testing.assert_close(full, individual)
    expected = torch.autograd.grad(full, x)[0]
    for a, b in [(0, 2), (2, 4)]:
        (fn(x[a:b], y[a:b]) / 2).backward()
    torch.testing.assert_close(x.grad, expected)


def test_factory_preserves_weighted_bce_and_ignores_weight_only_for_asl():
    weight = torch.arange(1., 10.)
    x, y = torch.zeros(2, 9), torch.ones(2, 9)
    bce = make_loss("weighted-bce", weight)
    assert isinstance(bce, torch.nn.BCEWithLogitsLoss)
    torch.testing.assert_close(bce(x, y), F.binary_cross_entropy_with_logits(x, y, pos_weight=weight))
    torch.testing.assert_close(make_loss("asl", weight)(x, y), make_loss("asl", weight * 100)(x, y))
    with pytest.raises(ValueError):
        make_loss("unknown", weight)


def test_cli_default_remains_weighted_bce():
    source = (Path(__file__).resolve().parents[1] / "scripts/train_prefix.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    loss_arg = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                    and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--loss")
    assert next(k.value.value for k in loss_arg.keywords if k.arg == "default") == "weighted-bce"
