"""T9 수식과 실제 목적함수 호출을 CPU Dummy 모델에서 검증한다."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from m4.consistency import bernoulli_symmetric_kl as kl, prefix_objective, validate_t9


def reference(a, b):
    p, q = a.sigmoid(), b.sigmoid()
    return .5 * (p*(F.logsigmoid(a)-F.logsigmoid(b))+(1-p)*(F.logsigmoid(-a)-F.logsigmoid(-b))
                 +q*(F.logsigmoid(b)-F.logsigmoid(a))+(1-q)*(F.logsigmoid(-b)-F.logsigmoid(-a))).mean()


def test_formula_gradients_and_mean():
    a = torch.linspace(-4, 4, 27).reshape(3, 9).requires_grad_()
    b = torch.flip(a.detach(), [1]).requires_grad_()
    torch.testing.assert_close(kl(a,b), reference(a,b))
    ga = torch.autograd.grad(kl(a,b),(a,b),retain_graph=True)
    gr = torch.autograd.grad(reference(a,b),(a,b))
    for x,y in zip(ga,gr):
        torch.testing.assert_close(x,y)
        assert x.abs().sum() > 0
    assert kl(a,a).item() == 0
    torch.testing.assert_close(kl(a,b),kl(b,a))
    torch.testing.assert_close(kl(a,b),kl(a.repeat(2,1),b.repeat(2,1)))


def test_production_fp32_gradient_finite_difference():
    a=torch.tensor([[.2]*9],requires_grad=True);b=torch.tensor([[-.3]*9],requires_grad=True)
    g=torch.autograd.grad(kl(a,b),a)[0][0,0].item()
    plus=a.detach().clone();minus=plus.clone();plus[0,0]+=.001;minus[0,0]-=.001
    numerical=((kl(plus,b)-kl(minus,b))/.002).item()
    assert g == pytest.approx(numerical,rel=.002,abs=1e-5)
    gaps=[kl(torch.full((2,9),d/2),torch.full((2,9),-d/2)).item() for d in [.1,.5,1.,2.]]
    assert gaps == sorted(gaps) and gaps[0] > 0


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32])
def test_extreme_logits_and_fp32_under_cpu_amp(dtype):
    a=torch.tensor([[-10000,-100,-20,-1,0,1,20,100,10000]],dtype=dtype,requires_grad=True)
    b=(-a.detach()).requires_grad_()
    with torch.autocast('cpu',dtype=torch.bfloat16):loss=kl(a,b)
    loss.backward()
    assert loss.dtype==torch.float32 and torch.isfinite(loss) and torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


class Dummy(torch.nn.Module):
    def __init__(self, cp=False):
        super().__init__();self.fc=torch.nn.Linear(9,9);self.cp=cp;self.inputs=[];self.masks=[]
    def block(self,x):
        dropped=F.dropout(x,p=.3,training=self.training)
        self.masks.append(dropped.detach().clone())
        return torch.tanh(self.fc(dropped))
    def forward(self,input_ids,attention_mask):
        self.inputs.append((input_ids,attention_mask))
        out=checkpoint(self.block,input_ids,use_reentrant=False,preserve_rng_state=True) if self.cp else self.block(input_ids)
        return SimpleNamespace(logits=out)


class SpyBCE(torch.nn.BCEWithLogitsLoss):
    def __init__(self):super().__init__(pos_weight=torch.linspace(1.8,3,9));self.targets=[]
    def forward(self,x,y):self.targets.append(y);return super().forward(x,y)


@pytest.mark.parametrize("lam",[0.,1.])
def test_forward_count_inputs_labels_and_single_update(lam):
    torch.manual_seed(42);model=Dummy();model.train();enc={'input_ids':torch.ones(16,9),'attention_mask':torch.ones(16,9)};target=torch.ones(16,9);fn=SpyBCE()
    opt=torch.optim.AdamW(model.parameters());updates=[];opt.register_step_post_hook(lambda *args:updates.append(1))
    loss,sup,raw,z=prefix_objective(model,enc,target,fn,lam)
    assert len(model.inputs)==len(fn.targets)==(2 if lam else 1)
    assert all(x is enc['input_ids'] and y is enc['attention_mask'] for x,y in model.inputs)
    assert all(y is target for y in fn.targets)
    if lam:assert not torch.equal(model.masks[0],model.masks[1])
    else:
        torch.testing.assert_close(loss,F.binary_cross_entropy_with_logits(z,target,pos_weight=fn.pos_weight))
        assert raw.item()==0
    torch.testing.assert_close(loss,sup+lam*raw)
    scaler=torch.amp.GradScaler('cpu');scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update()
    assert len(updates)==1 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_checkpoint_recompute_preserves_each_views_rng_and_gradient():
    torch.manual_seed(13);plain=Dummy(False);cp=Dummy(True);cp.load_state_dict(plain.state_dict())
    enc={'input_ids':torch.ones(16,9),'attention_mask':torch.ones(16,9)};target=torch.ones(16,9);fn=SpyBCE()
    torch.manual_seed(99);l0,*_=prefix_objective(plain,enc,target,fn,1);l0.backward()
    torch.manual_seed(99);l1,*_=prefix_objective(cp,enc,target,fn,1);assert len(cp.inputs)==2;l1.backward()
    torch.testing.assert_close(l0,l1)
    for p,q in zip(plain.parameters(),cp.parameters()):torch.testing.assert_close(p.grad,q.grad)
    assert len(cp.masks)==4
    torch.testing.assert_close(cp.masks[0],cp.masks[3]);torch.testing.assert_close(cp.masks[1],cp.masks[2])
    assert not torch.equal(cp.masks[0],cp.masks[1])
    assert plain.state_dict().keys()==cp.state_dict().keys()


def test_gradient_accumulation_applied_once():
    a=torch.zeros(4,9,requires_grad=True);y=torch.ones_like(a);fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(9)*2)
    baseline=fn(a,y);g=torch.autograd.grad(baseline,a)[0]
    for chunk in [a[:2],a[2:]]:
        loss=.5*(fn(chunk,y[:2])+fn(chunk,y[:2]))+kl(chunk,chunk)
        (loss/2).backward()
    torch.testing.assert_close(a.grad,g)


def test_fixed_settings_reject_silent_batch_change():
    validate_t9(SimpleNamespace(consistency_lambda=0.))
    with pytest.raises(ValueError):validate_t9(SimpleNamespace(consistency_lambda=1.0,pos_weight_power=.6))


@pytest.mark.parametrize('enabled,lam,kwargs',[(False,0.,None),(True,0.,{}),(True,1.,{'gradient_checkpointing_kwargs':{'use_reentrant':False,'preserve_rng_state':True}})])
def test_actual_training_checkpoint_branch(enabled,lam,kwargs):
    tree=ast.parse((HERE/'scripts/train_prefix.py').read_text(encoding='utf-8'))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='args.grad_ckpt')
    from unittest.mock import Mock
    model=Mock();env=dict(model=model,args=SimpleNamespace(grad_ckpt=enabled,consistency_lambda=lam))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_checkpoint_branch','exec'),env)
    if kwargs is None:model.gradient_checkpointing_enable.assert_not_called()
    else:model.gradient_checkpointing_enable.assert_called_once_with(**kwargs)


spec=importlib.util.spec_from_file_location('t9_runner',HERE/'scripts/prefix_experiment.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


@pytest.mark.parametrize('active',[False,True])
def test_runner_forwards_only_explicit_options_and_preserves_recipe(tmp_path,active):
    hashes={str(runner.BACKBONE/p):h for p,h in runner.BACKBONE_SHA256.items()};recipe=dict(runner.RECIPE)
    with patch.object(runner,'verify',return_value=({},None,None)),patch.object(runner,'read',return_value=recipe),patch.object(runner,'digest',side_effect=lambda p:hashes.get(str(p),'test')),patch.object(runner,'reserve',return_value=tmp_path),patch.object(runner,'save'),patch.object(runner,'invoke') as invoke,patch.object(runner,'write_evaluation'):
        runner.run(tmp_path,1. if active else 0.,active)
        _,args,_=invoke.call_args_list[0].args
        assert ('--consistency-lambda' in args)==active and ('--gradient-checkpointing' in args)==active
        if active:assert args[args.index('--consistency-lambda')+1]=='1.0'
        assert args[args.index('--pos-weight-power')+1]=='0.5'
    assert runner.RECIPE==recipe


def test_runner_combines_existing_sequential():
    with patch.object(runner.os,'chdir'),patch.object(runner,'run') as run,patch.object(runner,'sequential') as seq,patch('sys.argv',['runner','run','--out','new','--consistency-lambda','1.0','--gradient-checkpointing','--with-sequential']):
        runner.main();run.assert_called_once_with(Path('new'),1.,True,None);seq.assert_called_once_with(Path('new/sequential'),Path('new/training'),'cuda')
