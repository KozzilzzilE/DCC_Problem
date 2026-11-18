"""CPU synthetic network only; no RoBERTa construction or weights loaded."""
import sys
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup
import pytest

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H))
from m4.staged import StageRecorder, set_stage, validate
spec = importlib.util.spec_from_file_location('train_t4', H/'scripts/train_prefix.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.embeddings = nn.Linear(4, 4)
        self.base_model.encoder = nn.Module()
        self.base_model.encoder.layer = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.dropout = nn.Dropout(.1)
        self.classifier = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Dropout(.1), nn.Linear(4, 9))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        x = self.base_model.embeddings(x)
        for layer in self.base_model.encoder.layer:
            x = self.dropout(layer(x))
        return self.classifier(x)


def test_real_stage_control_adamw_and_schedule():
    torch.manual_seed(42)
    m = Tiny().train()
    opt = torch.optim.AdamW(train.llrd_groups(m, 5e-5, .8, .01))
    sch = get_linear_schedule_with_warmup(opt, 985, 9855)
    control = StageRecorder(m, opt, sch)
    original = [p.detach().clone() for p in m.base_model.parameters()]
    control.enter(1, 0)
    assert all(not p.requires_grad for p in m.base_model.parameters())
    assert all(p.requires_grad for p in m.classifier.parameters())
    def step():
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.ones(9)*2)(m(torch.ones(3,4)), torch.ones(3,9))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        opt.step();sch.step();opt.zero_grad(set_to_none=True)
    step(); step()
    assert all(torch.equal(p, old) for p,old in zip(m.base_model.parameters(),original))
    assert all(p.grad is None and p not in opt.state for p in m.base_model.parameters())
    head_states = {id(p): (opt.state[p],opt.state[p]['exp_avg'].clone(),opt.state[p]['step'].clone()) for p in m.classifier.parameters()}
    # Advance scheduler to the actual boundary without simulating model forwards.
    for _ in range(3283): opt.step();sch.step()
    before = sch.state_dict().copy();lrs=[g['lr'] for g in opt.param_groups]
    state = control.enter(2,3285)
    assert state['backbone_unchanged_stage1'] and state['head_changed_stage1'] and state['backbone_state_absent']
    assert sch.state_dict()==before and [g['lr'] for g in opt.param_groups]==lrs
    assert sch.last_epoch==3285
    for p in m.classifier.parameters():
        old,avg,stepnum=head_states[id(p)]
        assert opt.state[p] is old and torch.equal(old['exp_avg'],avg) and torch.equal(old['step'],stepnum)
    assert all(p.requires_grad for p in m.parameters())
    assert all(mod.training for mod in m.modules())
    assert [mod.p for mod in m.modules() if isinstance(mod,nn.Dropout)]==[.1,.1]
    m(torch.ones(3,4)).sum().backward()
    assert all(p.grad is not None for p in m.parameters())
    opt.step();sch.step();opt.zero_grad(set_to_none=True)
    assert all(p in opt.state for p in m.base_model.parameters())
    assert all(opt.state[p]['step']==1 for p in m.base_model.parameters())
    assert any(not torch.equal(p,old) for p,old in zip(m.base_model.parameters(),original))
    control.enter(3,6570)
    assert control.finish()['backbone_changed_final']
    assert m.calls==3


def test_default_does_not_freeze_or_change_groups():
    m=Tiny()
    before=[p.requires_grad for p in m.parameters()]
    validate(SimpleNamespace(freeze_backbone_epochs=0))
    gs=train.llrd_groups(m,5e-5,.8,.01)
    assert before==[p.requires_grad for p in m.parameters()]
    assert len([p for g in gs for p in g['params']])==len(list(m.parameters()))
    assert len({id(p) for g in gs for p in g['params']})==len(list(m.parameters()))


def test_boundary_and_recipe_rejection():
    m=Tiny()
    assert [set_stage(m,e,1)['backbone_frozen'] for e in [1,2,3]]==[True,False,False]
    a=SimpleNamespace(freeze_backbone_epochs=1, epochs=3,samples_per_call=2,include_full=False,batch_size=16,
       grad_accum=1,max_length=512,fmt='space',truncation_side='right',lr=5e-5,llrd=.8,pos_weight_power=.5,
       seed=42,count_classes=0,max_steps=0,grad_ckpt=False)
    validate(a)
    a.grad_ckpt=True
    with pytest.raises(ValueError):validate(a)


def test_training_main_synthetic_cpu_receipts(monkeypatch,tmp_path):
    import contextlib,json
    m=Tiny()
    raw_forward=m.forward
    m.forward=lambda input_ids,attention_mask:SimpleNamespace(logits=raw_forward(input_ids))
    m.save_pretrained=lambda *a,**k:None
    class Batch(dict):
        def to(self,device):return self
    class Tokenizer:
        def __call__(self,texts,**kw):return Batch(input_ids=torch.ones(len(texts),4),attention_mask=torch.ones(len(texts),4))
        def save_pretrained(self,*a):pass
    class TorchCPU:
        cuda=SimpleNamespace(manual_seed_all=lambda *a:None,reset_peak_memory_stats=lambda:None,
                             max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)
        amp=SimpleNamespace(GradScaler=lambda _:torch.amp.GradScaler('cpu'))
        def __getattr__(self,k):return getattr(torch,k)
        def device(self,*a):return torch.device('cpu')
        def autocast(self,*a,**kw):return contextlib.nullcontext()
    monkeypatch.setattr(train,'torch',TorchCPU())
    monkeypatch.setattr(train,'validate_inputs',lambda *a:None)
    monkeypatch.setattr(train,'provenance',lambda *a:{'synthetic':True})
    monkeypatch.setattr(train,'AutoTokenizer',SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()))
    monkeypatch.setattr(train,'AutoModelForSequenceClassification',SimpleNamespace(from_pretrained=lambda *a,**k:m))
    monkeypatch.setattr(train,'load_cache',lambda *a:list(range(16)))
    monkeypatch.setattr(train,'read_ids',lambda *a:[])
    monkeypatch.setattr(train,'label_matrix',lambda *a:__import__('numpy').ones((16,9)))
    monkeypatch.setattr(train,'prefix_texts',lambda *a:['a','b'])
    monkeypatch.setattr(sys,'argv',['train','--calls','dummy','--ids','fit','--backbone','dummy','--out',str(tmp_path/'new'),'--freeze-backbone-epochs','1'])
    assert train.main()==0
    q=json.loads((tmp_path/'new/run_config.json').read_text())
    assert [h['stage'] for h in q['history']]==['head_only','full','full']
    assert [h['global_step'] for h in q['history']]==[0,2,4]
    assert q['original_sample_exposures']==96 and q['backbone_sample_exposures']==64
    assert q['optimizer_updates']==6 and q['amp_skips']==0 and m.calls==6
    assert q['history'][0]['backbone_optimizer_state_parameters']==0
    assert q['history'][1]['backbone_optimizer_state_parameters']>0
    assert q['staged_verification']['transitions'][1]['backbone_unchanged_stage1']
    assert q['staged_verification']['backbone_changed_final']


def test_runner_dispatch_cpu_mock(monkeypatch,tmp_path):
    spec=importlib.util.spec_from_file_location('runner_t4',H/'scripts/prefix_experiment.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    calls=[]
    monkeypatch.setattr(runner,'run',lambda *a:calls.append(('run',a)))
    monkeypatch.setattr(runner,'sequential',lambda *a:calls.append(('sequential',a)))
    monkeypatch.setattr(sys,'argv',['runner','run','--out',str(tmp_path/'t4'),'--freeze-backbone-epochs','1','--with-sequential'])
    monkeypatch.chdir(H)
    runner.main()
    assert calls[0][1][1]==1 and calls[1][0]=='sequential'
