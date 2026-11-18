"""T4 training-only state control. No inference dependency."""
import hashlib
import subprocess
from pathlib import Path


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def provenance(root, backbone):
    cmd = ['git', '-c', 'safe.directory=' + root.as_posix(), '-C', str(root)]
    return dict(branch=subprocess.check_output(cmd + ['branch', '--show-current']).decode().strip(),
                commit=subprocess.check_output(cmd + ['rev-parse', 'HEAD']).decode().strip(),
                initial_weights_sha256=file_hash(Path(backbone) / 'model.safetensors'))


def validate(args):
    if not args.freeze_backbone_epochs:
        return
    expected = dict(epochs=3, samples_per_call=2, include_full=False, batch_size=16,
                    grad_accum=1, max_length=512, fmt='space', truncation_side='right',
                    lr=5e-5, llrd=.8, pos_weight_power=.5, seed=42, count_classes=0,
                    max_steps=0, grad_ckpt=False)
    changed = [k for k, v in expected.items() if getattr(args, k) != v]
    if changed:
        raise ValueError('T4 fixed recipe mismatch: ' + ', '.join(changed))


def validate_inputs(args, root):
    """T4 direct CLI also refuses Dev/all-data or a different backbone."""
    if not args.freeze_backbone_epochs:
        return
    import json
    control = root / 'outputs/mission4_prefix/localcontrol'
    cfg = json.loads((control / 'training/run_config.json').read_text(encoding='utf-8'))
    receipt = json.loads((control / 'initial.json').read_text(encoding='utf-8'))
    for key in ['calls', 'ids', 'backbone']:
        if getattr(args, key) is None or Path(getattr(args, key)).resolve() != Path(cfg[key]).resolve():
            raise ValueError('T4 requires original Fit inputs: ' + key)
    for path, key in [(Path(args.ids), 'fit_ids_sha256'),
                      (Path(args.backbone)/'model.safetensors', 'initial_weights_sha256'),
                      (Path(args.backbone)/'tokenizer.json', 'initial_tokenizer_sha256')]:
        if file_hash(path) != receipt[key]:
            raise ValueError('T4 protected input changed: ' + str(path))


def groups(model):
    backbone = list(model.base_model.parameters())
    head = list(model.classifier.parameters())
    ids = [id(p) for p in backbone + head]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}:
        raise ValueError('T4 requires disjoint, exhaustive base_model/classifier boundary')
    return backbone, head


def parameter_hash(params):
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_stage(model, epoch, freeze_epochs):
    backbone, head = groups(model)
    frozen = epoch <= freeze_epochs
    for p in backbone:
        p.requires_grad_(not frozen)
        if frozen:
            p.grad = None
    return dict(stage='head_only' if frozen else 'full', backbone_frozen=frozen,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad))


class StageRecorder:
    """Construct AFTER registering every baseline optimizer group, BEFORE epoch 1."""
    def __init__(self, model, optimizer, scheduler):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.optimizer_id, self.scheduler_id = id(optimizer), id(scheduler)
        self.backbone, self.head = groups(model)
        registered = [p for g in optimizer.param_groups for p in g['params']]
        assert len(registered) == len({id(p) for p in registered})
        assert {id(p) for p in registered} == {id(p) for p in model.parameters()}
        self.initial_backbone = parameter_hash(self.backbone)
        self.initial_head = parameter_hash(self.head)
        self.transitions = []

    def enter(self, epoch, step):
        head_states = {id(p): self.optimizer.state.get(p) for p in self.head}
        scheduler_epoch = self.scheduler.last_epoch
        lrs = [g['lr'] for g in self.optimizer.param_groups]
        state = set_stage(self.model, epoch, 1)
        state.update(epoch=epoch, global_step=step, scheduler_step=scheduler_epoch,
                     optimizer_identity_preserved=id(self.optimizer) == self.optimizer_id,
                     head_optimizer_state_preserved=all(self.optimizer.state.get(p) is head_states[id(p)] for p in self.head),
                     scheduler_preserved=id(self.scheduler) == self.scheduler_id and self.scheduler.last_epoch == scheduler_epoch,
                     learning_rates=lrs)
        if epoch == 2:
            state.update(backbone_unchanged_stage1=parameter_hash(self.backbone) == self.initial_backbone,
                         head_changed_stage1=parameter_hash(self.head) != self.initial_head,
                         backbone_state_absent=all(p not in self.optimizer.state for p in self.backbone))
            if not state['backbone_unchanged_stage1'] or not state['backbone_state_absent']:
                raise RuntimeError('Frozen backbone changed or acquired optimizer state')
        self.transitions.append(state)
        return state

    def finish(self):
        return dict(transitions=self.transitions,
                    backbone_changed_final=parameter_hash(self.backbone) != self.initial_backbone,
                    head_changed_final=parameter_hash(self.head) != self.initial_head,
                    backbone_state_parameters=sum(p in self.optimizer.state for p in self.backbone),
                    total_parameters=sum(p.numel() for p in self.model.parameters()),
                    backbone_parameters=sum(p.numel() for p in self.backbone),
                    head_parameters=sum(p.numel() for p in self.head))
