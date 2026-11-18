"""모델 실행 없이 T9 가중치·Sampling·불변 소스 및 보호 해시를 검증한다."""
import argparse
import ast
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import prefix_experiment as runner
from m4.data import load_cache,label_matrix,read_ids

ROOT=runner.HERE.parent
GIT=['git','-c','safe.directory='+ROOT.as_posix()]
def git(*args):return subprocess.check_output(GIT+list(args),cwd=ROOT).decode('utf-8')
def sampler(source):
    main=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in main.body if isinstance(n,ast.For) and ast.unparse(n.target)=='epoch');nodes=[]
    for node in loop.body:
        nodes.append(node)
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='order':break
    return compile(ast.Module(body=nodes,type_ignores=[]),'actual_sampling','exec')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert git('branch','--show-current').strip()=='mission4-prefix-t9-same-prefix-consistency'
    assert git('rev-parse','HEAD')==git('rev-parse','mission4-prefix')
    out=runner.reserve(args.out)
    protected=[p for d in [ROOT/'outputs/mission4_prefix/localcontrol',ROOT/'outputs/mission4_prefix/localsequential',runner.E002] for p in d.rglob('*') if p.is_file()]
    protected += [*runner.INPUT_SHA256,runner.POLICY,*runner.FIXED,*(runner.BACKBONE/p for p in runner.BACKBONE_SHA256),runner.MODEL1/'model.safetensors',runner.MODEL7/'model.safetensors']
    hashes={str(p):runner.digest(p) for p in protected}
    checks={str(p):hashes[str(p)]==v for p,v in runner.INPUT_SHA256.items()}
    checks.update({'TAPT_'+p:hashes[str(runner.BACKBONE/p)]==h for p,h in runner.BACKBONE_SHA256.items()})
    checks['policy']=hashes[str(runner.POLICY)]==runner.POLICY_SHA256
    checks['fixed_tables']=all(hashes[str(p)]==h for p,h in zip(runner.FIXED,runner.FIXED_SHA256))
    seq=runner.read(ROOT/'outputs/mission4_prefix/localsequential/sequential_timing.json')
    checks['fixed_models']=all(hashes[str(p/'model.safetensors')]==seq['model_hashes'][str(p)] for p in [runner.MODEL1,runner.MODEL7])
    cfg=runner.read(ROOT/'outputs/mission4_prefix/localcontrol/training/run_config.json')
    checks['local_recipe']=all(cfg[k]==v for k,v in dict(runner.RECIPE,grad_accum=1,grad_ckpt=False,count_classes=0,max_steps=0).items())
    for rel in ['m4/data.py','m4/decision.py','m4/halting.py','m4/streaming.py','m4/bundle.py','m4/simulate.py','scripts/pack_bundle.py','scripts/prefix_probs.py','inference.py','score.py']:
        checks[rel]=(runner.HERE/rel).read_text(encoding='utf-8')==git('show','mission4-prefix:mission4_symptom/'+rel)
    old=git('show','mission4-prefix:mission4_symptom/scripts/train_prefix.py');new=(runner.HERE/'scripts/train_prefix.py').read_text(encoding='utf-8')
    nodes={ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(new))}
    for prefix in ['random.seed','np.random.seed','torch.manual_seed','torch.cuda.manual_seed_all','calls =','labels =','prefixes =','pos =','pos_weight =','tokenizer =','tokenizer.truncation_side','model =','model.to','enc =','target =','optimizer =','scheduler =','scaler =','loss_fn =','scaler.scale','scaler.unscale_','torch.nn.utils.clip_grad_norm_','scaler.step','scaler.update','scheduler.step','optimizer.zero_grad']:
        base=[n for n in ast.walk(ast.parse(old)) if isinstance(n,(ast.Assign,ast.Expr)) and ast.unparse(n).startswith(prefix)]
        checks[prefix]=bool(base) and all(ast.dump(n,include_attributes=False) in nodes for n in base)
    oldrunner=ast.parse(git('show','mission4-prefix:mission4_symptom/scripts/prefix_experiment.py'));newrunner=ast.parse((runner.HERE/'scripts/prefix_experiment.py').read_text(encoding='utf-8'))
    for name in ['evaluate','write_evaluation','sequential','prepare','aligned']:
        get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        checks[name+'_unchanged']=ast.dump(get(oldrunner),include_attributes=False)==ast.dump(get(newrunner),include_attributes=False)
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    calls=load_cache(runner.CALLS,read_ids(runner.FIT));labels=label_matrix(calls).astype(np.float32);assert len(calls)==26280
    assert len(set(read_ids(runner.DEV)))==2920 and not set(read_ids(runner.FIT))&set(read_ids(runner.DEV))
    env=dict(np=np,labels=labels,args=SimpleNamespace(pos_weight_power=.5))
    weight_nodes=[n for n in ast.walk(ast.parse(new)) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['pos','pos_weight']]
    exec(compile(ast.Module(body=weight_nodes,type_ignores=[]),'actual_weights','exec'),env)
    np.testing.assert_array_equal(env['pos_weight'],np.array(cfg['pos_weight'],dtype=np.float32))
    history=[];legacy=runner.read(ROOT/'outputs/mission4_prefix/t8_cpu_verified/cpu_verification.json')
    for epoch in [1,2,3]:
        arrays=[]
        for source in [old,new]:
            envs=dict(np=np,args=SimpleNamespace(seed=42,samples_per_call=2,include_full=False),epoch=epoch,prefixes=[range(c.n_utt) for c in calls])
            exec(sampler(source),envs);arrays.append(np.asarray(envs['items'],dtype='<i8')[envs['order']])
        np.testing.assert_array_equal(*arrays);np.testing.assert_array_equal(np.bincount(arrays[0][:,0]),np.full(len(calls),2))
        sha=hashlib.sha256(arrays[0].tobytes()).hexdigest();assert sha==legacy['epochs'][epoch-1]['ordered_items_sha256']
        history.append({'epoch':epoch,'samples':len(arrays[0]),'views':2*len(arrays[0]),'ordered_items_sha256':sha,'positive_label_exposures':labels[arrays[0][:,0]].sum(0).astype(int).tolist()})
    assert sum(x['samples'] for x in history)==157680 and sum(x['views'] for x in history)==315360 and 157680//16==9855
    assert all(runner.digest(Path(p))==h for p,h in hashes.items())
    runner.save(out/'cpu_verification.json',{'passed':True,'checks':checks,'sampling_history':history,'pos_weight':env['pos_weight'].tolist(),
        'original_exposures':157680,'view_exposures':315360,'planned_optimizer_steps':9855,'baseline_commit':git('rev-parse','mission4-prefix').strip(),
        'not_run':['실제 모델 초기화/추론','CUDA AMP','GPU 메모리/시간','전체 Tokenization 재실행','실제 Bundle 생성']})
    runner.save(out/'protected_files.json',{'sha256':hashes,'unchanged':True})
    print('T9 CPU 사전 검증 통과: 실제 가중치·전체 Sampling·불변 코드 및 보호 해시')
if __name__=='__main__':main()
