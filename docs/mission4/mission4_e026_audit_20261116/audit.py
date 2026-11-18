#!/usr/bin/env python3
"""Read-only consistency checks of cached E026 Git blobs; no model inference."""
import hashlib, json, math, subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[2]
repo = root / 'github_review/DCC_Problem'
commit = subprocess.check_output(['git','rev-parse','5d7539b^{commit}'], cwd=repo, text=True).strip()
base = 'mission4_symptom/runs/E026_learned_halt05/wide/'
def read(name):
    path = base + name
    raw = subprocess.check_output(['git','show',f'{commit}:{path}'], cwd=repo)
    blob = subprocess.check_output(['git','rev-parse',f'{commit}:{path}'], cwd=repo, text=True).strip()
    return json.loads(raw), {'path':path, 'git_blob':blob, 'sha256':hashlib.sha256(raw).hexdigest()}
p, ps = read('policy.json'); r, rs = read('result.json')
run = r['runs'][0]; args = r['args']; scores = run['scores']
products = {k:{'reported':v['score'], 'calculated':v['macro_f1']*v['earliness'], 'residual':v['score']-v['macro_f1']*v['earliness']} for k,v in scores.items()}
diff = scores['learned']['score'] - scores['base']['score']; comparison = run['learned_vs_base']
checks = {
 'policy_beta_matches_result': p['beta'] == run['beta'] == 128.0,
 'policy_tau_matches_result': p['tau'] == run['tau'] == args['base_tau'] == args['ref_tau'] == 0.5,
 'policy_min_step': p['min_step'] == 5,
 'grid_contains_selection': any(x['beta']==p['beta'] and x['tau']==p['tau'] for x in p['info']['grid']),
 'fit_count_consistent': p['n_fit_calls'] == run['n_fit_calls'] == run['fit']['target']['n_calls'] == 26280,
 'eval_count_recorded': run['n_eval_calls'] == 2920,
 'product_matches': all(math.isclose(v['reported'],v['calculated'],abs_tol=1e-12) for v in products.values()),
 'diff_matches': math.isclose(diff,comparison['diff'],abs_tol=1e-12),
 'ci_ordered': comparison['lo'] <= comparison['diff'] <= comparison['hi'],
 'ci_includes_zero': comparison['lo'] <= 0 <= comparison['hi'],
 'settings': args['folds']==5 and args['seed']==0 and args['boot']==1000 and args['taus']==[0.5],
 'feature_names_match': p['feature_names']==r['feature_names'] and len(r['feature_names'])==r['n_features'],
}
out={'commit':commit,'sources':{'policy':ps,'result':rs},'checks':checks,'policy':{'beta':p['beta'],'tau':p['tau'],'min_step':p['min_step']},'settings':{'folds':args['folds'],'seed':args['seed'],'boot':args['boot'],'base_tau':args['base_tau'],'ref_tau':args['ref_tau'],'fit_tables':args['fit_tables'],'eval_tables':args['eval_tables']},'counts':{'fit_calls':run['n_fit_calls'],'fit_rows':run['fit']['target']['n_rows'],'eval_calls':run['n_eval_calls']},'products':products,'learned_vs_base':{'calculated_diff':diff,**comparison},'limitations':['No locally supplied per-call arrays, fit IDs, or OOF data were independently examined.','No independent rescore, fit-only retraining, leakage verification, or policy execution was performed.','Recorded CI method and data provenance were not independently verified.','This audit checks serialized values, not competition compliance or scientific adoption.']}
output=Path(__file__).with_name('audit.json'); output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
