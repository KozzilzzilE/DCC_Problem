"""감사용 검산: 정책 logit_shift, summary fit_only_shift, 각 멤버 run_config pos_weight 의 log 값을 비교해
m4/bundle.py learned_halt_setting()/shared_pos_weight() 의 tol=1e-5 검사를 통과하는지 본다. 읽기 전용."""
import json, math, subprocess
ROOT = "/Users/sunggyu/Desktop/이성규/DCC"
EXT = "/Volumes/로컬 디스크/DCC/private_experiments"
REPO = f"{ROOT}/github_review/DCC_Problem"
pol = json.load(open(f"{ROOT}/artifacts_private/runs/e026_fitonly_policy_v1_20261117/policy_fitonly_c128.json"))
summ = json.load(open(f"{ROOT}/artifacts_private/runs/e026_fitonly_policy_v1_20261117/summary.json"))
ps = pol["logit_shift"]; fs = summ["fit_only_shift"]
print("policy type/beta/tau/min_step:", pol["type"], pol["beta"], pol["tau"], pol["min_step"], "n_fit_calls", pol.get("n_fit_calls"))
print("policy logit_shift == summary fit_only_shift max|diff| =", max(abs(a-b) for a,b in zip(ps,fs)))
def git_json(path):
    out = subprocess.run(["git","-C",REPO,"show",f"origin/mission4-streaming:mission4_symptom/{path}"],capture_output=True,text=True)
    return json.loads(out.stdout) if out.returncode==0 else None
srcs = {
 "E002(git runs/E002_prefix_uniform)": git_json("runs/E002_prefix_uniform/run_config.json"),
 "E007(git runs/E007_prefix_uniform_s43)": git_json("runs/E007_prefix_uniform_s43/run_config.json"),
 "E016_final/shift(git)": git_json("runs/E016_final/shift/run_config.json"),
 "final/prefix_s42(git, 29,200)": git_json("runs/final/prefix_s42/run_config.json"),
 "final/prefix_s43(git, 29,200)": git_json("runs/final/prefix_s43/run_config.json"),
 "E001 inference_config(외장)": json.load(open(f"{EXT}/e026_priority1_checks_20261116/models/E001/inference_config.json")),
 "SJ42": json.load(open(f"{EXT}/m4_softjaccard_v1_20261116/s42/light/run_config.json")),
 "SJ43": json.load(open(f"{EXT}/m4_softjaccard_v1_20261116/s43/light/run_config.json")),
 "JM42": json.load(open(f"{EXT}/m4_jacc_macrof1_v1_20261116/s42/light/run_config.json")),
 "JM43": json.load(open(f"{EXT}/m4_jacc_macrof1_v1_20261116/s43/light/run_config.json")),
 "e002_oof fold0": json.load(open(f"{EXT}/e002_oof_20261117/fold0/run_config.json")),
}
CLS = ["고열","구토","두통","복통","어지러움","열상","오심","전신쇠약","호흡곤란"]
def pw(cfg):
    if cfg is None: return None
    w = cfg.get("pos_weight")
    if w is None:
        st = cfg.get("train_pos_weight_statistics")
        if isinstance(st, dict) and all(c in st for c in CLS): w = [st[c]["pos_weight"] for c in CLS]
    return None if w is None else [float(x) for x in w]
for name, cfg in srcs.items():
    w = pw(cfg)
    keys = {k: cfg.get(k) for k in ("fmt","max_length","truncation_side","pos_weight_power","ids","forbid_ids","calls","n_train_calls","epochs","seed") if cfg and k in cfg}
    if w is None:
        print(f"[{name}] pos_weight 기록 없음 (member_pos_weight -> None). keys={keys}")
        continue
    ls = [math.log(x) for x in w]
    g_pol = max(abs(a-b) for a,b in zip(ls, ps))
    print(f"[{name}] max|log(pw)-policy_shift|={g_pol:.3e} -> tol 1e-5 {'통과' if g_pol<=1e-5 else '불통과'} | {keys}")
    print("     pos_weight=", [round(x,6) for x in w])
print("policy shift exp =", [round(math.exp(x),6) for x in ps])
