"""4-2 학습형 중단 정책(기대 점수 + β·δ̂ ≥ τ)을 학습용 통화의 확률표로 학습하고, 판정용 확률표에서 채점한다.

    python scripts/train_halt_policy.py \
        --fit-tables runs/E011_oof/fold0/oof_prefix.npz ... runs/E011_oof/fold4/oof_prefix.npz \
        --eval-tables runs/E002_prefix_uniform/dev_prefix.npz \
        --shift-from runs/E002_prefix_uniform --base-tau 0.52 --out runs/E012_halt_policy/e002

절차 (정책의 설명과 수식은 m4/halting.py)
  1. 학습용 확률표로 목표 δ 를 만들고 LightGBM 을 학습한다.
  2. (β, τ) 는 학습용 통화 안에서 통화 단위 교차 학습 예측으로 고른다. 판정용 통화는 보지 않는다.
  3. 정책을 판정용 확률표에 그대로 적용해 4-2 점수를 낸다.
  4. 기준 정책(기대 점수 ≥ --base-tau)과의 차이를 통화 단위 짝 부트스트랩 95% 구간과 함께 낸다.
     "τ 만 학습용에서 다시 고른 정책"도 함께 낸다 (학습형의 이득이 τ 재선택 몫인지 가르는 기준).

확률표 여러 개를 줄 때
  --eval-tables : 같은 통화의 확률표들이어야 하고 확률을 평균한다 (모델 평균).
  --fit-tables  : 통화가 서로 겹치지 않으면 이어 붙이고(교차 학습 겹), 통화가 모두 같으면 평균한다.
학습용과 판정용 통화가 하나라도 겹치면 멈춘다.

리허설 (--rehearse-seed S)
  학습용 확률표가 아직 없을 때 쓴다. 판정용 확률표의 통화를 seed S 로 절반씩 나눠, 한쪽에서 학습·선택하고
  다른 쪽에서 채점한다. 방향을 바꿔 두 번 돌리고, 두 번의 판정 결과를 합친 점수도 낸다.
  --fit-tables 를 생략하면 판정용 확률표를 학습에도 쓴다 (통화는 절반으로 갈라져 겹치지 않는다).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4 import halting as hl  # noqa: E402
from m4 import simulate as sim  # noqa: E402
from m4.data import load_cache  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def merge_tables(paths, allow_concat: bool) -> sim.PrefixTable:
    """확률표 여러 개를 하나로. 통화가 모두 같으면 확률 평균, 서로 겹치지 않으면(allow_concat) 이어 붙인다."""
    tables = [sim.PrefixTable.load(p) for p in paths]
    first = tables[0]
    if all(t.call_ids == first.call_ids and np.array_equal(t.offsets, first.offsets) for t in tables[1:]):
        if len(tables) == 1:
            return first
        return sim.PrefixTable(first.call_ids, first.offsets, np.mean([t.probs for t in tables], axis=0).astype(np.float32), first.labels)
    ids = [i for t in tables for i in t.call_ids]
    if not allow_concat or len(set(ids)) != len(ids):
        raise SystemExit("확률표들의 통화가 같지도, 서로 겹치지 않지도 않습니다: " + ", ".join(map(str, paths)))
    offsets = np.concatenate([[0], np.cumsum(np.concatenate([t.n_utt for t in tables]))]).astype(np.int64)
    return sim.PrefixTable(ids, offsets, np.concatenate([t.probs for t in tables]), np.concatenate([t.labels for t in tables]))


def take_calls(table: sim.PrefixTable, keep: set) -> sim.PrefixTable:
    """keep 에 든 통화만 남긴 확률표 (순서 유지)."""
    idx = [i for i, c in enumerate(table.call_ids) if c in keep]
    n = table.n_utt[idx]
    rows = np.concatenate([np.arange(table.offsets[i], table.offsets[i + 1]) for i in idx])
    return sim.PrefixTable([table.call_ids[i] for i in idx], np.concatenate([[0], np.cumsum(n)]).astype(np.int64),
                           table.probs[rows], table.labels[idx])


def build(table: sim.PrefixTable, calls_by_id: dict, shift) -> hl.HaltData:
    return hl.HaltData.build(table, [calls_by_id[i] for i in table.call_ids], shift)


def run_once(fit: hl.HaltData, ev: hl.HaltData, args, shift, tag: str):
    """fit 으로 학습·선택하고 ev 에서 채점한다. (요약 dict, 정책, ev 의 통화별 중단 시점들)을 돌려준다."""
    overlap = set(fit.call_ids) & set(ev.call_ids)
    if overlap:
        raise SystemExit(f"학습용과 판정용 통화가 {len(overlap)}건 겹칩니다. 판정용 통화는 학습·선택에 쓰면 안 됩니다.")
    t0 = time.time()
    policy = hl.train_policy(fit, ref_tau=args.ref_tau, logit_shift=shift, folds=args.folds, seed=args.seed, threads=args.threads)
    info = policy.info
    print(f"[{tag}] 학습용 {len(fit)}통화 / 학습 행 {info['target']['n_rows']} ({time.time() - t0:.0f}초). "
          f"δ 교차 학습 R² {info['crossfit_r2']:.3f}, δ>0 비율 {info['target']['delta_pos_rate']:.3f}", flush=True)
    print(f"[{tag}] 학습용 안에서의 점수: 기준 τ={args.ref_tau:.2f} {info['fit_score_ref_tau']:.4f} / τ 만 선택(τ={info['fit_tau_only']:.2f}) "
          f"{info['fit_score_tau_only']:.4f} / 학습형(β={policy.beta:g}, τ={policy.tau:.2f}) {info['fit_score_selected']:.4f}", flush=True)

    halts = {"base": hl.first_halt(ev.value >= args.base_tau, ev),                 # 기준: τ 고정
             "tau_only": hl.first_halt(ev.value >= info["fit_tau_only"], ev),       # τ 만 학습용에서 고른 정책
             "learned": policy.halts(ev)}
    scores = {k: hl.score_halts(ev, h) for k, h in halts.items()}
    gain = booster_gain(policy)
    row = {"tag": tag, "n_fit_calls": len(fit), "n_eval_calls": len(ev), "beta": policy.beta, "tau": policy.tau,
           "tau_only": info["fit_tau_only"], "base_tau": args.base_tau, "ref_tau": args.ref_tau, "scores": scores,
           "learned_vs_base": hl.paired_bootstrap(ev, halts["learned"], halts["base"], args.boot),
           "learned_vs_tau_only": hl.paired_bootstrap(ev, halts["learned"], halts["tau_only"], args.boot),
           "tau_only_vs_base": hl.paired_bootstrap(ev, halts["tau_only"], halts["base"], args.boot),
           "fit": {k: v for k, v in info.items() if k != "grid"}, "top_features": gain[:12]}
    print_row(row)
    return row, policy, halts


def booster_gain(policy: hl.HaltPolicy):
    """특징 중요도 (분기 이득 합의 비율), 큰 순서."""
    g = policy.booster().feature_importance(importance_type="gain")
    total = max(float(g.sum()), 1e-12)
    order = np.argsort(-g)
    return [[policy.feature_names[i], round(float(g[i]) / total, 4)] for i in order]


def print_row(row) -> None:
    tag = row["tag"]
    one = row.get("beta") is not None                      # 합친 행은 방향마다 고른 값이 달라 값을 적지 않는다
    names = (("base", f"기준 τ={row['base_tau']:.2f}"),
             ("tau_only", f"τ 만 선택 τ={row['tau_only']:.2f}" if one else "τ 만 선택"),
             ("learned", f"학습형 β={row['beta']:g} τ={row['tau']:.2f}" if one else "학습형"))
    for key, name in names:
        s = row["scores"][key]
        print(f"[{tag}] 판정용 {name:<22} 4-2 {s['score']:.4f} = F1 {s['macro_f1']:.4f} x 조기 {s['earliness']:.4f}  "
              f"중앙 중단 {s['halt_median']:.0f}  끝까지 {s['never_halted']:.3f}", flush=True)
    for key, name in (("learned_vs_base", "학습형 − 기준"), ("learned_vs_tau_only", "학습형 − τ 만 선택"), ("tau_only_vs_base", "τ 만 선택 − 기준")):
        b = row[key]
        print(f"[{tag}]   {name:<16} {b['diff']:+.4f}  95% [{b['lo']:+.4f}, {b['hi']:+.4f}]  P(>0) {b['p_gt0']:.3f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-tables", nargs="+", default=None, help="정책 학습용 확률표 (겹치지 않는 통화면 이어 붙임, 같은 통화면 평균)")
    ap.add_argument("--eval-tables", nargs="+", required=True, help="판정용 확률표 (여러 개면 확률 평균)")
    ap.add_argument("--shift-from", required=True, help="pos_weight 를 읽을 run 폴더 (run_config.json)")
    ap.add_argument("--calls", default="runs/cache/train.jsonl", help="통화 원문 캐시 (Training 만. Validation 은 쓰지 않는다)")
    ap.add_argument("--out", required=True, help="결과 폴더 (result.json, policy.json)")
    ap.add_argument("--base-tau", type=float, default=0.52, help="판정에서 비교할 기준 정책의 τ (단일 0.52, 3개 평균 0.50)")
    ap.add_argument("--ref-tau", type=float, default=0.52, help="목표 δ 를 만들 때 학습용 확률표에 돌리는 기준 정책의 τ")
    ap.add_argument("--folds", type=int, default=5, help="(β, τ) 선택용 교차 학습 겹 수 (학습용 통화 안)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4, help="LightGBM CPU 스레드 수")
    ap.add_argument("--rehearse-seed", type=int, default=None, help="주면 판정용 통화를 절반씩 나눠 리허설한다")
    args = ap.parse_args()

    if Path(args.calls).name.startswith("val"):
        raise SystemExit("Validation 통화는 정책 학습·판정에 쓰지 않습니다.")
    if args.fit_tables is None and args.rehearse_seed is None:
        raise SystemExit("--fit-tables 가 필요합니다 (리허설이면 --rehearse-seed).")
    shift = np.log(json.loads((Path(args.shift_from) / "run_config.json").read_text(encoding="utf-8"))["pos_weight"])
    ev_table = merge_tables(args.eval_tables, allow_concat=False)
    fit_table = merge_tables(args.fit_tables, allow_concat=True) if args.fit_tables else ev_table
    need = set(ev_table.call_ids) | set(fit_table.call_ids)
    calls = {c.call_id: c for c in load_cache(Path(args.calls), need)}
    missing = need - set(calls)
    if missing:
        raise SystemExit(f"통화 원문에 없는 통화 {len(missing)}건 (예: {sorted(missing)[:3]})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"args": vars(args), "n_features": len(hl.FEATURE_NAMES), "feature_names": hl.FEATURE_NAMES, "runs": []}

    if args.rehearse_seed is None:
        t0 = time.time()
        fit, ev = build(fit_table, calls, shift), build(ev_table, calls, shift)
        print(f"특징 계산 {time.time() - t0:.0f}초 (학습용 {len(fit)}통화 {len(fit.X)}순간, 판정용 {len(ev)}통화 {len(ev.X)}순간)", flush=True)
        row, policy, _ = run_once(fit, ev, args, shift, "본실험")
        policy.save(out / "policy.json")
        result["runs"].append(row)
    else:
        ids = list(ev_table.call_ids)
        half = np.random.default_rng(args.rehearse_seed).permutation(len(ids)) % 2
        parts = [set(i for i, h in zip(ids, half) if h == k) for k in (0, 1)]
        ev_all = build(ev_table, calls, shift)
        same = fit_table is ev_table
        fit_all = ev_all if same else build(take_calls(fit_table, set(ids)), calls, shift)
        pooled = {k: np.zeros(len(ev_all), dtype=np.int64) for k in ("base", "tau_only", "learned")}
        for d in (0, 1):                                    # d: 판정에 쓰는 절반. 나머지 절반으로 학습한다
            fit_idx = np.array([i for i, c in enumerate(fit_all.call_ids) if c in parts[1 - d]])
            ev_idx = np.array([i for i, c in enumerate(ev_all.call_ids) if c in parts[d]])
            row, policy, halts = run_once(fit_all.subset(fit_idx), ev_all.subset(ev_idx), args, shift, f"리허설 {1 - d}→{d}")
            policy.save(out / f"policy_rehearse{d}.json")
            result["runs"].append(row)
            for k in pooled:
                pooled[k][ev_idx] = halts[k]
        # 두 방향의 판정 결과를 합친다: 모든 통화가 "자기를 보지 않은 정책"으로 한 번씩 채점된다
        row = {"tag": "리허설 합침", "n_eval_calls": len(ev_all), "base_tau": args.base_tau,
               "beta": None, "tau": None, "tau_only": None,
               "scores": {k: hl.score_halts(ev_all, h) for k, h in pooled.items()},
               "learned_vs_base": hl.paired_bootstrap(ev_all, pooled["learned"], pooled["base"], args.boot),
               "learned_vs_tau_only": hl.paired_bootstrap(ev_all, pooled["learned"], pooled["tau_only"], args.boot),
               "tau_only_vs_base": hl.paired_bootstrap(ev_all, pooled["tau_only"], pooled["base"], args.boot)}
        print_row(row)
        result["runs"].append(row)

    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"저장: {out / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
