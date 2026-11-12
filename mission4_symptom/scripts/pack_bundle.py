"""학습 결과(best_model 폴더)를 제출용 `.pt` 번들로 묶는다.

    python scripts/pack_bundle.py --members runs/E002_prefix_uniform --out ckpt/mission4.pt

멤버가 여러 개면 추론 때 확률을 평균한다. 확률 보정에 쓰는 log(pos_weight) 는 멤버들의 run_config.json 에서
읽는데, 평균 확률에 보정값 하나를 쓰므로 멤버끼리 값이 다르면 묶지 않고 멈춘다. 값이 적혀 있지 않은 멤버
(run_config.json 이 없거나 형식이 다른 경우)는 확인할 수 없다는 경고만 내고 묶는다.
maxent 규칙이면 증상 조합 사전분포를 학습에 쓴 통화의 라벨로 만들어 함께 넣는다. 통화 목록은 run_config.json 에
calls / ids 를 적어 둔 첫 멤버에서 읽고, `--calls` / `--prior-ids` 로 바꿀 수 있다 (dev·Validation 라벨로 만들면 안 된다).

4-2 중단 기준은 기본이 "기대 점수 >= --halt-tau" 이다. 학습형 정책을 넣으려면 정책 파일을 준다:

    python scripts/pack_bundle.py --members ... --out ... --halt-policy runs/E012_halt_policy/ens3/policy.json

정책을 학습할 때 쓴 보정값(logit_shift)이 번들의 값과 다르면 묶지 않고 멈춘다. `--halt-policy` 를 빼면 예전 방식 그대로다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4.bundle import learned_halt_setting, pack_bundle, shared_pos_weight, shared_prior_source  # noqa: E402
from m4.data import label_matrix, load_cache, read_ids  # noqa: E402
from m4.decision import label_set_prior  # noqa: E402


def read_run_config(member: Path) -> dict:
    """멤버 폴더의 run_config.json. 없으면 빈 설정으로 본다 (best_model 폴더만 받은 멤버)."""
    path = member / "run_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="run 폴더 (여러 개면 확률 평균)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", default="space")
    ap.add_argument("--truncation-side", default="right")
    ap.add_argument("--rule", default="maxent", choices=["maxent", "expected", "threshold"], help="4-1 결정 규칙")
    ap.add_argument("--calls", help="maxent 사전분포를 만들 라벨 캐시 (기본: 첫 멤버가 학습에 쓴 캐시)")
    ap.add_argument("--prior-ids", help="사전분포에 쓸 통화 목록 (기본: 첫 멤버가 학습에 쓴 목록)")
    ap.add_argument("--halt-type", default="expected", choices=["expected", "confidence", "stable", "fixed", "never"])
    ap.add_argument("--halt-tau", type=float, default=0.52)
    ap.add_argument("--halt-fallback-tau", type=float, default=0.50,
                    help="학습형 정책을 읽지 못할 때(lightgbm 없음) 쓸 기대 점수 기준 (3개 평균 0.50, 단일 모델 0.52)")
    ap.add_argument("--halt-policy", help="학습형 중단 정책 policy.json (scripts/train_halt_policy.py 산출물). "
                                          "주면 --halt-type / --halt-tau 대신 이 정책을 넣는다")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    if args.halt_policy and any(a.split("=")[0] in ("--halt-type", "--halt-tau") for a in sys.argv[1:]):
        raise SystemExit("--halt-policy 는 --halt-type / --halt-tau 와 함께 쓸 수 없습니다 (정책 파일의 값을 쓴다).")
    configs = [read_run_config(Path(m)) for m in args.members]
    names = [Path(m).name for m in args.members]
    # 사전분포 출처는 maxent 규칙이고 --calls / --prior-ids 중 비운 것이 있을 때만 학습 설정에서 찾는다
    need_source = args.rule == "maxent" and (not args.calls or args.prior_ids is None)
    try:
        weights, unknown = shared_pos_weight(configs, names)       # 멤버끼리 다르면 여기서 멈춘다
        calls_default, ids_default = shared_prior_source(configs, names) if need_source else (None, None)
    except ValueError as err:
        raise SystemExit(str(err))
    if weights is None:
        print("경고: 어느 멤버에도 pos_weight 가 없어 확률을 보정하지 않습니다.")
    elif unknown:
        print(f"경고: pos_weight 를 확인할 수 없는 멤버 {unknown} — 다른 멤버의 값으로 보정합니다. 같은 값으로 학습했는지 확인하십시오.")
    label_prior = None
    if args.rule == "maxent":
        # 사전분포는 모델이 학습에 쓴 통화의 라벨로만 만든다. 따로 주지 않으면 학습 설정에 적힌 캐시와 목록을 쓴다.
        calls_path = args.calls or calls_default
        ids_path = args.prior_ids if args.prior_ids is not None else ids_default
        if not calls_path:
            raise SystemExit("학습 설정에 라벨 캐시 경로가 없습니다. --calls 와 --prior-ids 를 직접 지정하십시오.")
        if "val" in Path(calls_path).name.lower():
            raise SystemExit(f"Validation 라벨로는 사전분포를 만들 수 없습니다: {calls_path}")
        keep = read_ids(Path(ids_path)) if ids_path else None      # 목록이 없으면 캐시 전체로 학습한 모델
        labels = label_matrix(load_cache(Path(calls_path), keep))
        print(f"사전분포 라벨: {calls_path} / 목록 {ids_path or '(캐시 전체)'} / {len(labels)}통화")
        prior, marg = label_set_prior(labels)
        # 어떤 라벨로 만들었는지 번들에 남긴다 (추론에는 쓰지 않는 기록)
        label_prior = {"prior": prior.tolist(), "marg": marg.tolist(),
                       "source": {"calls": str(calls_path), "ids": str(ids_path) if ids_path else None,
                                  "n_calls": int(len(labels))}}
    logit_shift = [math.log(w) for w in weights] if weights else None
    halt = {"type": args.halt_type, "tau": args.halt_tau, "min_step": 5}
    if args.halt_policy:
        # 학습형 정책: 정책을 학습할 때의 보정값이 번들의 보정값과 다르면 특징이 어긋나므로 멈춘다
        try:
            halt = learned_halt_setting(json.loads(Path(args.halt_policy).read_text(encoding="utf-8")), logit_shift)
        except ValueError as err:
            raise SystemExit(f"{args.halt_policy}: {err}")
        halt["source"] = str(args.halt_policy)             # 어느 파일에서 왔는지 (추론에는 쓰지 않는 기록)
        # 채점 환경에 lightgbm 이 없을 때 쓸 대체 기준 (m4/streaming.py 가 경고를 내고 이쪽으로 내려간다)
        halt["fallback"] = {"type": "expected", "tau": args.halt_fallback_tau, "min_step": 5}
        print(f"학습형 중단 정책: {args.halt_policy} / beta {halt['beta']:g}, tau {halt['tau']:.2f}, "
              f"min_step {halt['min_step']}, 특징 {len(halt['feature_names'])}개, 학습 통화 {halt.get('n_fit_calls')}")
    settings = {"fmt": args.fmt, "truncation_side": args.truncation_side,
                "logit_shift": logit_shift,
                "rule": {"type": args.rule}, "label_prior": label_prior,
                # 4-2 중단 시점에는 Macro-F1 에 맞는 임계값 규칙을 쓴다 (dev 비교 결과, EXPERIMENT_LOG E004)
                "halt_rule": {"type": "threshold", "threshold": 0.5, "force_top1": True},
                "halt": halt}
    path = pack_bundle(args.members, args.out, settings, note=args.note)
    print(f"{path} ({path.stat().st_size / 2**20:.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
