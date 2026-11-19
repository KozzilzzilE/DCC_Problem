"""학습 결과(best_model 폴더)를 제출용 `.pt` 번들로 묶는다.

    # 공유 번들 (4-1 · 4-2 동일 멤버)
    python scripts/pack_bundle.py --members runs/E002_prefix_uniform --out ckpt/mission4.pt

    # 미션 분리 번들 (방안 A: 4-1 SoftJaccard + 4-2 ENS3 + C128_v1_nonempty)
    python scripts/pack_bundle.py --members-4-1 <SJ42> <SJ43> --members-4-2 <E001> <E002> <E007> \
        --halt-policy policy.json --require-nonempty --out ckpt/mission4.pt

멤버가 여러 개면 추론 때 확률을 평균한다. 확률 보정에 쓰는 log(pos_weight) 는 멤버들의 run_config.json 에서
읽는데, 평균 확률에 보정값 하나를 쓰므로 멤버끼리 값이 다르면 묶지 않고 멈춘다. 값이 적혀 있지 않은 멤버
(run_config.json 이 없거나 형식이 다른 경우)는 확인할 수 없다는 경고만 내고 묶는다.
기본 결정 규칙은 대회 규정대로 임계값 0.5 고정(`--rule threshold`, `--force-top1` 없음, 빈 리스트 허용)이다.
4-2 중단 기준은 기본이 "기대 점수 >= 0.5 + 빈 예측 조기 중단 차단" 이다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4.bundle import (  # noqa: E402
    learned_halt_setting,
    pack_bundle,
    pack_mission_split_bundle,
    shared_pos_weight,
    shared_prior_source,
)
from m4.data import label_matrix, load_cache, read_ids  # noqa: E402
from m4.decision import label_set_prior  # noqa: E402


def read_run_config(member: Path, explicit: Optional[Path] = None) -> dict:
    """멤버의 run_config.json 을 읽는다. 명시 경로가 없으면 <member>/run_config.json 또는 상위 폴더를 본다."""
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"지정한 멤버 설정 파일을 찾을 수 없습니다: {explicit}")
        return json.loads(explicit.read_text(encoding="utf-8"))
    for cand in (member / "run_config.json", member.parent / "run_config.json"):
        if cand.is_file():
            return json.loads(cand.read_text(encoding="utf-8"))
    return {}


def _resolve_configs(members: Sequence[str], explicit_paths: Optional[Sequence[str]], flag_name: str) -> List[dict]:
    if explicit_paths is not None and len(explicit_paths) != len(members):
        raise SystemExit(f"{flag_name} 개수({len(explicit_paths)})가 멤버 수({len(members)})와 다릅니다.")
    return [
        read_run_config(Path(m), Path(explicit_paths[i]) if explicit_paths else None)
        for i, m in enumerate(members)
    ]


def _resolve_rule_and_shift(
    args: argparse.Namespace,
    member_dirs: Sequence[str],
    configs: Sequence[dict],
) -> Tuple[Optional[List[float]], dict, Optional[dict], bool]:
    names = [Path(m).name for m in member_dirs]
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
        label_prior = {"prior": prior.tolist(), "marg": marg.tolist(),
                       "source": {"calls": str(calls_path), "ids": str(ids_path) if ids_path else None,
                                  "n_calls": int(len(labels))}}
    logit_shift = [math.log(w) for w in weights] if weights else None
    top1 = bool(args.force_top1 and not args.no_force_top1)
    rule: Dict[str, object] = {"type": args.rule}
    if args.rule == "threshold":
        rule.update({"threshold": 0.5, "force_top1": top1})   # 보정하지 않은 확률(calibrated 없음)에 0.5 를 적용한다
    return logit_shift, rule, label_prior, top1


def _build_halt_setting(args: argparse.Namespace, logit_shift: Optional[List[float]]) -> Dict[str, object]:
    if args.halt_policy:
        raw_policy = json.loads(Path(args.halt_policy).read_text(encoding="utf-8"))
        if args.allow_empty_halt:
            req_nonempty = False
        elif args.require_nonempty:
            req_nonempty = True
        else:
            req_nonempty = bool(raw_policy.get("require_nonempty", False))
        raw_policy["require_nonempty"] = req_nonempty
        try:
            halt = learned_halt_setting(raw_policy, logit_shift)
        except ValueError as err:
            raise SystemExit(f"{args.halt_policy}: {err}")
        halt["source"] = Path(args.halt_policy).name         # 어느 파일에서 왔는지 (추론에는 쓰지 않는 기록)
        # 채점 환경에 lightgbm 이 없을 때 쓸 대체 기준 (m4/streaming.py 가 경고를 내고 이쪽으로 내려간다)
        halt["fallback"] = {"type": "expected", "tau": args.halt_fallback_tau, "min_step": 5,
                            "require_nonempty": req_nonempty}
        print(f"학습형 중단 정책: {args.halt_policy} / beta {halt['beta']:g}, tau {halt['tau']:.2f}, "
              f"min_step {halt['min_step']}, require_nonempty={req_nonempty}, 특징 {len(halt['feature_names'])}개, 학습 통화 {halt.get('n_fit_calls')}")
        return halt

    req_nonempty = not args.allow_empty_halt if (args.require_nonempty or args.allow_empty_halt) else True
    return {"type": args.halt_type, "tau": args.halt_tau, "min_step": 5, "require_nonempty": req_nonempty}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", help="공유 번들용 run/best_model 폴더 (4-1·4-2 동일 멤버)")
    ap.add_argument("--members-4-1", nargs="+", help="미션 분리 번들(방안 A): 4-1 전용 멤버 폴더")
    ap.add_argument("--members-4-2", nargs="+", help="미션 분리 번들(방안 A): 4-2 전용 멤버 폴더")
    ap.add_argument("--member-configs", nargs="+", help="--members 에 대응하는 run_config.json 경로(선택)")
    ap.add_argument("--member-configs-4-1", nargs="+", help="--members-4-1 에 대응하는 run_config.json 경로(선택)")
    ap.add_argument("--member-configs-4-2", nargs="+", help="--members-4-2 에 대응하는 run_config.json 경로(선택)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", default="space")
    ap.add_argument("--truncation-side", default="right")
    ap.add_argument("--rule", default="threshold", choices=["threshold", "maxent", "expected"], help="4-1 결정 규칙 (기본: threshold 0.5 고정)")
    ap.add_argument("--calls", help="maxent 사전분포를 만들 라벨 캐시 (기본: 첫 멤버가 학습에 쓴 캐시)")
    ap.add_argument("--prior-ids", help="사전분포에 쓸 통화 목록 (기본: 첫 멤버가 학습에 쓴 목록)")
    ap.add_argument("--halt-type", default="expected", choices=["expected", "confidence", "stable", "fixed", "never"])
    ap.add_argument("--halt-tau", type=float, default=0.5)
    ap.add_argument("--halt-fallback-tau", type=float, default=0.50,
                    help="학습형 정책을 읽지 못할 때(lightgbm 없음) 쓸 기대 점수 기준 (3개 평균 0.50)")
    ap.add_argument("--halt-policy", help="학습형 중단 정책 policy.json (scripts/train_halt_policy.py 산출물). "
                                          "주면 --halt-type / --halt-tau 대신 이 정책을 넣는다")
    ap.add_argument("--require-nonempty", action="store_true",
                    help="4-2 조기 중단에 보정 전 확률 >= 0.5 인 증상이 1개 이상일 때만 허용하는 게이트를 켠다")
    ap.add_argument("--allow-empty-halt", action="store_true",
                    help="4-2 조기 중단 시 빈 예측 차단을 끈다")
    ap.add_argument("--no-force-top1", action="store_true",
                    help="(기본 동작 유지용 호환 플래그) 0.5 를 넘는 증상이 없을 때 1위를 강제로 내지 않는다")
    ap.add_argument("--force-top1", action="store_true",
                    help="임계값 규칙에서 0.5 이상이 없을 때 1위 1개를 강제 출력한다 (대회 규정 위배 주의)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    if args.require_nonempty and args.allow_empty_halt:
        raise SystemExit("--require-nonempty 와 --allow-empty-halt 는 함께 쓸 수 없습니다.")
    if args.halt_policy and any(a.split("=")[0] in ("--halt-type", "--halt-tau") for a in sys.argv[1:]):
        raise SystemExit("--halt-policy 는 --halt-type / --halt-tau 와 함께 쓸 수 없습니다 (정책 파일의 값을 쓴다).")

    has_shared = bool(args.members)
    has_split = bool(args.members_4_1 or args.members_4_2)
    if has_shared and has_split:
        raise SystemExit("--members 와 --members-4-1/--members-4-2 는 함께 쓸 수 없습니다.")
    if not has_shared and not (args.members_4_1 and args.members_4_2):
        raise SystemExit("--members 또는 (--members-4-1 과 --members-4-2 모두) 를 지정하십시오.")

    if has_shared:
        configs = _resolve_configs(args.members, args.member_configs, "--member-configs")
        logit_shift, rule, label_prior, top1 = _resolve_rule_and_shift(args, args.members, configs)
        halt = _build_halt_setting(args, logit_shift)
        settings = {"fmt": args.fmt, "truncation_side": args.truncation_side,
                    "logit_shift": logit_shift,
                    "rule": rule, "label_prior": label_prior,
                    "halt_rule": {"type": "threshold", "threshold": 0.5, "force_top1": top1},
                    "halt": halt}
        path = pack_bundle(args.members, args.out, settings, note=args.note)
    else:
        top1 = bool(args.force_top1 and not args.no_force_top1)
        configs_41 = _resolve_configs(args.members_4_1, args.member_configs_4_1, "--member-configs-4-1")
        if args.rule == "threshold":
            # 4-1 임계값 규칙은 raw >= 0.5 만 사용하므로 logit_shift / label_prior 없이 묶는다
            rule_41 = {"type": "threshold", "threshold": 0.5, "force_top1": top1}
            shift_41, prior_41 = None, None
        else:
            shift_41, rule_41, prior_41, _ = _resolve_rule_and_shift(args, args.members_4_1, configs_41)
        settings_41 = {
            "fmt": args.fmt, "truncation_side": args.truncation_side,
            "logit_shift": shift_41, "rule": rule_41, "label_prior": prior_41,
            "halt_rule": {"type": "threshold", "threshold": 0.5, "force_top1": top1},
            "halt": {"type": "never"},
        }
        configs_42 = _resolve_configs(args.members_4_2, args.member_configs_4_2, "--member-configs-4-2")
        shift_42, rule_42, prior_42, _ = _resolve_rule_and_shift(args, args.members_4_2, configs_42)
        halt_42 = _build_halt_setting(args, shift_42)
        settings_42 = {
            "fmt": args.fmt, "truncation_side": args.truncation_side,
            "logit_shift": shift_42, "rule": rule_42, "label_prior": prior_42,
            "halt_rule": {"type": "threshold", "threshold": 0.5, "force_top1": top1},
            "halt": halt_42,
        }
        path = pack_mission_split_bundle(
            {"4-1": (args.members_4_1, settings_41), "4-2": (args.members_4_2, settings_42)},
            args.out, note=args.note,
        )

    print(f"{path} ({path.stat().st_size / 2**20:.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
