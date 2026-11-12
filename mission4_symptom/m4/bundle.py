"""제출용 가중치 파일(.pt) 하나에 모델과 추론 설정을 함께 담는다.

학습이 남긴 best_model 폴더(HF 형식)를 묶어 `.pt` 로 만들고, 추론 때는 이 파일만 읽는다.
입력 형식(화자 표시 여부, 잘림 방향)과 결정 규칙·중단 기준도 파일 안에 넣어, 경로 이름이나
실행 환경에 따라 추론 동작이 달라지지 않게 한다.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from . import SYMPTOM_CLASSES

BUNDLE_FORMAT = "dcc-mission4-bundle"
BUNDLE_VERSION = 1
WEIGHTS_FILE = "model.safetensors"

DEFAULT_SETTINGS: Dict[str, object] = {
    "fmt": "space",                 # 누적 입력 형식 (m4.data.format_utterance)
    "max_length": 512,
    "truncation_side": "right",
    # 학습 때 쓴 log(pos_weight). 확률 보정에 쓴다 (m4.decision.calibrate). 없으면 보정하지 않는다.
    "logit_shift": None,
    # 4-1 결정 규칙 (m4.decision.make_rule)
    "rule": {"type": "threshold", "threshold": 0.5, "force_top1": True, "max_labels": None},
    # maxent 규칙에 쓰는 증상 조합 사전분포 {"prior": 511개, "marg": 9개} (m4.decision.label_set_prior).
    # 그 모델의 학습에 쓴 통화의 라벨로만 만든다. 다른 규칙에서는 쓰지 않으며, 예전 번들에는 이 키가 없다.
    "label_prior": None,
    # 4-2 중단 시점의 결정 규칙. 4-2 는 전체를 모아 증상별 F1 을 내므로 4-1 규칙과 따로 둔다 (EXPERIMENT_LOG E004).
    "halt_rule": {"type": "threshold", "threshold": 0.5, "force_top1": True},
    # 4-2 중단 기준 (m4.streaming.should_halt). 제출 번들은 scripts/pack_bundle.py 가 expected·tau 0.52 로 덮어쓴다
    "halt": {"type": "confidence", "tau": 0.8, "min_step": 5},
}


def member_pos_weight(config: Dict[str, object]) -> Optional[List[float]]:
    """학습 설정 하나에서 클래스별 pos_weight 9개를 꺼낸다. 적혀 있지 않으면 None.

    본선 학습 스크립트는 `pos_weight` 목록으로 남기고, 예선(Mission 3) 학습 스크립트는
    `train_pos_weight_statistics` 에 증상 이름별로 남긴다. 두 형식을 모두 읽는다.
    """
    weights = config.get("pos_weight")
    if weights is None:
        stats = config.get("train_pos_weight_statistics")
        if not isinstance(stats, dict) or any(c not in stats for c in SYMPTOM_CLASSES):
            return None
        weights = [stats[c]["pos_weight"] for c in SYMPTOM_CLASSES]
    weights = [float(w) for w in weights]
    if len(weights) != len(SYMPTOM_CLASSES) or not all(w > 0 for w in weights):
        raise ValueError(f"pos_weight 는 양수 {len(SYMPTOM_CLASSES)}개여야 합니다: {weights}")
    return weights


def shared_pos_weight(configs: Sequence[Dict[str, object]], names: Sequence[str],
                      tol: float = 1e-5) -> Tuple[Optional[List[float]], List[str]]:
    """멤버들의 pos_weight 를 하나로 정한다. (pos_weight 또는 None, 값이 적혀 있지 않은 멤버 이름) 을 돌려준다.

    번들은 평균한 확률에 보정값 하나를 쓰므로, 값이 적힌 멤버끼리는 같아야 한다 (다르면 ValueError).
    tol 은 float32 로 저장된 값과 float64 값의 차이(1e-7 수준)만 넘기는 크기다.
    값이 없는 멤버는 확인할 길이 없으니 이름만 돌려주고, 묶는 쪽에서 알린다.
    """
    chosen: Optional[List[float]] = None
    chosen_name = ""
    missing: List[str] = []
    for config, name in zip(configs, names):
        weights = member_pos_weight(config)
        if weights is None:
            missing.append(name)
        elif chosen is None:
            chosen, chosen_name = weights, name
        elif max(abs(a - b) for a, b in zip(chosen, weights)) > tol:
            raise ValueError(f"멤버끼리 pos_weight 가 다릅니다 ({chosen_name} / {name}). "
                             "보정값 하나를 평균 확률에 쓰므로 함께 묶을 수 없습니다.")
    return chosen, missing


def shared_prior_source(configs: Sequence[Dict[str, object]], names: Sequence[str]
                        ) -> Tuple[Optional[str], Optional[str]]:
    """사전분포를 만들 (라벨 캐시, 통화 목록) 을 정한다. 학습 설정에 적어 둔 첫 멤버의 것을 쓴다.

    적어 둔 멤버끼리 서로 다르면 어느 쪽 라벨로 만들지 정할 수 없으므로 멈춘다 (ValueError).
    아무도 적어 두지 않았으면 (None, None).
    """
    chosen: Optional[Tuple[str, Optional[str]]] = None
    chosen_name = ""
    for config, name in zip(configs, names):
        if not config.get("calls"):
            continue
        source = (str(config["calls"]), str(config["ids"]) if config.get("ids") else None)
        if chosen is None:
            chosen, chosen_name = source, name
        elif source != chosen:
            raise ValueError(f"멤버끼리 학습 통화가 다릅니다 ({chosen_name}: {chosen} / {name}: {source}). "
                             "--calls 와 --prior-ids 로 직접 지정하십시오.")
    return chosen if chosen is not None else (None, None)


def pack_bundle(member_dirs: Sequence[Union[str, Path]], output: Union[str, Path],
                settings: Dict[str, object], note: str = "", half: bool = True) -> Path:
    """best_model 폴더들을 `.pt` 하나로 묶는다. half=True 면 가중치를 fp16 으로 저장해 크기를 줄인다."""
    import torch
    from safetensors.torch import load_file

    members = []
    for raw in member_dirs:
        path = Path(raw)
        model_dir = path / "best_model" if (path / "best_model").is_dir() else path
        files = {f.name: f.read_bytes() for f in sorted(model_dir.iterdir())
                 if f.is_file() and f.name != WEIGHTS_FILE}
        state = load_file(str(model_dir / WEIGHTS_FILE))
        if half:
            state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
        members.append({"name": path.name, "files": files, "state_dict": state})
    merged = json.loads(json.dumps(DEFAULT_SETTINGS))
    merged.update(settings)
    payload = {"format": BUNDLE_FORMAT, "version": BUNDLE_VERSION, "members": members,
               "settings": merged, "note": note}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output


def load_bundle(path: Union[str, Path], device) -> Tuple[List[Tuple[object, object]], Dict[str, object]]:
    """`.pt` 를 읽어 [(tokenizer, model)] 과 추론 설정을 돌려준다. 외부 다운로드는 하지 않는다."""
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"Mission 4 제출 번들(.pt)이 아닙니다: {path}")
    if payload.get("version") != BUNDLE_VERSION:
        raise ValueError(f"지원하지 않는 번들 버전입니다: {payload.get('version')}")
    settings = payload["settings"]
    members = []
    for member in payload["members"]:
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in member["files"].items():
                (Path(tmp) / name).write_bytes(data)
            tokenizer = AutoTokenizer.from_pretrained(tmp, local_files_only=True)
            config = AutoConfig.from_pretrained(tmp, local_files_only=True)
        tokenizer.truncation_side = settings["truncation_side"]
        model = AutoModelForSequenceClassification.from_config(config)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v
                               for k, v in member["state_dict"].items()}, strict=True)
        model.to(device).eval()
        if device.type == "cuda":
            model.half()            # GPU 에서는 fp16 으로 추론한다
        members.append((tokenizer, model))
    return members, settings
