"""제출용 가중치 파일(.pt) 하나에 모델과 추론 설정을 함께 담는다.

학습이 남긴 best_model 폴더(HF 형식)를 묶어 `.pt` 로 만들고, 추론 때는 이 파일만 읽는다.
입력 형식(화자 표시 여부, 잘림 방향)과 결정 규칙·중단 기준도 파일 안에 넣어, 경로 이름이나
실행 환경에 따라 추론 동작이 달라지지 않게 한다.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

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
    # 4-2 중단 시점의 결정 규칙. None 이면 4-1 과 같은 규칙을 쓴다.
    "halt_rule": None,
    # 4-2 중단 기준: min_step 번째 발화부터, 최대 확률이 tau 이상이면 중단한다
    "halt": {"type": "confidence", "tau": 0.8, "min_step": 5},
}


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
