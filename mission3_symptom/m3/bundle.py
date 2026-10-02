"""Mission 3 제출 번들: 학습한 모델 여러 개를 `.pt` 파일 하나로 묶고 다시 불러온다.

출제문제 12쪽은 가중치를 `.pt/.pth/.ckpt` 파일로, 11쪽은 `--ckpt_path {checkpoint file}` 하나로
추론하라고 한다. 그래서 앙상블 멤버 전부를 `torch.save` 한 dict 하나에 담는다.

    {"format": "dcc-mission3-bundle", "version": 1,
     "precision": "fp16" | "fp32",            # 추론 정밀도 (fp16 은 CUDA 에서만 적용)
     "members": [{"name": "final_s42",
                  "files": {"config.json": bytes, "tokenizer.json": bytes, ...},  # best_model 폴더의 설정 파일 원문
                  "state_dict": {이름: Tensor}}, ...],                             # 가중치 (fp32)
     "tfidf": None | {"weight": float, "joblib": bytes},   # 선택: Training 전용 TF-IDF 멤버와 혼합 가중치
     "note": str}

- 판정 임계값은 `m3.infer.DECISION_THRESHOLD`(0.5) 고정이다. 멤버 inference_config.json 의 threshold 0.5 는 기록용이며 읽지 않는다.
- 멤버는 균등 평균한다. 클래스별 가중치를 넣을 자리는 없다.
- 불러올 때 tokenizer/config 는 작은 임시 폴더에 풀어 Hugging Face 표준 로더로 읽고, 가중치는 메모리에서
  바로 넣는다. 인터넷에 접속하지 않는다.
"""
from __future__ import annotations

import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

BUNDLE_FORMAT = "dcc-mission3-bundle"
BUNDLE_VERSION = 1
WEIGHTS_FILE = "model.safetensors"
INFERENCE_CONFIG_FILE = "inference_config.json"
SUPPORTED_PRECISIONS = ("fp32", "fp16")


@dataclass(frozen=True)
class BundleMember:
    """번들 안의 트랜스포머 멤버 하나 (학습 run 하나의 best_model 폴더에 해당)."""

    name: str                       # 멤버 이름 (run 폴더 이름, 예: final_s42)
    files: Dict[str, bytes]         # best_model 폴더의 가중치 외 파일 원문 (config·tokenizer·inference_config 등)
    state_dict: Dict[str, object]   # 가중치 {파라미터 이름: Tensor} (fp32, mmap 으로 열림)

    def inference_config(self) -> Dict[str, object]:
        """학습이 남긴 inference_config.json (발화 경계·인코딩·max_length). 없으면 빈 dict."""
        raw = self.files.get(INFERENCE_CONFIG_FILE)
        return json.loads(raw.decode("utf-8")) if raw else {}


@dataclass(frozen=True)
class Bundle:
    """`load_bundle` 이 돌려주는, 검증을 마친 `.pt` 번들 내용."""

    path: Path                          # 읽은 .pt 파일 경로
    precision: str                      # 추론 정밀도 "fp16" | "fp32" (fp16 은 CUDA 에서만 적용)
    members: Tuple[BundleMember, ...]   # 확률을 균등 평균할 트랜스포머 멤버 (저장 순서 그대로)
    tfidf_weight: float                 # TF-IDF 블렌드 가중치 w (멤버가 없으면 0.0)
    tfidf_joblib: Optional[bytes]       # TF-IDF 멤버 joblib 원문 (없으면 None)
    note: str                           # 조립할 때 남긴 설명


def pack_bundle(
    member_dirs: Sequence[Union[str, Path]],
    output: Union[str, Path],
    precision: str = "fp16",
    tfidf_path: Optional[Union[str, Path]] = None,
    tfidf_weight: float = 0.0,
    note: str = "",
) -> Path:
    """학습 run 의 `best_model` 폴더(또는 그 부모 run 폴더)들을 `.pt` 하나로 묶는다."""
    import torch
    from safetensors.torch import load_file

    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"precision 은 {SUPPORTED_PRECISIONS} 중 하나여야 합니다: {precision!r}")
    if not member_dirs:
        raise ValueError("멤버가 하나 이상 있어야 합니다")
    members = []
    for raw in member_dirs:
        path = Path(raw)
        model_dir = path / "best_model" if (path / "best_model").is_dir() else path
        weights = model_dir / WEIGHTS_FILE
        if not weights.is_file() or not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"{model_dir} 에 config.json 과 {WEIGHTS_FILE} 이 있어야 합니다")
        files = {f.name: f.read_bytes() for f in sorted(model_dir.iterdir()) if f.is_file() and f.name != WEIGHTS_FILE}
        name = path.name if model_dir is not path else model_dir.name
        members.append({"name": name, "files": files, "state_dict": load_file(str(weights))})

    tfidf = None
    if tfidf_path is not None and tfidf_weight > 0:
        if not 0.0 < float(tfidf_weight) < 1.0:
            raise ValueError(f"tfidf_weight 는 0 초과 1 미만이어야 합니다: {tfidf_weight}")
        tfidf = {"weight": float(tfidf_weight), "joblib": Path(tfidf_path).read_bytes()}

    payload = {"format": BUNDLE_FORMAT, "version": BUNDLE_VERSION, "precision": precision,
               "members": members, "tfidf": tfidf, "note": note}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output


def load_bundle(path: Union[str, Path]) -> Bundle:
    """`.pt` 번들을 읽고 형식을 검증한다. 가중치는 mmap 으로 열어 필요할 때 메모리에 올린다."""
    import torch

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"Mission 3 제출 번들(.pt)이 아닙니다: {path}")
    if payload.get("version") != BUNDLE_VERSION:
        raise ValueError(f"지원하지 않는 번들 버전입니다: {payload.get('version')}")
    precision = payload.get("precision", "fp32")
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"번들 precision 이 올바르지 않습니다: {precision!r}")
    members = tuple(BundleMember(m["name"], dict(m["files"]), m["state_dict"]) for m in payload["members"])
    if not members:
        raise ValueError("번들에 멤버가 없습니다")
    tfidf = payload.get("tfidf")
    weight = float(tfidf["weight"]) if tfidf else 0.0
    if tfidf and not 0.0 < weight < 1.0:
        raise ValueError(f"번들의 TF-IDF 가중치가 올바르지 않습니다: {weight}")
    return Bundle(path, precision, members, weight, tfidf["joblib"] if tfidf else None, payload.get("note", ""))


def load_member_model(member: BundleMember):
    """번들 멤버 하나를 (tokenizer, model) 로 만든다. 저장 당시 best_model 폴더를 읽은 것과 같다."""
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    from .config import NUM_CLASSES
    from .model import validate_tokenizer_model_compatibility

    with tempfile.TemporaryDirectory() as tmp:
        for name, data in member.files.items():
            (Path(tmp) / name).write_bytes(data)
        tokenizer = AutoTokenizer.from_pretrained(tmp, local_files_only=True)
        config = AutoConfig.from_pretrained(tmp, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(member.state_dict, strict=True)
    if int(model.config.num_labels) != NUM_CLASSES:
        raise ValueError(f"번들 멤버 {member.name} 의 출력 클래스 수가 9가 아닙니다: {model.config.num_labels}")
    validate_tokenizer_model_compatibility(tokenizer, model)
    return tokenizer, model


def load_bundle_tfidf(bundle: Bundle):
    """번들에 TF-IDF 멤버가 있으면 불러온다. 없으면 None."""
    if bundle.tfidf_joblib is None:
        return None
    from .tfidf_member import load_tfidf_member

    return load_tfidf_member(io.BytesIO(bundle.tfidf_joblib))


def describe(bundle: Bundle) -> List[str]:
    """번들 구성을 사람이 읽을 줄 목록으로 요약한다 (경로·멤버 수·정밀도, 멤버별 파라미터 수, TF-IDF 가중치).

    추론 시작 로그와 `devsel.py assemble` 기록에 쓰며, 파라미터 수는 state_dict 의 원소 수를 더한 값이다.
    """
    lines = [f"번들: {bundle.path} (멤버 {len(bundle.members)}개, 정밀도 {bundle.precision})"]
    for member in bundle.members:
        params = sum(int(t.numel()) for t in member.state_dict.values())
        lines.append(f"  - {member.name}: 파라미터 {params:,}")
    if bundle.tfidf_joblib is not None:
        lines.append(f"  - TF-IDF 멤버: 전역 가중치 {bundle.tfidf_weight}")
    return lines
