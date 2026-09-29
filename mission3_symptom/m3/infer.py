"""Mission 3 제출 추론 — 대회 규정을 코드에 못박는다.

  - **입력**: 대화 본문(`utterances[].text`)만 사용한다. `m3.labels` 가 파싱 단계에서
    speaker / startAt / endAt / 인적사항을 원천 배제하므로 여기서 다시 거를 필요가 없다.
  - **결정 임계값**: 대회 규정대로 **모든 클래스 0.5 고정**이다. 임계값 파일을 읽지 않으며,
    번들(ensemble.json)도 임계값·클래스별 값을 받지 않는다.
  - **학습-추론 일치**: 발화 경계 표현(`utterance_sep_mode`), 인코딩(`encode_mode`),
    `max_length` 를 run_config 에서 복원한다. 이게 어긋나면 예외 없이 점수만 떨어진다.
  - **앙상블·블렌드**: `--ckpt_path` 가 `ensemble.json` 을 가리키면 트랜스포머 멤버 확률을
    균등 평균하고, Training 전용 TF-IDF 멤버와 9개 클래스 공통 가중치 하나로 섞는다.
    판정은 단일 모델과 같은 0.5 고정 임계값이며, 클래스별 가중치·임계값은 받지 않는다.

출력 CSV: `label file name`, `symptom`  (symptom 은 `"['두통', '복통']"` 형태의 String)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .config import (
    DEFAULT_UTTERANCE_SEP_MODE,
    NUM_CLASSES,
    TARGET_SYMPTOMS,
    UTTERANCE_SEP_MODES,
)
from .labels import read_transcript, verify_utterance_sep_mode
from .tfidf_member import load_tfidf_member

# 대회 규정 고정값. 체크포인트나 reports 에 저장된 보정 임계값이 있어도 사용하지 않는다.
DECISION_THRESHOLD = 0.5

OUTPUT_COLUMNS = ["label file name", "symptom"]

DEFAULT_BATCH_SIZE = 16
# fp32 가 기본이다. fp16(autocast)은 번들 ensemble.json 의 "precision" 이나 --precision 으로만 켜고
# CUDA 에서만 적용된다. RTX 5060 측정: 멤버 하나 forward 22초 -> 7.4초, 4멤버 번들 101초 -> 42초,
# Validation 3,640건 제출 행은 fp32 와 전부 같았다(멤버 seed42 단독은 32,760 칸 중 0~1 칸 차이).
SUPPORTED_PRECISIONS = ("fp32", "fp16")
DEFAULT_PRECISION = "fp32"
DEFAULT_MAX_LENGTH = 512
DEFAULT_ENCODE_MODE = "truncate"


def decision_threshold() -> float:
    """대회 규정 고정 임계값(0.5)을 돌려준다. 저장된 보정값은 무시한다."""
    return DECISION_THRESHOLD


def format_symptoms(symptoms: Sequence[str]) -> str:
    """제출 규격 문자열로 변환한다.

    증상 0개는 `"[]"`, 그 외에는 `"['두통', '복통']"` 처럼 Python 리스트 리터럴 형태다.
    순서는 `TARGET_SYMPTOMS`(가나다순)를 따른다.
    """
    unknown = [s for s in symptoms if s not in set(TARGET_SYMPTOMS)]
    if unknown:
        raise ValueError(f"9개 타겟 외 증상은 출력할 수 없습니다: {unknown}")
    ordered = [s for s in TARGET_SYMPTOMS if s in set(symptoms)]
    return str(ordered)


def symptoms_from_probabilities(
    probabilities: Sequence[float],
    threshold: float = DECISION_THRESHOLD,
) -> List[str]:
    """확률 벡터에 고정 임계값을 적용해 증상명 리스트를 만든다."""
    values = list(probabilities)
    if len(values) != NUM_CLASSES:
        raise ValueError(f"확률 벡터 길이가 9가 아닙니다: {len(values)}")
    return [TARGET_SYMPTOMS[i] for i, p in enumerate(values) if float(p) >= threshold]


def build_inference_config(training_config) -> Dict[str, object]:
    """학습 설정에서 추론이 반드시 복원해야 할 값만 추린다.

    `best_model/inference_config.json` 으로 저장되어, 부모 run 디렉터리 없이
    번들만 제출해도 학습과 같은 입력 표현·인코딩을 재현할 수 있게 한다.
    """
    return {
        "utterance_sep_mode": getattr(
            training_config, "utterance_sep_mode", DEFAULT_UTTERANCE_SEP_MODE),
        "encode_mode": getattr(training_config, "encode_mode", DEFAULT_ENCODE_MODE),
        "max_length": int(getattr(training_config, "max_length", DEFAULT_MAX_LENGTH)),
        "model_name_or_path": getattr(training_config, "model_name_or_path", None),
        "threshold": DECISION_THRESHOLD,
        "threshold_note": "대회 규정 고정값. class-wise threshold 는 제출에 사용하지 않는다.",
    }

@dataclass(frozen=True)
class InferenceSettings:
    """학습에서 복원한, 추론이 반드시 맞춰야 하는 설정."""

    model_dir: Path
    sep_mode: str
    encode_mode: str
    max_length: int
    config_path: Optional[Path]


def resolve_model_dir(ckpt_path: Union[str, Path]) -> Tuple[Path, Optional[Path]]:
    """`(model_dir, run_dir)` 을 돌려준다.

    run 디렉터리, `best_model` 디렉터리, 번들 안의 파일 경로를 모두 받아들인다.
    """
    path = Path(ckpt_path)
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"체크포인트 경로를 찾을 수 없습니다: {ckpt_path}")

    if (path / "best_model").is_dir():
        return path / "best_model", path
    if (path / "config.json").is_file():
        parent = path.parent
        run_dir = parent if (parent / "run_config.json").is_file() else None
        return path, run_dir

    raise FileNotFoundError(
        f"모델 번들을 찾지 못했습니다: {path}. "
        "run 디렉터리(best_model 을 포함) 또는 best_model 디렉터리를 지정하세요."
    )


def load_run_config(
    model_dir: Path,
    run_dir: Optional[Path],
) -> Tuple[Dict[str, object], Optional[Path]]:
    """추론 설정의 출처를 찾는다. 번들 안의 `inference_config.json` 이 우선한다."""
    candidates = [model_dir / "inference_config.json"]
    if run_dir is not None:
        candidates.append(run_dir / "run_config.json")
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8")), candidate
    return {}, None


def resolve_sep_mode(run_config: Dict[str, object]) -> str:
    """학습에 쓴 발화 경계 모드를 복원한다.

    기록이 없으면 학습 CSV 파일명에서 유추하고, 그마저 없으면 기존 기본값으로 떨어지되
    조용히 넘어가지 않고 경고를 남긴다.
    """
    mode = run_config.get("utterance_sep_mode")
    if isinstance(mode, str) and mode in UTTERANCE_SEP_MODES:
        return mode

    stem = Path(str(run_config.get("train_csv", ""))).stem.lower()
    for candidate in UTTERANCE_SEP_MODES:
        if candidate != DEFAULT_UTTERANCE_SEP_MODE and stem.endswith(f"_{candidate}"):
            print(
                f"[경고] run_config 에 utterance_sep_mode 가 없어 학습 CSV 이름에서 "
                f"{candidate!r} 로 추정했습니다."
            )
            return candidate

    print(
        "[경고] 발화 경계 모드를 확인하지 못해 기본값 "
        f"{DEFAULT_UTTERANCE_SEP_MODE!r} 를 사용합니다. "
        "학습 CSV 가 경계 토큰을 포함했다면 점수가 떨어집니다."
    )
    return DEFAULT_UTTERANCE_SEP_MODE


def resolve_settings(ckpt_path: Union[str, Path]) -> InferenceSettings:
    """체크포인트 경로에서 추론에 필요한 설정 일체를 복원한다."""
    model_dir, run_dir = resolve_model_dir(ckpt_path)
    run_config, config_path = load_run_config(model_dir, run_dir)

    encode_mode = run_config.get("encode_mode", DEFAULT_ENCODE_MODE)
    max_length = run_config.get("max_length", DEFAULT_MAX_LENGTH)
    return InferenceSettings(
        model_dir=model_dir,
        sep_mode=resolve_sep_mode(run_config),
        encode_mode=str(encode_mode) if encode_mode else DEFAULT_ENCODE_MODE,
        max_length=int(max_length) if max_length else DEFAULT_MAX_LENGTH,
        config_path=config_path,
    )


def read_texts(label_dir: Union[str, Path], sep_mode: str) -> Tuple[List[str], List[str]]:
    """라벨 폴더에서 `(파일명, 본문)` 을 파일명 순으로 읽는다."""
    directory = Path(label_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"라벨 폴더를 찾을 수 없습니다: {label_dir}")

    # 확장자 대소문자는 평가 데이터가 어떻게 오는지에 달렸다. `.JSON` 이면 못 찾고 죽는
    # 일이 없도록 소문자 비교로 모으고, 대소문자 구분 없는 파일시스템에서 중복되지 않게 한다.
    paths = sorted(
        {p.resolve(): p for p in directory.iterdir()
         if p.is_file() and p.suffix.lower() == ".json"}.values(),
        key=lambda p: p.name,
    )
    if not paths:
        raise FileNotFoundError(f"라벨 JSON 을 찾을 수 없습니다: {label_dir}")

    names: List[str] = []
    texts: List[str] = []
    for path in paths:
        names.append(path.name)
        # read_transcript 가 본문 외 메타데이터를 파싱 단계에서 버린다 (대회 규정).
        texts.append(read_transcript(path, sep_mode=sep_mode).text)
    return names, texts


ENSEMBLE_MANIFEST_NAME = "ensemble.json"
_MANIFEST_KEYS = {"members", "tfidf_member", "tfidf_weight", "note", "precision"}


def validate_precision(value) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_PRECISIONS:
        raise ValueError(f"precision 은 {list(SUPPORTED_PRECISIONS)} 중 하나여야 합니다: {value!r}")
    return value


@dataclass(frozen=True)
class EnsembleSpec:
    """`ensemble.json` 이 선언한 제출 번들. 클래스별 값이 들어갈 자리는 없다."""

    manifest_path: Path
    members: Tuple[Path, ...]
    tfidf_path: Optional[Path]
    tfidf_weight: float
    precision: str = DEFAULT_PRECISION


def find_ensemble_manifest(ckpt_path: Union[str, Path]) -> Optional[Path]:
    """`ckpt_path` 가 앙상블 번들(디렉터리 또는 `ensemble.json`)이면 그 경로, 아니면 None."""
    path = Path(ckpt_path)
    if path.is_file() and path.name == ENSEMBLE_MANIFEST_NAME:
        return path
    if path.is_dir() and (path / ENSEMBLE_MANIFEST_NAME).is_file():
        return path / ENSEMBLE_MANIFEST_NAME
    return None


def _resolve_relative(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _warn_if_outside(base: Path, path: Path, what: str) -> None:
    """제출은 `--ckpt_path` 폴더 하나다. 그 밖을 가리키면 폴더만 옮겼을 때 깨진다."""
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        print(
            f"[경고] {what} 이(가) 번들 밖을 가리킵니다: {path}. "
            "번들 폴더만 제출하면 찾을 수 없으니, 제출 전에 번들 안으로 복사하세요."
        )


def load_ensemble_spec(manifest_path: Union[str, Path]) -> EnsembleSpec:
    """`ensemble.json` 을 읽고 검증한다. 경로는 manifest 파일 위치 기준 상대경로도 받는다.

    형식:
        {"members": ["../run_a", "../run_b"],        # 트랜스포머 run 디렉터리, 균등 평균
         "tfidf_member": "../tfidf/tfidf_lr.joblib",  # 선택
         "tfidf_weight": 0.3,                          # tfidf_member 와 함께, 0 초과 1 미만 실수 하나
         "precision": "fp16",                          # 선택, 기본 fp32. fp16 은 CUDA 에서만 적용
         "note": "..."}                                # 선택
    임계값은 받지 않는다 (대회 규정 0.5 고정). 모르는 키가 있으면 거부한다.
    """
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{ENSEMBLE_MANIFEST_NAME} 은 JSON 객체여야 합니다: {manifest_path}")
    unknown = sorted(set(data) - _MANIFEST_KEYS)
    if unknown:
        raise ValueError(
            f"{ENSEMBLE_MANIFEST_NAME} 에 허용되지 않은 키가 있습니다: {unknown} "
            f"(허용: {sorted(_MANIFEST_KEYS)}; 임계값은 대회 규정상 0.5 고정)"
        )

    members = data.get("members")
    if not isinstance(members, list) or not members or not all(isinstance(m, str) and m for m in members):
        raise ValueError(f"members 는 비어 있지 않은 run 경로 목록이어야 합니다: {members!r}")
    base = manifest_path.parent
    resolved = []
    for member in members:
        path = _resolve_relative(base, member)
        resolve_model_dir(path)  # 없으면 FileNotFoundError
        _warn_if_outside(base, path, f"멤버 {member}")
        resolved.append(path)

    precision = validate_precision(data["precision"]) if "precision" in data else DEFAULT_PRECISION

    tfidf = data.get("tfidf_member")
    weight = data.get("tfidf_weight")
    if tfidf is None:
        if weight is not None:
            raise ValueError("tfidf_weight 는 tfidf_member 와 함께만 쓸 수 있습니다.")
        return EnsembleSpec(manifest_path, tuple(resolved), None, 0.0, precision)

    if not isinstance(tfidf, str) or not tfidf:
        raise ValueError(f"tfidf_member 는 파일 경로여야 합니다: {tfidf!r}")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError(f"tfidf_weight 는 9개 클래스 공통 실수 하나여야 합니다: {weight!r}")
    weight = float(weight)
    if not math.isfinite(weight) or not 0.0 < weight < 1.0:
        raise ValueError(f"tfidf_weight 는 0 초과 1 미만이어야 합니다: {weight}")
    tfidf_path = _resolve_relative(base, tfidf)
    if not tfidf_path.is_file():
        raise FileNotFoundError(f"TF-IDF 멤버 파일을 찾을 수 없습니다: {tfidf_path}")
    _warn_if_outside(base, tfidf_path, f"TF-IDF 멤버 {tfidf}")
    return EnsembleSpec(manifest_path, tuple(resolved), tfidf_path, weight, precision)


def inference_order(texts: Sequence[str]) -> List[int]:
    """긴 본문부터 처리하는 순서 (같은 길이는 입력 순서).

    짧은 것부터 하면 배치 텐서가 계속 커져 PyTorch 캐시가 이전 블록을 재사용하지 못한다.
    8GB 카드에서 실제 사용 0.7GB 에 예약 4.5GB, 배치 32 에서는 8GB 를 넘겨 20배 이상 느려졌다.
    가장 큰 배치를 먼저 잡으면 예약이 0.9GB 로 유지된다.
    """
    return sorted(range(len(texts)), key=lambda index: (-len(texts[index]), index))


def autocast_dtype(precision: str, device) -> Optional[object]:
    """fp16 은 CUDA 에서만. CPU 에서는 항상 fp32 다."""
    import torch

    validate_precision(precision)
    if precision == "fp16" and device.type == "cuda":
        return torch.float16
    return None


def _is_cuda_failure(exc: BaseException) -> bool:
    """CPU 로 내려서라도 끝낼 만한 GPU 쪽 실패인가 (allocator OOM, cuBLAS 할당 실패, 커널 없음 등)."""
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc)
    return isinstance(exc, RuntimeError) and any(
        key in message for key in ("CUDA", "CUBLAS", "cuDNN", "CUDNN", "out of memory"))


def _forward_rows(model, make_batch, indices: List[int], device, dtype):
    """묶음 하나의 확률. GPU 메모리가 부족하면 반으로 나눠 다시 한다 (한 건도 안 되면 예외를 올린다)."""
    import numpy as np
    import torch

    batch = logits = None
    try:
        batch = make_batch(indices, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(**batch).logits
        return torch.sigmoid(logits.float()).cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        if len(indices) == 1:
            raise
    # 재시도는 except 블록을 벗어난 뒤에 한다. 블록 안에서는 traceback 이 실패한 forward 의
    # 프레임(입력·활성값)을 붙잡고 있어 empty_cache 로도 메모리가 풀리지 않는다 (PyTorch FAQ).
    batch = logits = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
    middle = len(indices) // 2
    return np.concatenate([
        _forward_rows(model, make_batch, indices[:middle], device, dtype),
        _forward_rows(model, make_batch, indices[middle:], device, dtype),
    ])


def predict_all(model, make_batch, order: List[int], batch_size: int, device, dtype):
    """`order` 순서로 배치를 돌려 `(행 수, 9)` 확률과 마지막으로 쓴 device 를 돌려준다.

    채점은 1회 실행이라 GPU 쪽 실패로 죽으면 0점이다. 배치를 나눠도 한 건이 안 들어가거나
    CUDA 실행 오류(cuBLAS 할당 실패, 이 GPU 용 커널 없음 등)가 나면 모델을 CPU 로 내려 fp32 로
    끝까지 채운다. CPU 에서 난 오류나 CUDA 와 무관한 오류는 그대로 올린다.
    Windows 드라이버가 공유 메모리로 넘겨 느려지는 경우는 예외가 아니라서 여기서 잡히지 않는다.
    """
    import numpy as np
    import torch

    probabilities = np.zeros((len(order), NUM_CLASSES), dtype=np.float64)
    filled = np.zeros(len(order), dtype=bool)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            failure = None
            try:
                rows = _forward_rows(model, make_batch, chunk, device, dtype)
            except Exception as exc:  # noqa: BLE001 - CUDA 실패만 아래에서 처리하고 나머지는 다시 올린다
                if device.type != "cuda" or not _is_cuda_failure(exc):
                    raise
                failure = f"{type(exc).__name__}: {exc}"
            if failure is not None:
                # except 블록 밖에서 옮겨야 실패한 시도의 GPU 메모리가 풀린다.
                print(f"[경고] GPU 에서 추론하지 못해 CPU 로 전환합니다 (fp32): {failure}")
                device, dtype = torch.device("cpu"), None
                model.to(device)
                torch.cuda.empty_cache()
                rows = _forward_rows(model, make_batch, chunk, device, dtype)
            rows = np.asarray(rows)
            if rows.shape != (len(chunk), NUM_CLASSES):
                raise RuntimeError(
                    f"모델 출력 행 수가 배치와 다릅니다: {rows.shape} (기대 {(len(chunk), NUM_CLASSES)})")
            probabilities[chunk] = rows
            filled[chunk] = True
    if not filled.all():
        raise RuntimeError(f"예측이 누락된 행이 있습니다: {np.flatnonzero(~filled)[:5].tolist()}")
    return probabilities, device


def transformer_probabilities(
    label_dir: Union[str, Path],
    ckpt_path: Union[str, Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
    precision: str = DEFAULT_PRECISION,
):
    """트랜스포머 run 하나로 `(파일명 목록, (n, 9) 확률)` 을 만든다. 판정은 하지 않는다."""
    import numpy as np
    import torch

    from .dataset import encode_text
    from .model import load_saved_model

    settings = resolve_settings(ckpt_path)
    names, texts = read_texts(label_dir, settings.sep_mode)

    print(
        f"모델: {settings.model_dir}\n"
        f"설정 출처: {settings.config_path or '없음(기본값 사용)'}\n"
        f"발화 경계: {settings.sep_mode!r}  인코딩: {settings.encode_mode}  "
        f"max_length: {settings.max_length}\n"
        f"임계값: {DECISION_THRESHOLD:.3f} (대회 규정 고정)\n"
        f"대상: {len(names):,}건"
    )

    # 선언한 모드가 실제 본문에 반영됐는지 확인한다. 여기서 어긋나면 설정 복원이 잘못된
    # 것이지만, 채점은 1회 실행이라 중단시키지 않고 경고만 남긴다.
    try:
        verify_utterance_sep_mode(texts[:200], settings.sep_mode, source=str(label_dir))
    except ValueError as exc:
        print(f"[경고] {exc}")

    empty = sum(1 for text in texts if not text.strip())
    if empty:
        print(f"[경고] 본문이 비어 있는 통화 {empty:,}건은 특수 토큰만으로 추론됩니다.")

    tokenizer, model = load_saved_model(settings.model_dir)

    requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resolved_device = torch.device(requested)
        model.to(resolved_device)
    except (RuntimeError, AssertionError, ValueError) as exc:
        # CUDA OOM 이나 드라이버 문제로 죽으면 그대로 0점이다. CPU 로 내려서라도 끝낸다.
        print(f"[경고] {requested} 사용에 실패해 CPU 로 전환합니다: {exc}")
        resolved_device = torch.device("cpu")
        model.to(resolved_device)
    model.eval()
    dtype = autocast_dtype(precision, resolved_device)
    if dtype is not None:
        precision_label = "fp16 (autocast)"
    elif precision == "fp16":
        precision_label = "fp32 (fp16 요청됐지만 CUDA 가 아니라 fp32 로 실행)"
    else:
        precision_label = "fp32"
    print(f"정밀도: {precision_label}  배치: {batch_size}")

    def make_batch(indices: List[int], target_device):
        features = [
            encode_text(tokenizer, texts[index], settings.max_length, encode_mode=settings.encode_mode)
            for index in indices
        ]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        return {key: value.to(target_device) for key, value in batch.items()}

    # 길이가 비슷한 것끼리 묶어 padding 낭비를 줄이고(긴 것부터: 메모리 재사용), 결과는 원래 순서로 되돌린다.
    probabilities, resolved_device = predict_all(
        model, make_batch, inference_order(texts), batch_size, resolved_device, dtype)

    # 여러 멤버를 차례로 올릴 때 이전 모델이 GPU 메모리를 잡고 있지 않게 한다.
    del model
    # CPU 로 내려간 멤버도 앞서 GPU 에 잡아 둔 블록이 있을 수 있으니 반환 device 와 무관하게 비운다.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bad = ~np.isfinite(probabilities)
    if bad.any():
        rows = np.flatnonzero(bad.any(axis=1))
        print(
            f"[경고] 모델 출력에 NaN/inf 가 있는 파일 {len(rows):,}건: {[names[i] for i in rows[:5]]} "
            "-> 해당 클래스 확률을 0 으로 처리합니다 (0.5 미만이라 음성)."
        )
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
    return names, probabilities


def ensemble_probabilities(
    spec: EnsembleSpec,
    label_dir: Union[str, Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
    precision: Optional[str] = None,
):
    """멤버 확률 균등 평균 -> (선택) TF-IDF 전역 가중 블렌드. 판정은 하지 않는다."""
    description = f"트랜스포머 {len(spec.members)}개 균등 평균"
    if spec.tfidf_path is not None:
        description += f" + TF-IDF 멤버 (전역 가중치 {spec.tfidf_weight})"
    print(f"앙상블 번들: {spec.manifest_path}\n구성: {description}, 임계값 {DECISION_THRESHOLD:.3f} (대회 규정 고정)")

    names: Optional[List[str]] = None
    total = None
    for member in spec.members:
        member_names, probs = transformer_probabilities(
            label_dir, member, batch_size=batch_size, device=device,
            precision=precision or spec.precision)
        if names is None:
            names, total = list(member_names), probs.astype("float64")
        elif list(member_names) != names:
            raise RuntimeError(f"멤버마다 파일 순서가 다릅니다: {member}")
        else:
            total = total + probs
    blended = total / len(spec.members)

    if spec.tfidf_path is not None:
        # TF-IDF 멤버는 공백 결합 본문으로 학습했으므로 트랜스포머의 경계 모드와 무관하게 space 로 읽는다.
        tfidf_names, texts = read_texts(label_dir, "space")
        if list(tfidf_names) != names:
            raise RuntimeError("TF-IDF 멤버와 트랜스포머 멤버의 파일 순서가 다릅니다.")
        member = load_tfidf_member(spec.tfidf_path)
        blended = (1.0 - spec.tfidf_weight) * blended + spec.tfidf_weight * member.predict_proba(texts)
    return names, blended


def predict_directory(
    label_dir: Union[str, Path],
    ckpt_path: Union[str, Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
    precision: Optional[str] = None,
):
    """라벨 폴더 전체를 추론해 제출 규격 DataFrame 을 돌려준다.

    `ckpt_path` 가 run 디렉터리면 단일 모델, `ensemble.json` 번들이면 앙상블·블렌드다.
    어느 쪽이든 판정은 `symptoms_from_probabilities` 의 0.5 고정 임계값 하나로 한다.
    """
    import pandas as pd

    manifest = find_ensemble_manifest(ckpt_path)
    if manifest is None:
        names, probabilities = transformer_probabilities(
            label_dir, ckpt_path, batch_size=batch_size, device=device,
            precision=validate_precision(precision) if precision is not None else DEFAULT_PRECISION)
    else:
        names, probabilities = ensemble_probabilities(
            load_ensemble_spec(manifest), label_dir, batch_size=batch_size, device=device,
            precision=validate_precision(precision) if precision is not None else None)

    if len(probabilities) != len(names):
        raise RuntimeError(f"확률 행 수({len(probabilities)})와 파일 수({len(names)})가 다릅니다.")
    predictions = [format_symptoms(symptoms_from_probabilities(row)) for row in probabilities]

    return pd.DataFrame(
        {OUTPUT_COLUMNS[0]: list(names), OUTPUT_COLUMNS[1]: predictions},
        columns=OUTPUT_COLUMNS,
    )
