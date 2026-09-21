"""Mission 3 제출 추론 — 대회 규정을 코드에 못박는다.

  - **입력**: 대화 본문(`utterances[].text`)만 사용한다. `m3.labels` 가 파싱 단계에서
    speaker / startAt / endAt / 인적사항을 원천 배제하므로 여기서 다시 거를 필요가 없다.
  - **결정 임계값**: 대회 규정대로 **0.5 고정**이다. 학습 중 탐색한 class-wise threshold
    (`reports/best_thresholds.json` 포함)는 제출 경로에서 읽지 않는다.
  - **학습-추론 일치**: 발화 경계 표현(`utterance_sep_mode`), 인코딩(`encode_mode`),
    `max_length` 를 run_config 에서 복원한다. 이게 어긋나면 예외 없이 점수만 떨어진다.

출력 CSV: `label file name`, `symptom`  (symptom 은 `"['두통', '복통']"` 형태의 String)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .config import (
    DEFAULT_UTTERANCE_SEP_MODE,
    NUM_CLASSES,
    TARGET_SYMPTOMS,
    UTTERANCE_SEP_MODES,
)
from .labels import read_transcript

# 대회 규정 고정값. 체크포인트나 reports 에 저장된 보정 임계값이 있어도 사용하지 않는다.
DECISION_THRESHOLD = 0.5

OUTPUT_COLUMNS = ["label file name", "symptom"]

DEFAULT_BATCH_SIZE = 16
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
    paths = sorted(Path(label_dir).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"라벨 JSON 을 찾을 수 없습니다: {label_dir}")

    names: List[str] = []
    texts: List[str] = []
    for path in paths:
        names.append(path.name)
        # read_transcript 가 본문 외 메타데이터를 파싱 단계에서 버린다 (대회 규정).
        texts.append(read_transcript(path, sep_mode=sep_mode).text)
    return names, texts


def predict_directory(
    label_dir: Union[str, Path],
    ckpt_path: Union[str, Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
):
    """라벨 폴더 전체를 추론해 제출 규격 DataFrame 을 돌려준다."""
    import pandas as pd
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

    tokenizer, model = load_saved_model(settings.model_dir)
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(resolved_device)
    model.eval()

    # 길이가 비슷한 것끼리 묶어 padding 낭비를 줄이고, 결과는 원래 순서로 되돌린다.
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    predictions: List[Optional[str]] = [None] * len(texts)

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            features = [
                encode_text(
                    tokenizer,
                    texts[index],
                    settings.max_length,
                    encode_mode=settings.encode_mode,
                )
                for index in chunk
            ]
            batch = tokenizer.pad(features, padding=True, return_tensors="pt")
            batch = {key: value.to(resolved_device) for key, value in batch.items()}
            logits = model(**batch).logits
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            for offset, index in enumerate(chunk):
                predictions[index] = format_symptoms(
                    symptoms_from_probabilities(probabilities[offset])
                )

    missing = [names[i] for i, value in enumerate(predictions) if value is None]
    if missing:
        raise RuntimeError(f"예측이 누락된 파일이 있습니다: {missing[:5]}")

    return pd.DataFrame(
        {OUTPUT_COLUMNS[0]: names, OUTPUT_COLUMNS[1]: predictions},
        columns=OUTPUT_COLUMNS,
    )
