"""Training 전용 TF-IDF + LogisticRegression 보조 멤버.

제출 추론에서 트랜스포머 확률과 **9개 클래스 공통 가중치 하나**로 섞인다. 판정 임계값은
대회 규정대로 0.5 그대로다.

규정과 재현성을 위해 지키는 것:
  - 입력은 공백으로 이어 붙인 `utterances[].text` 뿐이다 (발화 경계 토큰 없음).
  - vectorizer 와 LR 은 Training CSV 로만 적합한다. Validation/Test 는 transform 만 한다.
  - 클래스별 LR 은 `class_weight='balanced'` 이고 정규화 강도 C 는 모든 클래스 공통이다.
    기본 C=0.15 는 처음에 Validation 오심 지표를 보며 탐색했고(하이퍼파라미터 선택),
    이후 Training 5-fold OOF 에서 오심 AUROC/AP 가 가장 높은 값으로 확인했다.
    OOF 의 macro 기준으로는 C=0.5 가 약간 더 높았다.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np

from .config import NUM_CLASSES, TARGET_SYMPTOMS

MEMBER_FORMAT = "m3-tfidf-lr"
MEMBER_VERSION = 1
DEFAULT_C = 0.15
DEFAULT_MIN_DF = 3


def build_vectorizers(min_df: int = DEFAULT_MIN_DF):
    """문자 2-4gram(char_wb) 과 공백 토큰 1-2gram 두 개의 TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=min_df,
        max_features=400_000, sublinear_tf=True,
    )
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=min_df,
        max_features=300_000, sublinear_tf=True, token_pattern=r"\S+",
    )
    return char, word


@dataclass
class TfidfLRMember:
    char_vectorizer: object
    word_vectorizer: object
    models: List[object]
    C: float
    min_df: int
    train_rows: int
    sklearn_version: str

    def _features(self, texts: Sequence[str]):
        from scipy import sparse

        texts = list(texts)
        return sparse.hstack(
            [self.char_vectorizer.transform(texts), self.word_vectorizer.transform(texts)]
        ).tocsr()

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """`(n, 9)` 양성 확률. 열 순서는 `TARGET_SYMPTOMS`."""
        features = self._features(texts)
        probs = np.zeros((features.shape[0], NUM_CLASSES), dtype=np.float64)
        for index, model in enumerate(self.models):
            probs[:, index] = model.predict_proba(features)[:, 1]
        return probs


def fit_tfidf_member(
    texts: Sequence[str],
    labels,
    C: float = DEFAULT_C,
    min_df: int = DEFAULT_MIN_DF,
) -> TfidfLRMember:
    """Training 본문과 9열 이진 라벨로 멤버를 적합한다."""
    import sklearn
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression

    texts = [str(text) for text in texts]
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape != (len(texts), NUM_CLASSES):
        raise ValueError(
            f"라벨 행렬은 (본문 수, {NUM_CLASSES}) 이어야 합니다: {labels.shape}, 본문 {len(texts)}건"
        )
    for index, symptom in enumerate(TARGET_SYMPTOMS):
        values = set(np.unique(labels[:, index]).tolist())
        if values != {0, 1}:
            raise ValueError(f"{symptom} 열에 양성과 음성이 모두 있어야 적합할 수 있습니다: {sorted(values)}")
    if not C > 0:
        raise ValueError(f"C 는 양수여야 합니다: {C}")

    char, word = build_vectorizers(min_df=min_df)
    features = sparse.hstack([char.fit_transform(texts), word.fit_transform(texts)]).tocsr()

    models = []
    for index in range(NUM_CLASSES):
        model = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=300)
        models.append(model.fit(features, labels[:, index].astype(int)))

    return TfidfLRMember(
        char_vectorizer=char, word_vectorizer=word, models=models, C=float(C),
        min_df=int(min_df), train_rows=len(texts), sklearn_version=sklearn.__version__,
    )


def save_tfidf_member(member: TfidfLRMember, path: Union[str, Path]) -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "format": MEMBER_FORMAT,
        "version": MEMBER_VERSION,
        "symptoms": list(TARGET_SYMPTOMS),
        "char_vectorizer": member.char_vectorizer,
        "word_vectorizer": member.word_vectorizer,
        "models": member.models,
        "C": member.C,
        "min_df": member.min_df,
        "train_rows": member.train_rows,
        "sklearn_version": member.sklearn_version,
    }, path)
    return path


def load_tfidf_member(path: Union[str, Path]) -> TfidfLRMember:
    """저장된 멤버를 읽는다. 형식과 증상 순서가 다르면 조용히 넘기지 않는다."""
    import joblib
    import sklearn

    payload = joblib.load(Path(path))
    if not isinstance(payload, dict) or payload.get("format") != MEMBER_FORMAT:
        raise ValueError(f"TF-IDF 멤버 형식이 아닙니다: {path}")
    if payload.get("version") != MEMBER_VERSION:
        raise ValueError(f"지원하지 않는 TF-IDF 멤버 버전입니다: {payload.get('version')}")
    if list(payload.get("symptoms", [])) != list(TARGET_SYMPTOMS):
        raise ValueError(f"증상 순서가 현재 설정과 다릅니다: {payload.get('symptoms')}")
    if len(payload["models"]) != NUM_CLASSES:
        raise ValueError(f"클래스별 모델 수가 {NUM_CLASSES} 가 아닙니다: {len(payload['models'])}")
    if payload.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"TF-IDF 멤버를 만든 scikit-learn {payload.get('sklearn_version')} 과 현재 "
            f"{sklearn.__version__} 가 다릅니다. 확률이 달라질 수 있습니다.",
            RuntimeWarning,
        )
    return TfidfLRMember(
        char_vectorizer=payload["char_vectorizer"],
        word_vectorizer=payload["word_vectorizer"],
        models=list(payload["models"]),
        C=float(payload["C"]),
        min_df=int(payload["min_df"]),
        train_rows=int(payload["train_rows"]),
        sklearn_version=str(payload["sklearn_version"]),
    )
