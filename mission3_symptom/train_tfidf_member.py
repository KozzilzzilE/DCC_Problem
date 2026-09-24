"""Training 전용 TF-IDF + LR 보조 멤버 학습 CLI.

    python mission3_symptom/train_tfidf_member.py \
        --train-csv <mission3_train.csv> \
        --output mission3_symptom/runs/tfidf_lr_c0.15/tfidf_lr.joblib

- 반드시 Training CSV 만 넣는다. Validation 을 넣으면 Validation 을 학습에 쓴 것이 된다 (실격 사유).
- 본문은 공백 결합(`space`) 이어야 한다. 제출 추론도 이 멤버에는 공백 결합 본문을 넣는다.
- 결과 옆에 같은 이름의 `.json` 으로 적합 출처(CSV 경로, 행 수, C, scikit-learn 버전)를 남긴다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args(argv=None):
    from m3.tfidf_member import DEFAULT_C, DEFAULT_MIN_DF

    p = argparse.ArgumentParser(description="Mission 3 Training 전용 TF-IDF + LR 보조 멤버")
    p.add_argument("--train-csv", required=True, help="Training CSV (text + 9개 라벨 열). Validation 금지")
    p.add_argument("--output", required=True, help="저장 경로 (.joblib)")
    p.add_argument("--C", type=float, default=DEFAULT_C, help="LR 정규화 강도 (9개 클래스 공통)")
    p.add_argument("--min-df", type=int, default=DEFAULT_MIN_DF)
    p.add_argument("--overwrite", action="store_true", help="기존 파일을 덮어쓴다")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from m3.config import TARGET_SYMPTOMS
    from m3.dataset import labels_from_dataframe, load_symptom_csv
    from m3.labels import verify_utterance_sep_mode
    from m3.tfidf_member import fit_tfidf_member, save_tfidf_member

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"이미 있습니다: {output} (--overwrite 로 덮어쓰기)")

    frame = load_symptom_csv(args.train_csv)
    texts = frame["text"].astype(str).tolist()
    # 경계 토큰이 섞인 CSV 로 적합하면 추론(공백 결합)과 입력이 어긋난다.
    try:
        verify_utterance_sep_mode(texts, "space", source=str(args.train_csv))
    except ValueError as exc:
        raise ValueError(f"TF-IDF 멤버는 space(공백 결합) 본문만 받습니다: {exc}") from None

    labels = labels_from_dataframe(frame).astype(int)
    member = fit_tfidf_member(texts, labels, C=args.C, min_df=args.min_df)
    save_tfidf_member(member, output)

    record = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "train_rows": member.train_rows,
        "C": member.C,
        "min_df": member.min_df,
        "class_weight": "balanced",
        "input": "space-joined utterances[].text",
        "sklearn_version": member.sklearn_version,
        "positives": {symptom: int(labels[:, i].sum()) for i, symptom in enumerate(TARGET_SYMPTOMS)},
        "note": "Training CSV 로만 적합. Validation/Test 는 transform 만 한다.",
    }
    output.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {output}  (Training {member.train_rows:,}행, C={member.C}, min_df={member.min_df})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
