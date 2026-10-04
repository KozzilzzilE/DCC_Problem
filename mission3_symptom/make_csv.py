"""원본 라벨 JSON 폴더에서 학습용 CSV 를 만든다 (학습 1단계).

대화 본문(`utterances[].text`)만 공백으로 이어 붙이고(발화 경계 모드 `space`), 9개 대상 증상만
0/1 컬럼으로 펼친다. 화자·시간·인적사항은 `m3.labels` 가 파싱 단계에서 버린다. 추론 경로도 같은
함수를 쓰므로 학습 입력과 추론 입력이 어긋나지 않는다.

    python make_csv.py --label-dir <data>/train/label --output data_csv/mission3_train.csv
    python make_csv.py --label-dir <data>/val/label   --output data_csv/mission3_val.csv

출력 컬럼: call_id, text, symptoms, label_vector, 고열 ... 호흡곤란 (utf-8-sig)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from m3.config import TARGET_SYMPTOMS  # noqa: E402
from m3.labels import load_transcripts_dir  # noqa: E402


def build_dataframe(label_dir: Path) -> pd.DataFrame:
    """라벨 폴더 하나를 CSV 행으로 바꾼다. JSON 이 하나도 없으면 실패한다."""
    records = load_transcripts_dir(label_dir, sep_mode="space")
    if not records:
        raise FileNotFoundError(f"파싱된 라벨 JSON 이 없습니다: {label_dir}")
    rows = []
    for record in records:
        row = {
            "call_id": record.call_id,
            "text": record.text,
            "symptoms": str(sorted(record.symptoms)),
            "label_vector": str(record.label_vector.tolist()),
        }
        row.update({symptom: int(record.label_vector[i]) for i, symptom in enumerate(TARGET_SYMPTOMS)})
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    """원본 라벨 JSON 폴더를 학습 CSV(call_id, text, 9개 라벨)로 바꿔 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label-dir", required=True, help="원본 라벨 JSON 폴더 (예: data/train/label)")
    parser.add_argument("--output", required=True, help="저장할 CSV 경로")
    args = parser.parse_args(argv)

    frame = build_dataframe(Path(args.label_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"{len(frame):,}행 -> {output}")


if __name__ == "__main__":
    main()
