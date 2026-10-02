"""Mission 2 (화자 이진 분류: 상황실 vs 신고자) 미션 폴더 단독 실행용 추론 스크립트.

실행 방법:
    python inference.py --audio_dir <wav 폴더> --label_dir <json 폴더> \\
                        --ckpt_path checkpoints/ --output ./outputs/mission2.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_CKPT = HERE / "checkpoints"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mission 2 화자 이진 분류 추론 스크립트")
    p.add_argument("--audio_dir", required=True, help="wav 오디오 파일이 있는 폴더 경로")
    p.add_argument("--label_dir", required=True, help="json 전사 라벨 파일이 있는 폴더 경로")
    p.add_argument("--ckpt_path", default=str(DEFAULT_CKPT), help="체크포인트 파일(.pt) 또는 디렉토리 경로")
    p.add_argument("--output", required=True, help="결과를 저장할 CSV 파일 경로")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from m2.infer import predict_directory

    started = time.perf_counter()
    df = predict_directory(args.audio_dir, args.label_dir, args.ckpt_path)
    elapsed = time.perf_counter() - started

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    n = len(df)
    per_item = (elapsed * 1000 / max(1, n))
    print(f"완료: {out} ({n}개 발화 추론 완료, 소요시간: {elapsed:.2f}초, 건당: {per_item:.2f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
