"""Mission 1 (신고자 성별 분류) — 미션 폴더 단독 실행용 추론 진입점.

    python inference.py --audio_dir <wav 폴더> --label_dir <json 폴더> \
                        --ckpt_path ckpt/w2v2_full.pt --output ./outputs/mission1.csv

이 파일이 있는 폴더(mission1_gender/)만 제출해도 동작하도록 만들었다.
  - 같은 폴더의 m1 패키지를 sys.path 에 넣는다 (루트 inference.py 처럼 다른 미션의
    의존성을 import 하지 않는다)
  - 사전학습 모델의 config/가중치는 체크포인트에 동봉돼 있어 인터넷 접속이 없다
    (HF_HUB_OFFLINE=1 로 검증)
  - 결정 임계값·피처 설정도 체크포인트에서 자동 복원된다

출력 CSV: [audio file name], [gender]  (gender 는 '남' / '여')
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# 한국어 Windows 콘솔(cp949)에서 진행 메시지 때문에 죽지 않게 한다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_CKPT = HERE / "ckpt" / "w2v2_full.pt"   # 제출 1안. 폴백: ckpt/resnet_aug_m80.pt


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mission 1 신고자 성별 분류 추론")
    p.add_argument("--audio_dir", required=True, help="wav 폴더")
    p.add_argument("--label_dir", required=True, help="json 라벨 폴더 (startAt/endAt/speaker 만 사용)")
    p.add_argument("--ckpt_path", default=str(DEFAULT_CKPT), help=f"체크포인트 (.pt). 기본 {DEFAULT_CKPT.name}")
    p.add_argument("--output", required=True, help="결과 CSV 경로 (예: ./outputs/mission1.csv)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="창(window) 단위 추론 배치 크기. 생략하면 갈래별 기본값(w2v2 32, resnet 128). "
                        "8 GB GPU 에서 w2v2 를 128 로 올리면 VRAM 초과로 10배 느려진다")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from m1.infer import predict_directory  # 폴더 안의 m1 패키지

    started = time.perf_counter()
    df = predict_directory(args.audio_dir, args.label_dir, args.ckpt_path, batch_size=args.batch_size)
    elapsed = time.perf_counter() - started

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    n = len(df)
    print(f"완료: {out}  ({n}행, 추론 {elapsed:.1f}초, 통화당 {elapsed * 1000 / max(1, n):.1f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
