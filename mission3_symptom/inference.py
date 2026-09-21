"""Mission 3 (환자 증상 인식) — 미션 폴더 단독 실행용 추론 진입점.

    python inference.py --label_dir <json 폴더> \
                        --ckpt_path runs/<run 이름> --output ./outputs/mission3.csv

이 파일이 있는 폴더(mission3_symptom/)만 제출해도 동작하도록 만들었다.
  - 같은 폴더의 m3 패키지만 import 한다. 루트 inference.py 처럼 librosa/torchvision 같은
    다른 미션의 의존성을 끌어오지 않는다 (Mission 3 는 텍스트 과제라 필요가 없다)
  - 모델 입력은 대화 본문(`utterances[].text`)만 사용한다. 화자·시간·인적사항은
    m3.labels 가 파싱 단계에서 원천 배제한다
  - 결정 임계값은 대회 규정대로 0.5 고정이다. 학습 중 탐색한 class-wise threshold 는
    제출 경로에서 사용하지 않는다
  - 발화 경계 표현과 인코딩 설정은 체크포인트의 run_config 에서 복원한다

출력 CSV: [label file name], [symptom]   (symptom 은 "['두통', '복통']" 형태의 String)
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mission 3 환자 증상 다중 라벨 추론")
    p.add_argument("--audio_dir", default=None,
                   help="받기만 하고 사용하지 않는다. Mission 3 는 대화 본문만 입력으로 허용된다")
    p.add_argument("--label_dir", required=True, help="json 라벨 폴더 (utterances[].text 만 사용)")
    p.add_argument("--ckpt_path", required=True,
                   help="run 디렉터리(best_model 포함) 또는 best_model 디렉터리")
    p.add_argument("--output", required=True, help="결과 CSV 경로 (예: ./outputs/mission3.csv)")
    p.add_argument("--batch_size", type=int, default=16, help="추론 배치 크기")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from m3.infer import predict_directory  # 폴더 안의 m3 패키지

    started = time.perf_counter()
    df = predict_directory(args.label_dir, args.ckpt_path, batch_size=args.batch_size)
    elapsed = time.perf_counter() - started

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    n = len(df)
    empty = int((df["symptom"] == "[]").sum())
    print(
        f"완료: {out}  ({n}행, 추론 {elapsed:.1f}초, 통화당 {elapsed * 1000 / max(1, n):.1f} ms)\n"
        f"증상 0개로 예측된 통화: {empty}건"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
