"""Mission 3 (환자 증상 인식) — 미션 폴더 단독 실행용 추론 진입점.

    python inference.py --audio_dir <wav 폴더> --label_dir <json 폴더> \
                        --ckpt_path ckpt/mission3.pt --output ./outputs/mission3.csv

    (--audio_dir 는 받기만 하고 읽지 않는다. --ckpt_path 는 제출 번들 .pt 파일(ckpt/mission3.pt),
     또는 모델 하나를 확인할 때 학습 run/best_model 폴더를 받는다)

이 파일이 있는 폴더(mission3_symptom/)만 제출해도 동작하도록 만들었다.
  - 같은 폴더의 m3 패키지만 import 한다. 루트 inference.py 처럼 librosa/torchvision 같은
    다른 미션의 의존성을 끌어오지 않는다 (Mission 3 는 텍스트 과제라 필요가 없다)
  - 모델 입력은 대화 본문(`utterances[].text`)만 사용한다. 화자·시간·인적사항은
    m3.labels 가 파싱 단계에서 원천 배제한다
  - 결정 임계값은 대회 규정대로 9개 클래스 모두 0.5 고정이다. 클래스별·튜닝된 임계값은
    어디서도 고르거나 불러오지 않는다
  - 발화 경계 표현과 인코딩 설정은 번들 멤버(또는 run 폴더)에 저장된 학습 설정에서 복원한다

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
    """추론 CLI 인자. 출제문제 11쪽 형식(--audio_dir, --label_dir, --ckpt_path, --output)을 받는다."""
    p = argparse.ArgumentParser(description="Mission 3 환자 증상 다중 라벨 추론")
    p.add_argument("--audio_dir", default=None,
                   help="받기만 하고 사용하지 않는다. Mission 3 는 대화 본문만 입력으로 허용된다")
    p.add_argument("--label_dir", required=True, help="json 라벨 폴더 (utterances[].text 만 사용)")
    p.add_argument("--ckpt_path", required=True,
                   help="제출 번들 .pt 파일(ckpt/mission3.pt), 또는 단일 모델 확인용 run 디렉터리(best_model 포함)/best_model 디렉터리")
    p.add_argument("--output", required=True, help="결과 CSV 경로 (예: ./outputs/mission3.csv)")
    p.add_argument("--batch_size", type=int, default=16, help="추론 배치 크기")
    p.add_argument("--precision", choices=("fp32", "fp16"), default=None,
                   help="지정하지 않으면 .pt 번들의 precision(제출 번들은 fp16)을, run 폴더면 fp32 를 쓴다. fp16 은 CUDA 에서만")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """label 폴더 전체를 추론해 제출 CSV(label file name, symptom)를 쓰고 걸린 시간을 출력한다."""
    args = parse_args(argv)

    from m3.infer import predict_directory  # 폴더 안의 m3 패키지

    started = time.perf_counter()
    df = predict_directory(args.label_dir, args.ckpt_path, batch_size=args.batch_size,
                           precision=args.precision)
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
