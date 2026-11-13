"""제출 폴더를 조립한다. 커밋된 파일(HEAD)만 내보내고, 가중치 번들은 해시를 대조해 복사한다.

    python scripts/build_submission.py --out <제출 폴더>/mission4_symptom

- 코드는 `git archive HEAD` 로 내보내므로 커밋하지 않은 수정은 들어가지 않는다 (줄바꿈도 저장소 그대로).
- 실험 분석 전용 스크립트, 실험 기록(docs, runs)은 넣지 않는다. 제출 규격이 요구하는 것만 넣는다.
- `SUBMISSION_README.md` 를 제출 폴더의 `README.md` 로 넣는다.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

# 제출에 넣는 커밋된 파일 (mission4_symptom 기준)
FILES = [
    "inference.py", "score.py", "requirements.txt", "model_train.ipynb",
    "m4/__init__.py", "m4/bundle.py", "m4/data.py", "m4/decision.py", "m4/halting.py", "m4/simulate.py",
    "m4/streaming.py",
    # 학습 코드
    "scripts/build_cache.py", "scripts/tapt_mlm.py", "scripts/train_prefix.py", "scripts/prefix_probs.py",
    "scripts/run_oof_tables.sh", "scripts/train_halt_policy.py", "scripts/run_final_training.sh",
    "scripts/pack_bundle.py", "scripts/build_notebook.py",
    "tests/test_bundle.py", "tests/test_decision.py", "tests/test_halting.py", "tests/test_simulate.py",
    "tests/test_streaming.py",
]
RENAME = {"SUBMISSION_README.md": "README.md"}
BUNDLE = "ckpt/mission4.pt"
BUNDLE_SHA256 = "ce35f219ab347989350cb25ec7c935b026144126e6356d45a5f15b6eb95af140"   # E016 번들, E017 에서 Validation 확인


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="만들 제출 폴더 (이미 있으면 비어 있어야 한다)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} 가 비어 있지 않습니다. 지우고 다시 실행하십시오.", file=sys.stderr)
        return 1

    if subprocess.run(["git", "status", "--porcelain", "--", *FILES, *RENAME], cwd=HERE,
                      capture_output=True, text=True, check=True).stdout.strip():
        print("제출 대상 파일에 커밋하지 않은 수정이 있습니다. 먼저 커밋하십시오.", file=sys.stderr)
        return 1

    paths = FILES + list(RENAME)
    tar = subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "--format=tar", "HEAD", "--", *paths],
                         cwd=HERE, capture_output=True, check=True).stdout
    prefix = subprocess.run(["git", "rev-parse", "--show-prefix"], cwd=HERE, capture_output=True, text=True,
                            check=True).stdout.strip()
    written = 0
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            rel = m.name[len(prefix):] if m.name.startswith(prefix) else m.name
            dest = out / RENAME.get(rel, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tf.extractfile(m).read())
            written += 1
    if written != len(paths):
        print(f"내보낸 파일 수가 다릅니다: {written} / {len(paths)}", file=sys.stderr)
        return 1

    src = HERE / BUNDLE
    if sha256(src) != BUNDLE_SHA256:
        print(f"{BUNDLE} 해시가 확인한 번들과 다릅니다.", file=sys.stderr)
        return 1
    (out / "ckpt").mkdir(exist_ok=True)
    shutil.copyfile(src, out / BUNDLE)
    if sha256(out / BUNDLE) != BUNDLE_SHA256:
        print("복사한 번들의 해시가 다릅니다.", file=sys.stderr)
        return 1

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True,
                            check=True).stdout.strip()
    print(f"완료: {out} (코드 {written}개, 커밋 {commit}, 번들 sha256 {BUNDLE_SHA256[:12]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
