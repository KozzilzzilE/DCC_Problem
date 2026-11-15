"""E020 1단계 표본: fit 통화(runs/cache/fit_ids.txt)에서 6,000건을 뽑는다.

    python runs/E020_audio/make_sample.py

규칙(사전에 정한 것): fit_ids.txt 를 파일 순서대로 읽고, np.random.default_rng(2026).permutation 으로
섞은 뒤 앞 6,000개를 sample_ids.txt 에 섞인 순서 그대로 쓴다. dev / Validation 은 읽지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # runs/E020_audio
ROOT = HERE.parent.parent                         # mission4_symptom
sys.path.insert(0, str(ROOT))

from m4.data import read_ids  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

N_SAMPLE = 6000
SEED = 2026


def main() -> int:
    fit = read_ids(ROOT / "runs/cache/fit_ids.txt")
    assert len(fit) == len(set(fit)), "fit_ids 에 중복이 있다"
    order = np.random.default_rng(SEED).permutation(len(fit))
    sample = [fit[i] for i in order[:N_SAMPLE]]
    (HERE / "sample_ids.txt").write_text("\n".join(sample) + "\n", encoding="utf-8")
    print(f"fit {len(fit)}건 중 {len(sample)}건 -> {HERE / 'sample_ids.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
