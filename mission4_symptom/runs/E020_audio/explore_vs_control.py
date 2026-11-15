"""[탐색] stage1 본 실행 뒤에 추가한 비교. 판정에 쓰지 않는다.

    python runs/E020_audio/explore_vs_control.py     # 결과: explore_vs_control.txt

왜: 1차 대조군(음성 블록만 섞은 T+A)이 T 보다 평균 AUC 0.006 낮아 0 근처가 아니었다. 즉 로지스틱 회귀에
65차원을 더하는 것만으로 생기는 손해가 있다. 같은 차원·같은 손해를 지는 대조군과 진짜 음성을 직접 비교해,
"차원 손해를 빼고 나면 음성이 남기는 것이 있는가"를 본다 (LR_T+A − LR_T+A_shuf).
stage1_oof.npz 의 겹 밖 예측을 그대로 쓰고, 부트스트랩 절차(통화 단위 짝, 2,000회, seed 2026)도 stage1 과 같다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from m4 import SYMPTOM_CLASSES  # noqa: E402
from stage1 import N_BOOT, SEED, auc_matrix  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PAIRS = [("LR_TpA", "LR_TpA_shuf", "T+A − 대조군 (1차 로지스틱)")]


def main() -> int:
    z = np.load(HERE / "stage1_oof.npz", allow_pickle=False)
    Y = z["labels"]
    n, k = Y.shape
    rng = np.random.default_rng(SEED)
    names = sorted({m for a, b, _ in PAIRS for m in (a, b)})
    boot = {m: np.zeros((N_BOOT, k)) for m in names}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for m in names:
            boot[m][b] = auc_matrix(z[m][idx], Y[idx])
    L = ["[탐색] stage1 본 실행 뒤 추가. 판정에 쓰지 않음. 같은 차원의 대조군과 진짜 음성 비교",
         f"통화 {n}건, 짝 부트스트랩 {N_BOOT}회 seed {SEED}", ""]
    for a, b, title in PAIRS:
        d = auc_matrix(z[a], Y) - auc_matrix(z[b], Y)
        db = boot[a] - boot[b]
        L.append(title)
        for j, name in enumerate(SYMPTOM_CLASSES):
            lo, hi = np.percentile(db[:, j], [2.5, 97.5])
            L.append(f"  {name:<6}{d[j]:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        lo, hi = np.percentile(db.mean(1), [2.5, 97.5])
        L.append(f"  {'평균':<6}{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
        L.append("")
    text = "\n".join(L) + "\n"
    (HERE / "explore_vs_control.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
