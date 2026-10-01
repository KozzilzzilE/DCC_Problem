"""Mission 3 figures. Matplotlib default colors. English labels.

Local:
    python figures.py

Colab: paste this file, then it writes figures/*.png
Korean font is not required.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

try:
    OUT = Path(__file__).resolve().parent / "figures"
except NameError:
    OUT = Path.cwd() / "figures"

SYMPTOMS = [
    "Fever",
    "Vomiting",
    "Headache",
    "Abdominal pain",
    "Dizziness",
    "Laceration",
    "Nausea",
    "Weakness",
    "Dyspnea",
]

C0, C1, C2, C3, C4, C5, C6, C7 = plt.rcParams["axes.prop_cycle"].by_key()["color"][:8]


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def _box(ax, x, y, w, h, text, face, tcolor="white", fs=10):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.1",
            facecolor=face,
            edgecolor="none",
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tcolor, fontsize=fs)


def fig_eval_flow() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 3.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title("Evaluation path", loc="left", fontweight="bold")
    boxes = [
        (0.2, 1.05, 2.55, 1.3, "Input\nutterances[].text\nspace-joined", C0),
        (3.1, 1.05, 2.4, 1.3, "Model\n9 sigmoid scores", C0),
        (5.85, 1.05, 2.6, 1.3, "Decision\nprob >= 0.5", C1),
        (8.8, 1.05, 2.5, 1.3, "Metric\nMacro F1", C0),
    ]
    for x, y, w, h, t, c in boxes:
        _box(ax, x, y, w, h, t, c)
    for x in (2.8, 5.55, 8.5):
        ax.annotate("", xy=(x + 0.25, 1.7), xytext=(x, 1.7), arrowprops=dict(arrowstyle="->", lw=1.6))


def fig_class_counts() -> None:
    train = [5186, 4463, 2905, 6771, 6334, 3465, 3341, 5310, 3927]
    val = [678, 537, 374, 838, 774, 463, 429, 651, 492]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    x = range(len(SYMPTOMS))
    ax.bar([i - 0.18 for i in x], train, 0.36, label="Training 29,200")
    ax.bar([i + 0.18 for i in x], val, 0.36, label="Validation 3,640")
    ax.set_xticks(list(x), SYMPTOMS, rotation=20, ha="right")
    ax.set_ylabel("Positive count")
    ax.set_title("Positives per class", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.set_ylim(0, 7800)


def fig_label_cardinality() -> None:
    train = [19560, 7137, 2161, 325, 17]
    val = [2419, 896, 280, 40, 5]
    labs = ["1", "2", "3", "4", "5"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = range(5)
    ax.bar([i - 0.18 for i in x], train, 0.36, label="Training")
    ax.bar([i + 0.18 for i in x], val, 0.36, label="Validation")
    ax.set_xticks(list(x), labs)
    ax.set_xlabel("Labels per call")
    ax.set_ylabel("Number of calls")
    ax.set_title("Label cardinality", loc="left", fontweight="bold")
    ax.legend(frameon=False)


def fig_token_length() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    names = ["Median", "p95", "Max"]
    train = [230, 460, 873]
    val = [233, 463, 807]
    x = range(3)
    ax.bar([i - 0.18 for i in x], train, 0.36, label="Training")
    ax.bar([i + 0.18 for i in x], val, 0.36, label="Validation")
    ax.axhline(512, color=C3, ls="--", lw=1.4, label="max_length 512")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("KLUE tokens")
    ax.set_title("Token length (Val >512: 2.69%)", loc="left", fontweight="bold")
    ax.legend(frameon=False)


def fig_cooccurrence() -> None:
    pairs = [
        ("Dizziness–Nausea", 1629),
        ("Vomiting–Dizziness", 1442),
        ("Vomiting–Abdominal pain", 1431),
        ("Vomiting–Nausea", 1291),
        ("Fever–Weakness", 937),
        ("Abdominal pain–Nausea", 843),
        ("Dizziness–Weakness", 773),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = range(len(pairs))
    ax.barh(list(y), [p[1] for p in pairs])
    ax.set_yticks(list(y), [p[0] for p in pairs])
    ax.invert_yaxis()
    ax.set_xlabel("Co-positive count (Training)")
    ax.set_title("Top co-occurrences", loc="left", fontweight="bold")


def fig_tokenizer() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    rows = [
        ("UNK rate (%)", [36.5, 0.05]),
        ("Val truncation (%)", [50.38, 5.38]),
        ("F1@0.5", [0.0, 0.5737]),
    ]
    for ax, (title, vals) in zip(axes, rows):
        ax.bar(["Before", "After"], vals, color=[C7, C0])
        ax.set_title(title)
        ymax = max(vals) * 1.25 if max(vals) else 1
        ax.set_ylim(0, ymax if ymax > 0 else 1)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("KoBERT tokenizer fix", fontweight="bold")


def fig_backbone() -> None:
    names = ["KoBERT\nfixed", "KoELECTRA", "KLUE-base", "KF-DeBERTa"]
    f1 = [0.5737, 0.5811, 0.6003, 0.6061]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(names, f1, color=[C7, C7, C0, C1])
    ax.set_ylim(0.54, 0.64)
    ax.set_ylabel("Macro F1@0.5")
    ax.set_title("Backbone comparison", loc="left", fontweight="bold")
    ax.text(3, 0.618, "186M; small gain vs cost\nkeep KLUE", ha="center", fontsize=9)
    for b, v in zip(bars, f1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)


def fig_posweight_macro() -> None:
    names = ["Plain BCE", "power=1", "power=0.5"]
    f1 = [0.5967, 0.6189, 0.6496]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.bar(names, f1, color=[C7, C1, C0])
    ax.set_ylim(0.56, 0.68)
    ax.set_ylabel("Macro F1@0.5")
    ax.set_title("Loss weighting (pos_weight)", loc="left", fontweight="bold")
    for b, v in zip(bars, f1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.4f}", ha="center")


def fig_posweight_per_class() -> None:
    plain = [0.687, 0.588, 0.518, 0.815, 0.645, 0.882, 0.032, 0.546, 0.660]
    p05 = [0.689, 0.590, 0.555, 0.819, 0.659, 0.884, 0.387, 0.590, 0.673]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    x = range(9)
    ax.bar([i - 0.18 for i in x], plain, 0.36, label="Plain BCE")
    ax.bar([i + 0.18 for i in x], p05, 0.36, label="power=0.5")
    ax.set_xticks(list(x), SYMPTOMS, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1@0.5")
    ax.set_title("Per-class F1@0.5", loc="left", fontweight="bold")
    ax.legend(frameon=False)


def fig_funnel() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    ax.set_title("Rejected vs adopted", loc="left", fontweight="bold")
    items = [
        (0.3, 3.4, 3.0, 1.25, "truncation\nno gain", C7, "white"),
        (3.7, 3.4, 3.0, 1.25, "pure-nausea sampling\n0.6003 -> 0.5855", C7, "white"),
        (7.1, 3.4, 3.4, 1.25, "structure / attention\nbelow baseline", C7, "white"),
        (2.4, 1.3, 6.2, 1.45, "Adopted: pos_weight + TF-IDF + TAPT/LLRD\nthreshold 0.5", C0, "white"),
    ]
    for x, y, w, h, t, c, tc in items:
        _box(ax, x, y, w, h, t, c, tc)
    ax.annotate("", xy=(5.5, 2.85), xytext=(5.5, 3.35), arrowprops=dict(arrowstyle="->", lw=1.8))


def fig_reject_adopt() -> None:
    labels = [
        "Head-tail / sliding",
        "Pure-nausea sampling",
        "Label attention",
        "pos_weight power=0.5",
        "TF-IDF (seed 42)",
        "TAPT20+LLRD0.8 single",
        "Submitted bundle",
    ]
    delta = [0.0, -0.0148, -0.0062, 0.0529, 0.0040, 0.0068, 0.0046]
    colors = [C7, C7, C7, C0, C1, C0, C0]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = range(len(labels))
    ax.barh(list(y), delta, color=colors)
    ax.set_yticks(list(y), labels)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("Delta Macro F1@0.5")
    ax.set_title("Rejected vs adopted", loc="left", fontweight="bold")
    ax.invert_yaxis()


def fig_tfidf_flow() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    ax.set_title("TF-IDF blend", loc="left", fontweight="bold")
    blocks = [
        (0.3, 2.3, 3.2, 1.25, "TEXT space-concat", C0, "white"),
        (4.0, 2.3, 3.4, 1.25, "KLUE (TAPT+LLRD)\nx 0.7", C0, "white"),
        (4.0, 0.35, 3.4, 1.25, "TF-IDF + LR\nx 0.3", C1, "white"),
        (8.2, 1.1, 2.9, 1.7, "blend >= 0.5\n9 labels", C0, "white"),
    ]
    for x, y, w, h, t, c, tc in blocks:
        _box(ax, x, y, w, h, t, c, tc)
    ax.annotate("", xy=(4.0, 2.9), xytext=(3.5, 2.9), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(4.0, 1.0), xytext=(1.9, 2.3), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(8.2, 2.3), xytext=(7.4, 2.9), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(8.2, 1.6), xytext=(7.4, 1.0), arrowprops=dict(arrowstyle="->", lw=1.5))


def fig_score_cost() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    pts = [
        (42, 0.6536, "pre-TAPT\nsingle+TF-IDF  0.6536", C7),
        (107, 0.6546, "pre-TAPT\n4-seed fp32  0.6546", C1),
        (42, 0.6593, "submitted\n4-seed fp16  0.6593", C0),
    ]
    for x, y, lab, c in pts:
        ax.scatter([x], [y], s=110, color=c, zorder=3)
        ax.annotate(lab, (x, y), xytext=(10, 10), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Val 3,640 inference time (s)")
    ax.set_ylabel("Macro F1@0.5")
    ax.set_title("Score vs latency", loc="left", fontweight="bold")
    ax.set_xlim(20, 130)
    ax.set_ylim(0.650, 0.664)


def fig_waterfall() -> None:
    labels = ["Plain", "pos_weight\n0.5", "+ TF-IDF", "pre-TAPT\n4-seed", "TAPT+LLRD\nsingle", "Submitted"]
    levels = [0.5967, 0.6496, 0.6536, 0.6546, 0.6543, 0.6593]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, levels, color=[C7, C0, C1, C7, C1, C0])
    ax.set_ylim(0.58, 0.68)
    ax.set_ylabel("Macro F1@0.5")
    ax.set_title("Performance path", loc="left", fontweight="bold")
    for i, v in enumerate(levels):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)


def fig_infer_opt() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.bar(["fp32", "fp16"], [107, 42], color=[C7, C0])
    ax.set_ylabel("Seconds (Val 3,640)")
    ax.set_title("Inference time")
    ax.set_ylim(0, 130)
    ax.text(0, 112, "101-113 s", ha="center", fontsize=9)
    ax.text(1, 47, "42 s", ha="center", fontsize=9)
    ax = axes[1]
    ax.bar(["Short-first", "Long-first"], [4.5, 0.9], color=[C7, C1])
    ax.set_ylabel("GPU reserved (GB)")
    ax.set_title("Memory")
    ax.set_ylim(0, 5.5)
    fig.suptitle("Inference optimization", fontweight="bold")


def _run() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = [
        fig_eval_flow,
        fig_class_counts,
        fig_label_cardinality,
        fig_token_length,
        fig_cooccurrence,
        fig_tokenizer,
        fig_backbone,
        fig_posweight_macro,
        fig_posweight_per_class,
        fig_funnel,
        fig_reject_adopt,
        fig_tfidf_flow,
        fig_score_cost,
        fig_waterfall,
        fig_infer_opt,
    ]
    names = [
        "01_eval_flow.png",
        "02_class_counts.png",
        "03_label_cardinality.png",
        "04_token_length.png",
        "05_cooccurrence_pairs.png",
        "06_tokenizer_fix.png",
        "07_backbone_f1.png",
        "08_posweight_macro.png",
        "09_posweight_per_class.png",
        "10_hypothesis_funnel.png",
        "11_reject_vs_adopt.png",
        "12_tfidf_blend_flow.png",
        "13_score_cost.png",
        "14_waterfall.png",
        "15_infer_opt.png",
    ]
    for name, fn in zip(names, mapping):
        fn()
        fig = plt.gcf()
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(name)
    print(f"-> {OUT}")


if __name__ == "__main__":
    _run()
