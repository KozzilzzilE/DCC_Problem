"""View bundled aggregate results; Python standard library only."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m2.recorded_results import load_snapshot, view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--format", choices=["text", "html"], default="text")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    snapshot = load_snapshot(args.results_dir)
    sections = [view(snapshot, s) for s in ("window", "architectures", "observations", "pilot200",
                "ablation200", "bandpass", "hypotheses", "hypothesis_seeds", "zcr", "zcr_reference", "ensemble",
                "final", "benchmark", "audit", "checkpoints")]
    if args.format == "html":
        body = "<!doctype html><meta charset='utf-8'><title>Mission 2 Recorded Results</title>" + "".join(s._repr_html_() for s in sections)
    else:
        body = "\n\n".join(map(str, sections))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
        print(f"Recorded results saved: {args.output}")
    else:
        print(body)


if __name__ == "__main__":
    main()
