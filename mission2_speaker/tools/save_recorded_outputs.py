"""Execute only tagged result-display cells and save their actual outputs.

This is a small standard-library executor for the bundled display cells, not a
Jupyter kernel and not a training/evaluation notebook executor.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def save_outputs(notebook_path, results_dir):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"RECORDED_RESULTS_OVERRIDE": str(results_dir.resolve())}
    count = 0
    original_cwd = Path.cwd()
    sys.path.insert(0, str(ROOT))
    try:
        os.chdir(ROOT)
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code" or "recorded-results" not in cell.get("metadata", {}).get("tags", []):
                continue
            code = "".join(cell["source"])
            tree = ast.parse(code)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            count += 1
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exec(compile(tree, str(notebook_path), "exec"), namespace)
                result = eval(compile(ast.Expression(last.value), str(notebook_path), "eval"), namespace) if last else None
            if namespace.get("RUN_STAGE") != "recorded_results":
                raise ValueError("Tagged display cells must keep RUN_STAGE='recorded_results'")
            outputs = []
            if stdout.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue().splitlines(keepends=True)})
            if result is not None:
                data = {"text/plain": [repr(result)]}
                if hasattr(result, "_repr_html_"):
                    data["text/html"] = [result._repr_html_()]
                outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": data})
            cell["execution_count"] = count
            cell["outputs"] = outputs
        if count == 0:
            raise ValueError("No recorded-results cells found")
        notebook.setdefault("metadata", {})["recorded_output_capture"] = {
            "executor": "tools/save_recorded_outputs.py (Python AST/standard library; no Jupyter kernel)",
            "scope": "Reading recorded aggregate artifacts only; no training, GPU, audio inference or official evaluation",
            "executed_display_cells": count,
        }
        temp = notebook_path.with_suffix(".ipynb.tmp")
        temp.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        temp.replace(notebook_path)
    finally:
        os.chdir(original_cwd)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "Mission2_Speaker_Classification.ipynb")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    count = save_outputs(args.notebook.resolve(), args.results_dir)
    print(f"Saved {count} recorded-result display cells: {args.notebook}")


if __name__ == "__main__":
    main()
