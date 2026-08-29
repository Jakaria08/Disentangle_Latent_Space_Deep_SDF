#!/usr/bin/env python3
"""Execute notebook code cells headlessly against completed evaluation artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/task5-direct-xdg-cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import nbformat


TASK = Path(__file__).resolve().parents[1]
NOTEBOOK = TASK / "notebooks" / "direct_mesh_results.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spiral-run", default="direct_mesh_spiral_optuna_main_v1_s42")
    parser.add_argument("--adaptive-run", default="direct_mesh_adaptive_optuna_main_v1_s42")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["DIRECT_MESH_SPIRAL_RUN"] = args.spiral_run
    os.environ["DIRECT_MESH_ADAPTIVE_RUN"] = args.adaptive_run
    os.environ["DIRECT_MESH_SPLIT"] = args.split
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "__notebook_check__"}
    count = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        exec(compile(cell.source, f"{NOTEBOOK}:cell-{index}", "exec"), namespace)
        plt.close("all")
        count += 1
    print(f"PASS: executed {count} code cells from {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
