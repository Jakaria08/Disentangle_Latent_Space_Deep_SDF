#!/usr/bin/env python3
"""Execute every code cell in one process as a dependency/plot smoke check.

This is not a replacement for an interactive notebook frontend and does not
save cell outputs. It provides a headless check on systems without nbconvert.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


TASK = Path(__file__).resolve().parents[1]
NOTEBOOK = TASK / "notebooks" / "velocity_reference_audit.ipynb"


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-task4-velocity")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-task4-velocity")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {}
    old_directory = Path.cwd()
    try:
        os.chdir(TASK.parents[2])
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            print(f"executing cell {index}", flush=True)
            exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(old_directory)
    print("all notebook code cells passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
