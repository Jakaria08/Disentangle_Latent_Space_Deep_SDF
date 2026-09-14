#!/usr/bin/env python3
"""Execute a notebook in place using jupyter_client (nbconvert is not required)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from jupyter_client import KernelManager


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path, default=EXPERIMENT_DIR / "all_methods_longitudinal_analysis.ipynb")
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.notebook.expanduser().resolve()
    notebook = nbformat.read(path, as_version=4)
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/mpl-all-visualization-notebook")
    environment.setdefault("XDG_CACHE_HOME", "/tmp/all-visualization-notebook-xdg")
    manager = KernelManager(kernel_name=args.kernel)
    manager.start_kernel(cwd=str(path.parent), env=environment)
    client = manager.client()
    client.start_channels()
    error: RuntimeError | None = None
    try:
        client.wait_for_ready(timeout=args.timeout)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            cell.outputs = []
            cell.execution_count = None
            message_id = client.execute(cell.source, allow_stdin=False, stop_on_error=True)
            while True:
                message = client.get_iopub_msg(timeout=args.timeout)
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                kind = message["header"]["msg_type"]
                content = message["content"]
                if kind == "status" and content.get("execution_state") == "idle":
                    break
                if kind == "execute_input":
                    cell.execution_count = content.get("execution_count")
                elif kind == "stream":
                    cell.outputs.append(nbformat.v4.new_output("stream", name=content["name"], text=content["text"]))
                elif kind in {"display_data", "execute_result"}:
                    values = {"data": content.get("data", {}), "metadata": content.get("metadata", {})}
                    if kind == "execute_result":
                        values["execution_count"] = content.get("execution_count")
                    cell.outputs.append(nbformat.v4.new_output(kind, **values))
                elif kind == "error":
                    cell.outputs.append(nbformat.v4.new_output(
                        "error", ename=content.get("ename", "Error"),
                        evalue=content.get("evalue", ""), traceback=content.get("traceback", []),
                    ))
                    error = RuntimeError(f"Notebook cell {index} failed: {content.get('ename')}: {content.get('evalue')}")
                elif kind == "clear_output":
                    cell.outputs = []
            if error is not None:
                break
    finally:
        nbformat.write(notebook, path)
        client.stop_channels()
        manager.shutdown_kernel(now=True)
    if error is not None:
        raise error
    print(f"Executed {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
