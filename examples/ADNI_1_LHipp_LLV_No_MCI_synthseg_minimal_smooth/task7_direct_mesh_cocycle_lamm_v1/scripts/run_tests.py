#!/usr/bin/env python3
"""Dependency-free runner for this task's assertion-style unit tests."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = TASK_ROOT / "tests"


def main() -> int:
    for path in (TEST_ROOT, TASK_ROOT / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    failures = []
    passed = []
    started = time.time()
    for path in sorted(TEST_ROOT.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            identifier = f"{path.name}::{name}"
            try:
                function()
                passed.append(identifier)
                print(f"PASS {identifier}", flush=True)
            except Exception as error:  # noqa: BLE001
                failures.append(
                    {"test": identifier, "error": f"{type(error).__name__}: {error}"}
                )
                print(
                    f"FAIL {identifier}: {type(error).__name__}: {error}", flush=True
                )
    report = {
        "status": "passed" if not failures else "failed",
        "passed": len(passed),
        "failed": len(failures),
        "elapsed_seconds": time.time() - started,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

