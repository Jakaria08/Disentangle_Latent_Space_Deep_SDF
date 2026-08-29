#!/usr/bin/env python3
"""Add scalar secondary tables to an already completed inference cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import build_analysis_cache as B


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    root = parser.parse_args().cache_dir.expanduser().resolve()
    manifest = B.read_json(root / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Cache is not complete")
    endpoint = pd.read_csv(root / "tables" / "current_endpoint_metrics.csv")
    registry = B.read_json(B.REGISTRY_PATH)
    B.write_csv(root / "tables" / "current_paired_tests.csv", B.paired_tests(endpoint))
    B.write_csv(root / "tables" / "current_consistency.csv", B.consistency_rows(registry))
    manifest["files"] = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name != "manifest.json")
    B.atomic_json(root / "manifest.json", manifest)
    print(f"POSTPROCESSED {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
