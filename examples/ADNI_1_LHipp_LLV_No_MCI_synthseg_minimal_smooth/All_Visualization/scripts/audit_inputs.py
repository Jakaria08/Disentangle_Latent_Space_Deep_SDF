#!/usr/bin/env python3
"""Audit all registered checkpoints, cached tables, and comparison inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import DEFAULT_REGISTRY, atomic_json, ensure_output_tree, output_root, registry, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def add(rows: list[dict[str, Any]], category: str, key: str, value: str | Path, kind: str) -> None:
    path = resolve(value)
    rows.append(
        {
            "category": category,
            "key": key,
            "kind": kind,
            "path": str(path),
            "exists": path.is_file() if kind == "file" else path.is_dir(),
        }
    )


def main() -> int:
    args = parse_args()
    config = registry(args.registry)
    root = output_root(config, args.output_root)
    ensure_output_tree(root)
    rows: list[dict[str, Any]] = []
    add(rows, "cohort", "master_manifest", config["cohort"]["master_manifest"], "file")
    for key, value in config["existing_caches"].items():
        add(rows, "cache", key, value, "directory")
    for method in config["latent_cocycle_methods"]:
        if "run_dir" in method:
            add(rows, "latent_cocycle", method["key"], method["run_dir"], "directory")
        for index, value in enumerate(method.get("run_dirs", []), start=1):
            add(rows, "latent_cocycle", f"{method['key']}_{index}", value, "directory")
    for key, method in config["pca_ode_baselines"]["methods"].items():
        run = resolve(method["run_dir"])
        add(rows, "pca_ode_baseline", f"{key}_run", run, "directory")
        add(rows, "pca_ode_baseline", f"{key}_checkpoint", run / "checkpoints" / "best.pt", "file")
    for method in config["direct_mesh_methods"]:
        add(rows, "direct_mesh", method["key"], method["checkpoint"], "file")
    for representation in config["inr_representations"]:
        run = resolve(representation["run_dir"])
        add(rows, "inr_representation", f"{representation['key']}_run", run, "directory")
        add(rows, "inr_representation", f"{representation['key']}_latents", run / representation["latent_export"] / "latents.pth", "file")
    add(rows, "brainode", "checkpoint", config["brainode"]["checkpoint"], "file")
    add(rows, "brainode", "comparison_cache", config["brainode"]["comparison_cache"], "directory")

    table = root / "tables" / "input_audit.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    missing = [row for row in rows if not row["exists"]]
    payload = {
        "schema_version": 1,
        "status": "complete" if not missing else "missing_inputs",
        "registered_inputs": len(rows),
        "present": len(rows) - len(missing),
        "missing": missing,
        "audit_table": str(table),
    }
    atomic_json(root / "input_audit.json", payload)
    print(json.dumps(payload, indent=2))
    return 1 if args.strict and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
