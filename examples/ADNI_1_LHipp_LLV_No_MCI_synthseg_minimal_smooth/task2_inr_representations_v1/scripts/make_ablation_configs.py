#!/usr/bin/env python3
"""Materialize named shared-grid ablations from one reviewed base config."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from shared_grid_common import load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_nested(config: dict, dotted_key: str, value) -> None:
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def main() -> None:
    args = parse_args()
    base = load_json(args.base)
    matrix = load_json(args.matrix)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for experiment in matrix["experiments"]:
        config = copy.deepcopy(base)
        suffix = experiment["suffix"]
        config["name"] = f"{base['name']}__{suffix}"
        config["description"] = experiment["description"]
        config["output_dir"] = f"{base['output_dir']}__{suffix}"
        for key, value in experiment["overrides"].items():
            set_nested(config, key, value)
        destination = output_dir / f"{suffix}.json"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Config exists; pass --overwrite: {destination}")
        write_json(destination, config)
        print(destination)


if __name__ == "__main__":
    main()
