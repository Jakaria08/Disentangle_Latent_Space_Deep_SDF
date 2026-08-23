#!/usr/bin/env python3
"""Collect the completed 3x3 matrix into one comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import common as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--run", action="append", type=Path, required=True, help="Repeat for the nine selected run directories.")
    parser.add_argument("--evaluation-name", default="evaluation")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Compare a selected subset instead of requiring the complete 3x3 matrix.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def companion_path(output: Path, label: str) -> Path:
    return output.with_name(f"{output.stem}_{label}{output.suffix}")


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    metadata = ["representation", "method", "split", "stratum", "diagnosis", "metric"]
    fieldnames = [key for key in metadata if any(key in row for row in rows)]
    fieldnames.extend(sorted({key for row in rows for key in row}.difference(fieldnames)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    C.validate_run_name(args.evaluation_name)
    rows = []
    pair_rows = []
    sequence_rows = []
    bootstrap_rows = []
    seen = set()
    for run in args.run:
        path = run.expanduser().resolve() / args.evaluation_name / args.split / "summary.json"
        report = C.read_json(path)
        if report["split"] != args.split:
            raise ValueError(f"Requested {args.split} but {path} contains {report['split']}")
        if bool(report.get("test_loaded_during_training", False)):
            raise ValueError(f"Training/test leakage recorded in {path}")
        key = (report["representation"], report["method"])
        if key in seen:
            raise ValueError(f"Duplicate representation/method: {key}")
        seen.add(key)
        metrics = report["pair_metrics"]["first_last_forward"]["groups"]["overall"]
        all_metrics = report["pair_metrics"]["all_forward"]["groups"]["overall"]
        defects = report["consistency_defects"]
        floor = report["representation_floor"]
        rows.append({
            "representation": report["representation"],
            "method": report["method"],
            "split": args.split,
            "representation_floor_rmse_mm": floor["coordinate_rmse_mm_mean"],
            "first_last_transport_coordinate_mae_mm": metrics["coordinate_mean"],
            "first_last_transport_euclidean_mm": metrics["euclidean_mean"],
            "first_last_end_to_end_rmse_mm": metrics["end_to_end_coordinate_rmse_mean"],
            "first_last_volume_relative": metrics["volume_relative_mean"],
            "first_last_nochange_coordinate_mae_mm": metrics["nochange_coordinate_mean"],
            "all_pair_transport_coordinate_mae_mm": all_metrics["coordinate_mean"],
            "all_pair_end_to_end_rmse_mm": all_metrics["end_to_end_coordinate_rmse_mean"],
            "semigroup_defect_mean": defects["relative_semigroup_defect_mean"],
            "inverse_defect_mean": defects["relative_inverse_defect_mean"],
            "checkpoint": report["checkpoint"],
        })
        for stratum, result in sorted(report["pair_metrics"].items()):
            for diagnosis, metrics_by_group in sorted(result["groups"].items()):
                if int(metrics_by_group.get("rows", 0)) == 0:
                    continue
                pair_rows.append({
                    "representation": report["representation"],
                    "method": report["method"],
                    "split": args.split,
                    "stratum": stratum,
                    "diagnosis": diagnosis,
                    **metrics_by_group,
                })
        sequences = report["sequence_metrics"]
        sequence_rows.append({
            "representation": report["representation"],
            "method": report["method"],
            "split": args.split,
            "diagnosis": "overall",
            "subjects": sequences["subjects"],
            **sequences["means"],
        })
        for diagnosis, metrics_by_group in sorted(sequences["by_diagnosis"].items()):
            sequence_rows.append({
                "representation": report["representation"],
                "method": report["method"],
                "split": args.split,
                "diagnosis": diagnosis,
                **metrics_by_group,
            })
        bootstrap = report.get("subject_bootstrap", {})
        for metric, interval in sorted(bootstrap.get("metrics", {}).items()):
            bootstrap_rows.append({
                "representation": report["representation"],
                "method": report["method"],
                "split": args.split,
                "metric": metric,
                "samples": bootstrap["samples"],
                "unit": bootstrap["unit"],
                **interval,
            })
    expected = {(representation, method) for representation in ("pca128", "spiralnet128", "adaptive128") for method in ("plain_ode", "brainode", "direct_c4")}
    if not args.allow_partial and seen != expected:
        raise ValueError(f"Expected full 3x3 matrix; missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    if args.allow_partial and len(seen) < 2:
        raise ValueError("A partial comparison requires at least two distinct representation/method runs")
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".csv":
        raise ValueError("--output must end in .csv")
    destinations = {
        "summary": output,
        "pairs": companion_path(output, "pairs"),
        "sequences": companion_path(output, "sequences"),
        "bootstrap": companion_path(output, "bootstrap"),
        "json": output.with_suffix(".json"),
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite {existing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(destinations["summary"], sorted(rows, key=lambda row: (row["representation"], row["method"])))
    write_rows(destinations["pairs"], sorted(pair_rows, key=lambda row: (row["representation"], row["method"], row["stratum"], row["diagnosis"])))
    write_rows(destinations["sequences"], sorted(sequence_rows, key=lambda row: (row["representation"], row["method"], row["diagnosis"])))
    write_rows(destinations["bootstrap"], sorted(bootstrap_rows, key=lambda row: (row["representation"], row["method"], row["metric"])))
    C.atomic_json(destinations["json"], {
        "split": args.split,
        "summary_rows": rows,
        "pair_rows": pair_rows,
        "sequence_rows": sequence_rows,
        "bootstrap_rows": bootstrap_rows,
    })
    print("WROTE " + ", ".join(str(path) for path in destinations.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
