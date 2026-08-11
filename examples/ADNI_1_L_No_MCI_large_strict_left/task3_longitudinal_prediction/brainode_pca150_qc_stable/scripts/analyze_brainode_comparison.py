#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core_brainode_common import TASK_DIR, load_config, resolve_repo_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize BrainODE evaluation metrics and compare them with SIREN flow summaries."
    )
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "core_brainode.json"),
    )
    parser.add_argument(
        "--evaluation-dir",
        default=None,
        help="Directory containing *_trajectory_metrics.csv from evaluate_brainode.py.",
    )
    parser.add_argument(
        "--siren-analysis-dir",
        default=None,
        help="SIREN flow analysis/checkpoint_best directory with *_summary.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for BrainODE and comparison summaries.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def median(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.median(values)) if values else float("nan")


def derive_fields(row: dict[str, str]) -> dict[str, Any]:
    source_condition = as_float(row, "source_condition")
    target_condition = as_float(row, "target_condition")
    length = int(float(row["length"]))
    return {
        **row,
        "diagnosis": "AD" if source_condition >= 0.5 else "CN",
        "pair_type": "adjacent" if length == 2 else "nonadjacent",
        "condition_change": abs(target_condition - source_condition) > 1.0e-6,
        "endpoint_vertex_mae": as_float(row, "endpoint_vertex_mae"),
        "no_change_vertex_mae": as_float(row, "no_change_vertex_mae"),
        "endpoint_vertex_mae_improvement": as_float(
            row, "endpoint_vertex_mae_improvement"
        ),
        "endpoint_pca_mse": as_float(row, "endpoint_pca_mse"),
        "no_change_pca_mse": as_float(row, "no_change_pca_mse"),
        "endpoint_pca_improvement": as_float(row, "endpoint_pca_improvement"),
        "gap_norm": as_float(row, "gap_norm"),
    }


def summarize_group(
    rows: list[dict[str, Any]],
    grouping: str,
    *,
    split: str = "",
    diagnosis: str = "",
    pair_type: str = "",
    record_set: str = "",
) -> dict[str, Any]:
    beats = [
        1.0 if row["endpoint_vertex_mae"] < row["no_change_vertex_mae"] else 0.0
        for row in rows
    ]
    return {
        "grouping": grouping,
        "split": split,
        "diagnosis": diagnosis,
        "pair_type": pair_type,
        "record_set": record_set,
        "rows": len(rows),
        "brainode_beats_no_change_fraction": mean(beats),
        "endpoint_vertex_mae_mean": mean(row["endpoint_vertex_mae"] for row in rows),
        "endpoint_vertex_mae_median": median(row["endpoint_vertex_mae"] for row in rows),
        "no_change_vertex_mae_mean": mean(row["no_change_vertex_mae"] for row in rows),
        "endpoint_vertex_mae_improvement_mean": mean(
            row["endpoint_vertex_mae_improvement"] for row in rows
        ),
        "endpoint_pca_mse_mean": mean(row["endpoint_pca_mse"] for row in rows),
        "no_change_pca_mse_mean": mean(row["no_change_pca_mse"] for row in rows),
        "endpoint_pca_improvement_mean": mean(
            row["endpoint_pca_improvement"] for row in rows
        ),
        "gap_norm_mean": mean(row["gap_norm"] for row in rows),
        "condition_change_rows": sum(1 for row in rows if row["condition_change"]),
    }


def grouped(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    result: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(str(row[key]) for key in keys)].append(row)
    return dict(result)


def brainode_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    summaries.append(summarize_group(rows, "overall"))
    for (split,), split_rows in sorted(grouped(rows, ("split",)).items()):
        summaries.append(summarize_group(split_rows, "split", split=split))
    for (diagnosis,), diagnosis_rows in sorted(grouped(rows, ("diagnosis",)).items()):
        summaries.append(
            summarize_group(diagnosis_rows, "diagnosis", diagnosis=diagnosis)
        )
    for (pair_type,), pair_rows in sorted(grouped(rows, ("pair_type",)).items()):
        summaries.append(summarize_group(pair_rows, "pair_type", pair_type=pair_type))
    for (record_set,), record_rows in sorted(grouped(rows, ("record_set",)).items()):
        summaries.append(
            summarize_group(record_rows, "record_set", record_set=record_set)
        )
    for (split, diagnosis), split_dx_rows in sorted(
        grouped(rows, ("split", "diagnosis")).items()
    ):
        summaries.append(
            summarize_group(
                split_dx_rows,
                "split_diagnosis",
                split=split,
                diagnosis=diagnosis,
            )
        )
    for (diagnosis, pair_type), dx_pair_rows in sorted(
        grouped(rows, ("diagnosis", "pair_type")).items()
    ):
        summaries.append(
            summarize_group(
                dx_pair_rows,
                "diagnosis_pair_type",
                diagnosis=diagnosis,
                pair_type=pair_type,
            )
        )
    for (split, diagnosis, pair_type), rows_for_key in sorted(
        grouped(rows, ("split", "diagnosis", "pair_type")).items()
    ):
        summaries.append(
            summarize_group(
                rows_for_key,
                "split_diagnosis_pair_type",
                split=split,
                diagnosis=diagnosis,
                pair_type=pair_type,
            )
        )
    return summaries


def load_brainode_rows(evaluation_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(evaluation_dir.glob("*_trajectory_metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_trajectory_metrics.csv files in {evaluation_dir}")
    rows: list[dict[str, Any]] = []
    seen = set()
    for path in paths:
        for row in read_csv(path):
            key = (
                row["split"],
                row["record_set"],
                row["subject_id"],
                row["start_visit_order"],
                row["length"],
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(derive_fields(row))
    return rows


def load_siren_rows(siren_dir: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    result: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for split in ("train", "val", "test"):
        path = siren_dir / f"{split}_summary.csv"
        if not path.is_file():
            continue
        for row in read_csv(path):
            result[
                (
                    split,
                    row.get("grouping", ""),
                    row.get("diagnosis", ""),
                    row.get("pair_type", ""),
                )
            ] = row
    return result


def compare_to_siren(
    brainode_summary: list[dict[str, Any]],
    siren_rows: dict[tuple[str, str, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    supported_groupings = {"split", "split_diagnosis", "split_diagnosis_pair_type"}
    for row in brainode_summary:
        if row["grouping"] not in supported_groupings:
            continue
        if not row["split"]:
            continue
        siren_grouping = {
            "split": "overall",
            "split_diagnosis": "diagnosis",
            "split_diagnosis_pair_type": "diagnosis_pair_type",
        }[row["grouping"]]
        siren = siren_rows.get(
            (
                row["split"],
                siren_grouping,
                row["diagnosis"],
                row["pair_type"],
            )
        )
        if siren is None:
            continue
        comparisons.append(
            {
                "split": row["split"],
                "grouping": row["grouping"],
                "diagnosis": row["diagnosis"],
                "pair_type": row["pair_type"],
                "brainode_rows": row["rows"],
                "brainode_beats_no_change_fraction": row[
                    "brainode_beats_no_change_fraction"
                ],
                "brainode_endpoint_vertex_mae_mean": row["endpoint_vertex_mae_mean"],
                "brainode_no_change_vertex_mae_mean": row["no_change_vertex_mae_mean"],
                "brainode_vertex_mae_improvement_mean": row[
                    "endpoint_vertex_mae_improvement_mean"
                ],
                "siren_rows": siren.get("rows", ""),
                "siren_model_beats_no_change_fraction": siren.get(
                    "model_beats_no_change_fraction", ""
                ),
                "siren_composed_beats_no_change_fraction": siren.get(
                    "composed_beats_no_change_fraction", ""
                ),
                "siren_model_target_sdf_l1_mean": siren.get(
                    "model_target_sdf_l1_mean", ""
                ),
                "siren_no_change_target_sdf_l1_mean": siren.get(
                    "no_change_target_sdf_l1_mean", ""
                ),
                "siren_sdf_l1_improvement_mean": siren.get(
                    "sdf_l1_improvement_mean", ""
                ),
                "metric_note": "BrainODE uses PCA inverse-transform vertex MAE; SIREN flow uses decoder SDF L1.",
            }
        )
    return comparisons


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    run_name = str(training["run_name"])
    output_root = resolve_repo_path(training["output_root"])
    evaluation_dir = (
        Path(args.evaluation_dir).expanduser()
        if args.evaluation_dir
        else output_root / run_name / "evaluation" / "best"
    )
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else TASK_DIR / "analysis" / "brainode_comparison"
    )
    siren_dir = (
        Path(args.siren_analysis_dir).expanduser()
        if args.siren_analysis_dir
        else resolve_repo_path(config["analysis"]["siren_flow_analysis_dir"])
    )

    rows = load_brainode_rows(evaluation_dir)
    summary_rows = brainode_summaries(rows)
    comparison_rows = compare_to_siren(summary_rows, load_siren_rows(siren_dir))

    write_csv(output_dir / "brainode_summary.csv", summary_rows)
    write_csv(output_dir / "brainode_vs_siren_flow.csv", comparison_rows)
    report = {
        "evaluation_dir": str(evaluation_dir),
        "siren_analysis_dir": str(siren_dir),
        "brainode_metric_rows": len(rows),
        "brainode_summary_rows": len(summary_rows),
        "comparison_rows": len(comparison_rows),
        "outputs": {
            "brainode_summary_csv": str(output_dir / "brainode_summary.csv"),
            "brainode_vs_siren_flow_csv": str(output_dir / "brainode_vs_siren_flow.csv"),
        },
    }
    write_json(output_dir / "analysis_summary.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
