#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from task2_common import TASK_DIR, load_config, summarize_values, write_json


METRICS = (
    "assd",
    "chamfer_l2_squared",
    "hd95",
    "volume_relative_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paired validation-set comparisons for configured INR decoders."
    )
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "pipeline.json"),
    )
    parser.add_argument(
        "--metrics-csv",
        default=str(
            TASK_DIR / "evaluation" / "metrics" / "reconstruction_per_scan.csv"
        ),
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated INR method names; defaults to all configured INR models.",
    )
    return parser.parse_args()


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted_sorted = [0.0] * len(p_values)
    total = len(p_values)
    running = 0.0
    for rank, original_index in enumerate(order):
        factor = total - rank
        running = max(running, min(1.0, factor * p_values[original_index]))
        adjusted_sorted[rank] = running
    adjusted = [0.0] * len(p_values)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    configured_models = list(config.get("inr_models", {}).keys())
    if args.models:
        models = [value.strip() for value in args.models.split(",") if value.strip()]
    else:
        models = configured_models
    unknown = set(models).difference(configured_models)
    if unknown:
        raise ValueError(f"Unknown INR models: {sorted(unknown)}")
    if len(models) < 2:
        raise ValueError("At least two INR models are required for comparison.")
    metrics_path = Path(args.metrics_csv)
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validation = [
        row
        for row in rows
        if row["split"] == "val" and row["method"] in models
    ]
    by_model = {
        model: {row["scan_id"]: row for row in validation if row["method"] == model}
        for model in models
    }
    paired_ids = sorted(
        set.intersection(*(set(by_model[model]) for model in models))
    )
    if not paired_ids:
        raise RuntimeError("No paired validation rows exist for the INR models.")

    report = {
        "selection_split": "val",
        "paired_scan_count": len(paired_ids),
        "models_compared": models,
        "models": {},
        "pairwise_tests": {},
        "selection_note": (
            "Lower is better for every reported metric. Use validation results for "
            "selection; test metrics must remain reporting-only."
        ),
    }
    table_rows = []
    for model in models:
        report["models"][model] = {}
        for metric in METRICS:
            values = [float(by_model[model][scan_id][metric]) for scan_id in paired_ids]
            stats = summarize_values(values)
            report["models"][model][metric] = stats
            table_rows.append({"model": model, "metric": metric, **stats})

    model_pairs = list(itertools.combinations(models, 2))
    for metric in METRICS:
        pair_reports = []
        raw_p_values = []
        for first_model, second_model in model_pairs:
            first = np.asarray(
                [float(by_model[first_model][scan_id][metric]) for scan_id in paired_ids]
            )
            second = np.asarray(
                [float(by_model[second_model][scan_id][metric]) for scan_id in paired_ids]
            )
            difference = second - first
            if np.allclose(difference, 0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(second, first, zero_method="wilcox")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            raw_p_values.append(p_value)
            pair_reports.append(
                {
                    "model_a": first_model,
                    "model_b": second_model,
                    "difference_definition": f"{second_model} - {first_model}",
                    "difference": summarize_values(difference),
                    f"{first_model}_lower_count": int(np.sum(first < second)),
                    f"{second_model}_lower_count": int(np.sum(second < first)),
                    "tie_count": int(np.sum(np.isclose(first, second))),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value": p_value,
                }
            )
        adjusted = holm_adjust(raw_p_values)
        report["pairwise_tests"][metric] = {}
        for pair_report, adjusted_p in zip(pair_reports, adjusted):
            pair_report["holm_adjusted_p_value"] = adjusted_p
            pair_key = f"{pair_report['model_a']}__vs__{pair_report['model_b']}"
            report["pairwise_tests"][metric][pair_key] = pair_report

    medians = {
        model: report["models"][model]["assd"]["median"] for model in models
    }
    report["validation_median_assd_rank"] = [
        {"model": model, "median_assd": medians[model]}
        for model in sorted(medians, key=medians.get)
    ]
    report["lower_validation_median_assd"] = min(medians, key=medians.get)
    output_dir = TASK_DIR / "evaluation" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "inr_model_comparison.json", report)
    with (output_dir / "inr_model_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
