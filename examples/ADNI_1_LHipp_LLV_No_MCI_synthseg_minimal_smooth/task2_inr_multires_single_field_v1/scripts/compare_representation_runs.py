#!/usr/bin/env python3
"""Paired comparison of Compact-style and multiresolution INR mesh metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from multires_common import require_bulk_path, stable_seed, write_csv, write_json


METRICS = ("assd_mm", "hd95_mm", "chamfer_l1_mm", "fscore_0_5mm", "fscore_1mm", "volume_relative_error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Baseline per_scan_metrics.csv (read-only; SSD is allowed).")
    parser.add_argument("--candidate", required=True, help="Candidate per_scan_metrics.csv on bulk disk.")
    parser.add_argument("--output-dir", required=True, help="Must be below /mnt/bulk10tb.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_inr(path: str) -> dict[tuple[str, str], dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["scan_id"], row["split"]): row for row in rows if row.get("method", "inr") == "inr"}


def cluster_bootstrap(
    values: np.ndarray, cluster_ids: np.ndarray, seed: int, repeats: int = 5000
) -> list[float]:
    rng = np.random.default_rng(seed)
    clusters = np.unique(cluster_ids)
    means = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = clusters[rng.integers(0, len(clusters), size=len(clusters))]
        positions = np.concatenate(
            [np.flatnonzero(cluster_ids == cluster) for cluster in sampled]
        )
        means[index] = values[positions].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    args = parse_args()
    output = require_bulk_path(args.output_dir)
    baseline = read_inr(args.baseline)
    candidate_path = require_bulk_path(args.candidate, "candidate metrics")
    candidate = read_inr(str(candidate_path))
    keys = sorted(set(baseline).intersection(candidate))
    if not keys:
        raise RuntimeError("No paired INR scan IDs were found.")
    rows = []
    summary = {}
    for split in ("train", "val", "test"):
        split_keys = [key for key in keys if key[1] == split]
        summary[split] = {}
        for metric in METRICS:
            differences = np.asarray(
                [float(candidate[key][metric]) - float(baseline[key][metric]) for key in split_keys],
                dtype=np.float64,
            )
            if not len(differences):
                continue
            cluster_ids = np.asarray(
                [candidate[key].get("subject_id", key[0]) for key in split_keys]
            )
            summary[split][metric] = {
                "count": len(differences),
                "candidate_minus_baseline_mean": float(differences.mean()),
                "candidate_minus_baseline_median": float(np.median(differences)),
                "subject_cluster_bootstrap_mean_95ci": cluster_bootstrap(
                    differences, cluster_ids, stable_seed(split + metric, args.seed)
                ),
            }
        for key in split_keys:
            row = {
                "scan_id": key[0],
                "subject_id": candidate[key].get("subject_id", key[0]),
                "split": split,
            }
            for metric in METRICS:
                row[f"baseline_{metric}"] = float(baseline[key][metric])
                row[f"candidate_{metric}"] = float(candidate[key][metric])
                row[f"candidate_minus_baseline_{metric}"] = row[f"candidate_{metric}"] - row[f"baseline_{metric}"]
            rows.append(row)
    write_csv(output / "paired_per_scan.csv", rows)
    write_json(
        output / "summary.json",
        {
            "baseline": str(Path(args.baseline).resolve()),
            "candidate": str(candidate_path),
            "definition": "candidate minus baseline; negative favors candidate for distance/error metrics and positive favors candidate for F-scores",
            "summary": summary,
        },
    )


if __name__ == "__main__":
    main()
