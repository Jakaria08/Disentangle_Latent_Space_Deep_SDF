#!/usr/bin/env python3
"""Measure population latent drift from the pretrained source checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ablation_common import atomic_write_json, require_bulk_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_training_latents(path: Path) -> tuple[list[str], np.ndarray, int]:
    payload = torch.load(path, map_location="cpu")
    ids = list(payload.get("training_scan_ids", []))
    latents = payload.get("latent_codes")
    if not ids or not isinstance(latents, torch.Tensor) or latents.ndim != 2:
        raise ValueError(f"Checkpoint has no ordered training latent table: {path}")
    if len(ids) != len(set(ids)) or len(ids) != len(latents):
        raise ValueError(f"Invalid training latent IDs in {path}.")
    array = latents.detach().cpu().float().numpy().astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite latent values in {path}.")
    return ids, array, int(payload.get("epoch", 0))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    output = require_bulk_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    source_path = require_bulk_path(args.source, "source checkpoint")
    candidate_path = require_bulk_path(args.candidate, "candidate checkpoint")
    output = require_bulk_path(args.output_dir)
    source_ids, source, source_epoch = load_training_latents(source_path)
    candidate_ids, candidate, candidate_epoch = load_training_latents(candidate_path)
    if source_ids != candidate_ids:
        raise ValueError("Candidate training scan order differs from the pretrained source.")
    if source.shape != candidate.shape:
        raise ValueError(f"Latent shapes differ: {source.shape} versus {candidate.shape}.")
    difference = candidate - source
    displacement = np.linalg.norm(difference, axis=1)
    source_centered = source - source.mean(axis=0, keepdims=True)
    candidate_centered = candidate - candidate.mean(axis=0, keepdims=True)
    cross = candidate_centered.T @ source_centered
    u, _singular, vt = np.linalg.svd(cross, full_matrices=False)
    aligned = candidate_centered @ (u @ vt)
    procrustes_residual = np.linalg.norm(aligned - source_centered) / max(
        np.linalg.norm(source_centered), 1.0e-12
    )
    source_distances = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
    candidate_distances = np.linalg.norm(candidate[:, None, :] - candidate[None, :, :], axis=2)
    upper = np.triu_indices(len(source), k=1)
    try:
        from scipy.stats import spearmanr

        pairwise_spearman = float(spearmanr(source_distances[upper], candidate_distances[upper]).statistic)
    except ImportError:
        pairwise_spearman = float("nan")
    correlations = []
    for dimension in range(source.shape[1]):
        if source[:, dimension].std() == 0.0 or candidate[:, dimension].std() == 0.0:
            correlations.append(float("nan"))
        else:
            correlations.append(float(np.corrcoef(source[:, dimension], candidate[:, dimension])[0, 1]))
    finite_correlations = np.asarray([value for value in correlations if np.isfinite(value)])
    rows = [
        {
            "scan_id": scan_id,
            "latent_l2_displacement": float(displacement[index]),
            "source_latent_norm": float(np.linalg.norm(source[index])),
            "candidate_latent_norm": float(np.linalg.norm(candidate[index])),
        }
        for index, scan_id in enumerate(source_ids)
    ]
    report = {
        "source_checkpoint": str(source_path),
        "candidate_checkpoint": str(candidate_path),
        "source_epoch": source_epoch,
        "candidate_epoch": candidate_epoch,
        "latent_shape": list(source.shape),
        "training_scan_order_exact_match": True,
        "displacement_l2": {
            "mean": float(displacement.mean()),
            "median": float(np.median(displacement)),
            "p95": float(np.quantile(displacement, 0.95)),
            "max": float(displacement.max()),
        },
        "per_dimension_pearson": {
            "finite_dimensions": int(len(finite_correlations)),
            "mean": float(finite_correlations.mean()) if len(finite_correlations) else float("nan"),
            "median": float(np.median(finite_correlations)) if len(finite_correlations) else float("nan"),
            "minimum": float(finite_correlations.min()) if len(finite_correlations) else float("nan"),
        },
        "pairwise_distance_spearman": pairwise_spearman,
        "orthogonal_procrustes_relative_residual": float(procrustes_residual),
        "source_global_mean": float(source.mean()),
        "source_global_std": float(source.std()),
        "candidate_global_mean": float(candidate.mean()),
        "candidate_global_std": float(candidate.std()),
    }
    atomic_write_json(output / "latent_drift_summary.json", report)
    write_csv(output / "latent_displacement_per_scan.csv", rows)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
