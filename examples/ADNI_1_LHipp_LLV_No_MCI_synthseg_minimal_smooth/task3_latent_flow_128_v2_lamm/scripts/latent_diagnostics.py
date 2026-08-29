#!/usr/bin/env python3
"""Train/validation latent flowability audit, including LAMM scale blocks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import activate

activate()
import common as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", action="append", default=[])
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def effective_rank(latent: np.ndarray) -> float:
    singular = np.linalg.svd(latent - latent.mean(axis=0), compute_uv=False)
    energy = np.square(singular)
    probabilities = energy / max(float(energy.sum()), 1.0e-12)
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def block_metrics(latent: np.ndarray, archive: dict[str, np.ndarray], first: int, last: int) -> dict[str, Any]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    times = archive["visit_time_years_from_baseline"].astype(np.float64)
    diagnoses = archive["subject_diagnoses"].astype(str)
    displacements: dict[str, list[float]] = {"CN": [], "AD": [], "overall": []}
    velocity_cosines: list[float] = []
    for subject_index in range(len(offsets) - 1):
        start, stop = int(offsets[subject_index]), int(offsets[subject_index + 1])
        z = latent[start:stop, first:last]
        dt = np.diff(times[start:stop])
        if np.any(dt <= 0.0):
            raise ValueError("Visits must be strictly chronological for latent diagnostics")
        velocity = np.diff(z, axis=0) / np.maximum(dt[:, None], 1.0e-8)
        speed = np.sqrt(np.mean(np.square(velocity), axis=1))
        diagnosis = str(diagnoses[subject_index])
        displacements[diagnosis].extend(speed.tolist())
        displacements["overall"].extend(speed.tolist())
        if len(velocity) > 1:
            numerator = np.sum(velocity[:-1] * velocity[1:], axis=1)
            denominator = np.linalg.norm(velocity[:-1], axis=1) * np.linalg.norm(velocity[1:], axis=1)
            valid = denominator > 1.0e-12
            velocity_cosines.extend((numerator[valid] / denominator[valid]).tolist())
    subset = latent[:, first:last]
    std = subset.std(axis=0)
    return {
        "dimensions": int(last - first),
        "effective_rank": effective_rank(subset),
        "near_constant_dimensions": int(np.sum(std < 1.0e-4)),
        "standard_deviation_median": float(np.median(std)),
        "standardized_rms_velocity_per_year": {
            group: float(np.mean(values)) if values else None for group, values in displacements.items()
        },
        "consecutive_velocity_cosine_mean": float(np.mean(velocity_cosines)) if velocity_cosines else None,
    }


def main() -> int:
    args = parse_args()
    registry = C.load_registry()
    names = args.representation or list(registry["representations"])
    result: dict[str, Any] = {"split": args.split, "representations": {}}
    for name in names:
        archive = C.load_archive(name, args.split, registry)
        latent = archive["visit_latent_standardized_128"].astype(np.float64)
        offsets = archive.get("latent_scale_offsets_128", np.asarray([0, C.LATENT_DIM]))
        scale_names = archive.get("latent_scale_names_128", np.asarray(["full"])).astype(str)
        if len(scale_names) != len(offsets) - 1:
            raise ValueError(f"Invalid scale metadata in prepared {name} archive")
        blocks = {
            str(scale_names[index]): block_metrics(latent, archive, int(first), int(last))
            for index, (first, last) in enumerate(zip(offsets[:-1], offsets[1:]))
        }
        result["representations"][name] = {
            "visits": len(latent),
            "full": block_metrics(latent, archive, 0, C.LATENT_DIM),
            "scale_blocks": blocks,
        }
    C.assert_finite_mapping(result)
    if args.output is None:
        import json

        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        destination = C.require_bulk_path(args.output, "latent diagnostics output")
        if destination.exists():
            raise FileExistsError(destination)
        C.atomic_json(destination, result)
        print(f"WROTE {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
