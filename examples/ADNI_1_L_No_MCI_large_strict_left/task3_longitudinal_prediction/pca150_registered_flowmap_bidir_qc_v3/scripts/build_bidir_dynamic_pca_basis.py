#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bidir_registered_pca_utils import (
    DEFAULT_EXPERIMENT_NAME,
    TASK_DIR,
    build_pair_records,
    decode_pca_np,
    experiment_root,
    load_pca_model,
    load_split_archive,
    mesh_volume_np,
    pca_latents,
    vertex_normals_np,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build longitudinal dynamic PCA bases from annualized training displacements."
    )
    parser.add_argument("--config", default=str(TASK_DIR / "configs" / "core_brainode.json"))
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--components", type=int, default=150)
    parser.add_argument("--dynamic-dims", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--pair-type", choices=("all", "adjacent", "nonadjacent"), default="adjacent")
    parser.add_argument("--max-gap-years", type=float, default=0.0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0.0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
    }


def main() -> int:
    args = parse_args()
    _, pca_model_dir, mean_flat, components, faces = load_pca_model(args.config, int(args.components))
    train_archive = load_split_archive("train")
    latents = pca_latents(train_archive, int(args.components))
    records = build_pair_records(
        train_archive,
        max_gap_years=float(args.max_gap_years),
        pair_type=str(args.pair_type),
    )
    if not records:
        raise RuntimeError("No training pair records were available for dynamic PCA.")

    velocities: list[np.ndarray] = []
    labels: list[int] = []
    log_volume_rates: dict[str, list[float]] = defaultdict(list)
    relative_volume_rates: dict[str, list[float]] = defaultdict(list)
    normal_rates_by_diag: dict[str, list[np.ndarray]] = defaultdict(list)
    template_normals = vertex_normals_np(mean_flat.reshape(-1, 3), faces)

    for record in records:
        source = latents[record.source_index]
        target = latents[record.target_index]
        delta_years = max(float(record.delta_years), 1.0e-6)
        velocity = (target - source) / delta_years
        velocities.append(velocity.astype(np.float32))
        labels.append(int(record.label_ad))

        vertices = decode_pca_np(np.stack([source, target], axis=0), mean_flat, components)
        volumes = mesh_volume_np(vertices, faces)
        log_rate = (np.log(max(float(volumes[1]), 1.0e-8)) - np.log(max(float(volumes[0]), 1.0e-8))) / delta_years
        relative_rate = (float(volumes[1]) - float(volumes[0])) / max(float(volumes[0]), 1.0e-8) / delta_years
        log_volume_rates[record.diagnosis].append(float(log_rate))
        relative_volume_rates[record.diagnosis].append(float(relative_rate))

        displacement_rate = (vertices[1] - vertices[0]) / delta_years
        normal_rate = np.sum(displacement_rate * template_normals, axis=1)
        normal_rates_by_diag[record.diagnosis].append(normal_rate.astype(np.float32))

    velocity_matrix = np.stack(velocities, axis=0).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    velocity_mean = velocity_matrix.mean(axis=0)
    velocity_std = np.maximum(velocity_matrix.std(axis=0), 1.0e-6)
    cn_mask = label_array == 0
    ad_mask = label_array == 1
    cn_velocity_mean = velocity_matrix[cn_mask].mean(axis=0) if cn_mask.any() else velocity_mean
    ad_velocity_mean = velocity_matrix[ad_mask].mean(axis=0) if ad_mask.any() else velocity_mean

    centered = velocity_matrix - velocity_mean[None, :]
    _, singular_values, vt = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
    total_energy = float(np.sum(singular_values**2))
    explained = (singular_values**2) / max(total_energy, 1.0e-12)

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else experiment_root(args.experiment_name) / "metadata"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    max_dim = int(max(args.dynamic_dims))
    if max_dim > vt.shape[0]:
        raise ValueError(f"Requested dynamic dim {max_dim}, but only {vt.shape[0]} modes exist.")

    basis_files: list[str] = []
    for dynamic_dim in [int(value) for value in args.dynamic_dims]:
        output = output_dir / f"dynamic_pca_velocity_k{dynamic_dim}.npz"
        np.savez_compressed(
            output,
            basis=vt[:dynamic_dim].astype(np.float32),
            singular_values=singular_values[:dynamic_dim].astype(np.float64),
            explained_variance_ratio=explained[:dynamic_dim].astype(np.float64),
            explained_variance_ratio_cumulative=np.asarray([explained[:dynamic_dim].sum()]),
            velocity_mean=velocity_mean.astype(np.float32),
            velocity_std=velocity_std.astype(np.float32),
            cn_velocity_mean=cn_velocity_mean.astype(np.float32),
            ad_velocity_mean=ad_velocity_mean.astype(np.float32),
            components=np.asarray([int(args.components)], dtype=np.int64),
            dynamic_dim=np.asarray([dynamic_dim], dtype=np.int64),
            pair_count=np.asarray([len(records)], dtype=np.int64),
        )
        basis_files.append(str(output))

    local_weights = np.zeros(mean_flat.shape[0] // 3, dtype=np.float32)
    if normal_rates_by_diag.get("AD") and normal_rates_by_diag.get("CN"):
        ad_mean = np.stack(normal_rates_by_diag["AD"], axis=0).mean(axis=0)
        cn_mean = np.stack(normal_rates_by_diag["CN"], axis=0).mean(axis=0)
        local_weights = np.abs(ad_mean - cn_mean).astype(np.float32)
        max_weight = float(local_weights.max())
        if max_weight > 0.0:
            local_weights = local_weights / max_weight
    local_weights_path = output_dir / "local_ad_cn_change_weights.npy"
    np.save(local_weights_path, local_weights.astype(np.float32))

    empirical = {
        "config": str(Path(args.config).resolve()),
        "pca_model_dir": str(pca_model_dir),
        "components": int(args.components),
        "pair_type": str(args.pair_type),
        "max_gap_years": float(args.max_gap_years),
        "pair_count": len(records),
        "basis_files": basis_files,
        "local_ad_cn_change_weights": str(local_weights_path),
        "log_volume_rate_per_year": {
            diagnosis: summarize(values) for diagnosis, values in log_volume_rates.items()
        },
        "relative_volume_rate_per_year": {
            diagnosis: summarize(values) for diagnosis, values in relative_volume_rates.items()
        },
        "velocity_norm_per_year": summarize(
            [float(np.linalg.norm(row)) for row in velocity_matrix.astype(np.float64)]
        ),
        "dynamic_pca_cumulative_energy": {
            str(int(value)): float(explained[: int(value)].sum()) for value in args.dynamic_dims
        },
    }
    empirical_path = output_dir / "empirical_rate_targets.json"
    with empirical_path.open("w", encoding="utf-8") as handle:
        json.dump(empirical, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(empirical, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
