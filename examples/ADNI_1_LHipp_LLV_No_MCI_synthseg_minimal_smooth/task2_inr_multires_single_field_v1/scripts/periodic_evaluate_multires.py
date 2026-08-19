#!/usr/bin/env python3
"""Fit held-out codes and evaluate multiresolution INR meshes against PCA."""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from multires_common import (
    choose_device,
    decode_latent_to_mesh,
    finite_difference_epsilon,
    fit_single_latent,
    load_config,
    load_checkpoint_training_scan_ids,
    load_decoder_checkpoint,
    load_json,
    load_manifest,
    load_sdf_arrays,
    numerical_spatial_gradient,
    read_scaling,
    require_bulk_path,
    resolve_repo_path,
    select_stratified_rows,
    stable_seed,
    write_csv,
    write_json,
)


DISTANCE_METRICS = (
    "assd_mm",
    "chamfer_l1_mm",
    "chamfer_l2_squared_mm2",
    "hd95_mm",
    "gt_to_prediction_mm",
    "prediction_to_gt_mm",
    "fscore_0_5mm",
    "fscore_1mm",
    "volume_absolute_error_mm3",
    "volume_relative_error",
    "normal_absolute_cosine",
    "normal_signed_cosine",
    "adjacent_face_angle_mean_degrees",
    "adjacent_face_angle_p95_degrees",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best_mesh")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=None)
    parser.add_argument("--per-split", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--latent-steps", type=int, default=None)
    parser.add_argument("--surface-points", type=int, default=None)
    parser.add_argument("--overwrite-meshes", action="store_true")
    return parser.parse_args()


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Empty or invalid mesh: {path}")
    return mesh


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int):
    points, faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def edge_qc(mesh: trimesh.Trimesh) -> tuple[int, int]:
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    return int(np.sum(counts == 1)), int(np.sum(counts > 2))


def surface_metrics(
    ground_truth: trimesh.Trimesh,
    predicted: trimesh.Trimesh,
    count: int,
    seed: int,
) -> dict[str, Any]:
    gt_points, gt_faces = sample_surface(ground_truth, count, seed)
    prediction_points, prediction_faces = sample_surface(predicted, count, seed + 1)
    gt_closest, gt_to_prediction, gt_target_faces = trimesh.proximity.ProximityQuery(
        predicted
    ).on_surface(gt_points)
    prediction_closest, prediction_to_gt, prediction_target_faces = trimesh.proximity.ProximityQuery(
        ground_truth
    ).on_surface(prediction_points)
    del gt_closest, prediction_closest
    mean_gt_to_prediction = float(gt_to_prediction.mean())
    mean_prediction_to_gt = float(prediction_to_gt.mean())

    gt_normals = ground_truth.face_normals[gt_faces]
    predicted_at_gt = predicted.face_normals[gt_target_faces]
    prediction_normals = predicted.face_normals[prediction_faces]
    gt_at_prediction = ground_truth.face_normals[prediction_target_faces]
    cosine = np.concatenate(
        (
            np.einsum("ij,ij->i", gt_normals, predicted_at_gt),
            np.einsum("ij,ij->i", prediction_normals, gt_at_prediction),
        )
    )
    adjacency_angles = np.degrees(np.asarray(predicted.face_adjacency_angles))
    gt_volume = abs(float(ground_truth.volume))
    predicted_volume = abs(float(predicted.volume))
    boundary_edges, nonmanifold_edges = edge_qc(predicted)
    result: dict[str, Any] = {
        "assd_mm": 0.5 * (mean_gt_to_prediction + mean_prediction_to_gt),
        "chamfer_l1_mm": mean_gt_to_prediction + mean_prediction_to_gt,
        "chamfer_l2_squared_mm2": float(
            np.square(gt_to_prediction).mean() + np.square(prediction_to_gt).mean()
        ),
        "hd95_mm": float(
            max(np.quantile(gt_to_prediction, 0.95), np.quantile(prediction_to_gt, 0.95))
        ),
        "gt_to_prediction_mm": mean_gt_to_prediction,
        "prediction_to_gt_mm": mean_prediction_to_gt,
        "ground_truth_volume_mm3": gt_volume,
        "predicted_volume_mm3": predicted_volume,
        "volume_absolute_error_mm3": abs(predicted_volume - gt_volume),
        "volume_relative_error": abs(predicted_volume - gt_volume) / max(gt_volume, 1.0e-12),
        "normal_absolute_cosine": float(np.abs(cosine).mean()),
        "normal_signed_cosine": float(cosine.mean()),
        "adjacent_face_angle_mean_degrees": float(adjacency_angles.mean()) if len(adjacency_angles) else 0.0,
        "adjacent_face_angle_p95_degrees": float(np.quantile(adjacency_angles, 0.95)) if len(adjacency_angles) else 0.0,
        "predicted_watertight": bool(predicted.is_watertight),
        "predicted_winding_consistent": bool(predicted.is_winding_consistent),
        "predicted_connected_components": int(len(predicted.split(only_watertight=False))),
        "predicted_euler_number": int(predicted.euler_number),
        "predicted_boundary_edges": boundary_edges,
        "predicted_nonmanifold_edges": nonmanifold_edges,
        "distance_backend": "sampled source points to exact target triangles",
    }
    for threshold, name in ((0.5, "fscore_0_5mm"), (1.0, "fscore_1mm")):
        recall = float(np.mean(gt_to_prediction <= threshold))
        precision = float(np.mean(prediction_to_gt <= threshold))
        result[name] = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return result


class PCAReconstructor:
    def __init__(self, model_dir: Path, coefficient_path: Path, count: int) -> None:
        self.mean = np.load(model_dir / "mean.npy")
        components = np.load(model_dir / "components_150.npy")
        self.components = components[:count]
        self.faces = np.load(model_dir / "faces.npy")
        archive = np.load(coefficient_path)
        source = archive["pca_150"][:, :count]
        self.coefficients = {
            str(scan_id): vector for scan_id, vector in zip(archive["scan_ids"], source)
        }

    def mesh(self, scan_id: str) -> trimesh.Trimesh:
        flat = self.mean + self.coefficients[scan_id] @ self.components
        return trimesh.Trimesh(
            vertices=flat.reshape(-1, 3), faces=self.faces, process=False
        )


def load_selection(
    config: dict[str, Any],
    rows: list[dict[str, str]],
    splits: list[str],
    count: int,
    source_seen_subjects: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    periodic = config["periodic_evaluation"]
    fixed = periodic.get("fixed_selection_json")
    if fixed:
        selection = load_json(resolve_repo_path(fixed))
    else:
        by_split = {}
        for offset, split in enumerate(splits):
            pool = [row for row in rows if row["split"] == split]
            if split == "val" and source_seen_subjects is not None:
                pool = [row for row in pool if row["subject_id"] not in source_seen_subjects]
            if not pool:
                raise ValueError(f"No eligible rows remain for periodic {split} evaluation.")
            chosen = select_stratified_rows(pool, min(count, len(pool)), int(config["seed"]) + offset)
            by_split[split] = [row["scan_id"] for row in chosen]
        selection = {
            "seed": int(config["seed"]),
            "strategy": "diagnosis_stratified_with_subject_coverage",
            "source_unseen_validation_only": source_seen_subjects is not None,
            "scan_ids_by_split": by_split,
        }
    by_id = {row["scan_id"]: row for row in rows}
    selected = []
    for split in splits:
        ids = selection["scan_ids_by_split"][split][:count]
        missing = [scan_id for scan_id in ids if scan_id not in by_id]
        if missing:
            raise KeyError(f"Fixed evaluation scans absent from manifest: {missing[:3]}")
        if split == "val" and source_seen_subjects is not None:
            contaminated = [
                scan_id for scan_id in ids
                if by_id[scan_id]["subject_id"] in source_seen_subjects
            ]
            if contaminated:
                raise ValueError(
                    "Saved validation selection contains source-pretraining subjects: "
                    f"{contaminated[:3]}. Remove the stale selection before evaluation."
                )
        selected.extend(by_id[scan_id] for scan_id in ids)
    return selected, selection


def field_gradient_metrics(
    decoder,
    latent: np.ndarray,
    sdf_path: str,
    config: dict[str, Any],
    epoch: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    pos, neg = load_sdf_arrays(sdf_path)
    candidates = np.concatenate((pos, neg), axis=0)
    band = float(config["field_diagnostics"]["target_band"])
    candidates = candidates[np.abs(candidates[:, 3]) <= band]
    count = int(config["field_diagnostics"]["points_per_scan"])
    rng = np.random.default_rng(seed)
    chosen = candidates[rng.integers(0, len(candidates), size=count)]
    xyz = torch.from_numpy(chosen[:, :3]).to(device)
    code = torch.from_numpy(latent.astype(np.float32)).reshape(1, -1).to(device).expand(count, -1)
    weights = [1.0] * len(decoder.grid_resolutions)
    epsilon = torch.as_tensor(
        finite_difference_epsilon(epoch, config, weights), device=device, dtype=xyz.dtype
    )
    with torch.no_grad():
        norm = torch.linalg.vector_norm(
            numerical_spatial_gradient(decoder, code, xyz, epsilon, weights), dim=1
        )
    return {
        "field_gradient_norm_mean": float(norm.mean().cpu()),
        "field_gradient_norm_median": float(norm.median().cpu()),
        "field_gradient_norm_p05": float(torch.quantile(norm, 0.05).cpu()),
        "field_gradient_norm_p95": float(torch.quantile(norm, 0.95).cpu()),
        "field_gradient_norm_absolute_error": float(torch.abs(norm - 1.0).mean().cpu()),
        "field_gradient_fraction_within_10pct": float((torch.abs(norm - 1.0) <= 0.1).float().mean().cpu()),
        "field_gradient_epsilon_x": float(epsilon[0].cpu()),
        "field_gradient_epsilon_y": float(epsilon[1].cpu()),
        "field_gradient_epsilon_z": float(epsilon[2].cpu()),
    }


def cluster_bootstrap(
    values: np.ndarray, cluster_ids: np.ndarray, seed: int, repeats: int = 2000
) -> list[float]:
    if not len(values):
        return [float("nan"), float("nan")]
    clusters = np.unique(cluster_ids)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = clusters[rng.integers(0, len(clusters), size=len(clusters))]
        positions = np.concatenate(
            [np.flatnonzero(cluster_ids == cluster) for cluster in sampled]
        )
        means[index] = values[positions].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"method:{row['method']}"].append(row)
        groups[f"split:{row['split']}/method:{row['method']}"].append(row)
        if row.get("diagnosis"):
            groups[f"diagnosis:{row['diagnosis']}/method:{row['method']}"].append(row)
    report = {}
    for name, group in groups.items():
        metrics = {}
        cluster_ids = np.asarray([row["subject_id"] for row in group])
        for metric in DISTANCE_METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            metrics[metric] = {
                "count": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "subject_cluster_bootstrap_mean_95ci": cluster_bootstrap(
                    values, cluster_ids, stable_seed(name + metric, seed)
                ),
            }
        report[name] = metrics
    return report


def paired_inr_minus_pca(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    lookup = {(row["scan_id"], row["split"], row["method"]): row for row in rows}
    result = {}
    for split in sorted({row["split"] for row in rows}):
        scan_ids = sorted(
            {row["scan_id"] for row in rows if row["split"] == split and row["method"] == "inr"}
        )
        split_result = {}
        for metric in DISTANCE_METRICS:
            differences = np.asarray(
                [
                    float(lookup[(scan_id, split, "inr")][metric])
                    - float(lookup[(scan_id, split, "pca")][metric])
                    for scan_id in scan_ids
                    if (scan_id, split, "pca") in lookup
                ],
                dtype=np.float64,
            )
            cluster_ids = np.asarray(
                [
                    lookup[(scan_id, split, "inr")]["subject_id"]
                    for scan_id in scan_ids
                    if (scan_id, split, "pca") in lookup
                ]
            )
            split_result[metric] = {
                "count": len(differences),
                "mean": float(differences.mean()) if len(differences) else float("nan"),
                "median": float(np.median(differences)) if len(differences) else float("nan"),
                "subject_cluster_bootstrap_mean_95ci": cluster_bootstrap(
                    differences, cluster_ids, stable_seed(split + metric, seed)
                ),
                "definition": "INR minus target-projected PCA",
            }
        result[split] = split_result
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    periodic = config["periodic_evaluation"]
    device = choose_device(args.device)
    decoder, payload, checkpoint = load_decoder_checkpoint(config, args.checkpoint, device)
    epoch = int(payload.get("epoch", 0))
    output_dir = require_bulk_path(
        args.output_dir
        or require_bulk_path(config["output_dir"]) / "manual_evaluation" / checkpoint.stem
    )
    splits = list(args.splits or periodic.get("splits", ["train", "val", "test"]))
    per_split = int(args.per_split or periodic.get("per_split", 100))
    resolution = int(args.resolution or periodic.get("resolution", 256))
    latent_steps = int(args.latent_steps or periodic.get("latent_steps", 500))
    surface_points = int(args.surface_points or periodic.get("surface_points", 30000))
    rows = load_manifest(config["manifest"])
    warm_start = config.get("decoder_warm_start", {})
    source_checkpoint = warm_start.get("checkpoint")
    restrict_validation = bool(periodic.get("source_unseen_validation_only", False))
    source_ids: set[str] = set()
    source_seen_subjects: set[str] = set()
    if source_checkpoint:
        source_ids = load_checkpoint_training_scan_ids(source_checkpoint)
        source_seen_subjects = {
            row["subject_id"] for row in rows if row["scan_id"] in source_ids
        }
    if restrict_validation and not source_checkpoint:
        raise ValueError("Source-unseen validation requires a source checkpoint.")
    selected, selection = load_selection(
        config,
        rows,
        splits,
        per_split,
        source_seen_subjects if restrict_validation else None,
    )
    write_json(output_dir / "evaluation_subset.json", selection)

    train_ids = list(payload["training_scan_ids"])
    latent_table = payload["latent_codes"].detach().cpu().numpy()
    learned = {scan_id: latent_table[index] for index, scan_id in enumerate(train_ids)}
    scaling = read_scaling(periodic["rescale_details_csv"])
    compare_pca = bool(periodic.get("compare_pca", True))
    pca = None
    if compare_pca:
        pca = PCAReconstructor(
            resolve_repo_path(periodic["pca_model_dir"]),
            resolve_repo_path(periodic["pca_coefficients"]),
            int(periodic.get("pca_components", 150)),
        )

    results = []
    latent_rows = []
    gradient_rows = []
    failures = []
    started = time.time()
    for number, row in enumerate(selected, start=1):
        try:
            if row["split"] == "train":
                latent = np.asarray(learned[row["scan_id"]], dtype=np.float32)
                latent_metrics = {"source": "learned_embedding", "steps_completed": 0}
            else:
                latent, fit_metrics = fit_single_latent(
                    decoder,
                    row["sdf_npz_path"],
                    int(config["latent_size"]),
                    config["latent_fit"],
                    float(config["clamp_distance"]),
                    device,
                    stable_seed(row["scan_id"], int(config["seed"])),
                    config["network_specs"],
                    steps_override=latent_steps,
                )
                latent_metrics = {"source": "frozen_decoder_fit", **fit_metrics}
            latent_rows.append({"scan_id": row["scan_id"], "split": row["split"], **latent_metrics})
            gradient_rows.append(
                {
                    "scan_id": row["scan_id"],
                    "split": row["split"],
                    **field_gradient_metrics(
                        decoder,
                        latent,
                        row["sdf_npz_path"],
                        config,
                        epoch,
                        stable_seed(row["scan_id"], int(config["seed"]) + 9001),
                        device,
                    ),
                }
            )

            inr_path = output_dir / "meshes" / "inr" / row["split"] / f"{row['scan_id']}.ply"
            if args.overwrite_meshes or not inr_path.is_file():
                decode_latent_to_mesh(
                    decoder,
                    latent,
                    inr_path,
                    resolution,
                    int(config["reconstruction"]["max_batch"]),
                    device,
                    scaling=scaling,
                )
                inr_mesh = load_mesh(inr_path)
            else:
                inr_mesh = load_mesh(inr_path)
            ground_truth = load_mesh(row.get("mesh_path_mm") or row["mesh_path"])
            common = {
                "scan_id": row["scan_id"],
                "subject_id": row["subject_id"],
                "split": row["split"],
                "diagnosis": row.get("diagnosis", ""),
                "source_pretrain_scan_seen": row["scan_id"] in source_ids,
                "source_pretrain_subject_seen": row["subject_id"] in source_seen_subjects,
            }
            metrics = surface_metrics(
                ground_truth,
                inr_mesh,
                surface_points,
                stable_seed(row["scan_id"], int(config["seed"])),
            )
            results.append({**common, "method": "inr", **metrics, "mesh_path": str(inr_path)})

            if pca is not None:
                pca_mesh = pca.mesh(row["scan_id"])
                pca_path = output_dir / "meshes" / "pca" / row["split"] / f"{row['scan_id']}.ply"
                if args.overwrite_meshes or not pca_path.is_file():
                    pca_path = require_bulk_path(pca_path, "PCA mesh")
                    pca_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = pca_path.with_name(f".{pca_path.stem}.tmp{pca_path.suffix}")
                    pca_mesh.export(temporary)
                    os.replace(temporary, pca_path)
                metrics = surface_metrics(
                    ground_truth,
                    pca_mesh,
                    surface_points,
                    stable_seed(row["scan_id"], int(config["seed"])),
                )
                results.append({**common, "method": "pca", **metrics, "mesh_path": str(pca_path)})
        except Exception as error:
            failures.append({"scan_id": row["scan_id"], "split": row["split"], "error": repr(error)})
        print(f"[{number:04d}/{len(selected):04d}] {row['split']} {row['scan_id']}", flush=True)

    write_csv(output_dir / "per_scan_metrics.csv", results)
    write_csv(output_dir / "latent_fit_metrics.csv", latent_rows)
    write_csv(output_dir / "field_gradient_metrics.csv", gradient_rows)
    write_json(output_dir / "failures.json", failures)
    requested_counts = {
        split: sum(row["split"] == split for row in selected) for split in splits
    }
    success_counts = {
        split: {
            method: sum(
                row["split"] == split and row["method"] == method for row in results
            )
            for method in (("inr", "pca") if pca is not None else ("inr",))
        }
        for split in splits
    }
    minimum_success_fraction = float(periodic.get("minimum_success_fraction", 0.95))
    for split, expected in requested_counts.items():
        for method, succeeded in success_counts[split].items():
            if succeeded < minimum_success_fraction * expected:
                raise RuntimeError(
                    f"Only {succeeded}/{expected} {split} {method} meshes evaluated; "
                    f"required fraction is {minimum_success_fraction:.3f}."
                )
    validation_assd = [
        float(row["assd_mm"])
        for row in results
        if row["split"] == "val" and row["method"] == "inr"
    ]
    if "val" in splits and not validation_assd:
        raise RuntimeError("No validation INR mesh succeeded; checkpoint selection is impossible.")
    report = {
        "architecture": "single_field_dense_multiresolution_sdf",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": epoch,
        "coordinate_space": "physical millimetres",
        "alignment": "none; one fixed inverse global normalization is applied to INR meshes",
        "requested_per_split": per_split,
        "evaluated_rows": len(results),
        "failed_scans": failures,
        "resolution": resolution,
        "surface_points": surface_points,
        "requested_counts": requested_counts,
        "success_counts": success_counts,
        "minimum_success_fraction": minimum_success_fraction,
        "seconds": time.time() - started,
        "definitions": {
            "assd_mm": "0.5 * (mean GT-to-prediction + mean prediction-to-GT)",
            "chamfer_l1_mm": "mean GT-to-prediction + mean prediction-to-GT",
            "chamfer_l2_squared_mm2": "sum of both directional mean squared distances",
            "hd95_mm": "maximum of both directional 95th-percentile distances",
            "confidence_intervals": "percentile bootstrap resampling subjects as clusters; repeated scans remain together",
        },
        "selection_metric": {
            "name": "validation INR ASSD in millimetres",
            "validation_inr_assd_mm": float(np.mean(validation_assd)) if validation_assd else None,
        },
        "summary": summarize(results, int(config["seed"])),
        "paired_inr_minus_pca": paired_inr_minus_pca(results, int(config["seed"])) if pca else {},
        "pca_interpretation": "train-only PCA basis with each evaluated target mesh projected to its own coefficient; oracle reconstruction, not longitudinal prediction",
        "test_policy": "monitor-only; test metrics never select a checkpoint",
    }
    write_json(output_dir / "summary.json", report)
    print(f"Evaluation complete: rows={len(results)} failures={len(failures)} output={output_dir}")


if __name__ == "__main__":
    main()
