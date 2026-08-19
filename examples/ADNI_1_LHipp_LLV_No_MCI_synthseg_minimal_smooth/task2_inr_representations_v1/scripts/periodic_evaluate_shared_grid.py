#!/usr/bin/env python3
"""Fit held-out codes, reconstruct fixed subsets, and compare INR with PCA."""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from shared_grid_common import (
    choose_device,
    decode_latent_to_mesh,
    fit_single_latent,
    load_checkpoint_training_scan_ids,
    load_config,
    load_decoder_checkpoint,
    load_manifest,
    load_json,
    resolve_repo_path,
    select_stratified_rows,
    stable_seed,
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
    parser.add_argument("--compare-pca", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--use-saved-selection",
        action="store_true",
        help="Reuse/create the experiment-level fixed evaluation_subset.json.",
    )
    parser.add_argument("--overwrite-meshes", action="store_true")
    return parser.parse_args()


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Empty or invalid mesh: {path}")
    return mesh


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    points, _faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float32)


def edge_qc(mesh: trimesh.Trimesh) -> tuple[int, int]:
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    return int(np.sum(counts == 1)), int(np.sum(counts > 2))


def surface_metrics(
    ground_truth: trimesh.Trimesh,
    predicted: trimesh.Trimesh,
    count: int,
    seed: int,
) -> dict[str, Any]:
    gt_points = sample_surface(ground_truth, count, seed)
    predicted_points = sample_surface(predicted, count, seed + 1)
    try:
        gt_to_prediction = trimesh.proximity.ProximityQuery(predicted).on_surface(gt_points)[1]
        prediction_to_gt = trimesh.proximity.ProximityQuery(ground_truth).on_surface(predicted_points)[1]
        distance_backend = "sampled source points to exact target triangles"
    except (ImportError, ModuleNotFoundError):
        prediction_tree = cKDTree(predicted_points)
        gt_tree = cKDTree(gt_points)
        gt_to_prediction = prediction_tree.query(gt_points, k=1, workers=-1)[0]
        prediction_to_gt = gt_tree.query(predicted_points, k=1, workers=-1)[0]
        distance_backend = "sampled point-cloud fallback"
    mean_gt_to_prediction = float(gt_to_prediction.mean())
    mean_prediction_to_gt = float(prediction_to_gt.mean())
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
        "predicted_watertight": bool(predicted.is_watertight),
        "predicted_winding_consistent": bool(predicted.is_winding_consistent),
        "predicted_connected_components": int(len(predicted.split(only_watertight=False))),
        "predicted_euler_number": int(predicted.euler_number),
        "predicted_boundary_edges": boundary_edges,
        "predicted_nonmanifold_edges": nonmanifold_edges,
        "distance_backend": distance_backend,
    }
    for threshold, name in ((0.5, "fscore_0_5mm"), (1.0, "fscore_1mm")):
        recall = float(np.mean(gt_to_prediction <= threshold))
        precision = float(np.mean(prediction_to_gt <= threshold))
        result[name] = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return result


def read_scaling(path: str | Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    keys = (
        "target_range_min",
        "range_global_min",
        "range_linear_scale_factor",
        "distance_unscale_factor",
    )
    values = {key: float(row[key]) for key in keys}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite mesh scaling in {path}")
    return values


def convert_normalized_mesh_to_mm(mesh: trimesh.Trimesh, scaling: dict[str, float]) -> None:
    mesh.vertices = (
        (mesh.vertices - scaling["target_range_min"])
        / scaling["range_linear_scale_factor"]
        + scaling["range_global_min"]
    ) * scaling["distance_unscale_factor"]


class PCAReconstructor:
    def __init__(self, model_dir: Path, coefficient_path: Path, component_count: int) -> None:
        self.mean = np.load(model_dir / "mean.npy")
        self.components = np.load(model_dir / f"components_{component_count}.npy")
        self.faces = np.load(model_dir / "faces.npy")
        archive = np.load(coefficient_path)
        coefficient_key = f"pca_{component_count}"
        self.coefficients = {
            str(scan_id): vector
            for scan_id, vector in zip(archive["scan_ids"], archive[coefficient_key])
        }
        if self.components.shape[0] != component_count:
            raise ValueError(
                f"Expected {component_count} PCA components, got {self.components.shape}."
            )

    def mesh(self, scan_id: str) -> trimesh.Trimesh:
        if scan_id not in self.coefficients:
            raise KeyError(f"PCA coefficients do not contain {scan_id}.")
        flat = self.mean + self.coefficients[scan_id] @ self.components
        vertices = flat.reshape(-1, 3)
        return trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)


def load_or_create_selection(
    rows: list[dict[str, str]],
    splits: list[str],
    per_split: int,
    seed: int,
    selection_path: Path | None,
    source_seen_subjects: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if selection_path is not None and selection_path.is_file():
        selection = load_json(selection_path)
        by_id = {row["scan_id"]: row for row in rows}
        selected = []
        for split in splits:
            ids = selection["scan_ids_by_split"][split][:per_split]
            missing = [scan_id for scan_id in ids if scan_id not in by_id]
            if missing:
                raise KeyError(f"Saved evaluation selection is absent from manifest: {missing[:3]}")
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

    selected = []
    by_split = {}
    for offset, split in enumerate(splits):
        pool = [row for row in rows if row["split"] == split]
        if split == "val" and source_seen_subjects is not None:
            pool = [row for row in pool if row["subject_id"] not in source_seen_subjects]
        if not pool:
            raise ValueError(f"No eligible rows remain for periodic {split} evaluation.")
        chosen = select_stratified_rows(pool, min(per_split, len(pool)), seed + offset)
        selected.extend(chosen)
        by_split[split] = [row["scan_id"] for row in chosen]
    selection = {
        "seed": seed,
        "strategy": "diagnosis_stratified_with_subject_coverage",
        "source_unseen_validation_only": source_seen_subjects is not None,
        "requested_per_split": per_split,
        "scan_ids_by_split": by_split,
    }
    if selection_path is not None:
        write_json(selection_path, selection)
    return selected, selection


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    if not len(values):
        return [math.nan, math.nan]
    indices = rng.integers(0, len(values), size=(1000, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[f"split:{row['split']}"] .append(row)
        groups[f"split:{row['split']}/method:{row['method']}"] .append(row)
        groups[f"diagnosis:{row.get('diagnosis', '')}/method:{row['method']}"] .append(row)
        groups[f"method:{row['method']}"] .append(row)
    result = {}
    for group_name, group in sorted(groups.items()):
        rng = np.random.default_rng(stable_seed(group_name, seed))
        numeric = {}
        for metric in DISTANCE_METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            numeric[metric] = {
                "count": int(len(values)),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "bootstrap_mean_95ci": bootstrap_interval(values, rng),
            }
        result[group_name] = numeric
    return result


def paired_summary(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    lookup = {(row["scan_id"], row["method"]): row for row in rows}
    scan_ids = sorted({row["scan_id"] for row in rows})
    result = {}
    for split in ("train", "val", "test"):
        pairs = [
            scan_id
            for scan_id in scan_ids
            if (scan_id, "inr") in lookup
            and (scan_id, "pca") in lookup
            and lookup[(scan_id, "inr")]["split"] == split
        ]
        if not pairs:
            continue
        rng = np.random.default_rng(stable_seed(f"paired:{split}", seed))
        result[split] = {}
        for metric in DISTANCE_METRICS:
            difference = np.asarray(
                [lookup[(scan_id, "inr")][metric] - lookup[(scan_id, "pca")][metric] for scan_id in pairs],
                dtype=np.float64,
            )
            result[split][metric] = {
                "definition": (
                    "INR minus PCA; positive favors INR"
                    if metric.startswith("fscore_")
                    else "INR minus PCA; negative favors INR"
                ),
                "count": len(pairs),
                "mean": float(difference.mean()),
                "median": float(np.median(difference)),
                "bootstrap_mean_95ci": bootstrap_interval(difference, rng),
            }
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    periodic = config["periodic_evaluation"]
    splits = list(args.splits or periodic["splits"])
    per_split = int(args.per_split or periodic["per_split"])
    resolution = int(args.resolution or periodic["resolution"])
    latent_steps = int(args.latent_steps or periodic["latent_steps"])
    surface_points = int(args.surface_points or periodic["surface_points"])
    compare_pca = bool(periodic["compare_pca"] if args.compare_pca is None else args.compare_pca)
    device = choose_device(args.device)
    decoder, payload, checkpoint = load_decoder_checkpoint(config, args.checkpoint, device)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else resolve_repo_path(config["output_dir"])
        / "manual_evaluation"
        / checkpoint.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_selection = (
        resolve_repo_path(config["output_dir"]) / "periodic_evaluation" / "evaluation_subset.json"
        if args.use_saved_selection
        else output_dir / "evaluation_subset.json"
    )
    manifest_rows = load_manifest(config["manifest"])
    warm_start = config.get("decoder_warm_start", {})
    source_checkpoint = warm_start.get("checkpoint", config.get("global_checkpoint"))
    source_ids: set[str] = set()
    source_seen_subjects: set[str] = set()
    restrict_validation = bool(periodic.get("source_unseen_validation_only", False))
    if source_checkpoint:
        source_ids = load_checkpoint_training_scan_ids(
            source_checkpoint,
            str(config.get("global_checkpoint_scan_id_suffix", "")),
        )
        source_seen_subjects = {
            row["subject_id"] for row in manifest_rows if row["scan_id"] in source_ids
        }
    if restrict_validation and not source_checkpoint:
        raise ValueError("Source-unseen validation requires a source checkpoint.")
    selected, selection = load_or_create_selection(
        manifest_rows,
        splits,
        per_split,
        int(periodic.get("selection_seed", config["seed"])),
        experiment_selection,
        source_seen_subjects if restrict_validation else None,
    )
    write_json(output_dir / "evaluation_subset.json", selection)

    training_ids = payload.get("training_scan_ids", [])
    learned_codes = payload.get("latent_codes")
    if learned_codes is None or len(training_ids) != len(learned_codes):
        raise ValueError("Checkpoint lacks an aligned train latent table.")
    learned_by_scan = dict(zip(training_ids, learned_codes.detach().cpu().numpy()))
    scaling = read_scaling(periodic["rescale_details_csv"])
    pca = None
    if compare_pca:
        pca = PCAReconstructor(
            resolve_repo_path(periodic["pca_model_dir"]),
            resolve_repo_path(periodic["pca_coefficients"]),
            int(periodic.get("pca_components", 150)),
        )

    results: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    for number, row in enumerate(selected, start=1):
        try:
            if row["split"] == "train":
                latent = np.asarray(learned_by_scan[row["scan_id"]], dtype=np.float32)
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
                    steps_override=latent_steps,
                    mesh_path=row["mesh_path"],
                    network_specs=config["network_specs"],
                )
                latent_metrics = {"source": "frozen_decoder_fit", **fit_metrics}
            latent_rows.append(
                {"scan_id": row["scan_id"], "split": row["split"], **latent_metrics}
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
                )
                inr_mesh = load_mesh(inr_path)
                convert_normalized_mesh_to_mm(inr_mesh, scaling)
                inr_mesh.export(inr_path)
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
            inr_metrics = surface_metrics(
                ground_truth,
                inr_mesh,
                surface_points,
                stable_seed(row["scan_id"], int(config["seed"])),
            )
            results.append({**common, "method": "inr", **inr_metrics, "mesh_path": str(inr_path)})

            if pca is not None:
                pca_mesh = pca.mesh(row["scan_id"])
                pca_path = output_dir / "meshes" / "pca" / row["split"] / f"{row['scan_id']}.ply"
                pca_path.parent.mkdir(parents=True, exist_ok=True)
                if args.overwrite_meshes or not pca_path.is_file():
                    pca_mesh.export(pca_path)
                pca_metrics = surface_metrics(
                    ground_truth,
                    pca_mesh,
                    surface_points,
                    stable_seed(row["scan_id"], int(config["seed"])),
                )
                if len(ground_truth.vertices) == len(pca_mesh.vertices):
                    pca_metrics["vertex_rmse_mm"] = float(
                        np.sqrt(np.mean(np.square(ground_truth.vertices - pca_mesh.vertices)))
                    )
                results.append({**common, "method": "pca", **pca_metrics, "mesh_path": str(pca_path)})
        except Exception as error:
            failures.append(
                {"scan_id": row["scan_id"], "split": row["split"], "error": repr(error)}
            )
        print(f"[{number:04d}/{len(selected):04d}] {row['split']} {row['scan_id']}", flush=True)

    write_csv(output_dir / "per_scan_metrics.csv", results)
    write_csv(output_dir / "latent_fit_metrics.csv", latent_rows)
    inr_validation = [
        float(row["assd_mm"])
        for row in results
        if row["split"] == "val" and row["method"] == "inr"
    ]
    if "val" in splits and not inr_validation:
        raise RuntimeError("No INR validation mesh succeeded; cannot select a mesh checkpoint.")
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "coordinate_space": "physical millimetres",
        "alignment": "none; one fixed inverse global normalization is applied to INR meshes",
        "resolution": resolution,
        "surface_points": surface_points,
        "requested_per_split": per_split,
        "evaluated_rows": len(results),
        "failed_scans": failures,
        "test_policy": "monitor-only; test metrics never select a checkpoint",
        "selection_metric": {
            "name": "validation INR ASSD in millimetres",
            "validation_inr_assd_mm": float(np.mean(inr_validation)) if inr_validation else math.nan,
        },
        "summary": summarize(results, int(config["seed"])),
        "paired_inr_minus_pca": paired_summary(results, int(config["seed"])),
        "seconds": time.time() - started,
        "definitions": {
            "assd_mm": "0.5 * (mean GT-to-prediction + mean prediction-to-GT)",
            "chamfer_l1_mm": "mean GT-to-prediction + mean prediction-to-GT",
            "chamfer_l2_squared_mm2": "mean squared distances in both directions, summed",
            "hd95_mm": "maximum of the two directional 95th percentiles",
        },
    }
    write_json(output_dir / "summary.json", report)
    print(
        f"Evaluation complete: scans={len(selected) - len(failures)} failures={len(failures)} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
