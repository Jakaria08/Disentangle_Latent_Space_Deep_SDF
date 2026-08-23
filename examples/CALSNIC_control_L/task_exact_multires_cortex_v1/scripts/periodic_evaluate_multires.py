#!/usr/bin/env python3
"""Evaluate CALSNIC multires triangle meshes against matched target-projected PCA."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import point_cloud_utils as pcu
import torch
import trimesh

from calsnic_common import load_sdf_space_mesh, mesh_sdf_to_mm, read_manifest
from delegate_multires import GENERIC_SCRIPTS

sys.path.insert(0, str(GENERIC_SCRIPTS))
from multires_common import (  # noqa: E402
    choose_device,
    decode_latent_to_mesh,
    fit_single_latent,
    load_config,
    load_decoder_checkpoint,
    require_bulk_path,
    stable_seed,
    write_csv,
    write_json,
)


METRICS = (
    "assd_mm",
    "chamfer_l1_mm",
    "chamfer_l2_squared_mm2",
    "hd95_mm",
    "gt_to_prediction_mm",
    "prediction_to_gt_mm",
    "fscore_0_5mm",
    "fscore_1mm",
    "fscore_2mm",
    "normal_absolute_cosine",
    "high_curvature_gt_to_prediction_mm",
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
    parser.add_argument("--overwrite-meshes", action="store_true")
    parser.add_argument("--confirm-test", action="store_true")
    return parser.parse_args()


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Invalid mesh: {path}")
    return mesh


def atomic_export(mesh: trimesh.Trimesh, path: Path) -> None:
    output = require_bulk_path(path, "evaluation mesh")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}")
    mesh.export(temporary)
    os.replace(temporary, output)


def edge_qc(mesh: trimesh.Trimesh) -> tuple[int, int]:
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    return int(np.sum(counts == 1)), int(np.sum(counts > 2))


def point_to_mesh(points: np.ndarray, mesh: trimesh.Trimesh):
    distance, face_index, barycentric = pcu.closest_points_on_mesh(
        np.asarray(points, dtype=np.float64),
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    return np.asarray(distance), np.asarray(face_index, dtype=np.int64), barycentric


def face_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    values = np.zeros(len(mesh.faces), dtype=np.float64)
    if len(mesh.face_adjacency):
        angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
        np.maximum.at(values, mesh.face_adjacency[:, 0], angles)
        np.maximum.at(values, mesh.face_adjacency[:, 1], angles)
    return values


def surface_metrics(gt: trimesh.Trimesh, prediction: trimesh.Trimesh, count: int, seed: int):
    gt_points, gt_source_faces = trimesh.sample.sample_surface(gt, count, seed=seed)
    prediction_points, prediction_source_faces = trimesh.sample.sample_surface(
        prediction, count, seed=seed + 1
    )
    gt_to_prediction, gt_target_faces, _ = point_to_mesh(gt_points, prediction)
    prediction_to_gt, prediction_target_faces, _ = point_to_mesh(prediction_points, gt)
    cosine = np.concatenate(
        (
            np.einsum("ij,ij->i", gt.face_normals[gt_source_faces], prediction.face_normals[gt_target_faces]),
            np.einsum("ij,ij->i", prediction.face_normals[prediction_source_faces], gt.face_normals[prediction_target_faces]),
        )
    )
    curvature = face_curvature(gt)
    threshold = float(np.quantile(curvature, 0.80))
    high_mask = curvature[gt_source_faces] >= threshold
    gt_volume, prediction_volume = abs(float(gt.volume)), abs(float(prediction.volume))
    boundary, nonmanifold = edge_qc(prediction)
    result: dict[str, Any] = {
        "assd_mm": float(0.5 * (gt_to_prediction.mean() + prediction_to_gt.mean())),
        "chamfer_l1_mm": float(gt_to_prediction.mean() + prediction_to_gt.mean()),
        "chamfer_l2_squared_mm2": float(np.square(gt_to_prediction).mean() + np.square(prediction_to_gt).mean()),
        "hd95_mm": float(max(np.quantile(gt_to_prediction, 0.95), np.quantile(prediction_to_gt, 0.95))),
        "gt_to_prediction_mm": float(gt_to_prediction.mean()),
        "prediction_to_gt_mm": float(prediction_to_gt.mean()),
        "normal_absolute_cosine": float(np.abs(cosine).mean()),
        "normal_signed_cosine": float(cosine.mean()),
        "high_curvature_gt_to_prediction_mm": float(gt_to_prediction[high_mask].mean()),
        "high_curvature_threshold_radians": threshold,
        "high_curvature_sample_count": int(high_mask.sum()),
        "ground_truth_volume_mm3": gt_volume,
        "predicted_volume_mm3": prediction_volume,
        "volume_absolute_error_mm3": abs(prediction_volume - gt_volume),
        "volume_relative_error": abs(prediction_volume - gt_volume) / max(gt_volume, 1.0e-12),
        "predicted_watertight": bool(prediction.is_watertight),
        "predicted_winding_consistent": bool(prediction.is_winding_consistent),
        "predicted_connected_components": int(len(prediction.split(only_watertight=False))),
        "predicted_euler_number": int(prediction.euler_number),
        "predicted_boundary_edges": boundary,
        "predicted_nonmanifold_edges": nonmanifold,
        "distance_backend": "sampled source points to exact target triangles via point_cloud_utils",
    }
    for threshold_mm, name in ((0.5, "fscore_0_5mm"), (1.0, "fscore_1mm"), (2.0, "fscore_2mm")):
        recall = float(np.mean(gt_to_prediction <= threshold_mm))
        precision = float(np.mean(prediction_to_gt <= threshold_mm))
        result[name] = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return result


class MatchedPCA:
    def __init__(self, directory: Path, components: int):
        metadata = json.loads((directory / "metadata.json").read_text())
        self.rank = int(metadata["rank"])
        self.count = min(int(components), self.rank)
        self.mean = np.load(directory / "mean_vertices.npy").reshape(-1)
        self.components = np.load(directory / "components.npy", mmap_mode="r")[: self.count].reshape(self.count, -1)
        self.faces = np.load(directory / "faces.npy")

    def reconstruct(self, row: dict[str, str]) -> trimesh.Trimesh:
        target = load_sdf_space_mesh(row)
        flat = np.asarray(target.vertices, dtype=np.float32).reshape(-1)
        coefficients = (flat - self.mean) @ self.components.T
        reconstructed = self.mean + coefficients @ self.components
        sdf_mesh = trimesh.Trimesh(vertices=reconstructed.reshape(-1, 3), faces=self.faces, process=False)
        return mesh_sdf_to_mm(sdf_mesh, row)


def cluster_bootstrap(values: np.ndarray, ids: np.ndarray, seed: int, repeats: int = 2000):
    if not len(values):
        return [float("nan"), float("nan")]
    clusters = np.unique(ids)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    for index in range(repeats):
        sampled = clusters[rng.integers(0, len(clusters), size=len(clusters))]
        positions = np.concatenate([np.flatnonzero(ids == subject) for subject in sampled])
        means[index] = values[positions].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(rows: list[dict[str, Any]], seed: int):
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"split:{row['split']}/method:{row['method']}"].append(row)
    report = {}
    for name, group in groups.items():
        subject_ids = np.asarray([row["subject_id"] for row in group])
        report[name] = {}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group])
            report[name][metric] = {
                "count": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "subject_bootstrap_mean_95ci": cluster_bootstrap(values, subject_ids, stable_seed(name + metric, seed)),
            }
    return report


def paired(rows: list[dict[str, Any]], seed: int):
    lookup = {(row["scan_id"], row["split"], row["method"]): row for row in rows}
    report = {}
    for split in sorted({row["split"] for row in rows}):
        scan_ids = sorted({row["scan_id"] for row in rows if row["split"] == split})
        report[split] = {}
        for metric in METRICS:
            pairs = [scan_id for scan_id in scan_ids if (scan_id, split, "inr") in lookup and (scan_id, split, "pca") in lookup]
            differences = np.asarray([float(lookup[(scan_id, split, "inr")][metric]) - float(lookup[(scan_id, split, "pca")][metric]) for scan_id in pairs])
            ids = np.asarray([lookup[(scan_id, split, "inr")]["subject_id"] for scan_id in pairs])
            report[split][metric] = {
                "count": len(differences),
                "mean_inr_minus_pca": float(differences.mean()) if len(differences) else float("nan"),
                "median_inr_minus_pca": float(np.median(differences)) if len(differences) else float("nan"),
                "subject_bootstrap_mean_95ci": cluster_bootstrap(differences, ids, stable_seed(split + metric, seed)),
                "fraction_inr_better": float(np.mean(differences < 0.0)) if len(differences) and not metric.startswith("fscore") and metric != "normal_absolute_cosine" else float(np.mean(differences > 0.0)) if len(differences) else float("nan"),
            }
    return report


def decode_mm(decoder, latent: np.ndarray, row: dict[str, str], output: Path, resolution: int, max_batch: int, device):
    temporary = output.parent / "_sdf_temporary" / f"{row['scan_id']}.ply"
    decode_latent_to_mesh(decoder, latent, temporary, resolution, max_batch, device, scaling=None)
    sdf_mesh = load_mesh(temporary)
    mm_mesh = mesh_sdf_to_mm(sdf_mesh, row)
    atomic_export(mm_mesh, output)
    temporary.unlink()
    return mm_mesh


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    periodic = config["periodic_evaluation"]
    splits = list(args.splits or periodic.get("splits", ["val"]))
    if "test" in splits and not args.confirm_test:
        raise PermissionError("Test evaluation is locked; add --confirm-test after model selection.")
    device = choose_device(args.device)
    decoder, payload, checkpoint = load_decoder_checkpoint(config, args.checkpoint, device)
    output = require_bulk_path(
        args.output_dir or Path(config["output_dir"]) / "manual_evaluation" / checkpoint.stem
    )
    per_split = int(args.per_split or periodic.get("per_split", 15))
    resolution = int(args.resolution or periodic.get("resolution", 256))
    latent_steps = int(args.latent_steps or periodic.get("latent_steps", 750))
    surface_points = int(args.surface_points or periodic.get("surface_points", 30000))
    rows = read_manifest(config["manifest"])
    selected = []
    for split in splits:
        candidates = [row for row in rows if row["split"] == split]
        selected.extend(candidates[: min(per_split, len(candidates))])
    train_ids = list(payload["training_scan_ids"])
    train_table = payload["latent_codes"].detach().cpu().numpy()
    learned = {scan_id: train_table[index] for index, scan_id in enumerate(train_ids)}
    pca = None
    if periodic.get("compare_pca", True):
        pca = MatchedPCA(Path(periodic["pca_model_dir"]), int(periodic.get("pca_components", 172)))

    results, latent_reports, failures = [], [], []
    started = time.time()
    for number, row in enumerate(selected, start=1):
        try:
            if row["split"] == "train":
                latent = np.asarray(learned[row["scan_id"]], dtype=np.float32)
                latent_report = {"source": "learned_embedding", "steps_completed": 0}
            else:
                latent, fit_report = fit_single_latent(
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
                latent_report = {"source": "frozen_decoder_fit", **fit_report}
            latent_reports.append({"scan_id": row["scan_id"], "split": row["split"], **latent_report})
            inr_path = output / "meshes" / "inr" / row["split"] / f"{row['scan_id']}.ply"
            inr_mesh = load_mesh(inr_path) if inr_path.is_file() and not args.overwrite_meshes else decode_mm(
                decoder, latent, row, inr_path, resolution, int(config["reconstruction"]["max_batch"]), device
            )
            ground_truth = load_mesh(row["mesh_path_mm"])
            common = {"scan_id": row["scan_id"], "subject_id": row["subject_id"], "split": row["split"], "diagnosis": row.get("diagnosis", "")}
            results.append({**common, "method": "inr", **surface_metrics(ground_truth, inr_mesh, surface_points, stable_seed(row["scan_id"], int(config["seed"]))), "mesh_path": str(inr_path)})
            if pca is not None:
                pca_path = output / "meshes" / "pca" / row["split"] / f"{row['scan_id']}.ply"
                pca_mesh = pca.reconstruct(row)
                if args.overwrite_meshes or not pca_path.is_file():
                    atomic_export(pca_mesh, pca_path)
                results.append({**common, "method": "pca", **surface_metrics(ground_truth, pca_mesh, surface_points, stable_seed(row["scan_id"], int(config["seed"]))), "mesh_path": str(pca_path)})
        except Exception as error:
            failures.append({"scan_id": row["scan_id"], "split": row["split"], "error": repr(error)})
        print(f"[{number}/{len(selected)}] {row['split']} {row['scan_id']}", flush=True)

    write_csv(output / "per_scan_metrics.csv", results)
    write_csv(output / "latent_fit_metrics.csv", latent_reports)
    write_json(output / "failures.json", failures)
    minimum = float(periodic.get("minimum_success_fraction", 1.0))
    for split in splits:
        expected = sum(row["split"] == split for row in selected)
        for method in (("inr", "pca") if pca is not None else ("inr",)):
            succeeded = sum(row["split"] == split and row["method"] == method for row in results)
            if succeeded < minimum * expected:
                raise RuntimeError(f"Only {succeeded}/{expected} {split} {method} evaluations succeeded.")
    validation = [float(row["assd_mm"]) for row in results if row["split"] == "val" and row["method"] == "inr"]
    if "val" in splits and not validation:
        raise RuntimeError("No validation INR mesh succeeded.")
    report = {
        "architecture": "single_field_dense_multiresolution_sdf",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "coordinate_space": "physical millimetres",
        "sdf_to_mm_transform": "per-subject inverse scaled-OBJ similarity transform",
        "splits": splits,
        "test_confirmed": bool(args.confirm_test),
        "resolution": resolution,
        "surface_points": surface_points,
        "seconds": time.time() - started,
        "failures": failures,
        "selection_metric": {"name": "validation INR ASSD in millimetres", "validation_inr_assd_mm": float(np.mean(validation)) if validation else None},
        "summary": summarize(results, int(config["seed"])),
        "paired_inr_minus_pca": paired(results, int(config["seed"])) if pca is not None else {},
        "pca_interpretation": "train-only basis; each target is projected to its own PCA coefficients (oracle reconstruction)",
        "test_policy": "manual confirmation required; test never selects an architecture or checkpoint",
    }
    write_json(output / "summary.json", report)
    print(f"Evaluation complete: rows={len(results)}, failures={len(failures)}, output={output}")


if __name__ == "__main__":
    main()
