#!/usr/bin/env python3
"""Separate validation/test evaluator for a frozen direct surface-cocycle checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import common as C
import data as D
import objectives as O
from conditional_spiral_unet import ConditionalSpiralUNet, vertex_normals
from train import build_model, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--surface-metrics", action="store_true")
    parser.add_argument(
        "--surface-max-pairs",
        type=int,
        default=None,
        help="Compute expensive surface metrics on a deterministic balanced subset; endpoint metrics still use all pairs.",
    )
    parser.add_argument("--surface-points", type=int, default=10000)
    parser.add_argument("--voxel-pitch-mm", type=float, default=0.5)
    parser.add_argument("--surface-seed", type=int, default=1701)
    parser.add_argument(
        "--save-vertex-maps",
        action="store_true",
        help="Cache diagnosis-wise observed/predicted normal-rate and error maps for the load-only notebook.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _surface_metric_function():
    scripts = C.TASK_ROOT.parent / "task3_pca_corrective_cocycle_128_v1" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from surface_metrics import metrics

    return metrics


def voxel_dice(left_vertices: np.ndarray, right_vertices: np.ndarray, faces: np.ndarray, pitch: float) -> float:
    import trimesh

    left = trimesh.Trimesh(vertices=left_vertices, faces=faces, process=False).voxelized(pitch).fill()
    right = trimesh.Trimesh(vertices=right_vertices, faces=faces, process=False).voxelized(pitch).fill()
    left_set = {tuple(row) for row in np.rint(np.asarray(left.points) / pitch).astype(np.int64)}
    right_set = {tuple(row) for row in np.rint(np.asarray(right.points) / pitch).astype(np.int64)}
    return float(2 * len(left_set & right_set) / max(len(left_set) + len(right_set), 1))


@torch.no_grad()
def endpoint_predictions(
    model: ConditionalSpiralUNet,
    split: D.PreparedSplit,
    rows: list[C.PairRow],
    batch_size: int,
) -> dict[tuple[int, int], np.ndarray]:
    output = {}
    for chunk in C.chunked(rows, batch_size):
        batch = split.pair_batch(chunk)
        prediction = model.transport(batch["source"], batch["source_age"], batch["target_age"], batch["label"])
        values = prediction.cpu().numpy()
        for index, row in enumerate(chunk):
            output[(row.source, row.target)] = values[index]
    return output


def balanced_surface_indices(records: list[dict[str, Any]], limit: int | None, seed: int) -> list[int]:
    """Balance an expensive-metric subset over diagnosis, interval type, and subject."""
    if limit is None or int(limit) >= len(records):
        return list(range(len(records)))
    if int(limit) <= 0:
        return []
    buckets: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = (str(record["diagnosis"]), str(record["pair_type"]), str(record["subject"]))
        buckets[key].append(index)
    rng = random.Random(int(seed))
    for values in buckets.values():
        rng.shuffle(values)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for diagnosis, pair_type, subject in buckets:
        strata[(diagnosis, pair_type)].append(subject)
    for subjects in strata.values():
        subjects.sort()
        rng.shuffle(subjects)
    selected: list[int] = []
    subject_cursor = {key: 0 for key in strata}
    ordered_strata = sorted(strata)
    while len(selected) < int(limit):
        progressed = False
        for stratum in ordered_strata:
            subjects = strata[stratum]
            for _ in range(len(subjects)):
                cursor = subject_cursor[stratum] % len(subjects)
                subject_cursor[stratum] += 1
                key = (stratum[0], stratum[1], subjects[cursor])
                if buckets[key]:
                    selected.append(buckets[key].pop())
                    progressed = True
                    break
            if len(selected) == int(limit):
                break
        if not progressed:
            break
    return sorted(selected)


def add_surface_metrics(
    records: list[dict[str, Any]],
    predictions: dict[tuple[int, int], np.ndarray],
    split: D.PreparedSplit,
    faces: np.ndarray,
    points: int,
    pitch: float,
    selected_indices: list[int],
) -> None:
    metric_function = _surface_metric_function()
    vertices = split.vertices.cpu().numpy()
    for index in selected_indices:
        record = records[index]
        key = (int(record["source_index"]), int(record["target_index"]))
        predicted = predictions[key]
        target = vertices[key[1]]
        metrics = metric_function(target, predicted, faces, int(points), seed=1701 + index)
        record.update(
            {
                "assd_mm": metrics["assd_mm"],
                "hd95_mm": metrics["hd95_mm"],
                "chamfer_l2_squared_mm2": metrics["chamfer_l2_squared_mm2"],
                "normal_signed_cosine": metrics["normal_signed_cosine"],
                "flipped_face_fraction": metrics["flipped_face_fraction_vs_ground_truth"],
                "mesh_dice": voxel_dice(predicted, target, faces, pitch),
            }
        )


def save_mean_vertex_maps(
    path: Path,
    records: list[dict[str, Any]],
    predictions: dict[tuple[int, int], np.ndarray],
    split: D.PreparedSplit,
    faces: np.ndarray,
) -> None:
    """Save one correspondence-based group-average surface map per method/diagnosis."""
    vertices = split.vertices.cpu().numpy()
    face_tensor = torch.as_tensor(faces, dtype=torch.long)
    sums = {
        diagnosis: {
            "observed_normal_rate": np.zeros(C.VERTEX_COUNT, dtype=np.float64),
            "predicted_normal_rate": np.zeros(C.VERTEX_COUNT, dtype=np.float64),
            "endpoint_error": np.zeros(C.VERTEX_COUNT, dtype=np.float64),
            "source_vertices": np.zeros((C.VERTEX_COUNT, 3), dtype=np.float64),
            "count": 0,
        }
        for diagnosis in ("CN", "AD")
    }
    for record in records:
        diagnosis = str(record["diagnosis"])
        source_index = int(record["source_index"])
        target_index = int(record["target_index"])
        source = vertices[source_index]
        target = vertices[target_index]
        predicted = predictions[(source_index, target_index)]
        normals = vertex_normals(torch.from_numpy(source[None]), face_tensor)[0].numpy()
        elapsed = max(float(record["delta_years"]), 1.0e-6)
        group = sums[diagnosis]
        group["observed_normal_rate"] += np.sum((target - source) * normals, axis=-1) / elapsed
        group["predicted_normal_rate"] += np.sum((predicted - source) * normals, axis=-1) / elapsed
        group["endpoint_error"] += np.linalg.norm(predicted - target, axis=-1)
        group["source_vertices"] += source
        group["count"] += 1
    arrays: dict[str, np.ndarray] = {"faces": faces.astype(np.int64)}
    for diagnosis, group in sums.items():
        count = max(int(group.pop("count")), 1)
        arrays[f"{diagnosis}_pairs"] = np.asarray([count], dtype=np.int64)
        for name, values in group.items():
            arrays[f"{diagnosis}_{name}"] = (values / count).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def velocity_metrics(model: ConditionalSpiralUNet, split: D.PreparedSplit, batch_size: int) -> tuple[list[dict], dict]:
    records = []
    predicted_blocks = []
    for start in range(0, len(split.scan_ids), batch_size):
        stop = min(start + batch_size, len(split.scan_ids))
        predicted = model.instantaneous_velocity(
            split.vertices[start:stop], split.ages[start:stop], split.labels[start:stop]
        )
        predicted_blocks.append(predicted)
    predicted_all = torch.cat(predicted_blocks)
    normals = vertex_normals(split.vertices, model.faces)
    for index in range(len(split.scan_ids)):
        reliability = float(split.velocity_reference_weight[index].cpu())
        if reliability <= 0.0:
            continue
        predicted = predicted_all[index]
        observed = split.velocity_reference[index]
        normal = normals[index]
        predicted_normal = torch.sum(predicted * normal, dim=-1)
        observed_normal = torch.sum(observed * normal, dim=-1)
        left = predicted.reshape(-1)
        right = observed.reshape(-1)
        cosine = torch.dot(left, right) / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(1.0e-8)
        left_centered = predicted_normal - predicted_normal.mean()
        right_centered = observed_normal - observed_normal.mean()
        correlation = torch.dot(left_centered, right_centered) / (
            torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
        ).clamp_min(1.0e-8)
        threshold = torch.quantile(observed_normal.abs(), 0.8)
        observed_hot = observed_normal.abs() >= threshold
        predicted_threshold = torch.quantile(predicted_normal.abs(), 0.8)
        predicted_hot = predicted_normal.abs() >= predicted_threshold
        hotspot_dice = 2.0 * torch.logical_and(observed_hot, predicted_hot).sum() / (
            observed_hot.sum() + predicted_hot.sum()
        ).clamp_min(1)
        records.append(
            {
                "scan_id": str(split.scan_ids[index]),
                "subject_id": str(split.subject_ids[index]),
                "diagnosis": str(split.diagnoses[index]),
                "age_years": float(split.ages[index].cpu()),
                "reference_reliability": reliability,
                "vector_rmse_mm_per_year": float(torch.sqrt(torch.mean((predicted - observed) ** 2)).cpu()),
                "normal_rmse_mm_per_year": float(torch.sqrt(torch.mean((predicted_normal - observed_normal) ** 2)).cpu()),
                "vector_cosine": float(cosine.cpu()),
                "normal_pearson": float(correlation.cpu()),
                "normal_sign_agreement": float((torch.sign(predicted_normal) == torch.sign(observed_normal)).float().mean().cpu()),
                "hotspot_dice": float(hotspot_dice.cpu()),
                "speed_ratio": float(
                    (torch.linalg.vector_norm(predicted, dim=-1).mean() / torch.linalg.vector_norm(observed, dim=-1).mean().clamp_min(1.0e-8)).cpu()
                ),
            }
        )
    groups = {}
    for diagnosis in ("CN", "AD", "overall"):
        current = records if diagnosis == "overall" else [row for row in records if row["diagnosis"] == diagnosis]
        if current:
            groups[diagnosis] = {
                "scans": len(current),
                **{
                    name: float(np.mean([row[name] for row in current]))
                    for name in (
                        "vector_rmse_mm_per_year",
                        "normal_rmse_mm_per_year",
                        "vector_cosine",
                        "normal_pearson",
                        "normal_sign_agreement",
                        "hotspot_dice",
                        "speed_ratio",
                    )
                },
            }
    return records, groups


def summarize_surface(records: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    numeric = [
        "mean_vertex_error_mm",
        "vertex_rmse_mm",
        "nochange_mean_vertex_error_mm",
        "volume_relative_error",
        "observed_log_volume_rate_per_year",
        "predicted_log_volume_rate_per_year",
        "volume_rate_abs_error_per_year",
        "assd_mm",
        "hd95_mm",
        "chamfer_l2_squared_mm2",
        "normal_signed_cosine",
        "flipped_face_fraction",
        "mesh_dice",
    ]
    for diagnosis in ("CN", "AD", "overall"):
        current = records if diagnosis == "overall" else [row for row in records if row["diagnosis"] == diagnosis]
        if not current:
            continue
        summary = {"pairs": len(current)}
        for name in numeric:
            available = [float(row[name]) for row in current if name in row and row[name] not in (None, "")]
            if available:
                summary[name] = float(np.mean(available))
                if len(available) != len(current):
                    summary[f"{name}_pairs"] = len(available)
        summary["error_to_nochange_ratio"] = summary["mean_vertex_error_mm"] / max(
            summary["nochange_mean_vertex_error_mm"], 1.0e-8
        )
        output[diagnosis] = summary
    return output


def main() -> int:
    args = parse_args()
    root = C.output_root(args.output_root)
    device = C.choose_device(args.device)
    checkpoint_path = C.resolve_path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload["config"]
    validate_config(config)
    split = D.load_split(args.split, root, device)
    rows = C.load_pairs(args.split, split.scan_ids, split.subject_ids, split.labels.cpu().numpy())
    if args.max_pairs is not None:
        rows = rows[: int(args.max_pairs)]
    model, statistics = build_model(config, root, device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    batch_size = int(args.batch_size or config["training"]["evaluation_batch_size"])
    endpoint = O.evaluate_rows(model, split, rows, batch_size)
    predictions = (
        endpoint_predictions(model, split, rows, batch_size)
        if args.surface_metrics or args.save_vertex_maps
        else {}
    )
    surface_indices: list[int] = []
    if args.surface_metrics:
        surface_indices = balanced_surface_indices(endpoint["records"], args.surface_max_pairs, args.surface_seed)
        add_surface_metrics(
            endpoint["records"],
            predictions,
            split,
            np.asarray(statistics["faces"]),
            args.surface_points,
            args.voxel_pitch_mm,
            surface_indices,
        )
    velocity_records, velocity_summary = velocity_metrics(model, split, batch_size)
    weighted_velocity_summary = O.evaluate_velocity_selection(
        model,
        split,
        int(config.get("selection", {}).get("velocity_validation_batch_size", batch_size)),
    )
    defects = O.evaluate_cocycle(
        model,
        split,
        rows,
        float(statistics["endpoint_scale_mm"]),
        batch_size,
        min(len(rows), int(config["selection"]["cocycle_validation_pairs"])),
    )
    evaluation_dir = checkpoint_path.parent.parent / "evaluation" / args.split
    if args.save_vertex_maps:
        save_mean_vertex_maps(
            evaluation_dir / "mean_vertex_maps.npz",
            endpoint["records"],
            predictions,
            split,
            np.asarray(statistics["faces"]),
        )
    write_csv(evaluation_dir / "per_pair_metrics.csv", endpoint["records"])
    write_csv(evaluation_dir / "instantaneous_velocity_metrics.csv", velocity_records)
    summary = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "operator": config["model"]["operator"],
        "epoch": int(payload["epoch"]),
        "subjects": len(set(split.subject_ids.astype(str))),
        "scans": len(split.scan_ids),
        "surface_metrics_enabled": bool(args.surface_metrics),
        "surface_metric_pairs": len(surface_indices),
        "surface_subset_balanced": bool(args.surface_metrics and len(surface_indices) < len(rows)),
        "surface_subset_seed": int(args.surface_seed),
        "vertex_maps_saved": bool(args.save_vertex_maps),
        "surface": summarize_surface(endpoint["records"]),
        "instantaneous_velocity": velocity_summary,
        "instantaneous_velocity_reliability_weighted": weighted_velocity_summary,
        "cocycle": defects,
        "adaptive_support": model.adaptive_support_report(),
        "test_data_loaded_by_training": False,
    }
    C.atomic_json(evaluation_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
