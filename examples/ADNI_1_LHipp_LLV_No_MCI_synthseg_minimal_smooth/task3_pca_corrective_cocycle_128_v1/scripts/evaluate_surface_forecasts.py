#!/usr/bin/env python3
"""Exact first-to-last surface, topology, and volume-trend comparison for fixed meshes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Import this first: it installs the verified rtree backend before trimesh
# snapshots its optional dependencies.
from surface_metrics import metrics as mesh_metrics

import trimesh

import c4_objective as O
import common as C
from evaluate import load_transport


ERROR_METRICS = (
    "prediction_coordinate_rmse_mm",
    "prediction_mean_vertex_euclidean_mm",
    "prediction_assd_mm",
    "prediction_hd95_mm",
    "prediction_chamfer_l1_mm",
    "prediction_chamfer_l2_squared_mm2",
    "prediction_volume_absolute_error_mm3",
    "prediction_volume_relative_error",
    "prediction_log_volume_rate_absolute_error_per_year",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Repeat as DISPLAY_NAME=/absolute/run/directory.",
    )
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--surface-points", type=int, default=10000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-meshes", action="store_true")
    parser.add_argument(
        "--subject-ids-npz",
        type=Path,
        default=None,
        help="Optional NPZ containing subject_ids; restrict every run to this matched cohort.",
    )
    parser.add_argument("--include-corrective-pca-branch", action="store_true")
    return parser.parse_args()


def stable_seed(text: str, base: int) -> int:
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return int((value + int(base)) % (2**31 - 1))


def parse_runs(values: list[str]) -> list[tuple[str, Path]]:
    output = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run must be DISPLAY=PATH: {value!r}")
        name, raw = value.split("=", 1)
        C.validate_run_name(name)
        path = Path(raw).expanduser().resolve()
        if not (path / "checkpoints" / "best.pt").is_file():
            raise FileNotFoundError(path / "checkpoints" / "best.pt")
        output.append((name, path))
    if len({name for name, _ in output}) != len(output):
        raise ValueError("Duplicate display name")
    return output


def selected_subjects(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        if "subject_ids" not in archive:
            raise KeyError(f"subject_ids missing from {path}")
        return set(archive["subject_ids"].astype(str))


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    value = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return abs(float(value.volume))


def coordinate_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "coordinate_rmse_mm": float(np.sqrt(np.square(delta).mean())),
        "mean_vertex_euclidean_mm": float(np.linalg.norm(delta, axis=-1).mean()),
    }


def export_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)


def evaluate_mesh(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    faces: np.ndarray,
    surface_points: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = coordinate_metrics(prediction, ground_truth)
    output.update(mesh_metrics(ground_truth, prediction, faces, surface_points, seed))
    return output


def prefix(values: dict[str, Any], name: str) -> dict[str, Any]:
    return {f"{name}_{key}": value for key, value in values.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    numeric = [
        key
        for key, value in rows[0].items()
        if key not in {"representation", "source_representation", "subject_id", "diagnosis", "source_scan_id", "target_scan_id", "split"}
        and isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
    ]
    for representation in sorted({str(row["representation"]) for row in rows}):
        selected = [row for row in rows if row["representation"] == representation]
        for diagnosis in ("CN", "AD", "overall"):
            group = selected if diagnosis == "overall" else [row for row in selected if row["diagnosis"] == diagnosis]
            if not group:
                continue
            record: dict[str, Any] = {
                "representation": representation,
                "diagnosis": diagnosis,
                "subjects": len(group),
            }
            for key in numeric:
                record[f"{key}_mean"] = mean(group, key)
                record[f"{key}_median"] = float(np.median([float(row[key]) for row in group]))
            for metric in ("assd_mm", "hd95_mm", "chamfer_l2_squared_mm2", "volume_relative_error"):
                prediction = record[f"prediction_{metric}_mean"]
                nochange = record[f"nochange_{metric}_mean"]
                record[f"prediction_{metric}_ratio_to_nochange"] = prediction / max(nochange, 1.0e-12)
            output.append(record)
    return output


def paired_bootstrap(
    rows: list[dict[str, Any]], samples: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["representation"])][str(row["subject_id"])] = row
    if "pca128" not in grouped:
        return []
    output = []
    for representation, current in sorted(grouped.items()):
        if representation == "pca128":
            continue
        subjects = sorted(set(current).intersection(grouped["pca128"]))
        if not subjects:
            continue
        for metric in ERROR_METRICS:
            differences = np.asarray(
                [float(current[s][metric]) - float(grouped["pca128"][s][metric]) for s in subjects],
                dtype=np.float64,
            )
            rng = np.random.default_rng(stable_seed(representation + metric, seed))
            indices = rng.integers(0, len(subjects), size=(samples, len(subjects)))
            estimates = differences[indices].mean(axis=1)
            output.append({
                "representation": representation,
                "baseline": "pca128",
                "metric": metric,
                "subjects": len(subjects),
                "current_minus_baseline_mean": float(differences.mean()),
                "ci95_low": float(np.quantile(estimates, 0.025)),
                "ci95_high": float(np.quantile(estimates, 0.975)),
                "fraction_subjects_better": float(np.mean(differences < 0.0)),
                "negative_favors_current": True,
            })
    return output


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise PermissionError("Test evaluation requires explicit --allow-test")
    if args.surface_points < 100:
        raise ValueError("Use at least 100 surface points")
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    runs = parse_runs(args.run)
    destination = C.require_bulk_path(args.output_dir, "surface comparison output")
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=False)
    keep_subjects = selected_subjects(args.subject_ids_npz)
    device = C.choose_device(args.device)
    registry = C.load_registry()
    raw_vertices = C.cached_vertices(args.split, registry)
    all_rows: list[dict[str, Any]] = []
    nochange_cache: dict[str, dict[str, Any]] = {}

    for display_name, run_dir in runs:
        resolved = C.read_json(run_dir / "resolved_config.json")
        config = resolved["config"]
        representation = str(config["representation"])
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if bool(checkpoint.get("test_data_loaded", False)):
            raise ValueError(f"Training/test leakage recorded in {checkpoint_path}")
        train_archive = C.load_archive(representation, "train", registry)
        archive = C.load_archive(representation, args.split, registry)
        geometry = C.build_geometry(representation, train_archive, device, registry)
        values = C.values_on_device(archive, device)
        O.attach_reference_geometry(values, geometry, int(config["training"].get("decoder_batch_size", 64)))
        transport = load_transport(config, checkpoint, device)
        pairs = C.first_last_pairs(archive)
        if keep_subjects is not None:
            pairs = [row for row in pairs if row.subject in keep_subjects]
        if args.max_subjects is not None:
            pairs = pairs[: int(args.max_subjects)]
        faces = geometry.faces.detach().cpu().numpy()

        for pair in pairs:
            source = pair.source
            target = pair.target
            latent = transport.transport(
                values["z"][source : source + 1],
                values["age"][source : source + 1],
                values["age"][target : target + 1],
                values["label"][source : source + 1],
            )
            prediction = geometry.vertices(latent)[0].detach().cpu().numpy()
            decoded_source = values["reference_vertices"][source].detach().cpu().numpy()
            decoded_target = values["reference_vertices"][target].detach().cpu().numpy()
            raw_source = np.asarray(raw_vertices[source], dtype=np.float32)
            raw_target = np.asarray(raw_vertices[target], dtype=np.float32)
            scan_id = str(archive["visit_scan_ids"][target])
            source_scan_id = str(archive["visit_scan_ids"][source])
            sample_seed = stable_seed(scan_id, args.seed)
            if pair.subject not in nochange_cache:
                nochange_cache[pair.subject] = evaluate_mesh(
                    raw_target, raw_source, faces, args.surface_points, sample_seed
                )
            predicted_values = evaluate_mesh(
                raw_target, prediction, faces, args.surface_points, sample_seed
            )
            floor_values = evaluate_mesh(
                raw_target, decoded_target, faces, args.surface_points, sample_seed
            )
            transport_values = coordinate_metrics(prediction, decoded_target)

            source_true_volume = mesh_volume(raw_source, faces)
            target_true_volume = mesh_volume(raw_target, faces)
            source_decoded_volume = mesh_volume(decoded_source, faces)
            predicted_volume = mesh_volume(prediction, faces)
            years = max(float(pair.delta_years), 1.0e-8)
            observed_rate = float(np.log(target_true_volume / source_true_volume) / years)
            predicted_rate_raw_anchor = float(np.log(predicted_volume / source_true_volume) / years)
            predicted_rate_decoded_anchor = float(np.log(predicted_volume / source_decoded_volume) / years)
            row: dict[str, Any] = {
                "representation": display_name,
                "source_representation": representation,
                "split": args.split,
                "subject_id": pair.subject,
                "diagnosis": pair.diagnosis,
                "source_scan_id": source_scan_id,
                "target_scan_id": scan_id,
                "followup_years": years,
                **prefix(predicted_values, "prediction"),
                **prefix(nochange_cache[pair.subject], "nochange"),
                **prefix(floor_values, "floor"),
                **prefix(transport_values, "transport"),
                "source_true_volume_mm3": source_true_volume,
                "target_true_volume_mm3": target_true_volume,
                "source_decoded_volume_mm3": source_decoded_volume,
                "prediction_volume_mm3": predicted_volume,
                "prediction_volume_signed_error_mm3": predicted_volume - target_true_volume,
                "prediction_volume_absolute_error_mm3": abs(predicted_volume - target_true_volume),
                "nochange_volume_absolute_error_mm3": abs(source_true_volume - target_true_volume),
                "observed_signed_log_volume_rate_per_year": observed_rate,
                "prediction_signed_log_volume_rate_raw_anchor_per_year": predicted_rate_raw_anchor,
                "prediction_signed_log_volume_rate_decoded_anchor_per_year": predicted_rate_decoded_anchor,
                "prediction_log_volume_rate_absolute_error_per_year": abs(predicted_rate_raw_anchor - observed_rate),
                "nochange_log_volume_rate_absolute_error_per_year": abs(observed_rate),
                "observed_volume_change_mm3_per_year": (target_true_volume - source_true_volume) / years,
                "prediction_volume_change_raw_anchor_mm3_per_year": (predicted_volume - source_true_volume) / years,
                "prediction_volume_change_decoded_anchor_mm3_per_year": (predicted_volume - source_decoded_volume) / years,
                "observed_annualized_percent_change": 100.0 * np.expm1(observed_rate),
                "prediction_annualized_percent_change_raw_anchor": 100.0 * np.expm1(predicted_rate_raw_anchor),
                "prediction_atrophy_direction_agreement": float(np.sign(predicted_rate_raw_anchor) == np.sign(observed_rate)),
            }
            all_rows.append(row)
            if args.write_meshes:
                export_mesh(destination / "meshes" / display_name / "prediction" / f"{scan_id}.ply", prediction, faces)
                export_mesh(destination / "meshes" / display_name / "floor" / f"{scan_id}.ply", decoded_target, faces)
                shared = destination / "meshes" / "ground_truth"
                target_path = shared / "target" / f"{scan_id}.ply"
                source_path = shared / "source" / f"{source_scan_id}.ply"
                if not target_path.exists():
                    export_mesh(target_path, raw_target, faces)
                if not source_path.exists():
                    export_mesh(source_path, raw_source, faces)

            if args.include_corrective_pca_branch and isinstance(geometry, C.FrozenCorrectiveGeometry):
                pca_prediction = geometry.pca_vertices(latent)[0].detach().cpu().numpy()
                pca_decoded_source = geometry.pca_vertices(
                    values["z"][source : source + 1]
                )[0].detach().cpu().numpy()
                pca_decoded_target = geometry.pca_vertices(
                    values["z"][target : target + 1]
                )[0].detach().cpu().numpy()
                pca_values = evaluate_mesh(raw_target, pca_prediction, faces, args.surface_points, sample_seed)
                pca_floor_values = evaluate_mesh(
                    raw_target, pca_decoded_target, faces, args.surface_points, sample_seed
                )
                pca_volume = mesh_volume(pca_prediction, faces)
                pca_decoded_source_volume = mesh_volume(pca_decoded_source, faces)
                pca_rate = float(np.log(pca_volume / source_true_volume) / years)
                pca_decoded_rate = float(
                    np.log(pca_volume / pca_decoded_source_volume) / years
                )
                branch = dict(row)
                branch["representation"] = f"{display_name}_same_flow_pca_decode"
                branch.update(prefix(pca_values, "prediction"))
                branch.update(prefix(pca_floor_values, "floor"))
                branch.update(
                    prefix(coordinate_metrics(pca_prediction, pca_decoded_target), "transport")
                )
                branch["source_decoded_volume_mm3"] = pca_decoded_source_volume
                branch["prediction_volume_mm3"] = pca_volume
                branch["prediction_volume_signed_error_mm3"] = pca_volume - target_true_volume
                branch["prediction_volume_absolute_error_mm3"] = abs(pca_volume - target_true_volume)
                branch["prediction_signed_log_volume_rate_raw_anchor_per_year"] = pca_rate
                branch[
                    "prediction_signed_log_volume_rate_decoded_anchor_per_year"
                ] = pca_decoded_rate
                branch["prediction_log_volume_rate_absolute_error_per_year"] = abs(pca_rate - observed_rate)
                branch["prediction_volume_change_raw_anchor_mm3_per_year"] = (pca_volume - source_true_volume) / years
                branch["prediction_volume_change_decoded_anchor_mm3_per_year"] = (
                    pca_volume - pca_decoded_source_volume
                ) / years
                branch["prediction_annualized_percent_change_raw_anchor"] = 100.0 * np.expm1(pca_rate)
                branch["prediction_atrophy_direction_agreement"] = float(np.sign(pca_rate) == np.sign(observed_rate))
                all_rows.append(branch)
                if args.write_meshes:
                    export_mesh(
                        destination / "meshes" / branch["representation"] / "prediction" / f"{scan_id}.ply",
                        pca_prediction,
                        faces,
                    )
        print(f"[{display_name}] evaluated {len(pairs)} first-to-last {args.split} subjects", flush=True)

    summary_rows = summarize(all_rows)
    bootstrap_rows = paired_bootstrap(all_rows, args.bootstrap_samples, args.seed)
    write_csv(destination / "per_subject.csv", all_rows)
    write_csv(destination / "summary.csv", summary_rows)
    if bootstrap_rows:
        write_csv(destination / "paired_bootstrap_vs_pca.csv", bootstrap_rows)
    C.atomic_json(destination / "summary.json", {
        "schema_version": 1,
        "split": args.split,
        "surface_points_per_direction": args.surface_points,
        "sampling": "deterministic common random numbers; exact point-to-triangle proximity",
        "volume_definition": "absolute watertight triangle-mesh volume in mm3",
        "trend_definition": "first-to-last signed log-volume rate per year; raw-source anchor is primary",
        "matched_subject_filter": sorted(keep_subjects) if keep_subjects is not None else None,
        "runs": {name: str(path) for name, path in runs},
        "summary_rows": summary_rows,
        "bootstrap_rows": bootstrap_rows,
        "test_was_explicitly_authorized": bool(args.split == "test" and args.allow_test),
    })
    print(f"WROTE {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
