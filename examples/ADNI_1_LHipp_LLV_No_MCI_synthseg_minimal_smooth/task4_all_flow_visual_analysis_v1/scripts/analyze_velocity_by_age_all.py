#!/usr/bin/env python3
"""Matched age-stratified surface velocity for every completed flow family."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-task4-all-age-velocity")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/task4-all-age-velocity-xdg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import build_analysis_cache as B


REPO = Path(__file__).resolve().parents[4]
TASK = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = TASK / "configs" / "age_velocity_all_methods.json"
TASK5_SCRIPTS = (
    REPO
    / "examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
    / "task5_direct_mesh_cocycle_spiral_unet_v1/scripts"
)
if str(TASK5_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TASK5_SCRIPTS))
import analyze_velocity_by_age as A


FULL_METHOD_ORDER = (
    "mesh_spiral",
    "mesh_adaptive",
    "latent_pca",
    "latent_spiral",
    "latent_adaptive",
    "lamm_n3",
)
ALL_METHOD_ORDER = (*FULL_METHOD_ORDER, "latent_inr")
COLORS = {
    "mesh_spiral": "#e76f00",
    "mesh_adaptive": "#8c2d91",
    "latent_pca": "#4c78a8",
    "latent_spiral": "#59a14f",
    "latent_adaptive": "#f2cf5b",
    "lamm_n3": "#e15759",
    "latent_inr": "#79706e",
}
LINESTYLES = {
    "mesh_spiral": "-",
    "mesh_adaptive": "-",
    "latent_pca": "--",
    "latent_spiral": "--",
    "latent_adaptive": "--",
    "lamm_n3": "-.",
    "latent_inr": ":",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run inference on eight balanced visits, validate contracts, and write nothing.",
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vertices, dtype=np.float64)
    triangles = vertices[:, faces]
    face_normals = np.cross(triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 0])
    for batch in range(len(vertices)):
        for corner in range(3):
            np.add.at(output[batch], faces[:, corner], face_normals[batch])
    norm = np.linalg.norm(output, axis=2, keepdims=True)
    return output / np.maximum(norm, 1.0e-12)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1.0e-12 else 0.0


def vector_record(
    method: str,
    label: str,
    family: str,
    definition: str,
    predicted: np.ndarray,
    observed: np.ndarray,
    normals: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    normal = np.asarray(normals, dtype=np.float64)
    predicted_normal = np.sum(predicted * normal, axis=1)
    observed_normal = np.sum(observed * normal, axis=1)
    predicted_flat = predicted.reshape(-1)
    observed_flat = observed.reshape(-1)
    denominator = float(np.linalg.norm(predicted_flat) * np.linalg.norm(observed_flat))
    return {
        "method": method,
        "method_label": label,
        "model_family": family,
        "velocity_definition": definition,
        **metadata,
        "predicted_speed_mm_per_year": float(np.linalg.norm(predicted, axis=1).mean()),
        "observed_speed_mm_per_year": float(np.linalg.norm(observed, axis=1).mean()),
        "predicted_inward_normal_mm_per_year": float(-predicted_normal.mean()),
        "observed_inward_normal_mm_per_year": float(-observed_normal.mean()),
        "vector_rmse_mm_per_year": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "zero_vector_rmse_mm_per_year": float(np.sqrt(np.mean(observed**2))),
        "normal_rmse_mm_per_year": float(np.sqrt(np.mean((predicted_normal - observed_normal) ** 2))),
        "zero_normal_rmse_mm_per_year": float(np.sqrt(np.mean(observed_normal**2))),
        "vector_cosine": float(np.dot(predicted_flat, observed_flat) / denominator) if denominator > 1.0e-12 else 0.0,
        "normal_pearson": safe_corr(predicted_normal, observed_normal),
        "normal_sign_agreement": float(np.mean(np.sign(predicted_normal) == np.sign(observed_normal))),
    }


def reference_metadata(reference: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    return {
        "scan_id": str(reference["scan_ids"][index]),
        "subject_id": str(reference["subject_ids"][index]),
        "diagnosis": str(reference["diagnoses"][index]),
        "age_years": float(reference["age_years"][index]),
        "reference_reliability": float(reference["velocity_reference_weight"][index]),
    }


def selected_reference_indices(
    reference: dict[str, np.ndarray], smoke: bool, allowed_scans: set[str] | None = None
) -> list[int]:
    candidates = [
        index
        for index, weight in enumerate(reference["velocity_reference_weight"])
        if float(weight) > 0.0
        and (allowed_scans is None or str(reference["scan_ids"][index]) in allowed_scans)
    ]
    if not smoke:
        return candidates
    output: list[int] = []
    groups = {
        diagnosis: [index for index in candidates if str(reference["diagnoses"][index]) == diagnosis]
        for diagnosis in ("CN", "AD")
    }
    for offset in range(4):
        for diagnosis in ("CN", "AD"):
            output.append(groups[diagnosis][offset])
    return sorted(output)


def load_reference(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in archive.files}
    task5_root = path.parent.parent
    statistics = torch.load(task5_root / "cache" / "training_statistics.pt", map_location="cpu", weights_only=False)
    faces = np.asarray(statistics["faces"], dtype=np.int64)
    if reference["vertices_mm"].shape[1:] != (2746, 3):
        raise ValueError("Unexpected direct-mesh reference topology")
    return reference, faces


def load_direct_records(
    config: dict[str, Any], selected_scans: set[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, spec in config["direct_mesh"].items():
        run = resolve(spec["run_dir"])
        path = run / "evaluation" / config["split"] / "age_velocity" / "per_visit_velocity.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing direct-mesh age-velocity table: {path}; run task5 analyze_velocity_by_age.py first"
            )
        frame = pd.read_csv(path, dtype={"scan_id": str, "subject_id": str})
        frame = frame[frame.scan_id.astype(str).isin(selected_scans)]
        for row in frame.to_dict("records"):
            row.update(
                {
                    "method": method,
                    "method_label": str(spec["label"]),
                    "model_family": "direct_mesh",
                    "velocity_definition": "explicit corresponding-mesh zero-horizon field",
                }
            )
            output.append(row)
    return output


def audit_checkpoint(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = read_json(run / "resolved_config.json")
    checkpoint = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    if bool(checkpoint.get("test_data_loaded", False)):
        raise ValueError(f"Training/test leakage recorded in {run}")
    return resolved["config"], checkpoint


def explicit_decoder_velocity(
    flow: torch.nn.Module,
    geometry: torch.nn.Module,
    latent: torch.Tensor,
    age: torch.Tensor,
    label: torch.Tensor,
    age_slope: float,
    batch_size: int,
) -> np.ndarray:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(latent), int(batch_size)):
        z = latent[start : start + batch_size]
        t = age[start : start + batch_size]
        d = label[start : start + batch_size]
        with torch.no_grad():
            latent_velocity = flow.average_velocity(z, t, t, d) * float(age_slope)
        z_input = z.detach().requires_grad_(True)
        _, velocity = torch.autograd.functional.jvp(
            geometry.vertices,
            z_input,
            latent_velocity.detach(),
            create_graph=False,
            strict=False,
        )
        chunks.append(velocity.detach().cpu())
    return torch.cat(chunks).numpy().astype(np.float64)


def current_explicit_records(
    config: dict[str, Any],
    reference: dict[str, np.ndarray],
    normals: np.ndarray,
    selected_scans: set[str],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    task = resolve(config["latent_128_task"])
    script_path, common, models = B.load_local_modules(task)
    output: list[dict[str, Any]] = []
    scan_to_reference = {str(scan): index for index, scan in enumerate(reference["scan_ids"])}
    try:
        for method, spec in config["latent_128"].items():
            run = resolve(spec["run_dir"])
            resolved, checkpoint = audit_checkpoint(run)
            archive = common.load_archive(str(spec["representation"]), config["split"])
            train = common.load_archive(str(spec["representation"]), "train")
            geometry = common.build_geometry(str(spec["representation"]), train, device)
            flow = B.load_direct_flow(models, resolved, checkpoint, device)
            scans = archive["visit_scan_ids"].astype(str)
            chosen = np.asarray([index for index, scan in enumerate(scans) if scan in selected_scans], dtype=np.int64)
            z = torch.from_numpy(archive["visit_latent_standardized_128"][chosen].astype(np.float32)).to(device)
            age = torch.from_numpy(archive["visit_age_norm_train"][chosen].astype(np.float32)).to(device)
            label = torch.from_numpy(archive["visit_label_ad"][chosen].astype(np.float32)).to(device)
            velocity = explicit_decoder_velocity(
                flow, geometry, z, age, label, B.age_slope_per_year(archive), batch_size
            )
            for local, archive_index in enumerate(chosen):
                ref = scan_to_reference[str(scans[archive_index])]
                output.append(
                    vector_record(
                        method,
                        str(spec["label"]),
                        "latent_explicit_decoder",
                        "latent zero-horizon field pushed through frozen decoder JVP",
                        velocity[local],
                        reference["velocity_reference_mm_per_year"][ref],
                        normals[ref],
                        reference_metadata(reference, ref),
                    )
                )
            del geometry, flow, z, velocity
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        B.unload_local_modules(script_path)
    return output


def lamm_n3_records(
    config: dict[str, Any],
    reference: dict[str, np.ndarray],
    normals: np.ndarray,
    selected_scans: set[str],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], float | None]:
    task = resolve(config["lamm_task"])
    # The LAMM task intentionally reuses the generic 128-D flow core and selects
    # its representations through an environment-owned registry.
    core_task = task.parent / "task3_latent_flow_128_v1"
    registry_key = "DEEP3DCOMP_LATENT_FLOW_REGISTRY"
    previous_registry = os.environ.get(registry_key)
    os.environ[registry_key] = str((task / "configs" / "representations.json").resolve())
    script_path, common, models = B.load_local_modules(core_task)
    spec = config["lamm_n3"]
    member_velocities: list[np.ndarray] = []
    chosen_scans: np.ndarray | None = None
    chosen_archive: dict[str, np.ndarray] | None = None
    try:
        for member in spec["members"]:
            run = resolve(member["run_dir"])
            resolved, checkpoint = audit_checkpoint(run)
            representation = str(member["representation"])
            archive = common.load_archive(representation, config["split"])
            train = common.load_archive(representation, "train")
            scans = archive["visit_scan_ids"].astype(str)
            chosen = np.asarray([index for index, scan in enumerate(scans) if scan in selected_scans], dtype=np.int64)
            if chosen_scans is None:
                chosen_scans = scans[chosen]
                chosen_archive = archive
            elif not np.array_equal(chosen_scans, scans[chosen]):
                raise ValueError("LAMM N3 member visit ordering is not aligned")
            geometry = common.build_geometry(representation, train, device)
            flow = B.load_direct_flow(models, resolved, checkpoint, device)
            z = torch.from_numpy(archive["visit_latent_standardized_128"][chosen].astype(np.float32)).to(device)
            age = torch.from_numpy(archive["visit_age_norm_train"][chosen].astype(np.float32)).to(device)
            label = torch.from_numpy(archive["visit_label_ad"][chosen].astype(np.float32)).to(device)
            member_velocities.append(
                explicit_decoder_velocity(
                    flow, geometry, z, age, label, B.age_slope_per_year(archive), batch_size
                )
            )
            del geometry, flow, z
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        B.unload_local_modules(script_path)
        if previous_registry is None:
            os.environ.pop(registry_key, None)
        else:
            os.environ[registry_key] = previous_registry
    if chosen_scans is None or chosen_archive is None:
        raise ValueError("No LAMM N3 visits selected")
    velocity = np.mean(np.stack(member_velocities), axis=0)
    scan_to_reference = {str(scan): index for index, scan in enumerate(reference["scan_ids"])}
    output = []
    for local, scan in enumerate(chosen_scans):
        ref = scan_to_reference[str(scan)]
        output.append(
            vector_record(
                str(spec["key"]),
                str(spec["label"]),
                "latent_lamm_n3_shape_ensemble",
                "three zero-horizon latent fields pushed through matched decoder JVPs, then averaged in mesh space",
                velocity[local],
                reference["velocity_reference_mm_per_year"][ref],
                normals[ref],
                reference_metadata(reference, ref),
            )
        )

    crosscheck: float | None = None
    existing_path = resolve(spec["existing_validation_subjects"])
    if existing_path.is_file() and len(selected_scans) == len(reference["scan_ids"]):
        existing = pd.read_csv(existing_path, dtype={"subject": str})
        offsets = chosen_archive["subject_visit_offsets"].astype(np.int64)
        baseline_scans = chosen_archive["visit_scan_ids"][offsets[:-1]].astype(str)
        predicted = {str(row["scan_id"]): float(row["predicted_speed_mm_per_year"]) for row in output}
        expected = dict(zip(existing.subject.astype(str), existing.mean_vertex_speed_mm_per_year.astype(float)))
        differences = []
        for subject, scan in zip(chosen_archive["subject_ids"].astype(str), baseline_scans):
            differences.append(abs(predicted[str(scan)] - expected[str(subject)]))
        crosscheck = float(max(differences, default=0.0))
        if crosscheck > 2.0e-5:
            raise ValueError(f"LAMM N3 baseline velocity cross-check failed: {crosscheck}")
    return output, crosscheck


def implicit_normal_velocity(
    geometry: torch.nn.Module,
    latent: torch.Tensor,
    latent_velocity: torch.Tensor,
    vertices_mm: np.ndarray,
    normals_mm: np.ndarray,
    epsilon_years: float,
) -> np.ndarray:
    xyz_np = ((vertices_mm / geometry.distance_unscale_factor) - geometry.range_global_min)
    xyz_np = xyz_np * geometry.range_linear_scale_factor + geometry.target_range_min
    points = torch.from_numpy(xyz_np.astype(np.float32)).to(latent.device).requires_grad_(True)
    baseline = geometry.sdf(latent, points)[0]
    gradient = torch.autograd.grad(baseline.sum(), points, create_graph=False)[0]
    with torch.no_grad():
        projected = points - baseline[:, None] * gradient / gradient.square().sum(dim=1, keepdim=True).clamp_min(1.0e-10)
    projected = projected.detach().requires_grad_(True)
    z_input = latent.detach().requires_grad_(True)
    baseline = geometry.sdf(z_input, projected)[0]
    gradient = torch.autograd.grad(baseline.sum(), projected, retain_graph=True, create_graph=False)[0]
    with torch.no_grad():
        epsilon = float(epsilon_years)
        future = geometry.sdf(z_input + epsilon * latent_velocity, projected)[0]
        past = geometry.sdf(z_input - epsilon * latent_velocity, projected)[0]
        sdf_rate = (future - past) / (2.0 * epsilon)
        gradient_norm = torch.linalg.vector_norm(gradient, dim=1).clamp_min(1.0e-8)
        gradient_unit = gradient / gradient_norm[:, None]
        raw_normals = torch.from_numpy(normals_mm.astype(np.float32)).to(latent.device)
        alignment = torch.sum(gradient_unit * raw_normals, dim=1)
        normal_speed = -sdf_rate * alignment / gradient_norm * float(geometry.linear_normalized_to_mm)
    return normal_speed.detach().cpu().numpy().astype(np.float64)


def inr_records(
    config: dict[str, Any],
    reference: dict[str, np.ndarray],
    normals: np.ndarray,
    selected_scans: set[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    task = resolve(config["inr_task"])
    script_path, common, models = B.load_local_modules(task)
    try:
        import importlib

        inr_geometry = importlib.import_module("inr_geometry")
        spec = config["inr"]
        run = resolve(spec["run_dir"])
        resolved, checkpoint = audit_checkpoint(run)
        archive = common.load_archive(config["split"])
        train = common.load_archive("train")
        geometry = inr_geometry.build_geometry(train, device)
        flow = B.load_direct_flow(models, resolved, checkpoint, device)
        scans = archive["visit_scan_ids"].astype(str)
        chosen = [index for index, scan in enumerate(scans) if scan in selected_scans]
        scan_to_reference = {str(scan): index for index, scan in enumerate(reference["scan_ids"])}
        slope = B.age_slope_per_year(archive)
        output = []
        for number, archive_index in enumerate(chosen):
            ref = scan_to_reference[str(scans[archive_index])]
            z = torch.from_numpy(archive["visit_latent_standardized_256"][archive_index : archive_index + 1].astype(np.float32)).to(device)
            age = torch.as_tensor(archive["visit_age_norm_train"][archive_index : archive_index + 1], dtype=torch.float32, device=device)
            label = torch.as_tensor(archive["visit_label_ad"][archive_index : archive_index + 1], dtype=torch.float32, device=device)
            with torch.no_grad():
                latent_velocity = flow.average_velocity(z, age, age, label) * float(slope)
            normal_speed = implicit_normal_velocity(
                geometry,
                z,
                latent_velocity,
                reference["vertices_mm"][ref],
                normals[ref],
                float(spec["surface_derivative_epsilon_years"]),
            )
            predicted = normal_speed[:, None] * normals[ref]
            output.append(
                vector_record(
                    str(spec["key"]),
                    str(spec["label"]),
                    "latent_implicit_decoder",
                    "centered small-time level-set normal derivative from latent field and frozen INR decoder",
                    predicted,
                    reference["velocity_reference_mm_per_year"][ref],
                    normals[ref],
                    reference_metadata(reference, ref),
                )
            )
            if (number + 1) % 20 == 0:
                print(f"INR velocity {number + 1}/{len(chosen)}", flush=True)
        return output
    finally:
        B.unload_local_modules(script_path)


def summarize_methods(
    records: list[dict[str, Any]],
    methods: Iterable[str],
    edges: np.ndarray,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method_index, method in enumerate(methods):
        current = [row for row in records if str(row["method"]) == method]
        if not current:
            continue
        label = str(current[0]["method_label"])
        family = str(current[0]["model_family"])
        definition = str(current[0]["velocity_definition"])
        for row in A.age_summary(current, edges, samples, seed + 10007 * method_index):
            output.append(
                {
                    "method": method,
                    "method_label": label,
                    "model_family": family,
                    "velocity_definition": definition,
                    **row,
                }
            )
    return output


def legacy_records(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, dtype={"scan_id": str, "subject_id": str})
    output: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        observed = float(row["observed_gt_normal_rms_mm_per_year"])
        predicted = float(row["model_normal_rms_mm_per_year"])
        output.append(
            {
                "method": str(row["method"]),
                "method_label": str(row["method_label"]),
                "model_family": "historical_different_protocol",
                "velocity_definition": str(row["generator_kind"]),
                "scan_id": str(row["scan_id"]),
                "subject_id": str(row["subject_id"]),
                "diagnosis": str(row["diagnosis"]),
                "age_years": float(row["age_years"]),
                "reference_reliability": 1.0,
                "predicted_speed_mm_per_year": predicted,
                "observed_speed_mm_per_year": observed,
                "predicted_inward_normal_mm_per_year": -float(row["model_normal_mean_mm_per_year"]),
                "observed_inward_normal_mm_per_year": -float(row["observed_gt_normal_mean_mm_per_year"]),
                "vector_rmse_mm_per_year": float(row["normal_rmse_mm_per_year"]),
                "zero_vector_rmse_mm_per_year": observed,
                "normal_rmse_mm_per_year": float(row["normal_rmse_mm_per_year"]),
                "zero_normal_rmse_mm_per_year": observed,
                "vector_cosine": float(row["normal_pearson"]),
                "normal_pearson": float(row["normal_pearson"]),
                "normal_sign_agreement": float(row["normal_sign_agreement"]),
            }
        )
    return output


def summary_rows(summary: list[dict[str, Any]], method: str, diagnosis: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in summary if row["method"] == method and row["diagnosis"] == diagnosis],
        key=lambda row: int(row["age_bin_index"]),
    )


def plot_speed(
    summary: list[dict[str, Any]], methods: Iterable[str], path: Path, title: str
) -> None:
    methods = list(methods)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.3), sharey=True)
    for axis, diagnosis in zip(axes, ("CN", "AD")):
        observed_rows = summary_rows(summary, methods[0], diagnosis)
        x = np.arange(len(observed_rows))
        observed = [float(row["observed_speed_mm_per_year"]) for row in observed_rows]
        axis.plot(x, observed, color="black", marker="o", linewidth=2.4, label="Observed velocity")
        for method in methods:
            rows = summary_rows(summary, method, diagnosis)
            if len(rows) != len(observed_rows):
                continue
            axis.plot(
                x,
                [float(row["predicted_speed_mm_per_year"]) for row in rows],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                marker="s",
                label=str(rows[0]["method_label"]),
            )
        axis.set_xticks(x, [str(row["age_bin"]) for row in observed_rows])
        axis.set_xlabel("Age interval (years)")
        axis.set_title(diagnosis)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean vertex speed (mm/year)")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(title + "\nObserved = fitted longitudinal-visit derivative", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_inward(
    summary: list[dict[str, Any]], methods: Iterable[str], path: Path, title: str
) -> None:
    methods = list(methods)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.3), sharey=True)
    for axis, diagnosis in zip(axes, ("CN", "AD")):
        observed_rows = summary_rows(summary, methods[0], diagnosis)
        x = np.arange(len(observed_rows))
        axis.plot(
            x,
            [float(row["observed_inward_normal_mm_per_year"]) for row in observed_rows],
            color="black",
            marker="o",
            linewidth=2.4,
            label="Observed velocity",
        )
        for method in methods:
            rows = summary_rows(summary, method, diagnosis)
            if len(rows) != len(observed_rows):
                continue
            axis.plot(
                x,
                [float(row["predicted_inward_normal_mm_per_year"]) for row in rows],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                marker="s",
                label=str(rows[0]["method_label"]),
            )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xticks(x, [str(row["age_bin"]) for row in observed_rows])
        axis.set_xlabel("Age interval (years)")
        axis.set_title(diagnosis)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Inward-normal velocity (mm/year; positive = shrinkage)")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(title + "\nNormal motion is comparable for explicit meshes and the implicit INR", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_normal_agreement(
    summary: list[dict[str, Any]], methods: Iterable[str], path: Path, title: str
) -> None:
    methods = list(methods)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex="col")
    for row_index, diagnosis in enumerate(("CN", "AD")):
        reference_rows = summary_rows(summary, methods[0], diagnosis)
        x = np.arange(len(reference_rows))
        for column, (metric, ylabel, reference_line) in enumerate(
            (
                ("normal_error_to_zero_ratio", "Normal error / zero-velocity error", 1.0),
                ("normal_pearson", "Normal-velocity spatial correlation", 0.0),
            )
        ):
            axis = axes[row_index, column]
            for method in methods:
                rows = summary_rows(summary, method, diagnosis)
                if len(rows) != len(reference_rows):
                    continue
                axis.plot(
                    x,
                    [float(item[metric]) for item in rows],
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    marker="o",
                    label=str(rows[0]["method_label"]),
                )
            axis.axhline(reference_line, color="black", linestyle="--", linewidth=0.8)
            axis.set_ylabel(f"{diagnosis}: {ylabel}")
            axis.grid(axis="y", alpha=0.25)
            axis.set_xticks(x, [str(item["age_bin"]) for item in reference_rows])
    axes[1, 0].set_xlabel("Age interval (years)")
    axes[1, 1].set_xlabel("Age interval (years)")
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(title + "\nBelow 1 is better for error ratio; correlation is best near 1", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_legacy(summary: list[dict[str, Any]], path: Path) -> None:
    methods = list(dict.fromkeys(str(row["method"]) for row in summary))
    colors = dict(zip(methods, ("#4c78a8", "#f58518", "#54a24b", "#e45756")))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for axis, diagnosis in zip(axes, ("CN", "AD")):
        reference = summary_rows(summary, methods[0], diagnosis)
        x = np.arange(len(reference))
        axis.plot(x, [float(row["observed_speed_mm_per_year"]) for row in reference], color="black", marker="o", linewidth=2.3, label="Observed velocity")
        for method in methods:
            rows = summary_rows(summary, method, diagnosis)
            axis.plot(x, [float(row["predicted_speed_mm_per_year"]) for row in rows], marker="s", color=colors[method], label=str(rows[0]["method_label"]))
        axis.set_xticks(x, [str(row["age_bin"]) for row in reference])
        axis.set_xlabel("Age interval (years)")
        axis.set_title(diagnosis)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Surface-normal RMS velocity (mm/year)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Historical ODE/BrainODE comparison by age (separate cohort and protocol)\n"
        "Displayed for context only; values must not be ranked against the primary validation cohort",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_records(records: list[dict[str, Any]], expected: set[str]) -> dict[str, Any]:
    methods = {str(row["method"]) for row in records}
    if methods != expected:
        raise ValueError(f"Method mismatch: expected={sorted(expected)}, actual={sorted(methods)}")
    numeric_fields = (
        "age_years",
        "reference_reliability",
        "predicted_speed_mm_per_year",
        "observed_speed_mm_per_year",
        "normal_rmse_mm_per_year",
        "normal_pearson",
    )
    for row in records:
        if not all(math.isfinite(float(row[field])) for field in numeric_fields):
            raise ValueError(f"Non-finite velocity record: {row['method']} {row['scan_id']}")
    return {
        method: {
            "visits": sum(str(row["method"]) == method for row in records),
            "subjects": len({str(row["subject_id"]) for row in records if str(row["method"]) == method}),
        }
        for method in sorted(methods)
    }


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    if config.get("split") != "val":
        raise ValueError("This analysis is validation-only; test must remain untouched")
    edges = A.validate_age_edges(config["age_bin_edges_years"])
    samples = int(args.bootstrap_samples or config["bootstrap_samples"])
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)

    reference, faces = load_reference(resolve(config["observed_reference"]))
    smoke_allowed = None
    if args.smoke:
        with np.load(resolve(config["inr"]["validation_archive"]), allow_pickle=False) as archive:
            smoke_allowed = set(archive["visit_scan_ids"].astype(str))
    selected = selected_reference_indices(reference, args.smoke, smoke_allowed)
    selected_scans = {str(reference["scan_ids"][index]) for index in selected}
    normals = vertex_normals(reference["vertices_mm"].astype(np.float64), faces)
    records = load_direct_records(config, selected_scans)
    records.extend(
        current_explicit_records(config, reference, normals, selected_scans, device, args.batch_size)
    )
    lamm_records, lamm_crosscheck = lamm_n3_records(
        config, reference, normals, selected_scans, device, args.batch_size
    )
    records.extend(lamm_records)
    records.extend(inr_records(config, reference, normals, selected_scans, device))
    counts = validate_records(records, set(ALL_METHOD_ORDER))

    if args.smoke:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "mode": "smoke_no_writes",
                    "methods": counts,
                    "lamm_n3_baseline_speed_max_abs_difference": lamm_crosscheck,
                    "test_data_loaded": False,
                },
                indent=2,
            )
        )
        return 0

    full_summary = summarize_methods(
        records, FULL_METHOD_ORDER, edges, samples, int(config["bootstrap_seed"])
    )
    inr_scans = {
        str(row["scan_id"]) for row in records if str(row["method"]) == "latent_inr"
    }
    common_records = [row for row in records if str(row["scan_id"]) in inr_scans]
    common_summary = summarize_methods(
        common_records, ALL_METHOD_ORDER, edges, samples, int(config["bootstrap_seed"]) + 500003
    )
    legacy = legacy_records(resolve(config["legacy_velocity_csv"]))
    legacy_methods = tuple(dict.fromkeys(str(row["method"]) for row in legacy))
    legacy_summary = summarize_methods(
        legacy, legacy_methods, edges, samples, int(config["bootstrap_seed"]) + 900001
    )

    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else resolve(config["output_root"])
    )
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing analysis: {output}; pass --force")
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "per_visit_velocity.csv", records)
    write_csv(tables / "full_cohort_age_summary.csv", full_summary)
    write_csv(tables / "strict_common_age_summary.csv", common_summary)
    write_csv(tables / "legacy_age_summary.csv", legacy_summary)
    plot_speed(
        full_summary,
        FULL_METHOD_ORDER,
        figures / "full_cohort_speed_by_age.png",
        "Surface instantaneous velocity by age: direct mesh and latent flows",
    )
    plot_inward(
        full_summary,
        FULL_METHOD_ORDER,
        figures / "full_cohort_inward_normal_by_age.png",
        "Inward-normal instantaneous velocity by age: full validation cohort",
    )
    plot_normal_agreement(
        full_summary,
        FULL_METHOD_ORDER,
        figures / "full_cohort_normal_agreement_by_age.png",
        "Normal-velocity agreement by age: full validation cohort",
    )
    plot_normal_agreement(
        common_summary,
        ALL_METHOD_ORDER,
        figures / "strict_common_normal_agreement_by_age.png",
        "Normal-velocity agreement on strict INR-shared visits\n"
        "100 visits; the oldest AD interval contains one subject",
    )
    plot_speed(
        common_summary,
        ALL_METHOD_ORDER,
        figures / "strict_common_speed_by_age.png",
        "Surface instantaneous velocity on strict INR-shared visits",
    )
    plot_inward(
        common_summary,
        ALL_METHOD_ORDER,
        figures / "strict_common_inward_normal_by_age.png",
        "Inward-normal velocity on strict INR-shared visits",
    )
    plot_legacy(legacy_summary, figures / "legacy_ode_brainode_velocity_by_age.png")

    report = {
        "schema_version": 1,
        "status": "complete",
        "split": "val",
        "test_data_loaded": False,
        "observed_reference": A.REFERENCE_DESCRIPTION,
        "age_bin_edges_years": edges.tolist(),
        "bootstrap_samples": samples,
        "bootstrap_unit": "subject",
        "primary_full_cohort_methods": list(FULL_METHOD_ORDER),
        "strict_common_methods": list(ALL_METHOD_ORDER),
        "method_counts": counts,
        "strict_common_scans": len(inr_scans),
        "lamm_n3_contract": "three latent velocities decoded separately and averaged only in corresponding mesh space",
        "lamm_n3_baseline_speed_max_abs_difference": lamm_crosscheck,
        "inr_contract": "implicit level-set normal velocity; tangential velocity is not identifiable",
        "legacy_contract": "separate historical cohort/protocol; displayed but not pooled or ranked with primary methods",
    }
    B.atomic_json(output / "summary.json", report)
    print(json.dumps({"output": str(output), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
