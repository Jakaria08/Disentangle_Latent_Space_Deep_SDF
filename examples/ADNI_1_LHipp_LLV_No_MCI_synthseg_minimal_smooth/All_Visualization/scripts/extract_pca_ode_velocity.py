#!/usr/bin/env python3
"""Map completed PCA plain-ODE and PCA BrainODE vector fields to mesh space.

The saved models operate on the same fixed PCA representation, visit ordering,
and normalized ages as the current latent PCA cocycle.  This script evaluates
the learned ODE field at each validation scan, pushes it through the frozen PCA
decoder Jacobian, and compares the resulting mm/year field with the same fitted
longitudinal surface reference used by the other methods.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
DEFAULT_REGISTRY = EXPERIMENT_DIR / "configs" / "model_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Evaluate eight balanced scans without writing files.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    output = np.zeros_like(vertices, dtype=np.float64)
    triangles = vertices[:, faces]
    face_normals = np.cross(
        triangles[:, :, 1] - triangles[:, :, 0],
        triangles[:, :, 2] - triangles[:, :, 0],
    )
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
    predicted: np.ndarray,
    observed: np.ndarray,
    normals: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    predicted_normal = np.sum(predicted * normals, axis=1)
    observed_normal = np.sum(observed * normals, axis=1)
    predicted_flat = predicted.reshape(-1)
    observed_flat = observed.reshape(-1)
    denominator = float(np.linalg.norm(predicted_flat) * np.linalg.norm(observed_flat))
    return {
        "method": method,
        "method_label": label,
        "model_family": "latent_pca_ode",
        "velocity_definition": "ODE diagonal vector field pushed through the frozen PCA decoder Jacobian",
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


def age_slope_per_year(archive: dict[str, np.ndarray]) -> float:
    age = archive["visit_age_years"].astype(np.float64)
    normalized = archive["visit_age_norm_train"].astype(np.float64)
    slope = float(np.polyfit(age, normalized, 1)[0])
    if not math.isfinite(slope) or slope <= 0.0:
        raise ValueError("Invalid normalized-age scale")
    return slope


def decoded_velocity(
    function: torch.nn.Module,
    geometry: torch.nn.Module,
    latent: torch.Tensor,
    age: torch.Tensor,
    diagnosis: torch.Tensor,
    age_slope: float,
    batch_size: int,
) -> np.ndarray:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(latent), int(batch_size)):
        z = latent[start : start + batch_size]
        t = age[start : start + batch_size]
        condition = diagnosis[start : start + batch_size]
        with torch.no_grad():
            latent_velocity = function(t, z, condition) * float(age_slope)
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


def load_august_modules(task_dir: Path):
    script_dir = task_dir / "scripts"
    for name in ("common", "models", "evaluate", "c4_objective"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(script_dir))
    common = importlib.import_module("common")
    models = importlib.import_module("models")
    return script_dir, common, models


def unload_august_modules(script_dir: Path) -> None:
    for name in ("common", "models", "evaluate", "c4_objective"):
        sys.modules.pop(name, None)
    sys.path[:] = [entry for entry in sys.path if entry != str(script_dir)]


def main() -> int:
    args = parse_args()
    config = read_json(args.registry)
    specification = config["pca_ode_baselines"]
    if specification.get("velocity_split") != "val":
        raise ValueError("PCA ODE velocity comparison must remain validation-only")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)

    reference_path = resolve(specification["velocity_reference"])
    with np.load(reference_path, allow_pickle=False) as loaded:
        reference = {key: loaded[key].copy() for key in loaded.files}
    statistics = torch.load(reference_path.parent / "training_statistics.pt", map_location="cpu", weights_only=False)
    faces = np.asarray(statistics["faces"], dtype=np.int64)
    normals = vertex_normals(reference["vertices_mm"].astype(np.float64), faces)
    reference_lookup = {str(scan): index for index, scan in enumerate(reference["scan_ids"])}

    task_dir = resolve(specification["task_dir"])
    script_dir, common, models = load_august_modules(task_dir)
    records: list[dict[str, Any]] = []
    try:
        archive = common.load_archive("pca128", "val")
        train = common.load_archive("pca128", "train")
        scans = archive["visit_scan_ids"].astype(str)
        chosen = [
            index
            for index, scan in enumerate(scans)
            if scan in reference_lookup and float(reference["velocity_reference_weight"][reference_lookup[scan]]) > 0.0
        ]
        if args.smoke:
            balanced: list[int] = []
            for diagnosis in ("CN", "AD"):
                candidates = [index for index in chosen if str(archive["visit_diagnoses"][index]) == diagnosis]
                balanced.extend(candidates[:4])
            chosen = sorted(balanced)
        selected = np.asarray(chosen, dtype=np.int64)
        geometry = common.build_geometry("pca128", train, device)
        latent = torch.from_numpy(archive["visit_latent_standardized_128"][selected].astype(np.float32)).to(device)
        age = torch.from_numpy(archive["visit_age_norm_train"][selected].astype(np.float32)).to(device)
        diagnosis = torch.from_numpy(archive["visit_label_ad"][selected].astype(np.float32)).to(device)
        slope = age_slope_per_year(archive)

        for method, method_spec in specification["methods"].items():
            run = resolve(method_spec["run_dir"])
            if "smoke" in run.name.lower():
                raise ValueError(f"Smoke checkpoint rejected: {run}")
            resolved = read_json(run / "resolved_config.json")
            run_config = resolved["config"]
            if run_config.get("representation") != "pca128" or run_config.get("method") != method_spec["method"]:
                raise ValueError(f"PCA ODE run contract mismatch: {run}")
            checkpoint = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
            if bool(checkpoint.get("test_data_loaded", False)):
                raise ValueError(f"Training/test leakage recorded in {run}")
            function = models.build_ode(run_config).to(device)
            function.load_state_dict(checkpoint["model_state_dict"], strict=True)
            function.eval()
            velocity = decoded_velocity(function, geometry, latent, age, diagnosis, slope, args.batch_size)
            for local, archive_index in enumerate(selected):
                scan = str(scans[archive_index])
                ref = reference_lookup[scan]
                records.append(
                    vector_record(
                        method,
                        str(method_spec["label"]),
                        velocity[local],
                        reference["velocity_reference_mm_per_year"][ref],
                        normals[ref],
                        {
                            "scan_id": scan,
                            "subject_id": str(archive["visit_subject_ids"][archive_index]),
                            "diagnosis": str(archive["visit_diagnoses"][archive_index]),
                            "age_years": float(archive["visit_age_years"][archive_index]),
                            "reference_reliability": float(reference["velocity_reference_weight"][ref]),
                        },
                    )
                )
            del function, velocity
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        unload_august_modules(script_dir)

    frame = pd.DataFrame(records).sort_values(["method", "subject_id", "age_years"]).reset_index(drop=True)
    counts = frame.groupby("method").agg(scans=("scan_id", "size"), subjects=("subject_id", "nunique")).to_dict("index")
    result = {
        "schema_version": 1,
        "status": "smoke_passed" if args.smoke else "complete",
        "split": "val",
        "test_data_loaded": False,
        "methods": counts,
        "rows": len(frame),
        "reference": str(reference_path),
        "definition": "learned ODE vector field evaluated at each scan and pushed through the frozen PCA decoder Jacobian",
    }
    if args.smoke:
        print(json.dumps(result, indent=2))
        return 0

    output = resolve(args.output_dir if args.output_dir is not None else specification["velocity_output"])
    output.mkdir(parents=True, exist_ok=True)
    table = output / "per_scan.csv"
    manifest = output / "summary.json"
    if not args.force and (table.exists() or manifest.exists()):
        raise FileExistsError(f"Refusing to overwrite PCA ODE velocity output: {output}; pass --force")
    temporary = table.with_name(f".{table.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, table)
    temporary_manifest = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    temporary_manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest)
    print(json.dumps({"output": str(output), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
