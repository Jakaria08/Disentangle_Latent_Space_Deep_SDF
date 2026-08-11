#!/usr/bin/env python3
"""Fit one independent, train-only PCA model for the current SynthSeg cohort.

Use exactly one of ``--structure hippocampus`` or ``--structure
lateral_ventricle``.  The script reads only that structure's QC-approved
physical-mm correspondence meshes and writes a PCA model, coefficient archives,
and held-out reconstruction checks under that same structure's experiment
directory.  It never combines structures and never trains Cocycle or Brain-ODE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trimesh
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = REPO_ROOT / "examples" / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "hippocampus": {
        "structure": "left_hippocampus",
        "experiment_root": BASE_ROOT / "hippocampus_pca_cocycle_v4",
        "manifest_name": "hippocampus_qc_keep_manifest.csv",
        "vertices": 2746,
        "faces": 5488,
    },
    "lateral_ventricle": {
        "structure": "left_lateral_ventricle",
        "experiment_root": BASE_ROOT / "lateral_ventricle_pca_cocycle_v4",
        "manifest_name": "lateral_ventricle_qc_keep_manifest.csv",
        "vertices": 8346,
        "faces": 16688,
    },
}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--input-contract", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-components", type=int, default=150)
    parser.add_argument("--evaluation-components", default="32,64,100,128,150")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and all mesh paths without writing/fitting PCA.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dimensions(value: str, maximum: int) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or result[0] < 1 or result[-1] > maximum:
        raise ValueError(f"Evaluation dimensions must lie in [1,{maximum}].")
    return result


def input_paths(args: argparse.Namespace, spec: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(spec["experiment_root"])
    manifest = args.manifest.expanduser().resolve() if args.manifest else root / "metadata" / spec["manifest_name"]
    contract = args.input_contract.expanduser().resolve() if args.input_contract else root / "configs" / "pca_input_contract.json"
    output = args.output_dir.expanduser().resolve() if args.output_dir else root / "pca"
    return manifest, contract, output


def load_and_validate(manifest_path: Path, contract_path: Path, spec: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not manifest_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError("Structure manifest and pca_input_contract.json are both required.")
    frame = pd.read_csv(manifest_path, dtype={"scan_id": str, "subject_id": str, "VISCODE": str})
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {
        "scan_id", "subject_id", "split", "diagnosis", "label_ad", "visit_order", "visit_month", "age_years", "age_norm_train",
        "mesh_path_mm", "vertex_count", "face_count", "structure",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Manifest lacks required columns: {missing}")
    if frame["scan_id"].duplicated().any() or frame.groupby("subject_id")["split"].nunique().gt(1).any():
        raise ValueError("Manifest has duplicate scans or subject split leakage.")
    if set(frame["split"].unique()) != set(SPLITS):
        raise ValueError("Manifest must contain train, val, and test scans.")
    if not frame["diagnosis"].isin(("CN", "AD")).all():
        raise ValueError("PCA input must contain only CN/AD scans.")
    if not (frame["label_ad"].astype(int) == frame["diagnosis"].map({"CN": 0, "AD": 1})).all():
        raise ValueError("label_ad does not match diagnosis.")
    if not frame["structure"].eq(spec["structure"]).all() or contract.get("structure") != spec["structure"]:
        raise ValueError("Manifest/contract structure does not match the requested PCA experiment.")
    if set(pd.to_numeric(frame["vertex_count"], errors="coerce").dropna().astype(int)) != {int(spec["vertices"])}:
        raise ValueError("Unexpected vertex count in manifest.")
    if set(pd.to_numeric(frame["face_count"], errors="coerce").dropna().astype(int)) != {int(spec["faces"])}:
        raise ValueError("Unexpected face count in manifest.")
    if (frame.groupby("subject_id")["scan_id"].nunique() < 2).any():
        raise ValueError("Every retained subject must have at least two scans.")
    if frame.duplicated(["subject_id", "visit_month"], keep=False).any():
        raise ValueError("Duplicate longitudinal visit time in manifest.")
    for _, group in frame.sort_values(["subject_id", "visit_month", "visit_order", "scan_id"]).groupby("subject_id", sort=False):
        if (group["visit_month"].diff().dropna() <= 0).any():
            raise ValueError("Non-increasing longitudinal visit time in manifest.")
    missing_paths = [path for path in frame["mesh_path_mm"].astype(str) if not Path(path).is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing {len(missing_paths)} mesh paths; first: {missing_paths[:3]}")
    return frame.sort_values(["split", "subject_id", "visit_order", "scan_id"], kind="stable").reset_index(drop=True), contract


def load_mesh_matrix(frame: pd.DataFrame, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    reference_faces: np.ndarray | None = None
    total = len(frame)
    for index, path_text in enumerate(frame["mesh_path_mm"].astype(str), start=1):
        path = Path(path_text)
        try:
            mesh = trimesh.load(path, force="mesh", process=False)
        except Exception as exc:
            raise RuntimeError(f"Could not load {path}: {exc}") from exc
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.shape != (int(spec["vertices"]), 3) or faces.shape != (int(spec["faces"]), 3):
            raise ValueError(f"Topology mismatch in {path}: {vertices.shape}, {faces.shape}")
        if not np.isfinite(vertices).all():
            raise ValueError(f"Non-finite vertices in {path}")
        if reference_faces is None:
            reference_faces = faces
        elif not np.array_equal(reference_faces, faces):
            raise ValueError(f"Face connectivity mismatch in {path}")
        vectors.append(vertices.reshape(-1))
        if index == 1 or index % 100 == 0 or index == total:
            print(f"  mesh load: {index:,}/{total:,}", flush=True)
    if reference_faces is None:
        raise RuntimeError("No meshes loaded.")
    return np.stack(vectors, axis=0), reference_faces


def volumes(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim == 2:
        points = points[None, ...]
    v0, v1, v2 = points[:, faces[:, 0]], points[:, faces[:, 1]], points[:, faces[:, 2]]
    return np.abs(np.einsum("bfi,bfi->bf", v0, np.cross(v1, v2)).sum(axis=1) / 6.0)


def evaluate(
    matrix: np.ndarray, frame: pd.DataFrame, model: PCA, faces: np.ndarray, component_list: list[int], batch_size: int, structure: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    scores = model.transform(matrix).astype(np.float32)
    observed = matrix.reshape(len(matrix), -1, 3)
    observed_volume = volumes(observed, faces)
    rows: list[dict[str, Any]] = []
    cumulative = np.cumsum(model.explained_variance_ratio_)
    for component_count in component_list:
        rmse = np.empty(len(matrix), dtype=np.float64)
        relative_volume_error = np.empty(len(matrix), dtype=np.float64)
        for start in range(0, len(matrix), batch_size):
            stop = min(start + batch_size, len(matrix))
            reconstructed = (
                model.mean_[None, :]
                + scores[start:stop, :component_count].astype(np.float64) @ model.components_[:component_count]
            ).reshape(stop - start, -1, 3)
            rmse[start:stop] = np.sqrt(np.mean((reconstructed - observed[start:stop]) ** 2, axis=(1, 2)))
            reconstructed_volume = volumes(reconstructed, faces)
            relative_volume_error[start:stop] = 100.0 * np.abs(reconstructed_volume - observed_volume[start:stop]) / np.maximum(observed_volume[start:stop], 1.0e-8)
        for split in SPLITS:
            mask = frame["split"].to_numpy() == split
            rows.append(
                {
                    "structure": structure,
                    "components": int(component_count),
                    "split": split,
                    "scans": int(mask.sum()),
                    "cumulative_explained_variance_train": float(cumulative[component_count - 1]),
                    "vertex_rmse_mm_mean": float(rmse[mask].mean()),
                    "vertex_rmse_mm_median": float(np.median(rmse[mask])),
                    "vertex_rmse_mm_p95": float(np.quantile(rmse[mask], 0.95)),
                    "volume_abs_relative_error_pct_mean": float(relative_volume_error[mask].mean()),
                    "volume_abs_relative_error_pct_median": float(np.median(relative_volume_error[mask])),
                    "volume_abs_relative_error_pct_p95": float(np.quantile(relative_volume_error[mask], 0.95)),
                }
            )
    return scores, rows


def write_coefficients(output: Path, frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    coefficient_dir = output / "coefficients"
    coefficient_dir.mkdir(parents=True, exist_ok=True)
    train_mask = frame["split"].to_numpy() == "train"
    mean = scores[train_mask].mean(axis=0).astype(np.float32)
    std = np.maximum(scores[train_mask].std(axis=0), 1.0e-6).astype(np.float32)
    standardized = ((scores - mean[None, :]) / std[None, :]).astype(np.float32)
    arrays: dict[str, np.ndarray] = {
        # np.asarray(..., dtype=np.str_) prevents pickle-dependent object
        # arrays in the .npz metadata and enables allow_pickle=False loaders.
        "scan_ids": np.asarray(frame["scan_id"].astype(str).tolist(), dtype=np.str_),
        "subject_ids": np.asarray(frame["subject_id"].astype(str).tolist(), dtype=np.str_),
        "splits": np.asarray(frame["split"].astype(str).tolist(), dtype=np.str_),
        "diagnoses": np.asarray(frame["diagnosis"].astype(str).tolist(), dtype=np.str_),
        "label_ad": frame["label_ad"].astype(np.int8).to_numpy(),
        "visit_orders": frame["visit_order"].astype(np.int16).to_numpy(),
        "visit_months": frame["visit_month"].astype(np.float32).to_numpy(),
        "age_years": frame["age_years"].astype(np.float32).to_numpy(),
        "age_norm_train": frame["age_norm_train"].astype(np.float32).to_numpy(),
        "pca_150": scores,
        "pca_standardized_150": standardized,
        "train_pca_mean_150": mean,
        "train_pca_std_150": std,
    }
    np.savez_compressed(coefficient_dir / "all_coefficients.npz", **arrays)
    for split in SPLITS:
        mask = frame["split"].to_numpy() == split
        np.savez_compressed(
            coefficient_dir / f"{split}_coefficients.npz",
            **{key: value[mask] if value.ndim >= 1 and value.shape[0] == len(frame) else value for key, value in arrays.items()},
        )
    return {
        "all": str(coefficient_dir / "all_coefficients.npz"),
        "splits": {split: str(coefficient_dir / f"{split}_coefficients.npz") for split in SPLITS},
        "standardization": "per-PC mean/std from train scans only",
    }


def main() -> int:
    args = parse_args()
    spec = EXPERIMENTS[args.structure]
    manifest_path, contract_path, output_dir = input_paths(args, spec)
    maximum = int(args.max_components)
    component_list = dimensions(args.evaluation_components, maximum)
    if maximum < 1:
        raise ValueError("max-components must be positive.")

    print("=" * 88)
    print(f"Train-only PCA | {spec['structure']} | no Cocycle/Brain-ODE training")
    print(f"Manifest: {manifest_path}")
    print(f"Output:   {output_dir}")
    print(f"PCA rank: {maximum}")
    print("=" * 88, flush=True)
    frame, contract = load_and_validate(manifest_path, contract_path, spec)
    if maximum > int(contract.get("max_components", 0)):
        raise ValueError("Requested PCA rank exceeds the approved input contract.")
    if args.validate_only:
        print(f"Validation-only passed: {len(frame):,} scans / {frame['subject_id'].nunique():,} subjects. No PCA was fitted.")
        return 0
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite populated PCA output directory: {output_dir}")

    started = time.time()
    matrix, faces = load_mesh_matrix(frame, spec)
    train_mask = frame["split"].to_numpy() == "train"
    train_matrix = matrix[train_mask]
    if maximum > min(train_matrix.shape):
        raise ValueError(f"PCA rank {maximum} exceeds training matrix shape {train_matrix.shape}.")
    print(f"Fitting randomized PCA on {train_matrix.shape[0]:,} train scans × {train_matrix.shape[1]:,} features…", flush=True)
    model = PCA(n_components=maximum, svd_solver="randomized", random_state=int(args.seed), iterated_power=5)
    model.fit(train_matrix)
    scores, metric_rows = evaluate(matrix, frame, model, faces, component_list, max(1, int(args.batch_size)), spec["structure"])

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / "mean.npy", model.mean_.astype(np.float32))
    np.save(model_dir / "components_150.npy", model.components_.astype(np.float32))
    np.save(model_dir / "explained_variance.npy", model.explained_variance_.astype(np.float64))
    np.save(model_dir / "explained_variance_ratio.npy", model.explained_variance_ratio_.astype(np.float64))
    np.save(model_dir / "singular_values.npy", model.singular_values_.astype(np.float64))
    np.save(model_dir / "faces.npy", faces.astype(np.int64))
    np.save(model_dir / "train_scan_ids.npy", frame.loc[train_mask, "scan_id"].astype(str).to_numpy())
    write_json(model_dir / "pca_model_summary.json", {
        "structure": spec["structure"], "fit_split": "train", "fit_scans": int(train_mask.sum()),
        "features": int(train_matrix.shape[1]), "max_components": maximum,
        "cumulative_explained_variance": {str(k): float(np.cumsum(model.explained_variance_ratio_)[k - 1]) for k in component_list},
    })
    coefficient_summary = write_coefficients(output_dir, frame, scores)
    write_csv(output_dir / "metrics" / "pca_reconstruction_summary.csv", metric_rows, list(metric_rows[0]))

    with np.load(output_dir / "coefficients" / "all_coefficients.npz", allow_pickle=False) as archive:
        saved_scores = archive["pca_150"]
        saved_standardized = archive["pca_standardized_150"]
        if saved_scores.shape != (len(frame), maximum) or saved_standardized.shape != (len(frame), maximum):
            raise RuntimeError("Saved coefficient archive has an invalid shape.")
        if not np.isfinite(saved_scores).all() or not np.isfinite(saved_standardized).all():
            raise RuntimeError("Saved coefficient archive contains non-finite values.")
    validation = {
        "passed": True,
        "source_meshes_modified": False,
        "pca_fitted": True,
        "cocycle_trained": False,
        "brainode_trained": False,
        "structure": spec["structure"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "input_contract": str(contract_path),
        "input_contract_sha256": sha256_file(contract_path),
        "scans": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "fit_scans": int(train_mask.sum()),
        "max_components": maximum,
        "evaluation_components": component_list,
        "coefficient_outputs": coefficient_summary,
        "elapsed_minutes": float((time.time() - started) / 60.0),
    }
    write_json(output_dir / "metadata" / "pca_fit_validation.json", validation)
    print("PCA complete and validated. Source meshes were not modified.")
    print(f"Validation report: {output_dir / 'metadata' / 'pca_fit_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
