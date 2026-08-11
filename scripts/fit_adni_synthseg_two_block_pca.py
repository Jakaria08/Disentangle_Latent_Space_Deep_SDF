#!/usr/bin/env python3
"""Fit train-only two-block PCA for strict no-MCI SynthSeg hippocampus + LV meshes.

Input: the validated PCA-Cocycle-v4 manifest bundle.  Output: separate PCA
bases for left hippocampus and left lateral ventricle, coefficient archives,
and reconstruction/variance validation.  This script does not train Cocycle or
Brain-ODE models.

The two structures must remain independent PCA blocks: they originate from
separate rigid/atlas registrations and have very different vertex counts.
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
TASK_ROOT = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
    / "task2_pca_cocycle_v4"
)
DEFAULT_MANIFEST = TASK_ROOT / "metadata" / "adni_synthseg_lhipp_llv_strict_no_mci_manifest.csv"
DEFAULT_CONTRACT = TASK_ROOT / "configs" / "pca_cocycle_v4_input_contract.json"

STRUCTURES: dict[str, dict[str, Any]] = {
    "left_hippocampus": {
        "short_name": "hippocampus",
        "path_column": "hippocampus_mesh_path_mm",
        "vertex_count": 2746,
        "face_count": 5488,
    },
    "left_lateral_ventricle": {
        "short_name": "lateral_ventricle",
        "path_column": "lateral_ventricle_mesh_path_mm",
        "vertex_count": 8346,
        "face_count": 16688,
    },
}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TASK_ROOT / "pca",
        help="New PCA output directory. Refuses to overwrite an existing PCA output.",
    )
    parser.add_argument("--max-components", type=int, default=150)
    parser.add_argument("--selected-components-per-structure", type=int, default=75)
    parser.add_argument("--evaluation-components", default="32,64,75,100,128,150")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the manifest, contract, subject splits, and all mesh paths without writing or fitting PCA.",
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


def parse_dimensions(value: str, maximum: int) -> list[int]:
    parsed = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not parsed or parsed[0] < 1 or parsed[-1] > maximum:
        raise ValueError(f"Evaluation dimensions must be within [1,{maximum}], got {parsed}.")
    return parsed


def load_input(manifest_path: Path, contract_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not manifest_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError("Both the PCA input manifest and its input contract are required.")
    manifest = pd.read_csv(manifest_path, dtype={"scan_id": str, "subject_id": str, "VISCODE": str})
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {
        "scan_id", "subject_id", "split", "diagnosis", "label_ad", "visit_order", "age_years", "age_norm_train",
        *[str(spec["path_column"]) for spec in STRUCTURES.values()],
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise KeyError(f"Manifest is missing required columns: {missing}")
    if manifest["scan_id"].duplicated().any():
        raise ValueError("Manifest has duplicate scans.")
    if set(manifest["split"].unique()) != set(SPLITS):
        raise ValueError("Manifest must contain train, val, and test splits.")
    if manifest.groupby("subject_id")["split"].nunique().gt(1).any():
        raise ValueError("Subject leakage in PCA input manifest.")
    if not manifest["diagnosis"].isin(["CN", "AD"]).all():
        raise ValueError("Only stable CN/AD scans are allowed.")
    if not (manifest["label_ad"].astype(int) == manifest["diagnosis"].map({"CN": 0, "AD": 1})).all():
        raise ValueError("label_ad does not match diagnosis.")
    if not bool(contract.get("longitudinal_policy", {}).get("strict_no_mci", False)):
        raise ValueError("Input contract is not strict no-MCI.")
    return manifest.sort_values(["split", "subject_id", "visit_order", "scan_id"], kind="stable").reset_index(drop=True), contract


def load_structure_matrix(manifest: pd.DataFrame, structure: str) -> tuple[np.ndarray, np.ndarray]:
    spec = STRUCTURES[structure]
    vertices_rows: list[np.ndarray] = []
    reference_faces: np.ndarray | None = None
    count = len(manifest)
    print(f"  Loading {structure} correspondence meshes ({count:,})…", flush=True)
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        path = Path(getattr(row, spec["path_column"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            mesh = trimesh.load(path, force="mesh", process=False)
        except Exception as exc:
            raise RuntimeError(f"Could not load {path}: {exc}") from exc
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.shape != (int(spec["vertex_count"]), 3) or faces.shape != (int(spec["face_count"]), 3):
            raise ValueError(f"Unexpected topology for {structure} {path.name}: {vertices.shape}, {faces.shape}")
        if not np.isfinite(vertices).all():
            raise ValueError(f"Non-finite vertex coordinates in {path}")
        if reference_faces is None:
            reference_faces = faces
        elif not np.array_equal(reference_faces, faces):
            raise ValueError(f"Correspondence face order mismatch for {structure}: {path}")
        vertices_rows.append(vertices.reshape(-1))
        if index == 1 or index % 100 == 0 or index == count:
            print(f"    {index:,}/{count:,}", flush=True)
    if reference_faces is None:
        raise RuntimeError(f"No meshes loaded for {structure}.")
    return np.stack(vertices_rows, axis=0), reference_faces


def validate_manifest_mesh_paths(manifest: pd.DataFrame) -> None:
    for structure, spec in STRUCTURES.items():
        missing = [
            str(path)
            for path in manifest[spec["path_column"]].astype(str)
            if not Path(path).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} {structure} mesh paths; first: {missing[:3]}")


def mesh_volumes(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Absolute signed mesh volume for one or a batch of vertex arrays."""
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim == 2:
        points = points[None, ...]
    v0, v1, v2 = points[:, faces[:, 0]], points[:, faces[:, 1]], points[:, faces[:, 2]]
    signed = np.einsum("bfi,bfi->bf", v0, np.cross(v1, v2)).sum(axis=1) / 6.0
    return np.abs(signed)


def evaluate_reconstruction(
    matrix: np.ndarray,
    labels: pd.DataFrame,
    pca: PCA,
    faces: np.ndarray,
    dimensions: list[int],
    batch_size: int,
    structure: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Evaluate vertex and volume reconstruction without writing reconstructed meshes."""
    scores = pca.transform(matrix).astype(np.float32)
    original = matrix.reshape(len(matrix), -1, 3)
    original_volume = mesh_volumes(original, faces)
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        rmse = np.empty(len(matrix), dtype=np.float64)
        volume_relative_error = np.empty(len(matrix), dtype=np.float64)
        for start in range(0, len(matrix), batch_size):
            end = min(start + batch_size, len(matrix))
            reconstruction = (
                pca.mean_[None, :]
                + scores[start:end, :dimension].astype(np.float64) @ pca.components_[:dimension]
            ).reshape(end - start, -1, 3)
            difference = reconstruction - original[start:end]
            rmse[start:end] = np.sqrt(np.mean(difference * difference, axis=(1, 2)))
            reconstructed_volume = mesh_volumes(reconstruction, faces)
            volume_relative_error[start:end] = 100.0 * np.abs(reconstructed_volume - original_volume[start:end]) / np.maximum(original_volume[start:end], 1.0e-8)
        for split in SPLITS:
            mask = labels["split"].to_numpy() == split
            rows.append(
                {
                    "structure": structure,
                    "components": int(dimension),
                    "split": split,
                    "scans": int(mask.sum()),
                    "vertex_rmse_mm_mean": float(rmse[mask].mean()),
                    "vertex_rmse_mm_median": float(np.median(rmse[mask])),
                    "vertex_rmse_mm_p95": float(np.quantile(rmse[mask], 0.95)),
                    "volume_abs_relative_error_pct_mean": float(volume_relative_error[mask].mean()),
                    "volume_abs_relative_error_pct_median": float(np.median(volume_relative_error[mask])),
                    "volume_abs_relative_error_pct_p95": float(np.quantile(volume_relative_error[mask], 0.95)),
                    "cumulative_explained_variance_train": float(np.cumsum(pca.explained_variance_ratio_)[dimension - 1]),
                }
            )
    return rows, {"scores": scores, "original_volume_mm3": original_volume.astype(np.float32)}


def fit_structure(
    *,
    structure: str,
    matrix: np.ndarray,
    faces: np.ndarray,
    manifest: pd.DataFrame,
    maximum: int,
    dimensions: list[int],
    seed: int,
    batch_size: int,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    train_mask = manifest["split"].to_numpy() == "train"
    train_matrix = matrix[train_mask]
    if maximum > min(train_matrix.shape):
        raise ValueError(f"{maximum} PCA components exceed training matrix dimensions {train_matrix.shape}.")
    print(f"  Fitting randomized PCA for {structure}: {train_matrix.shape[0]:,} train scans × {train_matrix.shape[1]:,} features…", flush=True)
    started = time.time()
    pca = PCA(n_components=maximum, svd_solver="randomized", random_state=seed, iterated_power=5)
    pca.fit(train_matrix)
    elapsed = time.time() - started
    metric_rows, evaluation = evaluate_reconstruction(
        matrix, manifest, pca, faces, dimensions, batch_size, structure
    )
    structure_dir = output_dir / "model" / structure
    structure_dir.mkdir(parents=True, exist_ok=True)
    np.save(structure_dir / "mean.npy", pca.mean_.astype(np.float32))
    np.save(structure_dir / f"components_{maximum}.npy", pca.components_.astype(np.float32))
    np.save(structure_dir / "explained_variance.npy", pca.explained_variance_.astype(np.float64))
    np.save(structure_dir / "explained_variance_ratio.npy", pca.explained_variance_ratio_.astype(np.float64))
    np.save(structure_dir / "singular_values.npy", pca.singular_values_.astype(np.float64))
    np.save(structure_dir / "faces.npy", faces.astype(np.int64))
    np.save(structure_dir / "train_scan_ids.npy", manifest.loc[train_mask, "scan_id"].astype(str).to_numpy())
    summary = {
        "structure": structure,
        "fit_split": "train",
        "fit_scans": int(train_mask.sum()),
        "features": int(train_matrix.shape[1]),
        "vertices": int(train_matrix.shape[1] // 3),
        "faces": int(faces.shape[0]),
        "max_components": int(maximum),
        "fitting_seconds": float(elapsed),
        "cumulative_explained_variance": {
            str(dimension): float(np.cumsum(pca.explained_variance_ratio_)[dimension - 1])
            for dimension in dimensions
        },
    }
    write_json(structure_dir / "pca_model_summary.json", summary)
    print(f"    complete in {elapsed / 60.0:.1f} min; variance at {maximum} PCs = {np.cumsum(pca.explained_variance_ratio_)[maximum - 1]:.4f}", flush=True)
    return summary, evaluation["scores"], metric_rows


def write_coefficient_archives(
    output_dir: Path,
    manifest: pd.DataFrame,
    scores_by_structure: dict[str, np.ndarray],
    selected_components: int,
) -> dict[str, Any]:
    coeff_dir = output_dir / "coefficients"
    coeff_dir.mkdir(parents=True, exist_ok=True)
    ordered_scores = [scores_by_structure[structure] for structure in STRUCTURES]
    selected = [scores[:, :selected_components] for scores in ordered_scores]
    train_mask = manifest["split"].to_numpy() == "train"
    train_mean = [values[train_mask].mean(axis=0).astype(np.float32) for values in selected]
    train_std = [np.maximum(values[train_mask].std(axis=0), 1.0e-6).astype(np.float32) for values in selected]
    combined_raw = np.concatenate(selected, axis=1).astype(np.float32)
    combined_standardized = np.concatenate(
        [(values - mean[None, :]) / std[None, :] for values, mean, std in zip(selected, train_mean, train_std)], axis=1
    ).astype(np.float32)
    metadata = {
        "scan_ids": manifest["scan_id"].astype(str).to_numpy(),
        "subject_ids": manifest["subject_id"].astype(str).to_numpy(),
        "splits": manifest["split"].astype(str).to_numpy(),
        "diagnoses": manifest["diagnosis"].astype(str).to_numpy(),
        "label_ad": manifest["label_ad"].astype(np.int8).to_numpy(),
        "visit_orders": manifest["visit_order"].astype(np.int16).to_numpy(),
        "age_years": manifest["age_years"].astype(np.float32).to_numpy(),
        "age_norm_train": manifest["age_norm_train"].astype(np.float32).to_numpy(),
    }
    max_components = int(next(iter(scores_by_structure.values())).shape[1])
    combined_raw_key = f"combined_raw_pca_{2 * selected_components}_{selected_components}_{selected_components}"
    combined_standardized_key = f"combined_standardized_pca_{2 * selected_components}_{selected_components}_{selected_components}"
    arrays: dict[str, np.ndarray] = {
        **metadata,
        f"left_hippocampus_pca_{max_components}": scores_by_structure["left_hippocampus"].astype(np.float32),
        f"left_lateral_ventricle_pca_{max_components}": scores_by_structure["left_lateral_ventricle"].astype(np.float32),
        combined_raw_key: combined_raw,
        combined_standardized_key: combined_standardized,
        f"left_hippocampus_train_mean_{selected_components}": train_mean[0],
        f"left_hippocampus_train_std_{selected_components}": train_std[0],
        f"left_lateral_ventricle_train_mean_{selected_components}": train_mean[1],
        f"left_lateral_ventricle_train_std_{selected_components}": train_std[1],
    }
    np.savez_compressed(coeff_dir / "all_coefficients.npz", **arrays)
    for split in SPLITS:
        mask = manifest["split"].to_numpy() == split
        np.savez_compressed(
            coeff_dir / f"{split}_coefficients.npz",
            **{
                name: values[mask] if values.ndim >= 1 and values.shape[0] == len(manifest) else values
                for name, values in arrays.items()
            },
        )
    return {
        "selected_components_per_structure": int(selected_components),
        "combined_model_dimension": int(combined_standardized.shape[1]),
        "combined_raw_key": combined_raw_key,
        "combined_standardized_key": combined_standardized_key,
        "coefficient_archives": [str(coeff_dir / "all_coefficients.npz"), *[str(coeff_dir / f"{split}_coefficients.npz") for split in SPLITS]],
    }


def validate_output(
    output_dir: Path,
    manifest: pd.DataFrame,
    maximum: int,
    selected: int,
) -> dict[str, Any]:
    archive_path = output_dir / "coefficients" / "all_coefficients.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        scan_ids = archive["scan_ids"].astype(str)
        combined_key = f"combined_standardized_pca_{2 * selected}_{selected}_{selected}"
        combined = archive[combined_key]
        if not np.array_equal(scan_ids, manifest["scan_id"].astype(str).to_numpy()):
            raise RuntimeError("Output coefficient scan order does not match the input manifest.")
        if combined.shape != (len(manifest), 2 * selected) or not np.isfinite(combined).all():
            raise RuntimeError("Output standardized coefficient matrix has invalid shape or non-finite values.")
    model_checks: dict[str, Any] = {}
    for structure, spec in STRUCTURES.items():
        model_dir = output_dir / "model" / structure
        mean = np.load(model_dir / "mean.npy")
        components = np.load(model_dir / f"components_{maximum}.npy")
        faces = np.load(model_dir / "faces.npy")
        expected_features = int(spec["vertex_count"]) * 3
        if mean.shape != (expected_features,) or components.shape != (maximum, expected_features) or faces.shape != (int(spec["face_count"]), 3):
            raise RuntimeError(f"Invalid saved PCA model shape for {structure}.")
        model_checks[structure] = {
            "mean_shape": list(mean.shape),
            "components_shape": list(components.shape),
            "faces_shape": list(faces.shape),
        }
    return {
        "passed": True,
        "coefficient_archive": str(archive_path),
        "scans": int(len(manifest)),
        "subjects": int(manifest["subject_id"].nunique()),
        "selected_combined_dimension": int(2 * selected),
        "model_checks": model_checks,
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    contract_path = args.input_contract.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    maximum = int(args.max_components)
    selected = int(args.selected_components_per_structure)
    dimensions = parse_dimensions(args.evaluation_components, maximum)
    if selected < 1 or selected > maximum:
        raise ValueError("selected-components-per-structure must lie within the fitted component range.")
    print("=" * 88)
    print("Train-only two-block physical-mm PCA (no Cocycle or Brain-ODE training)")
    print(f"Manifest: {manifest_path}")
    print(f"Output:   {output_dir}")
    print(f"PCA: {maximum} maximum PCs / {selected} selected PCs per structure")
    print("=" * 88, flush=True)
    started = time.time()
    manifest, contract = load_input(manifest_path, contract_path)
    if int(contract.get("representation_policy", {}).get("candidate_max_components_per_structure", maximum)) < maximum:
        raise ValueError("Requested PCA rank exceeds the rank authorised by the input contract.")
    validate_manifest_mesh_paths(manifest)
    print(f"Input validated: {len(manifest):,} scans / {manifest['subject_id'].nunique():,} subjects; "
          f"train={int((manifest['split'] == 'train').sum()):,} scans.", flush=True)
    if args.validate_only:
        print("Validation-only complete. No PCA output was written and no model was trained.")
        return 0
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"PCA output directory already contains files: {output_dir}. Choose a new --output-dir; existing output is protected."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_by_structure: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    for structure in STRUCTURES:
        matrix, faces = load_structure_matrix(manifest, structure)
        summary, scores, rows = fit_structure(
            structure=structure,
            matrix=matrix,
            faces=faces,
            manifest=manifest,
            maximum=maximum,
            dimensions=dimensions,
            seed=int(args.seed),
            batch_size=max(1, int(args.batch_size)),
            output_dir=output_dir,
        )
        summaries[structure] = summary
        scores_by_structure[structure] = scores
        metric_rows.extend(rows)
        del matrix

    coefficient_summary = write_coefficient_archives(output_dir, manifest, scores_by_structure, selected)
    metric_fields = list(metric_rows[0])
    write_csv(output_dir / "metrics" / "pca_reconstruction_summary.csv", metric_rows, metric_fields)
    validation = validate_output(output_dir, manifest, maximum, selected)
    validation.update(
        {
            "source_meshes_modified": False,
            "pca_fit_split": "train",
            "cocycle_trained": False,
            "brainode_trained": False,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "input_contract": str(contract_path),
            "input_contract_sha256": sha256_file(contract_path),
            "structures": summaries,
            "coefficients": coefficient_summary,
            "evaluation_components": dimensions,
            "elapsed_minutes": float((time.time() - started) / 60.0),
        }
    )
    write_json(output_dir / "metadata" / "pca_fit_validation.json", validation)
    print("=" * 88)
    print("PCA complete and validated. Source meshes were not modified.")
    print(f"Combined standardized feature dimension: {validation['selected_combined_dimension']}")
    print(f"Validation: {output_dir / 'metadata' / 'pca_fit_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
