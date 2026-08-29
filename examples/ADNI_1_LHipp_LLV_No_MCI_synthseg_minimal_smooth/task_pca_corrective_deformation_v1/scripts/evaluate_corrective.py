#!/usr/bin/env python3
"""Evaluate a corrective checkpoint against its exact matched PCA baseline."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

import common
from model import build_model
from surface_metrics import metrics as surface_metrics
import trimesh


DISTANCE_KEYS = (
    "coordinate_rmse_mm",
    "mean_vertex_euclidean_mm",
    "assd_mm",
    "hd95_mm",
    "chamfer_l1_mm",
    "chamfer_l2_squared_mm2",
    "volume_relative_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(common.DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["val"])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--surface-points", type=int, default=None)
    parser.add_argument("--max-shapes", type=int, default=None)
    parser.add_argument("--write-meshes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_seed(text: str, base: int) -> int:
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return int((value + int(base)) % (2**31 - 1))


def cluster_bootstrap_mean_ci(
    differences: np.ndarray,
    subject_ids: np.ndarray,
    seed: int,
    repeats: int,
) -> list[float]:
    subjects = np.unique(subject_ids)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(repeats), dtype=np.float64)
    positions = {subject: np.flatnonzero(subject_ids == subject) for subject in subjects}
    for index in range(int(repeats)):
        sampled = subjects[rng.integers(0, len(subjects), size=len(subjects))]
        chosen = np.concatenate([positions[subject] for subject in sampled])
        estimates[index] = float(differences[chosen].mean())
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


@torch.no_grad()
def predict(model, coefficients: torch.Tensor, batch_size: int = 32):
    corrected, pca = [], []
    model.eval()
    for start in range(0, len(coefficients), int(batch_size)):
        details = model.decode_with_details(coefficients[start : start + int(batch_size)])
        corrected.append(details["prediction"].detach().cpu())
        pca.append(details["pca_mesh"].detach().cpu())
    return torch.cat(corrected).numpy(), torch.cat(pca).numpy()


def coordinate_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "coordinate_rmse_mm": float(np.sqrt(np.square(difference).mean())),
        "mean_vertex_euclidean_mm": float(np.linalg.norm(difference, axis=-1).mean()),
    }


def export_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    common.makedirs(path.parent)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)


def aggregate(rows: list[dict], split: str, repeats: int, seed: int) -> dict:
    selected = [row for row in rows if row["split"] == split]
    subject_ids = np.asarray([row["subject_id"] for row in selected])
    summary = {"scans": len(selected), "subjects": int(len(np.unique(subject_ids)))}
    for key in DISTANCE_KEYS:
        pca_values = np.asarray([float(row[f"pca_{key}"]) for row in selected])
        corrected_values = np.asarray([float(row[f"corrected_{key}"]) for row in selected])
        difference = corrected_values - pca_values
        summary[key] = {
            "pca_mean": float(pca_values.mean()),
            "corrected_mean": float(corrected_values.mean()),
            "corrected_minus_pca_mean": float(difference.mean()),
            "corrected_minus_pca_median": float(np.median(difference)),
            "relative_improvement_pct": float(
                100.0 * (pca_values.mean() - corrected_values.mean()) / max(pca_values.mean(), 1e-12)
            ),
            "subject_cluster_bootstrap_difference_mean_95ci": cluster_bootstrap_mean_ci(
                difference,
                subject_ids,
                stable_seed(split + key, seed),
                repeats,
            ),
        }
    for key in (
        "normal_absolute_cosine",
        "normal_signed_cosine",
        "flipped_face_fraction_vs_ground_truth",
        "surface_area_ratio",
        "curvature_ratio_to_ground_truth",
    ):
        summary[key] = {
            "pca_mean": float(np.mean([float(row[f"pca_{key}"]) for row in selected])),
            "corrected_mean": float(
                np.mean([float(row[f"corrected_{key}"]) for row in selected])
            ),
        }
    summary["corrected_all_watertight"] = bool(
        all(str(row["corrected_predicted_watertight"]).lower() == "true" for row in selected)
    )
    summary["corrected_all_winding_consistent"] = bool(
        all(
            str(row["corrected_predicted_winding_consistent"]).lower() == "true"
            for row in selected
        )
    )
    return summary


def main() -> None:
    args = parse_args()
    if "test" in args.splits and not args.allow_test:
        raise PermissionError("Test evaluation requires the explicit --allow-test flag")
    config = common.load_config(args.config)
    device = common.choose_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    contract = common.load_pca_contract(config)
    common.checkpoint_contract_matches(checkpoint, contract)
    model = build_model(config, contract, device)
    model.load_state_dict(checkpoint["model_state"])

    output_dir = (
        common.require_bulk_path(args.output_dir, "evaluation directory")
        if args.output_dir
        else common.require_bulk_path(
            checkpoint_path.parent / "evaluation" / checkpoint_path.stem,
            "evaluation directory",
        )
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Evaluation directory is not empty: {output_dir}. Use --overwrite or a new path."
        )
    common.makedirs(output_dir)
    surface_points = int(
        args.surface_points or config["evaluation"]["surface_points"]
    )
    repeats = int(config["evaluation"]["bootstrap_repeats"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    rows: list[dict] = []

    for split in args.splits:
        data = common.load_split_tensors(split, contract, device, limit=args.max_shapes)
        corrected, pca = predict(model, data.coefficients)
        target = data.vertices_mm.detach().cpu().numpy()
        reencoded = model.pca_encode(torch.from_numpy(corrected).to(device))
        latent_error = float(torch.max(torch.abs(reencoded - data.coefficients)).detach().cpu())
        if latent_error > 2e-4:
            raise RuntimeError(f"Corrected meshes violate the PCA latent contract: {latent_error:.3e}")

        for index, manifest_row in enumerate(data.rows):
            scan_id = manifest_row["scan_id"]
            row = {
                "scan_id": scan_id,
                "subject_id": manifest_row.get("subject_id", scan_id),
                "split": split,
            }
            seed = stable_seed(scan_id, bootstrap_seed)
            for name, prediction in (("pca", pca[index]), ("corrected", corrected[index])):
                values = coordinate_metrics(prediction, target[index])
                values.update(
                    surface_metrics(
                        target[index],
                        prediction,
                        contract.faces,
                        surface_points=surface_points,
                        # Common random numbers: PCA and corrective surfaces must use the
                        # same deterministic samples, otherwise a sub-0.001 mm gain can be
                        # overwhelmed by Monte Carlo sampling noise.
                        seed=seed,
                    )
                )
                for key, value in values.items():
                    row[f"{name}_{key}"] = value
                if args.write_meshes:
                    export_mesh(
                        output_dir / "meshes" / split / name / f"{scan_id}.ply",
                        prediction,
                        contract.faces,
                    )
            for key in DISTANCE_KEYS:
                row[f"corrected_minus_pca_{key}"] = (
                    float(row[f"corrected_{key}"]) - float(row[f"pca_{key}"])
                )
            rows.append(row)
            print(
                f"[{split}] {index + 1:4d}/{len(data)} {scan_id} "
                f"RMSE {row['pca_coordinate_rmse_mm']:.6f} -> "
                f"{row['corrected_coordinate_rmse_mm']:.6f} mm; "
                f"ASSD {row['pca_assd_mm']:.6f} -> {row['corrected_assd_mm']:.6f} mm",
                flush=True,
            )

    summary = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "selected_model_is_exact_pca_fallback": bool(checkpoint["best_epoch"] == 0),
        "device": str(device),
        "surface_points_per_direction": surface_points,
        "splits": {
            split: aggregate(rows, split, repeats, bootstrap_seed) for split in args.splits
        },
        "distance_difference_definition": "corrected minus PCA; negative favors the corrective model",
        "test_was_explicitly_authorized": bool("test" in args.splits and args.allow_test),
        "data_contract": contract.summary(),
    }
    common.atomic_write_csv(output_dir / "per_scan_metrics.csv", rows)
    common.atomic_write_json(output_dir / "evaluation_summary.json", summary)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
