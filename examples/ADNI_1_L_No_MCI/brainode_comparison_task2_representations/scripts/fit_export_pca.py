#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import trimesh
from sklearn.decomposition import PCA

from task2_common import (
    TASK_DIR,
    load_config,
    load_manifest,
    load_obj_arrays,
    resolve_repo_path,
    rows_for_split,
    sha256_file,
    summarize_values,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit train-only PCA, export coefficients, and reconstruct meshes."
    )
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "pipeline.json"),
        help="Task 2 pipeline configuration.",
    )
    parser.add_argument(
        "--skip-meshes",
        action="store_true",
        help="Export PCA and metrics without writing reconstructed PLY files.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    pca_config = config["pca"]
    manifest_path = resolve_repo_path(config["manifest"])
    rows = load_manifest(manifest_path)
    train_rows = rows_for_split(rows, "train")
    expected_train = int(config["expected_counts"]["train"])
    if len(train_rows) != expected_train:
        raise RuntimeError(
            f"Expected {expected_train} training scans, found {len(train_rows)}."
        )

    model_dir = TASK_DIR / "pca" / "model"
    coefficient_dir = TASK_DIR / "pca" / "coefficients"
    per_scan_dir = coefficient_dir / "per_scan"
    metric_dir = TASK_DIR / "pca" / "metrics"
    for directory in (model_dir, coefficient_dir, per_scan_dir, metric_dir):
        directory.mkdir(parents=True, exist_ok=True)

    vertices_by_scan: dict[str, np.ndarray] = {}
    reference_faces = None
    for row in rows:
        vertices, faces = load_obj_arrays(row["mesh_path"])
        vertices_by_scan[row["scan_id"]] = vertices
        if reference_faces is None:
            reference_faces = faces
        elif not np.array_equal(reference_faces, faces):
            raise RuntimeError(f"Topology mismatch for {row['scan_id']}")
    if reference_faces is None:
        raise RuntimeError("No meshes were loaded.")

    train_matrix = np.stack(
        [vertices_by_scan[row["scan_id"]].reshape(-1) for row in train_rows],
        axis=0,
    )
    max_components = int(pca_config["max_components"])
    pca = PCA(
        n_components=max_components,
        svd_solver=str(pca_config.get("svd_solver", "full")),
    )
    pca.fit(train_matrix)

    np.save(model_dir / "mean.npy", pca.mean_.astype(np.float32))
    np.save(model_dir / "components_256.npy", pca.components_.astype(np.float32))
    np.save(model_dir / "explained_variance.npy", pca.explained_variance_.astype(np.float64))
    np.save(
        model_dir / "explained_variance_ratio.npy",
        pca.explained_variance_ratio_.astype(np.float64),
    )
    np.save(model_dir / "singular_values.npy", pca.singular_values_.astype(np.float64))
    np.save(model_dir / "faces.npy", reference_faces.astype(np.int32))
    with (model_dir / "pca_brainode.pkl").open("wb") as handle:
        pickle.dump(
            {
                "pca": pca,
                "faces": reference_faces.astype(np.int32),
                "fit_split": "train",
                "train_scan_ids": [row["scan_id"] for row in train_rows],
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    write_json(
        model_dir / "training_scan_ids.json",
        [row["scan_id"] for row in train_rows],
    )

    all_matrix = np.stack(
        [vertices_by_scan[row["scan_id"]].reshape(-1) for row in rows], axis=0
    )
    all_coefficients = pca.transform(all_matrix).astype(np.float32)
    train_coefficients = all_coefficients[
        [index for index, row in enumerate(rows) if row["split"] == "train"]
    ]
    coefficient_mean = train_coefficients.mean(axis=0)
    coefficient_std = train_coefficients.std(axis=0)
    np.save(model_dir / "coefficient_mean_train.npy", coefficient_mean)
    np.save(model_dir / "coefficient_std_train.npy", coefficient_std)

    coefficient_rows = []
    for index, row in enumerate(rows):
        path = per_scan_dir / f"{row['scan_id']}.npy"
        np.save(path, all_coefficients[index])
        coefficient_rows.append(
            {
                "scan_id": row["scan_id"],
                "image_id": row["image_id"],
                "subject_id": row["subject_id"],
                "split": row["split"],
                "diagnosis": row["diagnosis"],
                "label_ad": row["label_ad"],
                "visit_order": row["visit_order"],
                "age_norm": row["age_norm"],
                "coefficient_dimension": max_components,
                "coefficient_path": str(path),
            }
        )
    write_csv(
        coefficient_dir / "pca_coefficients.csv",
        list(coefficient_rows[0].keys()),
        coefficient_rows,
    )

    for split in ("train", "val", "test"):
        indices = [index for index, row in enumerate(rows) if row["split"] == split]
        np.savez_compressed(
            coefficient_dir / f"{split}_coefficients.npz",
            scan_ids=np.asarray([rows[index]["scan_id"] for index in indices]),
            coefficients=all_coefficients[indices],
        )

    report_components = [int(value) for value in pca_config["report_components"]]
    save_mesh_components = {
        int(value) for value in pca_config["save_mesh_components"]
    }
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    metric_rows: list[dict] = []
    summaries: dict[str, dict] = {}

    for component_count in report_components:
        reconstructed = (
            pca.mean_[None, :]
            + all_coefficients[:, :component_count].astype(np.float64)
            @ pca.components_[:component_count]
        )
        difference = reconstructed - all_matrix
        vertex_difference = difference.reshape(len(rows), -1, 3)
        rmse = np.sqrt(np.mean(vertex_difference**2, axis=(1, 2)))
        mae = np.mean(np.abs(vertex_difference), axis=(1, 2))

        for index, row in enumerate(rows):
            metric_rows.append(
                {
                    "scan_id": row["scan_id"],
                    "split": row["split"],
                    "diagnosis": row["diagnosis"],
                    "pca_components": component_count,
                    "vertex_rmse": float(rmse[index]),
                    "vertex_mae": float(mae[index]),
                }
            )
        summaries[str(component_count)] = {
            "cumulative_explained_variance": float(
                cumulative_variance[component_count - 1]
            ),
            "vertex_rmse": summarize_values(rmse),
            "vertex_mae": summarize_values(mae),
        }

        if component_count in save_mesh_components and not args.skip_meshes:
            output_dir = (
                TASK_DIR / "pca" / "reconstructed_meshes" / f"k{component_count}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            for index, row in enumerate(rows):
                vertices = reconstructed[index].reshape(-1, 3)
                mesh = trimesh.Trimesh(
                    vertices=vertices, faces=reference_faces, process=False
                )
                mesh.export(output_dir / f"{row['scan_id']}.ply")

    write_csv(
        metric_dir / "pca_vertex_reconstruction_per_scan.csv",
        list(metric_rows[0].keys()),
        metric_rows,
    )
    model_report = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "fit_split": "train",
        "train_scan_count": len(train_rows),
        "feature_count": int(train_matrix.shape[1]),
        "max_components": max_components,
        "primary_components": int(pca_config["primary_components"]),
        "report_components": report_components,
        "saved_mesh_components": (
            [] if args.skip_meshes else sorted(save_mesh_components)
        ),
        "component_summaries": summaries,
    }
    write_json(model_dir / "pca_config.json", model_report)
    write_json(metric_dir / "pca_dimension_summary.json", summaries)

    primary = str(int(pca_config["primary_components"]))
    print(
        f"PCA complete: {len(train_rows)} training scans, "
        f"{summaries[primary]['cumulative_explained_variance']:.6f} "
        f"variance at {primary} components."
    )
    print(f"Wrote PCA outputs to {TASK_DIR / 'pca'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
