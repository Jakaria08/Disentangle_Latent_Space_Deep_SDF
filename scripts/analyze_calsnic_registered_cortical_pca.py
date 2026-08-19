#!/usr/bin/env python3
"""Build a compact PCA model for registered CALSNIC cortical pial surfaces.

The source meshes are FreeSurfer fs6-resampled, volume-normalized, rigidly
registered bilateral pial surfaces.  They therefore have a common topology and
vertex correspondence, which makes vertex-coordinate PCA meaningful.

The script deliberately stores only the PCA model and compact diagnostics.  It
does not copy the input meshes or persist the temporary centered data matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


DEFAULT_SOURCE = Path(
    "/home/jakaria/CALSNIC/calsnic_pial_surface/mesh_dataset/"
    "pial_surface_volume_normalized_for_sex_rigid_reg"
)
DEFAULT_OUTPUT = Path("/mnt/bulk10tb/CALSNIC/registered_cortical_pca")
CONTROLS_FILE = Path("/home/jakaria/CALSNIC/calsnic_pial_surface/calsnic_controls_213.txt")
ALS_FILE = Path("/home/jakaria/CALSNIC/calsnic_pial_surface/calsnic_als_patients_233.txt")
REQUESTED_COMPONENTS = (128, 256, 512)
CURVE_COMPONENTS = (0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help=(
            "Hold out this fraction of meshes for reconstruction evaluation. "
            "PCA is fitted only on the remaining meshes."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed used for the held-out split."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory. This never touches the source meshes.",
    )
    return parser.parse_args()


def read_subject_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def split_paths(
    paths: list[Path], test_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Create a reproducible hold-out split without changing source ordering."""
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("--test-fraction must be in [0, 1)")
    if test_fraction == 0.0:
        return paths, []
    n_test = round(len(paths) * test_fraction)
    if n_test < 1 or len(paths) - n_test < 2:
        raise ValueError("The requested split leaves too few meshes for PCA fitting")
    rng = np.random.default_rng(seed)
    test_indices = set(rng.choice(len(paths), size=n_test, replace=False).tolist())
    train_paths = [path for index, path in enumerate(paths) if index not in test_indices]
    test_paths = [path for index, path in enumerate(paths) if index in test_indices]
    return train_paths, test_paths


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices and triangular faces while preserving PLY order exactly."""
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    for field in ("x", "y", "z"):
        if field not in vertex.dtype.names:
            raise ValueError(f"{path}: vertex field {field!r} is missing")
    vertices = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float32, copy=False
    )
    face_data = ply["face"].data["vertex_indices"]
    if not all(len(face) == 3 for face in face_data):
        raise ValueError(f"{path}: non-triangular faces are not supported")
    faces = np.asarray(face_data.tolist(), dtype=np.int32)
    return vertices, faces


def validate_and_mean(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Validate correspondence and calculate the vertexwise mean in one pass."""
    reference_faces: np.ndarray | None = None
    expected_shape: tuple[int, int] | None = None
    vertex_sum: np.ndarray | None = None

    for index, path in enumerate(paths, start=1):
        vertices, faces = read_mesh(path)
        if expected_shape is None:
            expected_shape = vertices.shape
            reference_faces = faces
            vertex_sum = np.zeros(expected_shape, dtype=np.float64)
        if vertices.shape != expected_shape:
            raise ValueError(
                f"{path}: vertex shape {vertices.shape} differs from {expected_shape}"
            )
        if not np.array_equal(faces, reference_faces):
            raise ValueError(f"{path}: face order/topology differs from the first mesh")
        if not np.isfinite(vertices).all():
            raise ValueError(f"{path}: contains non-finite vertex coordinates")
        assert vertex_sum is not None
        vertex_sum += vertices
        if index % 25 == 0 or index == len(paths):
            print(f"Validated and summed {index}/{len(paths)} meshes", flush=True)

    assert vertex_sum is not None and reference_faces is not None
    return (vertex_sum / len(paths)).astype(np.float32), reference_faces


def validate_against_reference(
    paths: list[Path], expected_shape: tuple[int, int], reference_faces: np.ndarray
) -> None:
    """Confirm held-out meshes have the same ordered topology as the training set."""
    for index, path in enumerate(paths, start=1):
        vertices, faces = read_mesh(path)
        if vertices.shape != expected_shape:
            raise ValueError(
                f"{path}: vertex shape {vertices.shape} differs from {expected_shape}"
            )
        if not np.array_equal(faces, reference_faces):
            raise ValueError(f"{path}: face order/topology differs from the training meshes")
        if not np.isfinite(vertices).all():
            raise ValueError(f"{path}: contains non-finite vertex coordinates")
        if index % 25 == 0 or index == len(paths):
            print(f"Validated held-out mesh {index}/{len(paths)}", flush=True)


def build_centered_matrix(
    paths: list[Path], mean_vertices: np.ndarray, work_dir: Path
) -> np.memmap:
    """Write the only temporary data matrix as a float32 memmap on the bulk disk."""
    n_samples = len(paths)
    n_features = mean_vertices.size
    matrix_path = work_dir / "centered_features.float32.dat"
    centered = np.memmap(
        matrix_path, mode="w+", dtype=np.float32, shape=(n_samples, n_features)
    )
    for index, path in enumerate(paths):
        vertices, _ = read_mesh(path)
        centered[index] = (vertices - mean_vertices).reshape(-1)
        if (index + 1) % 25 == 0 or index + 1 == n_samples:
            print(f"Loaded {index + 1}/{n_samples} meshes into temporary matrix", flush=True)
    centered.flush()
    return centered


def pca_from_centered_matrix(
    centered: np.memmap, output_dir: Path, n_vertices: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA through the sample Gram matrix and stream components to disk.

    With far fewer subjects than coordinates, the n-by-n Gram eigendecomposition
    is exact (up to floating point precision) and avoids a costly full SVD of a
    202-by-245,772 matrix.
    """
    n_samples, n_features = centered.shape
    gram = np.asarray(centered @ centered.T, dtype=np.float64) / (n_samples - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], a_min=0.0, a_max=None)
    eigenvectors = eigenvectors[:, order]

    # Numerical tolerance for a centred n-sample matrix.  The theoretical limit
    # is n-1, and this data set is expected to attain that limit.
    tolerance = max(float(eigenvalues[0]) * 1e-8, 1e-12)
    rank = min(n_samples - 1, int(np.count_nonzero(eigenvalues > tolerance)))
    if rank < 1:
        raise RuntimeError("No non-zero PCA modes were found")
    eigenvalues = eigenvalues[:rank]
    eigenvectors = eigenvectors[:, :rank]
    singular_values = np.sqrt(eigenvalues * (n_samples - 1))
    scores = (eigenvectors * singular_values[None, :]).astype(np.float32)

    component_path = output_dir / "components.npy"
    components = np.lib.format.open_memmap(
        component_path,
        mode="w+",
        dtype=np.float32,
        shape=(rank, n_vertices, 3),
    )
    component_flat = components.reshape(rank, n_features)
    block_width = 8192
    for start in range(0, n_features, block_width):
        stop = min(start + block_width, n_features)
        component_flat[:, start:stop] = (
            eigenvectors.T @ centered[:, start:stop] / singular_values[:, None]
        ).astype(np.float32)
        print(f"Wrote component features {start}:{stop}/{n_features}", flush=True)
    components.flush()
    del components

    np.save(output_dir / "explained_variance.npy", eigenvalues.astype(np.float32))
    np.save(output_dir / "singular_values.npy", singular_values.astype(np.float32))
    np.save(output_dir / "scores.npy", scores)
    return eigenvalues, singular_values, scores


def project_held_out_meshes(
    paths: list[Path],
    mean_vertices: np.ndarray,
    reference_faces: np.ndarray,
    component_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Project held-out meshes into the training-only PCA basis in memory."""
    n_features = mean_vertices.size
    held_out = np.empty((len(paths), n_features), dtype=np.float32)
    for index, path in enumerate(paths):
        vertices, faces = read_mesh(path)
        if vertices.shape != mean_vertices.shape or not np.array_equal(faces, reference_faces):
            raise ValueError(f"{path}: differs from the verified training topology")
        held_out[index] = (vertices - mean_vertices).reshape(-1)
        if (index + 1) % 25 == 0 or index + 1 == len(paths):
            print(f"Projected-input load {index + 1}/{len(paths)}", flush=True)
    components = np.load(component_path, mmap_mode="r").reshape(-1, n_features)
    scores = (held_out @ components.T).astype(np.float32)
    return held_out, scores


def requested_summary(
    centered: np.memmap,
    scores: np.ndarray,
    eigenvalues: np.ndarray,
    n_vertices: int,
) -> list[dict[str, float | int]]:
    """Compute in-sample residuals without materialising reconstructed meshes."""
    total_per_sample = np.einsum("ij,ij->i", centered, centered, dtype=np.float64)
    cumulative_energy = np.cumsum(scores.astype(np.float64) ** 2, axis=1)
    total_variance = float(eigenvalues.sum())
    rows: list[dict[str, float | int]] = []
    for requested in REQUESTED_COMPONENTS:
        effective = min(requested, scores.shape[1])
        residual_sse = np.maximum(total_per_sample - cumulative_energy[:, effective - 1], 0.0)
        vertex_rmse = np.sqrt(residual_sse / n_vertices)
        rows.append(
            {
                "requested_components": requested,
                "effective_components": effective,
                "explained_variance_ratio": float(eigenvalues[:effective].sum() / total_variance),
                "mean_vertex_rmse_normalized_units": float(vertex_rmse.mean()),
                "median_vertex_rmse_normalized_units": float(np.median(vertex_rmse)),
                "p95_vertex_rmse_normalized_units": float(np.quantile(vertex_rmse, 0.95)),
                "max_vertex_rmse_normalized_units": float(vertex_rmse.max()),
            }
        )
    return rows


def reconstruction_curve(
    centered: np.memmap, scores: np.ndarray, n_vertices: int
) -> list[dict[str, float | int]]:
    total_per_sample = np.einsum("ij,ij->i", centered, centered, dtype=np.float64)
    component_counts = sorted({k for k in CURVE_COMPONENTS if k < scores.shape[1]} | {scores.shape[1]})
    rows: list[dict[str, float | int]] = []
    for effective in component_counts:
        if effective == 0:
            residual_sse = total_per_sample
        else:
            captured = np.sum(scores[:, :effective].astype(np.float64) ** 2, axis=1)
            residual_sse = np.maximum(total_per_sample - captured, 0.0)
        vertex_rmse = np.sqrt(residual_sse / n_vertices)
        rows.append(
            {
                "effective_components": effective,
                "mean_vertex_rmse_normalized_units": float(vertex_rmse.mean()),
                "median_vertex_rmse_normalized_units": float(np.median(vertex_rmse)),
                "p95_vertex_rmse_normalized_units": float(np.quantile(vertex_rmse, 0.95)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(
    output_dir: Path,
    curve: list[dict[str, float | int]],
    eigenvalues: np.ndarray,
    summary: list[dict[str, float | int]],
) -> None:
    curve_x = [int(row["effective_components"]) for row in curve]
    curve_y = [float(row["mean_vertex_rmse_normalized_units"]) for row in curve]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    ax.plot(curve_x, curve_y, marker="o", color="#1f77b4")
    for row in summary:
        ax.axvline(
            int(row["effective_components"]), color="#d62728", alpha=0.28, linewidth=1
        )
    ax.set_xlabel("Effective PCA components")
    ax.set_ylabel("Mean per-vertex RMSE (normalized coordinate units)")
    ax.set_title("Registered cortical-surface PCA reconstruction error")
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "reconstruction_error_curve.png", dpi=160)
    plt.close(fig)

    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative * 100, color="#2ca02c")
    for row in summary:
        ax.axvline(
            int(row["effective_components"]), color="#d62728", alpha=0.28, linewidth=1
        )
    ax.set_xlabel("Effective PCA components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_ylim(0, 100.5)
    ax.set_title("Registered cortical-surface PCA variance explained")
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "explained_variance_curve.png", dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    n_total_meshes: int,
    n_training_meshes: int,
    n_evaluation_meshes: int,
    n_vertices: int,
    n_faces: int,
    rank: int,
    summary: list[dict[str, float | int]],
    selected_id: str,
    evaluation_label: str,
) -> None:
    lines = [
        "# CALSNIC registered cortical-surface PCA",
        "",
        f"- Available input: {n_total_meshes} healthy-control, bilateral pial meshes.",
        f"- PCA fitting set: {n_training_meshes} meshes.",
        f"- Reconstruction evaluation: {evaluation_label} ({n_evaluation_meshes} meshes).",
        f"- Common topology: {n_vertices:,} vertices and {n_faces:,} triangular faces per mesh.",
        "- Source: sex-volume-normalized, rigidly registered mesh directory.",
        f"- Maximum possible PCA rank for the training set is N - 1 = {rank}; the fitted rank is {rank}.",
        "- There are no CALSNIC `.sdf` files in the inspected CALSNIC directory.",
        "",
        "## Requested reconstruction dimensions",
        "",
        "| Requested | Effective | Variance explained | Mean vertex RMSE (normalized units) | 95th percentile (normalized units) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {requested_components} | {effective_components} | {explained_variance_ratio:.3%} | "
            "{mean_vertex_rmse_normalized_units:.4f} | {p95_vertex_rmse_normalized_units:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"The notebook visualizes the median-128D-error {evaluation_label} subject: `{selected_id}`.",
            "",
            f"The errors are measured on the {evaluation_label} set. "
            "Because PCA rank is limited by the number of centred training subjects, 256D and 512D requests are necessarily "
            f"clipped to the {rank} available PCA directions and should have identical reconstruction error.",
            "The input meshes have been eTIV-volume-normalized, so their coordinate units are not physical millimetres. "
            "Use the original per-subject normalization scale to convert a normalized-coordinate displacement approximately back to the original-coordinate scale.",
        ]
    )
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    all_paths = sorted(args.source_dir.glob("*.ply"))
    if not all_paths:
        raise FileNotFoundError(f"No .ply meshes found in {args.source_dir}")
    paths, held_out_paths = split_paths(all_paths, args.test_fraction, args.seed)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is non-empty. Use --overwrite to replace this analysis output."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_dir / "_work"
    work_dir.mkdir()
    centered: np.memmap | None = None

    controls = read_subject_ids(CONTROLS_FILE)
    als = read_subject_ids(ALS_FILE)
    all_subject_ids = [path.stem for path in all_paths]
    subject_ids = [path.stem for path in paths]
    held_out_ids = [path.stem for path in held_out_paths]
    print(
        f"Fitting PCA to {len(paths)} registered meshes from {args.source_dir}; "
        f"held-out evaluation meshes: {len(held_out_paths)}",
        flush=True,
    )

    try:
        mean_vertices, faces = validate_and_mean(paths)
        if held_out_paths:
            validate_against_reference(held_out_paths, mean_vertices.shape, faces)
        centered = build_centered_matrix(paths, mean_vertices, work_dir)
        eigenvalues, singular_values, scores = pca_from_centered_matrix(
            centered, args.output_dir, mean_vertices.shape[0]
        )
        if held_out_paths:
            evaluation_centered, evaluation_scores = project_held_out_meshes(
                held_out_paths,
                mean_vertices,
                faces,
                args.output_dir / "components.npy",
            )
            np.save(args.output_dir / "test_scores.npy", evaluation_scores)
            evaluation_paths = held_out_paths
            evaluation_ids = held_out_ids
            evaluation_label = "held-out test"
        else:
            evaluation_centered = centered
            evaluation_scores = scores
            evaluation_paths = paths
            evaluation_ids = subject_ids
            evaluation_label = "in-sample training"
        summary = requested_summary(
            evaluation_centered, evaluation_scores, eigenvalues, mean_vertices.shape[0]
        )
        curve = reconstruction_curve(
            evaluation_centered, evaluation_scores, mean_vertices.shape[0]
        )

        # Select a typical rather than unusually easy/hard mesh for notebook display.
        effective_128 = min(128, evaluation_scores.shape[1])
        total_per_sample = np.einsum(
            "ij,ij->i", evaluation_centered, evaluation_centered, dtype=np.float64
        )
        captured_128 = np.sum(
            evaluation_scores[:, :effective_128].astype(np.float64) ** 2, axis=1
        )
        rmse_128 = np.sqrt(
            np.maximum(total_per_sample - captured_128, 0.0) / mean_vertices.shape[0]
        )
        selected_index = int(np.argmin(np.abs(rmse_128 - np.median(rmse_128))))

        np.save(args.output_dir / "mean_vertices.npy", mean_vertices)
        np.save(args.output_dir / "faces.npy", faces)
        write_csv(args.output_dir / "reconstruction_error_summary.csv", summary)
        write_csv(args.output_dir / "reconstruction_error_curve.csv", curve)
        save_plots(args.output_dir, curve, eigenvalues, summary)

        metadata = {
            "source_dir": str(args.source_dir),
            "mesh_variant": "pial_surface_volume_normalized_for_sex_rigid_reg",
            "n_meshes": len(all_paths),
            "n_training_meshes": len(paths),
            "n_test_meshes": len(held_out_paths),
            "n_vertices": int(mean_vertices.shape[0]),
            "n_faces": int(faces.shape[0]),
            "n_features": int(mean_vertices.size),
            "pca_rank": int(len(eigenvalues)),
            "requested_components": list(REQUESTED_COMPONENTS),
            "effective_components": {
                str(k): min(k, int(len(eigenvalues))) for k in REQUESTED_COMPONENTS
            },
            "input_classes": {
                "controls": sum(subject in controls for subject in all_subject_ids),
                "als": sum(subject in als for subject in all_subject_ids),
                "unclassified": sum(
                    subject not in controls and subject not in als for subject in all_subject_ids
                ),
            },
            "split": {
                "evaluation": evaluation_label,
                "test_fraction": args.test_fraction,
                "seed": args.seed if held_out_paths else None,
                "training_subject_ids": subject_ids,
                "test_subject_ids": held_out_ids,
            },
            "representative_subject": {
                "id": evaluation_ids[selected_index],
                "index": selected_index,
                "selection": f"closest to the median 128D {evaluation_label} vertex RMSE",
                "source_ply": str(evaluation_paths[selected_index]),
            },
            "arrays": {
                "mean_vertices": "mean_vertices.npy",
                "components": "components.npy",
                "scores": "scores.npy",
                "explained_variance": "explained_variance.npy",
                "singular_values": "singular_values.npy",
                "faces": "faces.npy",
                "test_scores": "test_scores.npy" if held_out_paths else None,
            },
            "storage_note": "The temporary centred matrix is deleted after fitting; input meshes are not copied.",
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        write_report(
            args.output_dir,
            len(all_paths),
            len(paths),
            len(evaluation_paths),
            mean_vertices.shape[0],
            faces.shape[0],
            len(eigenvalues),
            summary,
            evaluation_ids[selected_index],
            evaluation_label,
        )
        # Close the memmap before its directory is removed.  On some filesystems
        # an open mapping prevents rmtree from removing the temporary file.
        centered.flush()
        centered._mmap.close()
        centered = None
        print(f"Finished. Results written to {args.output_dir}", flush=True)
    finally:
        # The model arrays have already been written; this is only a transient copy
        # of the centred meshes and is intentionally removed to limit storage use.
        if centered is not None:
            centered.flush()
            centered._mmap.close()
        if work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
