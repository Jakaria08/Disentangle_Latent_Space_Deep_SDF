#!/usr/bin/env python3
"""Numerically validate the spherical embedding and SPHARM fit/reconstruct round trip.

Run this before trusting ``fit_spharm_basis.py`` on real data. Checks, in order:

1. The 12 icosahedral anchors are found and pass their own internal structural
   asserts (``find_icosahedral_anchors`` already raises on any mismatch).
2. Zero triangle-orientation flips in the solved embedding, and a bounded
   area-distortion ratio (max/min triangle chord-area).
3. A synthetic round trip: a known spherical-harmonic function of a chosen
   degree, evaluated at the embedding's own angles, is fit and reconstructed;
   residual must collapse to numerical precision at and above the true
   degree, and must be strictly larger below it (confirms the fit is neither
   trivially exact nor broken).
4. The same round trip on real cohort data: fitting the shared basis to the
   PCA-stored mean shape's vertex positions at increasing degree must give
   monotonically non-increasing RMSE.

No output is written; this is a read-only numerical gate.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from spharm_common import DEFAULT_PCA_DIR, resolve_path
import spharm_embedding as se


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", default=str(DEFAULT_PCA_DIR / "faces.npy"))
    parser.add_argument("--mean-vertices", default=str(DEFAULT_PCA_DIR / "mean_vertices.npy"))
    parser.add_argument("--max-area-ratio", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"OK: {message}")


def main() -> None:
    args = parse_args()
    faces = np.load(resolve_path(args.faces))
    vertex_count = int(faces.max()) + 1

    angles, xyz, anchors = se.build_spherical_embedding(faces, vertex_count)
    check(angles.shape == (vertex_count, 2), f"embedding produced angles for all {vertex_count} vertices")
    check(len(anchors["vertex_ids"]) == 12, "exactly 12 icosahedral anchors recovered")

    signed = se.triangle_signed_areas(xyz, faces)
    flips = int(np.sum(signed <= 0))
    check(flips == 0, f"zero flipped/degenerate triangles out of {len(faces)}")

    areas = se.triangle_chord_areas(xyz, faces)
    ratio = float(areas.max() / areas.min())
    check(
        ratio <= args.max_area_ratio,
        f"area-distortion ratio {ratio:.1f} <= max-area-ratio {args.max_area_ratio:.0f}",
    )

    # --- Synthetic round trip -------------------------------------------------
    rng = np.random.default_rng(args.seed)
    true_degree = 6
    basis_true = se.real_sh_basis(angles[:, 0], angles[:, 1], true_degree)
    true_coefficients = rng.normal(size=(basis_true.shape[1], 3))
    synthetic = basis_true @ true_coefficients

    residuals = {}
    for degree in (2, 4, true_degree, true_degree + 2, true_degree + 6):
        basis = se.real_sh_basis(angles[:, 0], angles[:, 1], degree)
        solver = se.SharedBasisSolver(basis)
        reconstructed = solver.reconstruct(solver.fit(synthetic))
        residuals[degree] = float(np.sqrt(np.mean(np.sum((reconstructed - synthetic) ** 2, axis=1))))
    print("synthetic round-trip RMSE by degree:", residuals)
    check(residuals[2] > 1.0e-6, "degree below truth leaves a real (non-numerical) residual")
    check(residuals[true_degree] < 1.0e-8, f"degree == truth ({true_degree}) reconstructs to numerical precision")
    check(residuals[true_degree + 6] < 1.0e-8, "degree above truth stays at numerical precision (no overfitting blowup)")
    check(
        residuals[2] > residuals[4] > residuals[true_degree],
        "residual decreases monotonically toward the true degree",
    )

    # --- Real-data round trip on the PCA mean shape ---------------------------
    mean_vertices = np.load(resolve_path(args.mean_vertices)).reshape(vertex_count, 3)
    real_rmse = {}
    for degree in (4, 8, 16, 24, 32):
        basis = se.real_sh_basis(angles[:, 0], angles[:, 1], degree)
        solver = se.SharedBasisSolver(basis)
        reconstructed = solver.reconstruct(solver.fit(mean_vertices))
        real_rmse[degree] = float(np.sqrt(np.mean(np.sum((reconstructed - mean_vertices) ** 2, axis=1))))
    print("PCA mean-shape reconstruction RMSE (SDF-space units) by degree:", real_rmse)
    degrees_sorted = sorted(real_rmse)
    check(
        all(real_rmse[degrees_sorted[i]] >= real_rmse[degrees_sorted[i + 1]] for i in range(len(degrees_sorted) - 1)),
        "real-data reconstruction RMSE is monotonically non-increasing with degree",
    )

    print("All SPHARM embedding validation checks passed.")


if __name__ == "__main__":
    main()
