#!/usr/bin/env python3
"""Deterministic surface and topology metrics for corresponded hippocampus meshes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def _ensure_rtree_available() -> None:
    """Use the sibling INR environment's rtree backend when pytorch_geo lacks it.

    Training needs pytorch_geo/openmesh, while the existing exact-SDF environment owns the
    rtree/libspatialindex installation used by all prior ASSD evaluators. Import only that
    missing package from the sibling environment; do not prepend or replace the active
    environment's site-packages (which could change torch/numpy versions).
    """
    def usable() -> bool:
        try:
            from rtree import index

            return callable(getattr(index, "Index", None))
        except (ImportError, ModuleNotFoundError):
            return False

    if usable():
        return
    candidates = []
    override = os.environ.get("DEEP3DCOMP_RTREE_SITE_PACKAGES")
    if override:
        candidates.append(Path(override))
    executable = Path(sys.executable).resolve()
    if executable.parent.name == "bin" and executable.parent.parent.parent.name == "envs":
        environments = executable.parent.parent.parent
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidates.append(environments / "inr_sdf" / "lib" / version / "site-packages")
    for candidate in candidates:
        if candidate.exists():
            # This function runs before trimesh is imported.  Prepending only the
            # known sibling site-packages is necessary: appending it would let a
            # broken user-local optional import win.  Torch and NumPy are already
            # imported by the caller, so their active-environment versions remain
            # fixed in sys.modules.
            sys.path.insert(0, str(candidate))
            if usable():
                return
    raise ModuleNotFoundError(
        "Exact triangle-distance evaluation requires rtree. Install it in the active "
        "environment or set DEEP3DCOMP_RTREE_SITE_PACKAGES to a site-packages directory "
        "that contains rtree."
    )


_ensure_rtree_available()
import trimesh


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int):
    points, faces = trimesh.sample.sample_surface(mesh, int(count), seed=int(seed))
    return np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def discrete_curvature_per_mm(mesh: trimesh.Trimesh) -> float:
    if not len(mesh.face_adjacency):
        return 0.0
    angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    area = float(mesh.area)
    return float(0.5 * np.sum(lengths * angles) / max(area, 1e-12))


def metrics(
    ground_truth_vertices: np.ndarray,
    predicted_vertices: np.ndarray,
    faces: np.ndarray,
    surface_points: int,
    seed: int,
) -> dict:
    ground_truth = trimesh.Trimesh(
        vertices=np.asarray(ground_truth_vertices), faces=np.asarray(faces), process=False
    )
    predicted = trimesh.Trimesh(
        vertices=np.asarray(predicted_vertices), faces=np.asarray(faces), process=False
    )
    gt_points, gt_sample_faces = sample_surface(ground_truth, surface_points, seed)
    pred_points, pred_sample_faces = sample_surface(predicted, surface_points, seed + 1)
    _closest, gt_to_pred, gt_target_faces = trimesh.proximity.ProximityQuery(predicted).on_surface(
        gt_points
    )
    _closest, pred_to_gt, pred_target_faces = trimesh.proximity.ProximityQuery(ground_truth).on_surface(
        pred_points
    )
    gt_normals = ground_truth.face_normals[gt_sample_faces]
    predicted_at_gt = predicted.face_normals[gt_target_faces]
    predicted_normals = predicted.face_normals[pred_sample_faces]
    gt_at_predicted = ground_truth.face_normals[pred_target_faces]
    cosine = np.concatenate(
        (
            np.einsum("ij,ij->i", gt_normals, predicted_at_gt),
            np.einsum("ij,ij->i", predicted_normals, gt_at_predicted),
        )
    )
    corresponding_face_cosine = np.einsum(
        "ij,ij->i", ground_truth.face_normals, predicted.face_normals
    )
    gt_volume = abs(float(ground_truth.volume))
    pred_volume = abs(float(predicted.volume))
    return {
        "assd_mm": float(0.5 * (gt_to_pred.mean() + pred_to_gt.mean())),
        "hd95_mm": float(max(np.quantile(gt_to_pred, 0.95), np.quantile(pred_to_gt, 0.95))),
        "chamfer_l1_mm": float(gt_to_pred.mean() + pred_to_gt.mean()),
        "chamfer_l2_squared_mm2": float(
            np.square(gt_to_pred).mean() + np.square(pred_to_gt).mean()
        ),
        "normal_absolute_cosine": float(np.abs(cosine).mean()),
        "normal_signed_cosine": float(cosine.mean()),
        "flipped_face_fraction_vs_ground_truth": float(np.mean(corresponding_face_cosine < 0.0)),
        "volume_relative_error": float(abs(pred_volume - gt_volume) / max(gt_volume, 1e-12)),
        "surface_area_ratio": float(predicted.area / max(float(ground_truth.area), 1e-12)),
        "curvature_per_mm_mean": discrete_curvature_per_mm(predicted),
        "ground_truth_curvature_per_mm_mean": discrete_curvature_per_mm(ground_truth),
        "curvature_ratio_to_ground_truth": float(
            discrete_curvature_per_mm(predicted)
            / max(discrete_curvature_per_mm(ground_truth), 1e-12)
        ),
        "predicted_watertight": bool(predicted.is_watertight),
        "predicted_winding_consistent": bool(predicted.is_winding_consistent),
        "predicted_connected_components": int(len(predicted.split(only_watertight=False))),
        "predicted_euler_number": int(predicted.euler_number),
    }
