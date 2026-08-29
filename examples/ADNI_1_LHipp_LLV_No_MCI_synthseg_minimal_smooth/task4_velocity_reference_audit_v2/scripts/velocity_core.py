#!/usr/bin/env python3
"""Numerical core for the longitudinal velocity-reference audit.

The functions in this file are deliberately independent of the trained neural
models.  They operate on registered physical meshes in millimetres and keep the
visit, subject, and split boundaries explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


EPS = 1.0e-12


@dataclass(frozen=True)
class Candidate:
    name: str
    label: str
    alignment: str
    basis: str
    degree: int
    components: int | None = None
    robust: bool = False
    ridge: float = 0.0

    @property
    def minimum_fit_visits(self) -> int:
        return self.degree + 1


@dataclass
class Trajectory:
    candidate: Candidate
    time_center: float
    coefficients: np.ndarray
    output_shape: tuple[int, ...]
    basis_mean: np.ndarray | None = None
    basis_components: np.ndarray | None = None

    def _design(self, ages: np.ndarray) -> np.ndarray:
        centered = np.asarray(ages, dtype=np.float64).reshape(-1) - self.time_center
        return np.stack([centered**power for power in range(self.candidate.degree + 1)], axis=1)

    def predict(self, ages: np.ndarray | float) -> np.ndarray:
        query = np.atleast_1d(np.asarray(ages, dtype=np.float64))
        compact = self._design(query) @ self.coefficients
        if self.basis_components is not None:
            compact = compact @ self.basis_components + self.basis_mean
        return compact.reshape((len(query),) + self.output_shape)

    def derivative(self, ages: np.ndarray | float) -> np.ndarray:
        query = np.atleast_1d(np.asarray(ages, dtype=np.float64))
        centered = query - self.time_center
        design = np.zeros((len(query), self.candidate.degree + 1), dtype=np.float64)
        for power in range(1, self.candidate.degree + 1):
            design[:, power] = power * centered ** (power - 1)
        compact = design @ self.coefficients
        if self.basis_components is not None:
            compact = compact @ self.basis_components
        return compact.reshape((len(query),) + self.output_shape)


def candidates_from_config(config: dict) -> list[Candidate]:
    result = [
        Candidate("surface_linear_raw", "Linear surface (registered)", "raw", "surface", 1),
        Candidate("surface_linear_rigid", "Linear surface (rigid removed)", "rigid", "surface", 1),
        Candidate("surface_huber_rigid", "Huber surface (rigid removed)", "rigid", "surface", 1, robust=True),
    ]
    for count in config["pca_components"]:
        result.append(Candidate(f"pca_linear_{count}_rigid", f"PCA-{count} linear (rigid removed)", "rigid", "pca", 1, int(count)))
    for count in config["quadratic_components"]:
        result.append(Candidate(f"pca_quadratic_{count}_rigid", f"PCA-{count} quadratic (rigid removed)", "rigid", "pca", 2, int(count), ridge=1.0e-3))
    return result


def kabsch_align(moving: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigidly align ``moving`` to ``target`` without scale or reflection."""
    moving = np.asarray(moving, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (moving - moving_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    translation = target_center - moving_center @ rotation
    aligned = moving @ rotation + translation
    return aligned, rotation, translation


def generalized_rigid_alignment(meshes: np.ndarray, iterations: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Subject-level generalized Procrustes using rigid transforms only."""
    meshes = np.asarray(meshes, dtype=np.float64)
    template = meshes.mean(axis=0)
    aligned = meshes.copy()
    for _ in range(iterations):
        aligned = np.stack([kabsch_align(mesh, template)[0] for mesh in meshes])
        updated = aligned.mean(axis=0)
        if np.sqrt(np.mean((updated - template) ** 2)) < 1.0e-9:
            template = updated
            break
        template = updated
    aligned = np.stack([kabsch_align(mesh, template)[0] for mesh in meshes])
    return aligned, aligned.mean(axis=0)


def align_training_and_holdout(train: np.ndarray, holdout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the rigid template on training visits, then align the held-out visit."""
    aligned, template = generalized_rigid_alignment(train)
    held_aligned = kabsch_align(holdout, template)[0]
    return aligned, held_aligned


def _weighted_polynomial_fit(
    ages: np.ndarray,
    values: np.ndarray,
    degree: int,
    ridge: float,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    ages = np.asarray(ages, dtype=np.float64)
    center = float(ages.mean())
    centered = ages - center
    design = np.stack([centered**power for power in range(degree + 1)], axis=1)
    weights = np.ones(len(ages), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    root = np.sqrt(np.maximum(weights, EPS))[:, None]
    lhs = (design * root).T @ (design * root)
    penalty = np.eye(degree + 1, dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    lhs += penalty
    rhs = (design * root).T @ (values * root)
    coefficients = np.linalg.pinv(lhs, rcond=1.0e-12) @ rhs
    return center, coefficients


def fit_trajectory(
    ages: np.ndarray,
    values: np.ndarray,
    candidate: Candidate,
    pca_mean: np.ndarray | None = None,
    pca_components: np.ndarray | None = None,
    huber_delta: float = 1.345,
) -> Trajectory:
    ages = np.asarray(ages, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(ages) < candidate.minimum_fit_visits:
        raise ValueError(f"{candidate.name} needs at least {candidate.minimum_fit_visits} visits")
    if not np.all(np.diff(np.sort(ages)) > 0.0):
        raise ValueError("Ages must be unique within subject")
    output_shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    mean_used = None
    components_used = None
    compact = flat
    if candidate.basis == "pca":
        if pca_mean is None or pca_components is None or candidate.components is None:
            raise ValueError("PCA candidate requires a frozen mean and components")
        components_used = np.asarray(pca_components[: candidate.components], dtype=np.float64)
        mean_used = np.asarray(pca_mean, dtype=np.float64).reshape(-1)
        compact = (flat - mean_used) @ components_used.T
    center, coefficients = _weighted_polynomial_fit(ages, compact, candidate.degree, candidate.ridge)
    if candidate.robust:
        weights = np.ones(len(ages), dtype=np.float64)
        for _ in range(30):
            centered = ages - center
            design = np.stack([centered**power for power in range(candidate.degree + 1)], axis=1)
            residual = np.sqrt(np.mean((compact - design @ coefficients) ** 2, axis=1))
            median = float(np.median(residual))
            scale = 1.4826 * float(np.median(np.abs(residual - median))) + 1.0e-10
            cutoff = huber_delta * scale
            new_weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual - median), 1.0e-12))
            new_center, new_coefficients = _weighted_polynomial_fit(
                ages, compact, candidate.degree, candidate.ridge, new_weights
            )
            if np.max(np.abs(new_weights - weights)) < 1.0e-6:
                center, coefficients = new_center, new_coefficients
                break
            center, coefficients, weights = new_center, new_coefficients, new_weights
    return Trajectory(candidate, center, coefficients, output_shape, mean_used, components_used)


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    signed = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0
    return abs(float(signed))


def vertex_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_area = np.linalg.norm(cross, axis=1) * 0.5
    normals = np.zeros_like(vertices)
    np.add.at(normals, faces[:, 0], cross)
    np.add.at(normals, faces[:, 1], cross)
    np.add.at(normals, faces[:, 2], cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), EPS)
    signed = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0
    if signed < 0.0:
        normals *= -1.0
    area = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(area, faces.reshape(-1), np.repeat(face_area / 3.0, 3))
    return normals, area, abs(float(signed))


def prediction_metrics(predicted: np.ndarray, observed: np.ndarray, faces: np.ndarray, surface_distances: bool = True) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    distance = np.linalg.norm(predicted - observed, axis=1)
    result = {
        "vertex_rmse_mm": float(np.sqrt(np.mean(distance**2))),
        "mean_vertex_error_mm": float(np.mean(distance)),
        "volume_abs_error_mm3": abs(mesh_volume(predicted, faces) - mesh_volume(observed, faces)),
        "log_volume_abs_error": abs(np.log(max(mesh_volume(predicted, faces), EPS)) - np.log(max(mesh_volume(observed, faces), EPS))),
    }
    if surface_distances:
        left = cKDTree(predicted).query(observed, k=1)[0]
        right = cKDTree(observed).query(predicted, k=1)[0]
        joined = np.concatenate([left, right])
        result["assd_mm"] = float((left.mean() + right.mean()) / 2.0)
        result["hd95_mm"] = float(np.quantile(joined, 0.95))
    return result


def vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), EPS))


def weighted_field_metrics(model: np.ndarray, reference: np.ndarray, area: np.ndarray) -> dict[str, float]:
    model = np.asarray(model, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    weight = area / max(float(area.sum()), EPS)
    model_centered = model - np.sum(weight * model)
    reference_centered = reference - np.sum(weight * reference)
    pearson = np.sum(weight * model_centered * reference_centered) / max(
        np.sqrt(np.sum(weight * model_centered**2) * np.sum(weight * reference_centered**2)), EPS
    )
    model_rms = np.sqrt(np.sum(weight * model**2))
    reference_rms = np.sqrt(np.sum(weight * reference**2))
    cutoff_reference = np.quantile(np.abs(reference), 0.8)
    cutoff_model = np.quantile(np.abs(model), 0.8)
    hot_reference = np.abs(reference) >= cutoff_reference
    hot_model = np.abs(model) >= cutoff_model
    return {
        "normal_mae_mm_per_year": float(np.sum(weight * np.abs(model - reference))),
        "normal_rmse_mm_per_year": float(np.sqrt(np.sum(weight * (model - reference) ** 2))),
        "normal_pearson": float(pearson),
        "normal_sign_agreement": float(np.sum(weight * (np.sign(model) == np.sign(reference)))),
        "hotspot_dice": float(2 * np.logical_and(hot_reference, hot_model).sum() / max(hot_reference.sum() + hot_model.sum(), 1)),
        "model_to_reference_speed_ratio": float(model_rms / max(reference_rms, EPS)),
    }


def bootstrap_mean(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def subject_group_mean(fields: np.ndarray, subjects: np.ndarray, diagnoses: np.ndarray, group: str) -> np.ndarray:
    subject_fields = []
    for subject in np.unique(subjects[diagnoses == group]):
        subject_fields.append(fields[(subjects == subject) & (diagnoses == group)].mean(axis=0))
    if not subject_fields:
        raise ValueError(f"No fields for diagnosis {group}")
    return np.mean(np.stack(subject_fields), axis=0)


def consecutive_cosines(vectors: Iterable[np.ndarray]) -> list[float]:
    vectors = list(vectors)
    return [vector_cosine(vectors[index], vectors[index + 1]) for index in range(len(vectors) - 1)]
