#!/usr/bin/env python3
"""Voxel cognition-estimator and feedback utilities for PCA128 BrainODE.

The estimator consumes a solid voxelization of the anatomical surface.  It
does not consume age, subject identifier, diagnosis metadata, or the latent
vector.  This keeps the estimator comparable to the shape-based 3-D CNN in
BrainODE and prevents metadata shortcuts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn

import common as C
from models import rk4_step


@dataclass(frozen=True)
class VoxelGrid:
    """Shared physical-mm grid whose entries represent voxel centers."""

    resolution: int
    pitch_mm: float
    minimum_center_mm: tuple[float, float, float]
    padding_voxels: int

    @classmethod
    def from_training_vertices(
        cls,
        vertices_mm: np.ndarray,
        resolution: int,
        padding_voxels: int,
    ) -> "VoxelGrid":
        if int(resolution) < 16:
            raise ValueError("Voxel resolution must be at least 16")
        if int(padding_voxels) < 1 or 2 * int(padding_voxels) >= int(resolution) - 2:
            raise ValueError("Invalid voxel padding")
        vertices = np.asarray(vertices_mm, dtype=np.float64)
        if vertices.ndim != 3 or vertices.shape[2] != 3 or not np.isfinite(vertices).all():
            raise ValueError(f"Expected finite vertices [N,V,3], got {vertices.shape}")
        lower = vertices.min(axis=(0, 1))
        upper = vertices.max(axis=(0, 1))
        center = 0.5 * (lower + upper)
        usable_intervals = int(resolution) - 1 - 2 * int(padding_voxels)
        pitch = float(np.max(upper - lower) / float(usable_intervals))
        if not math.isfinite(pitch) or pitch <= 0.0:
            raise ValueError("Degenerate training bounds")
        minimum_center = center - 0.5 * float(int(resolution) - 1) * pitch
        return cls(
            resolution=int(resolution),
            pitch_mm=pitch,
            minimum_center_mm=tuple(float(value) for value in minimum_center),
            padding_voxels=int(padding_voxels),
        )

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "VoxelGrid":
        return cls(
            resolution=int(value["resolution"]),
            pitch_mm=float(value["pitch_mm"]),
            minimum_center_mm=tuple(float(item) for item in value["minimum_center_mm"]),
            padding_voxels=int(value["padding_voxels"]),
        )

    def mapping(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def maximum_center_mm(self) -> np.ndarray:
        return np.asarray(self.minimum_center_mm) + (self.resolution - 1) * self.pitch_mm

    @property
    def voxel_count(self) -> int:
        return int(self.resolution**3)

    def voxelize(self, vertices_mm: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        vertices = np.asarray(vertices_mm, dtype=np.float64)
        triangles = np.asarray(faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected vertices [V,3], got {vertices.shape}")
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False, validate=False)
        if not mesh.is_watertight:
            raise ValueError("Solid voxelization requires a watertight mesh")
        local = mesh.voxelized(pitch=self.pitch_mm).fill()
        points = np.asarray(local.points, dtype=np.float64)
        minimum = np.asarray(self.minimum_center_mm, dtype=np.float64)
        indices = np.rint((points - minimum) / self.pitch_mm).astype(np.int64)
        valid = np.all((indices >= 0) & (indices < self.resolution), axis=1)
        clipped = int(np.count_nonzero(~valid))
        mask = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.uint8)
        indices = indices[valid]
        if len(indices):
            mask[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        mesh_volume = float(abs(mesh.volume))
        voxel_volume = float(mask.sum()) * self.pitch_mm**3
        return mask, {
            "occupied_voxels": float(mask.sum()),
            "clipped_local_voxels": float(clipped),
            "mesh_volume_mm3": mesh_volume,
            "voxel_volume_mm3": voxel_volume,
            "voxel_to_mesh_volume_ratio": voxel_volume / max(mesh_volume, 1.0e-8),
        }


def pack_masks(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.ndim != 4:
        raise ValueError(f"Expected masks [N,R,R,R], got {masks.shape}")
    return np.packbits(masks.reshape(masks.shape[0], -1), axis=1, bitorder="little")


def unpack_masks(packed: np.ndarray, resolution: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim != 2:
        raise ValueError(f"Expected packed masks [N,K], got {packed.shape}")
    count = int(resolution) ** 3
    expected = (count + 7) // 8
    if packed.shape[1] != expected:
        raise ValueError(f"Packed mask width {packed.shape[1]} != expected {expected}")
    flat = np.unpackbits(packed, axis=1, count=count, bitorder="little")
    return flat.reshape(len(packed), int(resolution), int(resolution), int(resolution))


def voxel_archive_path(registry: dict[str, Any], split: str, resolution: int) -> Path:
    if split not in C.SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return C.output_root(registry) / "cognition_voxels" / "pca128" / f"voxel{int(resolution)}" / f"{split}.npz"


def load_voxel_archive(
    registry: dict[str, Any],
    split: str,
    resolution: int,
    expected_archive: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], VoxelGrid]:
    path = voxel_archive_path(registry, split, resolution)
    if not path.is_file():
        raise FileNotFoundError(f"Voxel archive is missing: {path}; run prepare_cognition_voxels.py")
    with np.load(path, allow_pickle=False) as loaded:
        archive = {key: loaded[key] for key in loaded.files}
    required = {
        "masks_packed",
        "visit_scan_ids",
        "visit_subject_ids",
        "visit_label_ad",
        "visit_diagnoses",
        "resolution",
        "pitch_mm",
        "minimum_center_mm",
        "padding_voxels",
        "occupied_voxels",
        "voxel_to_mesh_volume_ratio",
    }
    missing = sorted(required.difference(archive))
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    grid = VoxelGrid(
        resolution=int(archive["resolution"].item()),
        pitch_mm=float(archive["pitch_mm"].item()),
        minimum_center_mm=tuple(float(value) for value in archive["minimum_center_mm"]),
        padding_voxels=int(archive["padding_voxels"].item()),
    )
    if grid.resolution != int(resolution):
        raise ValueError(f"Voxel archive resolution mismatch: {grid.resolution} != {resolution}")
    rows = len(archive["visit_scan_ids"])
    if any(len(archive[key]) != rows for key in ("visit_subject_ids", "visit_label_ad", "visit_diagnoses")):
        raise ValueError(f"Voxel metadata length mismatch in {path}")
    masks = unpack_masks(archive["masks_packed"], grid.resolution)
    if not np.array_equal(masks.sum(axis=(1, 2, 3)), archive["occupied_voxels"]):
        raise ValueError(f"Packed mask occupancy mismatch in {path}")
    archive["masks"] = masks
    if expected_archive is not None:
        for key in ("visit_scan_ids", "visit_subject_ids", "visit_label_ad", "visit_diagnoses"):
            if not np.array_equal(archive[key].astype(str), expected_archive[key].astype(str)):
                raise ValueError(f"Voxel/PCA archive mismatch for {split}:{key}")
    return archive, grid


class VoxelCognitionCNN(nn.Module):
    """Small shape-only 3-D CNN returning an AD-like anatomy logit."""

    input_contract = "solid binary anatomical mask only; no age/diagnosis/latent metadata"

    def __init__(self, base_channels: int = 12, dropout: float = 0.15) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            groups = min(8, out_channels)
            while out_channels % groups:
                groups -= 1
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(groups, out_channels),
                nn.GELU(),
                nn.MaxPool3d(2),
            )

        width = int(base_channels)
        self.features = nn.Sequential(
            block(1, width),
            block(width, 2 * width),
            block(2 * width, 4 * width),
            block(4 * width, 8 * width),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(8 * width, 1),
        )

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        if masks.ndim == 4:
            masks = masks.unsqueeze(1)
        if masks.ndim != 5 or masks.shape[1] != 1:
            raise ValueError(f"Expected voxel masks [B,1,R,R,R], got {tuple(masks.shape)}")
        return self.classifier(self.features(masks.float())).reshape(-1)


def subject_balanced_weights(subject_ids: Iterable[str], labels: np.ndarray) -> np.ndarray:
    subjects = np.asarray(list(subject_ids)).astype(str)
    labels = np.asarray(labels, dtype=np.int64)
    if len(subjects) != len(labels) or set(np.unique(labels)).difference({0, 1}):
        raise ValueError("Invalid subject/label arrays")
    unique, counts = np.unique(subjects, return_counts=True)
    visits = dict(zip(unique.tolist(), counts.tolist()))
    subject_labels = {subject: int(labels[np.flatnonzero(subjects == subject)[0]]) for subject in unique}
    if any(np.any(labels[subjects == subject] != label) for subject, label in subject_labels.items()):
        raise ValueError("A subject has inconsistent diagnosis labels")
    class_subjects = {label: sum(value == label for value in subject_labels.values()) for label in (0, 1)}
    if min(class_subjects.values()) == 0:
        raise ValueError("Both CN and AD subjects are required")
    weights = np.asarray([
        1.0 / float(visits[subject]) / float(class_subjects[int(label)])
        for subject, label in zip(subjects, labels)
    ], dtype=np.float32)
    return weights / weights.mean()


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = max(len(labels), 1)
    value = 0.0
    for index in range(int(bins)):
        selected = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if index == int(bins) - 1 else probabilities < edges[index + 1]
        )
        if np.any(selected):
            value += float(np.count_nonzero(selected)) / total * abs(
                float(probabilities[selected].mean()) - float(labels[selected].mean())
            )
    return float(value)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("Invalid labels/probabilities")
    predictions = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "rows": float(len(labels)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(np.mean(predictions == labels)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece_10": expected_calibration_error(labels, probabilities, bins=10),
        "probability_mean_cn": float(probabilities[labels == 0].mean()),
        "probability_mean_ad": float(probabilities[labels == 1].mean()),
    }


def scan_and_subject_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    subject_ids: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    subject_ids = np.asarray(subject_ids).astype(str)
    unique = np.unique(subject_ids)
    subject_labels, subject_probabilities = [], []
    for subject in unique:
        selected = subject_ids == subject
        if len(np.unique(labels[selected])) != 1:
            raise ValueError(f"Inconsistent labels for {subject}")
        subject_labels.append(int(labels[selected][0]))
        subject_probabilities.append(float(probabilities[selected].mean()))
    return {
        "scan": binary_metrics(labels, probabilities),
        "subject": binary_metrics(np.asarray(subject_labels), np.asarray(subject_probabilities)),
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive calibration temperature on validation data only."""
    logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
    labels_tensor = torch.as_tensor(labels, dtype=torch.float64)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=80, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).clamp(0.05, 20.0))


class CognitionFeedbackTransport(nn.Module):
    """BrainODE transport with shape-derived condition refreshed each RK4 step."""

    def __init__(
        self,
        function: nn.Module,
        geometry: C.FrozenGeometry,
        estimator: VoxelCognitionCNN,
        grid: VoxelGrid,
        substeps: int,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.function = function
        self.geometry = geometry
        self.estimator = estimator
        self.grid = grid
        self.substeps = int(substeps)
        self.temperature = float(temperature)
        if self.substeps < 1 or self.temperature <= 0.0:
            raise ValueError("Invalid feedback integrator settings")

    @torch.no_grad()
    def estimate_condition(self, latent: torch.Tensor) -> torch.Tensor:
        vertices = self.geometry.vertices(latent).detach().cpu().numpy()
        faces = self.geometry.faces.detach().cpu().numpy()
        masks = np.stack([self.grid.voxelize(item, faces)[0] for item in vertices])
        tensor = torch.from_numpy(masks).to(next(self.estimator.parameters()).device, dtype=torch.float32)
        logits = self.estimator(tensor)
        return torch.sigmoid(logits / self.temperature).to(latent.device, dtype=latent.dtype)

    @torch.no_grad()
    def transport(self, latent, source_time, target_time, condition=None, context=None, context_time=None):
        del condition, context, context_time
        current = latent
        current_time = source_time.reshape(-1).to(latent)
        delta = (target_time.reshape(-1).to(latent) - current_time) / float(self.substeps)
        for _ in range(self.substeps):
            cognition = self.estimate_condition(current)
            current = rk4_step(self.function, current, current_time, delta, cognition)
            current_time = current_time + delta
        return current

    @torch.no_grad()
    def condition_path(self, latent: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.shape[0] != 1 or times.ndim != 1:
            raise ValueError("condition_path expects one case and one time vector")
        states = [latent]
        conditions = [self.estimate_condition(latent)]
        current = latent
        for index in range(1, len(times)):
            current = self.transport(current, times[index - 1 : index], times[index : index + 1])
            states.append(current)
            conditions.append(self.estimate_condition(current))
        return torch.cat(states), torch.cat(conditions)
