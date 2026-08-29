#!/usr/bin/env python3
"""Prepared fixed-topology longitudinal surface data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import common as C


def split_cache_path(split: str, root: Path | None = None) -> Path:
    if split not in C.SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return C.output_root(root) / "cache" / f"{split}_surfaces.npz"


def statistics_path(root: Path | None = None) -> Path:
    return C.output_root(root) / "cache" / "training_statistics.pt"


@dataclass
class PreparedSplit:
    split: str
    vertices: torch.Tensor
    ages: torch.Tensor
    years_from_baseline: torch.Tensor
    labels: torch.Tensor
    volumes: torch.Tensor
    velocity_reference: torch.Tensor
    velocity_reference_weight: torch.Tensor
    scan_ids: np.ndarray
    subject_ids: np.ndarray
    diagnoses: np.ndarray
    visit_orders: np.ndarray
    subject_visit_offsets: np.ndarray

    def to(self, device: torch.device | str) -> "PreparedSplit":
        return PreparedSplit(
            split=self.split,
            vertices=self.vertices.to(device),
            ages=self.ages.to(device),
            years_from_baseline=self.years_from_baseline.to(device),
            labels=self.labels.to(device),
            volumes=self.volumes.to(device),
            velocity_reference=self.velocity_reference.to(device),
            velocity_reference_weight=self.velocity_reference_weight.to(device),
            scan_ids=self.scan_ids,
            subject_ids=self.subject_ids,
            diagnoses=self.diagnoses,
            visit_orders=self.visit_orders,
            subject_visit_offsets=self.subject_visit_offsets,
        )

    def pair_batch(self, rows: list[C.PairRow], device: torch.device | None = None) -> dict[str, torch.Tensor]:
        source = torch.tensor([row.source for row in rows], dtype=torch.long, device=self.vertices.device)
        target = torch.tensor([row.target for row in rows], dtype=torch.long, device=self.vertices.device)
        middle = torch.tensor([row.intermediate for row in rows], dtype=torch.long, device=self.vertices.device)
        output = {
            "source_index": source,
            "target_index": target,
            "intermediate_index": middle,
            "source": self.vertices[source],
            "target": self.vertices[target],
            "source_age": self.ages[source],
            "target_age": self.ages[target],
            "source_years": self.years_from_baseline[source],
            "target_years": self.years_from_baseline[target],
            "label": self.labels[source],
            "source_volume": self.volumes[source],
            "target_volume": self.volumes[target],
            "velocity_reference": self.velocity_reference[source],
            "velocity_reference_weight": self.velocity_reference_weight[source],
            "target_velocity_reference": self.velocity_reference[target],
            "target_velocity_reference_weight": self.velocity_reference_weight[target],
        }
        if device is not None:
            output = {name: value.to(device) for name, value in output.items()}
        return output


def load_split(split: str, root: Path | None = None, device: torch.device | str = "cpu") -> PreparedSplit:
    path = split_cache_path(split, root)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared split missing: {path}; run prepare_data.py")
    with np.load(path, allow_pickle=False) as archive:
        prepared = PreparedSplit(
            split=split,
            vertices=torch.from_numpy(archive["vertices_mm"].astype(np.float32)),
            ages=torch.from_numpy(archive["age_years"].astype(np.float32)),
            years_from_baseline=torch.from_numpy(archive["years_from_baseline"].astype(np.float32)),
            labels=torch.from_numpy(archive["label_ad"].astype(np.float32)),
            volumes=torch.from_numpy(archive["volume_mm3"].astype(np.float32)),
            velocity_reference=torch.from_numpy(archive["velocity_reference_mm_per_year"].astype(np.float32)),
            velocity_reference_weight=torch.from_numpy(archive["velocity_reference_weight"].astype(np.float32)),
            scan_ids=archive["scan_ids"].astype(str),
            subject_ids=archive["subject_ids"].astype(str),
            diagnoses=archive["diagnoses"].astype(str),
            visit_orders=archive["visit_orders"].astype(np.int64),
            subject_visit_offsets=archive["subject_visit_offsets"].astype(np.int64),
        )
    if prepared.vertices.shape[1:] != (C.VERTEX_COUNT, 3):
        raise ValueError(f"Prepared vertices have wrong shape: {tuple(prepared.vertices.shape)}")
    if not torch.isfinite(prepared.vertices).all():
        raise ValueError(f"Non-finite vertices in {path}")
    return prepared.to(device)


def load_statistics(root: Path | None = None) -> dict[str, Any]:
    path = statistics_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Training statistics missing: {path}; run prepare_data.py")
    return torch.load(path, map_location="cpu", weights_only=False)


def verify_split_isolation(splits: dict[str, PreparedSplit]) -> None:
    seen_subjects: set[str] = set()
    seen_scans: set[str] = set()
    for name in C.SPLITS:
        if name not in splits:
            continue
        subjects = set(splits[name].subject_ids.astype(str))
        scans = set(splits[name].scan_ids.astype(str))
        if subjects & seen_subjects:
            raise ValueError(f"Subject leakage involving split {name}")
        if scans & seen_scans:
            raise ValueError(f"Scan leakage involving split {name}")
        seen_subjects.update(subjects)
        seen_scans.update(scans)
