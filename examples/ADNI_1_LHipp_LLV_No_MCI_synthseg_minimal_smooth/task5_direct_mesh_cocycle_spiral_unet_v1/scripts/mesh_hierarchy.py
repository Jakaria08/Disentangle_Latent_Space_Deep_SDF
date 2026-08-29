#!/usr/bin/env python3
"""Load the frozen hippocampus hierarchy and construct four-resolution spiral indices."""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np

import common as C


@dataclass
class MeshHierarchy:
    vertices: list[torch.Tensor]
    faces: list[torch.Tensor]
    spirals: list[torch.Tensor]
    dynamic_spirals: list[torch.Tensor]
    down: list[torch.Tensor]
    up: list[torch.Tensor]

    def to(self, device: torch.device | str) -> "MeshHierarchy":
        return MeshHierarchy(
            vertices=[value.to(device) for value in self.vertices],
            faces=[value.to(device) for value in self.faces],
            spirals=[value.to(device) for value in self.spirals],
            dynamic_spirals=[value.to(device) for value in self.dynamic_spirals],
            down=[value.to(device) for value in self.down],
            up=[value.to(device) for value in self.up],
        )

    @property
    def sizes(self) -> list[int]:
        return [int(value.shape[0]) for value in self.vertices]


def _ae_imports() -> tuple[Any, Any]:
    scripts = C.AE_TASK_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from spiral_common import preprocess_spiral, to_sparse

    return preprocess_spiral, to_sparse


def source_transform_path() -> Path:
    return C.AE_BULK_ROOT / "cache" / "transform_2-4-4.pkl"


def hierarchy_cache_path(root: Path | None = None) -> Path:
    return C.output_root(root) / "cache" / "hierarchy_2-4-4_seq13_d1.pt"


def build_hierarchy_payload(
    sequence_length: int = 13,
    dilation: int = 1,
    dynamic_lengths: tuple[int, int, int, int] = (2, 2, 121, 30),
) -> dict[str, Any]:
    path = source_transform_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Required frozen hierarchy is missing: {path}. Run the existing Spiral-AE preparation first."
        )
    with path.open("rb") as handle:
        transform = pickle.load(handle)
    preprocess_spiral, to_sparse = _ae_imports()
    vertices = [torch.as_tensor(value, dtype=torch.float32) for value in transform["vertices"]]
    faces = [torch.from_numpy(np.asarray(value, dtype=np.int64)) for value in transform["face"]]
    if [int(value.shape[0]) for value in vertices] != [2746, 1373, 344, 86]:
        raise ValueError("Frozen hierarchy has unexpected resolutions")
    spirals: list[torch.Tensor] = []
    dynamic: list[torch.Tensor] = []
    for level, (current_vertices, current_faces) in enumerate(zip(transform["vertices"], transform["face"])):
        count = int(current_vertices.shape[0])
        local_length = max(2, min(int(sequence_length), count // max(int(dilation), 1)))
        spirals.append(
            preprocess_spiral(current_faces, local_length, current_vertices, dilation).long().cpu()
        )
        dynamic_length = max(2, min(int(dynamic_lengths[level]), count))
        dynamic.append(
            preprocess_spiral(current_faces, dynamic_length, current_vertices, 1).long().cpu()
        )
    down = [to_sparse(value).coalesce().cpu() for value in transform["down_transform"]]
    up = [to_sparse(value).coalesce().cpu() for value in transform["up_transform"]]
    return {
        "schema_version": 1,
        "source_transform": str(path),
        "sequence_length": int(sequence_length),
        "dilation": int(dilation),
        "dynamic_lengths": list(dynamic_lengths),
        "vertices": vertices,
        "faces": faces,
        "spirals": spirals,
        "dynamic_spirals": dynamic,
        "down": down,
        "up": up,
    }


def save_hierarchy(root: Path | None = None, overwrite: bool = False) -> Path:
    path = hierarchy_cache_path(root)
    if path.exists() and not overwrite:
        return path
    payload = build_hierarchy_payload()
    C.atomic_torch_save(path, payload)
    return path


def load_hierarchy(root: Path | None = None, device: torch.device | str = "cpu") -> MeshHierarchy:
    path = hierarchy_cache_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared hierarchy missing: {path}; run prepare_data.py")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hierarchy = MeshHierarchy(
        vertices=payload["vertices"],
        faces=payload["faces"],
        spirals=payload["spirals"],
        dynamic_spirals=payload["dynamic_spirals"],
        down=payload["down"],
        up=payload["up"],
    )
    if hierarchy.sizes != [2746, 1373, 344, 86]:
        raise ValueError(f"Prepared hierarchy resolution mismatch: {hierarchy.sizes}")
    return hierarchy.to(device)
