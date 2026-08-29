#!/usr/bin/env python3
"""Shared data and safety utilities for PCA-conditioned corrective deformation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


TASK_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = TASK_ROOT.parent
REPO_ROOT = TASK_ROOT.parents[2]
DEFAULT_CONFIG = TASK_ROOT / "configs" / "pca128_corrective.json"
BULK_ROOT = Path("/mnt/bulk10tb")

SOURCE_TASK_ROOT = EXPERIMENT_ROOT / "task_spiral_ae_v1"
SOURCE_SCRIPTS = SOURCE_TASK_ROOT / "scripts"
if str(SOURCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_SCRIPTS))

import spiral_common as source_sc  # noqa: E402


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    return config


def require_bulk_path(value: str | Path, description: str = "runtime output") -> Path:
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(BULK_ROOT)
    except ValueError as error:
        raise ValueError(f"{description} must be below {BULK_ROOT}; refusing {path}") from error
    if path == BULK_ROOT:
        raise ValueError(f"{description} cannot be the bulk mount root itself")
    return path


def makedirs(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(path: str | Path, payload) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def atomic_write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    rows = list(rows)
    fieldnames = list(rows[0]) if rows else []
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def atomic_torch_save(path: str | Path, payload) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def choose_device(requested: str | None) -> torch.device:
    value = requested or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {value}, but CUDA is unavailable in this process. "
                "Run in the pytorch_geo environment on a GPU-enabled shell."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {value}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
    return device


@dataclass(frozen=True)
class PCAContract:
    mean: np.ndarray
    components: np.ndarray
    coefficient_mean: np.ndarray
    coefficient_std: np.ndarray
    residual_rms_mm: float
    coordinate_scale_mm: float
    faces: np.ndarray
    rows: list[dict]
    topology_hash: str
    mean_sha256: str
    components_sha256: str

    @property
    def latent_dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def n_vertices(self) -> int:
        return int(self.mean.size // 3)

    def summary(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "n_vertices": self.n_vertices,
            "features": int(self.mean.size),
            "residual_rms_mm": self.residual_rms_mm,
            "coordinate_scale_mm": self.coordinate_scale_mm,
            "topology_hash": self.topology_hash,
            "mean_sha256": self.mean_sha256,
            "components_sha256": self.components_sha256,
            "split_counts": {
                split: sum(row["split"] == split for row in self.rows)
                for split in source_sc.SPLITS
            },
        }


@dataclass
class SplitTensors:
    name: str
    coefficients: torch.Tensor
    vertices_mm: torch.Tensor
    rows: list[dict]

    def __len__(self) -> int:
        return int(self.vertices_mm.shape[0])


def _pca_paths(config: dict) -> tuple[Path, Path]:
    model_dir = resolve_path(config["pca_model_dir"])
    latent_dim = int(config["latent_dim"])
    return model_dir / "mean.npy", model_dir / f"components_{latent_dim}.npy"


def load_pca_contract(config: dict, rows: list[dict] | None = None) -> PCAContract:
    """Load the fixed PCA model and derive normalization from the full train split only."""
    rows = rows if rows is not None else source_sc.read_manifest()
    mean_path, components_path = _pca_paths(config)
    if not mean_path.exists() or not components_path.exists():
        raise FileNotFoundError(f"Missing PCA model files: {mean_path}, {components_path}")

    mean = np.load(mean_path).astype(np.float32).reshape(-1)
    components = np.load(components_path).astype(np.float32)
    if components.shape != (int(config["latent_dim"]), mean.size):
        raise ValueError(
            f"PCA shape mismatch: mean={mean.shape}, components={components.shape}, "
            f"latent_dim={config['latent_dim']}"
        )
    gram = components.astype(np.float64) @ components.astype(np.float64).T
    max_orthogonality_error = float(np.max(np.abs(gram - np.eye(len(components)))))
    if max_orthogonality_error > 5e-4:
        raise ValueError(f"PCA basis is not orthonormal; max error={max_orthogonality_error:.3e}")

    train = source_sc.load_split_vertices("train", rows=rows)
    train_flat = train.reshape(len(train), -1).astype(np.float32)
    if train_flat.shape[1] != mean.size:
        raise ValueError(f"Train feature count {train_flat.shape[1]} != PCA features {mean.size}")
    coefficients = (train_flat - mean) @ components.T
    reconstruction = coefficients @ components + mean
    residual = train_flat - reconstruction
    coefficient_mean = coefficients.mean(axis=0).astype(np.float32)
    coefficient_std = coefficients.std(axis=0).astype(np.float32)
    coefficient_std = np.maximum(coefficient_std, 1e-8)
    residual_rms = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))))
    coordinate_scale = float(
        np.sqrt(np.mean(np.square(train_flat - mean, dtype=np.float64)))
    )
    if not np.isfinite(residual_rms) or residual_rms <= 0.0:
        raise ValueError(f"Invalid train residual RMS: {residual_rms}")
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0.0:
        raise ValueError(f"Invalid coordinate scale: {coordinate_scale}")

    topology_hashes = {row["correspondence_topology_hash"] for row in rows}
    if len(topology_hashes) != 1:
        raise ValueError(f"Expected one topology hash, found {len(topology_hashes)}")
    faces = source_sc.load_faces(rows=rows).astype(np.int64)
    return PCAContract(
        mean=mean,
        components=components,
        coefficient_mean=coefficient_mean,
        coefficient_std=coefficient_std,
        residual_rms_mm=residual_rms,
        coordinate_scale_mm=coordinate_scale,
        faces=faces,
        rows=rows,
        topology_hash=next(iter(topology_hashes)),
        mean_sha256=sha256_file(mean_path),
        components_sha256=sha256_file(components_path),
    )


def load_split_tensors(
    split: str,
    contract: PCAContract,
    device: torch.device,
    limit: int | None = None,
) -> SplitTensors:
    if split not in source_sc.SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    vertices = source_sc.load_split_vertices(split, rows=contract.rows)
    selected_rows = source_sc.split_rows(contract.rows, split)
    if limit is not None:
        vertices = vertices[: int(limit)]
        selected_rows = selected_rows[: int(limit)]
    flat = vertices.reshape(len(vertices), -1).astype(np.float32)
    coefficients = (flat - contract.mean) @ contract.components.T
    return SplitTensors(
        name=split,
        coefficients=torch.from_numpy(coefficients).to(device).contiguous(),
        vertices_mm=torch.from_numpy(vertices.astype(np.float32)).to(device).contiguous(),
        rows=selected_rows,
    )


def pca_reconstruct_numpy(vertices: np.ndarray, contract: PCAContract) -> np.ndarray:
    flat = np.asarray(vertices, dtype=np.float32).reshape(len(vertices), -1)
    coefficients = (flat - contract.mean) @ contract.components.T
    reconstruction = coefficients @ contract.components + contract.mean
    return reconstruction.reshape(vertices.shape)


def split_subject_sets(rows: list[dict]) -> dict[str, set[str]]:
    return {
        split: {
            row.get("subject_id", row.get("scan_id", ""))
            for row in rows
            if row["split"] == split
        }
        for split in source_sc.SPLITS
    }


def assert_subject_disjoint(rows: list[dict]) -> None:
    subjects = split_subject_sets(rows)
    for index, left in enumerate(source_sc.SPLITS):
        for right in source_sc.SPLITS[index + 1 :]:
            overlap = subjects[left].intersection(subjects[right])
            if overlap:
                raise ValueError(
                    f"Subject leakage between {left} and {right}: {sorted(overlap)[:5]}"
                )


def checkpoint_contract_matches(checkpoint: dict, contract: PCAContract) -> None:
    saved = checkpoint.get("data_contract", {})
    current = contract.summary()
    for key in ("latent_dim", "n_vertices", "features", "topology_hash", "mean_sha256", "components_sha256"):
        if saved.get(key) != current.get(key):
            raise ValueError(
                f"Checkpoint data contract mismatch for {key}: "
                f"saved={saved.get(key)!r}, current={current.get(key)!r}"
            )


def run_directory(config: dict, output_dir: str | Path | None, smoke: bool = False) -> Path:
    if output_dir is not None:
        return require_bulk_path(output_dir, "run directory")
    root = require_bulk_path(config["output_root"], "output root")
    group = "smoke" if smoke else "runs"
    name = f"{config['experiment_name']}_s{int(config['training']['seed'])}"
    return require_bulk_path(root / group / name, "run directory")
