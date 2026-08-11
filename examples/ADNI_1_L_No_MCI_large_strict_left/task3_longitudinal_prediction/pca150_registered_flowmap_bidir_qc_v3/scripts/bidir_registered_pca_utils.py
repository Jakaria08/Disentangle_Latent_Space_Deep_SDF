#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / ".git").exists() or (path / "deep_sdf").is_dir():
            return path
    raise RuntimeError(f"Could not locate repository root from {start}")


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
TASK_PARENT = EXPERIMENT_DIR.parent
TASK_DIR = TASK_PARENT / "brainode_pca150_qc_stable"
REPO_ROOT = find_repo_root(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EXPERIMENT_NAME = "pca150_registered_flowmap_bidir_qc_v3"
SPLITS = ("train", "val", "test")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    return config


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class PairRecord:
    source_index: int
    target_index: int
    intermediate_index: int
    subject_id: str
    diagnosis: str
    label_ad: int
    source_scan_id: str
    target_scan_id: str
    source_visit_order: int
    target_visit_order: int
    source_age_norm: float
    target_age_norm: float
    source_age_years: float
    target_age_years: float
    delta_years: float
    abs_gap_years: float
    pair_type: str
    gap_bin: str
    direction: str


def experiment_root(name: str = DEFAULT_EXPERIMENT_NAME) -> Path:
    if str(name) == DEFAULT_EXPERIMENT_NAME:
        return EXPERIMENT_DIR
    return TASK_PARENT / str(name)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def split_archive_path(split: str) -> Path:
    return TASK_DIR / "dataset" / f"{split}_subject_sequences.npz"


def load_split_archive(split: str) -> dict[str, np.ndarray]:
    return load_npz(split_archive_path(split))


def load_all_archive() -> dict[str, np.ndarray]:
    return load_npz(TASK_DIR / "dataset" / "all_subject_sequences.npz")


def pca_latents(archive: dict[str, np.ndarray], components: int) -> np.ndarray:
    key = f"visit_pca_{int(components)}"
    if key in archive:
        return archive[key].astype(np.float32)
    if int(components) <= 256 and "visit_pca_256" in archive:
        return archive["visit_pca_256"][:, : int(components)].astype(np.float32)
    raise KeyError(f"Archive does not contain PCA coefficients for {components} components.")


def coefficient_stats(
    archive: dict[str, np.ndarray],
    components: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean_key = f"train_coefficient_mean_{int(components)}"
    std_key = f"train_coefficient_std_{int(components)}"
    if mean_key in archive and std_key in archive:
        mean = archive[mean_key].astype(np.float32)
        std = archive[std_key].astype(np.float32)
    else:
        mean = archive["train_coefficient_mean_256"][: int(components)].astype(np.float32)
        std = archive["train_coefficient_std_256"][: int(components)].astype(np.float32)
    return mean, np.maximum(std, np.float32(1.0e-6))


def load_pca_model(
    config_path: str | Path,
    components: int,
) -> tuple[dict[str, Any], Path, np.ndarray, np.ndarray, np.ndarray]:
    config = load_config(config_path)
    pca_model_dir = resolve_repo_path(config["task2"]["pca_model_dir"])
    mean_flat = np.load(pca_model_dir / "mean.npy").astype(np.float32)
    pca_components = np.load(pca_model_dir / "components_256.npy").astype(np.float32)[
        : int(components)
    ]
    faces = np.load(pca_model_dir / "faces.npy").astype(np.int64)
    return config, pca_model_dir, mean_flat, pca_components, faces


def decode_pca_np(
    coefficients: np.ndarray,
    mean_flat: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=np.float32)
    flat = coeff.reshape(-1, coeff.shape[-1]) @ components.astype(np.float32)
    flat = flat + mean_flat.astype(np.float32)
    return flat.reshape(*coeff.shape[:-1], -1, 3)


def decode_pca_torch(
    coefficients: torch.Tensor,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    flat = coefficients.reshape(-1, coefficients.shape[-1]) @ components
    flat = flat + mean_flat
    return flat.reshape(*coefficients.shape[:-1], -1, 3)


def mesh_volume_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim == 2:
        vertices = vertices[None, ...]
    f = faces.astype(np.int64)
    v0 = vertices[:, f[:, 0], :]
    v1 = vertices[:, f[:, 1], :]
    v2 = vertices[:, f[:, 2], :]
    signed = np.einsum("bfi,bfi->bf", v0, np.cross(v1, v2))
    return np.abs(signed.sum(axis=1) / 6.0)


def mesh_area_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim == 2:
        vertices = vertices[None, ...]
    f = faces.astype(np.int64)
    v0 = vertices[:, f[:, 0], :]
    v1 = vertices[:, f[:, 1], :]
    v2 = vertices[:, f[:, 2], :]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=2).sum(axis=1)


def mesh_volume_torch(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    f = faces.to(device=vertices.device, dtype=torch.long)
    v0 = vertices[:, f[:, 0], :]
    v1 = vertices[:, f[:, 1], :]
    v2 = vertices[:, f[:, 2], :]
    signed = torch.sum(v0 * torch.cross(v1, v2, dim=2), dim=2).sum(dim=1) / 6.0
    return torch.clamp(torch.abs(signed), min=1.0e-8)


def vertex_normals_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    f = faces.astype(np.int64)
    normals = np.zeros_like(vertices, dtype=np.float64)
    v0 = vertices[f[:, 0]]
    v1 = vertices[f[:, 1]]
    v2 = vertices[f[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for corner in range(3):
        np.add.at(normals, f[:, corner], face_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(norm, 1.0e-12)).astype(np.float32)


def gap_bin(delta_years: float, pair_type: str) -> str:
    if pair_type == "adjacent":
        return "adjacent"
    if abs(float(delta_years)) <= 2.0:
        return "short"
    return "long"


def build_pair_records(
    archive: dict[str, np.ndarray],
    *,
    max_gap_years: float = 0.0,
    pair_type: str = "all",
    include_backward: bool = False,
) -> list[PairRecord]:
    offsets = archive["subject_visit_offsets"]
    records: list[PairRecord] = []
    for subject_index in range(len(offsets) - 1):
        start = int(offsets[subject_index])
        end = int(offsets[subject_index + 1])
        for source_index in range(start, end - 1):
            for target_index in range(source_index + 1, end):
                source_order = int(archive["visit_orders"][source_index])
                target_order = int(archive["visit_orders"][target_index])
                current_pair_type = (
                    "adjacent" if target_order - source_order == 1 else "nonadjacent"
                )
                if pair_type != "all" and current_pair_type != pair_type:
                    continue
                source_age_years = float(archive["visit_continuous_age_years"][source_index])
                target_age_years = float(archive["visit_continuous_age_years"][target_index])
                delta_years = target_age_years - source_age_years
                if delta_years <= 1.0e-8:
                    continue
                if float(max_gap_years) > 0.0 and delta_years > float(max_gap_years):
                    continue
                intermediate_index = -1
                if target_index - source_index > 1:
                    intermediate_index = source_index + (target_index - source_index) // 2
                forward = PairRecord(
                        source_index=source_index,
                        target_index=target_index,
                        intermediate_index=intermediate_index,
                        subject_id=str(archive["visit_subject_ids"][source_index]),
                        diagnosis=str(archive["visit_diagnoses"][source_index]),
                        label_ad=int(archive["visit_label_ad"][source_index]),
                        source_scan_id=str(archive["visit_scan_ids"][source_index]),
                        target_scan_id=str(archive["visit_scan_ids"][target_index]),
                        source_visit_order=source_order,
                        target_visit_order=target_order,
                        source_age_norm=float(archive["visit_continuous_age_norm"][source_index]),
                        target_age_norm=float(archive["visit_continuous_age_norm"][target_index]),
                        source_age_years=source_age_years,
                        target_age_years=target_age_years,
                        delta_years=delta_years,
                        abs_gap_years=abs(delta_years),
                        pair_type=current_pair_type,
                        gap_bin=gap_bin(delta_years, current_pair_type),
                        direction="forward",
                )
                records.append(forward)
                if include_backward:
                    backward_intermediate_index = intermediate_index
                    records.append(
                        PairRecord(
                            source_index=target_index,
                            target_index=source_index,
                            intermediate_index=backward_intermediate_index,
                            subject_id=str(archive["visit_subject_ids"][source_index]),
                            diagnosis=str(archive["visit_diagnoses"][source_index]),
                            label_ad=int(archive["visit_label_ad"][source_index]),
                            source_scan_id=str(archive["visit_scan_ids"][target_index]),
                            target_scan_id=str(archive["visit_scan_ids"][source_index]),
                            source_visit_order=target_order,
                            target_visit_order=source_order,
                            source_age_norm=float(archive["visit_continuous_age_norm"][target_index]),
                            target_age_norm=float(archive["visit_continuous_age_norm"][source_index]),
                            source_age_years=target_age_years,
                            target_age_years=source_age_years,
                            delta_years=-delta_years,
                            abs_gap_years=abs(delta_years),
                            pair_type=current_pair_type,
                            gap_bin=gap_bin(delta_years, current_pair_type),
                            direction="backward",
                        )
                    )
    return records


def summarize_pair_records(records: Iterable[PairRecord]) -> dict[str, Any]:
    records = list(records)

    def count_by(values: Iterable[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return counts

    return {
        "pair_count": len(records),
        "pair_count_by_diagnosis": count_by(record.diagnosis for record in records),
        "pair_count_by_pair_type": count_by(record.pair_type for record in records),
        "pair_count_by_gap_bin": count_by(record.gap_bin for record in records),
        "pair_count_by_direction": count_by(record.direction for record in records),
    }


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")
