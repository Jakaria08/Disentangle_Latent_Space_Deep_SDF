#!/usr/bin/env python3
"""Shared utilities for structure-agnostic shared-grid INR experiments."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KIND_OFF_SURFACE = 0
KIND_SURFACE = 1
KIND_OFFSET = 2


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = load_json(config_path)
    config["_config_path"] = str(config_path)
    return config


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = resolve_repo_path(path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"scan_id", "subject_id", "split", "mesh_path", "sdf_npz_path"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    for row in rows:
        row["mesh_path"] = str(resolve_repo_path(row["mesh_path"]))
        row["sdf_npz_path"] = str(resolve_repo_path(row["sdf_npz_path"]))
    split_order = {"train": 0, "val": 1, "test": 2}
    return sorted(
        rows,
        key=lambda row: (
            split_order.get(row["split"], 99),
            row["subject_id"],
            float(row.get("visit_order", 0) or 0),
            row["scan_id"],
        ),
    )


def rows_for_split(rows: Iterable[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


def select_stratified_rows(
    rows: list[dict[str, str]], count: int, seed: int
) -> list[dict[str, str]]:
    """Deterministically sample scans with diagnosis balance and subject coverage."""
    if count >= len(rows):
        return list(rows)
    rng = np.random.default_rng(seed)
    diagnoses = sorted({row.get("diagnosis", "") for row in rows})
    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    selected_subjects: set[str] = set()
    pools: dict[str, list[dict[str, str]]] = {}
    for diagnosis in diagnoses:
        pool = [row for row in rows if row.get("diagnosis", "") == diagnosis]
        order = rng.permutation(len(pool))
        pools[diagnosis] = [pool[int(index)] for index in order]
    while len(selected) < count:
        changed = False
        for require_new_subject in (True, False):
            for diagnosis in diagnoses:
                match = next(
                    (
                        row
                        for row in pools[diagnosis]
                        if row["scan_id"] not in selected_ids
                        and (
                            not require_new_subject
                            or row["subject_id"] not in selected_subjects
                        )
                    ),
                    None,
                )
                if match is not None:
                    selected.append(match)
                    selected_ids.add(match["scan_id"])
                    selected_subjects.add(match["subject_id"])
                    changed = True
                    if len(selected) == count:
                        return selected
            if changed:
                break
        if not changed:
            break
    return selected


def seed_dataloader_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(text: str, base_seed: int = 0) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str | None = None) -> torch.device:
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def load_checkpoint_training_scan_ids(
    checkpoint: str | Path, source_scan_id_suffix: str = ""
) -> set[str]:
    """Return normalized scan IDs used to train a source checkpoint."""
    path = resolve_repo_path(checkpoint)
    payload = torch.load(path, map_location="cpu")
    source_ids = payload.get("training_scan_ids", payload.get("train_scan_ids", []))
    if not source_ids:
        raise ValueError(f"Checkpoint has no training scan provenance: {path}")
    if source_scan_id_suffix:
        return {
            scan_id[: -len(source_scan_id_suffix)]
            if scan_id.endswith(source_scan_id_suffix)
            else scan_id
            for scan_id in source_ids
        }
    return set(source_ids)


def build_decoder(config: dict[str, Any], device: torch.device):
    module = importlib.import_module(f"networks.{config['network_arch']}")
    decoder = module.Decoder(
        int(config["latent_size"]), **copy.deepcopy(config["network_specs"])
    )
    return decoder.to(device)


def checkpoint_path(config: dict[str, Any], checkpoint: str | Path) -> Path:
    value = Path(checkpoint)
    if value.is_file():
        return value
    name = str(checkpoint)
    name = name if name.endswith(".pth") else f"{name}.pth"
    return resolve_repo_path(config["output_dir"]) / "checkpoints" / name


def load_decoder_checkpoint(
    config: dict[str, Any], checkpoint: str | Path, device: torch.device
):
    path = checkpoint_path(config, checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device)
    decoder = build_decoder(config, device)
    decoder.load_state_dict(strip_module_prefix(payload.get("model_state_dict", payload)))
    decoder.eval()
    return decoder, payload, path


def warm_start_global_decoder(
    decoder,
    checkpoint: str | Path,
    device: torch.device,
    latent_embedding: torch.nn.Embedding | None = None,
    target_scan_ids: list[str] | None = None,
    source_scan_id_suffix: str = "",
) -> dict[str, Any]:
    path = resolve_repo_path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Global checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device)
    source = strip_module_prefix(payload.get("model_state_dict", payload))
    decoder.global_decoder.load_state_dict(source, strict=True)
    source_config = payload.get("config", {})
    if int(source_config.get("latent_size", decoder.latent_size)) != decoder.latent_size:
        raise ValueError("Global checkpoint latent dimension does not match the hybrid model.")
    report = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "source_epoch": payload.get("epoch"),
        "source_best_validation_l1": payload.get("best_validation_l1"),
        "latent_size": decoder.latent_size,
        "loaded_tensor_count": len(source),
        "network_specs": source_config.get("network_specs"),
        "transfer": "strict_global_decoder",
    }
    if latent_embedding is not None or target_scan_ids is not None:
        if latent_embedding is None or target_scan_ids is None:
            raise ValueError("latent_embedding and target_scan_ids must be provided together.")
        source_ids = payload.get("training_scan_ids", payload.get("train_scan_ids", []))
        source_latents = payload.get("latent_codes")
        if isinstance(source_latents, dict):
            source_latents = source_latents.get("weight")
        if source_latents is None or len(source_ids) != len(source_latents):
            raise ValueError("Source checkpoint has no aligned training latent table.")

        def normalize(scan_id: str) -> str:
            if source_scan_id_suffix and scan_id.endswith(source_scan_id_suffix):
                return scan_id[: -len(source_scan_id_suffix)]
            return scan_id

        source_lookup = {normalize(scan_id): index for index, scan_id in enumerate(source_ids)}
        matched = []
        with torch.no_grad():
            for target_index, scan_id in enumerate(target_scan_ids):
                source_index = source_lookup.get(scan_id)
                if source_index is not None:
                    latent_embedding.weight[target_index].copy_(
                        source_latents[source_index].to(
                            device=latent_embedding.weight.device,
                            dtype=latent_embedding.weight.dtype,
                        )
                    )
                    matched.append(scan_id)
        report.update(
            {
                "latent_transfer": "matched_scan_id",
                "source_scan_id_suffix_removed": source_scan_id_suffix,
                "target_training_scan_count": len(target_scan_ids),
                "matched_latent_count": len(matched),
                "random_latent_count": len(target_scan_ids) - len(matched),
            }
        )
    return report


def warm_start_full_decoder(
    decoder,
    checkpoint: str | Path,
    device: torch.device,
    expected_network_arch: str | None = None,
) -> dict[str, Any]:
    """Restore every decoder/grid tensor, deliberately excluding per-scan state.

    This is for a new-data fine-tuning run.  It is intentionally distinct from
    ``--resume``: the saved source latent table, optimizer moments, and RNG
    state are not compatible with a different training manifest and are never
    copied here.
    """
    path = resolve_repo_path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Decoder warm-start checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device)
    source = strip_module_prefix(payload.get("model_state_dict", payload))
    if not isinstance(source, dict):
        raise ValueError(f"Decoder warm-start checkpoint has no state dictionary: {path}")
    target = decoder.state_dict()
    missing = sorted(set(target).difference(source))
    unexpected = sorted(set(source).difference(target))
    mismatched = sorted(
        key
        for key in set(target).intersection(source)
        if tuple(target[key].shape) != tuple(source[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Full decoder warm-start is incompatible with the requested architecture: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatches={mismatched[:5]}."
        )
    source_config = payload.get("config", {})
    source_arch = source_config.get("network_arch")
    if expected_network_arch and source_arch and source_arch != expected_network_arch:
        raise ValueError(
            f"Decoder warm-start architecture is {source_arch!r}, expected "
            f"{expected_network_arch!r}."
        )
    source_latent_size = source_config.get("latent_size", decoder.latent_size)
    if int(source_latent_size) != decoder.latent_size:
        raise ValueError(
            f"Decoder warm-start latent size is {source_latent_size}, expected "
            f"{decoder.latent_size}."
        )
    decoder.load_state_dict(source, strict=True)
    source_ids = payload.get("training_scan_ids", payload.get("train_scan_ids", []))
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "source_epoch": payload.get("epoch"),
        "source_best_validation_l1": payload.get("best_validation_l1"),
        "source_best_mesh_assd": payload.get("best_mesh_assd"),
        "source_network_arch": source_arch,
        "latent_size": decoder.latent_size,
        "source_training_scan_count": len(source_ids),
        "loaded_tensor_count": len(source),
        "target_tensor_count": len(target),
        "state_dict_key_and_shape_match": True,
        "transfer": "strict_full_decoder",
        "latent_transfer": "disabled_new_pilot_codes",
        "optimizer_state_transfer": "disabled_new_optimizer",
        "rng_state_transfer": "disabled_new_run_seed",
    }


def load_sdf_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        if "pos" not in archive or "neg" not in archive:
            raise ValueError(f"SDF archive lacks pos/neg arrays: {path}")
        pos = np.asarray(archive["pos"], dtype=np.float32)
        neg = np.asarray(archive["neg"], dtype=np.float32)
    pos = pos[np.isfinite(pos).all(axis=1)]
    neg = neg[np.isfinite(neg).all(axis=1)]
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != 4 or neg.shape[1] != 4:
        raise ValueError(f"Unexpected SDF array shapes in {path}: {pos.shape}, {neg.shape}")
    if not len(pos) or not len(neg):
        raise ValueError(f"SDF archive has an empty sign partition: {path}")
    if np.any(pos[:, 3] < 0.0) or np.any(neg[:, 3] > 0.0):
        raise ValueError(f"SDF archive pos/neg sign contract is violated: {path}")
    return pos, neg


def load_obj_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def validate_manifest_contract(
    rows: list[dict[str, str]], grid_aabb: list[list[float]]
) -> dict[str, Any]:
    """Fail fast on split leakage, missing inputs, or meshes outside the local ROI."""
    scan_ids = [row["scan_id"] for row in rows]
    if len(scan_ids) != len(set(scan_ids)):
        raise ValueError("Manifest scan_id values are not unique.")
    subject_splits: dict[str, set[str]] = {}
    for row in rows:
        subject_splits.setdefault(row["subject_id"], set()).add(row["split"])
        for key in ("mesh_path", "sdf_npz_path"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f"Manifest {key} does not exist: {row[key]}")
    leaking_subjects = {
        subject: sorted(splits)
        for subject, splits in subject_splits.items()
        if len(splits) != 1
    }
    if leaking_subjects:
        raise ValueError(
            f"Subjects occur in multiple train/val/test splits: {dict(list(leaking_subjects.items())[:5])}"
        )

    population_min = np.full(3, np.inf, dtype=np.float64)
    population_max = np.full(3, -np.inf, dtype=np.float64)
    for row in rows:
        vertices, _faces = load_obj_arrays(row["mesh_path"])
        if not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError(f"Mesh has empty/non-finite vertices: {row['mesh_path']}")
        population_min = np.minimum(population_min, vertices.min(axis=0))
        population_max = np.maximum(population_max, vertices.max(axis=0))
    aabb = np.asarray(grid_aabb, dtype=np.float64)
    lower_margin = population_min - aabb[0]
    upper_margin = aabb[1] - population_max
    if np.any(lower_margin < 0.0) or np.any(upper_margin < 0.0):
        raise ValueError(
            "At least one mesh lies outside grid_aabb: "
            f"population=[{population_min.tolist()}, {population_max.tolist()}], "
            f"aabb={aabb.tolist()}"
        )
    return {
        "scan_count": len(rows),
        "subject_count": len(subject_splits),
        "split_scan_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "val", "test")
        },
        "split_subject_counts": {
            split: len({row["subject_id"] for row in rows if row["split"] == split})
            for split in ("train", "val", "test")
        },
        "population_mesh_min": population_min.tolist(),
        "population_mesh_max": population_max.tolist(),
        "grid_aabb": aabb.tolist(),
        "lower_margin": lower_margin.tolist(),
        "upper_margin": upper_margin.tolist(),
        "subject_split_disjoint": True,
        "all_mesh_and_sdf_paths_exist": True,
    }


class MeshSurfaceSampler:
    """Area and grid-cell balanced mesh sampling with outward normals."""

    def __init__(
        self,
        mesh_path: str | Path,
        grid_aabb: list[list[float]],
        grid_resolution: int | list[int],
    ) -> None:
        vertices, faces = load_obj_arrays(mesh_path)
        triangles = vertices[faces]
        crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        double_area = np.linalg.norm(crosses, axis=1)
        valid = double_area > 1.0e-12
        if not np.any(valid):
            raise ValueError(f"Mesh has no valid triangles: {mesh_path}")
        self.vertices = vertices
        self.faces = faces[valid]
        self.crosses = crosses[valid]
        self.double_area = double_area[valid]
        self.face_probabilities = self.double_area / self.double_area.sum()

        triangles = self.vertices[self.faces]
        signed_volume = np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum() / 6.0
        orientation = 1.0 if signed_volume >= 0.0 else -1.0
        vertex_normals = np.zeros_like(self.vertices, dtype=np.float64)
        oriented_crosses = self.crosses.astype(np.float64) * orientation
        for corner in range(3):
            np.add.at(vertex_normals, self.faces[:, corner], oriented_crosses)
        lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        self.vertex_normals = (vertex_normals / np.maximum(lengths, 1.0e-12)).astype(np.float32)

        resolution = (
            np.repeat(int(grid_resolution), 3)
            if isinstance(grid_resolution, int)
            else np.asarray(grid_resolution, dtype=np.int64)
        )
        minimum = np.asarray(grid_aabb[0], dtype=np.float64)
        maximum = np.asarray(grid_aabb[1], dtype=np.float64)
        centroids = self.vertices[self.faces].mean(axis=1)
        cell_xyz = np.floor((centroids - minimum) / (maximum - minimum) * (resolution - 1)).astype(np.int64)
        cell_xyz = np.clip(cell_xyz, 0, resolution - 2)
        cell_id = (
            cell_xyz[:, 0]
            + (resolution[0] - 1) * (
                cell_xyz[:, 1] + (resolution[1] - 1) * cell_xyz[:, 2]
            )
        )
        self.cell_faces = [
            np.flatnonzero(cell_id == value) for value in np.unique(cell_id)
        ]

    def _sample_face_indices(
        self, count: int, rng: np.random.Generator, balance_cells: bool
    ) -> np.ndarray:
        if not balance_cells:
            return rng.choice(len(self.faces), size=count, p=self.face_probabilities)
        chosen_cells = rng.integers(0, len(self.cell_faces), size=count)
        selected = np.empty(count, dtype=np.int64)
        for cell in np.unique(chosen_cells):
            positions = np.flatnonzero(chosen_cells == cell)
            candidates = self.cell_faces[int(cell)]
            probabilities = self.double_area[candidates]
            probabilities = probabilities / probabilities.sum()
            selected[positions] = rng.choice(
                candidates, size=len(positions), replace=True, p=probabilities
            )
        return selected

    def sample(
        self, count: int, rng: np.random.Generator, balance_cells: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        if count < 1:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
        face_indices = self._sample_face_indices(count, rng, balance_cells)
        faces = self.faces[face_indices]
        triangles = self.vertices[faces]
        normal_triangles = self.vertex_normals[faces]
        sqrt_u = np.sqrt(rng.random(count)).astype(np.float32)
        v = rng.random(count).astype(np.float32)
        weights = np.stack(
            (1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v), axis=1
        )
        points = np.sum(triangles * weights[:, :, None], axis=1)
        normals = np.sum(normal_triangles * weights[:, :, None], axis=1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
        return points.astype(np.float32), normals.astype(np.float32)


def _draw_rows(
    array: np.ndarray,
    count: int,
    rng: np.random.Generator,
    maximum_abs_sdf: float | None = None,
    grid_aabb: np.ndarray | None = None,
    grid_resolution: np.ndarray | None = None,
) -> np.ndarray:
    """Draw continuous NPZ samples, optionally balancing an oversample by grid cell."""
    if count < 1:
        return np.empty((0, 4), dtype=np.float32)
    if len(array) < 1:
        raise ValueError("Cannot sample from an empty SDF array.")
    candidate_count = count if grid_aabb is None else max(count * 4, count + 256)
    selected: list[np.ndarray] = []
    remaining = candidate_count
    attempts = 0
    while remaining > 0 and attempts < 100:
        draw_count = max(remaining * 2, 1024)
        draw = array[rng.integers(0, len(array), size=draw_count)]
        if maximum_abs_sdf is not None:
            draw = draw[np.abs(draw[:, 3]) <= maximum_abs_sdf]
        if len(draw):
            take = min(remaining, len(draw))
            selected.append(draw[:take])
            remaining -= take
        attempts += 1
    if remaining:
        eligible = (
            array
            if maximum_abs_sdf is None
            else array[np.abs(array[:, 3]) <= maximum_abs_sdf]
        )
        if not len(eligible):
            raise ValueError(
                f"No SDF samples satisfy |sdf| <= {maximum_abs_sdf}."
            )
        selected.append(eligible[rng.integers(0, len(eligible), size=remaining)])
    candidates = np.concatenate(selected, axis=0)
    if grid_aabb is None:
        return candidates[:count].astype(np.float32, copy=False)

    minimum, maximum = grid_aabb
    resolution = grid_resolution
    cells = np.floor(
        (candidates[:, :3] - minimum) / (maximum - minimum) * (resolution - 1)
    ).astype(np.int64)
    cells = np.clip(cells, 0, resolution - 2)
    cell_ids = cells[:, 0] + (resolution[0] - 1) * (
        cells[:, 1] + (resolution[1] - 1) * cells[:, 2]
    )
    order = np.argsort(cell_ids, kind="stable")
    sorted_ids = cell_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    lengths = np.diff(np.r_[starts, len(order)])
    groups = rng.integers(0, len(starts), size=count)
    offsets = np.floor(rng.random(count) * lengths[groups]).astype(np.int64)
    return candidates[order[starts[groups] + offsets]].astype(np.float32, copy=False)


def sample_continuous_sdf_pair(
    pos: np.ndarray,
    neg: np.ndarray,
    sampling: dict[str, Any],
    rng: np.random.Generator,
    grid_aabb: list[list[float]] | None = None,
    grid_resolution: int | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return distinct global and local continuous-SDF training sets."""
    global_near = int(sampling["global_near_samples_per_scene"])
    global_pos = int(sampling["global_positive_samples_per_scene"])
    global_neg = int(sampling["global_negative_samples_per_scene"])
    local_ultra = int(sampling["local_ultra_near_samples_per_scene"])
    local_pos = int(sampling["local_positive_samples_per_scene"])
    local_neg = int(sampling["local_negative_samples_per_scene"])
    if min(global_near, global_pos, global_neg, local_ultra, local_pos, local_neg) < 0:
        raise ValueError("SDF sample counts cannot be negative.")
    near_band = float(sampling.get("near_band", 0.1))
    ultra_band = float(sampling.get("ultra_near_band", 0.03))
    if not 0.0 < ultra_band <= near_band:
        raise ValueError("Expected 0 < ultra_near_band <= near_band.")
    near_pos_count = global_near // 2
    global_arrays = (
        _draw_rows(pos, near_pos_count, rng, near_band),
        _draw_rows(neg, global_near - near_pos_count, rng, near_band),
        _draw_rows(pos, global_pos, rng),
        _draw_rows(neg, global_neg, rng),
    )
    global_samples = np.concatenate(global_arrays, axis=0)
    rng.shuffle(global_samples)

    aabb = None if grid_aabb is None else np.asarray(grid_aabb, dtype=np.float32)
    resolution = None
    if grid_resolution is not None:
        resolution = (
            np.repeat(int(grid_resolution), 3)
            if isinstance(grid_resolution, int)
            else np.asarray(grid_resolution, dtype=np.int64)
        )
    ultra_pos_count = local_ultra // 2
    local_arrays = (
        _draw_rows(pos, ultra_pos_count, rng, ultra_band, aabb, resolution),
        _draw_rows(neg, local_ultra - ultra_pos_count, rng, ultra_band, aabb, resolution),
        _draw_rows(pos, local_pos, rng, near_band, aabb, resolution),
        _draw_rows(neg, local_neg, rng, near_band, aabb, resolution),
    )
    local_samples = np.concatenate(local_arrays, axis=0)
    rng.shuffle(local_samples)
    return global_samples.astype(np.float32), local_samples.astype(np.float32)


class ContinuousSDFDataset(Dataset):
    """Separate broad/global and dense/local samples from continuous SDF archives."""

    def __init__(self, rows: list[dict[str, str]], config: dict[str, Any]) -> None:
        self.rows = rows
        self.base_seed = int(config.get("seed", 0))
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.sampling = copy.deepcopy(config["sampling"])
        specs = config["network_specs"]
        self.grid_aabb = specs["grid_aabb"]
        self.grid_resolution = specs["grid_resolution"]
        self.loaded_sdf = None
        if bool(config.get("load_dataset_into_ram", False)):
            self.loaded_sdf = [load_sdf_arrays(row["sdf_npz_path"]) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(int(epoch))

    def __getitem__(self, index: int):
        rng = np.random.default_rng(
            stable_seed(
                self.rows[index].get("scan_id", str(index)),
                self.base_seed + int(self._epoch.item()),
            )
        )
        pos, neg = (
            load_sdf_arrays(self.rows[index]["sdf_npz_path"])
            if self.loaded_sdf is None
            else self.loaded_sdf[index]
        )
        global_samples, local_samples = sample_continuous_sdf_pair(
            pos,
            neg,
            self.sampling,
            rng,
            self.grid_aabb,
            self.grid_resolution,
        )
        return (
            torch.from_numpy(global_samples),
            torch.from_numpy(local_samples),
            index,
        )


# Kept as an import alias for older callers; the returned contract is now the
# Compact-SDF-style pair (global_samples, local_samples, scene_index).
HybridSDFDataset = ContinuousSDFDataset


def sample_balanced(
    pos: np.ndarray, neg: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    half = count // 2
    result = np.concatenate(
        (
            pos[rng.integers(0, len(pos), size=half)],
            neg[rng.integers(0, len(neg), size=count - half)],
        ),
        axis=0,
    )
    rng.shuffle(result)
    return result


def fit_single_latent(
    decoder,
    sdf_path: str | Path,
    latent_size: int,
    fit_config: dict[str, Any],
    clamp_distance: float,
    device: torch.device,
    seed: int,
    steps_override: int | None = None,
    mesh_path: str | Path | None = None,
    network_specs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one code with a frozen decoder using distinct global/local SDF sets."""
    rng = np.random.default_rng(seed)
    pos, neg = load_sdf_arrays(sdf_path)
    holdout_fraction = float(fit_config.get("holdout_fraction", 0.1))

    def split(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = rng.permutation(len(array))
        count = min(len(array) - 1, max(1, int(round(len(array) * holdout_fraction))))
        return array[order[count:]], array[order[:count]]

    pos_fit, pos_holdout = split(pos)
    neg_fit, neg_holdout = split(neg)
    latent = torch.empty(1, latent_size, device=device)
    torch.nn.init.normal_(latent, mean=0.0, std=float(fit_config.get("initial_std", 0.01)))
    latent.requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=float(fit_config.get("learning_rate", 0.005)))
    steps = int(steps_override or fit_config["steps"])
    if network_specs is None:
        raise ValueError("network_specs are required for local grid sampling.")
    sample_config = copy.deepcopy(fit_config["sampling"])
    reg = float(fit_config.get("code_regularization_lambda", 0.0))
    branch_weights = fit_config.get(
        "branch_weights", {"global": 1.0, "local": 1.0, "fused": 1.0}
    )
    bound = float(fit_config.get("code_bound", 0.0))
    patience = int(fit_config.get("early_stop_patience", steps))
    delta = float(fit_config.get("early_stop_min_delta", 0.0))
    best_loss = math.inf
    best_latent = None
    no_improvement = 0
    initial_l1 = None
    final_l1 = None
    decoder.eval()

    def objective(
        code: torch.Tensor,
        global_array: np.ndarray,
        local_array: np.ndarray,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        global_batch = torch.from_numpy(global_array).to(device)
        local_batch = torch.from_numpy(local_array).to(device)
        global_parts = decoder(
            torch.cat((code.expand(len(global_batch), -1), global_batch[:, :3]), dim=1),
            return_parts=True,
        )
        local_parts = decoder(
            torch.cat((code.expand(len(local_batch), -1), local_batch[:, :3]), dim=1),
            return_parts=True,
        )
        global_target = global_batch[:, 3:4].clamp(-clamp_distance, clamp_distance)
        local_target = local_batch[:, 3:4].clamp(-clamp_distance, clamp_distance)
        global_l1 = torch.abs(
            global_parts["global_sdf"].clamp(-clamp_distance, clamp_distance) - global_target
        ).mean()
        local_l1 = torch.abs(
            local_parts["local_sdf"].clamp(-clamp_distance, clamp_distance) - local_target
        ).mean()
        fused_global_l1 = torch.abs(
            global_parts["sdf"].clamp(-clamp_distance, clamp_distance) - global_target
        ).mean()
        fused_local_l1 = torch.abs(
            local_parts["sdf"].clamp(-clamp_distance, clamp_distance) - local_target
        ).mean()
        fused_l1 = 0.5 * (fused_global_l1 + fused_local_l1)
        loss = (
            float(branch_weights.get("global", 1.0)) * global_l1
            + float(branch_weights.get("local", 1.0)) * local_l1
            + float(branch_weights.get("fused", 1.0)) * fused_l1
            + reg * code.square().mean()
        )
        global_sign = (
            (global_parts["global_sdf"] >= 0) == (global_target >= 0)
        ).to(global_l1.dtype).mean()
        local_sign = (
            (local_parts["local_sdf"] >= 0) == (local_target >= 0)
        ).to(local_l1.dtype).mean()
        fused_sign = 0.5 * (
            ((global_parts["sdf"] >= 0) == (global_target >= 0)).to(fused_l1.dtype).mean()
            + ((local_parts["sdf"] >= 0) == (local_target >= 0)).to(fused_l1.dtype).mean()
        )
        return loss, {
            "global": global_l1,
            "local": local_l1,
            "fused": fused_l1,
            "fused_global": fused_global_l1,
            "fused_local": fused_local_l1,
            "global_sign": global_sign,
            "local_sign": local_sign,
            "fused_sign": fused_sign,
            "local_disagreement": torch.abs(
                local_parts["global_sdf"] - local_parts["local_sdf"]
            ).mean(),
            "local_gate_mean": local_parts["gate"].mean(),
        }

    for step in range(steps):
        global_array, local_array = sample_continuous_sdf_pair(
            pos_fit,
            neg_fit,
            sample_config,
            rng,
            network_specs["grid_aabb"],
            network_specs["grid_resolution"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss, terms = objective(latent, global_array, local_array)
        data_loss = terms["fused"]
        # Only the code is optimized here.  autograd.grad avoids allocating
        # gradients for the (usually much larger) frozen decoder.
        (latent_gradient,) = torch.autograd.grad(loss, latent)
        latent.grad = latent_gradient
        optimizer.step()
        if bound > 0:
            with torch.no_grad():
                norm = latent.norm(dim=1, keepdim=True)
                latent.mul_(torch.clamp(bound / (norm + 1.0e-12), max=1.0))
        value = float(loss.detach().cpu())
        initial_l1 = float(data_loss.detach().cpu()) if initial_l1 is None else initial_l1
        final_l1 = float(data_loss.detach().cpu())
        if value < best_loss - delta:
            best_loss = value
            best_latent = latent.detach().clone()
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            break
    if best_latent is None:
        raise RuntimeError(f"Latent optimization failed for {sdf_path}")
    evaluation_global, evaluation_local = sample_continuous_sdf_pair(
        pos_holdout,
        neg_holdout,
        sample_config,
        rng,
        network_specs["grid_aabb"],
        network_specs["grid_resolution"],
    )
    with torch.no_grad():
        _evaluation_objective, evaluation_terms = objective(
            best_latent, evaluation_global, evaluation_local
        )
    result = best_latent.cpu().numpy().reshape(-1).astype(np.float32)
    global_l1 = float(evaluation_terms["global"].cpu())
    local_l1 = float(evaluation_terms["local"].cpu())
    fused_l1 = float(evaluation_terms["fused"].cpu())
    return result, {
        "steps_requested": steps,
        "steps_completed": step + 1,
        "initial_data_l1": initial_l1,
        "final_data_l1": final_l1,
        "best_objective": best_loss,
        "heldout_sdf_l1": fused_l1,
        "heldout_global_l1": global_l1,
        "heldout_local_l1": local_l1,
        "heldout_fused_l1": fused_l1,
        "heldout_fused_global_l1": float(evaluation_terms["fused_global"].cpu()),
        "heldout_fused_local_l1": float(evaluation_terms["fused_local"].cpu()),
        "heldout_global_sign_accuracy": float(evaluation_terms["global_sign"].cpu()),
        "heldout_local_sign_accuracy": float(evaluation_terms["local_sign"].cpu()),
        "heldout_fused_sign_accuracy": float(evaluation_terms["fused_sign"].cpu()),
        "heldout_global_local_disagreement": float(evaluation_terms["local_disagreement"].cpu()),
        "heldout_local_gate_mean": float(evaluation_terms["local_gate_mean"].cpu()),
        "latent_norm": float(np.linalg.norm(result)),
        "finite": bool(np.isfinite(result).all()),
    }


def decode_latent_to_mesh(
    decoder,
    latent: np.ndarray,
    output_path: str | Path,
    resolution: int,
    max_batch: int,
    device: torch.device,
) -> dict[str, Any]:
    import trimesh
    from skimage.measure import marching_cubes

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    latent_tensor = torch.from_numpy(np.asarray(latent, dtype=np.float32)).reshape(1, -1).to(device)
    total = resolution**3
    values = np.empty(total, dtype=np.float32)
    step = 2.0 / (resolution - 1)
    decoder.eval()
    with torch.no_grad():
        for start in range(0, total, max_batch):
            stop = min(start + max_batch, total)
            index = torch.arange(start, stop, device=device)
            x = torch.div(index, resolution * resolution, rounding_mode="floor")
            y = torch.div(index, resolution, rounding_mode="floor") % resolution
            z = index % resolution
            xyz = torch.stack((x, y, z), dim=1).float() * step - 1.0
            prediction = decoder(torch.cat((latent_tensor.expand(len(index), -1), xyz), dim=1))
            values[start:stop] = prediction[:, 0].detach().cpu().numpy()
    volume = values.reshape(resolution, resolution, resolution)
    value_min, value_max = float(volume.min()), float(volume.max())
    if not value_min <= 0.0 <= value_max:
        raise RuntimeError(f"No zero level set: [{value_min}, {value_max}]")
    vertices, faces, _normals, _values = marching_cubes(
        volume, level=0.0, spacing=(step, step, step), method="lewiner"
    )
    vertices += np.asarray([-1.0, -1.0, -1.0], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(output)
    return {
        "mesh_path": str(output),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "sdf_grid_min": value_min,
        "sdf_grid_max": value_max,
    }
