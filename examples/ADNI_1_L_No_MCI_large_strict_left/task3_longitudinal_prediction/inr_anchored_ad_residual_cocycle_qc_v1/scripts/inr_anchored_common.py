"""Shared, read-only input and metric helpers for the anchored INR experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


def experiment_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "longitudinal_direct_flow.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repository root from experiment script.")


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base or experiment_dir()) / path


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if int(config.get("LatentSize", 0)) <= 0:
        raise ValueError("Config requires a positive LatentSize.")
    return config, path


def ensure_paths_exist(config: Mapping[str, Any], base: Path | None = None) -> dict[str, Path]:
    required = {
        "source_experiment": config["SourceExperimentDir"],
        "metadata": config["MetadataCsv"],
        "decoder": config["DecoderCheckpoint"],
        "base_flow": config["BaseFlowCheckpoint"],
    }
    for split, value in config["LatentArchives"].items():
        required[f"latents_{split}"] = value
    resolved = {name: resolve_path(value, base) for name, value in required.items()}
    missing = [f"{name}: {path}" for name, path in resolved.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required input(s) missing:\n" + "\n".join(missing))
    return resolved


def load_metadata_and_latents(config: Mapping[str, Any], base: Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the existing QC-clean contract and enforce exact scan/latent alignment."""
    paths = ensure_paths_exist(config, base)
    frame = pd.read_csv(paths["metadata"])
    required = {
        "split", "scan_id", "subject_id", "visit_order", "diagnosis", "label_ad",
        "continuous_age_norm", "continuous_age_years", "mesh_path", "sdf_npz_path",
        "left_mesh_volume_mm3",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"QC metadata missing columns: {missing}")
    frame = frame.copy()
    frame["scan_id"] = frame["scan_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["label_ad"] = frame["label_ad"].astype(int)
    frame["visit_order"] = frame["visit_order"].astype(int)
    frame["continuous_age_norm"] = frame["continuous_age_norm"].astype(float)
    frame["continuous_age_years"] = frame["continuous_age_years"].astype(float)
    if set(frame["split"].unique()) != {"train", "val", "test"}:
        raise ValueError(f"Expected train/val/test metadata, got {sorted(frame['split'].unique())}")
    if set(frame["diagnosis"].unique()) != {"CN", "AD"} or set(frame["label_ad"].unique()) != {0, 1}:
        raise ValueError("QC metadata must contain exactly CN/AD with labels 0/1.")
    if frame["scan_id"].duplicated().any():
        raise ValueError("Duplicate scan_id in QC metadata.")
    latent_map: dict[str, np.ndarray] = {}
    latent_size = int(config["LatentSize"])
    for split, path_value in config["LatentArchives"].items():
        archive = np.load(resolve_path(path_value, base), allow_pickle=False)
        if set(("scan_ids", "latents")).difference(archive.files):
            raise ValueError(f"Latent archive lacks scan_ids/latents: {path_value}")
        scan_ids, latents = archive["scan_ids"], archive["latents"]
        if latents.ndim != 2 or latents.shape[1] != latent_size or len(scan_ids) != len(latents):
            raise ValueError(f"Invalid latent archive shape at {path_value}: {latents.shape}")
        for scan_id, latent in zip(scan_ids, latents):
            key = Path(scan_id.decode() if isinstance(scan_id, bytes) else str(scan_id)).stem
            if key in latent_map:
                raise ValueError(f"Duplicate latent scan ID {key}")
            latent_map[key] = np.asarray(latent, dtype=np.float32)
    metadata_ids = set(frame["scan_id"])
    if metadata_ids != set(latent_map):
        missing_latents = sorted(metadata_ids.difference(latent_map))
        extra_latents = sorted(set(latent_map).difference(metadata_ids))
        raise ValueError(
            f"QC metadata/latents mismatch: missing={len(missing_latents)}, extra={len(extra_latents)}"
        )
    frame = frame.sort_values(
        ["split", "subject_id", "continuous_age_norm", "visit_order", "scan_id"]
    ).reset_index(drop=True)
    frame.insert(0, "global_index", np.arange(len(frame), dtype=np.int64))
    latents = np.stack([latent_map[scan_id] for scan_id in frame["scan_id"]], axis=0)
    return frame, latents


def read_obj_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif fields[0] == "f" and len(fields) >= 4:
                index = [int(value.split("/")[0]) - 1 for value in fields[1:4]]
                faces.append(index)
    if not vertices or not faces:
        raise ValueError(f"OBJ lacks vertices/faces: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def mesh_normals_and_areas(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    face_cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_area = 0.5 * np.linalg.norm(face_cross, axis=1)
    normals = np.zeros_like(vertices, dtype=np.float64)
    areas = np.zeros(len(vertices), dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_cross)
        np.add.at(areas, faces[:, corner], face_area / 3.0)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norm, 1.0e-12)
    return normals.astype(np.float32), areas.astype(np.float32)


def signed_mesh_volume(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Absolute tetrahedral volume for [B,V,3] vertices and shared [F,3] faces."""
    triangles = vertices[:, faces]  # [B,F,3,3]
    signed = torch.sum(triangles[:, :, 0] * torch.cross(triangles[:, :, 1], triangles[:, :, 2], dim=-1), dim=-1)
    return signed.sum(dim=1).abs() / 6.0


def decode_sdf(decoder: nn.Module, latents: torch.Tensor, points: torch.Tensor, chunk: int = 131072) -> torch.Tensor:
    if points.ndim != 3 or latents.ndim != 2 or points.shape[0] != latents.shape[0]:
        raise ValueError("decode_sdf expects latents [B,L] and points [B,N,3].")
    batch, count = points.shape[:2]
    joined = torch.cat(
        [latents[:, None, :].expand(batch, count, -1), points], dim=-1
    ).reshape(-1, latents.shape[1] + 3)
    values = [decoder(joined[start:start + chunk]) for start in range(0, len(joined), chunk)]
    return torch.cat(values, dim=0).reshape(batch, count)


def load_frozen_decoder(config: Mapping[str, Any], device: torch.device, base: Path | None = None) -> nn.Module:
    repository_root()
    source_specs = json.loads((resolve_path(config["SourceExperimentDir"], base) / "specs.json").read_text())
    architecture = __import__("networks." + str(source_specs["NetworkArch"]), fromlist=["Decoder"])
    decoder = architecture.Decoder(int(config["LatentSize"]), **source_specs["NetworkSpecs"]).to(device)
    payload = torch.load(resolve_path(config["DecoderCheckpoint"], base), map_location="cpu")
    state = payload["model_state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder


def load_frozen_base_flow(config: Mapping[str, Any], device: torch.device, base: Path | None = None) -> nn.Module:
    repository_root()
    from longitudinal_direct_flow import DirectAgeFlow

    source_dir = resolve_path(config["SourceExperimentDir"], base)
    specs = json.loads((source_dir / "specs.json").read_text())
    model = DirectAgeFlow(
        latent_size=int(specs["CodeLength"]),
        hidden_dims=[int(x) for x in specs["FlowHiddenDims"]],
        condition_dim=int(specs.get("ConditionDim", 1)),
        activation=str(specs.get("FlowActivation", "silu")),
        dropout=float(specs.get("FlowDropout", 0.0)),
        zero_initialize_output=bool(specs.get("FlowZeroInitializeOutput", True)),
        latent_condition_mode=str(specs.get("LatentConditionMode", "full")),
        latent_condition_dim=specs.get("LatentConditionDim"),
        include_delta_time_input=bool(specs.get("FlowIncludeDeltaTimeInput", False)),
    ).to(device)
    payload = torch.load(resolve_path(config["BaseFlowCheckpoint"], base), map_location="cpu")
    model.load_state_dict(payload["flow_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_forward_pair_table(frame: pd.DataFrame, source_mode: str) -> pd.DataFrame:
    mode = str(source_mode).strip().lower()
    if mode in {"anchor", "baseline", "one_shot"}:
        mode = "first_only"
    if mode not in {"all_forward_starts", "first_only", "adjacent"}:
        raise ValueError(f"Unknown pair mode {source_mode!r}")
    records: list[dict[str, Any]] = []
    for (split, subject_id), rows in frame.groupby(["split", "subject_id"], sort=True):
        rows = rows.sort_values(["continuous_age_norm", "visit_order", "scan_id"]).reset_index(drop=True)
        if len(rows) < 2:
            continue
        times = rows["continuous_age_norm"].to_numpy()
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"Non-increasing time in subject {subject_id}")
        if rows["label_ad"].nunique() != 1:
            raise ValueError(f"Changing diagnosis in subject {subject_id}")
        if mode == "first_only":
            pairs = [(0, j) for j in range(1, len(rows))]
        elif mode == "adjacent":
            pairs = [(j, j + 1) for j in range(len(rows) - 1)]
        else:
            pairs = [(i, j) for i in range(len(rows) - 1) for j in range(i + 1, len(rows))]
        for source_i, target_i in pairs:
            source, target = rows.iloc[source_i], rows.iloc[target_i]
            records.append({
                "split": split, "subject_id": str(subject_id), "diagnosis": str(source.diagnosis),
                "label_ad": int(source.label_ad), "source_index": int(source.global_index),
                "target_index": int(target.global_index), "source_scan_id": str(source.scan_id),
                "target_scan_id": str(target.scan_id), "source_time": float(source.continuous_age_norm),
                "target_time": float(target.continuous_age_norm), "source_age_years": float(source.continuous_age_years),
                "target_age_years": float(target.continuous_age_years),
                "gap_years": float(target.continuous_age_years - source.continuous_age_years),
                "is_reverse": False,
            })
    return pd.DataFrame.from_records(records)


def reverse_pair_table(pairs: pd.DataFrame) -> pd.DataFrame:
    reverse = pairs.copy()
    for left, right in [
        ("source_index", "target_index"), ("source_scan_id", "target_scan_id"),
        ("source_time", "target_time"), ("source_age_years", "target_age_years"),
    ]:
        reverse[left], reverse[right] = pairs[right].to_numpy(), pairs[left].to_numpy()
    reverse["gap_years"] = -pairs["gap_years"].to_numpy()
    reverse["is_reverse"] = True
    return reverse


def _balanced_sdf_samples(path: str | Path, count: int, seed: int | None) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        positive = np.asarray(payload["pos"], dtype=np.float32)
        negative = np.asarray(payload["neg"], dtype=np.float32)
    positive = positive[np.isfinite(positive).all(axis=1)]
    negative = negative[np.isfinite(negative).all(axis=1)]
    rng = np.random.default_rng(seed)
    half = count // 2
    return np.concatenate(
        [positive[rng.integers(len(positive), size=half)], negative[rng.integers(len(negative), size=count-half)]], axis=0
    ).astype(np.float32)


class CachedPairDataset(Dataset):
    """Pair dataset backed by this experiment's cache, not the source run."""

    def __init__(self, root: Path, pairs: pd.DataFrame, samples_per_target: int, deterministic: bool, seed: int) -> None:
        self.root = root
        self.pairs = pairs.reset_index(drop=True)
        self.samples = int(samples_per_target)
        self.deterministic, self.seed = bool(deterministic), int(seed)
        cached = np.load(root / "metadata" / "registered_meshes.npz", allow_pickle=False, mmap_mode="r")
        self.latents = cached["latents"]
        self.vertices = cached["vertices"]
        self.normals = cached["normals"]
        self.areas = cached["areas"]
        self.volumes = cached["volumes"]
        self.faces = cached["faces"]
        manifest = pd.read_csv(root / "metadata" / "scan_manifest.csv")
        self.sdf_paths = dict(zip(manifest.global_index.astype(int), manifest.sdf_npz_path.astype(str)))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.pairs.iloc[index]
        source_i, target_i = int(record.source_index), int(record.target_index)
        seed = None
        if self.deterministic:
            digest = hashlib.sha256(f"{self.seed}:{index}:{target_i}".encode()).digest()
            seed = int.from_bytes(digest[:8], "little")
        return {
            "source_latent": torch.from_numpy(np.array(self.latents[source_i], copy=True)),
            "target_latent": torch.from_numpy(np.array(self.latents[target_i], copy=True)),
            "source_time": torch.tensor(float(record.source_time), dtype=torch.float32),
            "target_time": torch.tensor(float(record.target_time), dtype=torch.float32),
            "condition": torch.tensor([float(record.label_ad)], dtype=torch.float32),
            "source_vertices": torch.from_numpy(np.array(self.vertices[source_i], copy=True)),
            "target_vertices": torch.from_numpy(np.array(self.vertices[target_i], copy=True)),
            "source_normals": torch.from_numpy(np.array(self.normals[source_i], copy=True)),
            "vertex_areas": torch.from_numpy(np.array(self.areas[source_i], copy=True)),
            "source_volume": torch.tensor(float(self.volumes[source_i]), dtype=torch.float32),
            "target_volume": torch.tensor(float(self.volumes[target_i]), dtype=torch.float32),
            "target_samples": torch.from_numpy(_balanced_sdf_samples(self.sdf_paths[target_i], self.samples, seed)),
            "gap_years": torch.tensor(float(record.gap_years), dtype=torch.float32),
            "label_ad": torch.tensor(int(record.label_ad), dtype=torch.long),
            "subject_id": str(record.subject_id), "source_scan_id": str(record.source_scan_id),
            "target_scan_id": str(record.target_scan_id),
        }


def load_cache_arrays(root: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    cache = np.load(root / "metadata" / "registered_meshes.npz", allow_pickle=False)
    basis = np.load(root / "basis" / "train_only_basis.npz", allow_pickle=False)
    return ({name: cache[name] for name in cache.files}, {name: basis[name] for name in basis.files})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
