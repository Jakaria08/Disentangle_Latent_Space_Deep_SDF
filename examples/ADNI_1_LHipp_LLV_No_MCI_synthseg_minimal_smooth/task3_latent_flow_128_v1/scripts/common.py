#!/usr/bin/env python3
"""Shared contracts for the isolated ADNI 128-D latent-flow experiments.

This module never discovers a "latest" checkpoint implicitly.  A representation
is a frozen, hashed snapshot declared in ``configs/representations.json``.  A
later AE winner is adopted by changing that registry and regenerating the
prepared archives under a new/cleared representation directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = TASK_ROOT.parents[2]
REGISTRY_PATH = TASK_ROOT / "configs" / "representations.json"
SPLITS = ("train", "val", "test")
LATENT_DIM = 128
REQUIRED_ARCHIVE_KEYS = {
    "subject_ids",
    "subject_splits",
    "subject_diagnoses",
    "subject_label_ad",
    "subject_visit_offsets",
    "visit_scan_ids",
    "visit_subject_ids",
    "visit_splits",
    "visit_diagnoses",
    "visit_label_ad",
    "visit_orders",
    "visit_months_from_baseline",
    "visit_time_years_from_baseline",
    "visit_age_years",
    "visit_age_norm_train",
    "visit_volume_mm3",
    "visit_latent_raw_128",
    "visit_latent_standardized_128",
    "train_latent_mean_128",
    "train_latent_std_128",
}
REQUIRED_PAIR_FIELDS = {
    "split",
    "diagnosis",
    "label_ad",
    "subject_id",
    "source_index",
    "target_index",
    "intermediate_index",
    "source_scan_id",
    "target_scan_id",
    "source_visit_order",
    "target_visit_order",
    "pair_type",
    "delta_years",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    with resolved.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {resolved}")
    return value


def atomic_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def atomic_torch_save(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {device.index} is unavailable")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_run_name(value: str) -> None:
    candidate = Path(value)
    if value in {"", ".", ".."} or candidate.name != value or "/" in value or "\\" in value:
        raise ValueError("Run name must be one safe directory-name component")


def load_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = read_json(path)
    if int(registry.get("latent_dim", -1)) != LATENT_DIM:
        raise ValueError(f"Registry must declare latent_dim={LATENT_DIM}")
    representations = registry.get("representations")
    if not isinstance(representations, dict) or set(representations) != {"pca128", "spiralnet128", "adaptive128"}:
        raise ValueError("Registry must define exactly pca128, spiralnet128, and adaptive128")
    return registry


def representation_spec(name: str, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = load_registry() if registry is None else registry
    if name not in registry["representations"]:
        raise KeyError(f"Unknown representation {name!r}")
    return dict(registry["representations"][name])


def output_root(registry: dict[str, Any] | None = None) -> Path:
    registry = load_registry() if registry is None else registry
    return resolve_path(registry["output_root"])


def source_sequence_path(split: str, registry: dict[str, Any] | None = None) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    registry = load_registry() if registry is None else registry
    return resolve_path(registry["source_sequence_root"]) / "dataset" / f"{split}_subject_sequences.npz"


def source_pair_path(split: str, registry: dict[str, Any] | None = None) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    registry = load_registry() if registry is None else registry
    return resolve_path(registry["source_sequence_root"]) / "pairs" / f"{split}_forward_pairs.csv"


def prepared_archive_path(representation: str, split: str, registry: dict[str, Any] | None = None) -> Path:
    return output_root(registry) / "representations" / representation / f"{split}_subject_sequences_128.npz"


def load_source_archive(split: str, registry: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    path = source_sequence_path(split, registry)
    with np.load(path, allow_pickle=False) as loaded:
        archive = {key: loaded[key] for key in loaded.files}
    if any(value.dtype == object for value in archive.values()):
        raise ValueError(f"Pickle-dependent array in {path}")
    return archive


def load_archive(representation: str, split: str, registry: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    path = prepared_archive_path(representation, split, registry)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared representation archive is missing: {path}. Run prepare_representations.py first.")
    with np.load(path, allow_pickle=False) as loaded:
        archive = {key: loaded[key] for key in loaded.files}
    missing = sorted(REQUIRED_ARCHIVE_KEYS.difference(archive))
    if missing:
        raise KeyError(f"{path} is missing fields: {missing}")
    if any(value.dtype == object for value in archive.values()):
        raise ValueError(f"Pickle-dependent array in {path}")
    visits = len(archive["visit_scan_ids"])
    subjects = len(archive["subject_ids"])
    raw = archive["visit_latent_raw_128"]
    standardized = archive["visit_latent_standardized_128"]
    if raw.shape != (visits, LATENT_DIM) or standardized.shape != (visits, LATENT_DIM):
        raise ValueError(f"Unexpected latent shape in {path}: raw={raw.shape}, standardized={standardized.shape}")
    if not np.isfinite(raw).all() or not np.isfinite(standardized).all():
        raise ValueError(f"Non-finite latent in {path}")
    if set(archive["visit_splits"].astype(str)) != {split}:
        raise ValueError(f"Split contamination in {path}")
    diagnoses = archive["visit_diagnoses"].astype(str)
    labels = archive["visit_label_ad"].astype(np.int64)
    if set(diagnoses).difference({"CN", "AD"}) or not np.array_equal(labels, (diagnoses == "AD").astype(np.int64)):
        raise ValueError(f"Diagnosis/label contract violation in {path}")
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    if offsets.shape != (subjects + 1,) or offsets[0] != 0 or offsets[-1] != visits:
        raise ValueError(f"Invalid subject offsets in {path}")
    source = load_source_archive(split, registry)
    if not np.array_equal(archive["visit_scan_ids"].astype(str), source["visit_scan_ids"].astype(str)):
        raise ValueError(f"Prepared/source scan order mismatch in {path}")
    return archive


def validate_split_isolation(archives: dict[str, dict[str, np.ndarray]]) -> None:
    seen_subjects: set[str] = set()
    seen_scans: set[str] = set()
    for split in SPLITS:
        archive = archives[split]
        subjects = set(archive["subject_ids"].astype(str))
        scans = set(archive["visit_scan_ids"].astype(str))
        if subjects & seen_subjects or scans & seen_scans:
            raise ValueError(f"Subject/scan leakage involving split {split}")
        seen_subjects.update(subjects)
        seen_scans.update(scans)


@dataclass(frozen=True)
class PairRow:
    source: int
    target: int
    intermediate: int
    subject: str
    diagnosis: str
    pair_type: str
    delta_years: float


def load_pairs(split: str, archive: dict[str, np.ndarray], registry: dict[str, Any] | None = None) -> list[PairRow]:
    path = source_pair_path(split, registry)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_PAIR_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"Pair schema mismatch in {path}")
        source_rows = list(reader)
    scans = archive["visit_scan_ids"].astype(str)
    subjects = archive["visit_subject_ids"].astype(str)
    diagnoses = archive["visit_diagnoses"].astype(str)
    labels = archive["visit_label_ad"].astype(np.int64)
    orders = archive["visit_orders"].astype(np.int64)
    output: list[PairRow] = []
    seen: set[tuple[int, int]] = set()
    for line, row in enumerate(source_rows, start=2):
        source = int(row["source_index"])
        target = int(row["target_index"])
        middle = int(row["intermediate_index"])
        if row["split"] != split or row["pair_type"] not in {"adjacent", "nonadjacent"}:
            raise ValueError(f"Invalid pair split/type in {path}:{line}")
        if not (0 <= source < len(scans) and 0 <= target < len(scans) and (middle == -1 or 0 <= middle < len(scans))):
            raise ValueError(f"Out-of-range pair index in {path}:{line}")
        if not (subjects[source] == subjects[target] == row["subject_id"]):
            raise ValueError(f"Cross-subject pair in {path}:{line}")
        if not (scans[source] == row["source_scan_id"] and scans[target] == row["target_scan_id"]):
            raise ValueError(f"Scan mismatch in {path}:{line}")
        if not (diagnoses[source] == diagnoses[target] == row["diagnosis"]):
            raise ValueError(f"Diagnosis mismatch in {path}:{line}")
        if not (labels[source] == labels[target] == int(row["label_ad"])):
            raise ValueError(f"Label mismatch in {path}:{line}")
        if orders[target] <= orders[source]:
            raise ValueError(f"Non-forward pair in {path}:{line}")
        if middle >= 0 and not (subjects[middle] == subjects[source] and orders[source] < orders[middle] < orders[target]):
            raise ValueError(f"Bad intermediate visit in {path}:{line}")
        key = (source, target)
        if key in seen:
            raise ValueError(f"Duplicate pair in {path}:{line}")
        seen.add(key)
        output.append(PairRow(source, target, middle, str(subjects[source]), str(diagnoses[source]), row["pair_type"], float(row["delta_years"])))
    if not output:
        raise ValueError(f"No pairs in {path}")
    return output


def collate_pairs(rows: list[PairRow]) -> dict[str, torch.Tensor]:
    return {
        "source": torch.tensor([row.source for row in rows], dtype=torch.long),
        "target": torch.tensor([row.target for row in rows], dtype=torch.long),
        "intermediate": torch.tensor([row.intermediate for row in rows], dtype=torch.long),
    }


def values_on_device(archive: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    latent = torch.from_numpy(archive["visit_latent_standardized_128"].astype(np.float32)).to(device)
    age = torch.from_numpy(archive["visit_age_norm_train"].astype(np.float32)).to(device)
    context = torch.empty_like(latent)
    context_age = torch.empty_like(age)
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    for subject_index in range(len(offsets) - 1):
        first, last = int(offsets[subject_index]), int(offsets[subject_index + 1])
        context[first:last] = latent[first]
        context_age[first:last] = age[first]
    return {
        "z": latent,
        "age": age,
        "years": torch.from_numpy(archive["visit_time_years_from_baseline"].astype(np.float32)).to(device),
        "label": torch.from_numpy(archive["visit_label_ad"].astype(np.float32)).to(device),
        "context": context,
        "context_age": context_age,
    }


def indexed(values: dict[str, torch.Tensor], raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    source = raw["source"].to(values["z"].device)
    target = raw["target"].to(values["z"].device)
    middle = raw["intermediate"].to(values["z"].device)
    return {
        "source": values["z"][source],
        "target": values["z"][target],
        "source_age": values["age"][source],
        "target_age": values["age"][target],
        "source_years": values["years"][source],
        "target_years": values["years"][target],
        "label": values["label"][source],
        "context": values["context"][source],
        "context_age": values["context_age"][source],
        "intermediate_index": middle,
    }


def _add_ae_import_path(registry: dict[str, Any]) -> Path:
    ae_scripts = resolve_path(registry["ae_source_root"]) / "scripts"
    if str(ae_scripts) not in sys.path:
        sys.path.insert(0, str(ae_scripts))
    return ae_scripts


def load_ae_model(name: str, device: torch.device, registry: dict[str, Any] | None = None) -> tuple[nn.Module, dict[str, Any]]:
    registry = load_registry() if registry is None else registry
    spec = representation_spec(name, registry)
    if spec["kind"] not in {"spiral_ae", "adaptive_ae"}:
        raise ValueError(f"{name} is not an AE representation")
    checkpoint_path = resolve_path(spec["checkpoint"])
    actual_hash = sha256(checkpoint_path)
    if actual_hash != spec["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch for {name}: {actual_hash}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(payload["latent_channels"]) != LATENT_DIM:
        raise ValueError(f"{name} checkpoint is not {LATENT_DIM}-D")
    _add_ae_import_path(registry)
    import spiral_common as ae_common
    from network import build_model

    ds_factors = [int(value) for value in payload["ds_factors"]]
    tag = ae_common.ds_tag(ds_factors)
    transform_path = resolve_path(registry["ae_bulk_root"]) / "cache" / f"transform_{tag}.pkl"
    if not transform_path.is_file():
        raise FileNotFoundError(
            f"Required read-only AE hierarchy cache is missing: {transform_path}. "
            "Refusing to generate files in the running AE task."
        )
    with transform_path.open("rb") as handle:
        transform = pickle.load(handle)
    spirals, dynamic, down, up = ae_common.build_spiral_stack(
        transform,
        int(payload["seq_length"]),
        int(payload["dilation"]),
        [int(value) for value in payload["dynamic_seq_lengths"]],
        device,
    )
    model = build_model(
        transform=transform,
        spiral_indices=spirals,
        dynamic_spiral_indices=dynamic,
        down_transform=down,
        up_transform=up,
        out_channels=[int(value) for value in payload["out_channels"]],
        latent_channels=LATENT_DIM,
        conv_type=str(payload["conv_type"]),
        adaptive_levels=int(payload["adaptive_levels"]),
        conv_types=[str(value) for value in payload["conv_types"]],
        dropout=float(payload.get("dropout", 0.0)),
        linear_skip=bool(payload.get("linear_skip", False)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def ae_normalization(registry: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    registry = load_registry() if registry is None else registry
    cache = resolve_path(registry["ae_bulk_root"]) / "cache"
    train_path = cache / "adni_train_V.npy"
    if not train_path.is_file():
        raise FileNotFoundError(f"Required read-only AE train cache is missing: {train_path}")
    train = np.load(train_path, mmap_mode="r")
    mean = np.asarray(train.mean(axis=0), dtype=np.float32)
    std = np.asarray(train.std(axis=0), dtype=np.float32)
    std = np.maximum(std, np.float32(1.0e-8))
    return mean, std


def cached_vertices(split: str, registry: dict[str, Any] | None = None, mmap: bool = True) -> np.ndarray:
    registry = load_registry() if registry is None else registry
    path = resolve_path(registry["ae_bulk_root"]) / "cache" / f"adni_{split}_V.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Required read-only AE vertex cache is missing: {path}")
    return np.load(path, mmap_mode="r" if mmap else None)


class FrozenGeometry(nn.Module):
    """Decoder interface mapping train-standardized latent coordinates to mm vertices."""

    def __init__(self, faces: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("faces", torch.as_tensor(faces, dtype=torch.long))

    def vertices(self, standardized_latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def volume_from_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        faces = self.faces.to(vertices.device)
        v0 = vertices[:, faces[:, 0], :]
        v1 = vertices[:, faces[:, 1], :]
        v2 = vertices[:, faces[:, 2], :]
        signed = torch.sum(v0 * torch.cross(v1, v2, dim=2), dim=2).sum(dim=1) / 6.0
        return torch.abs(signed).clamp_min(1.0e-8)

    def volume(self, standardized_latent: torch.Tensor) -> torch.Tensor:
        return self.volume_from_vertices(self.vertices(standardized_latent))


class Pca128Geometry(FrozenGeometry):
    def __init__(self, model_root: Path, latent_mean: np.ndarray, latent_std: np.ndarray) -> None:
        faces = np.load(model_root / "faces.npy", allow_pickle=False)
        super().__init__(faces)
        mean_flat = np.load(model_root / "mean.npy", allow_pickle=False).astype(np.float32)
        components = np.load(model_root / "components_150.npy", allow_pickle=False).astype(np.float32)[:LATENT_DIM]
        self.register_buffer("mean_flat", torch.from_numpy(mean_flat).view(1, -1))
        self.register_buffer("components", torch.from_numpy(components))
        self.register_buffer("latent_mean", torch.from_numpy(latent_mean.astype(np.float32)).view(1, -1))
        self.register_buffer("latent_std", torch.from_numpy(np.maximum(latent_std, 1.0e-8).astype(np.float32)).view(1, -1))

    def vertices(self, standardized_latent: torch.Tensor) -> torch.Tensor:
        raw = standardized_latent * self.latent_std + self.latent_mean
        flat = raw @ self.components + self.mean_flat
        return flat.reshape(standardized_latent.shape[0], -1, 3)


class FrozenAEGeometry(FrozenGeometry):
    def __init__(
        self,
        model: nn.Module,
        latent_mean: np.ndarray,
        latent_std: np.ndarray,
        mesh_mean: np.ndarray,
        mesh_std: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        super().__init__(faces)
        self.decoder = model
        self.register_buffer("latent_mean", torch.from_numpy(latent_mean.astype(np.float32)).view(1, -1))
        self.register_buffer("latent_std", torch.from_numpy(np.maximum(latent_std, 1.0e-8).astype(np.float32)).view(1, -1))
        self.register_buffer("mesh_mean", torch.from_numpy(mesh_mean.astype(np.float32)))
        self.register_buffer("mesh_std", torch.from_numpy(mesh_std.astype(np.float32)))

    def vertices(self, standardized_latent: torch.Tensor) -> torch.Tensor:
        raw = standardized_latent * self.latent_std + self.latent_mean
        normalized_vertices = self.decoder.decode(raw)
        return normalized_vertices * self.mesh_std + self.mesh_mean


def build_geometry(
    representation: str,
    train_archive: dict[str, np.ndarray],
    device: torch.device,
    registry: dict[str, Any] | None = None,
) -> FrozenGeometry:
    registry = load_registry() if registry is None else registry
    spec = representation_spec(representation, registry)
    latent_mean = train_archive["train_latent_mean_128"]
    latent_std = train_archive["train_latent_std_128"]
    if spec["kind"] == "pca":
        geometry: FrozenGeometry = Pca128Geometry(resolve_path(spec["pca_model_root"]), latent_mean, latent_std)
    else:
        model, _ = load_ae_model(representation, device, registry)
        mesh_mean, mesh_std = ae_normalization(registry)
        pca_model_root = resolve_path(registry["representations"]["pca128"]["pca_model_root"])
        faces = np.load(pca_model_root / "faces.npy", allow_pickle=False)
        geometry = FrozenAEGeometry(model, latent_mean, latent_std, mesh_mean, mesh_std, faces)
    geometry = geometry.to(device)
    geometry.eval()
    for parameter in geometry.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in geometry.parameters()):
        raise RuntimeError("Frozen decoder contains a trainable parameter")
    return geometry


def line_slope(times: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    centered = times - times.mean()
    return torch.sum(centered * (values - values.mean())) / torch.sum(centered.square()).clamp_min(1.0e-8)


def first_last_pairs(archive: dict[str, np.ndarray]) -> list[PairRow]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    subjects = archive["subject_ids"].astype(str)
    diagnoses = archive["subject_diagnoses"].astype(str)
    years = archive["visit_time_years_from_baseline"].astype(np.float64)
    return [
        PairRow(
            int(offsets[index]),
            int(offsets[index + 1] - 1),
            -1,
            str(subjects[index]),
            str(diagnoses[index]),
            "first_last",
            float(years[int(offsets[index + 1] - 1)] - years[int(offsets[index])]),
        )
        for index in range(len(subjects))
    ]


def safe_median(values: Iterable[np.ndarray], floor: float = 1.0e-6) -> float:
    items = list(values)
    if not items:
        return float(floor)
    merged = np.concatenate(items)
    merged = merged[np.isfinite(merged)]
    return float(max(np.median(merged), floor)) if merged.size else float(floor)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def assert_finite_mapping(mapping: dict[str, Any], prefix: str = "") -> None:
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            assert_finite_mapping(value, name)
        elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise RuntimeError(f"Non-finite value at {name}: {value}")
