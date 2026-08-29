#!/usr/bin/env python3
"""Shared immutable-data and filesystem contracts for the INR-256 task."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = TASK_ROOT.parents[2]
REGISTRY_PATH = TASK_ROOT / "configs" / "inr256_representation.json"
SPLITS = ("train", "val", "test")
LATENT_DIM = 256


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_json(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    value = read_json(path)
    if int(value.get("latent_dim", -1)) != LATENT_DIM:
        raise ValueError(f"Registry must declare latent_dim={LATENT_DIM}")
    if float(value.get("code_bound", 0.0)) <= 0.0:
        raise ValueError("Registry code_bound must be positive")
    return value


def output_root(registry: dict[str, Any] | None = None) -> Path:
    return resolve_path((registry or load_registry())["output_root"])


def archive_path(split: str, registry: dict[str, Any] | None = None) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split}")
    return output_root(registry) / "representations" / "inr256" / f"{split}_subject_sequences_256.npz"


def pair_path(split: str, registry: dict[str, Any] | None = None) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split}")
    return output_root(registry) / "pairs" / f"{split}_forward_pairs.csv"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(value: str, base: int) -> int:
    digest = hashlib.sha256(f"{base}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def atomic_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)


def atomic_torch_save(path: str | Path, value: Any) -> None:
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


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch cannot access a CUDA device")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device.index} is unavailable")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_run_name(value: str) -> None:
    candidate = Path(value)
    if not value or value in {".", ".."} or candidate.name != value or "/" in value or "\\" in value:
        raise ValueError("Run name must be one safe path component")


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def assert_finite_mapping(value: Any, prefix: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_mapping(item, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_mapping(item, f"{prefix}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"Non-finite value at {prefix}: {value}")


REQUIRED_KEYS = {
    "subject_ids", "subject_diagnoses", "subject_label_ad", "subject_visit_offsets",
    "visit_scan_ids", "visit_subject_ids", "visit_splits", "visit_diagnoses",
    "visit_label_ad", "visit_orders", "visit_months_from_baseline",
    "visit_time_years_from_baseline", "visit_age_years", "visit_age_norm_train",
    "visit_volume_mm3", "visit_latent_raw_256", "visit_latent_standardized_256",
    "train_latent_mean_256", "train_latent_std_256", "visit_sdf_xyz",
    "visit_sdf_values", "visit_mesh_path_mm", "visit_sdf_npz_path",
}


def load_archive(split: str, registry: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    path = archive_path(split, registry)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared archive missing: {path}; run prepare_inr256.py")
    with np.load(path, allow_pickle=False) as loaded:
        archive = {key: loaded[key] for key in loaded.files}
    missing = sorted(REQUIRED_KEYS.difference(archive))
    if missing:
        raise KeyError(f"{path} missing fields: {missing}")
    if any(array.dtype == object for array in archive.values()):
        raise ValueError(f"Pickle-dependent object array in {path}")
    visits = len(archive["visit_scan_ids"])
    if archive["visit_latent_raw_256"].shape != (visits, LATENT_DIM):
        raise ValueError("Raw latent shape mismatch")
    if archive["visit_latent_standardized_256"].shape != (visits, LATENT_DIM):
        raise ValueError("Standardized latent shape mismatch")
    if archive["visit_sdf_xyz"].shape[:2] != archive["visit_sdf_values"].shape:
        raise ValueError("SDF sample shape mismatch")
    if set(archive["visit_splits"].astype(str)) != {split}:
        raise ValueError(f"Split contamination in {path}")
    for key in ("visit_latent_raw_256", "visit_latent_standardized_256", "visit_sdf_xyz", "visit_sdf_values"):
        if not np.isfinite(archive[key]).all():
            raise ValueError(f"Non-finite values in {key}")
    return archive


def validate_split_isolation(archives: dict[str, dict[str, np.ndarray]]) -> None:
    subjects: set[str] = set()
    scans: set[str] = set()
    for split in SPLITS:
        current_subjects = set(archives[split]["subject_ids"].astype(str))
        current_scans = set(archives[split]["visit_scan_ids"].astype(str))
        if current_subjects & subjects or current_scans & scans:
            raise ValueError(f"Subject/scan leakage involving {split}")
        subjects.update(current_subjects)
        scans.update(current_scans)


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
    output: list[PairRow] = []
    path = pair_path(split, registry)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source, target = int(row["source_index"]), int(row["target_index"])
            if str(archive["visit_scan_ids"][source]) != row["source_scan_id"] or str(archive["visit_scan_ids"][target]) != row["target_scan_id"]:
                raise ValueError(f"Pair/archive index mismatch in {path}")
            output.append(PairRow(source, target, int(row["intermediate_index"]), row["subject_id"], row["diagnosis"], row["pair_type"], float(row["delta_years"])))
    if not output:
        raise ValueError(f"No pairs in {path}")
    return output


def first_last_pairs(archive: dict[str, np.ndarray]) -> list[PairRow]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    rows = []
    for index in range(len(offsets) - 1):
        start, end = int(offsets[index]), int(offsets[index + 1])
        if end - start < 2:
            continue
        rows.append(PairRow(start, end - 1, start + 1 if end - start > 2 else -1, str(archive["subject_ids"][index]), str(archive["subject_diagnoses"][index]), "first_last", float(archive["visit_time_years_from_baseline"][end - 1] - archive["visit_time_years_from_baseline"][start])))
    return rows


def values_on_device(archive: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    def tensor(key: str, dtype=torch.float32) -> torch.Tensor:
        return torch.from_numpy(np.asarray(archive[key])).to(device=device, dtype=dtype)
    return {
        "z": tensor("visit_latent_standardized_256"),
        "raw_z": tensor("visit_latent_raw_256"),
        "age": tensor("visit_age_norm_train"),
        "time_years": tensor("visit_time_years_from_baseline"),
        "label": tensor("visit_label_ad"),
        "volume": tensor("visit_volume_mm3"),
        "sdf_xyz": tensor("visit_sdf_xyz"),
        "sdf_values": tensor("visit_sdf_values"),
    }


def indexed(values: dict[str, torch.Tensor], rows: list[PairRow]) -> dict[str, torch.Tensor]:
    device = values["z"].device
    source = torch.tensor([row.source for row in rows], device=device, dtype=torch.long)
    target = torch.tensor([row.target for row in rows], device=device, dtype=torch.long)
    intermediate = torch.tensor([row.intermediate for row in rows], device=device, dtype=torch.long)
    valid_middle = intermediate >= 0
    safe_middle = intermediate.clamp_min(0)
    return {
        "source_index": source,
        "target_index": target,
        "intermediate_index": intermediate,
        "valid_middle": valid_middle,
        "source": values["z"][source],
        "target": values["z"][target],
        "middle": values["z"][safe_middle],
        "source_age": values["age"][source],
        "target_age": values["age"][target],
        "middle_age": values["age"][safe_middle],
        "source_years": values["time_years"][source],
        "target_years": values["time_years"][target],
        "label": values["label"][source],
        "source_volume": values["volume"][source],
        "target_volume": values["volume"][target],
        "source_xyz": values["sdf_xyz"][source],
        "source_sdf": values["sdf_values"][source],
        "target_xyz": values["sdf_xyz"][target],
        "target_sdf": values["sdf_values"][target],
    }
