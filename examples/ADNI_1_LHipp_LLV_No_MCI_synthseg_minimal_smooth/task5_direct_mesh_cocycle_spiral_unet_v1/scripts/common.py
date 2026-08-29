#!/usr/bin/env python3
"""Shared contracts and small utilities for the direct surface-cocycle task."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


TASK_ROOT = Path(__file__).resolve().parents[1]
COHORT_ROOT = TASK_ROOT.parent / "hippocampus_pca_cocycle_v4"
SOURCE_SEQUENCE_ROOT = COHORT_ROOT / "cocycle_v4"
SOURCE_MANIFEST = COHORT_ROOT / "metadata" / "hippocampus_qc_keep_manifest.csv"
AE_TASK_ROOT = TASK_ROOT.parent / "task_spiral_ae_v1"
AE_BULK_ROOT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1")
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "DEEP3DCOMP_DIRECT_MESH_ROOT",
        "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1",
    )
)
SPLITS = ("train", "val", "test")
VERTEX_COUNT = 2746
FACE_COUNT = 5488


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    candidate = (Path.cwd() / value).resolve()
    if candidate.exists():
        return candidate
    return (TASK_ROOT / value).resolve()


def output_root(value: str | Path | None = None) -> Path:
    path = DEFAULT_OUTPUT_ROOT if value is None else Path(value).expanduser().resolve()
    if path == Path("/mnt/bulk10tb") or path == Path("/"):
        raise ValueError(f"Refusing unsafe output root: {path}")
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested, but CUDA is unavailable")
    device = torch.device(requested)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def read_manifest() -> list[dict[str, str]]:
    with SOURCE_MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty source manifest: {SOURCE_MANIFEST}")
    if {int(row["vertex_count"]) for row in rows} != {VERTEX_COUNT}:
        raise ValueError("Unexpected vertex count in source manifest")
    if {int(row["face_count"]) for row in rows} != {FACE_COUNT}:
        raise ValueError("Unexpected face count in source manifest")
    if len({row["correspondence_topology_hash"] for row in rows}) != 1:
        raise ValueError("Source meshes do not share one topology hash")
    return rows


def source_sequence_path(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return SOURCE_SEQUENCE_ROOT / "dataset" / f"{split}_subject_sequences.npz"


def source_pair_path(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return SOURCE_SEQUENCE_ROOT / "pairs" / f"{split}_forward_pairs.csv"


@dataclass(frozen=True)
class PairRow:
    source: int
    target: int
    intermediate: int
    subject: str
    diagnosis: str
    pair_type: str
    delta_years: float


def load_pairs(split: str, scan_ids: np.ndarray, subjects: np.ndarray, labels: np.ndarray) -> list[PairRow]:
    with source_pair_path(split).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: list[PairRow] = []
    seen: set[tuple[int, int]] = set()
    for line, row in enumerate(rows, start=2):
        source = int(row["source_index"])
        target = int(row["target_index"])
        middle = int(row["intermediate_index"])
        if row["split"] != split or row["pair_type"] not in {"adjacent", "nonadjacent"}:
            raise ValueError(f"Invalid pair row at {source_pair_path(split)}:{line}")
        if not (0 <= source < len(scan_ids) and 0 <= target < len(scan_ids)):
            raise IndexError(f"Pair index outside split at line {line}")
        if str(scan_ids[source]) != row["source_scan_id"] or str(scan_ids[target]) != row["target_scan_id"]:
            raise ValueError(f"Pair scan order mismatch at line {line}")
        if str(subjects[source]) != str(subjects[target]) or str(subjects[source]) != row["subject_id"]:
            raise ValueError(f"Cross-subject pair at line {line}")
        if int(labels[source]) != int(labels[target]) or int(labels[source]) != int(row["label_ad"]):
            raise ValueError(f"Diagnosis-label mismatch at line {line}")
        if middle >= 0 and str(subjects[middle]) != str(subjects[source]):
            raise ValueError(f"Cross-subject intermediate visit at line {line}")
        key = (source, target)
        if key in seen:
            raise ValueError(f"Duplicate pair at line {line}")
        seen.add(key)
        output.append(
            PairRow(
                source=source,
                target=target,
                intermediate=middle,
                subject=str(row["subject_id"]),
                diagnosis=str(row["diagnosis"]),
                pair_type=str(row["pair_type"]),
                delta_years=float(row["delta_years"]),
            )
        )
    if not output:
        raise ValueError(f"No pairs for split {split}")
    return output


def balanced_pair_indices(rows: list[PairRow], samples: int, seed: int) -> list[int]:
    """Equal mass over diagnosis, interval type, and subjects within each stratum."""
    buckets: dict[tuple[str, str, str], list[int]] = {}
    for index, row in enumerate(rows):
        buckets.setdefault((row.diagnosis, row.pair_type, row.subject), []).append(index)
    strata: dict[tuple[str, str], list[str]] = {}
    for diagnosis, pair_type, subject in buckets:
        strata.setdefault((diagnosis, pair_type), []).append(subject)
    required = {(d, p) for d in ("CN", "AD") for p in ("adjacent", "nonadjacent")}
    if set(strata) != required:
        raise ValueError(f"Missing pair sampling strata: {sorted(required.difference(strata))}")
    rng = random.Random(int(seed))
    output = []
    ordered = sorted(required)
    while len(output) < int(samples):
        for key in ordered:
            subject = rng.choice(strata[key])
            output.append(rng.choice(buckets[(key[0], key[1], subject)]))
            if len(output) == int(samples):
                break
    rng.shuffle(output)
    return output


def chunked(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    current: list[Any] = []
    for value in values:
        current.append(value)
        if len(current) == int(size):
            yield current
            current = []
    if current:
        yield current

