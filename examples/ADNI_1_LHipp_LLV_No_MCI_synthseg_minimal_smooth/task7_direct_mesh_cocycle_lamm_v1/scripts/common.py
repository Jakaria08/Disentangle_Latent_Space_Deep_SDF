#!/usr/bin/env python3
"""Contracts for the isolated end-to-end direct LAMM surface-flow task."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parents[1]
COHORT_TASK_ROOT = TASK_ROOT.parent
SPIRAL_TASK_ROOT = COHORT_TASK_ROOT / "task5_direct_mesh_cocycle_spiral_unet_v1"
_BASE_PATH = SPIRAL_TASK_ROOT / "scripts" / "common.py"
_SPEC = importlib.util.spec_from_file_location("_direct_spiral_common_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load shared direct-mesh contracts from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

# Cohort and data contracts are deliberately identical to the direct Spiral task.
COHORT_ROOT = _BASE.COHORT_ROOT
SOURCE_SEQUENCE_ROOT = _BASE.SOURCE_SEQUENCE_ROOT
SOURCE_MANIFEST = _BASE.SOURCE_MANIFEST
AE_TASK_ROOT = _BASE.AE_TASK_ROOT
AE_BULK_ROOT = _BASE.AE_BULK_ROOT
SPLITS = _BASE.SPLITS
VERTEX_COUNT = _BASE.VERTEX_COUNT
FACE_COUNT = _BASE.FACE_COUNT
PairRow = _BASE.PairRow

DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "DEEP3DCOMP_DIRECT_LAMM_ROOT",
        "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task7_direct_mesh_cocycle_lamm_v1",
    )
)
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "DEEP3DCOMP_DIRECT_LAMM_DATA_ROOT",
        "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1",
    )
)
_DATA_ROOT_OVERRIDE: Path | None = None


def _safe_root(path: Path, purpose: str) -> Path:
    value = path.expanduser().resolve()
    if value in {Path("/"), Path("/mnt"), Path("/mnt/bulk10tb")}:
        raise ValueError(f"Refusing unsafe {purpose} root: {value}")
    return value


def output_root(value: str | Path | None = None) -> Path:
    return _safe_root(DEFAULT_OUTPUT_ROOT if value is None else Path(value), "output")


def configure_data_root(value: str | Path | None) -> Path:
    global _DATA_ROOT_OVERRIDE
    _DATA_ROOT_OVERRIDE = None if value is None else _safe_root(Path(value), "data")
    return data_root()


def data_root(value: str | Path | None = None) -> Path:
    if value is not None:
        return _safe_root(Path(value), "data")
    return _safe_root(_DATA_ROOT_OVERRIDE or DEFAULT_DATA_ROOT, "data")


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    candidate = (Path.cwd() / value).resolve()
    if candidate.exists():
        return candidate
    return (TASK_ROOT / value).resolve()


# Re-export the tested, path-independent utilities and the immutable cohort readers.
def read_json(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


atomic_json = _BASE.atomic_json
atomic_torch_save = _BASE.atomic_torch_save
sha256 = _BASE.sha256
set_seed = _BASE.set_seed
choose_device = _BASE.choose_device
parameter_count = _BASE.parameter_count
read_manifest = _BASE.read_manifest
source_sequence_path = _BASE.source_sequence_path
source_pair_path = _BASE.source_pair_path
load_pairs = _BASE.load_pairs
balanced_pair_indices = _BASE.balanced_pair_indices
chunked = _BASE.chunked


def contract_summary() -> dict[str, Any]:
    return {
        "task_root": str(TASK_ROOT),
        "output_root": str(output_root()),
        "data_root": str(data_root()),
        "source_manifest": str(SOURCE_MANIFEST),
        "vertex_count": VERTEX_COUNT,
        "face_count": FACE_COUNT,
    }
