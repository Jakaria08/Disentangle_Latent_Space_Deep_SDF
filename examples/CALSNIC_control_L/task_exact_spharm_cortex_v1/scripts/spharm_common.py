#!/usr/bin/env python3
"""Shared setup for the SPHARM task: reuse the multires task's manifest/mesh/eval helpers.

This task adds only spherical-embedding and SPHARM fit/evaluate code; it does
not rebuild the manifest, exact SDF archives, or PCA basis, all of which
already exist under ``task_exact_multires_cortex_v1``. Importing from there
(rather than copying) keeps a single source of truth for the manifest schema,
the SDF-space <-> millimetre transform, and the metric stack.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
CALSNIC_ROOT = TASK_DIR.parent
MULTIRES_SCRIPTS = CALSNIC_ROOT / "task_exact_multires_cortex_v1" / "scripts"
if not (MULTIRES_SCRIPTS / "calsnic_common.py").is_file():
    raise FileNotFoundError(f"Expected sibling task scripts at {MULTIRES_SCRIPTS}")
sys.path.insert(0, str(MULTIRES_SCRIPTS))

from calsnic_common import (  # noqa: E402
    DEFAULT_EXACT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    load_mesh,
    load_sdf_space_mesh,
    mesh_sdf_to_mm,
    read_manifest,
    require_bulk_path,
    resolve_path,
    sha256_file,
)

DEFAULT_PCA_DIR = DEFAULT_OUTPUT_ROOT / "pca" / "matched_left_train173"
DEFAULT_SPHARM_DIR = DEFAULT_OUTPUT_ROOT / "spharm" / "anchored_harmonic_all203"

__all__ = [
    "DEFAULT_EXACT_MANIFEST",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PCA_DIR",
    "DEFAULT_SPHARM_DIR",
    "MULTIRES_SCRIPTS",
    "atomic_write_csv",
    "atomic_write_json",
    "load_mesh",
    "load_sdf_space_mesh",
    "mesh_sdf_to_mm",
    "read_manifest",
    "require_bulk_path",
    "resolve_path",
    "sha256_file",
]
