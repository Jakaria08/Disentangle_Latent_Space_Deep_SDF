#!/usr/bin/env python3
"""Read the validated direct-mesh cache without duplicating 150 MB of surfaces."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import common as C


_BASE_PATH = C.SPIRAL_TASK_ROOT / "scripts" / "data.py"
_SPEC = importlib.util.spec_from_file_location("_direct_spiral_data_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load shared data implementation from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

PreparedSplit = _BASE.PreparedSplit


def split_cache_path(split: str, root: Path | None = None) -> Path:
    return _BASE.split_cache_path(split, C.data_root(root))


def statistics_path(root: Path | None = None) -> Path:
    return _BASE.statistics_path(C.data_root(root))


def load_split(split: str, root: Path | None = None, device="cpu") -> PreparedSplit:
    # Existing evaluators pass their output root here. Data ownership is intentionally
    # separate, so the configured direct-mesh data root always wins.
    del root
    return _BASE.load_split(split, C.data_root(), device)


def load_statistics(root: Path | None = None):
    del root
    return _BASE.load_statistics(C.data_root())


verify_split_isolation = _BASE.verify_split_isolation

