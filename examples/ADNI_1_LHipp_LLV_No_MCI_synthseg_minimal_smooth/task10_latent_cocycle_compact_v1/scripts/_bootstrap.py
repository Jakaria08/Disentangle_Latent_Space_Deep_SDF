#!/usr/bin/env python3
"""Load the tested latent-flow data/evaluation core without changing its sources."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[1]
CORE_SCRIPTS = TASK_ROOT.parent / "task3_latent_flow_128_v1" / "scripts"
REGISTRY = TASK_ROOT / "configs" / "representations.json"
ARCHIVE_ROOTS = {
    "pca128": Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/representations/pca128"),
    "spiralnet128": Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/representations/spiralnet128"),
    "adaptive128": Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/representations/adaptive128"),
    "lamm128": Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/representations/lamm128"),
}

_PATCHED = False


def _patch_lamm_builder(common: Any, registry: dict[str, Any]) -> None:
    common._add_ae_import_path(registry)
    common._add_lamm_import_path(registry)
    import train_lamm

    if getattr(train_lamm.build, "_task10_compatible", False):
        return
    original = train_lamm.build
    defaults = {"semi_amortized": 0, "sa_lr": 3.0e-3, "sa_weight": 1.0, "out_root": None}

    def build(args: Any, device: Any) -> Any:
        for key, value in defaults.items():
            if not hasattr(args, key):
                setattr(args, key, value)
        return original(args, device)

    build._task10_compatible = True
    train_lamm.build = build


def core() -> tuple[Any, Any]:
    """Return task3 ``common`` and ``c4_objective`` with read-only archive routing."""
    global _PATCHED
    os.environ["DEEP3DCOMP_LATENT_FLOW_REGISTRY"] = str(REGISTRY.resolve())
    if str(CORE_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(CORE_SCRIPTS))
    import c4_objective
    import common

    if not _PATCHED:
        def prepared_archive_path(
            representation: str, split: str, registry: dict[str, Any] | None = None
        ) -> Path:
            del registry
            if representation not in ARCHIVE_ROOTS:
                raise KeyError(f"Unknown prepared representation {representation!r}")
            return ARCHIVE_ROOTS[representation] / f"{split}_subject_sequences_128.npz"

        common.prepared_archive_path = prepared_archive_path
        _PATCHED = True
    registry = common.load_registry()
    if "lamm128" in registry["representations"]:
        _patch_lamm_builder(common, registry)
    return common, c4_objective
