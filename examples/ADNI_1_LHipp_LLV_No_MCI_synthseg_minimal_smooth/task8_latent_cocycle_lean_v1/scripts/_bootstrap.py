#!/usr/bin/env python3
"""Route this isolated task through the tested generic 128-D latent-flow core.

Training code here is new; data loading, frozen-decoder geometry, archive validation
and *all evaluation* are reused verbatim from ``task3_latent_flow_128_v1`` so that any
number reported here is directly comparable to the published direct_c4 baselines.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[1]
CORE_SCRIPTS = TASK_ROOT.parent / "task3_latent_flow_128_v1" / "scripts"
REGISTRY = TASK_ROOT / "configs" / "representations.json"

# Baseline runs to compare against, keyed by representation.
BASELINE_RUNS: dict[str, str] = {
    "pca128": "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/training/pca128/direct_c4/pca128_direct_c4_s42",
    "spiralnet128": "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/training/spiralnet128/direct_c4/spiralnet128_direct_c4_s42",
    "adaptive128": "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/training/adaptive128/direct_c4/adaptive128_direct_c4_s42",
    "lamm128": "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/training/lamm128/direct_c4/lamm128_direct_c4_s42",
}

REPRESENTATIONS = ("pca128", "spiralnet128", "adaptive128", "lamm128")

_PATCHED = False


def activate() -> None:
    os.environ["DEEP3DCOMP_LATENT_FLOW_REGISTRY"] = str(REGISTRY.resolve())
    if str(CORE_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(CORE_SCRIPTS))


def patch_lamm_builder(registry: dict[str, Any]) -> None:
    """Default the training-only fields that ``train_lamm.build`` gained after the
    frozen LAMM checkpoints were written.

    ``train_lamm.py`` acquired ``--semi-amortized``/``--sa-lr``/``--sa-weight`` and now
    echoes them from ``build``.  Checkpoints saved before that change carry no such args,
    so rebuilding their architecture raises AttributeError.  The three fields are
    training-only: they never reach a module constructor, and ``_load_lamm_model`` still
    finishes with ``load_state_dict(..., strict=True)``, which remains the real guarantee
    that the rebuilt decoder matches the checkpoint.  Patching here keeps the shared
    ``task_lamm_ae_v1`` and ``task3_latent_flow_128_v1`` sources untouched.
    """
    global _PATCHED
    if _PATCHED:
        return
    import common as C

    C._add_ae_import_path(registry)
    C._add_lamm_import_path(registry)
    import train_lamm

    original = train_lamm.build
    defaults = {"semi_amortized": 0, "sa_lr": 3.0e-3, "sa_weight": 1.0, "out_root": None}

    def build(args, device):  # noqa: ANN001, ANN202
        for key, value in defaults.items():
            if not hasattr(args, key):
                setattr(args, key, value)
        return original(args, device)

    train_lamm.build = build
    _PATCHED = True


def core():
    """Activate the core path and return ``(common, c4_objective)``."""
    activate()
    import c4_objective
    import common

    registry = common.load_registry()
    if any(
        str(spec.get("kind")) == "lamm_ae"
        for spec in registry["representations"].values()
    ):
        patch_lamm_builder(registry)
    return common, c4_objective
