#!/usr/bin/env python3
"""Route this isolated task through the tested generic 128-D flow core."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK_ROOT.parents[2]
CORE_SCRIPTS = TASK_ROOT.parent / "task3_latent_flow_128_v1" / "scripts"
SURFACE_SCRIPTS = (
    TASK_ROOT.parent / "task3_pca_corrective_cocycle_128_v1" / "scripts"
)
REGISTRY = TASK_ROOT / "configs" / "representations.json"


def activate() -> None:
    os.environ["DEEP3DCOMP_LATENT_FLOW_REGISTRY"] = str(REGISTRY.resolve())
    if str(CORE_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(CORE_SCRIPTS))


def run_core(script: str) -> None:
    activate()
    runpy.run_path(str(CORE_SCRIPTS / script), run_name="__main__")


def run_surface() -> None:
    activate()
    if str(SURFACE_SCRIPTS) not in sys.path:
        sys.path.insert(1, str(SURFACE_SCRIPTS))
    # The shared surface evaluator supports an optional corrective-PCA decoder.
    # This task has no such branch, so provide a never-matching compatibility
    # type while retaining the generic frozen-AE geometry.
    import common

    if not hasattr(common, "FrozenCorrectiveGeometry"):
        class FrozenCorrectiveGeometry:  # pragma: no cover - compatibility only
            pass

        common.FrozenCorrectiveGeometry = FrozenCorrectiveGeometry
    runpy.run_path(
        str(SURFACE_SCRIPTS / "evaluate_surface_forecasts.py"),
        run_name="__main__",
    )
