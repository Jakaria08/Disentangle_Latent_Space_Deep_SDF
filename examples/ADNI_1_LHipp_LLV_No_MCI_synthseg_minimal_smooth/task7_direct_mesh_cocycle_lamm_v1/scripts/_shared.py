#!/usr/bin/env python3
"""Paths to the validated direct-mesh objective and latest LAMM components."""

from __future__ import annotations

import sys
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]
COHORT_ROOT = TASK_ROOT.parent
SPIRAL_TASK = COHORT_ROOT / "task5_direct_mesh_cocycle_spiral_unet_v1"
SPIRAL_SCRIPTS = SPIRAL_TASK / "scripts"
LAMM_TASK = COHORT_ROOT / "task_lamm_ae_v1"
LAMM_SCRIPTS = LAMM_TASK / "scripts"


def activate_shared() -> None:
    """Add shared sources after this task so local contracts retain precedence."""
    for path in (SPIRAL_SCRIPTS, LAMM_SCRIPTS):
        value = str(path)
        if value not in sys.path:
            sys.path.append(value)

