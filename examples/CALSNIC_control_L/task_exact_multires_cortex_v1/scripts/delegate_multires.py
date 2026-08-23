#!/usr/bin/env python3
"""Delegate to the tested ADNI multires engine while retaining CALSNIC wrappers."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
GENERIC_SCRIPTS = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth"
    / "task2_inr_multires_single_field_v1"
    / "scripts"
)


def delegate(script_name: str) -> None:
    target = (GENERIC_SCRIPTS / script_name).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    sys.path.insert(0, str(GENERIC_SCRIPTS))
    runpy.run_path(str(target), run_name="__main__")
