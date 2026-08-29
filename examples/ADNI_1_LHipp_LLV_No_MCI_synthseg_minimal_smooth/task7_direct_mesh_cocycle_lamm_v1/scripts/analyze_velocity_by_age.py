#!/usr/bin/env python3
"""Run the shared CN/AD instantaneous surface-velocity age analysis for LAMM."""

from __future__ import annotations

import runpy

from _shared import SPIRAL_SCRIPTS, activate_shared


if __name__ == "__main__":
    activate_shared()
    runpy.run_path(
        str(SPIRAL_SCRIPTS / "analyze_velocity_by_age.py"), run_name="__main__"
    )

