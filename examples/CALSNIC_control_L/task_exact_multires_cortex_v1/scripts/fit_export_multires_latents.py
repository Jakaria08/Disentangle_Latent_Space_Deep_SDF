#!/usr/bin/env python3
"""Export CALSNIC multires latents with an explicit locked-test confirmation."""

from __future__ import annotations

import sys

from delegate_multires import delegate


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        delegate("fit_export_multires_latents.py")
    confirmed = "--confirm-test" in sys.argv
    if confirmed:
        sys.argv.remove("--confirm-test")
    if "--splits" in sys.argv:
        start = sys.argv.index("--splits") + 1
        splits = []
        for value in sys.argv[start:]:
            if value.startswith("--"):
                break
            splits.append(value)
    else:
        splits = ["train", "val", "test"]
    if "test" in splits and not confirmed:
        raise PermissionError("Test latent fitting is locked; add --confirm-test after model selection.")
    delegate("fit_export_multires_latents.py")
