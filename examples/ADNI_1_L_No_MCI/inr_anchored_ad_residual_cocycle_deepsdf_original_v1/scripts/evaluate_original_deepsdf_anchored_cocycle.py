#!/usr/bin/env python3
"""Export original-cohort DeepSDF base-versus-calibrated metrics."""

from __future__ import annotations

from original_deepsdf_adapter import experiment_dir

import evaluate_large_siren_anchored_cocycle as implementation  # type: ignore  # noqa: E402


implementation.experiment_dir = experiment_dir


if __name__ == "__main__":
    raise SystemExit(implementation.main())
