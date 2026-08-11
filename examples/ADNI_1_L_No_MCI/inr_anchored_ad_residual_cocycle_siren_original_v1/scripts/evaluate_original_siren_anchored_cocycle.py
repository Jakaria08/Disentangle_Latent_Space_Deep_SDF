#!/usr/bin/env python3
"""Evaluate original-cohort checkpoints with the shared metric implementation."""

from __future__ import annotations

from original_siren_adapter import experiment_dir

import evaluate_large_siren_anchored_cocycle as implementation  # type: ignore  # noqa: E402


implementation.experiment_dir = experiment_dir


if __name__ == "__main__":
    raise SystemExit(implementation.main())
