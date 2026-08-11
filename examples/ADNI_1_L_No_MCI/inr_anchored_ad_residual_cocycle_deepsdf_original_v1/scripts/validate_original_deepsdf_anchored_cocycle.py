#!/usr/bin/env python3
"""Validate original DeepSDF inputs, cache, and the anchor invariant."""

from __future__ import annotations

from original_deepsdf_adapter import experiment_dir, load_metadata_and_latents

import validate_large_siren_anchored_cocycle as implementation  # type: ignore  # noqa: E402


implementation.experiment_dir = experiment_dir
implementation.load_metadata_and_latents = load_metadata_and_latents


if __name__ == "__main__":
    raise SystemExit(implementation.main())
