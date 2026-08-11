#!/usr/bin/env python3
"""Build the original-cohort DeepSDF cache in this experiment directory."""

from __future__ import annotations

from original_deepsdf_adapter import ROOT, experiment_dir, load_metadata_and_latents
from inr_anchored_common import load_frozen_base_flow as shared_load_frozen_base_flow

import build_large_siren_cache as implementation  # type: ignore  # noqa: E402


implementation.experiment_dir = experiment_dir
implementation.load_metadata_and_latents = load_metadata_and_latents
implementation.load_frozen_base_flow = lambda config, device: shared_load_frozen_base_flow(config, device, ROOT)


if __name__ == "__main__":
    raise SystemExit(implementation.main())
