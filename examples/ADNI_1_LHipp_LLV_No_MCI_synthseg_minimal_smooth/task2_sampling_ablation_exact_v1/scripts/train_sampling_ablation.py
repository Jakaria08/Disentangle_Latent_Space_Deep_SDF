#!/usr/bin/env python3
"""Run the established multires trainer with the isolated ablation sampler."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPTS = SCRIPT_DIR.parent.parent / "task2_inr_multires_single_field_v1" / "scripts"
for path in (SCRIPT_DIR, BASE_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_multires_sdf as trainer  # noqa: E402
from sampling_dataset import SamplingAblationDataset  # noqa: E402


def main() -> None:
    # This is the only trainer hook: loss, Eikonal, optimizer, checkpointing,
    # validation, decoder, grids and latent handling stay in the established code.
    trainer.ContinuousSDFDataset = SamplingAblationDataset
    trainer.main()


if __name__ == "__main__":
    main()
