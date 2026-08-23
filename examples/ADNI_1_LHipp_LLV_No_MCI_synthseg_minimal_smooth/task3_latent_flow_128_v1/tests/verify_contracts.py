#!/usr/bin/env python3
"""Static/synthetic scientific-contract checks; requires no prepared data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "scripts"))

import common as C  # noqa: E402
from models import BrainODEAttentionFunc, build_ode  # noqa: E402
from train_c4 import validate_config as validate_c4  # noqa: E402
from train_latent_ode import validate_config as validate_ode  # noqa: E402


def main() -> int:
    registry = C.load_registry()
    assert set(registry["representations"]) == {"pca128", "spiralnet128", "adaptive128"}
    configs = sorted((TASK_ROOT / "configs").glob("*_s42.json"))
    assert len(configs) == 9, len(configs)
    for path in configs:
        config = json.loads(path.read_text())
        if config["method"] == "direct_c4":
            validate_c4(config)
            assert config["loss"]["coboundary_weight"] == 0.0
        else:
            validate_ode(config)
            assert set(config["loss"]) == {"latent_trajectory_mse_weight"}
            model = build_ode(config)
            assert sum(parameter.numel() for parameter in model.parameters()) > 0

    brain = BrainODEAttentionFunc(128, 32, 64, 0.0).eval()
    torch.manual_seed(19)
    latent = torch.randn(3, 128)
    time = torch.full((3,), 0.2)
    condition = torch.tensor([0.0, 1.0, 0.0])
    alone = brain(time[:1], latent[:1], condition[:1])
    together = brain(time, latent, condition)[:1]
    assert torch.equal(alone, together), torch.max(torch.abs(alone - together))
    print("CONTRACT CHECKS PASSED: 3x3 configs, latent-only ODE, direct/no-coboundary C4, singleton BrainODE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
