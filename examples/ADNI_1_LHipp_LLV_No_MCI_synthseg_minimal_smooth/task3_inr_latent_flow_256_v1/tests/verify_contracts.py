#!/usr/bin/env python3
"""Fast contracts for the isolated INR-256 task."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "scripts"))

import common as C  # noqa: E402
from inr_geometry import build_geometry  # noqa: E402
from models import BrainODEAttentionFunc, DirectC4Flow, build_ode  # noqa: E402


def main():
    registry = C.load_registry()
    archives = {split: C.load_archive(split, registry) for split in C.SPLITS}
    C.validate_split_isolation(archives)
    assert {split: len(archives[split]["visit_scan_ids"]) for split in C.SPLITS} == {"train": 401, "val": 100, "test": 100}
    assert {split: len(archives[split]["subject_ids"]) for split in C.SPLITS} == {"train": 91, "val": 20, "test": 20}
    configs = [C.read_json(TASK / "configs" / name) for name in ("inr256_direct_c4_s42.json", "inr256_plain_ode_s42.json", "inr256_brainode_s42.json")]
    assert {config["method"] for config in configs} == {"direct_c4", "plain_ode", "brainode"}
    assert all(int(config["model"]["latent_dim"]) == C.LATENT_DIM for config in configs)
    flow = DirectC4Flow(C.LATENT_DIM, 32, 1)
    z = torch.randn(4, C.LATENT_DIM)
    source, target, label = torch.zeros(4), torch.ones(4), torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.equal(flow.transport(z, source, target, label), z)
    brain = BrainODEAttentionFunc(C.LATENT_DIM, 32, 64, 0.0).eval()
    alone = brain(source[:1], z[:1], label[:1])
    together = brain(source, z, label)[:1]
    assert torch.allclose(alone, together, atol=1.0e-7, rtol=1.0e-7)
    geometry = build_geometry(archives["train"], torch.device("cpu"), registry)
    latent = torch.from_numpy(archives["train"]["visit_latent_standardized_256"][:1]).requires_grad_(True)
    xyz = torch.from_numpy(archives["train"]["visit_sdf_xyz"][:1, :8])
    loss = geometry.sdf(latent, xyz).mean()
    loss.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all() and float(latent.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in geometry.decoder.parameters())
    print("CONTRACT CHECKS PASSED: 601 scans, disjoint splits, 256-D models, frozen differentiable INR, singleton BrainODE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
