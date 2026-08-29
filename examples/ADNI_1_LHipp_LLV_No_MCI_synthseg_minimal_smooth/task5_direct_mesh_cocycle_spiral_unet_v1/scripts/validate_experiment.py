#!/usr/bin/env python3
"""Read-only implementation and prepared-data contract validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import common as C
import data as D
from conditional_spiral_unet import ConditionalSpiralUNet
from mesh_hierarchy import load_hierarchy
from mesh_layers import LearnablePrefixPool, adaptive_modules
from train import validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def adaptive_gradient_check(device: torch.device) -> dict:
    indices = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]], device=device)
    layer = LearnablePrefixPool(8, indices, initial_support=1.5).to(device)
    features = torch.randn(2, 4, 8, device=device, requires_grad=True)
    loss = layer(features).square().mean()
    loss.backward()
    gradient = float(layer.predictor.weight.grad.abs().sum().cpu())
    if not gradient > 0.0:
        raise AssertionError("Adaptive support predictor has zero gradient")
    before = layer.predictor.weight.detach().clone()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    optimizer.step()
    changed = float((layer.predictor.weight.detach() - before).abs().sum().cpu())
    if not changed > 0.0:
        raise AssertionError("Adaptive support predictor did not update")
    return {"gradient_l1": gradient, "parameter_change_l1": changed}


def model_contract(config: dict, root: Path, device: torch.device) -> dict:
    hierarchy = load_hierarchy(root, device)
    statistics = D.load_statistics(root)
    split = D.load_split("val", root, device)
    model = ConditionalSpiralUNet(hierarchy, config, statistics).to(device).eval()
    vertices = split.vertices[:1]
    age = split.ages[:1]
    label = split.labels[:1]
    with torch.no_grad():
        identity = model.transport(vertices, age, age, label)
        identity_error = float((identity - vertices).abs().max().cpu())
        if identity_error != 0.0:
            raise AssertionError(f"Identity is not exact: {identity_error}")
        model.velocity_head.bias[:3].fill_(0.01)
        model.velocity_head.bias[3:].fill_(0.02)
        velocity = model.instantaneous_velocity(vertices, age, label)
        epsilon = torch.full_like(age, 1.0e-3)
        finite = (model.transport(vertices, age, age + epsilon, label) - vertices) / epsilon.reshape(-1, 1, 1)
        finite_difference_error = float(torch.max(torch.abs(finite - velocity)).cpu())
        if finite_difference_error > 5.0e-4:
            raise AssertionError(f"Diagonal velocity finite-difference failure: {finite_difference_error}")
        cn = model.instantaneous_velocity(vertices, age, torch.zeros_like(label))
        ad = model.instantaneous_velocity(vertices, age, torch.ones_like(label))
        disease_difference = float((ad - cn).abs().mean().cpu())
        if disease_difference <= 0.0:
            raise AssertionError("Disease condition does not affect velocity")
    return {
        "config": config["name"],
        "operator": config["model"]["operator"],
        "parameters": C.parameter_count(model),
        "identity_max_abs_mm": identity_error,
        "finite_difference_max_abs_mm_per_year": finite_difference_error,
        "disease_condition_mean_abs_difference": disease_difference,
        "adaptive_modules": len(adaptive_modules(model)),
    }


def main() -> int:
    args = parse_args()
    root = C.output_root(args.output_root)
    device = C.choose_device(args.device)
    splits = {split: D.load_split(split, root) for split in C.SPLITS}
    D.verify_split_isolation(splits)
    results = []
    for name in (
        "spiral_direct_c4_s42.json",
        "adaptive_direct_c4_s42.json",
        "spiral_direct_c4_velocity_v2_s42.json",
        "adaptive_direct_c4_velocity_v2_s42.json",
    ):
        config = C.read_json(C.TASK_ROOT / "configs" / name)
        validate_config(config)
        results.append(model_contract(config, root, device))
    report = {
        "status": "passed",
        "split_scans": {name: len(value.scan_ids) for name, value in splits.items()},
        "split_subjects": {name: len(set(value.subject_ids.astype(str))) for name, value in splits.items()},
        "models": results,
        "adaptive_gradient": adaptive_gradient_check(device),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
