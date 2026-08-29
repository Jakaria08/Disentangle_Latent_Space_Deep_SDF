#!/usr/bin/env python3
"""Read-only validation of data, layout, identity, condition, and latent-width contracts."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

import common as C
import data as D
from train import build_model, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path, default=None)
    return parser.parse_args()


def check_model(config: dict, device: torch.device) -> dict:
    model, _ = build_model(config, C.output_root(), device)
    split = D.load_split("val", device=device)
    vertices = split.vertices[:1]
    age = split.ages[:1]
    label = split.labels[:1]
    model.eval()
    with torch.no_grad():
        latent, condition = model.encode(vertices, age, age, label)
        if latent.shape != (1, int(config["model"]["latent_dim"])):
            raise AssertionError(f"Wrong latent shape: {tuple(latent.shape)}")
        if not torch.isfinite(latent).all() or not torch.isfinite(condition).all():
            raise AssertionError("Non-finite encoder output")
        identity = model.transport(vertices, age, age, label)
        identity_error = float((identity - vertices).abs().max().cpu())
        if identity_error != 0.0:
            raise AssertionError(f"Identity is not exact: {identity_error}")
        model.set_test_velocity_bias(0.01, 0.02)
        velocity = model.instantaneous_velocity(vertices, age, label)
        epsilon = torch.full_like(age, 1.0e-3)
        finite = (
            model.transport(vertices, age, age + epsilon, label) - vertices
        ) / epsilon.reshape(-1, 1, 1)
        finite_error = float((finite - velocity).abs().max().cpu())
        if finite_error > 5.0e-4:
            raise AssertionError(f"Instantaneous finite-difference mismatch: {finite_error}")
        cn = model.instantaneous_velocity(vertices, age, torch.zeros_like(label))
        ad = model.instantaneous_velocity(vertices, age, torch.ones_like(label))
        disease_difference = float((ad - cn).abs().mean().cpu())
        if disease_difference <= 0.0:
            raise AssertionError("Diagnosis does not affect surface velocity")
    output = {
        "name": config["name"],
        "latent_shape": list(latent.shape),
        "latent_split": model.latent_split,
        "parameters": C.parameter_count(model),
        "parameter_breakdown": model.parameter_breakdown(),
        "layout_fingerprint": model.layout_fingerprint,
        "identity_max_abs_mm": identity_error,
        "finite_difference_max_abs_mm_per_year": finite_error,
        "disease_velocity_difference_mm_per_year": disease_difference,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def main() -> int:
    args = parse_args()
    C.configure_data_root(args.data_root)
    device = C.choose_device(args.device)
    splits = {name: D.load_split(name) for name in C.SPLITS}
    D.verify_split_isolation(splits)
    models = []
    for name in (
        "lamm_direct_c4_z128_s42.json",
        "lamm_direct_c4_z256_equal_s42.json",
        "lamm_direct_c4_z256_fine_s42.json",
    ):
        config = C.read_json(C.TASK_ROOT / "configs" / name)
        validate_config(config)
        models.append(check_model(config, device))
    report = {
        "status": "passed",
        "data_root": str(C.data_root()),
        "split_scans": {name: len(value.scan_ids) for name, value in splits.items()},
        "split_subjects": {
            name: len(set(value.subject_ids.astype(str))) for name, value in splits.items()
        },
        "models": models,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

