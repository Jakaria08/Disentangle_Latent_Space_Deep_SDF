#!/usr/bin/env python3
"""Run a tiny CPU/GPU forward-backward and checkpoint round-trip on real exact SDFs."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from calsnic_common import read_manifest
from delegate_multires import GENERIC_SCRIPTS

sys.path.insert(0, str(GENERIC_SCRIPTS))
from multires_common import (  # noqa: E402
    ContinuousSDFDataset,
    build_decoder,
    choose_device,
    load_config,
    require_bulk_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-report", default=None)
    return parser.parse_args()


def load_compute_loss():
    target = GENERIC_SCRIPTS / "train_multires_sdf.py"
    spec = importlib.util.spec_from_file_location("validated_generic_multires_train", target)
    if spec is None or spec.loader is None:
        raise ImportError(target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_loss


def tiny(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["load_dataset_into_ram"] = False
    result["sampling"].update(
        {
            "global_near_samples_per_scene": 16,
            "global_positive_samples_per_scene": 8,
            "global_negative_samples_per_scene": 8,
            "local_ultra_near_samples_per_scene": 16,
            "local_positive_samples_per_scene": 8,
            "local_negative_samples_per_scene": 8,
        }
    )
    result["eikonal"]["points_per_chunk"] = 8
    return result


def main() -> None:
    args = parse_args()
    config = tiny(load_config(args.config))
    rows = [row for row in read_manifest(config["manifest"]) if row["split"] == "train"][:2]
    if len(rows) != 2:
        raise ValueError("Implementation validation requires two training rows.")
    dataset = ContinuousSDFDataset(rows, config)
    dataset.set_epoch(int(config["total_epochs"]))
    samples = [dataset[index] for index in range(2)]
    broad = torch.stack([item[0] for item in samples])
    near = torch.stack([item[1] for item in samples])
    device = choose_device(args.device)
    broad, near = broad.to(device), near.to(device)
    model = build_decoder(config, device)
    codes = torch.randn(2, int(config["latent_size"]), device=device) * 0.01
    compute_loss = load_compute_loss()
    loss, metrics = compute_loss(model, codes, broad, near, int(config["total_epochs"]), config)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Tiny implementation loss is non-finite.")
    active_gradients = [parameter.grad for parameter in model.grids]
    if not all(gradient is not None and torch.isfinite(gradient).all() for gradient in active_gradients):
        raise RuntimeError("One or more active grids lack finite gradients.")
    restored = build_decoder(config, device)
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        xyz = torch.linspace(-0.2, 0.2, 21, device=device).reshape(7, 3)
        query = torch.cat((codes[:1].expand(7, -1), xyz), dim=1)
        weights = [1.0] * len(model.grid_resolutions)
        if not torch.equal(model(query, level_weights=weights), restored(query, level_weights=weights)):
            raise RuntimeError("In-memory checkpoint round-trip changed predictions.")
    report = {
        "passed": True,
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "scan_ids": [row["scan_id"] for row in rows],
        "broad_shape": list(broad.shape),
        "near_shape": list(near.shape),
        "loss": float(loss.detach().cpu()),
        "metrics": metrics,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "grid_parameters": sum(parameter.numel() for parameter in model.grids),
        "all_active_grid_gradients_finite": True,
        "checkpoint_roundtrip_exact": True,
    }
    output = require_bulk_path(
        args.output_report
        or Path(config["output_dir"]).parent / "implementation_validation" / f"{config['name']}_implementation.json"
    )
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
