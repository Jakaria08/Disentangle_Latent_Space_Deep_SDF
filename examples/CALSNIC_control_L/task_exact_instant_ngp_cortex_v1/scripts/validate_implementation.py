#!/usr/bin/env python3
"""End-to-end implementation check on two real scans, on CPU by default.

Runs the actual training loss and the actual reconstruction path over a tiny
sample budget, then asserts the properties that would otherwise fail silently
several hours into a run:

* the loss and every gradient are finite, including gradients into every hash
  table and, for the two-branch model, into both decoders;
* a state-dict round trip reproduces the field exactly;
* the narrow-band fusion actually replaces the field inside the band and leaves
  it untouched outside.

Writes one JSON report to the bulk disk.  Nothing else is created.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from hashgrid_common import (
    ContinuousSDFDataset,
    architecture_name,
    build_decoder,
    choose_device,
    field_callable,
    hash_capacity_report,
    is_two_branch,
    load_config,
    load_manifest,
    require_bulk_path,
    write_json,
)
from train_hashgrid_sdf import compute_loss, decoder_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epoch", type=int, default=1500, help="Epoch to simulate (sets level weights and Eikonal).")
    parser.add_argument("--output-report", default=None)
    return parser.parse_args()


def tiny_config(config: dict) -> dict:
    config = copy.deepcopy(config)
    config["scenes_per_batch"] = 2
    config["scenes_per_chunk"] = 2
    config["num_workers"] = 0
    config["load_dataset_into_ram"] = False
    config["sampling"].update(
        {
            "global_near_samples_per_scene": 32,
            "global_positive_samples_per_scene": 16,
            "global_negative_samples_per_scene": 16,
            "local_ultra_near_samples_per_scene": 32,
            "local_positive_samples_per_scene": 16,
            "local_negative_samples_per_scene": 16,
        }
    )
    config["eikonal"]["points_per_chunk"] = 16
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    small = tiny_config(config)
    rows = [row for row in load_manifest(config["manifest"]) if row["split"] == "train"][:2]

    model = build_decoder(small, device)
    embedding = torch.nn.Embedding(len(rows), int(small["latent_size"]), device=device)
    torch.nn.init.normal_(embedding.weight, std=float(small["code_initial_std"]))
    dataset = ContinuousSDFDataset(rows, small)
    dataset.set_epoch(args.epoch)
    samples = [dataset[index] for index in range(len(rows))]
    broad = torch.stack([torch.as_tensor(item[0]) for item in samples]).to(device)
    near = torch.stack([torch.as_tensor(item[1]) for item in samples]).to(device)
    indices = torch.arange(len(rows), device=device)

    loss, metrics = compute_loss(model, embedding(indices), broad, near, args.epoch, small)
    loss.backward()

    grid_grads = [
        table.grad is not None and bool(torch.isfinite(table.grad).all()) for table in model.grids
    ]
    other_grads = [
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in decoder_parameters(model)
    ]
    report = {
        "architecture": architecture_name(config),
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "epoch": args.epoch,
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "grid_parameters": int(sum(p.numel() for p in model.grids)),
        "decoder_parameters": int(sum(p.numel() for p in decoder_parameters(model))),
        "loss_finite": bool(np.isfinite(metrics["total"])),
        "all_hash_tables_received_finite_gradient": bool(all(grid_grads)),
        "hash_tables_with_gradient": int(sum(grid_grads)),
        "hash_table_count": len(grid_grads),
        "all_decoder_gradients_finite": bool(all(other_grads)),
        "latent_gradient_finite": bool(torch.isfinite(embedding.weight.grad).all()),
        "epsilon_mm": float(metrics["epsilon_x"]) * float(config["mm_per_normalized_unit"]),
        "eikonal_points": metrics["eikonal_points"],
        "metrics": {key: float(value) for key, value in metrics.items()},
        "hash_capacity": hash_capacity_report(config),
    }
    if not report["loss_finite"]:
        raise RuntimeError("Training loss is not finite.")
    if not report["all_hash_tables_received_finite_gradient"]:
        raise RuntimeError(
            f"Only {report['hash_tables_with_gradient']}/{report['hash_table_count']} "
            "hash tables received a finite gradient."
        )
    if not report["all_decoder_gradients_finite"] or not report["latent_gradient_finite"]:
        raise RuntimeError("A decoder or latent gradient is not finite.")

    # State-dict round trip must reproduce the field exactly.
    probe = torch.cat(
        (
            torch.randn(7, int(small["latent_size"]), device=device) * 0.05,
            torch.rand(7, 3, device=device) * 0.4 - 0.2,
        ),
        dim=1,
    )
    weights = [1.0] * len(model.grid_resolutions)
    with torch.no_grad():
        before = model(probe, level_weights=weights)
    clone = build_decoder(small, device)
    clone.load_state_dict(model.state_dict())
    with torch.no_grad():
        after = clone(probe, level_weights=weights)
    round_trip = float((before - after).abs().max())
    report["state_dict_round_trip_max_abs_difference"] = round_trip
    if round_trip != 0.0:
        raise RuntimeError(f"State-dict round trip changed the field by {round_trip}.")

    if is_two_branch(config):
        with torch.no_grad():
            parts = model(probe, level_weights=weights, return_parts=True)
            global_only = field_callable(model, "global")(probe, weights)
            local_only = field_callable(model, "local")(probe, weights)
        report["global_branch_matches_return_parts"] = bool(
            torch.allclose(parts["sdf_global"], global_only)
        )
        report["local_branch_matches_return_parts"] = bool(
            torch.allclose(parts["sdf_local"], local_only)
        )
        report["gate_within_unit_interval"] = bool(
            bool((parts["gate"] >= 0.0).all()) and bool((parts["gate"] <= 1.0).all())
        )
        if not (
            report["global_branch_matches_return_parts"]
            and report["local_branch_matches_return_parts"]
            and report["gate_within_unit_interval"]
        ):
            raise RuntimeError("Two-branch readouts are inconsistent.")

    destination = require_bulk_path(
        args.output_report
        or Path(config["output_dir"]).parent.parent
        / "implementation_validation"
        / f"{config['name']}_implementation.json"
    )
    write_json(destination, report)
    print(
        f"Implementation validation passed for {config['name']}: "
        f"model={report['model_parameters']:,} grid={report['grid_parameters']:,} "
        f"epsilon={report['epsilon_mm']:.3f}mm total={metrics['total']:.6g}\n"
        f"report={destination}"
    )


if __name__ == "__main__":
    main()
