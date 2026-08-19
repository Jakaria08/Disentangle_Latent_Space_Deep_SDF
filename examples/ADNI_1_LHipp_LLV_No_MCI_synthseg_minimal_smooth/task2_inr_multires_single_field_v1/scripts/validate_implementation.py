#!/usr/bin/env python3
"""Exercise data sampling, both loss paths, checkpoint reload, and mesh writing."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from multires_common import (
    ContinuousSDFDataset,
    atomic_torch_save,
    build_decoder,
    choose_device,
    decode_latent_to_mesh,
    load_config,
    load_manifest,
    read_scaling,
    require_bulk_path,
    write_json,
)
from train_multires_sdf import compute_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Primary no-Eikonal config.")
    parser.add_argument("--eik-config", required=True, help="Matched Eikonal ablation config.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def tiny_config(config: dict) -> dict:
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


class AnalyticSphere(torch.nn.Module):
    grid_resolutions = (8,)

    def forward(self, input_x, level_weights=None):
        del level_weights
        return torch.linalg.vector_norm(input_x[:, -3:], dim=1, keepdim=True) - 0.5


def main() -> None:
    args = parse_args()
    primary = tiny_config(load_config(args.config))
    eikonal = tiny_config(load_config(args.eik_config))
    if primary["network_specs"] != eikonal["network_specs"]:
        raise ValueError("Primary and Eikonal configs must use the identical architecture.")
    if primary["eikonal"]["enabled"] or not eikonal["eikonal"]["enabled"]:
        raise ValueError("Expected a no-Eikonal primary and an enabled Eikonal ablation.")
    device = choose_device(args.device)
    output = require_bulk_path(
        args.output_dir
        or Path(primary["_output_dir"]).parent / "implementation_validation"
    )

    rows = [row for row in load_manifest(primary["manifest"]) if row["split"] == "train"][:2]
    dataset = ContinuousSDFDataset(rows, primary)
    dataset.set_epoch(650)
    worker_count = int(primary["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=worker_count,
        persistent_workers=True,
    )
    broad, near, _indices = next(iter(loader))
    broad = broad.to(device)
    near = near.to(device)
    codes = torch.randn(2, int(primary["latent_size"]), device=device) * 0.01

    model = build_decoder(primary, device)
    model.zero_grad(set_to_none=True)
    primary_loss, primary_metrics = compute_loss(
        model, codes, broad, near, 650, primary
    )
    primary_loss.backward()
    if not torch.isfinite(primary_loss):
        raise RuntimeError("Primary loss is non-finite.")
    if not all(grid.grad is not None and torch.isfinite(grid.grad).all() for grid in model.grids):
        raise RuntimeError("Primary loss did not produce finite gradients for every active grid.")

    eik_model = build_decoder(eikonal, device)
    eik_model.load_state_dict(model.state_dict())
    eik_model.zero_grad(set_to_none=True)
    eik_loss, eik_metrics = compute_loss(
        eik_model, codes, broad, near, 650, eikonal
    )
    eik_loss.backward()
    if not torch.isfinite(eik_loss) or eik_metrics["eikonal_points"] != 8:
        raise RuntimeError("Numerical-gradient Eikonal validation failed.")

    checkpoint = output / "random_roundtrip_checkpoint.validation_only.pth"
    atomic_torch_save(
        {
            "validation_only": True,
            "model_state_dict": model.state_dict(),
            "config": {key: value for key, value in primary.items() if not key.startswith("_")},
        },
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location=device)
    restored = build_decoder(primary, device)
    restored.load_state_dict(payload["model_state_dict"])
    with torch.no_grad():
        query = torch.cat((codes[0:1].expand(7, -1), torch.linspace(-0.2, 0.2, 21, device=device).reshape(7, 3)), dim=1)
        before = model(query, level_weights=[1.0] * len(model.grid_resolutions))
        after = restored(query, level_weights=[1.0] * len(restored.grid_resolutions))
    if not torch.equal(before, after):
        raise RuntimeError("Checkpoint round-trip changed decoder predictions.")

    sphere_path = output / "synthetic_sphere_mesh.validation_only.ply"
    scaling = read_scaling(primary["periodic_evaluation"]["rescale_details_csv"])
    mesh_report = decode_latent_to_mesh(
        AnalyticSphere().to(device),
        np.zeros(int(primary["latent_size"]), dtype=np.float32),
        sphere_path,
        resolution=48,
        max_batch=32768,
        device=device,
        scaling=scaling,
    )
    if mesh_report["coordinate_space"] != "physical_mm":
        raise RuntimeError("Synthetic mesh was not written in final physical-mm coordinates.")
    report = {
        "passed": True,
        "validation_only": True,
        "device": str(device),
        "scans_sampled": [row["scan_id"] for row in rows],
        "multiprocessing_dataloader_workers": worker_count,
        "multiprocessing_dataloader_passed": True,
        "broad_shape": list(broad.shape),
        "near_shape": list(near.shape),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "grid_parameters": sum(parameter.numel() for parameter in model.grids),
        "primary_loss": float(primary_loss.detach().cpu()),
        "primary_metrics": primary_metrics,
        "eikonal_ablation_loss": float(eik_loss.detach().cpu()),
        "eikonal_ablation_metrics": eik_metrics,
        "checkpoint_roundtrip_exact": True,
        "checkpoint": str(checkpoint),
        "synthetic_mesh": mesh_report,
        "persistent_output_root": str(output),
    }
    write_json(output / "implementation_validation_report.json", report)
    print(f"Implementation validation passed; report={output / 'implementation_validation_report.json'}")


if __name__ == "__main__":
    main()
