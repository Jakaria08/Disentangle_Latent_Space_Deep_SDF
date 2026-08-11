#!/usr/bin/env python3
"""Write reproducible pair-level base-versus-calibrated SIREN metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from anchored_ad_residual_cocycle_model import AnchoredAdResidualCocycle
from inr_anchored_common import (
    CachedPairDataset, decode_sdf, experiment_dir, load_cache_arrays,
    load_frozen_base_flow, load_frozen_decoder, signed_mesh_volume, write_json,
)


def _build_model(config: dict[str, Any], root: Path, device: torch.device) -> tuple[AnchoredAdResidualCocycle, torch.nn.Module]:
    _, basis = load_cache_arrays(root)
    model = AnchoredAdResidualCocycle(
        base_flow=load_frozen_base_flow(config, device, root),
        feature_mean=torch.from_numpy(basis["feature_mean"]),
        feature_components=torch.from_numpy(basis["feature_components"]),
        residual_basis=torch.from_numpy(basis["residual_basis"]),
        hidden_dims=config["HeadHiddenDims"], dropout=float(config["Dropout"]),
        speed_lower=float(config["SpeedLower"]), speed_upper=float(config["SpeedUpper"]),
        speed_individual_log_span=float(config["SpeedIndividualLogSpan"]),
        residual_coefficient_span=float(config["ResidualCoefficientSpan"]),
    ).to(device)
    return model, load_frozen_decoder(config, device, root)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@torch.no_grad()
def _evaluate_pair_batch(
    model: AnchoredAdResidualCocycle, decoder: torch.nn.Module, batch: dict[str, Any], faces: torch.Tensor,
) -> list[dict[str, Any]]:
    source, target = batch["source_latent"], batch["target_latent"]
    s, t, condition = batch["source_time"], batch["target_time"], batch["condition"]
    anchored, aux = model.transport(source, s, t, condition, return_aux=True)
    base = aux["base_cn"] + condition * (aux["base_ad"] - aux["base_cn"])
    source_vertices, normals, target_vertices = batch["source_vertices"], batch["source_normals"], batch["target_vertices"]
    source_sdf = decode_sdf(decoder, source, source_vertices)
    samples = batch["target_samples"]
    gt_sdf = samples[:, :, 3]
    rows: list[dict[str, Any]] = []
    target_normal = ((target_vertices - source_vertices) * normals).sum(dim=-1)
    source_volume = batch["source_volume"].clamp_min(1.0e-6)
    target_volume = batch["target_volume"].clamp_min(1.0e-6)
    gt_rate = (target_volume / source_volume - 1.0) * 100.0 / batch["gap_years"].abs().clamp_min(0.05)
    for model_name, latent in (("base_direct_cocycle", base), ("anchored_ad_residual", anchored)):
        pred_sdf = decode_sdf(decoder, latent, samples[:, :, :3])
        surface_sdf = decode_sdf(decoder, latent, source_vertices)
        displacement = -(surface_sdf - source_sdf)
        predicted_vertices = source_vertices + displacement.unsqueeze(-1) * normals
        predicted_volume = signed_mesh_volume(predicted_vertices, faces).clamp_min(1.0e-6)
        predicted_rate = (predicted_volume / source_volume - 1.0) * 100.0 / batch["gap_years"].abs().clamp_min(0.05)
        sdf_l1 = (pred_sdf - gt_sdf).abs().mean(dim=1)
        normal_mae = (displacement - target_normal).abs().mean(dim=1)
        vertex_mae = (predicted_vertices - target_vertices).norm(dim=-1).mean(dim=1)
        for index in range(source.shape[0]):
            rows.append({
                "model": model_name, "split": "", "subject_id": batch["subject_id"][index],
                "source_scan_id": batch["source_scan_id"][index], "target_scan_id": batch["target_scan_id"][index],
                "label": "AD" if int(batch["label_ad"][index]) else "CN", "source_time": float(s[index]), "target_time": float(t[index]),
                "gap_years": float(batch["gap_years"][index]), "sdf_l1": float(sdf_l1[index]),
                "registered_normal_mae": float(normal_mae[index]), "registered_vertex_mae": float(vertex_mae[index]),
                "source_volume": float(source_volume[index]), "target_volume": float(target_volume[index]),
                "predicted_volume": float(predicted_volume[index]), "volume_relative_error_pct": float(((predicted_volume[index] / target_volume[index]) - 1.0).abs() * 100.0),
                "ground_truth_volume_rate_pct_per_year": float(gt_rate[index]),
                "predicted_volume_rate_pct_per_year": float(predicted_rate[index]),
                "volume_rate_absolute_error_pct_per_year": float((predicted_rate[index] - gt_rate[index]).abs()),
                "ad_speed": float(aux["speed"][index]),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    root = experiment_dir()
    config = json.loads(Path(args.config).read_text())
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device(args.device)
    model, decoder = _build_model(config, root, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    faces = torch.from_numpy(np.load(root / "metadata" / "registered_meshes.npz", allow_pickle=False)["faces"]).to(device)
    output_name = args.output_name or checkpoint_path.stem
    output = checkpoint_path.parent.parent / "evaluation" / output_name
    if output.exists():
        raise FileExistsError(f"Evaluation output exists; choose --output-name: {output}")
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for split in args.splits:
        pairs = pd.read_csv(root / "metadata" / f"pair_records_{split}.csv")
        if args.max_pairs is not None:
            pairs = pairs.iloc[:args.max_pairs].reset_index(drop=True)
        dataset = CachedPairDataset(root, pairs, int(config["SamplesPerTarget"]), deterministic=True, seed=int(config["ValidationSampleSeed"]))
        loader = DataLoader(dataset, batch_size=int(config["PairsPerBatch"]), shuffle=False, num_workers=0)
        for batch in loader:
            batch = _move(batch, device)
            batch_rows = _evaluate_pair_batch(model, decoder, batch, faces)
            for row in batch_rows:
                row["split"] = split
            rows.extend(batch_rows)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "pair_metrics.csv", index=False)
    metric_columns = [
        "sdf_l1", "registered_normal_mae", "registered_vertex_mae", "volume_relative_error_pct",
        "ground_truth_volume_rate_pct_per_year", "predicted_volume_rate_pct_per_year", "volume_rate_absolute_error_pct_per_year", "ad_speed",
    ]
    summary = frame.groupby(["split", "label", "model"], dropna=False)[metric_columns].agg(["mean", "median", "count"])
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "summary_metrics.csv", index=False)
    write_json(output / "summary.json", {
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "splits": args.splits, "pair_rows": int(len(frame)),
        "metric_direction": {
            "sdf_l1": "lower is better", "registered_normal_mae": "lower is better",
            "registered_vertex_mae": "lower is better", "volume_relative_error_pct": "lower is better",
            "volume_rate_absolute_error_pct_per_year": "lower is better",
            "ground_truth_volume_rate_pct_per_year": "reference only; sign follows mesh-volume convention",
            "predicted_volume_rate_pct_per_year": "compare against ground truth; not independently optimized as higher/lower",
            "ad_speed": "descriptive calibration factor, bounded by config",
        },
    })
    print(f"Wrote {len(frame)} pair-model rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
