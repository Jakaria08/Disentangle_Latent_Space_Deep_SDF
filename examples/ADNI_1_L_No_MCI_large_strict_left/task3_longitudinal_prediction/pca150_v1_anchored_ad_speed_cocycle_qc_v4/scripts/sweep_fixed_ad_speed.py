#!/usr/bin/env python3
"""Validation-only fixed AD speed sweep for the frozen-v1 calibration hypothesis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from train_v1_anchored_ad_speed import evaluate_records, evaluate_subject_slopes, make_model
from v1_speed_utils import EXPERIMENT_DIR, build_pair_records, load_config, load_split_archive, pca_latents, resolve_device, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "configs" / "v1_anchored_ad_speed_primary.json"))
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=["val"], default="val", help="Restricted to validation to protect the test set.")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def set_global_speed(model: torch.nn.Module, value: float) -> tuple[float, bool]:
    lower, upper = float(model.minimum_speed), float(model.maximum_speed)
    requested = float(value)
    if requested < lower or requested > upper:
        raise ValueError(f"Scale must be inside [{lower}, {upper}]")
    eps = 1.0e-6
    effective = min(max(requested, lower + eps * (upper - lower)), upper - eps * (upper - lower))
    clipped = effective != requested
    fraction = (effective - lower) / (upper - lower)
    with torch.no_grad():
        model.global_speed_logit.copy_(torch.tensor(math.log(fraction / (1.0 - fraction)), device=model.global_speed_logit.device))
    return effective, clipped


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else EXPERIMENT_DIR / "metadata"
    stats_path = metadata_dir / "speed_feature_stats.npz"
    if not stats_path.is_file():
        raise FileNotFoundError("Run prepare_v1_speed_metadata.py first")
    with np.load(stats_path, allow_pickle=False) as archive:
        metadata = {key: archive[key] for key in archive.files}
    device = resolve_device(args.device)
    archive = load_split_archive("val")
    records = build_pair_records(archive)
    latents = pca_latents(archive, int(config["components"]))
    rows = []
    for scale in args.scales:
        model = make_model(config=config, metadata=metadata, device=device)
        effective_scale, clipped = set_global_speed(model, float(scale))
        pair_metrics = evaluate_records(model=model, archive=archive, records=records, components=int(config["components"]), device=device, batch_size=int(args.batch_size))
        slope_metrics = evaluate_subject_slopes(model=model, archive=archive, latents=latents, device=device)
        rows.append(
            {
                "split": "val",
                "fixed_ad_speed": float(scale),
                "effective_ad_speed": float(effective_scale),
                "speed_was_clipped_to_open_interval": bool(clipped),
                "ad_pca_mse": pair_metrics["groups"]["AD"]["pca_mean"],
                "ad_vertex_mae": pair_metrics["groups"]["AD"]["vertex_mean"],
                "ad_volume_relative_error": pair_metrics["groups"]["AD"]["volume_relative_mean"],
                "ad_log_volume_rate_mae": pair_metrics["groups"]["AD"]["rate_mean"],
                "ad_subject_slope_mae": slope_metrics["ad_subject_slope_abs_error"],
                "ad_predicted_slope": slope_metrics["ad_predicted_slope_mean"],
                "ad_observed_slope": slope_metrics["ad_observed_slope_mean"],
                "overall_pca_mse": pair_metrics["groups"]["overall"]["pca_mean"],
                "overall_vertex_mae": pair_metrics["groups"]["overall"]["vertex_mean"],
                "cn_max_abs_delta_from_v1": pair_metrics["cn_max_abs_delta_from_v1"],
            }
        )
    output = Path(args.output) if args.output else EXPERIMENT_DIR / "analysis" / "fixed_ad_speed_val_sweep.csv"
    write_csv(output, rows)
    for row in rows:
        print(row)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
