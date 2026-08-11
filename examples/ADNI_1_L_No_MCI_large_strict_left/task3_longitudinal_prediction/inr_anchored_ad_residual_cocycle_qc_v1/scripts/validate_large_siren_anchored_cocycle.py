#!/usr/bin/env python3
"""Validate source inputs, cache shapes, and the frozen-base initialization invariant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anchored_ad_residual_cocycle_model import AnchoredAdResidualCocycle
from inr_anchored_common import (
    ensure_paths_exist, experiment_dir, load_cache_arrays, load_frozen_base_flow,
    load_metadata_and_latents, write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=None, help="Optional trained checkpoint to validate too.")
    args = parser.parse_args()
    root = experiment_dir()
    config = json.loads(Path(args.config).read_text())
    ensure_paths_exist(config, root)
    report: dict[str, object] = {"config": str(Path(args.config).resolve()), "device": args.device}
    frame, source_latents = load_metadata_and_latents(config, root)
    report["source_scan_count"] = int(len(frame))
    report["source_latent_shape"] = list(source_latents.shape)
    required = [root / "metadata" / "registered_meshes.npz", root / "basis" / "train_only_basis.npz", root / "metadata" / "input_contract.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run cache builder first; missing:\n" + "\n".join(missing))
    cache, basis = load_cache_arrays(root)
    if cache["latents"].shape != source_latents.shape or not np.allclose(cache["latents"], source_latents, atol=0.0, rtol=0.0):
        raise ValueError("Cached latents do not exactly match QC source latent archives.")
    if cache["vertices"].ndim != 3 or cache["vertices"].shape[0] != len(frame):
        raise ValueError("Invalid cached registered vertices.")
    if basis["feature_mean"].shape != (1, int(config["LatentSize"])):
        raise ValueError("Invalid train-only feature mean shape.")
    if basis["residual_basis"].shape[1] != int(config["LatentSize"]):
        raise ValueError("Invalid residual basis width.")
    device = torch.device(args.device)
    base = load_frozen_base_flow(config, device, root)
    model = AnchoredAdResidualCocycle(
        base_flow=base, feature_mean=torch.from_numpy(basis["feature_mean"]),
        feature_components=torch.from_numpy(basis["feature_components"]), residual_basis=torch.from_numpy(basis["residual_basis"]),
        hidden_dims=config["HeadHiddenDims"], dropout=float(config["Dropout"]),
        speed_lower=float(config["SpeedLower"]), speed_upper=float(config["SpeedUpper"]),
        speed_individual_log_span=float(config["SpeedIndividualLogSpan"]), residual_coefficient_span=float(config["ResidualCoefficientSpan"]),
    ).to(device).eval()
    latent = torch.from_numpy(cache["latents"][:3]).to(device)
    source_time = torch.from_numpy(cache["times"][:3]).to(device)
    target_time = source_time + 0.02
    with torch.no_grad():
        base_cn = base.transport(latent, source_time, target_time, torch.zeros(3, 1, device=device))
        base_ad = base.transport(latent, source_time, target_time, torch.ones(3, 1, device=device))
        new_cn = model.transport(latent, source_time, target_time, torch.zeros(3, 1, device=device))
        new_ad = model.transport(latent, source_time, target_time, torch.ones(3, 1, device=device))
    cn_error = float((new_cn - base_cn).abs().max().cpu())
    ad_error = float((new_ad - base_ad).abs().max().cpu())
    if cn_error > 1.0e-6 or ad_error > 1.0e-6:
        raise AssertionError(f"Initialization is not base-preserving: CN={cn_error}, AD={ad_error}")
    report.update({
        "cache_vertex_shape": list(cache["vertices"].shape), "cache_face_shape": list(cache["faces"].shape),
        "feature_pca_shape": list(basis["feature_components"].shape), "residual_basis_shape": list(basis["residual_basis"].shape),
        "initial_cn_max_abs_error_vs_base": cn_error, "initial_ad_max_abs_error_vs_base": ad_error,
    })
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        report["checkpoint"] = str(Path(args.checkpoint).resolve())
        report["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
    write_json(root / "metadata" / "validation_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
