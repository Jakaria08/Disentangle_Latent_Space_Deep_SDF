#!/usr/bin/env python3
"""Train the compact AD calibration while the SIREN decoder/base flow stay frozen."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from anchored_ad_residual_cocycle_model import AnchoredAdResidualCocycle
from inr_anchored_common import (
    CachedPairDataset, decode_sdf, experiment_dir, load_cache_arrays,
    load_frozen_base_flow, load_frozen_decoder, set_seed, signed_mesh_volume,
    write_json,
)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _surface_predictions(
    decoder: torch.nn.Module, source_latent: torch.Tensor, predicted_latent: torch.Tensor,
    vertices: torch.Tensor, normals: torch.Tensor, faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        source_sdf = decode_sdf(decoder, source_latent, vertices)
    predicted_sdf = decode_sdf(decoder, predicted_latent, vertices)
    normal_displacement = -(predicted_sdf - source_sdf)
    predicted_vertices = vertices + normal_displacement.unsqueeze(-1) * normals
    return normal_displacement, predicted_vertices, signed_mesh_volume(predicted_vertices, faces)


def _losses(
    model: AnchoredAdResidualCocycle, decoder: torch.nn.Module, batch: dict[str, Any],
    faces: torch.Tensor, residual_basis: torch.Tensor, config: dict[str, Any], *, training: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = config["LossWeights"]
    source, target = batch["source_latent"], batch["target_latent"]
    s, t, condition = batch["source_time"], batch["target_time"], batch["condition"]
    predicted, aux = model.transport(source, s, t, condition, return_aux=True)
    samples = batch["target_samples"]
    predicted_sdf = decode_sdf(decoder, predicted, samples[:, :, :3])
    target_sdf_loss = F.l1_loss(predicted_sdf, samples[:, :, 3])
    normal_displacement, predicted_vertices, predicted_volume = _surface_predictions(
        decoder, source, predicted, batch["source_vertices"], batch["source_normals"], faces
    )
    target_normal_displacement = ((batch["target_vertices"] - batch["source_vertices"]) * batch["source_normals"]).sum(dim=-1)
    vertex_count = normal_displacement.shape[1]
    sample_count = min(int(config["VerticesPerNormalLoss"]), vertex_count)
    if sample_count < vertex_count:
        indices = torch.randperm(vertex_count, device=source.device)[:sample_count]
        registered_normal = F.smooth_l1_loss(normal_displacement[:, indices], target_normal_displacement[:, indices])
    else:
        registered_normal = F.smooth_l1_loss(normal_displacement, target_normal_displacement)
    eps = 1.0e-6
    target_volume = batch["target_volume"].clamp_min(eps)
    source_volume = batch["source_volume"].clamp_min(eps)
    volume_log_ratio = F.smooth_l1_loss(torch.log(predicted_volume.clamp_min(eps) / source_volume), torch.log(target_volume / source_volume))
    gap_years = batch["gap_years"].abs().clamp_min(0.05)
    pred_rate = (predicted_volume / source_volume - 1.0) * 100.0 / gap_years
    target_rate = (target_volume / source_volume - 1.0) * 100.0 / gap_years
    volume_rate = F.smooth_l1_loss(pred_rate, target_rate)
    # Both paths use the same one-step map.  The observed midpoint is an
    # explicit consistency constraint; virtual points are random compositions.
    midpoint = 0.5 * (s + t)
    composed = model.transport(model.transport(source, s, midpoint, condition), midpoint, t, condition)
    observed_cocycle = F.smooth_l1_loss(composed, predicted)
    fraction = 0.2 + 0.6 * torch.rand_like(s)
    virtual_time = s + fraction * (t - s)
    virtual = model.transport(model.transport(source, s, virtual_time, condition), virtual_time, t, condition)
    virtual_cocycle = F.smooth_l1_loss(virtual, predicted)
    inverse = model.transport(predicted, t, s, condition)
    inverse_loss = F.smooth_l1_loss(inverse, source)
    with torch.no_grad():
        base_sdf = decode_sdf(decoder, aux["base_ad"], samples[:, :, :3])
        base_l1 = (base_sdf - samples[:, :, 3]).abs().mean(dim=1)
    new_l1 = (predicted_sdf - samples[:, :, 3]).abs().mean(dim=1)
    ad_mask = condition.reshape(-1) > 0.5
    if bool(ad_mask.any()):
        no_degradation = F.relu(new_l1[ad_mask] - (1.0 + float(config["NoDegradationTolerance"])) * base_l1[ad_mask]).mean()
        ad_volume_rate = F.smooth_l1_loss(pred_rate[ad_mask], target_rate[ad_mask])
        ad_sdf_l1 = new_l1[ad_mask].mean()
        ad_base_sdf_l1 = base_l1[ad_mask].mean()
    else:
        no_degradation = torch.zeros((), device=source.device)
        ad_volume_rate = pred_rate.sum() * 0.0
        ad_sdf_l1 = new_l1.sum() * 0.0
        ad_base_sdf_l1 = base_l1.sum() * 0.0
    dynamic_target = F.smooth_l1_loss((predicted - target) @ residual_basis.T, torch.zeros_like((predicted - target) @ residual_basis.T))
    regularization = model.regularization()
    terms = {
        "target_sdf": target_sdf_loss, "registered_normal": registered_normal,
        "volume_log_ratio": volume_log_ratio, "volume_rate": volume_rate,
        "observed_cocycle": observed_cocycle, "virtual_cocycle": virtual_cocycle,
        "inverse": inverse_loss, "no_degradation": no_degradation,
        "dynamic_target": dynamic_target, **regularization,
        "mean_speed": aux["speed"].mean(), "pred_volume_rate": pred_rate.mean(),
        "target_volume_rate": target_rate.mean(), "sdf_l1": new_l1.mean(), "base_sdf_l1": base_l1.mean(),
        "ad_volume_rate": ad_volume_rate, "ad_sdf_l1": ad_sdf_l1, "ad_base_sdf_l1": ad_base_sdf_l1,
        "ad_count": ad_mask.float().sum(),
    }
    total = sum(float(weights.get(name, 0.0)) * terms[name] for name in weights)
    return total, terms


@torch.no_grad()
def _validation(
    model: AnchoredAdResidualCocycle, decoder: torch.nn.Module, loader: DataLoader,
    faces: torch.Tensor, residual_basis: torch.Tensor, config: dict[str, Any], max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    aggregate: dict[str, float] = {}
    ad_volume_sum = 0.0
    ad_sdf_sum = 0.0
    ad_base_sdf_sum = 0.0
    ad_count = 0.0
    count = 0
    device = faces.device
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _to_device(batch, device)
        total, terms = _losses(model, decoder, batch, faces, residual_basis, config, training=False)
        for name, value in {"total": total, **terms}.items():
            aggregate[name] = aggregate.get(name, 0.0) + float(value.detach().cpu())
        batch_ad_count = float(terms["ad_count"].detach().cpu())
        ad_volume_sum += float(terms["ad_volume_rate"].detach().cpu()) * batch_ad_count
        ad_sdf_sum += float(terms["ad_sdf_l1"].detach().cpu()) * batch_ad_count
        ad_base_sdf_sum += float(terms["ad_base_sdf_l1"].detach().cpu()) * batch_ad_count
        ad_count += batch_ad_count
        count += 1
    if not count:
        raise RuntimeError("Validation loader produced no batches.")
    result = {name: value / count for name, value in aggregate.items()}
    if ad_count <= 0:
        raise RuntimeError("Validation split contains no AD pairs; cannot select the AD primary metric.")
    result["ad_volume_rate"] = ad_volume_sum / ad_count
    result["ad_sdf_l1"] = ad_sdf_sum / ad_count
    result["ad_base_sdf_l1"] = ad_base_sdf_sum / ad_count
    result["ad_count"] = ad_count
    return result


def _build_model(config: dict[str, Any], root: Path, device: torch.device) -> tuple[AnchoredAdResidualCocycle, torch.nn.Module, torch.Tensor]:
    _, basis = load_cache_arrays(root)
    base = load_frozen_base_flow(config, device, root)
    decoder = load_frozen_decoder(config, device, root)
    model = AnchoredAdResidualCocycle(
        base_flow=base, feature_mean=torch.from_numpy(basis["feature_mean"]),
        feature_components=torch.from_numpy(basis["feature_components"]), residual_basis=torch.from_numpy(basis["residual_basis"]),
        hidden_dims=config["HeadHiddenDims"], dropout=float(config["Dropout"]),
        speed_lower=float(config["SpeedLower"]), speed_upper=float(config["SpeedUpper"]),
        speed_individual_log_span=float(config["SpeedIndividualLogSpan"]),
        residual_coefficient_span=float(config["ResidualCoefficientSpan"]),
    ).to(device)
    return model, decoder, torch.from_numpy(basis["residual_basis"]).to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default="primary")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke-batches", type=int, default=None)
    args = parser.parse_args()
    root = experiment_dir()
    config = json.loads(Path(args.config).read_text())
    required = [root / "metadata" / "registered_meshes.npz", root / "basis" / "train_only_basis.npz"]
    if any(not path.exists() for path in required):
        raise FileNotFoundError("Run build_large_siren_cache.py before training.")
    run_dir = root / "runs" / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory exists; choose a new --run-name: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    device = torch.device(args.device)
    set_seed(int(config["Seed"]))
    forward_train = pd.read_csv(root / "metadata" / "pair_records_train.csv")
    backward_train = root / "metadata" / "pair_records_train_backward.csv"
    if bool(config.get("UseBackwardPairs", True)) and backward_train.exists():
        forward_train = pd.concat([forward_train, pd.read_csv(backward_train)], ignore_index=True)
    # Put AD pairs first only so a bounded smoke validation contains the
    # primary cohort. Full validation still uses every pair and is unchanged.
    val_pairs = pd.read_csv(root / "metadata" / "pair_records_val.csv").sort_values(
        ["label_ad", "subject_id", "source_time", "target_time"], ascending=[False, True, True, True]
    ).reset_index(drop=True)
    train_set = CachedPairDataset(root, forward_train, int(config["SamplesPerTarget"]), deterministic=False, seed=int(config["Seed"]))
    val_set = CachedPairDataset(root, val_pairs, int(config["SamplesPerTarget"]), deterministic=True, seed=int(config["ValidationSampleSeed"]))
    sampling_weights = torch.ones(len(forward_train), dtype=torch.double)
    sampling_weights[torch.as_tensor(forward_train.label_ad.to_numpy(dtype=np.int64)) == 1] = float(config["ADSamplingWeight"])
    sampler = WeightedRandomSampler(sampling_weights, num_samples=len(sampling_weights), replacement=True, generator=torch.Generator().manual_seed(int(config["Seed"])))
    train_loader = DataLoader(train_set, batch_size=int(config["PairsPerBatch"]), sampler=sampler, num_workers=int(config["NumWorkers"]), pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=int(config["PairsPerBatch"]), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model, decoder, residual_basis = _build_model(config, root, device)
    faces = torch.from_numpy(np.load(root / "metadata" / "registered_meshes.npz", allow_pickle=False)["faces"]).to(device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=float(config["LearningRate"]), weight_decay=float(config["WeightDecay"]))
    epochs = int(args.epochs or config["Epochs"])
    best_primary = float("inf")
    best_surface = float("inf")
    best_ad_volume = float("inf")
    patience = 0
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_totals: dict[str, float] = {}
        train_count = 0
        for batch_index, batch in enumerate(train_loader):
            if args.smoke_batches is not None and batch_index >= args.smoke_batches:
                break
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            total, terms = _losses(model, decoder, batch, faces, residual_basis, config, training=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
            optimizer.step()
            for name, value in {"total": total, **terms}.items():
                train_totals[name] = train_totals.get(name, 0.0) + float(value.detach().cpu())
            train_count += 1
        if not train_count:
            raise RuntimeError("Training loader produced no batches.")
        train_metrics = {f"train_{name}": value / train_count for name, value in train_totals.items()}
        val_metrics = {f"val_{name}": value for name, value in _validation(model, decoder, val_loader, faces, residual_basis, config, args.smoke_batches).items()}
        row = {"epoch": float(epoch), **train_metrics, **val_metrics, "elapsed_minutes": (time.time() - started) / 60.0}
        history.append(row)
        primary = val_metrics["val_ad_volume_rate"]
        surface = val_metrics["val_ad_sdf_l1"] + val_metrics["val_registered_normal"]
        feasible = val_metrics["val_ad_sdf_l1"] <= (1.0 + float(config["NoDegradationTolerance"])) * val_metrics["val_ad_base_sdf_l1"]
        state = {
            "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "config": config, "base_flow_checkpoint": str(config["BaseFlowCheckpoint"]),
            "decoder_checkpoint": str(config["DecoderCheckpoint"]), "basis_path": "basis/train_only_basis.npz",
            "validation": val_metrics,
        }
        torch.save(state, run_dir / "checkpoints" / "latest.pth")
        if primary < best_primary:
            best_primary, patience = primary, 0
            torch.save(state, run_dir / "checkpoints" / "best_primary.pth")
        else:
            patience += 1
        if feasible and surface < best_surface:
            best_surface = surface
            torch.save(state, run_dir / "checkpoints" / "best_feasible_surface.pth")
        if feasible and primary < best_ad_volume:
            best_ad_volume = primary
            torch.save(state, run_dir / "checkpoints" / "best_feasible_ad_volume.pth")
        print(f"epoch={epoch:03d} train={train_metrics['train_total']:.5f} val_ad_sdf={val_metrics['val_ad_sdf_l1']:.5f} val_ad_volume_rate={val_metrics['val_ad_volume_rate']:.5f} speed={val_metrics['val_mean_speed']:.3f}", flush=True)
        if args.smoke_batches is None and patience >= int(config["EarlyStoppingPatience"]):
            print(f"Early stopping at epoch {epoch}.")
            break
    columns = sorted({key for row in history for key in row})
    with (run_dir / "logs" / "epochs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(history)
    write_json(run_dir / "logs" / "summary.json", {
        "epochs_completed": len(history), "best_primary_val_volume_rate": best_primary,
        "best_surface": best_surface, "best_feasible_ad_volume": best_ad_volume,
        "elapsed_minutes": (time.time() - started) / 60.0, "smoke_batches": args.smoke_batches,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
