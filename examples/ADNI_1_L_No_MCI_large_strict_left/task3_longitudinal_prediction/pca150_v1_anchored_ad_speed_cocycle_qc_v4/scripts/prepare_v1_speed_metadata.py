#!/usr/bin/env python3
"""Build train-only feature/loss statistics and frozen-v1 validation references."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from v1_speed_utils import (
    EXPERIMENT_DIR,
    PairRecord,
    build_pair_records,
    coefficient_stats,
    decode_pca_torch,
    file_sha256,
    first_visit_sequence_starts,
    load_config,
    load_pca_model,
    load_split_archive,
    load_v1_flow,
    limit_records_stratified,
    mesh_volume_torch,
    pca_latents,
    resolve_device,
    resolve_repo_path,
    subject_end,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "configs" / "v1_anchored_ad_speed_primary.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def chunks(values: list[PairRecord], size: int) -> Iterable[list[PairRecord]]:
    for start in range(0, len(values), max(1, int(size))):
        yield values[start : start + max(1, int(size))]


def tensors_for_records(
    records: list[PairRecord],
    latents: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "source": torch.from_numpy(np.stack([latents[item.source_index] for item in records])).float().to(device),
        "target": torch.from_numpy(np.stack([latents[item.target_index] for item in records])).float().to(device),
        "source_norm": torch.tensor([item.source_age_norm for item in records], dtype=torch.float32, device=device),
        "target_norm": torch.tensor([item.target_age_norm for item in records], dtype=torch.float32, device=device),
        "source_years": torch.tensor([item.source_age_years for item in records], dtype=torch.float32, device=device),
        "target_years": torch.tensor([item.target_age_years for item in records], dtype=torch.float32, device=device),
        "condition": torch.tensor([float(item.label_ad) for item in records], dtype=torch.float32, device=device),
    }


@torch.no_grad()
def base_pair_statistics(
    *,
    records: list[PairRecord],
    latents: np.ndarray,
    base_flow: torch.nn.Module,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
    faces: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    scalars: list[np.ndarray] = []
    pca_errors: list[np.ndarray] = []
    vertex_errors: list[np.ndarray] = []
    true_rates: list[np.ndarray] = []
    detailed: list[dict[str, Any]] = []
    for current in chunks(records, batch_size):
        batch = tensors_for_records(current, latents, device)
        base = base_flow.transport(batch["source"], batch["source_norm"], batch["target_norm"], batch["condition"])
        source_vertices = decode_pca_torch(batch["source"], mean_flat, components)
        target_vertices = decode_pca_torch(batch["target"], mean_flat, components)
        base_vertices = decode_pca_torch(base, mean_flat, components)
        source_volume = mesh_volume_torch(source_vertices, faces)
        target_volume = mesh_volume_torch(target_vertices, faces)
        base_volume = mesh_volume_torch(base_vertices, faces)
        delta = batch["target_years"] - batch["source_years"]
        safe_delta = torch.where(delta.abs() < 1.0e-6, torch.full_like(delta, 1.0e-6), delta)
        base_rate = (torch.log(base_volume) - torch.log(source_volume)) / safe_delta
        true_rate = (torch.log(target_volume) - torch.log(source_volume)) / safe_delta
        displacement_rate = torch.linalg.norm(base - batch["source"], dim=1) / safe_delta.abs().clamp_min(1.0e-6)
        scalar = torch.stack(
            [
                batch["source_norm"],
                batch["target_norm"],
                torch.log1p(safe_delta.abs()),
                torch.log(source_volume),
                base_rate,
                displacement_rate,
            ],
            dim=1,
        )
        pca = torch.mean((base - batch["target"]) ** 2, dim=1)
        vertex = torch.mean(torch.abs(base_vertices - target_vertices), dim=(1, 2))
        scalars.append(scalar.cpu().numpy())
        pca_errors.append(pca.cpu().numpy())
        vertex_errors.append(vertex.cpu().numpy())
        true_rates.append(true_rate.cpu().numpy())
        for index, record in enumerate(current):
            detailed.append(
                {
                    "split": "",
                    "diagnosis": record.diagnosis,
                    "subject_id": record.subject_id,
                    "source_visit_order": record.source_visit_order,
                    "target_visit_order": record.target_visit_order,
                    "endpoint_pca_mse": float(pca[index].item()),
                    "endpoint_vertex_mae": float(vertex[index].item()),
                    "true_log_rate": float(true_rate[index].item()),
                }
            )
    return {
        "scalars": np.concatenate(scalars, axis=0) if scalars else np.zeros((0, 6), dtype=np.float32),
        "pca_errors": np.concatenate(pca_errors, axis=0) if pca_errors else np.zeros(0, dtype=np.float32),
        "vertex_errors": np.concatenate(vertex_errors, axis=0) if vertex_errors else np.zeros(0, dtype=np.float32),
        "true_rates": np.concatenate(true_rates, axis=0) if true_rates else np.zeros(0, dtype=np.float32),
    }, detailed


@torch.no_grad()
def base_subject_slopes(
    *,
    archive: dict[str, np.ndarray],
    latents: np.ndarray,
    base_flow: torch.nn.Module,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
    faces: torch.Tensor,
    device: torch.device,
    diagnosis: str,
) -> list[float]:
    slopes: list[float] = []
    for start in first_visit_sequence_starts(archive, diagnosis=diagnosis):
        end = subject_end(archive, start)
        source = torch.from_numpy(latents[start : start + 1]).float().to(device)
        targets = torch.from_numpy(latents[start + 1 : end]).float().to(device)
        source_norm = torch.full((targets.shape[0],), float(archive["visit_continuous_age_norm"][start]), device=device)
        target_norm = torch.from_numpy(archive["visit_continuous_age_norm"][start + 1 : end]).float().to(device)
        condition = torch.full((targets.shape[0],), float(archive["visit_label_ad"][start]), device=device)
        base = base_flow.transport(source.expand_as(targets), source_norm, target_norm, condition)
        target_volume = mesh_volume_torch(decode_pca_torch(targets, mean_flat, components), faces)
        base_volume = mesh_volume_torch(decode_pca_torch(base, mean_flat, components), faces)
        years = archive["visit_continuous_age_years"][start + 1 : end].astype(np.float64)
        source_year = float(archive["visit_continuous_age_years"][start])
        x = years - source_year
        if np.unique(x).size < 2:
            continue
        observed_slope = float(np.polyfit(x, np.log(target_volume.cpu().numpy()), 1)[0])
        predicted_slope = float(np.polyfit(x, np.log(base_volume.cpu().numpy()), 1)[0])
        slopes.append(observed_slope)
        _ = predicted_slope
    return slopes


def first_last_records(archive: dict[str, np.ndarray]) -> list[PairRecord]:
    all_records = build_pair_records(archive)
    offsets = archive["subject_visit_offsets"]
    target_by_start = {int(offsets[index]): int(offsets[index + 1] - 1) for index in range(len(offsets) - 1)}
    return [record for record in all_records if record.source_index in target_by_start and record.target_index == target_by_start[record.source_index]]


def grouped_reference(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for diagnosis in ("CN", "AD"):
        subset = [row for row in rows if row["diagnosis"] == diagnosis]
        output[diagnosis] = {
            "rows": len(subset),
            "endpoint_pca_mse": float(np.mean([row["endpoint_pca_mse"] for row in subset])) if subset else float("nan"),
            "endpoint_vertex_mae": float(np.mean([row["endpoint_vertex_mae"] for row in subset])) if subset else float("nan"),
        }
    output["overall"] = {
        "rows": len(rows),
        "endpoint_pca_mse": float(np.mean([row["endpoint_pca_mse"] for row in rows])) if rows else float("nan"),
        "endpoint_vertex_mae": float(np.mean([row["endpoint_vertex_mae"] for row in rows])) if rows else float("nan"),
    }
    output["split"] = split
    return output


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else EXPERIMENT_DIR / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    components_count = int(config["components"])
    base_checkpoint = resolve_repo_path(config["base_v1_checkpoint"])
    registered_tensors = resolve_repo_path(config["registered_mesh_tensors"])
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    if not registered_tensors.is_file():
        raise FileNotFoundError(registered_tensors)

    started = time.time()
    train_archive = load_split_archive("train")
    val_archive = load_split_archive("val")
    _, mean_flat_np, pca_components_np, faces_np = load_pca_model(components_count)
    mean_flat = torch.from_numpy(mean_flat_np).to(device)
    pca_components = torch.from_numpy(pca_components_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)
    base_flow, base_payload = load_v1_flow(base_checkpoint, device)

    train_latents = pca_latents(train_archive, components_count)
    val_latents = pca_latents(val_archive, components_count)
    train_ad = build_pair_records(train_archive, diagnosis="AD")
    val_all = build_pair_records(val_archive)
    val_first_last = first_last_records(val_archive)
    if args.max_train_pairs > 0:
        train_ad = train_ad[: int(args.max_train_pairs)]
    if args.max_val_pairs > 0:
        val_all = limit_records_stratified(val_all, int(args.max_val_pairs))
        val_first_last = limit_records_stratified(val_first_last, int(args.max_val_pairs))

    train_stats, _ = base_pair_statistics(
        records=train_ad,
        latents=train_latents,
        base_flow=base_flow,
        mean_flat=mean_flat,
        components=pca_components,
        faces=faces,
        device=device,
        batch_size=args.batch_size,
    )
    val_stats, val_rows = base_pair_statistics(
        records=val_all,
        latents=val_latents,
        base_flow=base_flow,
        mean_flat=mean_flat,
        components=pca_components,
        faces=faces,
        device=device,
        batch_size=args.batch_size,
    )
    _, first_last_rows = base_pair_statistics(
        records=val_first_last,
        latents=val_latents,
        base_flow=base_flow,
        mean_flat=mean_flat,
        components=pca_components,
        faces=faces,
        device=device,
        batch_size=args.batch_size,
    )
    for row in val_rows:
        row["split"] = "val"
    for row in first_last_rows:
        row["split"] = "val_first_last"
    slope_values = base_subject_slopes(
        archive=train_archive,
        latents=train_latents,
        base_flow=base_flow,
        mean_flat=mean_flat,
        components=pca_components,
        faces=faces,
        device=device,
        diagnosis="AD",
    )
    coefficient_mean, coefficient_std = coefficient_stats(train_archive, components_count)
    scalar_mean = train_stats["scalars"].mean(axis=0) if train_stats["scalars"].size else np.zeros(6, dtype=np.float32)
    scalar_std = train_stats["scalars"].std(axis=0) if train_stats["scalars"].size else np.ones(6, dtype=np.float32)
    scalar_std = np.maximum(scalar_std, 1.0e-6)
    loss_scales = {
        "pca": float(max(np.median(train_stats["pca_errors"]), 1.0e-6)),
        "vertex": float(max(np.median(train_stats["vertex_errors"]), 1.0e-6)),
        "rate": float(max(np.std(train_stats["true_rates"]), 0.02)),
        "slope": float(max(np.std(slope_values), 0.02)),
        "consistency": float(max(np.median(train_stats["pca_errors"]), 1.0e-6)),
    }
    np.savez_compressed(
        output_dir / "speed_feature_stats.npz",
        coefficient_mean=coefficient_mean.astype(np.float32),
        coefficient_std=coefficient_std.astype(np.float32),
        feature_scalar_mean=scalar_mean.astype(np.float32),
        feature_scalar_std=scalar_std.astype(np.float32),
    )
    references = {
        "val_all_pairs": grouped_reference(val_rows, "val"),
        "val_first_last": grouped_reference(first_last_rows, "val_first_last"),
        "loss_scales": loss_scales,
        "train_ad_pair_count": len(train_ad),
        "train_ad_subject_slope_count": len(slope_values),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": file_sha256(base_checkpoint),
        "base_checkpoint_epoch": int(base_payload["epoch"]),
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "frozen_v1_reference_metrics.json", references)
    write_json(
        output_dir / "input_manifest.json",
        {
            "config": str(resolve_repo_path(args.config)),
            "base_v1_checkpoint": str(base_checkpoint),
            "brainode_checkpoint": str(resolve_repo_path(config["brainode_checkpoint"])),
            "registered_mesh_tensors": str(registered_tensors),
            "pca_model_dir": str(load_pca_model(components_count)[0]),
            "components": components_count,
            "device": str(device),
            "max_train_pairs": int(args.max_train_pairs),
            "max_val_pairs": int(args.max_val_pairs),
        },
    )
    print(f"Wrote metadata to {output_dir}")
    print(f"AD train pairs: {len(train_ad)}; validation pairs: {len(val_all)}")
    print(f"Loss scales: {loss_scales}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
