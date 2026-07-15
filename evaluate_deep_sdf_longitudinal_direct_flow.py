#!/usr/bin/env python3
"""Evaluate and export diagnostics for the minimal no-MCI direct flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from longitudinal_direct_flow import direct_and_composed, per_row_latent_mse
from train_deep_sdf_longitudinal_direct_flow import (
    build_latent_regularizer_stats,
    build_data_contract,
    build_flow,
    build_loss,
    load_frozen_decoder,
    load_specs,
    make_loader,
    make_sequence_loader,
    sequence_losses_enabled,
    sequence_to_device,
    to_device,
    validate_loss_scope,
)


def load_trained_components(
    experiment_dir: Path,
    checkpoint_name: str,
    device: torch.device,
):
    specs = load_specs(experiment_dir)
    validate_loss_scope(specs)
    contract = build_data_contract(specs, experiment_dir)
    decoder, decoder_epoch = load_frozen_decoder(specs, experiment_dir, device)
    flow = build_flow(specs, device, contract)
    checkpoint_path = experiment_dir / "ModelParameters" / f"{checkpoint_name}.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing direct-flow checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    flow.load_state_dict(checkpoint["flow_state_dict"])
    flow.eval()
    return specs, contract, decoder, flow, checkpoint, decoder_epoch


def _as_list(value) -> List:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


def evaluate_split(
    *,
    split: str,
    specs: Mapping[str, object],
    contract,
    decoder: torch.nn.Module,
    flow,
    device: torch.device,
    virtual_ratios: Sequence[float],
    regularizer_stats,
) -> pd.DataFrame:
    """Return one diagnostic row for every chronological forward pair."""

    loader = make_loader(contract, split, specs, training=False)
    loss_module = build_loss(specs, decoder, flow)
    age_spec = specs.get("AgeNormalization", {})
    age_origin = float(age_spec.get("minimum_age_years", 57.0))
    age_range = float(age_spec.get("age_range_years", 34.0))
    rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = to_device(raw_batch, device)
            source_latent = batch["source_latent"]
            source_time = batch["source_time"]
            target_time = batch["target_time"]
            condition = batch["condition"]
            direct = flow.transport(
                source_latent,
                source_time,
                target_time,
                condition,
            )
            prediction_error = loss_module.decode_target_sdf_loss_per_row(
                direct,
                batch["target_samples"],
            )
            no_change_error = loss_module.decode_target_sdf_loss_per_row(
                source_latent,
                batch["target_samples"],
            )
            target_latent = batch["target_latent"]
            target_latent_mse = per_row_latent_mse(direct, target_latent)
            predicted_displacement = torch.linalg.vector_norm(
                direct - source_latent,
                dim=1,
            )
            real_displacement = torch.linalg.vector_norm(
                target_latent - source_latent,
                dim=1,
            )

            _, _, observed_composed = direct_and_composed(
                flow,
                source_latent,
                source_time,
                batch["observed_intermediate_time"],
                target_time,
                condition,
                direct_target=direct,
            )
            observed_error = per_row_latent_mse(direct, observed_composed)
            observed_mask = batch["observed_intermediate_mask"].view(-1)

            virtual_errors = []
            backward_virtual_errors = []
            future_extrapolation_errors = []
            for ratio in virtual_ratios:
                ratio_tensor = torch.full(
                    (source_latent.shape[0], 1),
                    float(ratio),
                    device=device,
                    dtype=source_latent.dtype,
                )
                virtual_time = source_time.view(-1, 1) + ratio_tensor * (
                    target_time.view(-1, 1) - source_time.view(-1, 1)
                )
                _, virtual_intermediate, virtual_composed = direct_and_composed(
                    flow,
                    source_latent,
                    source_time,
                    virtual_time,
                    target_time,
                    condition,
                    direct_target=direct,
                )
                virtual_errors.append(
                    per_row_latent_mse(direct, virtual_composed)
                )
                target_to_virtual = flow.transport(
                    target_latent,
                    target_time,
                    virtual_time,
                    condition,
                )
                backward_virtual_errors.append(
                    per_row_latent_mse(virtual_intermediate, target_to_virtual)
                )
                future_time = target_time.view(-1, 1) + ratio_tensor * (
                    target_time.view(-1, 1) - source_time.view(-1, 1)
                )
                future_direct = flow.transport(
                    source_latent,
                    source_time,
                    future_time,
                    condition,
                )
                future_composed = flow.transport(
                    direct,
                    target_time,
                    future_time,
                    condition,
                )
                future_extrapolation_errors.append(
                    per_row_latent_mse(future_direct, future_composed)
                )
            virtual_error = torch.stack(virtual_errors, dim=1).mean(dim=1)
            backward_virtual_error = torch.stack(
                backward_virtual_errors,
                dim=1,
            ).mean(dim=1)
            future_extrapolation_error = torch.stack(
                future_extrapolation_errors,
                dim=1,
            ).mean(dim=1)

            source_velocity = flow.average_velocity(
                source_latent,
                source_time,
                source_time,
                condition,
            )
            target_velocity = flow.average_velocity(
                direct,
                target_time,
                target_time,
                condition,
            )
            latent_mean = regularizer_stats.mean.to(
                device=device,
                dtype=direct.dtype,
            )
            latent_std = regularizer_stats.std.to(
                device=device,
                dtype=direct.dtype,
            )
            direct_zscore = torch.abs((direct - latent_mean) / latent_std)
            direct_zscore_p95 = torch.quantile(
                direct_zscore,
                0.95,
                dim=1,
            )
            direct_zscore_max = direct_zscore.max(dim=1).values
            normalized_gap = (
                target_time.view(-1) - source_time.view(-1)
            ).abs().clamp_min(1.0e-6)
            predicted_speed = (
                torch.linalg.vector_norm(direct - source_latent, dim=1)
                / normalized_gap
            )

            batch_size = source_latent.shape[0]
            source_order = _as_list(raw_batch["source_visit_order"])
            target_order = _as_list(raw_batch["target_visit_order"])
            for index in range(batch_size):
                source_norm = float(source_time[index].item())
                target_norm = float(target_time[index].item())
                model_error = float(prediction_error[index].item())
                baseline_error = float(no_change_error[index].item())
                has_observed = bool(observed_mask[index].item())
                rows.append(
                    {
                        "split": split,
                        "subject_id": raw_batch["subject_id"][index],
                        "diagnosis": raw_batch["diagnosis"][index],
                        "label_ad": int(condition[index].item()),
                        "source_scan_id": raw_batch["source_scan_id"][index],
                        "target_scan_id": raw_batch["target_scan_id"][index],
                        "source_visit_order": int(source_order[index]),
                        "target_visit_order": int(target_order[index]),
                        "pair_type": (
                            "adjacent"
                            if int(target_order[index]) - int(source_order[index]) == 1
                            else "nonadjacent"
                        ),
                        "source_age_norm": source_norm,
                        "target_age_norm": target_norm,
                        "source_age_years": age_origin + age_range * source_norm,
                        "target_age_years": age_origin + age_range * target_norm,
                        "gap_norm": target_norm - source_norm,
                        "gap_years": age_range * (target_norm - source_norm),
                        "model_target_sdf_l1": model_error,
                        "no_change_target_sdf_l1": baseline_error,
                        "sdf_l1_improvement": baseline_error - model_error,
                        "model_beats_no_change": model_error < baseline_error,
                        "target_latent_mse_diagnostic": float(
                            target_latent_mse[index].item()
                        ),
                        "predicted_displacement_l2": float(
                            predicted_displacement[index].item()
                        ),
                        "real_displacement_l2": float(
                            real_displacement[index].item()
                        ),
                        "displacement_magnitude_abs_error": float(
                            torch.abs(
                                predicted_displacement[index]
                                - real_displacement[index]
                            ).item()
                        ),
                        "has_observed_intermediate": has_observed,
                        "observed_cocycle_mse": (
                            float(observed_error[index].item())
                            if has_observed
                            else np.nan
                        ),
                        "virtual_cocycle_mse": float(
                            virtual_error[index].item()
                        ),
                        "backward_virtual_latent_mse": float(
                            backward_virtual_error[index].item()
                        ),
                        "future_extrapolation_cocycle_mse": float(
                            future_extrapolation_error[index].item()
                        ),
                        "predicted_latent_zscore_p95": float(
                            direct_zscore_p95[index].item()
                        ),
                        "predicted_latent_zscore_max": float(
                            direct_zscore_max[index].item()
                        ),
                        "predicted_speed_l2_per_normalized_age": float(
                            predicted_speed[index].item()
                        ),
                        "train_speed_guard_threshold_p95": float(
                            regularizer_stats.speed_percentile
                        ),
                        "source_velocity_l2_per_normalized_age": float(
                            torch.linalg.vector_norm(
                                source_velocity[index]
                            ).item()
                        ),
                        "target_velocity_l2_per_normalized_age": float(
                            torch.linalg.vector_norm(
                                target_velocity[index]
                            ).item()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "model_target_sdf_l1",
        "no_change_target_sdf_l1",
        "sdf_l1_improvement",
        "target_latent_mse_diagnostic",
        "predicted_displacement_l2",
        "real_displacement_l2",
        "displacement_magnitude_abs_error",
        "observed_cocycle_mse",
        "virtual_cocycle_mse",
        "backward_virtual_latent_mse",
        "future_extrapolation_cocycle_mse",
        "predicted_latent_zscore_p95",
        "predicted_latent_zscore_max",
        "predicted_speed_l2_per_normalized_age",
        "source_velocity_l2_per_normalized_age",
        "target_velocity_l2_per_normalized_age",
    ]
    grouping_sets: Iterable[tuple[str, List[str]]] = (
        ("overall", []),
        ("diagnosis", ["diagnosis"]),
        ("pair_type", ["pair_type"]),
        ("diagnosis_pair_type", ["diagnosis", "pair_type"]),
    )
    output: List[Dict[str, object]] = []
    for group_name, columns in grouping_sets:
        groups = [((), frame)] if not columns else frame.groupby(columns, dropna=False)
        for key, group in groups:
            key_values = key if isinstance(key, tuple) else (key,)
            row: Dict[str, object] = {
                "grouping": group_name,
                "rows": int(len(group)),
                "model_beats_no_change_fraction": float(
                    group["model_beats_no_change"].mean()
                ),
            }
            for column, value in zip(columns, key_values):
                row[column] = value
            for metric in metrics:
                row[f"{metric}_mean"] = float(group[metric].mean())
            output.append(row)
    return pd.DataFrame(output)


def trajectory_diagnostics(
    *,
    split: str,
    specs: Mapping[str, object],
    contract,
    flow,
    device: torch.device,
    regularizer_stats,
    subjects_per_diagnosis: int,
    future_years: float,
    points: int,
) -> pd.DataFrame:
    """Dense direct trajectories from each selected subject's first scan."""

    age_spec = specs.get("AgeNormalization", {})
    age_origin = float(age_spec.get("minimum_age_years", 57.0))
    age_range = float(age_spec.get("age_range_years", 34.0))
    split_metadata = contract.metadata.loc[
        contract.metadata["split"] == split
    ].copy()
    selected: List[str] = []
    for diagnosis in ("CN", "AD"):
        ids = (
            split_metadata.loc[split_metadata["diagnosis"] == diagnosis, "subject_id"]
            .drop_duplicates()
            .sort_values()
            .head(int(subjects_per_diagnosis))
        )
        selected.extend(ids.astype(str).tolist())

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for subject_id in selected:
            subject = split_metadata.loc[
                split_metadata["subject_id"].astype(str) == subject_id
            ].sort_values(["continuous_age_norm", "visit_order"])
            first = subject.iloc[0]
            last_observed_age = float(subject["continuous_age_years"].max())
            source_age = float(first["continuous_age_years"])
            ages = np.linspace(
                source_age,
                last_observed_age + float(future_years),
                int(points),
            )
            source_latent = contract.latent_maps[split][
                str(first["scan_id"])
            ].to(device).view(1, -1)
            source_time = torch.tensor(
                [[float(first["continuous_age_norm"])]],
                device=device,
                dtype=source_latent.dtype,
            )
            condition = torch.tensor(
                [[float(first["label_ad"])]],
                device=device,
                dtype=source_latent.dtype,
            )
            real_by_age = {
                round(float(row["continuous_age_years"]), 6): str(row["scan_id"])
                for _, row in subject.iterrows()
            }
            latent_mean = regularizer_stats.mean.to(
                device=device,
                dtype=source_latent.dtype,
            )
            latent_std = regularizer_stats.std.to(
                device=device,
                dtype=source_latent.dtype,
            )
            for age in ages:
                target_time = torch.tensor(
                    [[(float(age) - age_origin) / age_range]],
                    device=device,
                    dtype=source_latent.dtype,
                )
                predicted = flow.transport(
                    source_latent,
                    source_time,
                    target_time,
                    condition,
                )
                velocity = flow.average_velocity(
                    predicted,
                    target_time,
                    target_time,
                    condition,
                )
                z_score = torch.abs((predicted - latent_mean) / latent_std)
                real_scan_id = real_by_age.get(round(float(age), 6))
                real_latent_mse = np.nan
                if real_scan_id is not None:
                    real_latent = contract.latent_maps[split][real_scan_id].to(
                        device
                    ).view(1, -1)
                    real_latent_mse = float(
                        per_row_latent_mse(predicted, real_latent)[0].item()
                    )
                rows.append(
                    {
                        "split": split,
                        "subject_id": subject_id,
                        "diagnosis": str(first["diagnosis"]),
                        "age_years": float(age),
                        "age_norm": float(target_time.item()),
                        "region": (
                            "observed_range"
                            if float(age) <= last_observed_age
                            else "future_unvalidated"
                        ),
                        "is_real_scan_age": real_scan_id is not None,
                        "real_scan_id": real_scan_id or "",
                        "direct_displacement_l2": float(
                            torch.linalg.vector_norm(
                                predicted - source_latent
                            ).item()
                        ),
                        "latent_l2": float(
                            torch.linalg.vector_norm(predicted).item()
                        ),
                        "latent_zscore_p95": float(
                            torch.quantile(z_score.reshape(-1), 0.95).item()
                        ),
                        "latent_zscore_max": float(z_score.max().item()),
                        "instantaneous_velocity_l2_per_normalized_age": float(
                            torch.linalg.vector_norm(velocity).item()
                        ),
                        "train_speed_guard_threshold_p95": float(
                            regularizer_stats.speed_percentile
                        ),
                        "real_latent_mse_diagnostic": real_latent_mse,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_sequence_split(
    *,
    split: str,
    specs: Mapping[str, object],
    contract,
    decoder: torch.nn.Module,
    flow,
    device: torch.device,
    regularizer_stats,
) -> pd.DataFrame:
    """Return one row for every real target step in every rollout sequence."""

    loader = make_sequence_loader(contract, split, specs, training=False)
    loss_module = build_loss(specs, decoder, flow)
    rows: List[Dict[str, object]] = []
    latent_mean = regularizer_stats.mean.to(device=device)
    latent_std = regularizer_stats.std.to(device=device)
    with torch.no_grad():
        for raw_batch in loader:
            batch = sequence_to_device(raw_batch, device)
            latents = batch["latents"]
            times = batch["times"]
            age_years = batch["age_years"]
            visit_orders = batch["visit_orders"]
            condition = batch["condition"]
            target_samples = batch["target_samples"]
            step_mask = batch["step_mask"]
            scan_ids = batch["scan_ids"]
            subject_ids = batch["subject_id"]
            diagnoses = batch["diagnosis"]
            if not isinstance(latents, torch.Tensor):
                raise TypeError("Sequence latents must be a tensor")
            source_latent = latents[:, 0, :]
            source_time = times[:, 0]
            current_latent = source_latent
            previous_time = source_time
            for step_index in range(latents.shape[1] - 1):
                target_time = times[:, step_index + 1]
                samples = target_samples[:, step_index]
                rollout_latent = flow.transport(
                    current_latent,
                    previous_time,
                    target_time,
                    condition,
                )
                one_shot_latent = flow.transport(
                    source_latent,
                    source_time,
                    target_time,
                    condition,
                )
                rollout_error = loss_module.decode_target_sdf_loss_per_row(
                    rollout_latent,
                    samples,
                )
                one_shot_error = loss_module.decode_target_sdf_loss_per_row(
                    one_shot_latent,
                    samples,
                )
                no_change_error = loss_module.decode_target_sdf_loss_per_row(
                    source_latent,
                    samples,
                )
                cocycle_error = (
                    per_row_latent_mse(one_shot_latent, rollout_latent)
                    if step_index >= 1
                    else torch.full_like(rollout_error, float("nan"))
                )
                real_target_delta = latents[:, step_index + 1, :] - source_latent
                predicted_delta = rollout_latent - source_latent
                predicted_displacement = torch.linalg.vector_norm(
                    predicted_delta,
                    dim=1,
                )
                real_displacement = torch.linalg.vector_norm(
                    real_target_delta,
                    dim=1,
                )
                direction_cosine = torch.nn.functional.cosine_similarity(
                    predicted_delta,
                    real_target_delta,
                    dim=1,
                    eps=1.0e-8,
                )
                z_score = torch.abs(
                    (
                        rollout_latent
                        - latent_mean.to(dtype=rollout_latent.dtype)
                    )
                    / latent_std.to(dtype=rollout_latent.dtype)
                )
                z_p95 = torch.quantile(z_score, 0.95, dim=1)
                z_max = z_score.max(dim=1).values
                speed = (
                    torch.linalg.vector_norm(
                        rollout_latent - current_latent,
                        dim=1,
                    )
                    / (target_time - previous_time).abs().clamp_min(1.0e-6)
                )
                valid = step_mask[:, step_index].to(dtype=torch.bool)
                for row_index in range(latents.shape[0]):
                    if not bool(valid[row_index].item()):
                        continue
                    row_scan_ids = scan_ids[row_index]
                    rows.append(
                        {
                            "split": split,
                            "subject_id": subject_ids[row_index],
                            "diagnosis": diagnoses[row_index],
                            "label_ad": int(condition[row_index].item()),
                            "source_scan_id": row_scan_ids[0],
                            "target_scan_id": row_scan_ids[step_index + 1],
                            "step_index": int(step_index + 1),
                            "target_visit_order": int(
                                visit_orders[row_index, step_index + 1].item()
                            ),
                            "source_age_norm": float(source_time[row_index].item()),
                            "target_age_norm": float(target_time[row_index].item()),
                            "source_age_years": float(age_years[row_index, 0].item()),
                            "target_age_years": float(
                                age_years[row_index, step_index + 1].item()
                            ),
                            "gap_from_source_norm": float(
                                (
                                    target_time[row_index]
                                    - source_time[row_index]
                                ).item()
                            ),
                            "rollout_step_gap_norm": float(
                                (
                                    target_time[row_index]
                                    - previous_time[row_index]
                                ).item()
                            ),
                            "sequence_rollout_sdf_l1": float(
                                rollout_error[row_index].item()
                            ),
                            "sequence_one_shot_sdf_l1": float(
                                one_shot_error[row_index].item()
                            ),
                            "sequence_no_change_sdf_l1": float(
                                no_change_error[row_index].item()
                            ),
                            "sequence_rollout_improvement": float(
                                (
                                    no_change_error[row_index]
                                    - rollout_error[row_index]
                                ).item()
                            ),
                            "sequence_one_shot_improvement": float(
                                (
                                    no_change_error[row_index]
                                    - one_shot_error[row_index]
                                ).item()
                            ),
                            "sequence_cocycle_mse": float(
                                cocycle_error[row_index].item()
                            ),
                            "rollout_displacement_l2_from_source": float(
                                predicted_displacement[row_index].item()
                            ),
                            "real_displacement_l2_from_source": float(
                                real_displacement[row_index].item()
                            ),
                            "sequence_displacement_magnitude_abs_error": float(
                                torch.abs(
                                    predicted_displacement[row_index]
                                    - real_displacement[row_index]
                                ).item()
                            ),
                            "direction_cosine_from_source": float(
                                direction_cosine[row_index].item()
                            ),
                            "rollout_latent_zscore_p95": float(
                                z_p95[row_index].item()
                            ),
                            "rollout_latent_zscore_max": float(
                                z_max[row_index].item()
                            ),
                            "rollout_speed_l2_per_normalized_age": float(
                                speed[row_index].item()
                            ),
                            "train_speed_guard_threshold_p95": float(
                                regularizer_stats.speed_percentile
                            ),
                        }
                    )
                current_latent = rollout_latent
                previous_time = target_time
    return pd.DataFrame(rows)


def summarize_sequences(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "sequence_rollout_sdf_l1",
        "sequence_one_shot_sdf_l1",
        "sequence_no_change_sdf_l1",
        "sequence_rollout_improvement",
        "sequence_one_shot_improvement",
        "sequence_cocycle_mse",
        "rollout_displacement_l2_from_source",
        "real_displacement_l2_from_source",
        "sequence_displacement_magnitude_abs_error",
        "direction_cosine_from_source",
        "rollout_latent_zscore_p95",
        "rollout_latent_zscore_max",
        "rollout_speed_l2_per_normalized_age",
    ]
    grouping_sets: Iterable[tuple[str, List[str]]] = (
        ("overall", []),
        ("diagnosis", ["diagnosis"]),
        ("step_index", ["step_index"]),
        ("diagnosis_step_index", ["diagnosis", "step_index"]),
    )
    output: List[Dict[str, object]] = []
    for group_name, columns in grouping_sets:
        groups = [((), frame)] if not columns else frame.groupby(columns, dropna=False)
        for key, group in groups:
            key_values = key if isinstance(key, tuple) else (key,)
            row: Dict[str, object] = {
                "grouping": group_name,
                "rows": int(len(group)),
                "rollout_beats_no_change_fraction": float(
                    (group["sequence_rollout_sdf_l1"] < group["sequence_no_change_sdf_l1"]).mean()
                ),
                "one_shot_beats_no_change_fraction": float(
                    (group["sequence_one_shot_sdf_l1"] < group["sequence_no_change_sdf_l1"]).mean()
                ),
            }
            for column, value in zip(columns, key_values):
                row[column] = value
            for metric in metrics:
                row[f"{metric}_mean"] = float(group[metric].mean())
            output.append(row)
    return pd.DataFrame(output)


def export_example_meshes(
    *,
    split: str,
    specs: Mapping[str, object],
    contract,
    decoder: torch.nn.Module,
    flow,
    device: torch.device,
    output_dir: Path,
    subjects_per_diagnosis: int,
    future_years: float,
    resolution: int,
) -> pd.DataFrame:
    """Export baseline-to-real, virtual-intermediate, and future shapes."""

    if device.type != "cuda":
        raise RuntimeError("Mesh export requires CUDA because deep_sdf.mesh uses CUDA.")
    from deep_sdf import mesh as deep_sdf_mesh

    age_spec = specs.get("AgeNormalization", {})
    age_origin = float(age_spec.get("minimum_age_years", 57.0))
    age_range = float(age_spec.get("age_range_years", 34.0))
    metadata = contract.metadata.loc[
        contract.metadata["split"] == split
    ].copy()
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for diagnosis in ("CN", "AD"):
        subject_ids = (
            metadata.loc[metadata["diagnosis"] == diagnosis, "subject_id"]
            .drop_duplicates()
            .sort_values()
            .head(int(subjects_per_diagnosis))
        )
        for subject_id in subject_ids:
            subject = metadata.loc[
                metadata["subject_id"] == subject_id
            ].sort_values(["continuous_age_norm", "visit_order"])
            first, last = subject.iloc[0], subject.iloc[-1]
            source_latent = contract.latent_maps[split][str(first["scan_id"])].to(
                device
            ).view(1, -1)
            source_time = torch.tensor(
                [[float(first["continuous_age_norm"])]],
                device=device,
                dtype=source_latent.dtype,
            )
            condition = torch.tensor(
                [[float(first["label_ad"])]],
                device=device,
                dtype=source_latent.dtype,
            )
            ages_and_kinds = [
                (float(first["continuous_age_years"]), "baseline"),
                (
                    0.5
                    * (
                        float(first["continuous_age_years"])
                        + float(last["continuous_age_years"])
                    ),
                    "virtual_midpoint",
                ),
                (float(last["continuous_age_years"]), "last_real_target"),
                (
                    float(last["continuous_age_years"]) + float(future_years),
                    "future_unvalidated",
                ),
            ]
            for age, kind in ages_and_kinds:
                target_time = torch.tensor(
                    [[(age - age_origin) / age_range]],
                    device=device,
                    dtype=source_latent.dtype,
                )
                with torch.no_grad():
                    latent = flow.transport(
                        source_latent,
                        source_time,
                        target_time,
                        condition,
                    )
                stem = (
                    f"{split}_{diagnosis}_{subject_id}_{kind}_age_{age:.2f}"
                    .replace("/", "_")
                    .replace(" ", "_")
                )
                output_stem = mesh_dir / stem
                deep_sdf_mesh.create_mesh(
                    decoder,
                    latent,
                    filename=str(output_stem),
                    N=int(resolution),
                    max_batch=2**18,
                )
                rows.append(
                    {
                        "split": split,
                        "subject_id": str(subject_id),
                        "diagnosis": diagnosis,
                        "kind": kind,
                        "age_years": age,
                        "predicted_mesh": str(output_stem) + ".ply",
                        "baseline_real_mesh": str(first["mesh_path"]),
                        "last_real_mesh": str(last["mesh_path"]),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--experiment", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "both", "all"),
        default="val",
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--virtual-ratios", default="0.25,0.5,0.75")
    parser.add_argument("--subjects-per-diagnosis", type=int, default=1)
    parser.add_argument("--future-years", type=float, default=2.0)
    parser.add_argument("--trajectory-points", type=int, default=41)
    parser.add_argument("--export-example-meshes", action="store_true")
    parser.add_argument("--mesh-resolution", type=int, default=256)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment).resolve()
    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu was given but CUDA is unavailable")
        torch.cuda.set_device(int(args.gpu))
    device = torch.device(
        f"cuda:{torch.cuda.current_device()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    specs, contract, decoder, flow, checkpoint, decoder_epoch = (
        load_trained_components(
            experiment_dir,
            args.checkpoint,
            device,
        )
    )
    regularizer_stats = build_latent_regularizer_stats(
        contract,
        split="train",
        speed_percentile=float(specs.get("SpeedGuardPercentile", 95.0)),
    )
    ratios = [float(value) for value in args.virtual_ratios.split(",")]
    if args.split == "both":
        splits = ("val", "test")
    elif args.split == "all":
        splits = ("train", "val", "test")
    else:
        splits = (args.split,)
    output_dir = experiment_dir / "analysis" / f"checkpoint_{args.checkpoint}"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_pairs = []
    all_trajectories = []
    all_sequences = []
    for split in splits:
        pair_frame = evaluate_split(
            split=split,
            specs=specs,
            contract=contract,
            decoder=decoder,
            flow=flow,
            device=device,
            virtual_ratios=ratios,
            regularizer_stats=regularizer_stats,
        )
        pair_frame.to_csv(output_dir / f"{split}_pair_metrics.csv", index=False)
        summarize_pairs(pair_frame).to_csv(
            output_dir / f"{split}_summary.csv",
            index=False,
        )
        trajectory_frame = trajectory_diagnostics(
            split=split,
            specs=specs,
            contract=contract,
            flow=flow,
            device=device,
            regularizer_stats=regularizer_stats,
            subjects_per_diagnosis=args.subjects_per_diagnosis,
            future_years=args.future_years,
            points=args.trajectory_points,
        )
        trajectory_frame.to_csv(
            output_dir / f"{split}_trajectories.csv",
            index=False,
        )
        all_pairs.append(pair_frame)
        all_trajectories.append(trajectory_frame)
        if sequence_losses_enabled(specs):
            sequence_frame = evaluate_sequence_split(
                split=split,
                specs=specs,
                contract=contract,
                decoder=decoder,
                flow=flow,
                device=device,
                regularizer_stats=regularizer_stats,
            )
            sequence_frame.to_csv(
                output_dir / f"{split}_sequence_metrics.csv",
                index=False,
            )
            summarize_sequences(sequence_frame).to_csv(
                output_dir / f"{split}_sequence_summary.csv",
                index=False,
            )
            all_sequences.append(sequence_frame)
        if args.export_example_meshes:
            mesh_manifest = export_example_meshes(
                split=split,
                specs=specs,
                contract=contract,
                decoder=decoder,
                flow=flow,
                device=device,
                output_dir=output_dir,
                subjects_per_diagnosis=args.subjects_per_diagnosis,
                future_years=args.future_years,
                resolution=args.mesh_resolution,
            )
            mesh_manifest.to_csv(
                output_dir / f"{split}_mesh_manifest.csv",
                index=False,
            )

    combined_pairs = pd.concat(all_pairs, ignore_index=True)
    combined_trajectories = pd.concat(all_trajectories, ignore_index=True)
    combined_pairs.to_csv(output_dir / "all_pair_metrics.csv", index=False)
    combined_trajectories.to_csv(
        output_dir / "all_trajectories.csv",
        index=False,
    )
    combined_sequences = (
        pd.concat(all_sequences, ignore_index=True)
        if all_sequences
        else pd.DataFrame()
    )
    if not combined_sequences.empty:
        combined_sequences.to_csv(
            output_dir / "all_sequence_metrics.csv",
            index=False,
        )
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "decoder_epoch": int(decoder_epoch),
        "splits": list(splits),
        "pairs": int(len(combined_pairs)),
        "virtual_ratios": ratios,
        "mean_model_target_sdf_l1": float(
            combined_pairs["model_target_sdf_l1"].mean()
        ),
        "mean_no_change_target_sdf_l1": float(
            combined_pairs["no_change_target_sdf_l1"].mean()
        ),
        "mean_sdf_l1_improvement": float(
            combined_pairs["sdf_l1_improvement"].mean()
        ),
        "mean_displacement_magnitude_abs_error": float(
            combined_pairs["displacement_magnitude_abs_error"].mean()
        ),
        "model_beats_no_change_fraction": float(
            combined_pairs["model_beats_no_change"].mean()
        ),
        "mean_backward_virtual_latent_mse": float(
            combined_pairs["backward_virtual_latent_mse"].mean()
        ),
        "mean_future_extrapolation_cocycle_mse": float(
            combined_pairs["future_extrapolation_cocycle_mse"].mean()
        ),
        "mean_predicted_latent_zscore_p95": float(
            combined_pairs["predicted_latent_zscore_p95"].mean()
        ),
        "mean_predicted_speed_l2_per_normalized_age": float(
            combined_pairs["predicted_speed_l2_per_normalized_age"].mean()
        ),
        "future_trajectory_region_is_unvalidated": True,
    }
    if not combined_sequences.empty:
        summary.update(
            {
                "sequence_steps": int(len(combined_sequences)),
                "mean_sequence_rollout_sdf_l1": float(
                    combined_sequences["sequence_rollout_sdf_l1"].mean()
                ),
                "mean_sequence_no_change_sdf_l1": float(
                    combined_sequences["sequence_no_change_sdf_l1"].mean()
                ),
                "mean_sequence_rollout_improvement": float(
                    combined_sequences["sequence_rollout_improvement"].mean()
                ),
                "mean_sequence_displacement_magnitude_abs_error": float(
                    combined_sequences["sequence_displacement_magnitude_abs_error"].mean()
                ),
                "sequence_rollout_beats_no_change_fraction": float(
                    (
                        combined_sequences["sequence_rollout_sdf_l1"]
                        < combined_sequences["sequence_no_change_sdf_l1"]
                    ).mean()
                ),
                "mean_sequence_cocycle_mse": float(
                    combined_sequences["sequence_cocycle_mse"].mean()
                ),
                "mean_sequence_rollout_latent_zscore_p95": float(
                    combined_sequences["rollout_latent_zscore_p95"].mean()
                ),
            }
        )
    with (output_dir / "evaluation_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
