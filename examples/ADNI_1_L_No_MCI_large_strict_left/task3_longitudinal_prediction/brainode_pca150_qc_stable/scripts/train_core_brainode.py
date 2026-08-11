#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from brainode_model import (
    BrainODEWithCognition,
    ODEFuncWithAttention,
    cognition_bce_loss,
    integrate_autoregressive_rk4,
    integrate_sequence_rk4,
)
from core_brainode_common import (
    TASK_DIR,
    load_config,
    load_json,
    resolve_repo_path,
    write_json,
)


@dataclass
class TrajectoryRecord:
    subject_id: str
    direction: str
    start_visit_order: int
    length: int
    times: np.ndarray
    targets: np.ndarray
    conditions: np.ndarray
    condition: float


class TrajectoryDataset(Dataset):
    def __init__(self, records: list[TrajectoryRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "subject_id": record.subject_id,
            "direction": record.direction,
            "start_visit_order": record.start_visit_order,
            "length": record.length,
            "times": torch.from_numpy(record.times.copy()).float(),
            "targets": torch.from_numpy(record.targets.copy()).float(),
            "conditions": torch.from_numpy(record.conditions.copy()).float(),
            "condition": torch.tensor(record.condition, dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the core BrainODE PCA-150 attention ODE on Task 3 datasets."
    )
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "core_brainode.json"),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string. Defaults to the config value, then CUDA if available.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the configured number of epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the configured batch size.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the configured run name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/latest.pth in the resolved run directory if present.",
    )
    parser.add_argument(
        "--max-train-trajectories",
        type=int,
        default=None,
        help="Optional cap for train trajectories, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-val-trajectories",
        type=int,
        default=None,
        help="Optional cap for validation forward trajectories, useful for smoke tests.",
    )
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Skip TensorBoard writer creation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the configured random seed.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_npz_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def build_subject_sequences(archive: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    subject_ids = archive["subject_ids"]
    subject_visit_offsets = archive["subject_visit_offsets"]
    subject_cognition = archive["subject_cognition"]
    visit_times = archive["visit_continuous_age_norm"]
    visit_codes = archive["visit_pca_150"]
    visit_cognition = archive["visit_cognition"]
    sequences = []
    for subject_index, subject_id in enumerate(subject_ids.tolist()):
        start = int(subject_visit_offsets[subject_index])
        end = int(subject_visit_offsets[subject_index + 1])
        sequences.append(
            {
                "subject_id": str(subject_id),
                "condition": float(subject_cognition[subject_index]),
                "times": visit_times[start:end].astype(np.float32).copy(),
                "targets": visit_codes[start:end].astype(np.float32).copy(),
                "conditions": visit_cognition[start:end].astype(np.float32).copy(),
            }
        )
    return sequences


def build_trajectory_records(
    sequences: list[dict[str, Any]],
    include_backward: bool,
    include_length_one: bool,
    only_start_visit_zero: bool,
) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []
    for sequence in sequences:
        subject_id = sequence["subject_id"]
        times = sequence["times"]
        targets = sequence["targets"]
        conditions = sequence["conditions"]
        num_visits = len(times)
        for start_visit_order in range(num_visits):
            if only_start_visit_zero and start_visit_order != 0:
                continue
            forward_times = times[start_visit_order:]
            forward_targets = targets[start_visit_order:]
            forward_conditions = conditions[start_visit_order:]
            if include_length_one or len(forward_times) > 1:
                records.append(
                    TrajectoryRecord(
                        subject_id=subject_id,
                        direction="forward",
                        start_visit_order=start_visit_order,
                        length=len(forward_times),
                        times=forward_times,
                        targets=forward_targets,
                        conditions=forward_conditions,
                        condition=float(forward_conditions[0]),
                    )
                )
            if include_backward:
                backward_times = times[: start_visit_order + 1][::-1].copy()
                backward_targets = targets[: start_visit_order + 1][::-1].copy()
                backward_conditions = conditions[: start_visit_order + 1][::-1].copy()
                if include_length_one or len(backward_times) > 1:
                    records.append(
                        TrajectoryRecord(
                            subject_id=subject_id,
                            direction="backward",
                            start_visit_order=start_visit_order,
                            length=len(backward_times),
                            times=backward_times,
                            targets=backward_targets,
                            conditions=backward_conditions,
                            condition=float(backward_conditions[0]),
                        )
                    )
    return records


def collate_records(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "subject_id": [item["subject_id"] for item in batch],
        "direction": [item["direction"] for item in batch],
        "start_visit_order": torch.tensor(
            [item["start_visit_order"] for item in batch], dtype=torch.long
        ),
        "length": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "times": torch.stack([item["times"] for item in batch], dim=0),
        "targets": torch.stack([item["targets"] for item in batch], dim=0),
        "conditions": torch.stack([item["conditions"] for item in batch], dim=0),
        "condition": torch.stack([item["condition"] for item in batch], dim=0),
    }


def build_length_grouped_loaders(
    records: list[TrajectoryRecord],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> list[tuple[int, DataLoader]]:
    grouped: dict[int, list[TrajectoryRecord]] = {}
    for record in records:
        grouped.setdefault(record.length, []).append(record)
    loaders = []
    for length in sorted(grouped):
        dataset = TrajectoryDataset(grouped[length])
        loaders.append(
            (
                length,
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    collate_fn=collate_records,
                ),
            )
        )
    return loaders


def resolve_device(preferred: str | None) -> torch.device:
    if preferred:
        device = torch.device(preferred)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in the selected environment.")
    return device


def inverse_transform_pca(
    coefficients: torch.Tensor,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    original_shape = coefficients.shape[:-1]
    flat = coefficients.reshape(-1, coefficients.shape[-1])
    reconstructed = flat @ components + mean_flat
    return reconstructed.reshape(*original_shape, mean_flat.shape[0])


def endpoint_vertex_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    prediction_vertices = inverse_transform_pca(prediction, mean_flat, components)
    target_vertices = inverse_transform_pca(target, mean_flat, components)
    return torch.mean(torch.abs(prediction_vertices - target_vertices))


def trajectory_endpoint_pca_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return torch.mean((prediction[:, -1, :] - target[:, -1, :]) ** 2)


def full_trajectory_pca_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def apply_shared_augmentation(
    targets: torch.Tensor,
    noise_std: float,
    scale_min: float,
    scale_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = targets.shape[0]
    noise = torch.randn(batch_size, 1, targets.shape[-1], device=targets.device) * noise_std
    scale = torch.empty(batch_size, 1, 1, device=targets.device).uniform_(scale_min, scale_max)
    augmented_targets = (targets + noise) * scale
    return augmented_targets, noise, scale


def full_brainode_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "use_cognition_estimator": False,
        "use_autoregressive_rollout": False,
        "use_pseudo_cognitive_sampling": False,
        "shape_loss_lambda": 1.0,
        "cognition_loss_lambda": 0.0,
        "autoregressive_loss_lambda": 0.0,
        "pseudo_cognition_loss_lambda": 0.0,
        "cognition_hidden_dims": [256, 128],
        "cognition_dropout": 0.0,
        "require_transition_trajectories": False,
    }
    provided = dict(config.get("full_brainode", {}))
    return {**defaults, **provided}


def build_model(
    latent_dim: int,
    model_config: dict[str, Any],
    full_config: dict[str, Any],
) -> nn.Module:
    if bool(full_config.get("use_cognition_estimator", False)):
        return BrainODEWithCognition(
            latent_dim=latent_dim,
            condition_dim=int(model_config["condition_dim"]),
            attention_dim=int(model_config["attention_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            cognition_hidden_dims=tuple(
                int(value) for value in full_config.get("cognition_hidden_dims", [256, 128])
            ),
            cognition_dropout=float(full_config.get("cognition_dropout", 0.0)),
        )
    return ODEFuncWithAttention(
        latent_dim=latent_dim,
        condition_dim=int(model_config["condition_dim"]),
        attention_dim=int(model_config["attention_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
    )


def has_cognition_estimator(model: nn.Module) -> bool:
    return callable(getattr(model, "estimate_cognition", None))


def real_cognition_loss(
    model: nn.Module,
    targets: torch.Tensor,
    times: torch.Tensor,
    condition_targets: torch.Tensor,
) -> torch.Tensor:
    estimator = getattr(model, "estimate_cognition", None)
    if estimator is None:
        return targets.new_tensor(0.0)
    flat_targets = targets.reshape(-1, targets.shape[-1])
    flat_times = times.reshape(-1)
    flat_conditions = condition_targets.reshape(-1)
    prediction = estimator(flat_targets, flat_times)
    return cognition_bce_loss(prediction, flat_conditions)


def pseudo_cognition_loss(
    model: nn.Module,
    targets: torch.Tensor,
    times: torch.Tensor,
    condition_targets: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    estimator = getattr(model, "estimate_cognition", None)
    if estimator is None:
        return targets.new_tensor(0.0), 0
    start_condition = condition_targets[:, 0]
    end_condition = condition_targets[:, -1]
    transition_mask = (end_condition - start_condition).abs() > 1e-6
    if not bool(transition_mask.any()):
        return targets.new_tensor(0.0), 0
    selected_targets = targets[transition_mask]
    selected_times = times[transition_mask]
    selected_conditions = condition_targets[transition_mask]
    alpha = torch.rand(
        selected_targets.shape[0],
        1,
        device=targets.device,
        dtype=targets.dtype,
    )
    pseudo_latent = selected_targets[:, 0, :] + alpha * (
        selected_targets[:, -1, :] - selected_targets[:, 0, :]
    )
    pseudo_time = selected_times[:, 0] + alpha.squeeze(1) * (
        selected_times[:, -1] - selected_times[:, 0]
    )
    pseudo_condition = selected_conditions[:, 0] + alpha.squeeze(1) * (
        selected_conditions[:, -1] - selected_conditions[:, 0]
    )
    prediction = estimator(pseudo_latent, pseudo_time)
    return cognition_bce_loss(prediction, pseudo_condition), int(selected_targets.shape[0])


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_metric_value: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric_value": best_metric_value,
            "history": history,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]]]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    return (
        int(payload["epoch"]),
        float(payload.get("best_metric_value", math.inf)),
        list(payload.get("history", [])),
    )


def append_history_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def summarize_records(records: Iterable[TrajectoryRecord]) -> dict[str, Any]:
    count_by_length: dict[int, int] = {}
    count_by_direction: dict[str, int] = {}
    for record in records:
        count_by_length[record.length] = count_by_length.get(record.length, 0) + 1
        count_by_direction[record.direction] = count_by_direction.get(record.direction, 0) + 1
    return {
        "count": int(sum(count_by_length.values())),
        "count_by_length": {str(key): value for key, value in sorted(count_by_length.items())},
        "count_by_direction": dict(sorted(count_by_direction.items())),
    }


def train_one_epoch(
    model: nn.Module,
    loaders: list[tuple[int, DataLoader]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    integration_substeps: int,
    noise_std: float,
    scale_min: float,
    scale_max: float,
    gradient_clip_norm: float | None,
    full_config: dict[str, Any],
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_shape_loss = 0.0
    total_cognition_loss = 0.0
    total_autoregressive_loss = 0.0
    total_pseudo_cognition_loss = 0.0
    total_endpoint_loss = 0.0
    total_batches = 0
    total_trajectories = 0
    total_endpoints = 0
    total_transition_trajectories = 0
    for _, loader in loaders:
        for batch in loader:
            times = batch["times"].to(device)
            targets = batch["targets"].to(device)
            condition_targets = batch["conditions"].to(device)
            condition = batch["condition"].to(device)

            augmented_targets, _, _ = apply_shared_augmentation(
                targets=targets,
                noise_std=noise_std,
                scale_min=scale_min,
                scale_max=scale_max,
            )
            initial_state = augmented_targets[:, 0, :]
            prediction = integrate_sequence_rk4(
                func=model,
                initial_state=initial_state,
                times=times,
                condition=condition,
                substeps=integration_substeps,
            )
            shape_loss = full_trajectory_pca_mse(prediction, augmented_targets)
            loss = float(full_config["shape_loss_lambda"]) * shape_loss
            cognition_loss = targets.new_tensor(0.0)
            autoregressive_loss = targets.new_tensor(0.0)
            pseudo_loss = targets.new_tensor(0.0)
            pseudo_count = 0
            if bool(full_config.get("use_cognition_estimator", False)):
                cognition_loss = real_cognition_loss(
                    model=model,
                    targets=augmented_targets,
                    times=times,
                    condition_targets=condition_targets,
                )
                loss = loss + float(full_config["cognition_loss_lambda"]) * cognition_loss
            if bool(full_config.get("use_autoregressive_rollout", False)):
                autoregressive_prediction, _ = integrate_autoregressive_rk4(
                    func=model,
                    initial_state=initial_state,
                    times=times,
                    initial_condition=condition,
                    substeps=integration_substeps,
                )
                autoregressive_loss = full_trajectory_pca_mse(
                    autoregressive_prediction,
                    augmented_targets,
                )
                loss = loss + float(full_config["autoregressive_loss_lambda"]) * autoregressive_loss
            if bool(full_config.get("use_pseudo_cognitive_sampling", False)):
                pseudo_loss, pseudo_count = pseudo_cognition_loss(
                    model=model,
                    targets=augmented_targets,
                    times=times,
                    condition_targets=condition_targets,
                )
                loss = loss + float(full_config["pseudo_cognition_loss_lambda"]) * pseudo_loss
            if not loss.requires_grad:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip_norm is not None and gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

            batch_size = times.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_shape_loss += float(shape_loss.item()) * batch_size
            total_cognition_loss += float(cognition_loss.item()) * batch_size
            total_autoregressive_loss += float(autoregressive_loss.item()) * batch_size
            total_pseudo_cognition_loss += float(pseudo_loss.item()) * batch_size
            total_transition_trajectories += pseudo_count
            if targets.shape[1] > 1:
                endpoint_loss = trajectory_endpoint_pca_mse(prediction, augmented_targets)
                total_endpoint_loss += float(endpoint_loss.item()) * batch_size
                total_endpoints += batch_size
            total_batches += 1
            total_trajectories += batch_size

    return {
        "train_total_loss": total_loss / max(total_trajectories, 1),
        "train_full_pca_mse": total_shape_loss / max(total_trajectories, 1),
        "train_endpoint_pca_mse": total_endpoint_loss / max(total_endpoints, 1),
        "train_cognition_bce": total_cognition_loss / max(total_trajectories, 1),
        "train_autoregressive_pca_mse": total_autoregressive_loss / max(total_trajectories, 1),
        "train_pseudo_cognition_bce": total_pseudo_cognition_loss / max(total_trajectories, 1),
        "train_pseudo_transition_trajectories": float(total_transition_trajectories),
        "train_batches": float(total_batches),
        "train_trajectories": float(total_trajectories),
    }


@torch.no_grad()
def evaluate_forward_records(
    model: nn.Module,
    loaders: list[tuple[int, DataLoader]],
    device: torch.device,
    integration_substeps: int,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
    use_autoregressive_rollout: bool = False,
) -> dict[str, float]:
    model.eval()
    total_pca_mse = 0.0
    total_vertex_mae = 0.0
    total_cognition_bce = 0.0
    total_records = 0
    for _, loader in loaders:
        for batch in loader:
            times = batch["times"].to(device)
            targets = batch["targets"].to(device)
            condition_targets = batch["conditions"].to(device)
            condition = batch["condition"].to(device)
            initial_state = targets[:, 0, :]
            if use_autoregressive_rollout:
                prediction, _ = integrate_autoregressive_rk4(
                    func=model,
                    initial_state=initial_state,
                    times=times,
                    initial_condition=condition,
                    substeps=integration_substeps,
                )
            else:
                prediction = integrate_sequence_rk4(
                    func=model,
                    initial_state=initial_state,
                    times=times,
                    condition=condition,
                    substeps=integration_substeps,
                )
            batch_size = times.shape[0]
            total_pca_mse += float(trajectory_endpoint_pca_mse(prediction, targets).item()) * batch_size
            total_vertex_mae += float(
                endpoint_vertex_mae(
                    prediction[:, -1, :], targets[:, -1, :], mean_flat, components
                ).item()
            ) * batch_size
            if has_cognition_estimator(model):
                total_cognition_bce += float(
                    real_cognition_loss(model, targets, times, condition_targets).item()
                ) * batch_size
            total_records += batch_size
    return {
        "records": float(total_records),
        "pca_mse": total_pca_mse / max(total_records, 1),
        "vertex_mae": total_vertex_mae / max(total_records, 1),
        "cognition_bce": total_cognition_bce / max(total_records, 1),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    training_config = dict(config["training"])
    model_config = dict(config["model"])
    brainode_config = dict(config["brainode"])
    full_config = full_brainode_config(config)

    epochs = int(args.epochs if args.epochs is not None else training_config["epochs"])
    batch_size = int(
        args.batch_size if args.batch_size is not None else training_config["batch_size"]
    )
    run_name = str(args.run_name or training_config["run_name"])
    seed = int(args.seed if args.seed is not None else training_config["random_seed"])
    device = resolve_device(args.device or training_config.get("device"))
    output_root = resolve_repo_path(training_config["output_root"])
    run_dir = output_root / run_name
    checkpoints_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    history_jsonl_path = run_dir / "history.jsonl"
    history_json_path = run_dir / "history.json"
    status_path = run_dir / "training_status.json"
    resolved_config_path = run_dir / "resolved_config.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        resolved_config_path,
        {
            "config_path": str(Path(args.config).resolve()),
            "device": str(device),
            "epochs": epochs,
            "batch_size": batch_size,
            "run_name": run_name,
            "seed": seed,
            "config": config,
        },
    )
    write_json(
        status_path,
        {
            "status": "running",
            "run_dir": str(run_dir),
            "device": str(device),
            "started_at_unix": time.time(),
        },
    )

    writer = None
    try:
        set_random_seed(seed)

        validation_path = TASK_DIR / "metadata" / "validation_dataset.json"
        if not validation_path.is_file():
            raise FileNotFoundError(
                f"Missing dataset validation report: {validation_path}"
            )
        validation_report = load_json(validation_path)
        if validation_report.get("status") != "pass":
            raise RuntimeError("Task 3 dataset validation did not pass.")

        train_archive = load_npz_archive(TASK_DIR / "dataset" / "train_subject_sequences.npz")
        val_archive = load_npz_archive(TASK_DIR / "dataset" / "val_subject_sequences.npz")
        latent_dim = int(train_archive["visit_pca_150"].shape[1])
        if latent_dim != int(brainode_config["primary_components"]):
            raise ValueError(
                f"Dataset latent dimension {latent_dim} does not match configured primary_components "
                f"{brainode_config['primary_components']}."
            )

        train_sequences = build_subject_sequences(train_archive)
        val_sequences = build_subject_sequences(val_archive)
        train_records = build_trajectory_records(
            sequences=train_sequences,
            include_backward=True,
            include_length_one=False,
            only_start_visit_zero=False,
        )
        val_forward_records = build_trajectory_records(
            sequences=val_sequences,
            include_backward=False,
            include_length_one=False,
            only_start_visit_zero=False,
        )
        val_first_last_records = build_trajectory_records(
            sequences=val_sequences,
            include_backward=False,
            include_length_one=False,
            only_start_visit_zero=True,
        )

        if args.max_train_trajectories is not None:
            train_records = train_records[: int(args.max_train_trajectories)]
        if args.max_val_trajectories is not None:
            val_forward_records = val_forward_records[: int(args.max_val_trajectories)]
            val_first_last_records = val_first_last_records[: int(args.max_val_trajectories)]

        if not train_records:
            raise RuntimeError("No training trajectories are available.")
        if not val_forward_records:
            raise RuntimeError("No validation forward trajectories are available.")
        if not val_first_last_records:
            raise RuntimeError("No validation first-to-last trajectories are available.")
        transition_record_count = sum(
            int(abs(float(record.conditions[-1]) - float(record.conditions[0])) > 1e-6)
            for record in train_records
        )
        if (
            bool(full_config.get("require_transition_trajectories", False))
            and transition_record_count == 0
        ):
            raise RuntimeError(
                "Full BrainODE config requires transition trajectories, but none were found."
            )

        train_loaders = build_length_grouped_loaders(
            records=train_records,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(training_config["num_workers"]),
        )
        val_forward_loaders = build_length_grouped_loaders(
            records=val_forward_records,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(training_config["num_workers"]),
        )
        val_first_last_loaders = build_length_grouped_loaders(
            records=val_first_last_records,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(training_config["num_workers"]),
        )

        pca_model_dir = resolve_repo_path(config["task2"]["pca_model_dir"])
        components_256 = np.load(pca_model_dir / "components_256.npy").astype(np.float32)
        mean_flat = torch.from_numpy(
            np.load(pca_model_dir / "mean.npy").astype(np.float32)
        ).to(device)
        components = torch.from_numpy(components_256[:latent_dim]).to(device)

        model = build_model(
            latent_dim=latent_dim,
            model_config=model_config,
            full_config=full_config,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(training_config["scheduler_step_size"]),
            gamma=float(training_config["scheduler_gamma"]),
        )

        start_epoch = 0
        best_metric_name = str(training_config["best_metric"])
        best_metric_value = math.inf
        history: list[dict[str, Any]] = []
        latest_checkpoint_path = checkpoints_dir / "latest.pth"
        if args.resume and latest_checkpoint_path.is_file():
            start_epoch, best_metric_value, history = load_checkpoint(
                latest_checkpoint_path, model, optimizer, scheduler, device
            )

        if not args.disable_tensorboard:
            writer = SummaryWriter(log_dir=str(tensorboard_dir))

        record_summary = {
            "train": summarize_records(train_records),
            "val_forward": summarize_records(val_forward_records),
            "val_first_last": summarize_records(val_first_last_records),
            "train_transition_records": transition_record_count,
            "full_brainode": full_config,
        }
        write_json(run_dir / "record_summary.json", record_summary)

        for epoch in range(start_epoch + 1, epochs + 1):
            epoch_start = time.time()
            train_metrics = train_one_epoch(
                model=model,
                loaders=train_loaders,
                optimizer=optimizer,
                device=device,
                integration_substeps=int(training_config["integration_substeps"]),
                noise_std=float(training_config["noise_std"]),
                scale_min=float(training_config["scale_min"]),
                scale_max=float(training_config["scale_max"]),
                gradient_clip_norm=(
                    None
                    if training_config.get("gradient_clip_norm") is None
                    else float(training_config["gradient_clip_norm"])
                ),
                full_config=full_config,
            )
            val_forward_metrics = evaluate_forward_records(
                model=model,
                loaders=val_forward_loaders,
                device=device,
                integration_substeps=int(training_config["integration_substeps"]),
                mean_flat=mean_flat,
                components=components,
                use_autoregressive_rollout=bool(
                    full_config.get("use_autoregressive_rollout", False)
                ),
            )
            val_first_last_metrics = evaluate_forward_records(
                model=model,
                loaders=val_first_last_loaders,
                device=device,
                integration_substeps=int(training_config["integration_substeps"]),
                mean_flat=mean_flat,
                components=components,
                use_autoregressive_rollout=bool(
                    full_config.get("use_autoregressive_rollout", False)
                ),
            )
            scheduler.step()

            row = {
                "epoch": epoch,
                "seconds": time.time() - epoch_start,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **train_metrics,
                "val_forward_records": val_forward_metrics["records"],
                "val_forward_endpoint_pca_mse": val_forward_metrics["pca_mse"],
                "val_forward_endpoint_vertex_mae": val_forward_metrics["vertex_mae"],
                "val_forward_cognition_bce": val_forward_metrics["cognition_bce"],
                "val_first_last_records": val_first_last_metrics["records"],
                "val_first_last_pca_mse": val_first_last_metrics["pca_mse"],
                "val_first_last_vertex_mae": val_first_last_metrics["vertex_mae"],
                "val_first_last_cognition_bce": val_first_last_metrics["cognition_bce"],
            }
            history.append(row)
            append_history_row(history_jsonl_path, row)
            write_json(history_json_path, history)

            if writer is not None:
                for key, value in row.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, epoch)

            current_metric_value = float(row[best_metric_name])
            save_checkpoint(
                path=latest_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric_value=min(best_metric_value, current_metric_value),
                history=history,
                config=config,
            )
            if epoch % int(training_config["save_every"]) == 0:
                save_checkpoint(
                    path=checkpoints_dir / f"epoch_{epoch:04d}.pth",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric_value=min(best_metric_value, current_metric_value),
                    history=history,
                    config=config,
                )
            if current_metric_value < best_metric_value:
                best_metric_value = current_metric_value
                save_checkpoint(
                    path=checkpoints_dir / "best.pth",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric_value=best_metric_value,
                    history=history,
                    config=config,
                )

        best_epoch = None
        if history:
            best_epoch = min(history, key=lambda row: float(row[best_metric_name]))["epoch"]
        write_json(
            status_path,
            {
                "status": "complete",
                "run_dir": str(run_dir),
                "device": str(device),
                "epochs_requested": epochs,
                "epochs_completed": len(history),
                "best_metric": best_metric_name,
                "best_metric_value": best_metric_value,
                "best_epoch": best_epoch,
                "train_record_summary": record_summary["train"],
                "val_forward_record_summary": record_summary["val_forward"],
                "val_first_last_record_summary": record_summary["val_first_last"],
                "last_epoch": history[-1] if history else None,
                "resolved_config_path": str(resolved_config_path),
            },
        )
        if writer is not None:
            writer.close()
        print(
            json.dumps(
                {
                    "status": "complete",
                    "run_dir": str(run_dir),
                    "epochs_completed": len(history),
                    "best_metric": best_metric_name,
                    "best_metric_value": best_metric_value,
                    "best_epoch": best_epoch,
                },
                indent=2,
            )
        )
        return 0

    except Exception as exc:
        if writer is not None:
            writer.close()
        write_json(
            status_path,
            {
                "status": "failed",
                "run_dir": str(run_dir),
                "device": str(device),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
