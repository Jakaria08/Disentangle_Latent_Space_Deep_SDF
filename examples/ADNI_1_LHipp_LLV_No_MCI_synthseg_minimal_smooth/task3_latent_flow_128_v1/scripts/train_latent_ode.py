#!/usr/bin/env python3
"""Train plain ODE or BrainODE-style transport with latent-only supervision.

The frozen decoder is absent from the optimizer loss.  It is used under
``torch.no_grad`` only for validation checkpoint selection and reporting.
The trainer opens train and validation archives only; test is evaluator-only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import common as C
from models import build_ode, integrate_sequence_rk4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Gradient and validation checks; no files written.")
    parser.add_argument("--smoke", action="store_true", help="Bounded one-epoch run with limited train/validation records.")
    return parser.parse_args()


@dataclass(frozen=True)
class Trajectory:
    subject: str
    diagnosis: str
    direction: str
    times: np.ndarray
    targets: np.ndarray
    condition: float
    source_index: int
    target_index: int

    @property
    def length(self) -> int:
        return int(len(self.times))


class TrajectoryDataset(Dataset[Trajectory]):
    def __init__(self, rows: list[Trajectory]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Trajectory:
        return self.rows[index]


def collate(rows: list[Trajectory]) -> dict[str, Any]:
    return {
        "subject": [row.subject for row in rows],
        "diagnosis": [row.diagnosis for row in rows],
        "direction": [row.direction for row in rows],
        "times": torch.from_numpy(np.stack([row.times for row in rows])).float(),
        "targets": torch.from_numpy(np.stack([row.targets for row in rows])).float(),
        "condition": torch.tensor([row.condition for row in rows], dtype=torch.float32),
        "source_index": torch.tensor([row.source_index for row in rows], dtype=torch.long),
        "target_index": torch.tensor([row.target_index for row in rows], dtype=torch.long),
    }


def build_trajectories(
    archive: dict[str, np.ndarray],
    *,
    include_backward: bool,
    first_visit_only: bool,
) -> list[Trajectory]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    output: list[Trajectory] = []
    for subject_index in range(len(offsets) - 1):
        start, end = int(offsets[subject_index]), int(offsets[subject_index + 1])
        times = archive["visit_age_norm_train"][start:end].astype(np.float32)
        targets = archive["visit_latent_standardized_128"][start:end].astype(np.float32)
        diagnosis = str(archive["subject_diagnoses"][subject_index])
        condition = float(archive["subject_label_ad"][subject_index])
        subject = str(archive["subject_ids"][subject_index])
        visits = [0] if first_visit_only else list(range(len(times)))
        for visit in visits:
            if len(times[visit:]) > 1:
                output.append(Trajectory(
                    subject,
                    diagnosis,
                    "forward",
                    times[visit:].copy(),
                    targets[visit:].copy(),
                    condition,
                    start + visit,
                    end - 1,
                ))
            if include_backward and len(times[: visit + 1]) > 1:
                output.append(Trajectory(
                    subject,
                    diagnosis,
                    "backward",
                    times[: visit + 1][::-1].copy(),
                    targets[: visit + 1][::-1].copy(),
                    condition,
                    start + visit,
                    start,
                ))
    if not output:
        raise ValueError("No eligible trajectories")
    return output


def loaders(rows: list[Trajectory], batch_size: int, shuffle: bool, seed: int) -> list[tuple[int, DataLoader]]:
    grouped: dict[int, list[Trajectory]] = {}
    for row in rows:
        grouped.setdefault(row.length, []).append(row)
    return [
        (
            length,
            DataLoader(
                TrajectoryDataset(current),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                collate_fn=collate,
                generator=torch.Generator().manual_seed(seed + 1009 * length),
            ),
        )
        for length, current in sorted(grouped.items())
    ]


def shared_latent_augmentation(targets: torch.Tensor, training: dict[str, Any]) -> torch.Tensor:
    noise_std = float(training.get("noise_std", 0.0))
    scale_min = float(training.get("scale_min", 1.0))
    scale_max = float(training.get("scale_max", 1.0))
    if noise_std == 0.0 and scale_min == 1.0 and scale_max == 1.0:
        return targets
    noise = torch.randn(targets.shape[0], 1, targets.shape[-1], device=targets.device) * noise_std
    scale = torch.empty(targets.shape[0], 1, 1, device=targets.device).uniform_(scale_min, scale_max)
    return (targets + noise) * scale


def train_epoch(
    model: nn.Module,
    grouped_loaders: list[tuple[int, DataLoader]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: dict[str, Any],
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    total = 0.0
    endpoint_total = 0.0
    count = 0
    batch_count = 0
    for _, loader in grouped_loaders:
        for batch in loader:
            targets = batch["targets"].to(device)
            times = batch["times"].to(device)
            condition = batch["condition"].to(device)
            augmented = shared_latent_augmentation(targets, training)
            prediction = integrate_sequence_rk4(
                model,
                augmented[:, 0, :],
                times,
                condition,
                int(training["integration_substeps"]),
            )
            # This is intentionally the entire optimizer objective.
            latent_trajectory_mse = torch.mean((prediction - augmented) ** 2)
            endpoint = torch.mean((prediction[:, -1, :] - augmented[:, -1, :]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            latent_trajectory_mse.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            rows = int(targets.shape[0])
            total += float(latent_trajectory_mse.detach().cpu()) * rows
            endpoint_total += float(endpoint.detach().cpu()) * rows
            count += rows
            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                return {
                    "latent_trajectory_mse": total / max(count, 1),
                    "latent_endpoint_mse": endpoint_total / max(count, 1),
                    "trajectories": float(count),
                    "batches": float(batch_count),
                }
    return {
        "latent_trajectory_mse": total / max(count, 1),
        "latent_endpoint_mse": endpoint_total / max(count, 1),
        "trajectories": float(count),
        "batches": float(batch_count),
    }


def _empty_group() -> dict[str, list[float]]:
    return {name: [] for name in (
        "latent_mse",
        "transport_coordinate_mae",
        "transport_euclidean",
        "end_to_end_coordinate_mae",
        "end_to_end_coordinate_rmse",
        "end_to_end_euclidean",
        "volume_relative",
        "nochange_transport_coordinate_mae",
        "nochange_transport_euclidean",
        "nochange_end_to_end_coordinate_mae",
        "nochange_end_to_end_euclidean",
    )}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    geometry: C.FrozenGeometry,
    grouped_loaders: list[tuple[int, DataLoader]],
    raw_vertices_mm: np.ndarray,
    device: torch.device,
    substeps: int,
    max_records: int | None,
) -> dict[str, Any]:
    model.eval()
    geometry.eval()
    grouped = {diagnosis: _empty_group() for diagnosis in ("CN", "AD", "overall")}
    observed = 0
    for _, loader in grouped_loaders:
        for batch in loader:
            remaining = None if max_records is None else max_records - observed
            if remaining is not None and remaining <= 0:
                break
            targets = batch["targets"] if remaining is None else batch["targets"][:remaining]
            times = batch["times"] if remaining is None else batch["times"][:remaining]
            condition = batch["condition"] if remaining is None else batch["condition"][:remaining]
            source_indices = batch["source_index"] if remaining is None else batch["source_index"][:remaining]
            target_indices = batch["target_index"] if remaining is None else batch["target_index"][:remaining]
            diagnoses = batch["diagnosis"] if remaining is None else batch["diagnosis"][:remaining]
            targets = targets.to(device)
            prediction = integrate_sequence_rk4(model, targets[:, 0, :], times.to(device), condition.to(device), substeps)
            source_latent = targets[:, 0, :]
            target_latent = targets[:, -1, :]
            predicted_latent = prediction[:, -1, :]
            source_decoded = geometry.vertices(source_latent)
            target_decoded = geometry.vertices(target_latent)
            predicted_decoded = geometry.vertices(predicted_latent)
            real_target = torch.from_numpy(
                np.asarray(raw_vertices_mm[target_indices.numpy()], dtype=np.float32).copy()
            ).to(device)
            predicted_volume = geometry.volume_from_vertices(predicted_decoded)
            target_volume = geometry.volume_from_vertices(target_decoded)
            delta_transport = predicted_decoded - target_decoded
            delta_end = predicted_decoded - real_target
            delta_nochange_transport = source_decoded - target_decoded
            delta_nochange_end = source_decoded - real_target
            tensors = {
                "latent_mse": torch.mean((predicted_latent - target_latent) ** 2, dim=1),
                "transport_coordinate_mae": torch.mean(torch.abs(delta_transport), dim=(1, 2)),
                "transport_euclidean": torch.linalg.vector_norm(delta_transport, dim=2).mean(dim=1),
                "end_to_end_coordinate_mae": torch.mean(torch.abs(delta_end), dim=(1, 2)),
                "end_to_end_coordinate_rmse": torch.sqrt(torch.mean(delta_end.square(), dim=(1, 2))),
                "end_to_end_euclidean": torch.linalg.vector_norm(delta_end, dim=2).mean(dim=1),
                "volume_relative": torch.abs(predicted_volume - target_volume) / target_volume,
                "nochange_transport_coordinate_mae": torch.mean(torch.abs(delta_nochange_transport), dim=(1, 2)),
                "nochange_transport_euclidean": torch.linalg.vector_norm(delta_nochange_transport, dim=2).mean(dim=1),
                "nochange_end_to_end_coordinate_mae": torch.mean(torch.abs(delta_nochange_end), dim=(1, 2)),
                "nochange_end_to_end_euclidean": torch.linalg.vector_norm(delta_nochange_end, dim=2).mean(dim=1),
            }
            for index, diagnosis in enumerate(diagnoses):
                for bucket in (diagnosis, "overall"):
                    for name, tensor in tensors.items():
                        grouped[bucket][name].append(float(tensor[index].cpu()))
            observed += len(diagnoses)
        if max_records is not None and observed >= max_records:
            break
    summary: dict[str, Any] = {"records": observed, "groups": {}}
    for diagnosis, values in grouped.items():
        summary["groups"][diagnosis] = {
            f"{name}_mean": float(np.mean(items)) if items else float("nan")
            for name, items in values.items()
        } | {"records": len(values["latent_mse"])}
    macro_ratios = []
    for diagnosis in ("CN", "AD"):
        values = summary["groups"][diagnosis]
        coordinate_ratio = values["transport_coordinate_mae_mean"] / max(values["nochange_transport_coordinate_mae_mean"], 1.0e-12)
        euclidean_ratio = values["transport_euclidean_mean"] / max(values["nochange_transport_euclidean_mean"], 1.0e-12)
        macro_ratios.append(0.5 * (coordinate_ratio + euclidean_ratio))
    summary["selection_score"] = float(np.mean(macro_ratios))
    C.assert_finite_mapping(summary)
    return summary


def validate_config(config: dict[str, Any]) -> None:
    if config.get("method") not in {"plain_ode", "brainode"}:
        raise ValueError("ODE config method must be plain_ode or brainode")
    if config.get("representation") not in {"pca128", "spiralnet128", "adaptive128"}:
        raise ValueError("Unknown representation")
    if int(config["model"]["latent_dim"]) != C.LATENT_DIM:
        raise ValueError("ODE latent dimension must be 128")
    loss = config.get("loss", {})
    if set(loss) != {"latent_trajectory_mse_weight"} or float(loss["latent_trajectory_mse_weight"]) != 1.0:
        raise ValueError("ODE loss contract permits only unit-weight latent trajectory MSE")
    forbidden = {key for key in loss if any(word in key for word in ("vertex", "reconstruction", "volume", "cocycle"))}
    if forbidden:
        raise ValueError(f"Forbidden decoder/anatomy loss in ODE config: {sorted(forbidden)}")


def main() -> int:
    args = parse_args()
    config_path = C.resolve_path(args.config)
    config = C.read_json(config_path)
    validate_config(config)
    registry = C.load_registry()
    representation = str(config["representation"])
    # Test is deliberately not loaded or even validated here.
    train_archive = C.load_archive(representation, "train", registry)
    val_archive = C.load_archive(representation, "val", registry)
    if set(train_archive["subject_ids"].astype(str)) & set(val_archive["subject_ids"].astype(str)):
        raise ValueError("Train/validation subject leakage")
    device = C.choose_device(args.device)
    training = dict(config["training"])
    seed = int(args.seed if args.seed is not None else training["seed"])
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    if args.smoke:
        epochs = 1
    C.set_seed(seed)
    geometry = C.build_geometry(representation, train_archive, device, registry)
    model = build_ode(config).to(device)
    train_rows = build_trajectories(train_archive, include_backward=True, first_visit_only=False)
    val_forward_rows = build_trajectories(val_archive, include_backward=False, first_visit_only=False)
    val_first_last_rows = build_trajectories(val_archive, include_backward=False, first_visit_only=True)
    batch_size = int(training["batch_size"])
    train_loaders = loaders(train_rows, batch_size, True, seed)
    val_forward_loaders = loaders(val_forward_rows, batch_size, False, seed)
    val_first_last_loaders = loaders(val_first_last_rows, batch_size, False, seed)
    raw_val = C.cached_vertices("val", registry)
    max_train_batches = int(training.get("smoke_train_batches", 2)) if args.smoke else None
    max_val_records = int(training.get("smoke_val_records", 8)) if args.smoke else None

    print("=" * 96)
    print(f"{config['method']} | {representation} | device={device} | latent-only optimizer objective")
    print(f"parameters={C.parameter_count(model)} attention={getattr(model, 'attention_contract', 'none')}")
    print(f"train trajectories={len(train_rows)} val forward={len(val_forward_rows)} val first-last={len(val_first_last_rows)}")
    print("decoder in optimizer loss: no; test archive loaded: no")

    if args.dry_run:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
        metrics = train_epoch(model, train_loaders, optimizer, device, training, max_batches=1)
        validation = evaluate(
            model,
            geometry,
            val_first_last_loaders,
            raw_val,
            device,
            int(training["integration_substeps"]),
            int(training.get("smoke_val_records", 8)),
        )
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
        gradient_norm = math.sqrt(sum(float(torch.sum(gradient.detach().square()).cpu()) for gradient in gradients))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError("ODE gradient audit failed")
        print("DRY RUN PASSED — finite latent-only gradients and frozen-decoder validation; no files written.")
        print(json.dumps({"train": metrics, "gradient_l2_norm": gradient_norm, "validation": validation}, indent=2, sort_keys=True))
        return 0

    run_name = str(args.run_name or training["run_name"])
    if args.smoke and args.run_name is None:
        run_name = f"smoke_{run_name}"
    C.validate_run_name(run_name)
    output_dir = C.output_root(registry) / "training" / representation / str(config["method"]) / run_name
    checkpoint_dir = output_dir / "checkpoints"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if args.resume and not (checkpoint_dir / "latest.pt").is_file():
        raise FileNotFoundError(f"Cannot resume without {checkpoint_dir / 'latest.pt'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    C.atomic_json(output_dir / "resolved_config.json", {
        "config_path": str(config_path),
        "config": config,
        "representation_manifest": C.read_json(C.output_root(registry) / "representations" / representation / "manifest.json"),
        "device": str(device),
        "seed": seed,
        "epochs": epochs,
        "smoke": bool(args.smoke),
        "optimizer_objective": "standardized latent trajectory MSE only",
        "decoder_in_optimizer_loss": False,
        "test_data_loaded": False,
    })
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training["scheduler_step_size"]),
        gamma=float(training["scheduler_gamma"]),
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    if args.resume:
        payload = torch.load(checkpoint_dir / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = list(payload.get("history", []))
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload["best_selection_score"])
        best_epoch = int(payload["best_epoch"])
    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_epoch(model, train_loaders, optimizer, device, training, max_train_batches)
        val_forward = evaluate(model, geometry, val_forward_loaders, raw_val, device, int(training["integration_substeps"]), max_val_records)
        val_first_last = evaluate(model, geometry, val_first_last_loaders, raw_val, device, int(training["integration_substeps"]), max_val_records)
        scheduler.step()
        score = float(val_first_last["selection_score"])
        row = {
            "epoch": epoch,
            "elapsed_minutes": (time.time() - started) / 60.0,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "val_forward": val_forward,
            "val_first_last": val_first_last,
            "val_selection_score": score,
        }
        history.append(row)
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "representation": representation,
            "method": config["method"],
            "history": history,
            "best_selection_score": min(best_score, score),
            "best_epoch": epoch if score < best_score else best_epoch,
            "decoder_in_optimizer_loss": False,
            "test_data_loaded": False,
        }
        C.atomic_torch_save(checkpoint_dir / "latest.pt", payload)
        if score < best_score:
            best_score, best_epoch = score, epoch
            payload["best_selection_score"] = best_score
            payload["best_epoch"] = best_epoch
            C.atomic_torch_save(checkpoint_dir / "best.pt", payload)
        C.atomic_json(output_dir / "training_status.json", {
            "status": "running",
            "epoch": epoch,
            "epochs_requested": epochs,
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "decoder_in_optimizer_loss": False,
            "test_data_loaded": False,
        })
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"epoch {epoch:03d}/{epochs} latent_mse={train_metrics['latent_trajectory_mse']:.6f} val_score={score:.6f} best={best_score:.6f}@{best_epoch}", flush=True)
    C.atomic_json(output_dir / "training_status.json", {
        "status": "complete",
        "epoch": epochs,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "selected_checkpoint": str(checkpoint_dir / "best.pt"),
        "decoder_in_optimizer_loss": False,
        "test_data_loaded": False,
    })
    print(f"COMPLETE: {checkpoint_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
