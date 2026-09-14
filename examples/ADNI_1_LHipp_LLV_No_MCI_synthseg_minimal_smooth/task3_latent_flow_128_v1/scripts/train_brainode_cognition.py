#!/usr/bin/env python3
"""Train PCA128 BrainODE plus a shape-only voxel cognition estimator.

There are no converter subjects in this ADNI CN/AD split.  Consequently the
CNN is trained on observed CN-versus-AD anatomy only and no pseudo-cognition
targets are fabricated.  BrainODE is trained with the fixed subject diagnosis,
as in the no-pseudo estimator ablation; estimator feedback is evaluated only
after both components have been selected on validation data.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

import brainode_cognition as B
import common as C
import train_latent_ode as T
from models import build_ode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stage", choices=("all", "cognition", "ode"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="One gradient audit for each component; no writes.")
    parser.add_argument("--smoke", action="store_true", help="One bounded epoch per requested component.")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("method") != "brainode_cognition" or config.get("representation") != "pca128":
        raise ValueError("This experiment supports only pca128 brainode_cognition")
    model = config.get("ode_model", {})
    if int(model.get("latent_dim", -1)) != C.LATENT_DIM:
        raise ValueError("BrainODE cognition latent dimension must be 128")
    cognition = config.get("cognition_estimator", {})
    if cognition.get("input") != "solid_voxelized_shape_only":
        raise ValueError("Cognition estimator must be shape-only solid voxels")
    if int(cognition.get("output_dim", -1)) != 1:
        raise ValueError("Cognition estimator must emit one continuous CN-to-AD condition")
    if bool(cognition.get("uses_age", True)) or bool(cognition.get("uses_latent", True)):
        raise ValueError("Cognition estimator may not consume age or latent metadata")
    scientific = config.get("scientific_contract", {})
    required_false = ("mci_used", "converter_supervision", "pseudo_cognition_sampling", "test_used_for_selection")
    if any(bool(scientific.get(key, True)) for key in required_false):
        raise ValueError(f"Scientific contract requires false fields: {required_false}")
    loss = config.get("loss", {})
    if loss != {"ode_latent_trajectory_mse_weight": 1.0, "cognition_bce_weight": 1.0}:
        raise ValueError("Only latent trajectory MSE and separate cognition BCE are permitted")


def ode_config(config: dict[str, Any]) -> dict[str, Any]:
    return {"method": "brainode", "model": dict(config["ode_model"])}


class VoxelDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, masks: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> None:
        self.masks = torch.from_numpy(np.asarray(masks, dtype=np.uint8))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        self.weights = torch.from_numpy(np.asarray(weights, dtype=np.float32))
        if not (len(self.masks) == len(self.labels) == len(self.weights)):
            raise ValueError("Voxel dataset length mismatch")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"mask": self.masks[index], "label": self.labels[index], "weight": self.weights[index]}


def voxel_loader(
    voxel_archive: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    labels = voxel_archive["visit_label_ad"].astype(np.int64)
    weights = B.subject_balanced_weights(voxel_archive["visit_subject_ids"], labels)
    return DataLoader(
        VoxelDataset(voxel_archive["masks"], labels, weights),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def train_cognition_epoch(
    model: B.VoxelCognitionCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    weighted_total, unweighted_total, rows, batches = 0.0, 0.0, 0, 0
    for batch in loader:
        masks = batch["mask"].to(device, dtype=torch.float32)
        labels = batch["label"].to(device)
        weights = batch["weight"].to(device)
        logits = model(masks)
        raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = torch.sum(raw * weights) / torch.sum(weights).clamp_min(1.0e-8)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        optimizer.step()
        count = int(len(labels))
        weighted_total += float(loss.detach().cpu()) * count
        unweighted_total += float(raw.mean().detach().cpu()) * count
        rows += count
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return {
        "weighted_bce": weighted_total / max(rows, 1),
        "unweighted_bce": unweighted_total / max(rows, 1),
        "rows": float(rows),
        "batches": float(batches),
    }


@torch.no_grad()
def cognition_predictions(
    model: B.VoxelCognitionCNN,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> np.ndarray:
    model.eval()
    output = []
    for batch_index, batch in enumerate(loader):
        output.append(model(batch["mask"].to(device, dtype=torch.float32)).cpu().numpy())
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    return np.concatenate(output)


def cognition_selection(metrics: dict[str, Any]) -> float:
    subject = metrics["subject"]
    return float(subject["auroc"] - 0.05 * subject["brier"])


def run_cognition_training(
    config: dict[str, Any],
    train_voxels: dict[str, np.ndarray],
    val_voxels: dict[str, np.ndarray],
    device: torch.device,
    checkpoint_dir: Path | None,
    seed: int,
    smoke: bool,
    dry_run: bool,
    resume: bool,
) -> tuple[B.VoxelCognitionCNN, dict[str, Any]]:
    settings = config["cognition_estimator"]
    training = config["training"]
    model = B.VoxelCognitionCNN(int(settings["base_channels"]), float(settings["dropout"])).to(device)
    train_loader = voxel_loader(train_voxels, int(settings["batch_size"]), True, seed)
    val_loader = voxel_loader(val_voxels, int(settings["evaluation_batch_size"]), False, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=int(settings["scheduler_patience"]), min_lr=float(settings["minimum_learning_rate"])
    )
    epochs = 1 if smoke or dry_run else int(settings["epochs"])
    maximum_batches = 1 if dry_run else int(training["smoke_cognition_batches"]) if smoke else None
    history: list[dict[str, Any]] = []
    best_score, best_epoch, stale, start_epoch = -float("inf"), 0, 0, 1
    if resume:
        if checkpoint_dir is None or not (checkpoint_dir / "cognition_latest.pt").is_file():
            raise FileNotFoundError("Cannot resume cognition training without cognition_latest.pt")
        payload = torch.load(checkpoint_dir / "cognition_latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = list(payload["history"])
        best_score, best_epoch = float(payload["best_selection_score"]), int(payload["best_epoch"])
        start_epoch = int(payload["epoch"]) + 1
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_cognition_epoch(
            model, train_loader, optimizer, device, float(settings["gradient_clip_norm"]), maximum_batches
        )
        # Validation remains the full subject-held-out split even in smoke mode;
        # a leading partial batch can contain only one diagnosis and cannot
        # support AUROC or balanced-accuracy selection.
        logits = cognition_predictions(model, val_loader, device)
        count = len(logits)
        labels = val_voxels["visit_label_ad"][:count].astype(np.int64)
        subjects = val_voxels["visit_subject_ids"][:count]
        metrics = B.scan_and_subject_metrics(labels, torch.sigmoid(torch.from_numpy(logits)).numpy(), subjects)
        score = cognition_selection(metrics)
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation": metrics,
            "selection_score": score,
        }
        history.append(row)
        if dry_run:
            gradients = [value.grad for value in model.parameters() if value.grad is not None]
            norm = math.sqrt(sum(float(torch.sum(value.detach().square()).cpu()) for value in gradients))
            if not math.isfinite(norm) or norm <= 0.0:
                raise RuntimeError("Cognition gradient audit failed")
            return model, {"dry_run": True, "gradient_l2_norm": norm, "row": row, "temperature": 1.0}
        assert checkpoint_dir is not None
        improved = score > best_score
        stale = 0 if improved else stale + 1
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_selection_score": max(best_score, score),
            "best_epoch": epoch if improved else best_epoch,
            "input_contract": model.input_contract,
            "test_data_loaded": False,
            "pseudo_cognition_sampling": False,
        }
        C.atomic_torch_save(checkpoint_dir / "cognition_latest.pt", payload)
        if improved:
            best_score, best_epoch = score, epoch
            C.atomic_torch_save(checkpoint_dir / "cognition_best.pt", payload)
        C.atomic_json(checkpoint_dir.parent / "training_status.json", {
            "status": "running",
            "stage": "cognition",
            "epoch": epoch,
            "epochs_requested": epochs,
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "test_data_loaded": False,
        })
        print(f"cognition epoch {epoch:03d}/{epochs} bce={train_metrics['weighted_bce']:.6f} val_subject_auc={metrics['subject']['auroc']:.4f}", flush=True)
        if not smoke and stale >= int(settings["early_stopping_patience"]):
            break
    assert checkpoint_dir is not None
    best = torch.load(checkpoint_dir / "cognition_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    logits = cognition_predictions(model, val_loader, device)
    temperature = B.fit_temperature(logits, val_voxels["visit_label_ad"].astype(np.int64))
    calibrated = torch.sigmoid(torch.from_numpy(logits) / temperature).numpy()
    best["temperature"] = temperature
    best["calibrated_validation"] = B.scan_and_subject_metrics(
        val_voxels["visit_label_ad"].astype(np.int64), calibrated, val_voxels["visit_subject_ids"]
    )
    C.atomic_torch_save(checkpoint_dir / "cognition_best.pt", best)
    return model, best


def run_ode_training(
    config: dict[str, Any],
    train_archive: dict[str, np.ndarray],
    val_archive: dict[str, np.ndarray],
    geometry: C.FrozenGeometry,
    device: torch.device,
    checkpoint_dir: Path | None,
    seed: int,
    smoke: bool,
    dry_run: bool,
    resume: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    training = config["training"]
    model = build_ode(ode_config(config)).to(device)
    train_rows = T.build_trajectories(train_archive, include_backward=True, first_visit_only=False)
    val_rows = T.build_trajectories(val_archive, include_backward=False, first_visit_only=True)
    train_loaders = T.loaders(train_rows, int(training["ode_batch_size"]), True, seed)
    val_loaders = T.loaders(val_rows, int(training["ode_batch_size"]), False, seed)
    raw_val = C.cached_vertices("val")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["ode_learning_rate"]), weight_decay=float(training["ode_weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(training["ode_scheduler_step_size"]), gamma=float(training["ode_scheduler_gamma"])
    )
    ode_training = {
        "integration_substeps": int(training["integration_substeps"]),
        "noise_std": float(training["noise_std"]),
        "scale_min": float(training["scale_min"]),
        "scale_max": float(training["scale_max"]),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
    }
    epochs = 1 if smoke or dry_run else int(training["ode_epochs"])
    maximum_batches = 1 if dry_run else int(training["smoke_ode_batches"]) if smoke else None
    maximum_records = int(training["smoke_val_records"]) if smoke or dry_run else None
    history: list[dict[str, Any]] = []
    best_score, best_epoch, start_epoch = float("inf"), 0, 1
    if resume and checkpoint_dir is not None and (checkpoint_dir / "ode_latest.pt").is_file():
        payload = torch.load(checkpoint_dir / "ode_latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = list(payload["history"])
        best_score, best_epoch = float(payload["best_selection_score"]), int(payload["best_epoch"])
        start_epoch = int(payload["epoch"]) + 1
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = T.train_epoch(model, train_loaders, optimizer, device, ode_training, maximum_batches)
        validation = T.evaluate(
            model, geometry, val_loaders, raw_val, device, int(training["integration_substeps"]), maximum_records
        )
        scheduler.step()
        score = float(validation["selection_score"])
        row = {"epoch": epoch, "train": train_metrics, "validation": validation, "selection_score": score}
        history.append(row)
        if dry_run:
            gradients = [value.grad for value in model.parameters() if value.grad is not None]
            norm = math.sqrt(sum(float(torch.sum(value.detach().square()).cpu()) for value in gradients))
            if not math.isfinite(norm) or norm <= 0.0:
                raise RuntimeError("ODE gradient audit failed")
            return model, {"dry_run": True, "gradient_l2_norm": norm, "row": row}
        assert checkpoint_dir is not None
        improved = score < best_score
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_selection_score": min(best_score, score),
            "best_epoch": epoch if improved else best_epoch,
            "training_condition": "fixed observed CN=0 or AD=1 subject label",
            "optimizer_objective": "standardized PCA128 latent trajectory MSE only",
            "test_data_loaded": False,
            "decoder_in_optimizer_loss": False,
        }
        C.atomic_torch_save(checkpoint_dir / "ode_latest.pt", payload)
        if improved:
            best_score, best_epoch = score, epoch
            C.atomic_torch_save(checkpoint_dir / "ode_best.pt", payload)
        C.atomic_json(checkpoint_dir.parent / "training_status.json", {
            "status": "running",
            "stage": "ode",
            "epoch": epoch,
            "epochs_requested": epochs,
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "test_data_loaded": False,
        })
        print(f"ode epoch {epoch:03d}/{epochs} latent_mse={train_metrics['latent_trajectory_mse']:.6f} val_score={score:.6f}", flush=True)
    assert checkpoint_dir is not None
    best = torch.load(checkpoint_dir / "ode_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    return model, best


def main() -> int:
    args = parse_args()
    config_path = C.resolve_path(args.config)
    config = C.read_json(config_path)
    validate_config(config)
    registry = C.load_registry()
    train_archive = C.load_archive("pca128", "train", registry)
    val_archive = C.load_archive("pca128", "val", registry)
    if set(train_archive["subject_ids"].astype(str)) & set(val_archive["subject_ids"].astype(str)):
        raise ValueError("Train/validation subject leakage")
    resolution = int(config["cognition_estimator"]["voxel_resolution"])
    train_voxels, train_grid = B.load_voxel_archive(registry, "train", resolution, train_archive)
    val_voxels, val_grid = B.load_voxel_archive(registry, "val", resolution, val_archive)
    if train_grid != val_grid:
        raise ValueError("Train/validation voxel grids differ")
    device = C.choose_device(args.device)
    seed = int(args.seed if args.seed is not None else config["training"]["seed"])
    C.set_seed(seed)
    geometry = C.build_geometry("pca128", train_archive, device, registry)
    run_name = str(args.run_name or config["training"]["run_name"])
    if args.smoke and args.run_name is None:
        run_name = f"smoke_{run_name}"
    C.validate_run_name(run_name)
    output_dir = C.output_root(registry) / "training" / "pca128" / "brainode_cognition" / run_name
    checkpoint_dir = None if args.dry_run else output_dir / "checkpoints"
    if not args.dry_run:
        if output_dir.exists() and not args.resume:
            raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume for an interrupted stage")
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_path = output_dir / "resolved_config.json"
        if resolved_path.is_file():
            previous = C.read_json(resolved_path)
            if previous.get("config") != config or int(previous.get("seed", -1)) != seed:
                raise ValueError("Existing run config/seed differs; refusing mixed-stage output")
        C.atomic_json(output_dir / "resolved_config.json", {
            "config_path": str(config_path), "config": config, "device": str(device), "seed": seed,
            "smoke": bool(args.smoke), "test_data_loaded": False,
        })
        C.atomic_json(output_dir / "run_contract.json", {
            "representation": "pca128", "latent_dim": 128, "anatomy": "left_hippocampus",
            "cognition_input": B.VoxelCognitionCNN.input_contract,
            "ode_optimizer_condition": "fixed observed CN/AD label",
            "cognition_feedback_during_optimizer": False,
            "cognition_feedback_during_inference": True,
            "pseudo_cognition_sampling": False, "converter_supervision": False, "mci_used": False,
            "claim": "CN/AD anatomical-condition feedback; not CN-to-AD conversion prediction",
            "test_data_loaded": False,
        })
    print(f"PCA128 BrainODE cognition | stage={args.stage} | device={device} | no MCI/converters/pseudo targets")
    cognition_model: B.VoxelCognitionCNN | None = None
    cognition_payload: dict[str, Any] | None = None
    ode_model: nn.Module | None = None
    ode_payload: dict[str, Any] | None = None
    if args.stage in {"all", "cognition"}:
        cognition_model, cognition_payload = run_cognition_training(
            config, train_voxels, val_voxels, device, checkpoint_dir, seed, args.smoke, args.dry_run, args.resume
        )
    elif checkpoint_dir is not None:
        path = checkpoint_dir / "cognition_best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"ODE stage requires selected cognition estimator: {path}")
        cognition_payload = torch.load(path, map_location=device, weights_only=False)
        cognition_model = B.VoxelCognitionCNN(
            int(config["cognition_estimator"]["base_channels"]), float(config["cognition_estimator"]["dropout"])
        ).to(device)
        cognition_model.load_state_dict(cognition_payload["model_state_dict"])
    if args.stage in {"all", "ode"}:
        ode_model, ode_payload = run_ode_training(
            config, train_archive, val_archive, geometry, device, checkpoint_dir, seed, args.smoke, args.dry_run, args.resume
        )
    if args.dry_run:
        print("DRY RUN PASSED — finite separate CNN BCE and BrainODE latent gradients; no files written.")
        print(json.dumps({"cognition": cognition_payload, "ode": ode_payload}, indent=2, default=str))
        return 0
    if cognition_payload is not None and ode_payload is not None:
        assert checkpoint_dir is not None
        C.atomic_torch_save(checkpoint_dir / "combined_best.pt", {
            "config": config,
            "representation": "pca128",
            "method": "brainode_cognition",
            "cognition_state_dict": cognition_payload["model_state_dict"],
            "ode_state_dict": ode_payload["model_state_dict"],
            "temperature": float(cognition_payload.get("temperature", 1.0)),
            "cognition_best_epoch": int(cognition_payload["best_epoch"]),
            "ode_best_epoch": int(ode_payload["best_epoch"]),
            "voxel_grid": train_grid.mapping(),
            "test_data_loaded": False,
            "pseudo_cognition_sampling": False,
            "converter_supervision": False,
        })
    assert output_dir is not None
    C.atomic_json(output_dir / "training_status.json", {
        "status": "smoke_complete" if args.smoke else "complete",
        "completed_stage": args.stage,
        "combined_checkpoint_available": bool((output_dir / "checkpoints" / "combined_best.pt").is_file()),
        "test_data_loaded": False,
    })
    print(f"COMPLETE: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
