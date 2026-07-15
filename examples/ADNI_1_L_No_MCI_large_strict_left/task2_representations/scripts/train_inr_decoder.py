#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from task2_common import (
    TASK_DIR,
    build_decoder,
    choose_device,
    fit_single_latent,
    load_config,
    load_manifest,
    load_obj_arrays,
    load_sdf_arrays,
    resolve_repo_path,
    rows_for_split,
    set_global_seed,
    stable_seed,
    strip_module_prefix,
    write_json,
)


class ManifestSDFDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        samples_per_scene: int,
        surface_samples_per_scene: int = 0,
        positive_sdf_is_outside: bool = True,
        load_into_ram: bool = False,
    ):
        self.rows = rows
        self.samples_per_scene = int(samples_per_scene)
        self.surface_samples_per_scene = int(surface_samples_per_scene)
        self.off_surface_samples_per_scene = (
            self.samples_per_scene - self.surface_samples_per_scene
        )
        if self.surface_samples_per_scene < 0:
            raise ValueError("surface_samples_per_scene cannot be negative.")
        if self.off_surface_samples_per_scene < 1:
            raise ValueError("At least one off-surface sample per scene is required.")
        self.normal_sign = 1.0 if positive_sdf_is_outside else -1.0
        self.loaded_sdf = None
        self.loaded_surfaces = None
        self.loaded_bytes = 0
        if load_into_ram:
            started = time.time()
            self.loaded_sdf = []
            for index, row in enumerate(rows, start=1):
                pos, neg = load_sdf_arrays(row["sdf_npz_path"])
                self.loaded_sdf.append((pos, neg))
                self.loaded_bytes += pos.nbytes + neg.nbytes
                if index % 100 == 0 or index == len(rows):
                    print(
                        f"Preloaded SDF {index}/{len(rows)} "
                        f"({self.loaded_bytes / 1024**3:.2f} GiB)."
                    )
            print(
                f"Preloaded {len(rows)} SDF archives in {time.time() - started:.1f}s; "
                f"shared host-RAM payload={self.loaded_bytes / 1024**3:.2f} GiB."
            )
        if self.surface_samples_per_scene > 0:
            started = time.time()
            self.loaded_surfaces = [
                self._prepare_surface(row["mesh_path"]) for row in rows
            ]
            print(
                f"Preloaded surface triangles for {len(rows)} meshes in "
                f"{time.time() - started:.1f}s."
            )

    def _prepare_surface(self, mesh_path: str) -> dict[str, np.ndarray]:
        vertices, faces = load_obj_arrays(mesh_path)
        triangles = vertices[faces]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        double_area = np.linalg.norm(cross, axis=1)
        valid = double_area > 1e-12
        if not np.any(valid):
            raise ValueError(f"Mesh has no non-degenerate faces: {mesh_path}")
        triangles = triangles[valid]
        cross = cross[valid]
        double_area = double_area[valid]

        signed_volume = np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        ).sum() / 6.0
        outward_sign = 1.0 if signed_volume >= 0.0 else -1.0
        normals = cross / double_area[:, None]
        normals *= outward_sign * self.normal_sign
        probabilities = double_area / double_area.sum()
        return {
            "triangles": triangles.astype(np.float32),
            "normals": normals.astype(np.float32),
            "face_cdf": np.cumsum(probabilities, dtype=np.float64),
        }

    @staticmethod
    def _sample_surface(
        surface: dict[str, np.ndarray], count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        face_values = np.random.random(size=count)
        face_indices = np.searchsorted(
            surface["face_cdf"], face_values, side="right"
        )
        face_indices = np.minimum(
            face_indices, len(surface["triangles"]) - 1
        )
        triangles = surface["triangles"][face_indices]
        sqrt_u = np.sqrt(np.random.random(size=count)).astype(np.float32)
        v = np.random.random(size=count).astype(np.float32)
        weights = np.stack(
            (
                1.0 - sqrt_u,
                sqrt_u * (1.0 - v),
                sqrt_u * v,
            ),
            axis=1,
        )
        points = np.sum(triangles * weights[:, :, None], axis=1)
        normals = surface["normals"][face_indices]
        return points.astype(np.float32), normals.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        if self.loaded_sdf is None:
            pos, neg = load_sdf_arrays(self.rows[index]["sdf_npz_path"])
        else:
            pos, neg = self.loaded_sdf[index]
        half = self.off_surface_samples_per_scene // 2
        pos_index = np.random.randint(0, len(pos), size=half)
        neg_index = np.random.randint(
            0,
            len(neg),
            size=self.off_surface_samples_per_scene - half,
        )
        samples = np.concatenate((pos[pos_index], neg[neg_index]), axis=0)
        normals = np.zeros((len(samples), 3), dtype=np.float32)
        surface_mask = np.zeros(len(samples), dtype=np.bool_)

        if self.surface_samples_per_scene > 0:
            if self.loaded_surfaces is None:
                surface = self._prepare_surface(self.rows[index]["mesh_path"])
            else:
                surface = self.loaded_surfaces[index]
            surface_points, surface_normals = self._sample_surface(
                surface, self.surface_samples_per_scene
            )
            surface_samples = np.concatenate(
                (
                    surface_points,
                    np.zeros((self.surface_samples_per_scene, 1), dtype=np.float32),
                ),
                axis=1,
            )
            samples = np.concatenate((samples, surface_samples), axis=0)
            normals = np.concatenate((normals, surface_normals), axis=0)
            surface_mask = np.concatenate(
                (
                    surface_mask,
                    np.ones(self.surface_samples_per_scene, dtype=np.bool_),
                ),
                axis=0,
            )

        order = np.random.permutation(len(samples))
        return (
            torch.from_numpy(samples[order]),
            torch.from_numpy(normals[order]),
            torch.from_numpy(surface_mask[order]),
            index,
        )


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a train-only INR autodecoder with eikonal regularization."
    )
    parser.add_argument("--config", required=True, help="INR JSON configuration.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint name/path to resume, for example 'latest'.",
    )
    parser.add_argument("--device", default=None, help="Device such as cuda:0 or cpu.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the configured final epoch.",
    )
    parser.add_argument(
        "--scene-chunk-size",
        type=int,
        default=None,
        help=(
            "Override scenes processed simultaneously on the GPU. This changes "
            "peak memory, not the effective batch size."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one small optimization batch without writing checkpoints.",
    )
    return parser.parse_args()


def set_learning_rates(
    optimizer: torch.optim.Optimizer, config: dict[str, Any], epoch: int
) -> tuple[float, float]:
    step = int(config["learning_rate_step"])
    factor = float(config["learning_rate_factor"])
    multiplier = factor ** (epoch // step)
    network_lr = float(config["network_learning_rate"]) * multiplier
    latent_lr = float(config["latent_learning_rate"]) * multiplier
    optimizer.param_groups[0]["lr"] = network_lr
    optimizer.param_groups[1]["lr"] = latent_lr
    return network_lr, latent_lr


def eikonal_weight_for_epoch(config: dict[str, Any], epoch: int) -> float:
    target = float(config["eikonal_weight"])
    start_epoch = int(config.get("eikonal_start_epoch", 0))
    ramp_epochs = int(config.get("eikonal_ramp_epochs", 0))
    if epoch <= start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return target
    progress = min(1.0, max(0.0, (epoch - start_epoch) / ramp_epochs))
    return target * progress


def geometry_weight_for_epoch(config: dict[str, Any], epoch: int) -> float:
    start_epoch = int(config.get("geometry_start_epoch", 0))
    ramp_epochs = int(config.get("geometry_ramp_epochs", 0))
    if epoch <= start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, (epoch - start_epoch) / ramp_epochs))


def save_checkpoint(
    path: Path,
    epoch: int,
    decoder,
    latent_codes,
    optimizer,
    config: dict[str, Any],
    train_rows: list[dict[str, str]],
    best_validation_l1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": decoder.state_dict(),
            "latent_codes": latent_codes.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": serializable_config,
            "train_scan_ids": [row["scan_id"] for row in train_rows],
            "best_validation_l1": best_validation_l1,
        },
        path,
    )


def resolve_initial_decoder_checkpoint(config: dict[str, Any]) -> Path | None:
    value = config.get("initial_decoder_checkpoint")
    if not value:
        return None
    return resolve_repo_path(value)


def warm_start_decoder_from_checkpoint(
    decoder,
    checkpoint_path: Path,
    transfer_mode: str,
    device,
) -> dict[str, Any]:
    if transfer_mode not in {"strict", "partial"}:
        raise ValueError(
            "initial_decoder_transfer must be either 'strict' or 'partial', "
            f"got {transfer_mode!r}."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Initial decoder checkpoint does not exist: {checkpoint_path}"
        )

    payload = torch.load(checkpoint_path, map_location=device)
    source_state = strip_module_prefix(payload.get("model_state_dict", payload))
    if transfer_mode == "strict":
        decoder.load_state_dict(source_state)
        loaded_keys = sorted(source_state.keys())
        return {
            "checkpoint": str(checkpoint_path),
            "transfer_mode": transfer_mode,
            "source_epoch": payload.get("epoch"),
            "source_best_validation_l1": payload.get("best_validation_l1"),
            "exact_loaded": len(loaded_keys),
            "partial_loaded": 0,
            "skipped": 0,
            "details": [
                {
                    "key": key,
                    "status": "exact",
                    "shape": list(source_state[key].shape),
                }
                for key in loaded_keys
            ],
        }

    target_state = decoder.state_dict()
    merged_state = {}
    details = []
    exact_loaded = 0
    partial_loaded = 0
    skipped = 0
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            merged_state[key] = target_tensor
            skipped += 1
            details.append(
                {
                    "key": key,
                    "status": "missing_in_source",
                    "target_shape": list(target_tensor.shape),
                }
            )
            continue

        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            merged_state[key] = source_tensor.to(
                device=target_tensor.device, dtype=target_tensor.dtype
            )
            exact_loaded += 1
            details.append(
                {
                    "key": key,
                    "status": "exact",
                    "shape": list(target_tensor.shape),
                }
            )
            continue

        compatible_matrix = (
            source_tensor.ndim == 2
            and target_tensor.ndim == 2
            and source_tensor.shape[0] <= target_tensor.shape[0]
            and source_tensor.shape[1] <= target_tensor.shape[1]
        )
        compatible_vector = (
            source_tensor.ndim == 1
            and target_tensor.ndim == 1
            and source_tensor.shape[0] <= target_tensor.shape[0]
        )
        if compatible_matrix:
            copied = target_tensor.clone()
            source_on_target = source_tensor.to(
                device=target_tensor.device, dtype=target_tensor.dtype
            )
            copied[: source_tensor.shape[0], : source_tensor.shape[1]] = source_on_target
            if source_tensor.shape[1] < target_tensor.shape[1]:
                copied[:, source_tensor.shape[1] :] = 0.0
            merged_state[key] = copied
            partial_loaded += 1
            details.append(
                {
                    "key": key,
                    "status": "partial_top_left_zero_extra_columns",
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
            continue

        if compatible_vector:
            copied = target_tensor.clone()
            source_on_target = source_tensor.to(
                device=target_tensor.device, dtype=target_tensor.dtype
            )
            copied[: source_tensor.shape[0]] = source_on_target
            merged_state[key] = copied
            partial_loaded += 1
            details.append(
                {
                    "key": key,
                    "status": "partial_prefix",
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
            continue

        merged_state[key] = target_tensor
        skipped += 1
        details.append(
            {
                "key": key,
                "status": "shape_mismatch_skipped",
                "source_shape": list(source_tensor.shape),
                "target_shape": list(target_tensor.shape),
            }
        )

    decoder.load_state_dict(merged_state, strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "transfer_mode": transfer_mode,
        "source_epoch": payload.get("epoch"),
        "source_best_validation_l1": payload.get("best_validation_l1"),
        "exact_loaded": exact_loaded,
        "partial_loaded": partial_loaded,
        "skipped": skipped,
        "details": details,
    }


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validation_score(
    decoder,
    validation_rows: list[dict[str, str]],
    config: dict[str, Any],
    device,
    epoch: int,
) -> tuple[float, list[dict[str, Any]]]:
    seed = int(config["seed"])
    num_scans = min(int(config["validation_num_scans"]), len(validation_rows))
    rng = np.random.default_rng(seed)
    selected_indices = sorted(
        rng.choice(len(validation_rows), size=num_scans, replace=False).tolist()
    )
    selected = [validation_rows[index] for index in selected_indices]
    fit_config = dict(config["latent_fit"])
    fit_config["evaluation_samples"] = min(
        int(fit_config.get("evaluation_samples", 32768)), 8192
    )
    steps = int(config["validation_latent_steps"])
    details = []
    for row in selected:
        _latent, stats = fit_single_latent(
            decoder=decoder,
            sdf_path=row["sdf_npz_path"],
            latent_size=int(config["latent_size"]),
            fit_config=fit_config,
            clamp_distance=float(config["clamp_distance"]),
            device=device,
            seed=stable_seed(row["scan_id"], seed),
            steps_override=steps,
        )
        details.append(
            {
                "epoch": epoch,
                "scan_id": row["scan_id"],
                "heldout_sdf_l1": stats["heldout_sdf_l1"],
                "steps_completed": stats["steps_completed"],
            }
        )
    return float(np.mean([row["heldout_sdf_l1"] for row in details])), details


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    set_global_seed(seed)
    device = choose_device(args.device)
    if device.type != "cuda" and not args.smoke_test:
        print(
            "WARNING: CUDA is not active. Full 8x512 eikonal training on CPU will be very slow.",
            file=sys.stderr,
        )

    rows = load_manifest(config["manifest"])
    train_rows = rows_for_split(rows, "train")
    validation_rows = rows_for_split(rows, "val")
    if not train_rows or not validation_rows:
        raise RuntimeError("Training and validation rows are required.")

    run_config = dict(config)
    if args.scene_chunk_size is not None:
        if args.scene_chunk_size < 1:
            raise ValueError("--scene-chunk-size must be positive.")
        run_config["scene_chunk_size"] = int(args.scene_chunk_size)
    if args.smoke_test:
        smoke_surface_count = (
            86 if int(run_config.get("surface_samples_per_scene", 0)) > 0 else 0
        )
        run_config.update(
            {
                "scenes_per_batch": 2,
                "scene_chunk_size": 1,
                "samples_per_scene": 128,
                "surface_samples_per_scene": smoke_surface_count,
                "geometry_start_epoch": 0,
                "geometry_ramp_epochs": 0,
                "eikonal_start_epoch": 0,
                "eikonal_ramp_epochs": 0,
                "data_loader_workers": 0,
                "eikonal_points_per_scene": 128,
            }
        )
        train_rows = train_rows[:2]

    output_dir = resolve_repo_path(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    if not args.smoke_test:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "run_config.json",
            {
                key: value
                for key, value in run_config.items()
                if not key.startswith("_")
            },
        )
        write_json(
            output_dir / "training_scan_ids.json",
            [row["scan_id"] for row in train_rows],
        )

    decoder = build_decoder(run_config, device)
    initial_decoder_checkpoint = resolve_initial_decoder_checkpoint(run_config)
    if initial_decoder_checkpoint is not None and not args.resume:
        warm_start_report = warm_start_decoder_from_checkpoint(
            decoder=decoder,
            checkpoint_path=initial_decoder_checkpoint,
            transfer_mode=str(run_config.get("initial_decoder_transfer", "strict")),
            device=device,
        )
        print(
            "Warm-started decoder from "
            f"{initial_decoder_checkpoint} "
            f"with mode={warm_start_report['transfer_mode']}, "
            f"exact={warm_start_report['exact_loaded']}, "
            f"partial={warm_start_report['partial_loaded']}, "
            f"skipped={warm_start_report['skipped']}."
        )
        if not args.smoke_test:
            write_json(output_dir / "warm_start_report.json", warm_start_report)
    elif initial_decoder_checkpoint is not None and args.resume:
        print(
            "Resume checkpoint requested; ignoring initial_decoder_checkpoint "
            f"{initial_decoder_checkpoint}."
        )

    latent_size = int(run_config["latent_size"])
    code_bound = float(run_config["code_bound"])
    latent_codes = torch.nn.Embedding(
        len(train_rows), latent_size, max_norm=code_bound
    ).to(device)
    torch.nn.init.normal_(latent_codes.weight, mean=0.0, std=1.0 / np.sqrt(latent_size))

    optimizer = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": float(run_config["network_learning_rate"]),
            },
            {
                "params": latent_codes.parameters(),
                "lr": float(run_config["latent_learning_rate"]),
            },
        ]
    )
    start_epoch = 1
    best_validation_l1 = float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            name = args.resume if args.resume.endswith(".pth") else f"{args.resume}.pth"
            resume_path = checkpoint_dir / name
        payload = torch.load(resume_path, map_location=device)
        decoder.load_state_dict(strip_module_prefix(payload["model_state_dict"]))
        latent_codes.load_state_dict(payload["latent_codes"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_validation_l1 = float(
            payload.get("best_validation_l1", float("inf"))
        )
        print(f"Resuming from {resume_path} at epoch {start_epoch}.")

    dataset = ManifestSDFDataset(
        train_rows,
        samples_per_scene=int(run_config["samples_per_scene"]),
        surface_samples_per_scene=int(
            run_config.get("surface_samples_per_scene", 0)
        ),
        positive_sdf_is_outside=bool(
            run_config.get("positive_sdf_is_outside", True)
        ),
        load_into_ram=bool(run_config.get("load_dataset_into_ram", False)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_workers = int(run_config["data_loader_workers"])
    loader_options = {
        "dataset": dataset,
        "batch_size": int(run_config["scenes_per_batch"]),
        "shuffle": True,
        "num_workers": loader_workers,
        "drop_last": bool(run_config.get("drop_last", True)),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if loader_workers > 0:
        loader_options["persistent_workers"] = bool(
            run_config.get("persistent_workers", True)
        )
        loader_options["prefetch_factor"] = int(
            run_config.get("prefetch_factor", 2)
        )
    loader = DataLoader(**loader_options)

    final_epoch = int(args.epochs or run_config["epochs"])
    if args.smoke_test:
        final_epoch = 1
        start_epoch = 1
    scene_chunk_size = int(run_config["scene_chunk_size"])
    clamp_distance = float(run_config["clamp_distance"])
    reg_lambda = float(run_config["code_regularization_lambda"])
    reg_type = str(run_config.get("code_regularization_type", "l2_norm"))
    if reg_type not in {"l2_norm", "mean_squared"}:
        raise ValueError(
            f"Unsupported training code regularization type: {reg_type}. "
            "Expected 'l2_norm' or 'mean_squared'."
        )
    reg_warmup_epochs = int(run_config.get("code_regularization_warmup_epochs", 100))
    loss_mode = str(run_config.get("loss_mode", "balanced_sdf"))
    if loss_mode not in {"balanced_sdf", "surface_normal_sdf"}:
        raise ValueError(f"Unsupported loss_mode: {loss_mode!r}")
    surface_samples_per_scene = int(
        run_config.get("surface_samples_per_scene", 0)
    )
    if loss_mode == "surface_normal_sdf" and surface_samples_per_scene < 1:
        raise ValueError(
            "surface_normal_sdf requires surface_samples_per_scene > 0."
        )
    prediction_clamp_for_loss = bool(
        run_config.get("prediction_clamp_for_loss", True)
    )
    off_surface_sdf_weight = float(
        run_config.get("off_surface_sdf_weight", 1.0)
    )
    surface_sdf_weight = float(run_config.get("surface_sdf_weight", 0.0))
    surface_normal_weight = float(
        run_config.get("surface_normal_weight", 0.0)
    )
    geometry_start_epoch = int(run_config.get("geometry_start_epoch", 0))
    geometry_ramp_epochs = int(run_config.get("geometry_ramp_epochs", 0))
    eikonal_target_weight = float(run_config["eikonal_weight"])
    eikonal_start_epoch = int(run_config.get("eikonal_start_epoch", 0))
    eikonal_ramp_epochs = int(run_config.get("eikonal_ramp_epochs", 0))
    eikonal_on_raw_output = bool(
        run_config.get("eikonal_on_raw_output", False)
    )
    eikonal_loss_type = str(run_config.get("eikonal_loss_type", "squared"))
    if eikonal_loss_type not in {"l1", "squared"}:
        raise ValueError(
            "eikonal_loss_type must be either 'l1' or 'squared'."
        )
    normalized_sdf_mse_weight = float(
        run_config.get("normalized_sdf_mse_weight", 0.0)
    )
    sign_bce_weight = float(run_config.get("sign_bce_weight", 0.0))
    sign_logit_temperature = float(
        run_config.get("sign_logit_temperature", clamp_distance)
    )
    if sign_bce_weight > 0.0 and sign_logit_temperature <= 0.0:
        raise ValueError("sign_logit_temperature must be positive.")
    use_auxiliary_sdf_losses = bool(
        normalized_sdf_mse_weight > 0.0 or sign_bce_weight > 0.0
    )
    log_field_diagnostics = bool(
        run_config.get("log_field_diagnostics", False)
    )
    eikonal_count = int(run_config["eikonal_points_per_scene"])
    gradient_clip = float(run_config["gradient_clip_norm"])
    checkpoint_every = int(run_config["checkpoint_every"])
    additional_checkpoints = {
        int(value) for value in run_config.get("additional_checkpoints", [])
    }
    validation_every = int(run_config["validation_every"])

    def checkpoint_due(epoch_value: int) -> bool:
        return bool(
            epoch_value % checkpoint_every == 0
            or epoch_value in additional_checkpoints
            or epoch_value == final_epoch
        )

    print(
        f"Training {config['name']} on {len(train_rows)} scans with device={device}, "
        f"effective_batch={run_config['scenes_per_batch']}, "
        f"scene_chunk={scene_chunk_size}, samples_per_scene={run_config['samples_per_scene']}, "
        f"eikonal_points_per_scene={eikonal_count}, "
        f"eikonal_target_weight={eikonal_target_weight}, "
        f"eikonal_start_epoch={eikonal_start_epoch}, "
        f"eikonal_ramp_epochs={eikonal_ramp_epochs}, "
        f"eikonal_on_raw_output={eikonal_on_raw_output}, "
        f"eikonal_loss_type={eikonal_loss_type}, "
        f"loss_mode={loss_mode}, "
        f"surface_samples_per_scene={surface_samples_per_scene}, "
        f"geometry_start_epoch={geometry_start_epoch}, "
        f"geometry_ramp_epochs={geometry_ramp_epochs}, "
        f"prediction_clamp_for_loss={prediction_clamp_for_loss}, "
        f"normalized_sdf_mse_weight={normalized_sdf_mse_weight}, "
        f"sign_bce_weight={sign_bce_weight}, "
        f"dataset_in_ram={run_config.get('load_dataset_into_ram', False)}."
    )
    interrupted = False
    last_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, final_epoch + 1):
            last_epoch = epoch
            epoch_start = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            decoder.train()
            network_lr, latent_lr = set_learning_rates(optimizer, run_config, epoch)
            current_eikonal_weight = eikonal_weight_for_epoch(run_config, epoch)
            current_geometry_weight = geometry_weight_for_epoch(run_config, epoch)
            current_surface_sdf_weight = (
                surface_sdf_weight * current_geometry_weight
            )
            current_surface_normal_weight = (
                surface_normal_weight * current_geometry_weight
            )
            compute_eikonal = current_eikonal_weight > 0.0
            totals = {
                "loss": 0.0,
                "sdf": 0.0,
                "off_surface_sdf": 0.0,
                "surface_sdf": 0.0,
                "surface_normal": 0.0,
                "reconstruction": 0.0,
                "code": 0.0,
                "eikonal": 0.0,
            }
            diagnostic_totals = {
                "sign_accuracy": 0.0,
                "raw_output_abs_mean": 0.0,
                "raw_output_std": 0.0,
            }
            auxiliary_totals = {
                "normalized_sdf_mse": 0.0,
                "sign_bce": 0.0,
            }
            batches = 0

            for samples, sample_normals, surface_masks, scene_indices in loader:
                samples = samples.to(device, non_blocking=True)
                sample_normals = sample_normals.to(device, non_blocking=True)
                surface_masks = surface_masks.to(device, non_blocking=True)
                scene_indices = scene_indices.to(device, non_blocking=True)
                batch_scenes = len(scene_indices)
                optimizer.zero_grad(set_to_none=True)

                batch_values = {key: 0.0 for key in totals}
                diagnostic_batch_values = {
                    key: 0.0 for key in diagnostic_totals
                }
                auxiliary_batch_values = {
                    key: 0.0 for key in auxiliary_totals
                }
                for start in range(0, batch_scenes, scene_chunk_size):
                    stop = min(start + scene_chunk_size, batch_scenes)
                    chunk = samples[start:stop]
                    chunk_normals = sample_normals[start:stop]
                    chunk_surface_masks = surface_masks[start:stop]
                    indices = scene_indices[start:stop]
                    chunk_scenes, points_per_scene, _ = chunk.shape
                    codes = latent_codes(indices)

                    use_all_eikonal_points = bool(
                        compute_eikonal and eikonal_count >= points_per_scene
                    )
                    xyz = chunk[:, :, :3].reshape(-1, 3)
                    needs_surface_gradients = (
                        loss_mode == "surface_normal_sdf"
                        and current_surface_normal_weight > 0.0
                    )
                    needs_all_gradients = (
                        needs_surface_gradients or use_all_eikonal_points
                    )
                    if needs_all_gradients:
                        xyz = xyz.detach().requires_grad_(True)
                    target = chunk[:, :, 3:4].reshape(-1, 1)
                    target = target.clamp(-clamp_distance, clamp_distance)
                    target_normals = chunk_normals.reshape(-1, 3)
                    surface_mask = chunk_surface_masks.reshape(-1).bool()
                    off_surface_mask = ~surface_mask
                    expanded_codes = (
                        codes[:, None, :]
                        .expand(chunk_scenes, points_per_scene, latent_size)
                        .reshape(-1, latent_size)
                    )
                    prediction_raw = decoder(torch.cat((expanded_codes, xyz), dim=1))
                    prediction = prediction_raw.clamp(-clamp_distance, clamp_distance)
                    prediction_for_loss = (
                        prediction
                        if prediction_clamp_for_loss
                        else prediction_raw
                    )
                    sdf_loss = torch.mean(torch.abs(prediction_for_loss - target))
                    zero = prediction_raw.new_zeros(())

                    if loss_mode == "surface_normal_sdf":
                        off_surface_sdf_loss = torch.mean(
                            torch.abs(
                                prediction_for_loss[off_surface_mask]
                                - target[off_surface_mask]
                            )
                        )
                        surface_sdf_loss = torch.mean(
                            torch.abs(
                                prediction_for_loss[surface_mask]
                                - target[surface_mask]
                            )
                        )
                        reconstruction_loss = (
                            off_surface_sdf_weight * off_surface_sdf_loss
                            + current_surface_sdf_weight * surface_sdf_loss
                        )
                    else:
                        off_surface_sdf_loss = sdf_loss
                        surface_sdf_loss = zero
                        reconstruction_loss = sdf_loss

                    if use_auxiliary_sdf_losses:
                        normalized_sdf_mse = torch.mean(
                            (
                                (prediction_for_loss - target)
                                / clamp_distance
                            )
                            ** 2
                        )
                        sign_target = (target > 0).to(prediction_raw.dtype)
                        sign_bce = (
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                prediction_raw / sign_logit_temperature,
                                sign_target,
                            )
                        )
                        reconstruction_loss = (
                            reconstruction_loss
                            + normalized_sdf_mse_weight * normalized_sdf_mse
                            + sign_bce_weight * sign_bce
                        )
                    else:
                        normalized_sdf_mse = zero
                        sign_bce = zero

                    if reg_type == "mean_squared":
                        code_loss = torch.mean(codes.pow(2))
                    else:
                        code_loss = torch.linalg.vector_norm(
                            codes, dim=1
                        ).mean()

                    gradients_all = None
                    if needs_all_gradients:
                        gradient_prediction = (
                            prediction_raw
                            if eikonal_on_raw_output
                            or needs_surface_gradients
                            else prediction
                        )
                        gradients_all = torch.autograd.grad(
                            outputs=gradient_prediction,
                            inputs=xyz,
                            grad_outputs=torch.ones_like(gradient_prediction),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]

                    if needs_surface_gradients:
                        surface_gradients = gradients_all[surface_mask]
                        surface_normal_loss = torch.mean(
                            1.0
                            - torch.nn.functional.cosine_similarity(
                                surface_gradients,
                                target_normals[surface_mask],
                                dim=1,
                                eps=1e-8,
                            )
                        )
                    else:
                        surface_normal_loss = zero
                    reconstruction_loss = (
                        reconstruction_loss
                        + current_surface_normal_weight * surface_normal_loss
                    )

                    if compute_eikonal:
                        if use_all_eikonal_points:
                            gradients = gradients_all
                        else:
                            sample_index = torch.randint(
                                0,
                                points_per_scene,
                                size=(chunk_scenes, eikonal_count),
                                device=device,
                            )
                            gather_index = sample_index[:, :, None].expand(-1, -1, 3)
                            xyz_eikonal = torch.gather(
                                chunk[:, :, :3], dim=1, index=gather_index
                            )
                            xyz_eikonal = (
                                xyz_eikonal.reshape(-1, 3)
                                .detach()
                                .requires_grad_(True)
                            )
                            eikonal_codes = (
                                codes[:, None, :]
                                .expand(chunk_scenes, eikonal_count, latent_size)
                                .reshape(-1, latent_size)
                            )
                            eikonal_raw = decoder(
                                torch.cat((eikonal_codes, xyz_eikonal), dim=1)
                            )
                            eikonal_prediction = (
                                eikonal_raw
                                if eikonal_on_raw_output
                                else eikonal_raw.clamp(
                                    -clamp_distance, clamp_distance
                                )
                            )
                            gradients = torch.autograd.grad(
                                outputs=eikonal_prediction,
                                inputs=xyz_eikonal,
                                grad_outputs=torch.ones_like(
                                    eikonal_prediction
                                ),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True,
                            )[0]
                        gradient_error = (
                            torch.linalg.vector_norm(gradients, dim=1)
                            - 1.0
                        )
                        if eikonal_loss_type == "l1":
                            eikonal_loss = torch.mean(
                                torch.abs(gradient_error)
                            )
                        else:
                            eikonal_loss = torch.mean(gradient_error**2)
                    else:
                        eikonal_loss = zero
                    regularization_weight = reg_lambda * min(
                        1.0, epoch / max(1, reg_warmup_epochs)
                    )
                    loss = (
                        reconstruction_loss
                        + regularization_weight * code_loss
                        + current_eikonal_weight * eikonal_loss
                    )
                    weight = chunk_scenes / batch_scenes
                    (loss * weight).backward()

                    batch_values["loss"] += float(loss.detach().cpu()) * weight
                    batch_values["sdf"] += float(sdf_loss.detach().cpu()) * weight
                    batch_values["off_surface_sdf"] += (
                        float(off_surface_sdf_loss.detach().cpu()) * weight
                    )
                    batch_values["surface_sdf"] += (
                        float(surface_sdf_loss.detach().cpu()) * weight
                    )
                    batch_values["surface_normal"] += (
                        float(surface_normal_loss.detach().cpu()) * weight
                    )
                    batch_values["reconstruction"] += (
                        float(reconstruction_loss.detach().cpu()) * weight
                    )
                    batch_values["code"] += float(code_loss.detach().cpu()) * weight
                    batch_values["eikonal"] += (
                        float(eikonal_loss.detach().cpu()) * weight
                    )
                    if use_auxiliary_sdf_losses:
                        auxiliary_batch_values["normalized_sdf_mse"] += (
                            float(normalized_sdf_mse.detach().cpu()) * weight
                        )
                        auxiliary_batch_values["sign_bce"] += (
                            float(sign_bce.detach().cpu()) * weight
                        )
                    if log_field_diagnostics:
                        with torch.no_grad():
                            diagnostic_prediction = prediction_for_loss[
                                off_surface_mask
                            ]
                            diagnostic_target = target[off_surface_mask]
                            diagnostic_batch_values["sign_accuracy"] += (
                                float(
                                    (
                                        (diagnostic_prediction >= 0)
                                        == (diagnostic_target >= 0)
                                    )
                                    .to(torch.float32)
                                    .mean()
                                    .cpu()
                                )
                                * weight
                            )
                            diagnostic_batch_values[
                                "raw_output_abs_mean"
                            ] += (
                                float(prediction_raw.abs().mean().cpu())
                                * weight
                            )
                            diagnostic_batch_values["raw_output_std"] += (
                                float(prediction_raw.std().cpu()) * weight
                            )

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), gradient_clip)
                optimizer.step()
                with torch.no_grad():
                    norms = latent_codes.weight.norm(dim=1, keepdim=True)
                    latent_codes.weight.mul_(
                        torch.clamp(code_bound / (norms + 1e-12), max=1.0)
                    )
                for key in totals:
                    totals[key] += batch_values[key]
                if log_field_diagnostics:
                    for key in diagnostic_totals:
                        diagnostic_totals[key] += diagnostic_batch_values[key]
                if use_auxiliary_sdf_losses:
                    for key in auxiliary_totals:
                        auxiliary_totals[key] += auxiliary_batch_values[key]
                batches += 1

                if args.smoke_test:
                    break

            epoch_row = {
                "epoch": epoch,
                "loss": totals["loss"] / batches,
                "sdf_l1": totals["sdf"] / batches,
                "off_surface_sdf_l1": totals["off_surface_sdf"] / batches,
                "surface_sdf_l1": totals["surface_sdf"] / batches,
                "surface_normal": totals["surface_normal"] / batches,
                "reconstruction": totals["reconstruction"] / batches,
                "code_regularization_value": totals["code"] / batches,
                "code_regularization_type": reg_type,
                "eikonal": totals["eikonal"] / batches,
                "code_regularization_weight": reg_lambda
                * min(1.0, epoch / max(1, reg_warmup_epochs)),
                "network_lr": network_lr,
                "latent_lr": latent_lr,
                "seconds": time.time() - epoch_start,
            }
            epoch_row[
                "code_l2_norm"
                if reg_type == "l2_norm"
                else "code_mean_squared"
            ] = totals["code"] / batches
            if (
                eikonal_start_epoch > 0
                or eikonal_ramp_epochs > 0
                or eikonal_on_raw_output
            ):
                epoch_row["eikonal_weight"] = current_eikonal_weight
            if loss_mode == "surface_normal_sdf":
                epoch_row.update(
                    {
                        "off_surface_sdf_weight": off_surface_sdf_weight,
                        "geometry_weight": current_geometry_weight,
                        "surface_sdf_weight": current_surface_sdf_weight,
                        "surface_normal_weight": current_surface_normal_weight,
                    }
                )
            if log_field_diagnostics:
                for key, value in diagnostic_totals.items():
                    epoch_row[key] = value / batches
            if use_auxiliary_sdf_losses:
                epoch_row["normalized_sdf_mse"] = (
                    auxiliary_totals["normalized_sdf_mse"] / batches
                )
                epoch_row["normalized_sdf_mse_weight"] = (
                    normalized_sdf_mse_weight
                )
                epoch_row["sign_bce"] = auxiliary_totals["sign_bce"] / batches
                epoch_row["sign_bce_weight"] = sign_bce_weight
            if device.type == "cuda":
                epoch_row["cuda_peak_allocated_gb"] = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                )
                epoch_row["cuda_peak_reserved_gb"] = (
                    torch.cuda.max_memory_reserved(device) / 1024**3
                )
            print(json.dumps(epoch_row, sort_keys=True))
            if args.smoke_test:
                print("Smoke test passed; no checkpoints were written.")
                return 0
            append_csv(log_dir / "training_history.csv", epoch_row)

            if checkpoint_due(epoch):
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.pth",
                    epoch,
                    decoder,
                    latent_codes,
                    optimizer,
                    run_config,
                    train_rows,
                    best_validation_l1,
                )
                save_checkpoint(
                    checkpoint_dir / "latest.pth",
                    epoch,
                    decoder,
                    latent_codes,
                    optimizer,
                    run_config,
                    train_rows,
                    best_validation_l1,
                )

            if epoch % validation_every == 0 or epoch == final_epoch:
                score, details = validation_score(
                    decoder, validation_rows, run_config, device, epoch
                )
                for detail in details:
                    append_csv(log_dir / "validation_per_scan.csv", detail)
                validation_row = {
                    "epoch": epoch,
                    "mean_heldout_sdf_l1": score,
                    "scan_count": len(details),
                }
                append_csv(log_dir / "validation_history.csv", validation_row)
                print(json.dumps({"validation": validation_row}, sort_keys=True))
                if score < best_validation_l1:
                    best_validation_l1 = score
                    save_checkpoint(
                        checkpoint_dir / "best.pth",
                        epoch,
                        decoder,
                        latent_codes,
                        optimizer,
                        run_config,
                        train_rows,
                        best_validation_l1,
                    )
                if checkpoint_due(epoch):
                    save_checkpoint(
                        checkpoint_dir / "latest.pth",
                        epoch,
                        decoder,
                        latent_codes,
                        optimizer,
                        run_config,
                        train_rows,
                        best_validation_l1,
                    )
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted; saving interrupted.pth.", file=sys.stderr)
        if not args.smoke_test:
            save_checkpoint(
                checkpoint_dir / "interrupted.pth",
                max(0, last_epoch),
                decoder,
                latent_codes,
                optimizer,
                run_config,
                train_rows,
                best_validation_l1,
            )

    write_json(
        output_dir / "training_status.json",
        {
            "name": config["name"],
            "status": "interrupted" if interrupted else "complete",
            "final_epoch": int(max(0, last_epoch)),
            "best_validation_l1": (
                best_validation_l1 if np.isfinite(best_validation_l1) else None
            ),
            "device": str(device),
            "scene_chunk_size": scene_chunk_size,
            "effective_batch_size": int(run_config["scenes_per_batch"]),
            "eikonal_points_per_scene": eikonal_count,
            "surface_samples_per_scene": surface_samples_per_scene,
            "off_surface_samples_per_scene": (
                int(run_config["samples_per_scene"])
                - surface_samples_per_scene
            ),
            "loss_mode": loss_mode,
        },
    )
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
