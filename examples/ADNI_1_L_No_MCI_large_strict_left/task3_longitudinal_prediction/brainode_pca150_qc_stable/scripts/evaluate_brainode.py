#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brainode_model import integrate_autoregressive_rk4, integrate_sequence_rk4
from core_brainode_common import SPLITS, TASK_DIR, load_config, resolve_repo_path, write_json
from train_core_brainode import (
    TrajectoryRecord,
    build_model,
    build_subject_sequences,
    build_trajectory_records,
    full_brainode_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate core/full BrainODE checkpoints on train, val, or test trajectories."
    )
    parser.add_argument(
        "--config",
        default=str(TASK_DIR / "configs" / "core_brainode.json"),
    )
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument(
        "--split",
        default="test",
        choices=(*SPLITS, "all"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--first-last-only",
        action="store_true",
        help="Evaluate only baseline-to-last trajectories.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Evaluate every chronological source-target pair, matching direct-flow pair metrics.",
    )
    return parser.parse_args()


def load_npz_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def resolve_device(preferred: str | None) -> torch.device:
    if preferred:
        device = torch.device(preferred)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def resolve_checkpoint_path(
    checkpoint: str,
    config: dict[str, Any],
    run_name: str,
) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path
    if path.suffix == ".pth":
        resolved = resolve_repo_path(path)
        if resolved.is_file():
            return resolved
    output_root = resolve_repo_path(config["training"]["output_root"])
    name = checkpoint if checkpoint.endswith(".pth") else f"{checkpoint}.pth"
    return output_root / run_name / "checkpoints" / name


def inverse_transform_pca(
    coefficients: torch.Tensor,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    original_shape = coefficients.shape[:-1]
    flat = coefficients.reshape(-1, coefficients.shape[-1])
    reconstructed = flat @ components + mean_flat
    return reconstructed.reshape(*original_shape, mean_flat.shape[0])


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0.0, "mean": float("nan"), "median": float("nan")}
    return {
        "count": float(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_pairwise_records(sequences: list[dict[str, Any]]) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []
    for sequence in sequences:
        subject_id = sequence["subject_id"]
        times = sequence["times"]
        targets = sequence["targets"]
        conditions = sequence["conditions"]
        for source_index in range(len(times) - 1):
            for target_index in range(source_index + 1, len(times)):
                record_times = times[source_index : target_index + 1]
                record_targets = targets[source_index : target_index + 1]
                record_conditions = conditions[source_index : target_index + 1]
                records.append(
                    TrajectoryRecord(
                        subject_id=subject_id,
                        direction="forward",
                        start_visit_order=source_index,
                        length=len(record_times),
                        times=record_times,
                        targets=record_targets,
                        conditions=record_conditions,
                        condition=float(record_conditions[0]),
                    )
                )
    return records


@torch.no_grad()
def evaluate_records(
    model: torch.nn.Module,
    records: list[Any],
    device: torch.device,
    integration_substeps: int,
    mean_flat: torch.Tensor,
    components: torch.Tensor,
    use_autoregressive_rollout: bool,
    split: str,
    record_set: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for record in records:
        times = torch.from_numpy(record.times.copy()).float().unsqueeze(0).to(device)
        targets = torch.from_numpy(record.targets.copy()).float().unsqueeze(0).to(device)
        condition = torch.tensor([record.condition], dtype=torch.float32, device=device)
        initial_state = targets[:, 0, :]
        if use_autoregressive_rollout:
            prediction, predicted_conditions = integrate_autoregressive_rk4(
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
            predicted_conditions = condition.view(1, 1).expand(1, times.shape[1])

        endpoint_prediction = prediction[:, -1, :]
        endpoint_target = targets[:, -1, :]
        no_change = targets[:, 0, :]
        endpoint_pca_mse = torch.mean((endpoint_prediction - endpoint_target) ** 2, dim=1)
        no_change_pca_mse = torch.mean((no_change - endpoint_target) ** 2, dim=1)
        endpoint_vertices = inverse_transform_pca(endpoint_prediction, mean_flat, components)
        target_vertices = inverse_transform_pca(endpoint_target, mean_flat, components)
        no_change_vertices = inverse_transform_pca(no_change, mean_flat, components)
        endpoint_vertex_mae = torch.mean(
            torch.abs(endpoint_vertices - target_vertices),
            dim=1,
        )
        no_change_vertex_mae = torch.mean(
            torch.abs(no_change_vertices - target_vertices),
            dim=1,
        )
        rows.append(
            {
                "split": split,
                "record_set": record_set,
                "subject_id": record.subject_id,
                "direction": record.direction,
                "start_visit_order": int(record.start_visit_order),
                "length": int(record.length),
                "source_time_norm": float(record.times[0]),
                "target_time_norm": float(record.times[-1]),
                "gap_norm": float(record.times[-1] - record.times[0]),
                "source_condition": float(record.conditions[0]),
                "target_condition": float(record.conditions[-1]),
                "predicted_endpoint_condition": float(predicted_conditions[0, -1].item()),
                "endpoint_pca_mse": float(endpoint_pca_mse.item()),
                "no_change_pca_mse": float(no_change_pca_mse.item()),
                "endpoint_pca_improvement": float(
                    no_change_pca_mse.item() - endpoint_pca_mse.item()
                ),
                "endpoint_vertex_mae": float(endpoint_vertex_mae.item()),
                "no_change_vertex_mae": float(no_change_vertex_mae.item()),
                "endpoint_vertex_mae_improvement": float(
                    no_change_vertex_mae.item() - endpoint_vertex_mae.item()
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.first_last_only and args.pairwise:
        raise ValueError("--first-last-only and --pairwise are mutually exclusive.")
    config = load_config(args.config)
    full_config = full_brainode_config(config)
    training_config = dict(config["training"])
    model_config = dict(config["model"])
    brainode_config = dict(config["brainode"])
    run_name = str(args.run_name or training_config["run_name"])
    device = resolve_device(args.device or training_config.get("device"))
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, config, run_name)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    train_archive = load_npz_archive(TASK_DIR / "dataset" / "train_subject_sequences.npz")
    latent_dim = int(train_archive["visit_pca_150"].shape[1])
    model = build_model(
        latent_dim=latent_dim,
        model_config=model_config,
        full_config=full_config,
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])

    pca_model_dir = resolve_repo_path(config["task2"]["pca_model_dir"])
    mean_flat = torch.from_numpy(np.load(pca_model_dir / "mean.npy").astype(np.float32)).to(device)
    components_256 = np.load(pca_model_dir / "components_256.npy").astype(np.float32)
    components = torch.from_numpy(components_256[:latent_dim]).to(device)

    requested_splits = SPLITS if args.split == "all" else (args.split,)
    all_rows: list[dict[str, Any]] = []
    for split in requested_splits:
        archive = load_npz_archive(TASK_DIR / "dataset" / f"{split}_subject_sequences.npz")
        sequences = build_subject_sequences(archive)
        if args.pairwise:
            records = build_pairwise_records(sequences)
            record_set = "all_pairs"
        else:
            records = build_trajectory_records(
                sequences=sequences,
                include_backward=False,
                include_length_one=False,
                only_start_visit_zero=args.first_last_only,
            )
            record_set = "first_last" if args.first_last_only else "all_forward"
        all_rows.extend(
            evaluate_records(
                model=model,
                records=records,
                device=device,
                integration_substeps=int(training_config["integration_substeps"]),
                mean_flat=mean_flat,
                components=components,
                use_autoregressive_rollout=bool(
                    full_config.get("use_autoregressive_rollout", False)
                ),
                split=split,
                record_set=record_set,
            )
        )

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else resolve_repo_path(training_config["output_root"])
        / run_name
        / "evaluation"
        / checkpoint_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / f"{args.split}_trajectory_metrics.csv", all_rows)
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "record_count": len(all_rows),
        "use_autoregressive_rollout": bool(
            full_config.get("use_autoregressive_rollout", False)
        ),
        "endpoint_pca_mse": summarize([row["endpoint_pca_mse"] for row in all_rows]),
        "no_change_pca_mse": summarize([row["no_change_pca_mse"] for row in all_rows]),
        "endpoint_pca_improvement": summarize(
            [row["endpoint_pca_improvement"] for row in all_rows]
        ),
        "endpoint_vertex_mae": summarize([row["endpoint_vertex_mae"] for row in all_rows]),
        "no_change_vertex_mae": summarize([row["no_change_vertex_mae"] for row in all_rows]),
        "endpoint_vertex_mae_improvement": summarize(
            [row["endpoint_vertex_mae_improvement"] for row in all_rows]
        ),
    }
    write_json(output_dir / f"{args.split}_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
