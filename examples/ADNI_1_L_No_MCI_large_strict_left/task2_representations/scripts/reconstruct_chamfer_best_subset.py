#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from task2_common import (
    choose_device,
    decode_latent_to_mesh,
    fit_single_latent,
    load_config,
    load_decoder_checkpoint,
    load_manifest,
    resolve_repo_path,
    rows_for_split,
    stable_seed,
    write_json,
)


DEFAULT_CONFIG = (
    "examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/"
    "configs/inr_siren_eikonal_warmstart_no_skip.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a small train/val/test subset from the current best "
            "SIREN checkpoint and compute per-scan and average Chamfer metrics."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="INR JSON config.")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint name/path. Defaults to the current best checkpoint.",
    )
    parser.add_argument("--device", default=None, help="Device such as cuda:0 or cpu.")
    parser.add_argument(
        "--per-split",
        type=int,
        default=5,
        help="Number of scans per split. Default: 5.",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to evaluate. Default: train,val,test.",
    )
    parser.add_argument(
        "--selection",
        choices=("first", "random"),
        default="first",
        help="Subset selection mode. Default: first scans in manifest order.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--surface-points",
        type=int,
        default=30000,
        help="Surface points sampled per mesh for Chamfer. Default: 30000.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Marching-cubes grid resolution. Defaults to config reconstruction.resolution.",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help="Decoder grid batch size. Defaults to config reconstruction.max_batch.",
    )
    parser.add_argument(
        "--latent-steps",
        type=int,
        default=None,
        help="Override latent optimization steps for val/test and optionally train.",
    )
    parser.add_argument(
        "--fit-train-latents",
        action="store_true",
        help=(
            "Fit temporary train latents instead of using the train latent table "
            "stored in the checkpoint."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing meshes/metrics in the output directory when possible.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            "<experiment_output_dir>/quick_chamfer_best_subset."
        ),
    )
    return parser.parse_args()


def as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Empty mesh scene: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(loaded)}")
    return loaded


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    points, _faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float32)


def chamfer_metrics(
    ground_truth_points: np.ndarray,
    predicted_points: np.ndarray,
) -> dict[str, float]:
    pred_tree = cKDTree(predicted_points)
    gt_tree = cKDTree(ground_truth_points)
    gt_to_pred = pred_tree.query(ground_truth_points, k=1, workers=-1)[0]
    pred_to_gt = gt_tree.query(predicted_points, k=1, workers=-1)[0]
    return {
        "gt_to_pred_mean": float(np.mean(gt_to_pred)),
        "pred_to_gt_mean": float(np.mean(pred_to_gt)),
        "chamfer_l1": float(np.mean(gt_to_pred) + np.mean(pred_to_gt)),
        "chamfer_l2_squared": float(
            np.mean(gt_to_pred**2) + np.mean(pred_to_gt**2)
        ),
        "assd": float(0.5 * (np.mean(gt_to_pred) + np.mean(pred_to_gt))),
        "hd95": float(
            max(np.quantile(gt_to_pred, 0.95), np.quantile(pred_to_gt, 0.95))
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_rows(
    rows: list[dict[str, str]],
    splits: list[str],
    per_split: int,
    selection: str,
    seed: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    rng = np.random.default_rng(seed)
    for split in splits:
        split_rows = rows_for_split(rows, split)
        if len(split_rows) < per_split:
            raise ValueError(
                f"Split {split!r} only has {len(split_rows)} rows; "
                f"requested {per_split}."
            )
        if selection == "random":
            indices = sorted(rng.choice(len(split_rows), size=per_split, replace=False))
            selected.extend(split_rows[index] for index in indices)
        else:
            selected.extend(split_rows[:per_split])
    return selected


def checkpoint_train_latent_map(
    checkpoint_payload: dict[str, Any],
) -> dict[str, np.ndarray]:
    scan_ids = checkpoint_payload.get("train_scan_ids") or []
    latent_state = checkpoint_payload.get("latent_codes") or {}
    weights = latent_state.get("weight")
    if not scan_ids or weights is None:
        return {}
    if isinstance(weights, torch.Tensor):
        array = weights.detach().cpu().numpy().astype(np.float32)
    else:
        array = np.asarray(weights, dtype=np.float32)
    if len(scan_ids) != len(array):
        raise ValueError(
            "Checkpoint train_scan_ids and latent_codes.weight have different lengths: "
            f"{len(scan_ids)} vs {len(array)}."
        )
    return {scan_id: array[index] for index, scan_id in enumerate(scan_ids)}


def fit_or_load_latent(
    row: dict[str, str],
    decoder,
    config: dict[str, Any],
    device,
    train_latents: dict[str, np.ndarray],
    args: argparse.Namespace,
    latent_dir: Path,
) -> tuple[np.ndarray, dict[str, Any], str]:
    latent_dir.mkdir(parents=True, exist_ok=True)
    latent_path = latent_dir / f"{row['scan_id']}.npy"
    stats_path = latent_dir / f"{row['scan_id']}.json"
    can_use_checkpoint_train_latent = (
        row["split"] == "train"
        and not args.fit_train_latents
        and row["scan_id"] in train_latents
    )
    if can_use_checkpoint_train_latent:
        latent = train_latents[row["scan_id"]]
        stats = {
            "scan_id": row["scan_id"],
            "split": row["split"],
            "source": "checkpoint_train_latent",
            "latent_norm": float(np.linalg.norm(latent)),
        }
        np.save(latent_path, latent)
        write_json(stats_path, stats)
        return latent, stats, "checkpoint_train_latent"

    if args.skip_existing and latent_path.is_file() and stats_path.is_file():
        return np.load(latent_path), json.loads(stats_path.read_text()), "existing_fit"

    fit_config = dict(config["latent_fit"])
    if args.latent_steps is not None:
        fit_config["steps"] = int(args.latent_steps)
    latent, stats = fit_single_latent(
        decoder=decoder,
        sdf_path=row["sdf_npz_path"],
        latent_size=int(config["latent_size"]),
        fit_config=fit_config,
        clamp_distance=float(config["clamp_distance"]),
        device=device,
        seed=stable_seed(row["scan_id"], args.seed),
    )
    stats.update(
        {
            "scan_id": row["scan_id"],
            "split": row["split"],
            "source": "optimized_temporary_latent",
        }
    )
    np.save(latent_path, latent)
    write_json(stats_path, stats)
    return latent, stats, "optimized_temporary_latent"


def summarize(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "gt_to_pred_mean",
        "pred_to_gt_mean",
        "chamfer_l1",
        "chamfer_l2_squared",
        "assd",
        "hd95",
        "volume_absolute_error",
        "volume_relative_error",
    )
    summary: dict[str, Any] = {}
    groups = {"overall": metric_rows}
    for split in sorted({row["split"] for row in metric_rows}):
        groups[f"split:{split}"] = [
            row for row in metric_rows if row["split"] == split
        ]
    for group_name, rows in groups.items():
        summary[group_name] = {"count": len(rows)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            summary[group_name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return summary


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    decoder, checkpoint_payload, checkpoint_path = load_decoder_checkpoint(
        config, args.checkpoint, device
    )
    rows = load_manifest(config["manifest"])
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    invalid_splits = set(splits).difference({"train", "val", "test"})
    if invalid_splits:
        raise ValueError(f"Unknown split names: {sorted(invalid_splits)}")
    selected_rows = select_rows(
        rows=rows,
        splits=splits,
        per_split=int(args.per_split),
        selection=args.selection,
        seed=int(args.seed),
    )

    experiment_dir = resolve_repo_path(config["output_dir"])
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else experiment_dir / "quick_chamfer_best_subset"
    )
    latent_dir = output_dir / "latents"
    mesh_dir = output_dir / "meshes"
    point_dir = output_dir / "surface_points"
    report_dir = output_dir / "reports"
    for directory in (latent_dir, mesh_dir, point_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    resolution = int(args.resolution or config["reconstruction"]["resolution"])
    max_batch = int(args.max_batch or config["reconstruction"]["max_batch"])
    surface_points = int(args.surface_points)
    train_latents = checkpoint_train_latent_map(checkpoint_payload)

    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    print(
        f"Loaded checkpoint {checkpoint_path} "
        f"(epoch={checkpoint_payload.get('epoch')}, "
        f"best_validation_l1={checkpoint_payload.get('best_validation_l1')})."
    )
    print(
        f"Evaluating {len(selected_rows)} scans "
        f"({args.per_split} per split) at resolution={resolution}, "
        f"surface_points={surface_points} on {device}."
    )

    for index, row in enumerate(selected_rows, start=1):
        scan_id = row["scan_id"]
        try:
            latent, latent_stats, latent_source = fit_or_load_latent(
                row=row,
                decoder=decoder,
                config=config,
                device=device,
                train_latents=train_latents,
                args=args,
                latent_dir=latent_dir,
            )
            pred_mesh_path = mesh_dir / row["split"] / f"{scan_id}.ply"
            pred_report_path = report_dir / f"{scan_id}_decode.json"
            if not (
                args.skip_existing
                and pred_mesh_path.is_file()
                and pred_report_path.is_file()
            ):
                decode_stats = decode_latent_to_mesh(
                    decoder=decoder,
                    latent=latent,
                    output_path=pred_mesh_path,
                    resolution=resolution,
                    max_batch=max_batch,
                    device=device,
                )
                decode_stats.update(
                    {
                        "scan_id": scan_id,
                        "split": row["split"],
                        "checkpoint": str(checkpoint_path),
                        "latent_source": latent_source,
                    }
                )
                write_json(pred_report_path, decode_stats)

            gt_mesh = as_mesh(Path(row["mesh_path"]))
            pred_mesh = as_mesh(pred_mesh_path)
            gt_points_path = point_dir / f"{scan_id}_gt_points.npz"
            pred_points_path = point_dir / f"{scan_id}_pred_points.npz"
            if args.skip_existing and gt_points_path.is_file():
                with np.load(gt_points_path) as archive:
                    gt_points = np.asarray(archive["points"], dtype=np.float32)
            else:
                gt_points = sample_surface(
                    gt_mesh,
                    surface_points,
                    stable_seed(f"gt:{scan_id}", args.seed),
                )
                np.savez_compressed(gt_points_path, points=gt_points)
            if args.skip_existing and pred_points_path.is_file():
                with np.load(pred_points_path) as archive:
                    pred_points = np.asarray(archive["points"], dtype=np.float32)
            else:
                pred_points = sample_surface(
                    pred_mesh,
                    surface_points,
                    stable_seed(f"pred:{scan_id}", args.seed),
                )
                np.savez_compressed(pred_points_path, points=pred_points)

            metrics = chamfer_metrics(gt_points, pred_points)
            gt_volume = abs(float(gt_mesh.volume))
            pred_volume = abs(float(pred_mesh.volume))
            volume_absolute_error = abs(pred_volume - gt_volume)
            metric_row = {
                "scan_id": scan_id,
                "subject_id": row["subject_id"],
                "split": row["split"],
                "diagnosis": row["diagnosis"],
                "visit_order": row["visit_order"],
                "latent_source": latent_source,
                "heldout_sdf_l1": latent_stats.get("heldout_sdf_l1"),
                **metrics,
                "ground_truth_volume": gt_volume,
                "predicted_volume": pred_volume,
                "volume_absolute_error": volume_absolute_error,
                "volume_relative_error": volume_absolute_error / max(gt_volume, 1e-12),
                "predicted_watertight": bool(pred_mesh.is_watertight),
                "predicted_winding_consistent": bool(pred_mesh.is_winding_consistent),
                "ground_truth_mesh_path": row["mesh_path"],
                "predicted_mesh_path": str(pred_mesh_path),
            }
            metric_rows.append(metric_row)
            print(
                f"[{index}/{len(selected_rows)}] {scan_id} "
                f"split={row['split']} chamfer_l2_sq="
                f"{metric_row['chamfer_l2_squared']:.8f} "
                f"chamfer_l1={metric_row['chamfer_l1']:.8f} "
                f"latent={latent_source}"
            )
        except Exception as exc:
            failures.append({"scan_id": scan_id, "split": row["split"], "error": str(exc)})
            print(f"[{index}/{len(selected_rows)}] FAILED {scan_id}: {exc}")

    if not metric_rows:
        raise RuntimeError("No Chamfer metrics were produced.")

    summary = summarize(metric_rows)
    write_csv(output_dir / "per_scan_chamfer.csv", metric_rows)
    write_json(output_dir / "summary_chamfer.json", summary)
    write_json(
        output_dir / "run_info.json",
        {
            "config": str(Path(args.config).expanduser().resolve()),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_payload.get("epoch"),
            "checkpoint_best_validation_l1": checkpoint_payload.get(
                "best_validation_l1"
            ),
            "output_dir": str(output_dir),
            "splits": splits,
            "per_split": int(args.per_split),
            "selection": args.selection,
            "seed": int(args.seed),
            "surface_points": surface_points,
            "resolution": resolution,
            "max_batch": max_batch,
            "metric_definitions": {
                "chamfer_l1": "mean(gt_to_pred_distance) + mean(pred_to_gt_distance)",
                "chamfer_l2_squared": "mean(gt_to_pred_distance^2) + mean(pred_to_gt_distance^2)",
                "assd": "0.5 * chamfer_l1",
                "hd95": "max(q95(gt_to_pred_distance), q95(pred_to_gt_distance))",
            },
            "failure_count": len(failures),
            "failures": failures,
        },
    )
    print(json.dumps({"summary": summary, "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
