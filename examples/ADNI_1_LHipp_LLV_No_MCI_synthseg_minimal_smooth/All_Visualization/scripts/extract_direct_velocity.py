#!/usr/bin/env python3
"""Export per-scan cocycle-diagonal surface velocity for one direct mesh model."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


COHORT_TASK_ROOT = Path(__file__).resolve().parents[2]
LAMM_TASK = COHORT_TASK_ROOT / "task7_direct_mesh_cocycle_lamm_v1"
SPIRAL_TASK = COHORT_TASK_ROOT / "task5_direct_mesh_cocycle_spiral_unet_v1"
DEFAULT_DATA_ROOT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("lamm", "spiral"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def activate(family: str):
    scripts = (LAMM_TASK if family == "lamm" else SPIRAL_TASK) / "scripts"
    sys.path.insert(0, str(scripts))
    common = importlib.import_module("common")
    data = importlib.import_module("data")
    train = importlib.import_module("train")
    objectives = importlib.import_module("objectives")
    return common, data, train, objectives


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No velocity rows were generated")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise PermissionError("Test extraction requires --allow-test")
    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.name != "best.pt" or not checkpoint.is_file():
        raise FileNotFoundError(f"Expected selected best.pt: {checkpoint}")
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)

    common, data, train, objectives = activate(args.family)
    device = common.choose_device(args.device)
    data_root = args.data_root.expanduser().resolve()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if bool(payload.get("test_data_loaded", False)):
        raise ValueError("Checkpoint records test access during training")
    if args.family == "lamm":
        common.configure_data_root(data_root)
        split = data.load_split(args.split, device=device)
        train.validate_config(payload["config"])
        model, _ = train.build_model(payload["config"], LAMM_TASK, device)
    else:
        split = data.load_split(args.split, data_root, device)
        train.validate_config(payload["config"])
        model, _ = train.build_model(payload["config"], data_root, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    indices = objectives._velocity_scan_indices(split, None)
    rows: list[dict[str, Any]] = []
    for chunk in common.chunked(indices, int(args.batch_size)):
        selected = torch.as_tensor(chunk, dtype=torch.long, device=device)
        vertices = split.vertices[selected]
        observed = split.velocity_reference[selected]
        predicted = model.instantaneous_velocity(vertices, split.ages[selected], split.labels[selected])
        normals = objectives.vertex_normals(vertices, model.faces)
        predicted_normal = torch.sum(predicted * normals, dim=-1)
        observed_normal = torch.sum(observed * normals, dim=-1)
        vector_error = torch.sqrt(torch.mean((predicted - observed).square(), dim=(1, 2)))
        vector_zero = torch.sqrt(torch.mean(observed.square(), dim=(1, 2)))
        normal_error = torch.sqrt(torch.mean((predicted_normal - observed_normal).square(), dim=1))
        normal_zero = torch.sqrt(torch.mean(observed_normal.square(), dim=1))
        predicted_flat = predicted.reshape(len(chunk), -1)
        observed_flat = observed.reshape(len(chunk), -1)
        vector_cosine = torch.sum(predicted_flat * observed_flat, dim=1) / (
            torch.linalg.vector_norm(predicted_flat, dim=1) * torch.linalg.vector_norm(observed_flat, dim=1)
        ).clamp_min(1e-8)
        predicted_centered = predicted_normal - predicted_normal.mean(dim=1, keepdim=True)
        observed_centered = observed_normal - observed_normal.mean(dim=1, keepdim=True)
        normal_pearson = torch.sum(predicted_centered * observed_centered, dim=1) / (
            torch.linalg.vector_norm(predicted_centered, dim=1) * torch.linalg.vector_norm(observed_centered, dim=1)
        ).clamp_min(1e-8)
        sign = (torch.sign(predicted_normal) == torch.sign(observed_normal)).float().mean(dim=1)
        for local, index in enumerate(chunk):
            rows.append(
                {
                    "scan_id": str(split.scan_ids[index]),
                    "subject_id": str(split.subject_ids[index]),
                    "diagnosis": str(split.diagnoses[index]),
                    "age_years": float(split.ages[index].cpu()),
                    "reference_reliability": float(split.velocity_reference_weight[index].cpu()),
                    "predicted_speed_mm_per_year": float(torch.linalg.vector_norm(predicted[local], dim=-1).mean().cpu()),
                    "observed_speed_mm_per_year": float(torch.linalg.vector_norm(observed[local], dim=-1).mean().cpu()),
                    "predicted_inward_normal_mm_per_year": float((-predicted_normal[local].mean()).cpu()),
                    "observed_inward_normal_mm_per_year": float((-observed_normal[local].mean()).cpu()),
                    "vector_rmse_mm_per_year": float(vector_error[local].cpu()),
                    "zero_vector_rmse_mm_per_year": float(vector_zero[local].cpu()),
                    "normal_rmse_mm_per_year": float(normal_error[local].cpu()),
                    "zero_normal_rmse_mm_per_year": float(normal_zero[local].cpu()),
                    "vector_cosine": float(vector_cosine[local].cpu()),
                    "normal_pearson": float(normal_pearson[local].cpu()),
                    "normal_sign_agreement": float(sign[local].cpu()),
                    "method": args.name,
                    "method_label": args.name,
                    "model_family": "direct_mesh",
                    "velocity_definition": "cocycle diagonal field in corresponding surface coordinates",
                }
            )
    write_csv(destination / "per_scan.csv", rows)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "method": args.name,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "split": args.split,
        "scans": len(rows),
        "test_was_explicitly_authorized": bool(args.split == "test" and args.allow_test),
        "definition": "d/du Phi(X,t,u,d) at u=t; implemented by model.instantaneous_velocity",
        "reference": "observed longitudinal trajectory fitted from repeated aligned surfaces",
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
