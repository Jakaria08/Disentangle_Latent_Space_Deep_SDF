#!/usr/bin/env python3
"""Matched first-to-last and velocity evaluation for a direct LAMM/Spiral mesh flow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


TASK_ROOT = Path(__file__).resolve().parents[1]
COHORT_TASK_ROOT = TASK_ROOT.parent
LOCAL_SCRIPTS = TASK_ROOT / "scripts"
SPIRAL_TASK_ROOT = COHORT_TASK_ROOT / "task5_direct_mesh_cocycle_spiral_unet_v1"
SPIRAL_SCRIPTS = SPIRAL_TASK_ROOT / "scripts"
SURFACE_SCRIPTS = COHORT_TASK_ROOT / "task3_pca_corrective_cocycle_128_v1" / "scripts"
DEFAULT_DATA_ROOT = Path(
    "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("lamm", "spiral"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--surface-points", type=int, default=10000)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-velocity-scans", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def activate_family(family: str):
    selected = LOCAL_SCRIPTS if family == "lamm" else SPIRAL_SCRIPTS
    sys.path.insert(0, str(selected))
    C = importlib.import_module("common")
    D = importlib.import_module("data")
    # The isolated LAMM train module activates the tested direct-Spiral objective path.
    # Import it before resolving ``objectives`` so both families use that same code.
    T = importlib.import_module("train")
    O = importlib.import_module("objectives")
    return C, D, O, T


def stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base)) % (2**31 - 1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_surface(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics = (
        "prediction_assd_mm",
        "prediction_hd95_mm",
        "prediction_chamfer_l1_mm",
        "prediction_chamfer_l2_squared_mm2",
        "prediction_normal_signed_cosine",
        "prediction_normal_absolute_cosine",
        "prediction_flipped_face_fraction_vs_ground_truth",
        "nochange_assd_mm",
        "nochange_hd95_mm",
        "nochange_chamfer_l1_mm",
        "nochange_chamfer_l2_squared_mm2",
    )
    output: dict[str, dict[str, Any]] = {}
    for diagnosis in ("CN", "AD", "overall"):
        group = rows if diagnosis == "overall" else [r for r in rows if r["diagnosis"] == diagnosis]
        if not group:
            continue
        record: dict[str, Any] = {"subjects": len(group)}
        for metric in metrics:
            record[metric] = float(np.mean([float(row[metric]) for row in group]))
        for metric in ("assd_mm", "hd95_mm", "chamfer_l2_squared_mm2"):
            record[f"prediction_{metric}_ratio_to_nochange"] = record[
                f"prediction_{metric}"
            ] / max(record[f"nochange_{metric}"], 1.0e-12)
        output[diagnosis] = record
    return output


@torch.no_grad()
def surface_rows(model, split, rows, endpoint_records, points: int, seed: int):
    if str(SURFACE_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SURFACE_SCRIPTS))
    from surface_metrics import metrics as mesh_metrics

    faces = model.faces.detach().cpu().numpy()
    output: list[dict[str, Any]] = []
    for row, endpoint in zip(rows, endpoint_records):
        source = split.vertices[row.source : row.source + 1]
        source_age = split.ages[row.source : row.source + 1]
        target_age = split.ages[row.target : row.target + 1]
        label = split.labels[row.source : row.source + 1]
        prediction = model.transport(source, source_age, target_age, label)[0].cpu().numpy()
        source_np = source[0].cpu().numpy()
        target_np = split.vertices[row.target].cpu().numpy()
        sample_seed = stable_seed(str(split.scan_ids[row.target]), seed)
        predicted = mesh_metrics(target_np, prediction, faces, int(points), sample_seed)
        nochange = mesh_metrics(target_np, source_np, faces, int(points), sample_seed)
        record = dict(endpoint)
        record.update({f"prediction_{key}": value for key, value in predicted.items()})
        record.update({f"nochange_{key}": value for key, value in nochange.items()})
        output.append(record)
    return output


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise PermissionError("Test evaluation requires --allow-test")
    if args.surface_points < 100:
        raise ValueError("--surface-points must be at least 100")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if checkpoint_path.name != "best.pt":
        raise ValueError("Central comparison accepts only a selected best.pt checkpoint")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=False)

    C, D, O, T = activate_family(args.family)
    device = C.choose_device(args.device)
    data_root = args.data_root.expanduser().resolve()
    if args.family == "lamm":
        C.configure_data_root(data_root)
        split = D.load_split(args.split, device=device)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        T.validate_config(payload["config"])
        model, statistics = T.build_model(payload["config"], TASK_ROOT, device)
    else:
        split = D.load_split(args.split, data_root, device)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        T.validate_config(payload["config"])
        model, statistics = T.build_model(payload["config"], data_root, device)
    if bool(payload.get("test_data_loaded", False)):
        raise ValueError(f"Training/test leakage recorded in {checkpoint_path}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    first_last = O.first_last_rows(split)
    if args.max_subjects is not None:
        first_last = first_last[: int(args.max_subjects)]
    endpoint = O.evaluate_rows(
        model, split, first_last, int(args.batch_size), compute_flips=True
    )
    enriched = surface_rows(
        model,
        split,
        first_last,
        endpoint["records"],
        int(args.surface_points),
        int(args.seed),
    )
    velocity_limit = int(args.max_velocity_scans)
    velocity = O.evaluate_velocity_selection(
        model,
        split,
        int(args.batch_size),
        max_scans=velocity_limit if velocity_limit > 0 else None,
    )
    all_pairs = C.load_pairs(
        args.split,
        split.scan_ids,
        split.subject_ids,
        split.labels.detach().cpu().numpy(),
    )
    cocycle = O.evaluate_cocycle(
        model,
        split,
        all_pairs,
        float(statistics["endpoint_scale_mm"]),
        int(args.batch_size),
        min(len(all_pairs), int(payload["config"]["selection"]["cocycle_validation_pairs"])),
    )
    summary = {
        "schema_version": 1,
        "name": args.name,
        "family": f"direct_{args.family}",
        "checkpoint": str(checkpoint_path),
        "epoch": int(payload["epoch"]),
        "split": args.split,
        "subjects": len(first_last),
        "protocol": "one first-to-last prediction per held-out subject",
        "endpoint": endpoint["groups"],
        "surface": summarize_surface(enriched),
        "instantaneous_velocity": velocity,
        "cocycle": cocycle,
        "test_was_explicitly_authorized": bool(args.split == "test" and args.allow_test),
    }
    write_csv(destination / "per_subject.csv", enriched)
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
