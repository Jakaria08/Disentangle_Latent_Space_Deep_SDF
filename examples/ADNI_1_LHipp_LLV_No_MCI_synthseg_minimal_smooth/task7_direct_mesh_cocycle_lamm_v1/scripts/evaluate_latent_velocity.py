#!/usr/bin/env python3
"""Map latent-flow diagonal velocity through the decoder JVP and compare on surfaces."""

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


TASK_ROOT = Path(__file__).resolve().parents[1]
COHORT_TASK_ROOT = TASK_ROOT.parent
LATENT_SCRIPTS = COHORT_TASK_ROOT / "task3_latent_flow_128_v2_lamm" / "scripts"
DEFAULT_REFERENCE_ROOT = Path(
    "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-scans", type=int, default=0)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No positive-reliability velocity rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    left = vertices[:, faces[:, 1]] - vertices[:, faces[:, 0]]
    right = vertices[:, faces[:, 2]] - vertices[:, faces[:, 0]]
    face_normals = torch.cross(left, right, dim=-1)
    output = torch.zeros_like(vertices)
    for corner in range(3):
        output.index_add_(1, faces[:, corner], face_normals)
    return torch.nn.functional.normalize(output, dim=-1, eps=1.0e-8)


def weighted_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = np.asarray([float(row["reference_reliability"]) for row in rows])

    def weighted(name: str) -> float:
        return float(np.average([float(row[name]) for row in rows], weights=weights))

    vector_error = weighted("vector_rmse_mm_per_year")
    vector_zero = weighted("zero_vector_rmse_mm_per_year")
    normal_error = weighted("normal_rmse_mm_per_year")
    normal_zero = weighted("zero_normal_rmse_mm_per_year")
    predicted_speed = weighted("predicted_speed_mm_per_year")
    observed_speed = weighted("observed_speed_mm_per_year")
    vector_ratio = vector_error / max(vector_zero, 1.0e-8)
    normal_ratio = normal_error / max(normal_zero, 1.0e-8)
    return {
        "scans": len(rows),
        "reference_weight_sum": float(weights.sum()),
        "reference_reliability_mean": float(weights.mean()),
        "vector_rmse_mm_per_year": vector_error,
        "zero_vector_rmse_mm_per_year": vector_zero,
        "vector_error_to_zero_ratio": vector_ratio,
        "normal_rmse_mm_per_year": normal_error,
        "zero_normal_rmse_mm_per_year": normal_zero,
        "normal_error_to_zero_ratio": normal_ratio,
        "normalized_error_ratio": 0.5 * (vector_ratio + normal_ratio),
        "vector_cosine": weighted("vector_cosine"),
        "normal_pearson": weighted("normal_pearson"),
        "normal_sign_agreement": weighted("normal_sign_agreement"),
        "predicted_speed_mm_per_year": predicted_speed,
        "observed_speed_mm_per_year": observed_speed,
        "speed_ratio": predicted_speed / max(observed_speed, 1.0e-8),
    }


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise PermissionError("Test evaluation requires --allow-test")
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(LATENT_SCRIPTS))
    bootstrap = importlib.import_module("_bootstrap")
    bootstrap.activate()
    module = importlib.import_module("evaluate_n3_ensemble")
    C = importlib.import_module("common")
    device = C.choose_device(args.device)
    run_dir = args.run_dir.expanduser().resolve()
    member = module.load_member(args.name, run_dir, args.split, device)
    slope, residual = module.age_norm_per_year(member.train_archive)

    cache = args.reference_root.expanduser().resolve() / "cache" / f"{args.split}_surfaces.npz"
    with np.load(cache, allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in archive.files}
    latent_index = {
        str(scan_id): index for index, scan_id in enumerate(member.archive["visit_scan_ids"])
    }
    selected = [
        index
        for index, weight in enumerate(reference["velocity_reference_weight"])
        if float(weight) > 0.0 and str(reference["scan_ids"][index]) in latent_index
    ]
    if int(args.max_scans) > 0:
        selected = selected[: int(args.max_scans)]
    faces = member.geometry.faces.to(device)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(selected), int(args.batch_size)):
        ref_indices = selected[start : start + int(args.batch_size)]
        latent_indices = torch.as_tensor(
            [latent_index[str(reference["scan_ids"][index])] for index in ref_indices],
            dtype=torch.long,
            device=device,
        )
        z = member.values["z"][latent_indices]
        age = member.values["age"][latent_indices]
        label = member.values["label"][latent_indices]
        with torch.no_grad():
            latent_velocity = member.flow.average_velocity(z, age, age, label) * float(slope)
        z_for_jvp = z.detach().requires_grad_(True)
        decoded, predicted = torch.autograd.functional.jvp(
            member.geometry.vertices,
            z_for_jvp,
            latent_velocity.detach(),
            create_graph=False,
            strict=False,
        )
        observed = torch.from_numpy(
            np.asarray(reference["velocity_reference_mm_per_year"][ref_indices], dtype=np.float32)
        ).to(device)
        raw_vertices = torch.from_numpy(
            np.asarray(reference["vertices_mm"][ref_indices], dtype=np.float32)
        ).to(device)
        normals = vertex_normals(raw_vertices, faces)
        predicted_normal = torch.sum(predicted * normals, dim=-1)
        observed_normal = torch.sum(observed * normals, dim=-1)
        predicted_flat = predicted.flatten(1)
        observed_flat = observed.flatten(1)
        cosine = torch.sum(predicted_flat * observed_flat, dim=1) / (
            torch.linalg.vector_norm(predicted_flat, dim=1)
            * torch.linalg.vector_norm(observed_flat, dim=1)
        ).clamp_min(1.0e-8)
        pred_centered = predicted_normal - predicted_normal.mean(dim=1, keepdim=True)
        obs_centered = observed_normal - observed_normal.mean(dim=1, keepdim=True)
        correlation = torch.sum(pred_centered * obs_centered, dim=1) / (
            torch.linalg.vector_norm(pred_centered, dim=1)
            * torch.linalg.vector_norm(obs_centered, dim=1)
        ).clamp_min(1.0e-8)
        for local, ref_index in enumerate(ref_indices):
            rows.append(
                {
                    "scan_id": str(reference["scan_ids"][ref_index]),
                    "subject_id": str(reference["subject_ids"][ref_index]),
                    "diagnosis": str(reference["diagnoses"][ref_index]),
                    "age_years": float(reference["age_years"][ref_index]),
                    "reference_reliability": float(reference["velocity_reference_weight"][ref_index]),
                    "vector_rmse_mm_per_year": float(
                        torch.sqrt(torch.mean((predicted[local] - observed[local]).square())).cpu()
                    ),
                    "zero_vector_rmse_mm_per_year": float(
                        torch.sqrt(torch.mean(observed[local].square())).cpu()
                    ),
                    "normal_rmse_mm_per_year": float(
                        torch.sqrt(torch.mean((predicted_normal[local] - observed_normal[local]).square())).cpu()
                    ),
                    "zero_normal_rmse_mm_per_year": float(
                        torch.sqrt(torch.mean(observed_normal[local].square())).cpu()
                    ),
                    "vector_cosine": float(cosine[local].cpu()),
                    "normal_pearson": float(correlation[local].cpu()),
                    "normal_sign_agreement": float(
                        (torch.sign(predicted_normal[local]) == torch.sign(observed_normal[local]))
                        .float()
                        .mean()
                        .cpu()
                    ),
                    "predicted_speed_mm_per_year": float(
                        torch.linalg.vector_norm(predicted[local], dim=-1).mean().cpu()
                    ),
                    "observed_speed_mm_per_year": float(
                        torch.linalg.vector_norm(observed[local], dim=-1).mean().cpu()
                    ),
                    "decoded_source_coordinate_rmse_mm": float(
                        torch.sqrt(torch.mean((decoded[local] - raw_vertices[local]).square())).cpu()
                    ),
                }
            )
    groups = {}
    for diagnosis in ("CN", "AD", "overall"):
        current = rows if diagnosis == "overall" else [r for r in rows if r["diagnosis"] == diagnosis]
        if current:
            groups[diagnosis] = weighted_group(current)
    summary = {
        "schema_version": 1,
        "name": args.name,
        "family": "latent_flow_decoder_jvp",
        "run_dir": str(run_dir),
        "split": args.split,
        "definition": "diagonal latent flow phi(z,a,a,d), converted to years and mapped by decoder JVP",
        "reference": "reliability-weighted velocity fitted from repeated observed visits; not direct physical ground truth",
        "age_norm_per_year": float(slope),
        "age_normalization_linear_fit_max_residual": float(residual),
        "groups": groups,
    }
    write_csv(destination / "per_scan.csv", rows)
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
