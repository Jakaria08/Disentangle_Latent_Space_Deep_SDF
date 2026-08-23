#!/usr/bin/env python3
"""Unified validation/test evaluator for all 128-D transport methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import c4_objective as O
import common as C
from models import DirectC4Flow, build_ode, transport_rk4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--evaluation-name",
        default="evaluation",
        help="Safe output-directory name; use a new name to preserve an earlier evaluation.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write the report.")
    return parser.parse_args()


class ODETransport(nn.Module):
    def __init__(self, function: nn.Module, substeps: int) -> None:
        super().__init__()
        self.function = function
        self.substeps = int(substeps)

    def transport(self, latent, source_time, target_time, condition, context=None, context_time=None):
        del context, context_time
        return transport_rk4(self.function, latent, source_time, target_time, condition, self.substeps)


def load_transport(config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> nn.Module:
    if config["method"] in {"plain_ode", "brainode"}:
        function = build_ode(config).to(device)
        function.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return ODETransport(function, int(config["training"]["integration_substeps"])).to(device).eval()
    if config["method"] == "direct_c4":
        flow = DirectC4Flow(
            C.LATENT_DIM,
            int(config["model"]["width"]),
            int(config["model"]["residual_blocks"]),
            float(config["model"].get("dropout", 0.0)),
        ).to(device)
        flow.load_state_dict(checkpoint["flow_state_dict"], strict=True)
        return flow.eval()
    raise ValueError(f"Unknown method {config['method']}")


def reverse_rows(rows: list[C.PairRow]) -> list[C.PairRow]:
    return [
        C.PairRow(row.target, row.source, row.intermediate, row.subject, row.diagnosis, f"backward_{row.pair_type}", row.delta_years)
        for row in rows
    ]


def limited_pairs(rows: list[C.PairRow], limit: int | None, first_last: bool = False) -> list[C.PairRow]:
    if first_last:
        return O.balanced_first_last_subset(rows, limit)
    return O.balanced_subset(rows, limit)


def horizon_groups(rows: list[C.PairRow]) -> dict[str, list[C.PairRow]]:
    return {
        "le_1y": [row for row in rows if row.delta_years <= 1.0],
        "gt_1_le_2y": [row for row in rows if 1.0 < row.delta_years <= 2.0],
        "gt_2y": [row for row in rows if row.delta_years > 2.0],
    }


def representation_floor(
    values: dict[str, torch.Tensor],
    raw_vertices_mm: np.ndarray,
    geometry: C.FrozenGeometry,
    batch_size: int,
) -> dict[str, float]:
    rmse, mae, euclidean, volume = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(values["z"]), batch_size):
            decoded = values["reference_vertices"][start : start + batch_size]
            raw = torch.from_numpy(
                np.asarray(raw_vertices_mm[start : start + batch_size], dtype=np.float32).copy()
            ).to(decoded.device)
            delta = decoded - raw
            decoded_volume = geometry.volume_from_vertices(decoded)
            raw_volume = geometry.volume_from_vertices(raw)
            rmse.extend(torch.sqrt(torch.mean(delta.square(), dim=(1, 2))).cpu().tolist())
            mae.extend(torch.mean(torch.abs(delta), dim=(1, 2)).cpu().tolist())
            euclidean.extend(torch.linalg.vector_norm(delta, dim=2).mean(dim=1).cpu().tolist())
            volume.extend((torch.abs(decoded_volume - raw_volume) / raw_volume).cpu().tolist())
    return {
        "scans": len(rmse),
        "coordinate_rmse_mm_mean": float(np.mean(rmse)),
        "coordinate_mae_mm_mean": float(np.mean(mae)),
        "vertex_euclidean_mm_mean": float(np.mean(euclidean)),
        "volume_relative_error_mean": float(np.mean(volume)),
    }


@torch.no_grad()
def sequence_metrics(
    transport: nn.Module,
    geometry: C.FrozenGeometry,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
    raw_vertices_mm: np.ndarray,
    max_subjects: int | None,
) -> dict[str, Any]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    records = []
    subject_count = len(offsets) - 1 if max_subjects is None else min(len(offsets) - 1, max_subjects)
    for subject_index in range(subject_count):
        first, last = int(offsets[subject_index]), int(offsets[subject_index + 1])
        z = values["z"][first:last]
        age = values["age"][first:last]
        label = values["label"][first : first + 1]
        count = len(z) - 1
        direct = transport.transport(z[:1].expand(count, -1), age[:1].expand(count), age[1:], label.expand(count))
        rollout = []
        current = z[:1]
        for index in range(1, len(z)):
            current = transport.transport(current, age[index - 1 : index], age[index : index + 1], label)
            rollout.append(current)
        rollout = torch.cat(rollout)
        target_vertices = values["reference_vertices"][first + 1 : last]
        direct_vertices = geometry.vertices(direct)
        rollout_vertices = geometry.vertices(rollout)
        raw_target = torch.from_numpy(np.asarray(raw_vertices_mm[first + 1 : last], dtype=np.float32).copy()).to(z.device)
        reverse_ages = age[:-1].flip(0)
        direct_backward = transport.transport(
            z[-1:].expand(count, -1), age[-1:].expand(count), reverse_ages, label.expand(count)
        )
        rollout_backward = []
        current_backward = z[-1:]
        previous_age = age[-1:]
        for index in range(len(z) - 2, -1, -1):
            current_backward = transport.transport(
                current_backward, previous_age, age[index : index + 1], label
            )
            rollout_backward.append(current_backward)
            previous_age = age[index : index + 1]
        rollout_backward = torch.cat(rollout_backward)
        reverse_target_vertices = values["reference_vertices"][first : last - 1].flip(0)
        direct_backward_vertices = geometry.vertices(direct_backward)
        rollout_backward_vertices = geometry.vertices(rollout_backward)
        raw_reverse_target = torch.from_numpy(
            np.asarray(raw_vertices_mm[first : last - 1][::-1], dtype=np.float32).copy()
        ).to(z.device)
        records.append({
            "subject": str(archive["subject_ids"][subject_index]),
            "diagnosis": str(archive["subject_diagnoses"][subject_index]),
            "visits": int(len(z)),
            "forward_direct_transport_coordinate_mae": float(torch.mean(torch.abs(direct_vertices - target_vertices)).cpu()),
            "forward_rollout_transport_coordinate_mae": float(torch.mean(torch.abs(rollout_vertices - target_vertices)).cpu()),
            "forward_direct_end_to_end_coordinate_rmse": float(torch.sqrt(torch.mean((direct_vertices - raw_target).square())).cpu()),
            "forward_rollout_end_to_end_coordinate_rmse": float(torch.sqrt(torch.mean((rollout_vertices - raw_target).square())).cpu()),
            "forward_direct_rollout_latent_rmse": float(torch.sqrt(torch.mean((direct - rollout).square())).cpu()),
            "backward_direct_transport_coordinate_mae": float(torch.mean(torch.abs(direct_backward_vertices - reverse_target_vertices)).cpu()),
            "backward_rollout_transport_coordinate_mae": float(torch.mean(torch.abs(rollout_backward_vertices - reverse_target_vertices)).cpu()),
            "backward_direct_end_to_end_coordinate_rmse": float(torch.sqrt(torch.mean((direct_backward_vertices - raw_reverse_target).square())).cpu()),
            "backward_rollout_end_to_end_coordinate_rmse": float(torch.sqrt(torch.mean((rollout_backward_vertices - raw_reverse_target).square())).cpu()),
            "backward_direct_rollout_latent_rmse": float(torch.sqrt(torch.mean((direct_backward - rollout_backward).square())).cpu()),
        })
    metrics = [key for key in records[0] if key.startswith(("forward_", "backward_"))] if records else []
    return {
        "subjects": len(records),
        "means": {key: float(np.mean([row[key] for row in records])) for key in metrics},
        "by_diagnosis": {
            diagnosis: {
                key: float(np.mean([row[key] for row in records if row["diagnosis"] == diagnosis]))
                for key in metrics
            }
            for diagnosis in ("CN", "AD")
        },
        "subject_metrics": records,
    }


def subject_bootstrap(row_metrics: list[dict[str, Any]], samples: int, seed: int = 12345) -> dict[str, Any]:
    if samples <= 0 or not row_metrics:
        return {"samples": 0}
    keys = ("coordinate", "euclidean", "end_to_end_coordinate_rmse", "volume_relative")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in row_metrics:
        grouped.setdefault(str(row["subject"]), []).append(row)
    subjects = sorted(grouped)
    subject_values = {
        subject: {key: float(np.mean([row[key] for row in rows])) for key in keys}
        for subject, rows in grouped.items()
    }
    generator = np.random.default_rng(seed)
    draws = {key: [] for key in keys}
    for _ in range(samples):
        chosen = generator.choice(subjects, size=len(subjects), replace=True)
        for key in keys:
            draws[key].append(float(np.mean([subject_values[str(subject)][key] for subject in chosen])))
    return {
        "samples": samples,
        "unit": "subject",
        "metrics": {
            key: {
                "mean": float(np.mean([subject_values[subject][key] for subject in subjects])),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
            for key, values in draws.items()
        },
    }


def main() -> int:
    args = parse_args()
    C.validate_run_name(args.evaluation_name)
    run_dir = args.run_dir.expanduser().resolve()
    resolved = C.read_json(run_dir / "resolved_config.json")
    config = resolved["config"]
    representation = str(config["representation"])
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = C.choose_device(args.device)
    registry = C.load_registry()
    train_archive = C.load_archive(representation, "train", registry)
    archive = C.load_archive(representation, args.split, registry)
    geometry = C.build_geometry(representation, train_archive, device, registry)
    values = C.values_on_device(archive, device)
    O.attach_reference_geometry(values, geometry, int(config["training"].get("decoder_batch_size", 64)))
    raw_vertices = C.cached_vertices(args.split, registry)
    transport = load_transport(config, checkpoint, device)
    forward = C.load_pairs(args.split, archive, registry)
    first_last = C.first_last_pairs(archive)
    forward = limited_pairs(forward, args.max_pairs)
    first_last = limited_pairs(first_last, args.max_pairs, first_last=True)
    backward = reverse_rows(forward)
    first_last_backward = reverse_rows(first_last)
    batch_size = int(config["training"].get("evaluation_batch_size", config["training"].get("batch_size", 64)))

    categories = {
        "all_forward": forward,
        "adjacent_forward": [row for row in forward if row.pair_type == "adjacent"],
        "nonadjacent_forward": [row for row in forward if row.pair_type == "nonadjacent"],
        "first_last_forward": first_last,
        "all_backward": backward,
        "adjacent_backward": [row for row in backward if row.pair_type.endswith("adjacent") and "nonadjacent" not in row.pair_type],
        "nonadjacent_backward": [row for row in backward if row.pair_type.endswith("nonadjacent")],
        "first_last_backward": first_last_backward,
    }
    categories.update({f"horizon_{name}_forward": rows for name, rows in horizon_groups(forward).items() if rows})
    categories.update({f"horizon_{name}_backward": rows for name, rows in horizon_groups(backward).items() if rows})
    pair_metrics = {
        name: O.evaluate_pairs(transport, geometry, values, rows, raw_vertices, batch_size, include_rows=(name == "all_forward"))
        for name, rows in categories.items()
        if rows
    }
    train_values = C.values_on_device(train_archive, device)
    train_pairs = C.load_pairs("train", train_archive, registry)
    displacement = np.asarray([
        np.sqrt(np.mean((train_archive["visit_latent_standardized_128"][row.target] - train_archive["visit_latent_standardized_128"][row.source]) ** 2))
        for row in train_pairs
    ])
    defect_statistics = checkpoint.get("statistics", {"normalization_scales": {"displacement": float(max(np.median(displacement), 1.0e-6))}})
    defects = O.cocycle_defects(transport, values, forward, defect_statistics, batch_size)
    sequences = sequence_metrics(transport, geometry, values, archive, raw_vertices, args.max_subjects)
    floor = representation_floor(values, raw_vertices, geometry, batch_size)
    bootstrap = subject_bootstrap(pair_metrics["all_forward"].get("row_metrics", []), args.bootstrap_samples)
    report = {
        "run_dir": str(run_dir),
        "evaluation_name": args.evaluation_name,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "representation": representation,
        "method": config["method"],
        "representation_floor": floor,
        "pair_metrics": pair_metrics,
        "sequence_metrics": sequences,
        "consistency_defects": defects,
        "subject_bootstrap": bootstrap,
        "test_loaded_during_training": bool(checkpoint.get("test_data_loaded", False)),
        "source_meshes_modified": False,
    }
    C.assert_finite_mapping({key: value for key, value in report.items() if key not in {"pair_metrics", "sequence_metrics"}})
    if args.dry_run:
        print("EVALUATION DRY RUN PASSED — no files written.")
        print(json.dumps({key: value for key, value in report.items() if key != "pair_metrics"}, indent=2, sort_keys=True))
        return 0
    destination = run_dir / args.evaluation_name / args.split
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    C.atomic_json(destination / "summary.json", report)
    print(f"WROTE {destination / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
