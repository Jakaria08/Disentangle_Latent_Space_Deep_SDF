#!/usr/bin/env python3
"""Validation/test evaluator for INR-256 direct C4, plain ODE, and BrainODE."""

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
from inr_geometry import build_geometry
from models import DirectC4Flow, build_ode, transport_rk4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--evaluation-name", default="evaluation")
    parser.add_argument("--surface-resolution", type=int, default=128)
    parser.add_argument("--surface-samples", type=int, default=10000)
    parser.add_argument("--max-surface-pairs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class ODETransport(nn.Module):
    def __init__(self, function: nn.Module, substeps: int) -> None:
        super().__init__()
        self.function, self.substeps = function, int(substeps)

    def transport(self, latent, source_time, target_time, condition, context=None, context_time=None):
        del context, context_time
        return transport_rk4(self.function, latent, source_time, target_time, condition, self.substeps)


def load_transport(config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> nn.Module:
    if config["method"] == "direct_c4":
        model = DirectC4Flow(C.LATENT_DIM, int(config["model"]["width"]), int(config["model"]["residual_blocks"]), float(config["model"].get("dropout", 0.0))).to(device)
        model.load_state_dict(checkpoint["flow_state_dict"], strict=True)
        return model.eval()
    function = build_ode(config).to(device)
    function.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return ODETransport(function.eval(), int(config["training"]["integration_substeps"])).to(device).eval()


def reverse_rows(rows: list[C.PairRow]) -> list[C.PairRow]:
    return [C.PairRow(row.target, row.source, row.intermediate, row.subject, row.diagnosis, f"backward_{row.pair_type}", row.delta_years) for row in rows]


def surface_distances(first, second, samples: int, seed: int) -> dict[str, float]:
    import trimesh

    first_points, _ = trimesh.sample.sample_surface(first, int(samples), seed=seed)
    second_points, _ = trimesh.sample.sample_surface(second, int(samples), seed=seed + 1)
    _, first_to_second, _ = trimesh.proximity.ProximityQuery(second).on_surface(first_points)
    _, second_to_first, _ = trimesh.proximity.ProximityQuery(first).on_surface(second_points)
    return {
        "assd_mm": float(0.5 * (first_to_second.mean() + second_to_first.mean())),
        "hd95_mm": float(max(np.quantile(first_to_second, 0.95), np.quantile(second_to_first, 0.95))),
        "chamfer_l2_squared_mm2": float(np.mean(first_to_second ** 2) + np.mean(second_to_first ** 2)),
    }


@torch.no_grad()
def first_last_surfaces(transport, geometry, values, archive, rows, resolution: int, samples: int, limit: int | None) -> dict[str, Any]:
    import trimesh

    if limit is not None:
        rows = rows[: int(limit)]
    cache = {}
    records = []
    for number, row in enumerate(rows, start=1):
        source = values["z"][row.source:row.source + 1]
        target = values["z"][row.target:row.target + 1]
        source_age = values["age"][row.source:row.source + 1]
        target_age = values["age"][row.target:row.target + 1]
        label = values["label"][row.source:row.source + 1]
        prediction = transport.transport(source, source_age, target_age, label)
        source_key, target_key = ("fit", row.source), ("fit", row.target)
        if source_key not in cache:
            cache[source_key] = geometry.mesh(source, resolution)
        if target_key not in cache:
            cache[target_key] = geometry.mesh(target, resolution)
        predicted = geometry.mesh(prediction, resolution)
        source_mesh, target_fit = cache[source_key], cache[target_key]
        target_true = trimesh.load(str(archive["visit_mesh_path_mm"][row.target]), force="mesh", process=False)
        predicted_metrics = surface_distances(predicted, target_true, samples, 1000 + number)
        nochange_metrics = surface_distances(source_mesh, target_true, samples, 2000 + number)
        floor_metrics = surface_distances(target_fit, target_true, samples, 3000 + number)
        records.append({
            "subject": row.subject,
            "diagnosis": row.diagnosis,
            "source_scan_id": str(archive["visit_scan_ids"][row.source]),
            "target_scan_id": str(archive["visit_scan_ids"][row.target]),
            **{f"predicted_{key}": value for key, value in predicted_metrics.items()},
            **{f"nochange_{key}": value for key, value in nochange_metrics.items()},
            **{f"representation_floor_{key}": value for key, value in floor_metrics.items()},
            "predicted_volume_relative_error": float(abs(predicted.volume - target_true.volume) / max(abs(target_true.volume), 1.0)),
            "nochange_volume_relative_error": float(abs(source_mesh.volume - target_true.volume) / max(abs(target_true.volume), 1.0)),
            "representation_floor_volume_relative_error": float(abs(target_fit.volume - target_true.volume) / max(abs(target_true.volume), 1.0)),
            "latent_bound_excess": float(geometry.bound_excess(prediction)[0].cpu()),
        })
        print(f"surface {number:03d}/{len(rows):03d} {row.subject} {row.diagnosis}", flush=True)
    metrics = [key for key in records[0] if key.startswith(("predicted_", "nochange_", "representation_floor_"))] if records else []
    by_diagnosis = {}
    for diagnosis in ("CN", "AD", "overall"):
        current = records if diagnosis == "overall" else [row for row in records if row["diagnosis"] == diagnosis]
        by_diagnosis[diagnosis] = {f"{key}_mean": float(np.mean([row[key] for row in current])) for key in metrics} | {"pairs": len(current)}
    return {"resolution": int(resolution), "surface_samples": int(samples), "pairs": len(records), "groups": by_diagnosis, "row_metrics": records}


def main() -> int:
    args = parse_args()
    C.validate_run_name(args.evaluation_name)
    run_dir = args.run_dir.expanduser().resolve()
    resolved = C.read_json(run_dir / "resolved_config.json")
    config = resolved["config"]
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.split == "test" and bool(checkpoint.get("test_data_loaded", False)):
        raise ValueError("Checkpoint contract says test was loaded during training")
    registry = C.load_registry()
    train_archive = C.load_archive("train", registry)
    archive = C.load_archive(args.split, registry)
    device = C.choose_device(args.device)
    geometry = build_geometry(train_archive, device, registry)
    transport = load_transport(config, checkpoint, device)
    values = C.values_on_device(archive, device)
    forward = C.load_pairs(args.split, archive, registry)
    first_last = C.first_last_pairs(archive)
    backward = reverse_rows(forward)
    training = dict(config["training"])
    training.setdefault("field_samples_per_pair", 128)
    training.setdefault("decoder_point_chunk", 65536)
    training.setdefault("volume_samples", 512)
    training.setdefault("volume_temperature", 0.01)
    batch_size = int(training.get("evaluation_batch_size", min(16, int(training["batch_size"]))))
    proxy = {
        "all_forward": O.evaluate_pairs(transport, geometry, values, forward, training, batch_size, include_rows=True),
        "all_backward": O.evaluate_pairs(transport, geometry, values, backward, training, batch_size),
        "first_last_forward": O.evaluate_pairs(transport, geometry, values, first_last, training, batch_size),
        "first_last_backward": O.evaluate_pairs(transport, geometry, values, reverse_rows(first_last), training, batch_size),
    }
    scales = checkpoint.get("statistics", {}).get("normalization_scales", {})
    if "displacement" not in scales:
        train = C.load_pairs("train", train_archive, registry)
        displacement = [np.sqrt(np.mean((train_archive["visit_latent_standardized_256"][row.target] - train_archive["visit_latent_standardized_256"][row.source]) ** 2)) for row in train]
        scales["displacement"] = float(np.median(displacement))
    defects = O.cocycle_defects(transport, values, forward, float(scales["displacement"]), batch_size)
    surfaces = first_last_surfaces(transport, geometry, values, archive, first_last, args.surface_resolution, args.surface_samples, args.max_surface_pairs)
    report = {
        "run_dir": str(run_dir), "checkpoint": str(checkpoint_path), "split": args.split,
        "representation": "inr256", "method": config["method"], "proxy_pair_metrics": proxy,
        "first_last_surface_metrics": surfaces, "consistency_defects": defects,
        "test_loaded_during_training": bool(checkpoint.get("test_data_loaded", False)),
        "decoder_checkpoint_sha256": registry["decoder_checkpoint_sha256"],
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    destination = run_dir / args.evaluation_name / args.split
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    C.atomic_json(destination / "summary.json", report)
    print(f"WROTE {destination / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
