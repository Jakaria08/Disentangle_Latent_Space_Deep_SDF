#!/usr/bin/env python3
"""Read-only evaluation for independently trained direct Cocycle-V5 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_adni_synthseg_pca_cocycle_v4 import choose_device, load_archive, read_json, validate_pca_model
from train_adni_synthseg_pca_cocycle_v5 import (
    BASE_ROOT,
    EXPERIMENTS,
    CocyclePair,
    DirectDiagnosisResidualCocycleFlow,
    PcaGeometry,
    cocycle_defects,
    convert_pairs,
    default_config_path,
    first_last_pairs,
    indexed,
    resolve_path,
    training_statistics,
    values_on_device,
)
from train_adni_synthseg_pca_cocycle_v4 import load_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, choices=("hippocampus", "lateral_ventricle"))
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="c3")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test", "all"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and planned checkpoint/output paths without writing.")
    return parser.parse_args()


def output_directory(config_path: Path, run_name: str) -> Path:
    return config_path.parent.parent / "training" / run_name / "evaluation" / "cocycle_v5"


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def per_pair_metrics(
    *,
    flow: DirectDiagnosisResidualCocycleFlow,
    geometry: PcaGeometry,
    values: dict[str, torch.Tensor],
    archive: dict[str, np.ndarray],
    rows: list[CocyclePair],
    split: str,
    batch_size: int,
    progress_every_batches: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    batches = max(1, (len(rows) + batch_size - 1) // batch_size)
    for batch_index, start in enumerate(range(0, len(rows), batch_size), start=1):
        chunk = rows[start : start + batch_size]
        raw = {
            "source": torch.tensor([row.source for row in chunk], dtype=torch.long),
            "target": torch.tensor([row.target for row in chunk], dtype=torch.long),
            "intermediate": torch.tensor([row.intermediate for row in chunk], dtype=torch.long),
        }
        batch = indexed(values, raw)
        forward = flow.transport(batch["source"], batch["source_age"], batch["target_age"], batch["label"])
        backward = flow.transport(batch["target"], batch["target_age"], batch["source_age"], batch["label"])
        source_vertices = geometry.vertices(batch["source"])
        target_vertices = geometry.vertices(batch["target"])
        forward_vertices = geometry.vertices(forward)
        backward_vertices = geometry.vertices(backward)
        source_volume = geometry.volume_from_vertices(source_vertices)
        target_volume = geometry.volume_from_vertices(target_vertices)
        forward_volume = geometry.volume_from_vertices(forward_vertices)
        backward_volume = geometry.volume_from_vertices(backward_vertices)
        years = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
        forward_values = {
            "pca_mse": torch.mean((forward - batch["target"]).square(), dim=1),
            "vertex_coordinate_mae_mm": torch.mean(torch.abs(forward_vertices - target_vertices), dim=(1, 2)),
            "vertex_euclidean_mean_mm": torch.linalg.vector_norm(forward_vertices - target_vertices, dim=2).mean(dim=1),
            "volume_relative_error": torch.abs(forward_volume - target_volume) / target_volume,
            "log_volume_rate_abs_error": torch.abs((torch.log(forward_volume) - torch.log(target_volume)) / years),
        }
        backward_values = {
            "backward_pca_mse": torch.mean((backward - batch["source"]).square(), dim=1),
            "backward_vertex_coordinate_mae_mm": torch.mean(torch.abs(backward_vertices - source_vertices), dim=(1, 2)),
            "backward_vertex_euclidean_mean_mm": torch.linalg.vector_norm(backward_vertices - source_vertices, dim=2).mean(dim=1),
            "backward_volume_relative_error": torch.abs(backward_volume - source_volume) / source_volume,
            "backward_log_volume_rate_abs_error": torch.abs((torch.log(backward_volume) - torch.log(source_volume)) / years),
        }
        no_change = {
            "nochange_pca_mse": torch.mean((batch["source"] - batch["target"]).square(), dim=1),
            "nochange_vertex_coordinate_mae_mm": torch.mean(torch.abs(source_vertices - target_vertices), dim=(1, 2)),
            "nochange_vertex_euclidean_mean_mm": torch.linalg.vector_norm(source_vertices - target_vertices, dim=2).mean(dim=1),
            "nochange_volume_relative_error": torch.abs(source_volume - target_volume) / target_volume,
            "nochange_log_volume_rate_abs_error": torch.abs((torch.log(source_volume) - torch.log(target_volume)) / years),
        }
        middle_age = 0.5 * (batch["source_age"] + batch["target_age"])
        middle = flow.transport(batch["source"], batch["source_age"], middle_age, batch["label"])
        composed = flow.transport(middle, middle_age, batch["target_age"], batch["label"])
        inverse = flow.transport(forward, batch["target_age"], batch["source_age"], batch["label"])
        semi = torch.sqrt(torch.mean((forward - composed).square(), dim=1))
        inverse_error = torch.sqrt(torch.mean((inverse - batch["source"]).square(), dim=1))
        for index, row in enumerate(chunk):
            record: dict[str, Any] = {
                "split": split,
                "subject_id": row.subject,
                "diagnosis": row.diagnosis,
                "pair_type": row.pair_type,
                "source_scan_id": str(archive["visit_scan_ids"][row.source]),
                "target_scan_id": str(archive["visit_scan_ids"][row.target]),
                "source_age": float(batch["source_age"][index].cpu()),
                "target_age": float(batch["target_age"][index].cpu()),
                "delta_years": float(years[index].cpu()),
                "semigroup_defect_pca_rmse": float(semi[index].cpu()),
                "inverse_defect_pca_rmse": float(inverse_error[index].cpu()),
            }
            for values_dict in (forward_values, backward_values, no_change):
                record.update({name: float(value[index].cpu()) for name, value in values_dict.items()})
            records.append(record)
        if batch_index % max(1, progress_every_batches) == 0 or batch_index == batches:
            print(f"{split}: evaluated {batch_index}/{batches} batches ({len(records)}/{len(rows)} pairs)", flush=True)
    return records


def summarize(records: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    metrics = (
        "pca_mse", "vertex_coordinate_mae_mm", "vertex_euclidean_mean_mm", "volume_relative_error", "log_volume_rate_abs_error",
        "backward_pca_mse", "backward_vertex_coordinate_mae_mm", "backward_vertex_euclidean_mean_mm", "backward_volume_relative_error", "backward_log_volume_rate_abs_error",
        "nochange_pca_mse", "nochange_vertex_coordinate_mae_mm", "nochange_vertex_euclidean_mean_mm", "nochange_volume_relative_error", "nochange_log_volume_rate_abs_error",
        "semigroup_defect_pca_rmse", "inverse_defect_pca_rmse",
    )
    output: list[dict[str, Any]] = []
    splits = sorted({str(row["split"]) for row in records})
    for split in splits:
        for diagnosis in ("CN", "AD", "ALL"):
            selected = [row for row in records if row["split"] == split and (diagnosis == "ALL" or row["diagnosis"] == diagnosis)]
            if not selected:
                continue
            output.append({
                "protocol": protocol,
                "split": split,
                "diagnosis": diagnosis,
                "rows": len(selected),
                **{f"{metric}_mean": mean_or_nan([float(row[metric]) for row in selected]) for metric in metrics},
            })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config or default_config_path(args.structure, args.experiment))
    config = read_json(config_path)
    run_name = str(args.run_name or config["training"]["run_name"])
    run_dir = config_path.parent.parent / "training" / run_name
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best_shape.pt"
    output_dir = resolve_path(args.output_dir) if args.output_dir else output_directory(config_path, run_name)
    requested_splits = ("val", "test") if args.split == "all" else (args.split,)
    if args.dry_run:
        print(json.dumps({
            "config": str(config_path), "checkpoint": str(checkpoint_path), "output_dir": str(output_dir),
            "splits": requested_splits, "writes_files": False,
        }, indent=2), flush=True)
        return 0
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing selected Cocycle-V5 checkpoint: {checkpoint_path}")
    if output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_dir}. Use --output-dir for a new report.")

    device = choose_device(args.device)
    input_config = read_json(resolve_path(config["input_config"]))
    train_archive = load_archive(resolve_path(input_config["dataset"]["train_sequences"]), "train", 150)
    pca_model = validate_pca_model(input_config, 150)
    geometry = PcaGeometry(pca_model, train_archive["train_pca_mean_150"], train_archive["train_pca_std_150"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    flow = DirectDiagnosisResidualCocycleFlow(
        latent_dim=int(config["model"]["latent_dim"]), width=int(config["model"]["width"]), residual_blocks=int(config["model"]["residual_blocks"]),
    ).to(device)
    flow.load_state_dict(checkpoint["flow_state_dict"])
    flow.eval()
    geometry.eval()
    statistics = checkpoint.get("statistics")
    if not isinstance(statistics, dict):
        train_values = values_on_device(train_archive, device)
        train_pairs = convert_pairs(load_pairs(resolve_path(input_config["dataset"]["train_pairs"]), train_archive, "train"), train_archive)
        statistics = training_statistics(geometry, train_values, train_pairs, train_archive)

    all_records: list[dict[str, Any]] = []
    first_last_records: list[dict[str, Any]] = []
    defect_report: dict[str, Any] = {}
    for split in requested_splits:
        archive = load_archive(resolve_path(input_config["dataset"][f"{split}_sequences"]), split, 150)
        values = values_on_device(archive, device)
        pairs = convert_pairs(load_pairs(resolve_path(input_config["dataset"][f"{split}_pairs"]), archive, split), archive)
        print(f"Evaluating {args.structure} {args.experiment}: {split} all observed pairs ({len(pairs)})", flush=True)
        all_records.extend(per_pair_metrics(flow=flow, geometry=geometry, values=values, archive=archive, rows=pairs, split=split, batch_size=512, progress_every_batches=args.progress_every_batches))
        first_pairs = first_last_pairs(archive)
        print(f"Evaluating {split} first-to-last pairs ({len(first_pairs)})", flush=True)
        first_last_records.extend(per_pair_metrics(flow=flow, geometry=geometry, values=values, archive=archive, rows=first_pairs, split=split, batch_size=512, progress_every_batches=args.progress_every_batches))
        defect_report[split] = cocycle_defects(flow, values, pairs, statistics, 512)

    summary_rows = summarize(all_records, "all_observed_pairs") + summarize(first_last_records, "first_to_last")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "per_pair_metrics.csv", all_records)
    write_csv(output_dir / "first_last_metrics.csv", first_last_records)
    write_csv(output_dir / "summary.csv", summary_rows)
    with (output_dir / "cocycle_defects.json").open("w", encoding="utf-8") as handle:
        json.dump(defect_report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# Direct Cocycle-V5 evaluation\n\n")
        handle.write("This is a read-only evaluation of the selected `best_shape.pt` checkpoint. ")
        handle.write("It reports direct forward/backward observed-pair performance, first-to-last performance, and virtual-midpoint semigroup/inverse defects.\n")
    print("=" * 96, flush=True)
    print(f"Evaluation complete: {output_dir}", flush=True)
    for row in summary_rows:
        if row["diagnosis"] == "ALL":
            print(f"{row['split']} {row['protocol']}: vertex={row['vertex_coordinate_mae_mm_mean']:.6f} mm volume={row['volume_relative_error_mean']:.6f} semigroup={row['semigroup_defect_pca_rmse_mean']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
