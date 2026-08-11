#!/usr/bin/env python3
"""Run v1, v4 and BrainODE through one registered-mesh evaluator.

This closes the previous comparison gap: volume and local-change metrics are
computed from actual PCA predictions for all three models on identical pairs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_v1_anchored_ad_speed import make_model
from v1_speed_utils import (
    EXPERIMENT_DIR,
    SPLITS,
    TASK_DIR,
    PairRecord,
    build_pair_records,
    finite_mean,
    load_config,
    load_pca_model,
    load_split_archive,
    load_v1_flow,
    limit_records_stratified,
    pca_latents,
    prediction_metrics,
    resolve_device,
    resolve_repo_path,
    summarize_prediction_rows,
    vertex_normals_np,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "configs" / "v1_anchored_ad_speed_primary.json"))
    parser.add_argument("--run-name", default="v1_anchor_ad_speed_seed42")
    parser.add_argument("--checkpoint", default="best_feasible_volume")
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["val", "test"])
    parser.add_argument("--max-pairs-per-split", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def v4_checkpoint(run_name: str, value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    candidate = EXPERIMENT_DIR / "runs" / str(run_name) / "checkpoints" / (value if value.endswith(".pth") else f"{value}.pth")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def load_brainode(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    script_dir = TASK_DIR / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from brainode_model import ODEFuncWithAttention

    payload = torch.load(checkpoint, map_location="cpu")
    config = payload["config"]
    model_config = config["model"]
    model = ODEFuncWithAttention(
        latent_dim=int(config["brainode"]["primary_components"]),
        condition_dim=int(model_config["condition_dim"]),
        attention_dim=int(model_config["attention_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, int(config["training"]["integration_substeps"])


@torch.no_grad()
def brainode_prediction(model: torch.nn.Module, substeps: int, source: np.ndarray, record: PairRecord, device: torch.device) -> np.ndarray:
    from brainode_model import integrate_sequence_rk4

    state = torch.from_numpy(source[None, :]).float().to(device)
    times = torch.tensor([[record.source_age_norm, record.target_age_norm]], dtype=torch.float32, device=device)
    condition = torch.tensor([float(record.label_ad)], dtype=torch.float32, device=device)
    prediction = integrate_sequence_rk4(model, state, times, condition, substeps=substeps)
    return prediction[0, -1].cpu().numpy().astype(np.float32)


def summarize_pairwise(rows: list[dict[str, Any]], current: str, baseline: str) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str, int, int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["split"], row["diagnosis"], row["subject_id"], int(row["source_visit_order"]), int(row["target_visit_order"]), row["metric"])
        indexed.setdefault(key, {})[row["model"]] = row
    groups: dict[tuple[str, str, str], list[float]] = {}
    for key, models in indexed.items():
        if current not in models or baseline not in models:
            continue
        group = (key[0], key[1], key[-1])
        groups.setdefault(group, []).append(float(models[baseline]["value"]) - float(models[current]["value"]))
    return [
        {"split": split, "diagnosis": diagnosis, "metric": metric, "rows": len(values), "improvement_mean": finite_mean(values), "beats_baseline_fraction": finite_mean(1.0 if value > 0.0 else 0.0 for value in values), "current_model": current, "baseline_model": baseline}
        for (split, diagnosis, metric), values in groups.items()
    ]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoint = v4_checkpoint(args.run_name, args.checkpoint)
    v4_payload = torch.load(checkpoint, map_location=device)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else Path(v4_payload["metadata_dir"])
    with np.load(metadata_dir / "speed_feature_stats.npz", allow_pickle=False) as archive:
        metadata = {key: archive[key] for key in archive.files}
    v4 = make_model(config=v4_payload.get("config", config), metadata=metadata, device=device)
    v4.load_state_dict(v4_payload["model_state_dict"])
    v4.eval()
    v1, _ = load_v1_flow(resolve_repo_path(config["base_v1_checkpoint"]), device)
    brainode, substeps = load_brainode(resolve_repo_path(config["brainode_checkpoint"]), device)
    _, mean_flat, components, faces = load_pca_model(int(config["components"]))
    tensors = np.load(resolve_repo_path(config["registered_mesh_tensors"]), allow_pickle=False)
    normals = tensors["template_vertex_normals"].astype(np.float32) if "template_vertex_normals" in tensors.files else vertex_normals_np(np.zeros((mean_flat.size // 3, 3), dtype=np.float32), faces)
    output_dir = Path(args.output_dir) if args.output_dir else EXPERIMENT_DIR / "runs" / str(args.run_name) / "analysis" / f"checkpoint_{checkpoint.stem}" / "unified_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    metrics = ("endpoint_vertex_mae", "endpoint_pca_mse", "volume_relative_error", "log_volume_rate_abs_error", "local_normal_top_change_dice")
    for split in args.splits:
        archive = load_split_archive(split)
        latents = pca_latents(archive, int(config["components"]))
        records = build_pair_records(archive)
        if args.max_pairs_per_split > 0:
            records = limit_records_stratified(records, int(args.max_pairs_per_split))
        for record in records:
            source, target = latents[record.source_index], latents[record.target_index]
            with torch.no_grad():
                source_t = torch.from_numpy(source[None, :]).float().to(device)
                v1_prediction = v1.transport(source_t, torch.tensor([record.source_age_norm], device=device), torch.tensor([record.target_age_norm], device=device), torch.tensor([float(record.label_ad)], device=device))[0].cpu().numpy().astype(np.float32)
                v4_prediction, diagnostics = v4.transport(source_t, torch.tensor([record.source_age_norm], device=device), torch.tensor([record.target_age_norm], device=device), torch.tensor([record.source_age_years], device=device), torch.tensor([record.target_age_years], device=device), torch.tensor([float(record.label_ad)], device=device))
                v4_prediction_np = v4_prediction[0].cpu().numpy().astype(np.float32)
                speed = float(diagnostics["speed"][0].item())
            brain_prediction = brainode_prediction(brainode, substeps, source, record, device)
            models = (("v1", "full PCA direct cocycle flow", v1_prediction, 1.0), ("v4", "v1-anchored AD speed-calibrated flow", v4_prediction_np, speed), ("brainode", "attention neural ODE", brain_prediction, float("nan")))
            per_model: dict[str, dict[str, Any]] = {}
            for name, family, prediction, ad_speed in models:
                row = prediction_metrics(record=record, split=split, transport_method="direct", checkpoint=checkpoint.stem if name == "v4" else name, model_name=name, family=family, source=source, target=target, predicted=prediction, mean_flat=mean_flat, components=components, faces=faces, template_normals=normals, top_fraction=float(config["evaluation"]["top_change_fraction"]), ad_speed=ad_speed)
                prediction_rows.append(row)
                per_model[name] = row
            for metric in metrics:
                for name in ("v1", "v4", "brainode"):
                    comparison_rows.append({"split": split, "diagnosis": record.diagnosis, "subject_id": record.subject_id, "source_visit_order": record.source_visit_order, "target_visit_order": record.target_visit_order, "metric": metric, "model": name, "value": per_model[name][metric]})
    summaries: list[dict[str, Any]] = []
    for model_name in ("v1", "v4", "brainode"):
        for row in summarize_prediction_rows([item for item in prediction_rows if item["model"] == model_name]):
            row["model"] = model_name
            summaries.append(row)
    comparison_summary = summarize_pairwise(comparison_rows, "v4", "brainode") + summarize_pairwise(comparison_rows, "v4", "v1")
    write_csv(output_dir / "unified_registered_per_pair.csv", prediction_rows)
    write_csv(output_dir / "unified_registered_summary.csv", summaries)
    write_csv(output_dir / "unified_comparison_long.csv", comparison_rows)
    write_csv(output_dir / "unified_comparison_summary.csv", comparison_summary)
    write_json(output_dir / "run.json", {"v4_checkpoint": str(checkpoint), "brainode_checkpoint": str(resolve_repo_path(config["brainode_checkpoint"])), "v1_checkpoint": str(resolve_repo_path(config["base_v1_checkpoint"])), "splits": args.splits, "rows": len(prediction_rows), "device": str(device)})
    print(f"Wrote unified evaluation to {output_dir}; prediction rows={len(prediction_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
