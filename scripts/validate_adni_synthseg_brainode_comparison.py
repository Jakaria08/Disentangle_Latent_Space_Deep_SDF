#!/usr/bin/env python3
"""Validate BrainODE against the current strict SynthSeg longitudinal models.

This is a read-only model evaluator.  It extends the established independent
validator without changing its cohort, PCA decoder, pair definitions, or metric
implementations.  BrainODE is evaluated one case at a time, matching the prior
BrainODE evaluation protocol and preventing test predictions from depending on
the arbitrary composition of an evaluation batch.  A separate audit reports
how much the inherited cross-batch attention changes predictions in a group.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import validate_adni_synthseg_longitudinal_models as common
from train_adni_synthseg_pca_brainode import (
    ODEFuncWithAttention,
    PRIOR_BRAINODE_MODEL,
    integrate_sequence_rk4,
)
from train_adni_synthseg_pca_cocycle_v4 import BASE_ROOT, choose_device, read_json


STRUCTURE_RUNS = {
    structure: {**specification, "brainode": "brainode_attention_pca150_s42"}
    for structure, specification in common.STRUCTURE_RUNS.items()
}
METHOD_ORDER = [
    "no_change",
    "v1",
    "anchored_v4",
    "e3_selected",
    "brainode",
    "e3_latest_sensitivity",
]
METHOD_LABELS = {
    **common.METHOD_LABELS,
    "brainode": "BrainODE",
}
PRIMARY_METHODS = {*common.PRIMARY_METHODS, "brainode"}

# The reused reporting functions resolve these module-level constants at run
# time.  Set them once so all tables and figures use the extended method list.
common.STRUCTURE_RUNS = STRUCTURE_RUNS
common.METHOD_ORDER = METHOD_ORDER
common.METHOD_LABELS = METHOD_LABELS
common.PRIMARY_METHODS = PRIMARY_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--context-audit-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_ROOT / "longitudinal_model_validation" / "brainode_comparison_s42",
    )
    return parser.parse_args()


class BrainODEAdapter:
    """Transport adapter using independent, batch-size-one RK4 inference."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        checkpoint: Path,
        integration_substeps: int,
    ) -> None:
        self.name = "brainode"
        self.model = model
        self.device = device
        self.checkpoint = checkpoint
        self.integration_substeps = int(integration_substeps)
        self.model.eval()

    @torch.no_grad()
    def transport(
        self,
        z: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        source_years: torch.Tensor,
        target_years: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        del source_years, target_years
        predictions = []
        for index in range(z.shape[0]):
            times = torch.stack(
                (source_time[index : index + 1], target_time[index : index + 1]),
                dim=1,
            )
            trajectory = integrate_sequence_rk4(
                self.model,
                z[index : index + 1],
                times,
                label[index : index + 1],
                substeps=self.integration_substeps,
            )
            predictions.append(trajectory[:, -1, :])
        return torch.cat(predictions, dim=0)


def resolved(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = common.REPO_ROOT / path
    return path.resolve()


def load_brainode(
    structure: str,
    specification: dict[str, str],
    device: torch.device,
    reference_config: dict[str, Any],
) -> tuple[BrainODEAdapter, dict[str, Any], list[dict[str, Any]]]:
    run_dir = (
        BASE_ROOT
        / specification["folder"]
        / "brainode"
        / "training"
        / specification["brainode"]
    )
    checkpoint = run_dir / "checkpoints" / "best.pt"
    required = [
        checkpoint,
        run_dir / "resolved_config.json",
        run_dir / "final_report.json",
        run_dir / "history.jsonl",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    resolved_run = read_json(run_dir / "resolved_config.json")
    config = resolved_run["config"]
    input_config = resolved_run["input_config"]
    final_report = read_json(run_dir / "final_report.json")
    history = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if final_report.get("status") != "complete" or len(history) != int(final_report["epochs_completed"]):
        raise ValueError(f"Incomplete BrainODE run for {structure}")
    expected_structure = {
        "hippocampus": "left_hippocampus",
        "lateral_ventricle": "left_lateral_ventricle",
    }[structure]
    if config.get("structure") != expected_structure or final_report.get("structure") != expected_structure:
        raise ValueError(f"BrainODE structure mismatch for {structure}")
    scientific = config.get("scientific_contract", {})
    if not scientific.get("strict_no_mci") or scientific.get("test_loaded_during_training") is not False:
        raise ValueError(f"BrainODE training contract failed for {structure}")
    if int(config["representation"]["components"]) != 150:
        raise ValueError("BrainODE comparison requires PCA-150")

    # BrainODE must point at exactly the archives and PCA model used by the
    # other models; comparing merely similar cohorts is not sufficient.
    for split in ("train", "val", "test"):
        key = f"{split}_sequences"
        if resolved(input_config["dataset"][key]) != resolved(reference_config["dataset"][key]):
            raise ValueError(f"BrainODE/reference {key} mismatch for {structure}")
    for key in ("pca_model_dir", "pca_archive"):
        if resolved(input_config["representation"][key]) != resolved(reference_config["representation"][key]):
            raise ValueError(f"BrainODE/reference {key} mismatch for {structure}")

    payload = torch.load(checkpoint, map_location=device)
    common.finite_state_dict(payload["model_state_dict"], str(checkpoint))
    if payload.get("test_data_loaded") is not False:
        raise ValueError(f"BrainODE checkpoint reports test access for {structure}")
    actual_source_hash = common.file_sha256(PRIOR_BRAINODE_MODEL)
    if payload.get("brainode_model_source_sha256") != actual_source_hash:
        raise ValueError(f"BrainODE source hash mismatch for {structure}")

    metric = "val_first_last_endpoint_vertex_mae_mm"
    expected_best = min(history, key=lambda row: float(row[metric]))
    checkpoint_epoch = int(payload["epoch"])
    if checkpoint_epoch != int(expected_best["epoch"]) or checkpoint_epoch != int(final_report["best_epoch"]):
        raise ValueError(f"BrainODE best-checkpoint selection mismatch for {structure}")
    if not math.isclose(
        float(payload["best_metric_value"]),
        float(expected_best[metric]),
        rel_tol=1.0e-7,
        abs_tol=1.0e-10,
    ):
        raise ValueError(f"BrainODE best metric mismatch for {structure}")

    model = ODEFuncWithAttention(
        latent_dim=150,
        condition_dim=int(config["model"]["condition_dim"]),
        attention_dim=int(config["model"]["attention_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    adapter = BrainODEAdapter(
        model,
        device,
        checkpoint,
        int(config["training"]["integration_substeps"]),
    )
    report = {
        "status": "pass",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": common.file_sha256(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "validation_selection_metric": metric,
        "validation_selection_value": float(expected_best[metric]),
        "history_epochs": len(history),
        "strict_no_mci": True,
        "test_loaded_during_training": False,
        "same_archives_as_comparison_models": True,
        "same_pca_model_as_comparison_models": True,
        "model_source": str(PRIOR_BRAINODE_MODEL),
        "model_source_sha256": actual_source_hash,
        "primary_inference": "one case at a time, matching prior BrainODE evaluator",
        "cross_batch_attention_present": True,
    }
    return adapter, report, history


@torch.no_grad()
def batch_context_audit(
    adapter: BrainODEAdapter,
    archive: dict[str, np.ndarray],
    pairs: list[Any],
    geometry: common.Geometry,
    device: torch.device,
    sample_size: int,
) -> dict[str, Any]:
    count = min(int(sample_size), len(pairs))
    current = pairs[:count]
    z_all = torch.from_numpy(archive["visit_pca_standardized_150"].astype(np.float32)).to(device)
    time_all = torch.from_numpy(archive["visit_age_norm_train"].astype(np.float32)).to(device)
    label_all = torch.from_numpy(archive["visit_label_ad"].astype(np.float32)).to(device)
    source_index = torch.tensor([pair.source_index for pair in current], dtype=torch.long, device=device)
    target_index = torch.tensor([pair.target_index for pair in current], dtype=torch.long, device=device)
    source = z_all[source_index]
    source_time = time_all[source_index]
    target_time = time_all[target_index]
    label = label_all[source_index]
    zeros = torch.zeros_like(source_time)
    independent = adapter.transport(source, source_time, target_time, zeros, zeros, label)
    grouped_times = torch.stack((source_time, target_time), dim=1)
    grouped = integrate_sequence_rk4(
        adapter.model,
        source,
        grouped_times,
        label,
        substeps=adapter.integration_substeps,
    )[:, -1, :]
    independent_np = independent.cpu().numpy()
    grouped_np = grouped.cpu().numpy()
    independent_vertices = geometry.vertices(independent_np)
    grouped_vertices = geometry.vertices(grouped_np)
    euclidean = np.linalg.norm(grouped_vertices - independent_vertices, axis=2)
    return {
        "samples": count,
        "independent_batch_size": 1,
        "grouped_batch_size": count,
        "pca_coordinate_mae": float(np.mean(np.abs(grouped_np - independent_np))),
        "pca_rmse": float(np.sqrt(np.mean((grouped_np - independent_np) ** 2))),
        "vertex_coordinate_mae_mm": float(np.mean(np.abs(grouped_vertices - independent_vertices))),
        "vertex_euclidean_mean_mm": float(np.mean(euclidean)),
        "vertex_euclidean_max_mm": float(np.max(euclidean)),
        "interpretation": "Nonzero values mean prediction depends on other cases in the evaluation batch.",
    }


def bootstrap_comparisons(
    rows: list[dict[str, Any]],
    replicates: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    comparisons = [
        ("anchored_v4", "v1", "primary"),
        ("e3_selected", "v1", "primary"),
        ("brainode", "no_change", "primary"),
        ("brainode", "v1", "primary"),
        ("brainode", "anchored_v4", "primary"),
        ("brainode", "e3_selected", "primary"),
        ("e3_latest_sensitivity", "e3_selected", "exploratory_sensitivity"),
    ]
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    test_rows = [row for row in rows if row["split"] == "test"]
    for structure in STRUCTURE_RUNS:
        for diagnosis in ("CN", "AD", "ALL"):
            subset = [
                row for row in test_rows
                if row["structure"] == structure
                and (diagnosis == "ALL" or row["diagnosis"] == diagnosis)
            ]
            for current_method, baseline_method, role in comparisons:
                for metric in common.PAIR_ERROR_METRICS + common.PAIR_HIGHER_METRICS:
                    subject_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
                    for row in subset:
                        if row["method"] in {current_method, baseline_method} and math.isfinite(float(row[metric])):
                            subject_values[row["subject_id"]][row["method"]].append(float(row[metric]))
                    paired = []
                    for methods in subject_values.values():
                        if current_method in methods and baseline_method in methods:
                            current_value = float(np.mean(methods[current_method]))
                            baseline_value = float(np.mean(methods[baseline_method]))
                            improvement = (
                                current_value - baseline_value
                                if metric in common.PAIR_HIGHER_METRICS
                                else baseline_value - current_value
                            )
                            paired.append(improvement)
                    if not paired:
                        continue
                    values = np.asarray(paired, dtype=np.float64)
                    boot = np.asarray([
                        np.mean(rng.choice(values, size=values.size, replace=True))
                        for _ in range(replicates)
                    ])
                    output.append({
                        "structure": structure,
                        "diagnosis": diagnosis,
                        "current_method": current_method,
                        "baseline_method": baseline_method,
                        "comparison_role": role,
                        "metric": metric,
                        "subjects": len(values),
                        "improvement_mean": float(np.mean(values)),
                        "ci95_low": float(np.quantile(boot, 0.025)),
                        "ci95_high": float(np.quantile(boot, 0.975)),
                        "fraction_subjects_better": float(np.mean(values > 0.0)),
                        "higher_improvement_is_better": True,
                    })
    return output


def primary_rankings(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "vertex_coordinate_mae_mm_mean",
        "volume_relative_error_mean",
        "log_volume_rate_abs_error_mean",
        "local_normal_rate_mae_mm_per_year_mean",
    ]
    rows: list[dict[str, Any]] = []
    for structure in STRUCTURE_RUNS:
        for diagnosis in ("CN", "AD", "ALL"):
            candidates = [
                row for row in summary
                if row["structure"] == structure
                and row["split"] == "test"
                and row["diagnosis"] == diagnosis
                and row["method"] in PRIMARY_METHODS
            ]
            for metric in metrics:
                ordered = sorted(candidates, key=lambda row: float(row[metric]))
                for rank, row in enumerate(ordered, start=1):
                    rows.append({
                        "structure": structure,
                        "diagnosis": diagnosis,
                        "metric": metric,
                        "rank": rank,
                        "method": row["method"],
                        "value": float(row[metric]),
                    })
    return rows


def plot_brainode_histories(histories: dict[str, list[dict[str, Any]]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metric = "val_first_last_endpoint_vertex_mae_mm"
    for ax, (structure, history) in zip(axes, histories.items()):
        epochs = [int(row["epoch"]) for row in history]
        values = [float(row[metric]) for row in history]
        best_index = int(np.argmin(values))
        ax.plot(epochs, values, color="#4C78A8", linewidth=1.5)
        ax.scatter([epochs[best_index]], [values[best_index]], color="#E45756", s=45, zorder=3)
        ax.set_title(f"{STRUCTURE_RUNS[structure]['label']} BrainODE\nbest epoch {epochs[best_index]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation first-last vertex MAE (mm)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_markdown(
    report: dict[str, Any],
    summary: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
) -> str:
    lines = [
        "# BrainODE independent comparison",
        "",
        "This report evaluates BrainODE, no-change, V1, anchored V4, and selected E3 on the exact same locked strict CN/AD test pairs with one structure-specific PCA-150 decoder and one metric implementation. E3 latest remains exploratory.",
        "",
        "## Contract status",
        "",
        f"- Overall: **{report['status']}**",
        "- Strict no-MCI and subject/scan-disjoint splits: passed.",
        "- BrainODE training did not load test data: passed.",
        "- BrainODE archives and PCA model exactly match the comparison models: passed.",
        "- BrainODE primary inference is one case at a time, matching its previous evaluator.",
        "- The inherited attention operates across cases in a batch; see `brainode_batch_context_audit.json` for the measured sensitivity.",
        "",
        "## Locked test overview",
        "",
        "| Structure | Method | Dx | Vertex MAE (mm) | Volume rel. error | Log-rate error | Predicted mm³/year | Observed mm³/year |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for structure in STRUCTURE_RUNS:
        for method in METHOD_ORDER:
            for diagnosis in ("CN", "AD"):
                match = next((
                    row for row in summary
                    if row["structure"] == structure
                    and row["method"] == method
                    and row["split"] == "test"
                    and row["diagnosis"] == diagnosis
                ), None)
                if match is None:
                    continue
                lines.append(
                    f"| {STRUCTURE_RUNS[structure]['label']} | {METHOD_LABELS[method]} | {diagnosis} | "
                    f"{match['vertex_coordinate_mae_mm_mean']:.4f} | {match['volume_relative_error_mean']:.4f} | "
                    f"{match['log_volume_rate_abs_error_mean']:.4f} | "
                    f"{match['predicted_volume_change_mm3_per_year_mean']:.1f} | "
                    f"{match['observed_volume_change_mm3_per_year_mean']:.1f} |"
                )
    lines.extend(["", "## Primary winners on all test pairs", ""])
    for structure in STRUCTURE_RUNS:
        winners = [
            row for row in rankings
            if row["structure"] == structure and row["diagnosis"] == "ALL" and row["rank"] == 1
        ]
        lines.append(f"- {STRUCTURE_RUNS[structure]['label']}: " + "; ".join(
            f"{row['metric'].replace('_mean', '')} = {METHOD_LABELS[row['method']]} ({row['value']:.5g})"
            for row in winners
        ))
    lines.extend([
        "",
        "## Interpretation constraints",
        "",
        "- Test results compare already selected models; they must not be used to choose a new checkpoint.",
        "- Primary claims use `method_role=primary`; E3 latest is exploratory only.",
        "- Positive `improvement_mean` in the bootstrap table favors `current_method`.",
        "- Raw-mesh volume is an observed target only. Every model prediction is decoded through the same fixed PCA-150 representation.",
        "- BrainODE batch-context dependence is an architectural limitation and should be reported with its performance.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.bootstrap_replicates <= 0 or args.context_audit_size <= 1:
        raise ValueError("Batch sizes and bootstrap replicates must be positive; context audit must exceed one")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite validation output: {args.output_dir}")
    device = choose_device(args.device)
    print(f"BrainODE independent comparison | device={device} | output={args.output_dir}", flush=True)

    all_pair_rows: list[dict[str, Any]] = []
    all_trajectory_rows: list[dict[str, Any]] = []
    all_slope_rows: list[dict[str, Any]] = []
    contract_report: dict[str, Any] = {}
    context_audits: dict[str, Any] = {}
    histories: dict[str, list[dict[str, Any]]] = {}

    for structure, specification in STRUCTURE_RUNS.items():
        print(f"Loading and validating {specification['label']}...", flush=True)
        archives, pairs, geometry, adapters, structure_report = common.load_structure(
            structure, specification, device
        )
        v1_dir = (
            BASE_ROOT / specification["folder"] / "cocycle_v4" / "training" / specification["v1"]
        )
        reference_config = read_json(v1_dir / "config_used.json")
        brainode, brainode_report, history = load_brainode(
            structure, specification, device, reference_config
        )
        adapters["brainode"] = brainode
        structure_report["brainode"] = brainode_report
        contract_report[structure] = structure_report
        histories[structure] = history
        context_audits[structure] = batch_context_audit(
            brainode,
            archives["test"],
            pairs["test"],
            geometry,
            device,
            args.context_audit_size,
        )
        print(f"  BrainODE batch-context audit: {context_audits[structure]}", flush=True)

        for method in METHOD_ORDER:
            adapter = adapters[method]
            for split in ("val", "test"):
                print(f"  scoring {method:24s} {split} pairs={len(pairs[split])}", flush=True)
                all_pair_rows.extend(common.evaluate_method_pairs(
                    structure,
                    method,
                    adapter,
                    archives[split],
                    pairs[split],
                    geometry,
                    split,
                    device,
                    args.batch_size,
                ))
                trajectories, slopes = common.evaluate_trajectories(
                    structure,
                    method,
                    adapter,
                    archives[split],
                    geometry,
                    split,
                    device,
                )
                all_trajectory_rows.extend(trajectories)
                all_slope_rows.extend(slopes)

    print("Summarizing pairs, trajectories, slopes, and subject bootstraps...", flush=True)
    pair_summary = common.summarize_pair_rows(all_pair_rows, "diagnosis")
    gap_summary = common.summarize_pair_rows(all_pair_rows, "gap")
    slope_summary = common.summarize_slopes(all_slope_rows)
    bootstrap = bootstrap_comparisons(all_pair_rows, args.bootstrap_replicates)
    rankings = primary_rankings(pair_summary)
    report = {
        "status": "pass",
        "device": str(device),
        "structures": contract_report,
        "methods": {
            method: {
                "label": METHOD_LABELS[method],
                "role": "primary" if method in PRIMARY_METHODS else "exploratory_sensitivity",
            }
            for method in METHOD_ORDER
        },
        "pair_rows": len(all_pair_rows),
        "trajectory_rows": len(all_trajectory_rows),
        "subject_slope_rows": len(all_slope_rows),
        "bootstrap_replicates": args.bootstrap_replicates,
        "brainode_included": True,
        "brainode_primary_inference_batch_size": 1,
        "brainode_batch_context_audit": context_audits,
        "source_artifacts_modified": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    tables = args.output_dir / "tables"
    figures = args.output_dir / "figures"
    common.write_csv(tables / "per_pair_metrics.csv", all_pair_rows)
    common.write_csv(tables / "pair_summary.csv", pair_summary)
    common.write_csv(tables / "gap_summary.csv", gap_summary)
    common.write_csv(tables / "trajectory_predictions.csv", all_trajectory_rows)
    common.write_csv(tables / "subject_slopes.csv", all_slope_rows)
    common.write_csv(tables / "subject_slope_summary.csv", slope_summary)
    common.write_csv(tables / "paired_subject_bootstrap.csv", bootstrap)
    common.write_csv(tables / "primary_rankings.csv", rankings)
    common.write_json(args.output_dir / "brainode_batch_context_audit.json", context_audits)
    common.write_json(args.output_dir / "validation_report.json", report)
    figures.mkdir(parents=True, exist_ok=True)
    common.plot_metric_bars(pair_summary, figures / "test_metric_comparison.png")
    common.plot_volume_rates(pair_summary, figures / "test_volume_change_rates.png")
    common.plot_e3_histories(figures / "e3_validation_checkpoint_tradeoff.png")
    plot_brainode_histories(histories, figures / "brainode_validation_history.png")
    (args.output_dir / "README.md").write_text(
        build_markdown(report, pair_summary, rankings) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"Validation complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
