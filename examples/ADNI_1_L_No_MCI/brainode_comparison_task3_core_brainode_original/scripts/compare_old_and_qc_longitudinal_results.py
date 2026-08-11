#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]

OLD_FLOW_RUNS = {
    "old_siren_full256_direct": {
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_optimized",
        "family": "SIREN flow",
    },
    "old_siren_sequence_cocycle": {
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_sequence_cocycle_latent_guard",
        "family": "SIREN sequence/cocycle",
    },
    "old_siren_backward_virtual_shape": {
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_backward_virtual_shape",
        "family": "SIREN backward virtual-shape",
    },
    "old_siren_backward_timeflow_future": {
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_backward_timeflow_future",
        "family": "SIREN backward/future",
    },
    "old_siren_real_sdf_cocycle": {
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_real_sdf_observed_virtual_cocycle",
        "family": "SIREN real-SDF cocycle",
    },
    "old_deepsdf_full256_direct": {
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_deepsdf_optimized",
        "family": "DeepSDF flow",
    },
    "old_smallnet_siren": {
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_smallnet_siren_optimized",
        "family": "SIREN small-net flow",
    },
    "old_smallnet_deepsdf": {
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_smallnet_deepsdf_optimized",
        "family": "DeepSDF small-net flow",
    },
    "old_pca32_siren": {
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_pca32_siren_optimized",
        "family": "SIREN PCA32 flow",
    },
    "old_pca32_deepsdf": {
        "dir": "examples/ADNI_1_L_No_MCI/longitudinal_direct_pca32_deepsdf_optimized",
        "family": "DeepSDF PCA32 flow",
    },
}

QC_FLOW_RUNS = {
    "qc_siren_large_unfiltered": {
        "dir": "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle",
        "family": "SIREN real-SDF cocycle",
    },
    "qc_siren_drop_bad_min2": {
        "dir": "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2",
        "family": "SIREN QC direct/composed",
    },
    "qc_siren_seq_rollout": {
        "dir": "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_seq_rollout_v1",
        "family": "SIREN QC sequence rollout",
    },
    "qc_siren_local_decomp_volume": {
        "dir": "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_local_decomposed_flow_qc_volume_v1",
        "family": "SIREN QC local decomposed volume",
    },
}

BRAINODE_RUNS = {
    "old_brainode_pca150": {
        "dataset": "old_adni",
        "metrics": "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/training/core_attention_pca150/evaluation/best_pairwise/all_trajectory_metrics.csv",
    },
    "qc_brainode_pca150": {
        "dataset": "qc_large",
        "metrics": "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/brainode_pca150_qc_stable/training/core_attention_pca150_qc_stable/evaluation/best/all_trajectory_metrics.csv",
    },
}

OOD_ARTIFACT_DIRS = {
    "old_siren_full256_direct": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_optimized/analysis/notebook_best",
    "old_siren_sequence_cocycle": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_sequence_cocycle_latent_guard/analysis/notebook_best",
    "old_siren_backward_timeflow_future": "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_backward_timeflow_future/analysis/notebook_best",
    "qc_siren_drop_bad_min2": "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2/analysis/notebook_best",
    "qc_siren_seq_rollout": "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_seq_rollout_v1/analysis/notebook_best",
    "qc_siren_local_decomp_volume": "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_local_decomposed_flow_qc_volume_v1/analysis/notebook_best",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build old-ADNI and QC-large longitudinal method comparison tables."
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
            "analysis/old_vs_qc_longitudinal_comparison"
        ),
    )
    return parser.parse_args()


def repo_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def overall_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row.get("grouping") == "overall":
            return row
    return rows[0] if rows else None


def flow_metric_rows(dataset: str, runs: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for method, spec in runs.items():
        run_dir = repo_path(spec["dir"])
        summary_path = run_dir / "analysis" / "checkpoint_best" / "test_summary.csv"
        row = overall_row(read_csv(summary_path))
        if not row:
            output_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "family": spec["family"],
                    "metric_space": "decoder_sdf_l1",
                    "rows": "",
                    "beats_no_change_fraction": "",
                    "model_error_mean": "",
                    "no_change_error_mean": "",
                    "improvement_mean": "",
                    "composed_beats_no_change_fraction": "",
                    "status": "missing test_summary.csv",
                    "source_file": str(summary_path),
                    "metric_note": "SIREN/DeepSDF flow metrics use decoder SDF L1.",
                }
            )
            continue
        output_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "family": spec["family"],
                "metric_space": "decoder_sdf_l1",
                "rows": row.get("rows", ""),
                "beats_no_change_fraction": row.get("model_beats_no_change_fraction", ""),
                "model_error_mean": row.get("model_target_sdf_l1_mean", ""),
                "no_change_error_mean": row.get("no_change_target_sdf_l1_mean", ""),
                "improvement_mean": row.get("sdf_l1_improvement_mean", ""),
                "composed_beats_no_change_fraction": row.get(
                    "composed_beats_no_change_fraction", ""
                ),
                "status": "ok",
                "source_file": str(summary_path),
                "metric_note": "SIREN/DeepSDF flow metrics use decoder SDF L1.",
            }
        )
    return output_rows


def brainode_metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, spec in BRAINODE_RUNS.items():
        metric_path = repo_path(spec["metrics"])
        records = [r for r in read_csv(metric_path) if r.get("split") == "test"]
        if not records:
            rows.append(
                {
                    "dataset": spec["dataset"],
                    "method": method,
                    "family": "BrainODE PCA150",
                    "metric_space": "pca_inverse_vertex_mae",
                    "rows": "",
                    "beats_no_change_fraction": "",
                    "model_error_mean": "",
                    "no_change_error_mean": "",
                    "improvement_mean": "",
                    "composed_beats_no_change_fraction": "",
                    "status": "missing pairwise BrainODE metrics",
                    "source_file": str(metric_path),
                    "metric_note": "BrainODE metrics use PCA inverse-transform vertex MAE.",
                }
            )
            continue
        improvements = [as_float(r.get("endpoint_vertex_mae_improvement")) for r in records]
        rows.append(
            {
                "dataset": spec["dataset"],
                "method": method,
                "family": "BrainODE PCA150",
                "metric_space": "pca_inverse_vertex_mae",
                "rows": len(records),
                "beats_no_change_fraction": mean([1.0 if v > 0 else 0.0 for v in improvements]),
                "model_error_mean": mean([as_float(r.get("endpoint_vertex_mae")) for r in records]),
                "no_change_error_mean": mean(
                    [as_float(r.get("no_change_vertex_mae")) for r in records]
                ),
                "improvement_mean": mean(improvements),
                "composed_beats_no_change_fraction": "",
                "status": "ok",
                "source_file": str(metric_path),
                "metric_note": "BrainODE metrics use PCA inverse-transform vertex MAE.",
            }
        )
    return rows


def sequence_metric_rows(dataset: str, runs: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, spec in runs.items():
        path = repo_path(spec["dir"]) / "analysis" / "checkpoint_best" / "test_sequence_summary.csv"
        row = overall_row(read_csv(path))
        if not row:
            continue
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "rows": row.get("rows", ""),
                "rollout_beats_no_change_fraction": row.get(
                    "rollout_beats_no_change_fraction", ""
                ),
                "one_shot_beats_no_change_fraction": row.get(
                    "one_shot_beats_no_change_fraction", ""
                ),
                "sequence_rollout_sdf_l1_mean": row.get("sequence_rollout_sdf_l1_mean", ""),
                "sequence_one_shot_sdf_l1_mean": row.get("sequence_one_shot_sdf_l1_mean", ""),
                "sequence_no_change_sdf_l1_mean": row.get("sequence_no_change_sdf_l1_mean", ""),
                "sequence_rollout_improvement_mean": row.get(
                    "sequence_rollout_improvement_mean", ""
                ),
                "sequence_one_shot_improvement_mean": row.get(
                    "sequence_one_shot_improvement_mean", ""
                ),
                "source_file": str(path),
            }
        )
    return rows


def observed_volume_rate_rows(
    dataset: str,
    method: str,
    path: Path,
) -> list[dict[str, Any]]:
    rows = read_csv(path)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "test":
            continue
        grouped[
            (
                row.get("subject_id", ""),
                row.get("diagnosis", ""),
                row.get("transport_method", ""),
                row.get("split", ""),
            )
        ].append(row)

    rates: dict[tuple[str, str], list[float]] = defaultdict(list)
    subject_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (subject_id, diagnosis, transport, _split), visits in grouped.items():
        visits.sort(key=lambda r: as_float(r.get("years_from_baseline")))
        for previous, current in zip(visits, visits[1:]):
            v0 = as_float(previous.get("volume"))
            v1 = as_float(current.get("volume"))
            t0 = as_float(previous.get("years_from_baseline"))
            t1 = as_float(current.get("years_from_baseline"))
            if not (math.isfinite(v0) and math.isfinite(v1) and math.isfinite(t0) and math.isfinite(t1)):
                continue
            if v0 == 0.0 or t1 <= t0:
                continue
            atrophy_pct_per_year = ((v0 - v1) / abs(v0)) * 100.0 / (t1 - t0)
            key = (diagnosis, transport)
            rates[key].append(atrophy_pct_per_year)
            subject_sets[key].add(subject_id)

    output: list[dict[str, Any]] = []
    for (diagnosis, transport), values in sorted(rates.items()):
        output.append(
            {
                "dataset": dataset,
                "method": method,
                "split": "test",
                "diagnosis": diagnosis,
                "transport_method": transport,
                "intervals": len(values),
                "subjects": len(subject_sets[(diagnosis, transport)]),
                "mean_atrophy_pct_per_year": mean(values),
                "source_file": str(path),
            }
        )
    return output


def selected_speed_rows(dataset: str, method: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(path):
        if row.get("split") != "test":
            continue
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "split": row.get("split", ""),
                "diagnosis": row.get("diagnosis", ""),
                "transport_method": row.get("transport_method", ""),
                "intervals": row.get("intervals", ""),
                "subjects": row.get("subjects", ""),
                "mean_atrophy_pct_per_year": row.get("mean_atrophy_pct_per_year", ""),
                "source_file": str(path),
            }
        )
    return rows


def volume_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    old_volume_path = repo_path(
        "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_siren_optimized/"
        "analysis/notebook_best/batch_observed_age_volume_trend.csv"
    )
    rows.extend(
        observed_volume_rate_rows(
            dataset="old_adni",
            method="old_siren_full256_direct",
            path=old_volume_path,
        )
    )
    for method, rel in {
        "qc_siren_drop_bad_min2": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2/"
            "analysis/notebook_best/selected_speed_summary.csv"
        ),
        "qc_siren_seq_rollout": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_seq_rollout_v1/"
            "analysis/notebook_best/selected_speed_summary.csv"
        ),
        "qc_siren_local_decomp_volume": (
            "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
            "siren_no_skip_local_decomposed_flow_qc_volume_v1/"
            "analysis/notebook_best/selected_speed_summary.csv"
        ),
    }.items():
        rows.extend(selected_speed_rows("qc_large", method, repo_path(rel)))
    return rows


def ood_artifact_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patterns = (
        "*ood*",
        "*counterfactual*",
        "*volume_forecast*",
        "*volume_trend*",
        "*direct_vs_composed*",
        "*interpolation*",
    )
    for method, rel in OOD_ARTIFACT_DIRS.items():
        directory = repo_path(rel)
        if not directory.exists():
            rows.append(
                {
                    "method": method,
                    "artifact_type": "missing_directory",
                    "path": str(directory),
                    "status": "missing",
                }
            )
            continue
        seen: set[Path] = set()
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                rows.append(
                    {
                        "method": method,
                        "artifact_type": path.suffix.lstrip(".") or "file",
                        "path": str(path),
                        "status": "ok",
                    }
                )
    return rows


def main() -> int:
    args = parse_args()
    output_dir = repo_path(args.output_dir)

    method_rows = (
        flow_metric_rows("old_adni", OLD_FLOW_RUNS)
        + flow_metric_rows("qc_large", QC_FLOW_RUNS)
        + brainode_metric_rows()
    )
    method_fields = [
        "dataset",
        "method",
        "family",
        "metric_space",
        "rows",
        "beats_no_change_fraction",
        "model_error_mean",
        "no_change_error_mean",
        "improvement_mean",
        "composed_beats_no_change_fraction",
        "status",
        "source_file",
        "metric_note",
    ]
    write_csv(output_dir / "method_test_pair_comparison.csv", method_rows, method_fields)

    seq_rows = sequence_metric_rows("old_adni", OLD_FLOW_RUNS) + sequence_metric_rows(
        "qc_large", QC_FLOW_RUNS
    )
    seq_fields = [
        "dataset",
        "method",
        "rows",
        "rollout_beats_no_change_fraction",
        "one_shot_beats_no_change_fraction",
        "sequence_rollout_sdf_l1_mean",
        "sequence_one_shot_sdf_l1_mean",
        "sequence_no_change_sdf_l1_mean",
        "sequence_rollout_improvement_mean",
        "sequence_one_shot_improvement_mean",
        "source_file",
    ]
    write_csv(output_dir / "sequence_rollout_comparison.csv", seq_rows, seq_fields)

    vol_rows = volume_rows()
    vol_fields = [
        "dataset",
        "method",
        "split",
        "diagnosis",
        "transport_method",
        "intervals",
        "subjects",
        "mean_atrophy_pct_per_year",
        "source_file",
    ]
    write_csv(output_dir / "volume_trend_test_comparison.csv", vol_rows, vol_fields)

    artifact_rows = ood_artifact_rows()
    artifact_fields = ["method", "artifact_type", "path", "status"]
    write_csv(output_dir / "ood_future_artifact_manifest.csv", artifact_rows, artifact_fields)

    summary = {
        "output_dir": str(output_dir),
        "method_rows": len(method_rows),
        "sequence_rows": len(seq_rows),
        "volume_rows": len(vol_rows),
        "ood_future_artifacts": len(artifact_rows),
        "notes": [
            "BrainODE error is PCA inverse-transform vertex MAE; SIREN/DeepSDF flow error is decoder SDF L1.",
            "Missing method rows mean no checkpoint_best/test_summary.csv was present, not that the model is invalid.",
            "Volume trend rows mix directly computed old-ADNI volume rates and existing QC selected_speed_summary outputs.",
        ],
    }
    write_json(output_dir / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
