#!/usr/bin/env python3
"""Summarize anchor-flow evaluation by horizon, OOD status, and volume trend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def load_pairs(analysis_dir: Path) -> pd.DataFrame:
    all_path = analysis_dir / "all_pair_metrics.csv"
    if all_path.is_file():
        return pd.read_csv(all_path)
    frames = []
    for split in ("train", "val", "test"):
        path = analysis_dir / f"{split}_pair_metrics.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No pair metric CSVs found under {analysis_dir}.")
    return pd.concat(frames, ignore_index=True)


def add_protocol_columns(frame: pd.DataFrame, ood_horizon_years: float) -> pd.DataFrame:
    frame = frame.copy()
    bins = [0.0, 1.0, 2.0, 3.0, 5.0, np.inf]
    labels = ["0-1y", "1-2y", "2-3y", "3-5y", "5+y"]
    frame["horizon_bin"] = pd.cut(
        frame["gap_years"].astype(float),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    frame["ood_protocol"] = np.where(
        frame["gap_years"].astype(float) > float(ood_horizon_years),
        f"ood_gt_{ood_horizon_years:g}y",
        f"in_horizon_le_{ood_horizon_years:g}y",
    )
    return frame


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def summarize_group(frame: pd.DataFrame, columns: List[str], name: str) -> pd.DataFrame:
    candidate_metrics = [
        "model_target_sdf_l1",
        "composed_target_sdf_l1",
        "no_change_target_sdf_l1",
        "sdf_l1_improvement",
        "composed_sdf_l1_improvement",
        "composed_sdf_l1_gain_vs_direct",
        "direct_vs_composed_latent_mse",
        "future_extrapolation_cocycle_mse",
        "predicted_latent_zscore_p95",
        "composed_latent_zscore_p95",
        "model_proxy_log_volume_ratio",
        "composed_proxy_log_volume_ratio",
        "real_log_volume_ratio",
        "model_proxy_log_volume_ratio_abs_error",
        "composed_proxy_log_volume_ratio_abs_error",
        "model_proxy_annualized_log_volume_change",
        "composed_proxy_annualized_log_volume_change",
        "real_annualized_log_volume_change",
        "model_proxy_annualized_log_volume_change_abs_error",
        "composed_proxy_annualized_log_volume_change_abs_error",
        "cn_condition_proxy_log_volume_ratio",
        "ad_condition_proxy_log_volume_ratio",
        "real_final_log_volume_ratio",
        "model_final_log_volume_ratio",
        "composed_final_log_volume_ratio",
        "model_final_abs_error",
        "composed_final_abs_error",
    ]
    metrics = [metric for metric in candidate_metrics if metric in frame.columns]
    bool_metrics = [
        metric
        for metric in (
            "model_beats_no_change",
            "composed_beats_no_change",
            "model_volume_direction_correct",
            "composed_volume_direction_correct",
            "ad_more_atrophy_than_cn",
        )
        if metric in frame.columns
    ]
    groups: Iterable[tuple[object, pd.DataFrame]]
    groups = [((), frame)] if not columns else frame.groupby(columns, dropna=False)
    rows: List[Dict[str, object]] = []
    for key, group in groups:
        key_values = key if isinstance(key, tuple) else (key,)
        row: Dict[str, object] = {"grouping": name, "rows": int(len(group))}
        for column, value in zip(columns, key_values):
            row[column] = value
        for metric in metrics:
            row[f"{metric}_mean"] = finite_mean(group[metric])
        for metric in bool_metrics:
            row[f"{metric}_fraction"] = finite_mean(group[metric].astype(float))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_volume_subjects(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "split",
        "subject_id",
        "diagnosis",
        "gap_years",
        "model_proxy_log_volume_ratio",
        "composed_proxy_log_volume_ratio",
        "real_log_volume_ratio",
    }
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    rows = []
    for keys, group in frame.groupby(["split", "subject_id", "diagnosis"], sort=True):
        split, subject_id, diagnosis = keys
        longest = group.sort_values("gap_years").iloc[-1]
        rows.append(
            {
                "split": split,
                "subject_id": subject_id,
                "diagnosis": diagnosis,
                "longest_gap_years": float(longest["gap_years"]),
                "real_final_log_volume_ratio": float(longest["real_log_volume_ratio"]),
                "model_final_log_volume_ratio": float(
                    longest["model_proxy_log_volume_ratio"]
                ),
                "composed_final_log_volume_ratio": float(
                    longest["composed_proxy_log_volume_ratio"]
                ),
                "model_final_abs_error": abs(
                    float(longest["model_proxy_log_volume_ratio"])
                    - float(longest["real_log_volume_ratio"])
                ),
                "composed_final_abs_error": abs(
                    float(longest["composed_proxy_log_volume_ratio"])
                    - float(longest["real_log_volume_ratio"])
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--ood-horizon-years", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis).resolve()
    pairs = add_protocol_columns(load_pairs(analysis_dir), args.ood_horizon_years)
    pairs.to_csv(analysis_dir / "anchor_pair_metrics_with_protocol.csv", index=False)

    summaries = [
        summarize_group(pairs, ["split"], "split"),
        summarize_group(pairs, ["split", "diagnosis"], "split_diagnosis"),
        summarize_group(pairs, ["split", "horizon_bin"], "split_horizon"),
        summarize_group(
            pairs,
            ["split", "diagnosis", "horizon_bin"],
            "split_diagnosis_horizon",
        ),
        summarize_group(pairs, ["split", "ood_protocol"], "split_ood"),
        summarize_group(
            pairs,
            ["split", "diagnosis", "ood_protocol"],
            "split_diagnosis_ood",
        ),
    ]
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(analysis_dir / "anchor_protocol_summary.csv", index=False)

    subject_volume = summarize_volume_subjects(pairs)
    if not subject_volume.empty:
        subject_volume.to_csv(
            analysis_dir / "anchor_subject_volume_trend.csv",
            index=False,
        )
        volume_summary = summarize_group(
            subject_volume,
            ["split", "diagnosis"],
            "subject_final_volume",
        )
        volume_summary.to_csv(
            analysis_dir / "anchor_subject_volume_summary.csv",
            index=False,
        )

    payload = {
        "analysis_dir": str(analysis_dir),
        "pairs": int(len(pairs)),
        "ood_horizon_years": float(args.ood_horizon_years),
        "outputs": [
            "anchor_pair_metrics_with_protocol.csv",
            "anchor_protocol_summary.csv",
        ],
    }
    if not subject_volume.empty:
        payload["outputs"].extend(
            [
                "anchor_subject_volume_trend.csv",
                "anchor_subject_volume_summary.csv",
            ]
        )
    (analysis_dir / "anchor_protocol_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
