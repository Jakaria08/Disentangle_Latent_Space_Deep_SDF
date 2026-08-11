#!/usr/bin/env python3
"""Summarize direct-flow pair metrics by follow-up gap bins."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

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
        raise FileNotFoundError(
            f"No pair metric CSVs found under {analysis_dir}."
        )
    return pd.concat(frames, ignore_index=True)


def summarize_group(frame: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    candidate_metrics = [
        "model_target_sdf_l1",
        "composed_target_sdf_l1",
        "no_change_target_sdf_l1",
        "sdf_l1_improvement",
        "composed_sdf_l1_improvement",
        "composed_sdf_l1_gain_vs_direct",
        "direct_vs_composed_latent_mse",
        "composed_displacement_magnitude_abs_error",
        "observed_cocycle_mse",
        "virtual_cocycle_mse",
    ]
    metrics = [metric for metric in candidate_metrics if metric in frame.columns]
    groups: Iterable[tuple[object, pd.DataFrame]]
    groups = [((), frame)] if not columns else frame.groupby(columns, dropna=False)
    rows = []
    for key, group in groups:
        key_values = key if isinstance(key, tuple) else (key,)
        row = {
            "grouping": name,
            "rows": int(len(group)),
            "model_beats_no_change_fraction": float(
                group["model_beats_no_change"].mean()
            ),
        }
        if "composed_beats_no_change" in group.columns:
            row["composed_beats_no_change_fraction"] = float(
                group["composed_beats_no_change"].mean()
            )
        if "composed_beats_direct" in group.columns:
            row["composed_beats_direct_fraction"] = float(
                group["composed_beats_direct"].mean()
            )
        for column, value in zip(columns, key_values):
            row[column] = value
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(analysis_dir: Path) -> Path:
    frame = load_pairs(analysis_dir)
    if "gap_years" not in frame.columns:
        raise KeyError("Pair metrics must contain gap_years.")
    if (
        "model_target_sdf_l1" in frame.columns
        and "composed_target_sdf_l1" in frame.columns
        and "composed_sdf_l1_gain_vs_direct" not in frame.columns
    ):
        frame["composed_sdf_l1_gain_vs_direct"] = (
            frame["model_target_sdf_l1"] - frame["composed_target_sdf_l1"]
        )
    bins = [0.0, 1.0, 2.0, 4.0, np.inf]
    labels = ["0-1y", "1-2y", "2-4y", "4+y"]
    frame = frame.copy()
    frame["gap_bin"] = pd.cut(
        frame["gap_years"].astype(float),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    summaries = [
        summarize_group(frame, ["split"], "split"),
        summarize_group(frame, ["split", "diagnosis"], "split_diagnosis"),
        summarize_group(frame, ["split", "gap_bin"], "split_gap_bin"),
        summarize_group(
            frame,
            ["split", "diagnosis", "gap_bin"],
            "split_diagnosis_gap_bin",
        ),
        summarize_group(frame, ["gap_bin"], "all_gap_bin"),
        summarize_group(frame, ["diagnosis", "gap_bin"], "all_diagnosis_gap_bin"),
    ]
    output = pd.concat(summaries, ignore_index=True)
    output_path = analysis_dir / "gap_bin_summary.csv"
    output.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        required=True,
        help="Analysis checkpoint directory containing pair metric CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = summarize(Path(args.analysis).resolve())
    print(output_path)


if __name__ == "__main__":
    main()
