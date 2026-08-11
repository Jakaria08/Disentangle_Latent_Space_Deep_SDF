#!/usr/bin/env python3
"""Write one-shot anchor pair tables and horizon-OOD metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import pandas as pd


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "adni_no_mci_direct_flow_data.py").is_file():
            return parent
    raise RuntimeError("Could not locate repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adni_no_mci_direct_flow_data import build_forward_pairs, load_metadata


def resolve_path(path_value: str | Path, experiment_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (experiment_dir / path).resolve()


def horizon_bin(gap_years: float) -> str:
    if gap_years < 1.0:
        return "0-1y"
    if gap_years < 2.0:
        return "1-2y"
    if gap_years < 3.0:
        return "2-3y"
    if gap_years < 5.0:
        return "3-5y"
    return "5+y"


def rows_from_pairs(
    pairs,
    *,
    split: str,
    age_origin: float,
    age_range: float,
    ood_horizon_years: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for pair in pairs:
        gap_years = float(pair.age_gap_norm * age_range)
        rows.append(
            {
                "split": split,
                "subject_id": pair.subject_id,
                "diagnosis": pair.diagnosis,
                "label_ad": pair.label_ad,
                "source_scan_id": pair.source_scan_id,
                "target_scan_id": pair.target_scan_id,
                "source_visit_order": pair.source_visit_order,
                "target_visit_order": pair.target_visit_order,
                "pair_type": (
                    "adjacent"
                    if pair.target_visit_order - pair.source_visit_order == 1
                    else "nonadjacent"
                ),
                "source_age_norm": pair.source_time,
                "target_age_norm": pair.target_time,
                "source_age_years": age_origin + age_range * pair.source_time,
                "target_age_years": age_origin + age_range * pair.target_time,
                "gap_norm": pair.age_gap_norm,
                "gap_years": gap_years,
                "horizon_bin": horizon_bin(gap_years),
                "ood_protocol": (
                    f"ood_gt_{ood_horizon_years:g}y"
                    if gap_years > ood_horizon_years
                    else f"in_horizon_le_{ood_horizon_years:g}y"
                ),
                "has_observed_intermediate": pair.has_observed_intermediate,
                "observed_intermediate_count": len(pair.observed_intermediate_times),
                "source_mesh_volume_mm3": pair.source_mesh_volume_mm3,
                "target_mesh_volume_mm3": pair.target_mesh_volume_mm3,
                "target_sdf_path": pair.target_sdf_path,
            }
        )
    return rows


def summarize(frame: pd.DataFrame) -> Dict[str, object]:
    split_summary = {}
    for split, group in frame.groupby("split", sort=True):
        split_summary[split] = {
            "pairs": int(len(group)),
            "subjects": int(group["subject_id"].nunique()),
            "diagnosis_pairs": {
                key: int(value)
                for key, value in group["diagnosis"].value_counts().items()
            },
            "horizon_bins": {
                str(key): int(value)
                for key, value in group["horizon_bin"].value_counts().sort_index().items()
            },
            "ood_protocol": {
                str(key): int(value)
                for key, value in group["ood_protocol"].value_counts().items()
            },
            "gap_years": {
                "min": float(group["gap_years"].min()),
                "median": float(group["gap_years"].median()),
                "q75": float(group["gap_years"].quantile(0.75)),
                "q90": float(group["gap_years"].quantile(0.90)),
                "max": float(group["gap_years"].max()),
            },
        }
    return {
        "status": "pass",
        "total_pairs": int(len(frame)),
        "total_subjects": int(frame["subject_id"].nunique()),
        "splits": split_summary,
    }


def write_split(frame: pd.DataFrame, split: str, output_dir: Path) -> None:
    frame.loc[frame["split"] == split].to_csv(
        output_dir / f"anchor_pairs_{split}.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--experiment", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment).resolve()
    specs = json.loads((experiment_dir / "specs.json").read_text())
    metadata = load_metadata(
        resolve_path(specs["LongitudinalMetadataFile"], experiment_dir)
    )
    age_spec = specs.get("AgeNormalization", {})
    age_origin = float(age_spec.get("minimum_age_years", 57.0))
    age_range = float(age_spec.get("age_range_years", 34.0))
    protocol = specs.get("AnchorPredictionProtocol", {})
    ood_horizon_years = float(protocol.get("ood_horizon_years", 3.0))
    pair_source_mode = str(specs.get("PairSourceMode", "first_only"))

    frames = []
    for split in ("train", "val", "test"):
        split_frame = metadata.loc[metadata["split"] == split].copy()
        pairs = build_forward_pairs(split_frame, source_mode=pair_source_mode)
        frames.append(
            pd.DataFrame(
                rows_from_pairs(
                    pairs,
                    split=split,
                    age_origin=age_origin,
                    age_range=age_range,
                    ood_horizon_years=ood_horizon_years,
                )
            )
        )
    all_pairs = pd.concat(frames, ignore_index=True)
    output_dir = experiment_dir / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_pairs.to_csv(output_dir / "anchor_pairs_all.csv", index=False)
    for split in ("train", "val", "test"):
        write_split(all_pairs, split, output_dir)
    all_pairs.loc[
        (all_pairs["split"] == "train")
        & (all_pairs["gap_years"] <= ood_horizon_years)
    ].to_csv(output_dir / "ood_horizon_train_le3y.csv", index=False)
    all_pairs.loc[
        (all_pairs["split"].isin(["val", "test"]))
        & (all_pairs["gap_years"] > ood_horizon_years)
    ].to_csv(output_dir / "ood_horizon_eval_gt3y.csv", index=False)
    summary = summarize(all_pairs)
    (output_dir / "anchor_pair_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
