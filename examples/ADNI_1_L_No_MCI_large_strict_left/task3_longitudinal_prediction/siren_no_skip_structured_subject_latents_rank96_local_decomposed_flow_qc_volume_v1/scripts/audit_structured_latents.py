#!/usr/bin/env python3
"""Audit raw versus structured SIREN latents before flow training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fit_structured_siren_latents import (  # noqa: E402
    SPLITS,
    balanced_sdf_samples,
    choose_device,
    decode_sdf_loss,
    load_metadata,
    load_raw_latents,
    read_specs,
    resolve_path,
    split_arrays,
    stable_seed,
    write_json,
)
from train_deep_sdf_longitudinal_direct_flow import load_frozen_decoder  # noqa: E402


def load_structured_latents(
    experiment_dir: Path,
    expected_dim: int,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        path = experiment_dir / "latents" / f"{split}_latents.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing structured latent archive for {split}: {path}"
            )
        with np.load(path, allow_pickle=False) as payload:
            if "scan_ids" not in payload or "latents" not in payload:
                raise ValueError(f"Archive lacks scan_ids/latents: {path}")
            scan_ids = payload["scan_ids"].astype(str)
            latents = np.asarray(payload["latents"], dtype=np.float32)
        if latents.ndim != 2 or latents.shape[1] != int(expected_dim):
            raise ValueError(
                f"Expected structured latents [N,{expected_dim}], got {latents.shape}"
            )
        result[split] = {
            str(scan_id): np.asarray(latent, dtype=np.float32)
            for scan_id, latent in zip(scan_ids, latents)
        }
    return result


def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def safe_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else float("nan")


def safe_corr(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(left, right) / denom)


def annual_rate(
    source_value: float,
    target_value: float,
    source_age: float,
    target_age: float,
) -> float:
    gap = float(target_age - source_age)
    if not math.isfinite(gap) or abs(gap) < 1.0e-6:
        return float("nan")
    return float((target_value - source_value) / gap)


def build_audit_rows(
    frame: pd.DataFrame,
    raw: Mapping[str, Mapping[str, np.ndarray]],
    structured: Mapping[str, Mapping[str, np.ndarray]],
    *,
    latent_dim: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pair_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    for split in SPLITS:
        arrays = split_arrays(frame, split, raw[split], latent_dim)
        struct_map = structured[split]
        for subject_id, group in arrays.frame.groupby("subject_id", sort=True):
            group = group.sort_values(
                ["continuous_age_norm", "visit_order", "scan_id"]
            )
            rows = list(group.itertuples(index=False))
            raw_steps: list[np.ndarray] = []
            struct_steps: list[np.ndarray] = []
            mesh_rates: list[float] = []
            raw_speeds: list[float] = []
            struct_speeds: list[float] = []
            for source, target in zip(rows[:-1], rows[1:]):
                source_id = str(source.scan_id)
                target_id = str(target.scan_id)
                raw_delta = raw[split][target_id] - raw[split][source_id]
                struct_delta = struct_map[target_id] - struct_map[source_id]
                age_gap = float(target.continuous_age_years - source.continuous_age_years)
                age_gap = max(age_gap, 1.0e-6)
                source_volume = float(getattr(source, "left_mesh_volume_mm3", float("nan")))
                target_volume = float(getattr(target, "left_mesh_volume_mm3", float("nan")))
                volume_rate = annual_rate(
                    source_volume,
                    target_volume,
                    float(source.continuous_age_years),
                    float(target.continuous_age_years),
                )
                volume_pct_rate = (
                    100.0 * volume_rate / source_volume
                    if math.isfinite(volume_rate) and source_volume > 0.0
                    else float("nan")
                )
                raw_speed = float(np.linalg.norm(raw_delta) / age_gap)
                struct_speed = float(np.linalg.norm(struct_delta) / age_gap)
                raw_steps.append(raw_delta)
                struct_steps.append(struct_delta)
                mesh_rates.append(volume_pct_rate)
                raw_speeds.append(raw_speed)
                struct_speeds.append(struct_speed)
                pair_rows.append(
                    {
                        "split": split,
                        "subject_id": subject_id,
                        "diagnosis": str(source.diagnosis),
                        "source_scan_id": source_id,
                        "target_scan_id": target_id,
                        "source_age_years": float(source.continuous_age_years),
                        "target_age_years": float(target.continuous_age_years),
                        "gap_years": age_gap,
                        "mesh_volume_rate_mm3_per_year": volume_rate,
                        "mesh_volume_pct_rate_per_year": volume_pct_rate,
                        "mesh_volume_decreases": bool(
                            math.isfinite(source_volume)
                            and math.isfinite(target_volume)
                            and target_volume < source_volume
                        ),
                        "raw_delta_norm": float(np.linalg.norm(raw_delta)),
                        "structured_delta_norm": float(np.linalg.norm(struct_delta)),
                        "raw_speed_norm_per_year": raw_speed,
                        "structured_speed_norm_per_year": struct_speed,
                        "raw_structured_delta_cosine": cosine(raw_delta, struct_delta),
                        "source_raw_to_structured_l2": float(
                            np.linalg.norm(struct_map[source_id] - raw[split][source_id])
                        ),
                        "target_raw_to_structured_l2": float(
                            np.linalg.norm(struct_map[target_id] - raw[split][target_id])
                        ),
                    }
                )
            raw_consecutive_cosines = [
                cosine(left, right) for left, right in zip(raw_steps[:-1], raw_steps[1:])
            ]
            struct_consecutive_cosines = [
                cosine(left, right)
                for left, right in zip(struct_steps[:-1], struct_steps[1:])
            ]
            first = rows[0]
            last = rows[-1]
            first_volume = float(getattr(first, "left_mesh_volume_mm3", float("nan")))
            last_volume = float(getattr(last, "left_mesh_volume_mm3", float("nan")))
            subject_rows.append(
                {
                    "split": split,
                    "subject_id": subject_id,
                    "diagnosis": str(first.diagnosis),
                    "scan_count": int(len(rows)),
                    "first_age_years": float(first.continuous_age_years),
                    "last_age_years": float(last.continuous_age_years),
                    "followup_years": float(last.continuous_age_years - first.continuous_age_years),
                    "first_to_last_volume_pct_rate_per_year": (
                        100.0
                        * annual_rate(
                            first_volume,
                            last_volume,
                            float(first.continuous_age_years),
                            float(last.continuous_age_years),
                        )
                        / first_volume
                        if math.isfinite(first_volume) and first_volume > 0.0
                        else float("nan")
                    ),
                    "mean_raw_speed_norm_per_year": safe_mean(raw_speeds),
                    "mean_structured_speed_norm_per_year": safe_mean(struct_speeds),
                    "mean_raw_consecutive_delta_cosine": safe_mean(raw_consecutive_cosines),
                    "mean_structured_consecutive_delta_cosine": safe_mean(
                        struct_consecutive_cosines
                    ),
                    "raw_speed_vs_volume_rate_corr": safe_corr(raw_speeds, mesh_rates),
                    "structured_speed_vs_volume_rate_corr": safe_corr(
                        struct_speeds,
                        mesh_rates,
                    ),
                }
            )
    return pair_rows, subject_rows


def summarize_pairs(pair_rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(pair_rows)
    rows: list[dict[str, object]] = []
    if frame.empty:
        return rows
    for keys, group in frame.groupby(["split", "diagnosis"], dropna=False):
        split, diagnosis = keys
        rows.append(
            {
                "split": split,
                "diagnosis": diagnosis,
                "adjacent_pairs": int(len(group)),
                "subjects": int(group["subject_id"].nunique()),
                "mean_volume_pct_rate_per_year": safe_mean(
                    group["mesh_volume_pct_rate_per_year"]
                ),
                "median_volume_pct_rate_per_year": safe_median(
                    group["mesh_volume_pct_rate_per_year"]
                ),
                "volume_decrease_fraction": float(
                    group["mesh_volume_decreases"].astype(float).mean()
                ),
                "mean_raw_delta_norm": safe_mean(group["raw_delta_norm"]),
                "mean_structured_delta_norm": safe_mean(group["structured_delta_norm"]),
                "mean_raw_speed_norm_per_year": safe_mean(group["raw_speed_norm_per_year"]),
                "mean_structured_speed_norm_per_year": safe_mean(
                    group["structured_speed_norm_per_year"]
                ),
                "mean_raw_structured_delta_cosine": safe_mean(
                    group["raw_structured_delta_cosine"]
                ),
                "mean_source_raw_to_structured_l2": safe_mean(
                    group["source_raw_to_structured_l2"]
                ),
                "mean_target_raw_to_structured_l2": safe_mean(
                    group["target_raw_to_structured_l2"]
                ),
                "raw_speed_vs_volume_rate_corr": safe_corr(
                    group["raw_speed_norm_per_year"],
                    group["mesh_volume_pct_rate_per_year"],
                ),
                "structured_speed_vs_volume_rate_corr": safe_corr(
                    group["structured_speed_norm_per_year"],
                    group["mesh_volume_pct_rate_per_year"],
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_sdf(
    *,
    frame: pd.DataFrame,
    raw: Mapping[str, Mapping[str, np.ndarray]],
    structured: Mapping[str, Mapping[str, np.ndarray]],
    specs: Mapping[str, object],
    experiment_dir: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, object]]:
    if int(args.sdf_samples_per_scan) <= 0:
        return []
    device = choose_device(args)
    decoder, _epoch = load_frozen_decoder(specs, experiment_dir, device)
    decoder.eval()
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        split_frame = frame.loc[frame["split"] == split].copy()
        split_frame = split_frame.sort_values(
            ["subject_id", "continuous_age_norm", "visit_order", "scan_id"]
        )
        if int(args.max_sdf_scans_per_split) > 0:
            split_frame = split_frame.head(int(args.max_sdf_scans_per_split))
        for row in split_frame.itertuples(index=False):
            scan_id = str(row.scan_id)
            rng = np.random.default_rng(stable_seed(scan_id, int(args.seed)))
            samples_np = balanced_sdf_samples(
                str(row.sdf_npz_path),
                int(args.sdf_samples_per_scan),
                rng,
            )
            samples = torch.from_numpy(samples_np).view(1, len(samples_np), -1).to(device)
            raw_latent = torch.from_numpy(raw[split][scan_id]).view(1, -1).to(device)
            structured_latent = torch.from_numpy(structured[split][scan_id]).view(1, -1).to(device)
            with torch.no_grad():
                raw_l1 = float(
                    decode_sdf_loss(
                        decoder,
                        raw_latent,
                        samples,
                        clamp_distance=float(specs.get("ClampingDistance", 0.1)),
                        max_decoder_batch=int(args.max_decoder_batch),
                    ).detach().cpu()
                )
                structured_l1 = float(
                    decode_sdf_loss(
                        decoder,
                        structured_latent,
                        samples,
                        clamp_distance=float(specs.get("ClampingDistance", 0.1)),
                        max_decoder_batch=int(args.max_decoder_batch),
                    ).detach().cpu()
                )
            rows.append(
                {
                    "split": split,
                    "scan_id": scan_id,
                    "subject_id": str(row.subject_id),
                    "diagnosis": str(row.diagnosis),
                    "raw_sdf_l1": raw_l1,
                    "structured_sdf_l1": structured_l1,
                    "structured_minus_raw_sdf_l1": structured_l1 - raw_l1,
                    "raw_to_structured_l2": float(
                        np.linalg.norm(structured[split][scan_id] - raw[split][scan_id])
                    ),
                }
            )
    write_csv(output_dir / "sdf_scan_metrics.csv", rows)
    if rows:
        summary = []
        sdf_frame = pd.DataFrame(rows)
        for keys, group in sdf_frame.groupby(["split", "diagnosis"], dropna=False):
            split, diagnosis = keys
            summary.append(
                {
                    "split": split,
                    "diagnosis": diagnosis,
                    "scans": int(len(group)),
                    "mean_raw_sdf_l1": safe_mean(group["raw_sdf_l1"]),
                    "mean_structured_sdf_l1": safe_mean(group["structured_sdf_l1"]),
                    "mean_structured_minus_raw_sdf_l1": safe_mean(
                        group["structured_minus_raw_sdf_l1"]
                    ),
                    "median_structured_minus_raw_sdf_l1": safe_median(
                        group["structured_minus_raw_sdf_l1"]
                    ),
                }
            )
        write_csv(output_dir / "sdf_summary.csv", summary)
    return rows


def write_html_index(output_dir: Path, summary_rows: list[Mapping[str, object]]) -> None:
    table = pd.DataFrame(summary_rows).to_html(index=False, float_format=lambda x: f"{x:.6g}")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Structured Latent Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
    th {{ background: #f6f8fa; }}
    td:first-child, td:nth-child(2), th:first-child, th:nth-child(2) {{ text-align: left; }}
    code {{ background: #f6f8fa; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Structured Latent Audit</h1>
  <p>Summary by split and diagnosis. Detailed CSV files are in this directory.</p>
  {table}
  <p>Files: <code>adjacent_pair_metrics.csv</code>, <code>subject_metrics.csv</code>, <code>summary.csv</code>, optional <code>sdf_summary.csv</code>.</p>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit raw and structured SIREN latent longitudinal structure."
    )
    parser.add_argument(
        "--experiment",
        default=str(EXPERIMENT_DIR),
        help="Structured latent experiment directory.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sdf-samples-per-scan",
        type=int,
        default=0,
        help="Optional SDF samples per scan for raw vs structured reconstruction audit.",
    )
    parser.add_argument(
        "--max-sdf-scans-per-split",
        type=int,
        default=0,
        help="Limit SDF audit scans per split; 0 means all scans.",
    )
    parser.add_argument("--max-decoder-batch", type=int, default=65536)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment).resolve()
    specs = read_specs(experiment_dir)
    latent_dim = int(specs["CodeLength"])
    metadata_path = resolve_path(str(specs["LongitudinalMetadataFile"]), experiment_dir)
    frame = load_metadata(metadata_path)
    raw = load_raw_latents(specs, experiment_dir, latent_dim)
    structured = load_structured_latents(experiment_dir, latent_dim)
    output_dir = experiment_dir / "structured_latent_fit" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows, subject_rows = build_audit_rows(
        frame,
        raw,
        structured,
        latent_dim=latent_dim,
    )
    summary_rows = summarize_pairs(pair_rows)
    write_csv(output_dir / "adjacent_pair_metrics.csv", pair_rows)
    write_csv(output_dir / "subject_metrics.csv", subject_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    sdf_rows = evaluate_sdf(
        frame=frame,
        raw=raw,
        structured=structured,
        specs=specs,
        experiment_dir=experiment_dir,
        args=args,
        output_dir=output_dir,
    )
    write_json(
        output_dir / "audit_summary.json",
        {
            "experiment_dir": str(experiment_dir),
            "adjacent_pairs": int(len(pair_rows)),
            "subjects": int(len(subject_rows)),
            "sdf_scans_evaluated": int(len(sdf_rows)),
            "summary": summary_rows,
        },
    )
    write_html_index(output_dir, summary_rows)
    print(f"Structured latent audit written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
