#!/usr/bin/env python3
"""Export fixed-horizon direct/composed anchor-flow trajectory diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "evaluate_deep_sdf_longitudinal_direct_flow.py").is_file():
            return parent
    raise RuntimeError("Could not locate repository root.")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_deep_sdf_longitudinal_direct_flow import (
    load_trained_components,
    transport_composed_fixed_step,
)
from train_deep_sdf_longitudinal_direct_flow import (
    build_latent_regularizer_stats,
    build_loss,
    make_volume_probe_points,
    soft_volume_proxy,
)


def split_names(value: str) -> tuple[str, ...]:
    normalized = value.strip().lower()
    if normalized == "all":
        return ("train", "val", "test")
    if normalized == "both":
        return ("val", "test")
    if normalized not in {"train", "val", "test"}:
        raise ValueError("--split must be train, val, test, both, or all")
    return (normalized,)


def parse_horizons(raw: str | None, specs: Dict[str, object]) -> List[float]:
    if raw:
        return [float(value) for value in raw.split(",") if value.strip()]
    protocol = specs.get("AnchorPredictionProtocol", {})
    if isinstance(protocol, dict) and "long_horizon_years" in protocol:
        return [float(value) for value in protocol["long_horizon_years"]]
    return [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


def selected_subject_ids(
    metadata: pd.DataFrame,
    *,
    subjects_per_diagnosis: int,
) -> List[str]:
    selected: List[str] = []
    for diagnosis in ("CN", "AD"):
        ids = (
            metadata.loc[metadata["diagnosis"] == diagnosis, "subject_id"]
            .drop_duplicates()
            .sort_values()
            .head(int(subjects_per_diagnosis))
        )
        selected.extend(ids.astype(str).tolist())
    return selected


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "direct_vs_composed_latent_mse",
        "direct_latent_zscore_p95",
        "composed_latent_zscore_p95",
        "direct_proxy_log_volume_ratio",
        "composed_proxy_log_volume_ratio",
        "direct_proxy_volume",
        "composed_proxy_volume",
    ]
    metrics = [metric for metric in metrics if metric in frame.columns]
    rows: List[Dict[str, object]] = []
    for keys, group in frame.groupby(
        ["split", "source_diagnosis", "rollout_condition", "horizon_years"],
        dropna=False,
        sort=True,
    ):
        split, source_diagnosis, rollout_condition, horizon_years = keys
        row: Dict[str, object] = {
            "split": split,
            "source_diagnosis": source_diagnosis,
            "rollout_condition": rollout_condition,
            "horizon_years": float(horizon_years),
            "rows": int(len(group)),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = finite_mean(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def export_rows(
    *,
    split: str,
    specs: Dict[str, object],
    contract,
    decoder: torch.nn.Module,
    flow,
    device: torch.device,
    regularizer_stats,
    horizons: Iterable[float],
    subjects_per_diagnosis: int,
    composed_step_years: float,
) -> pd.DataFrame:
    metadata = contract.metadata.loc[contract.metadata["split"] == split].copy()
    subject_ids = selected_subject_ids(
        metadata,
        subjects_per_diagnosis=subjects_per_diagnosis,
    )
    age_spec = specs.get("AgeNormalization", {})
    age_origin = float(age_spec.get("minimum_age_years", 57.0))
    age_range = float(age_spec.get("age_range_years", 34.0))
    composed_step_norm = float(composed_step_years) / age_range
    latent_mean = regularizer_stats.mean.to(device=device)
    latent_std = regularizer_stats.std.to(device=device)
    loss_module = build_loss(specs, decoder, flow)
    volume_probe_points = make_volume_probe_points(specs, device)
    rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for subject_id in subject_ids:
            subject = metadata.loc[
                metadata["subject_id"].astype(str) == str(subject_id)
            ].sort_values(["continuous_age_norm", "visit_order", "scan_id"])
            first = subject.iloc[0]
            source_latent = contract.latent_maps[split][str(first["scan_id"])].to(
                device
            ).view(1, -1)
            source_time = torch.tensor(
                [[float(first["continuous_age_norm"])]],
                device=device,
                dtype=source_latent.dtype,
            )
            source_age_years = float(first["continuous_age_years"])
            source_volume = None
            if volume_probe_points is not None:
                source_volume = soft_volume_proxy(
                    loss_module,
                    source_latent,
                    volume_probe_points,
                    temperature=float(specs.get("VolumeProbeTemperature", 0.01)),
                    inside_sdf_sign=float(specs.get("VolumeProbeInsideSDFSign", -1.0)),
                    chunk_size=int(specs.get("VolumeProbeChunkSize", 4096)),
                )
            conditions = [
                ("observed_condition", float(first["label_ad"])),
                ("cn_counterfactual", 0.0),
                ("ad_counterfactual", 1.0),
            ]
            for horizon in horizons:
                target_age_years = source_age_years + float(horizon)
                target_time = torch.tensor(
                    [[(target_age_years - age_origin) / age_range]],
                    device=device,
                    dtype=source_latent.dtype,
                )
                for condition_name, condition_value in conditions:
                    condition = torch.tensor(
                        [[condition_value]],
                        device=device,
                        dtype=source_latent.dtype,
                    )
                    direct = flow.transport(
                        source_latent,
                        source_time,
                        target_time,
                        condition,
                    )
                    composed = transport_composed_fixed_step(
                        flow,
                        source_latent,
                        source_time,
                        target_time,
                        condition,
                        step_norm=composed_step_norm,
                    )
                    direct_z = torch.abs(
                        (direct - latent_mean.to(dtype=direct.dtype))
                        / latent_std.to(dtype=direct.dtype)
                    )
                    composed_z = torch.abs(
                        (composed - latent_mean.to(dtype=composed.dtype))
                        / latent_std.to(dtype=composed.dtype)
                    )
                    row: Dict[str, object] = {
                        "split": split,
                        "subject_id": str(subject_id),
                        "source_scan_id": str(first["scan_id"]),
                        "source_diagnosis": str(first["diagnosis"]),
                        "source_label_ad": int(first["label_ad"]),
                        "rollout_condition": condition_name,
                        "rollout_label_ad": int(condition_value),
                        "source_age_years": source_age_years,
                        "target_age_years": target_age_years,
                        "horizon_years": float(horizon),
                        "transport_method": "direct_and_composed",
                        "direct_displacement_l2": float(
                            torch.linalg.vector_norm(direct - source_latent).item()
                        ),
                        "composed_displacement_l2": float(
                            torch.linalg.vector_norm(composed - source_latent).item()
                        ),
                        "direct_vs_composed_latent_mse": float(
                            torch.mean((direct - composed) ** 2).item()
                        ),
                        "direct_latent_zscore_p95": float(
                            torch.quantile(direct_z.reshape(-1), 0.95).item()
                        ),
                        "direct_latent_zscore_max": float(direct_z.max().item()),
                        "composed_latent_zscore_p95": float(
                            torch.quantile(composed_z.reshape(-1), 0.95).item()
                        ),
                        "composed_latent_zscore_max": float(composed_z.max().item()),
                    }
                    if volume_probe_points is not None and source_volume is not None:
                        direct_volume = soft_volume_proxy(
                            loss_module,
                            direct,
                            volume_probe_points,
                            temperature=float(specs.get("VolumeProbeTemperature", 0.01)),
                            inside_sdf_sign=float(
                                specs.get("VolumeProbeInsideSDFSign", -1.0)
                            ),
                            chunk_size=int(specs.get("VolumeProbeChunkSize", 4096)),
                        )
                        composed_volume = soft_volume_proxy(
                            loss_module,
                            composed,
                            volume_probe_points,
                            temperature=float(specs.get("VolumeProbeTemperature", 0.01)),
                            inside_sdf_sign=float(
                                specs.get("VolumeProbeInsideSDFSign", -1.0)
                            ),
                            chunk_size=int(specs.get("VolumeProbeChunkSize", 4096)),
                        )
                        row.update(
                            {
                                "source_proxy_volume": float(source_volume.item()),
                                "direct_proxy_volume": float(direct_volume.item()),
                                "composed_proxy_volume": float(composed_volume.item()),
                                "direct_proxy_log_volume_ratio": float(
                                    torch.log(direct_volume / source_volume).item()
                                ),
                                "composed_proxy_log_volume_ratio": float(
                                    torch.log(composed_volume / source_volume).item()
                                ),
                            }
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--experiment", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", default="both")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--horizons-years", default=None)
    parser.add_argument("--subjects-per-diagnosis", type=int, default=25)
    parser.add_argument("--composed-step-years", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment).resolve()
    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu was given but CUDA is unavailable")
        torch.cuda.set_device(int(args.gpu))
    device = torch.device(
        f"cuda:{torch.cuda.current_device()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    specs, contract, decoder, flow, checkpoint, decoder_epoch = load_trained_components(
        experiment_dir,
        args.checkpoint,
        device,
    )
    regularizer_stats = build_latent_regularizer_stats(
        contract,
        split="train",
        speed_percentile=float(specs.get("SpeedGuardPercentile", 95.0)),
    )
    horizons = parse_horizons(args.horizons_years, specs)
    output_dir = experiment_dir / "analysis" / f"checkpoint_{args.checkpoint}"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        export_rows(
            split=split,
            specs=specs,
            contract=contract,
            decoder=decoder,
            flow=flow,
            device=device,
            regularizer_stats=regularizer_stats,
            horizons=horizons,
            subjects_per_diagnosis=int(args.subjects_per_diagnosis),
            composed_step_years=float(args.composed_step_years),
        )
        for split in split_names(args.split)
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(output_dir / "anchor_long_horizon_trajectories.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(output_dir / "anchor_long_horizon_summary.csv", index=False)
    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "decoder_epoch": int(decoder_epoch),
        "splits": list(split_names(args.split)),
        "horizons_years": horizons,
        "rows": int(len(frame)),
        "outputs": [
            "anchor_long_horizon_trajectories.csv",
            "anchor_long_horizon_summary.csv",
        ],
    }
    (output_dir / "anchor_long_horizon_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
