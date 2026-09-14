#!/usr/bin/env python3
"""Condition sweep for one trained checkpoint (BrainODE Fig. 3 / C.3 analog).

Every subject in the split starts from its own first visit and is transported 0-8 years ahead
twice: once with the control condition (d = 0) and once with the disease condition (d = 1).
Decoded mesh volumes give each model's predicted atrophy under both conditions from identical
baselines. That isolates what the condition itself does, which pair errors cannot show.

Outputs:
  <output>.csv   one row per subject x condition x horizon
  <output>.json  per condition and baseline group: mean log-volume change per horizon,
                 least-squares annual rate over 0-2 y and 0-8 y, the paired condition effect
                 rate(d=1) - rate(d=0), and the observed AD-minus-CN first-to-last rate gap
                 for reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import benchmark_common as bc
import dynamics_core as D
import evaluate_dynamics as EV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--trained-view", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-years", type=float, default=8.0)
    parser.add_argument("--step-years", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:2")
    return parser.parse_args()


def annual_rate(horizons: np.ndarray, log_change: np.ndarray, limit: float) -> float:
    mask = horizons <= limit + 1e-9
    h, y = horizons[mask], log_change[mask]
    return float(np.sum(h * y) / np.sum(h * h))  # least squares through the origin (log change is 0 at h = 0)


@torch.no_grad()
def main() -> int:
    args = parse_args()
    output = bc.require_bulk(args.output, "condition sweep")
    device = D.device(args.device)
    parts = D.core()
    C = parts["C"]
    transport, config, payload = D.load_trained_transport(args.checkpoint.resolve(), device)
    representation = config["representation"]
    trained_view = args.trained_view or payload.get("view")
    if trained_view is None:
        raise ValueError("pass --trained-view for checkpoints that do not record one")
    EV.normalization_guard(trained_view, args.view, representation)
    registry = D.view_registry(args.view)
    train_archive = C.load_archive(representation, "train", registry)
    archive = C.load_archive(representation, args.split, registry)
    geometry = C.build_geometry(representation, train_archive, device, registry)
    normalization = bc.read_json(D.view_root(args.view) / "view_manifest.json")["normalization"]
    per_year = 1.0 / (float(normalization["age_max_years"]) - float(normalization["age_min_years"]))

    offsets = archive["subject_visit_offsets"].astype(np.int64)
    first, last = offsets[:-1], offsets[1:] - 1
    z0 = torch.from_numpy(archive["visit_latent_standardized_128"][first].astype(np.float32)).to(device)
    age0 = torch.from_numpy(archive["visit_age_norm_train"][first].astype(np.float32)).to(device)
    horizons = np.round(np.arange(0.0, args.max_years + 1e-9, args.step_years), 4)
    subjects = archive["subject_ids"].astype(str)
    labels = archive["subject_diagnoses"].astype(str)
    cohorts = archive["subject_cohorts"].astype(str)

    rows = []
    for condition in (0.0, 1.0):
        d = torch.full((len(first),), condition, device=device)
        volumes = []
        for h in horizons:
            predicted = []
            for start in range(0, len(first), 256):
                sl = slice(start, start + 256)
                context = (z0[sl], age0[sl]) if D.transport_requires_context(config["method"]) else ()
                code = transport.transport(z0[sl], age0[sl], age0[sl] + float(h) * per_year, d[sl], *context)
                predicted.append(geometry.volume(code))
            volumes.append(torch.cat(predicted).double().cpu().numpy())
        volumes = np.stack(volumes)  # [H, S]
        log_change = np.log(volumes) - np.log(volumes[0:1])
        for hi, h in enumerate(horizons):
            for si in range(len(first)):
                rows.append({"subject_id": subjects[si], "baseline_label": labels[si], "cohort": cohorts[si], "condition": int(condition),
                             "horizon_years": float(h), "predicted_volume_mm3": float(volumes[hi, si]), "log_change_from_h0": float(log_change[hi, si])})
    frame = pd.DataFrame(rows)
    bc.atomic_csv(output, frame)

    years = archive["visit_time_years_from_baseline"].astype(np.float64)
    observed_rate = (np.log(archive["visit_volume_mm3"][last].astype(np.float64)) - np.log(archive["visit_volume_mm3"][first].astype(np.float64))) / np.maximum(years[last] - years[first], 1e-6)
    summary = {"checkpoint": str(args.checkpoint), "method": config["method"], "representation": representation, "view": args.view,
               "split": args.split, "subjects": int(len(first)), "horizons_years": horizons.tolist(), "groups": {}}
    for group in ("all", "CN", "AD"):
        member = np.ones(len(first), bool) if group == "all" else labels == group
        if not member.any():
            continue
        record = {"subjects": int(member.sum()), "observed_first_last_log_rate_mean": float(observed_rate[member].mean())}
        rates = {}
        for condition in (0, 1):
            curve = frame[(frame["condition"] == condition) & frame["baseline_label"].isin(labels[member])].groupby("horizon_years")["log_change_from_h0"].mean()
            record[f"d{condition}_mean_log_change_by_horizon"] = curve.round(6).tolist()
            rates[condition] = {limit: annual_rate(curve.index.to_numpy(), curve.to_numpy(), limit) for limit in (2.0, args.max_years)}
            record[f"d{condition}_annual_log_rate_0_2y"] = rates[condition][2.0]
            record[f"d{condition}_annual_log_rate_0_{args.max_years:g}y"] = rates[condition][args.max_years]
        record["condition_effect_annual_log_rate_0_2y"] = rates[1][2.0] - rates[0][2.0]
        summary["groups"][group] = record
    if "AD" in summary["groups"] and "CN" in summary["groups"]:
        gap = summary["groups"]["AD"]["observed_first_last_log_rate_mean"] - summary["groups"]["CN"]["observed_first_last_log_rate_mean"]
        summary["observed_ad_minus_cn_log_rate"] = gap
        effect = summary["groups"]["all"]["condition_effect_annual_log_rate_0_2y"]
        summary["condition_effect_over_observed_gap"] = effect / gap if abs(gap) > 1e-9 else float("nan")
    bc.atomic_json(output.with_suffix(".json"), summary)
    print(f"sweep {config['method']}/{representation}: condition effect 0-2y {summary['groups']['all']['condition_effect_annual_log_rate_0_2y']:.4f} "
          f"vs observed AD-CN gap {summary.get('observed_ad_minus_cn_log_rate', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
