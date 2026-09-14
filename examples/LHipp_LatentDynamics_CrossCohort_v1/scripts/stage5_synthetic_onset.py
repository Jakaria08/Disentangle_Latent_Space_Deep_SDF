#!/usr/bin/env python3
"""Stage 5 step 2, gate G5.6: synthetic onset recovery for the converter cocycle (PLAN Part 5B, E-C2).

Can a conversion onset be recovered from visits like the real converters', before any real converter is tested?

* generator: C0 = the P3 pooled direct_c4 checkpoint (stable subjects only). A synthetic converter starts from the
  first visit of a CN-stable validation subject and follows a real train converter's visit schedule and conversion
  window; its true onset is drawn uniformly inside the window, and the condition switches along the dose path
  c_i(tau) = sigmoid((tau - tau_i)/w);
* noise: Gaussian per code coordinate with the RMS residual of C0 on stable validation first-to-last pairs;
* recovery: grid posterior of the onset on the window (uniform prior, known noise) with the same frozen C0 heads;
  onset estimate = posterior median, 80% interval = 10-90% quantiles.

Reported: median/mean/90th-percentile absolute error in years, 80% interval coverage, and the error of simply
taking the window midpoint (what the window alone gives). Gate: median absolute error <= 0.5 y at w = 0.25 y.
Only train (schedules) and val (baselines, noise) are opened.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
import torch

import benchmark_common as bc
import converter_line as L
import dynamics_core as D
import train_converter_cocycle as TC

CONFIG = bc.read_json(bc.CONFIG_DIR / "converter_line.json")
SETTINGS = CONFIG["synthetic_recovery"]
OUTPUT_ROOT = bc.BULK_ROOT / "stage5_brainode_style" / "converter" / "synthetic_onset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--representation", required=True, choices=CONFIG["representations"])
    parser.add_argument("--seed", type=int, default=42, choices=CONFIG["seeds"])
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--subjects", type=int, default=int(SETTINGS["subjects"]))
    return parser.parse_args()


@torch.no_grad()
def c0_noise_sigma(flow, archive, values) -> float:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    groups = archive["subject_trajectory_groups"].astype(str)
    members = [i for i, g in enumerate(groups) if g in ("CN-stable", "AD-stable")]
    first = torch.as_tensor(offsets[members], device=values["z"].device)
    last = torch.as_tensor(offsets[np.asarray(members) + 1] - 1, device=values["z"].device)
    predicted = flow.transport(values["z"][first], values["age"][first], values["age"][last], values["label"][first])
    return float(torch.sqrt(torch.mean((predicted - values["z"][last]) ** 2)))


def schedules(archive) -> list[dict[str, Any]]:
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    years = archive["visit_time_years_from_baseline"].astype(np.float64)
    output = []
    for index, group in enumerate(archive["subject_trajectory_groups"].astype(str)):
        if group not in L.AD_CONVERTERS:
            continue
        start, end = int(offsets[index]), int(offsets[index + 1])
        labels = archive["visit_trajectory_labels"][start:end].tolist()
        first = labels.index("AD")
        output.append({"subject": str(archive["subject_ids"][index]), "years": years[start:end] - years[start],
                       "window_years": (years[start + first - 1] - years[start], years[start + first] - years[start])})
    return output


def recover(flow, width_years: float, age_range: float, sigma: float, plan, baselines, values, count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    width = width_years / age_range
    rows = []
    for number in range(count):
        schedule = plan[int(rng.integers(len(plan)))]
        visit = int(baselines[int(rng.integers(len(baselines)))])
        z0, age0 = values["z"][visit], float(values["age"][visit])
        ages = torch.as_tensor(age0 + schedule["years"][1:] / age_range, dtype=torch.float32, device=z0.device)
        a, b = (age0 + value / age_range for value in schedule["window_years"])
        truth = float(rng.uniform(a, b))
        clean = L.dosed_trajectory(flow, z0, age0, ages, torch.tensor([truth], device=z0.device), width)[0]
        observed = clean + torch.as_tensor(rng.normal(0.0, sigma, clean.shape), dtype=torch.float32, device=z0.device)
        grid, posterior = L.onset_posterior(flow, z0, age0, ages, observed, sigma, (a, b), width, int(SETTINGS["grid_points"]))
        summary = L.posterior_summary(grid, posterior, SETTINGS["interval"])
        rows.append({"synthetic": number, "schedule_subject": schedule["subject"], "visits": len(schedule["years"]),
                     "window_years": (b - a) * age_range, "true_onset_years": (truth - a) * age_range,
                     "estimate_years": (summary["median"] - a) * age_range,
                     "abs_error_years": abs(summary["median"] - truth) * age_range,
                     "midpoint_abs_error_years": abs(0.5 * (a + b) - truth) * age_range,
                     "interval_width_years": (summary["high"] - summary["low"]) * age_range,
                     "covered": bool(summary["low"] - 1e-6 <= truth <= summary["high"] + 1e-6)})
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    device = D.device(args.device)
    C = D.core()["C"]
    view = CONFIG["view"]
    registry = D.view_registry(view)
    train = C.load_archive(args.representation, "train", registry)
    val = C.load_archive(args.representation, "val", registry)
    D.assert_no_test_leakage(view, [train, val])
    checkpoint = TC.init_checkpoint(args.representation, args.seed)
    flow, _config, _payload = D.load_trained_transport(checkpoint, device)
    geometry = C.build_geometry(args.representation, train, device, registry)
    values = TC.prepared(val, geometry, device, 512)
    sigma = c0_noise_sigma(flow, val, values)
    normalization = bc.read_json(D.view_root(view) / "view_manifest.json")["normalization"]
    age_range = float(normalization["age_max_years"]) - float(normalization["age_min_years"])
    offsets = val["subject_visit_offsets"].astype(np.int64)
    baselines = [int(offsets[i]) for i, g in enumerate(val["subject_trajectory_groups"].astype(str)) if g == "CN-stable"]
    plan = schedules(train)

    output = bc.require_bulk(OUTPUT_ROOT)
    report: dict[str, Any] = {"checkpoint": str(checkpoint), "representation": args.representation, "seed": args.seed, "noise_sigma_code": sigma,
                              "schedules": len(plan), "baselines": len(baselines), "settings": SETTINGS, "widths": {}}
    for width_years in (float(CONFIG["dose"]["width_years"]), float(CONFIG["dose"]["width_years_sensitivity"])):
        frame = recover(flow, width_years, age_range, sigma, plan, baselines, values, args.subjects, int(SETTINGS["seed"]))
        name = f"{args.representation}_s{args.seed}_w{width_years:g}"
        bc.atomic_csv(output / f"{name}.csv", frame)
        report["widths"][f"{width_years:g}"] = {
            "median_abs_error_years": float(frame["abs_error_years"].median()), "mean_abs_error_years": float(frame["abs_error_years"].mean()),
            "p90_abs_error_years": float(frame["abs_error_years"].quantile(0.9)), "coverage_80": float(frame["covered"].mean()),
            "median_midpoint_abs_error_years": float(frame["midpoint_abs_error_years"].median()),
            "median_window_years": float(frame["window_years"].median()), "median_interval_width_years": float(frame["interval_width_years"].median()),
        }
    main_width = report["widths"][f"{float(CONFIG['dose']['width_years']):g}"]
    report["gate"] = {"id": "G5.6", "check": f"median |onset error| <= {SETTINGS['gate_median_abs_error_years']} y at w = {CONFIG['dose']['width_years']} y",
                      "passed": bool(main_width["median_abs_error_years"] <= float(SETTINGS["gate_median_abs_error_years"]))}
    report["gate_passed"] = report["gate"]["passed"]  # top-level flag for the orchestrator's done marker
    # The window midpoint alone is the no-information reference: recovery that is no better than it means the visits
    # do not localize the onset inside the window, even when the gate passes.
    report["recovery_better_than_window_midpoint"] = bool(main_width["median_abs_error_years"] < main_width["median_midpoint_abs_error_years"])
    bc.atomic_json(output / f"{args.representation}_s{args.seed}.json", report)
    print(f"G5.6 {'PASS' if report['gate']['passed'] else 'FAIL'} | sigma={sigma:.4f} | " + " | ".join(
        f"w={w}: median err {v['median_abs_error_years']:.3f} y (window midpoint {v['median_midpoint_abs_error_years']:.3f} y), 80% coverage {v['coverage_80']:.2f}"
        for w, v in report["widths"].items()))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
