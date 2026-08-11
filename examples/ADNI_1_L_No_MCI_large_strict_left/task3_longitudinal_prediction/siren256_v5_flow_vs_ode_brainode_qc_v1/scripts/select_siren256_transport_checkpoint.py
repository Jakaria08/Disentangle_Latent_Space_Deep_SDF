#!/usr/bin/env python3
"""Select one direct-flow checkpoint from validation-only surface evaluations."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

from siren256_common import read_json, write_json


def panel_metric(summary: dict[str, Any], panel: str, metric: str) -> float:
    try:
        return float(summary["subject_macro"][panel][metric])
    except (KeyError, TypeError, ValueError) as error:
        raise KeyError(f"Missing {panel}.{metric} in validation summary for {summary.get('checkpoint')}") from error


def first_last_metric(summary: dict[str, Any]) -> tuple[float, bool]:
    """Use all-pair geometry only for deliberately truncated smoke evaluations."""
    first = panel_metric(summary, "first_last_subject_macro", "registered_normal_mae")
    if math.isfinite(first):
        return first, False
    return panel_metric(summary, "all_subject_macro", "registered_normal_mae"), True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config = read_json(run / "config.json")
    manifest = read_json(run / "validation_candidates.json")
    candidates = [candidate for candidate in manifest.get("candidates", []) if candidate.get("shortlisted")]
    if not candidates:
        raise RuntimeError("No shortlisted validation candidates. Train with UseParetoValidationSelection=true.")
    summaries: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        tag = str(candidate["tag"])
        path = run / "evaluation" / "val" / tag / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Evaluate every shortlisted validation checkpoint first; missing {path}")
        summaries[tag] = read_json(path)
    baseline = next((candidate for candidate in candidates if candidate.get("nochange")), None)
    if baseline is None:
        raise RuntimeError("Validation shortlist does not include epoch-0 no-change baseline.")
    baseline_summary = summaries[str(baseline["tag"])]
    baseline_first, baseline_first_fallback = first_last_metric(baseline_summary)
    baseline_all = panel_metric(baseline_summary, "all_subject_macro", "registered_normal_mae")
    baseline_volume = panel_metric(baseline_summary, "all_subject_macro", "volume_relative_error")
    first_limit = float(config.get("ValidationSurfaceMaxFirstLastGeometryRatio", config.get("ParetoMaxFirstLastGeometryRatio", 1.0)))
    all_limit = float(config.get("ValidationSurfaceMaxAllPairGeometryRatio", config.get("ParetoMaxAllPairGeometryRatio", 1.0)))
    require_volume = bool(config.get("ValidationSurfaceRequireVolumeImprovement", True))
    assessed: list[dict[str, Any]] = []
    for candidate in candidates:
        tag = str(candidate["tag"])
        summary = summaries[tag]
        first, first_fallback = first_last_metric(summary)
        all_geometry = panel_metric(summary, "all_subject_macro", "registered_normal_mae")
        volume = panel_metric(summary, "all_subject_macro", "volume_relative_error")
        rate = panel_metric(summary, "all_subject_macro", "annual_log_volume_rate_mae")
        first_ratio, all_ratio, volume_ratio = first / max(baseline_first, 1.0e-12), all_geometry / max(baseline_all, 1.0e-12), volume / max(baseline_volume, 1.0e-12)
        feasible = bool(candidate.get("nochange")) or (first_ratio <= first_limit and all_ratio <= all_limit and (not require_volume or volume_ratio < 1.0))
        assessed.append({**candidate, "validation_surface_first_last_geometry": first, "validation_surface_first_last_fallback_to_all": first_fallback, "validation_surface_all_geometry": all_geometry, "validation_surface_volume_relative_error": volume, "validation_surface_annual_rate_mae": rate, "validation_surface_first_last_ratio": first_ratio, "validation_surface_all_geometry_ratio": all_ratio, "validation_surface_volume_ratio": volume_ratio, "validation_surface_feasible": feasible})
    learned = [candidate for candidate in assessed if not candidate["nochange"] and candidate["validation_surface_feasible"]]
    selected = min(learned, key=lambda candidate: (candidate["validation_surface_volume_relative_error"], candidate["validation_surface_first_last_geometry"], candidate["validation_surface_annual_rate_mae"], candidate["epoch"])) if learned else next(candidate for candidate in assessed if candidate["nochange"])
    source = run / "checkpoints" / f"{selected['checkpoint']}.pt"
    destination = run / "checkpoints" / "best.pt"
    if not source.exists():
        raise FileNotFoundError(source)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    result = {
        "selection_split": "val",
        "selection_rule": "min decoded validation volume error subject to decoded geometry gates; no-change fallback",
        "criteria": {"max_first_last_geometry_ratio": first_limit, "max_all_geometry_ratio": all_limit, "require_volume_improvement": require_volume},
        "baseline": {"checkpoint": baseline["checkpoint"], "first_last_geometry": baseline_first, "first_last_fallback_to_all": baseline_first_fallback, "all_geometry": baseline_all, "volume_relative_error": baseline_volume},
        "selected_checkpoint": selected["checkpoint"],
        "selected_epoch": int(selected["epoch"]),
        "selected_is_nochange": bool(selected["nochange"]),
        "candidates": assessed,
    }
    write_json(run / "selection" / "validation_surface_selection.json", result)
    report_path = run / "validation_report.json"
    if report_path.exists():
        report = read_json(report_path)
        report.update({"selection_status": "complete", "selected_checkpoint": "best.pt", "selected_epoch": int(selected["epoch"]), "validation_surface_selection": result})
        write_json(report_path, report)
    print(f"selected={selected['checkpoint']} epoch={selected['epoch']} nochange={selected['nochange']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
