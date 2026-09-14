#!/usr/bin/env python3
"""Stage 2 gate G2.10: pick a Latent ODE variant's setting from its validation-only search.

Reads every trial of the method's search job file. All trials must be complete. The winner is
the lowest best_validation_selection_score (ADNI val, never test). It is written to
configs/recipes/<method>_selected_overrides.json, which dynamics_core.load_recipe applies to
every later run of that method (every representation, view and seed). A report goes to
stage2_validation/reports/<method>_search.{json,md}.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import benchmark_common as bc
import dynamics_core as D


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="latent_ode", choices=D.LATENT_ODE_METHODS)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--device", default=None, help="Ignored; accepted because the orchestrator appends it.")
    args = parser.parse_args()
    spec = bc.read_json(args.jobs)
    trials = []
    for job in spec["jobs"]:
        if "overrides" not in job:
            continue
        status_path = Path(job["run_dir"]) / "training_status.json"
        status = bc.read_json(status_path) if status_path.is_file() else {}
        if status.get("status") != "complete":
            raise RuntimeError(f"{job['id']} is not complete ({status.get('status')}); refusing to select from a partial search")
        trials.append({"id": job["id"], "run_dir": job["run_dir"], "overrides": job["overrides"], "best_epoch": status["best_epoch"],
                       "stopped_epoch": status.get("epoch"), "score": float(status["best_validation_selection_score"]),
                       "minutes": status.get("elapsed_minutes")})
    trials.sort(key=lambda t: t["score"])
    winner = trials[0]
    bc.atomic_json(D.RECIPE_DIR / f"{args.method}_selected_overrides.json", {
        "method": args.method,
        "overrides": winner["overrides"],
        "selected_trial": winner["id"],
        "validation_selection_score": winner["score"],
        "selection": "lowest ADNI validation first-to-last selection score over the stage 2 search; test never used",
        "view": "p0_internal_adni", "representation": "pca128", "seed": 42, "written": time.strftime("%F %T"),
    })
    report = bc.require_bulk(D.VALIDATION_ROOT / "reports")
    bc.atomic_json(report / f"{args.method}_search.json", {"trials": trials, "winner": winner})
    keys = list(winner["overrides"])
    lines = [f"# {args.method} search (ADNI PCA-128, validation only)", "",
             "Score: macro CN/AD first-to-last decoded shape ratio vs no-change (1.0 = no-change; lower is better).", "",
             "| rank | trial | " + " | ".join(keys) + " | best epoch | stopped | val score | minutes |", "|" + "---|" * (len(keys) + 6)]
    for rank, t in enumerate(trials, start=1):
        minutes = f"{t['minutes']:.1f}" if isinstance(t["minutes"], (int, float)) else "-"
        lines.append(f"| {rank} | {t['id']} | " + " | ".join(str(t["overrides"][k]) for k in keys)
                     + f" | {t['best_epoch']} | {t['stopped_epoch']} | {t['score']:.5f} | {minutes} |")
    bc.atomic_write_text(report / f"{args.method}_search.md", "\n".join(lines) + "\n")
    print(f"{args.method} winner {winner['id']}: {winner['overrides']} score={winner['score']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
