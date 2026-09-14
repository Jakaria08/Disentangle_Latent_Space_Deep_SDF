#!/usr/bin/env python3
"""Stage 3 (ADNI internal matrix, protocol P0): job files and the results index.

Cells: 4 representations x 5 dynamics x seeds 42, 43, 44 = 60 seed-runs on p0_internal_adni.

* 54 are trained here with the stage 2 trainers and recipes (Latent ODE variants use their
  validation-selected settings automatically).
* 6 are the August seed-42 plain ODE / BrainODE anchors for PCA, SpiralNet and Adaptive, reused
  unchanged (G2.7a proved the evaluator reproduces them).
* The seed-42 cocycles are retrained rather than reused: the August trainer never saved the
  min-epoch-15 checkpoint the sensitivity analysis needs. The PCA retrain was bit-identical to its
  anchor in stage 2; the report records how close SpiralNet and Adaptive come.

Suites:
  training  every training run, then val and test evaluation of best.pt (and of best_min_epoch.pt
            for cocycles), plus val and test evaluation of the reused anchors
  analysis  condition sweeps for every seed-run, then the stage 3 report (built separately so the
            analysis scripts can change without touching running training jobs)

The test split is evaluated exactly once per checkpoint; evaluate_dynamics.py refuses to overwrite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import benchmark_common as bc
import dynamics_core as D
import stage2_build_jobs as S2

STAGE3_ROOT = bc.BULK_ROOT / "stage3_adni"
VIEW = "p0_internal_adni"
SEEDS = (42, 43, 44)
AUGUST = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version/training")


def reused_anchor(representation: str, method: str, seed: int) -> Path | None:
    if seed == 42 and representation != "lamm128" and method in ("plain_ode", "brainode"):
        summary = bc.read_json(AUGUST / representation / method / f"{representation}_{method}_s42" / "evaluation" / "test" / "summary.json")
        return Path(summary["checkpoint"])
    return None


def evaluation(job_id: str, checkpoint: Path, split: str, output: Path, after: list[str], trained_view: str | None = None) -> dict[str, Any]:
    argv = ["evaluate_dynamics.py", "--checkpoint", str(checkpoint), "--view", VIEW, "--split", split,
            "--bootstrap", "2000", "--output-dir", str(output)]
    if trained_view:
        argv += ["--trained-view", trained_view]
    return {"id": job_id, "argv": argv, "weight": 1, "after": after, "done_marker": str(output / "summary.json")}


def seed_runs() -> list[dict[str, Any]]:
    runs = []
    for seed in SEEDS:  # seed 42 first so a complete first replicate lands early
        for representation in bc.REPRESENTATIONS:
            for method in D.METHODS:
                key = f"{representation}__{method}__s{seed}"
                anchor = reused_anchor(representation, method, seed)
                if anchor is not None:
                    root = STAGE3_ROOT / "anchor_evaluations" / key
                    runs.append({"key": key, "representation": representation, "method": method, "seed": seed, "source": "august_anchor",
                                 "checkpoint": str(anchor), "trained_view": VIEW,
                                 "evaluations": {"val": str(root / "val"), "test": str(root / "test")}})
                    continue
                run_dir = D.RUNS_ROOT / VIEW / representation / method / f"{representation}_{method}_s{seed}"
                entry = {"key": key, "representation": representation, "method": method, "seed": seed, "source": "trained_stage3",
                         "run_dir": str(run_dir), "checkpoint": str(run_dir / "checkpoints" / "best.pt"), "trained_view": VIEW,
                         "evaluations": {split: str(run_dir / f"evaluation__{VIEW}" / split) for split in ("val", "test")}}
                if method == "direct_c4":
                    entry["min_epoch_checkpoint"] = str(run_dir / "checkpoints" / "best_min_epoch.pt")
                    entry["min_epoch_evaluations"] = {split: str(run_dir / f"evaluation_min_epoch__{VIEW}" / split) for split in ("val", "test")}
                runs.append(entry)
    return runs


def training_suite(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for run in runs:
        key = run["key"]
        if run["source"] == "august_anchor":
            for split in ("val", "test"):
                jobs.append(evaluation(f"eval_{split}__{key}", Path(run["checkpoint"]), split, Path(run["evaluations"][split]), [], VIEW))
            continue
        train = S2.train_job(f"train__{key}", run["method"], run["representation"], Path(run["run_dir"]).name, D.RUNS_ROOT, seed=run["seed"], view=VIEW)
        jobs.append(train)
        for split in ("val", "test"):
            jobs.append(evaluation(f"eval_{split}__{key}", Path(run["checkpoint"]), split, Path(run["evaluations"][split]), [train["id"]]))
        if "min_epoch_checkpoint" in run:
            for split in ("val", "test"):
                jobs.append(evaluation(f"eval_min_epoch_{split}__{key}", Path(run["min_epoch_checkpoint"]), split,
                                       Path(run["min_epoch_evaluations"][split]), [train["id"]]))
    return jobs


def analysis_suite(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for run in runs:
        output = STAGE3_ROOT / "condition_sweeps" / f"{run['key']}.csv"
        argv = ["stage3_condition_sweep.py", "--checkpoint", run["checkpoint"], "--trained-view", VIEW, "--view", VIEW,
                "--split", "test", "--output", str(output)]
        jobs.append({"id": f"sweep__{run['key']}", "argv": argv, "weight": 1, "after": [], "done_marker": str(output)})
    jobs.append({"id": "stage3_report", "argv": ["stage3_report_adni.py"], "weight": 1, "after": [j["id"] for j in jobs],
                 "done_marker": str(STAGE3_ROOT / "reports" / "stage3_adni_report.md")})
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suites", nargs="+", choices=("training", "analysis"))
    args = parser.parse_args()
    runs = seed_runs()
    bc.atomic_json(bc.require_bulk(STAGE3_ROOT / "results_index.json"), {"view": VIEW, "seeds": list(SEEDS), "runs": runs})
    print(f"results index: {len(runs)} seed-runs ({sum(r['source'] == 'august_anchor' for r in runs)} reused anchors)")
    for suite in args.suites:
        jobs = training_suite(runs) if suite == "training" else analysis_suite(runs)
        path = bc.require_bulk(STAGE3_ROOT / "jobs" / f"adni_{suite}.json")
        bc.atomic_json(path, {"name": f"adni_{suite}", "jobs": jobs})
        trains = sum(j["id"].startswith("train__") for j in jobs)
        print(f"{suite}: {len(jobs)} jobs ({trains} training) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
