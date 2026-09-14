#!/usr/bin/env python3
"""Stage 5, step 1: job files for the ablations (A1-A4) and the sensitivity arms.

Suites (job files and reports under stage5_brainode_style/; runs under runs/<view>/...):

  ablations    on p0_internal_adni; every training is followed by val and test evaluation and a
               test-split condition sweep
               A1  exact coboundary and volume coboundary v2, PCA + SpiralNet, s42, trained to completion
               A2  BrainODE-V (BrainODE field + cocycle objective), PCA, s42-s44
               A3  cocycle without the disease head, PCA, s42-s44
               A4  unconditional Latent ODE (faithful and residual), PCA, s42-s44
  sensitivity  pooled PCA basis (fit, encode, p3_pooled archives), then the five dynamics x
               s42-s44 on pca128_pooled with val and test evaluation; plus min-epoch-15 test
               evaluations of every stage-4 cocycle run (P2 internal, P2 cross-fit, P3, P4)

The BrainODE-matched age subset (65-95 at first visit) needs no jobs: the stage 5 report rescores
stored per-subject task rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import benchmark_common as bc
import dynamics_core as D
import stage2_build_jobs as S2

STAGE5_ROOT = bc.BULK_ROOT / "stage5_brainode_style"
STAGE4_INDEX = bc.BULK_ROOT / "stage4_crosscohort" / "results_index.json"
P0, P3 = "p0_internal_adni", "p3_pooled"
SEEDS = (42, 43, 44)
UNCONDITIONAL = ["--override", "model.condition_in_encoder=false", "--override", "model.condition_in_dynamics=false"]


def evaluation_jobs(key: str, run: dict[str, Any], view: str, after: list[str], splits=("val", "test"), name: str = "evaluation",
                    checkpoint: str | None = None) -> list[dict[str, Any]]:
    jobs = []
    for split in splits:
        output = Path(run["run_dir"]) / f"{name}__{view}" / split
        argv = ["evaluate_dynamics.py", "--checkpoint", checkpoint or run["checkpoint"], "--view", view, "--split", split,
                "--trained-view", run["view"], "--bootstrap", "2000", "--output-dir", str(output)]
        jobs.append({"id": f"{name}_{split}__{key}", "argv": argv, "weight": 1, "after": after, "done_marker": str(output / "summary.json")})
    return jobs


def training_job(key: str, run: dict[str, Any]) -> dict[str, Any]:
    marker = {"path": str(Path(run["run_dir"]) / "training_status.json"), "key": "status", "equals": "complete"}
    return {"id": f"train__{key}", "argv": run["train_argv"], "weight": run["weight"], "after": run.get("after", []), "done_marker": marker}


def run_entry(ablation: str, view: str, representation: str, method: str, run_name: str, seed: int, train_argv: list[str],
              weight: int = 1, after: list[str] | None = None) -> dict[str, Any]:
    run_dir = D.RUNS_ROOT / view / representation / method / run_name
    return {"ablation": ablation, "key": f"{ablation}__{representation}__{run_name}", "view": view, "representation": representation,
            "method": method, "seed": seed, "run_dir": str(run_dir), "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
            "train_argv": train_argv, "weight": weight, "after": after or []}


def ablation_runs() -> list[dict[str, Any]]:
    runs = []
    for variant, method in (("exact", "exact_coboundary_c4"), ("volume_v2", "volume_exact_coboundary_c4_v2")):
        for representation in ("pca128", "spiralnet128"):
            name = f"{representation}_{method}_complete_s42"
            argv = ["train_coboundary_ablation.py", "--variant", variant, "--view", P0, "--representation", representation,
                    "--seed", "42", "--run-name", name]
            runs.append(run_entry("A1", P0, representation, method, name, 42, argv, weight=1 if representation == "pca128" else 2))
    for ablation, method in (("A2", "brainode_v"), ("A3", "direct_c4_no_disease")):
        for seed in SEEDS:
            name = f"pca128_{method}_s{seed}"
            argv = ["train_cocycle.py", "--method", method, "--view", P0, "--representation", "pca128", "--seed", str(seed),
                    "--runs-root", str(D.RUNS_ROOT), "--run-name", name]
            runs.append(run_entry(ablation, P0, "pca128", method, name, seed, argv))
    for method in D.LATENT_ODE_METHODS:
        for seed in SEEDS:
            name = f"pca128_{method}_unconditional_s{seed}"
            argv = ["train_rubanova_latent_ode.py", "--method", method, "--view", P0, "--representation", "pca128", "--seed", str(seed),
                    "--runs-root", str(D.RUNS_ROOT), "--run-name", name, *UNCONDITIONAL]
            runs.append(run_entry("A4", P0, "pca128", method, name, seed, argv))
    return runs


def ablation_jobs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for run in runs:
        train = training_job(run["key"], run)
        jobs.append(train)
        jobs += evaluation_jobs(run["key"], run, P0, [train["id"]])
        output = STAGE5_ROOT / "ablations" / "condition_sweeps" / f"{run['key']}.csv"
        jobs.append({"id": f"sweep__{run['key']}", "weight": 1, "after": [train["id"]], "done_marker": str(output),
                     "argv": ["stage3_condition_sweep.py", "--checkpoint", run["checkpoint"], "--trained-view", P0, "--view", P0,
                              "--split", "test", "--output", str(output)]})
    return jobs


def pooled_runs() -> list[dict[str, Any]]:
    runs = []
    for seed in SEEDS:
        for method in D.METHODS:
            name = f"pca128_pooled_{method}_s{seed}"
            train = S2.train_job("unused", method, "pca128_pooled", name, D.RUNS_ROOT, seed=seed, view=P3)
            runs.append(run_entry("S-pooled-PCA", P3, "pca128_pooled", method, name, seed, train["argv"], after=["pooled_pca_basis"]))
    return runs


def min_epoch_runs() -> tuple[list[dict[str, Any]], list[str]]:
    runs, missing = [], []
    for run in bc.read_json(STAGE4_INDEX)["trained_runs"]:
        if run["method"] != "direct_c4" or run["blocked"]:
            continue
        checkpoint = Path(run["run_dir"]) / "checkpoints" / "best_min_epoch.pt"
        if checkpoint.is_file():
            runs.append(run | {"min_epoch_checkpoint": str(checkpoint)})
        else:
            missing.append(run["key"])
    return runs, missing


def sensitivity_jobs(pooled: list[dict[str, Any]], min_epoch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = [{"id": "pooled_pca_basis", "argv": ["stage5_pooled_pca.py"], "weight": 1, "after": [],
             "done_marker": {"path": str(STAGE5_ROOT / "sensitivity" / "pooled_pca" / "report.json"), "key": "status", "equals": "complete"}}]
    for run in pooled:
        train = training_job(run["key"], run)
        jobs.append(train)
        jobs += evaluation_jobs(run["key"], run, P3, [train["id"]])
    for run in min_epoch:
        jobs += evaluation_jobs(run["key"], run, run["view"], [], splits=("test",), name="evaluation_min_epoch",
                                checkpoint=run["min_epoch_checkpoint"])
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suites", nargs="+", choices=("ablations", "sensitivity"))
    args = parser.parse_args()
    ablations, pooled = ablation_runs(), pooled_runs()
    min_epoch, missing = min_epoch_runs()
    bc.atomic_json(bc.require_bulk(STAGE5_ROOT / "results_index.json"), {
        "ablation_runs": ablations, "pooled_pca_runs": pooled, "min_epoch_runs": min_epoch, "min_epoch_missing_checkpoint": missing,
        "age_subset": {"years": [65, 95], "rule": "age at the first visit, rescored from stored task_rows.csv in the stage 5 report"},
    })
    for suite in args.suites:
        jobs = ablation_jobs(ablations) if suite == "ablations" else sensitivity_jobs(pooled, min_epoch)
        path = bc.atomic_json(bc.require_bulk(STAGE5_ROOT / "jobs" / f"stage5_{suite}.json"), {"name": f"stage5_{suite}", "jobs": jobs})
        trainings = sum(job["id"].startswith("train__") for job in jobs)
        print(f"{suite}: {len(jobs)} jobs ({trainings} trainings) -> {path}")
    if missing:
        print(f"min-epoch checkpoints missing for {len(missing)} cocycle runs: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
