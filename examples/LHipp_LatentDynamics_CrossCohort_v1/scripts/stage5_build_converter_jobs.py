#!/usr/bin/env python3
"""Stage 5, step 2: job file for the converter line (PLAN Part 5B).

  gate          synthetic onset recovery G5.6 (PCA and Adaptive, s42); test evaluations depend on it
  c1 variants   train_converter_cocycle.py for every variant x its representations x seeds, then val and test
                evaluation with evaluate_converter.py (prefix-only and oracle onsets)
  comparators   C0 (P3 direct_c4) and BrainODE-core (P3 brainode), PCA and Adaptive x seeds, scored on the converter
                view with evaluate_dynamics.py --skip-pairs (task rows only; pair metrics need one label per subject)
  brainode_full PCA x seeds: estimator training, then val and test evaluation with feedback

Job file and index: stage5_brainode_style/jobs/stage5_converter.json, stage5_brainode_style/converter_index.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import benchmark_common as bc
import dynamics_core as D

STAGE5_ROOT = bc.BULK_ROOT / "stage5_brainode_style"
CONFIG = bc.read_json(bc.CONFIG_DIR / "converter_line.json")
VIEW, BASE = CONFIG["view"], CONFIG["base_view"]


def gate_jobs() -> list[dict[str, Any]]:
    jobs = []
    for representation in CONFIG["representations"]:
        marker = STAGE5_ROOT / "converter" / "synthetic_onset" / f"{representation}_s42.json"
        jobs.append({"id": f"gate_G5_6__{representation}", "argv": ["stage5_synthetic_onset.py", "--representation", representation, "--seed", "42"],
                     "weight": 1, "after": [], "done_marker": {"path": str(marker), "key": "gate_passed", "equals": True}})
    return jobs


def c1_runs() -> list[dict[str, Any]]:
    runs = []
    for variant, spec in CONFIG["variants"].items():
        for representation in spec["representations"]:
            for seed in CONFIG["seeds"]:
                name = f"{representation}_{variant}_s{seed}"
                run_dir = D.RUNS_ROOT / VIEW / representation / variant / name
                runs.append({"key": f"{variant}__{representation}__s{seed}", "variant": variant, "representation": representation, "seed": seed,
                             "run_dir": str(run_dir), "checkpoint": str(run_dir / "checkpoints" / "best.pt")})
    return runs


def comparator_runs() -> list[dict[str, Any]]:
    runs = []
    for method in ("direct_c4", "brainode"):
        for representation in CONFIG["representations"]:
            for seed in CONFIG["seeds"]:
                checkpoint = D.RUNS_ROOT / BASE / representation / method / f"{representation}_{method}_s{seed}" / "checkpoints" / "best.pt"
                runs.append({"key": f"{'c0' if method == 'direct_c4' else 'brainode_core'}__{representation}__s{seed}", "method": method,
                             "representation": representation, "seed": seed, "checkpoint": str(checkpoint),
                             "output_root": str(STAGE5_ROOT / "converter" / "comparators" / f"{representation}__{method}__s{seed}")})
    return runs


def brainode_full_runs() -> list[dict[str, Any]]:
    representation = CONFIG["brainode_full"]["representation"]
    return [{"key": f"brainode_full__{representation}__s{seed}", "seed": seed,
             "run_dir": str(D.RUNS_ROOT / VIEW / representation / "brainode_full" / f"{representation}_brainode_full_s{seed}")} for seed in CONFIG["seeds"]]


def jobs() -> list[dict[str, Any]]:
    output = gate_jobs()
    gates = [job["id"] for job in output]
    for run in c1_runs():
        train_id = f"train__{run['key']}"
        output.append({"id": train_id, "weight": 2 if run["representation"] != "pca128" else 1, "after": [],
                       "argv": ["train_converter_cocycle.py", "--variant", run["variant"], "--representation", run["representation"],
                                "--seed", str(run["seed"]), "--run-name", Path(run["run_dir"]).name],
                       "done_marker": {"path": str(Path(run["run_dir"]) / "training_status.json"), "key": "status", "equals": "complete"}})
        for split in ("val", "test"):
            summary = Path(run["run_dir"]) / f"evaluation__{VIEW}" / split / "summary.json"
            output.append({"id": f"evaluate_{split}__{run['key']}", "weight": 1, "after": [train_id] + (gates if split == "test" else []),
                           "argv": ["evaluate_converter.py", "--checkpoint", run["checkpoint"], "--split", split], "done_marker": str(summary)})
    for run in comparator_runs():
        for split in ("val", "test"):
            destination = Path(run["output_root"]) / split
            output.append({"id": f"evaluate_{split}__{run['key']}", "weight": 1, "after": gates if split == "test" else [],
                           "argv": ["evaluate_dynamics.py", "--checkpoint", run["checkpoint"], "--view", VIEW, "--trained-view", BASE, "--split", split,
                                    "--skip-pairs", "--bootstrap", "2000", "--output-dir", str(destination)],
                           "done_marker": str(destination / "summary.json")})
    for run in brainode_full_runs():
        train_id = f"train__{run['key']}"
        output.append({"id": train_id, "weight": 1, "after": [], "argv": ["train_brainode_full.py", "--seed", str(run["seed"])],
                       "done_marker": {"path": str(Path(run["run_dir"]) / "training_status.json"), "key": "status", "equals": "complete"}})
        for split in ("val", "test"):
            output.append({"id": f"evaluate_{split}__{run['key']}", "weight": 1, "after": [train_id] + (gates if split == "test" else []),
                           "argv": ["evaluate_brainode_full.py", "--run-dir", run["run_dir"], "--split", split],
                           "done_marker": str(Path(run["run_dir"]) / f"evaluation__{VIEW}" / split / "summary.json")})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    job_list = jobs()
    bc.atomic_json(bc.require_bulk(STAGE5_ROOT / "converter_index.json"), {
        "view": VIEW, "base_view": BASE, "c1_runs": c1_runs(), "comparator_runs": comparator_runs(), "brainode_full_runs": brainode_full_runs()})
    path = bc.atomic_json(bc.require_bulk(STAGE5_ROOT / "jobs" / "stage5_converter.json"), {"name": "stage5_converter", "jobs": job_list})
    print(f"converter: {len(job_list)} jobs ({sum(j['id'].startswith('train__') for j in job_list)} trainings) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
