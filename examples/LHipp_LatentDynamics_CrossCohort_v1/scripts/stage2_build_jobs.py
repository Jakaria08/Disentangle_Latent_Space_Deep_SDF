#!/usr/bin/env python3
"""Write stage 2 job files for orchestrate_dynamics.py.

Suites (outputs under stage2_validation/ on the bulk root):

  smoke             16 cells x ADNI view: 1-epoch training + validation evaluation (G2.9)
  anchor_retrain    PCA direct_c4 and PCA plain_ode retrained from scratch on the ADNI view
                    with the stage-3 recipe and seed 42, then evaluated on test (G2.7b)
  latent_ode_search           16 validation-only faithful Latent ODE trials on ADNI PCA-128 + selection (G2.10)
  latent_ode_residual_search  the same 16 trial settings for the residual variant + selection (G2.10)
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path
from typing import Any

import benchmark_common as bc
import dynamics_core as D

TRAINERS = {"direct_c4": "train_cocycle.py", "plain_ode": "train_ode_transport.py", "brainode": "train_ode_transport.py",
            "latent_ode": "train_rubanova_latent_ode.py", "latent_ode_residual": "train_rubanova_latent_ode.py"}
VIEW = "p0_internal_adni"


def weight(method: str, representation: str) -> int:
    """Capacity units: a decoder-in-loss cocycle on a neural decoder is the heaviest job."""
    return 2 if method == "direct_c4" and representation != "pca128" else 1


def train_job(job_id: str, method: str, representation: str, run_name: str, runs_root: Path, seed: int = 42,
              extra: list[str] | None = None, view: str = VIEW) -> dict[str, Any]:
    argv = [TRAINERS[method], "--view", view, "--representation", representation, "--seed", str(seed),
            "--runs-root", str(runs_root), "--run-name", run_name, *(extra or [])]
    if method in ("plain_ode", "brainode", *D.LATENT_ODE_METHODS):
        argv += ["--method", method]
    run_dir = runs_root / view / representation / method / run_name
    return {"id": job_id, "argv": argv, "weight": weight(method, representation), "after": [],
            "done_marker": {"path": str(run_dir / "training_status.json"), "key": "status", "equals": "complete"},
            "run_dir": str(run_dir)}


def eval_job(job_id: str, train: dict[str, Any], split: str, bootstrap: int, after: list[str], view: str = VIEW) -> dict[str, Any]:
    run_dir = Path(train["run_dir"])
    output = run_dir / f"evaluation__{view}" / split
    return {"id": job_id, "argv": ["evaluate_dynamics.py", "--checkpoint", str(run_dir / "checkpoints" / "best.pt"), "--view", view,
                                   "--split", split, "--bootstrap", str(bootstrap), "--output-dir", str(output)],
            "weight": 1, "after": after, "done_marker": str(output / "summary.json")}


def smoke() -> list[dict[str, Any]]:
    root = D.VALIDATION_ROOT / "smoke_runs"
    jobs = []
    for representation in bc.REPRESENTATIONS:
        for method in D.METHODS:
            tid = f"smoke_train__{representation}__{method}"
            train = train_job(tid, method, representation, f"smoke_{representation}_{method}", root, extra=["--smoke"])
            jobs += [train, eval_job(f"smoke_eval__{representation}__{method}", train, "val", 50, [tid])]
    return jobs


def anchor_retrain() -> list[dict[str, Any]]:
    root = D.VALIDATION_ROOT / "anchor_retrain"
    jobs = []
    for method in ("direct_c4", "plain_ode"):
        tid = f"retrain__pca128__{method}"
        train = train_job(tid, method, "pca128", f"pca128_{method}_s42_retrain", root)
        jobs += [train, eval_job(f"retrain_eval__pca128__{method}", train, "test", 2000, [tid])]
    jobs.append({"id": "retrain_compare", "argv": ["stage2_verify_anchors.py", "--retrain"], "weight": 1,
                 "after": [j["id"] for j in jobs if j["id"].startswith("retrain_eval")],
                 "done_marker": str(D.VALIDATION_ROOT / "reports" / "anchor_retrain_comparison.json")})
    return jobs


def latent_search(method: str) -> list[dict[str, Any]]:
    """Trial settings come from the recipe's grid and sampler seed; both variants share them."""
    recipe = bc.read_json(D.RECIPE_DIR / f"{method}.json")
    search = recipe["search"]
    keys = list(search["space"])
    grid = list(itertools.product(*(search["space"][key] for key in keys)))
    chosen = random.Random(int(search["sampler_seed"])).sample(grid, int(search["trials"]))
    root = D.VALIDATION_ROOT / f"{method}_search"
    prefix = "lres" if method == "latent_ode_residual" else "lode"
    jobs = []
    for index, values in enumerate(chosen):
        overrides = [item for key, value in zip(keys, values) for item in ("--override", f"{key}={value}")]
        job = train_job(f"{prefix}_trial_{index:02d}", method, search["representation"], f"trial_{index:02d}", root,
                        seed=int(search["training_seed"]), extra=overrides, view=search["view"])
        job["overrides"] = dict(zip(keys, values))
        jobs.append(job)
    jobs.append({"id": f"{prefix}_select",
                 "argv": ["stage2_select_latent_ode.py", "--method", method, "--jobs", str(D.VALIDATION_ROOT / "jobs" / f"{method}_search.json")],
                 "weight": 1, "after": [j["id"] for j in jobs],
                 "done_marker": str(D.RECIPE_DIR / f"{method}_selected_overrides.json")})
    return jobs


SUITES = {"smoke": smoke, "anchor_retrain": anchor_retrain,
          "latent_ode_search": lambda: latent_search("latent_ode"),
          "latent_ode_residual_search": lambda: latent_search("latent_ode_residual")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suites", nargs="+", choices=sorted(SUITES))
    args = parser.parse_args()
    for suite in args.suites:
        path = bc.require_bulk(D.VALIDATION_ROOT / "jobs" / f"{suite}.json")
        jobs = SUITES[suite]()
        bc.atomic_json(path, {"name": suite, "jobs": jobs})
        print(f"{suite}: {len(jobs)} jobs -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
