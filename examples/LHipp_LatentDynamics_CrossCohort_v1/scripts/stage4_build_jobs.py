#!/usr/bin/env python3
"""Stage 4 (cross-cohort protocols P1-P4c): results index and job files.

Protocols (views in configs/protocol_views.json):

  P1     zero-shot: every stage 3 ADNI seed-run (60 checkpoints) evaluated on AIBL, OASIS and CALSNIC,
         test split and whole cohort. CALSNIC is scored twice: with its labels (ALS -> disease condition)
         and with every subject fed the control condition (d = 0).
  P2     internal: 20 cells trained on each of AIBL, OASIS, CALSNIC (seed 42).
  P2-CF  5-fold subject cross-fit on AIBL and OASIS for PCA + the best learned representation (seed 42).
  P3     pooled ADNI+AIBL+OASIS: 20 cells x seeds 42, 43, 44.
  P4     leave-one-cohort-out: 20 cells x 3 held-out cohorts (seed 42).
  P4b    BrainODE Exp1 replica: P2 AIBL checkpoints evaluated on all of ADNI and OASIS.
  P4c    BrainODE Exp2 analog: P3 checkpoints evaluated on all of CALSNIC (labels and d = 0).

Best learned representation for cross-fit (rule fixed in PLAN.md before stage 3): lowest mean
one-shot error over the four non-faithful dynamics on ADNI test - Adaptive 0.3151 mm, LAMM 0.3153,
SpiralNet 0.3161. That is a near tie, and it is recorded as such.

Blocked (held back pending a decision): the cocycle on OASIS-only training views. OASIS training data
contain no AD subject with three or more visits, so there are no non-adjacent AD pairs, and the
cocycle's balanced pair sampler (August/task3 semantics) refuses to train. All other methods and
views are unaffected.

Suites:
  external   480 P1 evaluations (existing checkpoints; no training)
  training   266 training runs + their val/test evaluations + P4b/P4c transfer evaluations
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import benchmark_common as bc
import dynamics_core as D
import stage2_build_jobs as S2
import stage3_build_jobs as S3

STAGE4_ROOT = bc.BULK_ROOT / "stage4_crosscohort"
CROSSFIT_REPRESENTATIONS = ("pca128", "adaptive128")
TARGETS = ("aibl", "oasis", "calsnic")


def is_blocked(view: str, method: str) -> bool:
    return method == "direct_c4" and (view == "p2_internal_oasis" or view.startswith("p2_crossfit_oasis_"))


def evaluation(job_id: str, checkpoint: str, view: str, trained_view: str, split: str, output: Path,
               after: list[str], override: int | None = None) -> dict[str, Any]:
    argv = ["evaluate_dynamics.py", "--checkpoint", checkpoint, "--view", view, "--trained-view", trained_view,
            "--split", split, "--bootstrap", "2000", "--output-dir", str(output)]
    if override is not None:
        argv += ["--condition-override", str(override)]
    return {"id": job_id, "argv": argv, "weight": 1, "after": after, "done_marker": str(output / "summary.json")}


def trained_runs(include_blocked: bool = True) -> list[dict[str, Any]]:
    plan = [("P2", f"p2_internal_{c}", (42,), bc.REPRESENTATIONS) for c in TARGETS]
    plan += [("P3", "p3_pooled", (42,), bc.REPRESENTATIONS)]
    plan += [("P4", f"p4_loco_without_{c}", (42,), bc.REPRESENTATIONS) for c in ("adni", "aibl", "oasis")]
    plan += [("P2-CF", f"p2_crossfit_{c}_fold{k}", (42,), CROSSFIT_REPRESENTATIONS) for c in ("aibl", "oasis") for k in range(5)]
    plan += [("P3", "p3_pooled", (43, 44), bc.REPRESENTATIONS)]
    runs = []
    for protocol, view, seeds, representations in plan:
        for seed in seeds:
            for representation in representations:
                for method in D.METHODS:
                    blocked = is_blocked(view, method)
                    if blocked and not include_blocked:
                        continue
                    run_dir = D.RUNS_ROOT / view / representation / method / f"{representation}_{method}_s{seed}"
                    splits = ("test",) if protocol == "P2-CF" else ("val", "test")
                    runs.append({"key": f"{view}__{representation}__{method}__s{seed}", "protocol": protocol, "view": view,
                                 "representation": representation, "method": method, "seed": seed, "blocked": blocked,
                                 "run_dir": str(run_dir), "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
                                 "evaluations": {s: str(run_dir / f"evaluation__{view}" / s) for s in splits}})
    return runs


def external_evaluations() -> list[dict[str, Any]]:
    entries = []
    for run in bc.read_json(S3.STAGE3_ROOT / "results_index.json")["runs"]:
        for cohort in TARGETS:
            for scope in ("testsplit", "wholecohort"):
                view = f"p1_external_{cohort}_{scope}"
                for override in ((None, 0) if cohort == "calsnic" else (None,)):
                    suffix = "" if override is None else f"__d{override}"
                    entries.append({"key": f"{view}__{run['key']}{suffix}", "protocol": "P1", "view": view,
                                    "trained_view": "p0_internal_adni", "source_key": run["key"],
                                    "representation": run["representation"], "method": run["method"], "seed": run["seed"],
                                    "checkpoint": run["checkpoint"], "condition_override": override,
                                    "output": str(STAGE4_ROOT / "external_evaluations" / view / f"{run['key']}{suffix}" / "test")})
    return entries


def transfer_evaluations(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for run in runs:
        if run["blocked"]:
            continue
        if run["view"] == "p2_internal_aibl":
            targets = [("P4b", "p4b_exp1_aibl_to_adni_oasis", None)]
        elif run["view"] == "p3_pooled":
            targets = [("P4c", "p4c_exp2_pooled_to_calsnic", None), ("P4c", "p4c_exp2_pooled_to_calsnic", 0)]
        else:
            continue
        for protocol, view, override in targets:
            suffix = "" if override is None else f"__d{override}"
            entries.append({"key": f"{view}__{run['key']}{suffix}", "protocol": protocol, "view": view, "trained_view": run["view"],
                            "source_key": run["key"], "representation": run["representation"], "method": run["method"],
                            "seed": run["seed"], "checkpoint": run["checkpoint"], "condition_override": override,
                            "output": str(STAGE4_ROOT / "transfer_evaluations" / view / f"{run['key']}{suffix}" / "test")})
    return entries


def external_suite(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluation(f"ext__{e['key']}", e["checkpoint"], e["view"], e["trained_view"], "test", Path(e["output"]), [], e["condition_override"])
            for e in entries]


def training_suite(runs: list[dict[str, Any]], transfers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for run in runs:
        if run["blocked"]:
            continue
        train = S2.train_job(f"train__{run['key']}", run["method"], run["representation"], Path(run["run_dir"]).name,
                             D.RUNS_ROOT, seed=run["seed"], view=run["view"])
        jobs.append(train)
        for split, output in run["evaluations"].items():
            jobs.append(evaluation(f"eval_{split}__{run['key']}", run["checkpoint"], run["view"], run["view"], split, Path(output), [train["id"]]))
    for entry in transfers:
        jobs.append(evaluation(f"xfer__{entry['key']}", entry["checkpoint"], entry["view"], entry["trained_view"], "test",
                               Path(entry["output"]), [f"train__{entry['source_key']}"], entry["condition_override"]))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suites", nargs="+", choices=("external", "training"))
    args = parser.parse_args()
    runs = trained_runs()
    external = external_evaluations()
    transfers = transfer_evaluations(runs)
    bc.atomic_json(bc.require_bulk(STAGE4_ROOT / "results_index.json"), {
        "crossfit_representations": list(CROSSFIT_REPRESENTATIONS),
        "blocked_rule": "direct_c4 on p2_internal_oasis and p2_crossfit_oasis_*: no AD subject with >= 3 training visits",
        "trained_runs": runs, "external_evaluations": external, "transfer_evaluations": transfers})
    print(f"index: {len(runs)} training runs ({sum(r['blocked'] for r in runs)} blocked), "
          f"{len(external)} external evaluations, {len(transfers)} transfer evaluations")
    for suite in args.suites:
        jobs = external_suite(external) if suite == "external" else training_suite(runs, transfers)
        path = bc.require_bulk(STAGE4_ROOT / "jobs" / f"crosscohort_{suite}.json")
        bc.atomic_json(path, {"name": f"crosscohort_{suite}", "jobs": jobs})
        print(f"{suite}: {len(jobs)} jobs ({sum(j['id'].startswith('train__') for j in jobs)} training) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
