#!/usr/bin/env python3
"""Phase 1: build QC and cohort manifests for each target cohort.

This drives the two existing ADNI scripts rather than reimplementing them, so every cohort
is selected by exactly the same policy: hard geometry failures, strong adjacent-pair
failures, and a minimum of two visits per subject.  The QC flags are the corrected ones
(component tolerance, interval-stratified pair rule, culprit attribution); the ADNI
defaults are left untouched elsewhere.

Two cohorts need different switches, and both are configuration, not special-casing:

* OASIS keeps the ADNI ``strict_no_mci`` filter - its CDR-derived labels live on the same
  CN/MCI/AD axis.
* CALSNIC is ALS, so no baseline-CN/AD filter can apply; it uses ``--cohort-filter all``
  and its own Control/ALS labels, and the cohort is restricted to those two groups.

Nothing here touches a source mesh.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import xcohort_common as xc

PYTHON = "/home/jakaria/anaconda3/envs/inr_sdf/bin/python"
QC_SCRIPT = xc.REPO_ROOT / "scripts" / "adni_synthseg_strict_longitudinal_qc.py"
COHORT_SCRIPT = xc.REPO_ROOT / "scripts" / "prepare_adni_synthseg_separate_structure_cohorts.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohorts", nargs="+", default=None, help="Default: every target cohort.")
    parser.add_argument("--skip-mesh-qc", action="store_true", help="Reuse an existing mesh_qc directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in command], check=True, cwd=str(xc.REPO_ROOT))


def qc_command(spec: xc.CohortSpec, config: xc.Config, skip_mesh_qc: bool) -> list[str]:
    settings = config.qc_settings
    command = [
        PYTHON, QC_SCRIPT,
        "--source-root", spec.mesh_root,
        "--output-dir", spec.cohort_build_root / "qc",
        "--cohort-filter", spec.cohort_filter,
        "--multi-component-volume-tolerance-pct", settings["multi_component_volume_tolerance_pct"],
        "--pair-outlier-method", settings["pair_outlier_method"],
        "--pair-outlier-mad-multiplier", settings["pair_outlier_mad_multiplier"],
        "--pair-exclusion-policy", settings["pair_exclusion_policy"],
    ]
    if spec.keep_diagnosis_changers:
        command.append("--keep-diagnosis-changers")
    if skip_mesh_qc:
        command.append("--skip-mesh-qc")
    return command


def cohort_command(spec: xc.CohortSpec, config: xc.Config) -> list[str]:
    command = [
        PYTHON, COHORT_SCRIPT,
        "--qc-root", spec.cohort_build_root / "qc",
        "--source-root", spec.mesh_root,
        "--output-root", spec.cohort_build_root / "cohort",
        "--seed", config.seed,
        "--train-ratio", config.split_ratios["train"],
        "--val-ratio", config.split_ratios["val"],
        "--test-ratio", config.split_ratios["test"],
        "--pair-exclusion-policy", config.qc_settings["pair_exclusion_policy"],
        "--allowed-diagnoses", *spec.allowed_diagnoses,
        "--positive-diagnosis", spec.positive_diagnosis,
        "--negative-diagnosis", spec.negative_diagnosis,
    ]
    if spec.cohort_filter == "all":
        command.append("--allow-non-strict-subjects")
    if spec.drop_duplicate_visit_months:
        command.append("--drop-duplicate-visit-months")
    if spec.restrict_diagnoses:
        command += ["--restrict-diagnoses", *spec.restrict_diagnoses]
    return command


def main() -> int:
    args = parse_args()
    config = xc.load_config()
    names = args.cohorts or [c.name for c in config.targets()]
    summary: dict[str, object] = {}

    for name in names:
        spec = config.cohorts[name]
        if spec.already_prepared:
            print(f"\n[{name}] reference cohort, already prepared - skipping build.", flush=True)
            continue
        print("\n" + "=" * 88)
        print(f"[{name}] QC + cohort build   meshes: {spec.mesh_root}")
        print(f"[{name}] {spec.note}")
        print("=" * 88, flush=True)
        if not (spec.mesh_root / "manifests" / "selected_scans.csv").is_file():
            raise FileNotFoundError(
                f"{name}: mesh run has not produced {spec.mesh_root}/manifests/selected_scans.csv yet"
            )
        run(qc_command(spec, config, args.skip_mesh_qc), args.dry_run)
        run(cohort_command(spec, config), args.dry_run)

        if args.dry_run:
            continue
        rows = xc.read_manifest(spec, config)  # validates topology and splits
        counts = {split: len(xc.split_rows(rows, split)) for split in xc.SPLITS}
        subjects = len({row["subject_id"] for row in rows})
        summary[name] = {"scans": len(rows), "subjects": subjects, "splits": counts}
        print(f"[{name}] keep manifest OK: {len(rows)} scans / {subjects} subjects, splits {counts}", flush=True)

    if summary:
        xc.write_json(xc.TASK_ROOT / "reports" / "phase1_cohorts.json", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
