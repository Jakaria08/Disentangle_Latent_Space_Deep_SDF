#!/usr/bin/env python3
"""Stage 1e: audit the data foundation against gates G1.1-G1.8 (PLAN.md, Part 1).

Writes reports/stage1_audit.json and reports/stage1_audit.md under the stage 1 bulk root and
exits non-zero if any gate fails. It only reads what stages 1a-1d produced, plus the existing
ADNI archives and summaries it regresses against.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import benchmark_common as bc

AUGUST = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version/training")
LAMM_V3 = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/training")
ANCHOR_SUMMARIES = {
    "pca128": AUGUST / "pca128/direct_c4/pca128_direct_c4_s42_v2/evaluation/test/summary.json",
    "spiralnet128": AUGUST / "spiralnet128/direct_c4/spiralnet128_direct_c4_s42_v2/evaluation/test/summary.json",
    "adaptive128": AUGUST / "adaptive128/direct_c4/adaptive128_direct_c4_s42_v2/evaluation/test/summary.json",
    "lamm128": LAMM_V3 / "lamm128/direct_c4/lamm128_direct_c4_s42/evaluation/test/summary.json",
}
PHASE3_CSV = bc.REPO_ROOT / "examples/CrossCohort_LHipp_v1/reports/phase3_pca_protocols.csv"
TEST_PYTHON = "/home/jakaria/anaconda3/envs/inr_sdf/bin/python"

# (view, task) -> (subjects, positive-class subjects), from PLAN.md section 0.3.
EXPECTED_TASK_COUNTS = {
    ("p0_internal_adni", "one_shot_first"): (61, 28), ("p0_internal_adni", "all_prior_k"): (58, 25),
    ("p0_internal_adni", "four_shot"): (21, 1),
    ("p1_external_aibl_testsplit", "one_shot_first"): (15, 2), ("p1_external_aibl_testsplit", "all_prior_k"): (11, 0),
    ("p1_external_aibl_testsplit", "four_shot"): (0, 0),
    ("p1_external_aibl_wholecohort", "one_shot_first"): (148, 20), ("p1_external_aibl_wholecohort", "all_prior_k"): (114, 8),
    ("p1_external_aibl_wholecohort", "four_shot"): (0, 0),
    ("p1_external_oasis_testsplit", "one_shot_first"): (39, 1), ("p1_external_oasis_testsplit", "all_prior_k"): (20, 0),
    ("p1_external_oasis_testsplit", "four_shot"): (5, 0),
    ("p1_external_oasis_wholecohort", "one_shot_first"): (385, 6), ("p1_external_oasis_wholecohort", "all_prior_k"): (184, 0),
    ("p1_external_oasis_wholecohort", "four_shot"): (25, 0),
    ("p1_external_calsnic_testsplit", "one_shot_first"): (30, 14), ("p1_external_calsnic_testsplit", "all_prior_k"): (23, 12),
    ("p1_external_calsnic_testsplit", "four_shot"): (0, 0),
    ("p1_external_calsnic_wholecohort", "one_shot_first"): (302, 144), ("p1_external_calsnic_wholecohort", "all_prior_k"): (200, 89),
    ("p1_external_calsnic_wholecohort", "four_shot"): (0, 0),
}


def gate(gid: str, name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"id": gid, "name": name, "passed": bool(passed), "detail": detail}


def views_root(view: str) -> Path:
    return bc.STAGE1_ROOT / "views" / view


# --------------------------------------------------------------------------------------


def g11_topology(sources, registry) -> dict[str, Any]:
    import trimesh

    detail, ok = {}, True
    expected_hash = sources["topology"]["correspondence_topology_hash"]
    reference_faces = np.load(registry["faces_path"])
    pca_faces = np.load(bc.resolve(registry["representations"]["pca128"]["pca_model_root"]) / "faces.npy")
    detail["pca_faces_equal_registry_faces"] = bool(np.array_equal(reference_faces, pca_faces))
    ok &= detail["pca_faces_equal_registry_faces"]
    for cohort in sources["cohorts"]:
        frame = bc.read_strict_manifest(cohort, sources)  # raises on a topology violation
        faces = np.asarray(trimesh.load(frame["mesh_path_mm"].iloc[0], process=False).faces)
        same = bool(np.array_equal(faces, reference_faces))
        detail[f"{cohort}_ply_faces_equal_reference"] = same
        ok &= same
        inclusive = bc.STAGE1_ROOT / "cohorts" / cohort / "inclusive_manifest.csv"
        if inclusive.is_file():
            hashes = set(pd.read_csv(inclusive, usecols=["correspondence_topology_hash"])["correspondence_topology_hash"].astype(str))
            detail[f"{cohort}_inclusive_hash_ok"] = hashes == {expected_hash}
            ok &= detail[f"{cohort}_inclusive_hash_ok"]
    return gate("G1.1", "topology and faces identical to ADNI", ok, detail)


def g12_adni_codes(registry) -> dict[str, Any]:
    # Tolerance is relative to each code dimension's spread: LAMM codes are orders of magnitude
    # larger than SpiralNet's, so one absolute bound would be meaningless for one or the other.
    detail, ok = {}, True
    limit = 1.0e-3
    for rep, spec in registry["representations"].items():
        latents = bc.load_npz(bc.STAGE1_ROOT / "latents" / "adni" / f"{rep}.npz")
        keys = [bc.unqualify(k) for k in latents["scan_keys"].astype(str)]
        reference: dict[str, np.ndarray] = {}
        for split in bc.SPLITS:
            archive = bc.load_npz(Path(spec["adni_reference_archive_dir"]) / f"{split}_subject_sequences_128.npz")
            reference.update(zip(archive["visit_scan_ids"].astype(str), archive["visit_latent_raw_128"]))
        stored = np.stack([reference[k] for k in keys])
        copied_exact = float(np.abs(latents["codes"] - stored).max())
        difference = np.abs(latents["codes"] - latents["codes_recomputed"])
        relative = float((difference / np.maximum(stored.std(axis=0), 1.0e-12)).max())
        passed = copied_exact == 0.0 and relative <= limit
        detail[rep] = {
            "copied_vs_reference_max_abs": copied_exact,
            "recomputed_vs_reference_max_abs": float(difference.max()),
            "recomputed_vs_reference_max_relative_to_code_std": relative,
            "relative_limit": limit,
        }
        ok &= passed
    return gate("G1.2", "ADNI codes copied exactly; re-encoding reproduces them", ok, detail)


def g13_reconstruction(sources, registry) -> dict[str, Any]:
    detail, ok = {"by_cohort": {}}, True
    for cohort in sources["cohorts"]:
        strict = bc.read_strict_manifest(cohort, sources).set_index("scan_key")["split"]
        detail["by_cohort"][cohort] = {}
        for rep in registry["representations"]:
            latents = bc.load_npz(bc.STAGE1_ROOT / "latents" / cohort / f"{rep}.npz")
            split = pd.Series(latents["scan_keys"].astype(str)).map(strict).fillna("inclusive_only").to_numpy()
            rmse = latents["recon_coordinate_rmse_mm"]
            detail["by_cohort"][cohort][rep] = {s: float(rmse[split == s].mean()) for s in sorted(set(split))}
    for rep, spec in registry["representations"].items():
        observed = detail["by_cohort"]["adni"][rep]["val"]
        relative = abs(observed - spec["source_validation_rmse_mm"]) / spec["source_validation_rmse_mm"]
        detail[f"adni_val_{rep}"] = {"observed": observed, "published": spec["source_validation_rmse_mm"], "relative_diff": relative}
        ok &= relative <= 0.005
    if PHASE3_CSV.is_file():
        phase3 = pd.read_csv(PHASE3_CSV)
        external = phase3.loc[(phase3["protocol"] == "external") & (phase3["latent_k"].astype(str) == "128") & (phase3["split"] == "test")]
        for cohort in ("aibl", "oasis", "calsnic"):
            match = external.loc[external["eval_cohort"] == cohort, "vertex_rmse_mm_mean"]
            if match.empty:
                continue
            observed = detail["by_cohort"][cohort]["pca128"]["test"]
            detail[f"pca_external_{cohort}_vs_crosscohort_phase3"] = {"observed": observed, "phase3": float(match.iloc[0])}
            ok &= abs(observed - float(match.iloc[0])) <= 1.0e-4
    return gate("G1.3", "reconstruction matches published ADNI val and CrossCohort PCA external", ok, detail)


def g14_hygiene(sources) -> dict[str, Any]:
    views = bc.expand_views()
    problems: list[str] = []
    test_folds: dict[str, dict[str, int]] = {}
    for name, spec in views.items():
        archives = {s: bc.load_npz(views_root(name) / "dataset" / f"{s}_subject_sequences.npz") for s in bc.SPLITS}
        subjects = {s: set(a["subject_ids"].astype(str)) for s, a in archives.items()}
        for s, a in archives.items():
            for field in ("visit_age_norm_train", "visit_time_years_from_baseline", "visit_volume_mm3"):
                if not np.isfinite(a[field]).all():
                    problems.append(f"{name}/{s}: non-finite {field}")
        scans = {s: set(a["visit_scan_ids"].astype(str)) for s, a in archives.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if subjects[a] & subjects[b] or scans[a] & scans[b]:
                problems.append(f"{name}: {a}/{b} overlap")
        fitted_cohorts = set(archives["train"]["visit_cohorts"].astype(str)) | set(archives["val"]["visit_cohorts"].astype(str))
        if name.startswith("p4_loco_without_") and name.rsplit("_", 1)[1] in fitted_cohorts:
            problems.append(f"{name}: held-out cohort present in train/val")
        if "crossfit" not in spec:
            whole = {c for c, m in spec["test"] if m == "all"}
            if whole & fitted_cohorts:
                problems.append(f"{name}: whole-cohort test cohort {sorted(whole & fitted_cohorts)} also fitted")
        else:
            cohort = spec["crossfit"]["cohort"]
            for subject in subjects["test"]:
                test_folds.setdefault(cohort, {})[subject] = test_folds.setdefault(cohort, {}).get(subject, 0) + 1
        for rep in bc.REPRESENTATIONS:
            train = bc.load_npz(views_root(name) / "representations" / rep / "train_subject_sequences_128.npz")
            if not np.allclose(train["visit_latent_raw_128"].mean(axis=0), train["train_latent_mean_128"], atol=1e-4):
                problems.append(f"{name}/{rep}: standardization mean not fitted on train")
    for cohort, counts in test_folds.items():
        strict_subjects = set(bc.read_strict_manifest(cohort, sources)["subject_key"])
        if set(counts) != strict_subjects or set(counts.values()) != {1}:
            problems.append(f"{cohort}: cross-fit does not test every subject exactly once")
    for cohort, spec in sources["cohorts"].items():
        path = bc.STAGE1_ROOT / "cohorts" / cohort / "inclusive_manifest.csv"
        if not spec.get("inclusive_qc_root") or not path.is_file():
            continue
        inclusive = pd.read_csv(path, dtype={"subject_key": str}).drop_duplicates("subject_key").set_index("subject_key")["split"]
        strict = bc.read_strict_manifest(cohort, sources).drop_duplicates("subject_key").set_index("subject_key")["split"]
        shared = strict.index.intersection(inclusive.index)
        if not (strict.loc[shared] == inclusive.loc[shared]).all():
            problems.append(f"{cohort}: inclusive manifest moved a strict subject to another split")
    return gate("G1.4", "split, fold, LOCO and standardization hygiene", not problems, {"views_checked": len(views), "problems": problems})


def g15_counts() -> dict[str, Any]:
    detail, ok = {}, True
    for (view, task), (subjects, positive) in EXPECTED_TASK_COUNTS.items():
        table = pd.read_csv(views_root(view) / "tasks" / f"test_{task}.csv")
        observed = (int(len(table)), int((table["label_ad"] == 1).sum()) if len(table) else 0)
        detail[f"{view}/{task}"] = {"observed": observed, "expected": (subjects, positive)}
        ok &= observed == (subjects, positive)
    return gate("G1.5", "evaluable task counts reproduce PLAN section 0.3", ok, detail)


def g16_baselines() -> dict[str, Any]:
    path = bc.STAGE1_ROOT / "baselines" / "p0_internal_adni" / "test_baseline_rows.csv"
    rows = pd.read_csv(path)
    detail, ok = {}, True
    first = rows.loc[(rows["task"] == "one_shot_first") & (rows["baseline"] == "nochange_decoded")]
    for rep, summary_path in ANCHOR_SUMMARIES.items():
        if not summary_path.is_file():
            detail[rep] = "anchor summary not found; skipped"
            continue
        overall = bc.read_json(summary_path)["pair_metrics"]["first_last_forward"]["groups"]["overall"]
        mine = first.loc[first["representation"] == rep]
        comparison = {
            "subjects": (int(len(mine)), int(overall["rows"])),
            "coordinate_mae": (float(mine["coordinate_mae_mm"].mean()), float(overall["nochange_end_to_end_coordinate_mae_mean"])),
            "euclidean": (float(mine["euclidean_mm"].mean()), float(overall["nochange_end_to_end_euclidean_mean"])),
        }
        passed = comparison["subjects"][0] == comparison["subjects"][1] and all(
            abs(a - b) <= 1.0e-5 for key, (a, b) in comparison.items() if key != "subjects"
        )
        detail[rep] = {"mine_vs_anchor": comparison, "passed": passed}
        ok &= passed
    raw = rows.loc[(rows["task"] == "one_shot_first") & (rows["baseline"] == "nochange_raw"), "euclidean_mm"]
    detail["nochange_raw_one_shot_first_euclidean_mm"] = float(raw.mean())
    return gate("G1.6", "ADNI test no-change reproduces August/task3 anchor summaries", ok, detail)


def g17_inclusive(sources) -> dict[str, Any]:
    detail, ok = {}, True
    for cohort, spec in sources["cohorts"].items():
        if not spec.get("inclusive_qc_root"):
            continue
        report = bc.read_json(bc.STAGE1_ROOT / "cohorts" / cohort / "inclusive_report.json")
        groups = report["counts"]["subjects_by_group"]
        converters = int(groups.get("CN->AD", 0) + groups.get("MCI->AD", 0))
        detail[cohort] = {"passed": report["passed"], "counts": report["counts"], "ad_converters": converters,
                          "strict_scan_coverage": report["strict_scan_coverage"]}
        ok &= bool(report["passed"]) and converters > 0
    return gate("G1.7", "inclusive converter cohorts built and validated", ok, detail)


def g18_tests() -> dict[str, Any]:
    result = subprocess.run([TEST_PYTHON, str(bc.TASK_ROOT / "tests" / "test_stage1_data_foundation.py")],
                            capture_output=True, text=True)
    return gate("G1.8", "stage 1 unit and regression tests", result.returncode == 0, result.stdout.strip().splitlines()[-1:])


def markdown(gates: list[dict[str, Any]]) -> str:
    lines = ["# Stage 1 data foundation audit", "", "| gate | check | result |", "|---|---|---|"]
    lines += [f"| {g['id']} | {g['name']} | {'PASS' if g['passed'] else 'FAIL'} |" for g in gates]
    lines += ["", "Details are in `stage1_audit.json`.", ""]
    recon = next(g for g in gates if g["id"] == "G1.3")["detail"]["by_cohort"]
    lines += ["## Reconstruction coordinate RMSE (mm) by cohort and split", "", "| cohort | representation | splits |", "|---|---|---|"]
    for cohort, reps in recon.items():
        for rep, splits in reps.items():
            lines.append(f"| {cohort} | {rep} | " + ", ".join(f"{s} {v:.4f}" for s, v in splits.items()) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    sources = bc.load_cohort_sources()
    registry = bc.load_registry()
    gates = []
    for check in (lambda: g11_topology(sources, registry), lambda: g12_adni_codes(registry),
                  lambda: g13_reconstruction(sources, registry), lambda: g14_hygiene(sources), g15_counts,
                  g16_baselines, lambda: g17_inclusive(sources), g18_tests):
        try:
            gates.append(check())
        except Exception as error:  # noqa: BLE001 - a crashing check is a failed gate, never a silent pass
            gates.append(gate("G1.?", f"check crashed: {getattr(check, '__name__', 'lambda')}", False, f"{type(error).__name__}: {error}"))
        last = gates[-1]
        print(f"{last['id']} {'PASS' if last['passed'] else 'FAIL'}  {last['name']}", flush=True)
    reports = bc.require_bulk(bc.STAGE1_ROOT / "reports")
    bc.atomic_json(reports / "stage1_audit.json", {"passed": all(g["passed"] for g in gates), "gates": gates})
    bc.atomic_write_text(reports / "stage1_audit.md", markdown(gates) if any(g["id"] == "G1.3" for g in gates) else json.dumps(gates, indent=2))
    return 0 if all(g["passed"] for g in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
