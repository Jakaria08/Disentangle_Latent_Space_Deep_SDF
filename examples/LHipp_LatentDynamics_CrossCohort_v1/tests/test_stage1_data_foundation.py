#!/usr/bin/env python3
"""Stage 1 tests: pure data logic, configuration contracts, and regressions against ADNI.

Runs under pytest or directly (``python tests/test_stage1_data_foundation.py``), in any env
with numpy and pandas; no torch or GPU needed. The regression tests read the existing ADNI
cocycle_v4 dataset and pair files, which is how they prove the new view builder reproduces
the archives August_Version and task3 were trained on.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark_common as bc  # noqa: E402
import stage1_build_protocol_views as views  # noqa: E402

ADNI_COCYCLE = bc.REPO_ROOT / "examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/hippocampus_pca_cocycle_v4/cocycle_v4"


def strict_builder():
    path = bc.REPO_ROOT / "scripts" / "prepare_adni_synthseg_separate_structure_cohorts.py"
    spec = importlib.util.spec_from_file_location("strict_cohort_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# pure logic
# --------------------------------------------------------------------------------------


def test_split_sizes_matches_strict_builder():
    builder = strict_builder()
    ratios = (0.8, 0.1, 0.1)
    for total in range(3, 80):
        assert bc.split_sizes(total, ratios) == builder.split_sizes(total, ratios), total


def test_trajectory_groups():
    cases = {
        ("CN", "CN"): "CN-stable",
        ("AD", "AD", "AD"): "AD-stable",
        ("MCI", "MCI"): "MCI-stable",
        ("CN", "AD"): "CN->AD",
        ("CN", "MCI", "AD"): "CN->AD",
        ("CN", "CN", "MCI", "MCI", "AD"): "CN->AD",
        ("MCI", "AD"): "MCI->AD",
        ("CN", "MCI"): "CN->MCI",
        ("MCI", "CN"): "reverter",
        ("AD", "MCI"): "reverter",
        ("CN", "MCI", "CN"): "reverter",
    }
    for labels, expected in cases.items():
        assert bc.trajectory_group(labels) == expected, (labels, bc.trajectory_group(labels))


def test_conversion_window():
    times = [0.0, 1.5, 3.0, 4.5]
    assert bc.conversion_window(times, ["CN", "MCI", "AD", "AD"], "AD") == (1.5, 3.0)
    assert bc.conversion_window(times, ["CN", "MCI", "AD", "AD"], "MCI") == (0.0, 1.5)
    a, b = bc.conversion_window(times, ["AD", "AD", "AD", "AD"], "AD")
    assert math.isnan(a) and math.isnan(b), "conversion at baseline is not observed"
    a, b = bc.conversion_window(times, ["CN"] * 4, "AD")
    assert math.isnan(a) and math.isnan(b)


def test_stratified_split_is_deterministic_disjoint_and_pools_rare_strata():
    strata = {f"s{i}": ("CN-stable" if i < 40 else "CN->AD" if i < 50 else f"rare{i}") for i in range(52)}
    first = bc.stratified_split(strata, 42, (0.8, 0.1, 0.1), "t")
    second = bc.stratified_split(strata, 42, (0.8, 0.1, 0.1), "t")
    assert first == second
    assert set(first) == set(strata)
    assert set(first.values()) == {"train", "val", "test"}
    cn = [first[f"s{i}"] for i in range(40)]
    assert cn.count("val") == 4 and cn.count("test") == 4
    assert bc.stratified_split(strata, 7, (0.8, 0.1, 0.1), "t") != first


def test_crossfit_folds_are_balanced_and_deterministic():
    strata = {f"s{i}": f"{'CN' if i % 7 else 'AD'}|{bc.visit_count_bin(2 + i % 5)}" for i in range(149)}
    folds = bc.crossfit_folds(strata, 5, 42)
    assert folds == bc.crossfit_folds(strata, 5, 42)
    sizes = np.bincount(list(folds.values()), minlength=5)
    assert sizes.max() - sizes.min() <= 1, sizes
    for stratum in set(strata.values()):
        members = [folds[s] for s, v in strata.items() if v == stratum]
        counts = np.bincount(members, minlength=5)
        assert counts.max() - counts.min() <= 1, (stratum, counts)


def test_task_prefix_rules():
    tasks = bc.load_tasks()["tasks"]
    assert bc.task_prefix(1, tasks["one_shot_first"]) is None
    assert bc.task_prefix(4, tasks["one_shot_first"]) == [0]
    assert bc.task_prefix(4, tasks["one_shot_prev"]) == [2]
    assert bc.task_prefix(4, tasks["four_shot"]) is None
    assert bc.task_prefix(5, tasks["four_shot"]) == [0, 1, 2, 3]
    assert bc.task_prefix(7, tasks["four_shot"]) == [2, 3, 4, 5]
    assert bc.task_prefix(2, tasks["all_prior_k"]) is None
    assert bc.task_prefix(3, tasks["all_prior_k"]) == [0, 1]
    assert bc.task_prefix(4, tasks["all_prior_k"]) == [0, 1, 2]
    assert bc.task_prefix(9, tasks["all_prior_k"]) == [4, 5, 6, 7]


def test_linear_extrapolation_recovers_a_line_and_degrades_to_last_value():
    times = [0.0, 1.0, 2.5]
    values = np.stack([np.array([1.0, -2.0]) + t * np.array([0.5, 3.0]) for t in times])
    predicted = bc.linear_extrapolation(times, values, 4.0)
    assert np.allclose(predicted, [3.0, 10.0])
    assert np.allclose(bc.linear_extrapolation([2.0], values[:1], 9.0), values[0])


def test_population_drift_recovers_label_velocities():
    subjects, times, labels, codes = [], [], [], []
    for subject, label, velocity in (("a", 0, 1.0), ("b", 0, 1.0), ("c", 1, -3.0)):
        for t in (0.0, 0.5, 2.0):
            subjects.append(subject)
            times.append(t)
            labels.append(label)
            codes.append(np.full(4, 10.0 + velocity * t))
    drift = bc.population_drift(np.stack(codes), subjects, times, labels)
    assert np.allclose(drift[0], 1.0) and np.allclose(drift[1], -3.0)


def test_require_bulk_and_gpu_guards():
    for bad in (bc.REPO_ROOT / "examples", Path("/tmp/x")):
        try:
            bc.require_bulk(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"require_bulk accepted {bad}")
    bc.require_allowed_gpu("cuda:0")
    bc.require_allowed_gpu("cuda:2")
    bc.require_allowed_gpu("cpu")
    try:
        bc.require_allowed_gpu("cuda:1")
    except ValueError:
        pass
    else:
        raise AssertionError("GPU 1 must be refused")


def test_pinned_module_rejects_wrong_hash():
    pinned = bc.load_registry()["pinned_sources"]["train_lamm"]
    try:
        bc.load_pinned_git_module("never_imported", pinned["git_blob"], "0" * 64, [], bc.resolve(pinned["nominal_file"]))
    except ValueError:
        assert "never_imported" not in sys.modules
    else:
        raise AssertionError("a wrong pinned hash must be refused")


# --------------------------------------------------------------------------------------
# configuration contracts
# --------------------------------------------------------------------------------------


def test_configs_are_consistent():
    sources = bc.load_cohort_sources()
    registry = bc.load_registry()
    assert tuple(registry["representations"]) == bc.REPRESENTATIONS
    expanded = bc.expand_views()
    assert len(expanded) == 16 + 10, len(expanded)
    for name, spec in expanded.items():
        if "crossfit" in spec:
            info = spec["crossfit"]
            assert info["val_fold"] == (info["fold"] + 1) % info["folds"]
            assert info["cohort"] in sources["cohorts"]
            continue
        train_cohorts = {cohort for cohort, _ in spec["train"]}
        for split in bc.SPLITS:
            for cohort, member in spec[split]:
                assert cohort in sources["cohorts"], (name, cohort)
                assert member in ("train", "val", "test", "all"), (name, member)
                if split == "test" and member == "all":
                    assert cohort not in train_cohorts, f"{name}: whole-cohort test overlaps training cohort {cohort}"
        if name.startswith("p4_loco_without_"):
            held_out = name.rsplit("_", 1)[1]
            assert held_out not in train_cohorts and held_out not in {c for c, _ in spec["val"]}
    assert not sources["cohorts"]["calsnic"]["pooled_eligible"]
    assert "calsnic" not in {c for c, _ in expanded["p3_pooled"]["train"]}


def test_strict_manifest_counts_match_plan():
    # CALSNIC: 806 scans / 303 subjects in the manifest, minus CALSNIC2_EDM_C038 (no age at its scanned visits).
    expected = {"adni": (2583, 597), "aibl": (482, 148), "oasis": (1064, 385), "calsnic": (804, 302)}
    sources = bc.load_cohort_sources()
    for cohort, (scans, subjects) in expected.items():
        frame = bc.read_strict_manifest(cohort, sources)
        assert (len(frame), frame["subject_key"].nunique()) == (scans, subjects), cohort
        assert set(frame["label"]) == {0, 1}, cohort


# --------------------------------------------------------------------------------------
# view builder, synthetic and ADNI regression
# --------------------------------------------------------------------------------------


def synthetic_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for cohort, n_subjects in (("adni", 12), ("aibl", 9)):
        rows = []
        for s in range(n_subjects):
            split = bc.SPLITS[s % 3]
            label = s % 2
            for v in range(2 + s % 4):
                rows.append({
                    "cohort": cohort, "subject_key": bc.qualify(cohort, f"{s:02d}"), "scan_key": bc.qualify(cohort, f"{s:02d}_{v}"),
                    "split": split, "label": label, "diagnosis_source": "AD" if label else "CN",
                    "visit_month": 6.0 * v, "age_years": 65.0 + s + 0.5 * v,
                    "correspondence_volume_mm3": 3000.0 - v, "correspondence_surface_area_mm2": 1500.0,
                })
        frames[cohort] = pd.DataFrame(rows)
    return frames


def test_view_membership_archives_pairs_and_tasks_on_synthetic_data():
    frames = synthetic_frames()
    spec = {"train": [["adni", "train"], ["aibl", "train"]], "val": [["adni", "val"]], "test": [["aibl", "all"]]}
    try:
        views.view_membership(spec, frames, {})
    except ValueError as error:
        assert "more than one view split" in str(error)
    else:
        raise AssertionError("aibl train and aibl all must leak")
    spec = {"train": [["adni", "train"], ["adni", "test"]], "val": [["adni", "val"]], "test": [["aibl", "all"]]}
    rows = views.view_membership(spec, frames, {})
    train = rows.loc[rows["view_split"] == "train"]
    lo, hi = float(train["age_years"].min()), float(train["age_years"].max())
    archive = views.sequence_archive(train.copy(), lo, hi)
    offsets = archive["subject_visit_offsets"]
    assert offsets[0] == 0 and offsets[-1] == len(archive["visit_scan_ids"])
    assert archive["visit_age_norm_train"].min() == 0.0 and np.isclose(archive["visit_age_norm_train"].max(), 1.0)
    pairs = views.forward_pairs(archive, "train")
    lengths = np.diff(offsets)
    assert len(pairs) == int(sum(n * (n - 1) // 2 for n in lengths))
    wide = pairs.loc[pairs["target_index"] - pairs["source_index"] > 1]
    assert (wide["intermediate_index"] == wide["source_index"] + (wide["target_index"] - wide["source_index"]) // 2).all()
    test = views.sequence_archive(rows.loc[rows["view_split"] == "test"].copy(), lo, hi)
    tasks = bc.load_tasks()["tasks"]
    table = views.task_table(test, "all_prior_k", tasks["all_prior_k"])
    assert (table["n_visits"] >= 3).all() and (table["k"] == np.minimum(table["n_visits"] - 1, 4)).all()
    assert set(table["cohort"]) == {"aibl"}


def _adni_split_archive(split: str) -> dict[str, np.ndarray]:
    frame = bc.read_strict_manifest("adni")
    train = frame.loc[frame["split"] == "train"]
    rows = frame.loc[frame["split"] == split].assign(view_split=split, source_split=split)
    return views.sequence_archive(rows.copy(), float(train["age_years"].min()), float(train["age_years"].max()))


def test_view_archives_reproduce_adni_cocycle_v4_dataset():
    for split in bc.SPLITS:
        mine = _adni_split_archive(split)
        theirs = bc.load_npz(ADNI_COCYCLE / "dataset" / f"{split}_subject_sequences.npz")
        assert [bc.unqualify(s) for s in mine["visit_scan_ids"]] == theirs["visit_scan_ids"].astype(str).tolist(), split
        assert [bc.unqualify(s) for s in mine["subject_ids"]] == theirs["subject_ids"].astype(str).tolist(), split
        assert np.array_equal(mine["subject_visit_offsets"], theirs["subject_visit_offsets"])
        assert np.array_equal(mine["visit_label_ad"], theirs["visit_label_ad"])
        assert np.array_equal(mine["visit_orders"], theirs["visit_orders"])
        for key in ("visit_months_from_baseline", "visit_time_years_from_baseline", "visit_age_years", "visit_volume_mm3"):
            assert np.allclose(mine[key], theirs[key], atol=1e-4, rtol=0), (split, key)
        assert np.allclose(mine["visit_age_norm_train"], theirs["visit_age_norm_train"], atol=1e-6, rtol=0), split


def test_view_pairs_reproduce_adni_cocycle_v4_pairs():
    for split in bc.SPLITS:
        mine = views.forward_pairs(_adni_split_archive(split), split)
        theirs = pd.read_csv(ADNI_COCYCLE / "pairs" / f"{split}_forward_pairs.csv", dtype={"subject_id": str})
        assert len(mine) == len(theirs), split
        for column in ("source_index", "target_index", "intermediate_index", "label_ad", "source_visit_order", "target_visit_order"):
            assert np.array_equal(mine[column].to_numpy(), theirs[column].to_numpy()), (split, column)
        assert (mine["pair_type"].to_numpy() == theirs["pair_type"].to_numpy()).all()
        assert [bc.unqualify(s) for s in mine["source_scan_id"]] == theirs["source_scan_id"].astype(str).tolist()
        assert np.allclose(mine["delta_years"], theirs["delta_years"], atol=1e-9)


def run_all() -> int:
    failures = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001 - report every failure, then exit non-zero
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run_all())
