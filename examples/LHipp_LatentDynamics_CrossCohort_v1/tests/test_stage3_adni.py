#!/usr/bin/env python3
"""Stage 3 tests: statistics helpers and the ADNI matrix job/index contract. No torch or GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark_common as bc  # noqa: E402
import benchmark_stats as ST  # noqa: E402
import dynamics_core as D  # noqa: E402
import stage3_build_jobs as S3  # noqa: E402


def test_holm_matches_hand_computation():
    adjusted = ST.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adjusted["a"], 0.03) and np.isclose(adjusted["c"], 0.06) and np.isclose(adjusted["b"], 0.06)
    assert ST.holm({"x": 0.6, "y": 0.9}) == {"x": 1.0, "y": 1.0}


def test_paired_difference_detects_a_shift_and_not_a_null():
    rng = np.random.default_rng(0)
    base = rng.normal(0.3, 0.05, size=60)
    shifted = ST.paired_difference(base + 0.02 + rng.normal(0, 0.005, 60), base)
    assert shifted["ci95_low"] > 0 and shifted["wilcoxon_p"] < 1e-6
    null = ST.paired_difference(base, base)
    assert null["mean_difference"] == 0.0 and null["wilcoxon_p"] == 1.0


def test_bootstrap_and_correlations():
    values = np.random.default_rng(1).normal(1.0, 0.2, size=80)
    low, high = ST.bootstrap_mean_ci(values)
    assert low < values.mean() < high
    x = np.arange(20.0)
    assert np.isclose(ST.correlations(x, 2 * x + 1)["pearson_r"], 1.0)


def test_results_index_covers_the_matrix_once():
    runs = S3.seed_runs()
    assert len(runs) == 60 and len({r["key"] for r in runs}) == 60
    for rep in bc.REPRESENTATIONS:
        for method in D.METHODS:
            assert sorted(r["seed"] for r in runs if r["representation"] == rep and r["method"] == method) == [42, 43, 44]
    anchors = [r for r in runs if r["source"] == "august_anchor"]
    assert len(anchors) == 6
    assert all(r["seed"] == 42 and r["representation"] != "lamm128" and r["method"] in ("plain_ode", "brainode") for r in anchors)
    assert sum("min_epoch_checkpoint" in r for r in runs) == 12
    for r in runs:
        if r["source"] != "august_anchor":
            Path(r["run_dir"]).relative_to(D.RUNS_ROOT / "p0_internal_adni")


def test_training_suite_dependencies_and_unique_outputs():
    runs = S3.seed_runs()
    jobs = S3.training_suite(runs)
    ids = {j["id"] for j in jobs}
    assert len(ids) == len(jobs) == 54 + 144
    for job in jobs:
        assert set(job.get("after", [])) <= ids
        assert "--device" not in job["argv"]
    tests = [j for j in jobs if "--split" in j["argv"] and j["argv"][j["argv"].index("--split") + 1] == "test"]
    outputs = [j["argv"][j["argv"].index("--output-dir") + 1] for j in tests]
    assert len(outputs) == len(set(outputs)) == 60 + 12


def run_all() -> int:
    failures = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run_all())
