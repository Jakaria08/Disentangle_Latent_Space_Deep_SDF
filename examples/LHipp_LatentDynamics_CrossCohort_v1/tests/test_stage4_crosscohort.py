#!/usr/bin/env python3
"""Stage 4 tests: the cross-cohort results index and job contract. No torch or GPU."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmark_common as bc  # noqa: E402
import dynamics_core as D  # noqa: E402
import stage4_build_jobs as S4  # noqa: E402


def test_trained_runs_cover_every_protocol_once():
    runs = S4.trained_runs()
    assert len(runs) == 280 and len({r["key"] for r in runs}) == 280
    by_protocol = Counter(r["protocol"] for r in runs)
    assert by_protocol == {"P2": 60, "P2-CF": 100, "P3": 60, "P4": 60}
    pooled_seeds = sorted({r["seed"] for r in runs if r["protocol"] == "P3"})
    assert pooled_seeds == [42, 43, 44]
    assert {r["representation"] for r in runs if r["protocol"] == "P2-CF"} == set(S4.CROSSFIT_REPRESENTATIONS)
    for r in runs:
        Path(r["run_dir"]).relative_to(D.RUNS_ROOT / r["view"])


def test_blocked_rule_is_exactly_the_oasis_only_cocycle():
    blocked = [r for r in S4.trained_runs() if r["blocked"]]
    assert len(blocked) == 14
    assert all(r["method"] == "direct_c4" for r in blocked)
    assert all(r["view"] == "p2_internal_oasis" or r["view"].startswith("p2_crossfit_oasis_") for r in blocked)


def test_external_evaluations_use_adni_checkpoints_and_both_calsnic_conditions():
    external = S4.external_evaluations()
    assert len(external) == 60 * (2 + 2 + 4)
    assert all(e["trained_view"] == "p0_internal_adni" for e in external)
    calsnic = [e for e in external if "calsnic" in e["view"]]
    assert Counter(e["condition_override"] for e in calsnic) == {None: 120, 0: 120}
    assert len({e["output"] for e in external}) == len(external)


def test_training_suite_dependencies_transfers_and_no_blocked_jobs():
    runs = S4.trained_runs()
    transfers = S4.transfer_evaluations(runs)
    assert Counter(t["protocol"] for t in transfers) == {"P4b": 20, "P4c": 120}
    jobs = S4.training_suite(runs, transfers)
    ids = {j["id"] for j in jobs}
    assert sum(j["id"].startswith("train__") for j in jobs) == 266
    assert not any("p2_internal_oasis__" in j["id"] and "__direct_c4__" in j["id"] for j in jobs)
    for job in jobs:
        assert set(job.get("after", [])) <= ids
        assert "--device" not in job["argv"]
    for t in transfers:
        assert f"train__{t['source_key']}" in ids


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
