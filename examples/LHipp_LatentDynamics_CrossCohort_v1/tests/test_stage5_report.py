#!/usr/bin/env python3
"""Stage 5 (step 3) tests: the consolidated report's endpoint statistics, decision rule and gate states.

Pure pandas/numpy on synthetic frames (the plotting and loading paths are exercised by building the report):
    /home/jakaria/anaconda3/envs/pytorch_geo/bin/python tests/test_stage5_report.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import report_brainode_style as RB  # noqa: E402


def _synthetic_rows(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two representations x three models on 40 subjects (20 AD). model_b is 0.001 mm worse and captures AD perfectly."""
    rng = np.random.default_rng(seed)
    frames = []
    subjects = [f"s{i}" for i in range(40)]
    diagnosis = ["AD" if i < 20 else "CN" for i in range(40)]
    observed = np.where(np.asarray(diagnosis) == "AD", -0.05, -0.01) + rng.normal(0, 0.002, 40)
    error = 0.3 + rng.normal(0, 0.02, 40)
    for representation in ("pca128", "spiralnet128"):
        for method, error_shift, capture in (("direct_c4", 0.0, 0.5), ("plain_ode", 0.001, 1.0), ("brainode", 0.05, 1.0)):
            for seed_value in (42, 43):
                frames.append(pd.DataFrame({"protocol": "P0", "view": "p0_internal_adni", "override": -1, "cohort": "adni", "task": "one_shot_first",
                                            "variant": "averaged", "subject_id": subjects, "diagnosis": diagnosis, "method": method,
                                            "representation": representation, "seed": seed_value, "euclidean_mm": error + error_shift,
                                            "predicted_log_volume_rate": observed * capture, "observed_log_volume_rate": observed}))
    rows = pd.concat(frames, ignore_index=True)
    extra = rows[rows.representation == "pca128"].copy()
    rows = pd.concat([rows, extra.assign(view="p3_pooled", protocol="P3", cohort="aibl", task="all_prior_k"),
                      extra.assign(view="p1_external_aibl_wholecohort", protocol="P1", cohort="aibl")], ignore_index=True)
    baselines = pd.DataFrame({"view": "p0_internal_adni", "task": "one_shot_first", "representation": np.repeat(["pca128", "spiralnet128"], 40),
                              "subject_id": subjects * 2, "nochange_mm": 0.35})
    return rows, baselines


def test_ratio_ci_brackets_the_ratio_of_means():
    predicted, observed = np.linspace(-0.06, -0.02, 30), np.full(30, -0.05)
    point, low, high = RB.ratio_ci(predicted, observed)
    assert np.isclose(point, predicted.mean() / observed.mean()) and low <= point <= high
    assert all(np.isnan(v) for v in RB.ratio_ci(np.asarray([1.0]), np.asarray([1.0])))


def test_capture_distance_sign():
    observed = np.full(25, -0.05)
    closer, _low, high = RB.capture_distance_difference(observed * 0.95, observed * 0.5, observed)
    assert closer < 0 and high <= 0.0, "a model at 95% capture is closer to 100% than one at 50%"
    overshoot, *_ = RB.capture_distance_difference(observed * 1.5, observed * 0.9, observed)
    assert overshoot > 0, "150% capture is farther from 100% than 90%"


def test_endpoint_decision_rule():
    rows, baselines = _synthetic_rows()
    table, decisions = RB.table_endpoints(rows, baselines)
    p0 = table[table.endpoint_set == "P0 ADNI test"].set_index(["representation", "method"])
    assert bool(p0.loc[("pca128", "plain_ode"), "E1_non_inferior"]) and bool(p0.loc[("pca128", "plain_ode"), "E2_improves"])
    assert bool(p0.loc[("pca128", "plain_ode"), "preferred_over_cocycle"])
    assert not bool(p0.loc[("pca128", "brainode"), "E1_non_inferior"]), "0.05 mm worse exceeds the 2% margin (0.007 mm)"
    assert not bool(p0.loc[("pca128", "brainode"), "preferred_over_cocycle"])
    assert decisions[0]["preferred_over_cocycle"] == ["plain_ode"] and decisions[0]["ranking_by_E1"][0].startswith("direct_c4")
    assert {"E3 P3 AIBL all-prior-k", "E4 P1 AIBL whole cohort one-shot"} <= set(table.endpoint_set)


def test_markdown_renders_pending_gates():
    tables = {key: pd.DataFrame() for key in ("endpoints_e1_e4", "r_t5_composition", "r_t6_evaluable", "r_t7_per_dataset", "r_t1_aibl_regular",
                                              "r_t2_unified", "r_t8_cross_benchmark", "r_t3_ablations", "r_t3_converter", "fidelity_adni",
                                              "fidelity_external", "consistency_cost", "sensitivity_age_subset", "sensitivity_min_epoch",
                                              "sensitivity_pooled_pca", "context_brainode_published")}
    text = RB.markdown(tables, [], [{"id": "G5.7", "check": "x", "passed": None}, {"id": "G5.2", "check": "y", "passed": True}], [], ["missing run"])
    assert "| G5.7 | x | PENDING |" in text and "| G5.2 | y | PASS |" in text and "Partial draft" in text


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:  # noqa: BLE001 - report every failure, then exit non-zero
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
