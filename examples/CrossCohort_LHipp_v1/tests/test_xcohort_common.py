#!/usr/bin/env python3
"""Tests for the cross-cohort utilities.

Run: python tests/test_xcohort_common.py

These cover the parts where a silent error would invalidate every downstream number: PCA
correctness, the metric's units, the topology guard, and the result schema.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xcohort_common as xc  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}{(' - ' + detail) if detail and not condition else ''}")
    if not condition:
        FAILURES.append(name)


def test_config_loads() -> None:
    config = xc.load_config()
    check("config: four cohorts", set(config.cohorts) == {"adni", "aibl", "oasis", "calsnic"}, str(set(config.cohorts)))
    check("config: reference is adni", config.reference().name == "adni")
    check("config: calsnic excluded from pooled", not config.cohorts["calsnic"].pooled_eligible)
    check("config: pooled members are the AD-axis cohorts",
          {c.name for c in config.pooled_members()} == {"adni", "aibl", "oasis"})
    check("config: calsnic restricted to Control/ALS",
          config.cohorts["calsnic"].restrict_diagnoses == ("Control", "ALS"))
    check("config: adni manifest override resolves to an existing file",
          config.reference().keep_manifest.is_file(), str(config.reference().keep_manifest))


def test_pca_exact_on_low_rank_data() -> None:
    """PCA at the data's true rank must reconstruct it exactly."""
    rng = np.random.default_rng(0)
    n_samples, n_vertices, rank = 40, 50, 5
    basis = rng.normal(size=(rank, n_vertices * 3))
    coeffs = rng.normal(size=(n_samples, rank))
    flat = coeffs @ basis + rng.normal(size=(1, n_vertices * 3))
    vertices = flat.reshape(n_samples, n_vertices, 3)
    model = xc.fit_pca(vertices, rank)
    rec = model.reconstruct(vertices)
    check("pca: exact reconstruction at true rank", np.allclose(rec, vertices, atol=1e-8),
          f"max err {np.abs(rec - vertices).max():.2e}")
    check("pca: rank honoured", model.k == rank, f"k={model.k}")
    truncated = model.truncated(2)
    check("pca: truncation keeps the leading components", truncated.k == 2
          and np.allclose(truncated.components, model.components[:2]))
    worse = xc.vertex_rmse_mm(truncated.reconstruct(vertices), vertices).mean()
    better = xc.vertex_rmse_mm(rec, vertices).mean()
    check("pca: fewer components reconstruct worse", worse > better, f"{worse:.3e} vs {better:.3e}")


def test_pca_rank_capped_by_samples() -> None:
    rng = np.random.default_rng(1)
    vertices = rng.normal(size=(6, 20, 3))
    model = xc.fit_pca(vertices, 128)
    check("pca: rank capped by sample count", model.k <= 6, f"k={model.k}")


def test_metric_units() -> None:
    """Coordinate RMSE vs per-vertex Euclidean must differ by exactly sqrt(3)."""
    rng = np.random.default_rng(2)
    gt = rng.normal(size=(7, 30, 3))
    pred = gt + 0.01
    metrics = xc.reconstruction_metrics(pred, gt)
    ratio = metrics["vertex_euclidean_mm_mean"] / metrics["vertex_rmse_mm_mean"]
    check("metric: euclidean/rmse == sqrt(3)", abs(ratio - np.sqrt(3)) < 1e-9, f"ratio {ratio}")
    known = xc.vertex_rmse_mm(pred, gt)
    check("metric: constant offset gives that offset as RMSE",
          np.allclose(known, 0.01, atol=1e-9), f"{known[:2]}")


def test_result_schema() -> None:
    row = xc.result_row(protocol="internal", model="pca", eval_cohort="aibl")
    check("results: row has every field", set(row) == set(xc.RESULT_FIELDS))
    try:
        xc.result_row(nonsense=1)
        check("results: unknown field rejected", False, "no error raised")
    except KeyError:
        check("results: unknown field rejected", True)


def test_manifest_topology_guard() -> None:
    """A manifest from a different vertex space must be refused, not silently used."""
    config = xc.load_config()
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad_manifest.csv"
        bad.write_text(
            "scan_id,subject_id,split,diagnosis,vertex_count,correspondence_topology_hash,mesh_path_mm\n"
            "s1,sub1,train,CN,1234,deadbeef,/nonexistent.ply\n"
            "s2,sub2,val,CN,1234,deadbeef,/nonexistent.ply\n"
            "s3,sub3,test,CN,1234,deadbeef,/nonexistent.ply\n",
            encoding="utf-8",
        )
        spec = xc.CohortSpec(
            name="bogus", role="target", mesh_root=Path(tmp), already_prepared=False,
            cohort_filter="all", allowed_diagnoses=("any",), negative_diagnosis="CN",
            positive_diagnosis="AD", pooled_eligible=False, keep_manifest_override=bad,
        )
        try:
            xc.read_manifest(spec, config)
            check("manifest: wrong topology rejected", False, "no error raised")
        except ValueError as exc:
            check("manifest: wrong topology rejected", "topology hash" in str(exc))


def test_reference_pca_matches_published_adni() -> None:
    """End-to-end: the stored ADNI basis must reproduce ADNI's published PCA-128 numbers."""
    config = xc.load_config()
    reference = config.reference()
    if not reference.keep_manifest.is_file():
        check("reference pca: ADNI manifest present", False, "skipped, manifest missing")
        return
    rows = xc.read_manifest(reference, config)
    model = xc.load_reference_pca(reference).truncated(128)
    val = xc.load_vertices(reference, rows, "val")
    rmse = float(xc.vertex_rmse_mm(model.reconstruct(val), val).mean())
    check("reference pca: reproduces ADNI val 0.033668", abs(rmse - 0.033668) < 1e-5, f"got {rmse:.6f}")


def main() -> int:
    for test in (
        test_config_loads,
        test_pca_exact_on_low_rank_data,
        test_pca_rank_capped_by_samples,
        test_metric_units,
        test_result_schema,
        test_manifest_topology_guard,
        test_reference_pca_matches_published_adni,
    ):
        print(f"\n## {test.__name__}")
        test()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
