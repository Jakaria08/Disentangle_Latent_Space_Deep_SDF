#!/usr/bin/env python3
"""Small, non-training tests for the exact-SDF pilot pipeline."""

from __future__ import annotations

import importlib.util
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh


TASK_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = TASK_DIR / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_common import (  # noqa: E402
    atomic_write_npz,
    exact_signed_distance_outside_positive,
    relabel_arrays,
    require_bulk_path,
    restore_source_coordinate_order,
)


def load_script(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExactSDFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    def test_known_cube_distances_and_sign(self) -> None:
        xyz = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.6, 0.6, 0.0],
            ],
            dtype=np.float32,
        )
        actual = exact_signed_distance_outside_positive(self.mesh, xyz, chunk_size=2)
        expected = np.asarray([-0.5, 0.5, 0.0, np.sqrt(0.02)])
        np.testing.assert_allclose(actual, expected, atol=5.0e-8)

    def test_relabel_preserves_coordinates_with_partition_mapping(self) -> None:
        source_pos = np.asarray(
            [[1.0, 0.0, 0.0, 0.2], [0.5, 0.0, 0.0, 0.0]], dtype=np.float32
        )
        source_neg = np.asarray(
            [[0.0, 0.0, 0.0, -0.2], [0.4, 0.0, 0.0, -0.1]], dtype=np.float32
        )
        arrays, metrics = relabel_arrays(self.mesh, source_pos, source_neg, chunk_size=2)
        restored = restore_source_coordinate_order(
            arrays["pos"],
            arrays["neg"],
            arrays["pos_source_index"],
            arrays["neg_source_index"],
        )
        source_xyz = np.concatenate((source_pos[:, :3], source_neg[:, :3]), axis=0)
        self.assertEqual(restored.dtype, source_xyz.dtype)
        self.assertTrue(np.array_equal(restored, source_xyz))
        self.assertTrue(np.all(arrays["pos"][:, 3] >= 0.0))
        self.assertTrue(np.all(arrays["neg"][:, 3] < 0.0))
        self.assertEqual(metrics["query_count"], 4)

    def test_atomic_archive_round_trip_in_test_mode(self) -> None:
        arrays = {"pos": np.ones((2, 4), dtype=np.float32), "neg": -np.ones((2, 4), dtype=np.float32)}
        with tempfile.TemporaryDirectory(prefix="d3c_exact_sdf_test_") as directory:
            output = Path(directory) / "tiny.npz"
            atomic_write_npz(output, arrays, allow_non_bulk=True)
            with np.load(output, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["pos"], arrays["pos"])
                np.testing.assert_array_equal(archive["neg"], arrays["neg"])

    def test_persistent_ssd_output_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            require_bulk_path(TASK_DIR / "forbidden_output")

    def test_one_scan_relabel_and_independent_audit(self) -> None:
        relabel = load_script("relabel_exact_triangle_sdf")
        audit = load_script("audit_exact_triangle_sdf")
        with tempfile.TemporaryDirectory(prefix="d3c_exact_sdf_e2e_") as directory:
            root = Path(directory)
            mesh_path = root / "cube.obj"
            source_path = root / "source.npz"
            self.mesh.export(mesh_path)
            source_pos = np.asarray(
                [[1.0, 0.0, 0.0, 0.5], [0.7, 0.0, 0.0, 0.2]], dtype=np.float32
            )
            source_neg = np.asarray(
                [[0.0, 0.0, 0.0, -0.5], [0.4, 0.0, 0.0, -0.1]], dtype=np.float32
            )
            np.savez_compressed(source_path, pos=source_pos, neg=source_neg)
            row = {
                "scan_id": "cube_0",
                "subject_id": "cube",
                "split": "train",
                "structure": "synthetic",
                "mesh_path": str(mesh_path),
                "sdf_npz_path": str(source_path),
                "source_sdf_npz_path": str(source_path),
            }
            relabel_args = SimpleNamespace(
                overwrite=False,
                resume=False,
                chunk_size=2,
                zero_epsilon=1.0e-8,
                sign_check_points=4,
                sign_check_margin=1.0e-5,
                maximum_sign_disagreement_fraction=0.0,
                seed=19,
                allow_non_bulk_output=True,
            )
            exact_row, report = relabel.process_row(row, root, relabel_args, 0)
            self.assertEqual(report["status"], "written")
            audit_args = SimpleNamespace(
                points_per_scan=4,
                seed=23,
                chunk_size=2,
                absolute_tolerance=2.0e-6,
            )
            result = audit.audit_row(row, exact_row, audit_args, 0)
            self.assertTrue(result["coordinates_bitwise_identical"])
            self.assertLessEqual(result["sdf_absolute_error_max"], 2.0e-6)


class ConfigurationTests(unittest.TestCase):
    def test_static_paired_config_contract(self) -> None:
        launch = load_script("launch_pilot")
        report = launch.run_check(launch.load_matrix(), require_data=False)
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["compact_schedule"],
            {"latent_adapt": 100, "global": 0, "local": 0, "joint": 900},
        )
        self.assertEqual(
            report["decoder_warm_start"]["compact"]["latent_adapt_epochs"], 100
        )
        self.assertFalse(report["decoder_warm_start"]["compact"]["source_latents_loaded"])
        self.assertEqual(report["periodic_splits"], ["train", "val"])
        self.assertFalse(report["eikonal_enabled"])

    def test_subject_complete_selection(self) -> None:
        selection = load_script("build_pilot_manifest")
        rows = []
        for subject, count, diagnosis in (("a", 3, "CN"), ("b", 2, "AD"), ("c", 4, "CN")):
            for visit in range(count):
                rows.append(
                    {
                        "scan_id": f"{subject}_{visit}",
                        "subject_id": subject,
                        "diagnosis": diagnosis,
                        "visit_order": str(visit),
                        "correspondence_volume_mm3": str(1000 + visit),
                    }
                )
        selected, report = selection.select_subject_complete(rows, target_scans=5, seed=7)
        selected_subjects = {row["subject_id"] for row in selected}
        for subject in selected_subjects:
            self.assertEqual(
                sum(row["subject_id"] == subject for row in selected),
                sum(row["subject_id"] == subject for row in rows),
            )
        self.assertTrue(report["all_visits_retained_for_selected_subjects"])

    def test_paired_metric_comparison_writes_expected_delta(self) -> None:
        comparison = load_script("compare_paired_evaluations")
        with tempfile.TemporaryDirectory(prefix="d3c_paired_metrics_") as directory:
            root = Path(directory)
            approximate = root / "approx.csv"
            exact = root / "exact.csv"
            fields = ["scan_id", "subject_id", "split", "method", "assd_mm", "hd95_mm"]
            rows = [
                {"scan_id": "a0", "subject_id": "a", "split": "val", "method": "inr", "assd_mm": 0.2, "hd95_mm": 0.4},
                {"scan_id": "b0", "subject_id": "b", "split": "val", "method": "inr", "assd_mm": 0.3, "hd95_mm": 0.5},
            ]
            for path, shift in ((approximate, 0.0), (exact, -0.1)):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for row in rows:
                        item = dict(row)
                        item["assd_mm"] += shift
                        item["hd95_mm"] += shift
                        writer.writerow(item)
            output = root / "comparison"
            original_argv = sys.argv
            try:
                sys.argv = [
                    "compare_paired_evaluations.py",
                    "--family",
                    "compact",
                    "--split",
                    "val",
                    "--approx-metrics",
                    str(approximate),
                    "--exact-metrics",
                    str(exact),
                    "--output-dir",
                    str(output),
                    "--bootstrap-replicates",
                    "100",
                    "--allow-non-bulk-output",
                ]
                comparison.main()
            finally:
                sys.argv = original_argv
            summary = json.loads((output / "summary.json").read_text())
            self.assertAlmostEqual(
                summary["metrics"]["assd_mm"]["mean_exact_minus_approx"], -0.1
            )
            self.assertTrue(
                summary["metrics"]["assd_mm"]["ci_excludes_zero_in_improvement_direction"]
            )


if __name__ == "__main__":
    unittest.main()
