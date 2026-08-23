from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh


TASK = Path(__file__).resolve().parents[1]
SCRIPTS = TASK / "scripts"
REPO = TASK.parents[2]
sys.path.insert(0, str(SCRIPTS))

from build_manifest import select_balanced  # noqa: E402
from audit_exact_triangle_sdf import independent_ray_signed_distance  # noqa: E402
from calsnic_common import (  # noqa: E402
    exact_signed_distance,
    relabel_arrays,
    require_bulk_path,
    restore_source_order,
    sdf_vertices_to_mm,
)


def test_independent_audit_uses_global_containment_sign(monkeypatch):
    class SyntheticMesh:
        @staticmethod
        def contains(points):
            assert points.shape == (2, 3)
            return np.asarray([True, False])

    def fake_closest_point(_mesh, points):
        return points.copy(), np.asarray([0.25, 0.5]), np.asarray([3, 7])

    monkeypatch.setattr(trimesh.proximity, "closest_point", fake_closest_point)
    signed = independent_ray_signed_distance(SyntheticMesh(), np.zeros((2, 3)))
    assert np.array_equal(signed, [-0.25, 0.5])


def test_exact_relabel_preserves_coordinates_and_rebuilds_signs():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    query = np.asarray(
        [[0.8, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
        dtype=np.float32,
    )
    labels = exact_signed_distance(mesh, query)
    assert np.allclose(labels, [0.3, 0.3, -0.5, -0.4], atol=1.0e-7)
    source_pos = np.column_stack((query[labels >= 0.0], labels[labels >= 0.0])).astype(np.float32)
    source_neg = np.column_stack((query[labels < 0.0], labels[labels < 0.0])).astype(np.float32)
    arrays, report = relabel_arrays(mesh, source_pos, source_neg, chunk_size=2)
    restored = restore_source_order(
        arrays["pos"], arrays["neg"], arrays["pos_source_index"], arrays["neg_source_index"]
    )
    source_xyz = np.concatenate((source_pos[:, :3], source_neg[:, :3]), axis=0)
    assert restored.dtype == source_xyz.dtype
    assert np.array_equal(restored, source_xyz)
    assert np.all(arrays["pos"][:, 3] >= 0.0)
    assert np.all(arrays["neg"][:, 3] < 0.0)
    assert report["query_count"] == 4
    assert int(arrays["format_version"].item()) == 2


def test_subject_specific_sdf_to_mm_transform_roundtrip():
    row = {
        "scan_id": "synthetic",
        "mesh_center_x": "0.1",
        "mesh_center_y": "-0.2",
        "mesh_center_z": "0.3",
        "scaled_from_mm_scale": "0.01",
        "scaled_from_mm_tx": "0.5",
        "scaled_from_mm_ty": "-0.1",
        "scaled_from_mm_tz": "0.2",
    }
    mm = np.asarray([[10.0, 20.0, 30.0], [-4.0, 5.0, 8.0]])
    translation = np.asarray([0.5, -0.1, 0.2])
    center = np.asarray([0.1, -0.2, 0.3])
    sdf = mm * 0.01 + translation - center
    assert np.allclose(sdf_vertices_to_mm(sdf, row), mm)


def test_balanced_selection_is_deterministic_and_exact_size():
    records = []
    for index in range(40):
        records.append(
            {
                "scan_id": f"S{index:03d}",
                "study": f"study{index % 2}",
                "site": f"site{index % 5}",
                "sex": "F" if index % 2 else "M",
                "age_bin": str(index % 4),
                "etiv_bin": str((index // 2) % 4),
            }
        )
    first = select_balanced(records, 7, 123)
    second = select_balanced(records, 7, 123)
    assert len(first) == 7
    assert [row["scan_id"] for row in first] == [row["scan_id"] for row in second]
    assert len({row["scan_id"] for row in first}) == 7


def test_bulk_guard_rejects_workspace_output():
    with pytest.raises(ValueError):
        require_bulk_path(REPO / "forbidden-output")
    assert require_bulk_path("/tmp/calsnic-test", allow_non_bulk=True) == Path(
        "/tmp/calsnic-test"
    )


def test_configs_are_validation_only_during_periodic_evaluation():
    configs = sorted((TASK / "configs").glob("*.json"))
    assert {path.name for path in configs} == {
        "calsnic_control_L_multires64_z256_exact.json",
        "calsnic_control_L_multires128_z256_exact.json",
    }
    for path in configs:
        config = json.loads(path.read_text())
        assert config["periodic_evaluation"]["splits"] == ["val"]
        assert config["sdf_supervision"]["exact_triangle_distance"] is True
        assert Path(config["output_dir"]).is_relative_to(Path("/mnt/bulk10tb"))
        assert config["network_specs"]["grid_resolution"] == max(
            config["network_specs"]["grid_resolutions"]
        )


def test_shared_validator_applies_optional_mesh_center(tmp_path):
    mesh = trimesh.creation.box(extents=(0.4, 0.6, 0.8))
    mesh.apply_translation((4.0, -3.0, 2.0))
    mesh_path = tmp_path / "mesh.obj"
    mesh.export(mesh_path)
    sdf_path = tmp_path / "samples.npz"
    np.savez(
        sdf_path,
        pos=np.asarray([[0.9, 0.9, 0.9, 0.5]], dtype=np.float32),
        neg=np.asarray([[0.0, 0.0, 0.0, -0.1]], dtype=np.float32),
    )
    reference = REPO / "examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_multires_single_field_v1/scripts/multires_common.py"
    spec = importlib.util.spec_from_file_location("calsnic_test_shared_common", reference)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    center = mesh.bounds.mean(axis=0)
    report = module.validate_manifest_contract(
        [
            {
                "scan_id": "one",
                "subject_id": "one",
                "split": "train",
                "mesh_path": str(mesh_path),
                "sdf_npz_path": str(sdf_path),
                "mesh_center_x": str(center[0]),
                "mesh_center_y": str(center[1]),
                "mesh_center_z": str(center[2]),
            }
        ],
        [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    )
    assert report["optional_manifest_mesh_center_applied"] is True
    assert np.allclose(report["population_mesh_min"], [-0.2, -0.3, -0.4], atol=1e-5)
