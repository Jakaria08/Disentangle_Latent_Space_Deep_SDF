from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_validation_selection_never_reads_test_metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    analysis = load_module("all_visual_analysis", ROOT / "scripts" / "build_analysis.py")
    rows = []
    for method, assd, volume in (("a", 0.2, 0.03), ("b", 0.3, 0.01)):
        for diagnosis in ("CN", "AD"):
            rows.append({
                "method": method, "diagnosis": diagnosis,
                "prediction_assd_mm": assd,
                "endpoint_volume_rate_abs_error_per_year": volume,
                "velocity_normal_rmse_mm_per_year": 0.1,
                "prediction_flipped_face_fraction_vs_ground_truth": 0.0,
                "relative_cocycle_defect_mean": 0.001,
            })
    ranked = analysis.validation_ranking(pd.DataFrame(rows))
    assert ranked.selected_for_ood.sum() == 1
    assert set(ranked.method) == {"a", "b"}


def test_weighted_velocity_summary(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    analysis = load_module("all_visual_analysis_weighted", ROOT / "scripts" / "build_analysis.py")
    frame = pd.DataFrame({
        "method": ["m", "m"], "method_label": ["M", "M"], "diagnosis": ["CN", "CN"],
        "subject_id": ["1", "2"], "reference_reliability": [1.0, 3.0],
        "predicted_speed_mm_per_year": [1.0, 3.0], "observed_speed_mm_per_year": [2.0, 2.0],
        "predicted_inward_normal_mm_per_year": [0.0, 0.0], "observed_inward_normal_mm_per_year": [0.0, 0.0],
        "vector_rmse_mm_per_year": [1.0, 1.0], "zero_vector_rmse_mm_per_year": [2.0, 2.0],
        "normal_rmse_mm_per_year": [1.0, 1.0], "zero_normal_rmse_mm_per_year": [2.0, 2.0],
        "vector_cosine": [0.0, 1.0], "normal_pearson": [0.0, 1.0], "normal_sign_agreement": [0.5, 1.0],
    })
    result = analysis.summarize_velocity(frame, ["method", "method_label", "diagnosis"]).iloc[0]
    assert result.predicted_speed_mm_per_year == 2.5
    assert result.normal_error_to_observed_ratio == 0.5


def test_every_displayed_figure_has_preceding_markdown():
    notebook = json.loads((ROOT / "all_methods_longitudinal_analysis.ipynb").read_text(encoding="utf-8"))
    cells = notebook["cells"]
    for index, cell in enumerate(cells):
        text = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "figure(" in text:
            assert index > 0
            assert cells[index - 1]["cell_type"] == "markdown"


def test_every_code_cell_has_substantive_preceding_explanation():
    notebook = json.loads((ROOT / "all_methods_longitudinal_analysis.ipynb").read_text(encoding="utf-8"))
    cells = notebook["cells"]
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        assert index > 0
        preceding = cells[index - 1]
        assert preceding["cell_type"] == "markdown"
        assert len("".join(preceding.get("source", [])).split()) >= 20


def test_velocity_explanation_contains_derivation_and_denominators():
    notebook = json.loads((ROOT / "all_methods_longitudinal_analysis.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = (
        "J_D(z_t) v_z_per_year",
        "Jacobian-vector product",
        "61 validation subjects and 269 scans",
        "208 observed baseline-to-later-visit intervals",
        "51 eligible baseline surfaces",
        "1,122 predicted meshes and velocities per method",
        "There is no continuously measured instantaneous ground truth",
    )
    for phrase in required:
        assert phrase in text


def test_pca_ode_baselines_are_full_matched_runs():
    registry = json.loads((ROOT / "configs" / "model_registry.json").read_text(encoding="utf-8"))
    baselines = registry["pca_ode_baselines"]
    assert baselines["representation"] == "pca128"
    assert baselines["surface_split"] == "test"
    assert baselines["velocity_split"] == "val"
    assert set(baselines["methods"]) == {"pca_plain_ode", "pca_brainode"}
    assert {item["method"] for item in baselines["methods"].values()} == {"plain_ode", "brainode"}
    assert all("smoke" not in Path(item["run_dir"]).name.lower() for item in baselines["methods"].values())


def test_surface_progression_geometry_and_rigid_filter(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    progression = load_module("surface_progression_geometry", ROOT / "scripts" / "build_surface_progression.py")
    import numpy as np

    rng = np.random.default_rng(8)
    vertices = rng.normal(size=(40, 3))
    translation = np.array([0.2, -0.1, 0.05])
    omega = np.array([0.03, -0.02, 0.04])
    rigid = translation + np.cross(np.broadcast_to(omega, vertices.shape), vertices - vertices.mean(axis=0))
    residual = progression.remove_rigid_velocity(vertices, rigid)
    assert np.sqrt(np.mean(residual**2)) < 1.0e-10


def test_new_velocity_figures_replace_mixed_cohort_figures():
    notebook = json.loads((ROOT / "all_methods_longitudinal_analysis.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for old in (
        "velocity_speed_5year_latent.png",
        "velocity_speed_5year_direct.png",
        "velocity_disease_contrast_5year.png",
    ):
        assert old not in text
    for new in (
        "velocity_matched_cohort_summary.png",
        "velocity_interval_integrated.png",
        "velocity_ad_minus_cn_surface_map.png",
    ):
        assert new in text


def test_report_figures_keep_only_best_lamm(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    analysis = load_module("all_visual_one_lamm", ROOT / "scripts" / "build_analysis.py")
    frame = pd.DataFrame(
        {
            "method": ["lamm_n3", "lamm_global_256", "lamm_global_384", "mesh_spiral"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    filtered = analysis.one_lamm_for_plot(frame)
    assert set(filtered.method) == {"lamm_n3", "mesh_spiral"}


def test_without_direct_mesh_filter_removes_both_direct_methods(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    analysis = load_module("all_visual_without_direct", ROOT / "scripts" / "build_analysis.py")
    frame = pd.DataFrame(
        {
            "method": [
                "adaptive_spiral_mesh_direct",
                "spiral_mesh_direct",
                "latent_spiral",
                "lamm_n3",
            ]
        }
    )
    filtered = analysis.without_direct_mesh(frame)
    assert set(filtered.method) == {"latent_spiral", "lamm_n3"}


def test_without_direct_companions_immediately_follow_original_figures():
    notebook = json.loads((ROOT / "all_methods_longitudinal_analysis.ipynb").read_text(encoding="utf-8"))
    cells = notebook["cells"]
    pairs = (
        ("prediction_metrics_strict_test.png", "prediction_metrics_strict_test_without_direct_mesh.png"),
        ("prediction_vs_nochange.png", "prediction_vs_nochange_without_direct_mesh.png"),
        ("volume_rate_ad_cn.png", "volume_rate_ad_cn_without_direct_mesh.png"),
        ("velocity_matched_cohort_summary.png", "velocity_matched_cohort_summary_without_direct_mesh.png"),
        ("velocity_regional_inward.png", "velocity_regional_inward_without_direct_mesh.png"),
        ("velocity_interval_integrated.png", "velocity_interval_integrated_without_direct_mesh.png"),
        ("velocity_tangent_consistency.png", "velocity_tangent_consistency_without_direct_mesh.png"),
        ("velocity_paired_condition_trajectory.png", "velocity_paired_condition_trajectory_without_direct_mesh.png"),
        ("surface_inward_displacement_progression.png", "surface_inward_displacement_progression_without_direct_mesh.png"),
        ("surface_radial_narrowing_progression.png", "surface_radial_narrowing_progression_without_direct_mesh.png"),
        ("surface_nonuniform_area_progression.png", "surface_nonuniform_area_progression_without_direct_mesh.png"),
        ("velocity_ad_minus_cn_surface_map.png", "velocity_ad_minus_cn_surface_map_without_direct_mesh.png"),
        ("ood_best_vs_brainode_volume.png", "ood_brainode_only_volume_without_direct_mesh.png"),
    )
    sources = ["".join(cell.get("source", [])) for cell in cells]
    for original, companion in pairs:
        index = next(i for i, text in enumerate(sources) if original in text)
        assert cells[index + 1]["cell_type"] == "markdown"
        assert companion in sources[index + 2]


def test_surface_companion_method_set_has_no_direct_mesh():
    progression = load_module(
        "surface_progression_without_direct",
        ROOT / "scripts" / "build_surface_progression.py",
    )
    assert progression.WITHOUT_DIRECT_METHODS == [
        "latent_spiral",
        "latent_adaptive",
        "latent_pca",
        "lamm_n3",
        "pca_plain_ode",
        "pca_brainode",
    ]
