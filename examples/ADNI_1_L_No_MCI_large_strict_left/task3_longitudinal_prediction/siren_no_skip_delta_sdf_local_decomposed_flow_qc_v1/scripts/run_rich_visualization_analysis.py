#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / "deep_sdf").exists():
            return candidate
    raise RuntimeError(f"Could not find repository root from {start}")


def parse_args() -> argparse.Namespace:
    experiment = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate rich direct-flow visualization HTML/CSV outputs."
    )
    parser.add_argument("--experiment", default=str(experiment))
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--subjects-per-split", type=int, default=50)
    parser.add_argument("--min-scans", type=int, default=3)
    parser.add_argument("--batch-mesh-resolution", type=int, default=80)
    parser.add_argument("--case-mesh-resolution", type=int, default=96)
    parser.add_argument("--counterfactual-mesh-resolution", type=int, default=80)
    parser.add_argument("--mesh-max-batch", type=int, default=2**18)
    parser.add_argument("--ood-ages", default="92,95,100,105")
    parser.add_argument("--counterfactual-horizon-years", type=float, default=10.0)
    parser.add_argument("--counterfactual-step-years", type=float, default=2.0)
    parser.add_argument("--composed-step-years", type=float, default=0.5)
    parser.add_argument("--change-sample-count", type=int, default=3000)
    parser.add_argument("--skip-batch-volume-decode", action="store_true")
    parser.add_argument("--skip-shapes", action="store_true")
    parser.add_argument("--skip-mci-reference", action="store_true")
    return parser.parse_args()


def save_selected_subject_outputs(bundle, rich, direct_helpers, selected_subjects):
    selection_summary = rich.selection_summary_table(selected_subjects)
    paths = rich.save_tables(
        bundle.html_dir,
        {
            "selected_subjects_50_per_split": selected_subjects,
            "selected_subject_summary": selection_summary,
        },
    )
    selection_fig = rich.make_selection_figure(selected_subjects)
    paths["selection_figure"] = direct_helpers.save_figure(
        bundle,
        selection_fig,
        "selected_subjects_summary",
    )
    return paths


def save_metric_outputs(bundle, direct_helpers, rich) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    gap_path = bundle.analysis_dir / "gap_bin_summary.csv"
    if gap_path.is_file():
        gap_summary = pd.read_csv(gap_path)
        gap_fig = rich.make_gap_bin_figure(gap_summary)
        paths["gap_bin"] = direct_helpers.save_figure(
            bundle,
            gap_fig,
            "gap_bin_performance",
        )
    for split, fig in direct_helpers.build_all_dashboard_figures(bundle).items():
        paths[f"{split}_dashboard"] = direct_helpers.save_figure(
            bundle,
            fig,
            f"{split}_dashboard",
        )
    return paths


def save_volume_outputs(
    bundle,
    direct_helpers,
    rich,
    selected_subjects,
    args: argparse.Namespace,
) -> tuple[Dict[str, Path], pd.DataFrame]:
    if args.skip_batch_volume_decode:
        trend_path = bundle.html_dir / "selected_observed_age_volume_trend.csv"
        if not trend_path.is_file():
            print(
                "Skipping selected volume decode and no previous selected trend CSV exists."
            )
            return {}, pd.DataFrame()
        trend_frame = pd.read_csv(trend_path)
        subject_volume_summary = pd.read_csv(
            bundle.html_dir / "selected_subject_volume_summary.csv"
        )
        decode_status = pd.read_csv(bundle.html_dir / "selected_decode_status.csv")
    else:
        selected_volume_dataset = rich.build_selected_observed_age_volume_trend_dataset(
            bundle,
            selected_subjects,
            composed_step_years=args.composed_step_years,
            mesh_resolution=args.batch_mesh_resolution,
            mesh_max_batch=args.mesh_max_batch,
        )
        trend_frame = selected_volume_dataset["trend_frame"]
        subject_volume_summary = selected_volume_dataset["subject_summary"]
        decode_status = selected_volume_dataset["decode_status"]

    speed_frame = rich.adjacent_volume_speed_frame(trend_frame)
    speed_summary = rich.speed_summary_table(speed_frame)
    prediction_consistency = rich.prediction_consistency_table(subject_volume_summary)
    paths = rich.save_tables(
        bundle.html_dir,
        {
            "selected_observed_age_volume_trend": trend_frame,
            "selected_subject_volume_summary": subject_volume_summary,
            "selected_decode_status": decode_status,
            "selected_adjacent_volume_speed": speed_frame,
            "selected_speed_summary": speed_summary,
            "selected_prediction_consistency": prediction_consistency,
        },
    )
    abs_age_fig = rich.observed_age_volume_trend_figure(trend_frame)
    paths["absolute_age_volume_html"] = direct_helpers.save_figure(
        bundle,
        abs_age_fig,
        "selected_absolute_age_volume_trend",
    )
    elapsed_fig = rich.elapsed_relative_volume_trend_figure(trend_frame)
    paths["elapsed_relative_volume_html"] = direct_helpers.save_figure(
        bundle,
        elapsed_fig,
        "selected_elapsed_relative_volume_trend",
    )
    delta_fig = rich.make_final_delta_comparison_figure(subject_volume_summary)
    paths["final_delta_html"] = direct_helpers.save_figure(
        bundle,
        delta_fig,
        "selected_final_volume_delta_real_vs_prediction",
    )
    speed_fig = rich.make_speed_box_figure(speed_frame)
    paths["speed_box_html"] = direct_helpers.save_figure(
        bundle,
        speed_fig,
        "selected_volume_speed_real_vs_prediction",
    )
    return paths, speed_frame


def save_representative_interpolation_outputs(
    bundle,
    direct_helpers,
    rich,
    available_splits,
    args: argparse.Namespace,
    ood_ages: List[float],
) -> List[Path]:
    output_paths: List[Path] = []
    records = []
    for split in ("train", "val", "test"):
        if split not in available_splits:
            continue
        for diagnosis in ("CN", "AD"):
            try:
                pair_row = direct_helpers.select_representative_pair(
                    bundle,
                    split=split,
                    pair_type="nonadjacent",
                    diagnosis=diagnosis,
                    require_observed_intermediate=True,
                    max_source_age_years=min(ood_ages),
                )
                case = direct_helpers.build_pair_case(
                    bundle,
                    pair_row,
                    mesh_resolution=args.case_mesh_resolution,
                    mesh_max_batch=args.mesh_max_batch,
                )
                figures = {
                    "interpolation": direct_helpers.interpolation_figure(case),
                    "direct_vs_composed": direct_helpers.far_pair_direct_composed_figure(case),
                    "pair_volume": direct_helpers.pair_volume_trend_figure(case),
                }
                for name, fig in figures.items():
                    output_paths.append(
                        direct_helpers.save_figure(
                            bundle,
                            fig,
                            f"{split}_{diagnosis.lower()}_{name}",
                        )
                    )
                records.append(
                    {
                        "split": split,
                        "diagnosis": diagnosis,
                        "subject_id": case["subject_id"],
                        "source_scan_id": case["source_row"]["scan_id"],
                        "target_scan_id": case["target_row"]["scan_id"],
                        "status": "ok",
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "split": split,
                        "diagnosis": diagnosis,
                        "subject_id": "",
                        "source_scan_id": "",
                        "target_scan_id": "",
                        "status": f"{type(exc).__name__}: {exc}",
                    }
                )
    rich.save_tables(
        bundle.html_dir,
        {"representative_interpolation_cases": pd.DataFrame.from_records(records)},
    )
    return output_paths


def save_forecast_outputs(
    bundle,
    direct_helpers,
    rich,
    available_splits,
    args: argparse.Namespace,
    ood_ages: List[float],
) -> List[Path]:
    output_paths: List[Path] = []
    records = []
    for split in ("train", "val", "test"):
        if split not in available_splits:
            continue
        for diagnosis in ("CN", "AD"):
            try:
                pair_row = direct_helpers.select_representative_pair(
                    bundle,
                    split=split,
                    pair_type="nonadjacent",
                    diagnosis=diagnosis,
                    require_observed_intermediate=True,
                    max_source_age_years=min(ood_ages),
                )
                case = direct_helpers.build_subject_forecast_case(
                    bundle,
                    pair_row,
                    ood_ages_years=ood_ages,
                    composed_step_years=args.composed_step_years,
                    mesh_resolution=args.case_mesh_resolution,
                    mesh_max_batch=args.mesh_max_batch,
                )
                output_paths.append(
                    direct_helpers.save_figure(
                        bundle,
                        direct_helpers.subject_forecast_volume_figure(case),
                        f"{split}_{diagnosis.lower()}_subject_volume_forecast",
                    )
                )
                observed_fig, observed_summary = direct_helpers.observed_target_change_heatmap_figure(
                    case,
                    sample_count=args.change_sample_count,
                )
                output_paths.append(
                    direct_helpers.save_figure(
                        bundle,
                        observed_fig,
                        f"{split}_{diagnosis.lower()}_observed_target_change",
                    )
                )
                direct_ood_summary = pd.DataFrame()
                composed_ood_summary = pd.DataFrame()
                if case["ood_ages_years"]:
                    direct_ood_fig, direct_ood_summary = direct_helpers.ood_change_heatmap_figure(
                        case,
                        method="direct",
                        sample_count=args.change_sample_count,
                    )
                    output_paths.append(
                        direct_helpers.save_figure(
                            bundle,
                            direct_ood_fig,
                            f"{split}_{diagnosis.lower()}_ood_direct_change",
                        )
                    )
                    composed_ood_fig, composed_ood_summary = direct_helpers.ood_change_heatmap_figure(
                        case,
                        method="composed",
                        sample_count=args.change_sample_count,
                    )
                    output_paths.append(
                        direct_helpers.save_figure(
                            bundle,
                            composed_ood_fig,
                            f"{split}_{diagnosis.lower()}_ood_composed_change",
                        )
                    )
                records.append(
                    {
                        "split": split,
                        "diagnosis": diagnosis,
                        "subject_id": case["subject_id"],
                        "source_scan_id": case["source_row"]["scan_id"],
                        "source_age_years": float(case["source_age_years"]),
                        "last_real_age_years": float(case["last_real_age_years"]),
                        "ood_ages_years": ",".join(
                            f"{age:.1f}" for age in case["ood_ages_years"]
                        ),
                        "observed_mean_surface_shift": float(
                            observed_summary["mean_surface_shift"].mean()
                        )
                        if not observed_summary.empty
                        else float("nan"),
                        "direct_ood_mean_surface_shift": float(
                            direct_ood_summary["mean_surface_shift"].mean()
                        )
                        if not direct_ood_summary.empty
                        else float("nan"),
                        "composed_ood_mean_surface_shift": float(
                            composed_ood_summary["mean_surface_shift"].mean()
                        )
                        if not composed_ood_summary.empty
                        else float("nan"),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "split": split,
                        "diagnosis": diagnosis,
                        "subject_id": "",
                        "source_scan_id": "",
                        "source_age_years": float("nan"),
                        "last_real_age_years": float("nan"),
                        "ood_ages_years": "",
                        "observed_mean_surface_shift": float("nan"),
                        "direct_ood_mean_surface_shift": float("nan"),
                        "composed_ood_mean_surface_shift": float("nan"),
                        "status": f"{type(exc).__name__}: {exc}",
                    }
                )
    rich.save_tables(
        bundle.html_dir,
        {"representative_forecast_cases": pd.DataFrame.from_records(records)},
    )
    return output_paths


def save_counterfactual_outputs(
    bundle,
    direct_helpers,
    rich,
    selected_subjects,
    args: argparse.Namespace,
) -> List[Path]:
    output_paths: List[Path] = []
    records = []
    sources = rich.select_counterfactual_source_rows(
        bundle,
        selected_subjects,
        per_split_per_diagnosis=1,
    )
    for _, source in sources.iterrows():
        try:
            case = rich.build_counterfactual_condition_case(
                bundle,
                split=str(source["split"]),
                subject_id=str(source["subject_id"]),
                horizon_years=args.counterfactual_horizon_years,
                evaluation_step_years=args.counterfactual_step_years,
                mesh_resolution=args.counterfactual_mesh_resolution,
                mesh_max_batch=args.mesh_max_batch,
            )
            volume_fig = rich.counterfactual_volume_figure(case)
            output_paths.append(
                direct_helpers.save_figure(
                    bundle,
                    volume_fig,
                    f"{case['split']}_{case['source_diagnosis'].lower()}_{case['subject_id']}_counterfactual_volume",
                )
            )
            change_fig, _ = rich.counterfactual_final_change_figure(
                case,
                sample_count=args.change_sample_count,
            )
            output_paths.append(
                direct_helpers.save_figure(
                    bundle,
                    change_fig,
                    f"{case['split']}_{case['source_diagnosis'].lower()}_{case['subject_id']}_counterfactual_final_change",
                )
            )
            status = case["decode_status"]
            finals = (
                case["trend_frame"]
                .sort_values("age_years")
                .groupby("condition_name")
                .tail(1)
            )
            cn_final = finals.loc[finals["condition_name"] == "CN_condition", "volume"]
            ad_final = finals.loc[finals["condition_name"] == "AD_condition", "volume"]
            records.append(
                {
                    "split": case["split"],
                    "subject_id": case["subject_id"],
                    "source_diagnosis": case["source_diagnosis"],
                    "source_age_years": float(case["source_age_years"]),
                    "final_age_years": float(case["final_age_years"]),
                    "cn_condition_final_volume": float(cn_final.iloc[0])
                    if not cn_final.empty
                    else float("nan"),
                    "ad_condition_final_volume": float(ad_final.iloc[0])
                    if not ad_final.empty
                    else float("nan"),
                    "decode_success_fraction": float(
                        status["decode_success"].astype(bool).mean()
                    )
                    if not status.empty
                    else float("nan"),
                    "status": "ok",
                }
            )
        except Exception as exc:
            records.append(
                {
                    "split": str(source.get("split", "")),
                    "subject_id": str(source.get("subject_id", "")),
                    "source_diagnosis": str(source.get("diagnosis", "")),
                    "source_age_years": float("nan"),
                    "final_age_years": float("nan"),
                    "cn_condition_final_volume": float("nan"),
                    "ad_condition_final_volume": float("nan"),
                    "decode_success_fraction": float("nan"),
                    "status": f"{type(exc).__name__}: {exc}",
                }
            )
    rich.save_tables(
        bundle.html_dir,
        {"counterfactual_condition_summary": pd.DataFrame.from_records(records)},
    )
    return output_paths


def save_mci_reference_outputs(
    repo: Path,
    bundle,
    direct_helpers,
    rich,
    mci_helpers,
    speed_frame: pd.DataFrame,
) -> List[Path]:
    output_paths: List[Path] = []
    with_mci_root = (
        repo / "examples" / "ADNI_1_L_With_MCI" / "brainode_comparison_task1_manifest_original"
    )
    with_mci = mci_helpers.load_manifest(
        with_mci_root,
        mci_helpers.WITH_MCI_PREFIX,
        name="ADNI original with-MCI",
    )
    mci_visit_df = mci_helpers.build_visit_dataframe(with_mci.clean_df)
    mci_pair_df = mci_helpers.build_adjacent_pair_dataframe(with_mci.clean_df)
    mci_subject_df = mci_helpers.subject_mean_speed_dataframe(mci_pair_df)
    mci_hotspot = mci_helpers.shape_hotspot_summary_table(
        with_mci.clean_df,
        ["CN", "MCI", "AD"],
        top_fraction=0.10,
        focus="inward",
    )
    mci_similarity = mci_helpers.pairwise_shape_similarity_table(
        with_mci.clean_df,
        ["CN", "MCI", "AD"],
        top_fraction=0.10,
        focus="inward",
    )
    rich.save_tables(
        bundle.html_dir,
        {
            "mci_reference_visit_volume": mci_visit_df,
            "mci_reference_adjacent_speed": mci_pair_df,
            "mci_reference_subject_speed": mci_subject_df,
            "mci_reference_hotspot_summary": mci_hotspot,
            "mci_reference_hotspot_similarity": mci_similarity,
        },
    )
    comparison_fig = rich.make_mci_reference_speed_comparison_figure(
        speed_frame,
        mci_pair_df,
    )
    output_paths.append(
        direct_helpers.save_figure(bundle, comparison_fig, "mci_reference_speed_comparison")
    )
    mci_figures = []
    mci_figures.extend(mci_helpers.step2_figures(with_mci))
    mci_figures.extend(
        mci_helpers.common_longitudinal_figures(
            with_mci,
            ["CN", "MCI", "AD"],
            "with-MCI",
        )
    )
    mci_figures.extend(
        mci_helpers.local_shape_figures(
            with_mci.clean_df,
            "ADNI original with-MCI",
            ["CN", "MCI", "AD"],
        )
    )
    mci_figures.extend(
        mci_helpers.shape_difference_figures(
            with_mci.clean_df,
            "ADNI original with-MCI",
            ["CN", "MCI", "AD"],
            [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")],
        )
    )
    for idx, fig in enumerate(mci_figures, start=1):
        output_paths.append(
            direct_helpers.save_figure(bundle, fig, f"mci_reference_{idx:02d}")
        )
    return output_paths


def main() -> int:
    args = parse_args()
    experiment = Path(args.experiment).resolve()
    repo = find_repo_root(experiment)
    for extra in [
        repo,
        repo / "examples" / "ADNI_1_L_No_MCI",
        repo / "examples" / "ADNI_1_L_With_MCI",
        experiment / "scripts",
    ]:
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    import direct_flow_rich_notebook_helpers as direct_helpers
    import rich_visualization_helpers as rich
    import adni_original_speed_helpers as mci_helpers

    device = f"cuda:{args.gpu}" if args.gpu is not None else args.device
    ood_ages = [float(value) for value in args.ood_ages.split(",") if value.strip()]
    bundle = direct_helpers.load_bundle(experiment, checkpoint=args.checkpoint, device=device)
    print(f"Loaded {experiment}")
    print(f"Checkpoint {args.checkpoint}, epoch {bundle.checkpoint['epoch']}")
    print(f"Output directory: {bundle.html_dir}")

    available_splits = direct_helpers.available_summary_splits(bundle)
    metric_paths = save_metric_outputs(bundle, direct_helpers, rich)
    selected_subjects = rich.select_balanced_subjects(
        bundle,
        subjects_per_split=args.subjects_per_split,
        min_scans=args.min_scans,
    )
    selection_paths = save_selected_subject_outputs(
        bundle,
        rich,
        direct_helpers,
        selected_subjects,
    )
    trend_paths, speed_frame = save_volume_outputs(
        bundle,
        direct_helpers,
        rich,
        selected_subjects,
        args,
    )

    interpolation_paths: List[Path] = []
    forecast_paths: List[Path] = []
    counterfactual_paths: List[Path] = []
    if not args.skip_shapes:
        interpolation_paths = save_representative_interpolation_outputs(
            bundle,
            direct_helpers,
            rich,
            available_splits,
            args,
            ood_ages,
        )
        forecast_paths = save_forecast_outputs(
            bundle,
            direct_helpers,
            rich,
            available_splits,
            args,
            ood_ages,
        )
        counterfactual_paths = save_counterfactual_outputs(
            bundle,
            direct_helpers,
            rich,
            selected_subjects,
            args,
        )

    mci_paths: List[Path] = []
    if not args.skip_mci_reference:
        mci_paths = save_mci_reference_outputs(
            repo,
            bundle,
            direct_helpers,
            rich,
            mci_helpers,
            speed_frame,
        )

    index_path = rich.write_html_index(
        bundle.html_dir,
        title="Large Strict Direct-Flow Rich Analysis",
        sections=[
            ("Metric dashboards", list(metric_paths.values())),
            (
                "Subject selection and volume trend",
                [p for p in selection_paths.values() if str(p).endswith(".html")]
                + [p for p in trend_paths.values() if str(p).endswith(".html")],
            ),
            ("Representative interpolation pages", interpolation_paths),
            ("Representative observed and OOD forecast pages", forecast_paths),
            ("Counterfactual CN-vs-AD condition pages", counterfactual_paths),
            ("With-MCI ground-truth reference pages", mci_paths),
        ],
    )
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
