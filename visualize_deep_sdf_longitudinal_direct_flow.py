#!/usr/bin/env python3
"""Interactive dashboard for direct-flow training and evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


def _load_history(experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_path = experiment_dir / "Logs.pth"
    if not log_path.is_file():
        return pd.DataFrame(), pd.DataFrame()
    payload = torch.load(log_path, map_location="cpu")
    return (
        pd.DataFrame(payload.get("train", [])),
        pd.DataFrame(payload.get("validation", [])),
    )


def build_dashboard(
    experiment: str | Path,
    *,
    checkpoint: str = "best",
    split: str = "val",
) -> go.Figure:
    experiment_dir = Path(experiment).resolve()
    analysis_dir = experiment_dir / "analysis" / f"checkpoint_{checkpoint}"
    pair_path = analysis_dir / f"{split}_pair_metrics.csv"
    trajectory_path = analysis_dir / f"{split}_trajectories.csv"
    sequence_path = analysis_dir / f"{split}_sequence_metrics.csv"
    if not pair_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError(
            "Evaluation outputs are missing. Run "
            "evaluate_deep_sdf_longitudinal_direct_flow.py first."
        )
    pairs = pd.read_csv(pair_path)
    trajectories = pd.read_csv(trajectory_path)
    sequences = pd.read_csv(sequence_path) if sequence_path.is_file() else pd.DataFrame()
    train, validation = _load_history(experiment_dir)
    velocity_column = (
        "instantaneous_velocity_l2_per_normalized_age"
        if "instantaneous_velocity_l2_per_normalized_age" in trajectories.columns
        else "instantaneous_velocity_l2_per_year"
    )

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Training and validation target-SDF error",
            "Prediction versus copying the source shape",
            "Direct-versus-composed cocycle error",
            "Dense chronological-age trajectory diagnostics",
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.11,
    )

    if not train.empty:
        figure.add_trace(
            go.Scatter(
                x=train["epoch"],
                y=train["real_prediction"],
                name="train target SDF",
                mode="lines",
            ),
            row=1,
            col=1,
        )
        if "sequence_rollout_sdf" in train.columns:
            figure.add_trace(
                go.Scatter(
                    x=train["epoch"],
                    y=train["sequence_rollout_sdf"],
                    name="train sequence rollout SDF",
                    mode="lines",
                    line={"dash": "dot"},
                ),
                row=1,
                col=1,
            )
    if not validation.empty:
        figure.add_trace(
            go.Scatter(
                x=validation["epoch"],
                y=validation["real_prediction"],
                name="validation target SDF",
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )
        if "sequence_rollout_sdf" in validation.columns:
            figure.add_trace(
                go.Scatter(
                    x=validation["epoch"],
                    y=validation["sequence_rollout_sdf"],
                    name="validation sequence rollout SDF",
                    mode="lines+markers",
                    line={"dash": "dot"},
                ),
                row=1,
                col=1,
            )
        if "sequence_no_change_sdf" in validation.columns:
            figure.add_trace(
                go.Scatter(
                    x=validation["epoch"],
                    y=validation["sequence_no_change_sdf"],
                    name="validation sequence no-change",
                    mode="lines+markers",
                    line={"dash": "dashdot"},
                ),
                row=1,
                col=1,
            )
        figure.add_trace(
            go.Scatter(
                x=validation["epoch"],
                y=validation["no_change"],
                name="validation no-change",
                mode="lines+markers",
                line={"dash": "dash"},
            ),
            row=1,
            col=1,
        )

    grouped = (
        pairs.groupby(["diagnosis", "pair_type"], as_index=False)[
            ["model_target_sdf_l1", "no_change_target_sdf_l1"]
        ]
        .mean()
    )
    grouped["group"] = grouped["diagnosis"] + " / " + grouped["pair_type"]
    figure.add_trace(
        go.Bar(
            x=grouped["group"],
            y=grouped["model_target_sdf_l1"],
            name="predicted target",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=grouped["group"],
            y=grouped["no_change_target_sdf_l1"],
            name="copy source",
        ),
        row=1,
        col=2,
    )
    if not sequences.empty:
        sequence_grouped = (
            sequences.groupby("diagnosis", as_index=False)[
                ["sequence_rollout_sdf_l1", "sequence_no_change_sdf_l1"]
            ]
            .mean()
        )
        figure.add_trace(
            go.Bar(
                x=sequence_grouped["diagnosis"] + " / sequence",
                y=sequence_grouped["sequence_rollout_sdf_l1"],
                name="sequence rollout",
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Bar(
                x=sequence_grouped["diagnosis"] + " / sequence",
                y=sequence_grouped["sequence_no_change_sdf_l1"],
                name="sequence copy source",
            ),
            row=1,
            col=2,
        )

    observed = pairs.loc[pairs["has_observed_intermediate"].astype(bool)]
    figure.add_trace(
        go.Box(
            x=observed["diagnosis"],
            y=observed["observed_cocycle_mse"],
            name="observed intermediate",
            boxpoints="outliers",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Box(
            x=pairs["diagnosis"],
            y=pairs["virtual_cocycle_mse"],
            name="virtual intermediate",
            boxpoints="outliers",
        ),
        row=2,
        col=1,
    )
    optional_consistency_boxes = [
        ("backward_virtual_latent_mse", "backward virtual"),
        ("future_extrapolation_cocycle_mse", "future extrapolation"),
    ]
    for column, label in optional_consistency_boxes:
        if column in pairs.columns:
            figure.add_trace(
                go.Box(
                    x=pairs["diagnosis"],
                    y=pairs[column],
                    name=label,
                    boxpoints="outliers",
                ),
                row=2,
                col=1,
            )
    if not sequences.empty and "sequence_cocycle_mse" in sequences.columns:
        figure.add_trace(
            go.Box(
                x=sequences["diagnosis"],
                y=sequences["sequence_cocycle_mse"],
                name="sequence rollout cocycle",
                boxpoints="outliers",
            ),
            row=2,
            col=1,
        )

    for (subject_id, diagnosis), group in trajectories.groupby(
        ["subject_id", "diagnosis"]
    ):
        group = group.sort_values("age_years")
        observed_group = group.loc[group["region"] == "observed_range"]
        future_group = group.loc[group["region"] == "future_unvalidated"]
        label = f"{subject_id} ({diagnosis})"
        figure.add_trace(
            go.Scatter(
                x=observed_group["age_years"],
                y=observed_group[velocity_column],
                name=f"{label} observed range",
                mode="lines",
            ),
            row=2,
            col=2,
        )
        if not future_group.empty:
            bridge = pd.concat(
                [observed_group.tail(1), future_group],
                ignore_index=True,
            )
            figure.add_trace(
                go.Scatter(
                    x=bridge["age_years"],
                    y=bridge[velocity_column],
                    name=f"{label} future (no GT)",
                    mode="lines",
                    line={"dash": "dash"},
                ),
                row=2,
                col=2,
            )

    figure.update_yaxes(title_text="clamped SDF L1", row=1, col=1)
    figure.update_yaxes(title_text="clamped SDF L1", row=1, col=2)
    figure.update_yaxes(title_text="latent MSE", type="log", row=2, col=1)
    figure.update_yaxes(
        title_text="instantaneous latent velocity / normalized-age unit",
        row=2,
        col=2,
    )
    figure.update_xaxes(title_text="epoch", row=1, col=1)
    figure.update_xaxes(title_text="diagnosis / pair type", row=1, col=2)
    figure.update_xaxes(title_text="diagnosis", row=2, col=1)
    figure.update_xaxes(title_text="chronological age (years)", row=2, col=2)
    figure.update_layout(
        title=(
            f"No-MCI direct chronological-age flow — checkpoint {checkpoint}, "
            f"{split} split"
        ),
        barmode="group",
        height=950,
        width=1450,
        legend={"orientation": "h", "y": -0.13},
        template="plotly_white",
    )
    return figure


def _as_mesh(path: str | Path):
    import trimesh

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Mesh scene is empty: {path}")
        return trimesh.util.concatenate(geometries)
    return loaded


def _mesh_trace(path: str | Path, name: str, color: str, opacity: float):
    mesh = _as_mesh(path)
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        flatshading=False,
        showscale=False,
    )


def build_shape_figure(
    experiment: str | Path,
    *,
    checkpoint: str = "best",
    split: str = "val",
    subject_id: Optional[str] = None,
) -> go.Figure:
    experiment_dir = Path(experiment).resolve()
    manifest_path = (
        experiment_dir
        / "analysis"
        / f"checkpoint_{checkpoint}"
        / f"{split}_mesh_manifest.csv"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Mesh manifest is missing. Re-run evaluation with "
            "--export-example-meshes --gpu <id>."
        )
    manifest = pd.read_csv(manifest_path)
    if subject_id is None:
        subject_id = str(manifest.iloc[0]["subject_id"])
    rows = manifest.loc[manifest["subject_id"].astype(str) == str(subject_id)]
    kind_order = [
        "baseline",
        "virtual_midpoint",
        "last_real_target",
        "future_unvalidated",
    ]
    rows = rows.set_index("kind").loc[
        [kind for kind in kind_order if kind in set(rows["kind"])]
    ]
    figure = make_subplots(
        rows=1,
        cols=len(rows),
        specs=[[{"type": "scene"} for _ in range(len(rows))]],
        subplot_titles=[
            f"{kind}<br>age {float(row['age_years']):.2f}"
            for kind, row in rows.iterrows()
        ],
    )
    for column, (kind, row) in enumerate(rows.iterrows(), start=1):
        figure.add_trace(
            _mesh_trace(
                row["predicted_mesh"],
                f"predicted {kind}",
                "#d62728",
                0.78,
            ),
            row=1,
            col=column,
        )
        reference_path = None
        reference_name = None
        if kind == "baseline":
            reference_path = row["baseline_real_mesh"]
            reference_name = "real baseline"
        elif kind == "last_real_target":
            reference_path = row["last_real_mesh"]
            reference_name = "real last target"
        if reference_path is not None:
            figure.add_trace(
                _mesh_trace(
                    reference_path,
                    reference_name,
                    "#2ca02c",
                    0.34,
                ),
                row=1,
                col=column,
            )
    figure.update_layout(
        title=(
            f"Direct-flow decoded shapes — {subject_id}; red=prediction, "
            "green=real scan where available"
        ),
        width=max(1100, 360 * len(rows)),
        height=520,
        template="plotly_white",
    )
    for column in range(1, len(rows) + 1):
        figure.update_scenes(
            aspectmode="data",
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            row=1,
            col=column,
        )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--experiment", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--subject-id", default=None)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment).resolve()
    analysis_dir = experiment_dir / "analysis" / f"checkpoint_{args.checkpoint}"
    figure = build_dashboard(
        experiment_dir,
        checkpoint=args.checkpoint,
        split=args.split,
    )
    dashboard_path = analysis_dir / f"{args.split}_dashboard.html"
    figure.write_html(dashboard_path, include_plotlyjs=True)
    print(dashboard_path)
    manifest = analysis_dir / f"{args.split}_mesh_manifest.csv"
    if manifest.is_file():
        shape_figure = build_shape_figure(
            experiment_dir,
            checkpoint=args.checkpoint,
            split=args.split,
            subject_id=args.subject_id,
        )
        shape_path = analysis_dir / f"{args.split}_shape_viewer.html"
        shape_figure.write_html(shape_path, include_plotlyjs=True)
        print(shape_path)


if __name__ == "__main__":
    main(parse_args())
