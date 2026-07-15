from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree
import torch
import trimesh

from deep_sdf import mesh as deep_sdf_mesh
from evaluate_deep_sdf_longitudinal_direct_flow import load_trained_components
from visualize_deep_sdf_longitudinal_direct_flow import build_dashboard


@dataclass
class DirectFlowNotebookBundle:
    experiment_dir: Path
    checkpoint_name: str
    specs: Mapping[str, object]
    contract: object
    decoder: torch.nn.Module
    flow: torch.nn.Module
    checkpoint: Mapping[str, object]
    decoder_epoch: int
    device: torch.device

    @property
    def analysis_dir(self) -> Path:
        return self.experiment_dir / "analysis" / f"checkpoint_{self.checkpoint_name}"

    @property
    def html_dir(self) -> Path:
        return self.experiment_dir / "analysis" / f"notebook_{self.checkpoint_name}"


def choose_device(device: str = "auto") -> torch.device:
    normalized = str(device).strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if normalized == "cuda":
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device(normalized)
    raise ValueError(f"Unsupported device specifier: {device!r}")


def load_bundle(
    experiment: str | Path,
    *,
    checkpoint: str = "best",
    device: str = "auto",
) -> DirectFlowNotebookBundle:
    experiment_dir = Path(experiment).resolve()
    resolved_device = choose_device(device)
    if resolved_device.type == "cuda":
        torch.cuda.set_device(resolved_device.index or 0)
        resolved_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    specs, contract, decoder, flow, checkpoint_payload, decoder_epoch = (
        load_trained_components(
            experiment_dir,
            checkpoint,
            resolved_device,
        )
    )
    bundle = DirectFlowNotebookBundle(
        experiment_dir=experiment_dir,
        checkpoint_name=checkpoint,
        specs=specs,
        contract=contract,
        decoder=decoder,
        flow=flow,
        checkpoint=checkpoint_payload,
        decoder_epoch=int(decoder_epoch),
        device=resolved_device,
    )
    bundle.html_dir.mkdir(parents=True, exist_ok=True)
    return bundle


def available_summary_splits(bundle: DirectFlowNotebookBundle) -> List[str]:
    result = []
    for split in ("train", "val", "test"):
        if (bundle.analysis_dir / f"{split}_summary.csv").is_file():
            result.append(split)
    return result


def available_sequence_summary_splits(bundle: DirectFlowNotebookBundle) -> List[str]:
    result = []
    for split in ("train", "val", "test"):
        if (bundle.analysis_dir / f"{split}_sequence_summary.csv").is_file():
            result.append(split)
    return result


def require_summary_splits(
    bundle: DirectFlowNotebookBundle,
    splits: Sequence[str],
) -> None:
    missing = [
        split
        for split in splits
        if not (bundle.analysis_dir / f"{split}_summary.csv").is_file()
    ]
    if missing:
        split_arg = "all" if set(missing) == {"train", "val", "test"} else ",".join(missing)
        raise FileNotFoundError(
            "Missing evaluation outputs for splits "
            f"{missing}. Run:\n"
            f"/home/jakaria/anaconda3/envs/inr_sdf/bin/python "
            f"evaluate_deep_sdf_longitudinal_direct_flow.py -e {bundle.experiment_dir} "
            f"--checkpoint {bundle.checkpoint_name} --split all --gpu 0"
        )


def summary_table(bundle: DirectFlowNotebookBundle) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    sequence_splits = set(available_sequence_summary_splits(bundle))
    for split in available_summary_splits(bundle):
        summary = pd.read_csv(bundle.analysis_dir / f"{split}_summary.csv")
        overall = summary.loc[summary["grouping"] == "overall"].iloc[0]
        row = {
            "split": split,
            "rows": int(overall["rows"]),
            "model_target_sdf_l1": float(overall["model_target_sdf_l1_mean"]),
            "no_change_target_sdf_l1": float(overall["no_change_target_sdf_l1_mean"]),
            "sdf_l1_improvement": float(overall["sdf_l1_improvement_mean"]),
            "relative_improvement_percent": 100.0
            * float(overall["sdf_l1_improvement_mean"])
            / float(overall["no_change_target_sdf_l1_mean"]),
            "model_beats_no_change_fraction": float(
                overall["model_beats_no_change_fraction"]
            ),
        }
        if split in sequence_splits:
            sequence_summary = pd.read_csv(
                bundle.analysis_dir / f"{split}_sequence_summary.csv"
            )
            sequence_overall = sequence_summary.loc[
                sequence_summary["grouping"] == "overall"
            ].iloc[0]
            row.update(
                {
                    "sequence_rows": int(sequence_overall["rows"]),
                    "sequence_rollout_sdf_l1": float(
                        sequence_overall["sequence_rollout_sdf_l1_mean"]
                    ),
                    "sequence_no_change_sdf_l1": float(
                        sequence_overall["sequence_no_change_sdf_l1_mean"]
                    ),
                    "sequence_rollout_improvement": float(
                        sequence_overall["sequence_rollout_improvement_mean"]
                    ),
                    "sequence_rollout_beats_no_change_fraction": float(
                        sequence_overall["rollout_beats_no_change_fraction"]
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def diagnosis_summary_table(
    bundle: DirectFlowNotebookBundle,
    split: str,
) -> pd.DataFrame:
    frame = pd.read_csv(bundle.analysis_dir / f"{split}_summary.csv")
    return frame.loc[frame["grouping"] == "diagnosis"].reset_index(drop=True)


def pair_type_summary_table(
    bundle: DirectFlowNotebookBundle,
    split: str,
) -> pd.DataFrame:
    frame = pd.read_csv(bundle.analysis_dir / f"{split}_summary.csv")
    return frame.loc[frame["grouping"] == "pair_type"].reset_index(drop=True)


def sequence_summary_table(
    bundle: DirectFlowNotebookBundle,
    split: str,
) -> pd.DataFrame:
    path = bundle.analysis_dir / f"{split}_sequence_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run evaluation first for split={split!r}."
        )
    return pd.read_csv(path)


def sequence_diagnosis_summary_table(
    bundle: DirectFlowNotebookBundle,
    split: str,
) -> pd.DataFrame:
    frame = sequence_summary_table(bundle, split)
    return frame.loc[frame["grouping"] == "diagnosis"].reset_index(drop=True)


def dashboard_figure(
    bundle: DirectFlowNotebookBundle,
    split: str,
) -> go.Figure:
    return build_dashboard(
        bundle.experiment_dir,
        checkpoint=bundle.checkpoint_name,
        split=split,
    )


def save_figure(
    bundle: DirectFlowNotebookBundle,
    figure: go.Figure,
    stem: str,
) -> Path:
    output_path = bundle.html_dir / f"{stem}.html"
    figure.write_html(output_path, include_plotlyjs=True)
    return output_path


def clear_generated_html(bundle: DirectFlowNotebookBundle) -> List[Path]:
    removed: List[Path] = []
    for path in sorted(bundle.html_dir.glob("*.html")):
        path.unlink()
        removed.append(path)
    return removed


def age_origin_years(bundle: DirectFlowNotebookBundle) -> float:
    return float(bundle.specs["AgeNormalization"]["minimum_age_years"])


def age_range_years(bundle: DirectFlowNotebookBundle) -> float:
    return float(bundle.specs["AgeNormalization"]["age_range_years"])


def normalize_age_years(bundle: DirectFlowNotebookBundle, age_years: float) -> float:
    return (float(age_years) - age_origin_years(bundle)) / age_range_years(bundle)


def denormalize_age(bundle: DirectFlowNotebookBundle, age_norm: float) -> float:
    return age_origin_years(bundle) + age_range_years(bundle) * float(age_norm)


def normalized_step_from_years(bundle: DirectFlowNotebookBundle, step_years: float) -> float:
    return float(step_years) / age_range_years(bundle)


def _metadata_by_scan(bundle: DirectFlowNotebookBundle) -> pd.DataFrame:
    frame = bundle.contract.metadata.copy()
    return frame.set_index("scan_id", drop=False)


def scan_row(bundle: DirectFlowNotebookBundle, scan_id: str) -> pd.Series:
    by_scan = _metadata_by_scan(bundle)
    return by_scan.loc[str(scan_id)]


def subject_rows(
    bundle: DirectFlowNotebookBundle,
    subject_id: str,
    *,
    split: Optional[str] = None,
) -> pd.DataFrame:
    frame = bundle.contract.metadata
    mask = frame["subject_id"].astype(str) == str(subject_id)
    if split is not None:
        mask &= frame["split"].astype(str) == str(split)
    return frame.loc[mask].sort_values(
        ["continuous_age_norm", "visit_order", "scan_id"]
    ).reset_index(drop=True)


def load_pair_metrics(
    bundle: DirectFlowNotebookBundle,
    split: str,
) -> pd.DataFrame:
    path = bundle.analysis_dir / f"{split}_pair_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run evaluation first for split={split!r}."
        )
    return pd.read_csv(path)


def _condition_tensor(bundle: DirectFlowNotebookBundle, label_ad: int) -> torch.Tensor:
    return torch.tensor(
        [[float(label_ad)]],
        dtype=torch.float32,
        device=bundle.device,
    )


def _time_tensor(bundle: DirectFlowNotebookBundle, age_norm: float) -> torch.Tensor:
    return torch.tensor(
        [[float(age_norm)]],
        dtype=torch.float32,
        device=bundle.device,
    )


def _latent_tensor(
    bundle: DirectFlowNotebookBundle,
    split: str,
    scan_id: str,
) -> torch.Tensor:
    return (
        bundle.contract.latent_maps[str(split)][str(scan_id)]
        .to(bundle.device)
        .float()
        .view(1, -1)
    )


def load_mesh(mesh_like: str | Path | trimesh.Trimesh) -> trimesh.Trimesh:
    if isinstance(mesh_like, trimesh.Trimesh):
        return mesh_like
    loaded = trimesh.load(str(mesh_like), process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Empty mesh scene at {mesh_like}")
        return trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(loaded)!r}")
    return loaded


def mesh_volume(mesh_like: str | Path | trimesh.Trimesh) -> float:
    return float(abs(load_mesh(mesh_like).volume))


def mesh_volume_or_nan(
    mesh_like: Optional[str | Path | trimesh.Trimesh],
) -> float:
    if mesh_like is None:
        return float("nan")
    return mesh_volume(mesh_like)


def try_decode_mesh(
    bundle: DirectFlowNotebookBundle,
    latent: torch.Tensor,
    *,
    resolution: int = 96,
    max_batch: int = 2 ** 18,
    fallback_resolutions: Sequence[int] = (80, 72, 64),
) -> Tuple[Optional[trimesh.Trimesh], Optional[str]]:
    if bundle.device.type != "cuda":
        return None, (
            "Mesh decoding requires CUDA because deep_sdf.mesh.create_mesh uses GPU."
        )
    tried = [int(resolution), *[int(value) for value in fallback_resolutions]]
    last_error: Optional[str] = None
    for current_resolution in tried:
        try:
            root_logger = logging.getLogger()
            old_level = root_logger.level
            root_logger.setLevel(max(old_level, logging.CRITICAL))
            with torch.no_grad():
                try:
                    mesh = deep_sdf_mesh.create_mesh(
                        bundle.decoder,
                        latent,
                        N=int(current_resolution),
                        max_batch=int(max_batch),
                        return_trimesh=True,
                    )
                finally:
                    root_logger.setLevel(old_level)
        except Exception as exc:
            last_error = (
                f"{type(exc).__name__} at resolution {current_resolution}: {exc}"
            )
            continue
        if mesh is not None:
            return mesh, None
    if last_error is not None:
        return None, (
            "Marching cubes failed for every attempted resolution "
            f"{tried}. Last error: {last_error}"
        )
    return None, (
        "Marching cubes failed for every attempted resolution: "
        f"{tried}"
    )


def decode_mesh(
    bundle: DirectFlowNotebookBundle,
    latent: torch.Tensor,
    *,
    resolution: int = 96,
    max_batch: int = 2 ** 18,
    fallback_resolutions: Sequence[int] = (80, 72, 64),
) -> trimesh.Trimesh:
    mesh, error = try_decode_mesh(
        bundle,
        latent,
        resolution=resolution,
        max_batch=max_batch,
        fallback_resolutions=fallback_resolutions,
    )
    if mesh is not None:
        return mesh
    raise RuntimeError(error or "Mesh decoding failed.")


def transport_direct(
    bundle: DirectFlowNotebookBundle,
    latent: torch.Tensor,
    source_time: torch.Tensor,
    target_time: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return bundle.flow.transport(latent, source_time, target_time, condition)


def transport_composed_fixed_step(
    bundle: DirectFlowNotebookBundle,
    latent: torch.Tensor,
    *,
    start_age_years: float,
    end_age_years: float,
    label_ad: int,
    step_years: float = 0.5,
) -> torch.Tensor:
    current_latent = latent
    current_age = float(start_age_years)
    condition = _condition_tensor(bundle, label_ad)
    while current_age < float(end_age_years) - 1e-8:
        next_age = min(current_age + float(step_years), float(end_age_years))
        current_latent = transport_direct(
            bundle,
            current_latent,
            _time_tensor(bundle, normalize_age_years(bundle, current_age)),
            _time_tensor(bundle, normalize_age_years(bundle, next_age)),
            condition,
        )
        current_age = next_age
    return current_latent


def _mesh_trace(
    mesh_like: str | Path | trimesh.Trimesh,
    name: str,
    color: str,
    opacity: float,
    *,
    showlegend: bool,
) -> go.Mesh3d:
    mesh = load_mesh(mesh_like)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        flatshading=False,
        showscale=False,
        showlegend=showlegend,
    )


def mesh_panel_figure(
    panels: Sequence[Tuple[str, Sequence[Tuple[str, str | Path | trimesh.Trimesh, str, float]]]],
    *,
    title: str,
) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=len(panels),
        specs=[[{"type": "scene"} for _ in panels]],
        subplot_titles=[panel_title for panel_title, _ in panels],
    )
    showlegend = True
    for column, (_, entries) in enumerate(panels, start=1):
        for name, mesh_like, color, opacity in entries:
            figure.add_trace(
                _mesh_trace(
                    mesh_like,
                    name,
                    color,
                    opacity,
                    showlegend=showlegend,
                ),
                row=1,
                col=column,
            )
            showlegend = False
        figure.update_scenes(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
            row=1,
            col=column,
        )
    figure.update_layout(
        title=title,
        width=max(1200, 360 * len(panels)),
        height=520,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.08},
    )
    return figure


def volume_trend_figure(
    frame: pd.DataFrame,
    *,
    x_col: str,
    title: str,
) -> go.Figure:
    figure = go.Figure()
    for transport_method, group in frame.groupby("transport_method", sort=False):
        group = group.sort_values(x_col)
        figure.add_trace(
            go.Scatter(
                x=group[x_col],
                y=group["volume"],
                mode="lines+markers",
                name=str(transport_method),
            )
        )
    real = frame.loc[frame["transport_method"] == "real_observed"].sort_values(x_col)
    if not real.empty:
        figure.add_trace(
            go.Scatter(
                x=real[x_col],
                y=real["volume"],
                mode="markers",
                marker={"size": 10, "symbol": "diamond"},
                name="real_observed",
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title=x_col.replace("_", " "),
        yaxis_title="mesh volume",
        width=950,
        height=520,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.12},
    )
    return figure


def select_best_pair(
    bundle: DirectFlowNotebookBundle,
    *,
    diagnosis: str,
    pair_type: str,
    split_priority: Sequence[str] = ("test", "val", "train"),
    require_observed_intermediate: bool = False,
) -> pd.Series:
    for split in split_priority:
        path = bundle.analysis_dir / f"{split}_pair_metrics.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame = frame.loc[
            (frame["diagnosis"] == diagnosis) & (frame["pair_type"] == pair_type)
        ].copy()
        if require_observed_intermediate:
            frame = frame.loc[frame["has_observed_intermediate"].astype(bool)]
        if frame.empty:
            continue
        frame = frame.sort_values(
            ["sdf_l1_improvement", "model_beats_no_change"],
            ascending=[False, False],
        ).reset_index(drop=True)
        row = frame.iloc[0].copy()
        row["selected_split"] = split
        return row
    raise ValueError(
        f"No pair available for diagnosis={diagnosis!r}, pair_type={pair_type!r}."
    )


def _observed_intermediate_row(
    bundle: DirectFlowNotebookBundle,
    *,
    subject_id: str,
    split: str,
    source_age_norm: float,
    target_age_norm: float,
) -> pd.Series:
    rows = subject_rows(bundle, subject_id, split=split)
    middle = rows.loc[
        (rows["continuous_age_norm"] > float(source_age_norm))
        & (rows["continuous_age_norm"] < float(target_age_norm))
    ].copy()
    if middle.empty:
        raise ValueError(f"No observed intermediate rows for subject {subject_id!r}.")
    center = 0.5 * (float(source_age_norm) + float(target_age_norm))
    middle["distance_to_center"] = np.abs(middle["continuous_age_norm"] - center)
    middle = middle.sort_values(
        ["distance_to_center", "continuous_age_norm", "visit_order"]
    ).reset_index(drop=True)
    return middle.iloc[0]


def build_pair_case(
    bundle: DirectFlowNotebookBundle,
    pair_row: pd.Series,
    *,
    mesh_resolution: int = 96,
    mesh_max_batch: int = 2 ** 18,
) -> Dict[str, object]:
    split = str(pair_row["selected_split"])
    source_scan_id = str(pair_row["source_scan_id"])
    target_scan_id = str(pair_row["target_scan_id"])
    source_row = scan_row(bundle, source_scan_id)
    target_row = scan_row(bundle, target_scan_id)
    subject_id = str(source_row["subject_id"])
    label_ad = int(source_row["label_ad"])

    source_latent = _latent_tensor(bundle, split, source_scan_id)
    target_latent = _latent_tensor(bundle, split, target_scan_id)
    source_time = _time_tensor(bundle, float(source_row["continuous_age_norm"]))
    target_time = _time_tensor(bundle, float(target_row["continuous_age_norm"]))
    condition = _condition_tensor(bundle, label_ad)

    direct_target_latent = transport_direct(
        bundle,
        source_latent,
        source_time,
        target_time,
        condition,
    )

    virtual_mid_norm = 0.5 * (
        float(source_row["continuous_age_norm"]) + float(target_row["continuous_age_norm"])
    )
    virtual_mid_time = _time_tensor(bundle, virtual_mid_norm)
    virtual_mid_latent = transport_direct(
        bundle,
        source_latent,
        source_time,
        virtual_mid_time,
        condition,
    )

    observed_mid_row = None
    observed_mid_latent = None
    composed_target_latent = None
    if bool(pair_row["has_observed_intermediate"]):
        observed_mid_row = _observed_intermediate_row(
            bundle,
            subject_id=subject_id,
            split=split,
            source_age_norm=float(source_row["continuous_age_norm"]),
            target_age_norm=float(target_row["continuous_age_norm"]),
        )
        observed_mid_time = _time_tensor(
            bundle,
            float(observed_mid_row["continuous_age_norm"]),
        )
        observed_mid_latent = transport_direct(
            bundle,
            source_latent,
            source_time,
            observed_mid_time,
            condition,
        )
        composed_target_latent = transport_direct(
            bundle,
            observed_mid_latent,
            observed_mid_time,
            target_time,
            condition,
        )

    source_mesh = load_mesh(source_row["mesh_path"])
    target_mesh = load_mesh(target_row["mesh_path"])
    direct_target_mesh = decode_mesh(
        bundle,
        direct_target_latent,
        resolution=mesh_resolution,
        max_batch=mesh_max_batch,
    )
    virtual_mid_mesh = decode_mesh(
        bundle,
        virtual_mid_latent,
        resolution=mesh_resolution,
        max_batch=mesh_max_batch,
    )

    observed_mid_mesh = None
    observed_mid_gt_mesh = None
    composed_target_mesh = None
    if observed_mid_row is not None and observed_mid_latent is not None:
        observed_mid_mesh = decode_mesh(
            bundle,
            observed_mid_latent,
            resolution=mesh_resolution,
            max_batch=mesh_max_batch,
        )
        observed_mid_gt_mesh = load_mesh(observed_mid_row["mesh_path"])
        composed_target_mesh = decode_mesh(
            bundle,
            composed_target_latent,
            resolution=mesh_resolution,
            max_batch=mesh_max_batch,
        )

    return {
        "pair_row": pair_row,
        "split": split,
        "subject_id": subject_id,
        "diagnosis": str(source_row["diagnosis"]),
        "label_ad": label_ad,
        "source_row": source_row,
        "target_row": target_row,
        "source_mesh": source_mesh,
        "target_mesh": target_mesh,
        "source_latent": source_latent,
        "target_latent": target_latent,
        "source_time": source_time,
        "target_time": target_time,
        "condition": condition,
        "direct_target_latent": direct_target_latent,
        "direct_target_mesh": direct_target_mesh,
        "virtual_mid_norm": virtual_mid_norm,
        "virtual_mid_age_years": denormalize_age(bundle, virtual_mid_norm),
        "virtual_mid_latent": virtual_mid_latent,
        "virtual_mid_mesh": virtual_mid_mesh,
        "observed_mid_row": observed_mid_row,
        "observed_mid_latent": observed_mid_latent,
        "observed_mid_mesh": observed_mid_mesh,
        "observed_mid_gt_mesh": observed_mid_gt_mesh,
        "composed_target_latent": composed_target_latent,
        "composed_target_mesh": composed_target_mesh,
    }


def near_pair_figure(case: Mapping[str, object]) -> go.Figure:
    source_row = case["source_row"]
    target_row = case["target_row"]
    pair_row = case["pair_row"]
    panels = [
        (
            f"Source age {float(source_row['continuous_age_years']):.1f}",
            [("source", case["source_mesh"], "#7f7f7f", 0.85)],
        ),
        (
            "No-change vs ground truth",
            [
                ("no-change", case["source_mesh"], "#d62728", 0.72),
                ("ground truth", case["target_mesh"], "#2ca02c", 0.35),
            ],
        ),
        (
            "Direct one-shot vs ground truth",
            [
                ("direct", case["direct_target_mesh"], "#1f77b4", 0.72),
                ("ground truth", case["target_mesh"], "#2ca02c", 0.35),
            ],
        ),
    ]
    return mesh_panel_figure(
        panels,
        title=(
            f"{case['diagnosis']} adjacent pair | subject {case['subject_id']} | "
            f"{source_row['scan_id']} -> {target_row['scan_id']} | "
            f"SDF improvement {float(pair_row['sdf_l1_improvement']):.6f}"
        ),
    )


def far_pair_direct_composed_figure(case: Mapping[str, object]) -> go.Figure:
    if case["observed_mid_row"] is None:
        raise ValueError("Far-pair figure requires an observed intermediate.")
    observed_mid_row = case["observed_mid_row"]
    target_row = case["target_row"]
    pair_row = case["pair_row"]
    panels = [
        (
            f"Observed midpoint age {float(observed_mid_row['continuous_age_years']):.1f}",
            [
                ("predicted midpoint", case["observed_mid_mesh"], "#ff7f0e", 0.72),
                ("real midpoint", case["observed_mid_gt_mesh"], "#2ca02c", 0.35),
            ],
        ),
        (
            f"Direct final age {float(target_row['continuous_age_years']):.1f}",
            [
                ("direct final", case["direct_target_mesh"], "#1f77b4", 0.72),
                ("ground truth", case["target_mesh"], "#2ca02c", 0.35),
            ],
        ),
        (
            f"Composed final age {float(target_row['continuous_age_years']):.1f}",
            [
                ("composed final", case["composed_target_mesh"], "#d62728", 0.72),
                ("ground truth", case["target_mesh"], "#2ca02c", 0.35),
            ],
        ),
    ]
    return mesh_panel_figure(
        panels,
        title=(
            f"{case['diagnosis']} nonadjacent pair | subject {case['subject_id']} | "
            f"direct vs composed | SDF improvement {float(pair_row['sdf_l1_improvement']):.6f}"
        ),
    )


def interpolation_figure(case: Mapping[str, object]) -> go.Figure:
    panels: List[Tuple[str, Sequence[Tuple[str, str | Path | trimesh.Trimesh, str, float]]]] = [
        (
            f"Source age {float(case['source_row']['continuous_age_years']):.1f}",
            [("source", case["source_mesh"], "#7f7f7f", 0.82)],
        ),
        (
            f"Virtual midpoint age {float(case['virtual_mid_age_years']):.2f}",
            [("virtual midpoint", case["virtual_mid_mesh"], "#9467bd", 0.80)],
        ),
    ]
    if case["observed_mid_row"] is not None:
        observed_mid_row = case["observed_mid_row"]
        panels.append(
            (
                f"Observed midpoint age {float(observed_mid_row['continuous_age_years']):.1f}",
                [
                    ("predicted observed midpoint", case["observed_mid_mesh"], "#ff7f0e", 0.72),
                    ("real observed midpoint", case["observed_mid_gt_mesh"], "#2ca02c", 0.35),
                ],
            )
        )
    panels.append(
        (
            f"Target age {float(case['target_row']['continuous_age_years']):.1f}",
            [("ground truth target", case["target_mesh"], "#2ca02c", 0.72)],
        )
    )
    return mesh_panel_figure(
        panels,
        title=f"{case['diagnosis']} interpolation views | subject {case['subject_id']}",
    )


def pair_volume_trend_figure(case: Mapping[str, object]) -> go.Figure:
    rows = [
        {
            "transport_method": "real_observed",
            "age_years": float(case["source_row"]["continuous_age_years"]),
            "volume": mesh_volume(case["source_mesh"]),
        },
        {
            "transport_method": "direct",
            "age_years": float(case["target_row"]["continuous_age_years"]),
            "volume": mesh_volume(case["direct_target_mesh"]),
        },
        {
            "transport_method": "real_observed",
            "age_years": float(case["target_row"]["continuous_age_years"]),
            "volume": mesh_volume(case["target_mesh"]),
        },
        {
            "transport_method": "no_change",
            "age_years": float(case["target_row"]["continuous_age_years"]),
            "volume": mesh_volume(case["source_mesh"]),
        },
    ]
    if case["observed_mid_row"] is not None:
        rows.extend(
            [
                {
                    "transport_method": "direct_mid",
                    "age_years": float(case["observed_mid_row"]["continuous_age_years"]),
                    "volume": mesh_volume(case["observed_mid_mesh"]),
                },
                {
                    "transport_method": "real_observed",
                    "age_years": float(case["observed_mid_row"]["continuous_age_years"]),
                    "volume": mesh_volume(case["observed_mid_gt_mesh"]),
                },
                {
                    "transport_method": "composed",
                    "age_years": float(case["target_row"]["continuous_age_years"]),
                    "volume": mesh_volume(case["composed_target_mesh"]),
                },
            ]
        )
    rows.append(
        {
            "transport_method": "virtual_mid",
            "age_years": float(case["virtual_mid_age_years"]),
            "volume": mesh_volume(case["virtual_mid_mesh"]),
        }
    )
    frame = pd.DataFrame(rows)
    return volume_trend_figure(
        frame,
        x_col="age_years",
        title=f"{case['diagnosis']} pair volume trend | subject {case['subject_id']}",
    )


def select_subject_for_extrapolation(
    bundle: DirectFlowNotebookBundle,
    *,
    diagnosis: str,
    split_priority: Sequence[str] = ("test", "val", "train"),
) -> pd.Series:
    metadata = bundle.contract.metadata.copy()
    metadata = metadata.loc[metadata["diagnosis"] == diagnosis].copy()
    grouped_rows: List[Dict[str, object]] = []
    for split in split_priority:
        split_frame = metadata.loc[metadata["split"] == split].copy()
        if split_frame.empty:
            continue
        for subject_id, group in split_frame.groupby("subject_id", sort=True):
            group = group.sort_values(["continuous_age_norm", "visit_order"])
            grouped_rows.append(
                {
                    "split": split,
                    "subject_id": str(subject_id),
                    "diagnosis": diagnosis,
                    "num_scans": int(len(group)),
                    "age_span_years": float(
                        group["continuous_age_years"].max()
                        - group["continuous_age_years"].min()
                    ),
                    "last_age_years": float(group["continuous_age_years"].max()),
                }
            )
    if not grouped_rows:
        raise ValueError(f"No subject found for diagnosis={diagnosis!r}.")
    frame = pd.DataFrame(grouped_rows).sort_values(
        ["num_scans", "age_span_years", "last_age_years"],
        ascending=[False, False, False],
    )
    return frame.iloc[0]


def build_extrapolation_case(
    bundle: DirectFlowNotebookBundle,
    *,
    subject_id: str,
    split: str,
    horizon_years: float = 2.0,
    step_years: float = 0.5,
    mesh_resolution: int = 96,
    mesh_max_batch: int = 2 ** 18,
) -> Dict[str, object]:
    rows = subject_rows(bundle, subject_id, split=split)
    if rows.empty:
        raise ValueError(f"No rows for subject {subject_id!r} in split {split!r}.")
    start_row = rows.iloc[-1]
    label_ad = int(start_row["label_ad"])
    start_latent = _latent_tensor(bundle, split, str(start_row["scan_id"]))
    start_age_years = float(start_row["continuous_age_years"])
    final_age_years = start_age_years + float(horizon_years)

    real_rows = []
    for _, row in rows.iterrows():
        real_rows.append(
            {
                "transport_method": "real_observed",
                "age_years": float(row["continuous_age_years"]),
                "volume": mesh_volume(row["mesh_path"]),
            }
        )

    direct_rows = []
    composed_rows = []
    age_grid = np.arange(
        start_age_years,
        final_age_years + 1e-8,
        float(step_years),
    )
    if age_grid[-1] < final_age_years - 1e-8:
        age_grid = np.append(age_grid, final_age_years)

    direct_final_mesh = None
    composed_final_mesh = None
    for age_years in age_grid:
        direct_latent = transport_direct(
            bundle,
            start_latent,
            _time_tensor(bundle, normalize_age_years(bundle, start_age_years)),
            _time_tensor(bundle, normalize_age_years(bundle, float(age_years))),
            _condition_tensor(bundle, label_ad),
        )
        direct_mesh = decode_mesh(
            bundle,
            direct_latent,
            resolution=mesh_resolution,
            max_batch=mesh_max_batch,
        )
        direct_rows.append(
            {
                "transport_method": "direct",
                "age_years": float(age_years),
                "volume": mesh_volume(direct_mesh),
            }
        )
        composed_latent = transport_composed_fixed_step(
            bundle,
            start_latent,
            start_age_years=start_age_years,
            end_age_years=float(age_years),
            label_ad=label_ad,
            step_years=float(step_years),
        )
        composed_mesh = decode_mesh(
            bundle,
            composed_latent,
            resolution=mesh_resolution,
            max_batch=mesh_max_batch,
        )
        composed_rows.append(
            {
                "transport_method": "composed",
                "age_years": float(age_years),
                "volume": mesh_volume(composed_mesh),
            }
        )
        if abs(float(age_years) - final_age_years) < 1e-8:
            direct_final_mesh = direct_mesh
            composed_final_mesh = composed_mesh

    return {
        "subject_id": subject_id,
        "split": split,
        "diagnosis": str(start_row["diagnosis"]),
        "start_row": start_row,
        "rows": rows,
        "real_frame": pd.DataFrame(real_rows),
        "trend_frame": pd.DataFrame(real_rows + direct_rows + composed_rows),
        "direct_final_mesh": direct_final_mesh,
        "composed_final_mesh": composed_final_mesh,
        "final_age_years": float(final_age_years),
    }


def extrapolation_volume_figure(case: Mapping[str, object]) -> go.Figure:
    return volume_trend_figure(
        case["trend_frame"],
        x_col="age_years",
        title=(
            f"{case['diagnosis']} extrapolation volume trend | subject {case['subject_id']} | "
            f"start age {float(case['start_row']['continuous_age_years']):.1f}"
        ),
    )


def extrapolation_shape_figure(case: Mapping[str, object]) -> go.Figure:
    start_mesh = load_mesh(case["start_row"]["mesh_path"])
    panels = [
        (
            f"Latest observed age {float(case['start_row']['continuous_age_years']):.1f}",
            [("latest observed", start_mesh, "#7f7f7f", 0.82)],
        ),
        (
            f"Direct future age {float(case['final_age_years']):.1f}",
            [("direct future", case["direct_final_mesh"], "#1f77b4", 0.78)],
        ),
        (
            f"Composed future age {float(case['final_age_years']):.1f}",
            [("composed future", case["composed_final_mesh"], "#d62728", 0.78)],
        ),
    ]
    return mesh_panel_figure(
        panels,
        title=f"{case['diagnosis']} extrapolated future shapes | subject {case['subject_id']}",
    )


def build_all_dashboard_figures(
    bundle: DirectFlowNotebookBundle,
    *,
    preferred_splits: Sequence[str] = ("train", "val", "test"),
) -> Dict[str, go.Figure]:
    figures: Dict[str, go.Figure] = {}
    for split in preferred_splits:
        pair_path = bundle.analysis_dir / f"{split}_pair_metrics.csv"
        traj_path = bundle.analysis_dir / f"{split}_trajectories.csv"
        if pair_path.is_file() and traj_path.is_file():
            figures[split] = dashboard_figure(bundle, split)
    return figures


def _age_key(age_years: float) -> str:
    return f"{float(age_years):.6f}"


def _unique_sorted_ages(values: Iterable[float]) -> List[float]:
    ordered = sorted(float(value) for value in values)
    result: List[float] = []
    for age in ordered:
        if not result or abs(age - result[-1]) > 1e-6:
            result.append(age)
    return result


def select_representative_pair(
    bundle: DirectFlowNotebookBundle,
    *,
    split: str,
    pair_type: str = "nonadjacent",
    diagnosis: Optional[str] = None,
    require_observed_intermediate: bool = True,
    max_source_age_years: Optional[float] = None,
) -> pd.Series:
    frame = load_pair_metrics(bundle, split)
    frame = frame.loc[frame["pair_type"] == pair_type].copy()
    if diagnosis is not None:
        frame = frame.loc[frame["diagnosis"] == diagnosis].copy()
    if require_observed_intermediate:
        frame = frame.loc[frame["has_observed_intermediate"].astype(bool)].copy()
    if max_source_age_years is not None:
        younger = frame.loc[
            frame["source_age_years"].astype(float) < float(max_source_age_years)
        ].copy()
        if not younger.empty:
            frame = younger
    if frame.empty:
        raise ValueError(
            f"No representative pair for split={split!r}, pair_type={pair_type!r}, "
            f"diagnosis={diagnosis!r}."
        )
    frame = frame.sort_values(
        [
            "model_beats_no_change",
            "has_observed_intermediate",
            "sdf_l1_improvement",
            "gap_years",
            "source_age_years",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    row = frame.iloc[0].copy()
    row["selected_split"] = str(split)
    return row


def _future_subject_rows_from_source(
    bundle: DirectFlowNotebookBundle,
    *,
    subject_id: str,
    split: str,
    source_age_years: float,
) -> pd.DataFrame:
    rows = subject_rows(bundle, subject_id, split=split).copy()
    rows = rows.loc[rows["continuous_age_years"].astype(float) >= float(source_age_years) - 1e-6]
    return rows.reset_index(drop=True)


def _predict_meshes_from_source(
    bundle: DirectFlowNotebookBundle,
    *,
    source_latent: torch.Tensor,
    source_age_years: float,
    label_ad: int,
    target_ages_years: Sequence[float],
    composed_step_years: float,
    mesh_resolution: int,
    mesh_max_batch: int,
    source_mesh: trimesh.Trimesh,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, Optional[trimesh.Trimesh]],
    Dict[str, Optional[trimesh.Trimesh]],
    Dict[str, Optional[str]],
    Dict[str, Optional[str]],
]:
    source_time = _time_tensor(bundle, normalize_age_years(bundle, source_age_years))
    condition = _condition_tensor(bundle, label_ad)

    direct_latents: Dict[str, torch.Tensor] = {}
    composed_latents: Dict[str, torch.Tensor] = {}
    direct_meshes: Dict[str, Optional[trimesh.Trimesh]] = {}
    composed_meshes: Dict[str, Optional[trimesh.Trimesh]] = {}
    direct_errors: Dict[str, Optional[str]] = {}
    composed_errors: Dict[str, Optional[str]] = {}

    for age_years in _unique_sorted_ages(target_ages_years):
        key = _age_key(age_years)
        if abs(float(age_years) - float(source_age_years)) <= 1e-6:
            direct_latent = source_latent
            composed_latent = source_latent
            direct_mesh = source_mesh
            composed_mesh = source_mesh
            direct_error = None
            composed_error = None
        else:
            target_time = _time_tensor(bundle, normalize_age_years(bundle, float(age_years)))
            direct_latent = transport_direct(
                bundle,
                source_latent,
                source_time,
                target_time,
                condition,
            )
            direct_mesh, direct_error = try_decode_mesh(
                bundle,
                direct_latent,
                resolution=mesh_resolution,
                max_batch=mesh_max_batch,
            )
            composed_latent = transport_composed_fixed_step(
                bundle,
                source_latent,
                start_age_years=float(source_age_years),
                end_age_years=float(age_years),
                label_ad=int(label_ad),
                step_years=float(composed_step_years),
            )
            composed_mesh, composed_error = try_decode_mesh(
                bundle,
                composed_latent,
                resolution=mesh_resolution,
                max_batch=mesh_max_batch,
            )
        direct_latents[key] = direct_latent
        composed_latents[key] = composed_latent
        direct_meshes[key] = direct_mesh
        composed_meshes[key] = composed_mesh
        direct_errors[key] = direct_error
        composed_errors[key] = composed_error
    return (
        direct_latents,
        composed_latents,
        direct_meshes,
        composed_meshes,
        direct_errors,
        composed_errors,
    )


def build_subject_forecast_case(
    bundle: DirectFlowNotebookBundle,
    pair_row: pd.Series,
    *,
    ood_ages_years: Sequence[float] = (92.0, 95.0, 100.0, 105.0),
    composed_step_years: float = 0.5,
    mesh_resolution: int = 96,
    mesh_max_batch: int = 2 ** 18,
) -> Dict[str, object]:
    split = str(pair_row["selected_split"]) if "selected_split" in pair_row else str(pair_row["split"])
    source_scan_id = str(pair_row["source_scan_id"])
    target_scan_id = str(pair_row["target_scan_id"])
    source_row = scan_row(bundle, source_scan_id)
    target_row = scan_row(bundle, target_scan_id)
    subject_id = str(source_row["subject_id"])
    label_ad = int(source_row["label_ad"])
    source_age_years = float(source_row["continuous_age_years"])
    target_age_years = float(target_row["continuous_age_years"])

    future_rows = _future_subject_rows_from_source(
        bundle,
        subject_id=subject_id,
        split=split,
        source_age_years=source_age_years,
    )
    real_age_targets = _unique_sorted_ages(future_rows["continuous_age_years"].astype(float))
    filtered_ood_ages = _unique_sorted_ages(
        age for age in ood_ages_years if float(age) > source_age_years + 1e-6
    )
    prediction_ages = _unique_sorted_ages(list(real_age_targets) + list(filtered_ood_ages))

    source_latent = _latent_tensor(bundle, split, source_scan_id)
    source_mesh = load_mesh(source_row["mesh_path"])
    (
        direct_latents,
        composed_latents,
        direct_meshes,
        composed_meshes,
        direct_errors,
        composed_errors,
    ) = _predict_meshes_from_source(
        bundle,
        source_latent=source_latent,
        source_age_years=source_age_years,
        label_ad=label_ad,
        target_ages_years=prediction_ages,
        composed_step_years=composed_step_years,
        mesh_resolution=mesh_resolution,
        mesh_max_batch=mesh_max_batch,
        source_mesh=source_mesh,
    )

    real_meshes: Dict[str, trimesh.Trimesh] = {}
    real_rows_by_age: Dict[str, pd.Series] = {}
    for _, row in future_rows.iterrows():
        key = _age_key(float(row["continuous_age_years"]))
        real_meshes[key] = load_mesh(row["mesh_path"])
        real_rows_by_age[key] = row

    last_real_age_years = max(real_age_targets)
    volume_rows: List[Dict[str, object]] = []
    observed_volume_rows: List[Dict[str, object]] = []
    decode_status_rows: List[Dict[str, object]] = []
    observed_key_set = {_age_key(age) for age in real_age_targets}

    for age_years in prediction_ages:
        key = _age_key(age_years)
        region = "observed" if key in observed_key_set else "ood"
        decode_status_rows.append(
            {
                "age_years": float(age_years),
                "region": region,
                "transport_method": "direct",
                "decode_success": direct_meshes.get(key) is not None,
                "error": direct_errors.get(key),
            }
        )
        decode_status_rows.append(
            {
                "age_years": float(age_years),
                "region": region,
                "transport_method": "composed",
                "decode_success": composed_meshes.get(key) is not None,
                "error": composed_errors.get(key),
            }
        )

    for _, row in future_rows.iterrows():
        age_years = float(row["continuous_age_years"])
        key = _age_key(age_years)
        real_volume = mesh_volume(real_meshes[key])
        direct_volume = mesh_volume_or_nan(direct_meshes.get(key))
        composed_volume = mesh_volume_or_nan(composed_meshes.get(key))
        no_change_volume = mesh_volume(source_mesh)
        volume_rows.append(
            {
                "transport_method": "real_observed",
                "age_years": age_years,
                "volume": real_volume,
                "region": "observed",
            }
        )
        volume_rows.append(
            {
                "transport_method": "direct",
                "age_years": age_years,
                "volume": direct_volume,
                "region": "observed",
            }
        )
        volume_rows.append(
            {
                "transport_method": "composed",
                "age_years": age_years,
                "volume": composed_volume,
                "region": "observed",
            }
        )
        volume_rows.append(
            {
                "transport_method": "no_change",
                "age_years": age_years,
                "volume": no_change_volume,
                "region": "observed",
            }
        )
        if age_years > source_age_years + 1e-6:
            observed_volume_rows.append(
                {
                    "age_years": age_years,
                    "real_volume": real_volume,
                    "direct_volume": direct_volume,
                    "composed_volume": composed_volume,
                    "no_change_volume": no_change_volume,
                    "direct_decode_success": direct_meshes.get(key) is not None,
                    "composed_decode_success": composed_meshes.get(key) is not None,
                    "direct_abs_error": abs(direct_volume - real_volume) if np.isfinite(direct_volume) else float("nan"),
                    "composed_abs_error": abs(composed_volume - real_volume) if np.isfinite(composed_volume) else float("nan"),
                    "no_change_abs_error": abs(no_change_volume - real_volume),
                    "direct_signed_error": direct_volume - real_volume if np.isfinite(direct_volume) else float("nan"),
                    "composed_signed_error": composed_volume - real_volume if np.isfinite(composed_volume) else float("nan"),
                    "no_change_signed_error": no_change_volume - real_volume,
                    "direct_error": direct_errors.get(key),
                    "composed_error": composed_errors.get(key),
                }
            )

    for age_years in filtered_ood_ages:
        key = _age_key(age_years)
        volume_rows.append(
            {
                "transport_method": "direct",
                "age_years": float(age_years),
                "volume": mesh_volume_or_nan(direct_meshes.get(key)),
                "region": "ood",
            }
        )
        volume_rows.append(
            {
                "transport_method": "composed",
                "age_years": float(age_years),
                "volume": mesh_volume_or_nan(composed_meshes.get(key)),
                "region": "ood",
            }
        )
        volume_rows.append(
            {
                "transport_method": "no_change",
                "age_years": float(age_years),
                "volume": mesh_volume(source_mesh),
                "region": "ood",
            }
        )

    decode_status_table = pd.DataFrame(decode_status_rows).sort_values(
        ["age_years", "transport_method"]
    ).reset_index(drop=True)
    available_ood_ages_by_method = {
        "direct": [
            float(age)
            for age in filtered_ood_ages
            if direct_meshes.get(_age_key(float(age))) is not None
        ],
        "composed": [
            float(age)
            for age in filtered_ood_ages
            if composed_meshes.get(_age_key(float(age))) is not None
        ],
    }

    return {
        "bundle": bundle,
        "pair_row": pair_row,
        "split": split,
        "subject_id": subject_id,
        "diagnosis": str(source_row["diagnosis"]),
        "label_ad": label_ad,
        "source_row": source_row,
        "target_row": target_row,
        "source_mesh": source_mesh,
        "source_latent": source_latent,
        "source_age_years": source_age_years,
        "target_age_years": target_age_years,
        "future_rows": future_rows,
        "real_meshes": real_meshes,
        "real_rows_by_age": real_rows_by_age,
        "real_age_targets": real_age_targets,
        "ood_ages_years": filtered_ood_ages,
        "prediction_ages": prediction_ages,
        "direct_latents": direct_latents,
        "composed_latents": composed_latents,
        "direct_meshes": direct_meshes,
        "composed_meshes": composed_meshes,
        "direct_decode_errors": direct_errors,
        "composed_decode_errors": composed_errors,
        "decode_status_table": decode_status_table,
        "available_ood_ages_by_method": available_ood_ages_by_method,
        "volume_frame": pd.DataFrame(volume_rows),
        "observed_volume_table": pd.DataFrame(observed_volume_rows),
        "last_real_age_years": float(last_real_age_years),
    }


def subject_case_overview_table(case: Mapping[str, object]) -> pd.DataFrame:
    pair_row = case["pair_row"]
    status = case["decode_status_table"]
    direct_failures = int(
        ((status["transport_method"] == "direct") & (~status["decode_success"].astype(bool))).sum()
    )
    composed_failures = int(
        ((status["transport_method"] == "composed") & (~status["decode_success"].astype(bool))).sum()
    )
    return pd.DataFrame(
        [
            {
                "split": case["split"],
                "subject_id": case["subject_id"],
                "diagnosis": case["diagnosis"],
                "source_scan_id": case["source_row"]["scan_id"],
                "source_age_years": float(case["source_row"]["continuous_age_years"]),
                "target_scan_id": case["target_row"]["scan_id"],
                "target_age_years": float(case["target_row"]["continuous_age_years"]),
                "gap_years": float(pair_row["gap_years"]),
                "pair_type": pair_row["pair_type"],
                "sdf_l1_improvement": float(pair_row["sdf_l1_improvement"]),
                "model_beats_no_change": bool(pair_row["model_beats_no_change"]),
                "ood_ages_years": ", ".join(f"{age:.1f}" for age in case["ood_ages_years"]),
                "direct_decode_failures": direct_failures,
                "composed_decode_failures": composed_failures,
            }
        ]
    )


def subject_decode_status_table(case: Mapping[str, object]) -> pd.DataFrame:
    frame = case["decode_status_table"].copy()
    if frame.empty:
        return frame
    return frame.sort_values(["age_years", "transport_method"]).reset_index(drop=True)


def subject_forecast_volume_figure(case: Mapping[str, object]) -> go.Figure:
    frame = case["volume_frame"].copy()
    source_age_years = float(case["source_age_years"])
    last_real_age_years = float(case["last_real_age_years"])
    figure = go.Figure()

    style = {
        "real_observed": {"color": "#111111", "dash": "solid", "symbol": "diamond", "width": 3},
        "direct": {"color": "#1f77b4", "dash": "solid", "symbol": "circle", "width": 3},
        "composed": {"color": "#d62728", "dash": "dash", "symbol": "square", "width": 3},
        "no_change": {"color": "#7f7f7f", "dash": "dot", "symbol": "x", "width": 2},
    }

    for method in ("real_observed", "direct", "composed", "no_change"):
        group = frame.loc[frame["transport_method"] == method].copy()
        if group.empty:
            continue
        group = group.sort_values("age_years")
        spec = style[method]
        figure.add_trace(
            go.Scatter(
                x=group["age_years"],
                y=group["volume"],
                mode="lines+markers",
                name=method,
                line={"color": spec["color"], "dash": spec["dash"], "width": spec["width"]},
                marker={"symbol": spec["symbol"], "size": 8, "color": spec["color"]},
                customdata=np.stack([group["region"]], axis=-1),
                hovertemplate="age=%{x:.2f}<br>volume=%{y:.6f}<br>region=%{customdata[0]}<extra></extra>",
            )
        )

    if frame["age_years"].max() > last_real_age_years + 1e-6:
        figure.add_vrect(
            x0=last_real_age_years,
            x1=float(frame["age_years"].max()),
            fillcolor="#f2f2f2",
            opacity=0.25,
            line_width=0,
            annotation_text="OOD forecast",
            annotation_position="top left",
        )
    figure.add_vline(
        x=source_age_years,
        line_width=1,
        line_dash="dot",
        line_color="#555555",
    )
    figure.update_layout(
        title=(
            f"{case['split'].upper()} subject forecast volume trend | {case['diagnosis']} | "
            f"subject {case['subject_id']} | source age {source_age_years:.1f}"
        ),
        xaxis_title="age years",
        yaxis_title="mesh volume",
        width=1100,
        height=520,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.15},
    )
    return figure


def subject_observed_volume_error_table(case: Mapping[str, object]) -> pd.DataFrame:
    frame = case["observed_volume_table"].copy()
    if frame.empty:
        return frame
    ordered = [
        "age_years",
        "real_volume",
        "direct_volume",
        "composed_volume",
        "no_change_volume",
        "direct_abs_error",
        "composed_abs_error",
        "no_change_abs_error",
        "direct_signed_error",
        "composed_signed_error",
        "no_change_signed_error",
    ]
    return frame.loc[:, ordered].sort_values("age_years").reset_index(drop=True)


def _rigid_aligned_change(
    reference_mesh_like: str | Path | trimesh.Trimesh,
    moving_mesh_like: str | Path | trimesh.Trimesh,
    *,
    sample_count: int = 3000,
) -> Dict[str, object]:
    reference_mesh = load_mesh(reference_mesh_like)
    moving_mesh = load_mesh(moving_mesh_like)
    reference_points = reference_mesh.sample(sample_count)
    moving_points = moving_mesh.sample(sample_count)
    initial = np.eye(4, dtype=float)
    initial[:3, 3] = reference_points.mean(axis=0) - moving_points.mean(axis=0)
    try:
        matrix, _, cost = trimesh.registration.icp(
            moving_points,
            reference_points,
            initial=initial,
            threshold=1e-6,
            max_iterations=40,
            reflection=False,
            scale=False,
        )
    except Exception:
        matrix = initial
        cost = float("nan")
    aligned_vertices = trimesh.transform_points(np.asarray(moving_mesh.vertices), matrix)
    aligned_mesh = trimesh.Trimesh(
        vertices=aligned_vertices,
        faces=np.asarray(moving_mesh.faces),
        process=False,
    )
    tree = cKDTree(reference_points)
    distances, _ = tree.query(aligned_vertices, k=1)
    return {
        "aligned_mesh": aligned_mesh,
        "surface_shift": distances.astype(float),
        "mean_shift": float(np.mean(distances)),
        "p90_shift": float(np.quantile(distances, 0.90)),
        "p95_shift": float(np.quantile(distances, 0.95)),
        "max_shift": float(np.max(distances)),
        "icp_cost": float(cost),
    }


def _heat_mesh_trace(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
    *,
    name: str,
    cmin: float,
    cmax: float,
    show_scale: bool,
) -> go.Mesh3d:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=np.asarray(values, dtype=float),
        colorscale="Turbo",
        cmin=cmin,
        cmax=cmax,
        flatshading=False,
        name=name,
        showscale=show_scale,
        colorbar={"title": "surface shift"} if show_scale else None,
        hovertemplate="shift=%{intensity:.6f}<extra></extra>",
        showlegend=False,
    )


def message_figure(
    *,
    title: str,
    message: str,
) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(
        title=title,
        width=900,
        height=360,
        template="plotly_white",
    )
    return figure


def change_heatmap_figure(
    reference_mesh_like: str | Path | trimesh.Trimesh,
    comparisons: Sequence[Tuple[str, str | Path | trimesh.Trimesh]],
    *,
    title: str,
    sample_count: int = 3000,
) -> Tuple[go.Figure, pd.DataFrame]:
    summaries: List[Dict[str, object]] = []
    prepared: List[Tuple[str, Dict[str, object]]] = []
    for label, moving_mesh_like in comparisons:
        result = _rigid_aligned_change(
            reference_mesh_like,
            moving_mesh_like,
            sample_count=sample_count,
        )
        prepared.append((label, result))
        summaries.append(
            {
                "label": label,
                "mean_surface_shift": float(result["mean_shift"]),
                "p90_surface_shift": float(result["p90_shift"]),
                "p95_surface_shift": float(result["p95_shift"]),
                "max_surface_shift": float(result["max_shift"]),
                "icp_cost": float(result["icp_cost"]),
            }
        )

    summary_frame = pd.DataFrame(summaries)
    if summary_frame.empty:
        raise ValueError("No comparisons provided for change heatmap figure.")
    global_cmax = max(
        1e-8,
        max(float(np.quantile(result["surface_shift"], 0.99)) for _, result in prepared),
    )
    figure = make_subplots(
        rows=1,
        cols=len(prepared),
        specs=[[{"type": "scene"} for _ in prepared]],
        subplot_titles=[
            (
                f"{label}<br>"
                f"mean={float(result['mean_shift']):.4f}, "
                f"p95={float(result['p95_shift']):.4f}"
            )
            for label, result in prepared
        ],
    )
    for column, (label, result) in enumerate(prepared, start=1):
        figure.add_trace(
            _heat_mesh_trace(
                result["aligned_mesh"],
                result["surface_shift"],
                name=label,
                cmin=0.0,
                cmax=global_cmax,
                show_scale=(column == len(prepared)),
            ),
            row=1,
            col=column,
        )
        figure.update_scenes(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
            row=1,
            col=column,
        )
    figure.update_layout(
        title=title,
        width=max(1250, 420 * len(prepared)),
        height=560,
        template="plotly_white",
    )
    return figure, summary_frame


def observed_target_change_heatmap_figure(
    case: Mapping[str, object],
    *,
    sample_count: int = 3000,
) -> Tuple[go.Figure, pd.DataFrame]:
    target_age_years = float(case["target_age_years"])
    target_key = _age_key(target_age_years)
    target_mesh = case["real_meshes"][target_key]
    comparisons: List[Tuple[str, str | Path | trimesh.Trimesh]] = [
        (f"real target age {target_age_years:.1f}", target_mesh),
    ]
    direct_mesh = case["direct_meshes"].get(target_key)
    composed_mesh = case["composed_meshes"].get(target_key)
    if direct_mesh is not None:
        comparisons.append((f"direct target age {target_age_years:.1f}", direct_mesh))
    if composed_mesh is not None:
        comparisons.append((f"composed target age {target_age_years:.1f}", composed_mesh))
    return change_heatmap_figure(
        case["source_mesh"],
        comparisons,
        title=(
            f"{case['split'].upper()} target change map | {case['diagnosis']} | "
            f"subject {case['subject_id']} | rigid-aligned to source age {float(case['source_age_years']):.1f}"
        ),
        sample_count=sample_count,
    )


def ood_change_heatmap_figure(
    case: Mapping[str, object],
    *,
    method: str,
    sample_count: int = 3000,
) -> Tuple[go.Figure, pd.DataFrame]:
    if method not in {"direct", "composed"}:
        raise ValueError(f"Unsupported method {method!r}.")
    ood_ages = list(case["available_ood_ages_by_method"][method])
    if not ood_ages:
        message = (
            f"No decodable OOD meshes for {method}. "
            "See the decode-status table for the failed ages."
        )
        return (
            message_figure(
                title=(
                    f"{case['split'].upper()} OOD change maps | {case['diagnosis']} | "
                    f"subject {case['subject_id']} | {method}"
                ),
                message=message,
            ),
            pd.DataFrame(
                [
                    {
                        "label": "none",
                        "mean_surface_shift": float("nan"),
                        "p90_surface_shift": float("nan"),
                        "p95_surface_shift": float("nan"),
                        "max_surface_shift": float("nan"),
                        "icp_cost": float("nan"),
                        "message": message,
                    }
                ]
            ),
        )
    mesh_map = case["direct_meshes"] if method == "direct" else case["composed_meshes"]
    comparisons = [
        (f"{method} age {float(age_years):.1f}", mesh_map[_age_key(float(age_years))])
        for age_years in ood_ages
    ]
    return change_heatmap_figure(
        case["source_mesh"],
        comparisons,
        title=(
            f"{case['split'].upper()} OOD change maps | {case['diagnosis']} | subject {case['subject_id']} | "
            f"{method} forecast from source age {float(case['source_age_years']):.1f}"
        ),
        sample_count=sample_count,
    )


def select_anchor_scan(
    bundle: DirectFlowNotebookBundle,
    *,
    anchor_age_years: float,
    diagnosis: str,
    split_priority: Sequence[str] = ("train", "val", "test"),
) -> pd.Series:
    metadata = bundle.contract.metadata.copy()
    metadata = metadata.loc[metadata["diagnosis"] == diagnosis].copy()
    if metadata.empty:
        raise ValueError(f"No rows for diagnosis={diagnosis!r}.")
    split_rank = {split: rank for rank, split in enumerate(split_priority)}
    subject_counts = (
        metadata.groupby(["split", "subject_id"], sort=False)
        .size()
        .rename("num_subject_scans")
        .reset_index()
    )
    subject_spans = (
        metadata.groupby(["split", "subject_id"], sort=False)["continuous_age_years"]
        .agg(lambda series: float(np.max(series) - np.min(series)))
        .rename("subject_age_span_years")
        .reset_index()
    )
    metadata = metadata.merge(subject_counts, on=["split", "subject_id"], how="left")
    metadata = metadata.merge(subject_spans, on=["split", "subject_id"], how="left")
    metadata["anchor_distance_years"] = np.abs(
        metadata["continuous_age_years"].astype(float) - float(anchor_age_years)
    )
    metadata["split_rank"] = metadata["split"].map(split_rank).fillna(len(split_rank))
    metadata = metadata.sort_values(
        [
            "anchor_distance_years",
            "split_rank",
            "num_subject_scans",
            "subject_age_span_years",
            "continuous_age_years",
        ],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)
    return metadata.iloc[0]


def build_anchor_age_case(
    bundle: DirectFlowNotebookBundle,
    *,
    anchor_age_years: float,
    diagnosis: str,
    horizon_years: float = 20.0,
    evaluation_step_years: float = 2.0,
    composed_step_years: float = 0.5,
    mesh_resolution: int = 80,
    mesh_max_batch: int = 2 ** 18,
) -> Dict[str, object]:
    source_row = select_anchor_scan(
        bundle,
        anchor_age_years=anchor_age_years,
        diagnosis=diagnosis,
    )
    split = str(source_row["split"])
    source_scan_id = str(source_row["scan_id"])
    source_latent = _latent_tensor(bundle, split, source_scan_id)
    label_ad = int(source_row["label_ad"])
    actual_source_age_years = float(source_row["continuous_age_years"])
    condition = _condition_tensor(bundle, label_ad)

    if abs(actual_source_age_years - float(anchor_age_years)) <= 1e-6:
        anchor_latent = source_latent
    else:
        anchor_latent = transport_direct(
            bundle,
            source_latent,
            _time_tensor(bundle, normalize_age_years(bundle, actual_source_age_years)),
            _time_tensor(bundle, normalize_age_years(bundle, float(anchor_age_years))),
            condition,
        )
    if abs(actual_source_age_years - float(anchor_age_years)) <= 1e-6:
        anchor_mesh = load_mesh(source_row["mesh_path"])
        anchor_decode_error = None
        anchor_reference_mode = "real_anchor_scan"
    else:
        anchor_mesh, anchor_decode_error = try_decode_mesh(
            bundle,
            anchor_latent,
            resolution=mesh_resolution,
            max_batch=mesh_max_batch,
        )
        if anchor_mesh is not None:
            anchor_reference_mode = "decoded_anchor_latent"
        else:
            anchor_mesh = load_mesh(source_row["mesh_path"])
            anchor_reference_mode = "nearest_real_fallback"
    age_grid = np.arange(
        float(anchor_age_years),
        float(anchor_age_years) + float(horizon_years) + 1e-8,
        float(evaluation_step_years),
    )
    final_age_years = float(anchor_age_years) + float(horizon_years)
    if age_grid[-1] < final_age_years - 1e-8:
        age_grid = np.append(age_grid, final_age_years)
    age_grid = _unique_sorted_ages(age_grid.tolist())

    direct_meshes: Dict[str, Optional[trimesh.Trimesh]] = {}
    composed_meshes: Dict[str, Optional[trimesh.Trimesh]] = {}
    direct_errors: Dict[str, Optional[str]] = {}
    composed_errors: Dict[str, Optional[str]] = {}
    trend_rows: List[Dict[str, object]] = []
    decode_status_rows: List[Dict[str, object]] = []
    anchor_volume = mesh_volume_or_nan(anchor_mesh)

    for age_years in age_grid:
        key = _age_key(age_years)
        if abs(float(age_years) - float(anchor_age_years)) <= 1e-6:
            direct_mesh = anchor_mesh
            composed_mesh = anchor_mesh
            direct_error = anchor_decode_error
            composed_error = anchor_decode_error
        else:
            direct_latent = transport_direct(
                bundle,
                anchor_latent,
                _time_tensor(bundle, normalize_age_years(bundle, float(anchor_age_years))),
                _time_tensor(bundle, normalize_age_years(bundle, float(age_years))),
                condition,
            )
            direct_mesh, direct_error = try_decode_mesh(
                bundle,
                direct_latent,
                resolution=mesh_resolution,
                max_batch=mesh_max_batch,
            )
            composed_latent = transport_composed_fixed_step(
                bundle,
                anchor_latent,
                start_age_years=float(anchor_age_years),
                end_age_years=float(age_years),
                label_ad=label_ad,
                step_years=float(composed_step_years),
            )
            composed_mesh, composed_error = try_decode_mesh(
                bundle,
                composed_latent,
                resolution=mesh_resolution,
                max_batch=mesh_max_batch,
            )
        direct_meshes[key] = direct_mesh
        composed_meshes[key] = composed_mesh
        direct_errors[key] = direct_error
        composed_errors[key] = composed_error
        decode_status_rows.append(
            {
                "age_years": float(age_years),
                "transport_method": "direct",
                "decode_success": direct_mesh is not None,
                "error": direct_error,
            }
        )
        decode_status_rows.append(
            {
                "age_years": float(age_years),
                "transport_method": "composed",
                "decode_success": composed_mesh is not None,
                "error": composed_error,
            }
        )
        trend_rows.extend(
            [
                {
                    "transport_method": "anchor_reference",
                    "age_years": float(age_years),
                    "volume": anchor_volume,
                },
                {
                    "transport_method": "direct",
                    "age_years": float(age_years),
                    "volume": mesh_volume_or_nan(direct_mesh),
                },
                {
                    "transport_method": "composed",
                    "age_years": float(age_years),
                    "volume": mesh_volume_or_nan(composed_mesh),
                },
            ]
        )

    if abs(actual_source_age_years - float(anchor_age_years)) > 1e-6:
        trend_rows.append(
            {
                "transport_method": "nearest_real_scan",
                "age_years": actual_source_age_years,
                "volume": mesh_volume(source_row["mesh_path"]),
            }
        )

    return {
        "split": split,
        "diagnosis": diagnosis,
        "subject_id": str(source_row["subject_id"]),
        "source_row": source_row,
        "anchor_age_years": float(anchor_age_years),
        "actual_source_age_years": actual_source_age_years,
        "anchor_reference_mode": anchor_reference_mode,
        "anchor_decode_error": anchor_decode_error,
        "horizon_years": float(horizon_years),
        "final_age_years": final_age_years,
        "anchor_mesh": anchor_mesh,
        "direct_meshes": direct_meshes,
        "composed_meshes": composed_meshes,
        "direct_decode_errors": direct_errors,
        "composed_decode_errors": composed_errors,
        "decode_status_table": pd.DataFrame(decode_status_rows).sort_values(
            ["age_years", "transport_method"]
        ).reset_index(drop=True),
        "trend_frame": pd.DataFrame(trend_rows),
    }


def anchor_case_overview_table(case: Mapping[str, object]) -> pd.DataFrame:
    status = case["decode_status_table"]
    direct_failures = int(
        ((status["transport_method"] == "direct") & (~status["decode_success"].astype(bool))).sum()
    )
    composed_failures = int(
        ((status["transport_method"] == "composed") & (~status["decode_success"].astype(bool))).sum()
    )
    return pd.DataFrame(
        [
            {
                "split": case["split"],
                "diagnosis": case["diagnosis"],
                "subject_id": case["subject_id"],
                "nearest_real_scan_id": case["source_row"]["scan_id"],
                "nearest_real_age_years": float(case["actual_source_age_years"]),
                "anchor_age_years": float(case["anchor_age_years"]),
                "final_age_years": float(case["final_age_years"]),
                "horizon_years": float(case["horizon_years"]),
                "anchor_reference_mode": case["anchor_reference_mode"],
                "direct_decode_failures": direct_failures,
                "composed_decode_failures": composed_failures,
            }
        ]
    )


def anchor_decode_status_table(case: Mapping[str, object]) -> pd.DataFrame:
    frame = case["decode_status_table"].copy()
    if frame.empty:
        return frame
    return frame.sort_values(["age_years", "transport_method"]).reset_index(drop=True)


def anchor_volume_figure(case: Mapping[str, object]) -> go.Figure:
    frame = case["trend_frame"].copy().sort_values(["transport_method", "age_years"])
    figure = go.Figure()
    style = {
        "nearest_real_scan": {"color": "#111111", "dash": "solid", "symbol": "diamond", "width": 0},
        "anchor_reference": {"color": "#7f7f7f", "dash": "dot", "symbol": "x", "width": 2},
        "direct": {"color": "#1f77b4", "dash": "solid", "symbol": "circle", "width": 3},
        "composed": {"color": "#d62728", "dash": "dash", "symbol": "square", "width": 3},
    }
    for method in ("nearest_real_scan", "anchor_reference", "direct", "composed"):
        group = frame.loc[frame["transport_method"] == method].copy()
        if group.empty:
            continue
        spec = style[method]
        mode = "markers" if method == "nearest_real_scan" else "lines+markers"
        figure.add_trace(
            go.Scatter(
                x=group["age_years"],
                y=group["volume"],
                mode=mode,
                name=method,
                line={"color": spec["color"], "dash": spec["dash"], "width": spec["width"]},
                marker={"symbol": spec["symbol"], "size": 8, "color": spec["color"]},
                hovertemplate="age=%{x:.2f}<br>volume=%{y:.6f}<extra></extra>",
            )
        )
    figure.add_vline(
        x=float(case["anchor_age_years"]),
        line_width=1,
        line_dash="dot",
        line_color="#555555",
    )
    figure.update_layout(
        title=(
            f"{case['diagnosis']} anchor-age forecast | anchor {float(case['anchor_age_years']):.1f} -> "
            f"{float(case['final_age_years']):.1f} | nearest real age {float(case['actual_source_age_years']):.1f}"
        ),
        xaxis_title="age years",
        yaxis_title="mesh volume",
        width=1050,
        height=520,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.15},
    )
    return figure


def anchor_change_heatmap_figure(
    case: Mapping[str, object],
    *,
    sample_count: int = 3000,
) -> Tuple[go.Figure, pd.DataFrame]:
    final_key = _age_key(float(case["final_age_years"]))
    comparisons: List[Tuple[str, str | Path | trimesh.Trimesh]] = []
    direct_mesh = case["direct_meshes"].get(final_key)
    composed_mesh = case["composed_meshes"].get(final_key)
    if direct_mesh is not None:
        comparisons.append((f"direct age {float(case['final_age_years']):.1f}", direct_mesh))
    if composed_mesh is not None:
        comparisons.append((f"composed age {float(case['final_age_years']):.1f}", composed_mesh))
    if not comparisons:
        message = (
            "Neither direct nor composed forecast produced a decodable mesh at the final "
            "20-year horizon. See the decode-status table."
        )
        return (
            message_figure(
                title=(
                    f"{case['diagnosis']} 20-year change map | anchor {float(case['anchor_age_years']):.1f}"
                ),
                message=message,
            ),
            pd.DataFrame(
                [
                    {
                        "label": "none",
                        "mean_surface_shift": float("nan"),
                        "p90_surface_shift": float("nan"),
                        "p95_surface_shift": float("nan"),
                        "max_surface_shift": float("nan"),
                        "icp_cost": float("nan"),
                        "message": message,
                    }
                ]
            ),
        )
    return change_heatmap_figure(
        case["anchor_mesh"],
        comparisons,
        title=(
            f"{case['diagnosis']} 20-year change map | anchor {float(case['anchor_age_years']):.1f} | "
            f"rigid-aligned to anchor shape"
        ),
        sample_count=sample_count,
    )


def _split_subject_ids(
    bundle: DirectFlowNotebookBundle,
    split: str,
    *,
    limit: Optional[int] = None,
) -> List[str]:
    metadata = bundle.contract.metadata.copy()
    metadata = metadata.loc[metadata["split"].astype(str) == str(split)]
    subject_ids = sorted(metadata["subject_id"].astype(str).unique().tolist())
    if limit is not None:
        subject_ids = subject_ids[: int(limit)]
    return subject_ids


def build_observed_age_volume_trend_dataset(
    bundle: DirectFlowNotebookBundle,
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    subject_limits: Optional[Mapping[str, Optional[int]]] = None,
    mesh_resolution: int = 80,
    mesh_max_batch: int = 2 ** 18,
) -> Dict[str, pd.DataFrame]:
    subject_limits = dict(subject_limits or {})
    trend_rows: List[Dict[str, object]] = []
    subject_rows_out: List[Dict[str, object]] = []
    decode_rows: List[Dict[str, object]] = []

    for split in splits:
        limit = subject_limits.get(str(split))
        for subject_id in _split_subject_ids(bundle, str(split), limit=limit):
            rows = subject_rows(bundle, subject_id, split=str(split))
            if rows.empty:
                continue
            base_row = rows.iloc[0]
            label_ad = int(base_row["label_ad"])
            diagnosis = str(base_row["diagnosis"])
            base_scan_id = str(base_row["scan_id"])
            base_age_years = float(base_row["continuous_age_years"])
            base_age_norm = float(base_row["continuous_age_norm"])
            base_latent = _latent_tensor(bundle, str(split), base_scan_id)
            base_time = _time_tensor(bundle, base_age_norm)
            condition = _condition_tensor(bundle, label_ad)

            base_real_volume = mesh_volume(base_row["mesh_path"])
            base_recon_mesh, base_recon_error = try_decode_mesh(
                bundle,
                base_latent,
                resolution=mesh_resolution,
                max_batch=mesh_max_batch,
            )
            base_recon_volume = mesh_volume_or_nan(base_recon_mesh)

            per_subject_predicted: List[Tuple[float, float, bool]] = []
            per_subject_real: List[Tuple[float, float]] = []
            decode_success_count = 0

            for _, row in rows.iterrows():
                scan_id = str(row["scan_id"])
                age_years = float(row["continuous_age_years"])
                age_norm = float(row["continuous_age_norm"])
                years_from_baseline = age_years - base_age_years
                real_volume = mesh_volume(row["mesh_path"])
                per_subject_real.append((age_years, real_volume))

                if abs(age_norm - base_age_norm) <= 1e-8:
                    predicted_mesh = base_recon_mesh
                    predicted_error = base_recon_error
                else:
                    predicted_latent = transport_direct(
                        bundle,
                        base_latent,
                        base_time,
                        _time_tensor(bundle, age_norm),
                        condition,
                    )
                    predicted_mesh, predicted_error = try_decode_mesh(
                        bundle,
                        predicted_latent,
                        resolution=mesh_resolution,
                        max_batch=mesh_max_batch,
                    )
                predicted_volume = mesh_volume_or_nan(predicted_mesh)
                predicted_success = predicted_mesh is not None
                decode_success_count += int(predicted_success)
                per_subject_predicted.append(
                    (age_years, predicted_volume, predicted_success)
                )

                real_relative = (
                    real_volume / base_real_volume
                    if np.isfinite(base_real_volume) and base_real_volume > 0.0
                    else float("nan")
                )
                predicted_relative = (
                    predicted_volume / base_recon_volume
                    if np.isfinite(predicted_volume)
                    and np.isfinite(base_recon_volume)
                    and base_recon_volume > 0.0
                    else float("nan")
                )

                common = {
                    "split": str(split),
                    "subject_id": subject_id,
                    "diagnosis": diagnosis,
                    "label_ad": label_ad,
                    "scan_id": scan_id,
                    "base_scan_id": base_scan_id,
                    "visit_order": int(row["visit_order"]),
                    "age_years": age_years,
                    "years_from_baseline": years_from_baseline,
                    "base_age_years": base_age_years,
                }
                trend_rows.append(
                    {
                        **common,
                        "transport_method": "real_observed",
                        "volume": real_volume,
                        "relative_volume": real_relative,
                        "decode_success": True,
                    }
                )
                trend_rows.append(
                    {
                        **common,
                        "transport_method": "direct_from_base",
                        "volume": predicted_volume,
                        "relative_volume": predicted_relative,
                        "decode_success": predicted_success,
                    }
                )
                trend_rows.append(
                    {
                        **common,
                        "transport_method": "base_reconstruction_no_change",
                        "volume": base_recon_volume,
                        "relative_volume": 1.0
                        if np.isfinite(base_recon_volume)
                        else float("nan"),
                        "decode_success": base_recon_mesh is not None,
                    }
                )
                decode_rows.append(
                    {
                        "split": str(split),
                        "subject_id": subject_id,
                        "diagnosis": diagnosis,
                        "scan_id": scan_id,
                        "age_years": age_years,
                        "years_from_baseline": years_from_baseline,
                        "decode_success": predicted_success,
                        "error": predicted_error,
                    }
                )

            final_real_age, final_real_volume = per_subject_real[-1]
            _, final_predicted_volume, final_predicted_success = per_subject_predicted[-1]
            real_delta_pct = (
                100.0 * (final_real_volume - base_real_volume) / base_real_volume
                if base_real_volume > 0.0
                else float("nan")
            )
            predicted_delta_pct = (
                100.0
                * (final_predicted_volume - base_recon_volume)
                / base_recon_volume
                if np.isfinite(final_predicted_volume)
                and np.isfinite(base_recon_volume)
                and base_recon_volume > 0.0
                else float("nan")
            )
            sign_match = (
                np.sign(real_delta_pct) == np.sign(predicted_delta_pct)
                if np.isfinite(real_delta_pct)
                and np.isfinite(predicted_delta_pct)
                and abs(real_delta_pct) > 1e-8
                else False
            )
            subject_rows_out.append(
                {
                    "split": str(split),
                    "subject_id": subject_id,
                    "diagnosis": diagnosis,
                    "num_scans": int(len(rows)),
                    "base_age_years": base_age_years,
                    "final_age_years": final_real_age,
                    "base_real_volume": base_real_volume,
                    "base_recon_volume": base_recon_volume,
                    "base_recon_abs_error": abs(base_recon_volume - base_real_volume)
                    if np.isfinite(base_recon_volume)
                    else float("nan"),
                    "final_real_volume": final_real_volume,
                    "final_predicted_volume": final_predicted_volume,
                    "final_predicted_decode_success": final_predicted_success,
                    "final_abs_volume_error": abs(
                        final_predicted_volume - final_real_volume
                    )
                    if np.isfinite(final_predicted_volume)
                    else float("nan"),
                    "real_delta_pct": real_delta_pct,
                    "predicted_delta_pct": predicted_delta_pct,
                    "final_delta_sign_match": bool(sign_match),
                    "decode_success_fraction": decode_success_count / max(1, len(rows)),
                }
            )

    return {
        "trend_frame": pd.DataFrame(trend_rows),
        "subject_summary": pd.DataFrame(subject_rows_out),
        "decode_status": pd.DataFrame(decode_rows),
    }


def observed_age_volume_summary_table(dataset: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    subject_summary = dataset["subject_summary"].copy()
    rows: List[Dict[str, object]] = []
    for split, group in subject_summary.groupby("split", sort=False):
        rows.append(
            {
                "split": split,
                "subjects": int(len(group)),
                "mean_real_final_delta_pct": float(group["real_delta_pct"].mean()),
                "mean_predicted_final_delta_pct": float(
                    group["predicted_delta_pct"].mean()
                ),
                "final_delta_sign_match_fraction": float(
                    group["final_delta_sign_match"].mean()
                ),
                "mean_base_recon_abs_error": float(
                    group["base_recon_abs_error"].mean()
                ),
                "mean_final_abs_volume_error": float(
                    group["final_abs_volume_error"].mean()
                ),
                "mean_decode_success_fraction": float(
                    group["decode_success_fraction"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _binned_mean_frame(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    bin_width: float,
) -> pd.DataFrame:
    usable = frame.loc[np.isfinite(frame[y_col])].copy()
    usable["x_bin"] = np.round(usable[x_col].astype(float) / bin_width) * bin_width
    return (
        usable.groupby(["split", "transport_method", "x_bin"], sort=False)
        .agg(
            mean_value=(y_col, "mean"),
            std_value=(y_col, "std"),
            n=(y_col, "size"),
        )
        .reset_index()
    )


def observed_age_volume_trend_figure(
    trend_frame: pd.DataFrame,
    *,
    age_bin_width: float = 1.0,
) -> go.Figure:
    binned = _binned_mean_frame(
        trend_frame,
        x_col="age_years",
        y_col="volume",
        bin_width=age_bin_width,
    )
    return _split_trend_figure(
        binned,
        x_title="age years",
        y_title="mean mesh volume",
        title="Observed-age volume trend: real vs baseline reconstruction forecast",
    )


def elapsed_relative_volume_trend_figure(
    trend_frame: pd.DataFrame,
    *,
    elapsed_bin_width: float = 0.5,
) -> go.Figure:
    binned = _binned_mean_frame(
        trend_frame,
        x_col="years_from_baseline",
        y_col="relative_volume",
        bin_width=elapsed_bin_width,
    )
    return _split_trend_figure(
        binned,
        x_title="years from baseline",
        y_title="mean relative volume",
        title="Within-subject relative volume trend: real vs baseline reconstruction forecast",
    )


def _split_trend_figure(
    binned: pd.DataFrame,
    *,
    x_title: str,
    y_title: str,
    title: str,
) -> go.Figure:
    split_order = [split for split in ("train", "val", "test") if split in set(binned["split"])]
    method_order = [
        "real_observed",
        "direct_from_base",
        "base_reconstruction_no_change",
    ]
    style = {
        "real_observed": {"color": "#111111", "dash": "solid", "symbol": "diamond"},
        "direct_from_base": {"color": "#1f77b4", "dash": "solid", "symbol": "circle"},
        "base_reconstruction_no_change": {
            "color": "#7f7f7f",
            "dash": "dot",
            "symbol": "x",
        },
    }
    figure = make_subplots(
        rows=1,
        cols=len(split_order),
        shared_yaxes=False,
        subplot_titles=[split.upper() for split in split_order],
    )
    for col, split in enumerate(split_order, start=1):
        split_frame = binned.loc[binned["split"] == split]
        for method in method_order:
            group = split_frame.loc[
                split_frame["transport_method"] == method
            ].sort_values("x_bin")
            if group.empty:
                continue
            spec = style[method]
            figure.add_trace(
                go.Scatter(
                    x=group["x_bin"],
                    y=group["mean_value"],
                    mode="lines+markers",
                    name=method,
                    legendgroup=method,
                    showlegend=(col == 1),
                    line={"color": spec["color"], "dash": spec["dash"], "width": 3},
                    marker={"color": spec["color"], "symbol": spec["symbol"], "size": 8},
                    customdata=np.stack([group["n"]], axis=-1),
                    hovertemplate=(
                        f"{x_title}=%{{x:.2f}}<br>"
                        f"{y_title}=%{{y:.6f}}<br>"
                        "n=%{customdata[0]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        figure.update_xaxes(title_text=x_title, row=1, col=col)
        figure.update_yaxes(title_text=y_title if col == 1 else None, row=1, col=col)
    figure.update_layout(
        title=title,
        width=max(1100, 430 * max(1, len(split_order))),
        height=520,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.18},
    )
    return figure
