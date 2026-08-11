#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
OLD_HELPER_DIR = REPO_ROOT / "examples" / "ADNI_1_L_No_MCI"
for import_path in (SCRIPT_DIR, REPO_ROOT, OLD_HELPER_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import direct_flow_rich_notebook_helpers as flow_helpers  # noqa: E402
import evaluate_future_mesh_forecasts as fair_eval  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
    "analysis/future_mesh_rich_visualization"
)
DEFAULT_FUTURE_MESH_DIR = (
    "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
    "analysis/future_mesh_forecast_comparison"
)

RICH_MODELS = {
    "old_adni": [
        "old_brainode_pca150",
        "old_pca32_siren",
        "old_smallnet_deepsdf",
    ],
    "qc_large": [
        "qc_brainode_pca150",
        "qc_siren_latent_ode",
        "qc_siren_drop_bad_min2",
        "qc_siren_local_decomp_volume",
    ],
}

MODEL_LABELS = {
    "old_brainode_pca150": "BrainODE PCA150",
    "qc_brainode_pca150": "BrainODE PCA150",
    "old_pca32_siren": "PCA32 SIREN flow",
    "old_smallnet_deepsdf": "Small-net DeepSDF flow",
    "qc_siren_latent_ode": "QC SIREN latent ODE",
    "qc_siren_drop_bad_min2": "QC drop-bad SIREN flow",
    "qc_siren_local_decomp_volume": "QC local-decomposed volume SIREN",
}

DATASET_LABELS = {
    "old_adni": "Old ADNI no-MCI",
    "qc_large": "QC-large strict left",
}

PLOT_COLORS = {
    "source_gt": "#7f7f7f",
    "target_gt": "#111111",
    "brainode_endpoint": "#0f766e",
    "direct": "#1f77b4",
    "composed": "#d62728",
    "model_no_change": "#9467bd",
}


@dataclass
class MeshCase:
    dataset: str
    diagnosis: str
    split: str
    pair_type: str
    subject_id: str
    source_scan_id: str
    target_scan_id: str
    source_age_years: float
    target_age_years: float
    source_age_norm: float
    target_age_norm: float
    source_mesh_path: str
    target_mesh_path: str
    metrics: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a rich visual report for selected future-mesh cases from the "
            "BrainODE/SIREN/DeepSDF fair mesh evaluation."
        )
    )
    parser.add_argument("--future-mesh-dir", default=DEFAULT_FUTURE_MESH_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--pair-type", default="nonadjacent", choices=("nonadjacent", "adjacent", "all"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--mesh-resolution", type=int, default=80)
    parser.add_argument("--mesh-max-batch", type=int, default=2**18)
    parser.add_argument("--composed-step-years", type=float, default=0.5)
    parser.add_argument("--change-sample-count", type=int, default=3000)
    parser.add_argument(
        "--case-strategy",
        default="best_mean_improvement",
        choices=("best_mean_improvement", "median_brainode_chamfer"),
    )
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def load_future_metrics(future_mesh_dir: Path) -> pd.DataFrame:
    path = future_mesh_dir / "future_mesh_per_pair.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing future mesh metrics: {path}")
    frame = pd.read_csv(path, low_memory=False)
    numeric_cols = [
        "chamfer_l2_squared",
        "chamfer_l2_squared_improvement_vs_model_no_change",
        "assd",
        "hd95",
        "volume_relative_error",
        "surface_area_relative_error",
        "predicted_volume",
        "target_volume",
        "predicted_surface_area",
        "target_surface_area",
        "source_age_years",
        "target_age_years",
        "source_age_norm",
        "target_age_norm",
    ]
    for col in numeric_cols:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def required_transport(model: str) -> set[str]:
    if model in fair_eval.BRAINODE_MODELS:
        return {"brainode_endpoint", "model_no_change"}
    return {"direct", "composed", "model_no_change"}


def select_case_for_dataset_diagnosis(
    frame: pd.DataFrame,
    *,
    dataset: str,
    diagnosis: str,
    split: str,
    pair_type: str,
    strategy: str,
) -> MeshCase:
    models = RICH_MODELS[dataset]
    subset = frame.loc[
        (frame["dataset"].astype(str) == dataset)
        & (frame["split"].astype(str) == split)
        & (frame["diagnosis"].astype(str) == diagnosis)
        & (frame["model"].astype(str).isin(models))
    ].copy()
    if pair_type != "all":
        subset = subset.loc[subset["pair_type"].astype(str) == pair_type].copy()
    if subset.empty and pair_type != "all":
        subset = frame.loc[
            (frame["dataset"].astype(str) == dataset)
            & (frame["split"].astype(str) == split)
            & (frame["diagnosis"].astype(str) == diagnosis)
            & (frame["model"].astype(str).isin(models))
        ].copy()
    if subset.empty:
        raise ValueError(f"No rows for dataset={dataset}, diagnosis={diagnosis}, split={split}")

    pair_cols = ["source_scan_id", "target_scan_id"]
    candidates: list[dict[str, Any]] = []
    for (source_scan_id, target_scan_id), group in subset.groupby(pair_cols, sort=False):
        ok = True
        for model in models:
            model_transports = set(
                group.loc[group["model"].astype(str) == model, "transport_method"].astype(str)
            )
            if not required_transport(model).issubset(model_transports):
                ok = False
                break
        if not ok:
            continue
        prediction_rows = group.loc[group["transport_method"].astype(str) != "model_no_change"]
        brainode_rows = prediction_rows.loc[
            prediction_rows["model"].astype(str).str.contains("brainode", case=False, regex=False)
        ]
        mean_improvement = float(
            prediction_rows["chamfer_l2_squared_improvement_vs_model_no_change"].mean()
        )
        brainode_chamfer = float(brainode_rows["chamfer_l2_squared"].mean())
        candidates.append(
            {
                "source_scan_id": source_scan_id,
                "target_scan_id": target_scan_id,
                "mean_improvement": mean_improvement,
                "brainode_chamfer": brainode_chamfer,
                "rows": len(group),
            }
        )
    if not candidates:
        raise ValueError(
            f"No complete case with all rich models for dataset={dataset}, diagnosis={diagnosis}"
        )
    candidates_frame = pd.DataFrame(candidates)
    if strategy == "median_brainode_chamfer":
        median = float(candidates_frame["brainode_chamfer"].median())
        candidates_frame["score"] = -np.abs(candidates_frame["brainode_chamfer"] - median)
    else:
        candidates_frame["score"] = candidates_frame["mean_improvement"]
    selected = candidates_frame.sort_values(["score", "mean_improvement"], ascending=False).iloc[0]
    case_rows = subset.loc[
        (subset["source_scan_id"].astype(str) == str(selected["source_scan_id"]))
        & (subset["target_scan_id"].astype(str) == str(selected["target_scan_id"]))
    ].copy()
    first = case_rows.iloc[0]
    return MeshCase(
        dataset=dataset,
        diagnosis=diagnosis,
        split=str(first["split"]),
        pair_type=str(first["pair_type"]),
        subject_id=str(first["subject_id"]),
        source_scan_id=str(first["source_scan_id"]),
        target_scan_id=str(first["target_scan_id"]),
        source_age_years=float(first["source_age_years"]),
        target_age_years=float(first["target_age_years"]),
        source_age_norm=float(first["source_age_norm"]),
        target_age_norm=float(first["target_age_norm"]),
        source_mesh_path=str(first["source_mesh_path"]),
        target_mesh_path=str(first["target_mesh_path"]),
        metrics=case_rows,
    )


class BrainODEPredictor:
    def __init__(self, model_name: str, device: str, checkpoint: str) -> None:
        self.model_name = model_name
        self.spec = fair_eval.BRAINODE_MODELS[model_name]
        self.task_dir = repo_path(self.spec["task_dir"])
        modules = fair_eval.import_brainode_modules(self.task_dir)
        self.common = modules["common"]
        self.train = modules["train"]
        self.brainode_model = modules["brainode_model"]
        self.config = self.common.load_config(self.task_dir / "configs" / "core_brainode.json")
        self.training_config = dict(self.config["training"])
        self.full_config = self.train.full_brainode_config(self.config)
        self.device = fair_eval.resolve_device(device or self.training_config.get("device"))
        run_name = str(self.training_config["run_name"])
        self.checkpoint_path = fair_eval.resolve_brainode_checkpoint(
            task_dir=self.task_dir,
            config=self.config,
            common=self.common,
            checkpoint=checkpoint,
            run_name=run_name,
        )
        train_archive = fair_eval.load_npz(self.task_dir / "dataset" / "train_subject_sequences.npz")
        self.latent_dim = int(train_archive["visit_pca_150"].shape[1])
        self.model = self.train.build_model(
            latent_dim=self.latent_dim,
            model_config=dict(self.config["model"]),
            full_config=self.full_config,
        ).to(self.device)
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        pca_model_dir = self.common.resolve_repo_path(self.config["task2"]["pca_model_dir"])
        self.mean_flat = np.load(pca_model_dir / "mean.npy").astype(np.float32)
        self.components = np.load(pca_model_dir / "components_256.npy").astype(np.float32)[
            : self.latent_dim
        ]
        self.faces = np.load(pca_model_dir / "faces.npy").astype(np.int64)
        self._archive_cache: dict[str, dict[str, np.ndarray]] = {}

    def archive(self, split: str) -> dict[str, np.ndarray]:
        if split not in self._archive_cache:
            self._archive_cache[split] = fair_eval.load_npz(
                self.task_dir / "dataset" / f"{split}_subject_sequences.npz"
            )
        return self._archive_cache[split]

    @torch.no_grad()
    def predict(self, case: MeshCase) -> dict[str, trimesh.Trimesh]:
        archive = self.archive(case.split)
        scan_ids = [str(value) for value in archive["visit_scan_ids"].tolist()]
        source_index = scan_ids.index(case.source_scan_id)
        target_index = scan_ids.index(case.target_scan_id)
        if source_index >= target_index:
            raise ValueError(f"Invalid BrainODE pair for {case.source_scan_id} -> {case.target_scan_id}")
        time_slice = slice(source_index, target_index + 1)
        times_np = archive["visit_continuous_age_norm"][time_slice].astype(np.float32).copy()
        targets_np = archive["visit_pca_150"][time_slice].astype(np.float32).copy()
        condition_value = float(archive["visit_cognition"][source_index])
        times = torch.from_numpy(times_np).float().view(1, -1).to(self.device)
        targets = torch.from_numpy(targets_np).float().view(1, len(times_np), -1).to(self.device)
        condition = torch.tensor([condition_value], dtype=torch.float32, device=self.device)
        initial_state = targets[:, 0, :]
        if bool(self.full_config.get("use_autoregressive_rollout", False)):
            prediction, _ = self.brainode_model.integrate_autoregressive_rk4(
                func=self.model,
                initial_state=initial_state,
                times=times,
                initial_condition=condition,
                substeps=int(self.training_config["integration_substeps"]),
            )
        else:
            prediction = self.brainode_model.integrate_sequence_rk4(
                func=self.model,
                initial_state=initial_state,
                times=times,
                condition=condition,
                substeps=int(self.training_config["integration_substeps"]),
            )
        endpoint = prediction[0, -1, :].detach().cpu().numpy().astype(np.float32)
        source_code = targets_np[0].astype(np.float32)
        return {
            "brainode_endpoint": fair_eval.pca_mesh(
                endpoint,
                self.mean_flat,
                self.components,
                self.faces,
            ),
            "model_no_change": fair_eval.pca_mesh(
                source_code,
                self.mean_flat,
                self.components,
                self.faces,
            ),
        }


class FlowPredictor:
    def __init__(
        self,
        model_name: str,
        device: str,
        checkpoint: str,
        mesh_resolution: int,
        mesh_max_batch: int,
        composed_step_years: float,
    ) -> None:
        self.model_name = model_name
        self.spec = fair_eval.FLOW_MODELS[model_name]
        self.bundle = flow_helpers.load_bundle(
            repo_path(self.spec["experiment_dir"]),
            checkpoint=checkpoint,
            device=device,
        )
        self.mesh_resolution = int(mesh_resolution)
        self.mesh_max_batch = int(mesh_max_batch)
        self.composed_step_years = float(composed_step_years)

    @torch.no_grad()
    def predict(self, case: MeshCase) -> dict[str, trimesh.Trimesh]:
        source_latent = flow_helpers._latent_tensor(
            self.bundle,
            case.split,
            case.source_scan_id,
        )
        source_time = flow_helpers._time_tensor(self.bundle, case.source_age_norm)
        target_time = flow_helpers._time_tensor(self.bundle, case.target_age_norm)
        label_ad = int(
            self.bundle.contract.metadata.set_index("scan_id").loc[case.source_scan_id, "label_ad"]
        )
        condition = flow_helpers._condition_tensor(self.bundle, label_ad)
        direct_latent = flow_helpers.transport_direct(
            self.bundle,
            source_latent,
            source_time,
            target_time,
            condition,
        )
        composed_latent = flow_helpers.transport_composed_fixed_step(
            self.bundle,
            source_latent,
            start_age_years=case.source_age_years,
            end_age_years=case.target_age_years,
            label_ad=label_ad,
            step_years=self.composed_step_years,
        )
        output: dict[str, trimesh.Trimesh] = {}
        for name, latent in (
            ("direct", direct_latent),
            ("composed", composed_latent),
            ("model_no_change", source_latent),
        ):
            mesh, error = flow_helpers.try_decode_mesh(
                self.bundle,
                latent,
                resolution=self.mesh_resolution,
                max_batch=self.mesh_max_batch,
            )
            if mesh is None:
                raise RuntimeError(f"{self.model_name} {name} mesh decode failed: {error}")
            output[name] = mesh
        return output


def figure_html(fig: go.Figure, *, include_plotlyjs: bool) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True if include_plotlyjs else False,
        config={"displayModeBar": True, "responsive": True},
    )


def metric_table(case: MeshCase) -> str:
    cols = [
        "model",
        "transport_method",
        "chamfer_l2_squared",
        "chamfer_l2_squared_improvement_vs_model_no_change",
        "assd",
        "hd95",
        "volume_relative_error",
        "surface_area_relative_error",
    ]
    view = case.metrics.loc[:, cols].copy()
    view = view.loc[view["transport_method"].astype(str) != "model_no_change"]
    view["model"] = view["model"].map(lambda value: MODEL_LABELS.get(str(value), str(value)))
    for col in cols[2:]:
        view[col] = pd.to_numeric(view[col], errors="coerce").map(
            lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
        )
    return view.to_html(index=False, escape=True, classes="data-table")


def mesh_collection_for_case(
    *,
    case: MeshCase,
    predictors: dict[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, trimesh.Trimesh]]:
    source_mesh = fair_eval.load_mesh(case.source_mesh_path)
    target_mesh = fair_eval.load_mesh(case.target_mesh_path)
    collection: dict[str, dict[str, trimesh.Trimesh]] = {
        "ground_truth": {
            "source_gt": source_mesh,
            "target_gt": target_mesh,
        }
    }
    for model in RICH_MODELS[case.dataset]:
        meshes = predictors[model].predict(case)
        collection[model] = meshes
        mesh_dir = output_dir / "meshes" / case.dataset / case.diagnosis / safe_name(model)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        for method, mesh in meshes.items():
            path = mesh_dir / (
                f"{safe_name(case.subject_id)}__{safe_name(case.source_scan_id[:28])}"
                f"__to__{safe_name(case.target_scan_id[:28])}__{method}.ply"
            )
            mesh.export(path)
    return collection


def comparison_volume_area_figure(case: MeshCase, collection: dict[str, dict[str, trimesh.Trimesh]]) -> go.Figure:
    labels: list[str] = ["source GT", "target GT"]
    volumes: list[float] = [
        float(abs(collection["ground_truth"]["source_gt"].volume)),
        float(abs(collection["ground_truth"]["target_gt"].volume)),
    ]
    areas: list[float] = [
        float(collection["ground_truth"]["source_gt"].area),
        float(collection["ground_truth"]["target_gt"].area),
    ]
    colors: list[str] = [PLOT_COLORS["source_gt"], PLOT_COLORS["target_gt"]]

    for model in RICH_MODELS[case.dataset]:
        for method, mesh in collection[model].items():
            if method == "model_no_change":
                continue
            labels.append(f"{MODEL_LABELS[model]} {method}")
            volumes.append(float(abs(mesh.volume)))
            areas.append(float(mesh.area))
            colors.append(PLOT_COLORS.get(method, "#555555"))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Mesh volume", "Surface area"),
    )
    fig.add_trace(go.Bar(x=labels, y=volumes, marker_color=colors, name="volume"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=areas, marker_color=colors, name="area"), row=1, col=2)
    fig.update_xaxes(tickangle=35)
    fig.update_layout(
        title=(
            f"{DATASET_LABELS[case.dataset]} {case.diagnosis} case | "
            f"subject {case.subject_id} | target age {case.target_age_years:.2f}"
        ),
        template="plotly_white",
        height=520,
        width=1250,
        showlegend=False,
        margin=dict(l=50, r=30, t=80, b=170),
    )
    return fig


def model_mesh_panel(case: MeshCase, model: str, collection: dict[str, dict[str, trimesh.Trimesh]]) -> go.Figure:
    entries = [
        (
            f"source GT<br>{case.source_age_years:.2f}y",
            [("source GT", collection["ground_truth"]["source_gt"], PLOT_COLORS["source_gt"], 0.85)],
        ),
        (
            f"target GT<br>{case.target_age_years:.2f}y",
            [("target GT", collection["ground_truth"]["target_gt"], PLOT_COLORS["target_gt"], 0.85)],
        ),
    ]
    meshes = collection[model]
    for method, mesh in meshes.items():
        if method == "model_no_change":
            title = "model no-change"
        elif method == "brainode_endpoint":
            title = "BrainODE endpoint"
        else:
            title = method
        entries.append(
            (
                title,
                [(title, mesh, PLOT_COLORS.get(method, "#1f77b4"), 0.82)],
            )
        )
    fig = flow_helpers.mesh_panel_figure(
        entries,
        title=(
            f"{DATASET_LABELS[case.dataset]} | {case.diagnosis} | "
            f"{MODEL_LABELS[model]} | subject {case.subject_id}"
        ),
    )
    fig.update_layout(margin=dict(l=20, r=20, t=80, b=20))
    return fig


def model_change_heatmap(
    case: MeshCase,
    model: str,
    collection: dict[str, dict[str, trimesh.Trimesh]],
    sample_count: int,
) -> tuple[go.Figure, pd.DataFrame]:
    comparisons: list[tuple[str, trimesh.Trimesh]] = [
        (f"target GT {case.target_age_years:.2f}y", collection["ground_truth"]["target_gt"])
    ]
    for method, mesh in collection[model].items():
        if method == "model_no_change":
            continue
        label = "BrainODE endpoint" if method == "brainode_endpoint" else method
        comparisons.append((label, mesh))
    comparisons.append(("model no-change", collection[model]["model_no_change"]))
    return flow_helpers.change_heatmap_figure(
        collection["ground_truth"]["source_gt"],
        comparisons,
        title=(
            f"Surface change from source | {DATASET_LABELS[case.dataset]} | "
            f"{case.diagnosis} | {MODEL_LABELS[model]}"
        ),
        sample_count=sample_count,
    )


def write_case_page(
    *,
    case: MeshCase,
    collection: dict[str, dict[str, trimesh.Trimesh]],
    output_dir: Path,
    change_sample_count: int,
) -> tuple[Path, list[dict[str, Any]]]:
    page_name = f"{case.dataset}_{case.diagnosis.lower()}_{safe_name(case.subject_id)}.html"
    page_path = output_dir / page_name
    include_js = True
    parts: list[str] = []
    parts.append(
        "<section class='summary'>"
        f"<h2>{html.escape(DATASET_LABELS[case.dataset])} | {html.escape(case.diagnosis)} | "
        f"subject {html.escape(case.subject_id)}</h2>"
        f"<p>Split <code>{html.escape(case.split)}</code>, pair type <code>{html.escape(case.pair_type)}</code>, "
        f"source age {case.source_age_years:.2f}, target age {case.target_age_years:.2f}.</p>"
        f"<p><code>{html.escape(case.source_scan_id)}</code><br><code>{html.escape(case.target_scan_id)}</code></p>"
        "</section>"
    )
    parts.append("<section><h3>Future Mesh Metrics</h3>")
    parts.append(metric_table(case))
    parts.append("</section>")

    fig = comparison_volume_area_figure(case, collection)
    parts.append("<section><h3>Volume and Surface Area</h3>")
    parts.append(figure_html(fig, include_plotlyjs=include_js))
    include_js = False
    parts.append("</section>")

    heatmap_rows: list[dict[str, Any]] = []
    for model in RICH_MODELS[case.dataset]:
        parts.append(f"<section><h3>{html.escape(MODEL_LABELS[model])}</h3>")
        mesh_fig = model_mesh_panel(case, model, collection)
        parts.append("<div class='plot-block'>")
        parts.append(figure_html(mesh_fig, include_plotlyjs=include_js))
        include_js = False
        parts.append("</div>")
        heat_fig, heat_summary = model_change_heatmap(
            case,
            model,
            collection,
            change_sample_count,
        )
        heat_summary.insert(0, "dataset", case.dataset)
        heat_summary.insert(1, "diagnosis", case.diagnosis)
        heat_summary.insert(2, "subject_id", case.subject_id)
        heat_summary.insert(3, "model", model)
        heatmap_rows.extend(heat_summary.to_dict("records"))
        parts.append("<div class='plot-block'>")
        parts.append(figure_html(heat_fig, include_plotlyjs=include_js))
        include_js = False
        parts.append("</div>")
        parts.append("</section>")

    html_text = html_shell(
        title=f"{DATASET_LABELS[case.dataset]} {case.diagnosis} rich case",
        body="\n".join(parts),
        back_link="index.html",
    )
    page_path.write_text(html_text, encoding="utf-8")
    return page_path, heatmap_rows


def html_shell(title: str, body: str, back_link: str | None = None) -> str:
    back = f"<p><a href='{html.escape(back_link)}'>Back to rich report index</a></p>" if back_link else ""
    style = """
    <style>
      body{font-family:Arial,sans-serif;margin:28px auto;max-width:1480px;color:#1f2933;line-height:1.45}
      h1{font-size:28px;margin:0 0 10px} h2{font-size:22px;margin:0 0 8px}
      h3{font-size:18px;margin:0 0 12px}
      section{border-top:1px solid #d9e2ec;padding-top:18px;margin-top:26px}
      .summary{border-top:0;background:#f7f9fc;padding:16px 18px;border-left:4px solid #486581}
      .data-table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
      .data-table th,.data-table td{border:1px solid #d9e2ec;padding:6px 8px;vertical-align:top}
      .data-table th{background:#eef2f7;text-align:left}
      .plot-block{overflow-x:auto;overflow-y:hidden;margin:10px 0 24px;padding-bottom:8px}
      code{background:#eef2f7;padding:1px 4px;border-radius:3px}
      a{color:#1d4ed8;text-decoration:none} a:hover{text-decoration:underline}
      ul{line-height:1.7}
    </style>
    """
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>{style}</head><body>"
        f"{back}<h1>{html.escape(title)}</h1>{body}</body></html>"
    )


def make_predictors(args: argparse.Namespace) -> dict[str, Any]:
    predictors: dict[str, Any] = {}
    for models in RICH_MODELS.values():
        for model in models:
            if model in predictors:
                continue
            if model in fair_eval.BRAINODE_MODELS:
                predictors[model] = BrainODEPredictor(model, args.device, args.checkpoint)
            else:
                predictors[model] = FlowPredictor(
                    model,
                    args.device,
                    args.checkpoint,
                    args.mesh_resolution,
                    args.mesh_max_batch,
                    args.composed_step_years,
                )
    return predictors


def main() -> int:
    args = parse_args()
    future_mesh_dir = repo_path(args.future_mesh_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_future_metrics(future_mesh_dir)
    cases: list[MeshCase] = []
    selected_rows: list[dict[str, Any]] = []
    for dataset in ("old_adni", "qc_large"):
        for diagnosis in ("AD", "CN"):
            case = select_case_for_dataset_diagnosis(
                metrics,
                dataset=dataset,
                diagnosis=diagnosis,
                split=args.split,
                pair_type=args.pair_type,
                strategy=args.case_strategy,
            )
            cases.append(case)
            selected_rows.append(
                {
                    "dataset": case.dataset,
                    "diagnosis": case.diagnosis,
                    "split": case.split,
                    "pair_type": case.pair_type,
                    "subject_id": case.subject_id,
                    "source_scan_id": case.source_scan_id,
                    "target_scan_id": case.target_scan_id,
                    "source_age_years": case.source_age_years,
                    "target_age_years": case.target_age_years,
                }
            )

    predictors = make_predictors(args)
    page_paths: list[Path] = []
    all_heatmap_rows: list[dict[str, Any]] = []
    for case in cases:
        collection = mesh_collection_for_case(
            case=case,
            predictors=predictors,
            output_dir=output_dir,
        )
        page_path, heatmap_rows = write_case_page(
            case=case,
            collection=collection,
            output_dir=output_dir,
            change_sample_count=args.change_sample_count,
        )
        page_paths.append(page_path)
        all_heatmap_rows.extend(heatmap_rows)

    write_csv(output_dir / "selected_rich_cases.csv", selected_rows)
    write_csv(output_dir / "surface_change_heatmap_summary.csv", all_heatmap_rows)
    run_info = {
        "future_mesh_dir": str(future_mesh_dir),
        "output_dir": str(output_dir),
        "split": args.split,
        "pair_type": args.pair_type,
        "case_strategy": args.case_strategy,
        "device": args.device,
        "mesh_resolution": args.mesh_resolution,
        "change_sample_count": args.change_sample_count,
        "models": RICH_MODELS,
        "pages": [path.name for path in page_paths],
    }
    (output_dir / "rich_report_run.json").write_text(
        json.dumps(run_info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sections = []
    for dataset in ("old_adni", "qc_large"):
        paths = [path for path in page_paths if path.name.startswith(dataset)]
        links = "".join(
            f"<li><a href='{html.escape(path.name)}'>{html.escape(path.stem)}</a></li>"
            for path in paths
        )
        sections.append(f"<section><h2>{html.escape(DATASET_LABELS[dataset])}</h2><ul>{links}</ul></section>")
    model_text = "".join(
        f"<li><strong>{html.escape(DATASET_LABELS[dataset])}</strong>: "
        + ", ".join(html.escape(MODEL_LABELS[model]) for model in models)
        + "</li>"
        for dataset, models in RICH_MODELS.items()
    )
    body = (
        "<section class='summary'>"
        "<p>This report is a selected-case rich visualization layer for the finished "
        "future real-mesh evaluation. It does not replace the numeric all-pair CSVs; "
        "it decodes and saves representative meshes for visual inspection.</p>"
        f"<ul>{model_text}</ul>"
        "<p>Each case page separates metrics, volume/surface-area plots, 3D mesh panels, "
        "and source-aligned surface-change heatmaps.</p>"
        "</section>"
        + "\n".join(sections)
        + "<section><h2>Generated Data</h2><ul>"
        "<li><a href='selected_rich_cases.csv'>selected_rich_cases.csv</a></li>"
        "<li><a href='surface_change_heatmap_summary.csv'>surface_change_heatmap_summary.csv</a></li>"
        "<li><a href='rich_report_run.json'>rich_report_run.json</a></li>"
        "</ul></section>"
    )
    (output_dir / "index.html").write_text(
        html_shell("Rich Future Mesh Visualization Report", body),
        encoding="utf-8",
    )
    print(json.dumps(run_info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
