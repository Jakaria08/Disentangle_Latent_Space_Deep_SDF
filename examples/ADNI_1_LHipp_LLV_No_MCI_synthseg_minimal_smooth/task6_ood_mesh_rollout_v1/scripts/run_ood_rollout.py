#!/usr/bin/env python3
"""Age-75 to age-105 counterfactual rollout for selected mesh and latent cocycles.

This is a validation-only out-of-distribution stress test. It does not claim
that a geometrically valid prediction at age 105 is biologically validated.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-task6-ood-rollout")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/task6-ood-rollout-xdg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[4]
TASK = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = TASK / "configs" / "age75_to105.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run both models, sources, conditions and rollout forms to age 80 without writing.",
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def purge_modules(names: tuple[str, ...], script_path: Path) -> None:
    for name in names:
        sys.modules.pop(name, None)
    sys.path[:] = [item for item in sys.path if item != str(script_path)]


def choose_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {value}")
        torch.cuda.set_device(device)
    return device


def select_sources(split: Any, source_age: float, window: list[float]) -> list[dict[str, Any]]:
    output = []
    diagnoses = split.diagnoses.astype(str)
    ages = split.ages.detach().cpu().numpy().astype(np.float64)
    volumes = split.volumes.detach().cpu().numpy().astype(np.float64)
    for diagnosis in ("CN", "AD"):
        group = np.flatnonzero(diagnoses == diagnosis)
        candidates = group[(ages[group] >= float(window[0])) & (ages[group] <= float(window[1]))]
        if not len(candidates):
            raise ValueError(f"No {diagnosis} validation source in requested age window")
        group_log_volume = np.log(volumes[group])
        median = float(np.median(group_log_volume))
        scale = max(float(1.4826 * np.median(np.abs(group_log_volume - median))), 1.0e-4)
        score = (
            np.abs(ages[candidates] - source_age) / max(float(window[1] - window[0]) / 2.0, 1.0)
            + 0.25 * np.abs(np.log(volumes[candidates]) - median) / scale
        )
        index = int(candidates[int(np.argmin(score))])
        output.append(
            {
                "diagnosis": diagnosis,
                "index": index,
                "scan_id": str(split.scan_ids[index]),
                "subject_id": str(split.subject_ids[index]),
                "actual_age_years": float(ages[index]),
                "model_source_age_years": float(source_age),
                "volume_mm3": float(volumes[index]),
                "selection_score": float(np.min(score)),
                "vertices": split.vertices[index].detach().cpu().numpy().astype(np.float64),
            }
        )
    return output


def scenario_rows(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"source": source, "condition": condition, "label": float(condition == "AD")}
        for source in sources
        for condition in ("CN", "AD")
    ]


def load_direct(config: dict[str, Any], device: torch.device):
    spec = config["direct_mesh"]
    task = resolve(spec["task"])
    scripts = task / "scripts"
    sys.path.insert(0, str(scripts))
    checkpoint = torch.load(resolve(spec["run_dir"]) / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    if bool(checkpoint.get("test_data_loaded", False)):
        raise ValueError("Direct-mesh checkpoint records test access")
    common = importlib.import_module("common")
    data = importlib.import_module("data")
    train = importlib.import_module("train")
    root = common.output_root(None)
    model, statistics = train.build_model(checkpoint["config"], root, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    validation = data.load_split(config["split"], root, "cpu")
    training = data.load_split("train", root, "cpu")
    faces = np.asarray(statistics["faces"], dtype=np.int64)
    modules = ("common", "data", "train", "objectives", "conditional_spiral_unet", "mesh_hierarchy", "mesh_layers")
    purge_modules(modules, scripts)
    return model, validation, training, faces, checkpoint


def direct_rollout(
    model: torch.nn.Module,
    scenarios: list[dict[str, Any]],
    ages: list[int],
    step: int,
    device: torch.device,
) -> tuple[dict[tuple[int, str, int], np.ndarray], list[dict[str, Any]]]:
    source = torch.from_numpy(np.stack([row["source"]["vertices"] for row in scenarios]).astype(np.float32)).to(device)
    labels = torch.tensor([row["label"] for row in scenarios], dtype=torch.float32, device=device)
    baseline = float(ages[0])
    output: dict[tuple[int, str, int], np.ndarray] = {}
    failures: list[dict[str, Any]] = []
    with torch.no_grad():
        for target in ages:
            try:
                value = model.transport(
                    source,
                    torch.full((len(scenarios),), baseline, device=device),
                    torch.full((len(scenarios),), float(target), device=device),
                    labels,
                )
                values = value.detach().cpu().numpy().astype(np.float64)
                for index in range(len(scenarios)):
                    output[(index, "one_shot", int(target))] = values[index]
            except Exception as error:
                failures.append({"path": "one_shot", "age_years": target, "error": repr(error)})
        current = source.clone()
        output.update({(index, "composed", int(baseline)): source[index].detach().cpu().numpy().astype(np.float64) for index in range(len(scenarios))})
        for age in range(int(baseline), int(ages[-1]), int(step)):
            target = min(age + int(step), int(ages[-1]))
            try:
                current = model.transport(
                    current,
                    torch.full((len(scenarios),), float(age), device=device),
                    torch.full((len(scenarios),), float(target), device=device),
                    labels,
                )
                if target in ages:
                    values = current.detach().cpu().numpy().astype(np.float64)
                    for index in range(len(scenarios)):
                        output[(index, "composed", int(target))] = values[index]
            except Exception as error:
                failures.append({"path": "composed", "age_years": target, "error": repr(error)})
                break
    return output, failures


def load_latent(config: dict[str, Any], device: torch.device, source_scan_ids: list[str]):
    spec = config["latent_mesh"]
    task = resolve(spec["task"])
    scripts = task / "scripts"
    sys.path.insert(0, str(scripts))
    common = importlib.import_module("common")
    models = importlib.import_module("models")
    checkpoint = torch.load(resolve(spec["run_dir"]) / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    if bool(checkpoint.get("test_data_loaded", False)):
        raise ValueError("Latent checkpoint records test access")
    archive = common.load_archive(str(spec["representation"]), config["split"])
    training = common.load_archive(str(spec["representation"]), "train")
    geometry = common.build_geometry(str(spec["representation"]), training, device)
    model_config = checkpoint["config"]
    flow = models.DirectC4Flow(
        int(model_config["model"]["latent_dim"]),
        int(model_config["model"]["width"]),
        int(model_config["model"]["residual_blocks"]),
        float(model_config["model"].get("dropout", 0.0)),
    ).to(device)
    flow.load_state_dict(checkpoint["flow_state_dict"], strict=True)
    flow.eval()
    scan_to_index = {str(scan): index for index, scan in enumerate(archive["visit_scan_ids"].astype(str))}
    missing = [scan for scan in source_scan_ids if scan not in scan_to_index]
    if missing:
        raise ValueError(f"Selected direct source scans are absent from latent archive: {missing}")
    indices = [scan_to_index[scan] for scan in source_scan_ids]
    z = archive["visit_latent_standardized_128"][indices].astype(np.float32)
    age_coefficients = np.polyfit(
        archive["visit_age_years"].astype(np.float64),
        archive["visit_age_norm_train"].astype(np.float64),
        1,
    )
    modules = ("common", "models", "evaluate", "c4_objective")
    return flow, geometry, archive, training, z, age_coefficients, checkpoint, scripts, modules


def latent_rollout(
    flow: torch.nn.Module,
    geometry: torch.nn.Module,
    initial_by_source: np.ndarray,
    scenarios: list[dict[str, Any]],
    ages: list[int],
    step: int,
    age_coefficients: np.ndarray,
    device: torch.device,
) -> tuple[
    dict[tuple[int, str, int], np.ndarray],
    dict[tuple[int, str, int], np.ndarray],
    list[dict[str, Any]],
]:
    source_index = [0 if row["source"]["diagnosis"] == "CN" else 1 for row in scenarios]
    z0 = torch.from_numpy(initial_by_source[source_index]).to(device)
    labels = torch.tensor([row["label"] for row in scenarios], dtype=torch.float32, device=device)

    def normalize_age(age: float) -> float:
        return float(age_coefficients[0] * age + age_coefficients[1])

    latent_states: dict[tuple[int, str, int], np.ndarray] = {}
    mesh_states: dict[tuple[int, str, int], np.ndarray] = {}
    failures: list[dict[str, Any]] = []
    baseline = int(ages[0])
    with torch.no_grad():
        for target in ages:
            try:
                value = flow.transport(
                    z0,
                    torch.full((len(scenarios),), normalize_age(baseline), device=device),
                    torch.full((len(scenarios),), normalize_age(target), device=device),
                    labels,
                )
                values = value.detach().cpu().numpy().astype(np.float64)
                for index in range(len(scenarios)):
                    latent_states[(index, "one_shot", int(target))] = values[index]
            except Exception as error:
                failures.append({"stage": "flow", "path": "one_shot", "age_years": target, "error": repr(error)})
        current = z0.clone()
        for index in range(len(scenarios)):
            latent_states[(index, "composed", baseline)] = current[index].detach().cpu().numpy().astype(np.float64)
        for age in range(baseline, int(ages[-1]), int(step)):
            target = min(age + int(step), int(ages[-1]))
            try:
                current = flow.transport(
                    current,
                    torch.full((len(scenarios),), normalize_age(age), device=device),
                    torch.full((len(scenarios),), normalize_age(target), device=device),
                    labels,
                )
                if target in ages:
                    values = current.detach().cpu().numpy().astype(np.float64)
                    for index in range(len(scenarios)):
                        latent_states[(index, "composed", int(target))] = values[index]
            except Exception as error:
                failures.append({"stage": "flow", "path": "composed", "age_years": target, "error": repr(error)})
                break

    for key, latent in latent_states.items():
        index, path, age = key
        if not np.isfinite(latent).all():
            failures.append({"stage": "latent", "scenario": index, "path": path, "age_years": age, "error": "non-finite latent"})
            continue
        try:
            with torch.no_grad():
                vertices = geometry.vertices(torch.from_numpy(latent[None].astype(np.float32)).to(device))[0]
            value = vertices.detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(value).all():
                raise FloatingPointError("decoder returned non-finite vertices")
            mesh_states[key] = value
        except Exception as error:
            failures.append({"stage": "decoder", "scenario": index, "path": path, "age_years": age, "error": repr(error)})
    return mesh_states, latent_states, failures


def unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    return np.unique(np.sort(edges, axis=1), axis=0)


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    signed = np.sum(triangles[:, 0] * np.cross(triangles[:, 1], triangles[:, 2])) / 6.0
    return float(abs(signed))


def nearest_rmse(query: np.ndarray, reference_flat: np.ndarray, reference_square_mean: np.ndarray) -> float:
    flat = query.astype(np.float32, copy=False).reshape(-1)
    query_square = float(np.mean(flat * flat))
    cross = reference_flat @ flat / float(flat.size)
    mse = np.maximum(reference_square_mean + query_square - 2.0 * cross, 0.0)
    return float(np.sqrt(np.min(mse)))


def mesh_metrics(
    vertices: np.ndarray,
    source: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    train_flat: np.ndarray,
    train_square_mean: np.ndarray,
    guardrails: dict[str, float],
) -> dict[str, Any]:
    finite = bool(np.isfinite(vertices).all())
    if not finite:
        return {"finite": False, "structural_guardrail_pass": False}
    source_triangles = source[faces]
    triangles = vertices[faces]
    source_cross = np.cross(source_triangles[:, 1] - source_triangles[:, 0], source_triangles[:, 2] - source_triangles[:, 0])
    predicted_cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    predicted_cross_norm = np.linalg.norm(predicted_cross, axis=1)
    flips = np.logical_or(np.sum(source_cross * predicted_cross, axis=1) <= 0.0, predicted_cross_norm <= 1.0e-12)
    source_length = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    predicted_length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    edge_ratio = predicted_length / np.maximum(source_length, 1.0e-8)
    displacement = np.linalg.norm(vertices - source, axis=1)
    source_volume = mesh_volume(source, faces)
    volume = mesh_volume(vertices, faces)
    volume_ratio = volume / max(source_volume, 1.0e-8)
    flip_fraction = float(np.mean(flips))
    degenerate_fraction = float(np.mean(predicted_cross_norm <= 1.0e-12))
    p01, median, p99 = np.quantile(edge_ratio, [0.01, 0.5, 0.99])
    maximum_displacement = float(np.max(displacement))
    passed = bool(
        flip_fraction <= float(guardrails["maximum_flipped_face_fraction"])
        and degenerate_fraction <= float(guardrails["maximum_degenerate_face_fraction"])
        and float(guardrails["minimum_volume_ratio"]) <= volume_ratio <= float(guardrails["maximum_volume_ratio"])
        and p01 >= float(guardrails["minimum_edge_ratio_p01"])
        and p99 <= float(guardrails["maximum_edge_ratio_p99"])
        and maximum_displacement <= float(guardrails["maximum_vertex_displacement_mm"])
    )
    return {
        "finite": True,
        "volume_mm3": volume,
        "source_volume_mm3": source_volume,
        "volume_ratio": volume_ratio,
        "mean_vertex_displacement_mm": float(np.mean(displacement)),
        "p95_vertex_displacement_mm": float(np.quantile(displacement, 0.95)),
        "maximum_vertex_displacement_mm": maximum_displacement,
        "centroid_shift_mm": float(np.linalg.norm(vertices.mean(axis=0) - source.mean(axis=0))),
        "bounding_box_diagonal_mm": float(np.linalg.norm(np.ptp(vertices, axis=0))),
        "flipped_face_fraction": flip_fraction,
        "degenerate_face_fraction": degenerate_fraction,
        "edge_ratio_p01": float(p01),
        "edge_ratio_median": float(median),
        "edge_ratio_p99": float(p99),
        "nearest_training_mesh_rmse_mm": nearest_rmse(vertices, train_flat, train_square_mean),
        "structural_guardrail_pass": passed,
    }


def export_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    mesh.export(path, file_type="ply")


def make_html(
    path: Path,
    label: str,
    scenario: dict[str, Any],
    ages: list[int],
    meshes: dict[tuple[int, str, int], np.ndarray],
    scenario_index: int,
    faces: np.ndarray,
) -> list[int]:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    valid_ages = [
        age
        for age in ages
        if all(
            (scenario_index, rollout, int(age)) in meshes
            and np.isfinite(meshes[(scenario_index, rollout, int(age))]).all()
            for rollout in ("one_shot", "composed")
        )
    ]
    if not valid_ages:
        return []
    source = scenario["source"]["vertices"]
    all_vertices = [meshes[(scenario_index, rollout, age)] for age in valid_ages for rollout in ("one_shot", "composed")]
    stacked = np.concatenate(all_vertices, axis=0)
    lower, upper = stacked.min(axis=0), stacked.max(axis=0)
    center = 0.5 * (lower + upper)
    half = 0.55 * max(float(np.max(upper - lower)), 1.0)
    ranges = [[float(value - half), float(value + half)] for value in center]
    color_max = max(float(np.max(np.linalg.norm(value - source, axis=1))) for value in all_vertices)
    color_max = max(color_max, 1.0e-4)

    def trace(vertices: np.ndarray, title: str, scale: bool):
        return go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=np.linalg.norm(vertices - source, axis=1),
            colorscale="Viridis", cmin=0.0, cmax=color_max,
            colorbar={"title": "Displacement<br>(mm)"} if scale else None,
            showscale=scale, flatshading=True, name=title,
            hovertemplate="Displacement %{intensity:.3f} mm<extra></extra>",
        )

    first = valid_ages[0]
    figure = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("One-shot from age 75", "Annual composition"),
    )
    figure.add_trace(trace(meshes[(scenario_index, "one_shot", first)], "One-shot", False), row=1, col=1)
    figure.add_trace(trace(meshes[(scenario_index, "composed", first)], "Composed", True), row=1, col=2)
    frames = []
    for age in valid_ages:
        left = meshes[(scenario_index, "one_shot", age)]
        right = meshes[(scenario_index, "composed", age)]
        direct_ratio = mesh_volume(left, faces) / max(mesh_volume(source, faces), 1.0e-8)
        composed_ratio = mesh_volume(right, faces) / max(mesh_volume(source, faces), 1.0e-8)
        frames.append(
            go.Frame(
                name=str(age), traces=[0, 1],
                data=[trace(left, "One-shot", False), trace(right, "Composed", True)],
                layout=go.Layout(
                    title_text=(
                        f"{label}: {scenario['source']['diagnosis']} source under {scenario['condition']} condition — age {age}"
                        f"<br><sup>Volume/source: one-shot {direct_ratio:.3f}, composed {composed_ratio:.3f}</sup>"
                    )
                ),
            )
        )
    figure.frames = frames
    sliders = [{
        "active": 0,
        "currentvalue": {"prefix": "Target age: "},
        "steps": [
            {"label": str(age), "method": "animate", "args": [[str(age)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]}
            for age in valid_ages
        ],
    }]
    scene = {
        "xaxis": {"visible": False, "range": ranges[0]},
        "yaxis": {"visible": False, "range": ranges[1]},
        "zaxis": {"visible": False, "range": ranges[2]},
        "aspectmode": "cube",
        "camera": {"eye": {"x": 1.6, "y": 1.4, "z": 0.9}},
    }
    figure.update_layout(
        title=frames[0].layout.title.text,
        template="plotly_white", height=650, showlegend=False,
        sliders=sliders, scene=scene, scene2=scene,
        margin={"l": 0, "r": 0, "b": 20, "t": 95},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs="directory", full_html=True, auto_open=False)
    return valid_ages


def plot_trajectories(metrics: pd.DataFrame, divergences: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row_index, source in enumerate(("CN", "AD")):
        for method, color in (("mesh_adaptive", "#8c2d91"), ("latent_spiral", "#59a14f")):
            for condition, style in (("CN", "-"), ("AD", "--")):
                current = metrics[
                    metrics.method.eq(method)
                    & metrics.source_diagnosis.eq(source)
                    & metrics.condition.eq(condition)
                    & metrics.rollout.eq("one_shot")
                ].sort_values("age_years")
                axes[row_index, 0].plot(current.age_years, current.volume_ratio, color=color, linestyle=style, marker="o", label=f"{current.method_label.iloc[0]} / {condition}")
                composed = metrics[
                    metrics.method.eq(method)
                    & metrics.source_diagnosis.eq(source)
                    & metrics.condition.eq(condition)
                    & metrics.rollout.eq("composed")
                ].sort_values("age_years")
                axes[row_index, 0].plot(composed.age_years, composed.volume_ratio, color=color, linestyle=style, alpha=0.35)
        axes[row_index, 0].set_title(f"{source} source: volume/source")
        axes[row_index, 0].axhline(1.0, color="black", linewidth=0.7)
        current_divergence = divergences[divergences.source_diagnosis.eq(source)]
        for method, color in (("mesh_adaptive", "#8c2d91"), ("latent_spiral", "#59a14f")):
            for condition, style in (("CN", "-"), ("AD", "--")):
                current = current_divergence[current_divergence.method.eq(method) & current_divergence.condition.eq(condition)].sort_values("age_years")
                axes[row_index, 1].plot(current.age_years, current.mean_vertex_distance_mm, color=color, linestyle=style, marker="o", label=f"{current.method_label.iloc[0]} / {condition}")
        axes[row_index, 1].set_title(f"{source} source: composition disagreement")
        axes[row_index, 1].set_ylabel("One-shot vs composed (mm)")
        axes[row_index, 0].set_ylabel("Volume ratio")
        for axis in axes[row_index]:
            axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Target age (years)")
    axes[1, 1].set_xlabel("Target age (years)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.suptitle("Age-75 to age-105 OOD stress test\nOpaque = one-shot; faint = annually composed", y=1.02)
    figure.tight_layout()
    figure.savefig(figure_dir / "ood_volume_and_composition.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    if config.get("split") != "val":
        raise ValueError("OOD development analysis is validation-only")
    device = choose_device(args.device)
    ages = [int(value) for value in config["target_ages_years"]]
    if args.smoke:
        ages = ages[:2]
    if ages[0] != int(config["source_age_years"]) or ages[-1] <= ages[0]:
        raise ValueError("Invalid rollout age grid")
    step = int(config["composition_step_years"])
    if step <= 0 or any((age - ages[0]) % step for age in ages):
        raise ValueError("Saved ages must be reachable by the composition step")

    direct_model, validation, training, faces, direct_checkpoint = load_direct(config, device)
    sources = select_sources(validation, float(config["source_age_years"]), config["source_selection_window_years"])
    scenarios = scenario_rows(sources)
    direct_meshes, direct_failures = direct_rollout(direct_model, scenarios, ages, step, device)
    del direct_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    latent = load_latent(config, device, [row["scan_id"] for row in sources])
    flow, geometry, latent_archive, latent_training, z0, age_coefficients, latent_checkpoint, latent_scripts, latent_modules = latent
    latent_meshes, latent_states, latent_failures = latent_rollout(
        flow, geometry, z0, scenarios, ages, step, age_coefficients, device
    )
    latent_faces = geometry.faces.detach().cpu().numpy().astype(np.int64)
    if not np.array_equal(faces, latent_faces):
        raise ValueError("Selected direct and latent methods do not share identical faces")
    purge_modules(latent_modules, latent_scripts)
    del flow, geometry
    if device.type == "cuda":
        torch.cuda.empty_cache()

    expected = len(scenarios) * 2 * len(ages)
    if args.smoke:
        if len(direct_meshes) != expected or len(latent_meshes) != expected:
            raise RuntimeError(f"Smoke rollout incomplete: direct={len(direct_meshes)}, latent={len(latent_meshes)}, expected={expected}")
        print(json.dumps({
            "status": "smoke_passed", "split": "val", "test_data_loaded": False,
            "ages": ages, "scenarios": len(scenarios),
            "direct_meshes": len(direct_meshes), "latent_meshes": len(latent_meshes),
            "direct_failures": direct_failures, "latent_failures": latent_failures,
        }, indent=2))
        return 0

    output_root = resolve(args.output_root) if args.output_root else resolve(config["output_root"])
    report_dir = resolve(config["workspace_report_dir"])
    if (output_root / "summary.json").exists() and not args.force:
        raise FileExistsError(f"Completed output already exists; pass --force to replace files: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_vertices = training.vertices.detach().cpu().numpy().astype(np.float32)
    train_flat = train_vertices.reshape(len(train_vertices), -1)
    train_square_mean = np.mean(train_flat * train_flat, axis=1)
    train_latent = latent_training["visit_latent_standardized_128"].astype(np.float32)
    latent_square_mean = np.mean(train_latent * train_latent, axis=1)
    edges = unique_edges(faces)
    methods = {
        str(config["direct_mesh"]["key"]): (str(config["direct_mesh"]["label"]), direct_meshes, {}),
        str(config["latent_mesh"]["key"]): (str(config["latent_mesh"]["label"]), latent_meshes, latent_states),
    }
    metrics: list[dict[str, Any]] = []
    mesh_paths: dict[tuple[str, int, str, int], str] = {}
    latent_paths: list[str] = []
    for method, (label, meshes, method_latents) in methods.items():
        for scenario_index, scenario in enumerate(scenarios):
            for rollout in ("one_shot", "composed"):
                latent_values = []
                latent_ages = []
                for age in ages:
                    key = (scenario_index, rollout, age)
                    vertices = meshes.get(key)
                    base = {
                        "method": method, "method_label": label,
                        "source_diagnosis": scenario["source"]["diagnosis"],
                        "source_scan_id": scenario["source"]["scan_id"],
                        "source_subject_id": scenario["source"]["subject_id"],
                        "source_actual_age_years": scenario["source"]["actual_age_years"],
                        "model_source_age_years": config["source_age_years"],
                        "condition": scenario["condition"], "rollout": rollout,
                        "age_years": age, "horizon_years": age - ages[0],
                        "mesh_generated": vertices is not None,
                    }
                    if vertices is None:
                        metrics.append({**base, "finite": False, "structural_guardrail_pass": False})
                        continue
                    values = mesh_metrics(
                        vertices, scenario["source"]["vertices"], faces, edges,
                        train_flat, train_square_mean, config["geometric_guardrails"],
                    )
                    if method_latents and key in method_latents:
                        latent_value = method_latents[key]
                        values.update({
                            "latent_rms_standard_units": float(np.sqrt(np.mean(latent_value**2))),
                            "latent_max_abs_standard_units": float(np.max(np.abs(latent_value))),
                            "nearest_training_latent_rmse_standard_units": nearest_rmse(latent_value, train_latent, latent_square_mean),
                        })
                        latent_values.append(latent_value)
                        latent_ages.append(age)
                    mesh_path = (
                        output_root / "meshes" / method
                        / f"source_{scenario['source']['diagnosis'].lower()}_{scenario['source']['scan_id']}"
                        / f"condition_{scenario['condition'].lower()}" / rollout / f"age_{age:03d}.ply"
                    )
                    export_ply(mesh_path, vertices, faces)
                    mesh_paths[(method, scenario_index, rollout, age)] = str(mesh_path)
                    metrics.append({**base, **values, "mesh_path": str(mesh_path)})
                if latent_values:
                    latent_path = (
                        output_root / "latents" / method
                        / f"source_{scenario['source']['diagnosis'].lower()}_{scenario['source']['scan_id']}"
                        / f"condition_{scenario['condition'].lower()}" / f"{rollout}.npz"
                    )
                    latent_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(latent_path, ages_years=np.asarray(latent_ages), latent_standardized=np.stack(latent_values))
                    latent_paths.append(str(latent_path))

    metric_frame = pd.DataFrame(metrics)
    divergences: list[dict[str, Any]] = []
    for method, (label, meshes, _) in methods.items():
        for scenario_index, scenario in enumerate(scenarios):
            for age in ages:
                left = meshes.get((scenario_index, "one_shot", age))
                right = meshes.get((scenario_index, "composed", age))
                if left is None or right is None or not np.isfinite(left).all() or not np.isfinite(right).all():
                    continue
                distance = np.linalg.norm(left - right, axis=1)
                divergences.append({
                    "method": method, "method_label": label,
                    "source_diagnosis": scenario["source"]["diagnosis"],
                    "source_scan_id": scenario["source"]["scan_id"],
                    "condition": scenario["condition"], "age_years": age,
                    "mean_vertex_distance_mm": float(np.mean(distance)),
                    "p95_vertex_distance_mm": float(np.quantile(distance, 0.95)),
                    "maximum_vertex_distance_mm": float(np.max(distance)),
                    "absolute_volume_ratio_difference": abs(
                        mesh_volume(left, faces) - mesh_volume(right, faces)
                    ) / max(mesh_volume(scenario["source"]["vertices"], faces), 1.0e-8),
                })
    divergence_frame = pd.DataFrame(divergences)

    contrasts: list[dict[str, Any]] = []
    for method in methods:
        for source_diagnosis in ("CN", "AD"):
            for rollout in ("one_shot", "composed"):
                for age in ages:
                    subset = metric_frame[
                        metric_frame.method.eq(method)
                        & metric_frame.source_diagnosis.eq(source_diagnosis)
                        & metric_frame.rollout.eq(rollout)
                        & metric_frame.age_years.eq(age)
                    ]
                    if set(subset.condition) != {"CN", "AD"} or not subset.mesh_generated.all():
                        continue
                    cn = subset[subset.condition.eq("CN")].iloc[0]
                    ad = subset[subset.condition.eq("AD")].iloc[0]
                    contrasts.append({
                        "method": method, "method_label": str(cn.method_label),
                        "source_diagnosis": source_diagnosis, "rollout": rollout, "age_years": age,
                        "ad_minus_cn_volume_ratio": float(ad.volume_ratio - cn.volume_ratio),
                        "ad_condition_smaller_volume": bool(ad.volume_ratio <= cn.volume_ratio + 1.0e-8),
                    })

    html_records = []
    for method, (label, meshes, _) in methods.items():
        for scenario_index, scenario in enumerate(scenarios):
            name = f"{method}__source_{scenario['source']['diagnosis'].lower()}__condition_{scenario['condition'].lower()}.html"
            html_path = report_dir / name
            valid_ages = make_html(html_path, label, scenario, ages, meshes, scenario_index, faces)
            html_records.append({
                "method": method, "method_label": label,
                "source_diagnosis": scenario["source"]["diagnosis"],
                "condition": scenario["condition"],
                "html_path": str(html_path) if valid_ages else "",
                "ages_rendered": valid_ages,
            })

    table_dir = output_root / "tables"
    write_csv(table_dir / "trajectory_metrics.csv", metrics)
    write_csv(table_dir / "composition_disagreement.csv", divergences)
    write_csv(table_dir / "condition_contrast.csv", contrasts)
    write_csv(table_dir / "html_inventory.csv", html_records)
    plot_trajectories(metric_frame, divergence_frame, report_dir)

    training_age_range = [float(training.ages.min()), float(training.ages.max())]
    latent_training_age_range = [
        float(np.min(latent_training["visit_age_years"])),
        float(np.max(latent_training["visit_age_years"])),
    ]
    final = metric_frame[metric_frame.age_years.eq(ages[-1])]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "analysis_kind": "counterfactual_ood_stress_test_not_biological_validation",
        "split": "val",
        "test_data_loaded": False,
        "source_age_years": ages[0],
        "final_age_years": ages[-1],
        "target_ages_years": ages,
        "composition_step_years": step,
        "direct_training_age_range_years": training_age_range,
        "latent_training_age_range_years": latent_training_age_range,
        "out_of_training_age_support": bool(ages[-1] > max(training_age_range + latent_training_age_range)),
        "sources": [{key: value for key, value in row.items() if key not in {"vertices", "index"}} for row in sources],
        "methods": {
            str(config["direct_mesh"]["key"]): {
                "label": config["direct_mesh"]["label"], "checkpoint_epoch": int(direct_checkpoint["epoch"]),
            },
            str(config["latent_mesh"]["key"]): {
                "label": config["latent_mesh"]["label"], "checkpoint_epoch": int(latent_checkpoint["epoch"]),
            },
        },
        "expected_meshes": len(methods) * len(scenarios) * 2 * len(ages),
        "generated_meshes": int(metric_frame.mesh_generated.sum()),
        "structural_guardrail_passes": int(metric_frame.structural_guardrail_pass.fillna(False).sum()),
        "age105_structural_guardrail_passes": int(final.structural_guardrail_pass.fillna(False).sum()),
        "direct_failures": direct_failures,
        "latent_failures": latent_failures,
        "html_reports": html_records,
        "latent_state_files": latent_paths,
        "self_intersection_checked": False,
        "interpretation": (
            "Geometric validity and condition sensitivity at unsupported ages are stress-test diagnostics. "
            "They do not establish biological plausibility or calibrated age-105 prediction."
        ),
    }
    write_json(output_root / "summary.json", summary)
    write_json(report_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
