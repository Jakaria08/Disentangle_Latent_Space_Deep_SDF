#!/usr/bin/env python3
"""Generate the selected cocycle versus BrainODE AD trajectory through age 105."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-all-visualization-ood")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
COHORT_TASK_ROOT = EXPERIMENT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
REPO_SCRIPTS = REPO_ROOT / "scripts"
DEFAULT_REGISTRY = EXPERIMENT_DIR / "configs" / "model_registry.json"
DEFAULT_OUTPUT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/All_Visualization_v1")
DATA_ROOT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--subject-id", default="995")
    parser.add_argument("--condition", choices=("CN", "AD"), default="AD")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def choose_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable: {value}")
        torch.cuda.set_device(device)
    return device


def load_direct(method: str, checkpoint: Path, device: torch.device):
    lamm = method.startswith("lamm_global_") or method == "lamm_regional_tokens"
    family_task = COHORT_TASK_ROOT / ("task7_direct_mesh_cocycle_lamm_v1" if lamm else "task5_direct_mesh_cocycle_spiral_unet_v1")
    scripts = family_task / "scripts"
    sys.path.insert(0, str(scripts))
    for name in ("common", "data", "train", "objectives"):
        sys.modules.pop(name, None)
    common = importlib.import_module("common")
    data = importlib.import_module("data")
    train = importlib.import_module("train")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if bool(payload.get("test_data_loaded", False)):
        raise ValueError("Selected cocycle checkpoint records test access during training")
    if lamm:
        common.configure_data_root(DATA_ROOT)
        split = data.load_split("test", device=device)
        train.validate_config(payload["config"])
        model, _ = train.build_model(payload["config"], family_task, device)
    else:
        split = data.load_split("test", DATA_ROOT, device)
        train.validate_config(payload["config"])
        model, _ = train.build_model(payload["config"], DATA_ROOT, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, split, payload


def selected_direct(config: dict[str, Any], output: Path) -> tuple[str, str, Path]:
    ranking = read_json(output / "selected_best_method.json")
    method = ranking.get("method")
    methods = {entry["key"]: entry for entry in config["direct_mesh_methods"]}
    aliases = {
        "spiral_mesh_direct": "mesh_spiral",
        "adaptive_spiral_mesh_direct": "mesh_adaptive",
    }
    key = aliases.get(str(method), str(method))
    if key not in methods:
        # The interactive extrapolation requires direct vertices.  If the overall
        # selected entry is latent, use the best validation-ranked direct method
        # and record this narrower scope explicitly.
        import pandas as pd

        table = pd.read_csv(output / "tables" / "validation_best_method_ranking.csv")
        allowed = set(methods).union(aliases)
        direct = table[table.method.isin(allowed)].sort_values(["selection_rank_sum", "validation_macro_assd_mm"])
        if direct.empty:
            raise ValueError("No centrally ranked direct method is available for OOD mesh generation")
        method = str(direct.iloc[0].method)
        key = aliases.get(method, method)
    entry = methods[key]
    return str(method), str(entry["label"]), Path(entry["checkpoint"]).expanduser().resolve()


def volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    return float(abs(np.sum(triangles[:, 0] * np.cross(triangles[:, 1], triangles[:, 2])) / 6.0))


def structural_metrics(vertices: np.ndarray, source: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    source_triangles = source[faces]
    triangles = vertices[faces]
    source_cross = np.cross(source_triangles[:, 1] - source_triangles[:, 0], source_triangles[:, 2] - source_triangles[:, 0])
    predicted_cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    predicted_norm = np.linalg.norm(predicted_cross, axis=1)
    flip_fraction = float(np.mean((np.sum(source_cross * predicted_cross, axis=1) <= 0.0) | (predicted_norm <= 1e-12)))
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
    source_edges = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    predicted_edges = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    ratios = predicted_edges / np.maximum(source_edges, 1e-8)
    volume_ratio = volume(vertices, faces) / max(volume(source, faces), 1e-8)
    maximum_displacement = float(np.linalg.norm(vertices - source, axis=1).max())
    p01, p99 = np.quantile(ratios, [0.01, 0.99])
    passed = bool(
        np.isfinite(vertices).all() and flip_fraction <= 0.001 and float(np.mean(predicted_norm <= 1e-12)) == 0.0
        and 0.1 <= volume_ratio <= 1.5 and p01 >= 0.2 and p99 <= 5.0 and maximum_displacement <= 30.0
    )
    return {
        "volume_ratio": float(volume_ratio), "flipped_face_fraction": flip_fraction,
        "degenerate_face_fraction": float(np.mean(predicted_norm <= 1e-12)),
        "edge_ratio_p01": float(p01), "edge_ratio_p99": float(p99),
        "maximum_vertex_displacement_mm": maximum_displacement, "structural_guardrail_pass": passed,
    }


def export_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False).export(path)


@torch.no_grad()
def direct_trajectories(model, source: torch.Tensor, source_age: float, ages: list[float], label: float) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    device = source.device
    condition = torch.tensor([label], device=device, dtype=torch.float32)
    one_shot: dict[float, np.ndarray] = {}
    composed: dict[float, np.ndarray] = {}
    for target in ages:
        predicted = model.transport(
            source,
            torch.tensor([source_age], device=device),
            torch.tensor([target], device=device),
            condition,
        )
        one_shot[target] = predicted[0].cpu().numpy().astype(np.float64)
        current = source.clone()
        current_age = float(source_age)
        while current_age < target - 1e-7:
            next_age = min(current_age + 1.0, target)
            current = model.transport(
                current,
                torch.tensor([current_age], device=device),
                torch.tensor([next_age], device=device),
                condition,
            )
            current_age = next_age
        composed[target] = current[0].cpu().numpy().astype(np.float64)
    return one_shot, composed


def brainode_context(device: torch.device):
    sys.path.insert(0, str(REPO_SCRIPTS))
    module = importlib.import_module("build_adni_pca_cocycle_flow_vs_brainode_cache")
    context, provenance = module.load_context("hippocampus", device, ("test",))
    return module, context, provenance


@torch.no_grad()
def brainode_trajectory(module, context, archive: dict[str, np.ndarray], source_index: int, ages: list[float], condition: float) -> dict[float, np.ndarray]:
    train_age = archive["visit_age_years"].astype(np.float64)
    train_norm = archive["visit_age_norm_train"].astype(np.float64)
    slope, intercept = np.polyfit(train_age, train_norm, 1)
    raw_source = archive["visit_pca_150"][source_index].astype(np.float32)
    source_age = float(train_age[source_index])
    output: dict[float, np.ndarray] = {}
    for target in ages:
        values = torch.from_numpy(raw_source[None]).to(context.device)
        times = torch.tensor([slope * source_age + intercept, slope * target + intercept], dtype=torch.float32, device=context.device)
        diagnosis = torch.tensor([condition], dtype=torch.float32, device=context.device)
        predicted_raw = module.integrate_brainode(context.brainode, values, times, diagnosis, context.brainode_substeps)[-1]
        standardized = (predicted_raw.cpu().numpy().astype(np.float64) - context.geometry.score_mean) / context.geometry.score_std
        output[target] = context.geometry.vertices(standardized[None])[0]
    return output


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_html(
    path: Path,
    label: str,
    ages: list[float],
    faces: np.ndarray,
    source: np.ndarray,
    ground_truth: dict[float, np.ndarray],
    cocycle: dict[float, np.ndarray],
    brainode: dict[float, np.ndarray],
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    all_vertices = [source] + list(ground_truth.values()) + list(cocycle.values()) + list(brainode.values())
    stack = np.concatenate(all_vertices, axis=0)
    lower, upper = stack.min(axis=0), stack.max(axis=0)
    center = 0.5 * (lower + upper)
    half = 0.55 * max(float(np.max(upper - lower)), 1.0)
    ranges = [[float(value - half), float(value + half)] for value in center]
    color_max = max(float(np.linalg.norm(vertices - source, axis=1).max()) for vertices in all_vertices)

    def mesh(vertices: np.ndarray, name: str, show_scale: bool, opacity: float = 1.0):
        return go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=np.linalg.norm(vertices - source, axis=1), colorscale="Turbo",
            cmin=0.0, cmax=max(color_max, 1e-4), showscale=show_scale,
            colorbar={"title": "Change<br>(mm)"} if show_scale else None,
            flatshading=True, opacity=opacity, name=name,
            hovertemplate="Change from source %{intensity:.3f} mm<extra></extra>",
        )

    first = ages[0]
    first_gt = ground_truth.get(first, source)
    figure = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Observed surface (when available)", label, "BrainODE"),
    )
    figure.add_trace(mesh(first_gt, "Observed", False, 1.0 if first in ground_truth else 0.08), row=1, col=1)
    figure.add_trace(mesh(cocycle[first], label, False), row=1, col=2)
    figure.add_trace(mesh(brainode[first], "BrainODE", True), row=1, col=3)
    frames = []
    for age in ages:
        gt_available = age in ground_truth
        gt = ground_truth.get(age, source)
        frames.append(
            go.Frame(
                name=f"{age:.2f}", traces=[0, 1, 2],
                data=[mesh(gt, "Observed", False, 1.0 if gt_available else 0.08), mesh(cocycle[age], label, False), mesh(brainode[age], "BrainODE", True)],
                layout=go.Layout(
                    title_text=(
                        f"AD trajectory — age {age:.2f} years"
                        f"<br><sup>{'Observed scan available' if gt_available else 'No observed scan: prediction-only age'}; age 105 is an OOD stress test</sup>"
                    )
                ),
            )
        )
    figure.frames = frames
    scene = {
        "xaxis": {"visible": False, "range": ranges[0]},
        "yaxis": {"visible": False, "range": ranges[1]},
        "zaxis": {"visible": False, "range": ranges[2]},
        "aspectmode": "cube",
        "camera": {"eye": {"x": 1.6, "y": 1.3, "z": 0.9}},
    }
    figure.update_layout(
        title=frames[0].layout.title.text,
        template="plotly_white", height=650, showlegend=False,
        sliders=[{
            "currentvalue": {"prefix": "Target age: "},
            "steps": [
                {"label": f"{age:.1f}", "method": "animate", "args": [[f"{age:.2f}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]}
                for age in ages
            ],
        }],
        scene=scene, scene2=scene, scene3=scene,
        margin={"l": 0, "r": 0, "b": 20, "t": 95},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs="directory", full_html=True, auto_open=False)


def main() -> int:
    args = parse_args()
    output = args.output_root.expanduser().resolve()
    table_dir, figure_dir, html_dir = output / "tables", output / "figures", EXPERIMENT_DIR / "html"
    destination = html_dir / "best_cocycle_vs_brainode_ad_age105.html"
    if destination.exists() and not args.force:
        raise FileExistsError(f"Pass --force to replace {destination}")
    for directory in (table_dir, figure_dir, html_dir, output / "ood_meshes"):
        directory.mkdir(parents=True, exist_ok=True)
    config = read_json(args.registry)
    method, label, checkpoint = selected_direct(config, output)
    device = choose_device(args.device)
    model, split, payload = load_direct(method, checkpoint, device)
    direct_indices = np.flatnonzero(np.asarray(split.subject_ids).astype(str) == str(args.subject_id))
    if not len(direct_indices):
        raise ValueError(f"Subject {args.subject_id} absent from direct test split")
    direct_indices = direct_indices[np.argsort(split.ages[direct_indices].detach().cpu().numpy())]
    source_index = int(direct_indices[0])
    source_age = float(split.ages[source_index].cpu())
    source = split.vertices[source_index : source_index + 1]
    source_np = source[0].cpu().numpy().astype(np.float64)
    label_value = float(args.condition == "AD")
    observed_ages = [float(split.ages[index].cpu()) for index in direct_indices]
    scheduled = [85.0, 90.0, 95.0, 100.0, 105.0]
    ages = sorted({source_age, *observed_ages, *[age for age in scheduled if age > source_age]})
    one_shot, composed = direct_trajectories(model, source, source_age, ages, label_value)
    faces = model.faces.detach().cpu().numpy().astype(np.int64)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    module, context, brain_provenance = brainode_context(device)
    archive = context.archives["test"]
    brain_indices = np.flatnonzero(archive["visit_subject_ids"].astype(str) == str(args.subject_id))
    if not len(brain_indices):
        raise ValueError(f"Subject {args.subject_id} absent from BrainODE test archive")
    brain_indices = brain_indices[np.argsort(archive["visit_age_years"][brain_indices])]
    brain_source = int(brain_indices[0])
    if str(archive["visit_scan_ids"][brain_source]) != str(split.scan_ids[source_index]):
        raise ValueError("Direct and BrainODE source scan IDs do not match")
    brain = brainode_trajectory(module, context, archive, brain_source, ages, label_value)
    if not np.array_equal(faces, context.geometry.faces):
        raise ValueError("Best cocycle and BrainODE do not share the registered mesh faces")

    ground_truth: dict[float, np.ndarray] = {}
    for index in direct_indices:
        ground_truth[float(split.ages[index].cpu())] = split.vertices[index].cpu().numpy().astype(np.float64)
    rows: list[dict[str, Any]] = []
    mesh_root = output / "ood_meshes" / f"subject_{args.subject_id}"
    for age in ages:
        gt = ground_truth.get(age)
        disagreement = np.linalg.norm(one_shot[age] - composed[age], axis=1)
        best_structure = structural_metrics(composed[age], source_np, faces)
        brain_structure = structural_metrics(brain[age], source_np, faces)
        row = {
            "subject_id": str(args.subject_id), "condition": args.condition,
            "source_age_years": source_age, "target_age_years": age,
            "observed_mesh_available": gt is not None,
            "best_method": method, "best_method_label": label,
            "best_composed_volume_mm3": volume(composed[age], faces),
            "best_one_shot_volume_mm3": volume(one_shot[age], faces),
            "brainode_volume_mm3": volume(brain[age], faces),
            "one_shot_vs_composed_mean_vertex_mm": float(disagreement.mean()),
            "one_shot_vs_composed_p95_vertex_mm": float(np.quantile(disagreement, 0.95)),
            "out_of_distribution": age > float(config["training_age_support"]["maximum_years"]),
            **{f"best_{key}": value for key, value in best_structure.items()},
            **{f"brainode_{key}": value for key, value in brain_structure.items()},
        }
        if gt is not None:
            row.update({
                "observed_volume_mm3": volume(gt, faces),
                "best_mean_vertex_error_mm": float(np.linalg.norm(composed[age] - gt, axis=1).mean()),
                "brainode_mean_vertex_error_mm": float(np.linalg.norm(brain[age] - gt, axis=1).mean()),
            })
            export_mesh(mesh_root / "observed" / f"age_{age:06.2f}.ply", gt, faces)
        export_mesh(mesh_root / "best_composed" / f"age_{age:06.2f}.ply", composed[age], faces)
        export_mesh(mesh_root / "best_one_shot" / f"age_{age:06.2f}.ply", one_shot[age], faces)
        export_mesh(mesh_root / "brainode" / f"age_{age:06.2f}.ply", brain[age], faces)
        rows.append(row)
    write_rows(table_dir / "ood_best_vs_brainode_trajectory.csv", rows)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    age_values = np.asarray(ages)
    axis.plot(age_values, [volume(composed[a], faces) for a in ages], marker="o", label=f"{label} (annual composition)")
    axis.plot(age_values, [volume(brain[a], faces) for a in ages], marker="s", label="BrainODE")
    axis.scatter(list(ground_truth), [volume(ground_truth[a], faces) for a in ground_truth], color="black", zorder=4, label="Observed scans")
    axis.axvline(float(config["training_age_support"]["maximum_years"]), color="#e15759", linestyle="--", label="End of training-age support")
    axis.set_xlabel("Age (years)")
    axis.set_ylabel("Mesh volume (mm³)")
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "ood_best_vs_brainode_volume.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    make_html(destination, label, ages, faces, source_np, ground_truth, composed, brain)
    manifest = {
        "schema_version": 1, "status": "complete", "split": "test",
        "subject_id": str(args.subject_id), "diagnosis_condition": args.condition,
        "source_scan_id": str(split.scan_ids[source_index]), "source_age_years": source_age,
        "best_method": method, "best_method_label": label, "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]), "brainode_batch_size": 1,
        "ages": ages, "observed_ages": observed_ages,
        "ood_definition": f"age > {config['training_age_support']['maximum_years']}",
        "interpretation": "geometric extrapolation stress test, not biological ground truth beyond observed ages",
        "structural_guardrail": "volume ratio 0.1-1.5, flips <=0.001, no degenerate faces, edge p01>=0.2, edge p99<=5, max displacement<=30 mm",
        "html": str(destination), "brainode_provenance": brain_provenance,
    }
    (output / "ood_comparison_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
