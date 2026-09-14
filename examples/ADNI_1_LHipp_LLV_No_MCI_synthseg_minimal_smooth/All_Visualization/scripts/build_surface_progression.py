#!/usr/bin/env python3
"""Build matched, surface-space progression analyses from frozen checkpoints.

This script does not train or modify a model.  It evaluates the validation
cohort shared by the PCA cocycle, direct Spiral cocycles, PCA plain ODE, and
PCA BrainODE.  The output separates three questions that were previously mixed:

1. Does the diagonal field agree with fitted longitudinal surface change?
2. Does the diagonal field accumulate to the observed inter-visit change?
3. When the same baseline anatomy is assigned CN or AD conditioning, do the
   resulting trajectories separate smoothly in interpretable shape measures?

All physical velocities are rigid-body filtered before local shape metrics are
computed.  The observed instantaneous reference remains an estimate fitted to
repeated scans, not a directly measured continuous-time ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, NamedTuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-all-vis-progression")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/all-vis-progression-xdg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
COHORT_ROOT = EXPERIMENT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
DEFAULT_REGISTRY = EXPERIMENT_DIR / "configs" / "model_registry.json"
DEFAULT_DATA_ROOT = Path(
    "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1"
)

METHODS = {
    "mesh_spiral": "Direct Spiral cocycle",
    "mesh_adaptive": "Direct Adaptive cocycle",
    "latent_pca": "Latent PCA cocycle",
    "latent_spiral": "Latent Spiral cocycle",
    "latent_adaptive": "Latent Adaptive cocycle",
    "lamm_n3": "Latent LAMM N3 ensemble",
    "pca_plain_ode": "PCA plain ODE",
    "pca_brainode": "PCA BrainODE",
}
COLORS = {
    "mesh_spiral": "#0072B2",
    "mesh_adaptive": "#009E73",
    "latent_pca": "#56B4E9",
    "latent_spiral": "#9467BD",
    "latent_adaptive": "#8C564B",
    "lamm_n3": "#BCBD22",
    "pca_plain_ode": "#D55E00",
    "pca_brainode": "#CC79A7",
    "observed": "#111111",
    "CN": "#0072B2",
    "AD": "#D55E00",
}
DETAILED_METHODS = [
    "mesh_spiral",
    "mesh_adaptive",
    "latent_spiral",
    "latent_adaptive",
    "latent_pca",
    "lamm_n3",
    "pca_plain_ode",
    "pca_brainode",
]
WITHOUT_DIRECT_METHODS = [
    method for method in DETAILED_METHODS if method not in {"mesh_spiral", "mesh_adaptive"}
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--trajectory-years", type=float, default=5.0)
    parser.add_argument("--trajectory-step", type=float, default=0.5)
    parser.add_argument("--quadrature-points", type=int, default=7)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    signed = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
    if signed < 0.0:
        normals *= -1.0
    return normals


def kabsch(moving: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (moving - moving_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return moving @ rotation + target_center - moving_center @ rotation, rotation


def remove_rigid_velocity(vertices: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Remove the least-squares translation and infinitesimal rotation."""
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    x, y, z = centered.T
    design = np.zeros((3 * len(vertices), 6), dtype=np.float64)
    design[0::3, 0] = 1.0
    design[1::3, 1] = 1.0
    design[2::3, 2] = 1.0
    # omega cross x, with columns [omega_x, omega_y, omega_z].
    design[0::3, 4] = z
    design[0::3, 5] = -y
    design[1::3, 3] = -z
    design[1::3, 5] = x
    design[2::3, 3] = y
    design[2::3, 4] = -x
    fitted = design @ np.linalg.lstsq(design, velocity.reshape(-1), rcond=None)[0]
    return velocity - fitted.reshape(-1, 3)


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    signed = np.einsum(
        "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
    ).sum() / 6.0
    return abs(float(signed))


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )


def face_area_rates(vertices: np.ndarray, velocity: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangle = vertices[faces]
    rate = velocity[faces]
    edge1 = triangle[:, 1] - triangle[:, 0]
    edge2 = triangle[:, 2] - triangle[:, 0]
    cross = np.cross(edge1, edge2)
    unit = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1.0e-12)
    derivative = np.cross(rate[:, 1] - rate[:, 0], edge2) + np.cross(
        edge1, rate[:, 2] - rate[:, 0]
    )
    return 0.5 * np.einsum("ij,ij->i", unit, derivative)


class GeometryDefinition(NamedTuple):
    vertex_regions: np.ndarray
    face_regions: np.ndarray
    region_labels: tuple[str, ...]
    template: np.ndarray
    template_centerline: np.ndarray


def geometry_definition(template: np.ndarray, faces: np.ndarray) -> GeometryDefinition:
    centered = template - template.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    axis = right[0]
    coordinate = centered @ axis
    low, high = np.quantile(coordinate, [1.0 / 3.0, 2.0 / 3.0])
    vertex_regions = np.digitize(coordinate, [low, high]).astype(np.int64)
    face_regions = np.asarray(
        [np.bincount(vertex_regions[face], minlength=3).argmax() for face in faces], dtype=np.int64
    )

    # A reproducible template centreline: local centroid in long-axis slabs.
    edges = np.quantile(coordinate, np.linspace(0.0, 1.0, 25))
    bins = np.clip(np.digitize(coordinate, edges[1:-1]), 0, len(edges) - 2)
    centers = np.stack([template[bins == index].mean(axis=0) for index in range(len(edges) - 1)])
    for _ in range(3):
        padded = np.vstack((centers[0], centers, centers[-1]))
        centers = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
    centerline = centers[bins]

    # The wider end is reported as head-side, without claiming a subfield atlas.
    radius = np.linalg.norm(template - centerline, axis=1)
    if radius[vertex_regions == 0].mean() >= radius[vertex_regions == 2].mean():
        labels = ("Head-side third", "Middle third", "Tail-side third")
    else:
        labels = ("Tail-side third", "Middle third", "Head-side third")
    return GeometryDefinition(vertex_regions, face_regions, labels, template, centerline)


def baseline_radial_geometry(baseline: np.ndarray, definition: GeometryDefinition) -> tuple[np.ndarray, np.ndarray]:
    # Transfer the template centreline offsets to each already-corresponded baseline.
    local_shift = baseline.mean(axis=0) - definition.template.mean(axis=0)
    centerline = definition.template_centerline + local_shift
    radial = baseline - centerline
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1.0e-12)
    return centerline, radial


def region_mean(values: np.ndarray, regions: np.ndarray, region: int) -> float:
    selected = values[regions == region]
    return float(np.mean(selected)) if len(selected) else math.nan


def scan_region_rows(
    method: str,
    cache: dict[str, np.ndarray],
    predicted_fields: np.ndarray,
    faces: np.ndarray,
    definition: GeometryDefinition,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, vertices in enumerate(cache["vertices_mm"]):
        normals = vertex_normals(vertices, faces)
        predicted = remove_rigid_velocity(vertices, predicted_fields[index])
        observed = cache["velocity_reference_mm_per_year"][index].astype(np.float64)
        predicted_normal = np.einsum("ij,ij->i", predicted, normals)
        observed_normal = np.einsum("ij,ij->i", observed, normals)
        _, radial = baseline_radial_geometry(vertices, definition)
        predicted_radial = np.einsum("ij,ij->i", predicted, radial)
        observed_radial = np.einsum("ij,ij->i", observed, radial)
        areas = face_areas(vertices, faces)
        predicted_area = face_area_rates(vertices, predicted, faces)
        observed_area = face_area_rates(vertices, observed, faces)
        region_specs = [(-1, "Whole surface")] + list(enumerate(definition.region_labels))
        for region, label in region_specs:
            vertex_mask = np.ones(len(vertices), dtype=bool) if region < 0 else definition.vertex_regions == region
            face_mask = np.ones(len(faces), dtype=bool) if region < 0 else definition.face_regions == region
            rows.append(
                {
                    "method": method,
                    "method_label": METHODS[method],
                    "scan_id": str(cache["scan_ids"][index]),
                    "subject_id": str(cache["subject_ids"][index]),
                    "diagnosis": str(cache["diagnoses"][index]),
                    "age_years": float(cache["age_years"][index]),
                    "region": label,
                    "reference_reliability": float(cache["velocity_reference_weight"][index]),
                    "predicted_inward_mm_per_year": float(-predicted_normal[vertex_mask].mean()),
                    "observed_inward_mm_per_year": float(-observed_normal[vertex_mask].mean()),
                    "predicted_radial_narrowing_mm_per_year": float(-predicted_radial[vertex_mask].mean()),
                    "observed_radial_narrowing_mm_per_year": float(-observed_radial[vertex_mask].mean()),
                    "predicted_area_change_percent_per_year": float(
                        100.0 * predicted_area[face_mask].sum() / max(areas[face_mask].sum(), 1.0e-12)
                    ),
                    "observed_area_change_percent_per_year": float(
                        100.0 * observed_area[face_mask].sum() / max(areas[face_mask].sum(), 1.0e-12)
                    ),
                    "normal_rmse_mm_per_year": float(
                        np.sqrt(np.mean((predicted_normal[vertex_mask] - observed_normal[vertex_mask]) ** 2))
                    ),
                    "zero_normal_rmse_mm_per_year": float(
                        np.sqrt(np.mean(observed_normal[vertex_mask] ** 2))
                    ),
                }
            )
    return rows


def subject_equal_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.columns
        if column.startswith("predicted_") or column.startswith("observed_")
    ] + ["normal_rmse_mm_per_year", "zero_normal_rmse_mm_per_year"]
    numeric = [column for column in dict.fromkeys(numeric) if column in frame]
    subject_columns = group_columns + ["subject_id"]
    subjects = frame.groupby(subject_columns, as_index=False)[numeric].mean()
    result = subjects.groupby(group_columns, as_index=False)[numeric].mean()
    counts = subjects.groupby(group_columns, as_index=False).subject_id.nunique().rename(columns={"subject_id": "subjects"})
    result = result.merge(counts, on=group_columns)
    if {"normal_rmse_mm_per_year", "zero_normal_rmse_mm_per_year"} <= set(result):
        result["normal_error_to_zero_ratio"] = result.normal_rmse_mm_per_year / result.zero_normal_rmse_mm_per_year.clip(lower=1e-12)
    return result


def bootstrap_curve(
    frame: pd.DataFrame,
    value: str,
    groups: list[str],
    samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        per_subject = group.groupby("subject_id", as_index=False)[value].mean()
        values = per_subject[value].to_numpy(np.float64)
        if len(values) == 0:
            continue
        if len(values) == 1 or samples <= 0:
            low = high = float(values.mean())
        else:
            indices = rng.integers(0, len(values), size=(samples, len(values)))
            means = values[indices].mean(axis=1)
            low, high = np.quantile(means, [0.025, 0.975])
        row = dict(zip(groups, keys))
        row.update({"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "subjects": len(values)})
        rows.append(row)
    return pd.DataFrame(rows)


class Backend:
    method: str

    def source_state(self, indices: np.ndarray):
        raise NotImplementedError

    def state_at(self, source_state, source_age: np.ndarray, target_age: np.ndarray, condition: np.ndarray):
        raise NotImplementedError

    def mesh(self, state) -> np.ndarray:
        raise NotImplementedError

    def velocity(self, state, age: np.ndarray, condition: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return None


def clear_local_modules(script_path: Path) -> None:
    names = {
        "common", "data", "train", "objectives", "models", "evaluate", "c4_objective",
        "conditional_spiral_unet", "conditional_lamm_flow", "mesh_hierarchy", "layers",
    }
    for name in names:
        sys.modules.pop(name, None)
    sys.path[:] = [entry for entry in sys.path if entry != str(script_path)]


class DirectBackend(Backend):
    def __init__(self, method: str, checkpoint: Path, cache: dict[str, np.ndarray], device: torch.device):
        self.method = method
        self.device = device
        self.cache = cache
        self.script_path = COHORT_ROOT / "task5_direct_mesh_cocycle_spiral_unet_v1" / "scripts"
        clear_local_modules(self.script_path)
        sys.path.insert(0, str(self.script_path))
        common = importlib.import_module("common")
        data = importlib.import_module("data")
        train = importlib.import_module("train")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        split = data.load_split("val", DEFAULT_DATA_ROOT, device)
        train.validate_config(payload["config"])
        model, _ = train.build_model(payload["config"], DEFAULT_DATA_ROOT, device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model = model.eval()
        self.vertices = split.vertices

    def source_state(self, indices: np.ndarray) -> torch.Tensor:
        return self.vertices[torch.as_tensor(indices, dtype=torch.long, device=self.device)]

    def state_at(self, source_state, source_age, target_age, condition):
        source = torch.as_tensor(source_age, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(target_age, dtype=torch.float32, device=self.device)
        label = torch.as_tensor(condition, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model.transport(source_state, source, target, label)

    def mesh(self, state) -> np.ndarray:
        return state.detach().cpu().numpy().astype(np.float64)

    def velocity(self, state, age, condition) -> np.ndarray:
        time = torch.as_tensor(age, dtype=torch.float32, device=self.device)
        label = torch.as_tensor(condition, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            value = self.model.instantaneous_velocity(state, time, label)
        return value.detach().cpu().numpy().astype(np.float64)

    def close(self) -> None:
        del self.model, self.vertices
        clear_local_modules(self.script_path)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class PCABackend(Backend):
    def __init__(
        self,
        method: str,
        run_dir: Path,
        task_dir: Path,
        device: torch.device,
        representation: str = "pca128",
    ):
        self.method = method
        self.device = device
        self.script_path = task_dir / "scripts"
        clear_local_modules(self.script_path)
        sys.path.insert(0, str(self.script_path))
        common = importlib.import_module("common")
        models = importlib.import_module("models")
        archive = common.load_archive(representation, "val")
        train_archive = common.load_archive(representation, "train")
        self.archive = archive
        self.geometry = common.build_geometry(representation, train_archive, device)
        self.latent = torch.from_numpy(archive["visit_latent_standardized_128"].astype(np.float32)).to(device)
        fit = np.polyfit(archive["visit_age_years"].astype(np.float64), archive["visit_age_norm_train"].astype(np.float64), 1)
        self.age_slope, self.age_intercept = float(fit[0]), float(fit[1])
        resolved = read_json(run_dir / "resolved_config.json")["config"]
        checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
        if method == "latent_pca":
            flow = models.DirectC4Flow(
                128,
                int(resolved["model"]["width"]),
                int(resolved["model"]["residual_blocks"]),
                float(resolved["model"].get("dropout", 0.0)),
            ).to(device)
            flow.load_state_dict(checkpoint["flow_state_dict"], strict=True)
            self.flow = flow.eval()
            self.function = None
            self.transport_rk4 = None
            self.substeps = 0
        elif method in {"latent_spiral", "latent_adaptive"}:
            flow = models.DirectC4Flow(
                128,
                int(resolved["model"]["width"]),
                int(resolved["model"]["residual_blocks"]),
                float(resolved["model"].get("dropout", 0.0)),
            ).to(device)
            flow.load_state_dict(checkpoint["flow_state_dict"], strict=True)
            self.flow = flow.eval()
            self.function = None
            self.transport_rk4 = None
            self.substeps = 0
        else:
            function = models.build_ode(resolved).to(device)
            function.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.function = function.eval()
            self.flow = None
            self.transport_rk4 = models.transport_rk4
            self.substeps = max(8, int(resolved["training"]["integration_substeps"]))

    def normalized_age(self, age: np.ndarray) -> torch.Tensor:
        values = np.asarray(age, dtype=np.float64) * self.age_slope + self.age_intercept
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def source_state(self, indices: np.ndarray) -> torch.Tensor:
        return self.latent[torch.as_tensor(indices, dtype=torch.long, device=self.device)]

    def state_at(self, source_state, source_age, target_age, condition):
        source = self.normalized_age(source_age)
        target = self.normalized_age(target_age)
        label = torch.as_tensor(condition, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.flow is not None:
                return self.flow.transport(source_state, source, target, label)
            return self.transport_rk4(self.function, source_state, source, target, label, self.substeps)

    def mesh(self, state) -> np.ndarray:
        with torch.no_grad():
            mesh = self.geometry.vertices(state)
        return mesh.detach().cpu().numpy().astype(np.float64)

    def velocity(self, state, age, condition) -> np.ndarray:
        # Decoder JVP memory is substantially larger than ordinary decoding for
        # the adaptive hierarchy.  Chunking is mathematically sample-separable in
        # eval mode and prevents interval quadrature from materializing one very
        # large autograd graph for the full longitudinal cohort.
        chunk_size = 2 if self.method == "latent_adaptive" else 16
        age = np.asarray(age)
        condition = np.asarray(condition)
        output = []
        for start in range(0, len(state), chunk_size):
            stop = min(start + chunk_size, len(state))
            latent = state[start:stop]
            time = self.normalized_age(age[start:stop])
            label = torch.as_tensor(condition[start:stop], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                latent_velocity = (
                    self.flow.average_velocity(latent, time, time, label)
                    if self.flow is not None
                    else self.function(time, latent, label)
                ) * self.age_slope
            differentiable = latent.detach().requires_grad_(True)
            _, surface_velocity = torch.autograd.functional.jvp(
                self.geometry.vertices,
                differentiable,
                latent_velocity.detach(),
                create_graph=False,
                strict=False,
            )
            output.append(surface_velocity.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(output, axis=0)

    def close(self) -> None:
        del self.geometry, self.latent
        if self.flow is not None:
            del self.flow
        if self.function is not None:
            del self.function
        clear_local_modules(self.script_path)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class LAMMEnsembleBackend(Backend):
    """Three frozen LAMM-N3 latent flows, averaged after surface decoding."""

    def __init__(self, specification: dict[str, Any], device: torch.device):
        self.method = "lamm_n3"
        self.device = device
        self.script_path = COHORT_ROOT / "task3_latent_flow_128_v1" / "scripts"
        registry_path = COHORT_ROOT / "task3_latent_flow_128_v2_lamm" / "configs" / "representations.json"
        registry = read_json(registry_path)
        lamm_model = resolve(registry["lamm_source_root"]) / "scripts" / "lamm_model.py"
        expected_model_hash = registry["source_integrity"]["lamm_model_source_sha256"]
        if file_sha256(lamm_model) != expected_model_hash:
            raise ValueError("Pinned LAMM architecture source changed; refusing ensemble inference")

        # The local train_lamm.py has additional CLI/training options, but the pinned
        # lamm_model.py architecture is byte-identical.  Passing the explicit registry
        # avoids the default registry's builder-file hash check; every decoder and flow
        # checkpoint is then loaded strictly, which rejects architectural drift.
        self.registry = registry
        self.previous_registry = os.environ.get("DEEP3DCOMP_LATENT_FLOW_REGISTRY")
        os.environ["DEEP3DCOMP_LATENT_FLOW_REGISTRY"] = str(registry_path)
        clear_local_modules(self.script_path)
        sys.path.insert(0, str(self.script_path))
        common = importlib.import_module("common")
        models = importlib.import_module("models")
        lamm_scripts = resolve(registry["lamm_source_root"]) / "scripts"
        if str(lamm_scripts) not in sys.path:
            sys.path.insert(0, str(lamm_scripts))
        train_lamm = importlib.import_module("train_lamm")
        original_build_lamm = train_lamm.build

        def compatible_build_lamm(arguments, build_device):
            # These fields were added to the trainer after the frozen N3 registry
            # was written.  They only describe optional latent refinement during
            # training and do not alter the autoencoder architecture at inference.
            for name, value in {
                "semi_amortized": 0,
                "sa_lr": 3e-3,
                "sa_weight": 1.0,
            }.items():
                if not hasattr(arguments, name):
                    setattr(arguments, name, value)
            return original_build_lamm(arguments, build_device)

        train_lamm.build = compatible_build_lamm
        self.archives = []
        self.latents = []
        self.geometries = []
        self.flows = []
        for member in specification["members"]:
            representation = str(member["representation"])
            archive = common.load_archive(representation, "val", registry)
            train_archive = common.load_archive(representation, "train", registry)
            geometry = common.build_geometry(representation, train_archive, device, registry)
            run_dir = resolve(member["run_dir"])
            resolved = read_json(run_dir / "resolved_config.json")["config"]
            checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
            flow = models.DirectC4Flow(
                128,
                int(resolved["model"]["width"]),
                int(resolved["model"]["residual_blocks"]),
                float(resolved["model"].get("dropout", 0.0)),
            ).to(device)
            flow.load_state_dict(checkpoint["flow_state_dict"], strict=True)
            if self.archives and not np.array_equal(
                archive["visit_scan_ids"].astype(str), self.archives[0]["visit_scan_ids"].astype(str)
            ):
                raise ValueError("LAMM N3 members do not share visit ordering")
            self.archives.append(archive)
            self.latents.append(
                torch.from_numpy(archive["visit_latent_standardized_128"].astype(np.float32)).to(device)
            )
            self.geometries.append(geometry)
            self.flows.append(flow.eval())
        fit = np.polyfit(
            self.archives[0]["visit_age_years"].astype(np.float64),
            self.archives[0]["visit_age_norm_train"].astype(np.float64),
            1,
        )
        self.age_slope, self.age_intercept = float(fit[0]), float(fit[1])

    def normalized_age(self, age: np.ndarray) -> torch.Tensor:
        values = np.asarray(age, dtype=np.float64) * self.age_slope + self.age_intercept
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def source_state(self, indices: np.ndarray) -> list[torch.Tensor]:
        selected = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return [latent[selected] for latent in self.latents]

    def state_at(self, source_state, source_age, target_age, condition):
        source = self.normalized_age(source_age)
        target = self.normalized_age(target_age)
        label = torch.as_tensor(condition, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return [
                flow.transport(state, source, target, label)
                for flow, state in zip(self.flows, source_state)
            ]

    def mesh(self, state) -> np.ndarray:
        with torch.no_grad():
            values = [geometry.vertices(value) for geometry, value in zip(self.geometries, state)]
        return torch.stack(values).mean(dim=0).detach().cpu().numpy().astype(np.float64)

    def velocity(self, state, age, condition) -> np.ndarray:
        age = np.asarray(age)
        condition = np.asarray(condition)
        member_values = []
        for flow, geometry, member_state in zip(self.flows, self.geometries, state):
            chunks = []
            for start in range(0, len(member_state), 8):
                stop = min(start + 8, len(member_state))
                latent = member_state[start:stop]
                time = self.normalized_age(age[start:stop])
                label = torch.as_tensor(condition[start:stop], dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    latent_velocity = flow.average_velocity(latent, time, time, label) * self.age_slope
                differentiable = latent.detach().requires_grad_(True)
                _, surface_velocity = torch.autograd.functional.jvp(
                    geometry.vertices,
                    differentiable,
                    latent_velocity.detach(),
                    create_graph=False,
                    strict=False,
                )
                chunks.append(surface_velocity.detach().cpu())
            member_values.append(torch.cat(chunks, dim=0))
        return torch.stack(member_values).mean(dim=0).numpy().astype(np.float64)

    def close(self) -> None:
        del self.flows, self.geometries, self.latents, self.archives
        clear_local_modules(self.script_path)
        if self.previous_registry is None:
            os.environ.pop("DEEP3DCOMP_LATENT_FLOW_REGISTRY", None)
        else:
            os.environ["DEEP3DCOMP_LATENT_FLOW_REGISTRY"] = self.previous_registry
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def backend_specs(config: dict[str, Any]) -> list[tuple[str, str, Any, Path | None, str | None]]:
    direct = {item["key"]: resolve(item["checkpoint"]) for item in config["direct_mesh_methods"]}
    latent = {item["key"]: item for item in config["latent_cocycle_methods"]}
    ode = config["pca_ode_baselines"]
    lamm_representations = ("lamm128", "lamm128_n3_s2", "lamm128_n3_s3")
    lamm_specification = {
        "members": [
            {"representation": representation, "run_dir": run_dir}
            for representation, run_dir in zip(
                lamm_representations, latent["latent_lamm_n3"]["run_dirs"]
            )
        ]
    }
    return [
        ("direct", "mesh_spiral", direct["mesh_spiral"], None, None),
        ("direct", "mesh_adaptive", direct["mesh_adaptive"], None, None),
        ("pca", "latent_pca", resolve(latent["latent_pca"]["run_dir"]), COHORT_ROOT / "task3_latent_flow_128_v1", "pca128"),
        ("pca", "latent_spiral", resolve(latent["latent_spiral"]["run_dir"]), COHORT_ROOT / "task3_latent_flow_128_v1", "spiralnet128"),
        ("pca", "latent_adaptive", resolve(latent["latent_adaptive"]["run_dir"]), COHORT_ROOT / "task3_latent_flow_128_v1", "adaptive128"),
        ("lamm", "lamm_n3", lamm_specification, None, None),
        ("pca", "pca_plain_ode", resolve(ode["methods"]["pca_plain_ode"]["run_dir"]), resolve(ode["task_dir"]), "pca128"),
        ("pca", "pca_brainode", resolve(ode["methods"]["pca_brainode"]["run_dir"]), resolve(ode["task_dir"]), "pca128"),
    ]


def all_scan_velocity(backend: Backend, cache: dict[str, np.ndarray], batch_size: int) -> np.ndarray:
    output = []
    for start in range(0, len(cache["scan_ids"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(cache["scan_ids"])))
        state = backend.source_state(indices)
        output.append(backend.velocity(state, cache["age_years"][indices], cache["label_ad"][indices]))
    return np.concatenate(output, axis=0)


def interval_indices(cache: dict[str, np.ndarray], max_subjects: int | None) -> tuple[np.ndarray, np.ndarray]:
    source, target = [], []
    offsets = cache["subject_visit_offsets"].astype(int)
    count = len(offsets) - 1 if max_subjects is None else min(len(offsets) - 1, max_subjects)
    for subject in range(count):
        first, last = offsets[subject], offsets[subject + 1]
        for index in range(first + 1, last):
            if cache["age_years"][index] > cache["age_years"][first]:
                source.append(first)
                target.append(index)
    return np.asarray(source, dtype=int), np.asarray(target, dtype=int)


def interval_rows(
    backend: Backend,
    cache: dict[str, np.ndarray],
    faces: np.ndarray,
    definition: GeometryDefinition,
    quadrature_points: int,
    max_subjects: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources, targets = interval_indices(cache, max_subjects)
    source_age = cache["age_years"][sources].astype(np.float64)
    target_age = cache["age_years"][targets].astype(np.float64)
    diagnosis = cache["label_ad"][sources].astype(np.float64)
    spans = target_age - source_age
    fractions = np.linspace(0.0, 1.0, quadrature_points)
    integrated = np.zeros((len(sources), 4), dtype=np.float64)
    tangent_rows: list[dict[str, Any]] = []
    base_state = backend.source_state(sources)
    epsilon = 0.02
    for step, fraction in enumerate(fractions):
        ages = source_age + fraction * spans
        state = backend.state_at(base_state, source_age, ages, diagnosis)
        mesh = backend.mesh(state)
        velocity = backend.velocity(state, ages, diagnosis)
        weight = 0.5 if step in {0, len(fractions) - 1} else 1.0
        for index in range(len(sources)):
            clean = remove_rigid_velocity(mesh[index], velocity[index])
            normals = vertex_normals(cache["vertices_mm"][sources[index]], faces)
            normal = np.einsum("ij,ij->i", clean, normals)
            for region in range(4):
                mask = np.ones(len(normal), dtype=bool) if region == 0 else definition.vertex_regions == region - 1
                integrated[index, region] += weight * float(-normal[mask].mean())
        if 0.0 < fraction < 1.0:
            plus_age = np.minimum(ages + epsilon, target_age)
            minus_age = np.maximum(ages - epsilon, source_age)
            plus = backend.mesh(backend.state_at(base_state, source_age, plus_age, diagnosis))
            minus = backend.mesh(backend.state_at(base_state, source_age, minus_age, diagnosis))
            derivative = (plus - minus) / np.maximum((plus_age - minus_age)[:, None, None], 1e-12)
            for index in range(len(sources)):
                local_velocity = remove_rigid_velocity(mesh[index], velocity[index])
                local_derivative = remove_rigid_velocity(mesh[index], derivative[index])
                normals = vertex_normals(mesh[index], faces)
                difference = np.einsum("ij,ij->i", local_velocity - local_derivative, normals)
                tangent_rows.append(
                    {
                        "method": backend.method,
                        "method_label": METHODS[backend.method],
                        "subject_id": str(cache["subject_ids"][sources[index]]),
                        "diagnosis": str(cache["diagnoses"][sources[index]]),
                        "elapsed_years": float(fraction * spans[index]),
                        "path_fraction": float(fraction),
                        "normal_tangent_rmse_mm_per_year": float(np.sqrt(np.mean(difference**2))),
                    }
                )
    integrated /= max(quadrature_points - 1, 1)

    endpoint_state = backend.state_at(base_state, source_age, target_age, diagnosis)
    endpoint_meshes = backend.mesh(endpoint_state)
    rows: list[dict[str, Any]] = []
    labels = ("Whole surface",) + definition.region_labels
    for index, (source, target) in enumerate(zip(sources, targets)):
        baseline = cache["vertices_mm"][source].astype(np.float64)
        observed, _ = kabsch(cache["vertices_mm"][target].astype(np.float64), baseline)
        predicted, _ = kabsch(endpoint_meshes[index], baseline)
        normals = vertex_normals(baseline, faces)
        observed_rate = -np.einsum("ij,ij->i", observed - baseline, normals) / spans[index]
        endpoint_rate = -np.einsum("ij,ij->i", predicted - baseline, normals) / spans[index]
        for region, label in enumerate(labels):
            mask = np.ones(len(normals), dtype=bool) if region == 0 else definition.vertex_regions == region - 1
            observed_value = float(observed_rate[mask].mean())
            integrated_value = float(integrated[index, region])
            endpoint_value = float(endpoint_rate[mask].mean())
            rows.append(
                {
                    "method": backend.method,
                    "method_label": METHODS[backend.method],
                    "subject_id": str(cache["subject_ids"][source]),
                    "diagnosis": str(cache["diagnoses"][source]),
                    "source_scan_id": str(cache["scan_ids"][source]),
                    "target_scan_id": str(cache["scan_ids"][target]),
                    "followup_years": float(spans[index]),
                    "region": label,
                    "observed_interval_inward_mm_per_year": observed_value,
                    "integrated_diagonal_inward_mm_per_year": integrated_value,
                    "endpoint_inward_mm_per_year": endpoint_value,
                    "integrated_abs_error_mm_per_year": abs(integrated_value - observed_value),
                    "endpoint_abs_error_mm_per_year": abs(endpoint_value - observed_value),
                    "diagonal_endpoint_closure_abs_mm_per_year": abs(integrated_value - endpoint_value),
                }
            )
    return rows, tangent_rows


def shape_metrics(
    mesh: np.ndarray,
    baseline: np.ndarray,
    faces: np.ndarray,
    definition: GeometryDefinition,
) -> dict[str, float]:
    aligned, _ = kabsch(mesh, baseline)
    normals = vertex_normals(baseline, faces)
    centerline, radial_unit = baseline_radial_geometry(baseline, definition)
    baseline_radius = np.linalg.norm(baseline - centerline, axis=1)
    radius = np.linalg.norm(aligned - centerline, axis=1)
    volume_change = 100.0 * math.log(max(mesh_volume(aligned, faces), 1e-12) / max(mesh_volume(baseline, faces), 1e-12))
    area_change = 100.0 * math.log(max(face_areas(aligned, faces).sum(), 1e-12) / max(face_areas(baseline, faces).sum(), 1e-12))
    uniform_area = area_change - (2.0 / 3.0) * volume_change
    displacement = -np.einsum("ij,ij->i", aligned - baseline, normals)
    radial_change = radius - baseline_radius
    return {
        "log_volume_change_percent": volume_change,
        "log_surface_area_change_percent": area_change,
        "nonuniform_area_change_percent": uniform_area,
        "mean_inward_displacement_mm": float(displacement.mean()),
        "mean_radial_narrowing_mm": float(-radial_change.mean()),
        "head_side_radial_narrowing_mm": float(-region_mean(radial_change, definition.vertex_regions, definition.region_labels.index("Head-side third"))),
        "tail_side_radial_narrowing_mm": float(-region_mean(radial_change, definition.vertex_regions, definition.region_labels.index("Tail-side third"))),
    }


def paired_rows(
    backend: Backend,
    cache: dict[str, np.ndarray],
    faces: np.ndarray,
    definition: GeometryDefinition,
    years: float,
    step: float,
    max_subjects: int | None,
) -> list[dict[str, Any]]:
    offsets = cache["subject_visit_offsets"].astype(int)
    baselines = np.asarray([offsets[index] for index in range(len(offsets) - 1)], dtype=int)
    eligible = baselines[(cache["age_years"][baselines] >= 70.0) & (cache["age_years"][baselines] + years <= 96.0)]
    if max_subjects is not None:
        eligible = eligible[:max_subjects]
    source_age = cache["age_years"][eligible].astype(np.float64)
    base_state = backend.source_state(eligible)
    grid = np.arange(0.0, years + step * 0.5, step)
    rows: list[dict[str, Any]] = []
    for diagnosis, condition in (("CN", 0.0), ("AD", 1.0)):
        label = np.full(len(eligible), condition, dtype=np.float64)
        for elapsed in grid:
            ages = source_age + elapsed
            state = backend.state_at(base_state, source_age, ages, label)
            meshes = backend.mesh(state)
            velocity = backend.velocity(state, ages, label)
            for local, scan_index in enumerate(eligible):
                baseline = cache["vertices_mm"][scan_index].astype(np.float64)
                values = shape_metrics(meshes[local], baseline, faces, definition)
                clean = remove_rigid_velocity(meshes[local], velocity[local])
                aligned, rotation = kabsch(meshes[local], baseline)
                clean = clean @ rotation
                baseline_normals = vertex_normals(baseline, faces)
                instantaneous = float(-np.einsum("ij,ij->i", clean, baseline_normals).mean())
                rows.append(
                    {
                        "method": backend.method,
                        "method_label": METHODS[backend.method],
                        "subject_id": str(cache["subject_ids"][scan_index]),
                        "source_diagnosis": str(cache["diagnoses"][scan_index]),
                        "condition": diagnosis,
                        "baseline_age_years": float(source_age[local]),
                        "elapsed_years": float(elapsed),
                        "instantaneous_inward_mm_per_year": instantaneous,
                        **values,
                    }
                )
    return rows


def observed_progression_rows(
    cache: dict[str, np.ndarray], faces: np.ndarray, definition: GeometryDefinition, max_subjects: int | None
) -> list[dict[str, Any]]:
    offsets = cache["subject_visit_offsets"].astype(int)
    count = len(offsets) - 1 if max_subjects is None else min(len(offsets) - 1, max_subjects)
    rows: list[dict[str, Any]] = []
    for subject in range(count):
        first, last = offsets[subject], offsets[subject + 1]
        baseline = cache["vertices_mm"][first].astype(np.float64)
        for index in range(first, last):
            elapsed = float(cache["age_years"][index] - cache["age_years"][first])
            values = shape_metrics(cache["vertices_mm"][index].astype(np.float64), baseline, faces, definition)
            observed = cache["velocity_reference_mm_per_year"][index].astype(np.float64)
            normals = vertex_normals(cache["vertices_mm"][index], faces)
            instantaneous = float(-np.einsum("ij,ij->i", observed, normals).mean())
            rows.append(
                {
                    "subject_id": str(cache["subject_ids"][index]),
                    "diagnosis": str(cache["diagnoses"][index]),
                    "baseline_age_years": float(cache["age_years"][first]),
                    "elapsed_years": elapsed,
                    "instantaneous_inward_mm_per_year": instantaneous,
                    **values,
                }
            )
    return rows


def matched_overview(existing: pd.DataFrame) -> pd.DataFrame:
    anchor = set(existing.loc[existing.method.eq("latent_pca"), "scan_id"].astype(str))
    matched = []
    for method, group in existing.groupby("method"):
        if set(group.scan_id.astype(str)) == anchor:
            matched.append(group)
    frame = pd.concat(matched, ignore_index=True)
    summary = subject_equal_summary(frame, ["method", "method_label", "diagnosis"])
    overall = subject_equal_summary(frame, ["method", "method_label"])
    overall["diagnosis"] = "overall"
    return pd.concat([summary, overall[summary.columns]], ignore_index=True)


def plot_matched_summary(
    summary: pd.DataFrame, destination: Path, methods: list[str] | None = None
) -> None:
    methods = DETAILED_METHODS if methods is None else methods
    selected = summary[summary.method.isin(methods)]
    overall = selected[selected.diagnosis.eq("overall")].sort_values("normal_error_to_zero_ratio")
    disease = selected[selected.diagnosis.isin(["CN", "AD"])]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), constrained_layout=True)
    axes[0].barh(overall.method_label, overall.normal_error_to_zero_ratio, color="#4C78A8")
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Normal-velocity RMSE / zero-motion RMSE (lower is better)")
    axes[0].set_title("Matched validation scans only")
    order = overall.method.tolist()
    x = np.arange(len(order))
    width = 0.34
    for offset, diagnosis in ((-width / 2, "CN"), (width / 2, "AD")):
        current = disease[disease.diagnosis.eq(diagnosis)].set_index("method").reindex(order)
        axes[1].bar(x + offset, current.predicted_inward_normal_mm_per_year, width, label=diagnosis, color=COLORS[diagnosis])
    observed = disease.groupby("diagnosis").observed_inward_normal_mm_per_year.mean()
    axes[1].axhline(observed.get("CN", np.nan), color=COLORS["CN"], linestyle="--", linewidth=1.4)
    axes[1].axhline(observed.get("AD", np.nan), color=COLORS["AD"], linestyle="--", linewidth=1.4)
    axes[1].set_xticks(x, [METHODS.get(method, method) for method in order], rotation=35, ha="right")
    axes[1].set_ylabel("Mean inward surface velocity (mm/year)")
    axes[1].set_title("CN–AD separation; dashed lines are observed means")
    axes[1].legend(frameon=False)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_regional(
    summary: pd.DataFrame, destination: Path, methods: list[str] | None = None
) -> None:
    selected_methods = DETAILED_METHODS if methods is None else methods
    current = summary[summary.method.isin(selected_methods)]
    regions = [label for label in current.region.unique() if label != "Whole surface"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), sharey=True, constrained_layout=True)
    for axis, diagnosis in zip(axes, ("CN", "AD")):
        subset = current[current.diagnosis.eq(diagnosis)]
        x = np.arange(len(regions))
        offsets = np.linspace(-0.32, 0.32, len(selected_methods))
        for method, offset in zip(selected_methods, offsets):
            group = subset[subset.method.eq(method)].set_index("region").reindex(regions)
            axis.plot(x + offset, group.predicted_inward_mm_per_year, marker="o", linewidth=1.5, label=METHODS[method], color=COLORS[method])
        observed = subset.groupby("region").observed_inward_mm_per_year.mean().reindex(regions)
        axis.plot(x, observed, marker="D", linestyle="--", color="black", linewidth=2.0, label="Observed fitted change")
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_xticks(x, regions, rotation=18, ha="right")
        axis.set_title(diagnosis)
        axis.set_xlabel("Geometry-defined long-axis region")
    axes[0].set_ylabel("Inward normal velocity (mm/year)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_interval(
    interval: pd.DataFrame, destination: Path, methods: list[str] | None = None
) -> None:
    methods = DETAILED_METHODS if methods is None else methods
    whole = interval[
        interval.region.eq("Whole surface") & interval.method.isin(methods)
    ].copy()
    subject = whole.groupby(["method", "method_label", "subject_id", "diagnosis"], as_index=False).agg(
        observed=("observed_interval_inward_mm_per_year", "mean"),
        integrated=("integrated_diagonal_inward_mm_per_year", "mean"),
        closure=("diagonal_endpoint_closure_abs_mm_per_year", "mean"),
    )
    rows = []
    for (method, label), group in subject.groupby(["method", "method_label"]):
        error = np.sqrt(np.mean((group.integrated - group.observed) ** 2))
        zero = np.sqrt(np.mean(group.observed**2))
        correlation = np.corrcoef(group.integrated, group.observed)[0, 1] if len(group) > 2 else np.nan
        rows.append({"method": method, "method_label": label, "error_ratio": error / max(zero, 1e-12), "correlation": correlation, "closure": group.closure.mean()})
    metrics = pd.DataFrame(rows).sort_values("error_ratio")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1), constrained_layout=True)
    axes[0].barh(metrics.method_label, metrics.error_ratio, color=[COLORS[item] for item in metrics.method])
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Integrated-diagonal RMSE / zero-motion RMSE")
    axes[0].set_title("Does diagonal velocity match observed inter-visit change?")
    for method, group in subject.groupby("method"):
        axes[1].scatter(group.observed, group.integrated, s=25, alpha=0.6, label=METHODS[method], color=COLORS[method])
    limit = np.nanpercentile(np.abs(subject[["observed", "integrated"]].to_numpy()), 97.5)
    axes[1].plot([-limit, limit], [-limit, limit], color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlim(-limit, limit)
    axes[1].set_ylim(-limit, limit)
    axes[1].set_xlabel("Observed interval inward velocity (mm/year)")
    axes[1].set_ylabel("Integrated diagonal velocity (mm/year)")
    axes[1].set_title("Each point is one subject average")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tangent(
    tangent: pd.DataFrame, destination: Path, methods: list[str] | None = None
) -> None:
    methods = DETAILED_METHODS if methods is None else methods
    tangent = tangent[tangent.method.isin(methods)]
    subject = tangent.groupby(["method", "method_label", "subject_id", "path_fraction"], as_index=False).normal_tangent_rmse_mm_per_year.mean()
    curve = subject.groupby(["method", "method_label", "path_fraction"], as_index=False).normal_tangent_rmse_mm_per_year.mean()
    fig, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for method, group in curve.groupby("method"):
        group = group.sort_values("path_fraction")
        axis.plot(group.path_fraction, group.normal_tangent_rmse_mm_per_year, marker="o", label=METHODS[method], color=COLORS[method])
    axis.set_xlabel("Position between baseline and follow-up")
    axis.set_ylabel("Path-derivative vs diagonal-field RMSE (mm/year)")
    axis.set_title("Tangent consistency of the learned transition map")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_paired_velocity(
    paired: pd.DataFrame,
    observed: pd.DataFrame,
    summary: pd.DataFrame,
    destination: Path,
    methods: list[str] | None = None,
) -> None:
    methods = DETAILED_METHODS if methods is None else methods
    columns = 4 if len(methods) > 6 else 3
    rows = int(math.ceil(len(methods) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.8 * columns, 4.25 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    observed_copy = observed.copy()
    observed_copy["time_bin"] = (observed_copy.elapsed_years * 2.0).round() / 2.0
    obs = observed_copy.groupby(["diagnosis", "time_bin", "subject_id"], as_index=False).instantaneous_inward_mm_per_year.mean()
    obs = obs.groupby(["diagnosis", "time_bin"], as_index=False).instantaneous_inward_mm_per_year.mean()
    for axis, method in zip(axes.flat, methods):
        current = paired[paired.method.eq(method)]
        for diagnosis in ("CN", "AD"):
            group = current[current.condition.eq(diagnosis)].groupby("elapsed_years", as_index=False).instantaneous_inward_mm_per_year.mean()
            axis.plot(group.elapsed_years, group.instantaneous_inward_mm_per_year, color=COLORS[diagnosis], linewidth=2.2, label=f"{diagnosis}-conditioned")
            band = summary[(summary.method.eq(method)) & (summary.condition.eq(diagnosis))].sort_values("elapsed_years")
            axis.fill_between(
                band.elapsed_years, band.ci_low, band.ci_high,
                color=COLORS[diagnosis], alpha=0.13, linewidth=0,
            )
            factual = obs[obs.diagnosis.eq(diagnosis)]
            axis.scatter(factual.time_bin, factual.instantaneous_inward_mm_per_year, color=COLORS[diagnosis], marker="x", s=32, alpha=0.8, label=f"Observed {diagnosis}")
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_title(METHODS[method])
        axis.set_xlabel("Years from baseline")
    for row in range(rows):
        axes[row, 0].set_ylabel("Inward diagonal velocity (mm/year)")
    for axis in axes.flat[len(methods):]:
        axis.axis("off")
    handles, labels = axes.flat[len(methods) - 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_shape_progression(
    paired: pd.DataFrame,
    observed: pd.DataFrame,
    summary: pd.DataFrame,
    figure_dir: Path,
    methods: list[str] | None = None,
    filename_suffix: str = "",
) -> None:
    methods = DETAILED_METHODS if methods is None else methods
    columns = 4 if len(methods) > 6 else 3
    rows = int(math.ceil(len(methods) / columns))
    metrics = [
        ("mean_inward_displacement_mm", "Accumulated inward displacement (mm)", "surface_inward_displacement_progression.png"),
        ("mean_radial_narrowing_mm", "Mean radial narrowing (mm)", "surface_radial_narrowing_progression.png"),
        ("nonuniform_area_change_percent", "Area change beyond uniform volume scaling (%)", "surface_nonuniform_area_progression.png"),
    ]
    observed_copy = observed.copy()
    observed_copy["time_bin"] = (observed_copy.elapsed_years * 2.0).round() / 2.0
    for metric, ylabel, filename in metrics:
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(3.8 * columns, 4.25 * rows),
            sharex=True,
            sharey=True,
            constrained_layout=True,
            squeeze=False,
        )
        for axis, method in zip(axes.flat, methods):
            model = paired[paired.method.eq(method)]
            for diagnosis in ("CN", "AD"):
                group = model[model.condition.eq(diagnosis)].groupby("elapsed_years", as_index=False)[metric].mean()
                axis.plot(group.elapsed_years, group[metric], color=COLORS[diagnosis], linewidth=2.2, label=f"{diagnosis}-conditioned")
                band = summary[
                    (summary.method.eq(method))
                    & (summary.condition.eq(diagnosis))
                    & (summary.metric.eq(metric))
                ].sort_values("elapsed_years")
                axis.fill_between(
                    band.elapsed_years, band.ci_low, band.ci_high,
                    color=COLORS[diagnosis], alpha=0.13, linewidth=0,
                )
                factual = observed_copy[observed_copy.diagnosis.eq(diagnosis)].groupby("time_bin", as_index=False)[metric].mean()
                axis.scatter(factual.time_bin, factual[metric], color=COLORS[diagnosis], marker="x", s=26, alpha=0.8, label=f"Observed {diagnosis}")
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.set_title(METHODS[method])
            axis.set_xlabel("Years from baseline")
        for row in range(rows):
            axes[row, 0].set_ylabel(ylabel)
        for axis in axes.flat[len(methods):]:
            axis.axis("off")
        handles, labels = axes.flat[len(methods) - 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        path = Path(filename)
        destination = figure_dir / f"{path.stem}{filename_suffix}{path.suffix}"
        fig.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_surface_maps(
    region_fields: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    faces: np.ndarray,
    destination: Path,
    methods: list[str] | None = None,
) -> None:
    methods = DETAILED_METHODS if methods is None else methods
    template = cache["vertices_mm"].mean(axis=0).astype(np.float64)
    centered = template - template.mean(axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ basis[:2].T
    triangles = coordinates[faces]
    maps: dict[str, np.ndarray] = {}
    diagnoses = cache["diagnoses"].astype(str)
    for method, fields in region_fields.items():
        normals = np.stack([vertex_normals(mesh, faces) for mesh in cache["vertices_mm"]])
        clean_fields = np.stack(
            [remove_rigid_velocity(mesh, field) for mesh, field in zip(cache["vertices_mm"], fields)]
        )
        normal = -np.einsum("nij,nij->ni", clean_fields, normals)
        subject_frame = []
        for subject in np.unique(cache["subject_ids"]):
            selected = cache["subject_ids"].astype(str) == str(subject)
            subject_frame.append((diagnoses[selected][0], normal[selected].mean(axis=0)))
        ad = np.stack([value for diagnosis, value in subject_frame if diagnosis == "AD"]).mean(axis=0)
        cn = np.stack([value for diagnosis, value in subject_frame if diagnosis == "CN"]).mean(axis=0)
        maps[method] = ad - cn
    observed_fields = cache["velocity_reference_mm_per_year"].astype(np.float64)
    normals = np.stack([vertex_normals(mesh, faces) for mesh in cache["vertices_mm"]])
    observed_normal = -np.einsum("nij,nij->ni", observed_fields, normals)
    subject_frame = []
    for subject in np.unique(cache["subject_ids"]):
        selected = cache["subject_ids"].astype(str) == str(subject)
        subject_frame.append((diagnoses[selected][0], observed_normal[selected].mean(axis=0)))
    maps = {"observed": np.stack([v for d, v in subject_frame if d == "AD"]).mean(axis=0) - np.stack([v for d, v in subject_frame if d == "CN"]).mean(axis=0), **maps}
    order = ["observed"] + methods
    limit = max(np.quantile(np.abs(maps[item]), 0.98) for item in order)
    columns = 3
    rows = int(math.ceil(len(order) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.0 * columns, 4.0 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    labels = {"observed": "Observed fitted change", **METHODS}
    image = None
    for axis, method in zip(axes.flat, order):
        face_values = maps[method][faces].mean(axis=1)
        collection = PolyCollection(triangles, array=face_values, cmap="coolwarm", clim=(-limit, limit), edgecolors="none")
        axis.add_collection(collection)
        axis.autoscale_view()
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(labels[method])
        image = collection
    for axis in axes.flat[len(order):]:
        axis.axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, label="AD minus CN inward velocity (mm/year)")
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    config = read_json(args.registry)
    root = resolve(args.output_root) if args.output_root else resolve(config["output_root"])
    table_dir = root / "tables"
    figure_dir = root / "figures"
    outputs = [
        table_dir / "matched_velocity_summary.csv",
        table_dir / "regional_surface_velocity.csv",
        table_dir / "regional_surface_velocity_summary.csv",
        table_dir / "interval_integrated_velocity.csv",
        table_dir / "tangent_consistency.csv",
        table_dir / "paired_condition_progression.csv",
        table_dir / "observed_shape_progression.csv",
    ]
    if not args.smoke and not args.force and any(path.exists() for path in outputs):
        raise FileExistsError("Progression outputs already exist; pass --force to rebuild them")
    if args.quadrature_points < 3 or args.quadrature_points % 2 == 0:
        raise ValueError("--quadrature-points must be an odd integer of at least 3")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable: {device}")
        torch.cuda.set_device(device)

    cache_path = DEFAULT_DATA_ROOT / "cache" / "val_surfaces.npz"
    with np.load(cache_path, allow_pickle=False) as loaded:
        cache = {key: loaded[key].copy() for key in loaded.files}
    statistics = torch.load(DEFAULT_DATA_ROOT / "cache" / "training_statistics.pt", map_location="cpu", weights_only=False)
    faces = np.asarray(statistics["faces"], dtype=np.int64)
    definition = geometry_definition(cache["vertices_mm"].mean(axis=0), faces)
    max_subjects = 4 if args.smoke else args.max_subjects
    bootstrap_samples = min(args.bootstrap_samples, 100) if args.smoke else args.bootstrap_samples

    existing_path = table_dir / "instantaneous_velocity_per_scan.csv"
    existing = pd.read_csv(existing_path, dtype={"scan_id": str, "subject_id": str})
    matched = matched_overview(existing)
    region_rows: list[dict[str, Any]] = []
    interval_output: list[dict[str, Any]] = []
    tangent_output: list[dict[str, Any]] = []
    paired_output: list[dict[str, Any]] = []
    fields_for_maps: dict[str, np.ndarray] = {}

    specs = backend_specs(config)
    if args.smoke:
        specs = [
            item
            for item in specs
            if item[1] in {"latent_spiral", "latent_adaptive", "lamm_n3"}
        ]
    for family, method, run, task, representation in specs:
        print(f"Evaluating {METHODS[method]} ...", flush=True)
        if family == "direct":
            backend: Backend = DirectBackend(method, run, cache, device)
        elif family == "lamm":
            backend = LAMMEnsembleBackend(run, device)
        else:
            backend = PCABackend(method, run, task, device, representation)
        try:
            fields = all_scan_velocity(backend, cache, args.batch_size)
            region_rows.extend(scan_region_rows(method, cache, fields, faces, definition))
            fields_for_maps[method] = fields
            intervals, tangents = interval_rows(
                backend, cache, faces, definition, args.quadrature_points, max_subjects
            )
            interval_output.extend(intervals)
            tangent_output.extend(tangents)
            paired_output.extend(
                paired_rows(
                    backend, cache, faces, definition, args.trajectory_years,
                    args.trajectory_step, max_subjects,
                )
            )
        finally:
            backend.close()

    region = pd.DataFrame(region_rows)
    interval = pd.DataFrame(interval_output)
    tangent = pd.DataFrame(tangent_output)
    paired = pd.DataFrame(paired_output)
    observed = pd.DataFrame(observed_progression_rows(cache, faces, definition, max_subjects))
    region_summary = subject_equal_summary(region, ["method", "method_label", "diagnosis", "region"])
    paired_ci = bootstrap_curve(
        paired,
        "instantaneous_inward_mm_per_year",
        ["method", "method_label", "condition", "elapsed_years"],
        bootstrap_samples,
        1907,
    )
    shape_ci_parts = []
    for offset, metric in enumerate(
        (
            "mean_inward_displacement_mm",
            "mean_radial_narrowing_mm",
            "nonuniform_area_change_percent",
        )
    ):
        current = bootstrap_curve(
            paired,
            metric,
            ["method", "method_label", "condition", "elapsed_years"],
            bootstrap_samples,
            2203 + offset,
        )
        current["metric"] = metric
        shape_ci_parts.append(current)
    shape_ci = pd.concat(shape_ci_parts, ignore_index=True)

    if args.smoke:
        result = {
            "schema_version": 1,
            "status": "smoke_passed",
            "methods": sorted(region.method.unique()),
            "regional_rows": len(region),
            "interval_rows": len(interval),
            "tangent_rows": len(tangent),
            "paired_rows": len(paired),
            "finite": bool(np.isfinite(region.select_dtypes(include=[np.number])).all().all()),
        }
        if not result["finite"]:
            raise ValueError("Smoke output contains non-finite regional values")
        print(json.dumps(result, indent=2))
        return 0

    atomic_csv(matched, table_dir / "matched_velocity_summary.csv")
    atomic_csv(region, table_dir / "regional_surface_velocity.csv")
    atomic_csv(region_summary, table_dir / "regional_surface_velocity_summary.csv")
    atomic_csv(interval, table_dir / "interval_integrated_velocity.csv")
    atomic_csv(tangent, table_dir / "tangent_consistency.csv")
    atomic_csv(paired, table_dir / "paired_condition_progression.csv")
    atomic_csv(paired_ci, table_dir / "paired_condition_velocity_ci.csv")
    atomic_csv(shape_ci, table_dir / "paired_condition_shape_ci.csv")
    atomic_csv(observed, table_dir / "observed_shape_progression.csv")

    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_matched_summary(matched, figure_dir / "velocity_matched_cohort_summary.png")
    plot_regional(region_summary, figure_dir / "velocity_regional_inward.png")
    plot_interval(interval, figure_dir / "velocity_interval_integrated.png")
    plot_tangent(tangent, figure_dir / "velocity_tangent_consistency.png")
    plot_paired_velocity(paired, observed, paired_ci, figure_dir / "velocity_paired_condition_trajectory.png")
    plot_shape_progression(paired, observed, shape_ci, figure_dir)
    plot_surface_maps(fields_for_maps, cache, faces, figure_dir / "velocity_ad_minus_cn_surface_map.png")
    plot_matched_summary(
        matched,
        figure_dir / "velocity_matched_cohort_summary_without_direct_mesh.png",
        WITHOUT_DIRECT_METHODS,
    )
    plot_regional(
        region_summary,
        figure_dir / "velocity_regional_inward_without_direct_mesh.png",
        WITHOUT_DIRECT_METHODS,
    )
    plot_interval(
        interval,
        figure_dir / "velocity_interval_integrated_without_direct_mesh.png",
        WITHOUT_DIRECT_METHODS,
    )
    plot_tangent(
        tangent,
        figure_dir / "velocity_tangent_consistency_without_direct_mesh.png",
        WITHOUT_DIRECT_METHODS,
    )
    plot_paired_velocity(
        paired,
        observed,
        paired_ci,
        figure_dir / "velocity_paired_condition_trajectory_without_direct_mesh.png",
        WITHOUT_DIRECT_METHODS,
    )
    plot_shape_progression(
        paired,
        observed,
        shape_ci,
        figure_dir,
        WITHOUT_DIRECT_METHODS,
        "_without_direct_mesh",
    )
    plot_surface_maps(
        fields_for_maps,
        cache,
        faces,
        figure_dir / "velocity_ad_minus_cn_surface_map_without_direct_mesh.png",
        WITHOUT_DIRECT_METHODS,
    )

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "split": "validation",
        "test_data_loaded": False,
        "models_retrained": False,
        "methods": [item[1] for item in specs],
        "matched_overview_methods": sorted(matched.method.unique()),
        "subjects": int(pd.Series(cache["subject_ids"].astype(str)).nunique()),
        "scans": len(cache["scan_ids"]),
        "trajectory_baseline_age_window_years": [70.0, 91.0],
        "trajectory_horizon_years": args.trajectory_years,
        "quadrature_points": args.quadrature_points,
        "observed_instantaneous_reference": "subject-specific rigid-removed linear surface trajectory selected on validation; estimated, not directly measured",
        "regional_partition": list(definition.region_labels),
        "rigid_velocity_removed": True,
        "primary_claim_guardrail": "biological agreement and tangent consistency are reported separately",
    }
    atomic_json(manifest, root / "surface_progression_manifest.json")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
