from __future__ import annotations

import json
import math
import sys
import html
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
OLD_HELPER_DIR = REPO_ROOT / "examples" / "ADNI_1_L_No_MCI"
PCA_FLOW_SCRIPT_DIR = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_L_No_MCI_large_strict_left"
    / "task3_longitudinal_prediction"
    / "brainode_pca150_qc_stable"
    / "scripts"
)
for import_path in (SCRIPT_DIR, REPO_ROOT, OLD_HELPER_DIR, PCA_FLOW_SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import build_rich_future_mesh_report as rich_report  # noqa: E402
import direct_flow_rich_notebook_helpers as flow_helpers  # noqa: E402
import evaluate_future_mesh_forecasts as fair_eval  # noqa: E402
import evaluate_pca_cocycle_flow as pca_eval  # noqa: E402
from core_brainode_common import TASK_DIR as PCA_TASK_DIR  # noqa: E402
from core_brainode_common import load_config as pca_load_config  # noqa: E402
from core_brainode_common import resolve_repo_path as pca_resolve_repo_path  # noqa: E402


ANALYSIS_DIR = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_L_No_MCI"
    / "brainode_comparison_task3_core_brainode_original"
    / "analysis"
)
NOTEBOOK_DIR = ANALYSIS_DIR / "longitudinal_visual_notebooks"
CACHE_DIR = NOTEBOOK_DIR / "generated_cache"
TABLE_DIR = ANALYSIS_DIR / "unified_longitudinal_visual_report" / "tables"
FUTURE_MESH_DIR = ANALYSIS_DIR / "future_mesh_forecast_comparison"
PCA_FLOW_ANALYSIS_DIR = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_L_No_MCI_large_strict_left"
    / "task3_longitudinal_prediction"
    / "pca150_direct_cocycle_flow_qc_v1"
    / "analysis"
    / "checkpoint_best_val_endpoint_vertex_mae"
)
BEST_BRAINODE_DIR = ANALYSIS_DIR / "best_model_brainode_comparison"
QC_MM_REPORT = (
    Path("/home/jakaria/ADNI/ADNI_1_GO_Large/left_hippocampus_strict_no_mci/reports/prepare_summary.json")
)

QC_MODELS = (
    "qc_brainode_pca150",
    "pca150_direct_cocycle_flow",
    "qc_siren_drop_bad_min2",
    "qc_siren_latent_ode",
)
FLOW_MODELS = (
    "pca150_direct_cocycle_flow",
    "qc_siren_drop_bad_min2",
    "qc_siren_latent_ode",
)
MODEL_LABELS = {
    "qc_brainode_pca150": "BrainODE PCA150",
    "pca150_direct_cocycle_flow": "PCA150 cocycle flow",
    "qc_siren_drop_bad_min2": "SIREN cocycle flow",
    "qc_siren_latent_ode": "SIREN latent ODE",
}
MODEL_COLORS = {
    "qc_brainode_pca150": "#0f766e",
    "pca150_direct_cocycle_flow": "#2563eb",
    "qc_siren_drop_bad_min2": "#c2410c",
    "qc_siren_latent_ode": "#7c3aed",
}
TRANSPORT_LABELS = {
    "brainode_endpoint": "endpoint",
    "direct": "direct",
    "composed": "composed",
    "composed_observed": "composed",
    "composed_fixed_year_steps": "composed",
    "model_no_change": "no-change",
    "direct_from_baseline": "direct",
    "composed_observed_from_baseline": "composed",
}
CONDITION_STYLES = {
    "CN": {"dash": "solid"},
    "AD": {"dash": "dash"},
}
DEFAULT_TRANSPORT_PER_MODEL = {
    "qc_brainode_pca150": "brainode_endpoint",
    "pca150_direct_cocycle_flow": "direct",
    "qc_siren_drop_bad_min2": "direct",
    "qc_siren_latent_ode": "direct",
}
CACHED_SPLIT_TREND_TRANSPORTS = {
    "pca150_direct_cocycle_flow": {
        "transport_method": "direct_from_baseline",
        "transport_label": "direct from baseline",
    },
    "qc_siren_drop_bad_min2": {
        "transport_method": "direct_from_base",
        "transport_label": "direct from baseline",
    },
}
SURFACE_SAMPLE_COUNT = 16000
SMOOTHING_ITERATIONS = 8
SMOOTHING_ALPHA = 0.55
MM3_PER_CM3 = 1000.0
MM2_PER_CM2 = 100.0
VELOCITY_CACHE_VERSION = 2
VELOCITY_CONDITION_MODES = ("observed", "CN", "AD")


def repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def read_csv(path_like: str | Path) -> pd.DataFrame:
    path = repo_path(path_like)
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def read_json(path_like: str | Path) -> Any:
    path = repo_path(path_like)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_")


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(120,120,120,{float(alpha)})"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red},{green},{blue},{float(alpha)})"


def display_label_for_transport(value: str) -> str:
    return TRANSPORT_LABELS.get(str(value), str(value))


def short_error_text(error: Any, max_chars: int = 110) -> str:
    text = " ".join(str(error).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def summarize_failures(
    failures: Sequence[str],
    *,
    max_items: int = 6,
    prefix: str = "Skipped mesh decode",
) -> str:
    clean = [short_error_text(value, max_chars=140) for value in failures if str(value).strip()]
    if not clean:
        return ""
    shown = clean[:max_items]
    lines = [html.escape(item) for item in shown]
    if len(clean) > max_items:
        lines.append(f"... and {len(clean) - max_items} more")
    return f"{html.escape(prefix)}:<br>" + "<br>".join(lines)


def add_failure_annotation(
    fig: go.Figure,
    failures: Sequence[str],
    *,
    prefix: str = "Skipped mesh decode",
    x: float = 0.0,
    y: float = 1.13,
) -> None:
    text = summarize_failures(failures, prefix=prefix)
    if not text:
        return
    fig.add_annotation(
        text=text,
        x=x,
        y=y,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        bordercolor="#b91c1c",
        borderwidth=1,
        bgcolor="rgba(254,242,242,0.96)",
        font={"size": 11, "color": "#7f1d1d"},
    )


def add_note_annotation(
    fig: go.Figure,
    notes: Sequence[str],
    *,
    prefix: str,
    x: float = 0.0,
    y: float = 1.13,
) -> None:
    text = summarize_failures(notes, prefix=prefix)
    if not text:
        return
    fig.add_annotation(
        text=text,
        x=x,
        y=y,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        bordercolor="#64748b",
        borderwidth=1,
        bgcolor="rgba(248,250,252,0.96)",
        font={"size": 11, "color": "#334155"},
    )


def add_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "model" in out.columns:
        out["model_label"] = out["model"].map(lambda value: MODEL_LABELS.get(str(value), str(value)))
        out["model_color"] = out["model"].map(lambda value: MODEL_COLORS.get(str(value), "#444444"))
    if "transport_method" in out.columns:
        out["transport_label"] = out["transport_method"].map(display_label_for_transport)
    return out


def unique_sorted(values: Iterable[Any]) -> list[Any]:
    return sorted(set(values))


def array_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def chunk_slices(length: int, chunk_size: int) -> Iterable[slice]:
    start = 0
    while start < int(length):
        stop = min(start + int(chunk_size), int(length))
        yield slice(start, stop)
        start = stop


def l2_norm(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return np.linalg.norm(values, axis=1)


def safe_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    numer = np.sum(left_values * right_values, axis=1)
    denom = l2_norm(left_values) * l2_norm(right_values)
    return np.divide(numer, denom, out=np.full(numer.shape, np.nan, dtype=np.float64), where=denom > 1.0e-12)


def velocity_train_std(latents: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    split_values = frame["split"].astype(str).to_numpy()
    train = np.asarray(latents, dtype=np.float64)[split_values == "train"]
    if train.size == 0:
        train = np.asarray(latents, dtype=np.float64)
    std = np.nanstd(train, axis=0)
    std = np.asarray(std, dtype=np.float64)
    std[~np.isfinite(std) | (std < 1.0e-8)] = 1.0
    return std


def observed_latent_velocity(
    latents: np.ndarray,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values = np.asarray(latents, dtype=np.float64)
    velocities = np.full(values.shape, np.nan, dtype=np.float64)
    gaps = np.full(values.shape[0], np.nan, dtype=np.float64)
    methods = ["unavailable"] * values.shape[0]
    if frame.empty:
        return velocities, gaps, methods
    ordered_frame = frame.reset_index(drop=True)
    for _, group in ordered_frame.groupby(["split", "subject_id"], sort=False):
        group = group.sort_values(["age_years", "visit_order", "scan_id"])
        positions = group.index.to_numpy(dtype=int)
        if len(positions) < 2:
            continue
        ages = group["age_years"].to_numpy(dtype=np.float64)
        for local_index, position in enumerate(positions):
            if local_index == 0:
                source_local, target_local = 0, 1
                method = "forward"
            elif local_index == len(positions) - 1:
                source_local, target_local = len(positions) - 2, len(positions) - 1
                method = "backward"
            else:
                source_local, target_local = local_index - 1, local_index + 1
                method = "central"
            gap = float(ages[target_local] - ages[source_local])
            if not math.isfinite(gap) or gap <= 1.0e-8:
                continue
            velocities[position] = (values[positions[target_local]] - values[positions[source_local]]) / gap
            gaps[position] = gap
            methods[position] = method
    return velocities, gaps, methods


def mm_scale_summary() -> dict[str, float]:
    audit_path = TABLE_DIR / "whole_dataset_volume_audit_scan_volumes.csv"
    if audit_path.is_file():
        frame = pd.read_csv(audit_path, low_memory=False)
        frame = frame.loc[
            frame["dataset"].astype(str).eq("large_all")
            & frame["method"].astype(str).eq("ground_truth")
        ].copy()
        if {
            "metadata_mesh_volume_mm3",
            "volume",
        }.issubset(frame.columns):
            numer = pd.to_numeric(frame["metadata_mesh_volume_mm3"], errors="coerce")
            denom = pd.to_numeric(frame["volume"], errors="coerce")
            ratio = numer / denom.replace(0.0, np.nan)
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio):
                volume = float(ratio.median())
                distance = float(volume ** (1.0 / 3.0))
                return {
                    "distance_unscale_factor": distance,
                    "volume_unscale_factor": volume,
                    "surface_area_unscale_factor": float(distance**2),
                }

    payload = read_json(QC_MM_REPORT)
    if isinstance(payload, dict):
        left_group = payload.get("groups", {}).get("left", {}) if isinstance(payload.get("groups"), dict) else {}
        distance = finite_float(
            left_group.get("distance_unscale_factor", payload.get("distance_unscale_factor"))
        )
        volume = finite_float(
            left_group.get("volume_unscale_factor", payload.get("volume_unscale_factor"))
        )
        return {
            "distance_unscale_factor": distance,
            "volume_unscale_factor": volume,
            "surface_area_unscale_factor": distance**2 if math.isfinite(distance) else float("nan"),
        }

    return {
        "distance_unscale_factor": float("nan"),
        "volume_unscale_factor": float("nan"),
        "surface_area_unscale_factor": float("nan"),
    }


@dataclass(frozen=True)
class CaseKey:
    split: str
    diagnosis: str
    subject_id: str
    source_scan_id: str
    target_scan_id: str

    @property
    def name(self) -> str:
        return (
            f"{self.split}_{self.diagnosis}_{safe_name(self.subject_id)}_"
            f"{safe_name(self.source_scan_id)}__to__{safe_name(self.target_scan_id)}"
        )


@dataclass
class AnchorSpec:
    split: str
    diagnosis: str
    subject_id: str
    source_scan_id: str
    source_age_years: float
    final_observed_age_years: float
    source_mesh_path: str


class QCPcaFlowSupport:
    def __init__(self, device: str = "auto") -> None:
        self.device = pca_eval.resolve_device(device)
        checkpoint_path = (
            REPO_ROOT
            / "examples"
            / "ADNI_1_L_No_MCI_large_strict_left"
            / "task3_longitudinal_prediction"
            / "pca150_direct_cocycle_flow_qc_v1"
            / "checkpoints"
            / "best_val_endpoint_vertex_mae.pth"
        )
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.flow = pca_eval.make_flow_from_checkpoint(payload, components=150).to(self.device)
        self.flow.load_state_dict(payload["model_state_dict"])
        self.flow.eval()
        config = pca_load_config(PCA_TASK_DIR / "configs" / "core_brainode.json")
        pca_dir = pca_resolve_repo_path(config["task2"]["pca_model_dir"])
        self.mean_flat = np.load(pca_dir / "mean.npy").astype(np.float32)
        self.components = np.load(pca_dir / "components_256.npy").astype(np.float32)[:150]
        self.faces = np.load(pca_dir / "faces.npy").astype(np.int64)
        age_stats = read_json(pca_resolve_repo_path(config["task1"]["age_norm_stats"])) or {}
        self.age_min = finite_float(age_stats.get("age_min_train"))
        self.age_range = finite_float(age_stats.get("age_range_train"))
        self._archives: dict[str, dict[str, np.ndarray]] = {}
        self._latents: dict[str, np.ndarray] = {}
        self._scan_to_index: dict[str, dict[str, int]] = {}

    def normalize_age(self, age_years: float) -> float:
        return (float(age_years) - self.age_min) / self.age_range

    def archive(self, split: str) -> dict[str, np.ndarray]:
        if split not in self._archives:
            archive = pca_eval.load_npz(PCA_TASK_DIR / "dataset" / f"{split}_subject_sequences.npz")
            self._archives[split] = archive
            self._latents[split] = pca_eval.pca_latents(archive, 150).astype(np.float32)
            self._scan_to_index[split] = {
                str(scan_id): index for index, scan_id in enumerate(archive["visit_scan_ids"].tolist())
            }
        return self._archives[split]

    def scan_index(self, split: str, scan_id: str) -> int:
        self.archive(split)
        return int(self._scan_to_index[split][str(scan_id)])

    def source_latent(self, split: str, scan_id: str) -> np.ndarray:
        index = self.scan_index(split, scan_id)
        return self._latents[split][index].astype(np.float32)

    def condition_value(self, split: str, scan_id: str) -> float:
        archive = self.archive(split)
        index = self.scan_index(split, scan_id)
        return float(archive["visit_cognition"][index])

    def scan_latent_table(self, splits: Sequence[str] = ("train", "val", "test")) -> tuple[pd.DataFrame, np.ndarray]:
        rows: list[dict[str, Any]] = []
        latent_blocks: list[np.ndarray] = []
        for split in splits:
            archive = self.archive(str(split))
            latents = self._latents[str(split)].astype(np.float32)
            latent_blocks.append(latents)
            for index in range(latents.shape[0]):
                rows.append(
                    {
                        "split": array_text(archive["visit_splits"][index]),
                        "subject_id": array_text(archive["visit_subject_ids"][index]),
                        "scan_id": array_text(archive["visit_scan_ids"][index]),
                        "diagnosis": array_text(archive["visit_diagnoses"][index]),
                        "label_ad": int(archive["visit_label_ad"][index]),
                        "condition_observed": float(archive["visit_cognition"][index]),
                        "visit_order": int(archive["visit_orders"][index]),
                        "age_years": float(archive["visit_continuous_age_years"][index]),
                        "age_norm": float(archive["visit_continuous_age_norm"][index]),
                    }
                )
        frame = pd.DataFrame(rows)
        latents = np.vstack(latent_blocks).astype(np.float32) if latent_blocks else np.zeros((0, 150), dtype=np.float32)
        return frame.reset_index(drop=True), latents

    @torch.no_grad()
    def latent_velocity_batch(
        self,
        latents: np.ndarray,
        age_norms: np.ndarray,
        condition_values: np.ndarray,
        *,
        batch_size: int = 4096,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for slc in chunk_slices(len(latents), batch_size):
            latent = torch.from_numpy(np.asarray(latents[slc], dtype=np.float32)).to(self.device)
            time = torch.from_numpy(np.asarray(age_norms[slc], dtype=np.float32)).view(-1, 1).to(self.device)
            condition = torch.from_numpy(np.asarray(condition_values[slc], dtype=np.float32)).view(-1, 1).to(self.device)
            if hasattr(self.flow, "instantaneous_velocity_per_year"):
                velocity = self.flow.instantaneous_velocity_per_year(
                    latent,
                    time,
                    condition,
                    self.age_range,
                )
            else:
                velocity = self.flow.average_velocity(latent, time, time, condition) / float(self.age_range)
            outputs.append(velocity.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.zeros_like(latents, dtype=np.float32)

    def mesh_from_latent(self, latent: np.ndarray) -> trimesh.Trimesh:
        return pca_eval.pca_mesh(latent, self.mean_flat, self.components, self.faces)

    @torch.no_grad()
    def predict_pair_mesh(self, row: pd.Series) -> trimesh.Trimesh:
        split = str(row["split"])
        source_scan_id = str(row["source_scan_id"])
        target_scan_id = str(row["target_scan_id"])
        transport = str(row["transport_method"])
        archive = self.archive(split)
        source_index = self.scan_index(split, source_scan_id)
        target_index = self.scan_index(split, target_scan_id)
        source_latent = self._latents[split][source_index].astype(np.float32)
        if transport == "model_no_change":
            return self.mesh_from_latent(source_latent)
        condition = self.condition_value(split, source_scan_id)
        if transport == "direct":
            latent = pca_eval.predict_direct(
                self.flow,
                source_latent,
                float(row["source_age_norm"]),
                float(row["target_age_norm"]),
                condition,
                self.device,
            )
        elif transport == "composed_observed":
            time_slice = slice(source_index, target_index + 1)
            latent = pca_eval.predict_composed_observed(
                self.flow,
                self._latents[split][time_slice].astype(np.float32).copy(),
                archive["visit_continuous_age_norm"][time_slice].astype(np.float32).copy(),
                condition,
                self.device,
            )
        else:
            raise ValueError(f"Unsupported PCA-flow transport {transport!r}.")
        return self.mesh_from_latent(latent)

    @torch.no_grad()
    def forecast_mesh(
        self,
        *,
        split: str,
        source_scan_id: str,
        source_age_years: float,
        target_age_years: float,
        condition_value: float,
        transport: str,
    ) -> trimesh.Trimesh:
        source_latent = self.source_latent(split, source_scan_id)
        if abs(float(target_age_years) - float(source_age_years)) <= 1e-8 or transport == "model_no_change":
            return self.mesh_from_latent(source_latent)
        if transport == "direct":
            latent = pca_eval.predict_direct(
                self.flow,
                source_latent,
                self.normalize_age(source_age_years),
                self.normalize_age(target_age_years),
                float(condition_value),
                self.device,
            )
        elif transport == "composed":
            latent = pca_eval.predict_composed_year_steps(
                self.flow,
                source_latent,
                source_age_years=float(source_age_years),
                target_age_years=float(target_age_years),
                condition=float(condition_value),
                age_min_years=self.age_min,
                age_range_years=self.age_range,
                step_years=0.5,
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported PCA-flow anchor transport {transport!r}.")
        return self.mesh_from_latent(latent)


class QCBrainODESupport:
    def __init__(self, device: str = "auto", checkpoint: str = "best") -> None:
        self.predictor = rich_report.BrainODEPredictor("qc_brainode_pca150", device, checkpoint)
        self.device = self.predictor.device
        age_stats = read_json(
            pca_resolve_repo_path(self.predictor.config["task1"]["age_norm_stats"])
        ) or {}
        self.age_min = finite_float(age_stats.get("age_min_train"))
        self.age_range = finite_float(age_stats.get("age_range_train"))

    def archive(self, split: str) -> dict[str, np.ndarray]:
        return self.predictor.archive(split)

    def scan_index(self, split: str, scan_id: str) -> int:
        archive = self.archive(split)
        scan_ids = [str(value) for value in archive["visit_scan_ids"].tolist()]
        return scan_ids.index(str(scan_id))

    def scan_latent_table(self, splits: Sequence[str] = ("train", "val", "test")) -> tuple[pd.DataFrame, np.ndarray]:
        rows: list[dict[str, Any]] = []
        latent_blocks: list[np.ndarray] = []
        for split in splits:
            archive = self.archive(str(split))
            latents = archive["visit_pca_150"].astype(np.float32)
            latent_blocks.append(latents)
            for index in range(latents.shape[0]):
                rows.append(
                    {
                        "split": array_text(archive["visit_splits"][index]),
                        "subject_id": array_text(archive["visit_subject_ids"][index]),
                        "scan_id": array_text(archive["visit_scan_ids"][index]),
                        "diagnosis": array_text(archive["visit_diagnoses"][index]),
                        "label_ad": int(archive["visit_label_ad"][index]),
                        "condition_observed": float(archive["visit_cognition"][index]),
                        "visit_order": int(archive["visit_orders"][index]),
                        "age_years": float(archive["visit_continuous_age_years"][index]),
                        "age_norm": float(archive["visit_continuous_age_norm"][index]),
                    }
                )
        frame = pd.DataFrame(rows)
        latents = np.vstack(latent_blocks).astype(np.float32) if latent_blocks else np.zeros((0, 150), dtype=np.float32)
        return frame.reset_index(drop=True), latents

    @torch.no_grad()
    def latent_velocity_batch(
        self,
        latents: np.ndarray,
        age_norms: np.ndarray,
        condition_values: np.ndarray,
        *,
        batch_size: int = 4096,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for slc in chunk_slices(len(latents), batch_size):
            latent = torch.from_numpy(np.asarray(latents[slc], dtype=np.float32)).to(self.device)
            time = torch.from_numpy(np.asarray(age_norms[slc], dtype=np.float32)).to(self.device)
            condition = torch.from_numpy(np.asarray(condition_values[slc], dtype=np.float32)).to(self.device)
            velocity = self.predictor.model(time, latent, condition) / float(self.age_range)
            outputs.append(velocity.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.zeros_like(latents, dtype=np.float32)

    @torch.no_grad()
    def forecast_mesh(
        self,
        *,
        split: str,
        source_scan_id: str,
        target_age_years: float,
        condition_value: float,
    ) -> trimesh.Trimesh:
        archive = self.archive(split)
        source_index = self.scan_index(split, source_scan_id)
        source_age_norm = float(archive["visit_continuous_age_norm"][source_index])
        target_age_norm = float((float(target_age_years) - self.age_min) / self.age_range)
        source_code = archive["visit_pca_150"][source_index].astype(np.float32)
        if abs(source_age_norm - target_age_norm) <= 1e-8:
            endpoint = source_code
        else:
            times = torch.tensor(
                [[source_age_norm, target_age_norm]],
                dtype=torch.float32,
                device=self.device,
            )
            initial_state = torch.from_numpy(source_code).view(1, -1).to(self.device)
            condition = torch.tensor([float(condition_value)], dtype=torch.float32, device=self.device)
            if bool(self.predictor.full_config.get("use_autoregressive_rollout", False)):
                prediction, _ = self.predictor.brainode_model.integrate_autoregressive_rk4(
                    func=self.predictor.model,
                    initial_state=initial_state,
                    times=times,
                    initial_condition=condition,
                    substeps=int(self.predictor.training_config["integration_substeps"]),
                )
            else:
                prediction = self.predictor.brainode_model.integrate_sequence_rk4(
                    func=self.predictor.model,
                    initial_state=initial_state,
                    times=times,
                    condition=condition,
                    substeps=int(self.predictor.training_config["integration_substeps"]),
                )
            endpoint = prediction[0, -1, :].detach().cpu().numpy().astype(np.float32)
        return fair_eval.pca_mesh(
            endpoint,
            self.predictor.mean_flat,
            self.predictor.components,
            self.predictor.faces,
        )


class QCSirenForecastSupport:
    def __init__(self, model_name: str, device: str = "auto", checkpoint: str = "best") -> None:
        if model_name not in {"qc_siren_drop_bad_min2", "qc_siren_latent_ode"}:
            raise ValueError(f"Unsupported QC SIREN model {model_name!r}.")
        self.model_name = model_name
        self.bundle = flow_helpers.load_bundle(
            repo_path(fair_eval.FLOW_MODELS[model_name]["experiment_dir"]),
            checkpoint=checkpoint,
            device=device,
        )

    def source_row(self, split: str, scan_id: str) -> pd.Series:
        metadata = self.bundle.contract.metadata.copy()
        mask = (
            metadata["split"].astype(str).eq(str(split))
            & metadata["scan_id"].astype(str).eq(str(scan_id))
        )
        matches = metadata.loc[mask].copy()
        if matches.empty:
            raise KeyError(f"Missing {scan_id!r} in split {split!r} for {self.model_name}.")
        return matches.iloc[0]

    def scan_latent_table(self, splits: Sequence[str] = ("train", "val", "test")) -> tuple[pd.DataFrame, np.ndarray]:
        metadata = self.bundle.contract.metadata.copy()
        metadata = metadata.loc[metadata["split"].astype(str).isin([str(value) for value in splits])].copy()
        metadata = metadata.sort_values(["split", "subject_id", "continuous_age_years", "visit_order", "scan_id"]).reset_index(drop=True)
        rows: list[dict[str, Any]] = []
        latents: list[np.ndarray] = []
        for row in metadata.itertuples(index=False):
            split = str(getattr(row, "split"))
            scan_id = str(getattr(row, "scan_id"))
            latent = (
                self.bundle.contract.latent_maps[split][scan_id]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
                .reshape(-1)
            )
            latents.append(latent)
            rows.append(
                {
                    "split": split,
                    "subject_id": str(getattr(row, "subject_id")),
                    "scan_id": scan_id,
                    "diagnosis": str(getattr(row, "diagnosis")),
                    "label_ad": int(getattr(row, "label_ad")),
                    "condition_observed": float(getattr(row, "label_ad")),
                    "visit_order": int(getattr(row, "visit_order")),
                    "age_years": float(getattr(row, "continuous_age_years")),
                    "age_norm": float(getattr(row, "continuous_age_norm")),
                }
            )
        frame = pd.DataFrame(rows)
        latent_matrix = np.vstack(latents).astype(np.float32) if latents else np.zeros((0, 256), dtype=np.float32)
        return frame.reset_index(drop=True), latent_matrix

    @torch.no_grad()
    def latent_velocity_batch(
        self,
        latents: np.ndarray,
        age_norms: np.ndarray,
        condition_values: np.ndarray,
        *,
        batch_size: int = 4096,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        age_range = flow_helpers.age_range_years(self.bundle)
        for slc in chunk_slices(len(latents), batch_size):
            latent = torch.from_numpy(np.asarray(latents[slc], dtype=np.float32)).to(self.bundle.device)
            time = torch.from_numpy(np.asarray(age_norms[slc], dtype=np.float32)).view(-1, 1).to(self.bundle.device)
            condition = torch.from_numpy(np.asarray(condition_values[slc], dtype=np.float32)).view(-1, 1).to(self.bundle.device)
            if hasattr(self.bundle.flow, "instantaneous_velocity_per_year"):
                velocity = self.bundle.flow.instantaneous_velocity_per_year(
                    latent,
                    time,
                    condition,
                    age_range,
                )
            else:
                velocity = self.bundle.flow.average_velocity(latent, time, time, condition) / float(age_range)
            outputs.append(velocity.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.zeros_like(latents, dtype=np.float32)

    @torch.no_grad()
    def forecast_mesh(
        self,
        *,
        split: str,
        source_scan_id: str,
        source_age_years: float,
        target_age_years: float,
        condition_value: float,
        transport: str,
        mesh_resolution: int = 72,
        mesh_max_batch: int = 2**18,
    ) -> trimesh.Trimesh:
        source_latent = flow_helpers._latent_tensor(self.bundle, split, source_scan_id)
        if transport == "model_no_change" or abs(float(target_age_years) - float(source_age_years)) <= 1e-8:
            mesh, error = flow_helpers.try_decode_mesh(
                self.bundle,
                source_latent,
                resolution=mesh_resolution,
                max_batch=mesh_max_batch,
            )
            if mesh is None:
                raise RuntimeError(error or f"{self.model_name} source decode failed.")
            return mesh
        if transport == "direct":
            latent = flow_helpers.transport_direct(
                self.bundle,
                source_latent,
                flow_helpers._time_tensor(
                    self.bundle,
                    flow_helpers.normalize_age_years(self.bundle, float(source_age_years)),
                ),
                flow_helpers._time_tensor(
                    self.bundle,
                    flow_helpers.normalize_age_years(self.bundle, float(target_age_years)),
                ),
                flow_helpers._condition_tensor(self.bundle, int(condition_value)),
            )
        elif transport == "composed":
            latent = flow_helpers.transport_composed_fixed_step(
                self.bundle,
                source_latent,
                start_age_years=float(source_age_years),
                end_age_years=float(target_age_years),
                label_ad=int(condition_value),
                step_years=0.5,
            )
        else:
            raise ValueError(f"Unsupported transport {transport!r} for {self.model_name}.")
        mesh, error = flow_helpers.try_decode_mesh(
            self.bundle,
            latent,
            resolution=mesh_resolution,
            max_batch=mesh_max_batch,
        )
        if mesh is None:
            raise RuntimeError(error or f"{self.model_name} {transport} decode failed.")
        return mesh


@lru_cache(maxsize=4096)
def load_mesh_cached(path_like: str | Path) -> trimesh.Trimesh:
    return fair_eval.load_mesh(str(path_like))


def mesh_volume(mesh_like: str | Path | trimesh.Trimesh) -> float:
    mesh = load_mesh_cached(mesh_like) if not isinstance(mesh_like, trimesh.Trimesh) else mesh_like
    return float(abs(mesh.volume))


def mesh_area(mesh_like: str | Path | trimesh.Trimesh) -> float:
    mesh = load_mesh_cached(mesh_like) if not isinstance(mesh_like, trimesh.Trimesh) else mesh_like
    return float(mesh.area)


def mesh_for_solid_display(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    display_mesh = mesh.copy()
    notes: list[str] = []
    before_vertices = int(len(display_mesh.vertices))
    before_faces = int(len(display_mesh.faces))
    was_watertight = bool(getattr(display_mesh, "is_watertight", False))

    try:
        display_mesh.process(validate=True)
    except Exception as exc:
        notes.append(f"display cleanup skipped: {short_error_text(repr(exc), max_chars=90)}")

    try:
        filled = bool(display_mesh.fill_holes())
    except Exception:
        filled = False
    if filled:
        notes.append("filled small visual holes")

    try:
        display_mesh.fix_normals()
    except Exception:
        pass

    try:
        display_mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    after_vertices = int(len(display_mesh.vertices))
    after_faces = int(len(display_mesh.faces))
    if (after_vertices, after_faces) != (before_vertices, before_faces):
        notes.insert(
            0,
            (
                "cleaned display mesh "
                f"{before_vertices}v/{before_faces}f -> {after_vertices}v/{after_faces}f"
            ),
        )
    is_watertight = bool(getattr(display_mesh, "is_watertight", False))
    if is_watertight and not was_watertight:
        notes.append("watertight after display repair")
    elif not is_watertight:
        notes.append("mesh remains non-watertight; metrics still use original evaluation outputs")
    return display_mesh, notes


def _prepare_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def smooth_vertex_signal(values: np.ndarray, faces: np.ndarray, iterations: int, alpha: float) -> np.ndarray:
    edges = _prepare_edges(np.asarray(faces, dtype=np.int64))
    left = edges[:, 0]
    right = edges[:, 1]
    current = np.asarray(values, dtype=np.float64).copy()
    for _ in range(max(int(iterations), 0)):
        neighbor_sum = np.zeros_like(current)
        neighbor_count = np.zeros_like(current)
        np.add.at(neighbor_sum, left, current[right])
        np.add.at(neighbor_sum, right, current[left])
        np.add.at(neighbor_count, left, 1.0)
        np.add.at(neighbor_count, right, 1.0)
        averaged = np.divide(
            neighbor_sum,
            np.maximum(neighbor_count, 1.0),
            out=current.copy(),
            where=neighbor_count > 0.0,
        )
        current = (1.0 - float(alpha)) * current + float(alpha) * averaged
    return current


def aligned_surface_shift(
    reference_mesh_like: str | Path | trimesh.Trimesh,
    moving_mesh_like: str | Path | trimesh.Trimesh,
    *,
    sample_count: int = SURFACE_SAMPLE_COUNT,
    smoothing_iterations: int = SMOOTHING_ITERATIONS,
    smoothing_alpha: float = SMOOTHING_ALPHA,
) -> dict[str, Any]:
    reference_mesh = (
        load_mesh_cached(reference_mesh_like)
        if not isinstance(reference_mesh_like, trimesh.Trimesh)
        else reference_mesh_like
    )
    moving_mesh = (
        load_mesh_cached(moving_mesh_like)
        if not isinstance(moving_mesh_like, trimesh.Trimesh)
        else moving_mesh_like
    )
    reference_points = np.asarray(reference_mesh.sample(int(sample_count)), dtype=np.float64)
    moving_points = np.asarray(moving_mesh.sample(int(sample_count)), dtype=np.float64)
    initial = np.eye(4, dtype=np.float64)
    initial[:3, 3] = reference_points.mean(axis=0) - moving_points.mean(axis=0)
    try:
        matrix, _, icp_cost = trimesh.registration.icp(
            moving_points,
            reference_points,
            initial=initial,
            threshold=1e-6,
            max_iterations=60,
            reflection=False,
            scale=False,
        )
    except Exception:
        matrix = initial
        icp_cost = float("nan")
    aligned_vertices = trimesh.transform_points(np.asarray(moving_mesh.vertices), matrix)
    aligned_mesh = trimesh.Trimesh(
        vertices=aligned_vertices,
        faces=np.asarray(moving_mesh.faces),
        process=False,
    )
    tree = cKDTree(reference_points)
    distances, _ = tree.query(aligned_vertices, k=1)
    smoothed = smooth_vertex_signal(
        distances.astype(np.float64),
        np.asarray(aligned_mesh.faces),
        iterations=smoothing_iterations,
        alpha=smoothing_alpha,
    )
    return {
        "aligned_mesh": aligned_mesh,
        "raw_shift": distances.astype(np.float64),
        "surface_shift": smoothed.astype(np.float64),
        "mean_shift": float(np.mean(smoothed)),
        "p90_shift": float(np.quantile(smoothed, 0.90)),
        "p95_shift": float(np.quantile(smoothed, 0.95)),
        "max_shift": float(np.max(smoothed)),
        "icp_cost": float(icp_cost),
    }


def _mesh3d_trace(
    mesh: trimesh.Trimesh,
    *,
    name: str,
    color: str,
    opacity: float = 1.0,
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
        color=color,
        opacity=opacity,
        flatshading=True,
        lighting={
            "ambient": 0.62,
            "diffuse": 0.78,
            "specular": 0.18,
            "roughness": 0.72,
            "fresnel": 0.02,
        },
        lightposition={"x": 180, "y": 140, "z": 120},
        name=name,
        showlegend=False,
        hovertemplate=f"{name}<extra></extra>",
    )


def _heat_trace(
    mesh: trimesh.Trimesh,
    values: np.ndarray,
    *,
    cmin: float,
    cmax: float,
    name: str,
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
        intensitymode="vertex",
        colorscale="Viridis",
        cmin=cmin,
        cmax=cmax,
        opacity=1.0,
        flatshading=False,
        lighting={
            "ambient": 0.58,
            "diffuse": 0.82,
            "specular": 0.15,
            "roughness": 0.7,
            "fresnel": 0.02,
        },
        lightposition={"x": 170, "y": 130, "z": 110},
        showscale=show_scale,
        colorbar={"title": "shift (mm)"} if show_scale else None,
        name=name,
        showlegend=False,
        hovertemplate="shift=%{intensity:.6f}<extra></extra>",
    )


class LongitudinalVisualizationContext:
    def __init__(self, device: str = "auto") -> None:
        NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.mm_scale = mm_scale_summary()
        self.distance_scale_mm = self.mm_scale["distance_unscale_factor"]
        self.volume_scale_mm3 = self.mm_scale["volume_unscale_factor"]
        self.surface_area_scale_mm2 = self.mm_scale["surface_area_unscale_factor"]
        self._pca_support: QCPcaFlowSupport | None = None
        self._brainode_support: QCBrainODESupport | None = None
        self._flow_support: dict[str, QCSirenForecastSupport] = {}
        self.pair_metrics = self._load_pair_metrics()
        self.scan_volumes = self._load_scan_volumes()
        self.brainode_future_volume = self._load_brainode_future_volume()
        self.selected_volume_trends = self._load_selected_volume_trends()

    def _load_pair_metrics(self) -> pd.DataFrame:
        fair = read_csv(FUTURE_MESH_DIR / "future_mesh_per_pair.csv")
        fair = fair.loc[
            fair["dataset"].astype(str).eq("qc_large")
            & fair["model"].astype(str).isin(QC_MODELS)
        ].copy()
        pca = read_csv(PCA_FLOW_ANALYSIS_DIR / "pca_flow_per_pair.csv")
        pca = pca.loc[pca["dataset"].astype(str).eq("qc_large")].copy()
        combined = pd.concat([fair, pca], ignore_index=True, sort=False)
        numeric_cols = [
            "source_age_years",
            "target_age_years",
            "gap_years",
            "chamfer_l2_squared",
            "assd",
            "hd95",
            "volume_abs_error",
            "volume_relative_error",
            "surface_area_abs_error",
            "surface_area_relative_error",
            "endpoint_vertex_mae",
            "endpoint_vertex_rmse",
        ]
        for column in numeric_cols:
            if column in combined.columns:
                combined[column] = pd.to_numeric(combined[column], errors="coerce")
        return add_display_columns(combined)

    def _load_scan_volumes(self) -> pd.DataFrame:
        frame = read_csv(TABLE_DIR / "whole_dataset_volume_audit_scan_volumes.csv")
        if frame.empty:
            return frame
        mask = (
            frame["dataset"].astype(str).eq("large_all")
            & frame["model"].astype(str).eq("volume_audit_ground_truth")
            & frame["method"].astype(str).eq("ground_truth")
        )
        frame = frame.loc[mask].copy()
        numeric_cols = [
            "age_years",
            "visit_order",
            "volume",
            "metadata_mesh_volume_mm3",
            "mask_volume_mm3",
        ]
        for column in numeric_cols:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["subject_id"] = frame["subject_id"].astype(str)
        frame["split"] = frame["split"].astype(str)
        frame["diagnosis"] = frame["diagnosis"].astype(str)
        return frame

    def _load_brainode_future_volume(self) -> pd.DataFrame:
        frame = read_csv(BEST_BRAINODE_DIR / "brainode_conditional_future_volume.csv")
        if frame.empty:
            return frame
        frame = frame.loc[frame["dataset"].astype(str).eq("qc_large")].copy()
        for column in (
            "source_age_years",
            "target_age_years",
            "years_from_source",
            "predicted_volume",
            "source_pca_volume",
            "last_observed_pca_volume",
            "final_observed_age_years",
        ):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["subject_id"] = frame["subject_id"].astype(str)
        return frame

    def _load_selected_volume_trends(self) -> pd.DataFrame:
        frame = read_csv(TABLE_DIR / "selected_volume_trends.csv")
        if frame.empty:
            return frame
        frame = frame.loc[
            frame["dataset"].astype(str).eq("qc_large")
            & frame["model"].astype(str).isin(QC_MODELS)
        ].copy()
        numeric_cols = [
            "age_years",
            "visit_order",
            "predicted_volume",
            "observed_volume",
            "predicted_surface_area",
            "observed_surface_area",
            "volume_abs_error",
            "surface_area_abs_error",
        ]
        for column in numeric_cols:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["subject_id"] = frame["subject_id"].astype(str)
        return add_display_columns(frame)

    @property
    def pca_support(self) -> QCPcaFlowSupport:
        if self._pca_support is None:
            self._pca_support = QCPcaFlowSupport(device=self.device)
        return self._pca_support

    @property
    def brainode_support(self) -> QCBrainODESupport:
        if self._brainode_support is None:
            self._brainode_support = QCBrainODESupport(device=self.device)
        return self._brainode_support

    def flow_support(self, model_name: str) -> QCSirenForecastSupport:
        if model_name not in self._flow_support:
            self._flow_support[model_name] = QCSirenForecastSupport(model_name, device=self.device)
        return self._flow_support[model_name]

    def available_models(self) -> list[str]:
        return list(QC_MODELS)

    def _velocity_cache_paths(self) -> dict[str, Path]:
        base = CACHE_DIR / "instantaneous_latent_velocity"
        return {
            "per_scan": base / "instantaneous_latent_velocity_per_scan.csv",
            "summary": base / "instantaneous_latent_velocity_summary.csv",
            "condition_gap": base / "instantaneous_velocity_condition_gap.csv",
        }

    def _velocity_support(self, model: str) -> tuple[Any, str, str]:
        if model == "qc_brainode_pca150":
            return self.brainode_support, "pca150", "ode_vector_field"
        if model == "pca150_direct_cocycle_flow":
            return self.pca_support, "pca150", "cocycle_diagonal_generator"
        if model == "qc_siren_latent_ode":
            return self.flow_support(model), "siren256", "ode_vector_field"
        if model == "qc_siren_drop_bad_min2":
            return self.flow_support(model), "siren256", "cocycle_diagonal_generator"
        raise ValueError(f"Unsupported velocity model {model!r}.")

    def _condition_values_for_mode(self, scan_frame: pd.DataFrame, mode: str) -> np.ndarray:
        if mode == "observed":
            return pd.to_numeric(scan_frame["condition_observed"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if mode == "CN":
            return np.zeros(len(scan_frame), dtype=np.float32)
        if mode == "AD":
            return np.ones(len(scan_frame), dtype=np.float32)
        raise ValueError(f"Unsupported velocity condition mode {mode!r}.")

    def _build_instantaneous_velocity_for_model(self, model: str) -> pd.DataFrame:
        support, latent_family, generator_kind = self._velocity_support(model)
        print(f"  building instantaneous velocity: {MODEL_LABELS[model]}")
        scan_frame, latents = support.scan_latent_table()
        scan_frame = scan_frame.reset_index(drop=True)
        if scan_frame.empty:
            return pd.DataFrame()
        real_velocity, real_gap_years, real_method = observed_latent_velocity(latents, scan_frame)
        train_std = velocity_train_std(latents, scan_frame)
        real_velocity_z = real_velocity / train_std.reshape(1, -1)
        real_l2 = l2_norm(real_velocity)
        real_z_l2 = l2_norm(real_velocity_z)
        rows: list[dict[str, Any]] = []
        age_norms = pd.to_numeric(scan_frame["age_norm"], errors="coerce").to_numpy(dtype=np.float32)

        for condition_mode in VELOCITY_CONDITION_MODES:
            condition_values = self._condition_values_for_mode(scan_frame, condition_mode)
            model_velocity = support.latent_velocity_batch(
                latents,
                age_norms,
                condition_values,
            ).astype(np.float64)
            model_velocity_z = model_velocity / train_std.reshape(1, -1)
            error = model_velocity - real_velocity
            error_z = error / train_std.reshape(1, -1)
            model_l2 = l2_norm(model_velocity)
            model_z_l2 = l2_norm(model_velocity_z)
            error_l2 = l2_norm(error)
            error_z_l2 = l2_norm(error_z)
            cosine = safe_cosine(model_velocity, real_velocity)
            ratio = np.divide(
                model_l2,
                real_l2,
                out=np.full(model_l2.shape, np.nan, dtype=np.float64),
                where=real_l2 > 1.0e-12,
            )

            for index, scan in scan_frame.iterrows():
                rows.append(
                    {
                        "cache_version": VELOCITY_CACHE_VERSION,
                        "dataset": "qc_large",
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "model_priority": int(QC_MODELS.index(model)),
                        "latent_family": latent_family,
                        "generator_kind": generator_kind,
                        "split": str(scan["split"]),
                        "subject_id": str(scan["subject_id"]),
                        "scan_id": str(scan["scan_id"]),
                        "diagnosis": str(scan["diagnosis"]),
                        "label_ad": int(scan["label_ad"]),
                        "age_years": finite_float(scan["age_years"]),
                        "age_norm": finite_float(scan["age_norm"]),
                        "visit_order": int(scan["visit_order"]),
                        "condition_eval": condition_mode,
                        "condition_value": float(condition_values[index]),
                        "latent_dim": int(latents.shape[1]),
                        "real_velocity_method": real_method[index],
                        "real_velocity_gap_years": float(real_gap_years[index]),
                        "real_velocity_l2_per_year": float(real_l2[index]),
                        "real_velocity_z_l2_per_year": float(real_z_l2[index]),
                        "model_velocity_l2_per_year": float(model_l2[index]),
                        "model_velocity_z_l2_per_year": float(model_z_l2[index]),
                        "velocity_l2_error_per_year": float(error_l2[index]),
                        "velocity_z_l2_error_per_year": float(error_z_l2[index]),
                        "velocity_cosine_real_vs_model": float(cosine[index]),
                        "model_to_real_speed_ratio": float(ratio[index]),
                    }
                )
        return pd.DataFrame(rows)

    def summarize_instantaneous_velocity(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        summary = (
            frame.groupby(
                [
                    "dataset",
                    "model",
                    "model_label",
                    "model_priority",
                    "latent_family",
                    "generator_kind",
                    "split",
                    "diagnosis",
                    "condition_eval",
                ],
                sort=False,
            )
            .agg(
                rows=("scan_id", "size"),
                subjects=("subject_id", "nunique"),
                scans=("scan_id", "nunique"),
                real_velocity_l2_mean=("real_velocity_l2_per_year", "mean"),
                real_velocity_l2_median=("real_velocity_l2_per_year", "median"),
                real_velocity_z_l2_mean=("real_velocity_z_l2_per_year", "mean"),
                real_velocity_z_l2_median=("real_velocity_z_l2_per_year", "median"),
                model_velocity_l2_mean=("model_velocity_l2_per_year", "mean"),
                model_velocity_l2_median=("model_velocity_l2_per_year", "median"),
                model_velocity_z_l2_mean=("model_velocity_z_l2_per_year", "mean"),
                model_velocity_z_l2_median=("model_velocity_z_l2_per_year", "median"),
                velocity_l2_error_mean=("velocity_l2_error_per_year", "mean"),
                velocity_l2_error_median=("velocity_l2_error_per_year", "median"),
                velocity_z_l2_error_mean=("velocity_z_l2_error_per_year", "mean"),
                velocity_z_l2_error_median=("velocity_z_l2_error_per_year", "median"),
                velocity_cosine_mean=("velocity_cosine_real_vs_model", "mean"),
                velocity_cosine_median=("velocity_cosine_real_vs_model", "median"),
                model_to_real_speed_ratio_median=("model_to_real_speed_ratio", "median"),
            )
            .reset_index()
        )
        split_priority = {"train": 0, "val": 1, "test": 2}
        condition_priority = {"observed": 0, "CN": 1, "AD": 2}
        summary["split_priority"] = summary["split"].map(lambda value: split_priority.get(str(value), 99))
        summary["condition_priority"] = summary["condition_eval"].map(
            lambda value: condition_priority.get(str(value), 99)
        )
        return summary.sort_values(
            ["split_priority", "diagnosis", "model_priority", "condition_priority"]
        ).reset_index(drop=True)

    def condition_gap_instantaneous_velocity(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        key_cols = [
            "dataset",
            "model",
            "model_label",
            "model_priority",
            "latent_family",
            "generator_kind",
            "split",
            "subject_id",
            "scan_id",
            "diagnosis",
            "label_ad",
            "age_years",
            "age_norm",
            "visit_order",
        ]
        value_cols = [
            "model_velocity_l2_per_year",
            "model_velocity_z_l2_per_year",
        ]
        cn = frame.loc[frame["condition_eval"].astype(str).eq("CN"), key_cols + value_cols].copy()
        ad = frame.loc[frame["condition_eval"].astype(str).eq("AD"), key_cols + value_cols].copy()
        merged = cn.merge(ad, on=key_cols, suffixes=("_cn_condition", "_ad_condition"), how="inner")
        if merged.empty:
            return merged
        merged["ad_minus_cn_velocity_l2_per_year"] = (
            merged["model_velocity_l2_per_year_ad_condition"]
            - merged["model_velocity_l2_per_year_cn_condition"]
        )
        merged["ad_minus_cn_velocity_z_l2_per_year"] = (
            merged["model_velocity_z_l2_per_year_ad_condition"]
            - merged["model_velocity_z_l2_per_year_cn_condition"]
        )
        merged["ad_over_cn_velocity_z_ratio"] = np.divide(
            merged["model_velocity_z_l2_per_year_ad_condition"],
            merged["model_velocity_z_l2_per_year_cn_condition"],
            out=np.full(len(merged), np.nan, dtype=np.float64),
            where=merged["model_velocity_z_l2_per_year_cn_condition"].to_numpy(dtype=float) > 1.0e-12,
        )
        return merged.sort_values(["split", "diagnosis", "model_priority", "subject_id", "age_years"]).reset_index(drop=True)

    def build_instantaneous_velocity_tables(
        self,
        *,
        force_recompute: bool = False,
    ) -> dict[str, pd.DataFrame]:
        paths = self._velocity_cache_paths()
        required = {
            "cache_version",
            "model",
            "split",
            "diagnosis",
            "condition_eval",
            "model_velocity_z_l2_per_year",
            "real_velocity_z_l2_per_year",
            "velocity_cosine_real_vs_model",
        }
        if not force_recompute and all(path.is_file() for path in paths.values()):
            per_scan = pd.read_csv(paths["per_scan"], low_memory=False)
            if required.issubset(per_scan.columns) and set(per_scan["cache_version"].dropna().astype(int).unique()) == {VELOCITY_CACHE_VERSION}:
                return {
                    "per_scan": per_scan,
                    "summary": pd.read_csv(paths["summary"], low_memory=False),
                    "condition_gap": pd.read_csv(paths["condition_gap"], low_memory=False),
                }

        print("Building instantaneous latent velocity cache; this is latent-only and does not decode meshes.")
        frames = [self._build_instantaneous_velocity_for_model(model) for model in QC_MODELS]
        per_scan = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False)
        summary = self.summarize_instantaneous_velocity(per_scan)
        condition_gap = self.condition_gap_instantaneous_velocity(per_scan)
        for path, frame in (
            (paths["per_scan"], per_scan),
            (paths["summary"], summary),
            (paths["condition_gap"], condition_gap),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
        print(f"Saved instantaneous latent velocity cache to {paths['per_scan'].parent}")
        return {
            "per_scan": per_scan,
            "summary": summary,
            "condition_gap": condition_gap,
        }

    def instantaneous_velocity_summary_table(self) -> pd.DataFrame:
        tables = self.build_instantaneous_velocity_tables()
        return tables["summary"].copy()

    def instantaneous_velocity_condition_gap_summary(self) -> pd.DataFrame:
        gap = self.build_instantaneous_velocity_tables()["condition_gap"].copy()
        if gap.empty:
            return gap
        summary = (
            gap.groupby(
                [
                    "dataset",
                    "model",
                    "model_label",
                    "model_priority",
                    "latent_family",
                    "generator_kind",
                    "split",
                    "diagnosis",
                ],
                sort=False,
            )
            .agg(
                rows=("scan_id", "size"),
                subjects=("subject_id", "nunique"),
                scans=("scan_id", "nunique"),
                ad_minus_cn_velocity_z_mean=("ad_minus_cn_velocity_z_l2_per_year", "mean"),
                ad_minus_cn_velocity_z_median=("ad_minus_cn_velocity_z_l2_per_year", "median"),
                ad_minus_cn_velocity_l2_mean=("ad_minus_cn_velocity_l2_per_year", "mean"),
                ad_minus_cn_velocity_l2_median=("ad_minus_cn_velocity_l2_per_year", "median"),
                ad_over_cn_velocity_z_ratio_median=("ad_over_cn_velocity_z_ratio", "median"),
            )
            .reset_index()
        )
        split_priority = {"train": 0, "val": 1, "test": 2}
        summary["split_priority"] = summary["split"].map(lambda value: split_priority.get(str(value), 99))
        return summary.sort_values(["split_priority", "diagnosis", "model_priority"]).reset_index(drop=True)

    def plot_instantaneous_velocity_model_vs_real(self) -> go.Figure:
        summary = self.instantaneous_velocity_summary_table()
        observed = summary.loc[summary["condition_eval"].astype(str).eq("observed")].copy()
        fig = make_subplots(rows=1, cols=3, subplot_titles=["Train", "Val", "Test"])
        for col, split in enumerate(("train", "val", "test"), start=1):
            split_frame = observed.loc[observed["split"].astype(str).eq(split)].copy()
            for diagnosis, color in (("CN", "#2563eb"), ("AD", "#dc2626")):
                group = (
                    split_frame.loc[split_frame["diagnosis"].astype(str).eq(diagnosis)]
                    .sort_values("model_priority")
                    .copy()
                )
                if group.empty:
                    continue
                x = group["model_label"].tolist()
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=group["model_velocity_z_l2_median"],
                        name=f"{diagnosis} model velocity",
                        marker_color=color,
                        opacity=0.72,
                        legendgroup=f"{diagnosis}_model",
                        showlegend=(col == 1),
                    ),
                    row=1,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=group["real_velocity_z_l2_median"],
                        mode="markers",
                        name=f"{diagnosis} real velocity",
                        marker={"color": color, "size": 11, "symbol": "x", "line": {"width": 2}},
                        legendgroup=f"{diagnosis}_real",
                        showlegend=(col == 1),
                    ),
                    row=1,
                    col=col,
                )
            fig.update_xaxes(title_text="Model", tickangle=25, row=1, col=col)
            fig.update_yaxes(title_text="Train-std normalized latent speed / year", row=1, col=col)
        fig.update_layout(
            title="Instantaneous latent velocity: model field vs observed scan-to-scan velocity",
            template="plotly_white",
            barmode="group",
            width=1450,
            height=560,
            legend={"orientation": "h", "y": -0.22},
            margin={"l": 60, "r": 20, "t": 90, "b": 130},
        )
        return fig

    def plot_instantaneous_velocity_alignment(self) -> go.Figure:
        summary = self.instantaneous_velocity_summary_table()
        observed = summary.loc[summary["condition_eval"].astype(str).eq("observed")].copy()
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "Train: velocity error",
                "Val: velocity error",
                "Test: velocity error",
                "Train: direction cosine",
                "Val: direction cosine",
                "Test: direction cosine",
            ],
            vertical_spacing=0.16,
        )
        for col, split in enumerate(("train", "val", "test"), start=1):
            split_frame = observed.loc[observed["split"].astype(str).eq(split)].copy()
            for diagnosis, color in (("CN", "#2563eb"), ("AD", "#dc2626")):
                group = (
                    split_frame.loc[split_frame["diagnosis"].astype(str).eq(diagnosis)]
                    .sort_values("model_priority")
                    .copy()
                )
                if group.empty:
                    continue
                x = group["model_label"].tolist()
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=group["velocity_z_l2_error_median"],
                        name=f"{diagnosis} error",
                        marker_color=color,
                        opacity=0.76,
                        legendgroup=f"{diagnosis}_error",
                        showlegend=(col == 1),
                    ),
                    row=1,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=group["velocity_cosine_median"],
                        mode="lines+markers",
                        name=f"{diagnosis} cosine",
                        line={"color": color, "width": 3},
                        marker={"size": 8},
                        legendgroup=f"{diagnosis}_cosine",
                        showlegend=(col == 1),
                    ),
                    row=2,
                    col=col,
                )
            fig.update_xaxes(tickangle=25, row=1, col=col)
            fig.update_xaxes(title_text="Model", tickangle=25, row=2, col=col)
            fig.update_yaxes(title_text="Velocity error", row=1, col=col)
            fig.update_yaxes(title_text="Cosine", range=[-1.05, 1.05], row=2, col=col)
            fig.add_hline(y=0.0, line_dash="dot", line_color="#777777", row=2, col=col)
        fig.update_layout(
            title="Instantaneous velocity alignment with observed local latent displacement",
            template="plotly_white",
            barmode="group",
            width=1450,
            height=820,
            legend={"orientation": "h", "y": -0.16},
            margin={"l": 60, "r": 20, "t": 100, "b": 120},
        )
        return fig

    def plot_instantaneous_velocity_condition_gap(self) -> go.Figure:
        summary = self.instantaneous_velocity_condition_gap_summary()
        fig = make_subplots(rows=1, cols=3, subplot_titles=["Train", "Val", "Test"])
        for col, split in enumerate(("train", "val", "test"), start=1):
            split_frame = summary.loc[summary["split"].astype(str).eq(split)].copy()
            for diagnosis, color in (("CN", "#2563eb"), ("AD", "#dc2626")):
                group = (
                    split_frame.loc[split_frame["diagnosis"].astype(str).eq(diagnosis)]
                    .sort_values("model_priority")
                    .copy()
                )
                if group.empty:
                    continue
                fig.add_trace(
                    go.Bar(
                        x=group["model_label"],
                        y=group["ad_minus_cn_velocity_z_median"],
                        name=f"{diagnosis} source scans",
                        marker_color=color,
                        opacity=0.78,
                        legendgroup=diagnosis,
                        showlegend=(col == 1),
                    ),
                    row=1,
                    col=col,
                )
            fig.add_hline(y=0.0, line_dash="dot", line_color="#777777", row=1, col=col)
            fig.update_xaxes(title_text="Model", tickangle=25, row=1, col=col)
            fig.update_yaxes(title_text="AD-condition minus CN-condition velocity", row=1, col=col)
        fig.update_layout(
            title="Instantaneous disease-condition sensitivity from the same scan",
            template="plotly_white",
            barmode="group",
            width=1450,
            height=560,
            legend={"orientation": "h", "y": -0.22},
            margin={"l": 60, "r": 20, "t": 90, "b": 130},
        )
        return fig

    def plot_instantaneous_velocity_age_trend(self) -> go.Figure:
        frame = self.build_instantaneous_velocity_tables()["per_scan"].copy()
        frame = frame.loc[frame["condition_eval"].astype(str).eq("observed")].copy()
        if frame.empty:
            return go.Figure()
        frame["age_bin"] = pd.cut(
            pd.to_numeric(frame["age_years"], errors="coerce"),
            bins=[55, 65, 70, 75, 80, 85, 95],
            labels=["55-65", "65-70", "70-75", "75-80", "80-85", "85-95"],
            include_lowest=True,
        )
        summary = (
            frame.groupby(["model", "model_label", "model_priority", "diagnosis", "age_bin"], observed=True)
            .agg(
                rows=("scan_id", "size"),
                model_velocity_z_median=("model_velocity_z_l2_per_year", "median"),
                real_velocity_z_median=("real_velocity_z_l2_per_year", "median"),
            )
            .reset_index()
            .sort_values(["model_priority", "diagnosis", "age_bin"])
        )
        fig = make_subplots(rows=2, cols=2, subplot_titles=[MODEL_LABELS[model] for model in QC_MODELS])
        for idx, model in enumerate(QC_MODELS, start=1):
            row = 1 if idx <= 2 else 2
            col = 1 if idx in (1, 3) else 2
            model_frame = summary.loc[summary["model"].astype(str).eq(model)].copy()
            for diagnosis, color in (("CN", "#2563eb"), ("AD", "#dc2626")):
                group = model_frame.loc[model_frame["diagnosis"].astype(str).eq(diagnosis)].copy()
                if group.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=group["age_bin"].astype(str),
                        y=group["model_velocity_z_median"],
                        mode="lines+markers",
                        name=f"{diagnosis} model",
                        line={"color": color, "width": 3},
                        marker={"size": 8},
                        legendgroup=f"{diagnosis}_model",
                        showlegend=(idx == 1),
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=group["age_bin"].astype(str),
                        y=group["real_velocity_z_median"],
                        mode="markers",
                        name=f"{diagnosis} real",
                        marker={"color": color, "size": 10, "symbol": "x", "line": {"width": 2}},
                        legendgroup=f"{diagnosis}_real",
                        showlegend=(idx == 1),
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(title_text="Age bin (years)", row=row, col=col)
            fig.update_yaxes(title_text="Train-std normalized speed / year", row=row, col=col)
        fig.update_layout(
            title="Instantaneous velocity by age bin, observed condition",
            template="plotly_white",
            width=1350,
            height=860,
            legend={"orientation": "h", "y": -0.10},
            margin={"l": 60, "r": 20, "t": 90, "b": 100},
        )
        return fig

    def volume_cm3(self, value_mm3: Any) -> float:
        return finite_float(value_mm3) / MM3_PER_CM3

    def surface_area_cm2(self, value_mm2: Any) -> float:
        return finite_float(value_mm2) / MM2_PER_CM2

    def scan_volume_mm3_series(self, frame: pd.DataFrame) -> pd.Series:
        index = frame.index
        metadata = (
            pd.to_numeric(frame["metadata_mesh_volume_mm3"], errors="coerce")
            if "metadata_mesh_volume_mm3" in frame.columns
            else pd.Series(np.nan, index=index, dtype=float)
        )
        mask = (
            pd.to_numeric(frame["mask_volume_mm3"], errors="coerce")
            if "mask_volume_mm3" in frame.columns
            else pd.Series(np.nan, index=index, dtype=float)
        )
        scaled = (
            pd.to_numeric(frame["volume"], errors="coerce") * self.volume_scale_mm3
            if "volume" in frame.columns
            else pd.Series(np.nan, index=index, dtype=float)
        )
        return metadata.fillna(mask).fillna(scaled)

    def diagnosis_condition_value(self, diagnosis: str) -> float:
        return 1.0 if str(diagnosis).upper() == "AD" else 0.0

    def available_cached_split_trend_models(self) -> list[str]:
        frame = self.selected_volume_trends
        if frame.empty:
            return []
        available = []
        for model, config in CACHED_SPLIT_TREND_TRANSPORTS.items():
            mask = (
                frame["model"].astype(str).eq(model)
                & frame["transport_method"].astype(str).eq(str(config["transport_method"]))
            )
            if mask.any():
                available.append(model)
        return available

    def cached_split_trend_frame(self, model: str) -> pd.DataFrame:
        if model not in CACHED_SPLIT_TREND_TRANSPORTS:
            raise ValueError(f"No cached split-trend config for {model!r}.")
        config = CACHED_SPLIT_TREND_TRANSPORTS[model]
        all_model_rows = self.selected_volume_trends.loc[
            self.selected_volume_trends["model"].astype(str).eq(model)
        ].copy()
        frame = all_model_rows.loc[
            self.selected_volume_trends["model"].astype(str).eq(model)
            & self.selected_volume_trends["transport_method"].astype(str).eq(str(config["transport_method"]))
        ].copy()
        if frame.empty:
            return frame
        dedupe_cols = [
            "split",
            "diagnosis",
            "subject_id",
            "scan_id",
            "age_years",
            "transport_method",
        ]
        available_dedupe = [column for column in dedupe_cols if column in frame.columns]
        if available_dedupe:
            frame = frame.drop_duplicates(available_dedupe, keep="first").copy()
        frame = frame.reset_index(drop=True)
        frame["years_from_baseline"] = pd.to_numeric(frame["years_from_baseline"], errors="coerce")
        frame["age_years"] = pd.to_numeric(frame["age_years"], errors="coerce")
        if frame["years_from_baseline"].notna().sum() == 0:
            baseline_age = (
                frame.groupby(["split", "diagnosis", "subject_id"], sort=False)["age_years"]
                .transform("min")
            )
            frame["years_from_baseline"] = frame["age_years"] - baseline_age

        observed_raw = pd.to_numeric(frame["observed_volume"], errors="coerce")
        predicted_raw = pd.to_numeric(frame["predicted_volume"], errors="coerce")

        if observed_raw.notna().sum() == 0:
            observed_rows = all_model_rows.loc[
                all_model_rows["transport_method"].astype(str).eq("real_observed")
            ].copy()
            join_cols = ["split", "diagnosis", "subject_id", "scan_id"]
            observed_rows = observed_rows[join_cols + ["volume"]].rename(columns={"volume": "observed_volume_fallback"})
            frame = frame.merge(observed_rows, on=join_cols, how="left")
            observed_raw = pd.to_numeric(frame["observed_volume_fallback"], errors="coerce")

        if predicted_raw.notna().sum() == 0:
            predicted_raw = pd.to_numeric(frame["volume"], errors="coerce")

        frame["observed_volume_cm3"] = observed_raw * self.volume_scale_mm3 / MM3_PER_CM3
        frame["predicted_volume_cm3"] = predicted_raw * self.volume_scale_mm3 / MM3_PER_CM3
        baseline = (
            frame.sort_values(["split", "subject_id", "years_from_baseline", "age_years"])
            .groupby(["split", "diagnosis", "subject_id"], sort=False)
            .agg(
                baseline_observed_cm3=("observed_volume_cm3", "first"),
                baseline_age_years=("age_years", "first"),
                scan_count=("scan_id", "size"),
                max_followup_years=("years_from_baseline", "max"),
            )
            .reset_index()
        )
        frame = frame.merge(baseline, on=["split", "diagnosis", "subject_id"], how="left")
        denom = np.maximum(frame["baseline_observed_cm3"], 1.0e-8)
        frame["observed_rel_change_pct"] = 100.0 * (
            frame["observed_volume_cm3"] - frame["baseline_observed_cm3"]
        ) / denom
        frame["predicted_rel_change_pct"] = 100.0 * (
            frame["predicted_volume_cm3"] - frame["baseline_observed_cm3"]
        ) / denom
        frame["transport_display_label"] = str(config["transport_label"])
        return frame

    def cached_split_volume_trend_summary(self, model: str) -> pd.DataFrame:
        frame = self.cached_split_trend_frame(model)
        if frame.empty:
            return pd.DataFrame()
        summary = (
            frame.groupby(["split", "diagnosis"], sort=False)
            .agg(
                subjects=("subject_id", "nunique"),
                scans=("scan_id", "nunique"),
                median_followup_years=("years_from_baseline", "median"),
                max_followup_years=("years_from_baseline", "max"),
            )
            .reset_index()
        )
        summary["model"] = MODEL_LABELS.get(model, model)
        summary["transport"] = frame["transport_display_label"].iloc[0]
        return summary[
            [
                "model",
                "transport",
                "split",
                "diagnosis",
                "subjects",
                "scans",
                "median_followup_years",
                "max_followup_years",
            ]
        ]

    def plot_cached_split_volume_trends(self, model: str) -> go.Figure:
        frame = self.cached_split_trend_frame(model)
        if frame.empty:
            return go.Figure()
        grid = np.arange(0.0, 12.01, 1.0)
        fig = make_subplots(rows=1, cols=3, subplot_titles=["Train", "Val", "Test"])
        for col, split in enumerate(("train", "val", "test"), start=1):
            split_frame = frame.loc[frame["split"].astype(str).eq(split)].copy()
            for diagnosis, color in (("CN", "#2563eb"), ("AD", "#dc2626")):
                diag = split_frame.loc[split_frame["diagnosis"].astype(str).eq(diagnosis)].copy()
                if diag.empty:
                    continue
                for series_name, series_label, dash in (
                    ("observed_rel_change_pct", f"{diagnosis} GT", "solid"),
                    ("predicted_rel_change_pct", f"{diagnosis} prediction", "dash"),
                ):
                    subject_curves = []
                    for _, group in diag.groupby("subject_id", sort=False):
                        group = group.sort_values("years_from_baseline")
                        x = group["years_from_baseline"].to_numpy(dtype=float)
                        y = group[series_name].to_numpy(dtype=float)
                        valid_mask = np.isfinite(x) & np.isfinite(y)
                        x = x[valid_mask]
                        y = y[valid_mask]
                        if len(x) < 2:
                            continue
                        interp = np.interp(grid, x, y, left=np.nan, right=np.nan)
                        valid = (grid >= x.min()) & (grid <= x.max())
                        interp[~valid] = np.nan
                        subject_curves.append(interp)
                    if not subject_curves:
                        continue
                    matrix = np.vstack(subject_curves)
                    finite_mask = np.isfinite(matrix).any(axis=0)
                    mean = np.full(grid.shape, np.nan, dtype=float)
                    if finite_mask.any():
                        mean[finite_mask] = np.nanmean(matrix[:, finite_mask], axis=0)
                    fig.add_trace(
                        go.Scatter(
                            x=grid,
                            y=mean,
                            mode="lines+markers",
                            name=series_label,
                            legendgroup=series_label,
                            line={"color": color, "width": 3, "dash": dash},
                            marker={"size": 6},
                            showlegend=(col == 1),
                        ),
                        row=1,
                        col=col,
                    )
            fig.update_xaxes(title_text="Years from baseline", row=1, col=col)
            fig.update_yaxes(title_text="Relative volume change (%)", row=1, col=col)
        fig.update_layout(
            title=(
                f"{MODEL_LABELS.get(model, model)} observed-age trend summary | "
                f"{frame['transport_display_label'].iloc[0]}"
            ),
            template="plotly_white",
            width=1380,
            height=560,
            legend={"orientation": "h", "y": -0.18},
            margin={"l": 50, "r": 20, "t": 80, "b": 100},
        )
        return fig

    def case_forecast_transport(self, case_rows: pd.DataFrame, model: str) -> tuple[str, str]:
        if model == "qc_brainode_pca150":
            return "endpoint", "endpoint"
        best = self.best_pair_row(case_rows, model)
        raw = str(best["transport_method"])
        if model == "pca150_direct_cocycle_flow":
            return ("composed", "composed") if raw == "composed_observed" else ("direct", "direct")
        if raw.startswith("composed"):
            return "composed", "composed"
        if raw == "model_no_change":
            return "model_no_change", "no-change"
        return "direct", "direct"

    def case_followup_rows(self, case_key: CaseKey) -> pd.DataFrame:
        case_rows = self.pair_rows_for_case(case_key)
        first = case_rows.iloc[0]
        source_age_years = finite_float(first["source_age_years"])
        target_age_years = finite_float(first["target_age_years"])
        scans = self.scan_volumes.loc[
            self.scan_volumes["split"].astype(str).eq(case_key.split)
            & self.scan_volumes["subject_id"].astype(str).eq(case_key.subject_id)
        ].copy()
        scans = scans.sort_values("age_years")
        mask = (
            pd.to_numeric(scans["age_years"], errors="coerce").ge(source_age_years - 1.0e-8)
            & pd.to_numeric(scans["age_years"], errors="coerce").le(target_age_years + 1.0e-8)
        )
        return scans.loc[mask].copy()

    def _case_followup_cache_file(self, case_key: CaseKey) -> Path:
        return (
            CACHE_DIR
            / "case_followup_forecasts"
            / case_key.split
            / case_key.diagnosis
            / (
                f"{safe_name(case_key.subject_id)}__{safe_name(case_key.source_scan_id)}"
                f"__{safe_name(case_key.target_scan_id)}.csv"
            )
        )

    def build_case_followup_forecasts(
        self,
        case_key: CaseKey,
        *,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        cache_file = self._case_followup_cache_file(case_key)
        required_columns = {
            "kind",
            "model",
            "model_label",
            "source_scan_id",
            "target_scan_id",
            "future_scan_id",
            "future_age_years",
            "volume_mm3",
            "status",
            "error",
            "transport",
            "transport_label",
            "is_target_scan",
            "cache_device",
        }
        if cache_file.is_file() and not force_recompute:
            cached = pd.read_csv(cache_file)
            if required_columns.issubset(cached.columns):
                forecast_failures = cached.loc[
                    cached["kind"].astype(str).eq("forecast")
                    & cached["status"].astype(str).ne("ok")
                ]
                cache_matches_device = cached["cache_device"].astype(str).eq(str(self.device)).all()
                if cache_matches_device or forecast_failures.empty:
                    return cached
        case_rows = self.pair_rows_for_case(case_key)
        first = case_rows.iloc[0]
        source_scan_id = str(first["source_scan_id"])
        target_scan_id = str(first["target_scan_id"])
        source_age_years = finite_float(first["source_age_years"])
        target_age_years = finite_float(first["target_age_years"])
        diagnosis = str(first["diagnosis"])
        condition_value = self.diagnosis_condition_value(diagnosis)
        observed = self.case_followup_rows(case_key).copy()
        observed = observed.sort_values("age_years").reset_index(drop=True)
        observed_volume_mm3 = self.scan_volume_mm3_series(observed)
        rows: list[dict[str, Any]] = []
        for idx, gt_row in observed.iterrows():
            rows.append(
                {
                    "kind": "ground_truth",
                    "model": "ground_truth",
                    "model_label": "Ground truth",
                    "split": case_key.split,
                    "diagnosis": diagnosis,
                    "subject_id": case_key.subject_id,
                    "source_scan_id": source_scan_id,
                    "target_scan_id": target_scan_id,
                    "source_age_years": source_age_years,
                    "target_age_years": target_age_years,
                    "future_scan_id": str(gt_row["scan_id"]),
                    "future_age_years": finite_float(gt_row["age_years"]),
                    "visit_order": finite_float(gt_row["visit_order"]),
                    "volume_mm3": finite_float(observed_volume_mm3.iloc[idx]),
                    "status": "ok",
                    "error": "",
                    "transport": "observed",
                    "transport_label": "observed",
                    "is_target_scan": str(gt_row["scan_id"]) == target_scan_id,
                    "cache_device": str(self.device),
                }
            )
        for model in QC_MODELS:
            transport, transport_label = self.case_forecast_transport(case_rows, model)
            for _, gt_row in observed.iterrows():
                future_age_years = finite_float(gt_row["age_years"])
                future_scan_id = str(gt_row["scan_id"])
                try:
                    if model == "qc_brainode_pca150":
                        mesh = self.brainode_support.forecast_mesh(
                            split=case_key.split,
                            source_scan_id=source_scan_id,
                            target_age_years=future_age_years,
                            condition_value=condition_value,
                        )
                    elif model == "pca150_direct_cocycle_flow":
                        mesh = self.pca_support.forecast_mesh(
                            split=case_key.split,
                            source_scan_id=source_scan_id,
                            source_age_years=source_age_years,
                            target_age_years=future_age_years,
                            condition_value=condition_value,
                            transport="direct" if transport == "direct" else "composed",
                        )
                    else:
                        mesh = self.flow_support(model).forecast_mesh(
                            split=case_key.split,
                            source_scan_id=source_scan_id,
                            source_age_years=source_age_years,
                            target_age_years=future_age_years,
                            condition_value=condition_value,
                            transport=transport,
                        )
                    volume_mm3 = mesh_volume(mesh) * self.volume_scale_mm3
                    status = "ok"
                    error = ""
                except Exception as exc:
                    volume_mm3 = float("nan")
                    status = "mesh_failed"
                    error = repr(exc)
                rows.append(
                    {
                        "kind": "forecast",
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "split": case_key.split,
                        "diagnosis": diagnosis,
                        "subject_id": case_key.subject_id,
                        "source_scan_id": source_scan_id,
                        "target_scan_id": target_scan_id,
                        "source_age_years": source_age_years,
                        "target_age_years": target_age_years,
                        "future_scan_id": future_scan_id,
                        "future_age_years": future_age_years,
                        "visit_order": finite_float(gt_row["visit_order"]),
                        "volume_mm3": volume_mm3,
                        "status": status,
                        "error": error,
                        "transport": transport,
                        "transport_label": transport_label,
                        "is_target_scan": future_scan_id == target_scan_id,
                        "cache_device": str(self.device),
                    }
                )
        frame = pd.DataFrame(rows)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_file, index=False)
        return frame

    def plot_case_model_followup_forecast(
        self,
        case_key: CaseKey,
        model: str,
    ) -> tuple[go.Figure, pd.DataFrame]:
        frame = self.build_case_followup_forecasts(case_key)
        gt = frame.loc[frame["kind"].astype(str).eq("ground_truth")].copy()
        pred = frame.loc[
            frame["kind"].astype(str).eq("forecast") & frame["model"].astype(str).eq(model)
        ].copy()
        pred_ok = pred.loc[pred["status"].astype(str).eq("ok") & pred["volume_mm3"].notna()].copy()
        pred_ok = pred_ok.sort_values("future_age_years")
        failures = pred.loc[pred["status"].astype(str).ne("ok")].copy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=gt["future_age_years"],
                y=gt["volume_mm3"] / MM3_PER_CM3,
                mode="lines+markers",
                name="Ground truth",
                line={"color": "#111111", "width": 3},
                marker={
                    "size": [11 if bool(value) else 8 for value in gt["is_target_scan"].tolist()],
                    "symbol": ["diamond" if bool(value) else "circle" for value in gt["is_target_scan"].tolist()],
                },
            )
        )
        if not pred_ok.empty:
            fig.add_trace(
                go.Scatter(
                    x=pred_ok["future_age_years"],
                    y=pred_ok["volume_mm3"] / MM3_PER_CM3,
                    mode="lines+markers",
                    name=MODEL_LABELS[model],
                    line={"color": MODEL_COLORS[model], "width": 3},
                    marker={
                        "size": [11 if bool(value) else 8 for value in pred_ok["is_target_scan"].tolist()],
                        "symbol": ["diamond" if bool(value) else "circle" for value in pred_ok["is_target_scan"].tolist()],
                    },
                )
            )
        fig.add_vline(
            x=float(gt["future_age_years"].iloc[0]),
            line_dash="dot",
            line_color="#555555",
            annotation_text="source",
            annotation_position="top left",
        )
        if gt["is_target_scan"].any():
            target_age = float(gt.loc[gt["is_target_scan"], "future_age_years"].iloc[0])
            fig.add_vline(
                x=target_age,
                line_dash="dash",
                line_color="#888888",
                annotation_text="selected target",
                annotation_position="top right",
            )
        fig.update_layout(
            title=(
                f"{case_key.split.upper()} {case_key.diagnosis} {MODEL_LABELS[model]} | "
                f"source-to-observed follow-up predictions | subject {case_key.subject_id}"
            ),
            xaxis_title="Age (years)",
            yaxis_title="Volume (cm^3)",
            template="plotly_white",
            width=1120,
            height=520,
            legend={"orientation": "h", "y": -0.18},
            margin={"l": 50, "r": 20, "t": 150 if not failures.empty else 80, "b": 90},
        )
        if not failures.empty:
            failure_lines = [
                f"age {float(row.future_age_years):.2f} ({row.future_scan_id}): {row.error}"
                for row in failures.itertuples()
            ]
            add_failure_annotation(fig, failure_lines, prefix="Skipped follow-up predictions")
        table = gt[
            [
                "split",
                "diagnosis",
                "subject_id",
                "source_scan_id",
                "target_scan_id",
                "source_age_years",
                "target_age_years",
                "future_scan_id",
                "future_age_years",
                "is_target_scan",
                "volume_mm3",
            ]
        ].rename(columns={"volume_mm3": "ground_truth_volume_mm3"})
        table = table.merge(
            pred[
                [
                    "future_scan_id",
                    "future_age_years",
                    "model",
                    "model_label",
                    "transport",
                    "transport_label",
                    "volume_mm3",
                    "status",
                    "error",
                ]
            ].rename(columns={"volume_mm3": "predicted_volume_mm3"}),
            on=["future_scan_id", "future_age_years"],
            how="left",
        )
        table["ground_truth_volume_cm3"] = table["ground_truth_volume_mm3"] / MM3_PER_CM3
        table["predicted_volume_cm3"] = table["predicted_volume_mm3"] / MM3_PER_CM3
        table["abs_error_cm3"] = (
            table["predicted_volume_cm3"] - table["ground_truth_volume_cm3"]
        ).abs()
        ordered = [
            "split",
            "diagnosis",
            "subject_id",
            "model_label",
            "transport_label",
            "source_scan_id",
            "source_age_years",
            "target_scan_id",
            "target_age_years",
            "future_scan_id",
            "future_age_years",
            "is_target_scan",
            "ground_truth_volume_cm3",
            "predicted_volume_cm3",
            "abs_error_cm3",
            "status",
            "error",
        ]
        return fig, table[ordered].sort_values("future_age_years").reset_index(drop=True)

    def pair_rows_for_case(self, case_key: CaseKey) -> pd.DataFrame:
        mask = (
            self.pair_metrics["split"].astype(str).eq(case_key.split)
            & self.pair_metrics["diagnosis"].astype(str).eq(case_key.diagnosis)
            & self.pair_metrics["subject_id"].astype(str).eq(case_key.subject_id)
            & self.pair_metrics["source_scan_id"].astype(str).eq(case_key.source_scan_id)
            & self.pair_metrics["target_scan_id"].astype(str).eq(case_key.target_scan_id)
        )
        return self.pair_metrics.loc[mask].copy()

    def representative_cases(self) -> list[CaseKey]:
        rows = []
        required_by_model = {
            "qc_brainode_pca150": {"brainode_endpoint"},
            "pca150_direct_cocycle_flow": {"direct", "composed_observed"},
            "qc_siren_drop_bad_min2": {"direct", "composed"},
            "qc_siren_latent_ode": {"direct", "composed"},
        }
        for split in ("train", "val", "test"):
            for diagnosis in ("CN", "AD"):
                subset = self.pair_metrics.loc[
                    self.pair_metrics["split"].astype(str).eq(split)
                    & self.pair_metrics["diagnosis"].astype(str).eq(diagnosis)
                ].copy()
                candidates: list[dict[str, Any]] = []
                pair_cols = ["source_scan_id", "target_scan_id"]
                for _, group in subset.groupby(pair_cols, sort=False):
                    ok = True
                    for model, transports in required_by_model.items():
                        model_transports = set(
                            group.loc[group["model"].astype(str).eq(model), "transport_method"]
                            .astype(str)
                            .tolist()
                        )
                        if not transports.issubset(model_transports):
                            ok = False
                            break
                    if not ok:
                        continue
                    score_rows = []
                    for model in QC_MODELS:
                        model_group = group.loc[group["model"].astype(str).eq(model)].copy()
                        if model == "qc_brainode_pca150":
                            chosen = model_group.loc[
                                model_group["transport_method"].astype(str).eq("brainode_endpoint")
                            ].iloc[0]
                        else:
                            live = model_group.loc[
                                model_group["transport_method"].astype(str).isin({"direct", "composed", "composed_observed"})
                            ].copy()
                            chosen = live.sort_values("chamfer_l2_squared", ascending=True).iloc[0]
                        score_rows.append(chosen)
                    score_frame = pd.DataFrame(score_rows)
                    candidates.append(
                        {
                            "subject_id": str(group.iloc[0]["subject_id"]),
                            "source_scan_id": str(group.iloc[0]["source_scan_id"]),
                            "target_scan_id": str(group.iloc[0]["target_scan_id"]),
                            "gap_years": finite_float(group.iloc[0]["gap_years"]),
                            "mean_best_chamfer": float(score_frame["chamfer_l2_squared"].mean()),
                            "mean_best_assd": float(score_frame["assd"].mean()),
                        }
                    )
                if not candidates:
                    continue
                candidates_frame = pd.DataFrame(candidates).sort_values(
                    ["gap_years", "mean_best_chamfer", "mean_best_assd"],
                    ascending=[False, True, True],
                )
                chosen = candidates_frame.iloc[0]
                rows.append(
                    CaseKey(
                        split=split,
                        diagnosis=diagnosis,
                        subject_id=str(chosen["subject_id"]),
                        source_scan_id=str(chosen["source_scan_id"]),
                        target_scan_id=str(chosen["target_scan_id"]),
                    )
                )
        return rows

    def case_overview_table(self, case_key: CaseKey) -> pd.DataFrame:
        rows = self.pair_rows_for_case(case_key)
        summary_rows: list[dict[str, Any]] = []
        for model in QC_MODELS:
            best = self.best_pair_row(rows, model)
            summary_rows.append(
                {
                    "model": MODEL_LABELS[model],
                    "chosen_transport": display_label_for_transport(best["transport_method"]),
                    "gap_years": finite_float(best["gap_years"]),
                    "chamfer_l2_squared": finite_float(best["chamfer_l2_squared"]),
                    "assd_mm": finite_float(best["assd"]) * self.distance_scale_mm,
                    "hd95_mm": finite_float(best["hd95"]) * self.distance_scale_mm,
                    "volume_abs_error_cm3": self.volume_cm3(
                        finite_float(best["volume_abs_error"]) * self.volume_scale_mm3
                    ),
                    "surface_area_abs_error_cm2": self.surface_area_cm2(
                        finite_float(best["surface_area_abs_error"]) * self.surface_area_scale_mm2
                    ),
                }
            )
        return pd.DataFrame(summary_rows)

    def best_pair_row(self, case_rows: pd.DataFrame, model: str) -> pd.Series:
        model_rows = case_rows.loc[case_rows["model"].astype(str).eq(model)].copy()
        if model == "qc_brainode_pca150":
            return model_rows.loc[model_rows["transport_method"].astype(str).eq("brainode_endpoint")].iloc[0]
        live = model_rows.loc[
            model_rows["transport_method"].astype(str).isin({"direct", "composed", "composed_observed"})
        ].copy()
        live = live.sort_values(["chamfer_l2_squared", "assd"], ascending=[True, True])
        return live.iloc[0]

    def model_pair_summary(self) -> pd.DataFrame:
        rows = []
        for model in QC_MODELS:
            subset = self.pair_metrics.loc[
                self.pair_metrics["model"].astype(str).eq(model)
                & self.pair_metrics["transport_method"].astype(str).ne("model_no_change")
            ].copy()
            if subset.empty:
                continue
            for transport in unique_sorted(subset["transport_method"].astype(str)):
                group = subset.loc[subset["transport_method"].astype(str).eq(transport)].copy()
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "transport_method": transport,
                        "transport_label": display_label_for_transport(transport),
                        "rows": int(len(group)),
                        "assd_mm": float(group["assd"].mean()) * self.distance_scale_mm,
                        "hd95_mm": float(group["hd95"].mean()) * self.distance_scale_mm,
                        "volume_abs_error_cm3": self.volume_cm3(
                            float(group["volume_abs_error"].mean()) * self.volume_scale_mm3
                        ),
                        "surface_area_abs_error_cm2": self.surface_area_cm2(
                            float(group["surface_area_abs_error"].mean()) * self.surface_area_scale_mm2
                        ),
                        "volume_relative_error_pct": 100.0 * float(group["volume_relative_error"].mean()),
                        "surface_area_relative_error_pct": 100.0 * float(group["surface_area_relative_error"].mean()),
                    }
                )
        return pd.DataFrame(rows)

    def plot_reconstruction_metric_grid(self) -> go.Figure:
        summary = self.model_pair_summary()
        metrics = [
            ("assd_mm", "ASSD (mm)"),
            ("hd95_mm", "HD95 (mm)"),
            ("volume_abs_error_cm3", "Volume error (cm^3)"),
            ("surface_area_abs_error_cm2", "Surface area error (cm^2)"),
        ]
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[title for _, title in metrics],
        )
        for idx, (metric, title) in enumerate(metrics, start=1):
            row = 1 if idx <= 2 else 2
            col = 1 if idx in (1, 3) else 2
            for model in QC_MODELS:
                group = summary.loc[summary["model"].astype(str).eq(model)].copy()
                if group.empty:
                    continue
                fig.add_trace(
                    go.Bar(
                        x=group["transport_label"],
                        y=group[metric],
                        name=MODEL_LABELS[model],
                        marker_color=MODEL_COLORS[model],
                        legendgroup=model,
                        showlegend=(idx == 1),
                    ),
                    row=row,
                    col=col,
                )
            fig.update_yaxes(title_text=title, row=row, col=col)
        fig.update_layout(
            title="QC-large future-pair mesh error summary",
            template="plotly_white",
            barmode="group",
            width=1300,
            height=900,
            legend={"orientation": "h", "y": -0.08},
            margin={"l": 50, "r": 20, "t": 80, "b": 90},
        )
        return fig

    def ground_truth_relative_volume(self) -> pd.DataFrame:
        frame = self.scan_volumes.copy()
        if frame.empty:
            return frame
        frame = frame.sort_values(["split", "subject_id", "age_years"]).reset_index(drop=True)
        base = (
            frame.groupby(["split", "subject_id"], sort=False)
            .agg(
                baseline_age_years=("age_years", "min"),
                baseline_volume=("volume", "first"),
                diagnosis=("diagnosis", "first"),
            )
            .reset_index()
        )
        merged = frame.merge(base, on=["split", "subject_id", "diagnosis"], how="left")
        merged["years_from_baseline"] = merged["age_years"] - merged["baseline_age_years"]
        merged["relative_volume_pct"] = 100.0 * (
            merged["volume"] - merged["baseline_volume"]
        ) / np.maximum(merged["baseline_volume"], 1.0e-8)
        return merged

    def plot_ground_truth_cn_ad_aging(self) -> go.Figure:
        frame = self.ground_truth_relative_volume()
        if frame.empty:
            return go.Figure()
        grid = np.arange(0.0, 12.01, 1.0)
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["Train", "Val", "Test"],
        )
        for col, split in enumerate(("train", "val", "test"), start=1):
            split_frame = frame.loc[frame["split"].astype(str).eq(split)].copy()
            for diagnosis, color in (("CN", "#2563eb"), ("AD", "#dc2626")):
                diag = split_frame.loc[split_frame["diagnosis"].astype(str).eq(diagnosis)].copy()
                if diag.empty:
                    continue
                subject_curves = []
                for _, group in diag.groupby("subject_id", sort=False):
                    group = group.sort_values("years_from_baseline")
                    x = group["years_from_baseline"].to_numpy(dtype=float)
                    y = group["relative_volume_pct"].to_numpy(dtype=float)
                    if len(x) < 2:
                        continue
                    interp = np.interp(grid, x, y, left=np.nan, right=np.nan)
                    valid = (grid >= x.min()) & (grid <= x.max())
                    interp[~valid] = np.nan
                    subject_curves.append(interp)
                if not subject_curves:
                    continue
                matrix = np.vstack(subject_curves)
                finite_mask = np.isfinite(matrix).any(axis=0)
                mean = np.full(grid.shape, np.nan, dtype=float)
                q25 = np.full(grid.shape, np.nan, dtype=float)
                q75 = np.full(grid.shape, np.nan, dtype=float)
                if finite_mask.any():
                    reduced = matrix[:, finite_mask]
                    mean[finite_mask] = np.nanmean(reduced, axis=0)
                    q25[finite_mask] = np.nanquantile(reduced, 0.25, axis=0)
                    q75[finite_mask] = np.nanquantile(reduced, 0.75, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([grid, grid[::-1]]),
                        y=np.concatenate([q75, q25[::-1]]),
                        fill="toself",
                        fillcolor=hex_to_rgba(color, 0.14),
                        line={"color": "rgba(0,0,0,0)"},
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=grid,
                        y=mean,
                        mode="lines+markers",
                        name=diagnosis,
                        legendgroup=diagnosis,
                        line={"color": color, "width": 3},
                        marker={"size": 6},
                        showlegend=(col == 1),
                    ),
                    row=1,
                    col=col,
                )
            fig.update_xaxes(title_text="Years from baseline", row=1, col=col)
            fig.update_yaxes(title_text="Relative volume change (%)", row=1, col=col)
        fig.update_layout(
            title="Ground-truth CN vs AD hippocampus volume change",
            template="plotly_white",
            width=1380,
            height=520,
            legend={"orientation": "h", "y": -0.16},
            margin={"l": 50, "r": 20, "t": 80, "b": 90},
        )
        return fig

    def plot_ground_truth_start_end_rates(self) -> go.Figure:
        frame = self.scan_volumes.copy()
        if frame.empty:
            return go.Figure()
        rows = []
        for (split, subject_id), group in frame.groupby(["split", "subject_id"], sort=False):
            group = group.sort_values("age_years")
            if len(group) < 2:
                continue
            first = group.iloc[0]
            last = group.iloc[-1]
            gap = finite_float(last["age_years"]) - finite_float(first["age_years"])
            if gap <= 1.0e-8:
                continue
            pct = 100.0 * (finite_float(last["volume"]) - finite_float(first["volume"])) / max(
                finite_float(first["volume"]),
                1.0e-8,
            )
            rows.append(
                {
                    "split": str(split),
                    "subject_id": str(subject_id),
                    "diagnosis": str(first["diagnosis"]),
                    "followup_years": gap,
                    "annual_percent_change": pct / gap,
                }
            )
        summary = pd.DataFrame(rows)
        fig = go.Figure()
        for diagnosis, color in (("CN", "#2563eb"), ("AD", "#dc2626")):
            group = summary.loc[summary["diagnosis"].astype(str).eq(diagnosis)].copy()
            fig.add_trace(
                go.Box(
                    x=group["split"],
                    y=group["annual_percent_change"],
                    name=diagnosis,
                    marker_color=color,
                    boxmean=True,
                )
            )
        fig.update_layout(
            title="Ground-truth start-to-end annualized volume change",
            xaxis_title="Split",
            yaxis_title="Annualized relative change (% / year)",
            template="plotly_white",
            width=1000,
            height=520,
            legend={"orientation": "h", "y": -0.16},
        )
        return fig

    def plot_case_subject_history(self, case_key: CaseKey) -> go.Figure:
        rows = self.pair_rows_for_case(case_key)
        source = rows.iloc[0]
        gt = self.scan_volumes.loc[
            self.scan_volumes["split"].astype(str).eq(case_key.split)
            & self.scan_volumes["subject_id"].astype(str).eq(case_key.subject_id)
        ].copy()
        gt = gt.sort_values("age_years")
        gt_volume_cm3 = self.scan_volume_mm3_series(gt) / MM3_PER_CM3
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=gt["age_years"],
                y=gt_volume_cm3,
                mode="lines+markers",
                name="GT scans",
                line={"color": "#111111", "width": 3},
                marker={"size": 8},
            )
        )
        for model in QC_MODELS:
            best = self.best_pair_row(rows, model)
            fig.add_trace(
                go.Scatter(
                    x=[finite_float(best["target_age_years"])],
                    y=[self.volume_cm3(finite_float(best["predicted_volume"]) * self.volume_scale_mm3)],
                    mode="markers",
                    name=f"{MODEL_LABELS[model]} ({display_label_for_transport(str(best['transport_method']))})",
                    marker={"color": MODEL_COLORS[model], "size": 12, "symbol": "diamond"},
                )
            )
        fig.add_vline(
            x=finite_float(source["source_age_years"]),
            line_dash="dot",
            line_color="#555555",
            annotation_text="source",
            annotation_position="top left",
        )
        fig.add_vline(
            x=finite_float(source["target_age_years"]),
            line_dash="dash",
            line_color="#888888",
            annotation_text="target",
            annotation_position="top right",
        )
        fig.update_layout(
            title=(
                f"{case_key.split.upper()} {case_key.diagnosis} subject {case_key.subject_id} "
                "observed volume history with forecast target"
            ),
            xaxis_title="Age (years)",
            yaxis_title="Volume (cm^3)",
            template="plotly_white",
            width=1100,
            height=520,
            legend={"orientation": "h", "y": -0.18},
        )
        return fig

    def _mesh_cache_path(self, row: pd.Series) -> Path:
        return (
            CACHE_DIR
            / "pair_meshes"
            / str(row["model"])
            / str(row["split"])
            / safe_name(str(row["transport_method"]))
            / (
                f"{safe_name(row['source_scan_id'])}__to__{safe_name(row['target_scan_id'])}"
                f"__{safe_name(row['row_uid'])}.ply"
            )
        )

    def try_resolve_predicted_mesh(self, row: pd.Series) -> tuple[trimesh.Trimesh | None, str | None]:
        try:
            return self.resolve_predicted_mesh(row), None
        except Exception as exc:
            return None, repr(exc)

    def resolve_predicted_mesh(self, row: pd.Series) -> trimesh.Trimesh:
        existing = str(row.get("predicted_mesh_path", "") or "").strip()
        if existing and Path(existing).is_file():
            return load_mesh_cached(existing)
        cache_path = self._mesh_cache_path(row)
        if cache_path.is_file():
            return load_mesh_cached(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        model = str(row["model"])
        if model == "pca150_direct_cocycle_flow":
            mesh = self.pca_support.predict_pair_mesh(row)
        else:
            raise FileNotFoundError(
                f"Missing predicted mesh path for {model} and no local reconstruction handler was needed."
            )
        mesh.export(cache_path)
        return mesh

    def plot_case_mesh_panel(self, case_key: CaseKey) -> go.Figure:
        rows = self.pair_rows_for_case(case_key)
        first = rows.iloc[0]
        source_mesh, source_notes = mesh_for_solid_display(load_mesh_cached(str(first["source_mesh_path"])))
        target_mesh, target_notes = mesh_for_solid_display(load_mesh_cached(str(first["target_mesh_path"])))
        failures: list[str] = []
        display_notes: list[str] = []

        def record_display_notes(title: str, notes: Sequence[str]) -> None:
            if notes:
                clean_title = " ".join(str(title).replace("<br>", " ").split())
                display_notes.append(f"{clean_title}: {'; '.join(notes)}")

        record_display_notes("Source GT", source_notes)
        record_display_notes("Target GT", target_notes)
        tiles: list[tuple[str, trimesh.Trimesh, str]] = [
            (f"Source GT<br>{finite_float(first['source_age_years']):.2f}y", source_mesh, "#7f7f7f"),
            (f"Target GT<br>{finite_float(first['target_age_years']):.2f}y", target_mesh, "#111111"),
        ]
        for model in QC_MODELS:
            best = self.best_pair_row(rows, model)
            mesh, error = self.try_resolve_predicted_mesh(best)
            if mesh is None:
                failures.append(
                    f"{MODEL_LABELS[model]} ({display_label_for_transport(str(best['transport_method']))}): {error}"
                )
                continue
            mesh, mesh_notes = mesh_for_solid_display(mesh)
            title = f"{MODEL_LABELS[model]}<br>{display_label_for_transport(str(best['transport_method']))}"
            record_display_notes(title, mesh_notes)
            tiles.append((title, mesh, MODEL_COLORS[model]))
        fig = make_subplots(
            rows=1,
            cols=len(tiles),
            specs=[[{"type": "scene"} for _ in tiles]],
            subplot_titles=[title for title, _, _ in tiles],
        )
        for col, (_, mesh, color) in enumerate(tiles, start=1):
            fig.add_trace(_mesh3d_trace(mesh, name=str(col), color=color), row=1, col=col)
            fig.update_scenes(
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False,
                aspectmode="data",
                camera={"eye": {"x": 1.55, "y": 1.5, "z": 0.8}},
                row=1,
                col=col,
            )
        fig.update_layout(
            title=(
                f"{case_key.split.upper()} {case_key.diagnosis} mesh comparison | "
                f"subject {case_key.subject_id}"
            ),
            template="plotly_white",
            width=max(1500, 320 * len(tiles)),
            height=540,
            margin={"l": 10, "r": 10, "t": 180 if (failures or display_notes) else 80, "b": 10},
        )
        add_note_annotation(fig, display_notes, prefix="Display-only mesh cleanup", y=1.16)
        add_failure_annotation(
            fig,
            failures,
            prefix="Skipped mesh panel entries",
            y=1.30 if display_notes else 1.13,
        )
        return fig

    def plot_case_change_heatmaps(self, case_key: CaseKey) -> tuple[go.Figure, pd.DataFrame]:
        rows = self.pair_rows_for_case(case_key)
        first = rows.iloc[0]
        source_mesh, source_notes = mesh_for_solid_display(load_mesh_cached(str(first["source_mesh_path"])))
        target_mesh, target_notes = mesh_for_solid_display(load_mesh_cached(str(first["target_mesh_path"])))
        display_notes: list[str] = []

        def record_display_notes(title: str, notes: Sequence[str]) -> None:
            if notes:
                clean_title = " ".join(str(title).replace("<br>", " ").split())
                display_notes.append(f"{clean_title}: {'; '.join(notes)}")

        record_display_notes("Source GT", source_notes)
        record_display_notes("Target GT", target_notes)
        prepared = [("Target GT", aligned_surface_shift(source_mesh, target_mesh))]
        failures: list[str] = []
        for model in QC_MODELS:
            best = self.best_pair_row(rows, model)
            mesh, error = self.try_resolve_predicted_mesh(best)
            label = f"{MODEL_LABELS[model]} ({display_label_for_transport(str(best['transport_method']))})"
            if mesh is None:
                failures.append(f"{label}: {error}")
                continue
            mesh, mesh_notes = mesh_for_solid_display(mesh)
            record_display_notes(label, mesh_notes)
            prepared.append((label, aligned_surface_shift(source_mesh, mesh)))
        global_cmax = max(
            1.0e-8,
            max(float(np.quantile(item["surface_shift"], 0.995)) for _, item in prepared),
        )
        fig = make_subplots(
            rows=1,
            cols=len(prepared),
            specs=[[{"type": "scene"} for _ in prepared]],
            subplot_titles=[
                f"{label}<br>mean={item['mean_shift'] * self.distance_scale_mm:.2f} mm"
                for label, item in prepared
            ],
        )
        summary_rows = []
        for failed in failures:
            summary_rows.append(
                {
                    "label": failed.split(":", 1)[0],
                    "mean_shift_mm": float("nan"),
                    "p95_shift_mm": float("nan"),
                    "max_shift_mm": float("nan"),
                    "icp_cost": float("nan"),
                    "status": "mesh_failed",
                    "error": failed.split(":", 1)[1].strip() if ":" in failed else failed,
                }
            )
        for col, (label, item) in enumerate(prepared, start=1):
            fig.add_trace(
                _heat_trace(
                    item["aligned_mesh"],
                    item["surface_shift"] * self.distance_scale_mm,
                    cmin=0.0,
                    cmax=global_cmax * self.distance_scale_mm,
                    name=label,
                    show_scale=(col == len(prepared)),
                ),
                row=1,
                col=col,
            )
            fig.update_scenes(
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False,
                aspectmode="data",
                camera={"eye": {"x": 1.45, "y": 1.45, "z": 0.78}},
                row=1,
                col=col,
            )
            summary_rows.append(
                {
                    "label": label,
                    "mean_shift_mm": item["mean_shift"] * self.distance_scale_mm,
                    "p95_shift_mm": item["p95_shift"] * self.distance_scale_mm,
                    "max_shift_mm": item["max_shift"] * self.distance_scale_mm,
                    "icp_cost": item["icp_cost"],
                    "status": "ok",
                    "error": "",
                }
            )
        fig.update_layout(
            title=(
                f"{case_key.split.upper()} {case_key.diagnosis} surface-shift maps | "
                f"subject {case_key.subject_id}"
            ),
            template="plotly_white",
            width=max(1600, 340 * len(prepared)),
            height=560,
            margin={"l": 10, "r": 20, "t": 180 if (failures or display_notes) else 80, "b": 10},
        )
        add_note_annotation(fig, display_notes, prefix="Display-only heatmap mesh cleanup", y=1.16)
        add_failure_annotation(
            fig,
            failures,
            prefix="Skipped heatmaps",
            y=1.30 if display_notes else 1.13,
        )
        return fig, pd.DataFrame(summary_rows)

    def anchor_candidates(self) -> pd.DataFrame:
        frame = self.scan_volumes.copy()
        rows = []
        for (split, diagnosis, subject_id), group in frame.groupby(["split", "diagnosis", "subject_id"], sort=False):
            group = group.sort_values("age_years")
            if len(group) < 3:
                continue
            first = group.iloc[0]
            last = group.iloc[-1]
            rows.append(
                {
                    "split": str(split),
                    "diagnosis": str(diagnosis),
                    "subject_id": str(subject_id),
                    "scan_count": int(len(group)),
                    "followup_years": finite_float(last["age_years"]) - finite_float(first["age_years"]),
                    "source_scan_id": str(first["scan_id"]),
                    "source_age_years": finite_float(first["age_years"]),
                    "final_observed_age_years": finite_float(last["age_years"]),
                    "source_mesh_path": str(first["ground_truth_mesh_path"]),
                }
            )
        candidates = pd.DataFrame(rows)
        return candidates.sort_values(["split", "diagnosis", "followup_years", "scan_count"], ascending=[True, True, False, False])

    def select_anchor(self, *, split: str = "test", diagnosis: str = "CN") -> AnchorSpec:
        candidates = self.anchor_candidates()
        subset = candidates.loc[
            candidates["split"].astype(str).eq(split) & candidates["diagnosis"].astype(str).eq(diagnosis)
        ].copy()
        if subset.empty:
            raise ValueError(f"No anchor candidate for split={split!r}, diagnosis={diagnosis!r}.")
        chosen = subset.iloc[0]
        return AnchorSpec(
            split=str(chosen["split"]),
            diagnosis=str(chosen["diagnosis"]),
            subject_id=str(chosen["subject_id"]),
            source_scan_id=str(chosen["source_scan_id"]),
            source_age_years=finite_float(chosen["source_age_years"]),
            final_observed_age_years=finite_float(chosen["final_observed_age_years"]),
            source_mesh_path=str(chosen["source_mesh_path"]),
        )

    def _anchor_cache_file(
        self,
        *,
        split: str,
        subject_id: str,
        transport: str,
        horizons: Sequence[float],
        step_label: str,
    ) -> Path:
        horizon_key = "_".join(safe_name(value) for value in horizons)
        return CACHE_DIR / "anchor_forecasts" / f"{split}_{subject_id}_{transport}_{step_label}_{horizon_key}.csv"

    def _normalize_anchor_forecast_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            normalized = frame.copy()
            if "status" not in normalized.columns:
                normalized["status"] = pd.Series(dtype="object")
            if "error" not in normalized.columns:
                normalized["error"] = pd.Series(dtype="object")
            return normalized

        normalized = frame.copy()
        if "status" not in normalized.columns:
            normalized["status"] = "ok"
        else:
            normalized["status"] = normalized["status"].fillna("ok").astype(str)
        if "error" not in normalized.columns:
            normalized["error"] = ""
        else:
            normalized["error"] = normalized["error"].fillna("").astype(str)
        if "model_label" not in normalized.columns and "model" in normalized.columns:
            model_names = normalized["model"].astype(str)
            normalized["model_label"] = model_names.map(MODEL_LABELS).fillna(model_names)
        if "is_observed_future_age" not in normalized.columns and "future_age_years" in normalized.columns:
            gt_rows = normalized.loc[normalized["kind"].astype(str).eq("ground_truth")].copy()
            observed_future_age_set = {
                round(float(value), 6)
                for value in gt_rows["future_age_years"].tolist()
                if pd.notna(value)
            }
            normalized["is_observed_future_age"] = [
                round(float(value), 6) in observed_future_age_set if pd.notna(value) else False
                for value in normalized["future_age_years"].tolist()
            ]
        return normalized

    def build_anchor_forecasts(
        self,
        *,
        split: str = "test",
        diagnosis: str = "CN",
        transport: str = "direct",
        horizons_years: Sequence[float] = (0.0, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0),
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        anchor = self.select_anchor(split=split, diagnosis=diagnosis)
        cache_file = self._anchor_cache_file(
            split=split,
            subject_id=anchor.subject_id,
            transport=transport,
            horizons=horizons_years,
            step_label=f"{diagnosis}_{transport}",
        )
        if cache_file.is_file() and not force_recompute:
            cached = pd.read_csv(cache_file)
            cached = self._normalize_anchor_forecast_frame(cached)
            required_columns = {
                "kind",
                "model",
                "condition_label",
                "future_age_years",
                "volume_mm3",
                "is_observed_future_age",
                "status",
                "error",
            }
            if required_columns.issubset(cached.columns):
                cached.to_csv(cache_file, index=False)
                return cached
        rows: list[dict[str, Any]] = []
        gt = self.scan_volumes.loc[
            self.scan_volumes["split"].astype(str).eq(split)
            & self.scan_volumes["subject_id"].astype(str).eq(anchor.subject_id)
        ].copy()
        gt = gt.sort_values("age_years")
        observed_future_ages = unique_sorted(
            [
                finite_float(value)
                for value in gt["age_years"].tolist()
                if finite_float(value) >= anchor.source_age_years - 1.0e-8
            ]
        )
        observed_future_age_set = {round(float(value), 6) for value in observed_future_ages}
        for _, gt_row in gt.iterrows():
            gt_volume_mm3 = finite_float(
                self.scan_volume_mm3_series(pd.DataFrame([gt_row])).iloc[0]
            )
            rows.append(
                {
                    "kind": "ground_truth",
                    "model": "ground_truth",
                    "model_label": "Ground truth",
                    "condition_label": diagnosis,
                    "transport": "observed",
                    "subject_id": anchor.subject_id,
                    "source_scan_id": anchor.source_scan_id,
                    "source_diagnosis": diagnosis,
                    "source_age_years": anchor.source_age_years,
                    "future_age_years": finite_float(gt_row["age_years"]),
                    "horizon_years": finite_float(gt_row["age_years"]) - anchor.source_age_years,
                    "volume_mm3": gt_volume_mm3,
                    "is_beyond_observed": False,
                    "is_ood": False,
                    "is_observed_future_age": True,
                    "status": "ok",
                    "error": "",
                }
            )
        future_ages = unique_sorted(
            [anchor.source_age_years + float(h) for h in horizons_years] + observed_future_ages
        )
        print(
            f"Building anchor forecast for {split} {diagnosis} subject {anchor.subject_id} "
            f"from age {anchor.source_age_years:.2f} with transport={transport}"
        )
        for condition_label, condition_value in (("CN", 0.0), ("AD", 1.0)):
            print(f"  condition {condition_label}")
            for future_age_years in future_ages:
                horizon = float(future_age_years) - anchor.source_age_years
                is_beyond_observed = future_age_years > anchor.final_observed_age_years + 1.0e-6
                is_ood = future_age_years > 95.62053388 + 1.0e-6

                observed_age_flag = round(float(future_age_years), 6) in observed_future_age_set

                def append_forecast_row(
                    *,
                    model_name: str,
                    model_label: str,
                    transport_label: str,
                    mesh_builder,
                ) -> None:
                    try:
                        mesh = mesh_builder()
                        volume_mm3 = mesh_volume(mesh) * self.volume_scale_mm3
                        status = "ok"
                        error = ""
                    except Exception as exc:
                        volume_mm3 = float("nan")
                        status = "mesh_failed"
                        error = repr(exc)
                    rows.append(
                        {
                            "kind": "forecast",
                            "model": model_name,
                            "model_label": model_label,
                            "condition_label": condition_label,
                            "transport": transport_label,
                            "subject_id": anchor.subject_id,
                            "source_scan_id": anchor.source_scan_id,
                            "source_diagnosis": diagnosis,
                            "source_age_years": anchor.source_age_years,
                            "future_age_years": future_age_years,
                            "horizon_years": horizon,
                            "volume_mm3": volume_mm3,
                            "is_beyond_observed": is_beyond_observed,
                            "is_ood": is_ood,
                            "is_observed_future_age": observed_age_flag,
                            "status": status,
                            "error": error,
                        }
                    )

                append_forecast_row(
                    model_name="qc_brainode_pca150",
                    model_label=MODEL_LABELS["qc_brainode_pca150"],
                    transport_label="endpoint",
                    mesh_builder=lambda: self.brainode_support.forecast_mesh(
                        split=split,
                        source_scan_id=anchor.source_scan_id,
                        target_age_years=future_age_years,
                        condition_value=condition_value,
                    ),
                )

                append_forecast_row(
                    model_name="pca150_direct_cocycle_flow",
                    model_label=MODEL_LABELS["pca150_direct_cocycle_flow"],
                    transport_label=transport,
                    mesh_builder=lambda: self.pca_support.forecast_mesh(
                        split=split,
                        source_scan_id=anchor.source_scan_id,
                        source_age_years=anchor.source_age_years,
                        target_age_years=future_age_years,
                        condition_value=condition_value,
                        transport="direct" if transport == "direct" else "composed",
                    ),
                )

                for model_name in ("qc_siren_drop_bad_min2", "qc_siren_latent_ode"):
                    append_forecast_row(
                        model_name=model_name,
                        model_label=MODEL_LABELS[model_name],
                        transport_label=transport,
                        mesh_builder=lambda model_name=model_name: self.flow_support(model_name).forecast_mesh(
                            split=split,
                            source_scan_id=anchor.source_scan_id,
                            source_age_years=anchor.source_age_years,
                            target_age_years=future_age_years,
                            condition_value=condition_value,
                            transport=transport,
                        ),
                    )
        frame = self._normalize_anchor_forecast_frame(pd.DataFrame(rows))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_file, index=False)
        return frame

    def plot_anchor_forecasts(
        self,
        *,
        split: str = "test",
        diagnosis: str = "CN",
        transport: str = "direct",
    ) -> go.Figure:
        frame = self.build_anchor_forecasts(split=split, diagnosis=diagnosis, transport=transport)
        anchor = self.select_anchor(split=split, diagnosis=diagnosis)
        gt = frame.loc[frame["kind"].astype(str).eq("ground_truth")].copy()
        live = frame.loc[frame["kind"].astype(str).eq("forecast")].copy()
        failures = live.loc[live["status"].astype(str).ne("ok")].copy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=gt["future_age_years"],
                y=gt["volume_mm3"] / MM3_PER_CM3,
                mode="lines+markers",
                name="Ground truth",
                line={"color": "#111111", "width": 3},
                marker={"size": 8},
            )
        )
        for model in QC_MODELS:
            subset = live.loc[live["model"].astype(str).eq(model)].copy()
            for condition_label in ("CN", "AD"):
                group = subset.loc[subset["condition_label"].astype(str).eq(condition_label)].copy()
                group = group.loc[group["status"].astype(str).eq("ok") & group["volume_mm3"].notna()].copy()
                if group.empty:
                    continue
                group = group.sort_values("future_age_years").reset_index(drop=True)
                marker_sizes = [
                    9 if bool(value) else 6 for value in group["is_observed_future_age"].tolist()
                ]
                fig.add_trace(
                    go.Scatter(
                        x=group["future_age_years"],
                        y=group["volume_mm3"] / MM3_PER_CM3,
                        mode="lines+markers",
                        name=f"{MODEL_LABELS[model]} | {condition_label}",
                        line={
                            "color": MODEL_COLORS[model],
                            "width": 3,
                            "dash": CONDITION_STYLES[condition_label]["dash"],
                        },
                        marker={"size": marker_sizes},
                    )
                )
        observed_limit = float(gt["future_age_years"].max())
        future_limit = float(live["future_age_years"].max())
        if future_limit > observed_limit + 1.0e-6:
            fig.add_vrect(
                x0=observed_limit,
                x1=future_limit,
                fillcolor="#e5e7eb",
                opacity=0.22,
                line_width=0,
                annotation_text="Beyond observed follow-up",
                annotation_position="top left",
            )
        fig.update_layout(
            title=(
                f"{split.upper()} {diagnosis} baseline-source forecast | subject {anchor.subject_id} | "
                f"transport={transport} | solid=CN, dashed=AD | larger markers=observed follow-up ages"
            ),
            xaxis_title="Age (years)",
            yaxis_title="Volume (cm^3)",
            template="plotly_white",
            width=1250,
            height=560,
            legend={"orientation": "h", "y": -0.18},
            margin={"l": 50, "r": 20, "t": 150 if not failures.empty else 80, "b": 100},
        )
        if not failures.empty:
            failure_lines = [
                f"{MODEL_LABELS.get(str(row.model), str(row.model))} | {row.condition_label} | age {float(row.future_age_years):.2f}: {row.error}"
                for row in failures.itertuples()
            ]
            add_failure_annotation(fig, failure_lines, prefix="Skipped forecast meshes")
        return fig

    def plot_anchor_condition_gap(
        self,
        *,
        split: str = "test",
        diagnosis: str = "CN",
        transport: str = "direct",
    ) -> go.Figure:
        frame = self.build_anchor_forecasts(split=split, diagnosis=diagnosis, transport=transport)
        live = frame.loc[frame["kind"].astype(str).eq("forecast")].copy()
        failures = live.loc[live["status"].astype(str).ne("ok")].copy()
        fig = go.Figure()
        for model in QC_MODELS:
            cn = live.loc[
                live["model"].astype(str).eq(model)
                & live["condition_label"].astype(str).eq("CN")
                & live["status"].astype(str).eq("ok")
            ].copy()
            ad = live.loc[
                live["model"].astype(str).eq(model)
                & live["condition_label"].astype(str).eq("AD")
                & live["status"].astype(str).eq("ok")
            ].copy()
            merged = cn.merge(
                ad,
                on=["future_age_years", "horizon_years"],
                suffixes=("_cn", "_ad"),
                how="inner",
            )
            if merged.empty:
                continue
            merged["ad_minus_cn_cm3"] = (merged["volume_mm3_ad"] - merged["volume_mm3_cn"]) / MM3_PER_CM3
            fig.add_trace(
                go.Scatter(
                    x=merged["future_age_years"],
                    y=merged["ad_minus_cn_cm3"],
                    mode="lines+markers",
                    name=MODEL_LABELS[model],
                    line={"color": MODEL_COLORS[model], "width": 3},
                    marker={"size": 8},
                )
            )
        fig.add_hline(y=0.0, line_color="#777777", line_dash="dot")
        fig.update_layout(
            title=(
                f"{split.upper()} {diagnosis} anchor disease-conditioned gap | "
                "AD-condition volume minus CN-condition volume"
            ),
            xaxis_title="Age (years)",
            yaxis_title="AD - CN predicted volume (cm^3)",
            template="plotly_white",
            width=1150,
            height=520,
            legend={"orientation": "h", "y": -0.16},
            margin={"l": 50, "r": 20, "t": 150 if not failures.empty else 80, "b": 80},
        )
        if not failures.empty:
            failure_lines = [
                f"{MODEL_LABELS.get(str(row.model), str(row.model))} | {row.condition_label} | age {float(row.future_age_years):.2f}: {row.error}"
                for row in failures.itertuples()
            ]
            add_failure_annotation(fig, failure_lines, prefix="Gap plot skipped failed meshes")
        return fig

    def _anchor_mesh_cache_path(
        self,
        *,
        model: str,
        split: str,
        source_scan_id: str,
        source_diagnosis: str,
        condition_label: str,
        transport: str,
        future_age_years: float,
    ) -> Path:
        return (
            CACHE_DIR
            / "anchor_meshes"
            / model
            / split
            / source_diagnosis
            / condition_label
            / transport
            / f"{safe_name(source_scan_id)}__age_{safe_name(round(float(future_age_years), 3))}.ply"
        )

    def resolve_anchor_mesh(
        self,
        *,
        model: str,
        split: str,
        diagnosis: str,
        condition_label: str,
        transport: str,
        future_age_years: float,
    ) -> trimesh.Trimesh:
        anchor = self.select_anchor(split=split, diagnosis=diagnosis)
        cache_path = self._anchor_mesh_cache_path(
            model=model,
            split=split,
            source_scan_id=anchor.source_scan_id,
            source_diagnosis=diagnosis,
            condition_label=condition_label,
            transport=transport,
            future_age_years=future_age_years,
        )
        if cache_path.is_file():
            return load_mesh_cached(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        condition_value = 0.0 if condition_label == "CN" else 1.0
        if model == "qc_brainode_pca150":
            mesh = self.brainode_support.forecast_mesh(
                split=split,
                source_scan_id=anchor.source_scan_id,
                target_age_years=future_age_years,
                condition_value=condition_value,
            )
        elif model == "pca150_direct_cocycle_flow":
            mesh = self.pca_support.forecast_mesh(
                split=split,
                source_scan_id=anchor.source_scan_id,
                source_age_years=anchor.source_age_years,
                target_age_years=future_age_years,
                condition_value=condition_value,
                transport="direct" if transport == "direct" else "composed",
            )
        else:
            mesh = self.flow_support(model).forecast_mesh(
                split=split,
                source_scan_id=anchor.source_scan_id,
                source_age_years=anchor.source_age_years,
                target_age_years=future_age_years,
                condition_value=condition_value,
                transport=transport,
            )
        mesh.export(cache_path)
        return mesh

    def try_resolve_anchor_mesh(
        self,
        *,
        model: str,
        split: str,
        diagnosis: str,
        condition_label: str,
        transport: str,
        future_age_years: float,
    ) -> tuple[trimesh.Trimesh | None, str | None]:
        try:
            return (
                self.resolve_anchor_mesh(
                    model=model,
                    split=split,
                    diagnosis=diagnosis,
                    condition_label=condition_label,
                    transport=transport,
                    future_age_years=future_age_years,
                ),
                None,
            )
        except Exception as exc:
            return None, repr(exc)

    def plot_anchor_change_maps(
        self,
        *,
        split: str = "test",
        diagnosis: str = "CN",
        condition_label: str = "CN",
        transport: str = "direct",
        future_age_years: float | None = None,
    ) -> tuple[go.Figure, pd.DataFrame]:
        anchor = self.select_anchor(split=split, diagnosis=diagnosis)
        if future_age_years is None:
            future_age_years = anchor.source_age_years + 20.0
        source_mesh, source_notes = mesh_for_solid_display(load_mesh_cached(anchor.source_mesh_path))
        display_notes: list[str] = []
        if source_notes:
            display_notes.append(f"Source GT: {'; '.join(source_notes)}")
        prepared = []
        summary_rows = []
        failures: list[str] = []
        for model in QC_MODELS:
            mesh, error = self.try_resolve_anchor_mesh(
                model=model,
                split=split,
                diagnosis=diagnosis,
                condition_label=condition_label,
                transport=transport,
                future_age_years=float(future_age_years),
            )
            if mesh is None:
                failures.append(f"{MODEL_LABELS[model]}: {error}")
                summary_rows.append(
                    {
                        "model": MODEL_LABELS[model],
                        "condition_label": condition_label,
                        "future_age_years": future_age_years,
                        "mean_shift_mm": float("nan"),
                        "p95_shift_mm": float("nan"),
                        "status": "mesh_failed",
                        "error": error,
                    }
                )
                continue
            mesh, mesh_notes = mesh_for_solid_display(mesh)
            if mesh_notes:
                display_notes.append(f"{MODEL_LABELS[model]}: {'; '.join(mesh_notes)}")
            shift = aligned_surface_shift(source_mesh, mesh)
            prepared.append((model, shift))
            summary_rows.append(
                {
                    "model": MODEL_LABELS[model],
                    "condition_label": condition_label,
                    "future_age_years": future_age_years,
                    "mean_shift_mm": shift["mean_shift"] * self.distance_scale_mm,
                    "p95_shift_mm": shift["p95_shift"] * self.distance_scale_mm,
                    "status": "ok",
                    "error": "",
                }
            )
        if not prepared:
            fig = go.Figure()
            fig.add_annotation(
                text="No future meshes decoded successfully for this view.",
                x=0.5,
                y=0.6,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 16},
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(
                title=(
                    f"{split.upper()} {diagnosis} anchor shape -> {condition_label} future | "
                    f"{transport} | age {future_age_years:.1f}"
                ),
                template="plotly_white",
                width=900,
                height=420,
                margin={"l": 20, "r": 20, "t": 180 if (failures or display_notes) else 80, "b": 20},
            )
            add_note_annotation(fig, display_notes, prefix="Display-only anchor mesh cleanup", y=1.16)
            add_failure_annotation(
                fig,
                failures,
                prefix="Skipped anchor change maps",
                y=1.30 if display_notes else 1.13,
            )
            return fig, pd.DataFrame(summary_rows)
        cmax = max(
            1.0e-8,
            max(float(np.quantile(item["surface_shift"], 0.995)) for _, item in prepared),
        )
        fig = make_subplots(
            rows=1,
            cols=len(prepared),
            specs=[[{"type": "scene"} for _ in prepared]],
            subplot_titles=[
                f"{MODEL_LABELS[model]}<br>mean={item['mean_shift'] * self.distance_scale_mm:.2f} mm"
                for model, item in prepared
            ],
        )
        for col, (model, item) in enumerate(prepared, start=1):
            fig.add_trace(
                _heat_trace(
                    item["aligned_mesh"],
                    item["surface_shift"] * self.distance_scale_mm,
                    cmin=0.0,
                    cmax=cmax * self.distance_scale_mm,
                    name=MODEL_LABELS[model],
                    show_scale=(col == len(prepared)),
                ),
                row=1,
                col=col,
            )
            fig.update_scenes(
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False,
                aspectmode="data",
                camera={"eye": {"x": 1.45, "y": 1.45, "z": 0.78}},
                row=1,
                col=col,
            )
        fig.update_layout(
            title=(
                f"{split.upper()} {diagnosis} anchor shape -> {condition_label} future | "
                f"{transport} | age {future_age_years:.1f}"
            ),
            template="plotly_white",
            width=max(1400, 340 * len(prepared)),
            height=560,
            margin={"l": 10, "r": 20, "t": 180 if (failures or display_notes) else 80, "b": 10},
        )
        add_note_annotation(fig, display_notes, prefix="Display-only anchor mesh cleanup", y=1.16)
        add_failure_annotation(
            fig,
            failures,
            prefix="Skipped anchor change maps",
            y=1.30 if display_notes else 1.13,
        )
        return fig, pd.DataFrame(summary_rows)


def create_context(device: str = "auto") -> LongitudinalVisualizationContext:
    return LongitudinalVisualizationContext(device=device)
