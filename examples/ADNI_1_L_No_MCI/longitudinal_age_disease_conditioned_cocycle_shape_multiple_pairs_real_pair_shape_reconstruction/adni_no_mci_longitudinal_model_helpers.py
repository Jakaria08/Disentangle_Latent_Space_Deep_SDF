from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deep_sdf
from deep_sdf import mesh as deep_sdf_mesh
import deep_sdf.workspace as ws
import train_deep_sdf_longitudinal as longitudinal

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - plotly is optional for non-visual runs
    go = None


TRAINING_AGE_MIN = 57.0
TRAINING_AGE_MAX = 91.0
TRAINING_AGE_RANGE_YEARS = TRAINING_AGE_MAX - TRAINING_AGE_MIN
ONE_YEAR_NORM_DELTA = 1.0 / TRAINING_AGE_RANGE_YEARS
SIX_MONTH_NORM_DELTA = 0.5 / TRAINING_AGE_RANGE_YEARS
DEFAULT_ALIGN_MODE = "centroid"
DEFAULT_ALIGN_ITERS = 20
DEFAULT_ALIGN_TRIM_QUANTILE = 0.90
DEFAULT_SURFACE_SAMPLES = 30000
DEFAULT_ANCHOR_FIT_STEPS = 400
DEFAULT_ANCHOR_FIT_SAMPLES = 8192
DEFAULT_ANCHOR_FIT_LR = 5e-3
DEFAULT_ANCHOR_INIT_STD = 1e-2
DEFAULT_VELOCITY_EPS = 1e-4
DEFAULT_VELOCITY_METHOD = "finite_difference"


@dataclass
class LoadedModel:
    experiment_dir: Path
    specs: Dict[str, object]
    metadata: pd.DataFrame
    labels: Dict[str, object]
    device: torch.device
    checkpoint: str
    checkpoint_epoch: int
    decoder: torch.nn.Module
    flow: longitudinal.TemporalFlowMLP
    train_latents: Optional[torch.Tensor]
    align_mode: str
    align_iters: int
    align_trim_quantile: float


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def determine_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def resolve_path(path_str: str | Path, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def ensure_file(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _strip_model_suffix(name: str) -> str:
    return name[:-4] if name.endswith(".pth") else name


def list_model_checkpoints(experiment_dir: str | Path) -> List[str]:
    model_dir = Path(experiment_dir) / ws.model_params_subdir
    if not model_dir.is_dir():
        return []
    checkpoints = []
    for path in model_dir.glob("*.pth"):
        stem = _strip_model_suffix(path.name)
        if stem == "latest":
            continue
        checkpoints.append(stem)

    def _sort_key(name: str) -> Tuple[int, object]:
        return (0, int(name)) if name.isdigit() else (1, name)

    return sorted(set(checkpoints), key=_sort_key)


def canonical_checkpoint_name(experiment_dir: str | Path, checkpoint: str | int) -> str:
    ckpt = _strip_model_suffix(str(checkpoint).strip())
    model_path = Path(experiment_dir) / ws.model_params_subdir / f"{ckpt}.pth"
    if model_path.is_file():
        return ckpt
    if ckpt == "selected":
        selected_json = (
            Path(experiment_dir)
            / "analysis"
            / "validation"
            / "selected_checkpoint.json"
        )
        payload = json.loads(ensure_file(selected_json).read_text(encoding="utf-8"))
        selected = payload.get("selected_checkpoint", None)
        if selected is None:
            raise RuntimeError(f"selected_checkpoint.json has no selected_checkpoint: {selected_json}")
        return str(selected)
    raise FileNotFoundError(f"Checkpoint not found under {experiment_dir}: {ckpt}")


def load_metadata(experiment_dir: str | Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    experiment_dir = Path(experiment_dir).resolve()
    csv_path = ensure_file(
        experiment_dir / "metadata" / "adni_no_mci_longitudinal_records.csv"
    )
    labels_path = ensure_file(
        experiment_dir / "metadata" / "adni_no_mci_longitudinal_labels.pt"
    )
    frame = pd.read_csv(csv_path)
    labels = torch.load(labels_path, map_location="cpu")
    if not isinstance(labels, dict) or "records" not in labels:
        raise RuntimeError(f"Unexpected labels payload at {labels_path}")
    frame["visit_order"] = frame["visit_order"].astype(int)
    frame["label_ad"] = frame["label_ad"].astype(int)
    for column in (
        "months_from_baseline",
        "elapsed_years",
        "baseline_age_years",
        "continuous_age_years",
        "continuous_age_norm",
    ):
        frame[column] = frame[column].astype(float)
    return (
        frame.sort_values(["split", "subject_id", "visit_order", "scan_id"]).reset_index(
            drop=True
        ),
        labels,
    )


def _load_latent_checkpoint(experiment_dir: Path, checkpoint: str) -> Optional[torch.Tensor]:
    lat_path = experiment_dir / ws.latent_codes_subdir / f"{checkpoint}.pth"
    if not lat_path.is_file():
        return None
    payload = torch.load(lat_path, map_location="cpu")
    if "latent_codes" not in payload:
        return None
    latent_codes = payload["latent_codes"]
    if isinstance(latent_codes, dict):
        weight = latent_codes.get("weight", None)
    else:
        weight = latent_codes
    if weight is None:
        return None
    if torch.is_tensor(weight) and weight.dim() == 2:
        return weight.detach().cpu().float()
    return None


def load_model_bundle(
    experiment_dir: str | Path,
    checkpoint: str | int,
    device: str | torch.device | None = None,
) -> LoadedModel:
    experiment_dir = Path(experiment_dir).resolve()
    specs = ws.load_experiment_specifications(str(experiment_dir))
    checkpoint_name = canonical_checkpoint_name(experiment_dir, checkpoint)
    metadata, labels = load_metadata(experiment_dir)
    device_obj = determine_device(device)

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(specs["CodeLength"], **specs["NetworkSpecs"]).to(device_obj)
    flow = longitudinal.TemporalFlowMLP(
        specs["CodeLength"],
        specs["FlowHiddenDims"],
        age_condition_dim=int(specs.get("AgeConditionDim", 0) or 0),
    ).to(device_obj)
    epoch = longitudinal.load_model_and_flow(
        str(experiment_dir), checkpoint_name, decoder, flow
    )
    decoder.eval()
    flow.eval()

    return LoadedModel(
        experiment_dir=experiment_dir,
        specs=specs,
        metadata=metadata,
        labels=labels,
        device=device_obj,
        checkpoint=checkpoint_name,
        checkpoint_epoch=int(epoch),
        decoder=decoder,
        flow=flow,
        train_latents=_load_latent_checkpoint(experiment_dir, checkpoint_name),
        align_mode=str(specs.get("EvalChamferAlignMode", DEFAULT_ALIGN_MODE)),
        align_iters=int(specs.get("EvalChamferAlignIters", DEFAULT_ALIGN_ITERS)),
        align_trim_quantile=float(
            specs.get("EvalChamferAlignTrimQuantile", DEFAULT_ALIGN_TRIM_QUANTILE)
        ),
    )


def split_frame(metadata: pd.DataFrame, split: str) -> pd.DataFrame:
    return metadata.loc[metadata["split"] == split].copy().reset_index(drop=True)


def grouped_subject_rows(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    grouped = {}
    for subject_id, group in frame.groupby("subject_id", sort=True):
        grouped[str(subject_id)] = (
            group.sort_values(["visit_order", "continuous_age_norm", "scan_id"])
            .reset_index(drop=True)
        )
    return grouped


def select_subjects_by_diagnosis(frame: pd.DataFrame, per_group: int = 2) -> List[str]:
    selected: List[str] = []
    for diagnosis in ("CN", "AD"):
        subject_ids = sorted(frame.loc[frame["diagnosis"] == diagnosis, "subject_id"].unique())
        selected.extend([str(subject_id) for subject_id in subject_ids[:per_group]])
    return selected


def _record_from_scan_id(metadata: pd.DataFrame, scan_id: str) -> pd.Series:
    matches = metadata.loc[metadata["scan_id"] == scan_id]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one metadata row for scan_id={scan_id}, found {len(matches)}")
    return matches.iloc[0]


def _clone_samples_for_deterministic_fit(samples: Sequence[torch.Tensor], seed: int) -> List[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    cloned = []
    for tensor in samples:
        tensor_cpu = tensor.detach().cpu()
        perm = torch.randperm(tensor_cpu.shape[0], generator=generator)
        cloned.append(tensor_cpu.index_select(0, perm))
    return cloned


def build_observations_from_scan_ids(
    metadata: pd.DataFrame,
    scan_ids: Sequence[str],
    seed: int = 0,
) -> List[Dict[str, object]]:
    observations: List[Dict[str, object]] = []
    for offset, scan_id in enumerate(scan_ids):
        row = _record_from_scan_id(metadata, scan_id)
        sdf_samples = deep_sdf.data.read_sdf_samples_into_ram(str(row["sdf_npz_path"]))
        observations.append(
            {
                "scan_id": str(row["scan_id"]),
                "samples": _clone_samples_for_deterministic_fit(
                    sdf_samples, seed=seed + offset
                ),
                "time": float(row["continuous_age_norm"]),
                "age_cond": torch.tensor(
                    [float(row["label_ad"])], dtype=torch.float32
                ),
            }
        )
    observations.sort(key=lambda item: float(item["time"]))
    return observations


def fit_subject_anchor(
    bundle: LoadedModel,
    observation_scan_ids: Sequence[str],
    seed: int = 0,
    num_iterations: int = DEFAULT_ANCHOR_FIT_STEPS,
    num_samples: int = DEFAULT_ANCHOR_FIT_SAMPLES,
    lr: float = DEFAULT_ANCHOR_FIT_LR,
    init_std: float = DEFAULT_ANCHOR_INIT_STD,
    code_reg_lambda: Optional[float] = None,
) -> Tuple[torch.Tensor, List[float], List[Dict[str, object]]]:
    observations = build_observations_from_scan_ids(bundle.metadata, observation_scan_ids, seed=seed)
    seed_everything(seed)
    anchor, loss_hist = longitudinal.optimize_subject_anchor_from_observations(
        bundle.decoder,
        bundle.flow,
        int(bundle.specs["CodeLength"]),
        observations,
        float(bundle.specs["ClampingDistance"]),
        num_iterations=int(num_iterations),
        num_samples=int(num_samples),
        lr=float(lr),
        init_std=float(init_std),
        code_reg_lambda=float(
            bundle.specs.get("CodeRegularizationLambda", 0.0)
            if code_reg_lambda is None
            else code_reg_lambda
        ),
        code_bound=bundle.specs.get("CodeBound", None),
        use_pair_forward_consistency=False,
        pair_forward_lambda=0.0,
        pair_forward_pairs_per_iter=1,
        use_pair_backward_consistency=False,
        pair_backward_lambda=0.0,
        pair_backward_pairs_per_iter=1,
        use_general_cocycle_consistency=False,
        general_cocycle_lambda=0.0,
        general_cocycle_triplets_per_iter=1,
        use_age_conditioning=bool(bundle.specs.get("UseAgeConditioning", False)),
    )
    return anchor.to(bundle.device), loss_hist, observations


def anchor_baseline_time(observations: Sequence[Dict[str, object]]) -> float:
    return min(float(item["time"]) for item in observations)


def _to_time_tensor(value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([[float(value)]], device=device, dtype=dtype)


def _to_condition_tensor(label_ad: int | float | Sequence[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(label_ad, (list, tuple, np.ndarray)):
        values = torch.tensor(label_ad, device=device, dtype=dtype).view(1, -1)
    else:
        values = torch.tensor([[float(label_ad)]], device=device, dtype=dtype)
    return values


def transport_direct(
    bundle: LoadedModel,
    anchor: torch.Tensor,
    baseline_time: float,
    target_time: float,
    target_label_ad: int | float,
) -> torch.Tensor:
    s = _to_time_tensor(baseline_time, bundle.device, anchor.dtype)
    t = _to_time_tensor(target_time, bundle.device, anchor.dtype)
    cond = _to_condition_tensor(target_label_ad, bundle.device, anchor.dtype)
    with torch.no_grad():
        return longitudinal.apply_temporal_flow(bundle.flow, anchor, s, t, age_cond=cond)


def transport_composed(
    bundle: LoadedModel,
    anchor: torch.Tensor,
    baseline_time: float,
    intermediate_times: Sequence[float],
    target_time: float,
    target_label_ad: int | float,
) -> torch.Tensor:
    times = [float(baseline_time)] + [float(t) for t in intermediate_times] + [float(target_time)]
    z = anchor
    cond = _to_condition_tensor(target_label_ad, bundle.device, anchor.dtype)
    with torch.no_grad():
        for s_val, t_val in zip(times[:-1], times[1:]):
            s = _to_time_tensor(s_val, bundle.device, anchor.dtype)
            t = _to_time_tensor(t_val, bundle.device, anchor.dtype)
            z = longitudinal.apply_temporal_flow(bundle.flow, z, s, t, age_cond=cond)
    return z


def normalize_velocity_method(method: str) -> str:
    method_name = str(method).strip().lower()
    aliases = {
        "finite_difference": "finite_difference",
        "finite-difference": "finite_difference",
        "fd": "finite_difference",
        "direct_diagonal": "direct_diagonal",
        "direct-diagonal": "direct_diagonal",
        "direct": "direct_diagonal",
        "diagonal": "direct_diagonal",
    }
    if method_name not in aliases:
        raise ValueError(
            f"Unsupported velocity method: {method}. "
            "Expected one of finite_difference, fd, direct_diagonal, direct."
        )
    return aliases[method_name]


def latent_velocity_vector_direct(
    bundle: LoadedModel,
    latent: torch.Tensor,
    current_time: float,
    label_ad: int | float,
) -> torch.Tensor:
    s = _to_time_tensor(current_time, bundle.device, latent.dtype)
    cond = _to_condition_tensor(label_ad, bundle.device, latent.dtype)
    with torch.no_grad():
        return bundle.flow(latent, s, s, age_cond=cond)


def latent_velocity_vector(
    bundle: LoadedModel,
    latent: torch.Tensor,
    current_time: float,
    label_ad: int | float,
    eps: float = DEFAULT_VELOCITY_EPS,
    method: str = DEFAULT_VELOCITY_METHOD,
) -> torch.Tensor:
    method_name = normalize_velocity_method(method)
    if method_name == "direct_diagonal":
        return latent_velocity_vector_direct(bundle, latent, current_time, label_ad)
    s = _to_time_tensor(current_time, bundle.device, latent.dtype)
    t = _to_time_tensor(current_time + float(eps), bundle.device, latent.dtype)
    cond = _to_condition_tensor(label_ad, bundle.device, latent.dtype)
    with torch.no_grad():
        z_future = longitudinal.apply_temporal_flow(bundle.flow, latent, s, t, age_cond=cond)
    return (z_future - latent) / float(eps)


def _chunk_slices(length: int, chunk_size: int) -> Iterable[slice]:
    start = 0
    while start < length:
        stop = min(length, start + int(chunk_size))
        yield slice(start, stop)
        start = stop


def decode_mesh(
    bundle: LoadedModel,
    latent: torch.Tensor,
    resolution: int = 256,
    max_batch: int = 2 ** 18,
) -> trimesh.Trimesh:
    if bundle.device.type != "cuda":
        raise RuntimeError("Mesh extraction requires a CUDA device because deep_sdf.mesh.create_mesh uses CUDA tensors.")
    mesh_out = deep_sdf_mesh.create_mesh(
        bundle.decoder,
        latent,
        filename=None,
        N=int(resolution),
        max_batch=int(max_batch),
        return_trimesh=True,
    )
    if mesh_out is None:
        raise RuntimeError("Marching cubes failed to extract a zero-level mesh.")
    return mesh_out


def task2_reference_root(experiment_dir: str | Path) -> Path:
    bridge_specs = json.loads(
        ensure_file(Path(experiment_dir) / "pretrained_task2_deepsdf" / "specs.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint_path = Path(bridge_specs["SourceTask2Checkpoint"]).resolve()
    return checkpoint_path.parents[1]


def load_task2_fitted_latent(experiment_dir: str | Path, scan_id: str) -> torch.Tensor:
    root = task2_reference_root(experiment_dir)
    latent_path = ensure_file(root / "latents" / "per_scan" / f"{scan_id}.npy")
    latent = torch.from_numpy(np.load(latent_path)).float().view(1, -1)
    if latent.shape[1] != EXPECTED_CODE_LENGTH(experiment_dir):
        raise RuntimeError(
            f"Task 2 per-scan latent has unexpected width for scan {scan_id}: {latent.shape}"
        )
    return latent


def EXPECTED_CODE_LENGTH(experiment_dir: str | Path) -> int:
    specs = ws.load_experiment_specifications(str(Path(experiment_dir).resolve()))
    return int(specs["CodeLength"])


def load_task2_reference_mesh(
    experiment_dir: str | Path,
    scan_id: str,
    device: str | torch.device | None = None,
    resolution: int = 256,
    max_batch: int = 2 ** 18,
) -> trimesh.Trimesh:
    root = task2_reference_root(experiment_dir)
    mesh_path = root / "reconstructed_meshes" / f"{scan_id}.ply"
    if mesh_path.is_file():
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"Expected trimesh.Trimesh at {mesh_path}, got {type(loaded)}")
        return loaded

    bridge_dir = Path(experiment_dir).resolve() / "pretrained_task2_deepsdf"
    bridge_specs = ws.load_experiment_specifications(str(bridge_dir))
    arch = __import__("networks." + bridge_specs["NetworkArch"], fromlist=["Decoder"])
    device_obj = determine_device(device)
    decoder = arch.Decoder(bridge_specs["CodeLength"], **bridge_specs["NetworkSpecs"]).to(device_obj)
    longitudinal.load_pretrained_decoder(decoder, str(bridge_dir), "best")
    latent = load_task2_fitted_latent(experiment_dir, scan_id).to(device_obj)
    temp_bundle = LoadedModel(
        experiment_dir=Path(experiment_dir).resolve(),
        specs=bridge_specs,
        metadata=pd.DataFrame(),
        labels={},
        device=device_obj,
        checkpoint="best",
        checkpoint_epoch=-1,
        decoder=decoder,
        flow=longitudinal.TemporalFlowMLP(bridge_specs["CodeLength"], [1], age_condition_dim=0).to(device_obj),
        train_latents=None,
        align_mode=DEFAULT_ALIGN_MODE,
        align_iters=DEFAULT_ALIGN_ITERS,
        align_trim_quantile=DEFAULT_ALIGN_TRIM_QUANTILE,
    )
    return decode_mesh(temp_bundle, latent, resolution=resolution, max_batch=max_batch)


def load_mesh(mesh_like: str | Path | trimesh.Trimesh) -> trimesh.Trimesh:
    if isinstance(mesh_like, trimesh.Trimesh):
        return mesh_like
    loaded = trimesh.load(str(mesh_like), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh at {mesh_like}, got {type(loaded)}")
    return loaded


def mesh_volume(mesh_like: str | Path | trimesh.Trimesh) -> float:
    return float(abs(load_mesh(mesh_like).volume))


def vertex_area_weights(mesh_like: str | Path | trimesh.Trimesh) -> np.ndarray:
    mesh_obj = load_mesh(mesh_like)
    weights = np.zeros(len(mesh_obj.vertices), dtype=np.float64)
    face_areas = mesh_obj.area_faces
    faces = mesh_obj.faces
    for face_idx, face in enumerate(faces):
        area_share = float(face_areas[face_idx]) / 3.0
        weights[face[0]] += area_share
        weights[face[1]] += area_share
        weights[face[2]] += area_share
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(len(mesh_obj.vertices), 1.0 / max(1, len(mesh_obj.vertices)), dtype=np.float64)
    return weights


def deterministic_surface_samples(
    mesh_like: str | Path | trimesh.Trimesh,
    num_points: int = DEFAULT_SURFACE_SAMPLES,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh_obj = load_mesh(mesh_like)
    rng = np.random.default_rng(int(seed))
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    verts = np.asarray(mesh_obj.vertices, dtype=np.float64)
    triangles = verts[faces]
    face_areas = np.asarray(mesh_obj.area_faces, dtype=np.float64)
    probs = face_areas / np.clip(face_areas.sum(), 1e-12, None)
    face_idx = rng.choice(len(faces), size=int(num_points), p=probs)
    tri = triangles[face_idx]
    u = rng.random(int(num_points))
    v = rng.random(int(num_points))
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = tri[:, 0, :] + u[:, None] * (tri[:, 1, :] - tri[:, 0, :]) + v[:, None] * (tri[:, 2, :] - tri[:, 0, :])
    normals = np.asarray(mesh_obj.face_normals, dtype=np.float64)[face_idx]
    return points.astype(np.float32), normals.astype(np.float32), face_idx.astype(np.int64)


def kabsch_rigid(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean
    h_mat = src0.T @ dst0
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        vt_mat[-1, :] *= -1
        rot = vt_mat.T @ u_mat.T
    trans = dst_mean - src_mean @ rot.T
    return rot.astype(np.float32), trans.astype(np.float32)


def rigid_icp(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    max_iterations: int = DEFAULT_ALIGN_ITERS,
    trim_quantile: float = DEFAULT_ALIGN_TRIM_QUANTILE,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    if src.shape[0] < 8 or dst.shape[0] < 8:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    tree = KDTree(dst)
    rot_total = np.eye(3, dtype=np.float32)
    trans_total = np.zeros(3, dtype=np.float32)
    for _ in range(int(max_iterations)):
        src_current = src @ rot_total.T + trans_total
        distances, nn_idx = tree.query(src_current, k=1)
        nn = dst[nn_idx]
        if trim_quantile is not None and float(trim_quantile) < 1.0:
            threshold = np.quantile(distances, float(trim_quantile))
            keep = distances <= threshold
            if int(keep.sum()) >= 16:
                lhs = src_current[keep]
                rhs = nn[keep]
            else:
                lhs = src_current
                rhs = nn
        else:
            lhs = src_current
            rhs = nn
        delta_rot, delta_trans = kabsch_rigid(lhs, rhs)
        rot_total = delta_rot @ rot_total
        trans_total = trans_total @ delta_rot.T + delta_trans
        if np.linalg.norm(delta_trans) < tol and np.linalg.norm(delta_rot - np.eye(3)) < 1e-4:
            break
    return rot_total, trans_total


def align_points(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    mode: str = DEFAULT_ALIGN_MODE,
    max_iterations: int = DEFAULT_ALIGN_ITERS,
    trim_quantile: float = DEFAULT_ALIGN_TRIM_QUANTILE,
) -> np.ndarray:
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    mode_l = str(mode).lower()
    if mode_l in ("none", "off"):
        return src
    if mode_l in ("centroid", "translation", "translate"):
        return src + (dst.mean(axis=0) - src.mean(axis=0))
    if mode_l == "rigid":
        rot, trans = rigid_icp(src, dst, max_iterations=max_iterations, trim_quantile=trim_quantile)
        return src @ rot.T + trans
    raise ValueError(f"Unknown alignment mode: {mode}")


def deterministic_mesh_metrics(
    gt_mesh: str | Path | trimesh.Trimesh,
    pred_mesh: str | Path | trimesh.Trimesh,
    num_samples: int = DEFAULT_SURFACE_SAMPLES,
    seed: int = 0,
    align_mode: str = DEFAULT_ALIGN_MODE,
    align_iters: int = DEFAULT_ALIGN_ITERS,
    align_trim_quantile: float = DEFAULT_ALIGN_TRIM_QUANTILE,
) -> Dict[str, float]:
    gt_points, _, _ = deterministic_surface_samples(gt_mesh, num_points=num_samples, seed=seed)
    pred_points, _, _ = deterministic_surface_samples(pred_mesh, num_points=num_samples, seed=seed + 1)
    pred_points_aligned = align_points(
        pred_points,
        gt_points,
        mode=align_mode,
        max_iterations=align_iters,
        trim_quantile=align_trim_quantile,
    )
    pred_tree = KDTree(pred_points_aligned)
    gt_tree = KDTree(gt_points)
    d_gt_to_pred, _ = pred_tree.query(gt_points, k=1)
    d_pred_to_gt, _ = gt_tree.query(pred_points_aligned, k=1)
    sym = np.concatenate([d_gt_to_pred, d_pred_to_gt], axis=0)
    return {
        "chamfer_aligned": float(np.mean(d_gt_to_pred ** 2) + np.mean(d_pred_to_gt ** 2)),
        "assd_aligned": float(np.mean(sym)),
        "hd95_aligned": float(np.percentile(sym, 95.0)),
        "pred_volume": mesh_volume(pred_mesh),
        "target_volume": mesh_volume(gt_mesh),
        "pred_point_count": int(pred_points.shape[0]),
        "target_point_count": int(gt_points.shape[0]),
    }


def summarize_speed_map(
    mesh_like: str | Path | trimesh.Trimesh,
    yearly_normal_speed: np.ndarray,
) -> Dict[str, float]:
    mesh_obj = load_mesh(mesh_like)
    speed = np.asarray(yearly_normal_speed, dtype=np.float64).reshape(-1)
    if speed.shape[0] != len(mesh_obj.vertices):
        raise ValueError(
            f"Speed map length {speed.shape[0]} does not match vertex count {len(mesh_obj.vertices)}"
        )
    weights = vertex_area_weights(mesh_obj)
    weight_sum = float(np.clip(weights.sum(), 1e-12, None))
    rms = math.sqrt(float(np.sum(weights * (speed ** 2)) / weight_sum))
    mean_abs = float(np.sum(weights * np.abs(speed)) / weight_sum)
    net_volume_rate = float(np.sum(weights * speed))
    contracting = float(np.sum(weights[speed < 0.0]) / weight_sum)
    expanding = float(np.sum(weights[speed > 0.0]) / weight_sum)
    return {
        "area_weighted_rms_speed_per_year": rms,
        "mean_absolute_speed_per_year": mean_abs,
        "net_volume_rate_per_year": net_volume_rate,
        "contracting_area_fraction": contracting,
        "expanding_area_fraction": expanding,
    }


def bootstrap_subject_ci(
    frame: pd.DataFrame,
    value_col: str,
    group_col: str = "subject_id",
    iterations: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    if frame.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    subject_values = (
        frame.groupby(group_col, sort=True)[value_col].mean().dropna().to_numpy(dtype=float)
    )
    if subject_values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(iterations)):
        sample = rng.choice(subject_values, size=subject_values.size, replace=True)
        draws.append(float(sample.mean()))
    return {
        "count": int(subject_values.size),
        "mean": float(subject_values.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def _decoder_chunk_forward(
    decoder: torch.nn.Module,
    latent: torch.Tensor,
    xyz_chunk: torch.Tensor,
) -> torch.Tensor:
    latent_expanded = latent.expand(xyz_chunk.shape[0], -1)
    return decoder(torch.cat([latent_expanded, xyz_chunk], dim=1))


def implicit_surface_normal_velocity(
    bundle: LoadedModel,
    latent: torch.Tensor,
    current_time: float,
    label_ad: int | float,
    query_points: np.ndarray,
    eps: float = DEFAULT_VELOCITY_EPS,
    chunk_size: int = 4096,
    yearly: bool = True,
    method: str = DEFAULT_VELOCITY_METHOD,
) -> np.ndarray:
    points = np.asarray(query_points, dtype=np.float32)
    speeds: List[np.ndarray] = []
    dz_dt = latent_velocity_vector(
        bundle,
        latent,
        current_time,
        label_ad,
        eps=eps,
        method=method,
    )
    scale = 1.0 / TRAINING_AGE_RANGE_YEARS if yearly else 1.0

    for slc in _chunk_slices(len(points), chunk_size):
        xyz = torch.from_numpy(points[slc]).to(bundle.device)
        xyz = xyz.clone().detach().requires_grad_(True)
        z_rep = latent.expand(xyz.shape[0], -1).clone().detach().requires_grad_(True)
        sdf = bundle.decoder(torch.cat([z_rep, xyz], dim=1))
        grad_outputs = torch.ones_like(sdf)
        grad_x, grad_z = torch.autograd.grad(
            outputs=sdf,
            inputs=[xyz, z_rep],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )
        numerator = torch.sum(grad_z * dz_dt.expand_as(grad_z), dim=1)
        denom = torch.clamp(grad_x.norm(dim=1), min=1e-8)
        vn = -(numerator / denom) * scale
        speeds.append(vn.detach().cpu().numpy())
    return np.concatenate(speeds, axis=0)


def finite_step_surface_change(
    bundle: LoadedModel,
    latent_current: torch.Tensor,
    latent_future: torch.Tensor,
    query_points: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    points = np.asarray(query_points, dtype=np.float32)
    displacements: List[np.ndarray] = []
    for slc in _chunk_slices(len(points), chunk_size):
        xyz = torch.from_numpy(points[slc]).to(bundle.device)
        xyz = xyz.clone().detach().requires_grad_(True)
        z_current = latent_current.expand(xyz.shape[0], -1).clone().detach()
        z_future = latent_future.expand(xyz.shape[0], -1).clone().detach()
        sdf_current = bundle.decoder(torch.cat([z_current, xyz], dim=1))
        sdf_future = bundle.decoder(torch.cat([z_future, xyz], dim=1))
        grad_x = torch.autograd.grad(
            outputs=sdf_current,
            inputs=xyz,
            grad_outputs=torch.ones_like(sdf_current),
            create_graph=False,
            retain_graph=False,
        )[0]
        denom = torch.clamp(grad_x.norm(dim=1), min=1e-8)
        disp = -sdf_future.squeeze(1) / denom
        displacements.append(disp.detach().cpu().numpy())
    return np.concatenate(displacements, axis=0)


def yearly_speed_from_normalized_speed(speed: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(speed):
        speed = speed.detach().cpu().numpy()
    return np.asarray(speed, dtype=np.float64) / TRAINING_AGE_RANGE_YEARS


def forecast_time_delta_to_years(delta_time_norm: float) -> float:
    return float(delta_time_norm) * TRAINING_AGE_RANGE_YEARS


def one_year_latent(
    bundle: LoadedModel,
    latent: torch.Tensor,
    current_time: float,
    label_ad: int | float,
) -> torch.Tensor:
    s = _to_time_tensor(current_time, bundle.device, latent.dtype)
    t = _to_time_tensor(current_time + ONE_YEAR_NORM_DELTA, bundle.device, latent.dtype)
    cond = _to_condition_tensor(label_ad, bundle.device, latent.dtype)
    with torch.no_grad():
        return longitudinal.apply_temporal_flow(bundle.flow, latent, s, t, age_cond=cond)


def six_month_latent(
    bundle: LoadedModel,
    latent: torch.Tensor,
    current_time: float,
    label_ad: int | float,
) -> torch.Tensor:
    s = _to_time_tensor(current_time, bundle.device, latent.dtype)
    t = _to_time_tensor(current_time + SIX_MONTH_NORM_DELTA, bundle.device, latent.dtype)
    cond = _to_condition_tensor(label_ad, bundle.device, latent.dtype)
    with torch.no_grad():
        return longitudinal.apply_temporal_flow(bundle.flow, latent, s, t, age_cond=cond)


def observed_correspondence_speed(
    source_mesh: str | Path | trimesh.Trimesh,
    target_mesh: str | Path | trimesh.Trimesh,
    delta_years: float,
) -> Dict[str, object]:
    source = load_mesh(source_mesh)
    target = load_mesh(target_mesh)
    if source.vertices.shape != target.vertices.shape:
        raise ValueError(
            f"Observed correspondence speed requires matching vertex shapes, got {source.vertices.shape} and {target.vertices.shape}"
        )
    normals = np.asarray(source.vertex_normals, dtype=np.float64)
    displacement = np.asarray(target.vertices - source.vertices, dtype=np.float64)
    speed = np.sum(displacement * normals, axis=1) / max(float(delta_years), 1e-8)
    summary = summarize_speed_map(source, speed)
    summary.update({"speed": speed.astype(np.float32)})
    return summary


def model_speed_map_for_scan(
    bundle: LoadedModel,
    anchor: torch.Tensor,
    baseline_time: float,
    scan_row: pd.Series,
    yearly: bool = True,
) -> Dict[str, object]:
    current_time = float(scan_row["continuous_age_norm"])
    label_ad = int(scan_row["label_ad"])
    latent_now = transport_direct(bundle, anchor, baseline_time, current_time, label_ad)
    mesh_obj = load_mesh(scan_row["mesh_path"])
    speed = implicit_surface_normal_velocity(
        bundle,
        latent_now,
        current_time,
        label_ad,
        np.asarray(mesh_obj.vertices, dtype=np.float32),
        yearly=yearly,
    )
    summary = summarize_speed_map(mesh_obj, speed)
    summary.update({"latent": latent_now, "speed": speed.astype(np.float32), "mesh": mesh_obj})
    return summary


def compute_group_summary(
    frame: pd.DataFrame,
    metric_cols: Sequence[str],
    diagnosis_col: str = "diagnosis",
    subject_col: str = "subject_id",
    bootstrap_iterations: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cohort in ("all", "CN", "AD"):
        cohort_frame = frame if cohort == "all" else frame.loc[frame[diagnosis_col] == cohort]
        if cohort_frame.empty:
            continue
        row: Dict[str, object] = {
            "cohort": cohort,
            "num_rows": int(len(cohort_frame)),
            "num_subjects": int(cohort_frame[subject_col].nunique()),
        }
        for metric_idx, metric_col in enumerate(metric_cols):
            valid = cohort_frame[[subject_col, metric_col]].dropna()
            row[f"{metric_col}_mean"] = float(valid[metric_col].mean()) if not valid.empty else float("nan")
            ci = bootstrap_subject_ci(
                valid,
                metric_col,
                group_col=subject_col,
                iterations=bootstrap_iterations,
                seed=seed + metric_idx,
            )
            row[f"{metric_col}_ci_low"] = ci["ci_low"]
            row[f"{metric_col}_ci_high"] = ci["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def mesh_figure(
    mesh_like: str | Path | trimesh.Trimesh,
    scalars: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    colorscale: str = "Viridis",
) -> "go.Figure":
    if go is None:
        raise RuntimeError("plotly is not available in this environment")
    mesh_obj = load_mesh(mesh_like)
    verts = np.asarray(mesh_obj.vertices)
    faces = np.asarray(mesh_obj.faces)
    kwargs = {}
    if scalars is not None:
        kwargs.update(
            {
                "intensity": np.asarray(scalars, dtype=float),
                "colorscale": colorscale,
                "showscale": True,
            }
        )
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=1.0,
                **kwargs,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene_aspectmode="data",
        template="plotly_white",
    )
    return fig


def trajectory_figure(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str = "diagnosis",
    line_group_col: str = "subject_id",
    title: Optional[str] = None,
) -> "go.Figure":
    if go is None:
        raise RuntimeError("plotly is not available in this environment")
    fig = go.Figure()
    for diagnosis, group in frame.groupby(color_col, sort=True):
        for subject_id, subject_group in group.groupby(line_group_col, sort=True):
            ordered = subject_group.sort_values(x_col)
            fig.add_trace(
                go.Scatter(
                    x=ordered[x_col],
                    y=ordered[y_col],
                    mode="lines+markers",
                    name=f"{diagnosis}:{subject_id}",
                    legendgroup=str(diagnosis),
                    showlegend=False,
                )
            )
    fig.update_layout(title=title, template="plotly_white")
    return fig
