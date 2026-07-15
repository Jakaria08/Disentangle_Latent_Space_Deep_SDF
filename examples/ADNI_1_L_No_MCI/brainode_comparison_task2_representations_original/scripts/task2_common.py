#!/usr/bin/env python3
from __future__ import annotations

import csv
import copy
import hashlib
import importlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_task_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else TASK_DIR / path


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = load_json(config_path)
    config["_config_path"] = str(config_path)
    return config


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = resolve_repo_path(path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "scan_id",
        "filename",
        "subject_id",
        "image_id",
        "split",
        "diagnosis",
        "label_ad",
        "visit_order",
        "age_norm",
        "mesh_path",
        "sdf_npz_path",
    }
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    split_order = {"train": 0, "val": 1, "test": 2}
    return sorted(
        rows,
        key=lambda row: (
            split_order.get(row["split"], 99),
            row["subject_id"],
            int(row["visit_order"]),
            row["scan_id"],
        ),
    )


def rows_for_split(
    rows: Iterable[dict[str, str]], split: str
) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


def stable_seed(text: str, base_seed: int = 0) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def choose_device(requested: str | None = None):
    import torch

    if requested:
        device = torch.device(requested)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_decoder(config: dict[str, Any], device):
    module = importlib.import_module(f"networks.{config['network_arch']}")
    decoder = module.Decoder(
        int(config["latent_size"]), **copy.deepcopy(config["network_specs"])
    )
    return decoder.to(device)


def strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def checkpoint_path(output_dir: Path, checkpoint: str) -> Path:
    value = Path(checkpoint)
    if value.is_file():
        return value
    name = checkpoint if checkpoint.endswith(".pth") else f"{checkpoint}.pth"
    return output_dir / "checkpoints" / name


def load_decoder_checkpoint(
    config: dict[str, Any], checkpoint: str, device
):
    import torch

    output_dir = resolve_repo_path(config["output_dir"])
    path = checkpoint_path(output_dir, checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device)
    decoder = build_decoder(config, device)
    state = strip_module_prefix(payload.get("model_state_dict", payload))
    decoder.load_state_dict(state)
    decoder.eval()
    return decoder, payload, path


def load_obj_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                face = []
                for token in line.split()[1:4]:
                    face.append(int(token.split("/")[0]) - 1)
                faces.append(face)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def load_sdf_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        if "pos" not in archive or "neg" not in archive:
            raise ValueError(f"SDF archive lacks pos/neg arrays: {path}")
        pos = np.asarray(archive["pos"], dtype=np.float32)
        neg = np.asarray(archive["neg"], dtype=np.float32)
    pos = pos[np.isfinite(pos).all(axis=1)]
    neg = neg[np.isfinite(neg).all(axis=1)]
    return pos, neg


def sample_balanced_numpy(
    pos: np.ndarray,
    neg: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    half = count // 2
    pos_idx = rng.integers(0, len(pos), size=half, endpoint=False)
    neg_idx = rng.integers(0, len(neg), size=count - half, endpoint=False)
    samples = np.concatenate((pos[pos_idx], neg[neg_idx]), axis=0)
    rng.shuffle(samples)
    return samples


def split_fit_holdout(
    array: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(len(array))
    holdout_count = max(1, int(round(len(array) * fraction)))
    holdout_count = min(holdout_count, len(array) - 1)
    return array[order[holdout_count:]], array[order[:holdout_count]]


def evaluate_sdf_l1(
    decoder,
    latent,
    samples: np.ndarray,
    clamp_distance: float,
    device,
    batch_size: int = 65536,
) -> float:
    import torch

    losses = []
    decoder.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = torch.from_numpy(samples[start : start + batch_size]).to(device)
            xyz = batch[:, :3]
            target = batch[:, 3:4].clamp(-clamp_distance, clamp_distance)
            code = latent.expand(len(batch), -1)
            pred = decoder(torch.cat((code, xyz), dim=1))
            pred = pred.clamp(-clamp_distance, clamp_distance)
            losses.append(torch.abs(pred - target).detach().cpu().numpy())
    return float(np.concatenate(losses, axis=0).mean())


def fit_single_latent(
    decoder,
    sdf_path: str | Path,
    latent_size: int,
    fit_config: dict[str, Any],
    clamp_distance: float,
    device,
    seed: int,
    steps_override: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    rng = np.random.default_rng(seed)
    pos, neg = load_sdf_arrays(sdf_path)
    holdout_fraction = float(fit_config.get("holdout_fraction", 0.1))
    pos_fit, pos_holdout = split_fit_holdout(pos, holdout_fraction, rng)
    neg_fit, neg_holdout = split_fit_holdout(neg, holdout_fraction, rng)

    latent = torch.empty(1, latent_size, device=device)
    torch.nn.init.normal_(
        latent, mean=0.0, std=float(fit_config.get("initial_std", 0.01))
    )
    latent.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [latent], lr=float(fit_config.get("learning_rate", 0.005))
    )
    steps = int(steps_override or fit_config["steps"])
    samples_per_step = int(fit_config["samples_per_step"])
    reg_lambda = float(fit_config.get("code_regularization_lambda", 0.0))
    code_bound = fit_config.get("code_bound")
    patience = int(fit_config.get("early_stop_patience", steps))
    min_delta = float(fit_config.get("early_stop_min_delta", 0.0))
    best_loss = math.inf
    best_latent = None
    no_improvement = 0
    initial_data_loss = None
    final_data_loss = None

    decoder.eval()
    for step in range(steps):
        batch_np = sample_balanced_numpy(pos_fit, neg_fit, samples_per_step, rng)
        batch = torch.from_numpy(batch_np).to(device)
        xyz = batch[:, :3]
        target = batch[:, 3:4].clamp(-clamp_distance, clamp_distance)

        optimizer.zero_grad(set_to_none=True)
        pred = decoder(torch.cat((latent.expand(len(batch), -1), xyz), dim=1))
        pred = pred.clamp(-clamp_distance, clamp_distance)
        data_loss = torch.mean(torch.abs(pred - target))
        loss = data_loss + reg_lambda * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if code_bound is not None and float(code_bound) > 0:
            with torch.no_grad():
                norm = latent.norm(dim=1, keepdim=True)
                latent.mul_(torch.clamp(float(code_bound) / (norm + 1e-12), max=1.0))

        value = float(loss.detach().cpu())
        if initial_data_loss is None:
            initial_data_loss = float(data_loss.detach().cpu())
        final_data_loss = float(data_loss.detach().cpu())
        if value < best_loss - min_delta:
            best_loss = value
            best_latent = latent.detach().clone()
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            break

    if best_latent is None:
        raise RuntimeError(f"Latent optimization failed for {sdf_path}")
    latent = best_latent
    eval_count = int(fit_config.get("evaluation_samples", 32768))
    eval_samples = sample_balanced_numpy(
        pos_holdout, neg_holdout, eval_count, rng
    )
    heldout_l1 = evaluate_sdf_l1(
        decoder, latent, eval_samples, clamp_distance, device
    )
    result = latent.detach().cpu().numpy().reshape(-1).astype(np.float32)
    stats = {
        "steps_requested": steps,
        "steps_completed": step + 1,
        "initial_data_l1": initial_data_loss,
        "final_data_l1": final_data_loss,
        "best_objective": best_loss,
        "heldout_sdf_l1": heldout_l1,
        "latent_norm": float(np.linalg.norm(result)),
        "finite": bool(np.isfinite(result).all()),
    }
    return result, stats


def decode_latent_to_mesh(
    decoder,
    latent: np.ndarray,
    output_path: str | Path,
    resolution: int,
    max_batch: int,
    device,
) -> dict[str, Any]:
    import torch
    import trimesh
    from skimage.measure import marching_cubes

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    latent_tensor = torch.from_numpy(np.asarray(latent, dtype=np.float32))
    latent_tensor = latent_tensor.reshape(1, -1).to(device)
    total = resolution**3
    values = np.empty(total, dtype=np.float32)
    step = 2.0 / (resolution - 1)

    decoder.eval()
    with torch.no_grad():
        for start in range(0, total, max_batch):
            stop = min(start + max_batch, total)
            index = torch.arange(start, stop, device=device, dtype=torch.long)
            x = torch.div(index, resolution * resolution, rounding_mode="floor")
            y = torch.div(index, resolution, rounding_mode="floor") % resolution
            z = index % resolution
            xyz = torch.stack((x, y, z), dim=1).to(torch.float32)
            xyz = xyz * step - 1.0
            code = latent_tensor.expand(len(index), -1)
            pred = decoder(torch.cat((code, xyz), dim=1)).squeeze(1)
            values[start:stop] = pred.detach().cpu().numpy()

    volume = values.reshape(resolution, resolution, resolution)
    value_min = float(np.min(volume))
    value_max = float(np.max(volume))
    if not (value_min <= 0.0 <= value_max):
        raise RuntimeError(
            f"No zero level set for {output.name}: range [{value_min}, {value_max}]"
        )
    vertices, faces, _normals, _values = marching_cubes(
        volume, level=0.0, spacing=(step, step, step), method="lewiner"
    )
    vertices += np.asarray([-1.0, -1.0, -1.0], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(output)
    return {
        "mesh_path": str(output),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "sdf_grid_min": value_min,
        "sdf_grid_max": value_max,
    }


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
