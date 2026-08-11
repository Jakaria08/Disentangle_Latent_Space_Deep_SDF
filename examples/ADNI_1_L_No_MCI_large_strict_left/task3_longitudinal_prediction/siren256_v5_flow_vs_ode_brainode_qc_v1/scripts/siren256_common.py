"""Input contract, data access, and immutable helpers for the SIREN-256 study."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_dir() -> Path:
    root = root_dir()
    for parent in root.parents:
        if (parent / "networks").is_dir() and (parent / "longitudinal_direct_flow.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate Deep3DComp repository root.")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base or root_dir()) / path


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = read_json(path)
    common = config.get("CommonConfig")
    if common:
        shared = read_json(path.parent / str(common))
        shared.update(config)
        config = shared
    config["_config_path"] = str(path)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_prepared(root: Path | None = None) -> None:
    root = root or root_dir()
    required = [
        root / "metadata" / "input_contract.json",
        root / "metadata" / "scan_manifest.csv",
        root / "metadata" / "train_only_transport_basis.npz",
        root / "metadata" / "loss_scales.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Inputs are not prepared. Run prepare_siren256_v5_inputs.py first:\n" + "\n".join(missing))


def load_basis(root: Path | None = None) -> dict[str, np.ndarray]:
    root = root or root_dir()
    archive = np.load(root / "metadata" / "train_only_transport_basis.npz", allow_pickle=False)
    return {key: np.asarray(archive[key]) for key in archive.files}


def source_cache_path(config: dict[str, Any]) -> Path:
    return resolve(config["RegisteredMeshCache"])


def load_cache(config: dict[str, Any], *, mmap: bool = True) -> dict[str, np.ndarray]:
    archive = np.load(source_cache_path(config), allow_pickle=False, mmap_mode="r" if mmap else None)
    return {key: archive[key] for key in archive.files}


def _read_obj_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif fields[0] == "f" and len(fields) >= 4:
                faces.append([int(value.split("/")[0]) - 1 for value in fields[1:4]])
    if not vertices or not faces:
        raise ValueError(f"Mesh lacks vertices/faces: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def _mesh_normals_areas_volume(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    areas = np.zeros(len(vertices), dtype=np.float64)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
        np.add.at(areas, faces[:, corner], face_areas / 3.0)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    volume = abs(float(np.einsum("fi,fi->", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) / 6.0))
    return normals.astype(np.float32), areas.astype(np.float32), volume


def load_split_cache(config: dict[str, Any], splits: tuple[str, ...], root: Path | None = None) -> dict[str, Any]:
    """Build an in-memory cache from only the allowed split's raw inputs.

    This intentionally does *not* open the all-split registered-mesh archive.
    It reads only the allowed latent archives and registered OBJ paths, so the
    trainer cannot access test latent or mesh arrays through a shared cache.
    """
    root = root or root_dir()
    allowed = tuple(sorted(set(splits)))
    if not set(allowed).issubset({"train", "val", "test"}):
        raise ValueError(f"Unknown split selection: {splits}")
    manifest = pd.read_csv(root / "metadata" / "scan_manifest.csv")
    selected = manifest.loc[manifest.split.isin(allowed)].sort_values("cache_index").reset_index(drop=True)
    latent_by_id: dict[str, np.ndarray] = {}
    for split in allowed:
        archive = np.load(resolve(config["LatentArchives"][split]), allow_pickle=False)
        for scan_id, latent in zip(archive["scan_ids"], archive["latents"]):
            key = Path(str(scan_id)).stem
            latent_by_id[key] = np.asarray(latent, dtype=np.float32)
    if set(selected.scan_id.astype(str)) != set(latent_by_id):
        raise ValueError("Allowed metadata and latent archives are not exactly aligned.")
    vertices, normals, areas, volumes = [], [], [], []
    faces: np.ndarray | None = None
    for row in selected.itertuples(index=False):
        mesh_vertices, mesh_faces = _read_obj_mesh(row.mesh_path)
        if faces is None:
            faces = mesh_faces
        elif not np.array_equal(faces, mesh_faces) or mesh_vertices.shape != vertices[0].shape:
            raise ValueError(f"Registered topology mismatch: {row.scan_id}")
        mesh_normals, mesh_areas, mesh_volume = _mesh_normals_areas_volume(mesh_vertices, mesh_faces)
        vertices.append(mesh_vertices)
        normals.append(mesh_normals)
        areas.append(mesh_areas)
        volumes.append(mesh_volume)
    if faces is None:
        raise ValueError("No meshes in allowed split cache.")
    return {
        "scan_ids": selected.scan_id.astype(str).to_numpy(),
        "cache_index_to_local": {int(global_index): local for local, global_index in enumerate(selected.cache_index)},
        "latents": np.stack([latent_by_id[str(scan)] for scan in selected.scan_id], axis=0),
        "vertices": np.asarray(vertices, dtype=np.float32), "normals": np.asarray(normals, dtype=np.float32),
        "areas": np.asarray(areas, dtype=np.float32), "volumes": np.asarray(volumes, dtype=np.float32),
        "faces": faces.astype(np.int64), "times": selected.continuous_age_norm.to_numpy(dtype=np.float32),
        "labels": selected.label_ad.to_numpy(dtype=np.int64), "allowed_splits": allowed,
    }


def load_frozen_decoder(config: dict[str, Any], device: torch.device) -> nn.Module:
    repo_dir()
    source_specs = read_json(resolve(config["SourceExperimentDir"]) / "specs.json")
    architecture = __import__("networks." + str(source_specs["NetworkArch"]), fromlist=["Decoder"])
    decoder = architecture.Decoder(int(config["LatentSize"]), **source_specs["NetworkSpecs"]).to(device)
    payload = torch.load(resolve(config["DecoderCheckpoint"]), map_location="cpu")
    state = {key.removeprefix("module."): value for key, value in payload["model_state_dict"].items()}
    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder


def decode_sdf(decoder: nn.Module, latent: torch.Tensor, points: torch.Tensor, chunk: int = 131072) -> torch.Tensor:
    if latent.ndim != 2 or points.ndim != 3 or latent.shape[0] != points.shape[0] or points.shape[2] != 3:
        raise ValueError("decode_sdf requires latents [B,256] and points [B,N,3].")
    batch, count = points.shape[:2]
    joined = torch.cat((latent[:, None, :].expand(batch, count, -1), points), dim=2).reshape(-1, latent.shape[1] + 3)
    chunks = [decoder(joined[start : start + chunk]) for start in range(0, len(joined), chunk)]
    return torch.cat(chunks, dim=0).reshape(batch, count)


def signed_mesh_volume(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    triangles = vertices[:, faces]
    signed = (triangles[:, :, 0] * torch.cross(triangles[:, :, 1], triangles[:, :, 2], dim=-1)).sum(dim=-1).sum(dim=1) / 6.0
    return signed.abs().clamp_min(1.0e-8)


def sample_sdf(path: str, count: int, seed: int | None) -> np.ndarray:
    if count <= 0 or count % 2:
        raise ValueError("SDF sample count must be a positive even integer.")
    with np.load(path, allow_pickle=False) as archive:
        pos, neg = np.asarray(archive["pos"], dtype=np.float32), np.asarray(archive["neg"], dtype=np.float32)
    pos, neg = pos[np.isfinite(pos).all(axis=1)], neg[np.isfinite(neg).all(axis=1)]
    if not len(pos) or not len(neg):
        raise ValueError(f"No finite positive/negative SDF samples: {path}")
    rng = np.random.default_rng(seed)
    half = count // 2
    return np.concatenate((pos[rng.integers(len(pos), size=half)], neg[rng.integers(len(neg), size=count - half)])).astype(np.float32)


def load_pairs(split: str, *, include_backward: bool = False, root: Path | None = None) -> pd.DataFrame:
    root = root or root_dir()
    pairs = pd.read_csv(root / "metadata" / f"pair_records_{split}.csv")
    if include_backward:
        backward = root / "metadata" / f"pair_records_{split}_backward.csv"
        if not backward.exists() and split == "train":
            backward = root / "metadata" / "pair_records_train_backward.csv"
        if not backward.exists():
            raise FileNotFoundError(f"No backward pair table for split {split}.")
        pairs = pd.concat((pairs, pd.read_csv(backward)), ignore_index=True)
    return pairs.reset_index(drop=True)


class RegisteredPairDataset(Dataset[dict[str, Any]]):
    def __init__(self, config: dict[str, Any], cache: dict[str, Any], pairs: pd.DataFrame, samples: int, deterministic: bool, seed: int) -> None:
        self.pairs, self.samples, self.deterministic, self.seed = pairs.reset_index(drop=True), int(samples), bool(deterministic), int(seed)
        self.latents, self.vertices, self.normals, self.areas, self.volumes = (cache[key] for key in ("latents", "vertices", "normals", "areas", "volumes"))
        self.index_map = dict(cache.get("cache_index_to_local", {}))
        manifest = pd.read_csv(root_dir() / "metadata" / "scan_manifest.csv")
        self.sdf_paths = dict(zip(manifest.cache_index.astype(int), manifest.sdf_npz_path.astype(str)))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.pairs.iloc[index]
        source_global, target_global = int(item.source_cache_index), int(item.target_cache_index)
        if self.index_map and (source_global not in self.index_map or target_global not in self.index_map):
            raise KeyError("Pair accesses a scan outside the allowed split cache.")
        source, target = self.index_map.get(source_global, source_global), self.index_map.get(target_global, target_global)
        seed = None
        if self.deterministic:
            seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{index}:{target}".encode()).digest()[:8], "little")
        return {
            "source_latent": torch.from_numpy(np.array(self.latents[source], copy=True)),
            "target_latent": torch.from_numpy(np.array(self.latents[target], copy=True)),
            "source_vertices": torch.from_numpy(np.array(self.vertices[source], copy=True)),
            "target_vertices": torch.from_numpy(np.array(self.vertices[target], copy=True)),
            "source_normals": torch.from_numpy(np.array(self.normals[source], copy=True)),
            "vertex_areas": torch.from_numpy(np.array(self.areas[source], copy=True)),
            "source_volume": torch.tensor(float(self.volumes[source]), dtype=torch.float32),
            "target_volume": torch.tensor(float(self.volumes[target]), dtype=torch.float32),
            "source_time": torch.tensor(float(item.source_time), dtype=torch.float32),
            "target_time": torch.tensor(float(item.target_time), dtype=torch.float32),
            "gap_years": torch.tensor(float(item.gap_years), dtype=torch.float32),
            "condition": torch.tensor([float(item.label_ad)], dtype=torch.float32),
            "observed_cache_index": torch.tensor(int(item.observed_cache_index), dtype=torch.long),
            "observed_time": torch.tensor(float(item.observed_time), dtype=torch.float32),
            "target_samples": torch.from_numpy(sample_sdf(self.sdf_paths[target], self.samples, seed)),
            "subject_id": str(item.subject_id), "diagnosis": str(item.diagnosis),
            "is_first_last": bool(item.is_first_last), "is_adjacent": bool(item.is_adjacent),
        }


class BalancedPairBatchSampler(Sampler[list[int]]):
    """Equal CN/AD and adjacent/non-adjacent, then equal subject probability."""

    def __init__(self, pairs: pd.DataFrame, batch_size: int, batches: int, seed: int) -> None:
        self.batch_size, self.batches, self.seed, self.epoch = int(batch_size), int(batches), int(seed), 0
        self.groups: dict[tuple[int, int], dict[str, list[int]]] = {}
        for index, row in pairs.iterrows():
            key = (int(row.label_ad), int(bool(row.is_adjacent)))
            self.groups.setdefault(key, {}).setdefault(str(row.subject_id), []).append(int(index))
        if set(self.groups) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError(f"Training pair strata must all exist; got {sorted(self.groups)}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng, strata = np.random.default_rng(self.seed + 104729 * self.epoch), sorted(self.groups)
        for batch_index in range(self.batches):
            chosen: list[int] = []
            for slot in range(self.batch_size):
                group = self.groups[strata[(batch_index * self.batch_size + slot) % len(strata)]]
                subject = sorted(group)[int(rng.integers(len(group)))]
                choices = group[subject]
                chosen.append(choices[int(rng.integers(len(choices)))])
            yield chosen


def load_sequences(split: str, root: Path | None = None) -> list[dict[str, Any]]:
    root = root or root_dir()
    archive = np.load(root / "metadata" / f"subject_sequences_{split}.npz", allow_pickle=False)
    offsets, ids, indices, times, labels = (archive[key] for key in ("offsets", "subject_ids", "cache_indices", "times", "labels"))
    return [{"subject_id": str(ids[i]), "cache_indices": indices[offsets[i] : offsets[i + 1]], "times": times[offsets[i] : offsets[i + 1]], "label_ad": int(labels[i])} for i in range(len(ids))]


def balanced_sequence(rng: np.random.Generator, sequences: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [[item for item in sequences if int(item["label_ad"]) == label] for label in (0, 1)]
    selected = candidates[int(rng.integers(2))]
    return selected[int(rng.integers(len(selected)))]


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
