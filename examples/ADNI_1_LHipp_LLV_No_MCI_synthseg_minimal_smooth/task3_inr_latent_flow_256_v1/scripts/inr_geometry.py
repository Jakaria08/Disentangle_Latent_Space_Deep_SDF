#!/usr/bin/env python3
"""Frozen multiresolution INR adapter for standardized 256-D transport states."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import common as C


class FrozenINRGeometry(nn.Module):
    def __init__(self, train_archive: dict[str, np.ndarray], device: torch.device, registry: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.registry = C.load_registry() if registry is None else registry
        config = C.read_json(self.registry["decoder_config"])
        module = importlib.import_module(f"networks.{config['network_arch']}")
        specs = dict(config["network_specs"])
        specs.pop("grid_resolution", None)
        specs.pop("sampling_balance_resolution", None)
        decoder = module.Decoder(int(config["latent_size"]), **specs).to(device)
        payload = torch.load(C.resolve_path(self.registry["decoder_checkpoint"]), map_location=device, weights_only=False)
        state = payload.get("model_state_dict", payload)
        decoder.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
        decoder.eval()
        decoder.requires_grad_(False)
        self.decoder = decoder
        self.checkpoint_epoch = int(payload.get("epoch", 0))
        self.level_weights = tuple(float(value) for value in payload.get("level_weights", [1.0] * len(decoder.grid_resolutions)))
        self.code_bound = float(self.registry["code_bound"])
        self.register_buffer("latent_mean", torch.from_numpy(train_archive["train_latent_mean_256"].astype(np.float32)).reshape(1, -1))
        self.register_buffer("latent_std", torch.from_numpy(train_archive["train_latent_std_256"].astype(np.float32)).reshape(1, -1))
        with C.resolve_path(self.registry["rescale_details_csv"]).open(encoding="utf-8", newline="") as handle:
            scaling = next(csv.DictReader(handle))
        self.target_range_min = float(scaling["target_range_min"])
        self.range_global_min = float(scaling["range_global_min"])
        self.range_linear_scale_factor = float(scaling["range_linear_scale_factor"])
        self.distance_unscale_factor = float(scaling["distance_unscale_factor"])
        self.linear_normalized_to_mm = self.distance_unscale_factor / self.range_linear_scale_factor

    def train(self, mode: bool = True):
        super().train(False)
        self.decoder.eval()
        return self

    def raw_unprojected(self, standardized: torch.Tensor) -> torch.Tensor:
        return standardized * self.latent_std.to(standardized) + self.latent_mean.to(standardized)

    def raw_projected(self, standardized: torch.Tensor) -> torch.Tensor:
        raw = self.raw_unprojected(standardized)
        norm = torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(1.0e-12)
        factor = torch.clamp(self.code_bound / norm, max=1.0)
        return raw * factor

    def bound_excess(self, standardized: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(self.raw_unprojected(standardized), dim=1)
        return torch.relu(norm - self.code_bound)

    def sdf(self, standardized: torch.Tensor, xyz: torch.Tensor, point_chunk: int = 65536) -> torch.Tensor:
        if standardized.ndim != 2 or standardized.shape[1] != C.LATENT_DIM:
            raise ValueError(f"Expected standardized latent [B,{C.LATENT_DIM}]")
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0).expand(standardized.shape[0], -1, -1)
        if xyz.ndim != 3 or xyz.shape[0] != standardized.shape[0] or xyz.shape[2] != 3:
            raise ValueError("Expected xyz [B,P,3] or [P,3]")
        batch, points = xyz.shape[:2]
        raw = self.raw_projected(standardized)
        flat_xyz = xyz.reshape(-1, 3)
        flat_raw = raw[:, None, :].expand(batch, points, C.LATENT_DIM).reshape(-1, C.LATENT_DIM)
        predictions = []
        for start in range(0, len(flat_xyz), int(point_chunk)):
            stop = min(start + int(point_chunk), len(flat_xyz))
            predictions.append(self.decoder(torch.cat((flat_raw[start:stop], flat_xyz[start:stop]), dim=1), level_weights=self.level_weights)[:, 0])
        return torch.cat(predictions).reshape(batch, points)

    def soft_volume(self, standardized: torch.Tensor, samples: int, temperature: float, point_chunk: int = 65536) -> torch.Tensor:
        engine = torch.quasirandom.SobolEngine(3, scramble=True, seed=1729)
        xyz = engine.draw(int(samples)).to(device=standardized.device, dtype=standardized.dtype) * 2.0 - 1.0
        sdf = self.sdf(standardized, xyz, point_chunk)
        normalized_volume = 8.0 * torch.sigmoid(-sdf / float(temperature)).mean(dim=1)
        return normalized_volume * (self.linear_normalized_to_mm ** 3)

    @torch.no_grad()
    def mesh(self, standardized: torch.Tensor, resolution: int, point_chunk: int = 131072):
        import trimesh
        from skimage.measure import marching_cubes

        if standardized.shape != (1, C.LATENT_DIM):
            raise ValueError("mesh expects exactly one standardized latent")
        resolution = int(resolution)
        total = resolution ** 3
        step = 2.0 / (resolution - 1)
        values = np.empty(total, dtype=np.float32)
        for start in range(0, total, int(point_chunk)):
            stop = min(start + int(point_chunk), total)
            index = torch.arange(start, stop, device=standardized.device)
            x = torch.div(index, resolution * resolution, rounding_mode="floor")
            y = torch.div(index, resolution, rounding_mode="floor") % resolution
            z = index % resolution
            xyz = torch.stack((x, y, z), dim=1).to(standardized.dtype) * step - 1.0
            values[start:stop] = self.sdf(standardized, xyz, point_chunk)[:, :].reshape(-1).cpu().numpy()
        volume = values.reshape(resolution, resolution, resolution)
        if not float(volume.min()) <= 0.0 <= float(volume.max()):
            raise RuntimeError(f"No zero level set: [{volume.min()}, {volume.max()}]")
        vertices, faces, _, _ = marching_cubes(volume, level=0.0, spacing=(step, step, step), method="lewiner")
        vertices += np.asarray([-1.0, -1.0, -1.0], dtype=np.float32)
        vertices = ((vertices - self.target_range_min) / self.range_linear_scale_factor + self.range_global_min) * self.distance_unscale_factor
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def build_geometry(train_archive: dict[str, np.ndarray], device: torch.device, registry: dict[str, Any] | None = None) -> FrozenINRGeometry:
    return FrozenINRGeometry(train_archive, device, registry).to(device).eval()
