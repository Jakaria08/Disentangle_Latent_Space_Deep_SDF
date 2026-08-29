from __future__ import annotations

import copy

import numpy as np
import torch

from region_layout import RegionLayout, RegionScale


def synthetic_layout() -> RegionLayout:
    coarse_idx = torch.arange(12).reshape(2, 6)
    fine_idx = torch.arange(12).reshape(3, 4)
    scales = (
        RegionScale("coarse_2", coarse_idx, torch.ones_like(coarse_idx, dtype=torch.bool)),
        RegionScale("fine_3", fine_idx, torch.ones_like(fine_idx, dtype=torch.bool)),
    )
    return RegionLayout(scales=scales, n_vertices=12, fingerprint="synthetic", source="unit-test")


def synthetic_statistics() -> dict:
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [8, 9, 10],
            [8, 10, 11],
        ],
        dtype=np.int64,
    )
    return {
        "template_vertices_mm": np.zeros((12, 3), dtype=np.float32),
        "faces": faces,
        "coordinate_scale_mm": 1.0,
        "velocity_scale_mm_per_year": 0.1,
        "age_mean_years": 75.0,
        "age_std_years": 7.0,
    }


def synthetic_config(latent_dim: int) -> dict:
    split = [latent_dim // 2, latent_dim - latent_dim // 2]
    return {
        "name": f"synthetic_z{latent_dim}",
        "method": "direct_surface_cocycle",
        "model": {
            "operator": "lamm_mlpmixer",
            "region_scales": [2, 3],
            "token_dim": 16,
            "encoder_depth": 1,
            "decoder_depth": 1,
            "token_expansion": 2.0,
            "channel_expansion": 1.0,
            "latent_dim": latent_dim,
            "latent_split": split,
            "latent_width": 32,
            "latent_residual_blocks": 1,
            "condition_dim": 8,
            "time_frequencies": 1,
            "dropout": 0.0,
        },
    }


def make_batch(batch: int = 2):
    torch.manual_seed(11)
    vertices = torch.randn(batch, 12, 3)
    source = torch.tensor([70.0, 76.0])[:batch]
    target = torch.tensor([74.0, 81.0])[:batch]
    disease = torch.tensor([0.0, 1.0])[:batch]
    return vertices, source, target, disease

