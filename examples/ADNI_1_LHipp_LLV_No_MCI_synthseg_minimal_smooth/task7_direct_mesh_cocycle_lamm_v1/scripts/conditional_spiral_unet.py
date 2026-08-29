#!/usr/bin/env python3
"""Geometry compatibility used by the shared model-agnostic direct-mesh objectives.

The shared objective imports these two geometry functions from its historical model module.
Keeping this tiny local compatibility module avoids importing Spiral/torch-scatter into a
pure LAMM experiment. It does not instantiate or execute a Spiral network.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from conditional_lamm_flow import ConditionalLAMMFlow


# Retain the historical annotation name expected by the shared evaluator/objective.
ConditionalSpiralUNet = ConditionalLAMMFlow


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    triangles = vertices[:, faces]
    cross = torch.cross(
        triangles[:, :, 1] - triangles[:, :, 0],
        triangles[:, :, 2] - triangles[:, :, 0],
        dim=-1,
    )
    output = torch.zeros_like(vertices)
    for corner in range(3):
        indices = faces[:, corner].reshape(1, -1, 1).expand(vertices.shape[0], -1, 3)
        output.scatter_add_(1, indices, cross)
    return F.normalize(output, dim=-1, eps=1.0e-8)


def mesh_volume(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    triangles = vertices[:, faces]
    signed = torch.einsum(
        "bfi,bfi->bf",
        triangles[:, :, 0],
        torch.cross(triangles[:, :, 1], triangles[:, :, 2], dim=-1),
    ).sum(dim=1) / 6.0
    return signed.abs().clamp_min(1.0e-8)

