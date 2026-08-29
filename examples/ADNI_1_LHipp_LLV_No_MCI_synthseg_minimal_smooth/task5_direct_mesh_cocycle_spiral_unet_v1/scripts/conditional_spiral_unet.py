#!/usr/bin/env python3
"""Time- and diagnosis-conditioned direct Spiral U-Net surface cocycle."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mesh_hierarchy import MeshHierarchy
from mesh_layers import ConditionalMeshBlock, adaptive_support_report, sparse_pool


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


class TimeDiseaseEmbedding(nn.Module):
    def __init__(self, condition_dim: int, frequencies: int, age_mean: float, age_std: float) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.frequencies = int(frequencies)
        self.register_buffer("age_mean", torch.tensor(float(age_mean)))
        self.register_buffer("age_std", torch.tensor(max(float(age_std), 1.0e-6)))
        self.disease = nn.Embedding(2, 8)
        raw_dim = 6 + 4 * self.frequencies + 8
        self.network = nn.Sequential(
            nn.Linear(raw_dim, self.condition_dim),
            nn.SiLU(),
            nn.Linear(self.condition_dim, self.condition_dim),
            nn.SiLU(),
        )

    def forward(
        self, source_age: torch.Tensor, target_age: torch.Tensor, disease: torch.Tensor
    ) -> torch.Tensor:
        source = (source_age.reshape(-1, 1) - self.age_mean) / self.age_std
        target = (target_age.reshape(-1, 1) - self.age_mean) / self.age_std
        delta = (target_age.reshape(-1, 1) - source_age.reshape(-1, 1)) / self.age_std
        midpoint = 0.5 * (source + target)
        elapsed = (target_age.reshape(-1, 1) - source_age.reshape(-1, 1)).abs()
        raw = [source, target, delta, delta.abs(), midpoint, torch.log1p(elapsed)]
        for frequency in range(self.frequencies):
            scale = math.pi * float(2**frequency)
            raw.extend((torch.sin(scale * source), torch.cos(scale * source)))
            raw.extend((torch.sin(scale * target), torch.cos(scale * target)))
        label = disease.reshape(-1).round().long().clamp(0, 1)
        raw.append(self.disease(label))
        return self.network(torch.cat(raw, dim=1))


class ConditionalSpiralUNet(nn.Module):
    """Fully convolutional mesh U-Net; no global latent vector or ODE solver."""

    def __init__(self, hierarchy: MeshHierarchy, config: dict[str, Any], statistics: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        channels = [int(value) for value in model["channels"]]
        if len(channels) != 4:
            raise ValueError("model.channels must contain four resolutions")
        if hierarchy.sizes != [2746, 1373, 344, 86]:
            raise ValueError(f"Unexpected hierarchy: {hierarchy.sizes}")
        operator = str(model["operator"])
        if operator not in {"spiral", "adaptive_spiral"}:
            raise ValueError(f"Unknown operator {operator}")
        schedule = ["spiral"] * 4 if operator == "spiral" else ["spiral", "spiral", "adaptive", "adaptive"]
        self.operator = operator
        self.hierarchy = hierarchy
        self.register_buffer("faces", hierarchy.faces[0].long())
        self.register_buffer("template", torch.as_tensor(statistics["template_vertices_mm"], dtype=torch.float32))
        self.register_buffer("coordinate_scale", torch.tensor(float(statistics["coordinate_scale_mm"])))
        self.register_buffer("velocity_scale", torch.tensor(float(statistics["velocity_scale_mm_per_year"])))
        self.register_buffer("log_volume_mean", torch.tensor(float(statistics["log_volume_mean"])))
        self.register_buffer("log_volume_std", torch.tensor(float(statistics["log_volume_std"])))
        condition_dim = int(model["condition_dim"])
        self.condition = TimeDiseaseEmbedding(
            condition_dim,
            int(model.get("time_frequencies", 3)),
            float(statistics["age_mean_years"]),
            float(statistics["age_std_years"]),
        )
        dropout = float(model.get("dropout", 0.0))
        support = float(model.get("adaptive_initial_support", 9.0))
        input_channels = 10

        def block(level: int, cin: int, cout: int) -> ConditionalMeshBlock:
            return ConditionalMeshBlock(
                cin,
                cout,
                condition_dim,
                hierarchy.spirals[level],
                schedule[level],
                hierarchy.dynamic_spirals[level],
                support,
                dropout,
            )

        self.encoder = nn.ModuleList(
            [
                block(0, input_channels, channels[0]),
                block(1, channels[0], channels[1]),
                block(2, channels[1], channels[2]),
            ]
        )
        self.bottleneck = block(3, channels[2], channels[3])
        self.fusions = nn.ModuleList(
            [
                nn.Linear(channels[3] + channels[2], channels[2]),
                nn.Linear(channels[2] + channels[1], channels[1]),
                nn.Linear(channels[1] + channels[0], channels[0]),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                block(2, channels[2], channels[2]),
                block(1, channels[1], channels[1]),
                block(0, channels[0], channels[0]),
            ]
        )
        self.velocity_head = nn.Linear(channels[0], 6)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def input_features(self, vertices: torch.Tensor) -> torch.Tensor:
        template = self.template.unsqueeze(0).expand(vertices.shape[0], -1, -1)
        centered_template = self.template - self.template.mean(dim=0, keepdim=True)
        template_features = centered_template.unsqueeze(0).expand(vertices.shape[0], -1, -1)
        normals = vertex_normals(vertices, self.faces)
        log_volume = torch.log(mesh_volume(vertices, self.faces)).reshape(-1, 1, 1)
        log_volume = ((log_volume - self.log_volume_mean) / self.log_volume_std).expand(
            -1, vertices.shape[1], -1
        )
        return torch.cat(
            (
                (vertices - template) / self.coordinate_scale,
                template_features / self.coordinate_scale,
                normals,
                log_volume,
            ),
            dim=-1,
        )

    def encoded_features(
        self,
        vertices: torch.Tensor,
        source_age: torch.Tensor,
        target_age: torch.Tensor,
        disease: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.condition(source_age, target_age, disease)
        hidden = self.input_features(vertices)
        skips = []
        for level, layer in enumerate(self.encoder):
            hidden = layer(hidden, condition)
            skips.append(hidden)
            hidden = sparse_pool(hidden, self.hierarchy.down[level])
        hidden = self.bottleneck(hidden, condition)
        for decoder_index, level in enumerate((2, 1, 0)):
            hidden = sparse_pool(hidden, self.hierarchy.up[level])
            hidden = F.silu(self.fusions[decoder_index](torch.cat((hidden, skips[level]), dim=-1)))
            hidden = self.decoder[decoder_index](hidden, condition)
        return hidden

    def average_velocity(
        self,
        vertices: torch.Tensor,
        source_age: torch.Tensor,
        target_age: torch.Tensor,
        disease: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encoded_features(vertices, source_age, target_age, disease)
        cn_velocity, ad_residual = self.velocity_head(features).chunk(2, dim=-1)
        label = disease.reshape(-1, 1, 1).to(vertices)
        return self.velocity_scale * (cn_velocity + label * ad_residual)

    def instantaneous_velocity(
        self, vertices: torch.Tensor, age: torch.Tensor, disease: torch.Tensor
    ) -> torch.Tensor:
        return self.average_velocity(vertices, age, age, disease)

    def transport(
        self,
        vertices: torch.Tensor,
        source_age: torch.Tensor,
        target_age: torch.Tensor,
        disease: torch.Tensor,
    ) -> torch.Tensor:
        elapsed_years = (target_age.reshape(-1, 1, 1) - source_age.reshape(-1, 1, 1)).to(vertices)
        return vertices + elapsed_years * self.average_velocity(vertices, source_age, target_age, disease)

    def adaptive_support_report(self) -> list[dict[str, Any]]:
        return adaptive_support_report(self)
