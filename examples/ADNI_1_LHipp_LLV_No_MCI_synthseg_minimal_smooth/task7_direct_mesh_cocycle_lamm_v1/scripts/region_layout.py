#!/usr/bin/env python3
"""Recover the exact 43+86 latest-LAMM region partition from its selected checkpoint."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

import common as C


DEFAULT_LAYOUT_CHECKPOINT = Path(
    "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_lamm_ae_v1/"
    "studies/E1_N3_ml_s1/best.pt"
)


@dataclass(frozen=True)
class RegionScale:
    name: str
    member_idx: torch.Tensor
    member_mask: torch.Tensor

    @property
    def regions(self) -> int:
        return int(self.member_mask.shape[0])

    @property
    def width(self) -> int:
        return int(self.member_mask.shape[1])


@dataclass(frozen=True)
class RegionLayout:
    scales: tuple[RegionScale, ...]
    n_vertices: int
    fingerprint: str
    source: str

    @property
    def region_counts(self) -> list[int]:
        return [scale.regions for scale in self.scales]


def _fingerprint(scales: list[RegionScale]) -> str:
    digest = hashlib.sha256()
    for scale in scales:
        digest.update(scale.name.encode("utf-8"))
        digest.update(scale.member_idx.cpu().numpy().tobytes())
        digest.update(scale.member_mask.cpu().numpy().tobytes())
    return digest.hexdigest()


def validate_layout(layout: RegionLayout, expected_regions=(43, 86)) -> None:
    if layout.region_counts != [int(value) for value in expected_regions]:
        raise ValueError(
            f"Region counts {layout.region_counts} do not match {list(expected_regions)}"
        )
    for scale in layout.scales:
        if scale.member_idx.shape != scale.member_mask.shape:
            raise ValueError(f"Index/mask mismatch for {scale.name}")
        active = scale.member_idx[scale.member_mask]
        if active.numel() != layout.n_vertices:
            raise ValueError(
                f"{scale.name} has {active.numel()} active slots, expected {layout.n_vertices}"
            )
        if not torch.equal(torch.sort(active).values.cpu(), torch.arange(layout.n_vertices)):
            raise ValueError(f"{scale.name} does not partition every vertex exactly once")


def load_region_layout(
    checkpoint: str | Path | None = None,
    expected_regions=(43, 86),
    n_vertices: int = C.VERTEX_COUNT,
) -> RegionLayout:
    path = C.resolve_path(checkpoint or DEFAULT_LAYOUT_CHECKPOINT)
    if not path.is_file():
        raise FileNotFoundError(f"Latest-LAMM layout checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    scales: list[RegionScale] = []
    for index, regions in enumerate(expected_regions):
        mask_key = f"tokenizers.{index}.member_mask"
        index_key = f"tokenizers.{index}.member_idx"
        if mask_key not in state or index_key not in state:
            raise KeyError(f"Checkpoint lacks region buffers {mask_key!r}/{index_key!r}")
        mask = state[mask_key].detach().cpu().to(torch.bool)
        member_idx = state[index_key].detach().cpu().long().reshape(mask.shape)
        if int(mask.shape[0]) != int(regions):
            raise ValueError(f"Scale {index} has {mask.shape[0]} regions, expected {regions}")
        scales.append(
            RegionScale(
                name="coarse_43" if int(regions) == 43 else f"fine_{int(regions)}",
                member_idx=member_idx,
                member_mask=mask,
            )
        )
    layout = RegionLayout(
        scales=tuple(scales),
        n_vertices=int(n_vertices),
        fingerprint=_fingerprint(scales),
        source=str(path),
    )
    validate_layout(layout, expected_regions)
    return layout

