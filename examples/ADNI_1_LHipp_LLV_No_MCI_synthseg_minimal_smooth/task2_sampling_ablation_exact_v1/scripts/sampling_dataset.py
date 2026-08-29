#!/usr/bin/env python3
"""Training-only standard-band and explicit-shell exact-SDF samplers."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPTS = SCRIPT_DIR.parent.parent / "task2_inr_multires_single_field_v1" / "scripts"
if str(BASE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BASE_SCRIPTS))

from multires_common import (  # noqa: E402
    load_sdf_arrays,
    sample_continuous_sdf_pair,
    stable_seed,
)


def _count_plan(total: int, shells: list[dict[str, Any]]) -> list[int]:
    """Largest-remainder allocation that always sums exactly to ``total``."""
    if total < 0:
        raise ValueError("Sample total cannot be negative.")
    fractions = np.asarray([float(shell["fraction"]) for shell in shells], dtype=np.float64)
    if len(fractions) == 0 or np.any(fractions <= 0.0):
        raise ValueError("Every shell fraction must be positive.")
    if not np.isclose(fractions.sum(), 1.0, atol=1.0e-9):
        raise ValueError(f"Shell fractions must sum to one, found {fractions.sum():.12g}.")
    exact = fractions * total
    counts = np.floor(exact).astype(np.int64)
    remaining = total - int(counts.sum())
    if remaining:
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:remaining]] += 1
    return [int(value) for value in counts]


def _validate_shells(shells: list[dict[str, Any]], label: str) -> None:
    _count_plan(1000, shells)
    previous_upper = 0.0
    for index, shell in enumerate(shells):
        lower = float(shell["min_abs_sdf"])
        upper_value = shell.get("max_abs_sdf")
        upper = None if upper_value is None else float(upper_value)
        if lower < 0.0 or not math.isclose(lower, previous_upper, abs_tol=1.0e-12):
            raise ValueError(f"{label} shells must be ordered and contiguous at shell {index}.")
        if upper is not None and upper <= lower:
            raise ValueError(f"{label} shell {index} has a non-positive width.")
        if upper is None and index != len(shells) - 1:
            raise ValueError(f"Only the final {label} shell may have no upper bound.")
        previous_upper = lower if upper is None else upper


def validate_shell_sampling(sampling: dict[str, Any]) -> None:
    if sampling.get("mode", "standard_bands") != "shell_stratified":
        return
    shells = sampling.get("shells", {})
    if set(shells) != {"global", "local"}:
        raise ValueError("Shell sampling requires exactly global and local shell definitions.")
    _validate_shells(shells["global"], "global")
    _validate_shells(shells["local"], "local")


def _eligible(array: np.ndarray, lower: float, upper: float | None) -> np.ndarray:
    absolute = np.abs(array[:, 3])
    mask = absolute >= float(lower)
    if upper is not None:
        mask &= absolute < float(upper)
    return array[mask]


def _draw_rows(
    array: np.ndarray,
    count: int,
    rng: np.random.Generator,
    lower: float,
    upper: float | None,
    grid_aabb: np.ndarray | None = None,
    grid_resolution: np.ndarray | None = None,
) -> np.ndarray:
    if count == 0:
        return np.empty((0, 4), dtype=np.float32)
    candidates = _eligible(array, lower, upper)
    if not len(candidates):
        bound = "infinity" if upper is None else f"{upper:g}"
        raise ValueError(f"No samples satisfy {lower:g} <= |SDF| < {bound}.")
    if grid_aabb is None:
        chosen = candidates[rng.integers(0, len(candidates), size=count)]
        return chosen.astype(np.float32, copy=False)

    if grid_resolution is None:
        raise ValueError("A grid resolution is required when grid balancing is enabled.")
    minimum, maximum = grid_aabb
    cells = np.floor(
        (candidates[:, :3] - minimum) / (maximum - minimum) * (grid_resolution - 1)
    ).astype(np.int64)
    cells = np.clip(cells, 0, grid_resolution - 2)
    cell_ids = cells[:, 0] + (grid_resolution[0] - 1) * (
        cells[:, 1] + (grid_resolution[1] - 1) * cells[:, 2]
    )
    order = np.argsort(cell_ids, kind="stable")
    sorted_ids = cell_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    lengths = np.diff(np.r_[starts, len(order)])
    groups = rng.integers(0, len(starts), size=count)
    offsets = np.floor(rng.random(count) * lengths[groups]).astype(np.int64)
    chosen = candidates[order[starts[groups] + offsets]]
    return chosen.astype(np.float32, copy=False)


def _sample_shell_group(
    pos: np.ndarray,
    neg: np.ndarray,
    total: int,
    shells: list[dict[str, Any]],
    rng: np.random.Generator,
    grid_aabb: np.ndarray | None = None,
    grid_resolution: np.ndarray | None = None,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for count, shell in zip(_count_plan(total, shells), shells):
        positive = count // 2
        negative = count - positive
        lower = float(shell["min_abs_sdf"])
        upper_value = shell.get("max_abs_sdf")
        upper = None if upper_value is None else float(upper_value)
        parts.extend(
            (
                _draw_rows(pos, positive, rng, lower, upper, grid_aabb, grid_resolution),
                _draw_rows(neg, negative, rng, lower, upper, grid_aabb, grid_resolution),
            )
        )
    result = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    rng.shuffle(result)
    return result


def sample_shell_stratified_pair(
    pos: np.ndarray,
    neg: np.ndarray,
    sampling: dict[str, Any],
    rng: np.random.Generator,
    grid_aabb: list[list[float]],
    grid_resolution: int | list[int],
) -> tuple[np.ndarray, np.ndarray]:
    validate_shell_sampling(sampling)
    global_total = sum(
        int(sampling[key])
        for key in (
            "global_near_samples_per_scene",
            "global_positive_samples_per_scene",
            "global_negative_samples_per_scene",
        )
    )
    local_total = sum(
        int(sampling[key])
        for key in (
            "local_ultra_near_samples_per_scene",
            "local_positive_samples_per_scene",
            "local_negative_samples_per_scene",
        )
    )
    aabb = np.asarray(grid_aabb, dtype=np.float32)
    resolution = (
        np.repeat(int(grid_resolution), 3)
        if isinstance(grid_resolution, int)
        else np.asarray(grid_resolution, dtype=np.int64)
    )
    broad = _sample_shell_group(
        pos, neg, global_total, sampling["shells"]["global"], rng
    )
    near = _sample_shell_group(
        pos,
        neg,
        local_total,
        sampling["shells"]["local"],
        rng,
        grid_aabb=aabb,
        grid_resolution=resolution,
    )
    return broad, near


def shell_eligibility_report(
    pos: np.ndarray, neg: np.ndarray, sampling: dict[str, Any]
) -> dict[str, Any]:
    """Report candidate and requested counts without drawing a training batch."""
    validate_shell_sampling(sampling)
    totals = {
        "global": sum(
            int(sampling[key])
            for key in (
                "global_near_samples_per_scene",
                "global_positive_samples_per_scene",
                "global_negative_samples_per_scene",
            )
        ),
        "local": sum(
            int(sampling[key])
            for key in (
                "local_ultra_near_samples_per_scene",
                "local_positive_samples_per_scene",
                "local_negative_samples_per_scene",
            )
        ),
    }
    report: dict[str, Any] = {}
    for group in ("global", "local"):
        rows = []
        counts = _count_plan(totals[group], sampling["shells"][group])
        for requested, shell in zip(counts, sampling["shells"][group]):
            lower = float(shell["min_abs_sdf"])
            upper_value = shell.get("max_abs_sdf")
            upper = None if upper_value is None else float(upper_value)
            rows.append(
                {
                    "min_abs_sdf": lower,
                    "max_abs_sdf": upper,
                    "requested_total": requested,
                    "requested_positive": requested // 2,
                    "requested_negative": requested - requested // 2,
                    "eligible_positive": int(len(_eligible(pos, lower, upper))),
                    "eligible_negative": int(len(_eligible(neg, lower, upper))),
                }
            )
        report[group] = rows
    return report


class SamplingAblationDataset(Dataset):
    """Drop-in dataset for the existing multiresolution trainer."""

    def __init__(self, rows: list[dict[str, str]], config: dict[str, Any]) -> None:
        self.rows = rows
        self.base_seed = int(config.get("seed", 0))
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.sampling = copy.deepcopy(config["sampling"])
        self.mode = str(self.sampling.get("mode", "standard_bands"))
        if self.mode not in {"standard_bands", "shell_stratified"}:
            raise ValueError(f"Unknown sampling mode {self.mode!r}.")
        validate_shell_sampling(self.sampling)
        specs = config["network_specs"]
        self.grid_aabb = specs["grid_aabb"]
        self.grid_resolution = specs.get(
            "sampling_balance_resolution", specs.get("grid_resolution")
        )
        if self.grid_resolution is None:
            raise ValueError("No sampling balance resolution was configured.")
        self.loaded_sdf = None
        if bool(config.get("load_dataset_into_ram", False)):
            self.loaded_sdf = [load_sdf_arrays(row["sdf_npz_path"]) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(int(epoch))

    def __getitem__(self, index: int):
        rng = np.random.default_rng(
            stable_seed(
                self.rows[index].get("scan_id", str(index)),
                self.base_seed + int(self._epoch.item()),
            )
        )
        pos, neg = (
            load_sdf_arrays(self.rows[index]["sdf_npz_path"])
            if self.loaded_sdf is None
            else self.loaded_sdf[index]
        )
        if self.mode == "shell_stratified":
            broad, near = sample_shell_stratified_pair(
                pos,
                neg,
                self.sampling,
                rng,
                self.grid_aabb,
                self.grid_resolution,
            )
        else:
            broad, near = sample_continuous_sdf_pair(
                pos,
                neg,
                self.sampling,
                rng,
                self.grid_aabb,
                self.grid_resolution,
            )
        return torch.from_numpy(broad), torch.from_numpy(near), index
