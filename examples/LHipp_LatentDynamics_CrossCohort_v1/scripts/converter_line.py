#!/usr/bin/env python3
"""Stage 5 step 2: the converter cocycle C1 (PLAN Part 5B) - dose path, onsets, objective, onset rules, recovery.

Condition path of subject i with onset tau_i (normalized-age units, the transport's time axis):
    c_i(tau) = sigmoid((tau - tau_i) / w)
Per-leg average dose, the exact integral of c over [s, t] divided by (t - s):
    cbar_i(s, t) = w [softplus((t - tau_i)/w) - softplus((s - tau_i)/w)] / (t - s),   cbar_i(s, s) = c_i(s)
Transport: Phi_i(z, s, t) = z + (t - s) [v_CN(z,s,t) + cbar_i(s,t) v_AD(z,s,t)], i.e. task3's DirectC4Flow with the
condition replaced by cbar. Dose additivity (u - s) cbar(s,u) + (t - u) cbar(u,t) = (t - s) cbar(s,t) is exact, so
the time-varying condition adds no composition defect of its own. Stable subjects have a constant condition (0 or 1)
and C1 is then exactly direct_c4.

Onsets: a train converter's onset is tau_i = a_i + (b_i - a_i) sigmoid(theta_i), where [a_i, b_i] are the normalized
ages of the last visit before the first post-conversion label and of that first post-conversion visit. Validation
and test subjects never get a learned onset: their onset is fixed by a rule (prefix-only or oracle window).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import dynamics_core as D

STABLE_CN, STABLE_AD, CONVERTER = 0, 1, -1
MODE_CONSTANT, MODE_THETA, MODE_FIXED = 0, 1, 2
AD_CONVERTERS = ("CN->AD", "MCI->AD")
Condition = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------------------
# dose path
# --------------------------------------------------------------------------------------


def integrated_condition(source: torch.Tensor, target: torch.Tensor, onset: torch.Tensor, width: float) -> torch.Tensor:
    """Integral of sigmoid((tau - onset)/width) over [source, target]."""
    return width * (F.softplus((target - onset) / width) - F.softplus((source - onset) / width))


def average_dose(source: torch.Tensor, target: torch.Tensor, onset: torch.Tensor, width: float, eps: float = 1.0e-6) -> torch.Tensor:
    gap = target - source
    small = gap.abs() < eps
    safe = torch.where(small, torch.ones_like(gap), gap)
    return torch.where(small, torch.sigmoid((source - onset) / width), integrated_condition(source, target, onset, width) / safe)


# --------------------------------------------------------------------------------------
# per-subject condition tables
# --------------------------------------------------------------------------------------


@dataclass
class ConditionTable:
    """Per-visit condition specification (visits of one subject share their entries)."""

    mode: np.ndarray
    constant: np.ndarray
    theta_index: np.ndarray
    window_a: np.ndarray
    window_b: np.ndarray
    fixed_onset: np.ndarray
    partial: np.ndarray
    theta_count: int
    theta_subjects: list[str]

    def learned_onset_visits(self) -> int:
        return int(np.count_nonzero(self.mode == MODE_THETA))


def stable_groups(archive: dict[str, np.ndarray]) -> np.ndarray:
    """Per visit: 0 CN-stable, 1 AD-stable, -1 any trajectory with a change."""
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    output = np.full(int(offsets[-1]), CONVERTER, dtype=np.int64)
    for index, group in enumerate(archive["subject_trajectory_groups"].astype(str)):
        if group in ("CN-stable", "AD-stable"):
            output[offsets[index]:offsets[index + 1]] = STABLE_AD if group == "AD-stable" else STABLE_CN
    return output


def pair_rows(archive: dict[str, np.ndarray]) -> list:
    """All forward pairs with a midpoint intermediate (stage-1 packager rule) as task3 PairRows.

    Built from the archive instead of task3's ``load_pairs``, whose CSV contract assumes one binary diagnosis per
    subject. ``diagnosis`` is the stable group's label for stable subjects and "CN" for any trajectory with a change
    (none is AD at baseline in this view), which is what the balanced pair sampler stratifies on.
    """
    C = D.core()["C"]
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    years = archive["visit_time_years_from_baseline"].astype(np.float64)
    groups = archive["subject_trajectory_groups"].astype(str)
    rows = []
    for index, subject in enumerate(archive["subject_ids"].astype(str)):
        start, end = int(offsets[index]), int(offsets[index + 1])
        diagnosis = "AD" if groups[index] == "AD-stable" else "CN"
        for source in range(start, end - 1):
            for target in range(source + 1, end):
                gap = target - source
                rows.append(C.PairRow(source, target, source + gap // 2 if gap > 1 else -1, subject, diagnosis,
                                      "adjacent" if gap == 1 else "nonadjacent", float(years[target] - years[source])))
    return rows


def _first_at_or_beyond(labels: Sequence[str], target: str) -> int | None:
    order = {"CN": 0, "MCI": 1, "AD": 2}
    return next((position for position, label in enumerate(labels) if order[label] >= order[target]), None)


def build_condition_table(archive: dict[str, np.ndarray], rule: str, prefixes: dict[int, list[int]] | None = None,
                          partial_dose: bool = False, theta_count: int | None = None) -> ConditionTable:
    """Condition table for one archive.

    rule = "learned": train converters get a learned onset on their observed window (training only);
           "oracle":  converters get the midpoint of the window observed over all visits;
           "prefix":  a converter's onset is the midpoint of the window observed within ``prefixes[subject_index]``
                      (local visit positions); if the prefix shows no conversion it keeps its last prefix label, and if
                      its first prefix visit is already converted the condition is constant 1.
    Stable subjects are constant (CN 0, AD 1). CN->MCI subjects are constant 0 unless ``partial_dose``, in which
    case their MCI window drives kappa * dose.
    """
    if rule not in ("learned", "oracle", "prefix"):
        raise ValueError(rule)
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    visits = int(offsets[-1])
    ages = archive["visit_age_norm_train"].astype(np.float64)
    labels_all = archive["visit_trajectory_labels"].astype(str)
    table = ConditionTable(mode=np.full(visits, MODE_CONSTANT, np.int64), constant=np.zeros(visits, np.float32),
                           theta_index=np.full(visits, -1, np.int64), window_a=np.zeros(visits, np.float32),
                           window_b=np.zeros(visits, np.float32), fixed_onset=np.zeros(visits, np.float32),
                           partial=np.zeros(visits, bool), theta_count=0, theta_subjects=[])
    for index, group in enumerate(archive["subject_trajectory_groups"].astype(str)):
        start, end = int(offsets[index]), int(offsets[index + 1])
        members = slice(start, end)
        if group == "AD-stable":
            table.constant[members] = 1.0
            continue
        if group in AD_CONVERTERS:
            target, partial = "AD", False
        elif group == "CN->MCI" and partial_dose:
            target, partial = "MCI", True
        else:
            continue  # CN-stable, and CN->MCI without the partial dose: constant 0
        labels = labels_all[members].tolist()
        table.partial[members] = partial
        if rule == "prefix":
            local = list((prefixes or {}).get(index, []))
            if not local:
                # Not eligible for this task (e.g. all_prior_k needs >= 3 visits), so never scored; keep the
                # first-visit label so the table stays well defined.
                table.constant[members] = 1.0 if labels[0] == "AD" else 0.0
                continue
            seen = [labels[position] for position in local]
            first = _first_at_or_beyond(seen, target)
            if first is None:
                table.constant[members] = 1.0 if seen[-1] == "AD" else 0.0
                continue
            if first == 0:
                table.constant[members] = 1.0
                continue
            a, b = ages[start + local[first - 1]], ages[start + local[first]]
            table.mode[members], table.fixed_onset[members] = MODE_FIXED, 0.5 * (a + b)
            continue
        first = _first_at_or_beyond(labels, target)
        if first is None or first == 0:
            raise ValueError(f"{group} subject {index} has no observed conversion window")
        a, b = ages[start + first - 1], ages[start + first]
        if rule == "oracle":
            table.mode[members], table.fixed_onset[members] = MODE_FIXED, 0.5 * (a + b)
        else:
            table.mode[members], table.theta_index[members] = MODE_THETA, len(table.theta_subjects)
            table.window_a[members], table.window_b[members] = a, b
            table.theta_subjects.append(str(archive["subject_ids"][index]))
    table.theta_count = len(table.theta_subjects) if theta_count is None else int(theta_count)
    if rule != "learned" and table.learned_onset_visits():
        raise AssertionError("validation/test tables must not contain learned onsets")
    return table


# --------------------------------------------------------------------------------------
# C1
# --------------------------------------------------------------------------------------


class ConverterCocycle(nn.Module):
    """C1: a direct cocycle whose condition is each subject's per-leg average dose."""

    def __init__(self, flow: nn.Module, table: ConditionTable, width: float, partial_dose: bool = False) -> None:
        super().__init__()
        self.flow = flow
        self.width = float(width)
        self.partial_dose = bool(partial_dose)
        self.theta = nn.Parameter(torch.zeros(int(table.theta_count)))
        self.kappa_logit = nn.Parameter(torch.zeros(()), requires_grad=self.partial_dose)
        self.set_table(table)

    def set_table(self, table: ConditionTable) -> None:
        if int(table.theta_count) != self.theta.numel():
            raise ValueError(f"table expects {table.theta_count} onsets, model has {self.theta.numel()}")
        device = self.theta.device
        for name in ("mode", "theta_index"):
            self.register_buffer(name, torch.as_tensor(getattr(table, name), dtype=torch.long, device=device), persistent=False)
        for name in ("constant", "window_a", "window_b", "fixed_onset"):
            self.register_buffer(name, torch.as_tensor(getattr(table, name), dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("partial", torch.as_tensor(table.partial, dtype=torch.bool, device=device), persistent=False)

    def condition(self, visits: torch.Tensor, source_age: torch.Tensor, target_age: torch.Tensor) -> torch.Tensor:
        visits = visits.reshape(-1)
        mode = self.mode[visits]
        theta = self.theta[self.theta_index[visits].clamp_min(0)] if self.theta.numel() else torch.zeros_like(source_age)
        a, b = self.window_a[visits], self.window_b[visits]
        onset = torch.where(mode == MODE_THETA, a + (b - a) * torch.sigmoid(theta), self.fixed_onset[visits])
        dose = average_dose(source_age.reshape(-1), target_age.reshape(-1), onset, self.width)
        dose = torch.where(self.partial[visits], torch.sigmoid(self.kappa_logit) * dose, dose)
        return torch.where(mode == MODE_CONSTANT, self.constant[visits], dose)

    def onset_penalty(self) -> torch.Tensor:
        return (self.theta**2).mean() if self.theta.numel() else self.kappa_logit * 0.0

    def transport_visits(self, latent, source_age, target_age, visits):
        return self.flow.transport(latent, source_age, target_age, self.condition(visits, source_age, target_age))


def label_condition(values: dict[str, torch.Tensor]) -> Condition:
    """C0 / fixed-label semantics: the source visit's binary label on every leg."""
    return lambda visits, _source_age, _target_age: values["label"][visits.reshape(-1)]


class BoundCondition(nn.Module):
    """Adapter so task3 functions that call ``flow.transport(z, s, t, label)`` use a leg condition instead.

    ``bind(rows)`` must be called with exactly the rows of the next transport call (one chunk)."""

    def __init__(self, flow: nn.Module, condition: Condition) -> None:
        super().__init__()
        self.flow = flow
        self.condition = condition
        self.visits: torch.Tensor | None = None

    def bind(self, rows) -> "BoundCondition":
        self.visits = torch.as_tensor([int(row.source) for row in rows], dtype=torch.long)
        return self

    def transport(self, latent, source_age, target_age, _label=None, context=None, context_time=None):
        visits = self.visits.to(latent.device)
        return self.flow.transport(latent, source_age, target_age, self.condition(visits, source_age, target_age))


@torch.no_grad()
def evaluate_rows(flow: nn.Module, geometry, values, rows, raw_vertices, batch_size: int, condition: Condition) -> dict[str, Any]:
    """task3 ``evaluate_pairs`` metrics over ``rows`` with a leg condition (one call per chunk, recombined exactly)."""
    O = D.core()["O"]
    bound = BoundCondition(flow, condition)
    sums: dict[str, float] = {}
    count = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        result = O.evaluate_pairs(bound.bind(chunk), geometry, values, chunk, raw_vertices, len(chunk))["groups"]["overall"]
        for key, value in result.items():
            if key.endswith("_mean"):
                sums[key] = sums.get(key, 0.0) + value * result["rows"]
        count += result["rows"]
    return {key: value / max(count, 1) for key, value in sums.items()} | {"rows": count}


# --------------------------------------------------------------------------------------
# objective (task3 c4_objective with a leg condition)
# --------------------------------------------------------------------------------------


def pair_terms(flow, geometry, values, raw, statistics, condition: Condition) -> dict[str, torch.Tensor]:
    """task3 ``c4_objective.pair_terms`` with every leg's condition from ``condition(source_visits, s, t)``.

    Group-rate and disease-gap leaves use stable subjects only (``values["stable_group"]``); converters have no group
    rate target. With the source-visit label as condition on stable subjects this equals task3's function exactly.
    """
    O = D.core()["O"]
    batch = O.indexed(values, raw)
    visits = batch["source_index"]
    scales = statistics["normalization_scales"]

    def move(latent, source_age, target_age, members):
        return flow.transport(latent, source_age, target_age, condition(members, source_age, target_age))

    forward_prediction = move(batch["source"], batch["source_age"], batch["target_age"], visits)
    backward_prediction = move(batch["target"], batch["target_age"], batch["source_age"], visits)
    forward = O.shape_terms(forward_prediction, batch["target"], batch["target_vertices"], geometry, scales)
    backward = O.shape_terms(backward_prediction, batch["source"], batch["source_vertices"], geometry, scales)

    valid_middle = batch["intermediate_index"] >= 0
    observed = forward_prediction.sum() * 0.0
    if bool(valid_middle.any()):
        middle_age = values["age"][batch["intermediate_index"][valid_middle]]
        members = visits[valid_middle]
        forward_middle = move(batch["source"][valid_middle], batch["source_age"][valid_middle], middle_age, members)
        forward_composed = move(forward_middle, middle_age, batch["target_age"][valid_middle], members)
        backward_middle = move(batch["target"][valid_middle], batch["target_age"][valid_middle], middle_age, members)
        backward_composed = move(backward_middle, middle_age, batch["source_age"][valid_middle], members)
        observed = 0.5 * (
            O.mean_scaled_mse(forward_prediction[valid_middle], forward_composed, scales["latent"])
            + O.mean_scaled_mse(backward_prediction[valid_middle], backward_composed, scales["latent"])
        )

    ratio = torch.empty_like(batch["source_age"]).uniform_(0.2, 0.8)
    virtual_age = batch["source_age"] + ratio * (batch["target_age"] - batch["source_age"])
    forward_middle = move(batch["source"], batch["source_age"], virtual_age, visits)
    forward_composed = move(forward_middle, virtual_age, batch["target_age"], visits)
    backward_middle = move(batch["target"], batch["target_age"], virtual_age, visits)
    backward_composed = move(backward_middle, virtual_age, batch["source_age"], visits)
    virtual = 0.5 * (
        O.mean_scaled_mse(forward_prediction, forward_composed, scales["latent"])
        + O.mean_scaled_mse(backward_prediction, backward_composed, scales["latent"])
    )
    inverse_forward = move(forward_prediction, batch["target_age"], batch["source_age"], visits)
    inverse_backward = move(backward_prediction, batch["source_age"], batch["target_age"], visits)
    inverse = 0.5 * (
        O.mean_scaled_mse(inverse_forward, batch["source"], scales["latent"])
        + O.mean_scaled_mse(inverse_backward, batch["target"], scales["latent"])
    )

    predicted_forward_volume = geometry.volume_from_vertices(geometry.vertices(forward_prediction))
    predicted_backward_volume = geometry.volume_from_vertices(geometry.vertices(backward_prediction))
    log_forward = torch.log(predicted_forward_volume) - torch.log(batch["target_volume"])
    log_backward = torch.log(predicted_backward_volume) - torch.log(batch["source_volume"])
    volume = 0.5 * (
        F.smooth_l1_loss(log_forward / float(scales["volume_log"]), torch.zeros_like(log_forward))
        + F.smooth_l1_loss(log_backward / float(scales["volume_log"]), torch.zeros_like(log_backward))
    )
    years = (batch["target_years"] - batch["source_years"]).clamp_min(1.0e-6)
    observed_rate = (torch.log(batch["target_volume"]) - torch.log(batch["source_volume"])) / years
    rate_forward = (torch.log(predicted_forward_volume) - torch.log(batch["source_volume"])) / years
    rate_backward = (torch.log(predicted_backward_volume) - torch.log(batch["target_volume"])) / (-years)
    rate = 0.5 * (
        F.smooth_l1_loss((rate_forward - observed_rate) / float(scales["rate"]), torch.zeros_like(rate_forward))
        + F.smooth_l1_loss((rate_backward - observed_rate) / float(scales["rate"]), torch.zeros_like(rate_backward))
    )
    stable = values["stable_group"][visits]
    is_ad, is_cn = stable == STABLE_AD, stable == STABLE_CN
    group_rate = forward_prediction.sum() * 0.0
    disease_gap = forward_prediction.sum() * 0.0
    if bool(is_ad.any()) and bool(is_cn.any()):
        zero = torch.zeros((), device=rate_forward.device)
        losses = []
        for diagnosis, mask in (("CN", is_cn), ("AD", is_ad)):
            target = torch.tensor(float(statistics["group_log_volume_rate_targets"][diagnosis]), device=rate_forward.device)
            losses.append(F.smooth_l1_loss((rate_forward[mask].mean() - target) / float(scales["rate"]), zero))
        group_rate = torch.stack(losses).mean()
        target_gap = torch.tensor(float(statistics["ad_minus_cn_log_volume_rate_target"]), device=rate_forward.device)
        disease_gap = F.smooth_l1_loss(((rate_forward[is_ad].mean() - rate_forward[is_cn].mean()) - target_gap) / float(scales["rate"]), zero)
    return {
        "real_latent": 0.5 * (forward["latent"] + backward["latent"]),
        "real_vertex": 0.5 * (forward["vertex"] + backward["vertex"]),
        "real_coordinate": 0.5 * (forward["coordinate"] + backward["coordinate"]),
        "real_euclidean": 0.5 * (forward["euclidean"] + backward["euclidean"]),
        "observed_semigroup": observed, "virtual_semigroup": virtual, "inverse": inverse,
        "volume": volume, "rate": rate, "group_rate": group_rate, "disease_gap": disease_gap,
    }


def sequence_terms(flow, geometry, values, archive, start: int, statistics, condition: Condition) -> dict[str, torch.Tensor]:
    """task3 ``c4_objective.sequence_terms`` with a leg condition for the subject whose first visit is ``start``."""
    parts = D.core()
    C, O = parts["C"], parts["O"]
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    subject_index = int(np.searchsorted(offsets, int(start), side="right") - 1)
    end = int(offsets[subject_index + 1])
    z, ages, years = values["z"][start:end], values["age"][start:end], values["years"][start:end]
    reference_vertices, reference_volume = values["reference_vertices"][start:end], values["reference_volume"][start:end]
    if len(z) < 2:
        raise ValueError("Sequence needs at least two visits")
    scales = statistics["normalization_scales"]
    count = len(z) - 1
    one = torch.full((1,), int(start), dtype=torch.long, device=z.device)
    many = one.expand(count)

    def move(latent, source_age, target_age, members):
        return flow.transport(latent, source_age, target_age, condition(members, source_age, target_age))

    direct_forward = move(z[:1].expand(count, -1), ages[:1].expand(count), ages[1:], many)
    rollout_forward, current, previous_age = [], z[:1], ages[:1]
    for index in range(1, len(z)):
        current = move(current, previous_age, ages[index:index + 1], one)
        rollout_forward.append(current)
        previous_age = ages[index:index + 1]
    rollout_forward = torch.cat(rollout_forward)
    reverse_targets, reverse_vertices = z[:-1].flip(0), reference_vertices[:-1].flip(0)
    direct_backward = move(z[-1:].expand(count, -1), ages[-1:].expand(count), ages[:-1].flip(0), many)
    rollout_backward, current, previous_age = [], z[-1:], ages[-1:]
    for index in range(len(z) - 2, -1, -1):
        current = move(current, previous_age, ages[index:index + 1], one)
        rollout_backward.append(current)
        previous_age = ages[index:index + 1]
    rollout_backward = torch.cat(rollout_backward)
    shapes = (
        O.shape_terms(direct_forward, z[1:], reference_vertices[1:], geometry, scales),
        O.shape_terms(rollout_forward, z[1:], reference_vertices[1:], geometry, scales),
        O.shape_terms(direct_backward, reverse_targets, reverse_vertices, geometry, scales),
        O.shape_terms(rollout_backward, reverse_targets, reverse_vertices, geometry, scales),
    )
    semigroup = 0.5 * (O.mean_scaled_mse(direct_forward, rollout_forward, scales["latent"])
                       + O.mean_scaled_mse(direct_backward, rollout_backward, scales["latent"]))
    forward_volume = geometry.volume(torch.cat((z[:1], direct_forward)))
    backward_volume = geometry.volume(torch.cat((z[-1:], direct_backward)))
    slope_observed = C.line_slope(years, torch.log(reference_volume))
    slope_forward = C.line_slope(years, torch.log(forward_volume))
    slope_backward = C.line_slope(years.flip(0), torch.log(backward_volume))
    slope = 0.5 * (
        F.smooth_l1_loss((slope_forward - slope_observed) / float(scales["slope"]), torch.zeros_like(slope_forward))
        + F.smooth_l1_loss((slope_backward - slope_observed) / float(scales["slope"]), torch.zeros_like(slope_backward))
    )
    return {"sequence_latent": torch.stack([item["latent"] for item in shapes]).mean(),
            "sequence_vertex": torch.stack([item["vertex"] for item in shapes]).mean(),
            "sequence_semigroup": semigroup, "slope": slope}


@torch.no_grad()
def cocycle_defects(flow, values, rows, statistics, batch_size: int, condition: Condition) -> dict[str, float]:
    """task3 ``c4_objective.cocycle_defects`` with a leg condition."""
    parts = D.core()
    C, O = parts["C"], parts["O"]
    semigroup, inverse_values = [], []
    scale = float(statistics["normalization_scales"]["displacement"])
    for start in range(0, len(rows), batch_size):
        batch = O.indexed(values, C.collate_pairs(rows[start:start + batch_size]))
        visits, s, t = batch["source_index"], batch["source_age"], batch["target_age"]
        middle_age = 0.5 * (s + t)
        direct = flow.transport(batch["source"], s, t, condition(visits, s, t))
        middle = flow.transport(batch["source"], s, middle_age, condition(visits, s, middle_age))
        composed = flow.transport(middle, middle_age, t, condition(visits, middle_age, t))
        inverse = flow.transport(direct, t, s, condition(visits, t, s))
        semigroup.extend((torch.sqrt(torch.mean((direct - composed).square(), dim=1)) / scale).cpu().tolist())
        inverse_values.extend((torch.sqrt(torch.mean((inverse - batch["source"]).square(), dim=1)) / scale).cpu().tolist())
    return {"relative_semigroup_defect_mean": float(np.mean(semigroup)), "relative_semigroup_defect_p95": float(np.quantile(semigroup, 0.95)),
            "relative_inverse_defect_mean": float(np.mean(inverse_values)), "relative_inverse_defect_p95": float(np.quantile(inverse_values, 0.95))}


# --------------------------------------------------------------------------------------
# synthetic onset recovery (gate G5.6)
# --------------------------------------------------------------------------------------


@torch.no_grad()
def dosed_trajectory(flow, baseline: torch.Tensor, baseline_age: float, visit_ages: torch.Tensor, onsets: torch.Tensor, width: float) -> torch.Tensor:
    """Codes at ``visit_ages`` [V] transported from one baseline [D] for each onset in ``onsets`` [G] -> [G, V, D]."""
    grid, count = len(onsets), len(visit_ages)
    source = torch.full((grid * count,), float(baseline_age), device=baseline.device)
    target = visit_ages.repeat(grid)
    onset = onsets.repeat_interleave(count)
    dose = average_dose(source, target, onset, width)
    moved = flow.transport(baseline.expand(grid * count, -1), source, target, dose)
    return moved.reshape(grid, count, -1)


@torch.no_grad()
def onset_posterior(flow, baseline, baseline_age: float, visit_ages, observed, sigma: float, window: tuple[float, float],
                    width: float, grid_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid posterior of the onset on its window (uniform prior, Gaussian code noise with known sigma)."""
    grid = torch.linspace(float(window[0]), float(window[1]), int(grid_points), device=baseline.device)
    predicted = dosed_trajectory(flow, baseline, baseline_age, visit_ages, grid, width)
    log_likelihood = -0.5 * ((predicted - observed.unsqueeze(0)) ** 2).sum(dim=(1, 2)) / float(sigma) ** 2
    posterior = torch.softmax(log_likelihood.double(), dim=0)
    return grid.double().cpu().numpy(), posterior.cpu().numpy()


def posterior_summary(grid: np.ndarray, posterior: np.ndarray, interval: Sequence[float]) -> dict[str, float]:
    cumulative = np.cumsum(posterior)
    quantile = lambda q: float(grid[min(int(np.searchsorted(cumulative, q)), len(grid) - 1)])
    return {"median": quantile(0.5), "low": quantile(float(interval[0])), "high": quantile(float(interval[1]))}
