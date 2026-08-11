"""Frozen-SIREN geometry losses shared by all matched transports."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from siren256_common import decode_sdf, signed_mesh_volume


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return value.sum() * 0.0
    return value[mask].mean()


class C3GeometryLoss(nn.Module):
    def __init__(self, decoder: nn.Module, faces: torch.Tensor, loss_scales: dict[str, float], weights: dict[str, float], age_range_years: float, options: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.decoder, self.weights, self.age_range_years = decoder, dict(weights), float(age_range_years)
        self.options = dict(options or {})
        self.weight_schedule = dict(self.options.get("LossWeightSchedule", {}))
        self.current_epoch = 0
        self.normalized_geometry = bool(self.options.get("NormalizedGeometryHuber", False))
        self.volume_estimator = str(self.options.get("TrainingVolumeEstimator", "registered_normal_proxy")).lower()
        if self.volume_estimator not in {"registered_normal_proxy", "soft_occupancy"}:
            raise ValueError(f"Unknown TrainingVolumeEstimator: {self.volume_estimator!r}")
        self.register_buffer("faces", faces.long())
        self.scales = {key: max(float(value), 1.0e-8) for key, value in loss_scales.items()}
        if self.volume_estimator == "soft_occupancy":
            count = int(self.options.get("VolumeProbePoints", 4096))
            bounds = self.options.get("VolumeProbeBounds", [-1.05, 1.05])
            bounds_tensor = torch.as_tensor(bounds, dtype=torch.float32)
            if bounds_tensor.shape == (2,):
                low = torch.full((3,), float(bounds_tensor[0]))
                high = torch.full((3,), float(bounds_tensor[1]))
            elif bounds_tensor.shape == (3, 2):
                low, high = bounds_tensor[:, 0], bounds_tensor[:, 1]
            else:
                raise ValueError("VolumeProbeBounds must be [low,high] or [[x0,x1],[y0,y1],[z0,z1]].")
            if count <= 0 or not bool(torch.all(high > low)):
                raise ValueError("Invalid soft-occupancy probe configuration.")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.options.get("VolumeProbeSeed", 12345)))
            probes = low[None, :] + torch.rand(count, 3, generator=generator) * (high - low)[None, :]
        else:
            probes = torch.empty(0, 3, dtype=torch.float32)
        self.register_buffer("volume_probe_points", probes)

    def _scale(self, name: str) -> float:
        return self.scales.get(name, self.scales.get("latent", 1.0))

    def set_epoch(self, epoch: int) -> None:
        """Set the training epoch used by optional loss-weight curricula."""
        self.current_epoch = max(0, int(epoch))

    def effective_weight(self, name: str) -> float:
        """Return a deterministic scheduled weight without changing old configs."""
        final_weight = float(self.weights.get(name, 0.0))
        schedule = self.weight_schedule.get(name)
        if not isinstance(schedule, dict):
            return final_weight
        start_epoch = max(1, int(schedule.get("start_epoch", 1)))
        ramp_epochs = max(0, int(schedule.get("ramp_epochs", 0)))
        initial_weight = float(schedule.get("initial_weight", 0.0))
        scheduled_final = float(schedule.get("final_weight", final_weight))
        if self.current_epoch < start_epoch:
            return initial_weight
        if ramp_epochs == 0:
            return scheduled_final
        progress = min(1.0, float(self.current_epoch - start_epoch + 1) / float(ramp_epochs))
        return initial_weight + progress * (scheduled_final - initial_weight)

    def effective_weights(self) -> dict[str, float]:
        return {name: self.effective_weight(name) for name in self.weights}

    def _huber(self, name: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.normalized_geometry and name in {"registered_normal", "volume", "rate", "slope"}:
            residual = (prediction - target) / self._scale(name)
            return F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0)
        return F.smooth_l1_loss(prediction, target)

    def soft_volume(self, latent: torch.Tensor) -> torch.Tensor:
        if self.volume_estimator != "soft_occupancy" or not self.volume_probe_points.numel():
            raise RuntimeError("soft_volume is available only for the soft-occupancy estimator.")
        points = self.volume_probe_points.to(device=latent.device, dtype=latent.dtype)
        points = points[None, :, :].expand(latent.shape[0], -1, -1)
        sdf = decode_sdf(self.decoder, latent, points, chunk=int(self.options.get("VolumeProbeChunkSize", 131072)))
        temperature = float(self.options.get("VolumeProbeTemperature", 0.01))
        inside_sign = float(self.options.get("VolumeProbeInsideSDFSign", -1.0))
        if temperature <= 0.0:
            raise ValueError("VolumeProbeTemperature must be positive.")
        return torch.sigmoid(inside_sign * sdf / temperature).mean(dim=1).clamp_min(1.0e-8)

    def volume_from_prediction(self, source_latent: torch.Tensor, predicted_latent: torch.Tensor, source_volume: torch.Tensor, proxy_volume: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        source_volume = source_volume.clamp_min(1.0e-8)
        if self.volume_estimator == "soft_occupancy":
            with torch.no_grad():
                source_soft = self.soft_volume(source_latent)
            predicted_soft = self.soft_volume(predicted_latent)
            log_ratio = torch.log(predicted_soft / source_soft)
            return source_volume * torch.exp(log_ratio), log_ratio
        if proxy_volume is None:
            raise ValueError("registered_normal_proxy volume requires proxy_volume.")
        proxy_volume = proxy_volume.clamp_min(1.0e-8)
        return proxy_volume, torch.log(proxy_volume / source_volume)

    def normal_proxy(self, source_latent: torch.Tensor, predicted_latent: torch.Tensor, vertices: torch.Tensor, normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            source_sdf = decode_sdf(self.decoder, source_latent, vertices)
        predicted_sdf = decode_sdf(self.decoder, predicted_latent, vertices)
        displacement = -(predicted_sdf - source_sdf)
        proxy_vertices = vertices + displacement[..., None] * normals
        return displacement, proxy_vertices, signed_mesh_volume(proxy_vertices, self.faces)

    def _normal_loss(self, source_latent: torch.Tensor, predicted_latent: torch.Tensor, source_vertices: torch.Tensor, target_vertices: torch.Tensor, normals: torch.Tensor, vertex_count: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        displacement, proxy_vertices, volume = self.normal_proxy(source_latent, predicted_latent, source_vertices, normals)
        target_displacement = ((target_vertices - source_vertices) * normals).sum(dim=-1)
        if vertex_count and vertex_count < displacement.shape[1]:
            indices = torch.randperm(displacement.shape[1], device=displacement.device)[:vertex_count]
            normal = self._huber("registered_normal", displacement[:, indices], target_displacement[:, indices])
        else:
            normal = self._huber("registered_normal", displacement, target_displacement)
        return normal, volume, proxy_vertices

    def pair_terms(self, model: nn.Module, batch: dict[str, Any], vertex_count: int | None, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source, target = batch["source_latent"], batch["target_latent"]
        source_time, target_time, condition = batch["source_time"], batch["target_time"], batch["condition"]
        prediction = model.transport(source, source_time, target_time, condition)
        target_sdf = decode_sdf(self.decoder, prediction, batch["target_samples"][:, :, :3])
        terms: dict[str, torch.Tensor] = {
            "latent": F.smooth_l1_loss(prediction, target),
            "target_sdf": F.l1_loss(target_sdf, batch["target_samples"][:, :, 3]),
        }
        normal, proxy_volume, _ = self._normal_loss(source, prediction, batch["source_vertices"], batch["target_vertices"], batch["source_normals"], vertex_count)
        terms["registered_normal"] = normal
        source_volume, target_volume = batch["source_volume"].clamp_min(1.0e-8), batch["target_volume"].clamp_min(1.0e-8)
        _, predicted_log_ratio = self.volume_from_prediction(source, prediction, source_volume, proxy_volume)
        target_log_ratio = torch.log(target_volume / source_volume)
        terms["volume"] = self._huber("volume", predicted_log_ratio, target_log_ratio)
        gap = batch["gap_years"].abs().clamp_min(0.05)
        predicted_rate, target_rate = predicted_log_ratio / gap, target_log_ratio / gap
        terms["rate"] = self._huber("rate", predicted_rate, target_rate)
        valid = batch["observed_cache_index"] >= 0
        observed_time = batch["observed_time"].reshape(-1, 1)
        midpoint = model.transport(source, source_time, observed_time, condition)
        composed_observed = model.transport(midpoint, observed_time, target_time, condition)
        terms["observed_semigroup"] = _masked_mean(((prediction - composed_observed) ** 2).mean(dim=1), valid)
        ratio = torch.empty_like(source_time).uniform_(0.1, 0.9) if training else torch.full_like(source_time, 0.5)
        virtual_time = source_time + ratio * (target_time - source_time)
        virtual = model.transport(model.transport(source, source_time, virtual_time, condition), virtual_time, target_time, condition)
        terms["virtual_semigroup"] = F.mse_loss(prediction, virtual)
        terms["inverse"] = F.mse_loss(model.transport(prediction, target_time, source_time, condition), source)
        # C4-only population constraints.  They are train-derived because the
        # current batch contains no validation/test records during training.
        ad, cn = condition.reshape(-1) > 0.5, condition.reshape(-1) <= 0.5
        if bool(ad.any()) and bool(cn.any()):
            terms["group_rate"] = self._huber("rate", predicted_rate[ad].mean(), target_rate[ad].mean()) + self._huber("rate", predicted_rate[cn].mean(), target_rate[cn].mean())
            terms["disease_gap"] = self._huber("rate", predicted_rate[ad].mean() - predicted_rate[cn].mean(), target_rate[ad].mean() - target_rate[cn].mean())
        else:
            terms["group_rate"] = prediction.sum() * 0.0
            terms["disease_gap"] = prediction.sum() * 0.0
        terms["slope"] = prediction.sum() * 0.0
        return prediction, terms

    def sequence_terms(self, model: nn.Module, sequence: dict[str, torch.Tensor], vertex_count: int | None) -> dict[str, torch.Tensor]:
        latents, times, condition = sequence["latents"], sequence["times"], sequence["condition"]
        vertices, normals, volumes = sequence["vertices"], sequence["normals"], sequence["volumes"]
        if latents.shape[0] < 2:
            raise ValueError("A sequence requires at least two visits.")
        source, source_time = latents[:1], times[:1]
        current, previous_time = source, source_time
        rollout_latent, one_shot_latent, target_latent = [], [], []
        if self.volume_estimator == "soft_occupancy":
            with torch.no_grad():
                predicted_volumes = [self.soft_volume(source)]
        else:
            predicted_volumes = [volumes[:1]]
        normal_terms = []
        for index in range(1, latents.shape[0]):
            target_time = times[index : index + 1]
            current = model.transport(current, previous_time, target_time, condition)
            direct = model.transport(source, source_time, target_time, condition)
            normal_roll, proxy_volume_roll, _ = self._normal_loss(source, current, vertices[:1], vertices[index : index + 1], normals[:1], vertex_count)
            normal_direct, _, _ = self._normal_loss(source, direct, vertices[:1], vertices[index : index + 1], normals[:1], vertex_count)
            normal_terms.append(0.5 * (normal_roll + normal_direct))
            rollout_latent.append(current)
            one_shot_latent.append(direct)
            target_latent.append(latents[index : index + 1])
            if self.volume_estimator == "soft_occupancy":
                predicted_volumes.append(self.soft_volume(current))
            else:
                predicted_volumes.append(proxy_volume_roll)
            previous_time = target_time
        rollout, direct, target = torch.cat(rollout_latent), torch.cat(one_shot_latent), torch.cat(target_latent)
        predicted_log_volumes = torch.log(torch.cat(predicted_volumes).clamp_min(1.0e-8))
        target_log_volumes = torch.log(volumes.clamp_min(1.0e-8))
        centered_time = times - times.mean()
        denominator = (centered_time ** 2).sum().clamp_min(1.0e-8)
        predicted_slope = (centered_time * (predicted_log_volumes - predicted_log_volumes.mean())).sum() / denominator
        target_slope = (centered_time * (target_log_volumes - target_log_volumes.mean())).sum() / denominator
        zero = rollout.sum() * 0.0
        return {
            "sequence_latent": 0.5 * (F.smooth_l1_loss(rollout, target) + F.smooth_l1_loss(direct, target)),
            "sequence_geometry": torch.stack(normal_terms).mean(),
            "sequence_semigroup": F.mse_loss(rollout, direct),
            "slope": self._huber("slope", predicted_slope, target_slope),
            "latent": zero, "target_sdf": zero, "registered_normal": zero, "observed_semigroup": zero, "virtual_semigroup": zero, "inverse": zero, "volume": zero, "rate": zero, "group_rate": zero, "disease_gap": zero,
        }

    def total(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        total = next(iter(terms.values())).sum() * 0.0
        aliases = {"sequence_latent": "latent", "sequence_geometry": "registered_normal", "sequence_semigroup": "latent", "observed_semigroup": "latent", "virtual_semigroup": "latent", "inverse": "latent", "group_rate": "rate", "disease_gap": "rate"}
        for name, value in terms.items():
            scale_name = aliases.get(name, name)
            already_normalized = self.normalized_geometry and scale_name in {"registered_normal", "volume", "rate", "slope"}
            denominator = 1.0 if already_normalized else self._scale(scale_name)
            total = total + self.effective_weight(name) * value / denominator
        return total
