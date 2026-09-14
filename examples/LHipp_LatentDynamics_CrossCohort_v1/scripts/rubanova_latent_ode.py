#!/usr/bin/env python3
"""Latent ODE (Rubanova, Chen & Duvenaud, NeurIPS 2019) on 128-D representation codes.

Generative model, for one subject with condition d (0 control, 1 disease):

    z0 ~ N(0, I)                                   (z0 in R^{z0_dim}, at the first visit time t0)
    dz/dt = f_theta(z, t, d)                        (residual MLP vector field, fixed-step RK4)
    x_i ~ N(g(z(t_i)), sigma^2 I)                   (x_i = train-standardized code of visit i;
                                                     sigma fixed, as in Rubanova's released code)

Recognition model (ODE-RNN, run backward in time over the observed prefix x_1..x_k):

    h <- 0 at t_k; for i = k..1: h <- GRU(h, [x_i, d, t_{i+1} - t_i]); between visits
    h evolves by dh/dt = f_enc(h, t, d) from t_{i+1} back to t_i
    q(z0 | x_1..k, d) = N(mu(h), diag exp(logvar(h)))

Loss (extrapolation ELBO, per subject): -(1/n) sum_{i=1..n} log p(x_i | z(t_i)) + beta * KL(q || p),
with free bits (a per-dimension KL floor) and beta annealed 0 -> 1. The encoder sees the prefix,
the likelihood scores every visit, so the model must extrapolate.

Integration uses task3's RK4 (``models.integrate_sequence_rk4``), the same integrator as the
plain-ODE and BrainODE arms; ``torchdiffeq`` is not needed.

Residual variant (``model.residual = true``, method ``latent_ode_residual``; a documented
departure from the paper). The decoded latent trajectory predicts *change* relative to the last
observed visit (x_ref, t_ref) instead of the absolute code:

    x_hat(t) = x_ref + g(z(t)) - g(z(t_ref))

The faithful model must rebuild each subject's full 128-D code through z0, and with ~475
training subjects that reconstruction error exceeds two years of atrophy (stage 2 diagnostics:
validation 1.7-1.9x the no-change error). The residual variant keeps the ODE-RNN posterior, latent
ODE and ELBO, but anchors predictions on the observation, as the transport models are anchored.
With the zero-initialized vector field it predicts exactly no-change before training.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import dynamics_core as D


class LatentODE(nn.Module):
    method = "latent_ode"

    def __init__(
        self,
        *,
        obs_dim: int,
        z0_dim: int,
        encoder_hidden: int,
        encoder_ode_width: int,
        dynamics_width: int,
        dynamics_residual_blocks: int,
        decoder_width: int,
        dropout: float,
        observation_std: float,
        condition_in_encoder: bool,
        condition_in_dynamics: bool,
        substeps: int,
        residual: bool = False,
    ) -> None:
        super().__init__()
        M = D.core()["M"]
        self.residual = bool(residual)
        self.obs_dim = int(obs_dim)
        self.z0_dim = int(z0_dim)
        self.substeps = int(substeps)
        self.condition_in_encoder = bool(condition_in_encoder)
        self.condition_in_dynamics = bool(condition_in_dynamics)
        # A learned per-dimension sigma let the first search attempt lower its NLL by widening
        # sigma instead of reconstructing codes (code MSE stayed ~1 for every trial), so sigma is
        # fixed and searched instead.
        self.obs_std = float(observation_std)
        if self.obs_std <= 0.0:
            raise ValueError("observation_std must be positive")
        self.encoder_ode = M.PlainODEFunc(int(encoder_hidden), int(encoder_ode_width), 1, float(dropout))
        self.gru = nn.GRUCell(self.obs_dim + 2, int(encoder_hidden))
        self.posterior = nn.Sequential(
            nn.Linear(int(encoder_hidden), int(encoder_hidden)), nn.SiLU(), nn.Linear(int(encoder_hidden), 2 * self.z0_dim)
        )
        self.dynamics = M.PlainODEFunc(self.z0_dim, int(dynamics_width), int(dynamics_residual_blocks), float(dropout))
        self.decoder = nn.Sequential(
            nn.Linear(self.z0_dim, int(decoder_width)), nn.SiLU(),
            nn.Linear(int(decoder_width), int(decoder_width)), nn.SiLU(),
            nn.Linear(int(decoder_width), self.obs_dim),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LatentODE":
        model, training = config["model"], config["training"]
        return cls(
            obs_dim=int(model["latent_dim"]),
            z0_dim=int(model["z0_dim"]),
            encoder_hidden=int(model["encoder_hidden"]),
            encoder_ode_width=int(model["encoder_ode_width"]),
            dynamics_width=int(model["dynamics_width"]),
            dynamics_residual_blocks=int(model["dynamics_residual_blocks"]),
            decoder_width=int(model["decoder_width"]),
            dropout=float(model.get("dropout", 0.0)),
            observation_std=float(model["observation_std"]),
            condition_in_encoder=bool(model.get("condition_in_encoder", True)),
            condition_in_dynamics=bool(model.get("condition_in_dynamics", True)),
            substeps=int(training["integration_substeps"]),
            residual=bool(model.get("residual", False)),
        )

    # ----------------------------------------------------------------------------------

    def encode(self, obs: torch.Tensor, times: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior (mu, logvar) at each row's first observation.

        obs [B,K,D], times [B,K], mask [B,K] with every row's valid entries a left-aligned
        prefix (mask[:, 0] all True), condition [B].
        """
        if not bool(mask[:, 0].all()):
            raise ValueError("every row needs at least its first observation")
        batch, steps, _ = obs.shape
        rk4 = D.core()["M"].rk4_step
        d = condition.reshape(-1).to(obs)
        d_enc = d if self.condition_in_encoder else torch.zeros_like(d)
        h = obs.new_zeros(batch, self.gru.hidden_size)
        for k in range(steps - 1, -1, -1):
            valid = mask[:, k]
            if k < steps - 1:
                moving = valid & mask[:, k + 1]
                if bool(moving.any()):
                    start = times[:, k + 1]
                    delta = (times[:, k] - times[:, k + 1]) / float(self.substeps)
                    evolved, current = h, start
                    for _ in range(self.substeps):
                        evolved = rk4(self.encoder_ode, evolved, current, delta, d_enc)
                        current = current + delta
                    h = torch.where(moving[:, None], evolved, h)
                gap = torch.where(mask[:, k + 1], times[:, k + 1] - times[:, k], torch.zeros_like(times[:, k]))
            else:
                gap = torch.zeros_like(times[:, k])
            updated = self.gru(torch.cat((obs[:, k], d_enc[:, None], gap[:, None]), dim=1), h)
            h = torch.where(valid[:, None], updated, h)
        mu, logvar = self.posterior(h).chunk(2, dim=1)
        return mu, logvar.clamp(-12.0, 8.0)

    def decode_at(self, z0: torch.Tensor, t0: torch.Tensor, query_times: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Codes at query_times [B,Q] (non-decreasing, >= t0 or any single target) -> [B,Q,D]."""
        M = D.core()["M"]
        d = condition.reshape(-1).to(z0)
        d_dyn = d if self.condition_in_dynamics else torch.zeros_like(d)
        times = torch.cat((t0.reshape(-1, 1), query_times), dim=1).to(z0)
        states = M.integrate_sequence_rk4(self.dynamics, z0, times, d_dyn, self.substeps)[:, 1:, :]
        return self.decoder(states)

    def loss(
        self,
        obs: torch.Tensor,
        times: torch.Tensor,
        visit_mask: torch.Tensor,
        prefix_mask: torch.Tensor,
        condition: torch.Tensor,
        kl_weight: float,
        free_bits: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mu, logvar = self.encode(obs, times, prefix_mask, condition)
        z0 = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        # Padded query times repeat each row's last real time, so padding integrates over dt = 0.
        prediction = self.decode_at(z0, times[:, 0], times[:, 1:], condition)
        prediction = torch.cat((self.decoder(z0)[:, None, :], prediction), dim=1)
        if self.residual:
            rows = torch.arange(obs.shape[0], device=obs.device)
            last = prefix_mask.sum(dim=1) - 1
            # Difference first: (x_ref + a) - a is not x_ref in floating point; x_ref + (a - a) is.
            prediction = obs[rows, last][:, None, :] + (prediction - prediction[rows, last][:, None, :])
        std = self.obs_std
        nll_elements = 0.5 * ((obs - prediction) / std).square() + math.log(std) + 0.5 * math.log(2.0 * math.pi)
        nll_per_visit = nll_elements.sum(dim=2)
        weights = visit_mask.to(obs)
        nll = ((nll_per_visit * weights).sum(dim=1) / weights.sum(dim=1)).mean()
        kl_dims = 0.5 * (mu.square() + torch.exp(logvar) - 1.0 - logvar)
        kl_floored = torch.clamp(kl_dims.mean(dim=0), min=float(free_bits)).sum()
        total = nll + float(kl_weight) * kl_floored
        mse = (((obs - prediction).square().mean(dim=2) * weights).sum(dim=1) / weights.sum(dim=1)).mean()
        return total, {
            "loss": total.detach(),
            "nll": nll.detach(),
            "kl": kl_dims.sum(dim=1).mean().detach(),
            "kl_floored": kl_floored.detach(),
            "code_mse": mse.detach(),
        }

    # ----------------------------------------------------------------------------------

    def _from_z0(self, z0, obs, times, mask, condition, target_times) -> torch.Tensor:
        if not self.residual:
            return self.decode_at(z0, times[:, 0], target_times, condition)
        rows = torch.arange(obs.shape[0], device=obs.device)
        last = mask.sum(dim=1) - 1
        query = torch.cat((times[rows, last][:, None], target_times), dim=1)
        decoded = self.decode_at(z0, times[:, 0], query, condition)
        return obs[rows, last][:, None, :] + (decoded[:, 1:] - decoded[:, :1])

    @torch.no_grad()
    def predict(self, obs, times, mask, condition, target_times, samples: int = 0) -> torch.Tensor:
        """Codes at target_times [B,Q] from each row's observed prefix; posterior mean unless samples > 0."""
        mu, logvar = self.encode(obs, times, mask, condition)
        if samples <= 0:
            return self._from_z0(mu, obs, times, mask, condition, target_times)
        draws = [self._from_z0(mu + torch.randn_like(mu) * torch.exp(0.5 * logvar), obs, times, mask, condition, target_times)
                 for _ in range(samples)]
        return torch.stack(draws, dim=0)

    def transport(self, latent, source_time, target_time, condition, context=None, context_time=None) -> torch.Tensor:
        """One-observation adapter so pair-based evaluators treat the Latent ODE like a transport."""
        del context, context_time
        obs = latent[:, None, :]
        times = source_time.reshape(-1, 1).to(latent)
        mask = torch.ones(latent.shape[0], 1, dtype=torch.bool, device=latent.device)
        condition = condition.reshape(-1).to(latent)
        mu, _logvar = self.encode(obs, times, mask, condition)
        return self._from_z0(mu, obs, times, mask, condition, target_time.reshape(-1, 1).to(latent))[:, 0, :]
