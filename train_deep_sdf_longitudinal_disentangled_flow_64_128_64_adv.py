#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import signal
import sys
import os
import logging
import math
import json
import time
import copy
import random
import itertools
import numpy as np
import re

import deep_sdf
from deep_sdf import mesh, metrics, lr_scheduling, plotting, utils
from deep_sdf.loss import (
    CovarianceLoss,
    GradientMetricIsotropyLoss,
    IsometryLoss,
    GMMPriorLoss,
    select_near_surface_points,
)
import deep_sdf.workspace as ws
import reconstruct
from networks.longitudinal_disentangled_flow_64_128_64_adv import build_temporal_flow


class TeeStream:
    def __init__(self, stream, file_obj):
        self._stream = stream
        self._file_obj = file_obj

    def write(self, data):
        self._stream.write(data)
        self._file_obj.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._file_obj.flush()

    def isatty(self):
        return self._stream.isatty()


def save_model(experiment_directory, filename, decoder, temporal_flow, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": decoder.state_dict(),
            "flow_state_dict": temporal_flow.state_dict(),
        },
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def load_model_and_flow(experiment_directory, checkpoint, decoder, temporal_flow):
    filename = os.path.join(
        experiment_directory, ws.model_params_subdir, checkpoint + ".pth"
    )
    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename, map_location="cpu")
    state_dict = data["model_state_dict"]

    model_is_dp = isinstance(decoder, torch.nn.DataParallel)
    state_has_module = any(k.startswith("module.") for k in state_dict.keys())
    if model_is_dp and not state_has_module:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif (not model_is_dp) and state_has_module:
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    decoder.load_state_dict(state_dict)

    flow_state = data.get("flow_state_dict", None)
    if flow_state is None:
        logging.warning(
            "Checkpoint %s has no flow_state_dict. Temporal flow keeps initialized weights.",
            checkpoint,
        )
    else:
        temporal_flow.load_state_dict(flow_state)

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]

def load_pretrained_decoder(decoder, pretrained_dir, checkpoint):
    filename = os.path.join(pretrained_dir, ws.model_params_subdir, checkpoint + ".pth")
    if not os.path.isfile(filename):
        raise RuntimeError(f'pretrained model state dict "{filename}" does not exist')

    data = torch.load(filename, map_location="cpu")
    state_dict = data["model_state_dict"]

    model_is_dp = isinstance(decoder, torch.nn.DataParallel)
    state_has_module = any(k.startswith("module.") for k in state_dict.keys())

    if model_is_dp and not state_has_module:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif (not model_is_dp) and state_has_module:
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    decoder.load_state_dict(state_dict)
    return data.get("epoch", None)


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


class TemporalFlowMLP(torch.nn.Module):
    def __init__(self, latent_size, hidden_dims, age_condition_dim=0):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("Flow hidden dims must be non-empty.")
        self.age_condition_dim = max(0, int(age_condition_dim))
        dims = [latent_size + 2 + self.age_condition_dim] + list(hidden_dims) + [latent_size]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.ReLU(inplace=True))
        self.net = torch.nn.Sequential(*layers)

    def _prepare_age_condition(self, z, age_cond):
        if self.age_condition_dim <= 0:
            return None

        if age_cond is None:
            return torch.zeros(
                z.shape[0],
                self.age_condition_dim,
                device=z.device,
                dtype=z.dtype,
            )

        age = age_cond
        if age.dim() == 1:
            age = age.unsqueeze(1)
        age = age.to(device=z.device, dtype=z.dtype)

        if age.shape[1] == self.age_condition_dim:
            return age
        if age.shape[1] == 1 and self.age_condition_dim > 1:
            return age.repeat(1, self.age_condition_dim)

        raise ValueError(
            f"Invalid age condition width: got {age.shape[1]}, "
            f"expected {self.age_condition_dim}"
        )

    def forward(self, z, s, t, age_cond=None):
        if s.dim() == 1:
            s = s.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        parts = [z, s, t]
        if self.age_condition_dim > 0:
            parts.append(self._prepare_age_condition(z, age_cond))
        x = torch.cat(parts, dim=1)
        return self.net(x)


def apply_temporal_flow(temporal_flow, z, s, t, age_cond=None):
    return z + (t - s) * temporal_flow(z, s, t, age_cond=age_cond)


def has_disentangled_velocity_components(temporal_flow):
    flow = temporal_flow.module if hasattr(temporal_flow, "module") else temporal_flow
    return callable(getattr(flow, "velocity_components", None))


def get_disentangled_velocity_components(temporal_flow, z, s, t, age_cond=None):
    flow = temporal_flow.module if hasattr(temporal_flow, "module") else temporal_flow
    if not callable(getattr(flow, "velocity_components", None)):
        return None
    return flow.velocity_components(z, s, t, age_cond=age_cond)


def compute_disentangled_velocity_losses(
    temporal_flow,
    z,
    s,
    t,
    age_cond=None,
    use_disease_zero=False,
    disease_zero_lambda=0.0,
    use_residual=False,
    residual_lambda=0.0,
    use_residual_diagnosis_covariance=False,
    residual_diagnosis_covariance_lambda=0.0,
    use_disease_margin=False,
    disease_margin_lambda=0.0,
    disease_margin=0.05,
    use_adversarial_leakage=False,
    adversarial_age_lambda=0.0,
    adversarial_residual_lambda=0.0,
    adversarial_grl_lambda=1.0,
    use_disease_classification=False,
    disease_classification_lambda=0.0,
):
    zero = z.new_tensor(0.0)
    out = {
        "loss": zero,
        "disease_zero": zero,
        "residual": zero,
        "residual_diagnosis_covariance": zero,
        "disease_margin": zero,
        "age_adversarial": zero,
        "residual_adversarial": zero,
        "disease_classification": zero,
        "age_adversarial_bce": zero,
        "residual_adversarial_bce": zero,
        "disease_classification_bce": zero,
        "age_adversarial_acc": zero,
        "residual_adversarial_acc": zero,
        "disease_classification_acc": zero,
        "age_norm": zero,
        "disease_raw_norm": zero,
        "disease_norm": zero,
        "disease_raw_norm_healthy": zero,
        "disease_raw_norm_diseased": zero,
        "residual_norm": zero,
    }
    components = get_disentangled_velocity_components(
        temporal_flow, z, s, t, age_cond=age_cond
    )
    if components is None:
        return out

    diagnosis = components["diagnosis"].view(-1)
    v_age = components["age"]
    v_dis_raw = components["disease_raw"]
    v_dis = components["disease"]
    v_res = components["residual"]
    target = diagnosis.view(-1, 1)
    flow = temporal_flow.module if hasattr(temporal_flow, "module") else temporal_flow

    out["age_norm"] = torch.mean(torch.norm(v_age, dim=1)).detach()
    out["disease_raw_norm"] = torch.mean(torch.norm(v_dis_raw, dim=1)).detach()
    out["disease_norm"] = torch.mean(torch.norm(v_dis, dim=1)).detach()
    out["residual_norm"] = torch.mean(torch.norm(v_res, dim=1)).detach()
    healthy_mask = diagnosis < 0.5
    diseased_mask = diagnosis >= 0.5
    if torch.any(healthy_mask):
        out["disease_raw_norm_healthy"] = torch.mean(
            torch.norm(v_dis_raw[healthy_mask], dim=1)
        ).detach()
    if torch.any(diseased_mask):
        out["disease_raw_norm_diseased"] = torch.mean(
            torch.norm(v_dis_raw[diseased_mask], dim=1)
        ).detach()

    if use_disease_zero and float(disease_zero_lambda) > 0.0:
        if torch.any(healthy_mask):
            raw = v_dis_raw[healthy_mask]
            disease_zero = float(disease_zero_lambda) * torch.mean(raw.pow(2))
            out["disease_zero"] = disease_zero
            out["loss"] = out["loss"] + disease_zero

    if (
        use_disease_margin
        and float(disease_margin_lambda) > 0.0
        and torch.any(diseased_mask)
    ):
        raw_norm = torch.norm(v_dis_raw[diseased_mask], dim=1)
        margin = z.new_tensor(float(disease_margin))
        disease_margin_loss = float(disease_margin_lambda) * torch.mean(
            torch.relu(margin - raw_norm).pow(2)
        )
        out["disease_margin"] = disease_margin_loss
        out["loss"] = out["loss"] + disease_margin_loss

    if (
        use_disease_classification
        and float(disease_classification_lambda) > 0.0
        and callable(getattr(flow, "disease_logits", None))
    ):
        logits = flow.disease_logits(v_dis_raw)
        raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        weighted_bce = float(disease_classification_lambda) * raw_bce
        out["disease_classification"] = weighted_bce
        out["disease_classification_bce"] = raw_bce.detach()
        out["disease_classification_acc"] = (
            ((torch.sigmoid(logits.detach()) >= 0.5).float() == target).float().mean()
        )
        out["loss"] = out["loss"] + weighted_bce

    if use_adversarial_leakage and callable(getattr(flow, "age_leakage_logits", None)):
        if float(adversarial_age_lambda) > 0.0:
            logits = flow.age_leakage_logits(
                v_age, grl_lambda=float(adversarial_grl_lambda)
            )
            raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target
            )
            weighted_bce = float(adversarial_age_lambda) * raw_bce
            out["age_adversarial"] = weighted_bce
            out["age_adversarial_bce"] = raw_bce.detach()
            out["age_adversarial_acc"] = (
                ((torch.sigmoid(logits.detach()) >= 0.5).float() == target)
                .float()
                .mean()
            )
            out["loss"] = out["loss"] + weighted_bce

    if use_adversarial_leakage and callable(
        getattr(flow, "residual_leakage_logits", None)
    ):
        if float(adversarial_residual_lambda) > 0.0:
            logits = flow.residual_leakage_logits(
                v_res, grl_lambda=float(adversarial_grl_lambda)
            )
            raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target
            )
            weighted_bce = float(adversarial_residual_lambda) * raw_bce
            out["residual_adversarial"] = weighted_bce
            out["residual_adversarial_bce"] = raw_bce.detach()
            out["residual_adversarial_acc"] = (
                ((torch.sigmoid(logits.detach()) >= 0.5).float() == target)
                .float()
                .mean()
            )
            out["loss"] = out["loss"] + weighted_bce

    if use_residual and float(residual_lambda) > 0.0:
        residual = float(residual_lambda) * torch.mean(v_res.pow(2))
        out["residual"] = residual
        out["loss"] = out["loss"] + residual

    if (
        use_residual_diagnosis_covariance
        and float(residual_diagnosis_covariance_lambda) > 0.0
        and diagnosis.numel() >= 2
        and torch.any(diagnosis < 0.5)
        and torch.any(diagnosis >= 0.5)
    ):
        d_centered = diagnosis.view(-1, 1) - torch.mean(diagnosis)
        res_centered = v_res - torch.mean(v_res, dim=0, keepdim=True)
        cov = torch.mean(res_centered * d_centered, dim=0)
        cov_loss = float(residual_diagnosis_covariance_lambda) * torch.mean(cov.pow(2))
        out["residual_diagnosis_covariance"] = cov_loss
        out["loss"] = out["loss"] + cov_loss

    return out


def _sample_sorted_time_pairs(
    count,
    device,
    mode="uniform_01",
    time_values=None,
    dtype=torch.float32,
    uniform_low=0.0,
    uniform_high=1.0,
    mixed_adjacent_ratio=0.5,
    mixed_adjacent_max_gap_ratio=0.25,
    mixed_far_min_gap_ratio=0.5,
):
    if int(count) <= 0:
        return None, None

    if mode == "from_values":
        if time_values is None or time_values.numel() < 2:
            return None, None
        n = time_values.numel()
        idx_a = torch.randint(0, n, (int(count),), device=device)
        idx_b = torch.randint(0, n, (int(count),), device=device)
        s = time_values.index_select(0, idx_a).unsqueeze(1)
        t = time_values.index_select(0, idx_b).unsqueeze(1)
    elif mode == "mixed_adjacent_far":
        lo = float(uniform_low)
        hi = float(uniform_high)
        if hi < lo:
            lo, hi = hi, lo
        span = hi - lo
        if abs(span) <= 1e-12:
            return None, None

        adj_ratio = min(max(float(mixed_adjacent_ratio), 0.0), 1.0)
        n_adj = int(round(int(count) * adj_ratio))
        n_adj = max(0, min(int(count), n_adj))
        n_far = int(count) - n_adj
        adj_max = span * min(max(float(mixed_adjacent_max_gap_ratio), 0.0), 1.0)
        far_min = span * min(max(float(mixed_far_min_gap_ratio), 0.0), 1.0)

        def _collect_pairs(num_pairs, want_adjacent):
            if num_pairs <= 0:
                empty = torch.empty((0, 1), device=device, dtype=dtype)
                return empty, empty

            s_chunks = []
            t_chunks = []
            remaining = int(num_pairs)
            for _ in range(8):
                if remaining <= 0:
                    break
                draw = max(remaining * 2, 16)
                s_try = lo + span * torch.rand(draw, 1, device=device, dtype=dtype)
                t_try = lo + span * torch.rand(draw, 1, device=device, dtype=dtype)
                st_try = torch.sort(torch.cat([s_try, t_try], dim=1), dim=1).values
                s_try = st_try[:, 0:1]
                t_try = st_try[:, 1:2]
                gap = (t_try - s_try).squeeze(1)
                mask = gap <= adj_max if want_adjacent else gap >= far_min
                if mask.any():
                    s_sel = s_try[mask]
                    t_sel = t_try[mask]
                    take = min(remaining, s_sel.shape[0])
                    s_chunks.append(s_sel[:take])
                    t_chunks.append(t_sel[:take])
                    remaining -= take

            if remaining > 0:
                s_fill = lo + span * torch.rand(remaining, 1, device=device, dtype=dtype)
                t_fill = lo + span * torch.rand(remaining, 1, device=device, dtype=dtype)
                st_fill = torch.sort(torch.cat([s_fill, t_fill], dim=1), dim=1).values
                s_chunks.append(st_fill[:, 0:1])
                t_chunks.append(st_fill[:, 1:2])

            return torch.cat(s_chunks, dim=0), torch.cat(t_chunks, dim=0)

        s_adj, t_adj = _collect_pairs(n_adj, want_adjacent=True)
        s_far, t_far = _collect_pairs(n_far, want_adjacent=False)
        s = torch.cat([s_adj, s_far], dim=0)
        t = torch.cat([t_adj, t_far], dim=0)
        if s.shape[0] > 1:
            perm = torch.randperm(s.shape[0], device=device)
            s = s.index_select(0, perm)
            t = t.index_select(0, perm)
    else:
        lo = float(uniform_low)
        hi = float(uniform_high)
        if hi < lo:
            lo, hi = hi, lo
        if abs(hi - lo) <= 1e-12:
            return None, None
        s = lo + (hi - lo) * torch.rand(int(count), 1, device=device, dtype=dtype)
        t = lo + (hi - lo) * torch.rand(int(count), 1, device=device, dtype=dtype)

    st = torch.sort(torch.cat([s, t], dim=1), dim=1).values
    s = st[:, 0:1]
    t = st[:, 1:2]
    if mode != "from_values":
        return s, t
    valid = ((t - s).abs() > 1e-8).squeeze(1)
    if valid.sum().item() == 0:
        return None, None
    return s[valid], t[valid]


def _sample_sorted_time_triplets(
    count,
    device,
    mode="uniform_01",
    time_values=None,
    dtype=torch.float32,
    uniform_low=0.0,
    uniform_high=1.0,
):
    if int(count) <= 0:
        return None, None, None

    if mode == "from_values":
        if time_values is None or time_values.numel() < 3:
            return None, None, None
        n = time_values.numel()
        idx_a = torch.randint(0, n, (int(count),), device=device)
        idx_b = torch.randint(0, n, (int(count),), device=device)
        idx_c = torch.randint(0, n, (int(count),), device=device)
        a = time_values.index_select(0, idx_a).unsqueeze(1)
        b = time_values.index_select(0, idx_b).unsqueeze(1)
        c = time_values.index_select(0, idx_c).unsqueeze(1)
    else:
        lo = float(uniform_low)
        hi = float(uniform_high)
        if hi < lo:
            lo, hi = hi, lo
        if abs(hi - lo) <= 1e-12:
            return None, None, None
        scale = hi - lo
        a = lo + scale * torch.rand(int(count), 1, device=device, dtype=dtype)
        b = lo + scale * torch.rand(int(count), 1, device=device, dtype=dtype)
        c = lo + scale * torch.rand(int(count), 1, device=device, dtype=dtype)

    abc = torch.sort(torch.cat([a, b, c], dim=1), dim=1).values
    s = abc[:, 0:1]
    r = abc[:, 1:2]
    t = abc[:, 2:3]
    if mode != "from_values":
        return s, r, t
    valid = (((r - s).abs() > 1e-8) & ((t - r).abs() > 1e-8)).squeeze(1)
    if valid.sum().item() == 0:
        return None, None, None
    return s[valid], r[valid], t[valid]


def _apply_temporal_flow_interval(
    temporal_flow,
    z_start,
    t_start,
    t_end,
    max_dt=0.0,
    age_end_cond=None,
):
    dt_total = float(t_end) - float(t_start)
    if abs(dt_total) <= 1e-12:
        return z_start

    max_dt_val = float(max_dt) if max_dt is not None else 0.0
    if max_dt_val <= 0.0:
        s = torch.full(
            (z_start.shape[0], 1),
            float(t_start),
            device=z_start.device,
            dtype=z_start.dtype,
        )
        t = torch.full(
            (z_start.shape[0], 1),
            float(t_end),
            device=z_start.device,
            dtype=z_start.dtype,
        )
        return apply_temporal_flow(temporal_flow, z_start, s, t, age_cond=age_end_cond)

    num_steps = max(1, int(math.ceil(abs(dt_total) / max_dt_val)))
    step = dt_total / float(num_steps)
    z = z_start
    cur_t = float(t_start)
    for _ in range(num_steps):
        nxt_t = cur_t + step
        s = torch.full((z.shape[0], 1), cur_t, device=z.device, dtype=z.dtype)
        t = torch.full((z.shape[0], 1), nxt_t, device=z.device, dtype=z.dtype)
        z = apply_temporal_flow(temporal_flow, z, s, t, age_cond=age_end_cond)
        cur_t = nxt_t
    return z


def _sample_query_points(xyz_chunk, num_points):
    if xyz_chunk is None or xyz_chunk.numel() == 0:
        return None
    k = int(num_points)
    if k <= 0:
        return None
    n = int(xyz_chunk.shape[0])
    if n <= 0:
        return None
    k = min(k, n)
    if k == n:
        return xyz_chunk
    idx = torch.randint(0, n, (k,), device=xyz_chunk.device)
    return xyz_chunk.index_select(0, idx)


def _decode_sdf_on_shared_points(
    decoder,
    latents,
    query_xyz,
    clamp_min=None,
    clamp_max=None,
):
    if latents is None or query_xyz is None:
        return None
    if latents.numel() == 0 or query_xyz.numel() == 0:
        return None

    batch_size = int(latents.shape[0])
    num_points = int(query_xyz.shape[0])
    if batch_size <= 0 or num_points <= 0:
        return None

    z = latents.unsqueeze(1).expand(-1, num_points, -1).reshape(batch_size * num_points, -1)
    x = query_xyz.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * num_points, 3)
    pred = decoder(torch.cat([z, x], dim=1)).reshape(batch_size, num_points, -1)

    if clamp_min is not None and clamp_max is not None:
        pred = torch.clamp(pred, float(clamp_min), float(clamp_max))
    return pred


def _compute_shape_cocycle_sdf_loss(
    decoder,
    z_a,
    z_b,
    query_xyz,
    clamp_min=None,
    clamp_max=None,
):
    pred_a = _decode_sdf_on_shared_points(
        decoder,
        z_a,
        query_xyz,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    pred_b = _decode_sdf_on_shared_points(
        decoder,
        z_b,
        query_xyz,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    if pred_a is None or pred_b is None:
        return None
    return torch.mean(torch.abs(pred_a - pred_b))


def _build_subject_to_scan_indices(scan_to_subject_idx_cpu, num_subjects):
    subject_to_scan_indices = {subject_idx: [] for subject_idx in range(num_subjects)}
    for scan_idx, subject_idx in enumerate(scan_to_subject_idx_cpu.tolist()):
        subject_to_scan_indices[int(subject_idx)].append(int(scan_idx))
    return subject_to_scan_indices


def _sample_distinct_scan_pairs(scan_indices, pair_count):
    pairs = list(itertools.combinations([int(x) for x in scan_indices], 2))
    if len(pairs) == 0 or int(pair_count) <= 0:
        return []
    return random.sample(pairs, min(int(pair_count), len(pairs)))


def _load_auxiliary_sdf_samples(sdf_dataset, scan_idx, num_samples):
    scan_idx = int(scan_idx)
    if sdf_dataset.load_ram:
        return deep_sdf.data.unpack_sdf_samples_from_ram(
            sdf_dataset.loaded_data[scan_idx], int(num_samples)
        )
    filename = os.path.join(sdf_dataset.data_source, sdf_dataset.npyfiles[scan_idx])
    return deep_sdf.data.unpack_sdf_samples(filename, int(num_samples))


def _compute_real_scan_pair_direction_losses(
    decoder,
    temporal_flow,
    z_source,
    source_time,
    z_target,
    target_time,
    target_condition,
    target_sdf_data,
    clamp_min,
    clamp_max,
):
    device = z_source.device
    dtype = z_source.dtype
    source_t = torch.full((z_source.shape[0], 1), float(source_time), device=device, dtype=dtype)
    target_t = torch.full((z_source.shape[0], 1), float(target_time), device=device, dtype=dtype)
    transported = apply_temporal_flow(
        temporal_flow,
        z_source,
        source_t,
        target_t,
        age_cond=target_condition,
    )

    target_sdf_data = target_sdf_data.to(device=device, dtype=dtype, non_blocking=True)
    target_xyz = target_sdf_data[:, 0:3]
    target_sdf = target_sdf_data[:, 3].unsqueeze(1)
    target_sdf = torch.clamp(target_sdf, float(clamp_min), float(clamp_max))
    transported_expanded = transported.expand(target_xyz.shape[0], -1)
    pred_sdf = decoder(torch.cat([transported_expanded, target_xyz], dim=1))
    pred_sdf = torch.clamp(pred_sdf, float(clamp_min), float(clamp_max))

    reconstruction_loss = torch.mean(torch.abs(pred_sdf - target_sdf))
    latent_loss = torch.mean((transported - z_target) ** 2)
    return reconstruction_loss, latent_loss


def _compute_real_scan_pair_batch_losses(
    decoder,
    temporal_flow,
    lat_vecs,
    unique_subjects,
    subject_to_scan_indices,
    sdf_dataset,
    scan_to_time_cpu,
    scan_age_condition,
    subject_baseline_time,
    use_age_conditioning,
    pairs_per_subject,
    num_samples,
    use_forward,
    use_backward,
    clamp_min,
    clamp_max,
):
    forward_rec_sum = 0.0
    forward_latent_sum = 0.0
    backward_rec_sum = 0.0
    backward_latent_sum = 0.0
    forward_count = 0
    backward_count = 0
    auxiliary_sample_cache = {}

    def _samples_for_scan(scan_idx):
        scan_idx = int(scan_idx)
        if scan_idx not in auxiliary_sample_cache:
            auxiliary_sample_cache[scan_idx] = _load_auxiliary_sdf_samples(
                sdf_dataset, scan_idx, num_samples
            )
        return auxiliary_sample_cache[scan_idx]

    for subject_idx in unique_subjects:
        subject_idx_int = int(subject_idx.item())
        scan_pairs = _sample_distinct_scan_pairs(
            subject_to_scan_indices[subject_idx_int], pairs_per_subject
        )
        if len(scan_pairs) == 0:
            continue

        subject_anchor = lat_vecs(subject_idx.view(1))
        baseline_time_value = float(subject_baseline_time[subject_idx_int].item())
        baseline_time_tensor = torch.full(
            (1, 1),
            baseline_time_value,
            device=subject_anchor.device,
            dtype=subject_anchor.dtype,
        )

        for scan_a, scan_b in scan_pairs:
            if float(scan_to_time_cpu[scan_a].item()) <= float(
                scan_to_time_cpu[scan_b].item()
            ):
                forward_source_idx, forward_target_idx = scan_a, scan_b
            else:
                forward_source_idx, forward_target_idx = scan_b, scan_a

            source_time_value = float(scan_to_time_cpu[forward_source_idx].item())
            target_time_value = float(scan_to_time_cpu[forward_target_idx].item())
            source_time_tensor = torch.full(
                (1, 1),
                source_time_value,
                device=subject_anchor.device,
                dtype=subject_anchor.dtype,
            )
            target_time_tensor = torch.full(
                (1, 1),
                target_time_value,
                device=subject_anchor.device,
                dtype=subject_anchor.dtype,
            )
            source_condition = None
            target_condition = None
            if use_age_conditioning:
                source_condition = scan_age_condition[forward_source_idx].view(1, -1)
                target_condition = scan_age_condition[forward_target_idx].view(1, -1)

            z_source = apply_temporal_flow(
                temporal_flow,
                subject_anchor,
                baseline_time_tensor,
                source_time_tensor,
                age_cond=source_condition,
            )
            z_target = apply_temporal_flow(
                temporal_flow,
                subject_anchor,
                baseline_time_tensor,
                target_time_tensor,
                age_cond=target_condition,
            )

            if use_forward:
                target_sdf_data = _samples_for_scan(forward_target_idx)
                forward_rec, forward_latent = _compute_real_scan_pair_direction_losses(
                    decoder,
                    temporal_flow,
                    z_source,
                    source_time_value,
                    z_target,
                    target_time_value,
                    target_condition,
                    target_sdf_data,
                    clamp_min,
                    clamp_max,
                )
                forward_rec_sum = forward_rec_sum + forward_rec
                forward_latent_sum = forward_latent_sum + forward_latent
                forward_count += 1

            if use_backward:
                source_sdf_data = _samples_for_scan(forward_source_idx)
                backward_rec, backward_latent = _compute_real_scan_pair_direction_losses(
                    decoder,
                    temporal_flow,
                    z_target,
                    target_time_value,
                    z_source,
                    source_time_value,
                    source_condition,
                    source_sdf_data,
                    clamp_min,
                    clamp_max,
                )
                backward_rec_sum = backward_rec_sum + backward_rec
                backward_latent_sum = backward_latent_sum + backward_latent
                backward_count += 1

    return {
        "forward_reconstruction": (
            forward_rec_sum / forward_count if forward_count > 0 else None
        ),
        "forward_latent": (
            forward_latent_sum / forward_count if forward_count > 0 else None
        ),
        "backward_reconstruction": (
            backward_rec_sum / backward_count if backward_count > 0 else None
        ),
        "backward_latent": (
            backward_latent_sum / backward_count if backward_count > 0 else None
        ),
        "forward_count": forward_count,
        "backward_count": backward_count,
    }


def _parse_subject_and_timepoint(shape_name):
    # Starmen-style names: "...__sid-0001__...__tp-03__..."
    sid_match = re.search(r"__sid-(\d+)__", shape_name)
    tp_match = re.search(r"__tp-(\d+)__", shape_name)
    if sid_match is not None and tp_match is not None:
        return sid_match.group(1), int(tp_match.group(1))

    # Torus-style names: "ID_000_t0"
    torus_match = re.match(r"^(?:ID|id)_(\d+)_t(\d+)$", shape_name)
    if torus_match is not None:
        return torus_match.group(1), int(torus_match.group(2))

    raise RuntimeError(
        f"Could not parse sid/tp from shape name '{shape_name}'. "
        "Supported patterns are Starmen-style '__sid-0001__tp-03__' "
        "and Torus-style 'ID_000_t0'."
    )


def build_longitudinal_metadata(
    npyfiles,
    time_normalization_mode,
    scan_to_time_map=None,
    time_key_name="time",
):
    subject_ids = []
    timepoints = []
    missing_time = []
    for fpath in npyfiles:
        base = os.path.splitext(os.path.basename(fpath))[0]
        sid, tp = _parse_subject_and_timepoint(base)
        subject_ids.append(sid)
        if scan_to_time_map is not None:
            tval = scan_to_time_map.get(base, None)
            if tval is None:
                missing_time.append(base)
                timepoints.append(float(tp))
            else:
                timepoints.append(float(tval))
        else:
            timepoints.append(float(tp))

    if scan_to_time_map is not None and missing_time:
        preview = ", ".join(missing_time[:5])
        raise RuntimeError(
            f"Missing longitudinal time metadata '{time_key_name}' for {len(missing_time)} scans. "
            f"First missing keys: {preview}"
        )

    unique_subject_ids = sorted(set(subject_ids))
    subject_to_index = {sid: idx for idx, sid in enumerate(unique_subject_ids)}
    scan_to_subject_idx = torch.tensor(
        [subject_to_index[sid] for sid in subject_ids], dtype=torch.long
    )
    raw_time = torch.tensor(timepoints, dtype=torch.float32)
    normalized_time = torch.zeros_like(raw_time)

    mode = str(time_normalization_mode).lower()
    if mode == "none":
        normalized_time = raw_time
    elif mode == "global_minmax":
        t_min = float(raw_time.min())
        t_max = float(raw_time.max())
        denom = max(t_max - t_min, 1e-8)
        normalized_time = (raw_time - t_min) / denom
    elif mode == "subject_minmax":
        for sid in unique_subject_ids:
            mask = torch.tensor([x == sid for x in subject_ids], dtype=torch.bool)
            times_sid = raw_time[mask]
            t_min = float(times_sid.min())
            t_max = float(times_sid.max())
            denom = max(t_max - t_min, 1e-8)
            normalized_time[mask] = (times_sid - t_min) / denom
    else:
        raise ValueError(
            f"Unknown LongitudinalTimeNormalization='{time_normalization_mode}'. "
            "Use one of: none, global_minmax, subject_minmax."
        )

    return {
        "subject_ids": subject_ids,
        "timepoints_raw": raw_time,
        "timepoints": normalized_time,
        "scan_to_subject_idx": scan_to_subject_idx,
        "num_subjects": len(unique_subject_ids),
        "subject_to_index": subject_to_index,
    }


def _scan_key_from_path(path_str):
    return os.path.splitext(os.path.basename(path_str))[0]


def _load_scan_age_map_from_labels(labels_path, age_key):
    labels_obj = torch.load(labels_path, map_location="cpu")
    scan_to_age = {}

    if isinstance(labels_obj, dict) and "records" in labels_obj and isinstance(labels_obj["records"], list):
        for rec in labels_obj["records"]:
            if not isinstance(rec, dict):
                continue
            if age_key not in rec:
                continue
            mesh_path = rec.get("mesh_path", None)
            if mesh_path is None:
                continue
            scan_key = _scan_key_from_path(mesh_path)
            scan_to_age[scan_key] = float(rec[age_key])
    elif isinstance(labels_obj, dict):
        # Backward-compatible fallback for {"ID_000_t0": ...} maps.
        for key, value in labels_obj.items():
            if isinstance(value, dict):
                if age_key in value:
                    scan_to_age[str(key)] = float(value[age_key])
            elif torch.is_tensor(value):
                flat = value.detach().cpu().view(-1)
                if flat.numel() > 0:
                    scan_to_age[str(key)] = float(flat[0].item())
            else:
                try:
                    scan_to_age[str(key)] = float(value)
                except Exception:
                    pass
    else:
        raise RuntimeError(
            f"Unsupported labels format at {labels_path}: {type(labels_obj)}"
        )

    if len(scan_to_age) == 0:
        raise RuntimeError(
            f"No scan->age entries could be parsed from {labels_path} with key '{age_key}'."
        )

    return scan_to_age


def build_scan_age_tensor(npyfiles, scan_to_age_map, age_condition_key):
    vals = []
    missing = []
    for fpath in npyfiles:
        scan_key = _scan_key_from_path(fpath)
        age_val = scan_to_age_map.get(scan_key, None)
        if age_val is None:
            missing.append(scan_key)
            vals.append(0.0)
        else:
            vals.append(float(age_val))
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Missing age metadata for {len(missing)} scans using key '{age_condition_key}'. "
            f"First missing keys: {preview}"
        )
    return torch.tensor(vals, dtype=torch.float32)


def build_scan_condition_tensor(npyfiles, scan_to_age_map_dict, condition_keys):
    cond_cols = []
    for key in condition_keys:
        cond_cols.append(build_scan_age_tensor(npyfiles, scan_to_age_map_dict[key], key))
    if len(cond_cols) == 0:
        raise RuntimeError("Condition key list is empty.")
    return torch.stack(cond_cols, dim=1)


def compute_subject_baseline_time(
    scan_to_subject_idx_cpu,
    scan_to_time_cpu,
    num_subjects,
):
    time_inf = float("inf")
    subject_time = torch.full((num_subjects,), time_inf, dtype=torch.float32)

    for scan_idx in range(scan_to_subject_idx_cpu.numel()):
        subj = int(scan_to_subject_idx_cpu[scan_idx].item())
        tval = float(scan_to_time_cpu[scan_idx].item())
        if tval < float(subject_time[subj].item()):
            subject_time[subj] = tval

    if torch.isinf(subject_time).any():
        bad = torch.nonzero(torch.isinf(subject_time), as_tuple=False).view(-1)
        raise RuntimeError(
            f"Found {bad.numel()} subjects without baseline time metadata."
        )

    return subject_time


def compute_subject_baseline_time_and_condition(
    scan_to_subject_idx_cpu,
    scan_to_time_cpu,
    scan_to_condition_cpu,
    num_subjects,
):
    if scan_to_condition_cpu.dim() != 2:
        raise RuntimeError(
            "scan_to_condition_cpu must have shape [num_scans, condition_dim]"
        )
    time_inf = float("inf")
    subject_time = torch.full((num_subjects,), time_inf, dtype=torch.float32)
    cond_dim = int(scan_to_condition_cpu.shape[1])
    subject_condition = torch.zeros((num_subjects, cond_dim), dtype=torch.float32)

    for scan_idx in range(scan_to_subject_idx_cpu.numel()):
        subj = int(scan_to_subject_idx_cpu[scan_idx].item())
        tval = float(scan_to_time_cpu[scan_idx].item())
        if tval < float(subject_time[subj].item()):
            subject_time[subj] = tval
            subject_condition[subj] = scan_to_condition_cpu[scan_idx]

    if torch.isinf(subject_time).any():
        bad = torch.nonzero(torch.isinf(subject_time), as_tuple=False).view(-1)
        raise RuntimeError(
            f"Found {bad.numel()} subjects without baseline time/condition metadata."
        )

    return subject_time, subject_condition


def build_subject_condition_linear_fit(
    scan_to_subject_idx_cpu,
    scan_to_time_cpu,
    scan_to_condition_cpu,
    num_subjects,
):
    if scan_to_condition_cpu.dim() != 2:
        raise RuntimeError(
            "scan_to_condition_cpu must have shape [num_scans, condition_dim]"
        )
    cond_dim = int(scan_to_condition_cpu.shape[1])
    subject_t_min = torch.full((num_subjects,), float("inf"), dtype=torch.float32)
    subject_t_max = torch.full((num_subjects,), float("-inf"), dtype=torch.float32)
    cond_at_min = torch.zeros((num_subjects, cond_dim), dtype=torch.float32)
    cond_at_max = torch.zeros((num_subjects, cond_dim), dtype=torch.float32)

    for scan_idx in range(scan_to_subject_idx_cpu.numel()):
        subj = int(scan_to_subject_idx_cpu[scan_idx].item())
        tval = float(scan_to_time_cpu[scan_idx].item())
        cval = scan_to_condition_cpu[scan_idx]
        if tval < float(subject_t_min[subj].item()):
            subject_t_min[subj] = tval
            cond_at_min[subj] = cval
        if tval > float(subject_t_max[subj].item()):
            subject_t_max[subj] = tval
            cond_at_max[subj] = cval

    if torch.isinf(subject_t_min).any() or torch.isinf(subject_t_max).any():
        bad = torch.nonzero(
            torch.isinf(subject_t_min) | torch.isinf(subject_t_max), as_tuple=False
        ).view(-1)
        raise RuntimeError(
            f"Found {bad.numel()} subjects without condition/time metadata for linear fit."
        )

    dt = (subject_t_max - subject_t_min).unsqueeze(1)
    safe_dt = torch.clamp(dt, min=1e-8)
    slopes = (cond_at_max - cond_at_min) / safe_dt
    slopes = torch.where(dt <= 1e-8, torch.zeros_like(slopes), slopes)
    intercepts = cond_at_min - slopes * subject_t_min.unsqueeze(1)

    return {
        "t_min": subject_t_min,
        "t_max": subject_t_max,
        "slopes": slopes,
        "intercepts": intercepts,
    }


def subject_condition_at_time(subject_indices, sampled_time, subject_condition_fit, device, dtype):
    if subject_condition_fit is None:
        return None
    if sampled_time.dim() == 1:
        sampled_time = sampled_time.unsqueeze(1)

    fit_device = subject_condition_fit["t_min"].device
    if subject_indices.device != fit_device:
        subject_indices = subject_indices.to(fit_device)
    if sampled_time.device != fit_device:
        sampled_time = sampled_time.to(fit_device)

    t_min = subject_condition_fit["t_min"].index_select(0, subject_indices).to(
        device=device, dtype=dtype
    )
    t_max = subject_condition_fit["t_max"].index_select(0, subject_indices).to(
        device=device, dtype=dtype
    )
    slopes = subject_condition_fit["slopes"].index_select(0, subject_indices).to(
        device=device, dtype=dtype
    )
    intercepts = subject_condition_fit["intercepts"].index_select(0, subject_indices).to(
        device=device, dtype=dtype
    )

    tt = torch.clamp(sampled_time, min=t_min.unsqueeze(1), max=t_max.unsqueeze(1))
    return intercepts + slopes * tt


def _resolve_existing_file(path, experiment_directory=None):
    candidates = []
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if path is not None:
        candidates.append(path)
    if experiment_directory is not None and path is not None and not os.path.isabs(path):
        candidates.append(os.path.join(experiment_directory, path))
    if path is not None and not os.path.isabs(path):
        candidates.append(os.path.join(repo_root, path))
    if path is not None and not os.path.isabs(path):
        candidates.append(os.path.join(os.getcwd(), path))

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"Could not resolve file path '{path}'. Candidates checked: {candidates}"
    )


def _extract_latent_weight(latent_codes):
    if isinstance(latent_codes, dict):
        if "weight" not in latent_codes:
            raise RuntimeError(
                "Latent checkpoint uses dict format but has no 'weight' key."
            )
        latent_weight = latent_codes["weight"]
    elif torch.is_tensor(latent_codes):
        latent_weight = latent_codes
    else:
        raise RuntimeError(
            f"Unsupported latent checkpoint format type: {type(latent_codes)}"
        )

    if latent_weight.dim() == 3 and latent_weight.shape[1] == 1:
        latent_weight = latent_weight.squeeze(1)

    if latent_weight.dim() != 2:
        raise RuntimeError(
            f"Expected latent weight tensor with rank 2, got shape {tuple(latent_weight.shape)}"
        )
    return latent_weight.detach().cpu().float()


def load_pretrained_scan_latent_map(
    pretrained_experiment_dir, checkpoint, expected_latent_dim
):
    lat_ckpt_path = os.path.join(
        pretrained_experiment_dir, ws.latent_codes_subdir, checkpoint + ".pth"
    )
    if not os.path.isfile(lat_ckpt_path):
        raise FileNotFoundError(
            f'Pretrained latent checkpoint does not exist: "{lat_ckpt_path}"'
        )

    lat_data = torch.load(lat_ckpt_path, map_location="cpu")
    if "latent_codes" not in lat_data:
        raise RuntimeError(
            f'Invalid latent checkpoint "{lat_ckpt_path}": missing "latent_codes" key'
        )
    latent_weight = _extract_latent_weight(lat_data["latent_codes"])
    if latent_weight.shape[1] != int(expected_latent_dim):
        raise RuntimeError(
            "Pretrained latent dim mismatch: "
            f"{latent_weight.shape[1]} vs expected {expected_latent_dim}"
        )

    pretrained_specs = ws.load_experiment_specifications(pretrained_experiment_dir)
    pretrained_train_split_file = _resolve_existing_file(
        pretrained_specs["TrainSplit"], pretrained_experiment_dir
    )
    with open(pretrained_train_split_file, "r") as f:
        pretrained_train_split = json.load(f)

    if len(pretrained_train_split) != latent_weight.shape[0]:
        raise RuntimeError(
            "Pretrained split/latent count mismatch: "
            f"split has {len(pretrained_train_split)} scans, "
            f"latent table has {latent_weight.shape[0]} rows"
        )

    scan_latent_map = {}
    for idx, instance_name in enumerate(pretrained_train_split):
        scan_key = os.path.splitext(os.path.basename(instance_name))[0]
        if scan_key in scan_latent_map:
            raise RuntimeError(
                f"Duplicate scan key '{scan_key}' in pretrained train split."
            )
        scan_latent_map[scan_key] = latent_weight[idx]

    return scan_latent_map, lat_data.get("epoch", None), pretrained_train_split_file


def initialize_subject_anchors_from_pretrained_scan_latents(
    lat_vecs,
    npyfiles,
    scan_to_subject_idx_cpu,
    scan_latent_map,
    expected_scans_per_subject=10,
):
    num_subjects = lat_vecs.num_embeddings
    latent_dim = lat_vecs.embedding_dim
    sums = torch.zeros(num_subjects, latent_dim, dtype=torch.float32)
    counts = torch.zeros(num_subjects, dtype=torch.long)

    missing_scan_keys = []
    for scan_idx, fpath in enumerate(npyfiles):
        scan_key = os.path.splitext(os.path.basename(fpath))[0]
        latent = scan_latent_map.get(scan_key, None)
        if latent is None:
            missing_scan_keys.append(scan_key)
            continue
        subj_idx = int(scan_to_subject_idx_cpu[scan_idx].item())
        sums[subj_idx] += latent
        counts[subj_idx] += 1

    if missing_scan_keys:
        preview = ", ".join(missing_scan_keys[:5])
        raise RuntimeError(
            "Missing pretrained scan latents for current training scans. "
            f"First missing keys: {preview}"
        )

    zero_subjects = torch.nonzero(counts == 0, as_tuple=False).view(-1)
    if zero_subjects.numel() > 0:
        raise RuntimeError(
            f"Found {zero_subjects.numel()} subjects with zero scans in pretrained map."
        )

    if expected_scans_per_subject is not None:
        exp = int(expected_scans_per_subject)
        mismatched = torch.nonzero(counts != exp, as_tuple=False).view(-1)
        if mismatched.numel() > 0:
            bad_idx = int(mismatched[0].item())
            raise RuntimeError(
                "Unexpected scans-per-subject during pretrained init: "
                f"subject idx {bad_idx} has {int(counts[bad_idx].item())}, expected {exp}"
            )

    subject_means = sums / counts.unsqueeze(1).float()
    with torch.no_grad():
        lat_vecs.weight.data.copy_(
            subject_means.to(device=lat_vecs.weight.device, dtype=lat_vecs.weight.dtype)
        )
        if lat_vecs.max_norm is not None:
            bound = float(lat_vecs.max_norm)
            norms = lat_vecs.weight.data.norm(dim=1, keepdim=True)
            scale = torch.clamp(bound / (norms + 1e-12), max=1.0)
            lat_vecs.weight.data.mul_(scale)

    return counts, subject_means


def optimize_subject_anchor_from_observations(
    decoder,
    temporal_flow,
    latent_size,
    observations,
    clamp_dist,
    num_iterations,
    num_samples,
    lr,
    init_std,
    code_reg_lambda=0.0,
    code_bound=None,
    use_pair_forward_consistency=False,
    pair_forward_lambda=0.0,
    pair_forward_pairs_per_iter=1,
    use_pair_backward_consistency=False,
    pair_backward_lambda=0.0,
    pair_backward_pairs_per_iter=1,
    use_general_cocycle_consistency=False,
    general_cocycle_lambda=0.0,
    general_cocycle_triplets_per_iter=1,
    use_age_conditioning=False,
):
    if len(observations) == 0:
        raise ValueError("No observations provided for subject-anchor optimization.")

    device = next(decoder.parameters()).device
    decoder_was_training = decoder.training
    flow_was_training = temporal_flow.training
    decoder.eval()
    temporal_flow.eval()

    decoder_params = list(decoder.parameters())
    flow_params = list(temporal_flow.parameters())
    decoder_reqgrad = [p.requires_grad for p in decoder_params]
    flow_reqgrad = [p.requires_grad for p in flow_params]
    for p in decoder_params:
        p.requires_grad_(False)
    for p in flow_params:
        p.requires_grad_(False)

    try:
        anchor = torch.empty(1, latent_size, device=device).normal_(
            mean=0.0, std=float(init_std)
        )
        anchor.requires_grad_(True)
        optimizer = torch.optim.Adam([anchor], lr=float(lr))
        loss_l1 = torch.nn.L1Loss(reduction="mean")
        loss_hist = []
        observed_times = sorted({float(obs["time"]) for obs in observations})
        observed_times_tensor = torch.tensor(
            observed_times, device=device, dtype=anchor.dtype
        )
        observed_baseline_time = float(observed_times[0]) if len(observed_times) > 0 else 0.0

        def _condition_to_1d_tensor(cond_val):
            if cond_val is None:
                return None
            if torch.is_tensor(cond_val):
                return cond_val.detach().cpu().float().view(-1)
            if isinstance(cond_val, (list, tuple, np.ndarray)):
                return torch.tensor(cond_val, dtype=torch.float32).view(-1)
            return torch.tensor([float(cond_val)], dtype=torch.float32)

        condition_fit = None
        condition_fallback = None
        if use_age_conditioning:
            obs_pairs = []
            for obs in observations:
                cond_vec = _condition_to_1d_tensor(obs.get("age_cond", None))
                if cond_vec is not None:
                    obs_pairs.append((float(obs["time"]), cond_vec))
            if len(obs_pairs) > 0:
                obs_pairs = sorted(obs_pairs, key=lambda x: x[0])
                condition_fallback = obs_pairs[0][1]
            if len(obs_pairs) >= 2:
                cond_dim = int(obs_pairs[0][1].numel())
                for _, cond_vec in obs_pairs:
                    if int(cond_vec.numel()) != cond_dim:
                        raise RuntimeError(
                            "Inconsistent condition dimensions across observations in test anchor fitting."
                        )
                t_vals = torch.tensor([p[0] for p in obs_pairs], dtype=torch.float32)
                c_start = obs_pairs[0][1]
                c_end = obs_pairs[-1][1]
                dt = float(t_vals[-1].item() - t_vals[0].item())
                if dt <= 1e-8:
                    slope = torch.zeros_like(c_start)
                else:
                    slope = (c_end - c_start) / dt
                intercept = c_start - slope * float(t_vals[0].item())
                condition_fit = {
                    "t_min": float(t_vals.min().item()),
                    "t_max": float(t_vals.max().item()),
                    "slope": slope,
                    "intercept": intercept,
                }

        def _condition_from_sampled_time(tt):
            if not use_age_conditioning:
                return None
            if condition_fit is not None:
                tmin = float(condition_fit["t_min"])
                tmax = float(condition_fit["t_max"])
                slope = condition_fit["slope"].to(device=device, dtype=anchor.dtype)
                intercept = condition_fit["intercept"].to(device=device, dtype=anchor.dtype)
                tt_clamped = torch.clamp(tt, min=tmin, max=tmax)
                return intercept.unsqueeze(0) + tt_clamped * slope.unsqueeze(0)
            if condition_fallback is not None:
                fallback = condition_fallback.to(device=device, dtype=anchor.dtype).view(1, -1)
                return fallback.repeat(tt.shape[0], 1)
            return None

        for _ in range(int(num_iterations)):
            optimizer.zero_grad()
            loss = 0.0
            for obs in observations:
                sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
                    obs["samples"], int(num_samples)
                ).to(device)
                xyz = sdf_data[:, 0:3]
                sdf_gt = torch.clamp(
                    sdf_data[:, 3].unsqueeze(1), -float(clamp_dist), float(clamp_dist)
                )
                t_obs = float(obs["time"])
                t = torch.full(
                    (xyz.shape[0], 1), t_obs, device=device, dtype=anchor.dtype
                )
                s = torch.full_like(t, observed_baseline_time)
                anchor_expanded = anchor.expand(xyz.shape[0], -1)
                age_cond = None
                if use_age_conditioning:
                    obs_cond = _condition_to_1d_tensor(obs.get("age_cond", None))
                    if obs_cond is not None:
                        age_cond = obs_cond.to(device=device, dtype=anchor.dtype).view(1, -1)
                        age_cond = age_cond.repeat(xyz.shape[0], 1)
                z_t = apply_temporal_flow(
                    temporal_flow,
                    anchor_expanded,
                    s,
                    t,
                    age_cond=age_cond,
                )
                pred_sdf = decoder(torch.cat([z_t, xyz], dim=1))
                pred_sdf = torch.clamp(
                    pred_sdf, -float(clamp_dist), float(clamp_dist)
                )
                loss = loss + loss_l1(pred_sdf, sdf_gt)

            loss = loss / len(observations)
            reg_lambda = float(code_reg_lambda) if code_reg_lambda is not None else 0.0
            if reg_lambda > 0.0:
                loss = loss + reg_lambda * torch.mean(anchor.pow(2))

            if (
                observed_times_tensor.numel() >= 2
                and (
                    use_pair_forward_consistency
                    or use_pair_backward_consistency
                    or use_general_cocycle_consistency
                )
            ):
                if use_pair_forward_consistency and float(pair_forward_lambda) > 0.0:
                    s, t = _sample_sorted_time_pairs(
                        int(pair_forward_pairs_per_iter),
                        device=device,
                        mode="from_values",
                        time_values=observed_times_tensor,
                        dtype=anchor.dtype,
                    )
                    if s is not None:
                        anchor_rep = anchor.expand(s.shape[0], -1)
                        baseline = torch.full_like(s, observed_baseline_time)
                        age_s = _condition_from_sampled_time(s)
                        age_t = _condition_from_sampled_time(t)
                        z_s = apply_temporal_flow(temporal_flow, anchor_rep, baseline, s, age_cond=age_s)
                        z_t = apply_temporal_flow(temporal_flow, anchor_rep, baseline, t, age_cond=age_t)
                        z_st = apply_temporal_flow(temporal_flow, z_s, s, t, age_cond=age_t)
                        pair_fwd_raw = torch.mean((z_st - z_t) ** 2)
                        loss = loss + float(pair_forward_lambda) * pair_fwd_raw

                if use_pair_backward_consistency and float(pair_backward_lambda) > 0.0:
                    s, t = _sample_sorted_time_pairs(
                        int(pair_backward_pairs_per_iter),
                        device=device,
                        mode="from_values",
                        time_values=observed_times_tensor,
                        dtype=anchor.dtype,
                    )
                    if s is not None:
                        anchor_rep = anchor.expand(s.shape[0], -1)
                        baseline = torch.full_like(s, observed_baseline_time)
                        age_s = _condition_from_sampled_time(s)
                        age_t = _condition_from_sampled_time(t)
                        z_s = apply_temporal_flow(temporal_flow, anchor_rep, baseline, s, age_cond=age_s)
                        z_t = apply_temporal_flow(temporal_flow, anchor_rep, baseline, t, age_cond=age_t)
                        z_ts = apply_temporal_flow(temporal_flow, z_t, t, s, age_cond=age_s)
                        pair_bwd_raw = torch.mean((z_ts - z_s) ** 2)
                        loss = loss + float(pair_backward_lambda) * pair_bwd_raw

                if (
                    use_general_cocycle_consistency
                    and float(general_cocycle_lambda) > 0.0
                    and observed_times_tensor.numel() >= 3
                ):
                    s, r, t = _sample_sorted_time_triplets(
                        int(general_cocycle_triplets_per_iter),
                        device=device,
                        mode="from_values",
                        time_values=observed_times_tensor,
                        dtype=anchor.dtype,
                    )
                    if s is not None:
                        anchor_rep = anchor.expand(s.shape[0], -1)
                        baseline = torch.full_like(s, observed_baseline_time)
                        age_s = _condition_from_sampled_time(s)
                        age_r = _condition_from_sampled_time(r)
                        age_t = _condition_from_sampled_time(t)
                        z_s = apply_temporal_flow(
                            temporal_flow, anchor_rep, baseline, s, age_cond=age_s
                        )
                        z_sr = apply_temporal_flow(temporal_flow, z_s, s, r, age_cond=age_r)
                        z_srt = apply_temporal_flow(temporal_flow, z_sr, r, t, age_cond=age_t)
                        z_st = apply_temporal_flow(temporal_flow, z_s, s, t, age_cond=age_t)
                        cyc_gen_raw = torch.mean((z_srt - z_st) ** 2)
                        loss = loss + float(general_cocycle_lambda) * cyc_gen_raw

            loss.backward()
            optimizer.step()

            if code_bound is not None:
                with torch.no_grad():
                    bound = float(code_bound)
                    if bound > 0.0:
                        n = anchor.norm(dim=1, keepdim=True)
                        scale = torch.clamp(bound / (n + 1e-12), max=1.0)
                        anchor.mul_(scale)

            loss_hist.append(float(loss.detach().cpu().item()))

        return anchor.detach(), loss_hist
    finally:
        for p, req in zip(decoder_params, decoder_reqgrad):
            p.requires_grad_(req)
        for p, req in zip(flow_params, flow_reqgrad):
            p.requires_grad_(req)
        if decoder_was_training:
            decoder.train()
        if flow_was_training:
            temporal_flow.train()


def main_function(experiment_directory: str, continue_from, batch_split: int, gpu: int = None):

   
    
    logging.debug("running experiment " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + str(specs["Description"]))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This training script requires a CUDA-enabled PyTorch setup."
        )

    requested_gpu = gpu
    if requested_gpu is None:
        requested_gpu = get_spec_with_default(specs, "GpuId", None)
    if requested_gpu is not None:
        requested_gpu = int(requested_gpu)
        if requested_gpu < 0:
            raise RuntimeError(f"GpuId/--gpu must be >= 0, got {requested_gpu}")
        try:
            torch.cuda.set_device(requested_gpu)
        except Exception as e:
            raise RuntimeError(
                f"Unable to set CUDA device to index {requested_gpu}. "
                f"Visible CUDA devices: {torch.cuda.device_count()}. Error: {e}"
            ) from e
        logging.info("Using requested CUDA device index: %d", requested_gpu)
    else:
        logging.info("Using default CUDA device index: %d", torch.cuda.current_device())

    data_source = specs["DataSource"]
    train_split_file = _resolve_existing_file(
        specs["TrainSplit"], experiment_directory
    )
    test_split_file = _resolve_existing_file(
        specs["TestSplit"], experiment_directory
    )

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = lr_scheduling.get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", decoder, temporal_flow, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):
        save_model(experiment_directory, str(epoch) + ".pth", decoder, temporal_flow, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    # def signal_handler(sig, frame):
    #     logging.info("Stopping early...")
    #     sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch, loss_log):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch, loss_log)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    # signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)
    use_zero_displacement_loss = bool(
        get_spec_with_default(specs, "UseZeroDisplacementLoss", False)
    )
    zero_displacement_lambda = float(
        get_spec_with_default(specs, "ZeroDisplacementLambda", 0.0)
    )
    use_eikonal = get_spec_with_default(specs, "UseEikonal", False)
    use_covariance = get_spec_with_default(specs, "UseCovarianceLoss", False)
    lambda_cov = get_spec_with_default(specs, "CovarianceLossLambda", 1e-3)
    covariance_loss_fn = CovarianceLoss().cuda()

    use_gmm_prior = get_spec_with_default(specs, "UseGMMPriorLoss", False)
    gmm_lambda = get_spec_with_default(specs, "GMMLambda", 1e-4)
    gmm_k = get_spec_with_default(specs, "GMMK", 2)
    gmm_init_sigma = get_spec_with_default(specs, "GMMInitSigma", 0.5)
    gmm_min_sigma = get_spec_with_default(specs, "GMMMinSigma", 0.05)
    gmm_learn_pi = get_spec_with_default(specs, "GMMLearnPi", False)
    gmm_prior_loss_fn = None
    if use_gmm_prior:
        gmm_prior_loss_fn = GMMPriorLoss(
            K=gmm_k,
            latent_dim=latent_size,
            learn_pi=gmm_learn_pi,
            init_sigma=gmm_init_sigma,
            min_sigma=gmm_min_sigma,
        ).cuda()

    # Isometry loss configuration
    use_isometry = get_spec_with_default(specs, "UseIsometryLoss", False)
    lambda_iso = get_spec_with_default(specs, "IsometryLossLambda", 1e-3)
    iso_num_points = get_spec_with_default(specs, "IsometryNumPoints", 256)
    iso_num_probes = get_spec_with_default(specs, "IsometryNumProbes", 1)
    iso_compute_frequency = get_spec_with_default(specs, "IsometryComputeFrequency", 1)
    iso_scenes_per_batch = get_spec_with_default(specs, "IsometryScenesPerBatch", None)
    use_isometry_mixup = get_spec_with_default(specs, "UseIsometryMixup", False)
    iso_mixup_alpha = get_spec_with_default(specs, "IsometryMixupAlpha", 0.2)
    iso_mixup_prob = get_spec_with_default(specs, "IsometryMixupProb", 0.0)
    isometry_loss_fn = IsometryLoss(num_hutchinson_probes=iso_num_probes).cuda()

    use_grad_metric_iso = get_spec_with_default(specs, "UseGradMetricIsotropyLoss", False)
    grad_metric_iso_lambda = get_spec_with_default(specs, "GradMetricIsoLossLambda", 1.0)
    grad_metric_iso_alpha = get_spec_with_default(specs, "GradMetricIsoAlpha", 1.0)
    grad_metric_iso_normalize = get_spec_with_default(specs, "GradMetricIsoNormalize", True)
    grad_metric_iso_fn = None
    if use_grad_metric_iso:
        grad_metric_iso_fn = GradientMetricIsotropyLoss(
            alpha=grad_metric_iso_alpha, normalize=grad_metric_iso_normalize
        ).cuda()
    
    if use_isometry:
        logging.info(f"Isometry loss enabled: lambda={lambda_iso}, num_points={iso_num_points}, "
                     f"num_probes={iso_num_probes}, compute_freq={iso_compute_frequency}")
    if use_grad_metric_iso:
        logging.info(
            "Gradient metric isotropy enabled: "
            f"lambda={grad_metric_iso_lambda}, alpha={grad_metric_iso_alpha}, "
            f"normalize={grad_metric_iso_normalize}, num_points={iso_num_points}, "
            f"compute_freq={iso_compute_frequency}"
        )
    if use_gmm_prior:
        logging.info(
            "GMM prior enabled: "
            f"lambda={gmm_lambda}, K={gmm_k}, learn_pi={gmm_learn_pi}, "
            f"init_sigma={gmm_init_sigma}, min_sigma={gmm_min_sigma}"
        )

    flow_hidden_dims = get_spec_with_default(specs, "FlowHiddenDims", [256, 256])
    temporal_consistency_phase = int(
        get_spec_with_default(specs, "TemporalConsistencyPhase", 0)
    )
    if temporal_consistency_phase < 0 or temporal_consistency_phase > 4:
        logging.warning(
            "TemporalConsistencyPhase=%s is outside [0,4]. Clamping into range.",
            temporal_consistency_phase,
        )
        temporal_consistency_phase = max(0, min(4, temporal_consistency_phase))

    use_cocycle_loss = get_spec_with_default(specs, "UseCocycleLoss", True)
    cocycle_lambda = get_spec_with_default(specs, "CocycleLossLambda", 1e-2)
    cocycle_pairs_per_subject = max(
        1, int(get_spec_with_default(specs, "CocyclePairsPerSubject", 1))
    )
    use_cocycle_shape_loss = bool(
        get_spec_with_default(specs, "UseCocycleShapeLoss", False)
    )
    cocycle_shape_lambda = float(
        get_spec_with_default(specs, "CocycleShapeLossLambda", 0.0)
    )
    cocycle_shape_num_points = int(
        get_spec_with_default(specs, "CocycleShapeNumPoints", 512)
    )
    cocycle_shape_use_clamp = bool(
        get_spec_with_default(specs, "CocycleShapeUseClamp", True)
    )
    if cocycle_shape_num_points <= 0:
        logging.warning(
            "CocycleShapeNumPoints=%s is invalid; disabling cocycle shape loss.",
            cocycle_shape_num_points,
        )
        use_cocycle_shape_loss = False
    if use_cocycle_shape_loss and cocycle_shape_lambda <= 0.0:
        logging.info(
            "UseCocycleShapeLoss=true but CocycleShapeLossLambda<=0; shape cocycle term is inactive."
        )
    if use_cocycle_shape_loss and not use_cocycle_loss:
        logging.warning(
            "UseCocycleShapeLoss=true requires UseCocycleLoss=true. Disabling cocycle shape loss."
        )
        use_cocycle_shape_loss = False
    use_cocycle_backward_loss = get_spec_with_default(
        specs, "UseCocycleBackwardLoss", temporal_consistency_phase >= 4
    )
    cocycle_backward_lambda = float(
        get_spec_with_default(specs, "CocycleBackwardLossLambda", 1e-2)
    )
    cocycle_backward_pairs_per_subject = max(
        1, int(get_spec_with_default(specs, "CocycleBackwardPairsPerSubject", 1))
    )
    use_pair_forward_loss = get_spec_with_default(
        specs, "UsePairForwardLoss", temporal_consistency_phase >= 1
    )
    pair_forward_lambda = float(
        get_spec_with_default(specs, "PairForwardLossLambda", 1e-2)
    )
    pair_forward_pairs_per_subject = max(
        1, int(get_spec_with_default(specs, "PairForwardPairsPerSubject", 1))
    )
    use_pair_backward_loss = get_spec_with_default(
        specs, "UsePairBackwardLoss", temporal_consistency_phase >= 1
    )
    pair_backward_lambda = float(
        get_spec_with_default(specs, "PairBackwardLossLambda", 1e-2)
    )
    pair_backward_pairs_per_subject = max(
        1, int(get_spec_with_default(specs, "PairBackwardPairsPerSubject", 1))
    )
    consistency_pair_sampling_mode = str(
        get_spec_with_default(specs, "ConsistencyPairSamplingMode", "uniform_01")
    ).strip().lower()
    if consistency_pair_sampling_mode not in ("uniform_01", "mixed_adjacent_far"):
        logging.warning(
            "Unknown ConsistencyPairSamplingMode='%s'; using uniform_01.",
            consistency_pair_sampling_mode,
        )
        consistency_pair_sampling_mode = "uniform_01"
    consistency_pair_adjacent_ratio = float(
        get_spec_with_default(specs, "ConsistencyPairAdjacentRatio", 0.5)
    )
    consistency_pair_adjacent_max_gap_ratio = float(
        get_spec_with_default(specs, "ConsistencyPairAdjacentMaxGapRatio", 0.25)
    )
    consistency_pair_far_min_gap_ratio = float(
        get_spec_with_default(specs, "ConsistencyPairFarMinGapRatio", 0.5)
    )
    use_general_cocycle_loss = get_spec_with_default(
        specs, "UseGeneralCocycleLoss", temporal_consistency_phase >= 3
    )
    general_cocycle_lambda = float(
        get_spec_with_default(specs, "GeneralCocycleLossLambda", 1e-2)
    )
    general_cocycle_triplets_per_subject = max(
        1, int(get_spec_with_default(specs, "GeneralCocycleTripletsPerSubject", 1))
    )
    use_general_cocycle_backward_loss = get_spec_with_default(
        specs, "UseGeneralCocycleBackwardLoss", temporal_consistency_phase >= 4
    )
    general_cocycle_backward_lambda = float(
        get_spec_with_default(specs, "GeneralCocycleBackwardLossLambda", 1e-2)
    )
    general_cocycle_backward_triplets_per_subject = max(
        1,
        int(get_spec_with_default(specs, "GeneralCocycleBackwardTripletsPerSubject", 1)),
    )
    use_real_scan_pair_loss = bool(
        get_spec_with_default(specs, "UseRealScanPairLoss", False)
    )
    real_scan_pairs_per_subject = max(
        1, int(get_spec_with_default(specs, "RealScanPairsPerSubject", 1))
    )
    use_real_scan_pair_forward = bool(
        get_spec_with_default(specs, "UseRealScanPairForward", True)
    )
    use_real_scan_pair_backward = bool(
        get_spec_with_default(specs, "UseRealScanPairBackward", True)
    )
    real_scan_pair_num_samples = max(
        2, int(get_spec_with_default(specs, "RealScanPairNumSamples", 4096))
    )
    if real_scan_pair_num_samples % 2 != 0:
        real_scan_pair_num_samples += 1
    real_scan_pair_reconstruction_lambda = float(
        get_spec_with_default(specs, "RealScanPairReconstructionLambda", 1.0)
    )
    real_scan_pair_latent_lambda = float(
        get_spec_with_default(specs, "RealScanPairLatentLambda", 1e-2)
    )
    if use_real_scan_pair_loss and not (
        use_real_scan_pair_forward or use_real_scan_pair_backward
    ):
        logging.warning(
            "UseRealScanPairLoss=true but both real-pair directions are disabled. "
            "Disabling real scan pair loss."
        )
        use_real_scan_pair_loss = False
    if (
        use_real_scan_pair_loss
        and real_scan_pair_reconstruction_lambda <= 0.0
        and real_scan_pair_latent_lambda <= 0.0
    ):
        logging.warning(
            "UseRealScanPairLoss=true but both real-pair lambdas are non-positive. "
            "Disabling real scan pair loss."
        )
        use_real_scan_pair_loss = False
    if temporal_consistency_phase == 4:
        use_cocycle_loss = False
        use_cocycle_backward_loss = False
        use_pair_forward_loss = False
        use_pair_backward_loss = False
        use_general_cocycle_loss = True
        use_general_cocycle_backward_loss = True
        if use_cocycle_shape_loss:
            logging.info(
                "Temporal consistency phase 4 disables pair-based cocycle losses; "
                "cocycle shape loss is also disabled."
            )
            use_cocycle_shape_loss = False
        logging.info(
            "Temporal consistency phase 4 active: using only triplet forward/backward "
            "general cocycle losses; pairwise and pair-based cocycle losses are disabled."
        )
    time_normalization_mode = get_spec_with_default(
        specs, "LongitudinalTimeNormalization", "subject_minmax"
    )
    use_age_conditioning = bool(get_spec_with_default(specs, "UseAgeConditioning", False))
    age_condition_key = str(get_spec_with_default(specs, "AgeConditionKey", "age_norm"))
    age_condition_keys_raw = get_spec_with_default(specs, "AgeConditionKeys", None)
    if age_condition_keys_raw is None:
        age_condition_keys = [age_condition_key]
    elif isinstance(age_condition_keys_raw, str):
        age_condition_keys = [age_condition_keys_raw]
    elif isinstance(age_condition_keys_raw, (list, tuple)):
        age_condition_keys = [str(x) for x in age_condition_keys_raw]
    else:
        raise RuntimeError(
            "AgeConditionKeys must be a string or a list of strings when provided."
        )
    age_condition_keys = [k for k in age_condition_keys if k]
    if age_condition_key and age_condition_key not in age_condition_keys:
        age_condition_keys = [age_condition_key] + age_condition_keys
    age_metadata_file = get_spec_with_default(specs, "AgeMetadataFile", None)
    time_metadata_file = get_spec_with_default(
        specs, "LongitudinalTimeMetadataFile", age_metadata_file
    )
    time_metadata_key = str(get_spec_with_default(specs, "LongitudinalTimeKey", "time_index"))
    age_condition_dim = int(
        get_spec_with_default(
            specs,
            "AgeConditionDim",
            len(age_condition_keys) if use_age_conditioning else 0,
        )
    )
    if not use_age_conditioning:
        age_condition_dim = 0
    if use_age_conditioning and age_condition_dim <= 0:
        raise RuntimeError(
            "UseAgeConditioning=true requires AgeConditionDim >= 1"
        )
    if use_age_conditioning and len(age_condition_keys) == 0:
        raise RuntimeError(
            "UseAgeConditioning=true requires at least one condition key "
            "(AgeConditionKey or AgeConditionKeys)."
        )
    if use_age_conditioning and len(age_condition_keys) > 1 and age_condition_dim != len(age_condition_keys):
        raise RuntimeError(
            "For multi-key conditioning, AgeConditionDim must match len(AgeConditionKeys). "
            f"Got dim={age_condition_dim}, keys={age_condition_keys}"
        )
    logging.info("Temporal consistency phase: %d", temporal_consistency_phase)
    if use_cocycle_loss:
        logging.info(
            "Cocycle loss enabled: "
            f"lambda={cocycle_lambda}, pairs_per_subject={cocycle_pairs_per_subject}"
        )
    if use_cocycle_shape_loss:
        logging.info(
            "Cocycle shape loss enabled: lambda=%s, points=%s, clamp=%s",
            cocycle_shape_lambda,
            cocycle_shape_num_points,
            cocycle_shape_use_clamp,
        )
    if use_cocycle_backward_loss:
        logging.info(
            "Backward cocycle loss enabled: "
            f"lambda={cocycle_backward_lambda}, pairs_per_subject={cocycle_backward_pairs_per_subject}"
        )
    logging.info(
        "Consistency pair sampling mode=%s (adjacent_ratio=%.3f, adjacent_max_gap_ratio=%.3f, far_min_gap_ratio=%.3f)",
        consistency_pair_sampling_mode,
        consistency_pair_adjacent_ratio,
        consistency_pair_adjacent_max_gap_ratio,
        consistency_pair_far_min_gap_ratio,
    )
    if use_pair_forward_loss:
        logging.info(
            "Pair forward consistency enabled: lambda=%s, pairs_per_subject=%s",
            pair_forward_lambda,
            pair_forward_pairs_per_subject,
        )
    if use_pair_backward_loss:
        logging.info(
            "Pair backward consistency enabled: lambda=%s, pairs_per_subject=%s",
            pair_backward_lambda,
            pair_backward_pairs_per_subject,
        )
    if use_general_cocycle_loss:
        logging.info(
            "General cocycle consistency enabled: lambda=%s, triplets_per_subject=%s",
            general_cocycle_lambda,
            general_cocycle_triplets_per_subject,
        )
    if use_general_cocycle_backward_loss:
        logging.info(
            "General backward cocycle consistency enabled: lambda=%s, triplets_per_subject=%s",
            general_cocycle_backward_lambda,
            general_cocycle_backward_triplets_per_subject,
        )
    if use_real_scan_pair_loss:
        logging.info(
            "Real scan pair loss enabled: pairs_per_subject=%s, forward=%s, backward=%s, "
            "target_samples=%s, reconstruction_lambda=%s, latent_lambda=%s",
            real_scan_pairs_per_subject,
            use_real_scan_pair_forward,
            use_real_scan_pair_backward,
            real_scan_pair_num_samples,
            real_scan_pair_reconstruction_lambda,
            real_scan_pair_latent_lambda,
        )
    flow_model_type = str(
        get_spec_with_default(specs, "FlowModelType", "standard")
    ).strip().lower()
    use_disentangled_velocity_flow = flow_model_type in (
        "disentangled_velocity",
        "disentangled_flow",
    )
    use_velocity_disease_zero_loss = bool(
        get_spec_with_default(specs, "UseVelocityDiseaseZeroLoss", False)
    )
    velocity_disease_zero_lambda = float(
        get_spec_with_default(specs, "VelocityDiseaseZeroLambda", 0.0)
    )
    use_velocity_residual_loss = bool(
        get_spec_with_default(specs, "UseVelocityResidualLoss", False)
    )
    velocity_residual_lambda = float(
        get_spec_with_default(specs, "VelocityResidualLambda", 0.0)
    )
    use_velocity_residual_diagnosis_covariance_loss = bool(
        get_spec_with_default(
            specs, "UseVelocityResidualDiagnosisCovarianceLoss", False
        )
    )
    velocity_residual_diagnosis_covariance_lambda = float(
        get_spec_with_default(
            specs, "VelocityResidualDiagnosisCovarianceLambda", 0.0
        )
    )
    use_velocity_disease_margin_loss = bool(
        get_spec_with_default(specs, "UseVelocityDiseaseMarginLoss", False)
    )
    velocity_disease_margin_lambda = float(
        get_spec_with_default(specs, "VelocityDiseaseMarginLambda", 0.0)
    )
    velocity_disease_margin = float(
        get_spec_with_default(specs, "VelocityDiseaseMargin", 0.05)
    )
    use_velocity_adversarial_leakage_loss = bool(
        get_spec_with_default(specs, "UseVelocityAdversarialLeakageLoss", False)
    )
    velocity_adversarial_age_lambda = float(
        get_spec_with_default(specs, "VelocityAdversarialAgeLambda", 0.0)
    )
    velocity_adversarial_residual_lambda = float(
        get_spec_with_default(specs, "VelocityAdversarialResidualLambda", 0.0)
    )
    velocity_adversarial_grl_lambda = float(
        get_spec_with_default(specs, "VelocityAdversarialGRLLambda", 1.0)
    )
    use_velocity_disease_classification_loss = bool(
        get_spec_with_default(specs, "UseVelocityDiseaseClassificationLoss", False)
    )
    velocity_disease_classification_lambda = float(
        get_spec_with_default(specs, "VelocityDiseaseClassificationLambda", 0.0)
    )
    logging.info(
        "Longitudinal flow: type=%s, hidden_dims=%s, time_normalization=%s",
        flow_model_type,
        flow_hidden_dims,
        time_normalization_mode,
    )
    if use_disentangled_velocity_flow:
        logging.info(
            "Disentangled velocity flow enabled: age_dim=%s, disease_dim=%s, "
            "residual_dim=%s, age_scale=%s, residual_uses_diagnosis=%s",
            get_spec_with_default(specs, "VelocityAgeDim", 16),
            get_spec_with_default(specs, "VelocityDiseaseDim", 16),
            get_spec_with_default(
                specs,
                "VelocityResidualDim",
                int(latent_size)
                - int(get_spec_with_default(specs, "VelocityAgeDim", 16))
                - int(get_spec_with_default(specs, "VelocityDiseaseDim", 16)),
            ),
            get_spec_with_default(specs, "VelocityAgeEmbeddingScale", 0.02),
            get_spec_with_default(specs, "VelocityResidualUsesDiagnosis", False),
        )
        logging.info(
            "Disentangled velocity losses: disease_zero=%s(lambda=%s), "
            "residual=%s(lambda=%s), residual_diagnosis_covariance=%s(lambda=%s), "
            "disease_margin=%s(lambda=%s, margin=%s), "
            "adv_leakage=%s(age_lambda=%s, residual_lambda=%s, grl=%s), "
            "disease_cls=%s(lambda=%s)",
            use_velocity_disease_zero_loss,
            velocity_disease_zero_lambda,
            use_velocity_residual_loss,
            velocity_residual_lambda,
            use_velocity_residual_diagnosis_covariance_loss,
            velocity_residual_diagnosis_covariance_lambda,
            use_velocity_disease_margin_loss,
            velocity_disease_margin_lambda,
            velocity_disease_margin,
            use_velocity_adversarial_leakage_loss,
            velocity_adversarial_age_lambda,
            velocity_adversarial_residual_lambda,
            velocity_adversarial_grl_lambda,
            use_velocity_disease_classification_loss,
            velocity_disease_classification_lambda,
        )
    logging.info(
        "Age conditioning: enabled=%s, dim=%d, keys=%s, metadata=%s",
        use_age_conditioning,
        age_condition_dim,
        age_condition_keys,
        age_metadata_file,
    )
    logging.info(
        "Longitudinal time metadata: key=%s, file=%s",
        time_metadata_key,
        time_metadata_file,
    )
    logging.info(
        "Zero-displacement regularizer: enabled=%s, lambda=%s",
        use_zero_displacement_loss,
        zero_displacement_lambda,
    )
    code_bound = get_spec_with_default(specs, "CodeBound", None)

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    gpu_count = torch.cuda.device_count()
    logging.info("training with {} GPU(s)".format(gpu_count))

    use_data_parallel = bool(get_spec_with_default(specs, "UseDataParallel", False))
    if requested_gpu is not None and use_data_parallel:
        logging.info(
            "UseDataParallel=true ignored because explicit single-GPU selection is active (gpu=%d).",
            requested_gpu,
        )
        use_data_parallel = False
    if use_data_parallel and gpu_count > 1:
        decoder = torch.nn.DataParallel(decoder)
        logging.info("DataParallel enabled across %d GPUs.", gpu_count)
    elif use_data_parallel and gpu_count <= 1:
        logging.info("UseDataParallel=true but <=1 GPU visible; running without DataParallel.")
    else:
        logging.info("DataParallel disabled.")
    train_device = next(decoder.parameters()).device
    temporal_flow = build_temporal_flow(
        specs,
        latent_size,
        flow_hidden_dims,
        age_condition_dim=age_condition_dim,
    ).to(train_device)

    use_pretrained_sdf = get_spec_with_default(specs, "UsePretrainedSDFDecoder", False)
    pretrained_sdf_dir = get_spec_with_default(specs, "PretrainedSDFDecoderDir", None)
    pretrained_sdf_ckpt = get_spec_with_default(specs, "PretrainedSDFDecoderCheckpoint", "latest")
    use_pretrained_subject_anchor_init = get_spec_with_default(
        specs, "UsePretrainedSubjectAnchorInit", False
    )
    pretrained_subject_anchor_dir = get_spec_with_default(
        specs, "PretrainedSubjectAnchorInitDir", None
    )
    pretrained_subject_anchor_ckpt = get_spec_with_default(
        specs, "PretrainedSubjectAnchorInitCheckpoint", "latest"
    )
    pretrained_expected_scans_per_subject = get_spec_with_default(
        specs, "PretrainedSubjectAnchorExpectedScansPerSubject", 10
    )
    if use_pretrained_sdf:
        if continue_from is not None:
            logging.info(
                f"Skipping pretrained SDF load because continuing from checkpoint {continue_from}."
            )
        else:
            if pretrained_sdf_dir is None:
                raise RuntimeError(
                    "UsePretrainedSDFDecoder=true but PretrainedSDFDecoderDir is not set."
                )
            pretrained_epoch = load_pretrained_decoder(
                decoder, pretrained_sdf_dir, pretrained_sdf_ckpt
            )
            logging.info(
                f"Loaded pretrained SDF decoder from {pretrained_sdf_dir} "
                f"(checkpoint {pretrained_sdf_ckpt}, epoch {pretrained_epoch})."
            )

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 200)
    
    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    torus_path = get_spec_with_default(specs, "TorusPath", "/home/jakaria/torus_two_models_data/torus_two/obj_files")
    logging.info(f"Torus path: {torus_path}")
    if not os.path.exists(torus_path): 
        logging.error(f"Running w/o validation, since the specified Torus path does not exist: {torus_path}")
        torus_path = None
    load_ram = get_spec_with_default(specs, "LoadDatasetIntoRAM", False)
    if load_ram:
        logging.info(f"Loading SDF samples into memory because LoadDatasetIntoRAM=true")
    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=load_ram
    )
    scan_age_condition_cpu = None
    scan_age_condition = None
    subject_baseline_time = None
    subject_baseline_age = None
    subject_condition_fit = None
    scan_age_map = {}
    age_metadata_resolved = None
    time_metadata_resolved = None
    scan_time_map = None

    if time_metadata_file:
        time_metadata_resolved = _resolve_existing_file(
            time_metadata_file, experiment_directory
        )
        scan_time_map = _load_scan_age_map_from_labels(
            time_metadata_resolved, time_metadata_key
        )

    if use_age_conditioning:
        if age_metadata_file is None:
            raise RuntimeError(
                "UseAgeConditioning=true but AgeMetadataFile is not set in specs.json"
            )
        age_metadata_resolved = _resolve_existing_file(age_metadata_file, experiment_directory)
        for condition_key in age_condition_keys:
            scan_age_map[condition_key] = _load_scan_age_map_from_labels(
                age_metadata_resolved,
                condition_key,
            )

    longitudinal_meta = build_longitudinal_metadata(
        sdf_dataset.npyfiles,
        time_normalization_mode,
        scan_to_time_map=scan_time_map,
        time_key_name=time_metadata_key,
    )
    scan_to_subject_idx_cpu = longitudinal_meta["scan_to_subject_idx"]
    scan_to_subject_idx = scan_to_subject_idx_cpu.to(train_device)
    scan_to_time_cpu = longitudinal_meta["timepoints"]
    scan_to_time = scan_to_time_cpu.to(train_device)
    num_subjects = longitudinal_meta["num_subjects"]
    subject_to_scan_indices = _build_subject_to_scan_indices(
        scan_to_subject_idx_cpu, num_subjects
    )
    if use_real_scan_pair_loss:
        subjects_with_pairs = sum(
            1 for scan_indices in subject_to_scan_indices.values() if len(scan_indices) >= 2
        )
        if subjects_with_pairs == 0:
            raise RuntimeError(
                "UseRealScanPairLoss=true but no training subjects have at least two scans."
            )
        logging.info(
            "Real scan pair sampling can use %d/%d training subjects.",
            subjects_with_pairs,
            num_subjects,
        )
    subject_baseline_time_cpu = compute_subject_baseline_time(
        scan_to_subject_idx_cpu,
        scan_to_time_cpu,
        num_subjects,
    )
    subject_baseline_time = subject_baseline_time_cpu.to(train_device)
    consistency_time_sample_low = float(scan_to_time_cpu.min().item())
    consistency_time_sample_high = float(scan_to_time_cpu.max().item())

    if use_age_conditioning:
        scan_age_condition_cpu = build_scan_condition_tensor(
            sdf_dataset.npyfiles,
            scan_age_map,
            age_condition_keys,
        )
        scan_age_condition = scan_age_condition_cpu.to(train_device)
        (
            _subject_baseline_time_cpu_check,
            subject_baseline_age_cpu,
        ) = compute_subject_baseline_time_and_condition(
            scan_to_subject_idx_cpu,
            scan_to_time_cpu,
            scan_age_condition_cpu,
            num_subjects,
        )
        subject_condition_fit_cpu = build_subject_condition_linear_fit(
            scan_to_subject_idx_cpu,
            scan_to_time_cpu,
            scan_age_condition_cpu,
            num_subjects,
        )
        if not torch.allclose(_subject_baseline_time_cpu_check, subject_baseline_time_cpu):
            raise RuntimeError(
                "Inconsistent subject baseline time computed from time metadata."
            )
        subject_baseline_age = subject_baseline_age_cpu.to(train_device)
        subject_condition_fit = {
            key: val.to(train_device) for key, val in subject_condition_fit_cpu.items()
        }

    logging.info(
        "Longitudinal mapping built: %d scans -> %d subjects",
        len(sdf_dataset),
        num_subjects,
    )

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))
    
    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,         # to avoid unstable gradients in last batch
    )

    # Get train evaluation settings.
    eval_grid_res = get_spec_with_default(specs, "EvalGridResolution", 256)
    eval_train_scene_num = get_spec_with_default(specs, "EvalTrainSceneNumber", 10)
    eval_train_frequency = get_spec_with_default(specs, "EvalTrainFrequency", 300)
    eval_train_scene_idxs = random.sample(range(len(sdf_dataset)), min(eval_train_scene_num, len(sdf_dataset)))
    logging.debug(f"Plotting {eval_train_scene_num} shapes with indices {eval_train_scene_idxs}")

    # Get test evaluation settings.
    with open(test_split_file, "r") as f:
        test_split = json.load(f)
    eval_test_frequency = get_spec_with_default(specs, "EvalTestFrequency", 500)
    eval_test_scene_num = get_spec_with_default(specs, "EvalTestSceneNumber", 10)
    eval_test_subject_num = get_spec_with_default(
        specs, "EvalTestSubjectNumber", eval_test_scene_num
    )
    eval_test_optimization_steps = get_spec_with_default(specs, "EvalTestOptimizationSteps", 1000)
    eval_test_observed_timepoints = max(
        1, int(get_spec_with_default(specs, "EvalTestObservedTimepoints", 1))
    )
    eval_test_anchor_lr = get_spec_with_default(specs, "EvalTestAnchorLR", 5e-3)
    eval_test_anchor_num_samples = get_spec_with_default(
        specs, "EvalTestAnchorNumSamples", 16384
    )
    eval_test_anchor_init_std = get_spec_with_default(
        specs, "EvalTestAnchorInitStd", 0.01
    )
    eval_test_anchor_code_reg_lambda = get_spec_with_default(
        specs, "EvalTestAnchorCodeRegLambda", 1e-4
    )
    default_rollout_mode = "composed" if temporal_consistency_phase >= 2 else "direct"
    eval_test_rollout_mode = str(
        get_spec_with_default(specs, "EvalTestRolloutMode", default_rollout_mode)
    ).lower()
    if eval_test_rollout_mode not in ("direct", "composed"):
        logging.warning(
            "Unknown EvalTestRolloutMode=%s; falling back to 'direct'.",
            eval_test_rollout_mode,
        )
        eval_test_rollout_mode = "direct"
    eval_test_rollout_start = str(
        get_spec_with_default(specs, "EvalTestRolloutStart", "last_observed")
    ).lower()
    if eval_test_rollout_start not in ("zero", "last_observed"):
        logging.warning(
            "Unknown EvalTestRolloutStart=%s; falling back to 'last_observed'.",
            eval_test_rollout_start,
        )
        eval_test_rollout_start = "last_observed"
    eval_test_max_rollout_dt = float(
        get_spec_with_default(specs, "EvalTestMaxRolloutDt", 0.0)
    )
    use_test_pair_consistency = bool(
        get_spec_with_default(specs, "UseTestPairConsistency", False)
    )
    test_pair_forward_lambda = float(
        get_spec_with_default(specs, "TestPairForwardLambda", 0.0)
    )
    test_pair_forward_pairs_per_iter = max(
        1, int(get_spec_with_default(specs, "TestPairForwardPairsPerIter", 1))
    )
    test_pair_backward_lambda = float(
        get_spec_with_default(specs, "TestPairBackwardLambda", 0.0)
    )
    test_pair_backward_pairs_per_iter = max(
        1, int(get_spec_with_default(specs, "TestPairBackwardPairsPerIter", 1))
    )
    use_test_general_cocycle_consistency = bool(
        get_spec_with_default(specs, "UseTestGeneralCocycleConsistency", False)
    )
    test_general_cocycle_lambda = float(
        get_spec_with_default(specs, "TestGeneralCocycleLambda", 0.0)
    )
    test_general_cocycle_triplets_per_iter = max(
        1, int(get_spec_with_default(specs, "TestGeneralCocycleTripletsPerIter", 1))
    )
    logging.info(
        "Test rollout mode=%s, start=%s, max_dt=%s",
        eval_test_rollout_mode,
        eval_test_rollout_start,
        eval_test_max_rollout_dt,
    )
    if use_test_pair_consistency or use_test_general_cocycle_consistency:
        logging.info(
            "Test anchor consistency enabled: pair=%s (fwd_lambda=%s, bwd_lambda=%s), "
            "general_cocycle=%s (lambda=%s)",
            use_test_pair_consistency,
            test_pair_forward_lambda,
            test_pair_backward_lambda,
            use_test_general_cocycle_consistency,
            test_general_cocycle_lambda,
        )
    eval_chamfer_metric = get_spec_with_default(specs, "EvalChamferMetric", "chamfer")
    eval_chamfer_align_mode = get_spec_with_default(
        specs, "EvalChamferAlignMode", "centroid"
    )
    eval_chamfer_align_iters = int(
        get_spec_with_default(specs, "EvalChamferAlignIters", 20)
    )
    eval_chamfer_align_trim_quantile = float(
        get_spec_with_default(specs, "EvalChamferAlignTrimQuantile", 0.90)
    )
    eval_chamfer_metric_kwargs = {}
    if eval_chamfer_metric in (
        "chamfer_starmen_aligned",
        "chamfer_aligned_starmen",
        "chamfer_torus_aligned",
        "chamfer_aligned_torus",
        "chamfer_aligned",
    ):
        eval_chamfer_metric_kwargs = {
            "align_mode": eval_chamfer_align_mode,
            "align_max_iterations": eval_chamfer_align_iters,
            "align_trim_quantile": eval_chamfer_align_trim_quantile,
        }
    logging.info(
        "Eval Chamfer metric: %s%s",
        eval_chamfer_metric,
        (
            f" (align_mode={eval_chamfer_align_mode}, "
            f"iters={eval_chamfer_align_iters}, "
            f"trim_q={eval_chamfer_align_trim_quantile})"
            if eval_chamfer_metric_kwargs
            else ""
        ),
    )
    eval_test_filenames = deep_sdf.data.get_instance_filenames(data_source, test_split)
    test_longitudinal_meta = build_longitudinal_metadata(
        eval_test_filenames,
        time_normalization_mode,
        scan_to_time_map=scan_time_map,
        time_key_name=time_metadata_key,
    )
    test_scan_age_condition = None
    if use_age_conditioning:
        if len(scan_age_map) == 0:
            raise RuntimeError("Age conditioning enabled but scan_age_map is not initialized.")
        test_scan_age_condition = build_scan_condition_tensor(
            eval_test_filenames,
            scan_age_map,
            age_condition_keys,
        ).to(train_device)
    test_subject_to_scan_indices = {}
    for scan_idx, sid in enumerate(test_longitudinal_meta["subject_ids"]):
        test_subject_to_scan_indices.setdefault(sid, []).append(scan_idx)
    all_test_subject_ids = sorted(test_subject_to_scan_indices.keys())
    eval_test_subject_ids = random.sample(
        all_test_subject_ids, min(eval_test_subject_num, len(all_test_subject_ids))
    )
    logging.debug(
        "Longitudinal test eval uses %d subjects (observed timepoints=%d): %s",
        len(eval_test_subject_ids),
        eval_test_observed_timepoints,
        eval_test_subject_ids,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    # Avoid nn.Embedding(max_norm=...) because it renormalizes weights in-place
    # during forward passes, which can invalidate autograd when the embedding is
    # used multiple times in one optimization step.
    lat_vecs = torch.nn.Embedding(num_subjects, latent_size)
    lat_vecs = lat_vecs.cuda()
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    if use_pretrained_subject_anchor_init:
        if continue_from is not None:
            logging.info(
                "Skipping pretrained subject-anchor init because continuing from checkpoint %s.",
                continue_from,
            )
        else:
            if pretrained_subject_anchor_dir is None:
                raise RuntimeError(
                    "UsePretrainedSubjectAnchorInit=true but PretrainedSubjectAnchorInitDir is not set."
                )
            scan_latent_map, pretrained_lat_epoch, pretrained_split_file = (
                load_pretrained_scan_latent_map(
                    pretrained_subject_anchor_dir,
                    pretrained_subject_anchor_ckpt,
                    latent_size,
                )
            )
            counts, _ = initialize_subject_anchors_from_pretrained_scan_latents(
                lat_vecs,
                sdf_dataset.npyfiles,
                scan_to_subject_idx_cpu,
                scan_latent_map,
                expected_scans_per_subject=pretrained_expected_scans_per_subject,
            )
            logging.info(
                "Initialized subject anchors from pretrained scan latents at %s "
                "(checkpoint %s, epoch %s, split %s).",
                pretrained_subject_anchor_dir,
                pretrained_subject_anchor_ckpt,
                str(pretrained_lat_epoch),
                pretrained_split_file,
            )
            logging.info(
                "Pretrained init scan-count stats per subject: min=%d, max=%d, mean=%.2f",
                int(counts.min().item()),
                int(counts.max().item()),
                float(counts.float().mean().item()),
            )
            logging.info(
                "Post-init subject anchor mean magnitude: %.6f",
                float(get_mean_latent_vector_magnitude(lat_vecs).item()),
            )

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [{
            "params": list(decoder.parameters()) + list(temporal_flow.parameters()),
            "lr": lr_schedules[0].get_learning_rate(0),
        },
        {
            "params": list(lat_vecs.parameters()) + (list(gmm_prior_loss_fn.parameters()) if use_gmm_prior else []),
            "lr": lr_schedules[1].get_learning_rate(0),
        }]
    )

    summary_writer = SummaryWriter(log_dir=os.path.join(experiment_directory, ws.tb_logs_dir))

    loss_log = []               # per-batch
    loss_log_epoch = []         # per-epoch
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = load_model_and_flow(
            experiment_directory, continue_from, decoder, temporal_flow
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )
        # TODO test this
        for i, lrs in enumerate(lr_schedules):
            if isinstance(lrs, lr_scheduling.StepLearningRateOnPlateauSchedule):
                lrs.last_lr = optimizer_all.param_groups[i]["lr"]

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder+flow parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
            + sum(p.data.nelement() for p in temporal_flow.parameters())
        )
    )
    logging.info(
        "Number of subject anchor parameters: {} (# subjects {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )
    
    # Global batch counter for iso compute frequency
    global_batch_idx = 0
    
    try:
        train_chamfer_dists_log = []
        test_chamfer_dists_log = []
        for epoch in range(start_epoch, num_epochs + 1):
            

            epoch_time_start = time.time()
            epoch_losses = []
            epoch_sdf_losses = []
            epoch_reg_losses = []
            epoch_eikonal_losses = []
            epoch_cov_losses = []
            epoch_gmm_losses = []
            epoch_gmm_nlls = []
            epoch_gmm_entropies = []
            epoch_iso_losses = []
            epoch_grad_metric_iso_losses = []
            epoch_iso_g1_losses = []
            epoch_iso_g2_losses = []
            epoch_cocycle_losses = []
            epoch_cocycle_shape_losses = []
            epoch_cocycle_backward_losses = []
            epoch_pair_forward_losses = []
            epoch_pair_backward_losses = []
            epoch_general_cocycle_losses = []
            epoch_general_cocycle_backward_losses = []
            epoch_real_pair_forward_reconstruction_losses = []
            epoch_real_pair_forward_latent_losses = []
            epoch_real_pair_backward_reconstruction_losses = []
            epoch_real_pair_backward_latent_losses = []
            epoch_velocity_disease_zero_losses = []
            epoch_velocity_residual_losses = []
            epoch_velocity_residual_diagnosis_covariance_losses = []
            epoch_velocity_disease_margin_losses = []
            epoch_velocity_age_adversarial_losses = []
            epoch_velocity_residual_adversarial_losses = []
            epoch_velocity_disease_classification_losses = []
            epoch_velocity_age_adversarial_bces = []
            epoch_velocity_residual_adversarial_bces = []
            epoch_velocity_disease_classification_bces = []
            epoch_velocity_age_adversarial_accs = []
            epoch_velocity_residual_adversarial_accs = []
            epoch_velocity_disease_classification_accs = []
            epoch_velocity_age_norms = []
            epoch_velocity_disease_raw_norms = []
            epoch_velocity_disease_raw_healthy_norms = []
            epoch_velocity_disease_raw_diseased_norms = []
            epoch_velocity_disease_norms = []
            epoch_velocity_residual_norms = []

            logging.info("epoch {}...".format(epoch))

            # Required because evaluation puts the decoder into 'eval' mode.
            decoder.train()
            temporal_flow.train()

            adjust_learning_rate(lr_schedules, optimizer_all, epoch, loss_log_epoch)
            for sdf_data, indices in sdf_loader:
                global_batch_idx += 1
                # logging.debug(f"time for dataloading: {(time.time() - TIME)*1000:.3f} ms"); TIME = time.time()
                # Process the input data
                sdf_data = sdf_data.reshape(-1, 4).to(train_device, non_blocking=True)
                indices = indices.to(train_device, non_blocking=True)

                num_sdf_samples = sdf_data.shape[0]

                sdf_data.requires_grad = False

                xyz = sdf_data[:, 0:3]
                xyz.requires_grad = True
                sdf_gt = sdf_data[:, 3].unsqueeze(1)

                if enforce_minmax:
                    sdf_gt = torch.clamp(sdf_gt, minT, maxT)

                indices_batch = indices

                xyz = torch.chunk(xyz, batch_split)
                indices = torch.chunk(
                    indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                    batch_split,
                )

                sdf_gt = torch.chunk(sdf_gt, batch_split)

                batch_loss_tb = 0.0
                sdf_loss_tb = 0.0
                reg_loss_tb = 0.0
                eikonal_loss_tb = 0.0
                cov_loss_tb = 0.0
                gmm_loss_tb = 0.0
                gmm_nll_tb = 0.0
                gmm_entropy_tb = 0.0
                iso_loss_tb = 0.0
                grad_metric_iso_loss_tb = 0.0
                iso_g1_tb = 0.0
                iso_g2_tb = 0.0
                cocycle_loss_tb = 0.0
                cocycle_shape_loss_tb = 0.0
                cocycle_backward_loss_tb = 0.0
                pair_forward_loss_tb = 0.0
                pair_backward_loss_tb = 0.0
                general_cocycle_loss_tb = 0.0
                general_cocycle_backward_loss_tb = 0.0
                real_pair_forward_reconstruction_loss_tb = 0.0
                real_pair_forward_latent_loss_tb = 0.0
                real_pair_backward_reconstruction_loss_tb = 0.0
                real_pair_backward_latent_loss_tb = 0.0
                velocity_disease_zero_loss_tb = 0.0
                velocity_residual_loss_tb = 0.0
                velocity_residual_diagnosis_covariance_loss_tb = 0.0
                velocity_disease_margin_loss_tb = 0.0
                velocity_age_adversarial_loss_tb = 0.0
                velocity_residual_adversarial_loss_tb = 0.0
                velocity_disease_classification_loss_tb = 0.0
                velocity_age_adversarial_bce_tb = 0.0
                velocity_residual_adversarial_bce_tb = 0.0
                velocity_disease_classification_bce_tb = 0.0
                velocity_age_adversarial_acc_tb = 0.0
                velocity_residual_adversarial_acc_tb = 0.0
                velocity_disease_classification_acc_tb = 0.0
                velocity_age_norm_tb = 0.0
                velocity_disease_raw_norm_tb = 0.0
                velocity_disease_raw_healthy_norm_tb = 0.0
                velocity_disease_raw_diseased_norm_tb = 0.0
                velocity_disease_norm_tb = 0.0
                velocity_residual_norm_tb = 0.0

                optimizer_all.zero_grad()

                for i in range(batch_split):

                    unique_indices = torch.unique(indices[i])
                    scan_subject_indices = scan_to_subject_idx.index_select(0, indices[i])
                    scan_times = scan_to_time.index_select(0, indices[i]).unsqueeze(1)
                    start_times = subject_baseline_time.index_select(
                        0, scan_subject_indices
                    ).unsqueeze(1)
                    subject_anchor_batch = lat_vecs(scan_subject_indices)
                    scan_ages = None
                    if use_age_conditioning:
                        scan_ages = scan_age_condition.index_select(0, indices[i])
                    batch_vecs = apply_temporal_flow(
                        temporal_flow,
                        subject_anchor_batch,
                        start_times,
                        scan_times,
                        age_cond=scan_ages,
                    )
                    input = torch.cat([batch_vecs, xyz[i]], dim=1)
                    
                    # NN optimization
                    pred_sdf = decoder(input)

                    if enforce_minmax:
                        pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                    chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples
                    sdf_loss_tb += chunk_loss.item()

                    if do_code_regularization:
                        unique_subjects = torch.unique(scan_subject_indices)
                        unique_anchor_vecs = lat_vecs(unique_subjects)
                        l2_size_loss = torch.mean(torch.norm(unique_anchor_vecs, dim=1))
                        reg_loss = code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    
                        chunk_loss = chunk_loss + reg_loss.cuda()
                        reg_loss_tb += reg_loss.item()

                    if use_zero_displacement_loss and zero_displacement_lambda > 0.0:
                        unique_subjects = torch.unique(scan_subject_indices)
                        unique_anchor_vecs = lat_vecs(unique_subjects)
                        baseline_t = subject_baseline_time.index_select(0, unique_subjects).unsqueeze(1)
                        baseline_age = None
                        if use_age_conditioning:
                            baseline_age = subject_baseline_age.index_select(0, unique_subjects)
                        baseline_components = get_disentangled_velocity_components(
                            temporal_flow,
                            unique_anchor_vecs,
                            baseline_t,
                            baseline_t,
                            age_cond=baseline_age,
                        )
                        if baseline_components is None:
                            baseline_velocity = temporal_flow(
                                unique_anchor_vecs,
                                baseline_t,
                                baseline_t,
                                age_cond=baseline_age,
                            )
                        else:
                            baseline_velocity = torch.cat(
                                [
                                    baseline_components["disease_raw"],
                                    baseline_components["residual"],
                                ],
                                dim=1,
                            )
                        zero_disp_loss = zero_displacement_lambda * torch.mean(baseline_velocity ** 2)
                        chunk_loss = chunk_loss + zero_disp_loss
                        reg_loss_tb += float(zero_disp_loss.item())

                    if use_disentangled_velocity_flow:
                        unique_scan_indices = torch.unique(indices[i])
                        unique_scan_subjects = scan_to_subject_idx.index_select(
                            0, unique_scan_indices
                        )
                        unique_anchors = lat_vecs(unique_scan_subjects)
                        unique_start_times = subject_baseline_time.index_select(
                            0, unique_scan_subjects
                        ).unsqueeze(1)
                        unique_scan_times = scan_to_time.index_select(
                            0, unique_scan_indices
                        ).unsqueeze(1)
                        unique_scan_ages = None
                        if use_age_conditioning:
                            unique_scan_ages = scan_age_condition.index_select(
                                0, unique_scan_indices
                            )
                        velocity_losses = compute_disentangled_velocity_losses(
                            temporal_flow,
                            unique_anchors,
                            unique_start_times,
                            unique_scan_times,
                            age_cond=unique_scan_ages,
                            use_disease_zero=use_velocity_disease_zero_loss,
                            disease_zero_lambda=velocity_disease_zero_lambda,
                            use_residual=use_velocity_residual_loss,
                            residual_lambda=velocity_residual_lambda,
                            use_residual_diagnosis_covariance=(
                                use_velocity_residual_diagnosis_covariance_loss
                            ),
                            residual_diagnosis_covariance_lambda=(
                                velocity_residual_diagnosis_covariance_lambda
                            ),
                            use_disease_margin=use_velocity_disease_margin_loss,
                            disease_margin_lambda=velocity_disease_margin_lambda,
                            disease_margin=velocity_disease_margin,
                            use_adversarial_leakage=(
                                use_velocity_adversarial_leakage_loss
                            ),
                            adversarial_age_lambda=(
                                velocity_adversarial_age_lambda
                            ),
                            adversarial_residual_lambda=(
                                velocity_adversarial_residual_lambda
                            ),
                            adversarial_grl_lambda=(
                                velocity_adversarial_grl_lambda
                            ),
                            use_disease_classification=(
                                use_velocity_disease_classification_loss
                            ),
                            disease_classification_lambda=(
                                velocity_disease_classification_lambda
                            ),
                        )
                        chunk_loss = chunk_loss + velocity_losses["loss"]
                        velocity_disease_zero_loss_tb += float(
                            velocity_losses["disease_zero"].detach().item()
                        )
                        velocity_residual_loss_tb += float(
                            velocity_losses["residual"].detach().item()
                        )
                        velocity_residual_diagnosis_covariance_loss_tb += float(
                            velocity_losses[
                                "residual_diagnosis_covariance"
                            ].detach().item()
                        )
                        velocity_disease_margin_loss_tb += float(
                            velocity_losses["disease_margin"].detach().item()
                        )
                        velocity_age_adversarial_loss_tb += float(
                            velocity_losses["age_adversarial"].detach().item()
                        )
                        velocity_residual_adversarial_loss_tb += float(
                            velocity_losses["residual_adversarial"].detach().item()
                        )
                        velocity_disease_classification_loss_tb += float(
                            velocity_losses["disease_classification"].detach().item()
                        )
                        velocity_age_adversarial_bce_tb += float(
                            velocity_losses["age_adversarial_bce"].detach().item()
                        )
                        velocity_residual_adversarial_bce_tb += float(
                            velocity_losses["residual_adversarial_bce"].detach().item()
                        )
                        velocity_disease_classification_bce_tb += float(
                            velocity_losses[
                                "disease_classification_bce"
                            ].detach().item()
                        )
                        velocity_age_adversarial_acc_tb += float(
                            velocity_losses["age_adversarial_acc"].detach().item()
                        )
                        velocity_residual_adversarial_acc_tb += float(
                            velocity_losses[
                                "residual_adversarial_acc"
                            ].detach().item()
                        )
                        velocity_disease_classification_acc_tb += float(
                            velocity_losses[
                                "disease_classification_acc"
                            ].detach().item()
                        )
                        velocity_age_norm_tb += float(
                            velocity_losses["age_norm"].detach().item()
                        )
                        velocity_disease_raw_norm_tb += float(
                            velocity_losses["disease_raw_norm"].detach().item()
                        )
                        velocity_disease_raw_healthy_norm_tb += float(
                            velocity_losses[
                                "disease_raw_norm_healthy"
                            ].detach().item()
                        )
                        velocity_disease_raw_diseased_norm_tb += float(
                            velocity_losses[
                                "disease_raw_norm_diseased"
                            ].detach().item()
                        )
                        velocity_disease_norm_tb += float(
                            velocity_losses["disease_norm"].detach().item()
                        )
                        velocity_residual_norm_tb += float(
                            velocity_losses["residual_norm"].detach().item()
                        )

                    if (
                        use_cocycle_loss
                        or use_cocycle_backward_loss
                        or use_pair_forward_loss
                        or use_pair_backward_loss
                        or use_general_cocycle_loss
                        or use_general_cocycle_backward_loss
                    ):
                        unique_subjects = torch.unique(scan_subject_indices)
                        if unique_subjects.numel() > 0:
                            subject_anchors = lat_vecs(unique_subjects)
                            subject_count = subject_anchors.shape[0]
                            baseline_s = subject_baseline_time.index_select(
                                0, unique_subjects
                            ).unsqueeze(1)

                            def _sampled_condition(tt):
                                if not use_age_conditioning:
                                    return None
                                return subject_condition_at_time(
                                    unique_subjects,
                                    tt,
                                    subject_condition_fit,
                                    device=subject_anchors.device,
                                    dtype=subject_anchors.dtype,
                                )

                            if use_pair_forward_loss and pair_forward_lambda > 0.0:
                                pair_fwd_raw_loss = 0.0
                                pair_fwd_samples = 0
                                for _ in range(pair_forward_pairs_per_subject):
                                    s, t = _sample_sorted_time_pairs(
                                        subject_count,
                                        device=subject_anchors.device,
                                        mode=consistency_pair_sampling_mode,
                                        dtype=subject_anchors.dtype,
                                        uniform_low=consistency_time_sample_low,
                                        uniform_high=consistency_time_sample_high,
                                        mixed_adjacent_ratio=consistency_pair_adjacent_ratio,
                                        mixed_adjacent_max_gap_ratio=consistency_pair_adjacent_max_gap_ratio,
                                        mixed_far_min_gap_ratio=consistency_pair_far_min_gap_ratio,
                                    )
                                    if s is None:
                                        continue
                                    cond_s = _sampled_condition(s)
                                    cond_t = _sampled_condition(t)
                                    z_s = apply_temporal_flow(
                                        temporal_flow,
                                        subject_anchors,
                                        baseline_s,
                                        s,
                                        age_cond=cond_s,
                                    )
                                    z_t = apply_temporal_flow(
                                        temporal_flow,
                                        subject_anchors,
                                        baseline_s,
                                        t,
                                        age_cond=cond_t,
                                    )
                                    z_st = apply_temporal_flow(
                                        temporal_flow, z_s, s, t, age_cond=cond_t
                                    )
                                    pair_fwd_raw_loss = pair_fwd_raw_loss + torch.mean(
                                        (z_st - z_t) ** 2
                                    )
                                    pair_fwd_samples += 1
                                if pair_fwd_samples > 0:
                                    pair_fwd_raw_loss = pair_fwd_raw_loss / pair_fwd_samples
                                    pair_fwd_loss = pair_forward_lambda * pair_fwd_raw_loss
                                    chunk_loss = chunk_loss + pair_fwd_loss
                                    pair_forward_loss_tb += pair_fwd_loss.item()

                            if use_pair_backward_loss and pair_backward_lambda > 0.0:
                                pair_bwd_raw_loss = 0.0
                                pair_bwd_samples = 0
                                for _ in range(pair_backward_pairs_per_subject):
                                    s, t = _sample_sorted_time_pairs(
                                        subject_count,
                                        device=subject_anchors.device,
                                        mode=consistency_pair_sampling_mode,
                                        dtype=subject_anchors.dtype,
                                        uniform_low=consistency_time_sample_low,
                                        uniform_high=consistency_time_sample_high,
                                        mixed_adjacent_ratio=consistency_pair_adjacent_ratio,
                                        mixed_adjacent_max_gap_ratio=consistency_pair_adjacent_max_gap_ratio,
                                        mixed_far_min_gap_ratio=consistency_pair_far_min_gap_ratio,
                                    )
                                    if s is None:
                                        continue
                                    cond_s = _sampled_condition(s)
                                    cond_t = _sampled_condition(t)
                                    z_s = apply_temporal_flow(
                                        temporal_flow,
                                        subject_anchors,
                                        baseline_s,
                                        s,
                                        age_cond=cond_s,
                                    )
                                    z_t = apply_temporal_flow(
                                        temporal_flow,
                                        subject_anchors,
                                        baseline_s,
                                        t,
                                        age_cond=cond_t,
                                    )
                                    z_ts = apply_temporal_flow(
                                        temporal_flow, z_t, t, s, age_cond=cond_s
                                    )
                                    pair_bwd_raw_loss = pair_bwd_raw_loss + torch.mean(
                                        (z_ts - z_s) ** 2
                                    )
                                    pair_bwd_samples += 1
                                if pair_bwd_samples > 0:
                                    pair_bwd_raw_loss = pair_bwd_raw_loss / pair_bwd_samples
                                    pair_bwd_loss = pair_backward_lambda * pair_bwd_raw_loss
                                    chunk_loss = chunk_loss + pair_bwd_loss
                                    pair_backward_loss_tb += pair_bwd_loss.item()

                            if use_cocycle_loss and cocycle_lambda > 0.0:
                                cocycle_raw_loss = 0.0
                                cocycle_samples = 0
                                cocycle_shape_raw_loss = 0.0
                                cocycle_shape_samples = 0
                                for _ in range(cocycle_pairs_per_subject):
                                    r, t = _sample_sorted_time_pairs(
                                        subject_count,
                                        device=subject_anchors.device,
                                        mode=consistency_pair_sampling_mode,
                                        dtype=subject_anchors.dtype,
                                        uniform_low=consistency_time_sample_low,
                                        uniform_high=consistency_time_sample_high,
                                        mixed_adjacent_ratio=consistency_pair_adjacent_ratio,
                                        mixed_adjacent_max_gap_ratio=consistency_pair_adjacent_max_gap_ratio,
                                        mixed_far_min_gap_ratio=consistency_pair_far_min_gap_ratio,
                                    )
                                    if r is None:
                                        continue
                                    cond_r = _sampled_condition(r)
                                    cond_t = _sampled_condition(t)
                                    z_r = apply_temporal_flow(
                                        temporal_flow,
                                        subject_anchors,
                                        baseline_s,
                                        r,
                                        age_cond=cond_r,
                                    )
                                    z_rt = apply_temporal_flow(
                                        temporal_flow, z_r, r, t, age_cond=cond_t
                                    )
                                    z_t = apply_temporal_flow(
                                        temporal_flow,
                                        subject_anchors,
                                        baseline_s,
                                        t,
                                        age_cond=cond_t,
                                    )
                                    cocycle_raw_loss = cocycle_raw_loss + torch.mean(
                                        (z_rt - z_t) ** 2
                                    )
                                    cocycle_samples += 1
                                    if (
                                        use_cocycle_shape_loss
                                        and cocycle_shape_lambda > 0.0
                                    ):
                                        query_xyz = _sample_query_points(
                                            xyz[i].detach(), cocycle_shape_num_points
                                        )
                                        clamp_min_shape = minT if (cocycle_shape_use_clamp and enforce_minmax) else None
                                        clamp_max_shape = maxT if (cocycle_shape_use_clamp and enforce_minmax) else None
                                        cocycle_shape_raw = _compute_shape_cocycle_sdf_loss(
                                            decoder,
                                            z_rt,
                                            z_t,
                                            query_xyz,
                                            clamp_min=clamp_min_shape,
                                            clamp_max=clamp_max_shape,
                                        )
                                        if cocycle_shape_raw is not None:
                                            cocycle_shape_raw_loss = (
                                                cocycle_shape_raw_loss + cocycle_shape_raw
                                            )
                                            cocycle_shape_samples += 1
                                if cocycle_samples > 0:
                                    cocycle_raw_loss = cocycle_raw_loss / cocycle_samples
                                    cocycle_loss = cocycle_lambda * cocycle_raw_loss
                                    chunk_loss = chunk_loss + cocycle_loss
                                    cocycle_loss_tb += cocycle_loss.item()
                                if cocycle_shape_samples > 0:
                                    cocycle_shape_raw_loss = (
                                        cocycle_shape_raw_loss / cocycle_shape_samples
                                    )
                                    cocycle_shape_loss = (
                                        cocycle_shape_lambda * cocycle_shape_raw_loss
                                    )
                                    chunk_loss = chunk_loss + cocycle_shape_loss
                                    cocycle_shape_loss_tb += cocycle_shape_loss.item()

                            if use_cocycle_backward_loss and cocycle_backward_lambda > 0.0:
                                cocycle_backward_raw_loss = 0.0
                                cocycle_backward_samples = 0
                                for _ in range(cocycle_backward_pairs_per_subject):
                                    r, t = _sample_sorted_time_pairs(
                                        subject_count,
                                        device=subject_anchors.device,
                                        mode=consistency_pair_sampling_mode,
                                        dtype=subject_anchors.dtype,
                                        uniform_low=consistency_time_sample_low,
                                        uniform_high=consistency_time_sample_high,
                                        mixed_adjacent_ratio=consistency_pair_adjacent_ratio,
                                        mixed_adjacent_max_gap_ratio=consistency_pair_adjacent_max_gap_ratio,
                                        mixed_far_min_gap_ratio=consistency_pair_far_min_gap_ratio,
                                    )
                                    if r is None:
                                        continue
                                    cond_r = _sampled_condition(r)
                                    cond_t = _sampled_condition(t)
                                    z_t = apply_temporal_flow(
                                        temporal_flow,
                                        subject_anchors,
                                        baseline_s,
                                        t,
                                        age_cond=cond_t,
                                    )
                                    z_tr = apply_temporal_flow(
                                        temporal_flow, z_t, t, r, age_cond=cond_r
                                    )
                                    z_r = apply_temporal_flow(
                                        temporal_flow,
                                        subject_anchors,
                                        baseline_s,
                                        r,
                                        age_cond=cond_r,
                                    )
                                    cocycle_backward_raw_loss = (
                                        cocycle_backward_raw_loss + torch.mean((z_tr - z_r) ** 2)
                                    )
                                    cocycle_backward_samples += 1
                                if cocycle_backward_samples > 0:
                                    cocycle_backward_raw_loss = (
                                        cocycle_backward_raw_loss / cocycle_backward_samples
                                    )
                                    cocycle_backward_loss = (
                                        cocycle_backward_lambda * cocycle_backward_raw_loss
                                    )
                                    chunk_loss = chunk_loss + cocycle_backward_loss
                                    cocycle_backward_loss_tb += cocycle_backward_loss.item()

                            if use_general_cocycle_loss and general_cocycle_lambda > 0.0:
                                general_cocycle_raw_loss = 0.0
                                general_cocycle_samples = 0
                                for _ in range(general_cocycle_triplets_per_subject):
                                    s, r, t = _sample_sorted_time_triplets(
                                        subject_count,
                                        device=subject_anchors.device,
                                        mode="uniform_01",
                                        dtype=subject_anchors.dtype,
                                        uniform_low=consistency_time_sample_low,
                                        uniform_high=consistency_time_sample_high,
                                    )
                                    if s is None:
                                        continue
                                    cond_s = _sampled_condition(s)
                                    cond_r = _sampled_condition(r)
                                    cond_t = _sampled_condition(t)
                                    z_s = apply_temporal_flow(
                                        temporal_flow,
                                        subject_anchors,
                                        baseline_s,
                                        s,
                                        age_cond=cond_s,
                                    )
                                    z_sr = apply_temporal_flow(
                                        temporal_flow, z_s, s, r, age_cond=cond_r
                                    )
                                    z_srt = apply_temporal_flow(
                                        temporal_flow, z_sr, r, t, age_cond=cond_t
                                    )
                                    z_st = apply_temporal_flow(
                                        temporal_flow, z_s, s, t, age_cond=cond_t
                                    )
                                    general_cocycle_raw_loss = (
                                        general_cocycle_raw_loss + torch.mean((z_srt - z_st) ** 2)
                                    )
                                    general_cocycle_samples += 1
                                if general_cocycle_samples > 0:
                                    general_cocycle_raw_loss = (
                                        general_cocycle_raw_loss / general_cocycle_samples
                                    )
                                    general_cocycle_loss = (
                                        general_cocycle_lambda * general_cocycle_raw_loss
                                    )
                                    chunk_loss = chunk_loss + general_cocycle_loss
                                    general_cocycle_loss_tb += general_cocycle_loss.item()

                            if (
                                use_general_cocycle_backward_loss
                                and general_cocycle_backward_lambda > 0.0
                            ):
                                general_cocycle_backward_raw_loss = 0.0
                                general_cocycle_backward_samples = 0
                                for _ in range(general_cocycle_backward_triplets_per_subject):
                                    s, r, t = _sample_sorted_time_triplets(
                                        subject_count,
                                        device=subject_anchors.device,
                                        mode="uniform_01",
                                        dtype=subject_anchors.dtype,
                                        uniform_low=consistency_time_sample_low,
                                        uniform_high=consistency_time_sample_high,
                                    )
                                    if s is None:
                                        continue
                                    cond_s = _sampled_condition(s)
                                    cond_r = _sampled_condition(r)
                                    cond_t = _sampled_condition(t)
                                    z_t = apply_temporal_flow(
                                        temporal_flow,
                                        subject_anchors,
                                        baseline_s,
                                        t,
                                        age_cond=cond_t,
                                    )
                                    z_tr = apply_temporal_flow(
                                        temporal_flow, z_t, t, r, age_cond=cond_r
                                    )
                                    z_trs = apply_temporal_flow(
                                        temporal_flow, z_tr, r, s, age_cond=cond_s
                                    )
                                    z_ts = apply_temporal_flow(
                                        temporal_flow, z_t, t, s, age_cond=cond_s
                                    )
                                    general_cocycle_backward_raw_loss = (
                                        general_cocycle_backward_raw_loss
                                        + torch.mean((z_trs - z_ts) ** 2)
                                    )
                                    general_cocycle_backward_samples += 1
                                if general_cocycle_backward_samples > 0:
                                    general_cocycle_backward_raw_loss = (
                                        general_cocycle_backward_raw_loss
                                        / general_cocycle_backward_samples
                                    )
                                    general_cocycle_backward_loss = (
                                        general_cocycle_backward_lambda
                                        * general_cocycle_backward_raw_loss
                                    )
                                    chunk_loss = chunk_loss + general_cocycle_backward_loss
                                    general_cocycle_backward_loss_tb += (
                                        general_cocycle_backward_loss.item()
                                    )

                    if use_real_scan_pair_loss:
                        unique_subjects = torch.unique(scan_subject_indices)
                        real_pair_losses = _compute_real_scan_pair_batch_losses(
                            decoder=decoder,
                            temporal_flow=temporal_flow,
                            lat_vecs=lat_vecs,
                            unique_subjects=unique_subjects,
                            subject_to_scan_indices=subject_to_scan_indices,
                            sdf_dataset=sdf_dataset,
                            scan_to_time_cpu=scan_to_time_cpu,
                            scan_age_condition=scan_age_condition,
                            subject_baseline_time=subject_baseline_time,
                            use_age_conditioning=use_age_conditioning,
                            pairs_per_subject=real_scan_pairs_per_subject,
                            num_samples=real_scan_pair_num_samples,
                            use_forward=use_real_scan_pair_forward,
                            use_backward=use_real_scan_pair_backward,
                            clamp_min=minT,
                            clamp_max=maxT,
                        )

                        if real_pair_losses["forward_count"] > 0:
                            real_pair_forward_reconstruction_loss = (
                                real_scan_pair_reconstruction_lambda
                                * real_pair_losses["forward_reconstruction"]
                            )
                            real_pair_forward_latent_loss = (
                                real_scan_pair_latent_lambda
                                * real_pair_losses["forward_latent"]
                            )
                            chunk_loss = (
                                chunk_loss
                                + real_pair_forward_reconstruction_loss
                                + real_pair_forward_latent_loss
                            )
                            real_pair_forward_reconstruction_loss_tb += float(
                                real_pair_forward_reconstruction_loss.item()
                            )
                            real_pair_forward_latent_loss_tb += float(
                                real_pair_forward_latent_loss.item()
                            )

                        if real_pair_losses["backward_count"] > 0:
                            real_pair_backward_reconstruction_loss = (
                                real_scan_pair_reconstruction_lambda
                                * real_pair_losses["backward_reconstruction"]
                            )
                            real_pair_backward_latent_loss = (
                                real_scan_pair_latent_lambda
                                * real_pair_losses["backward_latent"]
                            )
                            chunk_loss = (
                                chunk_loss
                                + real_pair_backward_reconstruction_loss
                                + real_pair_backward_latent_loss
                            )
                            real_pair_backward_reconstruction_loss_tb += float(
                                real_pair_backward_reconstruction_loss.item()
                            )
                            real_pair_backward_latent_loss_tb += float(
                                real_pair_backward_latent_loss.item()
                            )
                    
                    # Isometry/metric losses computation
                    if (use_isometry or use_grad_metric_iso) and (global_batch_idx % iso_compute_frequency == 0):
                        # Use the underlying decoder (unwrap DataParallel if needed)
                        decoder_for_iso = decoder.module if hasattr(decoder, 'module') else decoder
                        iso_device = next(decoder_for_iso.parameters()).device
                        iso_loss_sum = 0.0
                        iso_g1_sum = 0.0
                        iso_g2_sum = 0.0
                        grad_metric_iso_loss_sum = 0.0
                        iso_scene_count = 0

                        iso_indices = unique_indices
                        if (
                            iso_scenes_per_batch is not None
                            and iso_scenes_per_batch > 0
                            and unique_indices.numel() > iso_scenes_per_batch
                        ):
                            perm = torch.randperm(
                                unique_indices.numel(), device=unique_indices.device
                            )
                            iso_indices = unique_indices[perm[:iso_scenes_per_batch]]

                        for scene_idx in iso_indices:
                            mask = indices[i] == scene_idx
                            if mask.sum() == 0:
                                continue

                            xyz_scene = xyz[i][mask]
                            sdf_scene = sdf_gt[i][mask]

                            # Select near-surface points for isometry computation
                            iso_points = select_near_surface_points(
                                xyz_scene.detach(),
                                sdf_scene.detach(),
                                clamp_dist,
                                iso_num_points,
                            ).to(iso_device)

                            # Latent for this scene (all entries are identical)
                            sample_latent = batch_vecs[mask][:1]
                            if use_isometry_mixup and unique_indices.numel() > 1:
                                if torch.rand(1).item() < iso_mixup_prob:
                                    idx_pool = unique_indices[unique_indices != scene_idx]
                                    if idx_pool.numel() > 0:
                                        rand_scene_idx = idx_pool[
                                            torch.randint(
                                                0,
                                                idx_pool.numel(),
                                                (1,),
                                                device=idx_pool.device,
                                            )
                                        ]
                                        mix_mask = indices[i] == rand_scene_idx
                                        mix_latent = batch_vecs[mix_mask][:1]
                                        if mix_latent.numel() > 0:
                                            mix_latent = mix_latent.to(sample_latent.device)
                                            mix_alpha = torch.distributions.Beta(
                                                iso_mixup_alpha, iso_mixup_alpha
                                            ).sample((1,)).to(sample_latent.device)
                                            sample_latent = (
                                                mix_alpha * sample_latent + (1.0 - mix_alpha) * mix_latent
                                            )

                            sample_latent = sample_latent.to(iso_device)
                            iso_latent_expanded = sample_latent.expand(iso_num_points, -1).to(iso_device)  # [K, m]

                            if use_isometry:
                                iso_loss = lambda_iso * isometry_loss_fn(
                                    decoder_for_iso,
                                    iso_latent_expanded,
                                    iso_points,
                                    latent_size,
                                )
                                iso_g1 = getattr(isometry_loss_fn, "last_g1", None)
                                iso_g2 = getattr(isometry_loss_fn, "last_g2", None)
                                if iso_g1 is not None:
                                    iso_g1_sum += iso_g1.item()
                                if iso_g2 is not None:
                                    iso_g2_sum += iso_g2.item()
                                iso_loss_sum = iso_loss_sum + iso_loss

                            if use_grad_metric_iso:
                                grad_metric_iso_loss = grad_metric_iso_lambda * grad_metric_iso_fn(
                                    decoder_for_iso,
                                    iso_latent_expanded,
                                    iso_points,
                                    latent_size,
                                )
                                grad_metric_iso_loss_sum = (
                                    grad_metric_iso_loss_sum + grad_metric_iso_loss
                                )
                            iso_scene_count += 1

                        if iso_scene_count > 0:
                            if use_isometry:
                                iso_loss = iso_loss_sum / iso_scene_count
                                chunk_loss = chunk_loss + iso_loss
                                iso_loss_tb += iso_loss.item()
                                iso_g1_tb += iso_g1_sum / iso_scene_count
                                iso_g2_tb += iso_g2_sum / iso_scene_count
                            if use_grad_metric_iso:
                                grad_metric_iso_loss = grad_metric_iso_loss_sum / iso_scene_count
                                chunk_loss = chunk_loss + grad_metric_iso_loss
                                grad_metric_iso_loss_tb += grad_metric_iso_loss.item()
                    
                    summary_writer.add_scalar(
                        "Loss/train_vanilla",
                        float(chunk_loss.detach().item()),
                        global_step=epoch,
                    )
                    if use_eikonal:
                        grad_outputs = torch.ones_like(pred_sdf, requires_grad=True)
                        gradients = torch.autograd.grad(pred_sdf, [xyz[i]], grad_outputs=grad_outputs, create_graph=True, allow_unused=True, retain_graph=True)[0]
                        eikonal_loss = 0.002 * ((1. - torch.linalg.vector_norm(gradients, dim=1))**2).mean()
                        chunk_loss += eikonal_loss
                        eikonal_loss_tb += eikonal_loss.item()

                    chunk_loss.backward()

                    batch_loss_tb += chunk_loss.item()
                    # Print batch loss
                #print(f"Batch loss: {batch_loss_tb}")                    
                logging.debug("loss = {}".format(batch_loss_tb))
                if use_covariance:
                    cov_indices = scan_to_subject_idx.index_select(0, indices_batch)
                    if cov_indices.device != lat_vecs.weight.device:
                        cov_indices = cov_indices.to(lat_vecs.weight.device)
                    cov_indices = torch.unique(cov_indices)
                    cov_latents = lat_vecs.weight.index_select(0, cov_indices)
                    cov_loss = lambda_cov * covariance_loss_fn(cov_latents)
                    cov_loss.backward()
                    cov_loss_tb = cov_loss.item()
                    batch_loss_tb += cov_loss_tb
                if use_gmm_prior:
                    gmm_indices = scan_to_subject_idx.index_select(0, indices_batch)
                    if gmm_indices.device != lat_vecs.weight.device:
                        gmm_indices = gmm_indices.to(lat_vecs.weight.device)
                    gmm_indices = torch.unique(gmm_indices)
                    if gmm_indices.numel() > 0:
                        gmm_latents = lat_vecs.weight.index_select(0, gmm_indices)
                        gmm_loss_raw = gmm_prior_loss_fn(gmm_latents)
                        gmm_loss = gmm_lambda * gmm_loss_raw
                        gmm_loss.backward()
                        gmm_loss_tb = gmm_loss.item()
                        batch_loss_tb += gmm_loss_tb
                        gmm_nll = getattr(gmm_prior_loss_fn, "last_nll", None)
                        gmm_entropy = getattr(gmm_prior_loss_fn, "last_avg_entropy", None)
                        if gmm_nll is not None:
                            gmm_nll_tb = gmm_nll.item()
                        if gmm_entropy is not None:
                            gmm_entropy_tb = gmm_entropy.item()

                loss_log.append(batch_loss_tb)
                epoch_losses.append(batch_loss_tb)
                epoch_sdf_losses.append(sdf_loss_tb)
                epoch_reg_losses.append(reg_loss_tb)
                epoch_eikonal_losses.append(eikonal_loss_tb)
                epoch_cov_losses.append(cov_loss_tb)
                epoch_gmm_losses.append(gmm_loss_tb)
                epoch_gmm_nlls.append(gmm_nll_tb)
                epoch_gmm_entropies.append(gmm_entropy_tb)
                epoch_iso_losses.append(iso_loss_tb)
                epoch_grad_metric_iso_losses.append(grad_metric_iso_loss_tb)
                epoch_iso_g1_losses.append(iso_g1_tb)
                epoch_iso_g2_losses.append(iso_g2_tb)
                epoch_cocycle_losses.append(cocycle_loss_tb)
                epoch_cocycle_shape_losses.append(cocycle_shape_loss_tb)
                epoch_cocycle_backward_losses.append(cocycle_backward_loss_tb)
                epoch_pair_forward_losses.append(pair_forward_loss_tb)
                epoch_pair_backward_losses.append(pair_backward_loss_tb)
                epoch_general_cocycle_losses.append(general_cocycle_loss_tb)
                epoch_general_cocycle_backward_losses.append(
                    general_cocycle_backward_loss_tb
                )
                epoch_real_pair_forward_reconstruction_losses.append(
                    real_pair_forward_reconstruction_loss_tb
                )
                epoch_real_pair_forward_latent_losses.append(
                    real_pair_forward_latent_loss_tb
                )
                epoch_real_pair_backward_reconstruction_losses.append(
                    real_pair_backward_reconstruction_loss_tb
                )
                epoch_real_pair_backward_latent_losses.append(
                    real_pair_backward_latent_loss_tb
                )
                epoch_velocity_disease_zero_losses.append(
                    velocity_disease_zero_loss_tb
                )
                epoch_velocity_residual_losses.append(velocity_residual_loss_tb)
                epoch_velocity_residual_diagnosis_covariance_losses.append(
                    velocity_residual_diagnosis_covariance_loss_tb
                )
                epoch_velocity_disease_margin_losses.append(
                    velocity_disease_margin_loss_tb
                )
                epoch_velocity_age_adversarial_losses.append(
                    velocity_age_adversarial_loss_tb
                )
                epoch_velocity_residual_adversarial_losses.append(
                    velocity_residual_adversarial_loss_tb
                )
                epoch_velocity_disease_classification_losses.append(
                    velocity_disease_classification_loss_tb
                )
                epoch_velocity_age_adversarial_bces.append(
                    velocity_age_adversarial_bce_tb
                )
                epoch_velocity_residual_adversarial_bces.append(
                    velocity_residual_adversarial_bce_tb
                )
                epoch_velocity_disease_classification_bces.append(
                    velocity_disease_classification_bce_tb
                )
                epoch_velocity_age_adversarial_accs.append(
                    velocity_age_adversarial_acc_tb
                )
                epoch_velocity_residual_adversarial_accs.append(
                    velocity_residual_adversarial_acc_tb
                )
                epoch_velocity_disease_classification_accs.append(
                    velocity_disease_classification_acc_tb
                )
                epoch_velocity_age_norms.append(velocity_age_norm_tb)
                epoch_velocity_disease_raw_norms.append(
                    velocity_disease_raw_norm_tb
                )
                epoch_velocity_disease_raw_healthy_norms.append(
                    velocity_disease_raw_healthy_norm_tb
                )
                epoch_velocity_disease_raw_diseased_norms.append(
                    velocity_disease_raw_diseased_norm_tb
                )
                epoch_velocity_disease_norms.append(velocity_disease_norm_tb)
                epoch_velocity_residual_norms.append(velocity_residual_norm_tb)

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(decoder.parameters()) + list(temporal_flow.parameters()),
                        grad_clip,
                        norm_type=2,
                    )

                optimizer_all.step()
                if code_bound is not None:
                    with torch.no_grad():
                        bound = float(code_bound)
                        if bound > 0.0:
                            n = lat_vecs.weight.norm(dim=1, keepdim=True)
                            scale = torch.clamp(bound / (n + 1e-12), max=1.0)
                            lat_vecs.weight.mul_(scale)

            # LOG EPOCH
            seconds_elapsed = time.time() - epoch_time_start
            timing_log.append(seconds_elapsed)
            # Log epoch losses.
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_sdf_loss = sum(epoch_sdf_losses) / len(epoch_sdf_losses)
            epoch_reg_loss = sum(epoch_reg_losses) / len(epoch_reg_losses)
            epoch_eikonal_loss = sum(epoch_eikonal_losses) / len(epoch_eikonal_losses)
            epoch_cov_loss = sum(epoch_cov_losses) / len(epoch_cov_losses)
            epoch_gmm_loss = sum(epoch_gmm_losses) / len(epoch_gmm_losses)
            epoch_gmm_nll = sum(epoch_gmm_nlls) / len(epoch_gmm_nlls)
            epoch_gmm_entropy = sum(epoch_gmm_entropies) / len(epoch_gmm_entropies)
            epoch_iso_loss_contrib = sum(epoch_iso_losses) / len(epoch_iso_losses)
            epoch_grad_metric_iso_loss = (
                sum(epoch_grad_metric_iso_losses) / len(epoch_grad_metric_iso_losses)
            )
            epoch_cocycle_loss = sum(epoch_cocycle_losses) / len(epoch_cocycle_losses)
            epoch_cocycle_shape_loss = (
                sum(epoch_cocycle_shape_losses) / len(epoch_cocycle_shape_losses)
            )
            epoch_cocycle_backward_loss = (
                sum(epoch_cocycle_backward_losses) / len(epoch_cocycle_backward_losses)
            )
            epoch_pair_forward_loss = (
                sum(epoch_pair_forward_losses) / len(epoch_pair_forward_losses)
            )
            epoch_pair_backward_loss = (
                sum(epoch_pair_backward_losses) / len(epoch_pair_backward_losses)
            )
            epoch_general_cocycle_loss = (
                sum(epoch_general_cocycle_losses) / len(epoch_general_cocycle_losses)
            )
            epoch_general_cocycle_backward_loss = (
                sum(epoch_general_cocycle_backward_losses)
                / len(epoch_general_cocycle_backward_losses)
            )
            epoch_real_pair_forward_reconstruction_loss = (
                sum(epoch_real_pair_forward_reconstruction_losses)
                / len(epoch_real_pair_forward_reconstruction_losses)
            )
            epoch_real_pair_forward_latent_loss = (
                sum(epoch_real_pair_forward_latent_losses)
                / len(epoch_real_pair_forward_latent_losses)
            )
            epoch_real_pair_backward_reconstruction_loss = (
                sum(epoch_real_pair_backward_reconstruction_losses)
                / len(epoch_real_pair_backward_reconstruction_losses)
            )
            epoch_real_pair_backward_latent_loss = (
                sum(epoch_real_pair_backward_latent_losses)
                / len(epoch_real_pair_backward_latent_losses)
            )
            epoch_velocity_disease_zero_loss = (
                sum(epoch_velocity_disease_zero_losses)
                / len(epoch_velocity_disease_zero_losses)
            )
            epoch_velocity_residual_loss = (
                sum(epoch_velocity_residual_losses)
                / len(epoch_velocity_residual_losses)
            )
            epoch_velocity_residual_diagnosis_covariance_loss = (
                sum(epoch_velocity_residual_diagnosis_covariance_losses)
                / len(epoch_velocity_residual_diagnosis_covariance_losses)
            )
            epoch_velocity_disease_margin_loss = (
                sum(epoch_velocity_disease_margin_losses)
                / len(epoch_velocity_disease_margin_losses)
            )
            epoch_velocity_age_adversarial_loss = (
                sum(epoch_velocity_age_adversarial_losses)
                / len(epoch_velocity_age_adversarial_losses)
            )
            epoch_velocity_residual_adversarial_loss = (
                sum(epoch_velocity_residual_adversarial_losses)
                / len(epoch_velocity_residual_adversarial_losses)
            )
            epoch_velocity_disease_classification_loss = (
                sum(epoch_velocity_disease_classification_losses)
                / len(epoch_velocity_disease_classification_losses)
            )
            epoch_velocity_age_adversarial_bce = (
                sum(epoch_velocity_age_adversarial_bces)
                / len(epoch_velocity_age_adversarial_bces)
            )
            epoch_velocity_residual_adversarial_bce = (
                sum(epoch_velocity_residual_adversarial_bces)
                / len(epoch_velocity_residual_adversarial_bces)
            )
            epoch_velocity_disease_classification_bce = (
                sum(epoch_velocity_disease_classification_bces)
                / len(epoch_velocity_disease_classification_bces)
            )
            epoch_velocity_age_adversarial_acc = (
                sum(epoch_velocity_age_adversarial_accs)
                / len(epoch_velocity_age_adversarial_accs)
            )
            epoch_velocity_residual_adversarial_acc = (
                sum(epoch_velocity_residual_adversarial_accs)
                / len(epoch_velocity_residual_adversarial_accs)
            )
            epoch_velocity_disease_classification_acc = (
                sum(epoch_velocity_disease_classification_accs)
                / len(epoch_velocity_disease_classification_accs)
            )
            epoch_velocity_age_norm = (
                sum(epoch_velocity_age_norms) / len(epoch_velocity_age_norms)
            )
            epoch_velocity_disease_raw_norm = (
                sum(epoch_velocity_disease_raw_norms)
                / len(epoch_velocity_disease_raw_norms)
            )
            epoch_velocity_disease_raw_healthy_norm = (
                sum(epoch_velocity_disease_raw_healthy_norms)
                / len(epoch_velocity_disease_raw_healthy_norms)
            )
            epoch_velocity_disease_raw_diseased_norm = (
                sum(epoch_velocity_disease_raw_diseased_norms)
                / len(epoch_velocity_disease_raw_diseased_norms)
            )
            epoch_velocity_disease_norm = (
                sum(epoch_velocity_disease_norms)
                / len(epoch_velocity_disease_norms)
            )
            epoch_velocity_residual_norm = (
                sum(epoch_velocity_residual_norms) / len(epoch_velocity_residual_norms)
            )

            print(f"Epoch {epoch} total loss: {epoch_loss}")
            print(f"Epoch {epoch} sdf loss (weighted): {epoch_sdf_loss}")
            if do_code_regularization:
                print(f"Epoch {epoch} code regularizer loss (weighted): {epoch_reg_loss}")
            if use_eikonal:
                print(f"Epoch {epoch} eikonal loss (weighted): {epoch_eikonal_loss}")
            if use_covariance:
                print(f"Epoch {epoch} covariance loss (weighted): {epoch_cov_loss}")
            if use_gmm_prior:
                print(f"Epoch {epoch} gmm loss (weighted): {epoch_gmm_loss}")
                print(f"Epoch {epoch} gmm nll (raw): {epoch_gmm_nll}")
                print(f"Epoch {epoch} gmm assign entropy: {epoch_gmm_entropy}")
                with torch.no_grad():
                    gmm_sigma = torch.exp(gmm_prior_loss_fn.log_sigma).clamp_min(gmm_min_sigma)
                    print(f"Epoch {epoch} gmm sigma mean: {gmm_sigma.mean().item()}")
                    print(f"Epoch {epoch} gmm sigma min: {gmm_sigma.min().item()}")
            if use_isometry:
                print(f"Epoch {epoch} isometry loss (weighted): {epoch_iso_loss_contrib}")
                iso_count = sum(1 for x in epoch_iso_losses if x > 0)
                epoch_iso_g1 = sum(epoch_iso_g1_losses) / max(1, iso_count)
                epoch_iso_g2 = sum(epoch_iso_g2_losses) / max(1, iso_count)
                print(f"Epoch {epoch} isometry G1: {epoch_iso_g1}")
                print(f"Epoch {epoch} isometry G2: {epoch_iso_g2}")
            if use_grad_metric_iso:
                print(
                    f"Epoch {epoch} grad metric iso loss (weighted): "
                    f"{epoch_grad_metric_iso_loss}"
                )
            if use_cocycle_loss:
                print(f"Epoch {epoch} cocycle loss (weighted): {epoch_cocycle_loss}")
            if use_cocycle_shape_loss and cocycle_shape_lambda > 0.0:
                print(
                    f"Epoch {epoch} cocycle shape loss (weighted): {epoch_cocycle_shape_loss}"
                )
            if use_cocycle_backward_loss:
                print(
                    f"Epoch {epoch} cocycle backward loss (weighted): "
                    f"{epoch_cocycle_backward_loss}"
                )
            if use_pair_forward_loss:
                print(
                    f"Epoch {epoch} pair forward consistency loss (weighted): "
                    f"{epoch_pair_forward_loss}"
                )
            if use_pair_backward_loss:
                print(
                    f"Epoch {epoch} pair backward consistency loss (weighted): "
                    f"{epoch_pair_backward_loss}"
                )
            if use_general_cocycle_loss:
                print(
                    f"Epoch {epoch} general cocycle loss (weighted): "
                    f"{epoch_general_cocycle_loss}"
                )
            if use_general_cocycle_backward_loss:
                print(
                    f"Epoch {epoch} general backward cocycle loss (weighted): "
                    f"{epoch_general_cocycle_backward_loss}"
                )
            if use_real_scan_pair_loss and use_real_scan_pair_forward:
                print(
                    f"Epoch {epoch} real pair forward reconstruction loss (weighted): "
                    f"{epoch_real_pair_forward_reconstruction_loss}"
                )
                print(
                    f"Epoch {epoch} real pair forward latent loss (weighted): "
                    f"{epoch_real_pair_forward_latent_loss}"
                )
            if use_real_scan_pair_loss and use_real_scan_pair_backward:
                print(
                    f"Epoch {epoch} real pair backward reconstruction loss (weighted): "
                    f"{epoch_real_pair_backward_reconstruction_loss}"
                )
                print(
                    f"Epoch {epoch} real pair backward latent loss (weighted): "
                    f"{epoch_real_pair_backward_latent_loss}"
                )
            if use_disentangled_velocity_flow:
                if use_velocity_disease_zero_loss:
                    print(
                        f"Epoch {epoch} velocity disease-zero loss (weighted): "
                        f"{epoch_velocity_disease_zero_loss}"
                    )
                if use_velocity_residual_loss:
                    print(
                        f"Epoch {epoch} velocity residual loss (weighted): "
                        f"{epoch_velocity_residual_loss}"
                    )
                if use_velocity_residual_diagnosis_covariance_loss:
                    print(
                        f"Epoch {epoch} velocity residual-diagnosis covariance loss "
                        f"(weighted): {epoch_velocity_residual_diagnosis_covariance_loss}"
                    )
                if use_velocity_disease_margin_loss:
                    print(
                        f"Epoch {epoch} velocity disease margin loss (weighted): "
                        f"{epoch_velocity_disease_margin_loss}"
                    )
                if use_velocity_adversarial_leakage_loss:
                    print(
                        f"Epoch {epoch} velocity age adversarial loss (weighted): "
                        f"{epoch_velocity_age_adversarial_loss}"
                    )
                    print(
                        f"Epoch {epoch} velocity residual adversarial loss (weighted): "
                        f"{epoch_velocity_residual_adversarial_loss}"
                    )
                    print(
                        f"Epoch {epoch} velocity age leakage BCE/acc: "
                        f"{epoch_velocity_age_adversarial_bce}/"
                        f"{epoch_velocity_age_adversarial_acc}"
                    )
                    print(
                        f"Epoch {epoch} velocity residual leakage BCE/acc: "
                        f"{epoch_velocity_residual_adversarial_bce}/"
                        f"{epoch_velocity_residual_adversarial_acc}"
                    )
                if use_velocity_disease_classification_loss:
                    print(
                        f"Epoch {epoch} velocity disease classification loss "
                        f"(weighted): {epoch_velocity_disease_classification_loss}"
                    )
                    print(
                        f"Epoch {epoch} velocity disease classifier BCE/acc: "
                        f"{epoch_velocity_disease_classification_bce}/"
                        f"{epoch_velocity_disease_classification_acc}"
                    )
                print(f"Epoch {epoch} velocity age norm: {epoch_velocity_age_norm}")
                print(
                    f"Epoch {epoch} velocity disease raw norm: "
                    f"{epoch_velocity_disease_raw_norm}"
                )
                print(
                    f"Epoch {epoch} velocity disease raw norm healthy: "
                    f"{epoch_velocity_disease_raw_healthy_norm}"
                )
                print(
                    f"Epoch {epoch} velocity disease raw norm diseased: "
                    f"{epoch_velocity_disease_raw_diseased_norm}"
                )
                print(
                    f"Epoch {epoch} velocity disease gated norm: "
                    f"{epoch_velocity_disease_norm}"
                )
                print(
                    f"Epoch {epoch} velocity residual norm: "
                    f"{epoch_velocity_residual_norm}"
                )

            print(f"Epoch {epoch} time (s): {seconds_elapsed:.2f}")
            loss_log_epoch.append(epoch_loss)
            summary_writer.add_scalar("Loss/train", epoch_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_sdf", epoch_sdf_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_reg", epoch_reg_loss, global_step=epoch)
            if use_eikonal:
                summary_writer.add_scalar("Loss/train_eikonal", epoch_eikonal_loss, global_step=epoch)
            if use_covariance:
                summary_writer.add_scalar("Loss/train_covariance", epoch_cov_loss, global_step=epoch)
            if use_cocycle_loss:
                summary_writer.add_scalar("Loss/train_cocycle", epoch_cocycle_loss, global_step=epoch)
            if use_cocycle_shape_loss and cocycle_shape_lambda > 0.0:
                summary_writer.add_scalar(
                    "Loss/train_cocycle_shape",
                    epoch_cocycle_shape_loss,
                    global_step=epoch,
                )
            if use_cocycle_backward_loss:
                summary_writer.add_scalar(
                    "Loss/train_cocycle_backward",
                    epoch_cocycle_backward_loss,
                    global_step=epoch,
                )
            if use_pair_forward_loss:
                summary_writer.add_scalar(
                    "Loss/train_pair_forward",
                    epoch_pair_forward_loss,
                    global_step=epoch,
                )
            if use_pair_backward_loss:
                summary_writer.add_scalar(
                    "Loss/train_pair_backward",
                    epoch_pair_backward_loss,
                    global_step=epoch,
                )
            if use_general_cocycle_loss:
                summary_writer.add_scalar(
                    "Loss/train_general_cocycle",
                    epoch_general_cocycle_loss,
                    global_step=epoch,
                )
            if use_general_cocycle_backward_loss:
                summary_writer.add_scalar(
                    "Loss/train_general_cocycle_backward",
                    epoch_general_cocycle_backward_loss,
                    global_step=epoch,
                )
            if use_real_scan_pair_loss and use_real_scan_pair_forward:
                summary_writer.add_scalar(
                    "Loss/train_real_pair_forward_reconstruction",
                    epoch_real_pair_forward_reconstruction_loss,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "Loss/train_real_pair_forward_latent",
                    epoch_real_pair_forward_latent_loss,
                    global_step=epoch,
                )
            if use_real_scan_pair_loss and use_real_scan_pair_backward:
                summary_writer.add_scalar(
                    "Loss/train_real_pair_backward_reconstruction",
                    epoch_real_pair_backward_reconstruction_loss,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "Loss/train_real_pair_backward_latent",
                    epoch_real_pair_backward_latent_loss,
                    global_step=epoch,
                )
            if use_disentangled_velocity_flow:
                summary_writer.add_scalar(
                    "Loss/train_velocity_disease_zero",
                    epoch_velocity_disease_zero_loss,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "Loss/train_velocity_residual",
                    epoch_velocity_residual_loss,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "Loss/train_velocity_residual_diagnosis_covariance",
                    epoch_velocity_residual_diagnosis_covariance_loss,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "Loss/train_velocity_disease_margin",
                    epoch_velocity_disease_margin_loss,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "Loss/train_velocity_adv_age",
                    epoch_velocity_age_adversarial_loss,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "Loss/train_velocity_adv_residual",
                    epoch_velocity_residual_adversarial_loss,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "Loss/train_velocity_disease_classification",
                    epoch_velocity_disease_classification_loss,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "VelocityProbeBCE/age_leakage",
                    epoch_velocity_age_adversarial_bce,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "VelocityProbeBCE/residual_leakage",
                    epoch_velocity_residual_adversarial_bce,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "VelocityProbeBCE/disease_classification",
                    epoch_velocity_disease_classification_bce,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "VelocityProbeAcc/age_leakage",
                    epoch_velocity_age_adversarial_acc,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "VelocityProbeAcc/residual_leakage",
                    epoch_velocity_residual_adversarial_acc,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "VelocityProbeAcc/disease_classification",
                    epoch_velocity_disease_classification_acc,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "VelocityNorm/age", epoch_velocity_age_norm, global_step=epoch
                )
                summary_writer.add_scalar(
                    "VelocityNorm/disease_raw",
                    epoch_velocity_disease_raw_norm,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "VelocityNorm/disease_raw_healthy",
                    epoch_velocity_disease_raw_healthy_norm,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "VelocityNorm/disease_raw_diseased",
                    epoch_velocity_disease_raw_diseased_norm,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "VelocityNorm/disease_gated",
                    epoch_velocity_disease_norm,
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "VelocityNorm/residual",
                    epoch_velocity_residual_norm,
                    global_step=epoch,
                )
            if use_gmm_prior:
                summary_writer.add_scalar("Loss/train_gmm", epoch_gmm_loss, global_step=epoch)
                summary_writer.add_scalar("Loss/train_gmm_nll", epoch_gmm_nll, global_step=epoch)
                summary_writer.add_scalar("Loss/train_gmm_entropy", epoch_gmm_entropy, global_step=epoch)
            if use_isometry:
                iso_count = sum(1 for x in epoch_iso_losses if x > 0)
                summary_writer.add_scalar(
                    "Loss/train_isometry",
                    sum(epoch_iso_losses) / max(1, iso_count),
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "Loss/train_isometry_G1",
                    sum(epoch_iso_g1_losses) / max(1, iso_count),
                    global_step=epoch,
                )
                summary_writer.add_scalar(
                    "Loss/train_isometry_G2",
                    sum(epoch_iso_g2_losses) / max(1, iso_count),
                    global_step=epoch,
                )
            if use_grad_metric_iso:
                summary_writer.add_scalar(
                    "Loss/train_grad_metric_iso",
                    epoch_grad_metric_iso_loss,
                    global_step=epoch,
                )
            # Log learning rate.
            lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])
            summary_writer.add_scalar("Learning Rate/Params", lr_log[-1][0], global_step=epoch)
            summary_writer.add_scalar("Learning Rate/Latent", lr_log[-1][1], global_step=epoch)
            # Log latent vector length.
            mlm = get_mean_latent_vector_magnitude(lat_vecs)
            lat_mag_log.append(mlm)
            summary_writer.add_scalar("Mean Latent Magnitude/train", mlm, global_step=epoch)
            append_parameter_magnitudes(param_mag_log, decoder)
            append_parameter_magnitudes(param_mag_log, temporal_flow)
            # Log weights and gradient flow.
            grad_norms = []
            for _name, _param in decoder.named_parameters():
                if _name.startswith("module.decoder."):
                    _name = _name[15:]
                summary_writer.add_scalar(f"WeightsNorm/{_name}", _param.norm(p=2).item(), global_step=epoch)
                if hasattr(_param, "grad") and _param.grad is not None:
                    grad_norm = _param.grad.detach().norm(p=2)
                    summary_writer.add_scalar(f"GradsNorm/{_name}.grad", grad_norm.item(), global_step=epoch)
                    grad_norms.append(grad_norm)
            for _name, _param in temporal_flow.named_parameters():
                summary_writer.add_scalar(
                    f"WeightsNorm/flow.{_name}", _param.norm(p=2).item(), global_step=epoch
                )
                if hasattr(_param, "grad") and _param.grad is not None:
                    grad_norm = _param.grad.detach().norm(p=2)
                    summary_writer.add_scalar(
                        f"GradsNorm/flow.{_name}.grad", grad_norm.item(), global_step=epoch
                    )
                    grad_norms.append(grad_norm)
            if grad_norms:
                summary_writer.add_scalar(
                    "GradsNorm/allNetParams.grad",
                    torch.norm(torch.stack(grad_norms), p=2).item(),
                    global_step=epoch,
                )
            if lat_vecs.weight.grad is not None:
                summary_writer.add_scalar(
                    "GradsNorm/allLatParams.grad",
                    torch.norm(lat_vecs.weight.grad.detach(), p=2).item(),
                    global_step=epoch,
                )

            # Save checkpoint.
            if epoch in checkpoints:
                save_checkpoints(epoch)

            if epoch % log_frequency == 0:
                save_latest(epoch)
                save_logs(
                    experiment_directory,
                    loss_log,
                    lr_log,
                    timing_log,
                    lat_mag_log,
                    param_mag_log,
                    epoch,
                )
            
        
                # EVALUATION 
            if torus_path is not None:
                
                # Only if the path to the GT meshes exists.
                if epoch % eval_train_frequency == 0:
                    logging.info(f"Train Evaluation Started...")
                    # Training-set evaluation: Reconstruct mesh from learned latent and compute metrics.
                    chamfer_dists = []
                    chamfer_dists_all = []
                    eval_train_time_start = time.time()
                    for index in eval_train_scene_idxs:
                        index_tensor = torch.tensor(
                            [index], dtype=torch.long, device=lat_vecs.weight.device
                        )
                        subject_idx = scan_to_subject_idx.index_select(0, index_tensor)
                        eval_time = scan_to_time.index_select(0, index_tensor).unsqueeze(1)
                        eval_start = subject_baseline_time.index_select(
                            0, subject_idx
                        ).unsqueeze(1)
                        subject_anchor = lat_vecs(subject_idx)
                        eval_age = None
                        if use_age_conditioning:
                            eval_age = scan_age_condition.index_select(0, index_tensor)
                        lat_vec = apply_temporal_flow(
                            temporal_flow,
                            subject_anchor,
                            eval_start,
                            eval_time,
                            age_cond=eval_age,
                        )
                        save_name = os.path.basename(sdf_dataset.npyfiles[index]).split(".npz")[0]
                        path = os.path.join(experiment_directory, ws.tb_logs_dir, ws.tb_logs_train_reconstructions, save_name)
                        if not os.path.exists(path):
                            os.makedirs(path)

                        start = time.time()
                        with torch.no_grad():
                            train_mesh = mesh.create_mesh(
                                decoder, 
                                lat_vec, 
                                N=eval_grid_res, 
                                max_batch=int(2 ** 18), 
                                filename=os.path.join(path, f"epoch={epoch}"),
                                return_trimesh=True,
                            )
                        logging.debug("[Train eval] Total time to create training mesh: {}".format(time.time() - start))

                        if train_mesh is not None:
                            gt_mesh_path = f"{torus_path}/{save_name}.obj"
                            cd, cd_all = metrics.compute_metric(
                                gt_mesh=gt_mesh_path,
                                gen_mesh=train_mesh,
                                metric=eval_chamfer_metric,
                                **eval_chamfer_metric_kwargs,
                            )
                            chamfer_dists.append(cd)
                            chamfer_dists_all.append(cd_all)
                        
                        del train_mesh, save_name

                    if chamfer_dists:
                        logging.debug(f"Chamfer distance mean: {sum(chamfer_dists)/len(chamfer_dists)} from {chamfer_dists}.")            
                        summary_writer.add_scalar("Mean Chamfer Dist/train", sum(chamfer_dists)/len(chamfer_dists), epoch)
                        fig, percentiles = plotting.plot_dist_violin(np.concatenate(chamfer_dists_all, axis=0))
                        summary_writer.add_figure("CD Percentiles/train dists", fig, global_step=epoch)
                        for p in [75, 90, 99]:
                            if p in percentiles:
                                summary_writer.add_scalar(f"CD Percentiles/train {p}th", percentiles[p], global_step=epoch)
                    summary_writer.add_scalar(
                        "Time/train eval per shape (sec)",
                        (time.time() - eval_train_time_start) / max(1, len(eval_train_scene_idxs)),
                        epoch,
                    )
                    # End of eval train.
                
                if epoch % eval_test_frequency == 0:
                    logging.info(f"Test Evaluation Started...")
                    # Longitudinal test-set evaluation:
                    # infer one subject anchor from early scans, then forecast later timepoints.
                    eval_test_time_start = time.time()
                    test_err_sum = 0.0
                    fitted_subject_count = 0
                    predicted_scan_count = 0
                    chamfer_dists = []
                    chamfer_dists_all = []
                    test_loss_hists = []
                    mesh_label_names = []
                    test_latents = []
                    subject_loss_labels = []
                    for sid in eval_test_subject_ids:
                        subject_scan_indices = test_subject_to_scan_indices[sid]
                        subject_scan_indices = sorted(
                            subject_scan_indices,
                            key=lambda idx: float(
                                test_longitudinal_meta["timepoints_raw"][idx].item()
                            ),
                        )
                        if len(subject_scan_indices) < 2:
                            logging.warning(
                                "Skipping test subject %s: need at least 2 scans for longitudinal eval.",
                                sid,
                            )
                            continue

                        observed_count = min(
                            eval_test_observed_timepoints, len(subject_scan_indices) - 1
                        )
                        observed_indices = subject_scan_indices[:observed_count]
                        target_indices = subject_scan_indices[observed_count:]

                        observations = []
                        for scan_idx in observed_indices:
                            test_fname = eval_test_filenames[scan_idx]
                            test_fpath = _resolve_existing_file(test_fname)
                            if not os.path.isfile(test_fpath):
                                raise FileNotFoundError(
                                    f"Missing test SDF file: {test_fpath}"
                                )
                            test_sdf_samples = deep_sdf.data.read_sdf_samples_into_ram(
                                test_fpath
                            )
                            test_sdf_samples[0] = test_sdf_samples[0][
                                torch.randperm(test_sdf_samples[0].shape[0])
                            ]
                            test_sdf_samples[1] = test_sdf_samples[1][
                                torch.randperm(test_sdf_samples[1].shape[0])
                            ]
                            t_obs = float(
                                test_longitudinal_meta["timepoints"][scan_idx].item()
                            )
                            obs_item = {"samples": test_sdf_samples, "time": t_obs}
                            if use_age_conditioning:
                                obs_item["age_cond"] = (
                                    test_scan_age_condition[scan_idx].detach().cpu()
                                )
                            observations.append(obs_item)

                        start = time.time()
                        subject_anchor, anchor_loss_hist = (
                            optimize_subject_anchor_from_observations(
                                decoder,
                                temporal_flow,
                                latent_size,
                                observations,
                                clamp_dist,
                                num_iterations=int(eval_test_optimization_steps),
                                num_samples=int(eval_test_anchor_num_samples),
                                lr=float(eval_test_anchor_lr),
                                init_std=float(eval_test_anchor_init_std),
                                code_reg_lambda=float(eval_test_anchor_code_reg_lambda),
                                code_bound=code_bound,
                                use_pair_forward_consistency=use_test_pair_consistency,
                                pair_forward_lambda=float(test_pair_forward_lambda),
                                pair_forward_pairs_per_iter=int(
                                    test_pair_forward_pairs_per_iter
                                ),
                                use_pair_backward_consistency=use_test_pair_consistency,
                                pair_backward_lambda=float(test_pair_backward_lambda),
                                pair_backward_pairs_per_iter=int(
                                    test_pair_backward_pairs_per_iter
                                ),
                                use_general_cocycle_consistency=use_test_general_cocycle_consistency,
                                general_cocycle_lambda=float(test_general_cocycle_lambda),
                                general_cocycle_triplets_per_iter=int(
                                    test_general_cocycle_triplets_per_iter
                                ),
                                use_age_conditioning=use_age_conditioning,
                            )
                        )
                        logging.debug(
                            "[Test eval] Subject %s anchor optimization time: %s",
                            sid,
                            time.time() - start,
                        )
                        if len(anchor_loss_hist) > 0 and not np.isnan(anchor_loss_hist[-1]):
                            test_err_sum += anchor_loss_hist[-1]
                            fitted_subject_count += 1
                        test_loss_hists.append(anchor_loss_hist)
                        subject_loss_labels.append(f"sid-{sid}")
                        target_indices = sorted(
                            target_indices,
                            key=lambda idx: float(
                                test_longitudinal_meta["timepoints_raw"][idx].item()
                            ),
                        )
                        subject_baseline_t = float(
                            test_longitudinal_meta["timepoints"][
                                subject_scan_indices[0]
                            ].item()
                        )
                        rollout_z_prev = None
                        rollout_t_prev = None
                        if eval_test_rollout_mode == "composed":
                            if (
                                eval_test_rollout_start == "last_observed"
                                and len(observed_indices) > 0
                            ):
                                t_roll_start = float(
                                    test_longitudinal_meta["timepoints"][
                                        observed_indices[-1]
                                    ].item()
                                )
                                roll_start_age = None
                                if use_age_conditioning:
                                    roll_start_age = test_scan_age_condition[
                                        observed_indices[-1]
                                    ].view(1, -1)
                                rollout_z_prev = _apply_temporal_flow_interval(
                                    temporal_flow,
                                    subject_anchor,
                                    t_start=subject_baseline_t,
                                    t_end=t_roll_start,
                                    max_dt=eval_test_max_rollout_dt,
                                    age_end_cond=roll_start_age,
                                )
                                rollout_t_prev = t_roll_start
                            else:
                                rollout_z_prev = subject_anchor
                                rollout_t_prev = subject_baseline_t

                        for scan_idx in target_indices:
                            test_fname = eval_test_filenames[scan_idx]
                            save_name = os.path.basename(test_fname).split(".npz")[0]
                            mesh_label_names.append(save_name)
                            path = os.path.join(
                                experiment_directory,
                                ws.tb_logs_dir,
                                ws.tb_logs_test_reconstructions,
                                save_name,
                            )
                            if not os.path.exists(path):
                                os.makedirs(path)

                            t_target = float(
                                test_longitudinal_meta["timepoints"][scan_idx].item()
                            )
                            age_target = None
                            if use_age_conditioning:
                                age_target = test_scan_age_condition[scan_idx].view(1, -1)
                            if eval_test_rollout_mode == "composed":
                                if rollout_t_prev is None or rollout_z_prev is None:
                                    rollout_z_prev = subject_anchor
                                    rollout_t_prev = subject_baseline_t
                                if t_target < float(rollout_t_prev) - 1e-8:
                                    # Fallback for unexpected ordering.
                                    z_target = _apply_temporal_flow_interval(
                                        temporal_flow,
                                        subject_anchor,
                                        t_start=subject_baseline_t,
                                        t_end=t_target,
                                        max_dt=eval_test_max_rollout_dt,
                                        age_end_cond=age_target,
                                    )
                                else:
                                    z_target = _apply_temporal_flow_interval(
                                        temporal_flow,
                                        rollout_z_prev,
                                        t_start=float(rollout_t_prev),
                                        t_end=t_target,
                                        max_dt=eval_test_max_rollout_dt,
                                        age_end_cond=age_target,
                                    )
                                    rollout_z_prev = z_target.detach()
                                    rollout_t_prev = t_target
                            else:
                                z_target = _apply_temporal_flow_interval(
                                    temporal_flow,
                                    subject_anchor,
                                    t_start=subject_baseline_t,
                                    t_end=t_target,
                                    max_dt=eval_test_max_rollout_dt,
                                    age_end_cond=age_target,
                                )
                            test_latents.append(z_target.detach())

                            start = time.time()
                            with torch.no_grad():
                                test_mesh = mesh.create_mesh(
                                    decoder,
                                    z_target,
                                    N=eval_grid_res,
                                    max_batch=int(2 ** 18),
                                    filename=os.path.join(path, f"epoch={epoch}"),
                                    return_trimesh=True,
                                )
                            logging.debug(
                                "[Test eval] Total time to create test mesh: %s",
                                time.time() - start,
                            )

                            if test_mesh is not None:
                                gt_mesh_path = f"{torus_path}/{save_name}.obj"
                                cd, cd_all = metrics.compute_metric(
                                    gt_mesh=gt_mesh_path,
                                    gen_mesh=test_mesh,
                                    metric=eval_chamfer_metric,
                                    **eval_chamfer_metric_kwargs,
                                )
                                chamfer_dists.append(cd)
                                chamfer_dists_all.append(cd_all)
                                predicted_scan_count += 1

                    if chamfer_dists:
                        logging.debug(f"Test Chamfer distance mean: {sum(chamfer_dists)/len(chamfer_dists)} from {chamfer_dists}.")            
                        summary_writer.add_scalar("Mean Chamfer Dist/test", sum(chamfer_dists)/len(chamfer_dists), epoch)
                        if fitted_subject_count > 0:
                            summary_writer.add_scalar(
                                "Loss/test",
                                test_err_sum / fitted_subject_count,
                                epoch,
                            )
                        if len(test_latents) > 0:
                            mlm = torch.mean(
                                torch.norm(torch.cat(test_latents, dim=0), dim=1)
                            )
                            summary_writer.add_scalar(
                                "Mean Latent Magnitude/test", mlm, global_step=epoch
                            )
                        if len(test_loss_hists) > 0:
                            fig = plotting.plot_train_stats(
                                loss_hists=test_loss_hists, labels=subject_loss_labels
                            )
                            summary_writer.add_figure(
                                "Loss/test optimization curves", fig, epoch
                            )
                        fig, percentiles = plotting.plot_dist_violin(np.concatenate(chamfer_dists_all, axis=0))
                        summary_writer.add_figure("CD Percentiles/test dists", fig, global_step=epoch)
                        for p in [75, 90, 99]:
                            if p in percentiles:
                                summary_writer.add_scalar(f"CD Percentiles/test {p}th", percentiles[p], global_step=epoch)
                    summary_writer.add_scalar(
                        "Time/test eval per shape (sec)",
                        (time.time() - eval_test_time_start) / max(1, predicted_scan_count),
                        epoch,
                    )
                    # End of eval test.

            summary_writer.add_scalar("Time/epoch (min)", (time.time()-epoch_time_start)/60, epoch)
            summary_writer.flush() 
               
            # End of epoch.
    except KeyboardInterrupt as e:
        logging.error(f"Received KeyboardInterrupt. Cleaning up and ending training.")
    finally:
        exception_in_flight = sys.exc_info()[0] is not None
        # Calculate model size.
        param_size = 0
        param_cnt = 0
        for param in decoder.parameters():
            param_size += param.nelement() * param.element_size()
            param_cnt += param.nelement()
        for param in temporal_flow.parameters():
            param_size += param.nelement() * param.element_size()
            param_cnt += param.nelement()
        buffer_size = 0
        for buffer in decoder.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        for buffer in temporal_flow.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        model_size_mb = (param_size + buffer_size) / 1024**2
        
        # Log hparams and graph to TensorBoard.
        writer_hparams = {
            **{k: v if isinstance(v, (int, float, str, bool)) else str(v) for k, v in specs.items() if not isinstance(v, dict)},
            # Add the NetworkSpecs dict.
            **{k: v if not isinstance(v, list) else str(v) for k, v in specs["NetworkSpecs"].items()},
            # Add the LR schedule dicts.                                                           
            **{f"net_lr_schedule.{k}": v for k, v in specs["LearningRateSchedule"][0].items()},
            **{f"lat_lr_schedule.{k}": v for k, v in specs["LearningRateSchedule"][1].items()},
            # Final LR values.
            "last_net_lr": optimizer_all.param_groups[0]["lr"],
            "last_lat_lr": optimizer_all.param_groups[1]["lr"],
            # Storage values in MB.
            "model_size_mb": model_size_mb,
            "model_param_cnt": param_cnt,
            "single_latent_size_mb": sum(p.nelement()*p.element_size() for p in lat_vecs.parameters()),
            # "NumEpochs": specs["NumEpochs"],
            # "CodeLength": specs["CodeLength"],
            # "CodeRegularization": str(do_code_regularization),
            # "CodeRegularizationLambda": code_reg_lambda,
        }
        train_results = {
            "best_train_loss" : min(loss_log) if len(loss_log) else -1,
            "best_train_cd" : min(train_chamfer_dists_log) if len(train_chamfer_dists_log) else -1,
            "best_test_cd" : min(test_chamfer_dists_log) if len(test_chamfer_dists_log) else -1,
        }
        summary_writer.add_hparams(writer_hparams, train_results, run_name='.')
        write_model_graph = bool(get_spec_with_default(specs, "WriteModelGraph", False))
        if write_model_graph and not exception_in_flight and "input" in locals():
            try:
                summary_writer.add_graph(decoder, input)
            except Exception as e:
                logging.warning("Skipping add_graph due to error: %s", str(e))
        elif write_model_graph and exception_in_flight:
            logging.warning("Skipping add_graph because an exception is already in flight.")
        summary_writer.flush()    
        summary_writer.close()
        # End of training.

if __name__ == "__main__":

    #python train_deep_sdf.py -e examples/torus_bump_rotate

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        type=int,
        default=None,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )
    arg_parser.add_argument(
        "--gpu",
        dest="gpu",
        type=int,
        default=None,
        help="CUDA device index to use for this run (e.g., 0 or 1). "
        + "If omitted, uses specs.json GpuId when present, else PyTorch default.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()
    runtime_specs = {}

    specs_path = os.path.join(args.experiment_directory, "specs.json")
    if os.path.isfile(specs_path):
        with open(specs_path, "r") as spec_f:
            runtime_specs = json.load(spec_f)

        # Allow logging level control from specs.json when CLI flags are omitted.
        # CLI --debug/--quiet still take precedence.
        if not args.debug and not args.quiet:
            spec_debug = bool(runtime_specs.get("DebugLogging", False))
            spec_quiet = bool(runtime_specs.get("QuietLogging", False))
            if spec_debug:
                args.debug = True
            elif spec_quiet:
                args.quiet = True

    if args.batch_split is None:
        args.batch_split = int(runtime_specs.get("BatchSplit", 1))
    if int(args.batch_split) <= 0:
        raise RuntimeError(f"batch_split must be >= 1, got {args.batch_split}")

    if args.gpu is None and "GpuId" in runtime_specs:
        args.gpu = int(runtime_specs["GpuId"])
    if args.gpu is not None and int(args.gpu) < 0:
        raise RuntimeError(f"gpu must be >= 0, got {args.gpu}")

    if args.logfile is None:
        args.logfile = os.path.join(args.experiment_directory, "train.log")
    terminal_log_path = os.path.join(args.experiment_directory, "terminal.log")

    terminal_log_file = open(terminal_log_path, "a", buffering=1)
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    sys.stdout = TeeStream(stdout_orig, terminal_log_file)
    sys.stderr = TeeStream(stderr_orig, terminal_log_file)
    try:
        deep_sdf.configure_logging(args)
        main_function(
            args.experiment_directory,
            args.continue_from,
            int(args.batch_split),
            gpu=None if args.gpu is None else int(args.gpu),
        )
    finally:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        terminal_log_file.close()
