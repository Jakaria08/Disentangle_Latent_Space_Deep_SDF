#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import os
import json
import time
import logging
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import tree

import deep_sdf
from deep_sdf import lr_scheduling, loss as deep_sdf_loss, mesh, metrics
from deep_sdf import loss_subset as deep_sdf_loss_subset
import deep_sdf.workspace as ws
from sdf_utils import sap as sap_metric
from sdf_utils import sap_latent_subset as sap_subset
from sdf_utils import dci_latent_subset as dci_subset
from sdf_utils import mig_latent_subset as mig_subset

from networks import residual_mlp_vae, pointnet_vae
import reconstruct


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _get_module_state(module):
    if isinstance(module, torch.nn.DataParallel):
        return module.module.state_dict()
    return module.state_dict()


def _load_module_state(module, state_dict):
    module.load_state_dict(_strip_module_prefix(state_dict))


def _get_vae_decoder(vae_module):
    if isinstance(vae_module, torch.nn.DataParallel):
        return vae_module.module.decoder
    return vae_module.decoder


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def resolve_spec_path(experiment_directory, spec_path):
    if spec_path is None:
        return None
    if os.path.isabs(spec_path):
        return spec_path
    return os.path.join(experiment_directory, spec_path)


def save_model(experiment_directory, filename, vae, sdf_decoder, epoch):
    model_params_dir = ws.get_model_params_dir(experiment_directory, True)
    torch.save(
        {
            "epoch": epoch,
            "vae_state_dict": _get_module_state(vae),
            "sdf_decoder_state_dict": _get_module_state(sdf_decoder),
        },
        os.path.join(model_params_dir, filename),
    )


def load_model(experiment_directory, filename, vae, sdf_decoder):
    full_filename = os.path.join(ws.get_model_params_dir(experiment_directory), filename)
    if not os.path.isfile(full_filename):
        raise Exception('model state dict "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename, map_location="cpu")
    if "vae_state_dict" not in data or "sdf_decoder_state_dict" not in data:
        raise Exception("model file is missing VAE or SDF decoder state")

    _load_module_state(vae, data["vae_state_dict"])
    _load_module_state(sdf_decoder, data["sdf_decoder_state_dict"])

    return data["epoch"]


def save_optimizer(experiment_directory, filename, optimizer, epoch):
    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)
    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):
    full_filename = os.path.join(ws.get_optimizer_params_dir(experiment_directory), filename)
    if not os.path.isfile(full_filename):
        raise Exception('optimizer state dict "{}" does not exist'.format(full_filename))
    data = torch.load(full_filename, map_location="cpu")
    optimizer.load_state_dict(data["optimizer_state_dict"])
    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_codes, epoch):
    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)
    latent_codes = latent_codes.detach().cpu()
    embedding = torch.nn.Embedding(latent_codes.shape[0], latent_codes.shape[1])
    embedding.weight.data.copy_(latent_codes)
    torch.save(
        {"epoch": epoch, "latent_codes": embedding.state_dict()},
        os.path.join(latent_codes_dir, filename),
    )


def save_logs(
    experiment_directory,
    loss_log,
    loss_log_epoch,
    sdf_loss_log_epoch,
    sdf_reg_log_epoch,
    vae_recon_log_epoch,
    vae_kl_log_epoch,
    vae_lat_mag_log,
    snnl_log_epoch,
    snnl_age_log_epoch,
    attr_log_epoch,
    cov_log_epoch,
    corr_leak_log_epoch,
    cross_cov_log_epoch,
    rank_log_epoch,
    matchstd_log_epoch,
    matchstd_std0_log_epoch,
    matchstd_stdref_log_epoch,
    sens_log_epoch,
    sens_delta_log_epoch,
    lr_log,
    timing_log,
    epoch,
):
    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "loss_epoch": loss_log_epoch,
            "sdf_loss_epoch": sdf_loss_log_epoch,
            "sdf_reg_epoch": sdf_reg_log_epoch,
            "vae_recon_epoch": vae_recon_log_epoch,
            "vae_kl_epoch": vae_kl_log_epoch,
            "vae_latent_magnitude": vae_lat_mag_log,
            "snnl_epoch": snnl_log_epoch,
            "snnl_age_epoch": snnl_age_log_epoch,
            "attr_epoch": attr_log_epoch,
            "cov_epoch": cov_log_epoch,
            "corr_leak_epoch": corr_leak_log_epoch,
            "cross_cov_epoch": cross_cov_log_epoch,
            "rank_epoch": rank_log_epoch,
            "matchstd_epoch": matchstd_log_epoch,
            "matchstd_std0_epoch": matchstd_std0_log_epoch,
            "matchstd_stdref_epoch": matchstd_stdref_log_epoch,
            "sens_epoch": sens_log_epoch,
            "sens_delta_epoch": sens_delta_log_epoch,
            "learning_rate": lr_log,
            "timing": timing_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):
    full_filename = os.path.join(experiment_directory, ws.logs_filename)
    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename, map_location="cpu")
    return (
        data["loss"],
        data["loss_epoch"],
        data["sdf_loss_epoch"],
        data["sdf_reg_epoch"],
        data["vae_recon_epoch"],
        data["vae_kl_epoch"],
        data["vae_latent_magnitude"],
        data.get("snnl_epoch", []),
        data.get("snnl_age_epoch", []),
        data.get("attr_epoch", []),
        data.get("cov_epoch", []),
        data.get("corr_leak_epoch", []),
        data.get("cross_cov_epoch", []),
        data.get("rank_epoch", []),
        data.get("matchstd_epoch", []),
        data.get("matchstd_std0_epoch", []),
        data.get("matchstd_stdref_epoch", []),
        data.get("sens_epoch", []),
        data.get("sens_delta_epoch", []),
        data["learning_rate"],
        data["timing"],
        data["epoch"],
    )


def clip_logs(
    loss_log,
    loss_log_epoch,
    sdf_loss_log_epoch,
    sdf_reg_log_epoch,
    vae_recon_log_epoch,
    vae_kl_log_epoch,
    vae_lat_mag_log,
    snnl_log_epoch,
    snnl_age_log_epoch,
    attr_log_epoch,
    cov_log_epoch,
    corr_leak_log_epoch,
    cross_cov_log_epoch,
    rank_log_epoch,
    matchstd_log_epoch,
    matchstd_std0_log_epoch,
    matchstd_stdref_log_epoch,
    sens_log_epoch,
    sens_delta_log_epoch,
    lr_log,
    timing_log,
    epoch,
):
    if len(loss_log_epoch) > 0:
        iters_per_epoch = len(loss_log) // len(loss_log_epoch)
        loss_log = loss_log[: (iters_per_epoch * epoch)]
    loss_log_epoch = loss_log_epoch[:epoch]
    sdf_loss_log_epoch = sdf_loss_log_epoch[:epoch]
    sdf_reg_log_epoch = sdf_reg_log_epoch[:epoch]
    vae_recon_log_epoch = vae_recon_log_epoch[:epoch]
    vae_kl_log_epoch = vae_kl_log_epoch[:epoch]
    vae_lat_mag_log = vae_lat_mag_log[:epoch]
    snnl_log_epoch = snnl_log_epoch[:epoch]
    snnl_age_log_epoch = snnl_age_log_epoch[:epoch]
    attr_log_epoch = attr_log_epoch[:epoch]
    cov_log_epoch = cov_log_epoch[:epoch]
    corr_leak_log_epoch = corr_leak_log_epoch[:epoch]
    cross_cov_log_epoch = cross_cov_log_epoch[:epoch]
    rank_log_epoch = rank_log_epoch[:epoch]
    matchstd_log_epoch = matchstd_log_epoch[:epoch]
    matchstd_std0_log_epoch = matchstd_std0_log_epoch[:epoch]
    matchstd_stdref_log_epoch = matchstd_stdref_log_epoch[:epoch]
    sens_log_epoch = sens_log_epoch[:epoch]
    sens_delta_log_epoch = sens_delta_log_epoch[:epoch]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]

    return (
        loss_log,
        loss_log_epoch,
        sdf_loss_log_epoch,
        sdf_reg_log_epoch,
        vae_recon_log_epoch,
        vae_kl_log_epoch,
        vae_lat_mag_log,
        snnl_log_epoch,
        snnl_age_log_epoch,
        attr_log_epoch,
        cov_log_epoch,
        corr_leak_log_epoch,
        cross_cov_log_epoch,
        rank_log_epoch,
        matchstd_log_epoch,
        matchstd_std0_log_epoch,
        matchstd_stdref_log_epoch,
        sens_log_epoch,
        sens_delta_log_epoch,
        lr_log,
        timing_log,
    )


def load_latent_codes_from_file(latent_path):
    if not os.path.isfile(latent_path):
        raise Exception('latent state file "{}" does not exist'.format(latent_path))

    data = torch.load(latent_path, map_location="cpu")
    latent_data = data["latent_codes"] if isinstance(data, dict) and "latent_codes" in data else data

    if isinstance(latent_data, torch.Tensor):
        if latent_data.dim() == 3 and latent_data.size(1) == 1:
            latent_data = latent_data[:, 0, :]
        elif latent_data.dim() != 2:
            raise Exception("latent tensor has unexpected shape")
        return latent_data

    if isinstance(latent_data, dict):
        if "weight" in latent_data:
            return latent_data["weight"]
        # Accept dicts that map name -> latent tensor (e.g., test_latents.pt)
        if all(isinstance(v, torch.Tensor) for v in latent_data.values()):
            return latent_data
        raise Exception("latent state dict missing weight")

    raise Exception("unrecognized latent code format")


def load_sdf_decoder_weights(model_path, sdf_decoder):
    if model_path is None:
        return
    if not os.path.isfile(model_path):
        raise Exception('SDF decoder model file "{}" does not exist'.format(model_path))

    data = torch.load(model_path, map_location="cpu")
    if isinstance(data, dict):
        if "sdf_decoder_state_dict" in data:
            state = data["sdf_decoder_state_dict"]
        else:
            state = data.get("model_state_dict", data.get("state_dict", data))
    else:
        state = data
    state = _strip_module_prefix(state)
    sdf_decoder.load_state_dict(state)


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def compute_vae_latents(vae, surface_points, batch_size, device):
    was_training = vae.training
    vae.eval()
    latent_chunks = []
    with torch.no_grad():
        total = surface_points.shape[0] if isinstance(surface_points, torch.Tensor) else len(surface_points)
        for start in range(0, total, batch_size):
            if isinstance(surface_points, torch.Tensor):
                chunk = surface_points[start : start + batch_size].to(device)
            else:
                chunk_np = np.stack(surface_points[start : start + batch_size], axis=0)
                chunk = torch.as_tensor(chunk_np).to(device)
            out = vae(chunk)
            latent_chunks.append(out["mu"].detach().cpu())
    if was_training:
        vae.train()
    return torch.cat(latent_chunks, dim=0)


def reconstruct_latents_for_dataset(
    dataset,
    sdf_decoder,
    data_source,
    latent_size,
    clamp_dist,
    num_samples,
    num_iterations,
    lr,
    l2reg,
    init_std,
    scene_indices=None,
):
    if dataset is None:
        return None, float("nan")

    decoder_was_training = sdf_decoder.training
    sdf_decoder.eval()

    latents = torch.full((len(dataset), latent_size), float("nan"))
    losses = []
    indices = (
        scene_indices if scene_indices is not None else range(len(dataset))
    )
    for scene_idx in indices:
        npy_path = dataset.npyfiles[scene_idx]
        sdf_path = os.path.join(data_source, npy_path)
        if not os.path.isfile(sdf_path):
            logging.warning("Missing SDF file for test latent reconstruction: %s", sdf_path)
            continue
        sdf_samples = deep_sdf.data.read_sdf_samples_into_ram(sdf_path)
        if isinstance(sdf_samples, (list, tuple)) and len(sdf_samples) >= 2:
            sdf_samples[0] = sdf_samples[0][torch.randperm(sdf_samples[0].shape[0])]
            sdf_samples[1] = sdf_samples[1][torch.randperm(sdf_samples[1].shape[0])]
        loss_hist, latent = reconstruct.reconstruct(
            sdf_decoder,
            int(num_iterations),
            latent_size,
            sdf_samples,
            init_std,
            clamp_dist,
            num_samples=int(num_samples),
            lr=lr,
            l2reg=l2reg,
            return_loss_hist=True,
        )
        if loss_hist:
            losses.append(loss_hist[-1])
        latents[scene_idx] = latent.detach().cpu()

    if decoder_was_training:
        sdf_decoder.train()

    if torch.isnan(latents).all():
        return None, float("nan")

    valid_losses = [loss for loss in losses if not np.isnan(loss)]
    mean_loss = float(np.mean(valid_losses)) if valid_losses else float("nan")
    return latents, mean_loss


def _unpack_batch(batch):
    if len(batch) == 3:
        sdf_data, indices, labels = batch
        surface_points = None
    elif len(batch) == 4:
        sdf_data, indices, labels, surface_points = batch
    elif len(batch) == 2:
        sdf_data, indices = batch
        labels = None
        surface_points = None
    else:
        raise ValueError("Unexpected batch structure from DataLoader")
    return sdf_data, indices, labels, surface_points


def _resolve_labels_path(data_source, labels_file):
    if labels_file is None:
        return None
    if os.path.isabs(labels_file):
        return labels_file
    return os.path.join(data_source, labels_file)


def _load_label_map(labels_path, npyfiles):
    if labels_path is None:
        return None
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"labels file not found: {labels_path}")
    labels = torch.load(labels_path, map_location="cpu")
    if isinstance(labels, dict):
        return labels
    if hasattr(labels, "__len__") and len(labels) == len(npyfiles):
        label_map = {}
        for idx, npy_path in enumerate(npyfiles):
            base_name = os.path.splitext(os.path.basename(npy_path))[0]
            label_map[base_name] = labels[idx]
        return label_map
    logging.warning("labels are not a dict and length does not match filenames.")
    return {}


def _labels_for_indices(npyfiles, label_map, indices):
    if label_map is None:
        return None
    labels = []
    label_len = None
    for idx in indices.tolist():
        base_name = os.path.splitext(os.path.basename(npyfiles[idx]))[0]
        label = label_map.get(base_name) if isinstance(label_map, dict) else None
        if label is None:
            labels.append(None)
            continue
        label_t = torch.as_tensor(label).view(-1)
        if label_len is None:
            label_len = label_t.numel()
        elif label_t.numel() != label_len:
            raise Exception("Label length mismatch across samples.")
        labels.append(label_t)
    if label_len is None:
        return None
    filled = []
    for label in labels:
        if label is None:
            filled.append(torch.full((label_len,), float("nan")))
        else:
            filled.append(label)
    return torch.stack(filled, dim=0)


def _summarize_labels(npyfiles, label_map, label_index):
    if label_map is None:
        return {
            "total": len(npyfiles),
            "missing": len(npyfiles),
            "valid": 0,
            "unique": {},
        }
    missing = 0
    values = []
    for npy_path in npyfiles:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        label = label_map.get(base_name) if isinstance(label_map, dict) else None
        if label is None:
            missing += 1
            continue
        label_t = torch.as_tensor(label).view(-1)
        if label_t.numel() <= label_index:
            missing += 1
            continue
        values.append(float(label_t[label_index].item()))
    if not values:
        return {
            "total": len(npyfiles),
            "missing": missing,
            "valid": 0,
            "unique": {},
        }
    vals = np.array(values, dtype=float)
    valid_mask = np.isfinite(vals) & (vals != -1)
    vals = vals[valid_mask]
    uniques, counts = np.unique(vals, return_counts=True)
    return {
        "total": len(npyfiles),
        "missing": missing,
        "valid": int(valid_mask.sum()),
        "unique": {float(u): int(c) for u, c in zip(uniques, counts)},
        "min": float(vals.min()) if vals.size else float("nan"),
        "max": float(vals.max()) if vals.size else float("nan"),
    }


def _collect_label_values(npyfiles, label_map, label_index):
    if label_map is None:
        return None
    values = []
    for npy_path in npyfiles:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        label = label_map.get(base_name) if isinstance(label_map, dict) else None
        if label is None:
            values.append(float("nan"))
            continue
        label_t = torch.as_tensor(label).view(-1)
        if label_index >= label_t.numel():
            values.append(float("nan"))
        else:
            values.append(float(label_t[label_index].item()))
    return np.asarray(values, dtype=float)


def _best_threshold_accuracy(values, labels):
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    y = labels[order]
    n = y.size
    pos = (y == 1).astype(np.int64)
    neg = (y == 0).astype(np.int64)
    prefix_pos = np.cumsum(pos)
    prefix_neg = np.cumsum(neg)
    total_pos = prefix_pos[-1]
    total_neg = prefix_neg[-1]
    acc_left0 = (prefix_neg + (total_pos - prefix_pos)) / float(n)
    acc_left1 = (prefix_pos + (total_neg - prefix_neg)) / float(n)
    return float(max(acc_left0.max(), acc_left1.max()))


def main_function(experiment_directory: str, continue_from, batch_split: int):

    logging.debug("running experiment " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + str(specs.get("Description", "(none)")))

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    test_split_file = get_spec_with_default(specs, "TestSplit", None)

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    latent_codes_path = get_spec_with_default(specs, "PretrainedLatentPath", None)
    if latent_codes_path is None:
        latent_codes_path = get_spec_with_default(specs, "LatentCodesPath", None)
    latent_codes_path = resolve_spec_path(experiment_directory, latent_codes_path)
    if latent_codes_path is None:
        raise Exception("PretrainedLatentPath or LatentCodesPath must be set in specs")

    teacher_latents = load_latent_codes_from_file(latent_codes_path)
    teacher_latents = teacher_latents.float()

    code_length = get_spec_with_default(specs, "CodeLength", teacher_latents.shape[1])
    if code_length != teacher_latents.shape[1]:
        raise Exception(
            "CodeLength does not match pretrained latent dimensionality: {} vs {}".format(
                code_length, teacher_latents.shape[1]
            )
        )

    latent_size = code_length

    sdf_decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    pretrained_sdf_path = get_spec_with_default(specs, "PretrainedSDFDecoderPath", None)
    if pretrained_sdf_path is None:
        pretrained_sdf_path = get_spec_with_default(specs, "PretrainedDecoderPath", None)
    pretrained_sdf_path = resolve_spec_path(experiment_directory, pretrained_sdf_path)
    if pretrained_sdf_path is not None:
        logging.info("Loading pretrained SDF decoder from: {}".format(pretrained_sdf_path))
        load_sdf_decoder_weights(pretrained_sdf_path, sdf_decoder)

    train_sdf_decoder = get_spec_with_default(specs, "TrainSDFDecoder", False)
    set_requires_grad(sdf_decoder, train_sdf_decoder)

    vae_input_dim = get_spec_with_default(specs, "VAEInputDim", latent_size)
    if vae_input_dim != latent_size:
        raise Exception("VAEInputDim must match pretrained latent size")

    vae_latent_dim = get_spec_with_default(specs, "VAELatentDim", 16)
    vae_encoder_dims = get_spec_with_default(specs, "VAEEncoderHiddenDims", [256, 128])
    vae_decoder_dims = get_spec_with_default(specs, "VAEDecoderHiddenDims", [128, 256, 256])
    vae_blocks = get_spec_with_default(specs, "VAEBlocks", 1)
    vae_activation = get_spec_with_default(specs, "VAEActivation", "gelu")
    vae_dropout = get_spec_with_default(specs, "VAEDropout", 0.0)
    vae_layernorm = get_spec_with_default(specs, "VAELayerNorm", True)
    use_kl = get_spec_with_default(specs, "UseKLLoss", True)

    guided_contrastive_loss = get_spec_with_default(specs, "GuidedContrastiveLoss", False)
    attribute_loss = get_spec_with_default(specs, "AttributeLoss", False)
    label_task_type = get_spec_with_default(specs, "LabelTaskType", None)
    if label_task_type is not None:
        label_task_type = str(label_task_type).lower()
    if "SNNLType" in specs:
        snnl_type = specs["SNNLType"]
    elif label_task_type in ("classification", "class", "cls", "binary"):
        snnl_type = "cls"
    elif label_task_type in ("regression", "reg", "continuous"):
        snnl_type = "reg_exact"
    else:
        snnl_type = "reg_exact"
    snnl_temp = get_spec_with_default(specs, "SNNLTemp", 181.0)
    snnl_weight = get_spec_with_default(specs, "SNNLWeight", 0.5)
    attr_weight = get_spec_with_default(specs, "AttributeWeight", 0.5)
    covariance_loss = get_spec_with_default(specs, "CovarianceLoss", False)
    covariance_lambda = get_spec_with_default(specs, "CovarianceLossLambda", 1.0)
    label_index = get_spec_with_default(specs, "LabelIndex", 0)
    attribute_latent_index = get_spec_with_default(specs, "AttributeLatentIndex", 0)
    attribute_subset = get_spec_with_default(specs, "AttributeSubset", None)
    snnl_target_dim = get_spec_with_default(specs, "SNNLTargetDim", 0)
    snnl_reg_threshold = get_spec_with_default(specs, "SNNLRegThreshold", 0.05)
    snnl_reg_pos_mode = get_spec_with_default(specs, "SNNLRegPosMode", "threshold")
    snnl_reg_topk_frac = get_spec_with_default(specs, "SNNLRegTopkFrac", 0.1)
    snnl_reg_use_adaptive_T = get_spec_with_default(specs, "SNNLRegUseAdaptiveT", True)
    snnl_reg_normalize_z = get_spec_with_default(specs, "SNNLRegNormalizeZ", True)
    age_snnl_reg_loss = get_spec_with_default(specs, "AgeSNNLRegLoss", False)
    age_snnl_reg_weight = get_spec_with_default(specs, "AgeSNNLRegWeight", 0.5)
    age_snnl_reg_label_index = get_spec_with_default(specs, "AgeSNNLRegLabelIndex", 1)
    age_snnl_reg_target_dim = get_spec_with_default(specs, "AgeSNNLRegTargetDim", 1)
    age_snnl_reg_temp = get_spec_with_default(specs, "AgeSNNLRegTemp", snnl_temp)
    age_snnl_reg_threshold = get_spec_with_default(
        specs, "AgeSNNLRegThreshold", snnl_reg_threshold
    )
    age_snnl_reg_pos_mode = get_spec_with_default(
        specs, "AgeSNNLRegPosMode", snnl_reg_pos_mode
    )
    age_snnl_reg_topk_frac = get_spec_with_default(
        specs, "AgeSNNLRegTopkFrac", snnl_reg_topk_frac
    )
    age_snnl_reg_use_adaptive_T = get_spec_with_default(
        specs, "AgeSNNLRegUseAdaptiveT", snnl_reg_use_adaptive_T
    )
    age_snnl_reg_normalize_z = get_spec_with_default(
        specs, "AgeSNNLRegNormalizeZ", snnl_reg_normalize_z
    )
    disease_latent_count = get_spec_with_default(specs, "DiseaseLatentCount", None)
    age_latent_count = get_spec_with_default(specs, "AgeLatentCount", None)
    other_latent_count = get_spec_with_default(specs, "OtherLatentCount", None)
    disease_label_index = get_spec_with_default(specs, "DiseaseLabelIndex", label_index)
    age_label_index = get_spec_with_default(
        specs, "AgeLabelIndex", age_snnl_reg_label_index
    )
    kl_subset_weights = get_spec_with_default(specs, "KLSubsetWeights", None)
    kl_disease_weight = float(get_spec_with_default(specs, "KLDiseaseWeight", 1.0))
    kl_age_weight = float(get_spec_with_default(specs, "KLAgeWeight", 1.0))
    kl_other_weight = float(get_spec_with_default(specs, "KLOtherWeight", 1.0))
    if isinstance(kl_subset_weights, dict):
        kl_disease_weight = float(kl_subset_weights.get("disease", kl_disease_weight))
        kl_age_weight = float(kl_subset_weights.get("age", kl_age_weight))
        kl_other_weight = float(kl_subset_weights.get("other", kl_other_weight))
    corr_leakage_loss = get_spec_with_default(specs, "CorrLeakageLoss", False)
    corr_leakage_lambda = get_spec_with_default(specs, "CorrLeakageLambda", 1.0)
    cross_cov_loss = get_spec_with_default(specs, "CrossCovLoss", False)
    cross_cov_lambda = get_spec_with_default(specs, "CrossCovLambda", 1.0)
    sensitivity_loss = get_spec_with_default(specs, "SensitivityLoss", False)
    sensitivity_eps = get_spec_with_default(specs, "SensitivityEps", 0.02)
    sensitivity_eta = get_spec_with_default(specs, "SensitivityEta", 0.0025)
    sensitivity_weight = get_spec_with_default(specs, "SensitivityWeight", 0.1)
    sensitivity_target_dim = get_spec_with_default(specs, "SensitivityLatentIndex", 0)
    sensitivity_subset = get_spec_with_default(specs, "SensitivitySubset", "disease")
    rank_loss = get_spec_with_default(specs, "RankLoss", False)
    rank_margin = get_spec_with_default(specs, "RankLossMargin", 0.5)
    rank_weight = get_spec_with_default(specs, "RankLossWeight", 0.1)
    rank_target_dim = get_spec_with_default(specs, "RankLossTargetDim", 0)
    rank_cn_label = get_spec_with_default(specs, "RankLossCNLabel", 1)
    rank_subset = get_spec_with_default(specs, "RankLossSubset", "disease")
    matchstd_loss = get_spec_with_default(specs, "MatchStdLoss", False)
    matchstd_weight = get_spec_with_default(specs, "MatchStdWeight", 0.1)
    matchstd_target_dim = get_spec_with_default(specs, "MatchStdTargetDim", 0)
    matchstd_eps = get_spec_with_default(specs, "MatchStdEps", 1e-6)
    matchstd_subset = get_spec_with_default(specs, "MatchStdSubset", "disease")
    leakage_target_dim = get_spec_with_default(
        specs, "LeakageTargetDim", attribute_latent_index
    )
    label_mix_enabled = get_spec_with_default(specs, "LabelMixing", False)
    pseudo_labels_file = get_spec_with_default(specs, "PseudoLabelsFile", "pseudo_label.pt")
    real_labels_file = get_spec_with_default(specs, "RealLabelsFile", "labels.pt")
    mix_pseudo_start = get_spec_with_default(specs, "LabelMixPseudoRatioStart", 1.0)
    mix_unlabeled_start = get_spec_with_default(specs, "LabelMixUnlabeledRatioStart", 0.0)
    label_mix_stratified = get_spec_with_default(specs, "LabelMixStratified", False)
    mix_real_start = 1.0 - float(mix_pseudo_start) - float(mix_unlabeled_start)
    if mix_pseudo_start < 0.0 or mix_unlabeled_start < 0.0 or mix_real_start < 0.0:
        raise RuntimeError(
            "Invalid label mix ratios (pseudo {}, unlabeled {}, real {}).".format(
                mix_pseudo_start, mix_unlabeled_start, mix_real_start
            )
        )
    eval_test_reconstruct = get_spec_with_default(specs, "EvalTestReconstructLatents", False)
    eval_test_start_epoch = get_spec_with_default(specs, "EvalTestStartEpoch", 1)
    train_latent_holdout_frac = float(
        get_spec_with_default(specs, "TrainLatentHoldoutFraction", 0.0)
    )
    train_latent_holdout_seed = get_spec_with_default(specs, "TrainLatentHoldoutSeed", 0)

    if (
        disease_latent_count is None
        or age_latent_count is None
        or other_latent_count is None
    ):
        raise RuntimeError(
            "DiseaseLatentCount, AgeLatentCount, and OtherLatentCount must be set for latent subset training."
        )
    disease_latent_count = int(disease_latent_count)
    age_latent_count = int(age_latent_count)
    other_latent_count = int(other_latent_count)
    if disease_latent_count + age_latent_count + other_latent_count != vae_latent_dim:
        raise RuntimeError(
            "Latent subset counts must sum to VAELatentDim ({}): disease={} age={} other={}".format(
                vae_latent_dim, disease_latent_count, age_latent_count, other_latent_count
            )
        )
    disease_dims = list(range(0, disease_latent_count))
    age_dims = list(range(disease_latent_count, disease_latent_count + age_latent_count))
    other_dims = list(
        range(
            disease_latent_count + age_latent_count,
            disease_latent_count + age_latent_count + other_latent_count,
        )
    )

    def _subset_to_dims(name):
        name = str(name).lower()
        if name == "disease":
            return disease_dims
        if name == "age":
            return age_dims
        if name == "other":
            return other_dims
        raise RuntimeError(f"Unknown subset name: {name}")
    logging.info(
        "Latent subsets: disease=%s age=%s other=%s",
        disease_dims,
        age_dims,
        other_dims,
    )
    logging.info(
        "KL subset weights: disease=%.3f age=%.3f other=%.3f",
        kl_disease_weight,
        kl_age_weight,
        kl_other_weight,
    )

    compute_sap = get_spec_with_default(specs, "ComputeSAP", False)
    compute_dci = get_spec_with_default(specs, "ComputeDCI", True)
    compute_mig = get_spec_with_default(specs, "ComputeMIG", True)
    if "SAPRegression" in specs:
        sap_regression = get_spec_with_default(specs, "SAPRegression", False)
    elif label_task_type in ("classification", "class", "cls", "binary"):
        sap_regression = False
    elif label_task_type in ("regression", "reg", "continuous"):
        sap_regression = True
    else:
        sap_regression = get_spec_with_default(specs, "SAPRegression", False)
    if "SAPContinuousFactors" in specs:
        sap_continuous = get_spec_with_default(specs, "SAPContinuousFactors", True)
    elif label_task_type in ("classification", "class", "cls", "binary"):
        sap_continuous = False
    elif label_task_type in ("regression", "reg", "continuous"):
        sap_continuous = True
    else:
        sap_continuous = get_spec_with_default(specs, "SAPContinuousFactors", True)
    sap_nb_bins = get_spec_with_default(specs, "SAPNumBins", 10)
    sap_label_indices = get_spec_with_default(specs, "SAPLabelIndices", None)
    sap_corr_extra_frequency = get_spec_with_default(specs, "SAPCORRExtraFrequency", 0)
    sap_corr_labels_file = get_spec_with_default(specs, "SAPCORRLabelsFile", "labels.pt")
    compute_sap_age = get_spec_with_default(specs, "ComputeSAPAge", False)
    sap_age_label_indices = get_spec_with_default(specs, "SAPAgeLabelIndices", None)
    sap_age_regression = get_spec_with_default(specs, "SAPAgeRegression", True)
    sap_age_continuous = get_spec_with_default(
        specs, "SAPAgeContinuousFactors", True
    )
    sap_age_nb_bins = get_spec_with_default(specs, "SAPAgeNumBins", sap_nb_bins)
    sap_age_corr_labels_file = get_spec_with_default(
        specs, "SAPAgeCORRLabelsFile", sap_corr_labels_file
    )
    sap_debug_predictions = get_spec_with_default(specs, "SAPDebugPredictions", False)
    sap_debug_pred_samples = int(get_spec_with_default(specs, "SAPDebugPredSamples", 0))
    sap_kumar_holdout = get_spec_with_default(specs, "SAPKumarHoldout", False)
    sap_kumar_holdout_frac = float(get_spec_with_default(specs, "SAPKumarHoldoutFrac", 0.8))
    sap_kumar_holdout_seed = get_spec_with_default(specs, "SAPKumarHoldoutSeed", 0)
    if sap_debug_predictions:
        logging.warning(
            "Subset training: SAPDebugPredictions is per-latent; subset script ignores it."
        )
    if sap_kumar_holdout:
        logging.info("Subset training: SAPKumarHoldout enabled (subset-level holdout SAP).")



    use_labels = get_spec_with_default(specs, "ReturnLabels", None)
    if use_labels is None:
        use_labels = (
            guided_contrastive_loss
            or attribute_loss
            or corr_leakage_loss
            or rank_loss
            or age_snnl_reg_loss
            or compute_sap
            or compute_sap_age
        )
    labels_filename = get_spec_with_default(specs, "LabelsFile", "labels.pt")
    warn_missing_labels = get_spec_with_default(specs, "WarnMissingLabels", True)

    encoder_type = get_spec_with_default(specs, "EncoderType", "pointnet2")
    encoder_type_norm = str(encoder_type).lower()
    if encoder_type_norm in ("residual_mlp", "mlp", "latent", "latent_mlp"):
        vae_input_mode = "latent"
        vae = residual_mlp_vae.ResidualMLPVAE(
            input_dim=vae_input_dim,
            latent_dim=vae_latent_dim,
            encoder_hidden_dims=vae_encoder_dims,
            decoder_hidden_dims=vae_decoder_dims,
            num_blocks=vae_blocks,
            activation=vae_activation,
            dropout=vae_dropout,
            use_layernorm=vae_layernorm,
            use_kl=use_kl,
        ).cuda()
    else:
        vae_input_mode = "points"
        vae = pointnet_vae.PointNetLatentVAE(
            latent_dim=vae_latent_dim,
            output_dim=vae_input_dim,
            encoder_type=encoder_type,
            decoder_hidden_dims=vae_decoder_dims,
            decoder_blocks=vae_blocks,
            decoder_activation=vae_activation,
            decoder_dropout=vae_dropout,
            decoder_layernorm=vae_layernorm,
            use_kl=use_kl,
        ).cuda()

    if torch.cuda.device_count() > 1:
        vae = torch.nn.DataParallel(vae)
        sdf_decoder = torch.nn.DataParallel(sdf_decoder)

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))
    if sensitivity_loss:
        logging.info(
            "SensitivityLoss enabled: eps=%.6f eta=%.6f weight=%.6f target_dim=%d (debug: target Δcode >= eta)",
            float(sensitivity_eps),
            float(sensitivity_eta),
            float(sensitivity_weight),
            int(sensitivity_target_dim),
        )
    if rank_loss:
        logging.info(
            "RankLoss enabled: margin=%.6f weight=%.6f target_dim=%d cn_label=%d",
            float(rank_margin),
            float(rank_weight),
            int(rank_target_dim),
            int(rank_cn_label),
        )
    if matchstd_loss:
        logging.info(
            "MatchStdLoss enabled: weight=%.6f target_dim=%d eps=%.6f",
            float(matchstd_weight),
            int(matchstd_target_dim),
            float(matchstd_eps),
        )

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 200)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    test_split = None
    if test_split_file is not None:
        with open(test_split_file, "r") as f:
            test_split = json.load(f)

    load_ram = get_spec_with_default(specs, "LoadDatasetIntoRAM", False)
    if load_ram:
        logging.info("Loading SDF samples into memory because LoadDatasetIntoRAM=true")

    data_source_mesh = get_spec_with_default(specs, "DataSourceMesh", None)
    surface_point_count = get_spec_with_default(specs, "SurfacePointCount", 2048)
    return_surface_points = get_spec_with_default(specs, "ReturnSurfacePoints", True)
    if vae_input_mode == "points" and not return_surface_points:
        raise RuntimeError("ReturnSurfacePoints must be True for point-based encoders.")
    if vae_input_mode == "latent":
        return_surface_points = False

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source,
        train_split,
        num_samp_per_scene,
        load_ram=load_ram,
        return_labels=use_labels,
        labels_filename=labels_filename,
        data_source_mesh=data_source_mesh,
        return_surface_points=return_surface_points,
        surface_point_count=surface_point_count,
        warn_missing_labels=warn_missing_labels,
    )

    num_scenes = len(sdf_dataset)
    if teacher_latents.shape[0] != num_scenes:
        raise Exception(
            "Pretrained latent count does not match number of scenes: {} vs {}".format(
                teacher_latents.shape[0], num_scenes
            )
        )
    train_indices = list(range(num_scenes))
    holdout_indices = []
    if train_latent_holdout_frac > 0.0:
        if train_latent_holdout_frac >= 1.0:
            raise RuntimeError("TrainLatentHoldoutFraction must be < 1.0.")
        holdout_count = int(round(num_scenes * train_latent_holdout_frac))
        if holdout_count <= 0 or holdout_count >= num_scenes:
            raise RuntimeError(
                "TrainLatentHoldoutFraction yields empty train/holdout split."
            )
        rng = random.Random(train_latent_holdout_seed)
        shuffled = list(range(num_scenes))
        rng.shuffle(shuffled)
        holdout_indices = sorted(shuffled[:holdout_count])
        train_indices = sorted(shuffled[holdout_count:])
        logging.info(
            "Using train latent holdout split: train=%d holdout=%d (frac=%.3f seed=%s)",
            len(train_indices),
            len(holdout_indices),
            train_latent_holdout_frac,
            str(train_latent_holdout_seed),
        )

    test_dataset = None
    test_latents = None
    if test_split is not None:
        test_dataset = deep_sdf.data.SDFSamples(
            data_source,
            test_split,
            num_samp_per_scene,
            load_ram=load_ram,
            return_labels=use_labels,
            labels_filename=labels_filename,
            data_source_mesh=data_source_mesh,
            return_surface_points=return_surface_points,
            surface_point_count=surface_point_count,
            warn_missing_labels=warn_missing_labels,
        )
        test_latents_path = get_spec_with_default(specs, "TestLatentPath", None)
        test_latents_path = resolve_spec_path(experiment_directory, test_latents_path)
        if (
            vae_input_mode == "latent"
            and not eval_test_reconstruct
            and test_latents_path is None
        ):
            raise RuntimeError(
                "EncoderType=residual_mlp requires TestLatentPath for test eval "
                "(or set EvalTestReconstructLatents=true / disable test eval)."
            )
        if test_latents_path is None:
            if not eval_test_reconstruct:
                logging.info(
                    "TestSplit provided without TestLatentPath; test eval will run without VAE recon loss."
                )
        else:
            if eval_test_reconstruct:
                logging.info(
                    "EvalTestReconstructLatents enabled; ignoring TestLatentPath."
                )
            else:
                test_latents = load_latent_codes_from_file(test_latents_path)
                if isinstance(test_latents, dict):
                    missing = []
                    ordered = []
                    for npy_path in test_dataset.npyfiles:
                        base_name = os.path.splitext(os.path.basename(npy_path))[0]
                        if base_name not in test_latents:
                            missing.append(base_name)
                            continue
                        ordered.append(test_latents[base_name].detach().cpu())
                    if missing:
                        raise Exception(
                            "Test latent dict missing {} entries (e.g., {}).".format(
                                len(missing),
                                missing[0],
                            )
                        )
                    if not ordered:
                        raise Exception("No test latents matched test dataset.")
                    test_latents = torch.stack(ordered, dim=0)
                    if test_latents.dim() == 3 and test_latents.size(1) == 1:
                        test_latents = test_latents[:, 0, :]
                    elif test_latents.dim() == 3 and test_latents.size(2) == 1:
                        test_latents = test_latents[:, :, 0]
                test_latents = test_latents.float()
                if test_latents.shape[0] != len(test_dataset):
                    raise Exception(
                        "Test latent count does not match number of test scenes: {} vs {}".format(
                            test_latents.shape[0], len(test_dataset)
                        )
                    )

    def _select_vae_inputs(dataset, eval_latents, scene_indices=None):
        if vae_input_mode == "points":
            if dataset is None or not getattr(dataset, "surface_points", None):
                return None
            inputs = dataset.surface_points
            if scene_indices is not None:
                indices = [int(idx) for idx in scene_indices]
                inputs = [inputs[idx] for idx in indices]
            return inputs
        if eval_latents is None:
            return None
        if scene_indices is not None:
            return eval_latents[scene_indices]
        return eval_latents

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    if (
        guided_contrastive_loss
        or attribute_loss
        or corr_leakage_loss
        or age_snnl_reg_loss
        or compute_sap
        or compute_sap_age
    ) and not use_labels:
        raise Exception("Label-based losses/SAP requested but ReturnLabels is disabled.")

    sap_corr_label_map = None
    sap_age_label_map = None
    if compute_sap or (sap_corr_extra_frequency is not None and sap_corr_extra_frequency > 0):
        sapcorr_path = _resolve_labels_path(data_source, sap_corr_labels_file)
        sap_corr_label_map = _load_label_map(sapcorr_path, sdf_dataset.npyfiles)
    if compute_sap_age:
        if (
            sap_age_corr_labels_file == sap_corr_labels_file
            and sap_corr_label_map is not None
        ):
            sap_age_label_map = sap_corr_label_map
        else:
            sap_age_path = _resolve_labels_path(data_source, sap_age_corr_labels_file)
            sap_age_label_map = _load_label_map(sap_age_path, sdf_dataset.npyfiles)

    pseudo_label_map = None
    real_label_map = None
    if label_mix_enabled:
        pseudo_path = _resolve_labels_path(data_source, pseudo_labels_file)
        real_path = _resolve_labels_path(data_source, real_labels_file)
        if mix_pseudo_start > 0.0:
            pseudo_label_map = _load_label_map(pseudo_path, sdf_dataset.npyfiles)
        if mix_real_start > 0.0:
            real_label_map = _load_label_map(real_path, sdf_dataset.npyfiles)

    train_dataset = sdf_dataset
    if holdout_indices:
        train_dataset = data_utils.Subset(sdf_dataset, train_indices)

    sdf_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    eval_train_frequency = get_spec_with_default(specs, "EvalTrainFrequency", 0)
    eval_test_frequency = get_spec_with_default(specs, "EvalTestFrequency", 0)
    eval_train_scene_num = get_spec_with_default(specs, "EvalTrainSceneNumber", 0)
    eval_test_scene_num = get_spec_with_default(specs, "EvalTestSceneNumber", 0)
    eval_test_optimization_steps = get_spec_with_default(specs, "EvalTestOptimizationSteps", 1000)
    eval_test_latent_lr = get_spec_with_default(specs, "EvalTestLatentLR", 5e-3)
    eval_test_latent_l2reg = get_spec_with_default(specs, "EvalTestLatentL2Reg", True)
    eval_test_latent_init_std = get_spec_with_default(specs, "EvalTestLatentInitStd", 0.01)
    eval_test_num_samples = get_spec_with_default(specs, "EvalTestNumSamples", num_samp_per_scene)
    mesh_train_scene_num = get_spec_with_default(specs, "EvalMeshTrainSceneNumber", 10)
    mesh_test_scene_num = get_spec_with_default(specs, "EvalMeshTestSceneNumber", 10)
    eval_grid_res = get_spec_with_default(specs, "EvalGridResolution", 256)
    eval_max_batch = get_spec_with_default(specs, "EvalMaxBatch", int(2 ** 18))
    eval_gt_mesh_dir = get_spec_with_default(specs, "EvalGTMeshDir", None)
    eval_gt_mesh_dir = resolve_spec_path(experiment_directory, eval_gt_mesh_dir)
    eval_gt_mesh_ext = get_spec_with_default(specs, "EvalGTMeshExt", ".obj")
    eval_gt_mesh_samples = get_spec_with_default(specs, "EvalGTMeshSamples", 30000)

    def select_eval_indices(dataset, scene_count):
        if dataset is None:
            return []
        if scene_count is None or scene_count <= 0:
            return list(range(len(dataset)))
        count = min(scene_count, len(dataset))
        return random.sample(range(len(dataset)), count)

    def build_eval_loader(dataset, scene_count, split_name):
        if dataset is None:
            return None
        if scene_count is None or scene_count <= 0:
            scene_count = len(dataset)
        scene_count = min(scene_count, len(dataset))
        if scene_count == len(dataset):
            indices = list(range(len(dataset)))
        else:
            indices = random.sample(range(len(dataset)), scene_count)
        logging.debug("Eval {} scene indices: {}".format(split_name, indices))
        subset = data_utils.Subset(dataset, indices)
        return data_utils.DataLoader(
            subset,
            batch_size=scene_per_batch,
            shuffle=False,
            num_workers=num_data_loader_threads,
            drop_last=False,
        )

    def build_eval_loader_from_indices(dataset, indices, split_name):
        if dataset is None or indices is None or len(indices) == 0:
            return None
        logging.debug("Eval {} scene indices: {}".format(split_name, indices))
        subset = data_utils.Subset(dataset, indices)
        return data_utils.DataLoader(
            subset,
            batch_size=scene_per_batch,
            shuffle=False,
            num_workers=num_data_loader_threads,
            drop_last=False,
        )

    def select_indices_from_pool(index_pool, scene_count):
        if not index_pool:
            return []
        if scene_count is None or scene_count <= 0 or scene_count >= len(index_pool):
            return list(index_pool)
        return random.sample(index_pool, scene_count)

    def select_mesh_indices(dataset, scene_count):
        if dataset is None or scene_count is None or scene_count <= 0:
            return []
        count = min(scene_count, len(dataset))
        return random.sample(range(len(dataset)), count)

    eval_train_loader = None
    eval_train_holdout_loader = None
    train_holdout_eval_indices = None
    train_eval_indices = None
    if eval_train_frequency is not None and eval_train_frequency > 0:
        if holdout_indices:
            train_eval_indices = select_indices_from_pool(
                train_indices, eval_train_scene_num
            )
            eval_train_loader = build_eval_loader_from_indices(
                sdf_dataset, train_eval_indices, "train"
            )
            train_holdout_eval_indices = select_indices_from_pool(
                holdout_indices, eval_train_scene_num
            )
            eval_train_holdout_loader = build_eval_loader_from_indices(
                sdf_dataset, train_holdout_eval_indices, "train_holdout_eval"
            )
        else:
            eval_train_loader = build_eval_loader(
                sdf_dataset, eval_train_scene_num, "train"
            )

    eval_test_scene_idxs = select_eval_indices(test_dataset, eval_test_scene_num)
    holdout_eval_scene_idxs = select_indices_from_pool(
        holdout_indices, eval_test_scene_num
    )
    eval_test_loader = None
    if eval_test_frequency is not None and eval_test_frequency > 0:
        if test_dataset is None:
            logging.warning(
                "EvalTestFrequency set but test dataset missing; skipping test evaluation."
            )
        elif eval_test_scene_idxs:
            eval_test_loader = build_eval_loader_from_indices(
                test_dataset, eval_test_scene_idxs, "test"
            )
        else:
            logging.warning(
                "EvalTestFrequency set but no eval test indices; skipping test evaluation."
            )
    eval_holdout_loader = None

    eval_train_scene_idxs = (
        select_indices_from_pool(train_indices, mesh_train_scene_num)
        if holdout_indices
        else select_mesh_indices(sdf_dataset, mesh_train_scene_num)
    )
    mesh_test_scene_idxs = select_mesh_indices(test_dataset, mesh_test_scene_num)
    holdout_mesh_scene_idxs = select_indices_from_pool(
        holdout_indices, mesh_test_scene_num
    )

    sap_train_loader = None
    sap_test_loader = None
    if compute_sap and sap_corr_extra_frequency is not None and sap_corr_extra_frequency > 0:
        if holdout_indices:
            sap_train_loader = build_eval_loader_from_indices(
                sdf_dataset, train_indices, "train_sap"
            )
        else:
            sap_train_loader = build_eval_loader(sdf_dataset, 0, "train_sap")
        if test_dataset is not None:
            sap_test_loader = build_eval_loader(test_dataset, 0, "test_sap")

    lr_schedules = lr_scheduling.get_learning_rate_schedules(specs)

    vae_lr = lr_schedules[0].get_learning_rate(0)
    params = [{"params": vae.parameters(), "lr": vae_lr}]

    if train_sdf_decoder:
        sdf_lr_schedule = lr_schedules[1] if len(lr_schedules) > 1 else lr_schedules[0]
        params.append({"params": sdf_decoder.parameters(), "lr": sdf_lr_schedule.get_learning_rate(0)})

    optimizer = torch.optim.Adam(params)

    summary_writer = SummaryWriter(log_dir=os.path.join(experiment_directory, ws.tb_logs_dir))

    snn_loss_fn = None
    if guided_contrastive_loss:
        snnl_type_norm = str(snnl_type).lower()
        if snnl_type_norm in ("reg", "reg_fast", "regloss"):
            snn_loss_fn = deep_sdf_loss_subset.SNNRegLossExactGroup(
                T=snnl_temp,
                target_dims=disease_dims,
                threshold=snnl_reg_threshold,
                pos_mode=snnl_reg_pos_mode,
                topk_frac=snnl_reg_topk_frac,
                use_adaptive_T=snnl_reg_use_adaptive_T,
                normalize_z=snnl_reg_normalize_z,
            )
        elif snnl_type_norm in ("reg_exact", "regexact", "regloss_exact"):
            snn_loss_fn = deep_sdf_loss_subset.SNNRegLossExactGroup(
                T=snnl_temp,
                target_dims=disease_dims,
                threshold=snnl_reg_threshold,
                pos_mode=snnl_reg_pos_mode,
                topk_frac=snnl_reg_topk_frac,
                use_adaptive_T=snnl_reg_use_adaptive_T,
                normalize_z=snnl_reg_normalize_z,
            )
        elif snnl_type_norm in ("cls", "class", "classification"):
            snn_loss_fn = deep_sdf_loss_subset.SNNLossClsGroup(
                T=snnl_temp,
                target_dims=disease_dims,
                normalize_z=snnl_reg_normalize_z,
                use_adaptive_T=snnl_reg_use_adaptive_T,
            )
        else:
            raise ValueError(f"Unsupported SNNLType: {snnl_type}")
    age_snnl_reg_fn = (
        deep_sdf_loss_subset.SNNRegLossExactGroup(
            T=age_snnl_reg_temp,
            target_dims=age_dims,
            threshold=age_snnl_reg_threshold,
            pos_mode=age_snnl_reg_pos_mode,
            topk_frac=age_snnl_reg_topk_frac,
            use_adaptive_T=age_snnl_reg_use_adaptive_T,
            normalize_z=age_snnl_reg_normalize_z,
        )
        if age_snnl_reg_loss
        else None
    )
    attr_loss_fn = deep_sdf_loss.AttributeLoss() if attribute_loss else None
    sens_loss_fn = (
        deep_sdf_loss_subset.SensitivityGroupLoss(
            eps=sensitivity_eps,
            eta=sensitivity_eta,
            target_dims=_subset_to_dims(sensitivity_subset),
        )
        if sensitivity_loss
        else None
    )
    rank_loss_fn = (
        deep_sdf_loss_subset.RankLossGroup(
            margin=rank_margin,
            target_dims=_subset_to_dims(rank_subset),
            cn_label=rank_cn_label,
        )
        if rank_loss
        else None
    )
    matchstd_loss_fn = (
        deep_sdf_loss_subset.MatchStdGroup(
            target_dims=_subset_to_dims(matchstd_subset),
            eps=matchstd_eps,
        )
        if matchstd_loss
        else None
    )
    cov_loss_fn = (
        deep_sdf_loss_subset.CovarianceSubsetLoss(
            subsets={"disease": disease_dims, "age": age_dims, "other": other_dims},
            beta=covariance_lambda,
        )
        if covariance_loss
        else None
    )

    loss_log = []
    loss_log_epoch = []
    sdf_loss_log_epoch = []
    sdf_reg_log_epoch = []
    vae_recon_log_epoch = []
    vae_kl_log_epoch = []
    vae_lat_mag_log = []
    snnl_log_epoch = []
    snnl_age_log_epoch = []
    attr_log_epoch = []
    cov_log_epoch = []
    corr_leak_log_epoch = []
    cross_cov_log_epoch = []
    rank_log_epoch = []
    matchstd_log_epoch = []
    matchstd_std0_log_epoch = []
    matchstd_stdref_log_epoch = []
    sens_log_epoch = []
    sens_delta_log_epoch = []
    lr_log = []
    timing_log = []
    last_test_eval_sdf = None
    last_test_sap = None
    last_test_latent_recon = None
    last_train_eval_sdf = None
    last_train_sap = None
    last_train_eval_epoch = None
    last_test_eval_epoch = None
    last_train_cd = None
    last_test_cd = None

    start_epoch = 1

    if continue_from is not None:
        logging.info('continuing from "{}"'.format(continue_from))

        model_epoch = load_model(
            experiment_directory, continue_from + ".pth", vae, sdf_decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer
        )
        for i, lrs in enumerate(lr_schedules):
            if isinstance(lrs, lr_scheduling.StepLearningRateOnPlateauSchedule):
                lrs.last_lr = optimizer.param_groups[i]["lr"]

        (
            loss_log,
            loss_log_epoch,
            sdf_loss_log_epoch,
            sdf_reg_log_epoch,
            vae_recon_log_epoch,
            vae_kl_log_epoch,
            vae_lat_mag_log,
            snnl_log_epoch,
            snnl_age_log_epoch,
            attr_log_epoch,
            cov_log_epoch,
            corr_leak_log_epoch,
            cross_cov_log_epoch,
            rank_log_epoch,
            matchstd_log_epoch,
            matchstd_std0_log_epoch,
            matchstd_stdref_log_epoch,
            sens_log_epoch,
            sens_delta_log_epoch,
            lr_log,
            timing_log,
            log_epoch,
        ) = load_logs(experiment_directory)

        if not (model_epoch == optimizer_epoch and model_epoch == log_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, log_epoch
                )
            )

        if model_epoch < log_epoch:
            (
                loss_log,
                loss_log_epoch,
                sdf_loss_log_epoch,
                sdf_reg_log_epoch,
                vae_recon_log_epoch,
                vae_kl_log_epoch,
                vae_lat_mag_log,
                snnl_log_epoch,
                snnl_age_log_epoch,
                attr_log_epoch,
                cov_log_epoch,
                corr_leak_log_epoch,
                cross_cov_log_epoch,
                rank_log_epoch,
                matchstd_log_epoch,
                matchstd_std0_log_epoch,
                matchstd_stdref_log_epoch,
                sens_log_epoch,
                sens_delta_log_epoch,
                lr_log,
                timing_log,
            ) = clip_logs(
                loss_log,
                loss_log_epoch,
                sdf_loss_log_epoch,
                sdf_reg_log_epoch,
                vae_recon_log_epoch,
                vae_kl_log_epoch,
                vae_lat_mag_log,
                snnl_log_epoch,
                snnl_age_log_epoch,
                attr_log_epoch,
                cov_log_epoch,
                corr_leak_log_epoch,
                cross_cov_log_epoch,
                rank_log_epoch,
                matchstd_log_epoch,
                matchstd_std0_log_epoch,
                matchstd_stdref_log_epoch,
                sens_log_epoch,
                sens_delta_log_epoch,
                lr_log,
                timing_log,
                model_epoch,
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of VAE parameters: {}".format(
            sum(p.data.nelement() for p in vae.parameters())
        )
    )
    logging.info(
        "Number of SDF decoder parameters: {}".format(
            sum(p.data.nelement() for p in sdf_decoder.parameters())
        )
    )

    def adjust_learning_rate(lr_schedules, optimizer, epoch, loss_log_epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            schedule = lr_schedules[min(i, len(lr_schedules) - 1)]
            param_group["lr"] = schedule.get_learning_rate(epoch, loss_log_epoch)

    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", vae, sdf_decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer, epoch)
        latent_batch = get_spec_with_default(specs, "LatentExportBatchSize", 1024)
        device = next(vae.parameters()).device
        vae_inputs = _select_vae_inputs(sdf_dataset, teacher_latents)
        if vae_inputs is None:
            raise RuntimeError("Unable to export latents: VAE inputs are missing.")
        vae_latents = compute_vae_latents(vae, vae_inputs, latent_batch, device)
        save_latent_vectors(experiment_directory, "latest.pth", vae_latents, epoch)

    def save_checkpoints(epoch):
        filename = str(epoch) + ".pth"
        save_model(experiment_directory, filename, vae, sdf_decoder, epoch)
        save_optimizer(experiment_directory, filename, optimizer, epoch)
        latent_batch = get_spec_with_default(specs, "LatentExportBatchSize", 1024)
        device = next(vae.parameters()).device
        vae_inputs = _select_vae_inputs(sdf_dataset, teacher_latents)
        if vae_inputs is None:
            raise RuntimeError("Unable to export latents: VAE inputs are missing.")
        vae_latents = compute_vae_latents(vae, vae_inputs, latent_batch, device)
        save_latent_vectors(experiment_directory, filename, vae_latents, epoch)

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

    recon_loss_type = get_spec_with_default(specs, "VAEReconLoss", "mse")
    vae_recon_weight = get_spec_with_default(specs, "VAEReconWeight", 1.0)
    vae_kl_weight = get_spec_with_default(specs, "VAEKLWeight", 1.0)
    vae_kl_warmup_epochs = get_spec_with_default(specs, "KLWarmupEpochs", 0)
    sdf_loss_weight = get_spec_with_default(specs, "SDFLossWeight", 1.0)

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)
    code_reg_warmup_epochs = get_spec_with_default(specs, "CodeRegularizationWarmupEpochs", 100)


    def run_eval(eval_loader, eval_latents, epoch, split_label, kl_weight, code_reg_weight):
        if eval_loader is None:
            return

        vae_was_training = vae.training
        sdf_was_training = sdf_decoder.training
        vae.eval()
        sdf_decoder.eval()

        device = next(vae.parameters()).device
        eval_losses = []
        eval_sdf_losses = []
        eval_sdf_reg_losses = []
        eval_vae_recon = []
        eval_vae_kl = []
        eval_vae_lat_mag = []

        with torch.no_grad():
            for batch in eval_loader:
                sdf_data, indices, _labels, surface_points = _unpack_batch(batch)
                sdf_data = sdf_data.reshape(sdf_data.shape[0], -1, 4)

                xyz = sdf_data[:, :, 0:3].to(device)
                sdf_gt = sdf_data[:, :, 3].unsqueeze(-1).to(device)
                if vae_input_mode == "points":
                    if surface_points is None:
                        raise RuntimeError("Surface points required for point-based encoder.")
                    vae_in = surface_points.to(device)
                    teacher_batch = (
                        eval_latents[indices].to(device) if eval_latents is not None else None
                    )
                else:
                    if eval_latents is None:
                        raise RuntimeError("Latent inputs required for latent encoder.")
                    teacher_batch = eval_latents[indices].to(device)
                    vae_in = teacher_batch

                if enforce_minmax:
                    sdf_gt = torch.clamp(sdf_gt, minT, maxT)

                indices = indices.long()
                vae_out = vae(vae_in)
                mu = vae_out["mu"]
                logvar = vae_out["logvar"]
                z_hat = vae_out["z_hat"]
                if teacher_batch is not None:
                    vae_total, vae_recon, vae_kl = residual_mlp_vae.vae_loss(
                        z_hat,
                        teacher_batch,
                        mu,
                        logvar,
                        recon_weight=vae_recon_weight,
                        kl_weight=kl_weight,
                        recon_loss=recon_loss_type,
                    )
                else:
                    vae_total = torch.tensor(0.0, device=device)
                    vae_recon = torch.tensor(float("nan"), device=device)
                    vae_kl = torch.tensor(float("nan"), device=device)

                latent_per_sample, xyz_flat = residual_mlp_vae.expand_latent_to_points(
                    z_hat, xyz
                )
                sdf_gt_flat = sdf_gt.reshape(-1, 1)

                num_sdf_samples = float(sdf_gt_flat.shape[0])

                latent_chunks = torch.chunk(latent_per_sample, batch_split)
                xyz_chunks = torch.chunk(xyz_flat, batch_split)
                sdf_gt_chunks = torch.chunk(sdf_gt_flat, batch_split)

                batch_sdf_loss = 0.0
                batch_sdf_reg = 0.0

                for i in range(batch_split):
                    sdf_input = torch.cat([latent_chunks[i], xyz_chunks[i]], dim=1)
                    pred_sdf = sdf_decoder(sdf_input)

                    if enforce_minmax:
                        pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                    chunk_total, chunk_sdf, chunk_reg = residual_mlp_vae.deep_sdf_loss(
                        pred_sdf,
                        sdf_gt_chunks[i],
                        latent_chunks[i],
                        code_reg_lambda=code_reg_lambda,
                        code_reg_weight=code_reg_weight,
                    )

                    chunk_scale = float(pred_sdf.shape[0]) / num_sdf_samples
                    chunk_sdf = chunk_sdf * chunk_scale
                    chunk_reg = chunk_reg * chunk_scale

                    batch_sdf_loss += chunk_sdf.item()
                    batch_sdf_reg += chunk_reg.item()

                batch_total_loss = sdf_loss_weight * (batch_sdf_loss + batch_sdf_reg)
                if teacher_batch is not None:
                    batch_total_loss += vae_total.item()
                eval_losses.append(batch_total_loss)
                eval_sdf_losses.append(batch_sdf_loss)
                eval_sdf_reg_losses.append(batch_sdf_reg)
                if teacher_batch is not None:
                    eval_vae_recon.append(vae_recon.item())
                    eval_vae_kl.append(vae_kl.item())
                eval_vae_lat_mag.append(torch.mean(torch.norm(mu, dim=1)).item())

        eval_metrics = None
        if eval_losses:
            eval_loss = sum(eval_losses) / len(eval_losses)
            eval_sdf_loss = sum(eval_sdf_losses) / len(eval_sdf_losses)
            eval_sdf_reg = sum(eval_sdf_reg_losses) / len(eval_sdf_reg_losses)
            eval_vae_recon_loss = sum(eval_vae_recon) / len(eval_vae_recon) if eval_vae_recon else float("nan")
            eval_vae_kl_loss = sum(eval_vae_kl) / len(eval_vae_kl) if eval_vae_kl else float("nan")
            eval_vae_lat_mag = sum(eval_vae_lat_mag) / len(eval_vae_lat_mag)
            eval_metrics = {
                "eval_loss": eval_loss,
                "eval_sdf_loss": eval_sdf_loss,
                "eval_sdf_reg": eval_sdf_reg,
                "eval_vae_recon": eval_vae_recon_loss,
                "eval_vae_kl": eval_vae_kl_loss,
                "eval_vae_lat_mag": eval_vae_lat_mag,
            }

            logging.info(
                "{} eval loss: {:.6f} | sdf: {:.6f} | sdf_reg: {:.6f} | "
                "vae_recon: {:.6f} | vae_kl: {:.6f}".format(
                    split_label,
                    eval_loss,
                    eval_sdf_loss,
                    eval_sdf_reg,
                    eval_vae_recon_loss,
                    eval_vae_kl_loss,
                )
            )

            summary_writer.add_scalar(f"Loss/{split_label}", eval_loss, global_step=epoch)
            summary_writer.add_scalar(
                f"Loss/{split_label}_sdf", eval_sdf_loss, global_step=epoch
            )
            summary_writer.add_scalar(
                f"Loss/{split_label}_reg", eval_sdf_reg, global_step=epoch
            )
            summary_writer.add_scalar(
                f"Loss/{split_label}_vae_recon", eval_vae_recon_loss, global_step=epoch
            )
            summary_writer.add_scalar(
                f"Loss/{split_label}_vae_kl", eval_vae_kl_loss, global_step=epoch
            )
            summary_writer.add_scalar(
                f"Mean Latent Magnitude/{split_label}", eval_vae_lat_mag, global_step=epoch
            )

        if vae_was_training:
            vae.train()
        else:
            vae.eval()

        if sdf_was_training:
            sdf_decoder.train()
        else:
            sdf_decoder.eval()

        return eval_metrics

    def _collect_factors_codes(
        eval_loader, eval_latents, split_label, label_map, npyfiles, label_indices=None
    ):
        if eval_loader is None:
            return None, None
        if label_map is None:
            logging.warning("Metrics skipped for {}: SAPCORRLabelsFile is missing.".format(split_label))
            return None, None

        device = next(vae.parameters()).device
        codes_vae = []
        factors = []

        vae_was_training = vae.training
        vae.eval()
        with torch.no_grad():
            for batch in eval_loader:
                _sdf_data, indices, _labels, surface_points = _unpack_batch(batch)
                labels = _labels_for_indices(npyfiles, label_map, indices)
                if labels is None:
                    continue
                indices = indices.long()
                labels = labels.view(labels.shape[0], -1)
                if vae_input_mode == "points":
                    if surface_points is None:
                        raise RuntimeError("Surface points required for point-based encoder.")
                    vae_in = surface_points.to(device)
                else:
                    if eval_latents is None:
                        raise RuntimeError("Latent inputs required for latent encoder.")
                    vae_in = eval_latents[indices].to(device)
                vae_out = vae(vae_in)
                mu = vae_out["mu"]

                codes_vae.append(mu.detach().cpu())
                factors.append(labels.detach().cpu())

        if vae_was_training:
            vae.train()

        if not factors:
            logging.warning("Metrics skipped for {}: no labels found.".format(split_label))
            return None, None

        factors_np = torch.cat(factors, dim=0).numpy()
        codes_vae_np = torch.cat(codes_vae, dim=0).numpy()
        if label_indices is not None:
            indices = label_indices
            if isinstance(indices, int):
                indices = [indices]
            factors_np = factors_np[:, indices]

        mask = np.all(np.isfinite(factors_np), axis=1)
        mask &= np.all(factors_np != -1, axis=1)
        if mask.sum() < 2:
            logging.warning(
                "Metrics skipped for {}: insufficient valid labels.".format(split_label)
            )
            return None, None

        return factors_np[mask], codes_vae_np[mask]

    def _pca1_scores_np(x):
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        B, D = x.shape
        if B == 0:
            return np.zeros((0,))
        x_centered = x - x.mean(axis=0, keepdims=True)
        if D == 1:
            return x_centered[:, 0]
        if B <= 1:
            return np.zeros((B,))
        cov = (x_centered.T @ x_centered) / max(B - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        v1 = eigvecs[:, -1]
        return x_centered @ v1

    def _subset_corrs_from_labels(latents_np, labels_np):
        out = {
            "pca": {"disease": None, "age": None, "other": None},
            "mean": {"disease": None, "age": None, "other": None},
        }
        if labels_np is None or latents_np is None:
            return out
        if labels_np.shape[0] != latents_np.shape[0]:
            return out
        mask = np.isfinite(labels_np) & (labels_np != -1)
        if mask.sum() < 2:
            return out
        latents = latents_np[mask]
        y = labels_np[mask].astype(float)
        y = (y - y.mean()) / (y.std() + 1e-8)
        for name, dims in (("disease", disease_dims), ("age", age_dims), ("other", other_dims)):
            if not dims:
                out["pca"][name] = None
                out["mean"][name] = None
                continue
            x = latents[:, dims]
            s_pca = _pca1_scores_np(x)
            if np.std(s_pca) == 0 or np.std(y) == 0:
                out["pca"][name] = float("nan")
            else:
                s_pca = (s_pca - s_pca.mean()) / (s_pca.std() + 1e-8)
                out["pca"][name] = float((s_pca * y).mean())
            s_mean = x.mean(axis=1)
            if np.std(s_mean) == 0 or np.std(y) == 0:
                out["mean"][name] = float("nan")
            else:
                s_mean = (s_mean - s_mean.mean()) / (s_mean.std() + 1e-8)
                out["mean"][name] = float((s_mean * y).mean())
        return out

    def _regression_holdout_gap(y_vals, codes_sub):
        mask = np.isfinite(y_vals) & np.all(np.isfinite(codes_sub), axis=1)
        if mask.sum() < 4:
            return None
        y_valid = y_vals[mask]
        x_valid = codes_sub[mask]
        n_samples = y_valid.shape[0]
        test_size = max(1, int(round((1.0 - sap_kumar_holdout_frac) * n_samples)))
        train_size = n_samples - test_size
        if train_size < 2:
            return None
        idx = np.arange(n_samples)
        try:
            idx_train, idx_test = train_test_split(
                idx,
                test_size=test_size,
                train_size=train_size,
                random_state=sap_kumar_holdout_seed,
                shuffle=True,
            )
        except ValueError:
            return None

        scores = []
        for d in range(x_valid.shape[1]):
            x_train = x_valid[idx_train, d].reshape(-1, 1)
            x_test = x_valid[idx_test, d].reshape(-1, 1)
            y_train = y_valid[idx_train]
            y_test = y_valid[idx_test]
            if np.std(x_train) == 0 or np.std(y_train) == 0:
                scores.append(np.nan)
                continue
            regr = LinearRegression()
            regr.fit(x_train, y_train)
            y_pred = regr.predict(x_test)
            try:
                score = float(r2_score(y_test, y_pred))
            except ValueError:
                score = np.nan
            scores.append(score)

        vals = np.array(scores, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return None
        vals_sorted = np.sort(vals)
        return {
            "gap": float(vals_sorted[-1] - vals_sorted[-2]),
            "best": float(vals_sorted[-1]),
        }

    def _classification_holdout_gap(factors_sub, codes_sub, cfg):
        try:
            train_acc, test_acc, _ = sap_metric.sap_classification_holdout_predictions(
                factors_sub,
                codes_sub,
                continuous_factors=cfg.get("continuous_factors", True),
                nb_bins=cfg.get("nb_bins", 10),
                train_frac=sap_kumar_holdout_frac,
                random_state=sap_kumar_holdout_seed,
                pred_sample_n=0,
            )
        except Exception:
            return None
        if test_acc is None or test_acc.shape[0] == 0:
            return None
        vals = np.array(test_acc[0], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return None
        vals_sorted = np.sort(vals)
        return {
            "gap": float(vals_sorted[-1] - vals_sorted[-2]),
            "best": float(vals_sorted[-1]),
        }

    def _sap_holdout_for_subset(factors, codes, dims, cfg):
        if factors is None or codes is None:
            return None
        label_indices = cfg.get("label_indices")
        if label_indices is None:
            return None
        if isinstance(label_indices, int):
            label_indices = [label_indices]
        if len(label_indices) != 1:
            return None
        y = factors[:, label_indices[0]]
        x = codes[:, dims]
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            return None
        if cfg.get("regression", True):
            return _regression_holdout_gap(y, x)
        return _classification_holdout_gap(y.reshape(-1, 1), x, cfg)

    def _prepare_classification_labels(factors, cfg):
        label_indices = cfg.get("label_indices")
        if label_indices is None:
            return None
        if isinstance(label_indices, int):
            label_indices = [label_indices]
        if len(label_indices) != 1:
            return None
        y = factors[:, label_indices[0]]
        if cfg.get("continuous_factors", False):
            y = sap_metric.minmax_scale(y)
            y = sap_metric.get_bin_index(y, cfg.get("nb_bins", 10))
        return np.asarray(y).reshape(-1)

    def _subset_scores_classification_old(factors, codes, subsets, cfg):
        scores = {}
        if factors is None or codes is None:
            return scores
        y = _prepare_classification_labels(factors, cfg)
        if y is None:
            return scores
        if np.unique(y).size < 2:
            for name in subsets:
                scores[name] = 0.0
            return scores
        for name, dims in subsets.items():
            x = codes[:, dims]
            if x.size == 0 or x.shape[0] < 4:
                scores[name] = 0.0
                continue
            best_score, best_sp = 0, 0
            for sp in range(1, 10):
                clf = tree.DecisionTreeClassifier(max_depth=sp)
                try:
                    cv = cross_val_score(clf, x, y, cv=5).mean()
                except Exception:
                    cv = 0.0
                if cv > best_score:
                    best_score, best_sp = cv, sp
            clf = tree.DecisionTreeClassifier(max_depth=best_sp)
            try:
                clf.fit(x, y)
                y_pred = clf.predict(x)
                scores[name] = float(accuracy_score(y, y_pred))
            except Exception:
                scores[name] = 0.0
        return scores

    def _subset_scores_classification_holdout(factors, codes, subsets, cfg):
        scores = {}
        if factors is None or codes is None:
            return scores
        y = _prepare_classification_labels(factors, cfg)
        if y is None:
            return scores
        if np.unique(y).size < 2:
            for name in subsets:
                scores[name] = 0.0
            return scores
        n_samples = y.shape[0]
        test_size = max(1, int(round((1.0 - sap_kumar_holdout_frac) * n_samples)))
        train_size = n_samples - test_size
        if train_size < 2:
            for name in subsets:
                scores[name] = 0.0
            return scores
        try:
            idx = np.arange(n_samples)
            idx_train, idx_test = train_test_split(
                idx,
                test_size=test_size,
                train_size=train_size,
                random_state=sap_kumar_holdout_seed,
                stratify=y if np.unique(y).size > 1 else None,
            )
        except Exception:
            idx_train = np.arange(0, train_size)
            idx_test = np.arange(train_size, n_samples)
        for name, dims in subsets.items():
            x = codes[:, dims]
            if x.size == 0 or x.shape[0] < 4:
                scores[name] = 0.0
                continue
            x_train = x[idx_train]
            y_train = y[idx_train]
            x_test = x[idx_test]
            y_test = y[idx_test]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                scores[name] = 0.0
                continue
            best_score, best_sp = 0, 0
            for sp in range(1, 10):
                clf = tree.DecisionTreeClassifier(max_depth=sp)
                try:
                    cv = cross_val_score(clf, x_train, y_train, cv=5).mean()
                except Exception:
                    cv = 0.0
                if cv > best_score:
                    best_score, best_sp = cv, sp
            clf = tree.DecisionTreeClassifier(max_depth=best_sp)
            try:
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                scores[name] = float(accuracy_score(y_test, y_pred))
            except Exception:
                scores[name] = 0.0
        return scores

    def _subset_scores_classification_locatello(factors, codes, subsets, cfg):
        scores = {}
        if factors is None or codes is None:
            return scores
        y = _prepare_classification_labels(factors, cfg)
        if y is None:
            return scores
        if np.unique(y).size < 2:
            for name in subsets:
                scores[name] = 0.0
            return scores
        n_samples = y.shape[0]
        test_size = max(1, int(round((1.0 - sap_kumar_holdout_frac) * n_samples)))
        train_size = n_samples - test_size
        if train_size < 2:
            for name in subsets:
                scores[name] = 0.0
            return scores
        try:
            idx = np.arange(n_samples)
            idx_train, idx_test = train_test_split(
                idx,
                test_size=test_size,
                train_size=train_size,
                random_state=sap_kumar_holdout_seed,
                stratify=y if np.unique(y).size > 1 else None,
            )
        except Exception:
            idx_train = np.arange(0, train_size)
            idx_test = np.arange(train_size, n_samples)
        for name, dims in subsets.items():
            x = codes[:, dims]
            if x.size == 0 or x.shape[0] < 4:
                scores[name] = 0.0
                continue
            x_train = x[idx_train]
            y_train = y[idx_train]
            x_test = x[idx_test]
            y_test = y[idx_test]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                scores[name] = 0.0
                continue
            scaler = StandardScaler()
            try:
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                clf = LinearSVC(C=0.01, max_iter=5000)
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                scores[name] = float(accuracy_score(y_test, y_pred))
            except Exception:
                scores[name] = 0.0
        return scores

    def _subset_scores_regression_old(factors, codes, subsets, cfg):
        scores = {}
        if factors is None or codes is None:
            return scores
        label_indices = cfg.get("label_indices")
        if label_indices is None:
            return scores
        if isinstance(label_indices, int):
            label_indices = [label_indices]
        if len(label_indices) != 1:
            return scores
        y = factors[:, label_indices[0]]
        for name, dims in subsets.items():
            x = codes[:, dims]
            if x.size == 0 or x.shape[0] < 4:
                scores[name] = 0.0
                continue
            try:
                regr = LinearRegression()
                regr.fit(x, y)
                y_pred = regr.predict(x)
                scores[name] = float(r2_score(y, y_pred))
            except Exception:
                scores[name] = 0.0
        return scores

    def _subset_scores_regression_holdout(factors, codes, subsets, cfg, use_scaler=False):
        scores = {}
        if factors is None or codes is None:
            return scores
        label_indices = cfg.get("label_indices")
        if label_indices is None:
            return scores
        if isinstance(label_indices, int):
            label_indices = [label_indices]
        if len(label_indices) != 1:
            return scores
        y = factors[:, label_indices[0]]
        n_samples = y.shape[0]
        test_size = max(1, int(round((1.0 - sap_kumar_holdout_frac) * n_samples)))
        train_size = n_samples - test_size
        if train_size < 2:
            for name in subsets:
                scores[name] = 0.0
            return scores
        idx = np.arange(n_samples)
        try:
            idx_train, idx_test = train_test_split(
                idx,
                test_size=test_size,
                train_size=train_size,
                random_state=sap_kumar_holdout_seed,
                shuffle=True,
            )
        except Exception:
            idx_train = np.arange(0, train_size)
            idx_test = np.arange(train_size, n_samples)
        for name, dims in subsets.items():
            x = codes[:, dims]
            if x.size == 0 or x.shape[0] < 4:
                scores[name] = 0.0
                continue
            x_train = x[idx_train]
            x_test = x[idx_test]
            y_train = y[idx_train]
            y_test = y[idx_test]
            if use_scaler:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
            try:
                regr = LinearRegression()
                regr.fit(x_train, y_train)
                y_pred = regr.predict(x_test)
                scores[name] = float(r2_score(y_test, y_pred))
            except Exception:
                scores[name] = 0.0
        return scores

    def _corr_from_preds(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 2:
            return 0.0
        yt = y_true[mask]
        yp = y_pred[mask]
        if np.std(yt) == 0 or np.std(yp) == 0:
            return 0.0
        return float(np.corrcoef(yt, yp)[0, 1])

    def _subset_corr_classification_old(factors, codes, subsets, cfg):
        corrs = {}
        if factors is None or codes is None:
            return corrs
        y = _prepare_classification_labels(factors, cfg)
        if y is None or np.unique(y).size < 2:
            for name in subsets:
                corrs[name] = 0.0
            return corrs
        for name, dims in subsets.items():
            x = codes[:, dims]
            if x.size == 0 or x.shape[0] < 4:
                corrs[name] = 0.0
                continue
            best_score, best_sp = 0, 0
            for sp in range(1, 10):
                clf = tree.DecisionTreeClassifier(max_depth=sp)
                try:
                    cv = cross_val_score(clf, x, y, cv=5).mean()
                except Exception:
                    cv = 0.0
                if cv > best_score:
                    best_score, best_sp = cv, sp
            clf = tree.DecisionTreeClassifier(max_depth=best_sp)
            try:
                clf.fit(x, y)
                y_pred = clf.predict(x)
                corrs[name] = _corr_from_preds(y, y_pred)
            except Exception:
                corrs[name] = 0.0
        return corrs

    def _subset_corr_classification_holdout(factors, codes, subsets, cfg):
        corrs = {}
        if factors is None or codes is None:
            return corrs
        y = _prepare_classification_labels(factors, cfg)
        if y is None or np.unique(y).size < 2:
            for name in subsets:
                corrs[name] = 0.0
            return corrs
        n_samples = y.shape[0]
        test_size = max(1, int(round((1.0 - sap_kumar_holdout_frac) * n_samples)))
        train_size = n_samples - test_size
        if train_size < 2:
            for name in subsets:
                corrs[name] = 0.0
            return corrs
        try:
            idx = np.arange(n_samples)
            idx_train, idx_test = train_test_split(
                idx,
                test_size=test_size,
                train_size=train_size,
                random_state=sap_kumar_holdout_seed,
                stratify=y if np.unique(y).size > 1 else None,
            )
        except Exception:
            idx_train = np.arange(0, train_size)
            idx_test = np.arange(train_size, n_samples)
        for name, dims in subsets.items():
            x = codes[:, dims]
            if x.size == 0 or x.shape[0] < 4:
                corrs[name] = 0.0
                continue
            x_train = x[idx_train]
            y_train = y[idx_train]
            x_test = x[idx_test]
            y_test = y[idx_test]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                corrs[name] = 0.0
                continue
            best_score, best_sp = 0, 0
            for sp in range(1, 10):
                clf = tree.DecisionTreeClassifier(max_depth=sp)
                try:
                    cv = cross_val_score(clf, x_train, y_train, cv=5).mean()
                except Exception:
                    cv = 0.0
                if cv > best_score:
                    best_score, best_sp = cv, sp
            clf = tree.DecisionTreeClassifier(max_depth=best_sp)
            try:
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                corrs[name] = _corr_from_preds(y_test, y_pred)
            except Exception:
                corrs[name] = 0.0
        return corrs

    def _subset_corr_classification_locatello(factors, codes, subsets, cfg):
        corrs = {}
        if factors is None or codes is None:
            return corrs
        y = _prepare_classification_labels(factors, cfg)
        if y is None or np.unique(y).size < 2:
            for name in subsets:
                corrs[name] = 0.0
            return corrs
        n_samples = y.shape[0]
        test_size = max(1, int(round((1.0 - sap_kumar_holdout_frac) * n_samples)))
        train_size = n_samples - test_size
        if train_size < 2:
            for name in subsets:
                corrs[name] = 0.0
            return corrs
        try:
            idx = np.arange(n_samples)
            idx_train, idx_test = train_test_split(
                idx,
                test_size=test_size,
                train_size=train_size,
                random_state=sap_kumar_holdout_seed,
                stratify=y if np.unique(y).size > 1 else None,
            )
        except Exception:
            idx_train = np.arange(0, train_size)
            idx_test = np.arange(train_size, n_samples)
        for name, dims in subsets.items():
            x = codes[:, dims]
            if x.size == 0 or x.shape[0] < 4:
                corrs[name] = 0.0
                continue
            x_train = x[idx_train]
            y_train = y[idx_train]
            x_test = x[idx_test]
            y_test = y[idx_test]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                corrs[name] = 0.0
                continue
            scaler = StandardScaler()
            try:
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                clf = LinearSVC(C=0.01, max_iter=5000)
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                corrs[name] = _corr_from_preds(y_test, y_pred)
            except Exception:
                corrs[name] = 0.0
        return corrs

    def _subset_corr_regression_old(factors, codes, subsets, cfg):
        corrs = {}
        if factors is None or codes is None:
            return corrs
        label_indices = cfg.get("label_indices")
        if label_indices is None:
            return corrs
        if isinstance(label_indices, int):
            label_indices = [label_indices]
        if len(label_indices) != 1:
            return corrs
        y = factors[:, label_indices[0]]
        for name, dims in subsets.items():
            x = codes[:, dims]
            if x.size == 0 or x.shape[0] < 4:
                corrs[name] = 0.0
                continue
            try:
                regr = LinearRegression()
                regr.fit(x, y)
                y_pred = regr.predict(x)
                corrs[name] = _corr_from_preds(y, y_pred)
            except Exception:
                corrs[name] = 0.0
        return corrs

    def _subset_corr_regression_holdout(factors, codes, subsets, cfg, use_scaler=False):
        corrs = {}
        if factors is None or codes is None:
            return corrs
        label_indices = cfg.get("label_indices")
        if label_indices is None:
            return corrs
        if isinstance(label_indices, int):
            label_indices = [label_indices]
        if len(label_indices) != 1:
            return corrs
        y = factors[:, label_indices[0]]
        n_samples = y.shape[0]
        test_size = max(1, int(round((1.0 - sap_kumar_holdout_frac) * n_samples)))
        train_size = n_samples - test_size
        if train_size < 2:
            for name in subsets:
                corrs[name] = 0.0
            return corrs
        idx = np.arange(n_samples)
        try:
            idx_train, idx_test = train_test_split(
                idx,
                test_size=test_size,
                train_size=train_size,
                random_state=sap_kumar_holdout_seed,
                shuffle=True,
            )
        except Exception:
            idx_train = np.arange(0, train_size)
            idx_test = np.arange(train_size, n_samples)
        for name, dims in subsets.items():
            x = codes[:, dims]
            if x.size == 0 or x.shape[0] < 4:
                corrs[name] = 0.0
                continue
            x_train = x[idx_train]
            x_test = x[idx_test]
            y_train = y[idx_train]
            y_test = y[idx_test]
            if use_scaler:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
            try:
                regr = LinearRegression()
                regr.fit(x_train, y_train)
                y_pred = regr.predict(x_test)
                corrs[name] = _corr_from_preds(y_test, y_pred)
            except Exception:
                corrs[name] = 0.0
        return corrs

    def _subset_gap(scores):
        if not scores:
            return 0.0, None
        vals = [float(v) for v in scores.values()]
        vals_sorted = sorted(vals, reverse=True)
        if len(vals_sorted) < 2:
            return 0.0, None
        return float(vals_sorted[0] - vals_sorted[1]), vals_sorted

    def compute_disentanglement_metrics(
        eval_loader, eval_latents, epoch, split_label, label_map, npyfiles
    ):
        if eval_loader is None or (not compute_sap and not compute_sap_age):
            return {}

        def _fmt(v):
            if v is None:
                return "0.0000"
            try:
                if np.isnan(v):
                    return "0.0000"
            except Exception:
                pass
            return f"{v:.4f}"

        # disease labels/codes
        factors_d, codes_d = (None, None)
        if compute_sap:
            factors_d, codes_d = _collect_factors_codes(
                eval_loader,
                eval_latents,
                split_label,
                label_map,
                npyfiles,
                None,
            )
        # age labels/codes (may come from a different labels file)
        factors_a, codes_a = (None, None)
        if compute_sap_age:
            if sap_age_label_map is label_map and factors_d is not None:
                factors_a, codes_a = factors_d, codes_d
            else:
                factors_a, codes_a = _collect_factors_codes(
                    eval_loader,
                    eval_latents,
                    split_label,
                    sap_age_label_map,
                    npyfiles,
                    None,
                )

        subsets_all = {"disease": disease_dims, "age": age_dims, "other": other_dims}
        sap_scores_d = {}
        sap_holdout_scores_d = {}
        sap_loc_scores_d = {}
        dci_scores_d = {}
        mig_scores_d = {}
        pred_scores_d = {}
        sap_scores_a = {}
        sap_holdout_scores_a = {}
        sap_loc_scores_a = {}
        dci_scores_a = {}
        mig_scores_a = {}
        pred_scores_a = {}

        if factors_d is None and factors_a is None:
            return {}

        # disease subset metrics
        if compute_sap and factors_d is not None:
            cfg_d = {
                "disease": {
                    "label_indices": [disease_label_index],
                    "regression": sap_regression,
                    "continuous_factors": sap_continuous,
                    "nb_bins": sap_nb_bins,
                },
                "age": {
                    "label_indices": [disease_label_index],
                    "regression": sap_regression,
                    "continuous_factors": sap_continuous,
                    "nb_bins": sap_nb_bins,
                },
                "other": {
                    "label_indices": [disease_label_index],
                    "regression": sap_regression,
                    "continuous_factors": sap_continuous,
                    "nb_bins": sap_nb_bins,
                },
            }
            sap_scores_d.update(
                _subset_scores_classification_old(factors_d, codes_d, subsets_all, cfg_d["disease"])
            )
            sap_holdout_scores_d.update(
                _subset_scores_classification_holdout(
                    factors_d, codes_d, subsets_all, cfg_d["disease"]
                )
            )
            sap_loc_scores_d.update(
                _subset_scores_classification_locatello(
                    factors_d, codes_d, subsets_all, cfg_d["disease"]
                )
            )
            corr_sap_d_old = _subset_corr_classification_old(
                factors_d, codes_d, subsets_all, cfg_d["disease"]
            )
            corr_sap_d_hold = _subset_corr_classification_holdout(
                factors_d, codes_d, subsets_all, cfg_d["disease"]
            )
            corr_sap_d_loc = _subset_corr_classification_locatello(
                factors_d, codes_d, subsets_all, cfg_d["disease"]
            )
            pred_scores_d.update(dci_subset.predictability_by_subset(factors_d, codes_d, subsets_all, cfg_d))
            if compute_dci:
                dci_scores_d.update(dci_subset.dci_by_subset(factors_d, codes_d, subsets_all, cfg_d))
            if compute_mig:
                mig_scores_d.update(mig_subset.mig_by_subset(factors_d, codes_d, subsets_all, cfg_d))
        else:
            corr_sap_d_old = {k: 0.0 for k in subsets_all}
            corr_sap_d_hold = {k: 0.0 for k in subsets_all}
            corr_sap_d_loc = {k: 0.0 for k in subsets_all}

        # age subset metrics
        if compute_sap_age and factors_a is not None:
            cfg_a = {
                "disease": {
                    "label_indices": [age_label_index],
                    "regression": sap_age_regression,
                    "continuous_factors": sap_age_continuous,
                    "nb_bins": sap_age_nb_bins,
                },
                "age": {
                    "label_indices": [age_label_index],
                    "regression": sap_age_regression,
                    "continuous_factors": sap_age_continuous,
                    "nb_bins": sap_age_nb_bins,
                },
                "other": {
                    "label_indices": [age_label_index],
                    "regression": sap_age_regression,
                    "continuous_factors": sap_age_continuous,
                    "nb_bins": sap_age_nb_bins,
                },
            }
            sap_scores_a.update(
                _subset_scores_regression_old(factors_a, codes_a, subsets_all, cfg_a["age"])
            )
            sap_holdout_scores_a.update(
                _subset_scores_regression_holdout(
                    factors_a, codes_a, subsets_all, cfg_a["age"], use_scaler=False
                )
            )
            sap_loc_scores_a.update(
                _subset_scores_regression_holdout(
                    factors_a, codes_a, subsets_all, cfg_a["age"], use_scaler=True
                )
            )
            corr_sap_a_old = _subset_corr_regression_old(
                factors_a, codes_a, subsets_all, cfg_a["age"]
            )
            corr_sap_a_hold = _subset_corr_regression_holdout(
                factors_a, codes_a, subsets_all, cfg_a["age"], use_scaler=False
            )
            corr_sap_a_loc = _subset_corr_regression_holdout(
                factors_a, codes_a, subsets_all, cfg_a["age"], use_scaler=True
            )
            pred_scores_a.update(dci_subset.predictability_by_subset(factors_a, codes_a, subsets_all, cfg_a))
            if compute_dci:
                dci_scores_a.update(dci_subset.dci_by_subset(factors_a, codes_a, subsets_all, cfg_a))
            if compute_mig:
                mig_scores_a.update(mig_subset.mig_by_subset(factors_a, codes_a, subsets_all, cfg_a))
        else:
            corr_sap_a_old = {k: 0.0 for k in subsets_all}
            corr_sap_a_hold = {k: 0.0 for k in subsets_all}
            corr_sap_a_loc = {k: 0.0 for k in subsets_all}

        def _subset_to_label_corr(factors, codes, dims, label_idx, mode="pca"):
            if factors is None or codes is None:
                return None
            y = factors[:, label_idx]
            x = codes[:, dims]
            if x.ndim != 2 or x.shape[0] != y.shape[0]:
                return None
            if mode == "mean":
                s = x.mean(axis=1)
            else:
                s = _pca1_scores_np(x)
            # standardize
            y = (y - y.mean()) / (y.std() + 1e-8)
            s = (s - s.mean()) / (s.std() + 1e-8)
            corr = float((s * y).mean())
            return corr

        corr_subset_disease_pca = {
            "disease": _subset_to_label_corr(
                factors_d, codes_d, disease_dims, disease_label_index, mode="pca"
            ),
            "age": _subset_to_label_corr(
                factors_d, codes_d, age_dims, disease_label_index, mode="pca"
            ),
            "other": _subset_to_label_corr(
                factors_d, codes_d, other_dims, disease_label_index, mode="pca"
            ),
        }
        corr_subset_disease_mean = {
            "disease": _subset_to_label_corr(
                factors_d, codes_d, disease_dims, disease_label_index, mode="mean"
            ),
            "age": _subset_to_label_corr(
                factors_d, codes_d, age_dims, disease_label_index, mode="mean"
            ),
            "other": _subset_to_label_corr(
                factors_d, codes_d, other_dims, disease_label_index, mode="mean"
            ),
        }
        corr_subset_age_pca = {
            "disease": _subset_to_label_corr(
                factors_a, codes_a, disease_dims, age_label_index, mode="pca"
            ),
            "age": _subset_to_label_corr(
                factors_a, codes_a, age_dims, age_label_index, mode="pca"
            ),
            "other": _subset_to_label_corr(
                factors_a, codes_a, other_dims, age_label_index, mode="pca"
            ),
        }
        corr_subset_age_mean = {
            "disease": _subset_to_label_corr(
                factors_a, codes_a, disease_dims, age_label_index, mode="mean"
            ),
            "age": _subset_to_label_corr(
                factors_a, codes_a, age_dims, age_label_index, mode="mean"
            ),
            "other": _subset_to_label_corr(
                factors_a, codes_a, other_dims, age_label_index, mode="mean"
            ),
        }

        gap_d_old, _ = _subset_gap(sap_scores_d)
        gap_d_hold, _ = _subset_gap(sap_holdout_scores_d)
        gap_d_loc, _ = _subset_gap(sap_loc_scores_d)
        gap_a_old, _ = _subset_gap(sap_scores_a)
        gap_a_hold, _ = _subset_gap(sap_holdout_scores_a)
        gap_a_loc, _ = _subset_gap(sap_loc_scores_a)

        # TensorBoard logging
        if compute_sap:
            summary_writer.add_scalar(
                f"SAPSubsetGap/vae_{split_label}_disease_old", gap_d_old, global_step=epoch
            )
            summary_writer.add_scalar(
                f"SAPSubsetGap/vae_{split_label}_disease_holdout", gap_d_hold, global_step=epoch
            )
            summary_writer.add_scalar(
                f"SAPSubsetGap/vae_{split_label}_disease_locatello", gap_d_loc, global_step=epoch
            )
            for subset_name, val in sap_scores_d.items():
                summary_writer.add_scalar(
                    f"SAPSubsetScore/vae_{split_label}_disease_old_{subset_name}",
                    float(val),
                    global_step=epoch,
                )
            for subset_name, val in sap_holdout_scores_d.items():
                summary_writer.add_scalar(
                    f"SAPSubsetScore/vae_{split_label}_disease_holdout_{subset_name}",
                    float(val),
                    global_step=epoch,
                )
            for subset_name, val in sap_loc_scores_d.items():
                summary_writer.add_scalar(
                    f"SAPSubsetScore/vae_{split_label}_disease_locatello_{subset_name}",
                    float(val),
                    global_step=epoch,
                )
            for subset_name, val in corr_sap_d_old.items():
                summary_writer.add_scalar(
                    f"CorrSAP/vae_{split_label}_disease_old_{subset_name}",
                    float(val),
                    global_step=epoch,
                )
            for subset_name, val in corr_sap_d_hold.items():
                summary_writer.add_scalar(
                    f"CorrSAP/vae_{split_label}_disease_holdout_{subset_name}",
                    float(val),
                    global_step=epoch,
                )
            for subset_name, val in corr_sap_d_loc.items():
                summary_writer.add_scalar(
                    f"CorrSAP/vae_{split_label}_disease_locatello_{subset_name}",
                    float(val),
                    global_step=epoch,
                )
        if compute_sap_age:
            summary_writer.add_scalar(
                f"SAPSubsetGap/vae_{split_label}_age_old", gap_a_old, global_step=epoch
            )
            summary_writer.add_scalar(
                f"SAPSubsetGap/vae_{split_label}_age_holdout", gap_a_hold, global_step=epoch
            )
            summary_writer.add_scalar(
                f"SAPSubsetGap/vae_{split_label}_age_locatello", gap_a_loc, global_step=epoch
            )
            for subset_name, val in sap_scores_a.items():
                summary_writer.add_scalar(
                    f"SAPSubsetScore/vae_{split_label}_age_old_{subset_name}",
                    float(val),
                    global_step=epoch,
                )
            for subset_name, val in sap_holdout_scores_a.items():
                summary_writer.add_scalar(
                    f"SAPSubsetScore/vae_{split_label}_age_holdout_{subset_name}",
                    float(val),
                    global_step=epoch,
                )
            for subset_name, val in sap_loc_scores_a.items():
                summary_writer.add_scalar(
                    f"SAPSubsetScore/vae_{split_label}_age_locatello_{subset_name}",
                    float(val),
                    global_step=epoch,
                )
            for subset_name, val in corr_sap_a_old.items():
                summary_writer.add_scalar(
                    f"CorrSAP/vae_{split_label}_age_old_{subset_name}",
                    float(val),
                    global_step=epoch,
                )
            for subset_name, val in corr_sap_a_hold.items():
                summary_writer.add_scalar(
                    f"CorrSAP/vae_{split_label}_age_holdout_{subset_name}",
                    float(val),
                    global_step=epoch,
                )
            for subset_name, val in corr_sap_a_loc.items():
                summary_writer.add_scalar(
                    f"CorrSAP/vae_{split_label}_age_locatello_{subset_name}",
                    float(val),
                    global_step=epoch,
                )
        if "disease" in dci_scores_d:
            summary_writer.add_scalar(
                f"DCI/vae_{split_label}_disease_disentanglement",
                dci_scores_d["disease"]["disentanglement"],
                global_step=epoch,
            )
            summary_writer.add_scalar(
                f"DCI/vae_{split_label}_disease_completeness",
                dci_scores_d["disease"]["completeness"],
                global_step=epoch,
            )
            summary_writer.add_scalar(
                f"DCI/vae_{split_label}_disease_informativeness",
                dci_scores_d["disease"]["informativeness"],
                global_step=epoch,
            )
        if "age" in dci_scores_d:
            summary_writer.add_scalar(
                f"DCI/vae_{split_label}_age_disentanglement",
                dci_scores_d["age"]["disentanglement"],
                global_step=epoch,
            )
            summary_writer.add_scalar(
                f"DCI/vae_{split_label}_age_completeness",
                dci_scores_d["age"]["completeness"],
                global_step=epoch,
            )
            summary_writer.add_scalar(
                f"DCI/vae_{split_label}_age_informativeness",
                dci_scores_d["age"]["informativeness"],
                global_step=epoch,
            )
        if "other" in dci_scores_d:
            summary_writer.add_scalar(
                f"DCI/vae_{split_label}_other_disentanglement",
                dci_scores_d["other"]["disentanglement"],
                global_step=epoch,
            )
            summary_writer.add_scalar(
                f"DCI/vae_{split_label}_other_completeness",
                dci_scores_d["other"]["completeness"],
                global_step=epoch,
            )
            summary_writer.add_scalar(
                f"DCI/vae_{split_label}_other_informativeness",
                dci_scores_d["other"]["informativeness"],
                global_step=epoch,
            )
        if "disease" in mig_scores_d:
            summary_writer.add_scalar(
                f"MIG/vae_{split_label}_disease", mig_scores_d["disease"]["mig"], global_step=epoch
            )
        if "age" in mig_scores_d:
            summary_writer.add_scalar(
                f"MIG/vae_{split_label}_age", mig_scores_d["age"]["mig"], global_step=epoch
            )
        if "other" in mig_scores_d:
            summary_writer.add_scalar(
                f"MIG/vae_{split_label}_other", mig_scores_d["other"]["mig"], global_step=epoch
            )
        if "disease" in pred_scores_d:
            summary_writer.add_scalar(
                f"Predict/vae_{split_label}_disease_acc", pred_scores_d["disease"], global_step=epoch
            )
        if "age" in pred_scores_d:
            summary_writer.add_scalar(
                f"Predict/vae_{split_label}_age_r2", pred_scores_d["age"], global_step=epoch
            )
        if "other" in pred_scores_d:
            summary_writer.add_scalar(
                f"Predict/vae_{split_label}_other", pred_scores_d["other"], global_step=epoch
            )
        for subset_name, val in corr_subset_disease_pca.items():
            if val is not None:
                summary_writer.add_scalar(
                    f"Corr/vae_{split_label}_{subset_name}_disease", val, global_step=epoch
                )
        for subset_name, val in corr_subset_age_pca.items():
            if val is not None:
                summary_writer.add_scalar(
                    f"Corr/vae_{split_label}_{subset_name}_age", val, global_step=epoch
                )
        for subset_name, val in corr_subset_disease_mean.items():
            if val is not None:
                summary_writer.add_scalar(
                    f"CorrMean/vae_{split_label}_{subset_name}_disease",
                    val,
                    global_step=epoch,
                )
        for subset_name, val in corr_subset_age_mean.items():
            if val is not None:
                summary_writer.add_scalar(
                    f"CorrMean/vae_{split_label}_{subset_name}_age",
                    val,
                    global_step=epoch,
                )

        # Console tables
        def _log_subset_pred_table(title, scores_old, scores_hold, scores_loc, gap_old, gap_hold, gap_loc):
            rows = [
                [
                    "disease",
                    _fmt(scores_old.get("disease")),
                    _fmt(scores_hold.get("disease")),
                    _fmt(scores_loc.get("disease")),
                ],
                [
                    "age",
                    _fmt(scores_old.get("age")),
                    _fmt(scores_hold.get("age")),
                    _fmt(scores_loc.get("age")),
                ],
                [
                    "other",
                    _fmt(scores_old.get("other")),
                    _fmt(scores_hold.get("other")),
                    _fmt(scores_loc.get("other")),
                ],
            ]
            header = ["subset", "old", "holdout", "locatello"]
            col_widths = [max(len(str(row[i])) for row in ([header] + rows)) for i in range(len(header))]
            lines = [" | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(header))]
            lines.append("-+-".join("-" * w for w in col_widths))
            for row in rows:
                lines.append(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(header))))
            logging.info("Epoch %d %s subset prediction (%s):\n%s", epoch, title, split_label, "\n".join(lines))
            logging.info(
                "Epoch %d %s SAP gap (%s): old=%.4f holdout=%.4f locatello=%.4f",
                epoch,
                title,
                split_label,
                gap_old,
                gap_hold,
                gap_loc,
            )

        def _log_corr_table(title, corr_old, corr_hold, corr_loc, corr_pca, corr_mean):
            rows = [
                [
                    "sap_old",
                    _fmt(corr_old.get("disease")),
                    _fmt(corr_old.get("age")),
                    _fmt(corr_old.get("other")),
                ],
                [
                    "sap_holdout",
                    _fmt(corr_hold.get("disease")),
                    _fmt(corr_hold.get("age")),
                    _fmt(corr_hold.get("other")),
                ],
                [
                    "sap_locatello",
                    _fmt(corr_loc.get("disease")),
                    _fmt(corr_loc.get("age")),
                    _fmt(corr_loc.get("other")),
                ],
                [
                    "pca",
                    _fmt(corr_pca.get("disease")),
                    _fmt(corr_pca.get("age")),
                    _fmt(corr_pca.get("other")),
                ],
                [
                    "mean",
                    _fmt(corr_mean.get("disease")),
                    _fmt(corr_mean.get("age")),
                    _fmt(corr_mean.get("other")),
                ],
            ]
            header = ["type", "disease", "age", "other"]
            col_widths = [max(len(str(row[i])) for row in ([header] + rows)) for i in range(len(header))]
            lines = [" | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(header))]
            lines.append("-+-".join("-" * w for w in col_widths))
            for row in rows:
                lines.append(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(header))))
            logging.info("Epoch %d %s correlation (%s):\n%s", epoch, title, split_label, "\n".join(lines))

        def _log_metrics_table(title, dci_scores, mig_scores):
            rows = [
                [
                    "disease",
                    _fmt(dci_scores.get("disease", {}).get("disentanglement") if "disease" in dci_scores else None),
                    _fmt(dci_scores.get("disease", {}).get("completeness") if "disease" in dci_scores else None),
                    _fmt(dci_scores.get("disease", {}).get("informativeness") if "disease" in dci_scores else None),
                    _fmt(mig_scores.get("disease", {}).get("mig") if "disease" in mig_scores else None),
                ],
                [
                    "age",
                    _fmt(dci_scores.get("age", {}).get("disentanglement") if "age" in dci_scores else None),
                    _fmt(dci_scores.get("age", {}).get("completeness") if "age" in dci_scores else None),
                    _fmt(dci_scores.get("age", {}).get("informativeness") if "age" in dci_scores else None),
                    _fmt(mig_scores.get("age", {}).get("mig") if "age" in mig_scores else None),
                ],
                [
                    "other",
                    _fmt(dci_scores.get("other", {}).get("disentanglement") if "other" in dci_scores else None),
                    _fmt(dci_scores.get("other", {}).get("completeness") if "other" in dci_scores else None),
                    _fmt(dci_scores.get("other", {}).get("informativeness") if "other" in dci_scores else None),
                    _fmt(mig_scores.get("other", {}).get("mig") if "other" in mig_scores else None),
                ],
            ]
            header = ["subset", "DCI_dis", "DCI_comp", "DCI_info", "MIG"]
            col_widths = [max(len(str(row[i])) for row in ([header] + rows)) for i in range(len(header))]
            lines = [" | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(header))]
            lines.append("-+-".join("-" * w for w in col_widths))
            for row in rows:
                lines.append(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(header))))
            logging.info("Epoch %d %s metrics (%s):\n%s", epoch, title, split_label, "\n".join(lines))

        if compute_sap:
            _log_subset_pred_table(
                "Disease-label",
                sap_scores_d,
                sap_holdout_scores_d,
                sap_loc_scores_d,
                gap_d_old,
                gap_d_hold,
                gap_d_loc,
            )
            _log_corr_table(
                "Disease-label",
                corr_sap_d_old,
                corr_sap_d_hold,
                corr_sap_d_loc,
                corr_subset_disease_pca,
                corr_subset_disease_mean,
            )
            _log_metrics_table(
                "Disease-label",
                dci_scores_d,
                mig_scores_d,
            )
        if compute_sap_age:
            _log_subset_pred_table(
                "Age-label",
                sap_scores_a,
                sap_holdout_scores_a,
                sap_loc_scores_a,
                gap_a_old,
                gap_a_hold,
                gap_a_loc,
            )
            _log_corr_table(
                "Age-label",
                corr_sap_a_old,
                corr_sap_a_hold,
                corr_sap_a_loc,
                corr_subset_age_pca,
                corr_subset_age_mean,
            )
            _log_metrics_table(
                "Age-label",
                dci_scores_a,
                mig_scores_a,
            )

        return {
            "sap": {"disease": gap_d_old, "age": gap_a_old},
            "sap_holdout": {"disease": gap_d_hold, "age": gap_a_hold},
            "sap_locatello": {"disease": gap_d_loc, "age": gap_a_loc},
            "corr_sap": {
                "disease": {"old": corr_sap_d_old, "holdout": corr_sap_d_hold, "locatello": corr_sap_d_loc},
                "age": {"old": corr_sap_a_old, "holdout": corr_sap_a_hold, "locatello": corr_sap_a_loc},
            },
            "sap_subset_scores": {
                "disease": {
                    "old": sap_scores_d,
                    "holdout": sap_holdout_scores_d,
                    "locatello": sap_loc_scores_d,
                },
                "age": {
                    "old": sap_scores_a,
                    "holdout": sap_holdout_scores_a,
                    "locatello": sap_loc_scores_a,
                },
            },
            "dci": {"disease_label": dci_scores_d, "age_label": dci_scores_a},
            "mig": {"disease_label": mig_scores_d, "age_label": mig_scores_a},
            "predict": {"disease_label": pred_scores_d, "age_label": pred_scores_a},
        }

    def generate_eval_meshes(dataset, eval_latents, scene_indices, split_label, epoch):
        if dataset is None or not scene_indices:
            return

        vae_was_training = vae.training
        sdf_was_training = sdf_decoder.training
        vae.eval()
        sdf_decoder.eval()

        device = next(vae.parameters()).device
        if split_label == "train":
            recon_dir = ws.tb_logs_train_reconstructions
        else:
            recon_dir = ws.tb_logs_test_reconstructions

        with torch.no_grad():
            for scene_idx in scene_indices:
                if vae_input_mode == "points":
                    surface_points = dataset.surface_points[scene_idx]
                    vae_in = torch.as_tensor(surface_points).unsqueeze(0).to(device)
                else:
                    if eval_latents is None:
                        raise RuntimeError("Latent inputs required for latent encoder.")
                    vae_in = eval_latents[scene_idx : scene_idx + 1].to(device)
                vae_out = vae(vae_in)
                z_hat = vae_out["z_hat"]

                save_name = os.path.basename(dataset.npyfiles[scene_idx]).split(".npz")[0]
                out_dir = os.path.join(
                    experiment_directory, ws.tb_logs_dir, recon_dir, save_name
                )
                os.makedirs(out_dir, exist_ok=True)

                mesh.create_mesh(
                    sdf_decoder,
                    z_hat,
                    N=eval_grid_res,
                    max_batch=eval_max_batch,
                    filename=os.path.join(out_dir, f"epoch={epoch}"),
                    return_trimesh=False,
                )

        if vae_was_training:
            vae.train()
        else:
            vae.eval()

        if sdf_was_training:
            sdf_decoder.train()
        else:
            sdf_decoder.eval()

    def compute_chamfer_for_scenes(dataset, eval_latents, scene_indices, split_label, epoch):
        if (
            dataset is None
            or not scene_indices
            or eval_gt_mesh_dir is None
        ):
            return None

        vae_was_training = vae.training
        sdf_was_training = sdf_decoder.training
        vae.eval()
        sdf_decoder.eval()

        device = next(vae.parameters()).device
        chamfer_dists = []
        for scene_idx in scene_indices:
            base_name = os.path.splitext(os.path.basename(dataset.npyfiles[scene_idx]))[0]
            gt_path = os.path.join(eval_gt_mesh_dir, base_name + eval_gt_mesh_ext)
            if not os.path.isfile(gt_path):
                logging.warning("GT mesh missing for chamfer: %s", gt_path)
                continue
            if vae_input_mode == "points":
                surface_points = dataset.surface_points[scene_idx]
                vae_in = torch.as_tensor(surface_points).unsqueeze(0).to(device)
            else:
                if eval_latents is None:
                    raise RuntimeError("Latent inputs required for latent encoder.")
                vae_in = eval_latents[scene_idx : scene_idx + 1].to(device)
            with torch.no_grad():
                vae_out = vae(vae_in)
                z_hat = vae_out["z_hat"]
            gen_mesh = mesh.create_mesh(
                sdf_decoder,
                z_hat,
                N=eval_grid_res,
                max_batch=eval_max_batch,
                return_trimesh=True,
            )
            if gen_mesh is None:
                continue
            cd, _ = metrics.compute_metric(
                gt_mesh=gt_path,
                gen_mesh=gen_mesh,
                num_mesh_samples=eval_gt_mesh_samples,
                metric="chamfer",
            )
            chamfer_dists.append(cd)

        if vae_was_training:
            vae.train()
        else:
            vae.eval()
        if sdf_was_training:
            sdf_decoder.train()
        else:
            sdf_decoder.eval()

        if not chamfer_dists:
            return None
        mean_cd = sum(chamfer_dists) / len(chamfer_dists)
        summary_writer.add_scalar(
            f"Chamfer/{split_label}", mean_cd, global_step=epoch
        )
        return mean_cd

    def compute_latent_label_correlation(
        dataset, eval_latents, epoch, split_label, label_map, scene_indices=None
    ):
        if compute_sap or compute_sap_age:
            # correlations are already logged via compute_disentanglement_metrics
            return
        if dataset is None:
            return
        labels_d = _collect_label_values(dataset.npyfiles, label_map, disease_label_index)
        label_map_age = sap_age_label_map if sap_age_label_map is not None else label_map
        labels_a = _collect_label_values(dataset.npyfiles, label_map_age, age_label_index)
        if labels_d is None and labels_a is None:
            return
        vae_inputs = _select_vae_inputs(dataset, eval_latents, scene_indices)
        if vae_inputs is None:
            logging.warning(
                "Correlation skipped ({}): VAE inputs unavailable.".format(split_label)
            )
            return

        if scene_indices is not None:
            scene_indices = [int(idx) for idx in scene_indices]
            if labels_d is not None:
                labels_d = labels_d[scene_indices]
            if labels_a is not None:
                labels_a = labels_a[scene_indices]

        latent_batch = get_spec_with_default(specs, "LatentExportBatchSize", 1024)
        device = next(vae.parameters()).device
        vae_latents = compute_vae_latents(
            vae, vae_inputs, latent_batch, device
        ).cpu().numpy()

        if labels_d is not None and vae_latents.shape[0] != labels_d.shape[0]:
            logging.warning(
                "Correlation skipped ({}): latent count {} != disease label count {}".format(
                    split_label, vae_latents.shape[0], labels_d.shape[0]
                )
            )
            return
        if labels_a is not None and vae_latents.shape[0] != labels_a.shape[0]:
            logging.warning(
                "Correlation skipped ({}): latent count {} != age label count {}".format(
                    split_label, vae_latents.shape[0], labels_a.shape[0]
                )
            )
            return

        corr_d = _subset_corrs_from_labels(vae_latents, labels_d)
        corr_a = _subset_corrs_from_labels(vae_latents, labels_a)

        for subset_name, val in corr_d["pca"].items():
            if val is not None:
                summary_writer.add_scalar(
                    f"Corr/vae_{split_label}_{subset_name}_disease", val, global_step=epoch
                )
        for subset_name, val in corr_a["pca"].items():
            if val is not None:
                summary_writer.add_scalar(
                    f"Corr/vae_{split_label}_{subset_name}_age", val, global_step=epoch
                )
        for subset_name, val in corr_d["mean"].items():
            if val is not None:
                summary_writer.add_scalar(
                    f"CorrMean/vae_{split_label}_{subset_name}_disease",
                    val,
                    global_step=epoch,
                )
        for subset_name, val in corr_a["mean"].items():
            if val is not None:
                summary_writer.add_scalar(
                    f"CorrMean/vae_{split_label}_{subset_name}_age",
                    val,
                    global_step=epoch,
                )

    def print_latent_diagnosis_table(
        dataset, eval_latents, epoch, split_label, label_map, scene_indices=None
    ):
        if compute_sap or compute_sap_age:
            # subset table already printed in compute_disentanglement_metrics
            return
        if dataset is None:
            return
        labels_d = _collect_label_values(dataset.npyfiles, label_map, disease_label_index)
        label_map_age = sap_age_label_map if sap_age_label_map is not None else label_map
        labels_a = _collect_label_values(dataset.npyfiles, label_map_age, age_label_index)
        if labels_d is None and labels_a is None:
            return
        vae_inputs = _select_vae_inputs(dataset, eval_latents, scene_indices)
        if vae_inputs is None:
            logging.warning(
                "Latent table skipped ({}): VAE inputs unavailable.".format(split_label)
            )
            return

        if scene_indices is not None:
            scene_indices = [int(idx) for idx in scene_indices]
            if labels_d is not None:
                labels_d = labels_d[scene_indices]
            if labels_a is not None:
                labels_a = labels_a[scene_indices]

        latent_batch = get_spec_with_default(specs, "LatentExportBatchSize", 1024)
        device = next(vae.parameters()).device
        vae_latents = compute_vae_latents(
            vae, vae_inputs, latent_batch, device
        ).cpu().numpy()

        if labels_d is not None and vae_latents.shape[0] != labels_d.shape[0]:
            logging.warning(
                "Latent table skipped ({}): latent count {} != disease label count {}".format(
                    split_label, vae_latents.shape[0], labels_d.shape[0]
                )
            )
            return
        if labels_a is not None and vae_latents.shape[0] != labels_a.shape[0]:
            logging.warning(
                "Latent table skipped ({}): latent count {} != age label count {}".format(
                    split_label, vae_latents.shape[0], labels_a.shape[0]
                )
            )
            return

        corr_d = _subset_corrs_from_labels(vae_latents, labels_d)
        corr_a = _subset_corrs_from_labels(vae_latents, labels_a)

        def _fmt(v):
            if v is None:
                return "n/a"
            try:
                if np.isnan(v):
                    return "n/a"
            except Exception:
                pass
            return f"{v:.4f}"

        rows = [
            [
                "disease",
                _fmt(corr_d["pca"].get("disease")),
                _fmt(corr_d["mean"].get("disease")),
                _fmt(corr_a["pca"].get("disease")),
                _fmt(corr_a["mean"].get("disease")),
            ],
            [
                "age",
                _fmt(corr_d["pca"].get("age")),
                _fmt(corr_d["mean"].get("age")),
                _fmt(corr_a["pca"].get("age")),
                _fmt(corr_a["mean"].get("age")),
            ],
            [
                "other",
                _fmt(corr_d["pca"].get("other")),
                _fmt(corr_d["mean"].get("other")),
                _fmt(corr_a["pca"].get("other")),
                _fmt(corr_a["mean"].get("other")),
            ],
        ]
        header = [
            "subset",
            "corr_disease_pca",
            "corr_disease_mean",
            "corr_age_pca",
            "corr_age_mean",
        ]
        col_widths = [max(len(str(row[i])) for row in ([header] + rows)) for i in range(len(header))]
        lines = [" | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(header))]
        lines.append("-+-".join("-" * w for w in col_widths))
        for row in rows:
            lines.append(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(header))))
        logging.info("Epoch %d subset correlation table (%s):\n%s", epoch, split_label, "\n".join(lines))

    def log_eval_debug(eval_loader, dataset, eval_latents, epoch, split_label, label_map):
        if eval_loader is None or dataset is None:
            return
        label_summary = _summarize_labels(dataset.npyfiles, label_map, label_index)
        logging.info(
            "Epoch %d debug (%s): samples=%d surface_points=%s expected_points=%d label_summary=%s",
            epoch,
            split_label,
            len(dataset),
            "set" if getattr(dataset, "surface_points", None) else "missing",
            surface_point_count,
            label_summary,
        )
        device = next(vae.parameters()).device
        try:
            batch = next(iter(eval_loader))
        except StopIteration:
            logging.warning("Epoch %d debug (%s): eval loader is empty.", epoch, split_label)
            return
        sdf_data, indices, labels, surface_points = _unpack_batch(batch)
        batch_size = indices.shape[0]
        indices = indices.long()
        teacher_batch = None
        if eval_latents is not None:
            teacher_batch = eval_latents[indices].to(device)
        if vae_input_mode == "points":
            if surface_points is None:
                logging.warning(
                    "Epoch %d debug (%s): surface_points missing in batch.",
                    epoch,
                    split_label,
                )
                return
            surface_points = torch.as_tensor(surface_points).to(device)
            if surface_points.dim() != 3:
                logging.warning(
                    "Epoch %d debug (%s): surface_points dim=%d shape=%s",
                    epoch,
                    split_label,
                    surface_points.dim(),
                    tuple(surface_points.shape),
                )
            else:
                logging.info(
                    "Epoch %d debug (%s): surface_points shape=%s min=%.4f max=%.4f",
                    epoch,
                    split_label,
                    tuple(surface_points.shape),
                    float(surface_points.min().item()),
                    float(surface_points.max().item()),
                )
            vae_in = surface_points
        else:
            if teacher_batch is None:
                logging.warning(
                    "Epoch %d debug (%s): latent inputs missing.",
                    epoch,
                    split_label,
                )
                return
            logging.info(
                "Epoch %d debug (%s): latent_inputs shape=%s min=%.4f max=%.4f",
                epoch,
                split_label,
                tuple(teacher_batch.shape),
                float(teacher_batch.min().item()),
                float(teacher_batch.max().item()),
            )
            vae_in = teacher_batch

        vae_out = vae(vae_in)
        mu = vae_out["mu"]
        z_hat = vae_out["z_hat"]
        if teacher_batch is not None:
            recon_mse = F.mse_loss(z_hat, teacher_batch).item()
            logging.info(
                "Epoch %d debug (%s): teacher_batch=%s z_hat=%s recon_mse=%.6f",
                epoch,
                split_label,
                tuple(teacher_batch.shape),
                tuple(z_hat.shape),
                recon_mse,
            )
        else:
            logging.info(
                "Epoch %d debug (%s): z_hat=%s (teacher_batch missing)",
                epoch,
                split_label,
                tuple(z_hat.shape),
            )
        logging.info(
            "Epoch %d debug (%s): mu[0] mean=%.6f std=%.6f",
            epoch,
            split_label,
            float(mu[:, 0].mean().item()),
            float(mu[:, 0].std().item()),
        )
        if labels is None:
            logging.warning("Epoch %d debug (%s): labels missing in batch.", epoch, split_label)
            return
        labels = labels.to(device).view(labels.shape[0], -1)
        if labels.shape[1] <= label_index:
            logging.warning(
                "Epoch %d debug (%s): label_index=%d out of bounds for labels shape=%s",
                epoch,
                split_label,
                label_index,
                tuple(labels.shape),
            )
            return
        label_values = labels[:, label_index].to(torch.float32)
        valid_mask = torch.isfinite(label_values) & (label_values != -1)
        if valid_mask.any():
            vals = label_values[valid_mask]
            logging.info(
                "Epoch %d debug (%s): age stats min=%.4f max=%.4f mean=%.4f std=%.4f",
                epoch,
                split_label,
                float(vals.min().item()),
                float(vals.max().item()),
                float(vals.mean().item()),
                float(vals.std().item()),
            )
        else:
            logging.warning(
                "Epoch %d debug (%s): no valid age labels in batch.",
                epoch,
                split_label,
            )
        logging.info(
            "Epoch %d debug (%s): batch_size=%d valid_labels=%d label_values unique=%s",
            epoch,
            split_label,
            batch_size,
            int(valid_mask.sum().item()),
            torch.unique(label_values[valid_mask]).tolist() if valid_mask.any() else [],
        )
        if guided_contrastive_loss:
            if valid_mask.sum().item() > 1:
                y_vals = label_values[valid_mask]
                snnl_val = snn_loss_fn(mu[valid_mask], y_vals).item()
                if isinstance(snn_loss_fn, deep_sdf_loss.SNNRegLossExact):
                    y = y_vals.view(-1, 1)
                    bsize = y.shape[0]
                    offdiag = ~torch.eye(bsize, dtype=torch.bool, device=y.device)
                    abs_dy = torch.abs(y - y.t())
                    if snn_loss_fn.pos_mode == "topk":
                        abs_dy = abs_dy.masked_fill(~offdiag, float("inf"))
                        k = max(1, int(round(snn_loss_fn.topk_frac * (bsize - 1))))
                        thr_i = abs_dy.kthvalue(k, dim=1).values.unsqueeze(1)
                        same = abs_dy <= thr_i
                    else:
                        same = abs_dy <= snn_loss_fn.threshold
                    same = same & offdiag
                    pos_pairs = int(same.sum().item())
                    logging.info(
                        "Epoch %d debug (%s): snnl_pos_pairs=%d avg_pos_per_sample=%.2f mode=%s thr=%.4f topk_frac=%.3f",
                        epoch,
                        split_label,
                        pos_pairs,
                        float(pos_pairs) / float(bsize) if bsize else 0.0,
                        snn_loss_fn.pos_mode,
                        snn_loss_fn.threshold,
                        snn_loss_fn.topk_frac,
                    )
                    if pos_pairs == 0:
                        logging.warning(
                            "Epoch %d debug (%s): no positive pairs for SNNRegLossExact; increase threshold/topk_frac.",
                            epoch,
                            split_label,
                        )
                logging.info(
                    "Epoch %d debug (%s): snnl_loss=%.6f temp=%.3f",
                    epoch,
                    split_label,
                    snnl_val,
                    snnl_temp,
                )
            else:
                logging.info(
                    "Epoch %d debug (%s): snnl skipped (valid_labels=%d)",
                    epoch,
                    split_label,
                    int(valid_mask.sum().item()),
                )

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            epoch_time_start = time.time()

            epoch_losses = []
            epoch_sdf_losses = []
            epoch_sdf_reg_losses = []
            epoch_vae_recon = []
            epoch_vae_kl = []
            epoch_vae_lat_mag = []
            epoch_snnl = []
            epoch_snnl_age = []
            epoch_attr = []
            epoch_cov = []
            epoch_corr_leak = []
            epoch_cross_cov = []
            epoch_rank = []
            epoch_matchstd = []
            epoch_matchstd_std0 = []
            epoch_matchstd_stdref = []
            epoch_sens = []
            epoch_sens_delta = []

            logging.info("epoch {}...".format(epoch))

            vae.train()
            if train_sdf_decoder:
                sdf_decoder.train()
            else:
                sdf_decoder.eval()

            device = next(vae.parameters()).device

            adjust_learning_rate(lr_schedules, optimizer, epoch, loss_log_epoch)

            if use_kl:
                kl_weight = vae_kl_weight * residual_mlp_vae.linear_warmup(
                    epoch, vae_kl_warmup_epochs
                )
            else:
                kl_weight = 0.0
            if do_code_regularization:
                if code_reg_warmup_epochs <= 0:
                    code_reg_weight = 1.0
                else:
                    code_reg_weight = min(1.0, epoch / float(code_reg_warmup_epochs))
            else:
                code_reg_weight = 0.0

            for batch in sdf_loader:
                sdf_data, indices, labels, surface_points = _unpack_batch(batch)
                sdf_data = sdf_data.reshape(sdf_data.shape[0], -1, 4)

                sdf_data.requires_grad = False

                xyz = sdf_data[:, :, 0:3].to(device)
                sdf_gt = sdf_data[:, :, 3].unsqueeze(-1).to(device)

                if enforce_minmax:
                    sdf_gt = torch.clamp(sdf_gt, minT, maxT)

                indices = indices.long()
                teacher_batch = teacher_latents[indices].to(device)

                if vae_input_mode == "points":
                    if surface_points is None:
                        raise RuntimeError(
                            "Surface points required for point-based encoder."
                        )
                    vae_in = torch.as_tensor(surface_points).to(device)
                else:
                    vae_in = teacher_batch

                vae_out = vae(vae_in)
                mu = vae_out["mu"]
                logvar = vae_out["logvar"]
                z_hat = vae_out["z_hat"]

                if use_kl and (
                    kl_disease_weight != 1.0
                    or kl_age_weight != 1.0
                    or kl_other_weight != 1.0
                ):
                    # reconstruction
                    if recon_loss_type == "l1":
                        recon = F.l1_loss(z_hat, teacher_batch, reduction="mean")
                    elif recon_loss_type == "mse":
                        recon = F.mse_loss(z_hat, teacher_batch, reduction="mean")
                    else:
                        raise ValueError(f"Unsupported recon_loss: {recon_loss_type}")

                    # weighted KL per subset
                    kl_terms = 0.5 * (
                        mu.pow(2) + logvar.exp() - 1.0 - logvar
                    )  # [B,D]
                    kl_dim = kl_terms.mean(dim=0)  # [D]
                    weights = mu.new_ones(mu.shape[1])
                    weights[disease_dims] = kl_disease_weight
                    weights[age_dims] = kl_age_weight
                    weights[other_dims] = kl_other_weight
                    kl = (kl_dim * weights).mean()

                    vae_recon = recon
                    vae_kl = kl
                    vae_total = vae_recon_weight * recon + kl_weight * kl
                else:
                    vae_total, vae_recon, vae_kl = residual_mlp_vae.vae_loss(
                        z_hat,
                        teacher_batch,
                        mu,
                        logvar,
                        recon_weight=vae_recon_weight,
                        kl_weight=kl_weight,
                        recon_loss=recon_loss_type,
                    )

                snnl_loss_val = 0.0
                age_snnl_loss_val = 0.0
                attr_loss_val = 0.0
                cov_loss_val = 0.0
                corr_leak_loss_val = 0.0
                cross_cov_loss_val = 0.0
                rank_loss_val = 0.0
                matchstd_loss_val = 0.0
                matchstd_std0_val = 0.0
                matchstd_stdref_val = 0.0
                sens_loss_val = 0.0
                sens_delta_val = 0.0
                if use_labels:
                    label_values = None
                    if label_mix_enabled:
                        pseudo_ratio = float(mix_pseudo_start)
                        unlabeled_ratio = float(mix_unlabeled_start)
                        if pseudo_ratio < 0.0 or unlabeled_ratio < 0.0:
                            raise RuntimeError("Label mix ratios must be >= 0.")
                        if pseudo_ratio + unlabeled_ratio > 1.0:
                            raise RuntimeError(
                                "Label mix ratios exceed 1.0 (pseudo {} + unlabeled {}).".format(
                                    pseudo_ratio, unlabeled_ratio
                                )
                            )
                        real_ratio = 1.0 - pseudo_ratio - unlabeled_ratio

                        batch_size = indices.shape[0]
                        if label_mix_stratified:
                            k_real = int(round(real_ratio * batch_size))
                            k_pseudo = int(round(pseudo_ratio * batch_size))
                            if k_real + k_pseudo > batch_size:
                                overflow = k_real + k_pseudo - batch_size
                                if k_pseudo >= overflow:
                                    k_pseudo -= overflow
                                else:
                                    overflow -= k_pseudo
                                    k_pseudo = 0
                                    k_real = max(0, k_real - overflow)
                            perm = torch.randperm(batch_size, device=mu.device)
                            real_mask = torch.zeros(
                                batch_size, device=mu.device, dtype=torch.bool
                            )
                            pseudo_mask = torch.zeros(
                                batch_size, device=mu.device, dtype=torch.bool
                            )
                            if k_real > 0:
                                real_mask[perm[:k_real]] = True
                            if k_pseudo > 0:
                                pseudo_mask[perm[k_real : k_real + k_pseudo]] = True
                        else:
                            rand = torch.rand(batch_size, device=mu.device)
                            real_mask = rand < real_ratio
                            pseudo_mask = (rand >= real_ratio) & (
                                rand < (real_ratio + pseudo_ratio)
                            )
                        label_values = torch.full(
                            (batch_size,), float("nan"), device=mu.device
                        )

                        if pseudo_ratio > 0.0 and pseudo_mask.any():
                            pseudo_labels = _labels_for_indices(
                                sdf_dataset.npyfiles, pseudo_label_map, indices
                            )
                            if pseudo_labels is None:
                                raise RuntimeError(
                                    "Label mixing enabled but pseudo labels are missing."
                                )
                            pseudo_labels = pseudo_labels.to(mu.device).view(
                                pseudo_labels.shape[0], -1
                            )
                            if pseudo_labels.shape[1] <= disease_label_index:
                                raise RuntimeError(
                                    "Pseudo labels missing label_index {} (shape {}).".format(
                                        disease_label_index, pseudo_labels.shape
                                    )
                                )
                            label_values[pseudo_mask] = pseudo_labels[
                                pseudo_mask, disease_label_index
                            ].to(torch.float32)

                        if real_ratio > 0.0 and real_mask.any():
                            real_labels = _labels_for_indices(
                                sdf_dataset.npyfiles, real_label_map, indices
                            )
                            if real_labels is None:
                                raise RuntimeError(
                                    "Label mixing enabled but real labels are missing."
                                )
                            real_labels = real_labels.to(mu.device).view(
                                real_labels.shape[0], -1
                            )
                            if real_labels.shape[1] <= disease_label_index:
                                raise RuntimeError(
                                    "Real labels missing label_index {} (shape {}).".format(
                                        disease_label_index, real_labels.shape
                                    )
                                )
                            label_values[real_mask] = real_labels[
                                real_mask, disease_label_index
                            ].to(torch.float32)
                    else:
                        if labels is None:
                            raise RuntimeError("Label-based losses enabled but labels are missing in batch.")
                        labels = labels.to(mu.device).view(labels.shape[0], -1)
                        if labels.shape[1] <= disease_label_index:
                            raise RuntimeError(
                                "Labels missing label_index {} (shape {}).".format(
                                    disease_label_index, labels.shape
                                )
                            )
                        label_values = labels[:, disease_label_index].to(torch.float32)

                    valid_mask = torch.isfinite(label_values) & (label_values != -1)
                    if valid_mask.any():
                        if guided_contrastive_loss and valid_mask.sum().item() > 1:
                            snnl_loss = snn_loss_fn(
                                mu[valid_mask], label_values[valid_mask]
                            )
                            vae_total = vae_total + (snnl_weight * snnl_loss)
                            snnl_loss_val = snnl_loss.item()
                        if attribute_loss:
                            if attribute_subset:
                                attr_latent = mu[valid_mask][:, _subset_to_dims(attribute_subset)].mean(dim=1)
                            else:
                                attr_latent = mu[valid_mask, attribute_latent_index]
                            attr_loss = attr_loss_fn(
                                attr_latent, label_values[valid_mask]
                            )
                            vae_total = vae_total + (attr_weight * attr_loss)
                            attr_loss_val = attr_loss.item()
                        if corr_leakage_loss:
                            leak_loss = deep_sdf_loss_subset.corr_leakage_penalty_group(
                                mu[valid_mask],
                                label_values[valid_mask],
                                disease_dims,
                            )
                            vae_total = vae_total + (corr_leakage_lambda * leak_loss)
                            corr_leak_loss_val = leak_loss.item()
                        if cross_cov_loss:
                            cross_loss = deep_sdf_loss_subset.cross_cov_penalty_group(
                                mu[valid_mask],
                                disease_dims,
                            )
                            vae_total = vae_total + (cross_cov_lambda * cross_loss)
                            cross_cov_loss_val = cross_loss.item()
                        if rank_loss:
                            rank_loss_val_t = rank_loss_fn(
                                mu[valid_mask], label_values[valid_mask]
                            )
                            vae_total = vae_total + (rank_weight * rank_loss_val_t)
                            rank_loss_val = rank_loss_val_t.item()
                    if age_snnl_reg_loss:
                        age_labels = labels
                        if age_labels is None:
                            raise RuntimeError(
                                "Age SNNL enabled but labels are missing in batch."
                            )
                        age_labels = age_labels.to(mu.device).view(age_labels.shape[0], -1)
                        if age_labels.shape[1] <= age_label_index:
                            raise RuntimeError(
                                "Labels missing age label_index {} (shape {}).".format(
                                    age_label_index, age_labels.shape
                                )
                            )
                        age_label_values = age_labels[:, age_label_index].to(
                            torch.float32
                        )
                        age_valid_mask = torch.isfinite(age_label_values) & (
                            age_label_values != -1
                        )
                        if age_valid_mask.any() and age_valid_mask.sum().item() > 1:
                            age_snnl_loss = age_snnl_reg_fn(
                                mu[age_valid_mask], age_label_values[age_valid_mask]
                            )
                            vae_total = vae_total + (age_snnl_reg_weight * age_snnl_loss)
                            age_snnl_loss_val = age_snnl_loss.item()

                if matchstd_loss:
                    matchstd_loss_t, std0_t, stdref_t = matchstd_loss_fn(mu)
                    vae_total = vae_total + (matchstd_weight * matchstd_loss_t)
                    matchstd_loss_val = matchstd_loss_t.item()
                    matchstd_std0_val = float(std0_t.item())
                    matchstd_stdref_val = float(stdref_t.item())

                if sensitivity_loss:
                    decoder = _get_vae_decoder(vae)
                    sens_loss, sens_delta = sens_loss_fn(mu, decoder)
                    vae_total = vae_total + (sensitivity_weight * sens_loss)
                    sens_loss_val = sens_loss.item()
                    sens_delta_val = sens_delta.item()

                if covariance_loss:
                    cov_loss = cov_loss_fn(mu, logvar)
                    vae_total = vae_total + cov_loss
                    cov_loss_val = cov_loss.item()

                latent_per_sample, xyz_flat = residual_mlp_vae.expand_latent_to_points(
                    z_hat, xyz
                )
                sdf_gt_flat = sdf_gt.reshape(-1, 1)

                num_sdf_samples = float(sdf_gt_flat.shape[0])

                latent_chunks = torch.chunk(latent_per_sample, batch_split)
                xyz_chunks = torch.chunk(xyz_flat, batch_split)
                sdf_gt_chunks = torch.chunk(sdf_gt_flat, batch_split)

                optimizer.zero_grad()

                batch_sdf_loss = 0.0
                batch_sdf_reg = 0.0

                for i in range(batch_split):
                    sdf_input = torch.cat([latent_chunks[i], xyz_chunks[i]], dim=1)
                    pred_sdf = sdf_decoder(sdf_input)

                    if enforce_minmax:
                        pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                    chunk_total, chunk_sdf, chunk_reg = residual_mlp_vae.deep_sdf_loss(
                        pred_sdf,
                        sdf_gt_chunks[i],
                        latent_chunks[i],
                        code_reg_lambda=code_reg_lambda,
                        code_reg_weight=code_reg_weight,
                    )

                    chunk_scale = float(pred_sdf.shape[0]) / num_sdf_samples
                    chunk_total = chunk_total * chunk_scale
                    chunk_sdf = chunk_sdf * chunk_scale
                    chunk_reg = chunk_reg * chunk_scale

                    (sdf_loss_weight * chunk_total).backward(retain_graph=True)

                    batch_sdf_loss += chunk_sdf.item()
                    batch_sdf_reg += chunk_reg.item()

                vae_total.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip, norm_type=2)
                    if train_sdf_decoder:
                        torch.nn.utils.clip_grad_norm_(sdf_decoder.parameters(), grad_clip, norm_type=2)

                optimizer.step()

                batch_total_loss = sdf_loss_weight * (batch_sdf_loss + batch_sdf_reg) + vae_total.item()
                loss_log.append(batch_total_loss)
                epoch_losses.append(batch_total_loss)
                epoch_sdf_losses.append(batch_sdf_loss)
                epoch_sdf_reg_losses.append(batch_sdf_reg)
                epoch_vae_recon.append(vae_recon.item())
                epoch_vae_kl.append(vae_kl.item())
                epoch_vae_lat_mag.append(torch.mean(torch.norm(mu, dim=1)).item())
                if guided_contrastive_loss:
                    epoch_snnl.append(snnl_loss_val)
                if age_snnl_reg_loss:
                    epoch_snnl_age.append(age_snnl_loss_val)
                if attribute_loss:
                    epoch_attr.append(attr_loss_val)
                if covariance_loss:
                    epoch_cov.append(cov_loss_val)
                if corr_leakage_loss:
                    epoch_corr_leak.append(corr_leak_loss_val)
                if cross_cov_loss:
                    epoch_cross_cov.append(cross_cov_loss_val)
                if rank_loss:
                    epoch_rank.append(rank_loss_val)
                if matchstd_loss:
                    epoch_matchstd.append(matchstd_loss_val)
                    epoch_matchstd_std0.append(matchstd_std0_val)
                    epoch_matchstd_stdref.append(matchstd_stdref_val)
                if sensitivity_loss:
                    epoch_sens.append(sens_loss_val)
                    epoch_sens_delta.append(sens_delta_val)

            seconds_elapsed = time.time() - epoch_time_start
            timing_log.append(seconds_elapsed)

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_sdf_loss = sum(epoch_sdf_losses) / len(epoch_sdf_losses)
            epoch_sdf_reg = sum(epoch_sdf_reg_losses) / len(epoch_sdf_reg_losses)
            epoch_vae_recon_loss = sum(epoch_vae_recon) / len(epoch_vae_recon)
            epoch_vae_kl_loss = sum(epoch_vae_kl) / len(epoch_vae_kl)
            epoch_vae_lat_mag = sum(epoch_vae_lat_mag) / len(epoch_vae_lat_mag)
            epoch_snnl_loss = sum(epoch_snnl) / len(epoch_snnl) if epoch_snnl else 0.0
            epoch_snnl_age_loss = (
                sum(epoch_snnl_age) / len(epoch_snnl_age) if epoch_snnl_age else 0.0
            )
            epoch_attr_loss = sum(epoch_attr) / len(epoch_attr) if epoch_attr else 0.0
            epoch_cov_loss = sum(epoch_cov) / len(epoch_cov) if epoch_cov else 0.0
            epoch_corr_leak_loss = (
                sum(epoch_corr_leak) / len(epoch_corr_leak) if epoch_corr_leak else 0.0
            )
            epoch_cross_cov_loss = (
                sum(epoch_cross_cov) / len(epoch_cross_cov) if epoch_cross_cov else 0.0
            )
            epoch_rank_loss = sum(epoch_rank) / len(epoch_rank) if epoch_rank else 0.0
            epoch_matchstd_loss = (
                sum(epoch_matchstd) / len(epoch_matchstd) if epoch_matchstd else 0.0
            )
            epoch_matchstd_std0 = (
                sum(epoch_matchstd_std0) / len(epoch_matchstd_std0) if epoch_matchstd_std0 else 0.0
            )
            epoch_matchstd_stdref = (
                sum(epoch_matchstd_stdref) / len(epoch_matchstd_stdref) if epoch_matchstd_stdref else 0.0
            )
            epoch_sens_loss = sum(epoch_sens) / len(epoch_sens) if epoch_sens else 0.0
            epoch_sens_delta = (
                sum(epoch_sens_delta) / len(epoch_sens_delta) if epoch_sens_delta else 0.0
            )
            epoch_sdf_weighted = sdf_loss_weight * (epoch_sdf_loss + epoch_sdf_reg)
            epoch_vae_recon_weighted = vae_recon_weight * epoch_vae_recon_loss
            epoch_vae_kl_weighted = kl_weight * epoch_vae_kl_loss

            sens_log = ""
            if sensitivity_loss:
                sens_log = " | sens: {:.6f} | sens_delta: {:.6f}".format(
                    epoch_sens_loss, epoch_sens_delta
                )
                logging.info(
                    "Sensitivity debug (epoch %d): delta=%.6f target_eta=%.6f",
                    epoch,
                    epoch_sens_delta,
                    float(sensitivity_eta),
                )

            if use_kl:
                logging.info(
                    "Epoch {} loss: {:.6f} | sdf: {:.6f} | sdf_reg: {:.6f} | "
                    "vae_recon: {:.6f} | vae_kl: {:.6f} | "
                    "weighted -> sdf: {:.6f} | vae_recon: {:.6f} | vae_kl: {:.6f}{}".format(
                        epoch,
                        epoch_loss,
                        epoch_sdf_loss,
                        epoch_sdf_reg,
                        epoch_vae_recon_loss,
                        epoch_vae_kl_loss,
                        epoch_sdf_weighted,
                        epoch_vae_recon_weighted,
                        epoch_vae_kl_weighted,
                        sens_log,
                    )
                )
            else:
                logging.info(
                    "Epoch {} loss: {:.6f} | sdf: {:.6f} | sdf_reg: {:.6f} | "
                    "vae_recon: {:.6f} | weighted -> sdf: {:.6f} | vae_recon: {:.6f}{}".format(
                        epoch,
                        epoch_loss,
                        epoch_sdf_loss,
                        epoch_sdf_reg,
                        epoch_vae_recon_loss,
                        epoch_sdf_weighted,
                        epoch_vae_recon_weighted,
                        sens_log,
                    )
                )
            if (
                guided_contrastive_loss
                or age_snnl_reg_loss
                or attribute_loss
                or covariance_loss
                or corr_leakage_loss
                or cross_cov_loss
                or rank_loss
                or matchstd_loss
            ):
                extra_parts = []
                if guided_contrastive_loss:
                    extra_parts.append(f"snnl: {epoch_snnl_loss:.6f}")
                if age_snnl_reg_loss:
                    extra_parts.append(f"snnl_age: {epoch_snnl_age_loss:.6f}")
                if attribute_loss:
                    extra_parts.append(f"attr: {epoch_attr_loss:.6f}")
                if covariance_loss:
                    extra_parts.append(f"cov: {epoch_cov_loss:.6f}")
                if corr_leakage_loss:
                    extra_parts.append(f"leak: {epoch_corr_leak_loss:.6f}")
                if cross_cov_loss:
                    extra_parts.append(f"cross_cov: {epoch_cross_cov_loss:.6f}")
                if rank_loss:
                    extra_parts.append(f"rank: {epoch_rank_loss:.6f}")
                if matchstd_loss:
                    extra_parts.append(f"matchstd: {epoch_matchstd_loss:.6f}")
                if extra_parts:
                    logging.info(
                        "Epoch {} extra losses: {}".format(
                            epoch, " | ".join(extra_parts)
                        )
                    )

            loss_log_epoch.append(epoch_loss)
            sdf_loss_log_epoch.append(epoch_sdf_loss)
            sdf_reg_log_epoch.append(epoch_sdf_reg)
            vae_recon_log_epoch.append(epoch_vae_recon_loss)
            vae_kl_log_epoch.append(epoch_vae_kl_loss)
            vae_lat_mag_log.append(epoch_vae_lat_mag)
            snnl_log_epoch.append(epoch_snnl_loss)
            snnl_age_log_epoch.append(epoch_snnl_age_loss)
            attr_log_epoch.append(epoch_attr_loss)
            cov_log_epoch.append(epoch_cov_loss)
            corr_leak_log_epoch.append(epoch_corr_leak_loss)
            cross_cov_log_epoch.append(epoch_cross_cov_loss)
            rank_log_epoch.append(epoch_rank_loss)
            matchstd_log_epoch.append(epoch_matchstd_loss)
            matchstd_std0_log_epoch.append(epoch_matchstd_std0)
            matchstd_stdref_log_epoch.append(epoch_matchstd_stdref)
            sens_log_epoch.append(epoch_sens_loss)
            sens_delta_log_epoch.append(epoch_sens_delta)

            summary_writer.add_scalar("Loss/train", epoch_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_sdf", epoch_sdf_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_reg", epoch_sdf_reg, global_step=epoch)
            summary_writer.add_scalar("Loss/train_vae_recon", epoch_vae_recon_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_vae_kl", epoch_vae_kl_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_vae_total", epoch_vae_recon_loss + epoch_vae_kl_loss, global_step=epoch)
            summary_writer.add_scalar("Mean Latent Magnitude/train", epoch_vae_lat_mag, global_step=epoch)
            summary_writer.add_scalar("KL/warmup", kl_weight, global_step=epoch)
            if guided_contrastive_loss:
                summary_writer.add_scalar("Loss/train_snnl", epoch_snnl_loss, global_step=epoch)
            if age_snnl_reg_loss:
                summary_writer.add_scalar(
                    "Loss/train_snnl_age", epoch_snnl_age_loss, global_step=epoch
                )
            if attribute_loss:
                summary_writer.add_scalar("Loss/train_attr", epoch_attr_loss, global_step=epoch)
            if covariance_loss:
                summary_writer.add_scalar("Loss/train_cov", epoch_cov_loss, global_step=epoch)
            if corr_leakage_loss:
                summary_writer.add_scalar("Loss/train_leak", epoch_corr_leak_loss, global_step=epoch)
            if cross_cov_loss:
                summary_writer.add_scalar("Loss/train_cross_cov", epoch_cross_cov_loss, global_step=epoch)
            if rank_loss:
                summary_writer.add_scalar("Loss/train_rank", epoch_rank_loss, global_step=epoch)
            if matchstd_loss:
                summary_writer.add_scalar("Loss/train_matchstd", epoch_matchstd_loss, global_step=epoch)
                summary_writer.add_scalar("Metric/train_matchstd_std0", epoch_matchstd_std0, global_step=epoch)
                summary_writer.add_scalar("Metric/train_matchstd_stdref", epoch_matchstd_stdref, global_step=epoch)
            if sensitivity_loss:
                summary_writer.add_scalar("Loss/train_sensitivity", epoch_sens_loss, global_step=epoch)
                summary_writer.add_scalar(
                    "Metric/train_sensitivity_delta", epoch_sens_delta, global_step=epoch
                )

            lr_log.append([group["lr"] for group in optimizer.param_groups])
            summary_writer.add_scalar("Learning Rate/VAE", optimizer.param_groups[0]["lr"], global_step=epoch)
            if train_sdf_decoder and len(optimizer.param_groups) > 1:
                summary_writer.add_scalar("Learning Rate/SDFDecoder", optimizer.param_groups[1]["lr"], global_step=epoch)

            if epoch in checkpoints:
                save_checkpoints(epoch)

            if epoch % log_frequency == 0:
                save_latest(epoch)
                save_logs(
                    experiment_directory,
                    loss_log,
                    loss_log_epoch,
                    sdf_loss_log_epoch,
                    sdf_reg_log_epoch,
                    vae_recon_log_epoch,
                    vae_kl_log_epoch,
                    vae_lat_mag_log,
                    snnl_log_epoch,
                    snnl_age_log_epoch,
                    attr_log_epoch,
                    cov_log_epoch,
                    corr_leak_log_epoch,
                    cross_cov_log_epoch,
                    rank_log_epoch,
                    matchstd_log_epoch,
                    matchstd_std0_log_epoch,
                    matchstd_stdref_log_epoch,
                    sens_log_epoch,
                    sens_delta_log_epoch,
                    lr_log,
                    timing_log,
                    epoch,
                )

            if (
                eval_train_loader is not None
                and eval_train_frequency is not None
                and eval_train_frequency > 0
                and epoch % eval_train_frequency == 0
            ):
                eval_metrics = run_eval(
                    eval_train_loader,
                    teacher_latents,
                    epoch,
                    "eval_train",
                    kl_weight,
                    code_reg_weight,
                )
                if eval_metrics is not None:
                    last_train_eval_sdf = eval_metrics.get("eval_sdf_loss")
                    last_train_eval_epoch = epoch
                def _run_label_metrics(eval_loader, split_label, scene_indices):
                    if eval_loader is None:
                        return None
                    metrics = compute_disentanglement_metrics(
                        eval_loader,
                        teacher_latents,
                        epoch,
                        split_label,
                        sap_corr_label_map,
                        sdf_dataset.npyfiles,
                    )
                    compute_latent_label_correlation(
                        sdf_dataset,
                        teacher_latents,
                        epoch,
                        split_label,
                        sap_corr_label_map,
                        scene_indices=scene_indices,
                    )
                    print_latent_diagnosis_table(
                        sdf_dataset,
                        teacher_latents,
                        epoch,
                        split_label,
                        sap_corr_label_map,
                        scene_indices=scene_indices,
                    )
                    log_eval_debug(
                        eval_loader,
                        sdf_dataset,
                        teacher_latents,
                        epoch,
                        split_label,
                        sap_corr_label_map,
                    )
                    return metrics

                train_eval_indices_use = None
                if train_eval_indices is not None:
                    train_eval_indices_use = train_eval_indices
                elif hasattr(eval_train_loader.dataset, "indices"):
                    train_eval_indices_use = eval_train_loader.dataset.indices
                train_metrics = _run_label_metrics(
                    eval_train_loader, "train", train_eval_indices_use
                )
                if train_metrics and train_metrics.get("sap") is not None:
                    last_train_sap = train_metrics["sap"]
                if eval_train_holdout_loader is not None:
                    _run_label_metrics(
                        eval_train_holdout_loader,
                        "train_holdout",
                        train_holdout_eval_indices,
                    )
                generate_eval_meshes(
                    sdf_dataset,
                    teacher_latents,
                    eval_train_scene_idxs,
                    "train",
                    epoch,
                )
                if eval_gt_mesh_dir is None:
                    logging.error("EvalGTMeshDir not set; skipping train Chamfer.")
                else:
                    train_cd = compute_chamfer_for_scenes(
                        sdf_dataset,
                        teacher_latents,
                        eval_train_scene_idxs,
                        "train",
                        epoch,
                    )
                    if train_cd is not None:
                        last_train_cd = train_cd
                        logging.info(
                            "Epoch %d train chamfer: %.6f (mesh_count=%d)",
                            epoch,
                            train_cd,
                            len(eval_train_scene_idxs),
                        )
                    else:
                        logging.info(
                            "Epoch %d train chamfer: n/a (mesh_count=%d)",
                            epoch,
                            len(eval_train_scene_idxs),
                        )

            if (
                sap_corr_extra_frequency is not None
                and sap_corr_extra_frequency > 0
                and epoch % sap_corr_extra_frequency == 0
            ):
                if compute_sap:
                    if sap_train_loader is not None:
                        train_metrics_extra = compute_disentanglement_metrics(
                            sap_train_loader,
                            teacher_latents,
                            epoch,
                            "train_extra",
                            sap_corr_label_map,
                            sdf_dataset.npyfiles,
                        )
                        if train_metrics_extra and train_metrics_extra.get("sap") is not None:
                            last_train_sap = train_metrics_extra["sap"]
                if (
                    eval_train_loader is not None
                    and last_train_eval_epoch != epoch
                ):
                    eval_metrics = run_eval(
                        eval_train_loader,
                        teacher_latents,
                        epoch,
                        "eval_train_extra",
                        kl_weight,
                        code_reg_weight,
                    )
                    if eval_metrics is not None:
                        last_train_eval_sdf = eval_metrics.get("eval_sdf_loss")
                        last_train_eval_epoch = epoch
                if any(
                    metric is not None
                    for metric in (
                        last_train_eval_sdf,
                        last_train_sap,
                        last_train_cd,
                        last_test_eval_sdf,
                        last_test_sap,
                        last_test_cd,
                    )
                ):
                    def _fmt_sap_summary(val):
                        if val is None:
                            return "n/a"
                        if isinstance(val, dict):
                            parts = []
                            if "disease" in val:
                                parts.append("disease={:.4f}".format(val["disease"]))
                            if "age" in val:
                                parts.append("age={:.4f}".format(val["age"]))
                            return "{" + ", ".join(parts) + "}" if parts else "n/a"
                        try:
                            return "{:.6f}".format(val)
                        except Exception:
                            return str(val)

                    logging.info(
                        "Epoch {} extra summary: train_sdf_loss={} train_sap={} train_cd={} test_sdf_loss={} test_sap={} test_cd={}".format(
                            epoch,
                            "{:.6f}".format(last_train_eval_sdf)
                            if last_train_eval_sdf is not None
                            else "n/a",
                            _fmt_sap_summary(last_train_sap),
                            "{:.6f}".format(last_train_cd)
                            if last_train_cd is not None
                            else "n/a",
                            "{:.6f}".format(last_test_eval_sdf)
                            if last_test_eval_sdf is not None
                            else "n/a",
                            _fmt_sap_summary(last_test_sap),
                            "{:.6f}".format(last_test_cd)
                            if last_test_cd is not None
                            else "n/a",
                        )
                    )

            if (
                eval_test_frequency is not None
                and eval_test_frequency > 0
                and epoch % eval_test_frequency == 0
            ):
                if eval_test_loader is None:
                    logging.warning(
                        "EvalTestFrequency set but no test eval loader; skipping eval."
                    )
                elif epoch < eval_test_start_epoch:
                    logging.info(
                        "Skipping test eval at epoch %d (start epoch %d).",
                        epoch,
                        eval_test_start_epoch,
                    )
                else:
                    if eval_test_loader is not None:
                        logging.info(
                            "Test eval status: dataset=%s latents=%s gt_mesh_dir=%s",
                            "ok" if test_dataset is not None else "missing",
                            "set" if test_latents is not None else "none",
                            eval_gt_mesh_dir if eval_gt_mesh_dir is not None else "missing",
                        )
                        test_sdf_loss = None
                        test_sap = None
                        test_cd = None
                        if eval_test_reconstruct:
                            subset_indices = (
                                eval_test_scene_idxs if eval_test_scene_idxs else None
                            )
                            test_latents, test_latent_recon = reconstruct_latents_for_dataset(
                                test_dataset,
                                sdf_decoder,
                                data_source,
                                latent_size,
                                clamp_dist,
                                eval_test_num_samples,
                                eval_test_optimization_steps,
                                eval_test_latent_lr,
                                eval_test_latent_l2reg,
                                eval_test_latent_init_std,
                                scene_indices=subset_indices,
                            )
                            last_test_latent_recon = test_latent_recon
                            summary_writer.add_scalar(
                                "Loss/test_latent_recon", test_latent_recon, global_step=epoch
                            )

                        if test_latents is not None:
                            logging.info("Test latents shape: %s", tuple(test_latents.shape))
                        else:
                            logging.info(
                                "Test latents not provided; skipping VAE recon loss on test."
                            )

                        try:
                            sample_idx = (
                                eval_test_scene_idxs[0] if eval_test_scene_idxs else 0
                            )
                            device = next(vae.parameters()).device
                            if vae_input_mode == "points":
                                if test_dataset is not None and getattr(test_dataset, "surface_points", None):
                                    sample_points = torch.as_tensor(
                                        test_dataset.surface_points[sample_idx]
                                    ).unsqueeze(0).to(device)
                                    with torch.no_grad():
                                        vae_out = vae(sample_points)
                                    logging.info(
                                        "Test VAE shapes: points=%s mu=%s z_hat=%s",
                                        tuple(sample_points.shape),
                                        tuple(vae_out["mu"].shape),
                                        tuple(vae_out["z_hat"].shape),
                                    )
                            else:
                                if test_latents is not None:
                                    sample_latent = test_latents[sample_idx : sample_idx + 1].to(device)
                                    with torch.no_grad():
                                        vae_out = vae(sample_latent)
                                    logging.info(
                                        "Test VAE shapes: latent_in=%s mu=%s z_hat=%s",
                                        tuple(sample_latent.shape),
                                        tuple(vae_out["mu"].shape),
                                        tuple(vae_out["z_hat"].shape),
                                    )
                        except Exception as exc:
                            logging.warning("Test VAE shape logging failed: %s", exc)

                        subset_indices = (
                            eval_test_scene_idxs if eval_test_scene_idxs else None
                        )
                        compute_latent_label_correlation(
                            test_dataset,
                            test_latents,
                            epoch,
                            "test",
                            sap_corr_label_map,
                            scene_indices=subset_indices,
                        )
                        print_latent_diagnosis_table(
                            test_dataset,
                            test_latents,
                            epoch,
                            "test",
                            sap_corr_label_map,
                            scene_indices=subset_indices,
                        )
                        eval_metrics = run_eval(
                            eval_test_loader,
                            test_latents,
                            epoch,
                            "eval_test",
                            kl_weight,
                            code_reg_weight,
                        )
                        if eval_metrics is not None:
                            last_test_eval_sdf = eval_metrics.get("eval_sdf_loss")
                            last_test_eval_epoch = epoch
                            test_sdf_loss = last_test_eval_sdf
                        test_metrics = compute_disentanglement_metrics(
                            eval_test_loader,
                            test_latents,
                            epoch,
                            "test",
                            sap_corr_label_map,
                            test_dataset.npyfiles if test_dataset is not None else [],
                        )
                        if test_metrics and test_metrics.get("sap") is not None:
                            last_test_sap = test_metrics["sap"]
                            test_sap = test_metrics["sap"]
                        elif compute_sap:
                            logging.error(
                                "Test SAP unavailable; check SAPCORRLabelsFile or LabelIndex."
                            )
                        if vae_input_mode == "latent" and test_latents is None:
                            logging.error(
                                "Test latents missing; skipping test mesh generation."
                            )
                        else:
                            generate_eval_meshes(
                                test_dataset,
                                test_latents,
                                mesh_test_scene_idxs,
                                "test",
                                epoch,
                            )
                        if eval_gt_mesh_dir is None:
                            logging.error("EvalGTMeshDir not set; skipping test Chamfer.")
                        else:
                            if vae_input_mode == "latent" and test_latents is None:
                                logging.error(
                                    "Test latents missing; skipping test Chamfer."
                                )
                            else:
                                test_cd = compute_chamfer_for_scenes(
                                    test_dataset,
                                    test_latents,
                                    mesh_test_scene_idxs,
                                    "test",
                                    epoch,
                                )
                            if test_cd is not None:
                                last_test_cd = test_cd

                        def _fmt_metric(val):
                            if val is None:
                                return "n/a"
                            if isinstance(val, dict):
                                parts = []
                                if "disease" in val:
                                    parts.append("disease={:.4f}".format(val["disease"]))
                                if "age" in val:
                                    parts.append("age={:.4f}".format(val["age"]))
                                return "{" + ", ".join(parts) + "}" if parts else "n/a"
                            try:
                                return "{:.6f}".format(val)
                            except Exception:
                                return str(val)

                        logging.info(
                            "Epoch %d test summary: eval_count=%d mesh_count=%d "
                            "test_sdf_loss=%s test_sap=%s test_cd=%s test_latent_recon=%s",
                            epoch,
                            len(eval_test_scene_idxs) if eval_test_scene_idxs else 0,
                            len(mesh_test_scene_idxs) if mesh_test_scene_idxs else 0,
                            _fmt_metric(test_sdf_loss),
                            _fmt_metric(test_sap),
                            _fmt_metric(test_cd),
                            _fmt_metric(last_test_latent_recon),
                        )

            summary_writer.add_scalar("Time/epoch (min)", (time.time() - epoch_time_start) / 60, epoch)
            summary_writer.flush()

    except KeyboardInterrupt:
        logging.error("Received KeyboardInterrupt. Cleaning up and ending training.")
    finally:
        summary_writer.flush()
        summary_writer.close()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a Residual MLP VAE + DeepSDF")
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
        help="A snapshot to continue from. This can be 'latest' to continue "
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))
