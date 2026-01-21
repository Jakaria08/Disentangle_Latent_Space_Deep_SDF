#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import os
import json
import time
import logging
import random
import numpy as np

import deep_sdf
from deep_sdf import lr_scheduling, loss as deep_sdf_loss, mesh, metrics
import deep_sdf.workspace as ws
from sdf_utils import sap as sap_metric
from sdf_utils import dci as dci_metric
from sdf_utils import mig as mig_metric

from networks import residual_mlp_vae
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
    attr_log_epoch,
    cov_log_epoch,
    corr_leak_log_epoch,
    cross_cov_log_epoch,
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
            "attr_epoch": attr_log_epoch,
            "cov_epoch": cov_log_epoch,
            "corr_leak_epoch": corr_leak_log_epoch,
            "cross_cov_epoch": cross_cov_log_epoch,
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
        data.get("attr_epoch", []),
        data.get("cov_epoch", []),
        data.get("corr_leak_epoch", []),
        data.get("cross_cov_epoch", []),
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
    attr_log_epoch,
    cov_log_epoch,
    corr_leak_log_epoch,
    cross_cov_log_epoch,
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
    attr_log_epoch = attr_log_epoch[:epoch]
    cov_log_epoch = cov_log_epoch[:epoch]
    corr_leak_log_epoch = corr_leak_log_epoch[:epoch]
    cross_cov_log_epoch = cross_cov_log_epoch[:epoch]
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
        attr_log_epoch,
        cov_log_epoch,
        corr_leak_log_epoch,
        cross_cov_log_epoch,
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


def compute_vae_latents(vae, teacher_latents, batch_size, device):
    was_training = vae.training
    vae.eval()
    latent_chunks = []
    with torch.no_grad():
        for start in range(0, teacher_latents.shape[0], batch_size):
            chunk = teacher_latents[start : start + batch_size].to(device)
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
    elif len(batch) == 2:
        sdf_data, indices = batch
        labels = None
    else:
        raise ValueError("Unexpected batch structure from DataLoader")
    return sdf_data, indices, labels


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
    snnl_temp = get_spec_with_default(specs, "SNNLTemp", 181.0)
    snnl_weight = get_spec_with_default(specs, "SNNLWeight", 0.5)
    attr_weight = get_spec_with_default(specs, "AttributeWeight", 0.5)
    covariance_loss = get_spec_with_default(specs, "CovarianceLoss", False)
    covariance_lambda = get_spec_with_default(specs, "CovarianceLossLambda", 1.0)
    label_index = get_spec_with_default(specs, "LabelIndex", 0)
    attribute_latent_index = get_spec_with_default(specs, "AttributeLatentIndex", 0)
    snnl_target_dim = get_spec_with_default(specs, "SNNLTargetDim", 0)
    corr_leakage_loss = get_spec_with_default(specs, "CorrLeakageLoss", False)
    corr_leakage_lambda = get_spec_with_default(specs, "CorrLeakageLambda", 1.0)
    cross_cov_loss = get_spec_with_default(specs, "CrossCovLoss", False)
    cross_cov_lambda = get_spec_with_default(specs, "CrossCovLambda", 1.0)
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

    compute_sap = get_spec_with_default(specs, "ComputeSAP", False)
    sap_regression = get_spec_with_default(specs, "SAPRegression", False)
    sap_continuous = get_spec_with_default(specs, "SAPContinuousFactors", True)
    sap_nb_bins = get_spec_with_default(specs, "SAPNumBins", 10)
    sap_label_indices = get_spec_with_default(specs, "SAPLabelIndices", None)
    sap_corr_extra_frequency = get_spec_with_default(specs, "SAPCORRExtraFrequency", 0)
    sap_corr_labels_file = get_spec_with_default(specs, "SAPCORRLabelsFile", "labels.pt")



    use_labels = get_spec_with_default(specs, "ReturnLabels", None)
    if use_labels is None:
        use_labels = (
            guided_contrastive_loss
            or attribute_loss
            or corr_leakage_loss
            or compute_sap
        )
    labels_filename = get_spec_with_default(specs, "LabelsFile", "labels.pt")

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

    if torch.cuda.device_count() > 1:
        vae = torch.nn.DataParallel(vae)
        sdf_decoder = torch.nn.DataParallel(sdf_decoder)

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

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

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source,
        train_split,
        num_samp_per_scene,
        load_ram=load_ram,
        return_labels=use_labels,
        labels_filename=labels_filename,
    )

    num_scenes = len(sdf_dataset)
    if teacher_latents.shape[0] != num_scenes:
        raise Exception(
            "Pretrained latent count does not match number of scenes: {} vs {}".format(
                teacher_latents.shape[0], num_scenes
            )
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
        )
        test_latents_path = get_spec_with_default(specs, "TestLatentPath", None)
        test_latents_path = resolve_spec_path(experiment_directory, test_latents_path)
        if test_latents_path is None:
            if not eval_test_reconstruct:
                logging.warning(
                    "TestSplit provided but TestLatentPath not set; skipping test evaluation."
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

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    if (
        guided_contrastive_loss
        or attribute_loss
        or corr_leakage_loss
        or compute_sap
    ) and not use_labels:
        raise Exception("Label-based losses/SAP requested but ReturnLabels is disabled.")

    sap_corr_label_map = None
    if compute_sap or (sap_corr_extra_frequency is not None and sap_corr_extra_frequency > 0):
        sapcorr_path = _resolve_labels_path(data_source, sap_corr_labels_file)
        sap_corr_label_map = _load_label_map(sapcorr_path, sdf_dataset.npyfiles)

    pseudo_label_map = None
    real_label_map = None
    if label_mix_enabled:
        pseudo_path = _resolve_labels_path(data_source, pseudo_labels_file)
        real_path = _resolve_labels_path(data_source, real_labels_file)
        if mix_pseudo_start > 0.0:
            pseudo_label_map = _load_label_map(pseudo_path, sdf_dataset.npyfiles)
        if mix_real_start > 0.0:
            real_label_map = _load_label_map(real_path, sdf_dataset.npyfiles)

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
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

    def select_mesh_indices(dataset, scene_count):
        if dataset is None or scene_count is None or scene_count <= 0:
            return []
        count = min(scene_count, len(dataset))
        return random.sample(range(len(dataset)), count)

    eval_train_loader = None
    if eval_train_frequency is not None and eval_train_frequency > 0:
        eval_train_loader = build_eval_loader(
            sdf_dataset, eval_train_scene_num, "train"
        )

    eval_test_scene_idxs = select_eval_indices(test_dataset, eval_test_scene_num)
    eval_test_loader = None
    if eval_test_frequency is not None and eval_test_frequency > 0:
        if test_dataset is None or (test_latents is None and not eval_test_reconstruct):
            logging.warning(
                "EvalTestFrequency set but test data or latents missing; skipping test evaluation."
            )
        elif eval_test_scene_idxs:
            eval_test_loader = build_eval_loader_from_indices(
                test_dataset, eval_test_scene_idxs, "test"
            )
        else:
            logging.warning(
                "EvalTestFrequency set but no eval test indices; skipping test evaluation."
            )

    eval_train_scene_idxs = select_mesh_indices(sdf_dataset, mesh_train_scene_num)
    mesh_test_scene_idxs = select_mesh_indices(test_dataset, mesh_test_scene_num)

    sap_train_loader = None
    sap_test_loader = None
    if compute_sap and sap_corr_extra_frequency is not None and sap_corr_extra_frequency > 0:
        sap_train_loader = build_eval_loader(sdf_dataset, 0, "train_sap")
        if test_dataset is not None and test_latents is not None:
            sap_test_loader = build_eval_loader(test_dataset, 0, "test_sap")

    lr_schedules = lr_scheduling.get_learning_rate_schedules(specs)

    vae_lr = lr_schedules[0].get_learning_rate(0)
    params = [{"params": vae.parameters(), "lr": vae_lr}]

    if train_sdf_decoder:
        sdf_lr_schedule = lr_schedules[1] if len(lr_schedules) > 1 else lr_schedules[0]
        params.append({"params": sdf_decoder.parameters(), "lr": sdf_lr_schedule.get_learning_rate(0)})

    optimizer = torch.optim.Adam(params)

    summary_writer = SummaryWriter(log_dir=os.path.join(experiment_directory, ws.tb_logs_dir))

    snn_loss_fn = (
        deep_sdf_loss.SNNLossCls(T=snnl_temp, target_dim=snnl_target_dim)
        if guided_contrastive_loss
        else None
    )
    attr_loss_fn = deep_sdf_loss.AttributeLoss() if attribute_loss else None
    cov_loss_fn = (
        deep_sdf_loss.DIPVAEIILoss(beta=covariance_lambda)
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
    attr_log_epoch = []
    cov_log_epoch = []
    corr_leak_log_epoch = []
    cross_cov_log_epoch = []
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
            attr_log_epoch,
            cov_log_epoch,
            corr_leak_log_epoch,
            cross_cov_log_epoch,
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
                attr_log_epoch,
                cov_log_epoch,
                corr_leak_log_epoch,
                cross_cov_log_epoch,
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
                attr_log_epoch,
                cov_log_epoch,
                corr_leak_log_epoch,
                cross_cov_log_epoch,
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
        vae_latents = compute_vae_latents(vae, teacher_latents, latent_batch, device)
        save_latent_vectors(experiment_directory, "latest.pth", vae_latents, epoch)

    def save_checkpoints(epoch):
        filename = str(epoch) + ".pth"
        save_model(experiment_directory, filename, vae, sdf_decoder, epoch)
        save_optimizer(experiment_directory, filename, optimizer, epoch)
        latent_batch = get_spec_with_default(specs, "LatentExportBatchSize", 1024)
        device = next(vae.parameters()).device
        vae_latents = compute_vae_latents(vae, teacher_latents, latent_batch, device)
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
        if eval_loader is None or eval_latents is None:
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
                sdf_data, indices, _labels = _unpack_batch(batch)
                sdf_data = sdf_data.reshape(sdf_data.shape[0], -1, 4)

                xyz = sdf_data[:, :, 0:3].to(device)
                sdf_gt = sdf_data[:, :, 3].unsqueeze(-1).to(device)

                if enforce_minmax:
                    sdf_gt = torch.clamp(sdf_gt, minT, maxT)

                indices = indices.long()
                teacher_batch = eval_latents[indices].to(device)

                vae_out = vae(teacher_batch)
                mu = vae_out["mu"]
                logvar = vae_out["logvar"]
                z_hat = vae_out["z_hat"]

                vae_total, vae_recon, vae_kl = residual_mlp_vae.vae_loss(
                    z_hat,
                    teacher_batch,
                    mu,
                    logvar,
                    recon_weight=vae_recon_weight,
                    kl_weight=kl_weight,
                    recon_loss=recon_loss_type,
                )

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

                batch_total_loss = (
                    sdf_loss_weight * (batch_sdf_loss + batch_sdf_reg) + vae_total.item()
                )
                eval_losses.append(batch_total_loss)
                eval_sdf_losses.append(batch_sdf_loss)
                eval_sdf_reg_losses.append(batch_sdf_reg)
                eval_vae_recon.append(vae_recon.item())
                eval_vae_kl.append(vae_kl.item())
                eval_vae_lat_mag.append(torch.mean(torch.norm(mu, dim=1)).item())

        eval_metrics = None
        if eval_losses:
            eval_loss = sum(eval_losses) / len(eval_losses)
            eval_sdf_loss = sum(eval_sdf_losses) / len(eval_sdf_losses)
            eval_sdf_reg = sum(eval_sdf_reg_losses) / len(eval_sdf_reg_losses)
            eval_vae_recon_loss = sum(eval_vae_recon) / len(eval_vae_recon)
            eval_vae_kl_loss = sum(eval_vae_kl) / len(eval_vae_kl)
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

    def _collect_factors_codes(eval_loader, eval_latents, split_label, label_map, npyfiles):
        if eval_loader is None or eval_latents is None:
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
                _sdf_data, indices, _labels = _unpack_batch(batch)
                labels = _labels_for_indices(npyfiles, label_map, indices)
                if labels is None:
                    continue
                indices = indices.long()
                labels = labels.view(labels.shape[0], -1)
                teacher_batch = eval_latents[indices].to(device)
                vae_out = vae(teacher_batch)
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
        if sap_label_indices is not None:
            indices = sap_label_indices
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

    def compute_disentanglement_metrics(
        eval_loader, eval_latents, epoch, split_label, label_map, npyfiles
    ):
        if eval_loader is None or eval_latents is None or not compute_sap:
            return {}

        factors_np, codes_vae_np = _collect_factors_codes(
            eval_loader, eval_latents, split_label, label_map, npyfiles
        )
        if factors_np is None:
            return {}

        sap_vae = sap_metric.sap(
            factors_np,
            codes_vae_np,
            continuous_factors=sap_continuous,
            nb_bins=sap_nb_bins,
            regression=sap_regression,
        )
        dci_scores = dci_metric.dci(
            factors_np,
            codes_vae_np,
            continuous_factors=sap_continuous,
        )
        mig_scores = mig_metric.mig(
            factors_np,
            codes_vae_np,
            continuous_factors=sap_continuous,
            continuous_codes=True,
            nb_bins=sap_nb_bins,
        )

        summary_writer.add_scalar(f"SAP/vae_{split_label}", sap_vae, global_step=epoch)
        summary_writer.add_scalar(
            f"DCI/vae_{split_label}_disentanglement",
            dci_scores["disentanglement"],
            global_step=epoch,
        )
        summary_writer.add_scalar(
            f"DCI/vae_{split_label}_completeness",
            dci_scores["completeness"],
            global_step=epoch,
        )
        summary_writer.add_scalar(
            f"DCI/vae_{split_label}_informativeness",
            dci_scores["informativeness"],
            global_step=epoch,
        )
        summary_writer.add_scalar(
            f"MIG/vae_{split_label}",
            mig_scores["mig"],
            global_step=epoch,
        )

        logging.info(
            "Epoch {} metrics ({}): SAP={:.6f} DCI(d,c,i)=({:.6f},{:.6f},{:.6f}) MIG={:.6f}".format(
                epoch,
                split_label,
                sap_vae,
                dci_scores["disentanglement"],
                dci_scores["completeness"],
                dci_scores["informativeness"],
                mig_scores["mig"],
            )
        )
        return {
            "sap": sap_vae,
            "dci": dci_scores,
            "mig": mig_scores,
        }

    def generate_eval_meshes(dataset, eval_latents, scene_indices, split_label, epoch):
        if dataset is None or eval_latents is None or not scene_indices:
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
                teacher_latent = eval_latents[scene_idx : scene_idx + 1].to(device)
                vae_out = vae(teacher_latent)
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
            or eval_latents is None
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
            teacher_latent = eval_latents[scene_idx : scene_idx + 1].to(device)
            with torch.no_grad():
                vae_out = vae(teacher_latent)
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
        if dataset is None or eval_latents is None:
            return
        labels_np = _collect_label_values(dataset.npyfiles, label_map, label_index)
        if labels_np is None:
            return

        latent_batch = get_spec_with_default(specs, "LatentExportBatchSize", 1024)
        device = next(vae.parameters()).device
        vae_latents = compute_vae_latents(vae, eval_latents, latent_batch, device).cpu().numpy()

        if scene_indices is not None:
            scene_indices = np.asarray(scene_indices, dtype=int)
            labels_np = labels_np[scene_indices]
            vae_latents = vae_latents[scene_indices]

        if vae_latents.shape[0] != labels_np.shape[0]:
            logging.warning(
                "Correlation skipped ({}): latent count {} != label count {}".format(
                    split_label, vae_latents.shape[0], labels_np.shape[0]
                )
            )
            return

        latent0 = vae_latents[:, 0]
        mask = np.isfinite(labels_np) & (labels_np != -1)
        if mask.sum() < 2:
            logging.warning(
                "Correlation skipped ({}): insufficient valid labels.".format(split_label)
            )
            return

        latent0 = latent0[mask]
        labels_np = labels_np[mask]
        if np.std(latent0) == 0 or np.std(labels_np) == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(latent0, labels_np)[0, 1])

        summary_writer.add_scalar(
            f"Correlation/{split_label}_latent0_label", corr, global_step=epoch
        )
        logging.info(
            "Epoch {} correlation ({}): latent0 vs label[{}] = {:.6f}".format(
                epoch, split_label, label_index, corr
            )
        )

    def print_latent_diagnosis_table(
        dataset, eval_latents, epoch, split_label, label_map, scene_indices=None
    ):
        if dataset is None or eval_latents is None:
            return
        labels_np = _collect_label_values(dataset.npyfiles, label_map, label_index)
        if labels_np is None:
            return

        latent_batch = get_spec_with_default(specs, "LatentExportBatchSize", 1024)
        device = next(vae.parameters()).device
        vae_latents = compute_vae_latents(vae, eval_latents, latent_batch, device).cpu().numpy()

        if scene_indices is not None:
            scene_indices = np.asarray(scene_indices, dtype=int)
            labels_np = labels_np[scene_indices]
            vae_latents = vae_latents[scene_indices]

        if vae_latents.shape[0] != labels_np.shape[0]:
            logging.warning(
                "Latent table skipped ({}): latent count {} != label count {}".format(
                    split_label, vae_latents.shape[0], labels_np.shape[0]
                )
            )
            return

        mask = np.isfinite(labels_np) & (labels_np != -1)
        if mask.sum() < 2:
            logging.warning(
                "Latent table skipped ({}): insufficient valid labels.".format(split_label)
            )
            return

        labels_np = labels_np[mask].astype(int)
        latents = vae_latents[mask]

        logging.info(
            "Epoch {} latent vs diagnosis table ({}):".format(epoch, split_label)
        )
        logging.info("  dim | corr | best_acc")
        for dim in range(latents.shape[1]):
            x = latents[:, dim]
            if np.std(x) == 0 or np.std(labels_np) == 0:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(x, labels_np)[0, 1])
            acc = _best_threshold_accuracy(x, labels_np)
            logging.info("  {:>3d} | {:>6.3f} | {:>8.3f}".format(dim, corr, acc))

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
            epoch_attr = []
            epoch_cov = []
            epoch_corr_leak = []
            epoch_cross_cov = []

            logging.info("epoch {}...".format(epoch))

            vae.train()
            if train_sdf_decoder:
                sdf_decoder.train()
            else:
                sdf_decoder.eval()

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
                sdf_data, indices, labels = _unpack_batch(batch)
                sdf_data = sdf_data.reshape(sdf_data.shape[0], -1, 4)

                sdf_data.requires_grad = False

                xyz = sdf_data[:, :, 0:3].cuda()
                sdf_gt = sdf_data[:, :, 3].unsqueeze(-1).cuda()

                if enforce_minmax:
                    sdf_gt = torch.clamp(sdf_gt, minT, maxT)

                indices = indices.long()
                teacher_batch = teacher_latents[indices].cuda()

                vae_out = vae(teacher_batch)
                mu = vae_out["mu"]
                logvar = vae_out["logvar"]
                z_hat = vae_out["z_hat"]

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
                attr_loss_val = 0.0
                cov_loss_val = 0.0
                corr_leak_loss_val = 0.0
                cross_cov_loss_val = 0.0
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
                            if pseudo_labels.shape[1] <= label_index:
                                raise RuntimeError(
                                    "Pseudo labels missing label_index {} (shape {}).".format(
                                        label_index, pseudo_labels.shape
                                    )
                                )
                            label_values[pseudo_mask] = pseudo_labels[
                                pseudo_mask, label_index
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
                            if real_labels.shape[1] <= label_index:
                                raise RuntimeError(
                                    "Real labels missing label_index {} (shape {}).".format(
                                        label_index, real_labels.shape
                                    )
                                )
                            label_values[real_mask] = real_labels[
                                real_mask, label_index
                            ].to(torch.float32)
                    else:
                        if labels is None:
                            raise RuntimeError("Label-based losses enabled but labels are missing in batch.")
                        labels = labels.to(mu.device).view(labels.shape[0], -1)
                        if labels.shape[1] <= label_index:
                            raise RuntimeError(
                                "Labels missing label_index {} (shape {}).".format(
                                    label_index, labels.shape
                                )
                            )
                        label_values = labels[:, label_index].to(torch.float32)

                    valid_mask = torch.isfinite(label_values) & (label_values != -1)
                    if valid_mask.any():
                        if guided_contrastive_loss and valid_mask.sum().item() > 1:
                            snnl_loss = snn_loss_fn(
                                mu[valid_mask], label_values[valid_mask]
                            )
                            vae_total = vae_total + (snnl_weight * snnl_loss)
                            snnl_loss_val = snnl_loss.item()
                        if attribute_loss:
                            attr_latent = mu[valid_mask, attribute_latent_index]
                            attr_loss = attr_loss_fn(
                                attr_latent, label_values[valid_mask]
                            )
                            vae_total = vae_total + (attr_weight * attr_loss)
                            attr_loss_val = attr_loss.item()
                        if corr_leakage_loss:
                            leak_loss = deep_sdf_loss.corr_leakage_penalty(
                                mu[valid_mask],
                                label_values[valid_mask],
                                leakage_target_dim,
                            )
                            vae_total = vae_total + (corr_leakage_lambda * leak_loss)
                            corr_leak_loss_val = leak_loss.item()
                        if cross_cov_loss:
                            cross_loss = deep_sdf_loss.cross_cov_penalty(
                                mu[valid_mask],
                                leakage_target_dim,
                            )
                            vae_total = vae_total + (cross_cov_lambda * cross_loss)
                            cross_cov_loss_val = cross_loss.item()

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
                if attribute_loss:
                    epoch_attr.append(attr_loss_val)
                if covariance_loss:
                    epoch_cov.append(cov_loss_val)
                if corr_leakage_loss:
                    epoch_corr_leak.append(corr_leak_loss_val)
                if cross_cov_loss:
                    epoch_cross_cov.append(cross_cov_loss_val)

            seconds_elapsed = time.time() - epoch_time_start
            timing_log.append(seconds_elapsed)

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_sdf_loss = sum(epoch_sdf_losses) / len(epoch_sdf_losses)
            epoch_sdf_reg = sum(epoch_sdf_reg_losses) / len(epoch_sdf_reg_losses)
            epoch_vae_recon_loss = sum(epoch_vae_recon) / len(epoch_vae_recon)
            epoch_vae_kl_loss = sum(epoch_vae_kl) / len(epoch_vae_kl)
            epoch_vae_lat_mag = sum(epoch_vae_lat_mag) / len(epoch_vae_lat_mag)
            epoch_snnl_loss = sum(epoch_snnl) / len(epoch_snnl) if epoch_snnl else 0.0
            epoch_attr_loss = sum(epoch_attr) / len(epoch_attr) if epoch_attr else 0.0
            epoch_cov_loss = sum(epoch_cov) / len(epoch_cov) if epoch_cov else 0.0
            epoch_corr_leak_loss = (
                sum(epoch_corr_leak) / len(epoch_corr_leak) if epoch_corr_leak else 0.0
            )
            epoch_cross_cov_loss = (
                sum(epoch_cross_cov) / len(epoch_cross_cov) if epoch_cross_cov else 0.0
            )
            epoch_sdf_weighted = sdf_loss_weight * (epoch_sdf_loss + epoch_sdf_reg)
            epoch_vae_recon_weighted = vae_recon_weight * epoch_vae_recon_loss
            epoch_vae_kl_weighted = kl_weight * epoch_vae_kl_loss

            if use_kl:
                logging.info(
                    "Epoch {} loss: {:.6f} | sdf: {:.6f} | sdf_reg: {:.6f} | "
                    "vae_recon: {:.6f} | vae_kl: {:.6f} | "
                    "weighted -> sdf: {:.6f} | vae_recon: {:.6f} | vae_kl: {:.6f}".format(
                        epoch,
                        epoch_loss,
                        epoch_sdf_loss,
                        epoch_sdf_reg,
                        epoch_vae_recon_loss,
                        epoch_vae_kl_loss,
                        epoch_sdf_weighted,
                        epoch_vae_recon_weighted,
                        epoch_vae_kl_weighted,
                    )
                )
            else:
                logging.info(
                    "Epoch {} loss: {:.6f} | sdf: {:.6f} | sdf_reg: {:.6f} | "
                    "vae_recon: {:.6f} | weighted -> sdf: {:.6f} | vae_recon: {:.6f}".format(
                        epoch,
                        epoch_loss,
                        epoch_sdf_loss,
                        epoch_sdf_reg,
                        epoch_vae_recon_loss,
                        epoch_sdf_weighted,
                        epoch_vae_recon_weighted,
                    )
                )
            if (
                guided_contrastive_loss
                or attribute_loss
                or covariance_loss
                or corr_leakage_loss
                or cross_cov_loss
            ):
                logging.info(
                    "Epoch {} extra losses: snnl: {:.6f} | attr: {:.6f} | cov: {:.6f} | leak: {:.6f} | cross_cov: {:.6f}".format(
                        epoch,
                        epoch_snnl_loss,
                        epoch_attr_loss,
                        epoch_cov_loss,
                        epoch_corr_leak_loss,
                        epoch_cross_cov_loss,
                    )
                )

            loss_log_epoch.append(epoch_loss)
            sdf_loss_log_epoch.append(epoch_sdf_loss)
            sdf_reg_log_epoch.append(epoch_sdf_reg)
            vae_recon_log_epoch.append(epoch_vae_recon_loss)
            vae_kl_log_epoch.append(epoch_vae_kl_loss)
            vae_lat_mag_log.append(epoch_vae_lat_mag)
            snnl_log_epoch.append(epoch_snnl_loss)
            attr_log_epoch.append(epoch_attr_loss)
            cov_log_epoch.append(epoch_cov_loss)
            corr_leak_log_epoch.append(epoch_corr_leak_loss)
            cross_cov_log_epoch.append(epoch_cross_cov_loss)

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
            if attribute_loss:
                summary_writer.add_scalar("Loss/train_attr", epoch_attr_loss, global_step=epoch)
            if covariance_loss:
                summary_writer.add_scalar("Loss/train_cov", epoch_cov_loss, global_step=epoch)
            if corr_leakage_loss:
                summary_writer.add_scalar("Loss/train_leak", epoch_corr_leak_loss, global_step=epoch)
            if cross_cov_loss:
                summary_writer.add_scalar("Loss/train_cross_cov", epoch_cross_cov_loss, global_step=epoch)

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
                    attr_log_epoch,
                    cov_log_epoch,
                    corr_leak_log_epoch,
                    cross_cov_log_epoch,
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
                train_metrics = compute_disentanglement_metrics(
                    eval_train_loader,
                    teacher_latents,
                    epoch,
                    "train",
                    sap_corr_label_map,
                    sdf_dataset.npyfiles,
                )
                if train_metrics and train_metrics.get("sap") is not None:
                    last_train_sap = train_metrics["sap"]
                train_eval_indices = None
                if hasattr(eval_train_loader.dataset, "indices"):
                    train_eval_indices = eval_train_loader.dataset.indices
                compute_latent_label_correlation(
                    sdf_dataset,
                    teacher_latents,
                    epoch,
                    "train",
                    sap_corr_label_map,
                    scene_indices=train_eval_indices,
                )
                print_latent_diagnosis_table(
                    sdf_dataset,
                    teacher_latents,
                    epoch,
                    "train",
                    sap_corr_label_map,
                    scene_indices=train_eval_indices,
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
                    logging.info(
                        "Epoch {} extra summary: train_sdf_loss={} train_sap={} train_cd={} test_sdf_loss={} test_sap={} test_cd={}".format(
                            epoch,
                            "{:.6f}".format(last_train_eval_sdf)
                            if last_train_eval_sdf is not None
                            else "n/a",
                            "{:.6f}".format(last_train_sap)
                            if last_train_sap is not None
                            else "n/a",
                            "{:.6f}".format(last_train_cd)
                            if last_train_cd is not None
                            else "n/a",
                            "{:.6f}".format(last_test_eval_sdf)
                            if last_test_eval_sdf is not None
                            else "n/a",
                            "{:.6f}".format(last_test_sap)
                            if last_test_sap is not None
                            else "n/a",
                            "{:.6f}".format(last_test_cd)
                            if last_test_cd is not None
                            else "n/a",
                        )
                    )

            if (
                eval_test_loader is not None
                and eval_test_frequency is not None
                and eval_test_frequency > 0
                and epoch % eval_test_frequency == 0
            ):
                logging.info(
                    "Test eval status: dataset=%s latents=%s gt_mesh_dir=%s",
                    "ok" if test_dataset is not None else "missing",
                    "set" if test_latents is not None else "none",
                    eval_gt_mesh_dir if eval_gt_mesh_dir is not None else "missing",
                )
                if epoch < eval_test_start_epoch:
                    logging.info(
                        "Skipping test eval at epoch %d (start epoch %d).",
                        epoch,
                        eval_test_start_epoch,
                    )
                else:
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

                    if test_latents is None:
                        logging.error("Test latents unavailable; skipping test eval.")
                    else:
                        logging.info(
                            "Test latents shape: %s",
                            tuple(test_latents.shape),
                        )
                        try:
                            if test_latents.shape[0] > 0:
                                sample_idx = (
                                    eval_test_scene_idxs[0]
                                    if eval_test_scene_idxs
                                    else 0
                                )
                                device = next(vae.parameters()).device
                                sample_latent = test_latents[
                                    sample_idx : sample_idx + 1
                                ].to(device)
                                with torch.no_grad():
                                    vae_out = vae(sample_latent)
                                logging.info(
                                    "Test VAE shapes: teacher=%s mu=%s z_hat=%s",
                                    tuple(sample_latent.shape),
                                    tuple(vae_out["mu"].shape),
                                    tuple(vae_out["z_hat"].shape),
                                )
                        except Exception as exc:
                            logging.warning(
                                "Test VAE shape logging failed: %s", exc
                            )
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
                        return "n/a" if val is None else "{:.6f}".format(val)

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
