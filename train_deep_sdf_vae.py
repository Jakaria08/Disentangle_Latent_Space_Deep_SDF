#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
torch.autograd.set_detect_anomaly(True)
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
import numpy as np
import torch.autograd as autograd
import warnings

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

import deep_sdf
from deep_sdf import mesh, metrics, lr_scheduling, plotting, utils, loss, data
import deep_sdf.workspace as ws
import reconstruct
import networks.sdf_vae as vae
from torch.utils.tensorboard import SummaryWriter
from sdf_utils import sap as sap_metric

guided_contrastive_loss = False
guided_contrastive_loss_cls = False
unsupervised_contrastive_loss = False
attribute_loss = False
kl_div_loss = False
jacobian_loss = False
dip_vae_loss = False
PretrainedModel = False
annealing_epochs = 1
beta_final = 0.001
temp = 2
temp_reg = 2 # change this?
w_cls = 0.01
w_reg = 0.005
threshold = 0.1
w_code_reg = 0.8
w_jacobian = 1e-3

def calculate_correlations_all(latent_vectors, labels_cls, labels_reg):
    z = latent_vectors.detach().cpu().to(torch.float64)
    y_cls = labels_cls.detach().cpu().flatten().to(torch.float64)
    y_age = labels_reg.detach().cpu().flatten().to(torch.float64)

    # Direct correlations for supervised dimensions
    z0_disease_corr = torch.corrcoef(torch.stack([z[:, 0], y_cls]))[0, 1]
    z1_age_corr = torch.corrcoef(torch.stack([z[:, 1], y_age]))[0, 1]

    # Other dimensions (excluding supervised ones)
    other_idx = list(range(2, z.shape[1]))  # [2, 3, 4, ..., D-1]
    
    # Calculate correlations for all other dimensions
    age_corr_others = {}
    disease_corr_others = {}
    
    for d in other_idx:
        age_corr_others[int(d)] = torch.corrcoef(torch.stack([z[:, d], y_age]))[0, 1].item()
        disease_corr_others[int(d)] = torch.corrcoef(torch.stack([z[:, d], y_cls]))[0, 1].item()
    
    # Find second highest correlations (by absolute value)
    age_corrs_abs = sorted(age_corr_others.values(), key=abs, reverse=True)
    disease_corrs_abs = sorted(disease_corr_others.values(), key=abs, reverse=True)
    
    # Get second highest (or first if only one dimension exists)
    second_highest_age_corr = age_corrs_abs[1] if len(age_corrs_abs) > 1 else age_corrs_abs[0] if age_corrs_abs else 0.0
    second_highest_disease_corr = disease_corrs_abs[1] if len(disease_corrs_abs) > 1 else disease_corrs_abs[0] if disease_corrs_abs else 0.0

    return z1_age_corr.item(), z0_disease_corr.item(), second_highest_age_corr, second_highest_disease_corr

def calculate_correlations(latent_vectors, labels_cls, labels_reg):
    """
    Calculate Pearson correlation between latent dimensions and labels
    """
    
    # Ensure tensors are on CPU and detached
    z = latent_vectors.detach().cpu()
    labels_cls = labels_cls.detach().cpu().flatten()
    labels_reg = labels_reg.detach().cpu().flatten()
    
    # Calculate correlations
    z0_disease_corr = torch.corrcoef(torch.stack([z[:, 0], labels_cls]))[0, 1]
    z1_age_corr = torch.corrcoef(torch.stack([z[:, 1], labels_reg]))[0, 1]
    
    return z0_disease_corr.item(), z1_age_corr.item()

def jacobian_penalty_JJT(surface_points, autoencoder):
    """
    Computes the Jacobian penalty ||JJ^T - I||^2_F for all shapes in the batch,
    encouraging the encoder to preserve local geometry.
    """
    # Get original shape info
    batch_size, num_points, dims = surface_points.shape
    
    # For computational efficiency, we'll compute the Jacobian penalties per shape
    # and then average them
    total_penalty = 0.0
    
    # Loop through each shape in the batch
    for i in range(batch_size):
        # Take a single shape
        single_shape = surface_points[i:i+1]  # Keep batch dimension: [1, 2048, 3]
        single_shape.requires_grad_(True)
        
        # Wrapper function for Jacobian calculation
        def encoder_wrapper(shape_input):
            # Shape_input has shape [1, 2048, 3]
            if kl_div_loss:
                mu, _ = autoencoder.encoder(shape_input)
                return mu[0]  # Remove batch dimension
            else:
                z = autoencoder.encoder(shape_input)
                return z[0]  # Remove batch dimension
        
        # Compute the Jacobian: J = d(z)/d(x) for this shape
        J = autograd.functional.jacobian(
            encoder_wrapper, 
            single_shape, 
            create_graph=True)
        
        # Reshape J to proper dimensions for matrix multiplication
        # J will have shape [latent_dim, 1, 2048, 3]
        # We want to flatten the last dimensions to [latent_dim, 2048*3]
        J = J.reshape(J.shape[0], -1)
        
        # Compute JJ^T for this shape
        JJT = J @ J.transpose(0, 1)
        
        # Create identity matrix of appropriate size
        latent_dim = JJT.shape[0]
        I = torch.eye(latent_dim).to(JJT.device)
        
        # Calculate penalty for this shape
        shape_penalty = torch.norm(JJT - I, p='fro')**2
        
        # Add to total
        total_penalty += shape_penalty
    
    # Return average penalty across all shapes
    return total_penalty / batch_size


def kl_divergence_loss(mu, logvar):
    logvar = torch.clamp(logvar, min=-3, max=3)  # Clamp logvar to prevent numerical issues
    #mu = torch.clamp(mu, min=-3, max=3)  # Clamp mu to prevent numerical issues
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def get_encoder_params_dir(experiment_directory, create=False):
    encoder_params_dir = os.path.join(experiment_directory, "EncoderParameters")
    if create:
        os.makedirs(encoder_params_dir, exist_ok=True)
    return encoder_params_dir


def save_encoder(experiment_directory, filename, encoder, epoch):
    encoder_params_dir = get_encoder_params_dir(experiment_directory, True)
    state = encoder.state_dict()
    torch.save(
        {"epoch": epoch, "encoder_state_dict": state},
        os.path.join(encoder_params_dir, filename),
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


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_sdf_decoder_weights(pretrained_path, sdf_decoder):
    if pretrained_path is None:
        return
    if not os.path.isfile(pretrained_path):
        raise Exception(f'pretrained SDF decoder "{pretrained_path}" does not exist')

    data = torch.load(pretrained_path, map_location="cpu")
    if isinstance(data, dict):
        if "model_state_dict" in data:
            state = data["model_state_dict"]
        elif "sdf_decoder_state_dict" in data:
            state = data["sdf_decoder_state_dict"]
        else:
            state = data
    else:
        state = data

    state = _strip_module_prefix(state)
    missing, unexpected = sdf_decoder.load_state_dict(state, strict=False)
    if missing:
        logging.warning(f"SDF decoder missing keys: {missing}")
    if unexpected:
        logging.warning(f"SDF decoder unexpected keys: {unexpected}")


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


def _discrete_entropy(y, eps=1e-12):
    values, counts = np.unique(y, return_counts=True)
    probs = counts.astype(np.float64) / max(len(y), 1)
    return float(-(probs * np.log(probs + eps)).sum())


def _safe_corrcoef(x, y):
    if x.shape[0] != y.shape[0]:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _best_acc_1d(z, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=200, solver="liblinear")
    clf.fit(z.reshape(-1, 1), y)
    y_pred = clf.predict(z.reshape(-1, 1))
    return float(accuracy_score(y, y_pred))


def compute_latent_label_table(latents, labels):
    # latents: [N, D], labels: [N]
    corrs = []
    accs = []
    for d in range(latents.shape[1]):
        z = latents[:, d]
        corrs.append(_safe_corrcoef(z, labels))
        accs.append(_best_acc_1d(z, labels))
    return corrs, accs


def compute_sap_score(latents, factors, continuous_factors):
    if latents.size == 0 or factors.size == 0:
        return float("nan")
    return float(
        sap_metric.sap(
            factors=factors,
            codes=latents,
            continuous_factors=continuous_factors,
            regression=continuous_factors,
        )
    )


def compute_dci_metrics(latents, factors, discrete_factors=True, eps=1e-12):
    # factors: [N, F], latents: [N, D]
    n_factors = factors.shape[1]
    n_latents = latents.shape[1]
    importance = np.zeros((n_factors, n_latents), dtype=np.float64)
    informativeness = []

    for f in range(n_factors):
        y = factors[:, f]
        if discrete_factors:
            if len(np.unique(y)) < 2:
                informativeness.append(float("nan"))
                continue
            clf = LogisticRegression(max_iter=500, solver="liblinear")
            clf.fit(latents, y)
            y_pred = clf.predict(latents)
            informativeness.append(accuracy_score(y, y_pred))
            importance[f, :] = np.abs(clf.coef_).reshape(-1)
        else:
            reg = LinearRegression()
            reg.fit(latents, y)
            y_pred = reg.predict(latents)
            informativeness.append(_safe_corrcoef(y, y_pred))
            importance[f, :] = np.abs(reg.coef_).reshape(-1)

    code_importance = importance.sum(axis=0)
    if code_importance.sum() < eps:
        return 0.0, 0.0, 0.0, importance

    p = importance / (code_importance[None, :] + eps)
    entropy_per_code = -np.sum(p * np.log(p + eps), axis=0) / max(np.log(n_factors + eps), eps)
    disentanglement_per_code = 1.0 - entropy_per_code
    disentanglement = float(
        np.sum(disentanglement_per_code * code_importance) / (code_importance.sum() + eps)
    )

    factor_importance = importance.sum(axis=1)
    if factor_importance.sum() < eps:
        completeness = 0.0
    else:
        p_f = importance / (factor_importance[:, None] + eps)
        entropy_per_factor = -np.sum(p_f * np.log(p_f + eps), axis=1) / max(np.log(n_latents + eps), eps)
        completeness = float(np.mean(1.0 - entropy_per_factor))

    informativeness_mean = float(np.nanmean(informativeness)) if informativeness else 0.0
    return disentanglement, completeness, informativeness_mean, importance


def compute_mig_score(latents, factors, discrete_factors=True, eps=1e-12):
    # factors: [N, F], latents: [N, D]
    n_factors = factors.shape[1]
    n_latents = latents.shape[1]
    mi_matrix = np.zeros((n_factors, n_latents), dtype=np.float64)
    migs = []

    for f in range(n_factors):
        y = factors[:, f]
        if discrete_factors:
            mi = mutual_info_classif(latents, y, discrete_features=False)
            H = _discrete_entropy(y, eps=eps)
        else:
            mi = mutual_info_regression(latents, y)
            H = float(np.var(y))

        mi_matrix[f, :] = mi
        mi_sorted = np.sort(mi)[::-1]
        gap = mi_sorted[0] - (mi_sorted[1] if n_latents > 1 else 0.0)
        migs.append(gap / (H + eps))

    return float(np.mean(migs)) if migs else 0.0, mi_matrix


def log_latent_table(prefix, corrs, accs):
    logging.info(f"{prefix} latent vs diagnosis table:")
    logging.info("  dim | corr | best_acc")
    for idx, (c, a) in enumerate(zip(corrs, accs)):
        logging.info(f"{idx:5d} | {c: .3f} | {a: .3f}")


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory: str, continue_from, batch_split: int):


    logging.debug("running experiment " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + str(specs["Description"]))

    data_source = specs["DataSource"]
    data_source_mesh = specs["DataSourceMesh"]
    train_split_file = specs["TrainSplit"]
    test_split_file = specs["TestSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]
    decoder_specs = specs["NetworkSpecs"]

    encoder_type = get_spec_with_default(specs, "EncoderType", "resnet_pointnet")
    use_labels = get_spec_with_default(specs, "UseLabels", True)
    label_index = get_spec_with_default(specs, "LabelIndex", 0)
    reg_label_index = get_spec_with_default(specs, "RegLabelIndex", None)
    use_regression_label = get_spec_with_default(specs, "UseRegressionLabel", False)

    guided_contrastive_loss = get_spec_with_default(specs, "GuidedContrastiveLoss", False)
    guided_contrastive_loss_cls = get_spec_with_default(specs, "GuidedContrastiveLossCls", guided_contrastive_loss)
    guided_contrastive_loss_reg = get_spec_with_default(specs, "GuidedContrastiveLossReg", False)
    attribute_loss = get_spec_with_default(specs, "AttributeLoss", False)
    kl_div_loss = get_spec_with_default(specs, "UseKLLoss", False)
    jacobian_loss = get_spec_with_default(specs, "JacobianLoss", False)
    dip_vae_loss = get_spec_with_default(specs, "DIPVAE", False)

    covariance_loss = get_spec_with_default(specs, "CovarianceLoss", False)
    covariance_loss_lambda = get_spec_with_default(specs, "CovarianceLossLambda", 0.0)
    corr_leakage_loss = get_spec_with_default(specs, "CorrLeakageLoss", False)
    corr_leakage_lambda = get_spec_with_default(specs, "CorrLeakageLambda", 0.0)
    cross_cov_loss = get_spec_with_default(specs, "CrossCovLoss", False)
    cross_cov_lambda = get_spec_with_default(specs, "CrossCovLambda", 0.0)
    leakage_target_dim = get_spec_with_default(specs, "LeakageTargetDim", 0)

    temp = get_spec_with_default(specs, "SNNLTemp", 2.0)
    temp_reg = get_spec_with_default(specs, "SNNLRegTemp", 2.0)
    w_cls = get_spec_with_default(specs, "SNNLWeight", 0.01)
    w_reg = get_spec_with_default(specs, "SNNLRegWeight", 0.005)
    threshold = get_spec_with_default(specs, "SNNLRegThreshold", 0.1)

    attribute_weight = get_spec_with_default(specs, "AttributeWeight", w_cls)
    attribute_latent_index = get_spec_with_default(specs, "AttributeLatentIndex", 0)
    reg_attribute_latent_index = get_spec_with_default(specs, "RegAttributeLatentIndex", 1)

    beta_final = get_spec_with_default(specs, "VAEKLWeight", beta_final)
    annealing_epochs = get_spec_with_default(specs, "KLWarmupEpochs", annealing_epochs)
    w_code_reg = get_spec_with_default(specs, "CodeRegWeight", w_code_reg)
    w_jacobian = get_spec_with_default(specs, "JacobianWeight", w_jacobian)
    dip_vae_weight = get_spec_with_default(specs, "DIPVAEWeight", 1.0)

    save_encoder_only = get_spec_with_default(specs, "SaveEncoderOnly", False)

    compute_sap = get_spec_with_default(specs, "ComputeSAP", False)
    sap_label_indices = get_spec_with_default(specs, "SAPLabelIndices", [label_index])
    sap_continuous = get_spec_with_default(specs, "SAPContinuousFactors", False)
    compute_dci = get_spec_with_default(specs, "ComputeDCI", False)
    compute_mig = get_spec_with_default(specs, "ComputeMIG", False)

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
        logging.info("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)
        if save_encoder_only:
            save_encoder(experiment_directory, "latest.pth", decoder.encoder, epoch)

    def save_checkpoints(epoch):
        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)
        if save_encoder_only:
            save_encoder(experiment_directory, str(epoch) + ".pth", decoder.encoder, epoch)

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
    use_eikonal = get_spec_with_default(specs, "UseEikonal", False)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    #decoder_old = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()
    decoder = vae.SDFVAE(
        latent_size,
        num_samp_per_scene,
        decoder_specs,
        kl_div_loss,
        encoder_type=encoder_type,
    ).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    pretrained_sdf_path = get_spec_with_default(specs, "PretrainedSDFDecoderPath", None)
    if pretrained_sdf_path:
        logging.info(f"Loading pretrained SDF decoder from: {pretrained_sdf_path}")
        load_sdf_decoder_weights(pretrained_sdf_path, decoder.decoder)

    train_sdf_decoder = get_spec_with_default(specs, "TrainSDFDecoder", True)
    if not train_sdf_decoder:
        logging.info("Freezing SDF decoder weights (TrainSDFDecoder=false).")
        for param in decoder.decoder.parameters():
            param.requires_grad = False

    #decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 200)
    
    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    torus_path = get_spec_with_default(specs, "TorusPath", "/home/jakaria/CALSNIC/calsnic_pial_surface/mesh_dataset/pial_surface/scaled_obj_files")
    #torus_path =  get_spec_with_default(specs, "TorusPath", "/home/jakaria/final_classification_dataset_femur_original/all_mesh/scaled_obj_files")
    logging.info(f"Torus path: {torus_path}")
    if not os.path.exists(torus_path): 
        logging.error(f"Running w/o validation, since the specified Torus path does not exist: {torus_path}")
        torus_path = None
    load_ram = get_spec_with_default(specs, "LoadDatasetIntoRAM", False)
    if load_ram:
        logging.info(f"Loading SDF samples into memory because LoadDatasetIntoRAM=true")
    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, data_source_mesh, train_split, num_samp_per_scene, load_ram=load_ram
    )
    test_dataset = deep_sdf.data.SDFSamples(
        data_source, data_source_mesh, test_split, num_samp_per_scene, load_ram=load_ram
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
    eval_train_scene_num = get_spec_with_default(specs, "EvalTrainSceneNumber", 0)
    eval_mesh_train_scene_num = get_spec_with_default(specs, "EvalMeshTrainSceneNumber", 10)
    eval_train_frequency = get_spec_with_default(specs, "EvalTrainFrequency", 200)
    eval_train_scene_idxs = list(range(len(sdf_dataset)))
    if eval_train_scene_num and eval_train_scene_num > 0:
        eval_train_scene_idxs = random.sample(
            range(len(sdf_dataset)), min(eval_train_scene_num, len(sdf_dataset))
        )
    eval_mesh_train_scene_idxs = random.sample(
        range(len(sdf_dataset)), min(eval_mesh_train_scene_num, len(sdf_dataset))
    )
    logging.debug(f"Train eval scenes: {len(eval_train_scene_idxs)}; mesh scenes: {len(eval_mesh_train_scene_idxs)}")

    # Get test evaluation settings.
    eval_test_frequency = get_spec_with_default(specs, "EvalTestFrequency", 500)
    eval_test_scene_num = get_spec_with_default(specs, "EvalTestSceneNumber", 0)
    eval_mesh_test_scene_num = get_spec_with_default(specs, "EvalMeshTestSceneNumber", 10)
    eval_test_optimization_steps = get_spec_with_default(specs, "EvalTestOptimizationSteps", 1000)
    eval_test_scene_idxs = list(range(len(test_split)))
    if eval_test_scene_num and eval_test_scene_num > 0:
        eval_test_scene_idxs = random.sample(
            range(len(test_split)), min(eval_test_scene_num, len(test_split))
        )
    eval_mesh_test_scene_idxs = random.sample(
        range(len(test_split)), min(eval_mesh_test_scene_num, len(test_split))
    )

    eval_batch_size = get_spec_with_default(specs, "EvalBatchSize", scene_per_batch)
    if eval_train_scene_num and eval_train_scene_num > 0:
        eval_train_dataset = torch.utils.data.Subset(sdf_dataset, eval_train_scene_idxs)
    else:
        eval_train_dataset = sdf_dataset
    eval_train_loader = data_utils.DataLoader(
        eval_train_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    if eval_test_scene_num and eval_test_scene_num > 0:
        eval_test_dataset = torch.utils.data.Subset(test_dataset, eval_test_scene_idxs)
    else:
        eval_test_dataset = test_dataset
    eval_test_loader = data_utils.DataLoader(
        eval_test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    def _extract_label_column(labels_np, idx):
        if labels_np.ndim == 1:
            return labels_np
        if idx is None or idx >= labels_np.shape[1]:
            return None
        return labels_np[:, idx]

    def collect_latents_and_labels(eval_loader):
        decoder.eval()
        all_latents = []
        all_labels = []
        sdf_losses = []
        with torch.no_grad():
            for sdf_data, _indices, labels, _filenames, surface_points in eval_loader:
                sdf_data = sdf_data.reshape(-1, 4)
                num_sdf_samples = sdf_data.shape[0]
                xyz = sdf_data[:, 0:3].cuda()
                sdf_gt = sdf_data[:, 3].unsqueeze(1).cuda()
                surface_points = surface_points.float().cuda()

                if kl_div_loss:
                    pred_sdf, mu, logvar, z = decoder(surface_points, xyz)
                    latents = mu
                else:
                    pred_sdf, z = decoder(surface_points, xyz)
                    latents = z

                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                sdf_loss = loss_l1(pred_sdf, sdf_gt) / num_sdf_samples
                sdf_losses.append(sdf_loss.item())

                all_latents.append(latents.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        latents_np = np.concatenate(all_latents, axis=0) if all_latents else np.zeros((0, latent_size))
        labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,))
        return latents_np, labels_np, float(np.mean(sdf_losses)) if sdf_losses else float("nan")

    def compute_eval_metrics(latents_np, labels_np, prefix):
        metrics_out = {}
        if not use_labels or labels_np.size == 0:
            return metrics_out

        labels_cls_np = _extract_label_column(labels_np, label_index)
        if labels_cls_np is None:
            return metrics_out

        corrs, accs = compute_latent_label_table(latents_np, labels_cls_np)
        metrics_out["corrs"] = corrs
        metrics_out["accs"] = accs
        metrics_out["corr0"] = corrs[0] if corrs else float("nan")

        factors = []
        if sap_label_indices:
            for idx in sap_label_indices:
                col = _extract_label_column(labels_np, idx)
                if col is not None:
                    factors.append(col)
        if factors:
            factors_np = np.stack(factors, axis=1)
            if compute_sap:
                metrics_out["sap"] = compute_sap_score(latents_np, factors_np, sap_continuous)
            if compute_dci:
                dci_d, dci_c, dci_i, _ = compute_dci_metrics(
                    latents_np, factors_np, discrete_factors=not sap_continuous
                )
                metrics_out["dci"] = (dci_d, dci_c, dci_i)
            if compute_mig:
                metrics_out["mig"] = compute_mig_score(
                    latents_np, factors_np, discrete_factors=not sap_continuous
                )[0]

        log_latent_table(prefix, corrs, accs)
        return metrics_out

    def compute_chamfer_for_indices(dataset, indices, prefix):
        gt_mesh_dir = get_spec_with_default(specs, "EvalGTMeshDir", None)
        gt_mesh_ext = get_spec_with_default(specs, "EvalGTMeshExt", ".obj")
        if not gt_mesh_dir:
            logging.error("EvalGTMeshDir not set; skipping Chamfer.")
            return float("nan")

        chamfer_dists = []
        chamfer_dists_all = []
        for index in indices:
            surface_points = torch.as_tensor(dataset.surface_points[index]).unsqueeze(0).float().cuda()
            with torch.no_grad():
                if kl_div_loss:
                    mu, _ = decoder.encoder(surface_points)
                    lat_vec = mu
                else:
                    lat_vec = decoder.encoder(surface_points)
            lat_vec = lat_vec.detach()
            save_name = os.path.splitext(os.path.basename(dataset.npyfiles[index]))[0]
            path = os.path.join(
                experiment_directory,
                ws.tb_logs_dir,
                ws.tb_logs_train_reconstructions if prefix == "train" else ws.tb_logs_test_reconstructions,
                save_name,
            )
            os.makedirs(path, exist_ok=True)
            with torch.no_grad():
                gen_mesh = mesh.create_mesh(
                    decoder.decoder,
                    lat_vec,
                    N=eval_grid_res,
                    max_batch=int(2 ** 18),
                    filename=os.path.join(path, f"epoch={epoch}"),
                    return_trimesh=True,
                )
            if gen_mesh is None:
                continue
            gt_mesh_path = os.path.join(gt_mesh_dir, f"{save_name}{gt_mesh_ext}")
            cd, cd_all = metrics.compute_metric(
                gt_mesh=gt_mesh_path, gen_mesh=gen_mesh, metric="chamfer"
            )
            chamfer_dists.append(cd)
            chamfer_dists_all.append(cd_all)

        if chamfer_dists_all:
            fig, percentiles = plotting.plot_dist_violin(np.concatenate(chamfer_dists_all, axis=0))
            summary_writer.add_figure(f"CD Percentiles/{prefix}", fig, global_step=epoch)
            for p in [75, 90, 99]:
                if p in percentiles:
                    summary_writer.add_scalar(f"CD Percentiles/{prefix} {p}th", percentiles[p], global_step=epoch)
        return float(np.mean(chamfer_dists)) if chamfer_dists else float("nan")

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
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

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [{
            "params": decoder.parameters(),
            "lr": lr_schedules[0].get_learning_rate(0),
        },
        {
            "params": lat_vecs.parameters(),
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

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
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
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )
    
    try:
        train_chamfer_dists_log = []
        test_chamfer_dists_log = []
        for epoch in range(start_epoch, num_epochs + 1):
            
            # Calculate current β
            if epoch < annealing_epochs:
                beta = beta_final * (epoch / annealing_epochs)  # Linear increase
            else:
                beta = beta_final  # After annealing, use final β

            epoch_time_start = time.time()
            epoch_losses = []
            epoch_sdf_losses = []
            epoch_reg_losses = []
            epoch_eikonal_losses = []
            epoch_snnl = []
            epoch_snnl_reg = []
            epoch_attr = []
            epoch_attr_reg = []
            epoch_loss_kl = []
            epoch_jacobian_loss = []
            epoch_dip_vae_loss = []
            epoch_cov_loss = []
            epoch_leak_loss = []
            epoch_cross_cov_loss = []

            logging.info("epoch {}...".format(epoch))

            # Required because evaluation puts the decoder into 'eval' mode.
            decoder.train()

            adjust_learning_rate(lr_schedules, optimizer_all, epoch, loss_log_epoch)
            for sdf_data, indices, labels, filenames, surface_points in sdf_loader:
                # logging.debug(f"time for dataloading: {(time.time() - TIME)*1000:.3f} ms"); TIME = time.time()
                # Process the input data
                sdf_data = sdf_data.reshape(-1, 4)

                num_sdf_samples = sdf_data.shape[0]
                #logging.info(f"sdf_data shape: {sdf_data.shape}")
                #logging.info(f"label shape: {labels.shape}")
                #logging.info(f"indices shape: {indices.shape}")
                #logging.info(f"indices: {indices}")

                sdf_data.requires_grad = False

                xyz = sdf_data[:, 0:3]
                xyz.requires_grad = True

                #labels for torus bump

                #labels_cls = labels[:,0] # bump or no bump binary label
                #labels_reg = labels[:,2] # torus scale

                labels_cls = None
                labels_reg = None
                if use_labels and label_index is not None:
                    labels_cls = labels[:, label_index]
                if use_regression_label and reg_label_index is not None:
                    labels_reg = labels[:, reg_label_index]

                if labels_cls is not None:
                    labels_cls = labels_cls.to(torch.float32).to(torch.device("cuda"))
                    labels_cls.requires_grad = False
                if labels_reg is not None:
                    labels_reg = labels_reg.to(torch.float32).to(torch.device("cuda"))
                    labels_reg.requires_grad = False

                sdf_gt = sdf_data[:, 3].unsqueeze(1)

                if enforce_minmax:
                    sdf_gt = torch.clamp(sdf_gt, minT, maxT)

                xyz = torch.chunk(xyz, batch_split)
                #logging.info(f"xyz[0] shape: {xyz[0].shape}")
                surface_points = torch.chunk(surface_points, batch_split)
                #logging.info(f"Surface points[0] shape: {surface_points[0].shape}")

                indices_z = torch.chunk(indices, batch_split)
                
                indices = torch.chunk(
                    indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                    batch_split,
                )

                if labels_cls is not None:
                    labels_cls = torch.chunk(labels_cls, batch_split)
                else:
                    labels_cls = [None] * batch_split
                if labels_reg is not None:
                    labels_reg = torch.chunk(labels_reg, batch_split)
                else:
                    labels_reg = [None] * batch_split

                sdf_gt = torch.chunk(sdf_gt, batch_split)

                batch_loss_tb = 0.0
                sdf_loss_tb = 0.0
                reg_loss_tb = 0.0
                eikonal_loss_tb = 0.0
                snnl = 0.0
                snnl_reg = 0.0
                attr_loss = 0.0
                attr_loss_reg = 0.0
                loss_kl = 0.0
                jacobian_loss_val = 0.0
                dip_vae_loss_val = 0.0
                cov_loss_val = 0.0
                leak_loss_val = 0.0
                cross_cov_loss_val = 0.0

                optimizer_all.zero_grad()

                for i in range(batch_split):
                    z = lat_vecs(indices_z[i])
                    batch_vecs = z.unsqueeze(1).repeat(1, num_samp_per_scene, 1).view(-1, latent_size)
                    #print(f"Batch vecs device: {batch_vecs[i].device}")
                    #print(f"z vecs device: {z[i].device}")
                    #batch_vecs = lat_vecs(indices[i])
                    #z_for_c_loss = lat_vecs(indices_z[i])
                    labels_cls_i = labels_cls[i].unsqueeze(-1) if labels_cls[i] is not None else None
                    labels_reg_i = labels_reg[i].unsqueeze(-1) if labels_reg[i] is not None else None

                    #logging.info(f"batch_vecs shape: {batch_vecs.shape}")
                    #logging.info(f"latent vecs z (for loss) shape: {z_for_c_loss.shape}")
                    #logging.info(f"xyz shape: {xyz[i].shape}")
                    #logging.info(f"indices shape: {indices[i].shape}")
                    #logging.info(f"indices: {indices[i]}")
                    #logging.info(f"labels_cls shape: {labels_cls.shape}")
                    #logging.info(f"labels_reg: {labels_reg}")
                    #logging.info(f"labels_cls: {labels_cls}")
                    #logging.info(f"xyz shape: {xyz[i].shape}")
                    #logging.info(f"surface_points shape: {surface_points[i].shape}")
                    
                    input = torch.cat([batch_vecs, xyz[i]], dim=1)
                    #print(f"input device: {input.device}")
                    #logging.info(f"input shape: {input.shape}")
                    
                    # NN optimization
                    if kl_div_loss:
                        pred_sdf, mu, logvar, z = decoder(surface_points[i], xyz[i])
                    else:
                        pred_sdf, z = decoder(surface_points[i], xyz[i])

                    if enforce_minmax:
                        pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                    chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples
                    sdf_loss_tb += chunk_loss.item()

                    if do_code_regularization:
                        l2_size_loss = torch.sum(torch.norm(z, dim=1))
                        reg_loss = (
                            code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                        ) / num_sdf_samples
                    
                        chunk_loss = chunk_loss + w_code_reg * reg_loss.cuda()
                        reg_loss_tb += reg_loss.item()
                    
                    summary_writer.add_scalar("Loss/train_vanilla", chunk_loss, global_step=epoch)
                    if use_eikonal:
                        grad_outputs = torch.ones_like(pred_sdf, requires_grad=True)
                        gradients = torch.autograd.grad(pred_sdf, [xyz[i]], grad_outputs=grad_outputs, create_graph=True, allow_unused=True, retain_graph=True)[0]
                        eikonal_loss = 0.002 * ((1. - torch.linalg.vector_norm(gradients, dim=1))**2).mean()
                        chunk_loss += eikonal_loss
                        eikonal_loss_tb += eikonal_loss.item()

                    if jacobian_loss:
                        j_penalty = jacobian_penalty_JJT(surface_points[i], decoder)
                        jacobian_loss_t = w_jacobian * j_penalty
                        chunk_loss += jacobian_loss_t
                        jacobian_loss_val += jacobian_loss_t.item()

                        
                    if kl_div_loss:
                        kl_loss = kl_divergence_loss(mu, logvar)
                        chunk_loss += beta * kl_loss
                        loss_kl += kl_loss.item()

                    if guided_contrastive_loss:
                        if guided_contrastive_loss_cls and labels_cls_i is not None:
                            SNN_Loss = loss.SNNLossCls(T=temp, lam1=1.0, lam2=2.0, target_dim=attribute_latent_index)
                            loss_snn = SNN_Loss(z, labels_cls_i)
                            chunk_loss += loss_snn * w_cls
                            snnl += loss_snn.item()

                        if guided_contrastive_loss_reg and labels_reg_i is not None:
                            SNN_Loss_Reg = loss.SNNRegLossExact(
                                T=temp_reg,
                                lam1=1,
                                lam2=2.0,
                                threshold=threshold,
                                target_dim=reg_attribute_latent_index,
                                normalize_z=True,
                                use_adaptive_T=True,
                                pos_mode="topk",
                            )
                            loss_snn_reg = SNN_Loss_Reg(z, labels_reg_i)
                            chunk_loss += loss_snn_reg * w_reg
                            snnl_reg += loss_snn_reg.item()
                        

                    if attribute_loss:
                        loss_attr = loss.AttributeLoss()
                        if labels_cls_i is not None:
                            loss_attr_cls = loss_attr(z[:, attribute_latent_index], labels_cls_i)
                            chunk_loss += loss_attr_cls * attribute_weight
                            attr_loss += loss_attr_cls.item()
                        if labels_reg_i is not None:
                            loss_attr_reg = loss_attr(z[:, reg_attribute_latent_index], labels_reg_i)
                            chunk_loss += loss_attr_reg * attribute_weight
                            attr_loss_reg += loss_attr_reg.item()

                    if covariance_loss:
                        cov_loss_fn = loss.CovarianceLoss()
                        loss_cov = cov_loss_fn(z) * covariance_loss_lambda
                        chunk_loss += loss_cov
                        cov_loss_val += loss_cov.item()

                    if corr_leakage_loss and labels_cls_i is not None:
                        loss_leak = loss.corr_leakage_penalty(z, labels_cls_i, leakage_target_dim)
                        chunk_loss += loss_leak * corr_leakage_lambda
                        leak_loss_val += (loss_leak * corr_leakage_lambda).item()

                    if cross_cov_loss:
                        loss_cross = loss.cross_cov_penalty(z, leakage_target_dim)
                        chunk_loss += loss_cross * cross_cov_lambda
                        cross_cov_loss_val += (loss_cross * cross_cov_lambda).item()

                    if dip_vae_loss == True and kl_div_loss:
                        dip_vae_loss_II = loss.DIPVAEIILoss()
                        loss_dip_vae_II = dip_vae_loss_II(mu, logvar) * dip_vae_weight
                        chunk_loss += loss_dip_vae_II
                        dip_vae_loss_val += loss_dip_vae_II.item()
                        
                    chunk_loss.backward()

                    batch_loss_tb += chunk_loss.item()
                    # Print batch loss
                #print(f"SNNL Loss: {snnl}")
                #print(f"Batch loss: {batch_loss_tb}") 
                #print(f"kl loss: {loss_kl}")                   
                logging.debug("loss = {}".format(batch_loss_tb))
                loss_log.append(batch_loss_tb)
                epoch_losses.append(batch_loss_tb)
                epoch_sdf_losses.append(sdf_loss_tb)
                epoch_reg_losses.append(reg_loss_tb)
                epoch_eikonal_losses.append(eikonal_loss_tb)
                epoch_snnl.append(snnl)
                epoch_attr.append(attr_loss)
                epoch_snnl_reg.append(snnl_reg)
                epoch_attr_reg.append(attr_loss_reg)
                epoch_loss_kl.append(loss_kl)
                epoch_jacobian_loss.append(jacobian_loss_val)
                epoch_dip_vae_loss.append(dip_vae_loss_val)
                epoch_cov_loss.append(cov_loss_val)
                epoch_leak_loss.append(leak_loss_val)
                epoch_cross_cov_loss.append(cross_cov_loss_val)

                if grad_clip is not None:

                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip, norm_type=2)

                optimizer_all.step()

            # LOG EPOCH
            seconds_elapsed = time.time() - epoch_time_start
            timing_log.append(seconds_elapsed)
            # Log epoch losses.
            epoch_loss = sum(epoch_losses)/len(epoch_losses)
            loss_log_epoch.append(epoch_loss)
            summary_writer.add_scalar("Loss/train", epoch_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_sdf", sum(epoch_sdf_losses)/len(epoch_sdf_losses), global_step=epoch)
            summary_writer.add_scalar("Loss/train_reg", sum(epoch_reg_losses)/len(epoch_reg_losses), global_step=epoch)
            if use_eikonal:
                summary_writer.add_scalar("Loss/train_eikonal", sum(epoch_eikonal_losses)/len(epoch_eikonal_losses), global_step=epoch)
            
            if jacobian_loss:
                summary_writer.add_scalar("Loss/train_jacobian", sum(epoch_jacobian_loss)/len(epoch_jacobian_loss), global_step=epoch)

            if guided_contrastive_loss:
                if guided_contrastive_loss_cls:
                    summary_writer.add_scalar("Loss/train_snnl", sum(epoch_snnl)/len(epoch_snnl), global_step=epoch)
                if guided_contrastive_loss_reg:
                    summary_writer.add_scalar("Loss/train_snnl_reg", sum(epoch_snnl_reg)/len(epoch_snnl_reg), global_step=epoch)

            if kl_div_loss:
                summary_writer.add_scalar("Loss/train_kl", sum(epoch_loss_kl)/len(epoch_loss_kl), global_step=epoch)
            
            if attribute_loss:
                summary_writer.add_scalar("Loss/train_attr", sum(epoch_attr)/len(epoch_attr), global_step=epoch)
                summary_writer.add_scalar("Loss/train_attr_reg", sum(epoch_attr_reg)/len(epoch_attr_reg), global_step=epoch)

            if dip_vae_loss:
                summary_writer.add_scalar("Loss/train_dip_vae", sum(epoch_dip_vae_loss)/len(epoch_dip_vae_loss), global_step=epoch)

            if covariance_loss:
                summary_writer.add_scalar("Loss/train_covariance", sum(epoch_cov_loss)/len(epoch_cov_loss), global_step=epoch)
            if corr_leakage_loss:
                summary_writer.add_scalar("Loss/train_corr_leakage", sum(epoch_leak_loss)/len(epoch_leak_loss), global_step=epoch)
            if cross_cov_loss:
                summary_writer.add_scalar("Loss/train_cross_cov", sum(epoch_cross_cov_loss)/len(epoch_cross_cov_loss), global_step=epoch)

            # Log learning rate.
            lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])
            summary_writer.add_scalar("Learning Rate/Params", lr_log[-1][0], global_step=epoch)
            summary_writer.add_scalar("Learning Rate/Latent", lr_log[-1][1], global_step=epoch)
            # Log latent vector length.
            #mlm = get_mean_latent_vector_magnitude(lat_vecs)
            #lat_mag_log.append(mlm)
            #summary_writer.add_scalar("Mean Latent Magnitude/train", mlm, global_step=epoch)
            append_parameter_magnitudes(param_mag_log, decoder)

            print(f"Epoch Loss: {epoch_loss}")
            print(f"Epoch SDF Loss: {sum(epoch_sdf_losses)/len(epoch_sdf_losses)}")
            if guided_contrastive_loss:
                if guided_contrastive_loss_cls:
                    print(f"SNNL Loss: {sum(epoch_snnl)/len(epoch_snnl)}")
                if guided_contrastive_loss_reg:
                    print(f"SNNL Reg Loss: {sum(epoch_snnl_reg)/len(epoch_snnl_reg)}")
            if attribute_loss:
                print(f"Attribute Loss: {sum(epoch_attr)/len(epoch_attr)}")
                print(f"Attribute Reg Loss: {sum(epoch_attr_reg)/len(epoch_attr_reg)}")
            if kl_div_loss:
                print(f"KL Loss: {sum(epoch_loss_kl)/len(epoch_loss_kl)}")
            if jacobian_loss:
                print(f"Jacobian Loss: {sum(epoch_jacobian_loss)/len(epoch_jacobian_loss)}")
            if use_eikonal:
                print(f"Eikonal Loss: {sum(epoch_eikonal_losses)/len(epoch_eikonal_losses)}")
            if dip_vae_loss:
                print(f"DIP VAE Loss: {sum(epoch_dip_vae_loss)/len(epoch_dip_vae_loss)}")
            if covariance_loss:
                print(f"Covariance Loss: {sum(epoch_cov_loss)/len(epoch_cov_loss)}")
            if corr_leakage_loss:
                print(f"Corr Leakage Loss: {sum(epoch_leak_loss)/len(epoch_leak_loss)}")
            if cross_cov_loss:
                print(f"Cross Cov Loss: {sum(epoch_cross_cov_loss)/len(epoch_cross_cov_loss)}")
            if do_code_regularization:
                print(f"Reg Loss: {sum(epoch_reg_losses)/len(epoch_reg_losses)}")
            

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
            summary_writer.add_scalar(f"GradsNorm/allNetParams.grad", torch.norm(torch.stack(grad_norms), p=2).item(), global_step=epoch)
            #summary_writer.add_scalar(f"GradsNorm/allLatParams.grad", torch.norm(lat_vecs.weight.grad.detach(), p=2).item(), global_step=epoch)

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
                if epoch % eval_train_frequency == 0:
                    logging.info(f"Train Evaluation Started at epoch {epoch}...")
                    train_latents, train_labels, train_sdf_loss = collect_latents_and_labels(eval_train_loader)
                    train_metrics = compute_eval_metrics(train_latents, train_labels, "train")
                    if "sap" in train_metrics:
                        summary_writer.add_scalar("SAP/train", train_metrics["sap"], global_step=epoch)
                        logging.info(f"Epoch {epoch} SAP (train): {train_metrics['sap']:.6f}")
                    if "dci" in train_metrics:
                        dci_d, dci_c, dci_i = train_metrics["dci"]
                        summary_writer.add_scalar("DCI/train_d", dci_d, global_step=epoch)
                        summary_writer.add_scalar("DCI/train_c", dci_c, global_step=epoch)
                        summary_writer.add_scalar("DCI/train_i", dci_i, global_step=epoch)
                        logging.info(f"Epoch {epoch} DCI (train): d={dci_d:.6f} c={dci_c:.6f} i={dci_i:.6f}")
                    if "mig" in train_metrics:
                        summary_writer.add_scalar("MIG/train", train_metrics["mig"], global_step=epoch)
                        logging.info(f"Epoch {epoch} MIG (train): {train_metrics['mig']:.6f}")
                    if "corr0" in train_metrics:
                        logging.info(f"Epoch {epoch} correlation (train): latent0 vs label[0] = {train_metrics['corr0']:.6f}")

                    train_cd = compute_chamfer_for_indices(sdf_dataset, eval_mesh_train_scene_idxs, "train")
                    if not np.isnan(train_cd):
                        summary_writer.add_scalar("Mean Chamfer Dist/train", train_cd, global_step=epoch)
                        logging.info(f"Epoch {epoch} train chamfer: {train_cd:.6f}")
                    summary_writer.add_scalar("Loss/train_eval_sdf", train_sdf_loss, global_step=epoch)
                    logging.info(
                        f"Epoch {epoch} train summary: eval_count={len(eval_train_scene_idxs)} "
                        f"mesh_count={len(eval_mesh_train_scene_idxs)} train_sdf_loss={train_sdf_loss:.6f} "
                        f"train_cd={train_cd if not np.isnan(train_cd) else 'n/a'}"
                    )

                if epoch % eval_test_frequency == 0:
                    logging.info(f"Test Evaluation Started at epoch {epoch}...")
                    test_latents, test_labels, test_sdf_loss = collect_latents_and_labels(eval_test_loader)
                    test_metrics = compute_eval_metrics(test_latents, test_labels, "test")
                    if "sap" in test_metrics:
                        summary_writer.add_scalar("SAP/test", test_metrics["sap"], global_step=epoch)
                        logging.info(f"Epoch {epoch} SAP (test): {test_metrics['sap']:.6f}")
                    if "dci" in test_metrics:
                        dci_d, dci_c, dci_i = test_metrics["dci"]
                        summary_writer.add_scalar("DCI/test_d", dci_d, global_step=epoch)
                        summary_writer.add_scalar("DCI/test_c", dci_c, global_step=epoch)
                        summary_writer.add_scalar("DCI/test_i", dci_i, global_step=epoch)
                        logging.info(f"Epoch {epoch} DCI (test): d={dci_d:.6f} c={dci_c:.6f} i={dci_i:.6f}")
                    if "mig" in test_metrics:
                        summary_writer.add_scalar("MIG/test", test_metrics["mig"], global_step=epoch)
                        logging.info(f"Epoch {epoch} MIG (test): {test_metrics['mig']:.6f}")
                    if "corr0" in test_metrics:
                        logging.info(f"Epoch {epoch} correlation (test): latent0 vs label[0] = {test_metrics['corr0']:.6f}")

                    test_cd = compute_chamfer_for_indices(test_dataset, eval_mesh_test_scene_idxs, "test")
                    if not np.isnan(test_cd):
                        summary_writer.add_scalar("Mean Chamfer Dist/test", test_cd, global_step=epoch)
                        logging.info(f"Epoch {epoch} test chamfer: {test_cd:.6f}")
                    summary_writer.add_scalar("Loss/test_eval_sdf", test_sdf_loss, global_step=epoch)
                    logging.info(
                        f"Epoch {epoch} test summary: eval_count={len(eval_test_scene_idxs)} "
                        f"mesh_count={len(eval_mesh_test_scene_idxs)} test_sdf_loss={test_sdf_loss:.6f} "
                        f"test_cd={test_cd if not np.isnan(test_cd) else 'n/a'}"
                    )

            summary_writer.add_scalar("Time/epoch (min)", (time.time()-epoch_time_start)/60, epoch)
            summary_writer.flush() 
               
            # End of epoch.
    except KeyboardInterrupt as e:
        logging.error(f"Received KeyboardInterrupt. Cleaning up and ending training.")
    finally:
        # Calculate model size.
        param_size = 0
        param_cnt = 0
        for param in decoder.parameters():
            param_size += param.nelement() * param.element_size()
            param_cnt += param.nelement()
        buffer_size = 0
        for buffer in decoder.buffers():
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
            "best_train_loss" : min(loss_log),
            "best_train_cd" : min(train_chamfer_dists_log) if len(train_chamfer_dists_log) else -1,
            "best_test_cd" : min(test_chamfer_dists_log) if len(test_chamfer_dists_log) else -1,
        }
        summary_writer.add_hparams(writer_hparams, train_results, run_name='.')
        #summary_writer.add_graph(decoder, input)        
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
