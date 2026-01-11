#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import os
import json
import time
import logging

import deep_sdf
from deep_sdf import lr_scheduling
import deep_sdf.workspace as ws

from networks import residual_mlp_vae


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
        if "weight" not in latent_data:
            raise Exception("latent state dict missing weight")
        return latent_data["weight"]

    raise Exception("unrecognized latent code format")


def load_sdf_decoder_weights(model_path, sdf_decoder):
    if model_path is None:
        return
    if not os.path.isfile(model_path):
        raise Exception('SDF decoder model file "{}" does not exist'.format(model_path))

    data = torch.load(model_path, map_location="cpu")
    state = data.get("model_state_dict", data)
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


def main_function(experiment_directory: str, continue_from, batch_split: int):

    logging.debug("running experiment " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + str(specs.get("Description", "(none)")))

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

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

    vae = residual_mlp_vae.ResidualMLPVAE(
        input_dim=vae_input_dim,
        latent_dim=vae_latent_dim,
        encoder_hidden_dims=vae_encoder_dims,
        decoder_hidden_dims=vae_decoder_dims,
        num_blocks=vae_blocks,
        activation=vae_activation,
        dropout=vae_dropout,
        use_layernorm=vae_layernorm,
    ).cuda()

    if torch.cuda.device_count() > 1:
        vae = torch.nn.DataParallel(vae)
        sdf_decoder = torch.nn.DataParallel(sdf_decoder)

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 200)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    load_ram = get_spec_with_default(specs, "LoadDatasetIntoRAM", False)
    if load_ram:
        logging.info("Loading SDF samples into memory because LoadDatasetIntoRAM=true")

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=load_ram
    )

    num_scenes = len(sdf_dataset)
    if teacher_latents.shape[0] != num_scenes:
        raise Exception(
            "Pretrained latent count does not match number of scenes: {} vs {}".format(
                teacher_latents.shape[0], num_scenes
            )
        )

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    lr_schedules = lr_scheduling.get_learning_rate_schedules(specs)

    vae_lr = lr_schedules[0].get_learning_rate(0)
    params = [{"params": vae.parameters(), "lr": vae_lr}]

    if train_sdf_decoder:
        sdf_lr_schedule = lr_schedules[1] if len(lr_schedules) > 1 else lr_schedules[0]
        params.append({"params": sdf_decoder.parameters(), "lr": sdf_lr_schedule.get_learning_rate(0)})

    optimizer = torch.optim.Adam(params)

    summary_writer = SummaryWriter(log_dir=os.path.join(experiment_directory, ws.tb_logs_dir))

    loss_log = []
    loss_log_epoch = []
    sdf_loss_log_epoch = []
    sdf_reg_log_epoch = []
    vae_recon_log_epoch = []
    vae_kl_log_epoch = []
    vae_lat_mag_log = []
    lr_log = []
    timing_log = []

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

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            epoch_time_start = time.time()

            epoch_losses = []
            epoch_sdf_losses = []
            epoch_sdf_reg_losses = []
            epoch_vae_recon = []
            epoch_vae_kl = []
            epoch_vae_lat_mag = []

            logging.info("epoch {}...".format(epoch))

            vae.train()
            if train_sdf_decoder:
                sdf_decoder.train()
            else:
                sdf_decoder.eval()

            adjust_learning_rate(lr_schedules, optimizer, epoch, loss_log_epoch)

            kl_weight = vae_kl_weight * residual_mlp_vae.linear_warmup(
                epoch, vae_kl_warmup_epochs
            )
            if do_code_regularization:
                if code_reg_warmup_epochs <= 0:
                    code_reg_weight = 1.0
                else:
                    code_reg_weight = min(1.0, epoch / float(code_reg_warmup_epochs))
            else:
                code_reg_weight = 0.0

            for sdf_data, indices in sdf_loader:
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

            seconds_elapsed = time.time() - epoch_time_start
            timing_log.append(seconds_elapsed)

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_sdf_loss = sum(epoch_sdf_losses) / len(epoch_sdf_losses)
            epoch_sdf_reg = sum(epoch_sdf_reg_losses) / len(epoch_sdf_reg_losses)
            epoch_vae_recon_loss = sum(epoch_vae_recon) / len(epoch_vae_recon)
            epoch_vae_kl_loss = sum(epoch_vae_kl) / len(epoch_vae_kl)
            epoch_vae_lat_mag = sum(epoch_vae_lat_mag) / len(epoch_vae_lat_mag)

            logging.info("Epoch {} loss: {}".format(epoch, epoch_loss))

            loss_log_epoch.append(epoch_loss)
            sdf_loss_log_epoch.append(epoch_sdf_loss)
            sdf_reg_log_epoch.append(epoch_sdf_reg)
            vae_recon_log_epoch.append(epoch_vae_recon_loss)
            vae_kl_log_epoch.append(epoch_vae_kl_loss)
            vae_lat_mag_log.append(epoch_vae_lat_mag)

            summary_writer.add_scalar("Loss/train", epoch_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_sdf", epoch_sdf_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_reg", epoch_sdf_reg, global_step=epoch)
            summary_writer.add_scalar("Loss/train_vae_recon", epoch_vae_recon_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_vae_kl", epoch_vae_kl_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_vae_total", epoch_vae_recon_loss + epoch_vae_kl_loss, global_step=epoch)
            summary_writer.add_scalar("Mean Latent Magnitude/train", epoch_vae_lat_mag, global_step=epoch)
            summary_writer.add_scalar("KL/warmup", kl_weight, global_step=epoch)

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
                    lr_log,
                    timing_log,
                    epoch,
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
