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
import numpy as np

import deep_sdf
from deep_sdf import mesh, metrics, lr_scheduling, plotting, utils, loss
import deep_sdf.workspace as ws
import reconstruct

from torch.utils.tensorboard import SummaryWriter

guided_contrastive_loss = False
beta = 0.000015
temp = 181
w_cls = 0.5

def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
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


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def main_function(experiment_directory: str, continue_from, batch_split: int):

   

    logging.debug("running experiment " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + str(specs["Description"]))

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    test_split_file = specs["TestSplit"]

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
        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", mu, epoch)

    def save_checkpoints(epoch):
        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", mu, epoch)

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

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)
    
    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    torus_path = get_spec_with_default(specs, "TorusPath", "/home/jakaria//torus_two_models_data/torus_two/obj_files")
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
    eval_train_frequency = get_spec_with_default(specs, "EvalTrainFrequency", 20)
    eval_train_scene_idxs = random.sample(range(len(sdf_dataset)), min(eval_train_scene_num, len(sdf_dataset)))
    logging.debug(f"Plotting {eval_train_scene_num} shapes with indices {eval_train_scene_idxs}")

    # Get test evaluation settings.
    with open(test_split_file, "r") as f:
        test_split = json.load(f)
    eval_test_frequency = get_spec_with_default(specs, "EvalTestFrequency", 20)
    eval_test_scene_num = get_spec_with_default(specs, "EvalTestSceneNumber", 10)
    eval_test_optimization_steps = get_spec_with_default(specs, "EvalTestOptimizationSteps", 1000)
    eval_test_filenames = deep_sdf.data.get_instance_filenames(data_source, test_split)
    eval_test_filenames = random.sample(eval_test_filenames, min(eval_test_scene_num, len(eval_test_filenames)))

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)
    '''
    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )
    '''
    mu = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        mu.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )
    logvar = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        logvar.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )
    #using mu instead of lat_vecs
    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(mu)
        )
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [{
            "params": decoder.parameters(),
            "lr": lr_schedules[0].get_learning_rate(0),
        },
        {
            "params": mu.parameters(),
            "lr": lr_schedules[1].get_learning_rate(0),
        },
                {
            "params": logvar.parameters(),
            "lr": lr_schedules[2].get_learning_rate(0),
        }
        ]
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
            experiment_directory, continue_from + ".pth", mu
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
            mu.num_embeddings * mu.embedding_dim,
            mu.num_embeddings,
            mu.embedding_dim,
        )
    )
    
    try:
        train_chamfer_dists_log = []
        test_chamfer_dists_log = []
        for epoch in range(start_epoch, num_epochs + 1):
            

            epoch_time_start = time.time()
            epoch_losses = []
            epoch_sdf_losses = []
            epoch_reg_losses = []
            epoch_eikonal_losses = []
            epoch_snnl = []
            epoch_kl_losses = []

            logging.info("epoch {}...".format(epoch))

            # Required because evaluation puts the decoder into 'eval' mode.
            decoder.train()

            adjust_learning_rate(lr_schedules, optimizer_all, epoch, loss_log_epoch)
            for sdf_data, indices, labels in sdf_loader:
                # logging.debug(f"time for dataloading: {(time.time() - TIME)*1000:.3f} ms"); TIME = time.time()
                # Process the input data
                sdf_data = sdf_data.reshape(-1, 4)

                num_sdf_samples = sdf_data.shape[0]
                #logging.info(f"sdf_data shape: {sdf_data.shape}")
                #logging.info(f"labels shape: {labels.shape}")
                #logging.info(f"indices: {indices}")
                sdf_data.requires_grad = False

                xyz = sdf_data[:, 0:3]
                xyz.requires_grad = True
                labels = labels[:,0] # Bump or no Bump, binary label
                labels = labels.to(torch.float32)
                #labels = labels.cuda()
                #logging.info(f"labels shape: {labels.shape}")
                #logging.info(f"labels data type: {labels.dtype}")
                #logging.info(f"xyz data type: {xyz.dtype}")
                labels.requires_grad = True
                sdf_gt = sdf_data[:, 3].unsqueeze(1)

                if enforce_minmax:
                    sdf_gt = torch.clamp(sdf_gt, minT, maxT)

                xyz = torch.chunk(xyz, batch_split)
                '''
                indices = torch.chunk(
                    indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                    batch_split,
                )
                '''

                indices = torch.chunk(indices, batch_split)

                labels = torch.chunk(
                    labels,
                    batch_split,
                )

                #labels = torch.chunk(labels, batch_split)

                sdf_gt = torch.chunk(sdf_gt, batch_split)

                batch_loss_tb = 0.0
                sdf_loss_tb = 0.0
                reg_loss_tb = 0.0
                eikonal_loss_tb = 0.0
                snnl = 0.0
                kl_loss_tb = 0.0

                optimizer_all.zero_grad()

                for i in range(batch_split):

                    #batch_vecs = lat_vecs(indices[i])
                    batch_mu = mu(indices[i])
                    batch_logvar = logvar(indices[i])
                    #batch_mu = mu[i]
                    #batch_logvar = logvar[i]
                    labels = labels[i].unsqueeze(-1)
                    labels = labels.cuda()
                    logging.info(f"batch_mu shape: {batch_mu.shape}")
                    logging.info(f"xyz shape: {xyz[i].shape}")
                    logging.info(f"indices shape: {indices[i].shape}")
                    logging.info(f"Device: {batch_mu.device}")
                    logging.info(f"labels Device: {labels.device}")
                    z = reparameterize(batch_mu, batch_logvar)
                    z_repeated = z.unsqueeze(1).repeat(1, num_samp_per_scene, 1).view(-1, latent_size)
                    input = torch.cat([z_repeated, xyz[i]], dim=1)
                    logging.info(f"input shape: {input.shape}")
                    # NN optimization
                    pred_sdf = decoder(input)

                    if enforce_minmax:
                        pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                    chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples
                    # beta is not used for now
                    kl_loss = kl_divergence(batch_mu, batch_logvar) / num_sdf_samples
                    chunk_loss += kl_loss
                    sdf_loss_tb += chunk_loss.item()
                    kl_loss_tb += kl_loss.item()

                    if do_code_regularization:
                        l2_size_loss = torch.sum(torch.norm(z, dim=1))
                        reg_loss = (
                            code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                        ) / num_sdf_samples
                    
                        chunk_loss = chunk_loss + reg_loss.cuda()
                        reg_loss_tb += reg_loss.item()
                    
                    summary_writer.add_scalar("Loss/train_vanilla", chunk_loss, global_step=epoch)
                    if use_eikonal:
                        grad_outputs = torch.ones_like(pred_sdf, requires_grad=True)
                        gradients = torch.autograd.grad(pred_sdf, [xyz[i]], grad_outputs=grad_outputs, create_graph=True, allow_unused=True, retain_graph=True)[0]
                        eikonal_loss = 0.002 * ((1. - torch.linalg.vector_norm(gradients, dim=1))**2).mean()
                        chunk_loss += eikonal_loss
                        eikonal_loss_tb += eikonal_loss.item()

                    if guided_contrastive_loss:
                        #Classification Loss
                        #SNN_Loss = loss.SNNLoss(temp)
                        z = z.cuda()
                        SNN_Loss = loss.Old_SNNLoss(temp)
                        loss_snn = SNN_Loss(z, labels)
                        chunk_loss += loss_snn * w_cls
                        #print(loss_snn.item())
                        snnl += loss_snn.item()
                        '''
                        #Regression Loss
                        SNN_Loss_Reg = SNNRegLoss(temp, threshold)
                        loss_snn_reg = SNN_Loss_Reg(z, label[:, :, 2])
                        loss += loss_snn_reg * w_cls
                        #print(loss_snn.item())
                        snnl_reg += loss_snn_reg.item()
                        '''
                    chunk_loss.backward()

                    batch_loss_tb += chunk_loss.item()
                    # Print batch loss
                print(f"Batch loss: {batch_loss_tb}")   
                #print(f"SNNL: {snnl}")   
                print(f"KL Loss: {kl_loss_tb}")              
                logging.debug("loss = {}".format(batch_loss_tb))
                loss_log.append(batch_loss_tb)
                epoch_losses.append(batch_loss_tb)
                epoch_sdf_losses.append(sdf_loss_tb)
                epoch_kl_losses.append(kl_loss_tb)
                epoch_reg_losses.append(reg_loss_tb)
                epoch_eikonal_losses.append(eikonal_loss_tb)
                epoch_snnl.append(snnl)

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
            summary_writer.add_scalar("Loss/train_kl", sum(epoch_kl_losses)/len(epoch_kl_losses), global_step=epoch)
            summary_writer.add_scalar("Loss/train_reg", sum(epoch_reg_losses)/len(epoch_reg_losses), global_step=epoch)
            if use_eikonal:
                summary_writer.add_scalar("Loss/train_eikonal", sum(epoch_eikonal_losses)/len(epoch_eikonal_losses), global_step=epoch)
            if guided_contrastive_loss:
                summary_writer.add_scalar("Loss/train_snnl", sum(epoch_snnl)/len(epoch_snnl), global_step=epoch)
            # Log learning rate.
            lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])
            summary_writer.add_scalar("Learning Rate/Params", lr_log[-1][0], global_step=epoch)
            summary_writer.add_scalar("Learning Rate/Latent", lr_log[-1][1], global_step=epoch)
            # Log latent vector length.
            mlm = get_mean_latent_vector_magnitude(mu)
            lat_mag_log.append(mlm)
            summary_writer.add_scalar("Mean Latent Magnitude/train", mlm, global_step=epoch)
            append_parameter_magnitudes(param_mag_log, decoder)
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
            summary_writer.add_scalar(f"GradsNorm/allLatParams.grad", torch.norm(mu.weight.grad.detach(), p=2).item(), global_step=epoch)

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
            logging.info(f"Epoch {epoch} completed in {seconds_elapsed:.2f} seconds.")
            logging.info(f"Checking for torus path for evaluation...")
            if torus_path is not None:
                logging.info(f"Torus path: {torus_path}")
                # Only if the path to the GT meshes exists.
                if epoch % eval_train_frequency == 0:
                    logging.info(f"Train Evaluation Started...")
                    # Training-set evaluation: Reconstruct mesh from learned latent and compute metrics.
                    chamfer_dists = []
                    chamfer_dists_all = []
                    eval_train_time_start = time.time()
                    for index in eval_train_scene_idxs:
                        lat_vec = mu(torch.LongTensor([index])).cuda()
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
                            cd, cd_all = metrics.compute_metric(gt_mesh=gt_mesh_path, gen_mesh=train_mesh, metric="chamfer")
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
                    summary_writer.add_scalar("Time/train eval per shape (sec)", (time.time()-eval_train_time_start)/len(eval_test_filenames), epoch)
                    # End of eval train.
                
                if epoch % eval_test_frequency == 0:
                    logging.info(f"Test Evaluation Started...")
                    # Test-set evaluation: Reconstruct latent and mesh from GT sdf values and compute metrics.
                    eval_test_time_start = time.time()
                    test_err_sum = 0.
                    chamfer_dists = []
                    chamfer_dists_all = []
                    test_loss_hists = []
                    mesh_label_names = []
                    test_latents = []
                    for test_fname in eval_test_filenames:
                        save_name = os.path.basename(sdf_dataset.npyfiles[index]).split(".npz")[0]
                        mesh_label_names.append(save_name)
                        path = os.path.join(experiment_directory, ws.tb_logs_dir, ws.tb_logs_test_reconstructions, save_name)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        test_fpath = os.path.join(data_source, ws.sdf_samples_subdir, test_fname)
                        test_sdf_samples = deep_sdf.data.read_sdf_samples_into_ram(test_fpath)
                        test_sdf_samples[0] = test_sdf_samples[0][torch.randperm(test_sdf_samples[0].shape[0])]
                        test_sdf_samples[1] = test_sdf_samples[1][torch.randperm(test_sdf_samples[1].shape[0])]

                        start = time.time()
                        test_loss_hist, test_latent = reconstruct.reconstruct(
                            decoder,
                            int(eval_test_optimization_steps),
                            latent_size,
                            test_sdf_samples,
                            0.01,  # [emp_mean,emp_var],
                            0.1,
                            num_samples=8000,
                            lr=5e-3,
                            l2reg=True,
                            return_loss_hist=True
                        )
                        logging.debug("[Test eval] Total reconstruction time: {}".format(time.time() - start))
                        if not np.isnan(test_loss_hist[-1]):
                            test_err_sum += test_loss_hist[-1]
                        test_loss_hists.append(test_loss_hist)
                        test_latents.append(test_latent)

                        start = time.time()
                        with torch.no_grad():
                            test_mesh = mesh.create_mesh(
                                decoder, 
                                test_latent, 
                                N=eval_grid_res, 
                                max_batch=int(2 ** 18), 
                                filename=os.path.join(path, f"epoch={epoch}"),
                                return_trimesh=True,
                            )
                        logging.debug("[Test eval] Total time to create test mesh: {}".format(time.time() - start))

                        if test_mesh is not None:
                            gt_mesh_path = f"{torus_path}/{save_name}.obj"
                            cd, cd_all = metrics.compute_metric(gt_mesh=gt_mesh_path, gen_mesh=test_mesh, metric="chamfer")
                            chamfer_dists.append(cd)
                            chamfer_dists_all.append(cd_all)

                        del test_sdf_samples, test_mesh

                    if chamfer_dists:
                        logging.debug(f"Test Chamfer distance mean: {sum(chamfer_dists)/len(chamfer_dists)} from {chamfer_dists}.")            
                        summary_writer.add_scalar("Mean Chamfer Dist/test", sum(chamfer_dists)/len(chamfer_dists), epoch)
                        summary_writer.add_scalar("Loss/test", test_err_sum/len(eval_test_filenames), epoch)
                        mlm = torch.mean(torch.norm(torch.cat(test_latents, dim=0), dim=1))
                        summary_writer.add_scalar("Mean Latent Magnitude/test", mlm, global_step=epoch)
                        fig = plotting.plot_train_stats(loss_hists=test_loss_hists, labels=mesh_label_names)
                        summary_writer.add_figure("Loss/test optimization curves", fig, epoch)
                        fig, percentiles = plotting.plot_dist_violin(np.concatenate(chamfer_dists_all, axis=0))
                        summary_writer.add_figure("CD Percentiles/test dists", fig, global_step=epoch)
                        for p in [75, 90, 99]:
                            if p in percentiles:
                                summary_writer.add_scalar(f"CD Percentiles/test {p}th", percentiles[p], global_step=epoch)
                    summary_writer.add_scalar("Time/test eval per shape (sec)", (time.time()-eval_test_time_start)/len(eval_test_filenames), epoch)
                    # End of eval test.

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
            "single_latent_size_mb": sum(p.nelement()*p.element_size() for p in mu.parameters()),
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
        summary_writer.add_graph(decoder, input)        
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
