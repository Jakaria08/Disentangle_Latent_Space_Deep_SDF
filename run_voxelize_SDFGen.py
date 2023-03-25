# Voxelizes all shapes in some ShapeNet split file with a list of given voxel grid resolutions.
# Saves the Chamfer distance of the reconstruction from Marching Cubes when using a dense and
# a sparse voxel grid.

import os
import json
import logging 
import argparse
import psutil
import time
from deep_sdf import metrics, utils, mesh
import trimesh
import pandas as pd
from tqdm import tqdm
import mesh_to_sdf
import skimage
import random
import copy
import math
from pqdm.processes import pqdm


def get_meshes_targets_and_specific_args(split_path: str, output_dir: str, voxel_resolution: int):
    # Prepare all input and output mesh files.
    with open(split_path, "r") as f:
        split = json.load(f)
        dataset_name = list(split.keys())[0]
        synset_id = list(split[dataset_name].keys())[0]
        shape_ids = split[dataset_name][synset_id]

    meshes_targets_and_specific_args = []
    file_not_found_cnt = 0
    for shape_id in shape_ids:
        input_obj_paths = [
            # Path that works with ShapeNetCore.v2
            os.path.join(input_dir, synset_id, shape_id, "models/model_normalized.obj"), 
            # Path that works with the DeepSDF dataset structure.
            os.path.join(input_dir, synset_id, shape_id + ".obj")
        ]
        existing_paths = [p for p in input_obj_paths if os.path.exists(p)]  # Should contain only one value.
        if not existing_paths:
            file_not_found_cnt += 1
            continue
        meshes_targets_and_specific_args.append({
            "input_obj_path": existing_paths[0],
            "output_obj_path": os.path.join(output_dir, synset_id, shape_id + ".ply"),
            "voxel_resolution": voxel_resolution,
        })
        os.makedirs(os.path.join(output_dir, synset_id), exist_ok=True)

    # Logging to terminal.
    logging.info(f"Voxelizing a total of {len(shape_ids)-file_not_found_cnt} shapes.")
    if file_not_found_cnt:
        logging.info(f"Could not find {file_not_found_cnt} out of {len(shape_ids)} shapes.")
    return meshes_targets_and_specific_args


def run_voxelize(mtsa: dict):
    
    PADDING = 3

    input_obj_path = mtsa["input_obj_path"]
    input_shape_id = input_obj_path.split(os.sep)[-3]
    input_class_id = input_obj_path.split(os.sep)[-4]
    input_shapenet_path = os.sep.join(input_obj_path.split(os.sep)[:-4])
    output_obj_path = mtsa["output_obj_path"]
    voxel_resolution = mtsa["voxel_resolution"]
    if os.path.exists(output_obj_path):
        return
    
    
    logging.debug(f"Voxelizing mesh: {input_obj_path}")

    # Extract voxel grid.
    voxel_dict = mesh.get_SDFGen_voxels(
        input_shape_id, voxel_resolution, PADDING, input_shapenet_path, input_class_id
    )
    voxels = voxel_dict["voxels"]
    voxel_size = voxel_dict["voxel_size"]
    # Reconstruct mesh from voxel grid.
    try:
        reconstruction, zero_level_dist = mesh.get_mesh_from_SDFGen_voxels(voxels, voxel_size, voxel_dict["centroid"], voxel_dict["scale"])
    except ValueError:
        # ValueError('Surface level must be within volume data range.')
        logging.debug(f"Caught ValueError at voxel-res {voxel_resolution} ({input_obj_path})")
        return
    # Compute reconstruction quality.
    gt_mesh = utils.as_mesh(trimesh.load(input_obj_path))
    cd, _ = metrics.compute_metric(gt_mesh, reconstruction, metric="chamfer")

    vert_cnt = len(reconstruction.vertices)
    num_dense_voxels = (voxel_resolution+2)**3

    # Compute sparse voxel grid.
    sparse_vox = copy.deepcopy(voxels)
    # Drop all voxels further than two voxel diagonals.
    sparse_vox[abs(sparse_vox)>2*math.sqrt(2*voxel_size**2)] = 1
    # Reconstruct mesh from sparse voxel grid.
    try:
        sparse_reconstruction, zero_level_dist_sparse = mesh.get_mesh_from_SDFGen_voxels(sparse_vox, voxel_size, voxel_dict["centroid"], voxel_dict["scale"])
    except ValueError:
        # ValueError('Surface level must be within volume data range.')
        logging.debug(f"Caught ValueError at voxel-res {voxel_resolution} ({input_obj_path})")
        return
    sparse_cd, _ = metrics.compute_metric(gt_mesh, sparse_reconstruction, metric="chamfer")
    num_sparse_voxels = num_dense_voxels - len(sparse_vox[sparse_vox == 1.0])

    logging.debug(f"Reduced with voxel-res {voxel_resolution} to chamfer distance of {cd:4f}. Reduced Mesh has {vert_cnt} Vertices.")
    with open(output_obj_path, "wb+") as f:
        f.write(trimesh.exchange.ply.export_ply(reconstruction))

    # Save results to logs.
    return [
        input_obj_path, 
        output_obj_path,
        voxel_resolution, 
        len(gt_mesh.vertices),
        vert_cnt,
        cd,
        sparse_cd,
        num_sparse_voxels,
        num_dense_voxels
    ]
        
if __name__ == "__main__":

    output_dir = "data/voxelize_sdf_gen/vox"      # This needs to be changed to where you want your data to be extracted to!
    input_dir = "/mnt/hdd/ShapeNetCore.v2"
    split_path = "examples/splits/sv2_planes_test.json"

    voxel_resolutions = [
        16,
        24,
        32,
        48, 
        64, 
        80,
        96, 
        112,
        128, 
        144,
        160, 
        176,
        192, 
        208,
        224,
        256,
        288,
        320
    ]

    # Setup args and logging.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        "--n_jobs",
        dest="n_jobs",
        default=1,
        help="Number of processes to run in parallel.",
    )
    utils.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    utils.configure_logging(args)

    for voxel_resolution in voxel_resolutions:  
        this_output_dir = output_dir + f"_res={voxel_resolution}"
        os.makedirs(this_output_dir, exist_ok=True)  
        logging.info(f"Voxelizing with resolution={voxel_resolution} into {this_output_dir}")
        meshes_targets_and_specific_args = get_meshes_targets_and_specific_args(split_path, this_output_dir, voxel_resolution)
        logs = []
        stopping_early = False
        try:
            logs = pqdm(meshes_targets_and_specific_args, run_voxelize, n_jobs=int(args.n_jobs))
        except KeyboardInterrupt:
            logging.info("Cleaning up and exiting. This might take a few minutes.")
            stopping_early = True
        finally:
            df_output_path = os.path.join(this_output_dir, "run_voxelize_logs.csv")
            logs_df = pd.DataFrame(
                [_ for _ in logs if _], 
                columns=[
                    "input_obj_path", 
                    "output_obj_path", 
                    "voxel_resolution", 
                    "gt_vertices", 
                    "decimated_vertices", 
                    "cd", 
                    "sparse_cd", 
                    "num_sparse_voxels", 
                    "num_dense_voxels"
                ],
            )
            if os.path.exists(df_output_path):
                logs_df_old = pd.read_csv(df_output_path)
                logs_df_all = pd.concat([logs_df_old, logs_df], ignore_index=True, axis=0)
                logs_df_all.to_csv(df_output_path)
            else:
                logs_df.to_csv(df_output_path, index=False)
            if stopping_early:
                break
