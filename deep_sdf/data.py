#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import time
import numpy as np
import os
import random
import torch
import torch.utils.data
import logging
import deep_sdf.workspace as ws
import trimesh
from typing import Tuple, List


def get_instance_filenames(data_source, split):
    npzfiles = []
    for instance_name in split:
        # Remove .obj extension
        instance_name_without_extension = os.path.splitext(instance_name)[0]
        instance_filename = os.path.join(data_source, instance_name_without_extension + ".npz")

        if not os.path.isfile(
            os.path.join(data_source, instance_filename)
        ):
            # raise RuntimeError(
            #     'Requested non-existent file "' + instance_filename + "'"
            # )
            logging.warning(
                "Requested non-existent file '{}'".format(instance_filename)
            )
        npzfiles += [instance_filename]

        # Add 5 augmented versions of the npz file
        '''
        for i in range(0, 5, 2):
            augmented_filename = instance_name_without_extension + f"_transformed_{i}.npz"
            if os.path.isfile(os.path.join(data_source, augmented_filename)):
                npzfiles.append(augmented_filename)
            else:
                logging.warning(f"Augmented file not found: '{augmented_filename}'")
        '''
    return npzfiles

def get_mesh_paths(data_source, split):
    mesh_paths = []
    for instance_name in split:
        instance_name_without_extension = os.path.splitext(instance_name)[0]
        instance_filename = os.path.join(data_source, instance_name)

        if not os.path.isfile(
            os.path.join(data_source, instance_name)
        ):
            # raise RuntimeError(
            #     'Requested non-existent file "' + instance_filename + "'"
            # )
            logging.warning(
                "Requested non-existent file '{}'".format(instance_filename)
            )
        mesh_paths += [instance_filename]
        '''
        for i in range(0, 5, 2):
            augmented_filename = instance_name_without_extension + f"_transformed_{i}.obj"
            augmented_filename_full = os.path.join(data_source, augmented_filename)
            if os.path.isfile(os.path.join(data_source, augmented_filename)):
                mesh_paths.append(augmented_filename_full)
            else:
                logging.warning(f"Augmented file not found: '{augmented_filename}'")
                print(data_source, augmented_filename)
        '''
    return mesh_paths


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns tuple containing a tensor of positive and a tensor of negative SDF samples."""
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])
    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples

def get_surface_points(mesh_path, num_points=4096):
    mesh = trimesh.load(mesh_path)
    points = mesh.sample(num_points)
    return points

class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        data_source_mesh,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        num_points=2048,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.data_source_mesh = data_source_mesh
        self.npyfiles = get_instance_filenames(data_source, split)
        self.mesh_paths = get_mesh_paths(data_source_mesh, split)
        self.labels = self.load_labels()

        self.surface_points = []
        for mesh_path in self.mesh_paths:
            points = get_surface_points(mesh_path, num_points)
            self.surface_points.append(points)

        logging.debug(f"Loaded {len(self.surface_points)} surface points")
        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram
        TIME = time.time()
        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )
        logging.debug(f"Time for loading into RAM: {(time.time() - TIME)*1000} ms"); TIME = time.time()

    def load_labels(self):
        labels = torch.load(self.data_source + "/labels.pt")

        expanded_labels = {}
        for key, value in labels.items():
            expanded_labels[key] = value
            '''
            # Add augmented labels
            for i in range(0, 5, 2):
                augmented_key = f"{key}_transformed_{i}"
                expanded_labels[augmented_key] = value
            '''
        return expanded_labels
    
    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        TIME = time.time()
        filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        
        label = self.labels[os.path.splitext(os.path.basename(self.npyfiles[idx]))[0]]
        label = torch.tensor(label)
        surface_point = self.surface_points[idx]
        
        if self.load_ram:
            retval = (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx, label, filename, surface_point
            )
        else:
            retval = unpack_sdf_samples(filename, self.subsample), idx, label, filename, surface_point
        
        logging.debug(f"Time for getting item: {(time.time() - TIME)*1000} ms"); TIME = time.time()
        return retval
