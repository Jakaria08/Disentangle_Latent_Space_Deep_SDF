# Semi-supervised Disentanglement in Medical Shapes

DeepSDF-based pipeline for learning an SDF auto-decoder and a second-stage MLP‑VAE over latent codes.

## Overview

1. Preprocess meshes into SDF samples.
2. Stage 1: Train the SDF auto-decoder.
3. Stage 2: Train the MLP‑VAE on Stage 1 latent codes.

## File Organization

The scripts assume a shared organizational structure so that outputs from one step can be used by later steps.

### Data Layout

```text
<data_source_name>/
    .datasources.json
    SdfSamples/
        <dataset_name>/
            <class_name>/
                <instance_name>.npz
    SurfaceSamples/
        <dataset_name>/
            <class_name>/
                <instance_name>.ply
```

Split files (JSON) define subsets of the unified data source. See `examples/splits/`.

The file `datasources.json` stores a mapping from dataset names to paths. If data is moved, update this file accordingly.

### Experiment Layout

```text
<experiment_name>/
    specs.json
    Logs.pth
    LatentCodes/
        <Epoch>.pth
    ModelParameters/
        <Epoch>.pth
    OptimizerParameters/
        <Epoch>.pth
    Reconstructions/
        <Epoch>/
            Codes/
                <MeshId>.pth
            Meshes/
                <MeshId>.pth
    Evaluations/
        Chamfer/
            <Epoch>.json
        EarthMoversDistance/
            <Epoch>.json
    TensorBoard/
        ReconstructionsTrain/
            <classID_shapeID>/
                <epoch=Epoch>.ply
        ReconstructionsTest/
            <classID_shapeID>/
                <epoch=Epoch>.ply
        TensorBoardLogFile
```

The only file required to begin an experiment is `specs.json`, which sets parameters, architecture, and data paths.

## Preprocessing

The preprocessing code is in C++ and requires:

- CLI11
- Pangolin
- nanoflann
- Eigen3

Build:

```
mkdir build
cd build
cmake ..
make -j
```

Headless rendering (optional):

```
export PANGOLIN_WINDOW_URI=headless://
```

Preprocess SDF samples:

```
python preprocess_data.py --data_dir [path to sdf data folder] --source [path to mesh dataset folder] --name <dataset_name> --split examples/splits/<split>.json --skip
```

Preprocess surface samples (for evaluation):

```
python preprocess_data.py --data_dir [path to sdf data folder] --source [path to mesh dataset folder] --name <dataset_name> --split examples/splits/<split>.json --surface --skip
```

## Training

Stage 1: Train SDF auto-decoder:

```
python train_deep_sdf.py -e examples/<experiment_folder>
```

Stage 2: Train MLP‑VAE on latent codes:

```
python train_MLP_VAE_deep_sdf.py -e examples/<experiment_folder>
```

To resume training:

```
python train_deep_sdf.py -e examples/<experiment_folder> --continue <epoch>
```

## Reconstruction

```
python reconstruct.py -e examples/<experiment_folder> -c <epoch> --split examples/splits/<split>.json -d [path to sdf data folder] --skip
```

## Evaluation

```
python evaluate.py -e examples/<experiment_folder> -c <epoch> -d [path to sdf data folder] -s examples/splits/<split>.json
```
